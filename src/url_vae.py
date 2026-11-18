"""The character-level URL VAE: model, training and scoring.

The model (the generative model of the project)
    A URL is a sequence of characters c_1 .. c_T over 97 ids (src/loaders.py).
    Encoder   a GRU reads the characters and gives the mean and log-variance of a Gaussian q(z | x), z of size `latent`.
    Decoder   an autoregressive GRU reads z and the previous characters and predicts the next character
              p(c_t | c_<t, z) (teacher forcing: the true previous character is the input during training).
              A start token (id 97, input only) is the first input. No end token is predicted, because the loss
              and the scores count the real characters only, so length enters through the number of characters and not
              through a model of where the URL stops. Padding is masked everywhere.
    Training  maximizes the ELBO  sum_t log p(c_t | c_<t, z) - beta * KL(q(z | x) || N(0, I)).
              Posterior collapse (the decoder ignores z, so KL goes to 0) is fought with KL annealing (beta grows
              linearly from 0 to `beta` over `anneal_epochs`) and a little word dropout (a share `word_dropout` of the
              decoder's input characters is replaced by the unknown id, which pushes the decoder to use z). The KL term
              is logged every epoch (`history`).
    Fitted on the benign TRAIN rows only. The best epoch is the one with the lowest negative ELBO per character on the
    benign VALIDATION rows (no malicious row and no label is used to choose it).
    Batches     hold URLs of similar length (`bucket_batches`; a random batch of 512 URLs always holds a long one, since the
               mean URL has 49 characters and the mean longest in a random batch is 200). The loss divides by a fixed number
               of characters, so the weight of a character does not depend on the length of the batch it falls in.

Scores saved for every item (higher means more likely malicious), both per character so URL length does not enter
through summation:
    nll_per_char       reconstruction negative log-likelihood per character, with z = the posterior mean mu(x)
    neg_elbo_per_char  (reconstruction NLL + KL) per character. The reconstruction term is averaged over
                       `score_samples` draws of z from q(z | x) (fixed random numbers, so a run repeats). The KL weight is 1
                       here whatever `beta` was in training, so it is the true negative ELBO.

Use (see 4_url_vae.ipynb, which calls `run_url_vae` in src/runner.py):

    model, history = train_vae(train, validation_benign, cfg, seed, device)
    scores = score_vae(model, test, cfg, device)              # DataFrame: nll_per_char, neg_elbo_per_char, kl, n_chars
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from src.loaders import PAD_ID, UNK_ID, VOCAB_SIZE
from src.nn_common import CharGRUEncoder, Tensors, batch_slices, epoch_batches, set_seed, trim

START_ID = VOCAB_SIZE                      # decoder input only, never a target
LOGVAR_RANGE = (-12.0, 8.0)                # keeps exp(logvar) finite
SCORE_NAMES = ("nll_per_char", "neg_elbo_per_char")


@dataclass
class VAEConfig:
    """Every setting of the URL VAE. The defaults are a first guess, not tuned. Tune on validation data only."""
    # network (4_url_vae.ipynb sweeps latent and beta)
    emb_dim: int = 32
    hidden: int = 64             # GRU size of the encoder and of the decoder
    latent: int = 16
    bidirectional: bool = False  # encoder only
    # training
    epochs: int = 8
    batch_size: int = 512
    lr: float = 2e-3
    grad_clip: float = 5.0
    beta: float = 1.0            # target weight of the KL term
    anneal_epochs: float = 3.0   # beta grows linearly from 0 to `beta` over this many epochs
    word_dropout: float = 0.1
    bucket_batches: int = 50     # batches of similar URL length (see nn_common.epoch_batches). 0 = random batches
    # scoring
    score_samples: int = 4       # draws of z for the negative ELBO

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "VAEConfig":
        return cls(**{k: v for k, v in d.items() if k in {f.name for f in fields(cls)}})


# ----------------------------------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------------------------------
class URLVAE(nn.Module):
    """GRU encoder, Gaussian latent, autoregressive GRU decoder (see the top of this file)."""

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = CharGRUEncoder(cfg.emb_dim, cfg.hidden, cfg.bidirectional)
        self.to_mu = nn.Linear(self.encoder.out_dim, cfg.latent)
        self.to_logvar = nn.Linear(self.encoder.out_dim, cfg.latent)
        self.dec_embedding = nn.Embedding(VOCAB_SIZE + 1, cfg.emb_dim)          # the extra id is the start token
        self.z_to_h = nn.Linear(cfg.latent, cfg.hidden)
        self.dec_gru = nn.GRU(cfg.emb_dim + cfg.latent, cfg.hidden, batch_first=True)
        self.out = nn.Linear(cfg.hidden, VOCAB_SIZE)

    def encode(self, ids: torch.Tensor, lengths: torch.Tensor):
        """Mean and log-variance of q(z | x) for a batch of trimmed ids (B, T) with their true lengths (B,)."""
        h = self.encoder(ids, lengths)
        return self.to_mu(h), self.to_logvar(h).clamp(*LOGVAR_RANGE)

    def decode(self, ids: torch.Tensor, z: torch.Tensor, word_dropout: float = 0.0) -> torch.Tensor:
        """Logits (B, T, 97) of every character given z and the previous characters (teacher forcing)."""
        prev = ids[:, :-1]
        if word_dropout > 0.0:
            drop = (torch.rand(prev.shape, device=prev.device) < word_dropout) & (prev != PAD_ID)
            prev = torch.where(drop, torch.full_like(prev, UNK_ID), prev)
        start = torch.full((ids.shape[0], 1), START_ID, dtype=ids.dtype, device=ids.device)
        x = self.dec_embedding(torch.cat([start, prev], dim=1))                    # (B, T, emb)
        x = torch.cat([x, z.unsqueeze(1).expand(-1, x.shape[1], -1)], dim=2)
        out, _ = self.dec_gru(x, torch.tanh(self.z_to_h(z)).unsqueeze(0).contiguous())
        return self.out(out)

    def rec_nll(self, ids: torch.Tensor, z: torch.Tensor, word_dropout: float = 0.0) -> torch.Tensor:
        """Reconstruction negative log-likelihood summed over the real characters of every URL, shape (B,)."""
        logits = self.decode(ids, z, word_dropout)
        ce = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), ids.reshape(-1), ignore_index=PAD_ID, reduction="none")
        return ce.view(ids.shape).sum(dim=1)


def kl_to_prior(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL( N(mu, diag(exp(logvar))) || N(0, I) ) in nats, shape (B,)."""
    return 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=1)


# ----------------------------------------------------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def score_vae(model: URLVAE, data: Tensors, device: torch.device, seed: int = 0, samples: int | None = None,
              batch_size: int = 2048) -> pd.DataFrame:
    """Per-URL scores in the row order of `data`: nll_per_char, neg_elbo_per_char, kl (nats) and n_chars.

    URLs are visited from short to long so each batch is trimmed to its longest URL. The noise for the negative ELBO
    comes from a generator seeded with `seed`, so the same model scores the same data the same way.
    """
    samples = model.cfg.score_samples if samples is None else samples
    was_training = model.training
    model.eval()
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    n = len(data)
    nll_mu = np.zeros(n)
    rec_mean = np.zeros(n)
    kl = np.zeros(n)
    order = torch.argsort(data.lengths).cpu().numpy()
    for idx in batch_slices(n, batch_size, order):
        sel = torch.from_numpy(idx).to(device)
        lens = data.lengths[sel]
        ids = trim(data.ids[sel], lens)
        mu, logvar = model.encode(ids, lens)
        nll_mu[idx] = model.rec_nll(ids, mu).double().cpu().numpy()
        std = (0.5 * logvar).exp()
        rec = torch.zeros(len(idx), device=device, dtype=torch.float64)
        for _ in range(samples):
            eps = torch.randn(mu.shape, device=device, generator=gen)
            rec += model.rec_nll(ids, mu + eps * std).double()
        rec_mean[idx] = (rec / samples).cpu().numpy()
        kl[idx] = kl_to_prior(mu, logvar).double().cpu().numpy()
    model.train(was_training)
    n_chars = data.lengths.cpu().numpy().astype(float)
    return pd.DataFrame({"nll_per_char": nll_mu / n_chars, "neg_elbo_per_char": (rec_mean + kl) / n_chars,
                         "kl": kl, "n_chars": n_chars})


# ----------------------------------------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------------------------------------
def train_vae(train: Tensors, validation: Tensors, cfg: VAEConfig, seed: int, device: torch.device, log=print):
    """Fit the VAE on `train` (benign train URLs) and keep the epoch with the lowest validation negative ELBO per char.

    `validation` must be benign validation URLs. Returns (model with the best weights, history DataFrame). The history
    has one row per epoch: beta, the training reconstruction per character and KL per URL, and on the validation URLs the
    negative ELBO per character, the NLL per character (z = mean), the KL per URL, plus the epoch time.
    """
    set_seed(seed)
    model = URLVAE(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    n = len(train)
    lengths = train.lengths.cpu().numpy()
    mean_chars = float(lengths.mean())               # a fixed divisor, so every character has the same weight in every batch
    steps_per_epoch = int(np.ceil(n / cfg.batch_size))
    anneal_steps = max(1.0, cfg.anneal_epochs * steps_per_epoch)
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    step, best, best_state, best_epoch, rows = 0, float("inf"), None, 0, []
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        model.train()
        rec_sum = torch.zeros((), dtype=torch.float64, device=device)      # summed on the device, read once per epoch
        kl_sum = torch.zeros((), dtype=torch.float64, device=device)
        for idx in epoch_batches(lengths, cfg.batch_size, gen, cfg.bucket_batches):
            sel = torch.from_numpy(idx).to(device)
            lens = train.lengths[sel]
            ids = trim(train.ids[sel], lens)
            beta_t = cfg.beta * min(1.0, step / anneal_steps)
            mu, logvar = model.encode(ids, lens)
            z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
            rec = model.rec_nll(ids, z, cfg.word_dropout)
            kl = kl_to_prior(mu, logvar)
            loss = (rec + beta_t * kl).sum() / (len(idx) * mean_chars)          # per character, KL weighted by beta_t
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)   # stop at once on NaN or inf
            opt.step()
            step += 1
            rec_sum += rec.detach().sum().double()
            kl_sum += kl.detach().sum().double()
        rec_sum, kl_sum, chars = float(rec_sum), float(kl_sum), float(lengths.sum())
        val = score_vae(model, validation, device, seed=seed, samples=1)
        row = {"epoch": epoch, "beta": beta_t, "train_rec_per_char": rec_sum / chars, "train_kl": kl_sum / n,
               "val_neg_elbo_per_char": float(val["neg_elbo_per_char"].mean()),
               "val_nll_per_char": float(val["nll_per_char"].mean()), "val_kl": float(val["kl"].mean()),
               "seconds": time.time() - t0}
        rows.append(row)
        if row["val_neg_elbo_per_char"] < best:
            best, best_epoch = row["val_neg_elbo_per_char"], epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        log(f"  epoch {epoch:2d}/{cfg.epochs}  beta {beta_t:.2f}  train rec/char {row['train_rec_per_char']:.3f}  "
            f"KL {row['train_kl']:.1f}  | val -ELBO/char {row['val_neg_elbo_per_char']:.3f}  val KL {row['val_kl']:.1f}"
            f"{'  (best)' if best_epoch == epoch else ''}  {row['seconds']:.0f}s")
    model.load_state_dict(best_state)
    history = pd.DataFrame(rows)
    history["best"] = history["epoch"] == best_epoch
    return model, history


# ----------------------------------------------------------------------------------------------------------------------
# Saving the model
# ----------------------------------------------------------------------------------------------------------------------
def save_vae(folder: Path | str, model: URLVAE, history: pd.DataFrame, seed: int) -> Path:
    """Write model.pt (weights and settings) and history.csv in `folder` (data/artifacts/url_vae/seed<k>, git-ignored)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": model.cfg.to_dict(), "seed": int(seed)}, folder / "model.pt")
    history.to_csv(folder / "history.csv", index=False)
    return folder / "model.pt"


def load_vae(folder: Path | str, device: torch.device) -> URLVAE:
    """Load a model saved by save_vae."""
    blob = torch.load(Path(folder) / "model.pt", map_location=device)
    model = URLVAE(VAEConfig.from_dict(blob["config"])).to(device)
    model.load_state_dict(blob["state_dict"])
    return model.eval()
