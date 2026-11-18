"""The supervised reference: the URL VAE's encoder plus one linear layer, trained with labels.

It is the project's "ceiling" context, not a contribution (the course asks for a generative model). It answers one
question: how much do labels and malicious examples add over a model that only saw benign URLs?

    Network    CharGRUEncoder (the same class and sizes as the VAE encoder, see src/nn_common.py) -> one linear layer ->
               one logit. Reusing the encoder matches the capacity, so a gap to the VAE does not come from network size.
    Training   binary cross-entropy on the benign AND malicious train rows (more than twice as many rows as the VAE sees).
    Selection  the epoch with the lowest validation cross-entropy. This uses validation LABELS (the VAE's selection does
               not), which is one more reason to read the reference as context and not as a fair upper bound.
    Score      `prob_malicious`, the predicted probability of malicious (higher means more likely malicious). It is
               computed from the logit in float64 so very confident URLs do not tie at exactly 1.

Read its results with the caveats in 5_supervised_reference.ipynb: the training data differs from the VAE's, and a labeled
model can use the shortcuts (missing scheme, IP host) directly.

Use (see 5_supervised_reference.ipynb, which calls `run_supervised` in src/runner.py):

    model, history = train_classifier(train, validation, cfg, seed, device)
    prob = score_classifier(model, test, device)               # numpy array, one probability per URL
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

from src.nn_common import CharGRUEncoder, Tensors, batch_slices, epoch_batches, set_seed, trim
from src.url_vae import VAEConfig


@dataclass
class SupervisedConfig:
    """Settings of the reference. The encoder sizes must equal the VAE's (use `SupervisedConfig.matching`)."""
    emb_dim: int = 32
    hidden: int = 64
    bidirectional: bool = False
    epochs: int = 6
    batch_size: int = 512
    lr: float = 1e-3
    grad_clip: float = 5.0
    bucket_batches: int = 50     # batches of similar URL length (see nn_common.epoch_batches). 0 = random batches

    @classmethod
    def matching(cls, vae: VAEConfig, **overrides) -> "SupervisedConfig":
        """A reference whose encoder has exactly the sizes of the VAE encoder `vae`."""
        return cls(emb_dim=vae.emb_dim, hidden=vae.hidden, bidirectional=vae.bidirectional, **overrides)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SupervisedConfig":
        return cls(**{k: v for k, v in d.items() if k in {f.name for f in fields(cls)}})


class URLClassifier(nn.Module):
    """The encoder of the VAE and one linear layer that gives the logit of "malicious"."""

    def __init__(self, cfg: SupervisedConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = CharGRUEncoder(cfg.emb_dim, cfg.hidden, cfg.bidirectional)
        self.head = nn.Linear(self.encoder.out_dim, 1)

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(ids, lengths)).squeeze(1)


@torch.no_grad()
def logits_of(model: URLClassifier, data: Tensors, device: torch.device, batch_size: int = 2048) -> np.ndarray:
    """Logits for every URL of `data`, in its row order (URLs are visited short to long for speed)."""
    was_training = model.training
    model.eval()
    out = np.zeros(len(data))
    order = torch.argsort(data.lengths).cpu().numpy()
    for idx in batch_slices(len(data), batch_size, order):
        sel = torch.from_numpy(idx).to(device)
        lens = data.lengths[sel]
        out[idx] = model(trim(data.ids[sel], lens), lens).double().cpu().numpy()
    model.train(was_training)
    return out


def score_classifier(model: URLClassifier, data: Tensors, device: torch.device) -> np.ndarray:
    """Predicted probability of malicious for every URL of `data` (float64, computed from the logit)."""
    return 1.0 / (1.0 + np.exp(-logits_of(model, data, device)))


def train_classifier(train: Tensors, validation: Tensors, cfg: SupervisedConfig, seed: int, device: torch.device,
                     log=print):
    """Fit the reference on `train` (benign and malicious train URLs) and keep the epoch with the lowest validation loss.

    Returns (model with the best weights, history DataFrame with train and validation cross-entropy per epoch).
    """
    set_seed(seed)
    model = URLClassifier(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    y_train = torch.from_numpy(train.labels.astype("float32")).to(device)
    y_val = validation.labels.astype("float64")
    n = len(train)
    lengths = train.lengths.cpu().numpy()
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    best, best_state, best_epoch, rows = float("inf"), None, 0, []
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum = 0.0
        for idx in epoch_batches(lengths, cfg.batch_size, gen, cfg.bucket_batches):
            sel = torch.from_numpy(idx).to(device)
            lens = train.lengths[sel]
            loss = F.binary_cross_entropy_with_logits(model(trim(train.ids[sel], lens), lens), y_train[sel])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)   # stop at once on NaN or inf
            opt.step()
            loss_sum += float(loss.detach()) * len(idx)
        z = logits_of(model, validation, device)
        val_loss = float(np.mean(np.logaddexp(0.0, z) - y_val * z))              # stable binary cross-entropy
        rows.append({"epoch": epoch, "train_bce": loss_sum / n, "val_bce": val_loss, "seconds": time.time() - t0})
        if val_loss < best:
            best, best_epoch = val_loss, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        log(f"  epoch {epoch:2d}/{cfg.epochs}  train BCE {rows[-1]['train_bce']:.4f}  val BCE {val_loss:.4f}"
            f"{'  (best)' if best_epoch == epoch else ''}  {rows[-1]['seconds']:.0f}s")
    model.load_state_dict(best_state)
    history = pd.DataFrame(rows)
    history["best"] = history["epoch"] == best_epoch
    return model, history


def save_classifier(folder: Path | str, model: URLClassifier, history: pd.DataFrame, seed: int) -> Path:
    """Write model.pt and history.csv in `folder` (data/artifacts/url_supervised/seed<k>, git-ignored)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": model.cfg.to_dict(), "seed": int(seed)}, folder / "model.pt")
    history.to_csv(folder / "history.csv", index=False)
    return folder / "model.pt"


def load_classifier(folder: Path | str, device: torch.device) -> URLClassifier:
    """Load a model saved by save_classifier."""
    blob = torch.load(Path(folder) / "model.pt", map_location=device)
    model = URLClassifier(SupervisedConfig.from_dict(blob["config"])).to(device)
    model.load_state_dict(blob["state_dict"])
    return model.eval()
