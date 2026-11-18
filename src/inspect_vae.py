"""Look inside a trained URL VAE: what it finds surprising, where URLs sit in its latent space, and what it generates.

These functions only compute the numbers behind the report figures (the drawing is in src/plots.py). They read a trained
model and never train or score anything for the results tables.

    char_nll(model, data, device)               negative log-likelihood of every character of every URL (z = mu)
    surprise_examples(test, scores, ...)        pick example URLs: false alarms, typical, caught and missed
    example_nll(model, test, enc, positions, device)   char_nll of the picked test URLs, in the order of `positions`
    latent_means(model, data, device)           the posterior mean mu of every URL, one row per URL
    latent_sample(model, test, enc, device)     mu of a random, class-balanced sample of the test URLs
    pca_2d(x)                                   the first two principal components and the variance they explain
    active_units(mu)                            how many latent dimensions carry information (a collapse check)
    prior_samples(model, n, max_chars, ...)     URLs written by the decoder from z drawn from the prior N(0, I)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.loaders import PAD_ID, VOCAB_SIZE, decode_url
from src.nn_common import Encoded, Tensors, batch_slices, to_tensors, trim
from src.url_vae import START_ID, URLVAE


@torch.no_grad()
def char_nll(model: URLVAE, data: Tensors, device: torch.device, batch_size: int = 1024) -> list[np.ndarray]:
    """Negative log-likelihood in nats of every real character of every URL, given z = mu. One array per URL (row order)."""
    was_training = model.training
    model.eval()
    out: list = [None] * len(data)
    order = torch.argsort(data.lengths).cpu().numpy()
    for idx in batch_slices(len(data), batch_size, order):
        sel = torch.from_numpy(idx).to(device)
        lens = data.lengths[sel]
        ids = trim(data.ids[sel], lens)
        mu, _ = model.encode(ids, lens)
        logits = model.decode(ids, mu)
        ce = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), ids.reshape(-1).long(), ignore_index=PAD_ID, reduction="none")
        ce = ce.view(ids.shape).cpu().numpy()
        for row, i in enumerate(idx):
            out[i] = ce[row, :int(lens[row])]
    model.train(was_training)
    return out


def surprise_examples(test: pd.DataFrame, score: np.ndarray, per_group: int = 2, max_len: int = 70, min_len: int = 12) -> pd.DataFrame:
    """Example test URLs for the surprise figure, chosen by their score and never by looking at the characters.

    Four groups, among URLs of `min_len` to `max_len` characters (the longest so a row fits on a page, the shortest so an example
    is a URL and not a few characters): the benign URLs with the highest
    score (false alarms), the benign URLs nearest the benign median (typical), the malicious URLs with the highest score
    (caught) and the malicious URLs with the lowest score (missed). Returns the chosen rows of `test` with `group` and
    `score` columns and `position` (the row number in `test`).
    """
    t = test.reset_index(drop=True).assign(score=np.asarray(score, dtype=float), position=np.arange(len(test)))
    t = t[t["url_len"].between(min_len, max_len)]
    benign, malicious = t[t["label"] == 0], t[t["label"] == 1]
    typical = (benign["score"] - benign["score"].median()).abs().nsmallest(per_group).index
    picks = [("benign, highest score (false alarms)", benign.nlargest(per_group, "score")),
             ("benign, typical score", benign.loc[typical]),
             ("malicious, highest score (caught)", malicious.nlargest(per_group, "score")),
             ("malicious, lowest score (missed)", malicious.nsmallest(per_group, "score"))]
    return pd.concat([p.assign(group=name) for name, p in picks], ignore_index=True)


@torch.no_grad()
def latent_means(model: URLVAE, data: Tensors, device: torch.device, batch_size: int = 2048) -> np.ndarray:
    """The posterior mean mu of every URL, shape (n_urls, latent), in the row order of `data`."""
    was_training = model.training
    model.eval()
    out = np.zeros((len(data), model.cfg.latent))
    order = torch.argsort(data.lengths).cpu().numpy()
    for idx in batch_slices(len(data), batch_size, order):
        sel = torch.from_numpy(idx).to(device)
        lens = data.lengths[sel]
        out[idx] = model.encode(trim(data.ids[sel], lens), lens)[0].cpu().numpy()
    model.train(was_training)
    return out


def example_nll(model: URLVAE, test: pd.DataFrame, enc: Encoded, positions, device: torch.device) -> list[np.ndarray]:
    """`char_nll` of the test rows at `positions` (row numbers of `test`, as `enc` was made from `test`), in that order."""
    positions = np.asarray(positions)
    mask = np.zeros(len(test), dtype=bool)
    mask[positions] = True
    by_row = dict(zip(np.flatnonzero(mask), char_nll(model, to_tensors(test, enc, mask, device), device)))
    return [by_row[p] for p in positions]


def latent_sample(model: URLVAE, test: pd.DataFrame, enc: Encoded, device: torch.device, per_class: int = 2000, seed: int = 0):
    """Posterior means of `per_class` random test URLs of each class. Returns (row numbers of `test`, ascending; mu)."""
    rng = np.random.default_rng(seed)
    picks = [rng.choice(np.flatnonzero(test["label"].to_numpy() == k), per_class, replace=False) for k in (0, 1)]
    positions = np.sort(np.concatenate([p[:per_class] for p in picks]))
    mask = np.zeros(len(test), dtype=bool)
    mask[positions] = True
    return positions, latent_means(model, to_tensors(test, enc, mask, device), device)


def pca_2d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project onto the first two principal components. Returns (coordinates (n, 2), share of variance of each of the two)."""
    x = np.asarray(x, dtype=float)
    centred = x - x.mean(axis=0)
    _, s, vt = np.linalg.svd(centred, full_matrices=False)
    var = s ** 2
    return centred @ vt[:2].T, var[:2] / var.sum()


def active_units(mu: np.ndarray, threshold: float = 0.01) -> int:
    """Number of latent dimensions whose posterior mean varies across URLs by more than `threshold` (in variance).

    A dimension that does not vary carries no information about the URL. Zero active units means posterior collapse.
    """
    return int((np.var(np.asarray(mu), axis=0) > threshold).sum())


@torch.no_grad()
def _sample_ids(model: URLVAE, z: torch.Tensor, max_chars: int, gen: torch.Generator, temperature: float):
    """Write one URL per row of z, one character at a time. Returns (ids (n, max_chars), summed log-probability (n,))."""
    n = z.shape[0]
    h = torch.tanh(model.z_to_h(z)).unsqueeze(0).contiguous()
    prev = torch.full((n, 1), START_ID, dtype=torch.long, device=z.device)
    ids, logp = [], torch.zeros(n, device=z.device)
    for _ in range(max_chars):
        x = torch.cat([model.dec_embedding(prev), z.unsqueeze(1)], dim=2)
        out, h = model.dec_gru(x, h)
        logits = model.out(out[:, 0]) / temperature
        logits[:, PAD_ID] = float("-inf")                    # padding is never a character
        probs = F.softmax(logits, dim=1)
        nxt = torch.multinomial(probs, 1, generator=gen)
        logp += torch.log(probs.gather(1, nxt))[:, 0]
        ids.append(nxt)
        prev = nxt
    return torch.cat(ids, dim=1), logp


@torch.no_grad()
def prior_samples(model: URLVAE, n: int, max_chars: int, device: torch.device, seed: int = 0, temperature: float = 1.0) -> list[str]:
    """`n` URLs written by the decoder from z drawn from the prior N(0, I), the first `max_chars` characters of each.

    The model has no end-of-URL token (it is trained on the real characters only), so a sample has no natural end and is cut
    at `max_chars`.
    """
    was_training = model.training
    model.eval()
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    z = torch.randn(n, model.cfg.latent, device=device, generator=gen)
    ids, _ = _sample_ids(model, z, max_chars, gen, temperature)
    model.train(was_training)
    return [decode_url(row) for row in ids.cpu().numpy()]
