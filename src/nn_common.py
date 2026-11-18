"""PyTorch pieces shared by the URL VAE and the supervised reference.

Keeping them here has one purpose: the VAE and its supervised reference read a URL with the SAME encoder, so a gap
between them does not come from network size (the reference is "the VAE's encoder plus one linear layer").

    set_seed, get_device       reproducibility and the device (a local NVIDIA GPU when there is one)
    encode_items               turn the URLs of an items table into one uint8 array, once per run
    Tensors, to_tensors        a subset of that array (and the URL lengths) on the device
    batch_slices               index batches over a subset, in random or length order
    CharGRUEncoder             embedding + GRU, returns one vector per URL

Model input (src/loaders.py): 256 ids per URL, 0 = padding, 1 = unknown, 2 to 96 = printable ASCII. A URL is never
empty (checked by checks.data_checks), so every length is at least 1. The padding is masked in every loss.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from src.loaders import MAX_URL_LEN, PAD_ID, VOCAB_SIZE, encode_urls


# ----------------------------------------------------------------------------------------------------------------------
# Seed and device
# ----------------------------------------------------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (CPU and GPU) so a run with the same seed repeats."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str = "auto") -> torch.device:
    """'auto' (the default) uses the GPU when torch can see one and the CPU otherwise. 'cpu' forces the CPU. 'cuda' forces
    the GPU and stops with a clear message if torch cannot see one."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("No GPU is visible to PyTorch (torch.cuda.is_available() is False). Install a CUDA build of "
                           "torch and restart the kernel, or use device='auto' or 'cpu'.")
    return torch.device(name)


# ----------------------------------------------------------------------------------------------------------------------
# Data on the device
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class Encoded:
    """Every URL of an items table as ids, in the row order of that table."""
    ids: np.ndarray        # uint8, shape (n, 256)
    lengths: np.ndarray    # int64, number of real characters (1 to 256)


def encode_items(items: pd.DataFrame) -> Encoded:
    """Encode all URLs of `items` (about 20 seconds for the full table). Do it once and reuse it for every seed."""
    ids = encode_urls(items["url"]).astype(np.uint8)
    lengths = (ids != PAD_ID).sum(axis=1).astype(np.int64)
    if len(lengths) and lengths.min() < 1:
        raise ValueError("an empty URL reached the model input")
    return Encoded(ids=ids, lengths=lengths)


@dataclass
class Tensors:
    """A subset of the encoded URLs on the device, with the matching labels and item ids (kept on the CPU)."""
    ids: torch.Tensor        # uint8 (n, 256), on the device
    lengths: torch.Tensor    # int64 (n,), on the device
    labels: np.ndarray       # int (n,)
    item_id: np.ndarray      # int (n,)

    def __len__(self) -> int:
        return int(self.ids.shape[0])


def to_tensors(items: pd.DataFrame, enc: Encoded, mask, device: torch.device) -> Tensors:
    """The rows of `items` where `mask` is true (a boolean array over the rows), as tensors on the device."""
    m = np.asarray(mask, dtype=bool)
    return Tensors(ids=torch.from_numpy(enc.ids[m]).to(device), lengths=torch.from_numpy(enc.lengths[m]).to(device),
                   labels=items["label"].to_numpy()[m], item_id=items["item_id"].to_numpy()[m])


def batch_slices(n: int, batch_size: int, order: np.ndarray | torch.Tensor | None = None):
    """Yield index arrays of at most `batch_size` positions. `order` is the visiting order (default 0 .. n-1)."""
    order = np.arange(n) if order is None else np.asarray(order)
    for start in range(0, n, batch_size):
        yield order[start:start + batch_size]


def epoch_batches(lengths: np.ndarray, batch_size: int, gen: torch.Generator, chunk_batches: int = 0) -> list[np.ndarray]:
    """The batches of one training epoch, as index arrays, in the order to train on them.

    chunk_batches = 0: shuffle all rows and cut them into batches. Every batch is padded to its longest URL, and a random
    batch nearly always holds a long one, so most of the computed positions are padding.
    chunk_batches > 0: shuffle all rows, cut the shuffle into chunks of `chunk_batches` batches, sort each chunk by URL
    length, cut it into batches of similar length, then shuffle the batches. Same rows and the same number of batches per
    epoch, but little padding. Every batch is still a random draw from a small window of the shuffle, and the order of the
    batches is random.
    """
    n = len(lengths)
    order = torch.randperm(n, generator=gen).numpy()
    if chunk_batches <= 0:
        return list(batch_slices(n, batch_size, order))
    size = batch_size * int(chunk_batches)
    batches: list[np.ndarray] = []
    for start in range(0, n, size):
        chunk = order[start:start + size]
        chunk = chunk[np.argsort(lengths[chunk], kind="stable")]
        batches += list(batch_slices(len(chunk), batch_size, chunk))
    return [batches[i] for i in torch.randperm(len(batches), generator=gen).numpy()]


def trim(ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Cut the padding columns that no URL of the batch uses, and return the ids as int64 for the embedding."""
    return ids[:, :int(lengths.max())].long()


# ----------------------------------------------------------------------------------------------------------------------
# The encoder
# ----------------------------------------------------------------------------------------------------------------------
class CharGRUEncoder(nn.Module):
    """Embedding and a GRU over the characters. Returns the last hidden state(s) of the GRU as one vector per URL.

    With `bidirectional` the forward and backward states are joined, so the output size is 2 * hidden. The URL is
    packed with its true length, so the padding never reaches the GRU.
    """

    def __init__(self, emb_dim: int = 64, hidden: int = 256, bidirectional: bool = True):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, emb_dim, padding_idx=PAD_ID)
        self.gru = nn.GRU(emb_dim, hidden, batch_first=True, bidirectional=bidirectional)
        self.out_dim = hidden * (2 if bidirectional else 1)

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(self.embedding(ids), lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)                                   # (directions, batch, hidden)
        return h.transpose(0, 1).reshape(ids.shape[0], -1)


__all__ = ["set_seed", "get_device", "Encoded", "encode_items", "Tensors", "to_tensors", "batch_slices", "trim",
           "CharGRUEncoder", "MAX_URL_LEN"]
