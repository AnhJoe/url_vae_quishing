"""Turn an item into the array a model reads, the same way for everyone.

The URL model reads a URL as 256 integers (decided in 1_eda.ipynb section 6):
    - id 0 is padding, id 1 is the unknown token, ids 2 to 96 are the 95 printable ASCII symbols (space to tilde),
    - a URL longer than 256 characters is cut off at 256 (the `truncated` column of the items table marks these),
    - a shorter URL is padded with 0 at the end. The model must mask the padding in its loss.
    - No start or end token is added. A sequence model that needs them adds its own and says so in its notebook.

Typical use:

    ids = encode_urls(items_subset["url"])               # int16 array, shape (n, 256)
    text = decode_url(ids[0])                            # back to a string (padding dropped, unknown token as "?")
"""
from __future__ import annotations

import numpy as np

MAX_URL_LEN = 256
PAD_ID, UNK_ID = 0, 1
FIRST_CHAR_ID = 2
CHARS = "".join(chr(c) for c in range(32, 127))              # the 95 printable ASCII symbols
VOCAB_SIZE = FIRST_CHAR_ID + len(CHARS)                      # 97 ids in total
_ID_OF = {ch: FIRST_CHAR_ID + i for i, ch in enumerate(CHARS)}


def encode_url(url: str, max_len: int = MAX_URL_LEN) -> np.ndarray:
    """One URL as an int16 array of length max_len (ids described at the top of this file)."""
    out = np.full(max_len, PAD_ID, dtype=np.int16)
    for i, ch in enumerate(url[:max_len]):
        out[i] = _ID_OF.get(ch, UNK_ID)
    return out


def encode_urls(urls, max_len: int = MAX_URL_LEN) -> np.ndarray:
    """Many URLs as one int16 array of shape (n, max_len)."""
    urls = list(urls)
    return np.stack([encode_url(u, max_len) for u in urls]) if urls else np.zeros((0, max_len), dtype=np.int16)


def decode_url(ids) -> str:
    """Turn ids back into text. Padding is dropped and the unknown token is shown as a question mark."""
    return "".join(CHARS[i - FIRST_CHAR_ID] if i >= FIRST_CHAR_ID else "?" for i in np.asarray(ids).tolist() if i != PAD_ID)
