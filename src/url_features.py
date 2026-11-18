"""URL string features, computed from the URL text alone.

1_eda.ipynb builds the manifest with `url_features` (every column the items table and the floors need, plus the artefact
flags used in the cleaning log) and `norm_url` (the normalised key used to count near-duplicate URLs). Nothing here looks
at the label, so the same functions can score a new URL later.

    feat = url_features(df)          df has a `url` column; returns one row of features per URL, same index as df
    key  = norm_url(df["url"])       lower-cased, no scheme, no leading www., no trailing slash
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd

SCHEME_RE = r"^[A-Za-z][A-Za-z0-9+.\-]*://"
SECOND_LEVEL = {"co", "com", "org", "net", "gov", "edu", "ac", "or", "ne", "go"}


def norm_url(s: pd.Series) -> pd.Series:
    """Normalised URL key: strip spaces and quotes, lower-case, drop the scheme and a leading www., drop trailing slashes."""
    s = s.str.strip().str.strip("'\"").str.lower()
    s = s.str.replace(r"^[a-z][a-z0-9+.\-]*://", "", regex=True)
    s = s.str.replace(r"^www\.", "", regex=True)
    return s.str.rstrip("/")


def shannon(s: str) -> float:
    """Shannon entropy of the characters of `s`, in bits."""
    n = len(s)
    if n == 0:
        return 0.0
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


def approx_domain(host: str) -> str:
    """Approximate registered domain without a public-suffix list: last two labels, three when the pattern looks like
    example.co.uk. Good enough for grouping, not for security claims."""
    labels = host.split(".")
    if len(labels) <= 2:
        return host
    if len(labels[-1]) == 2 and labels[-2] in SECOND_LEVEL:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def url_features(df: pd.DataFrame) -> pd.DataFrame:
    """One row of length, scheme, host, path and artefact features per URL of `df["url"]` (index kept)."""
    u = df["url"]
    out = pd.DataFrame(index=df.index)
    out["url_len"] = u.str.len().astype("int32")
    scheme = u.str.extract(r"^([A-Za-z][A-Za-z0-9+.\-]*)://", expand=False).str.lower()
    out["scheme"] = np.select([scheme.eq("http"), scheme.eq("https"), scheme.notna()], ["http", "https", "other"], default="none")

    out["has_scheme"] = out["scheme"].ne("none")
    out["is_https"] = out["scheme"].eq("https")

    rest = u.str.replace(SCHEME_RE, "", regex=True)
    authority = rest.str.extract(r"^([^/?#]*)", expand=False).fillna("").str.strip(" '\"")
    out["has_userinfo"] = authority.str.contains("@", regex=False)
    hostport = authority.str.replace(r"^.*@", "", regex=True)
    out["has_port"] = hostport.str.contains(r":\d+$", regex=True)
    host = hostport.str.replace(r":\d*$", "", regex=True).str.lower().str.strip("[]")
    out["host"] = host
    out["host_is_ip"] = host.str.match(r"^\d{1,3}(?:\.\d{1,3}){3}$") | hostport.str.startswith("[")
    out["n_host_labels"] = host.str.count(r"\.") + 1
    out["host_hyphens"] = host.str.count("-")
    out["tld"] = np.where(out["host_is_ip"], "(ip)", host.str.rsplit(".", n=1).str[-1])
    out["domain"] = [approx_domain(h) for h in host]

    tail = rest.str.extract(r"^[^/?#]*(.*)$", expand=False).fillna("")
    path = tail.str.extract(r"^([^?#]*)", expand=False).fillna("")
    out["path_len"] = path.str.len().astype("int32")
    out["path_depth"] = path.str.count("/").astype("int16")
    out["has_query"] = tail.str.contains("?", regex=False)
    out["n_digits"] = u.str.count(r"\d").astype("int32")
    out["digit_ratio"] = out["n_digits"] / out["url_len"].clip(lower=1)
    out["n_upper"] = u.str.count(r"[A-Z]").astype("int32")
    out["n_pct"] = u.str.count("%").astype("int16")
    out["entropy"] = [shannon(s) for s in u]

    out["non_ascii"] = u.map(str.isascii).eq(False)
    out["edge_space_or_quote"] = u.str.contains(r"^[\s'\"]|[\s'\"]$", regex=True)
    out["html_entity"] = u.str.contains(r"&(?:amp|lt|gt|quot);", regex=True)
    out["bracket_artefact"] = u.str.contains(r"\]\(|\[|\]", regex=True)
    out["backslash"] = u.str.contains("\\", regex=False)
    return out
