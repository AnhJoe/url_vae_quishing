"""Build the shared items table that every model and the evaluation framework read.

One row is one URL (an "item"). Every split, stratum and floor feature lives in this one table, so all team members
train, tune and score on identical data and the evaluation framework can group results without touching raw files.

notebooks/2_data_processing.ipynb runs this step and checks the result. The same step can be run from the project root:

    python -m src.prepare_data                  # builds data/processed/items.csv.gz and items_summary.json

The model notebooks (3_floors, 4_url_vae, 5_supervised_reference, 6_results) read the table with `load_items` (below), never with a bare `pd.read_csv`.

Input (written by 1_eda.ipynb, under data/processed/):
    manifest.csv.gz           one row per URL with the cleaning decision (`keep`) and the URL features

Output: data/processed/items.csv.gz, one row per KEPT row (1,005,700 rows), in manifest order (all benign rows, then all
malicious rows). `item_id` is the row number in that order and is the key that score files use (see src/evaluate.py).

Table schema
------------
Identity and label
    item_id        int      0 .. n-1, position in the table. Stable as long as the manifest and split rule do not change.
    cls            str      "benign" or "malicious", taken from the class folder that holds the CSV (never from a file name).
    label          int      0 for benign, 1 for malicious. Malicious is the positive (anomalous) class.
    index          int      row number in the source CSV of that class. (cls, index) is unique.
    url            str      the URL exactly as encoded in the QR code (no cleaning, may lack a scheme).

Split (rule in src/splits.py, decided in 1_eda.ipynb section 5)
    split          str      "train", "validation" or "test". Benign and malicious rows are each split 70/15/15,
                            grouped by domain (all rows of one domain share a split), seed 0.
    domain         str      approximate registered domain, the grouping key. For an IP host it is the last two numbers
                            of the address (for example "115.240"). Empty for 15 rows. Bootstrap CIs resample this key.
    host           str      host name or address of the URL.

URL features (floor inputs and strata)
    url_len        int      number of characters in the URL.
    url_len_bin    str      "1-32", "33-64", "65-128", "129-256" or ">256". Stratum for URL results. The last bin is
                            exactly the truncated URLs.
    truncated      bool     True if url_len is above 256, the maximum length of the URL model (cut off at 256).
    scheme         str      "http", "https", "other" or "none" (no scheme). Stratum for URL results.
    has_scheme     bool     the URL starts with a scheme.
    is_https       bool     the scheme is https.
    host_is_ip     bool     the host is an IP address. Almost all such rows are malicious, so this group is reported as a
                            share only (the benign side is too small for a stratified AUROC).
    n_host_labels  int      number of dot-separated labels in the host.
    path_len       int      length of the path part.
    digit_ratio    float    digits divided by url_len.

Floor inputs: floor A uses url_len, floor B adds has_scheme, host_is_ip, n_host_labels, path_len and digit_ratio. All are
columns of this table, fitted on benign train rows only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.splits import DEFAULT_FRACS, DEFAULT_SEED, SPLIT_NAMES, grouped_split

ITEMS_FILE = "items.csv.gz"
MAX_URL_LEN = 256                                             # decided in 1_eda.ipynb section 6
URL_LEN_EDGES = [0, 32, 64, 128, MAX_URL_LEN, np.inf]
URL_LEN_LABELS = ["1-32", "33-64", "65-128", "129-256", ">256"]
STRING_COLUMNS = ["cls", "url", "split", "domain", "host", "scheme", "url_len_bin"]
BOOL_COLUMNS = ["truncated", "has_scheme", "is_https", "host_is_ip"]
ITEM_COLUMNS = ["item_id", "cls", "label", "index", "url", "split", "domain", "host",
                "url_len", "url_len_bin", "truncated", "scheme", "has_scheme", "is_https", "host_is_ip",
                "n_host_labels", "path_len", "digit_ratio"]


def load_items(proc_dir: Path | str, split: str | None = None, cls: str | None = None) -> pd.DataFrame:
    """Read items.csv.gz and optionally keep one split or one class. Text columns keep empty strings instead of NaN."""
    t = pd.read_csv(Path(proc_dir) / ITEMS_FILE, keep_default_na=False,
                    dtype={c: "string" for c in STRING_COLUMNS} | {c: "bool" for c in BOOL_COLUMNS})
    for c in STRING_COLUMNS:
        t[c] = t[c].astype(object)
    if split is not None:
        t = t[t["split"] == split]
    if cls is not None:
        t = t[t["cls"] == cls]
    return t.reset_index(drop=True)


def build_items(manifest: pd.DataFrame, fracs=DEFAULT_FRACS, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Combine the cleaned manifest and the grouped split into the items table."""
    rows = manifest[manifest["keep"]].reset_index(drop=True)
    rows = rows.assign(split_id=-1)
    for cls in ["benign", "malicious"]:                        # each class is split on its own, same rule as splits.py
        m = rows["cls"] == cls
        rows.loc[m, "split_id"] = grouped_split(rows.loc[m, "domain"], fracs, seed)
    rows["split"] = rows["split_id"].map(dict(enumerate(SPLIT_NAMES)))
    rows["item_id"] = np.arange(len(rows))
    rows["url_len_bin"] = pd.cut(rows["url_len"], URL_LEN_EDGES, labels=URL_LEN_LABELS).astype(object)
    rows["truncated"] = rows["url_len"] > MAX_URL_LEN
    return rows[ITEM_COLUMNS]


def check_items(items: pd.DataFrame) -> dict:
    """Checks that must hold before the table is saved. Returns the counts used in items_summary.json."""
    assert items["item_id"].is_unique and (items["item_id"].to_numpy() == np.arange(len(items))).all()
    assert not items.duplicated(["cls", "index"]).any()
    assert items["split"].isin(SPLIT_NAMES).all()
    assert (items["label"] == (items["cls"] == "malicious").astype(int)).all()
    for cls, g in items.groupby("cls"):                        # a domain never sits in two splits of the same class
        assert (g.groupby("domain")["split"].nunique() <= 1).all(), f"domain split across splits in {cls}"
    return {s: {c: int(n) for c, n in g["cls"].value_counts().items()} for s, g in items.groupby("split")}


def data_version(items: pd.DataFrame) -> str:
    """Short hash of which row is in which split. It changes if the rows, the order or the split change."""
    h = hashlib.sha1()
    h.update("\n".join(items["cls"] + "/" + items["index"].astype(str) + "/" + items["split"]).encode())
    return h.hexdigest()[:12]


RAW_CLASS_DIRS = {"benign": "QR_All_benign", "malicious": "QR_All_Malicious"}


def source_files(root: Path) -> dict:
    """File name, size and SHA-256 of the two source CSVs under data/raw, so a result can name the data it came from.
    Empty if the raw files are not there."""
    out = {}
    for cls, folder in RAW_CLASS_DIRS.items():
        found = sorted((root / "data" / "raw" / folder).glob("all_generated_urls_*.csv"))
        if not found:
            return {}
        digest = hashlib.sha256()
        with open(found[0], "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        out[cls] = {"file": found[0].name, "bytes": found[0].stat().st_size, "sha256": digest.hexdigest()}
    return out


def write_sources(root: Path) -> dict:
    """Add the source file checksums to an existing items_summary.json (the items table is not touched)."""
    path = root / "data" / "processed" / "items_summary.json"
    summary = json.loads(path.read_text())
    summary["sources"] = source_files(root)
    path.write_text(json.dumps(summary, indent=2))
    return summary["sources"]


def run_items(root: Path) -> None:
    """Build items.csv.gz and items_summary.json from the manifest."""
    proc = root / "data" / "processed"
    manifest = pd.read_csv(proc / "manifest.csv.gz", keep_default_na=False)
    items = build_items(manifest)
    counts = check_items(items)
    items.to_csv(proc / ITEMS_FILE, index=False)

    summary = {
        "rows": int(len(items)), "data_version": data_version(items),
        "split_rule": {"key": "domain", "fractions": DEFAULT_FRACS, "seed": DEFAULT_SEED},
        "max_url_len": MAX_URL_LEN, "url_len_bins": URL_LEN_LABELS,
        "rows_by_split_and_class": counts,
        "truncated_by_class": {c: int(g["truncated"].sum()) for c, g in items.groupby("cls")},
        "sources": source_files(root),
    }
    (proc / "items_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"saved {proc / ITEMS_FILE}: {len(items):,} items, data version {summary['data_version']}")
    print("saved", proc / "items_summary.json")


def main() -> None:
    p = argparse.ArgumentParser(description="Build the shared items table (see the top of this file for the schema).")
    p.add_argument("--root", type=Path, default=Path.cwd(), help="project root (the folder that contains src and data)")
    p.add_argument("--sources-only", action="store_true", help="only add the source file checksums to items_summary.json")
    a = p.parse_args()
    if a.sources_only:
        print(json.dumps(write_sources(a.root), indent=2))
    else:
        run_items(a.root)


if __name__ == "__main__":
    main()
