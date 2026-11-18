"""Floors: benign-only Gaussian detectors that use a few trivial properties of an item and no learned model.

A floor is fitted on the benign TRAIN rows only. Its score is the distance of an item from the benign mean, so an item
with unusual feature values scores high (higher means more likely malicious, as for every score in the project).

    one feature     the absolute z-score |x - mean| / std  (score name `abs_z`)
    several         the Mahalanobis distance (score name `mahalanobis`), which accounts for how the features vary
                    together, so an unusual combination of ordinary values also scores high

Both are the same formula: the distance after standardizing each feature and whitening with the benign correlation
matrix. With one feature the correlation matrix is 1, so the distance is the absolute z-score.

A small ridge is added to the diagonal of the correlation matrix before it is inverted. It matters only when a feature
is almost constant or two features are almost copies of each other (for example `host_is_ip` among benign URLs). The
ridge is applied on the standardized scale, so it does not depend on the units of the features.

The two floors of the project are defined in `FLOORS` below (decided in 1_eda.ipynb section 6). `run_floors` in
src/runner.py fits and scores both and saves the scores, which is all 3_floors.ipynb does.

Use on its own:

    floor = fit_floor(benign_train, FLOORS["floorA_url"]["features"])
    val_scores = floor_scores(floor, validation)
    test_scores = floor_scores(floor, test)

Then save and evaluate the scores with `save_and_evaluate` (src/evaluate.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_RIDGE = 1e-6

# model name -> score name and the items-table columns the floor uses. Floor A is the headline floor.
FLOORS = {
    "floorA_url": {"score": "abs_z", "features": ["url_len"]},
    "floorB_url": {"score": "mahalanobis",
                   "features": ["url_len", "has_scheme", "host_is_ip", "n_host_labels", "path_len", "digit_ratio"]},
}


def fit_floor(train: pd.DataFrame, features: list[str], ridge: float = DEFAULT_RIDGE) -> dict:
    """Fit the benign Gaussian floor on `train` (pass the benign train rows only). Returns a dict of its parameters."""
    features = list(features)
    x = train[features].to_numpy(dtype="float64")
    if len(x) < 2:
        raise ValueError("a floor needs at least two training rows")
    if not np.isfinite(x).all():
        raise ValueError(f"floor features have missing or infinite values in the training rows: {features}")
    mean = x.mean(axis=0)
    std = x.std(axis=0, ddof=1)
    if (std <= 0).any():
        raise ValueError(f"a floor feature is constant in the training rows: {dict(zip(features, std))}")
    corr = np.atleast_2d(np.corrcoef((x - mean) / std, rowvar=False))
    if len(features) > 1:
        corr = corr + ridge * np.eye(len(features))
    return {"features": features, "mean": mean, "std": std, "corr": corr, "corr_inv": np.linalg.inv(corr),
            "ridge": ridge if len(features) > 1 else 0.0, "n_train": int(len(x))}


def floor_scores(floor: dict, rows: pd.DataFrame) -> np.ndarray:
    """Distance of every row from the benign mean: absolute z-score for one feature, Mahalanobis distance for several."""
    x = rows[floor["features"]].to_numpy(dtype="float64")
    if not np.isfinite(x).all():
        raise ValueError(f"floor features have missing or infinite values: {floor['features']}")
    z = (x - floor["mean"]) / floor["std"]
    return np.sqrt(np.einsum("ij,jk,ik->i", z, floor["corr_inv"], z))


def floor_summary(floor: dict) -> pd.DataFrame:
    """Mean and standard deviation of each feature in the benign training rows, for display."""
    return pd.DataFrame({"mean": floor["mean"], "std": floor["std"]}, index=floor["features"])
