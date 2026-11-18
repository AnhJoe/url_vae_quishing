"""Run a model on the items table for one or more seeds and save its scores. The model notebooks call these functions.

Every function takes the items table, trains or fits on the TRAIN rows, scores every validation and test item, and saves
the scores with `save_and_evaluate` (src/evaluate.py), so every model is evaluated the same way. Each returns the metrics
table (one row per score and seed).

    run_floors(items, results_dir)                       fit and score floor A and floor B (no randomness, seed 0)
    run_url_vae(items, results_dir, artifacts_dir, ...)  train the URL VAE for each seed, save both scores per seed
    run_supervised(items, results_dir, artifacts_dir, ...)  train the supervised reference for each seed
    run_tuning(items, artifacts_dir, base_cfg)           sweep the VAE's latent size and beta on seed 0 (no labels, no test)

Every saved run has a run record (`run.json`, next to its score files) with the model settings and the data version (a hash
of which row sits in which split). A seed is skipped when its score files exist AND its record matches the run being asked
for, so a long multi-seed run can be restarted after an interruption. If the settings or the data changed (or the scores have
no record), the seed is trained again and the log says what changed. `overwrite=True` trains every seed again. The model
weights and the training history of every seed go to `data/artifacts/<model>/seed<k>/` (git-ignored). The record also
names the environment (Python, numpy, pandas and torch versions, the device) and, when the notebook passes
`sources=summary["sources"]`, the checksums of the two source CSVs. Those are for the report and are never compared.

Command line (from the project root), useful for long GPU runs outside a notebook:

    python -m src.runner floors
    python -m src.runner tune
    python -m src.runner url_vae
    python -m src.runner url_supervised --epochs 6
"""
from __future__ import annotations

import argparse
import json
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import evaluate_run, run_dir, save_and_evaluate
from src.floors import FLOORS, fit_floor, floor_scores
from src.nn_common import Encoded, encode_items, get_device, to_tensors
from src.prepare_data import data_version, load_items
from src.supervised import (SupervisedConfig, save_classifier, score_classifier, train_classifier)
from src.url_vae import SCORE_NAMES, VAEConfig, save_vae, score_vae, train_vae

DEFAULT_SEEDS = (0,)


def _rows(items: pd.DataFrame, split: str, cls: str | None = None) -> np.ndarray:
    """Boolean mask over `items` for one split, and one class if given."""
    m = (items["split"] == split).to_numpy()
    return m & (items["cls"] == cls).to_numpy() if cls else m


RECORD_FILE = "run.json"


def _environment(dev=None) -> dict:
    """Versions and device of this run, for the record. Only the settings and the data version are ever compared."""
    env = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
    try:
        import torch
    except ImportError:
        return env
    env["torch"] = torch.__version__
    if dev is not None:
        env["device"] = str(dev)
        if dev.type == "cuda":
            env["gpu"], env["cuda"] = torch.cuda.get_device_name(dev), torch.version.cuda
    return env


def _record(model: str, seed: int, version: str, n_items: int, config: dict, env: dict | None = None,
            sources: dict | None = None) -> dict:
    """What a run depends on: its settings and the data (which row sits in which split). Saved next to its scores.

    `environment` and `sources` (checksums of the source CSVs) are for reading later and are not compared."""
    out = {"model": model, "seed": int(seed), "data_version": version, "n_items": int(n_items), "config": config}
    if env:
        out["environment"] = env
    if sources:
        out["sources"] = sources
    return out


def _write_record(results_dir, model: str, scores, seed: int, record: dict) -> None:
    for s in scores:
        (run_dir(results_dir, model, s, seed) / RECORD_FILE).write_text(json.dumps(record, indent=2))


def _changes(saved: dict, now: dict) -> str:
    """Plain words for what differs between a saved run record and the run being asked for ('' if nothing differs)."""
    old_cfg, new_cfg = saved.get("config", {}), now["config"]
    out = [f"{k} {old_cfg.get(k)} to {new_cfg.get(k)}" for k in sorted(set(old_cfg) | set(new_cfg)) if old_cfg.get(k) != new_cfg.get(k)]
    if saved.get("data_version") != now["data_version"]:
        out.append(f"data version {saved.get('data_version')} to {now['data_version']}")
    return ", ".join(out)


def _status(results_dir, model: str, scores, seed: int, record: dict) -> tuple[str, str]:
    """("done", "") when every score file exists with a run record that matches `record`, ("missing", "") when there are no
    score files, and ("stale", reason) when they exist but come from other settings or data, or have no record."""
    folders = [run_dir(results_dir, model, s, seed) for s in scores]
    have = [all((f / name).exists() for name in ("validation.csv.gz", "test.csv.gz")) for f in folders]
    if not any(have):
        return "missing", ""
    if not all(have):
        return "stale", "some score files are missing"
    for f in folders:
        if not (f / RECORD_FILE).exists():
            return "stale", "the score files have no run record"
        changes = _changes(json.loads((f / RECORD_FILE).read_text()), record)
        if changes:
            return "stale", changes
    return "done", ""


def _existing(results_dir, items, model: str, scores, seed: int) -> pd.DataFrame:
    return pd.concat([evaluate_run(run_dir(results_dir, model, s, seed), items) for s in scores], ignore_index=True)


def quick_subset(items: pd.DataFrame, per_group: int = 3000, seed: int = 0) -> pd.DataFrame:
    """A small copy of the items table for a trial run: `per_group` random rows of each split and class.

    All benign train rows with an IP-address host are kept, because floor B needs that feature to vary. The rows keep their
    item_id, so the scores of a trial run are saved and evaluated like any other run. Save them to a separate folder
    (for example results/quick) so they never mix with the real results.
    """
    parts = [g.sample(min(per_group, len(g)), random_state=seed) for _, g in items.groupby(["split", "cls"])]
    parts.append(items[(items["split"] == "train") & (items["cls"] == "benign") & items["host_is_ip"]])
    return pd.concat(parts).drop_duplicates("item_id").sort_values("item_id").reset_index(drop=True)


def trial_setup(items: pd.DataFrame, results_dir: Path | str, artifacts_dir: Path | str, quick: bool, per_group: int = 3000):
    """For a notebook's QUICK switch: the items, results folder and artifacts folder to run on.

    quick=False returns the inputs unchanged. quick=True returns a small slice (see quick_subset) and `quick` sub-folders,
    so a trial never mixes with the real results.
    """
    if not quick:
        return items, Path(results_dir), Path(artifacts_dir)
    return quick_subset(items, per_group), Path(results_dir) / "quick", Path(artifacts_dir) / "quick"


def _saved_config(artifacts_dir: Path | str, model: str, config_cls):
    """The settings saved with the seed 0 model of `model` (the development run), or the defaults if it was not trained."""
    path = Path(artifacts_dir) / model / "seed0" / "model.pt"
    if not path.exists():
        return config_cls()
    import torch
    return config_cls.from_dict(torch.load(path, map_location="cpu")["config"])


def matching_reference_config(artifacts_dir: Path | str, vae_model: str = "url_vae", **overrides) -> SupervisedConfig:
    """Supervised reference settings whose encoder has the sizes of the trained VAE (read from its saved seed 0 model).

    Falls back to the default VAE sizes if no VAE has been trained yet.
    """
    return SupervisedConfig.matching(_saved_config(artifacts_dir, vae_model, VAEConfig), **overrides)


def final_configs(artifacts_dir: Path | str) -> tuple[VAEConfig, SupervisedConfig]:
    """Settings for the final multi-seed runs (6_results.ipynb): those of the development runs, so tuning happens once.

    The VAE settings come from the seed 0 model saved by 4_url_vae.ipynb. The reference takes its training settings from the
    seed 0 model saved by 5_supervised_reference.ipynb and its encoder sizes from the VAE. A model that was not trained yet
    gets the defaults.
    """
    vae = _saved_config(artifacts_dir, "url_vae", VAEConfig)
    ref = _saved_config(artifacts_dir, "url_supervised", SupervisedConfig)
    return vae, SupervisedConfig.matching(vae, epochs=ref.epochs, batch_size=ref.batch_size, lr=ref.lr,
                                          grad_clip=ref.grad_clip, bucket_batches=ref.bucket_batches)


# ----------------------------------------------------------------------------------------------------------------------
# Floors
# ----------------------------------------------------------------------------------------------------------------------
def run_floors(items: pd.DataFrame, results_dir: Path | str, log=print, sources: dict | None = None) -> pd.DataFrame:
    """Fit floor A and floor B on the benign train rows, score validation and test, save and evaluate. Seed 0 only."""
    benign_train = items[_rows(items, "train", "benign")]
    val, test = items[_rows(items, "validation")], items[_rows(items, "test")]
    out = []
    version = data_version(items)
    for model, spec in FLOORS.items():
        floor = fit_floor(benign_train, spec["features"])
        out.append(save_and_evaluate(results_dir, items, model, spec["score"], 0,
                                     validation=(val["item_id"], floor_scores(floor, val)),
                                     test=(test["item_id"], floor_scores(floor, test))))
        _write_record(results_dir, model, [spec["score"]], 0,
                      _record(model, 0, version, len(items), dict(spec), _environment(), sources))
        log(f"{model}: fitted on {floor['n_train']:,} benign train rows, features {spec['features']}")
    return pd.concat(out, ignore_index=True)


# ----------------------------------------------------------------------------------------------------------------------
# URL VAE
# ----------------------------------------------------------------------------------------------------------------------
def run_url_vae(items: pd.DataFrame, results_dir: Path | str, artifacts_dir: Path | str, cfg: VAEConfig | None = None,
                seeds=DEFAULT_SEEDS, device: str = "auto", overwrite: bool = False, encoded: Encoded | None = None,
                model_name: str = "url_vae", log=print, trained: dict | None = None, sources: dict | None = None):
    """Train the URL VAE on the benign train rows for every seed and save both scores of every seed.

    Returns (metrics table, dict seed -> training history). `encoded` can be passed to reuse an `encode_items(items)` result.
    `model_name` changes the results folder, so a variant (another latent size, for example) does not overwrite the main run.
    `trained` maps a seed to a (model, history) that is already trained with exactly these settings, for example
    `Tuning.winners()`: that seed is scored and saved without training again. `sources` is `summary["sources"]`.
    """
    import torch
    cfg = cfg or VAEConfig()
    dev = get_device(device)
    enc = encoded or encode_items(items)
    train = to_tensors(items, enc, _rows(items, "train", "benign"), dev)
    val_benign = to_tensors(items, enc, _rows(items, "validation", "benign"), dev)
    val, test = (to_tensors(items, enc, _rows(items, s), dev) for s in ("validation", "test"))
    log(f"URL VAE on {dev}: {len(train):,} benign train URLs, {len(val_benign):,} benign validation URLs, {cfg.epochs} epochs")
    metrics, histories = [], {}
    version, env = data_version(items), _environment(dev)
    for seed in seeds:
        folder = Path(artifacts_dir) / model_name / f"seed{seed}"
        record = _record(model_name, seed, version, len(items), cfg.to_dict(), env, sources)
        state, why = ("missing", "") if overwrite else _status(results_dir, model_name, SCORE_NAMES, seed, record)
        if state == "done":
            log(f"seed {seed}: score files and run record match, skipped (overwrite=True to rerun)")
            metrics.append(_existing(results_dir, items, model_name, SCORE_NAMES, seed))
            if (folder / "history.csv").exists():
                histories[seed] = pd.read_csv(folder / "history.csv")
            continue
        if state == "stale":
            log(f"seed {seed}: the saved run is out of date ({why}), training again")
        if trained and seed in trained:
            model, history = trained[seed]
            log(f"seed {seed}: using the model trained in the tuning sweep (same settings, no training)")
        else:
            log(f"seed {seed}")
            model, history = train_vae(train, val_benign, cfg, seed, dev, log)
        s_val, s_test = score_vae(model, val, dev, seed), score_vae(model, test, dev, seed)
        for name in SCORE_NAMES:
            metrics.append(save_and_evaluate(results_dir, items, model_name, name, seed,
                                             validation=(val.item_id, s_val[name].to_numpy()),
                                             test=(test.item_id, s_test[name].to_numpy())))
        save_vae(folder, model, history, seed)
        _write_record(results_dir, model_name, SCORE_NAMES, seed, record)      # last, so a record means the seed finished
        histories[seed] = history
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return pd.concat(metrics, ignore_index=True), histories


# ----------------------------------------------------------------------------------------------------------------------
# Tuning sweep
# ----------------------------------------------------------------------------------------------------------------------
TUNE_LATENT, TUNE_BETA = (8, 16, 32), (0.01, 0.1, 1.0)
TUNE_MODEL = "url_vae_tuning"


@dataclass
class Tuning:
    """The result of `run_tuning`: one row per candidate, the winner, and where its model is saved."""
    table: pd.DataFrame           # one row per candidate with its best-epoch benign validation numbers; `selected` marks the winner
    best: str                     # name of the winning candidate
    best_config: VAEConfig
    folder: Path                  # artifacts folder of the winner (model.pt, history.csv)
    seed: int

    def winners(self, device: str = "auto") -> dict:
        """{seed: (model, history)} of the winner, to pass as `trained=` to `run_url_vae` so that seed is not trained twice."""
        from src.url_vae import load_vae
        return {self.seed: (load_vae(self.folder, get_device(device)), pd.read_csv(self.folder / "history.csv"))}


def _candidates(base: VAEConfig, latent, beta) -> list[tuple[str, VAEConfig]]:
    """The base settings, then one change at a time: each latent size and each beta that differs from the base."""
    out = [("base", base)]
    for v in latent:
        if v != base.latent:
            out.append((f"latent{v}", VAEConfig(**{**base.to_dict(), "latent": v})))
    for b in beta:
        if b != base.beta:
            out.append((f"beta{b:g}", VAEConfig(**{**base.to_dict(), "beta": b})))
    return out


def run_tuning(items: pd.DataFrame, artifacts_dir: Path | str, base_cfg: VAEConfig | None = None, latent=TUNE_LATENT,
               beta=TUNE_BETA, seed: int = 0, device: str = "auto", overwrite: bool = False, encoded: Encoded | None = None,
               model_name: str = TUNE_MODEL, log=print) -> Tuning:
    """Train the VAE once per candidate on seed `seed` and keep the one with the lowest benign validation negative ELBO.

    One change at a time from `base_cfg`: each latent size in `latent` and each beta in `beta`. The choice uses benign train
    and benign validation rows only: no label, no malicious row and no test row, and no scores are saved for the test set.
    The negative ELBO is the true one (KL weight 1) whatever beta the candidate trained with, so candidates are comparable.
    It also means the criterion favors beta = 1 by construction (that is the objective beta = 1 optimizes), so the beta part of
    the sweep can only show that a smaller beta is not needed for density, never that it helps detection.
    Candidates are saved under `<artifacts_dir>/<model_name>/<name>/` with a run record, and a candidate that is already
    saved with the same settings and data is not trained again. Pass `Tuning.winners()` to `run_url_vae` to reuse the winner.

    It is a one-seed, one-at-a-time sweep with one noisy validation draw per epoch: differences between candidates that
    are small compared with the seed-to-seed spread are not evidence that one is better.
    """
    import torch
    base = base_cfg or VAEConfig()
    dev = get_device(device)
    enc = encoded or encode_items(items)
    train = to_tensors(items, enc, _rows(items, "train", "benign"), dev)
    val_benign = to_tensors(items, enc, _rows(items, "validation", "benign"), dev)
    version, env = data_version(items), _environment(dev)
    rows = []
    for name, cfg in _candidates(base, latent, beta):
        folder = Path(artifacts_dir) / model_name / name
        record = _record(model_name, seed, version, len(items), cfg.to_dict(), env)
        saved = folder / RECORD_FILE
        if not overwrite and (folder / "model.pt").exists() and saved.exists() and not _changes(json.loads(saved.read_text()), record):
            log(f"{name}: already trained with these settings, skipped")
            history = pd.read_csv(folder / "history.csv")
        else:
            log(f"{name}: latent {cfg.latent}, beta {cfg.beta}")
            model, history = train_vae(train, val_benign, cfg, seed, dev, log)
            save_vae(folder, model, history, seed)
            saved.write_text(json.dumps(record, indent=2))
            del model
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        best = history.loc[history["best"]].iloc[0]
        rows.append({"candidate": name, "latent": cfg.latent, "beta": cfg.beta, "best_epoch": int(best["epoch"]),
                     "val_neg_elbo_per_char": float(best["val_neg_elbo_per_char"]),
                     "val_nll_per_char": float(best["val_nll_per_char"]), "val_kl": float(best["val_kl"]),
                     "minutes": float(history["seconds"].sum()) / 60})
    table = pd.DataFrame(rows)
    winner = table.loc[table["val_neg_elbo_per_char"].idxmin(), "candidate"]
    table["selected"] = table["candidate"] == winner
    configs = dict(_candidates(base, latent, beta))
    table.to_csv(Path(artifacts_dir) / model_name / "tuning.csv", index=False)
    return Tuning(table=table, best=winner, best_config=configs[winner], folder=Path(artifacts_dir) / model_name / winner, seed=seed)


# ----------------------------------------------------------------------------------------------------------------------
# Supervised reference
# ----------------------------------------------------------------------------------------------------------------------
def run_supervised(items: pd.DataFrame, results_dir: Path | str, artifacts_dir: Path | str,
                   cfg: SupervisedConfig | None = None, seeds=DEFAULT_SEEDS, device: str = "auto",
                   overwrite: bool = False, encoded: Encoded | None = None, model_name: str = "url_supervised",
                   log=print, sources: dict | None = None):
    """Train the supervised reference on the benign and malicious train rows for every seed and save its scores.

    Returns (metrics table, dict seed -> training history). Pass `cfg=SupervisedConfig.matching(vae_cfg)` so the encoder
    has the same sizes as the VAE's.
    """
    import torch
    cfg = cfg or SupervisedConfig()
    dev = get_device(device)
    enc = encoded or encode_items(items)
    train = to_tensors(items, enc, _rows(items, "train"), dev)
    val, test = (to_tensors(items, enc, _rows(items, s), dev) for s in ("validation", "test"))
    log(f"Supervised reference on {dev}: {len(train):,} train URLs, {len(val):,} validation URLs, {cfg.epochs} epochs")
    metrics, histories = [], {}
    version, env = data_version(items), _environment(dev)
    for seed in seeds:
        folder = Path(artifacts_dir) / model_name / f"seed{seed}"
        record = _record(model_name, seed, version, len(items), cfg.to_dict(), env, sources)
        state, why = ("missing", "") if overwrite else _status(results_dir, model_name, ["prob_malicious"], seed, record)
        if state == "done":
            log(f"seed {seed}: score files and run record match, skipped (overwrite=True to rerun)")
            metrics.append(_existing(results_dir, items, model_name, ["prob_malicious"], seed))
            if (folder / "history.csv").exists():
                histories[seed] = pd.read_csv(folder / "history.csv")
            continue
        if state == "stale":
            log(f"seed {seed}: the saved run is out of date ({why}), training again")
        log(f"seed {seed}")
        model, history = train_classifier(train, val, cfg, seed, dev, log)
        metrics.append(save_and_evaluate(results_dir, items, model_name, "prob_malicious", seed,
                                         validation=(val.item_id, score_classifier(model, val, dev)),
                                         test=(test.item_id, score_classifier(model, test, dev))))
        save_classifier(folder, model, history, seed)
        _write_record(results_dir, model_name, ["prob_malicious"], seed, record)   # last, so a record means the seed finished
        histories[seed] = history
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return pd.concat(metrics, ignore_index=True), histories


# ----------------------------------------------------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Run floors, the VAE tuning sweep, the URL VAE or the supervised reference.")
    p.add_argument("model", choices=["floors", "tune", "url_vae", "url_supervised"])
    p.add_argument("--root", type=Path, default=Path.cwd(), help="project root (the folder that contains src and data)")
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--epochs", type=int, default=None, help="override the default number of epochs")
    p.add_argument("--device", default="auto")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    items = load_items(a.root / "data" / "processed")
    results, artifacts = a.root / "results", a.root / "data" / "artifacts"
    if a.model == "floors":
        out = run_floors(items, results)
    elif a.model == "tune":
        out = run_tuning(items, artifacts, VAEConfig(**({"epochs": a.epochs} if a.epochs else {})), device=a.device,
                         overwrite=a.overwrite).table
    elif a.model == "url_vae":
        cfg = VAEConfig(**({"epochs": a.epochs} if a.epochs else {}))
        out, _ = run_url_vae(items, results, artifacts, cfg, a.seeds, a.device, a.overwrite)
    else:
        cfg = SupervisedConfig.matching(VAEConfig(), **({"epochs": a.epochs} if a.epochs else {}))
        out, _ = run_supervised(items, results, artifacts, cfg, a.seeds, a.device, a.overwrite)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
