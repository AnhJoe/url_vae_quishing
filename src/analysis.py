"""The evaluation layer on top of src/evaluate.py: confidence intervals, strata, paired comparisons, seeds, success criteria.

src/evaluate.py turns one score file into point metrics. This module answers the questions the report needs, from the same
score files (results/<model>/<score>/seed<k>/), and never recomputes a model:

    seed_table()           one row of metrics per run, and `seed_summary()` the mean, spread and range over seeds
    metrics_ci()           AUROC, AUPRC and recall at 1% / 5% FPR with 95% confidence intervals
    compare()              paired AUROC difference between a model and a reference (a floor), overall and per seed
    strata()               AUROC inside each stratum (URL-length bin, scheme) and the paired difference there
    criteria()             the agreed success criteria, fixed on validation before the test set is touched
    choose_score()         which of a model's scores is the headline one, chosen on validation AUROC

How the uncertainty is computed (standard cluster bootstrap)
    Rows that share a domain are not independent (one site has many URLs), so the bootstrap resamples whole DOMAINS of the
    test split with replacement, `n_boot` times (the `domain` column of the items table). In each replicate every row gets
    the weight "how many times its domain was drawn", and each metric is recomputed with those weights. The 95% interval
    is the 2.5th and 97.5th percentile over the replicates. A paired difference uses the SAME replicate for both scores, so
    it measures whether one score beats the other on the same resampled data. The alarm thresholds stay fixed at the
    validation values (they are chosen before the test set), so the interval covers test-set sampling only.
    Seeds: for a model with several seeds, the headline number is the mean over seeds computed inside every replicate, so
    the interval is about the test data. The spread across seeds (min and max) is reported next to it, never mixed in.

Strata and the rule for "enough items"
    A stratum gets an AUROC only if it has at least `min_per_class` benign AND malicious test items (default 100, our
    choice for a stable AUROC) from at least `min_domains` different domains in each class (default 10). The domain rule
    matters because the intervals resample whole domains, so a stratum of many rows from a few domains has an unstable
    interval. The others are listed with their counts and no AUROC. The IP-host group is reported as
    counts only (`ip_host_table`) because almost no benign URL has an IP host.

Typical use (see 6_results.ipynb):

    an = Analysis(RESULTS_DIR, items)
    score = an.choose_score("url_vae")
    an.metrics_ci(), an.compare("url_vae", score, "floorA_url", "abs_z"), an.strata("url_vae", score), an.criteria("url_vae")
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import FPR_LEVELS, evaluate_run, load_scores, run_dir, threshold_at_fpr

N_BOOT = 1000
MIN_PER_CLASS = 100
MIN_DOMAINS = 10
STRATA_COLUMNS = ("url_len_bin", "scheme")
FLOOR_A = ("floorA_url", "abs_z")
FLOOR_B = ("floorB_url", "mahalanobis")
REFERENCE = ("url_supervised", "prob_malicious")
VAE_SCORES = ("nll_per_char", "neg_elbo_per_char")


# ----------------------------------------------------------------------------------------------------------------------
# Weighted metrics. With every weight equal to 1 they equal the functions of src/evaluate.py (checked in checks.py).
# ----------------------------------------------------------------------------------------------------------------------
class ScoreView:
    """One score on one set of rows, prepared once so a metric with new row weights costs one pass over the rows."""

    def __init__(self, y, score):
        self.y = np.asarray(y).astype(bool)
        self.score = np.asarray(score, dtype=float)
        self.uniq, self.inv = np.unique(self.score, return_inverse=True)  # distinct scores, and the group of every row
        self.k = len(self.uniq)
        self.pos = self.y.astype(float)
        self.neg = 1.0 - self.pos

    def groups(self, w):
        """Weighted malicious and benign counts of every distinct score value, in ascending score order."""
        return (np.bincount(self.inv, weights=w * self.pos, minlength=self.k),
                np.bincount(self.inv, weights=w * self.neg, minlength=self.k))

    def cut(self, threshold: float) -> int:
        """Index of the first score group strictly above the threshold (groups from there on are flagged)."""
        return int(np.searchsorted(self.uniq, threshold, side="right"))

    @staticmethod
    def auroc_of(wp, wn) -> float:
        """Weighted probability that a malicious row scores above a benign row (ties count half)."""
        tp, tn = wp.sum(), wn.sum()
        if tp == 0 or tn == 0:
            return float("nan")
        below = np.cumsum(wn) - wn
        return float((wp * (below + 0.5 * wn)).sum() / (tp * tn))

    @staticmethod
    def auprc_of(wp, wn) -> float:
        """Weighted average precision (one cut per distinct score)."""
        wp, wn = wp[::-1], wn[::-1]                                       # from the highest score down
        total_p = wp.sum()
        if total_p == 0:
            return float("nan")
        tp = np.cumsum(wp)
        seen = tp + np.cumsum(wn)
        precision = np.divide(tp, seen, out=np.zeros_like(tp), where=seen > 0)
        return float((wp / total_p * precision).sum())

    @staticmethod
    def recall_of(wp, cut: int) -> float:
        """Weighted share of malicious rows in the groups from `cut` on."""
        total = wp.sum()
        return float(wp[cut:].sum() / total) if total > 0 else float("nan")

    def auroc(self, w) -> float:
        return self.auroc_of(*self.groups(w))

    def auprc(self, w) -> float:
        return self.auprc_of(*self.groups(w))

    def recall_at(self, threshold: float, w) -> float:
        """Weighted share of malicious rows with a score strictly above the threshold."""
        return self.recall_of(self.groups(w)[0], self.cut(threshold))

    def subset(self, mask) -> "ScoreView":
        return ScoreView(self.y[mask], self.score[mask])


def _bootstrap(groups, fn, n_cols: int, n_boot: int, seed: int) -> np.ndarray:
    """Core of the domain bootstrap: fn(weights) returns n_cols numbers, evaluated on `n_boot` resamples of the domains."""
    codes, uniq = pd.factorize(pd.Series(np.asarray(groups)))
    n_dom = len(uniq)
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, n_cols))
    for b in range(n_boot):
        counts = np.bincount(rng.integers(0, n_dom, n_dom), minlength=n_dom)
        out[b] = fn(counts[codes].astype(float))
    return out


def domain_bootstrap(groups, stats: dict, n_boot: int = N_BOOT, seed: int = 0) -> pd.DataFrame:
    """Resample whole domains `n_boot` times and evaluate every function in `stats` on each replicate.

    `groups` is the domain of every row. `stats` maps a name to a function of the row weights (an array with one weight per
    row) that returns a number. Returns a DataFrame with one row per replicate and one column per name. The same seed gives
    the same replicates, so different calls are paired.
    """
    fns = list(stats.values())
    out = _bootstrap(groups, lambda w: [f(w) for f in fns], len(fns), n_boot, seed)
    return pd.DataFrame(out, columns=list(stats))


def percentile_ci(values, level: float = 0.95) -> tuple[float, float]:
    """Percentile interval of the bootstrap replicates (NaN replicates are ignored)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan")
    a = (1 - level) / 2
    return float(np.quantile(v, a)), float(np.quantile(v, 1 - a))


# ----------------------------------------------------------------------------------------------------------------------
# The analysis object
# ----------------------------------------------------------------------------------------------------------------------
def list_runs(results_dir: Path | str) -> pd.DataFrame:
    """Every run folder found under results_dir (model, score, seed), with both score files present."""
    rows = []
    for folder in sorted(Path(results_dir).glob("*/*/seed*")):
        if (folder / "validation.csv.gz").exists() and (folder / "test.csv.gz").exists():
            rows.append({"model": folder.parts[-3], "score": folder.parts[-2], "seed": int(folder.name.removeprefix("seed"))})
    return pd.DataFrame(rows, columns=["model", "score", "seed"])


class Analysis:
    """Reads the saved scores of any run and answers the questions of the report (see the top of this file)."""

    def __init__(self, results_dir: Path | str, items: pd.DataFrame, n_boot: int = N_BOOT, boot_seed: int = 0,
                 min_per_class: int = MIN_PER_CLASS, min_domains: int = MIN_DOMAINS):
        self.results_dir, self.items = Path(results_dir), items
        self.n_boot, self.boot_seed, self.min_per_class = n_boot, boot_seed, min_per_class
        self.min_domains = min_domains
        self.test = items[items["split"] == "test"].reset_index(drop=True)
        self.y = self.test["label"].to_numpy()
        self._views: dict = {}
        self._thr: dict = {}
        self._per_run: pd.DataFrame | None = None
        self._cache: dict = {}

    # finding and loading runs
    def seeds(self, model: str, score: str) -> list[int]:
        runs = list_runs(self.results_dir)
        s = runs[(runs["model"] == model) & (runs["score"] == score)]["seed"].tolist()
        if not s:
            raise FileNotFoundError(f"no runs of {model}/{score} under {self.results_dir}")
        return sorted(s)

    def view(self, model: str, score: str, seed: int) -> ScoreView:
        """The test scores of one run as a ScoreView, aligned with `self.test`. Also stores its validation thresholds."""
        key = (model, score, seed)
        if key not in self._views:
            folder = run_dir(self.results_dir, model, score, seed)
            test = load_scores(folder / "test.csv.gz", self.items, "test")
            val = load_scores(folder / "validation.csv.gz", self.items, "validation")
            self._views[key] = ScoreView(self.y, test["score"])
            benign = val.loc[val["label"] == 0, "score"]
            self._thr[key] = {f: threshold_at_fpr(benign, f) for f in FPR_LEVELS}
        return self._views[key]

    def thresholds(self, model: str, score: str, seed: int) -> dict:
        self.view(model, score, seed)
        return self._thr[(model, score, seed)]

    # point metrics for every run
    def seed_table(self) -> pd.DataFrame:
        """One row of point metrics (src/evaluate.py evaluate_run) for every run found under results_dir."""
        if self._per_run is None:
            runs = list_runs(self.results_dir)
            rows = [evaluate_run(run_dir(self.results_dir, r.model, r.score, r.seed), self.items) for r in runs.itertuples()]
            self._per_run = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        return self._per_run

    def seed_summary(self) -> pd.DataFrame:
        """Per model and score: number of seeds, mean and standard deviation of every metric, and the AUROC range."""
        t = self.seed_table()
        cols = ["auroc", "auprc", "recall_at_1pct_fpr", "recall_at_5pct_fpr", "validation_auroc"]
        g = t.groupby(["model", "score"])
        out = g[cols].mean()
        out.insert(0, "n_seeds", g.size())
        for c in ("auroc", "auprc"):
            out[f"{c}_sd"] = g[c].std(ddof=1)
        out["auroc_min"], out["auroc_max"] = g["auroc"].min(), g["auroc"].max()
        return out.reset_index()

    def choose_score(self, model: str, scores=VAE_SCORES) -> str:
        """The headline score of a model: the one with the higher mean VALIDATION AUROC over its seeds (fixed before testing)."""
        t = self.seed_table()
        t = t[(t["model"] == model) & (t["score"].isin(scores))]
        if t.empty:
            raise FileNotFoundError(f"no runs of {model} with scores {list(scores)}")
        return str(t.groupby("score")["validation_auroc"].mean().idxmax())

    # metrics with confidence intervals
    def _seed_metrics(self, model, score, seeds):
        """A function of the row weights that returns [AUROC, AUPRC, recall at 1% FPR, recall at 5% FPR] averaged over seeds."""
        views = [self.view(model, score, s) for s in seeds]
        cuts = [[v.cut(self._thr[(model, score, s)][f]) for f in FPR_LEVELS] for v, s in zip(views, seeds)]

        def fn(w):
            rows = []
            for v, c in zip(views, cuts):
                wp, wn = v.groups(w)
                rows.append([v.auroc_of(wp, wn), v.auprc_of(wp, wn)] + [v.recall_of(wp, k) for k in c])
            return np.mean(rows, axis=0)
        return fn

    def metrics_ci(self, runs=None) -> pd.DataFrame:
        """Test AUROC, AUPRC and recall at 1% and 5% FPR (mean over seeds) with domain-bootstrap 95% intervals.

        `runs` is a list of (model, score) pairs; the default is every model and score found. Columns: value, ci_low and
        ci_high for each metric, the number of seeds, and the lowest and highest seed AUROC.
        """
        if runs is None:
            runs = list_runs(self.results_dir)[["model", "score"]].drop_duplicates().itertuples(index=False, name=None)
        names = ["auroc", "auprc", "recall_at_1pct_fpr", "recall_at_5pct_fpr"]
        one = np.ones(len(self.test))
        rows = []
        for model, score in runs:
            seeds = self.seeds(model, score)
            fn = self._seed_metrics(model, score, seeds)
            point = fn(one)
            boot = pd.DataFrame(_bootstrap(self.test["domain"], fn, len(names), self.n_boot, self.boot_seed), columns=names)
            row = {"model": model, "score": score, "n_seeds": len(seeds)}
            for n, p in zip(names, point):
                lo, hi = percentile_ci(boot[n])
                row.update({n: p, f"{n}_lo": lo, f"{n}_hi": hi})
            seed_auc = [self.view(model, score, s).auroc(one) for s in seeds]
            row.update({"auroc_seed_min": min(seed_auc), "auroc_seed_max": max(seed_auc)})
            rows.append(row)
        return pd.DataFrame(rows)

    # paired comparison
    def compare(self, model: str, score: str, ref_model: str = FLOOR_A[0], ref_score: str = FLOOR_A[1],
                ref_seed: int = 0) -> pd.DataFrame:
        """AUROC of `model` minus AUROC of a reference, on the same resampled domains.

        Rows: "mean over seeds" (the headline) and one row per seed. `ci_low` above 0 means the model beats the reference.
        """
        key = ("compare", model, score, ref_model, ref_score, ref_seed)
        if key in self._cache:
            return self._cache[key].copy()
        seeds = self.seeds(model, score)
        ref = self.view(ref_model, ref_score, ref_seed)
        views = [self.view(model, score, s) for s in seeds]
        names = ["mean over seeds"] + [f"seed {s}" for s in seeds]

        def aucs(w):
            r = ref.auroc(w)
            a = np.array([v.auroc(w) for v in views])
            return np.r_[a.mean(), a], r

        def fn(w):
            a, r = aucs(w)
            return a - r

        one = np.ones(len(self.test))
        point, r0 = aucs(one)
        boot = _bootstrap(self.test["domain"], fn, len(names), self.n_boot, self.boot_seed)
        rows = []
        for i, name in enumerate(names):
            lo, hi = percentile_ci(boot[:, i])
            rows.append({"run": name, "model_auroc": point[i], "reference_auroc": r0, "difference": point[i] - r0,
                         "ci_low": lo, "ci_high": hi, "beats_reference": bool(lo > 0)})
        self._cache[key] = pd.DataFrame(rows)
        return self._cache[key].copy()

    # strata
    def strata_counts(self, columns=STRATA_COLUMNS) -> pd.DataFrame:
        """Test items and domains per stratum and class, and whether the stratum has enough of both for an AUROC."""
        rows = []
        for col in columns:
            for value, g in self.test.groupby(col):
                nb, nm = int((g["label"] == 0).sum()), int((g["label"] == 1).sum())
                db, dm = int(g.loc[g["label"] == 0, "domain"].nunique()), int(g.loc[g["label"] == 1, "domain"].nunique())
                rows.append({"stratum_column": col, "stratum": value, "n_benign": nb, "n_malicious": nm,
                             "domains_benign": db, "domains_malicious": dm,
                             "eligible": min(nb, nm) >= self.min_per_class and min(db, dm) >= self.min_domains})
        return pd.DataFrame(rows)

    def strata(self, model: str, score: str, ref_model: str = FLOOR_A[0], ref_score: str = FLOOR_A[1],
               ref_seed: int = 0, columns=STRATA_COLUMNS) -> pd.DataFrame:
        """AUROC of the model (mean over seeds) and of the reference inside every stratum, and their paired difference.

        Strata without enough items of both classes keep their counts and have no AUROC (`eligible` is False).
        """
        ckey = ("strata", model, score, ref_model, ref_score, ref_seed, tuple(columns))
        if ckey in self._cache:
            return self._cache[ckey].copy()
        seeds = self.seeds(model, score)
        views = [self.view(model, score, s) for s in seeds]
        ref = self.view(ref_model, ref_score, ref_seed)
        table = self.strata_counts(columns)
        masks, stats = {}, {}
        for r in table[table["eligible"]].itertuples():
            m = (self.test[r.stratum_column] == r.stratum).to_numpy()
            key = (r.stratum_column, r.stratum)
            masks[key] = m
            sv, sr = [v.subset(m) for v in views], ref.subset(m)
            stats[key] = (lambda m, sv, sr: lambda w: float(np.mean([v.auroc(w[m]) for v in sv]) - sr.auroc(w[m])))(m, sv, sr)
        boot = domain_bootstrap(self.test["domain"], stats, self.n_boot, self.boot_seed) if stats else None
        one = np.ones(len(self.test))
        out = []
        for r in table.itertuples():
            row = r._asdict()
            row.pop("Index")
            key = (r.stratum_column, r.stratum)
            if key in masks:
                m = masks[key]
                auc_m = float(np.mean([v.subset(m).auroc(one[m]) for v in views]))
                auc_r = ref.subset(m).auroc(one[m])
                lo, hi = percentile_ci(boot[key])
                row.update({"model_auroc": auc_m, "reference_auroc": auc_r, "difference": auc_m - auc_r,
                            "ci_low": lo, "ci_high": hi, "beats_reference": bool(lo > 0)})
            else:
                row.update({"model_auroc": np.nan, "reference_auroc": np.nan, "difference": np.nan,
                            "ci_low": np.nan, "ci_high": np.nan, "beats_reference": np.nan})
            out.append(row)
        self._cache[ckey] = pd.DataFrame(out)
        return self._cache[ckey].copy()

    def ip_host_table(self) -> pd.DataFrame:
        """Test items with an IP-address host, per class. Reported as counts and shares only (no AUROC is computed)."""
        t = self.test.assign(ip=self.test["host_is_ip"])
        g = t.groupby("cls")["ip"].agg(["sum", "size"]).rename(columns={"sum": "ip_host_items", "size": "items"})
        g["share_of_class"] = g["ip_host_items"] / g["items"]
        return g.reset_index()

    # success criteria
    def criteria(self, model: str = "url_vae", score: str | None = None, floor=FLOOR_A, floor_b=FLOOR_B,
                 reference=REFERENCE) -> pd.DataFrame:
        """The success criteria of the proposal for one model score, each with its numbers and pass or FAIL.

        Beats its floor            paired AUROC difference vs floor A, 95% interval above 0 (mean over seeds)
        Holds within strata        the same in a majority of the strata that have enough items
        Seed robustness            the lowest seed still beats floor A (its own interval above 0); only for a model with
                                   more than one seed (with one seed the row is left out, because no spread can be measured)
        Floor B alongside          the same comparison with floor B (information only)
        Gap to the reference       share of the floor-to-reference AUROC gap the model closes (descriptive, no threshold)
        Not meeting a criterion is a finding, not a failure of the project.
        """
        score = score or self.choose_score(model)
        rows = []
        cmp_a = self.compare(model, score, *floor)
        head = cmp_a.iloc[0]
        rows.append({"criterion": "Beats its floor (floor A)", "result": "pass" if head["beats_reference"] else "FAIL",
                     "value": head["difference"],
                     "detail": f"AUROC {head['model_auroc']:.3f} vs {head['reference_auroc']:.3f}, difference "
                               f"{head['difference']:+.3f}, 95% CI [{head['ci_low']:+.3f}, {head['ci_high']:+.3f}]"})
        st = self.strata(model, score, *floor)
        elig = st[st["eligible"]]
        wins = int(elig["beats_reference"].astype(bool).sum())
        rows.append({"criterion": "Holds within strata", "result": "pass" if len(elig) and wins * 2 > len(elig) else "FAIL",
                     "value": wins / len(elig) if len(elig) else np.nan,
                     "detail": f"beats floor A in {wins} of {len(elig)} strata with at least {self.min_per_class} items and "
                               f"{self.min_domains} domains of each class ({len(st) - len(elig)} strata too small, listed in strata())"})
        per_seed = cmp_a.iloc[1:]
        if len(per_seed) > 1:
            low = per_seed.loc[per_seed["ci_low"].idxmin()]
            rows.append({"criterion": "Seed robustness", "result": "pass" if bool((per_seed["ci_low"] > 0).all()) else "FAIL",
                         "value": float(per_seed["ci_low"].min()),
                         "detail": f"{len(per_seed)} seeds; the weakest ({low['run']}) has difference {low['difference']:+.3f}, "
                                   f"95% CI low {low['ci_low']:+.3f}"})
        head_b = self.compare(model, score, *floor_b).iloc[0]
        rows.append({"criterion": "Floor B alongside (information)", "result": "info", "value": head_b["difference"],
                     "detail": f"AUROC {head_b['model_auroc']:.3f} vs {head_b['reference_auroc']:.3f}, difference "
                               f"{head_b['difference']:+.3f}, 95% CI [{head_b['ci_low']:+.3f}, {head_b['ci_high']:+.3f}]"})
        rows.append(self._gap_row(model, score, floor, reference))
        return pd.DataFrame(rows)

    def _gap_row(self, model, score, floor, reference) -> dict:
        """(model - floor) / (reference - floor), with a domain-bootstrap interval. The reference is a different model."""
        name = "Gap to the supervised reference closed (descriptive)"
        try:
            ref_seeds = self.seeds(*reference)
        except FileNotFoundError:
            return {"criterion": name, "result": "info", "value": np.nan, "detail": "no supervised reference runs found yet"}
        seeds = self.seeds(model, score)
        mv = [self.view(model, score, s) for s in seeds]
        rv = [self.view(*reference, s) for s in ref_seeds]
        fv = self.view(*floor, 0)

        def stat(w):
            f = fv.auroc(w)
            gap = np.mean([v.auroc(w) for v in rv]) - f
            return float((np.mean([v.auroc(w) for v in mv]) - f) / gap) if abs(gap) > 1e-9 else float("nan")

        one = np.ones(len(self.test))
        boot = domain_bootstrap(self.test["domain"], {"gap": stat}, self.n_boot, self.boot_seed)["gap"]
        lo, hi = percentile_ci(boot)
        f, m, r = fv.auroc(one), np.mean([v.auroc(one) for v in mv]), np.mean([v.auroc(one) for v in rv])
        return {"criterion": name, "result": "info", "value": stat(one),
                "detail": f"floor {f:.3f}, model {m:.3f}, reference {r:.3f}; share of the gap closed {stat(one):.2f}, "
                          f"95% CI [{lo:.2f}, {hi:.2f}]"}
