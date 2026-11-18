"""Checks that a model notebook runs before it trains anything.

Each function returns a `Checks` object. Call `.report()` on it in a notebook cell: it shows a table of named checks
and raises an error if any of them failed, so a notebook never trains on data that does not match what it assumes.

    data_checks(items, summary)                the items table is the one 2_data_processing builds (counts, splits, labels,
                                               URL columns)
    url_input_checks(items)                    encode_urls gives the ids the model expects (padding, cut-off, unknown token)
    evaluation_checks(items)                   made-up scores go through save_and_evaluate and give the metrics they must
    model_checks(items)                        the URL VAE and the supervised reference behave as designed on a tiny model
                                               (shapes, padding and causality, KL, loss at initialization, learning, scores)
    analysis_checks(items)                     the weighted metrics, the domain bootstrap, the paired comparison and the
                                               strata rule give the answers they must on made-up scores

What the checks cover and what they do not: they test the shared pipeline (files, splits, input arrays, metrics) and the
code of the models and of the analysis, never how good a trained model is. They read a few thousand URLs and use tiny
networks on the CPU, so they run in seconds to a minute.

Typical use in a model notebook:

    items = load_items(PROC_DIR)
    summary = json.loads((PROC_DIR / "items_summary.json").read_text())
    checks.data_checks(items, summary).report()
"""
from __future__ import annotations

import json
import tempfile

import numpy as np
import pandas as pd

from src.evaluate import auprc, auroc, save_and_evaluate
from src.floors import FLOORS
from src.loaders import MAX_URL_LEN, VOCAB_SIZE, decode_url, encode_urls
from src.prepare_data import data_version
from src.splits import SPLIT_NAMES

URL_FLOOR_FEATURES = FLOORS["floorB_url"]["features"]            # floor B uses every floor feature


# ----------------------------------------------------------------------------------------------------------------------
# A list of named checks
# ----------------------------------------------------------------------------------------------------------------------
class Checks:
    """A list of named pass/fail checks (plus information rows). `report()` shows the table and raises if any failed."""

    def __init__(self, title: str):
        self.title = title
        self.rows: list[dict] = []

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.rows.append({"check": name, "result": "pass" if bool(passed) else "FAIL", "detail": str(detail)})

    def info(self, name: str, detail: str) -> None:
        self.rows.append({"check": name, "result": "info", "detail": str(detail)})

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=["check", "result", "detail"])

    def report(self) -> None:
        t = self.table()
        _show(t, self.title)
        failed = t.loc[t["result"] == "FAIL", "check"].tolist()
        if failed:
            raise AssertionError(f"{len(failed)} check(s) failed: " + "; ".join(failed))


def _show(t: pd.DataFrame, title: str) -> None:
    """Display the table in a notebook (wrapped text, failures in red). Falls back to plain text elsewhere."""
    try:
        from IPython.display import display
        styled = (t.style.hide(axis="index").set_caption(title)
                  .set_table_styles([
                      {"selector": "caption", "props": [("caption-side", "top"), ("text-align", "left"), ("font-weight", "bold"), ("padding", "4px 0 8px 0")]},
                      {"selector": "th, td", "props": [("white-space", "normal"), ("overflow-wrap", "anywhere"), ("max-width", "520px"), ("text-align", "left"), ("vertical-align", "top")]},
                  ])
                  .map(lambda v: "color: #b00020; font-weight: bold" if v == "FAIL" else "", subset=["result"]))
        display(styled)
    except Exception:
        print(title)
        print(t.to_string(index=False))


def _refused(fn) -> bool:
    """True if calling fn raises ValueError (the evaluation functions refuse malformed scores with ValueError)."""
    try:
        fn()
    except ValueError:
        return True
    return False


# ----------------------------------------------------------------------------------------------------------------------
# The items table
# ----------------------------------------------------------------------------------------------------------------------
def data_checks(items: pd.DataFrame, summary: dict) -> Checks:
    """Check the whole items table (load_items with no filter) against items_summary.json and against itself."""
    c = Checks("Data checks")
    t = items
    c.add("row count matches items_summary.json", len(t) == summary["rows"], f"{len(t):,} rows")
    version = data_version(t)
    c.add("data version matches items_summary.json", version == summary["data_version"],
          f"{version}. Everyone on the team should see the same value")
    c.add("item_id equals the row position", bool((t["item_id"].to_numpy() == np.arange(len(t))).all()))
    c.add("label matches the class folder", bool((t["label"] == (t["cls"] == "malicious").astype(int)).all()))

    counts = {s: {k: int(v) for k, v in g["cls"].value_counts().items()} for s, g in t.groupby("split")}
    c.add("rows per split and class match items_summary.json", counts == summary["rows_by_split_and_class"],
          "; ".join(f"{s}: {counts[s].get('benign', 0):,} benign, {counts[s].get('malicious', 0):,} malicious" for s in SPLIT_NAMES))
    c.add("every split has benign and malicious rows", all(counts.get(s, {}).get(k, 0) > 0 for s in SPLIT_NAMES for k in ("benign", "malicious")))
    split_domains = t.groupby(["cls", "domain"])["split"].nunique()
    c.add("no domain appears in two splits of the same class", int(split_domains.max()) == 1)
    share = t.groupby("split")["label"].mean()
    overall = t["label"].mean()
    c.add("malicious share of every split is within 1 point of the overall share", bool((share - overall).abs().max() < 0.01),
          ", ".join(f"{s} {share[s]:.3f}" for s in SPLIT_NAMES) + f" (overall {overall:.3f})")
    c.add("no empty URL, and url_len equals the URL length", bool((t["url"].str.len() > 0).all() and (t["url_len"] == t["url"].str.len()).all()))
    c.add(f"truncated is true exactly for URLs longer than {MAX_URL_LEN}", bool((t["truncated"] == (t["url_len"] > MAX_URL_LEN)).all()),
          f"{int(t['truncated'].sum()):,} truncated")

    missing = int(t[URL_FLOOR_FEATURES].isna().sum().sum())
    c.add("URL floor features have no missing values", missing == 0, ", ".join(URL_FLOOR_FEATURES))
    return c


# ----------------------------------------------------------------------------------------------------------------------
# Model input
# ----------------------------------------------------------------------------------------------------------------------
def url_input_checks(items: pd.DataFrame, n: int = 3000, seed: int = 0) -> Checks:
    """Encode a few thousand URLs with encode_urls and check the ids against the URLs they came from.

    Also checks URLs with characters outside printable ASCII (the unknown token) and URLs longer than the maximum length.
    """
    c = Checks("URL input checks")
    sample = items.sample(n, random_state=seed)
    urls = sample["url"].tolist()
    ids = encode_urls(urls)
    shown = lambda u: "".join(ch if 32 <= ord(ch) < 127 else "?" for ch in u[:MAX_URL_LEN])     # what decode_url must return

    c.add(f"array shape is (n, {MAX_URL_LEN}) and the type is int16", ids.shape == (n, MAX_URL_LEN) and ids.dtype == np.int16, f"{ids.shape}, {ids.dtype}")
    c.add(f"ids are between 0 and {VOCAB_SIZE - 1} (vocabulary size {VOCAB_SIZE})", int(ids.min()) >= 0 and int(ids.max()) < VOCAB_SIZE, f"smallest {ids.min()}, largest {ids.max()}")
    nonpad = ids != 0
    c.add("padding (id 0) only comes after the last character", bool((np.diff(nonpad.astype(int), axis=1) <= 0).all()))
    c.add(f"each row has min(URL length, {MAX_URL_LEN}) real ids", bool((nonpad.sum(axis=1) == np.minimum(sample["url_len"].to_numpy(), MAX_URL_LEN)).all()))
    c.add("decode_url(encode_urls(url)) gives back the URL (cut at the maximum length, unknown characters as ?)",
          all(decode_url(row) == shown(u) for row, u in zip(ids, urls)))

    odd = items.loc[~items["url"].str.fullmatch(r"[ -~]*")].head(20)
    odd_ids = encode_urls(odd["url"])
    c.add("a URL with a character outside printable ASCII gets the unknown token (id 1)", len(odd) > 0 and bool(((odd_ids == 1).sum(axis=1) > 0).all()), f"{len(odd)} URLs tried")
    long = items.loc[items["truncated"]].head(20)
    long_ids = encode_urls(long["url"])
    c.add(f"a URL longer than {MAX_URL_LEN} is cut off at {MAX_URL_LEN}", len(long) > 0 and bool(((long_ids != 0).sum(axis=1) == MAX_URL_LEN).all())
          and all(decode_url(row) == shown(u) for row, u in zip(long_ids, long["url"])), f"{len(long)} URLs tried")
    c.info("unknown token share", f"{100 * (ids == 1).sum() / nonpad.sum():.4f}% of the real ids in this sample")
    return c


# ----------------------------------------------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------------------------------------------
def evaluation_checks(items: pd.DataFrame, seed: int = 0) -> Checks:
    """Run made-up scores through save_and_evaluate in a temporary folder and compare with the metrics they must give.

    Random scores must give AUROC near 0.5 and AUPRC near the malicious share. Perfect scores (the label itself) must give
    1 everywhere, and reversed scores must give AUROC 0. Malformed score files must be refused.
    """
    c = Checks("Evaluation checks (made-up scores in a temporary folder)")
    val, test = (items[items["split"] == s] for s in ("validation", "test"))
    rng = np.random.default_rng(seed)

    def run(val_scores, test_scores, val_ids=None, test_ids=None):
        with tempfile.TemporaryDirectory() as tmp:
            return save_and_evaluate(tmp, items, "check", "made_up", seed,
                                     validation=(val["item_id"] if val_ids is None else val_ids, val_scores),
                                     test=(test["item_id"] if test_ids is None else test_ids, test_scores)).iloc[0]

    r = run(rng.normal(size=len(val)), rng.normal(size=len(test)))
    c.add("random scores give AUROC near 0.5", abs(r["auroc"] - 0.5) < 0.01, f"AUROC {r['auroc']:.4f}")
    c.add("random scores give AUPRC near the malicious share", abs(r["auprc"] - r["prevalence"]) < 0.01, f"AUPRC {r['auprc']:.4f}, malicious share {r['prevalence']:.4f}")
    c.add("random scores flag about 1% and 5% of test benign items at the validation thresholds",
          abs(r["test_fpr_at_1pct"] - 0.01) < 0.004 and abs(r["test_fpr_at_5pct"] - 0.05) < 0.01,
          f"{100 * r['test_fpr_at_1pct']:.2f}% and {100 * r['test_fpr_at_5pct']:.2f}%")

    p = run(val["label"].to_numpy(float), test["label"].to_numpy(float))
    c.add("perfect scores give AUROC 1, AUPRC 1 and recall 1 with no false alarms",
          p["auroc"] == 1 and p["auprc"] == 1 and p["recall_at_1pct_fpr"] == 1 and p["recall_at_5pct_fpr"] == 1 and p["test_fpr_at_1pct"] == 0,
          f"AUROC {p['auroc']:.3f}, AUPRC {p['auprc']:.3f}, recall at 1% FPR {p['recall_at_1pct_fpr']:.3f}")
    q = run(-val["label"].to_numpy(float), -test["label"].to_numpy(float))
    c.add("reversed scores give AUROC 0 (higher must mean more malicious)", q["auroc"] == 0, f"AUROC {q['auroc']:.3f}")

    scores = rng.normal(size=len(val))
    t_scores = rng.normal(size=len(test))
    c.add("a validation file that misses one item is refused", _refused(lambda: run(scores[1:], t_scores, val_ids=val["item_id"].to_numpy()[1:])))
    c.add("a validation file with an item that is not in the split is refused",
          _refused(lambda: run(np.append(scores, 0.0), t_scores, val_ids=np.append(val["item_id"].to_numpy(), -1))))
    c.add("scores with a missing value (NaN) are refused", _refused(lambda: run(np.where(np.arange(len(val)) == 0, np.nan, scores), t_scores)))
    return c


# ----------------------------------------------------------------------------------------------------------------------
# Models (tiny networks on the CPU)
# ----------------------------------------------------------------------------------------------------------------------
def model_checks(items: pd.DataFrame, seed: int = 0) -> Checks:
    """Check the code of the URL VAE and the supervised reference on a tiny network and a few dozen URLs."""
    import torch
    from src.nn_common import encode_items, get_device, to_tensors, trim
    from src.supervised import SupervisedConfig, URLClassifier, score_classifier, train_classifier
    from src.url_vae import START_ID, URLVAE, VAEConfig, kl_to_prior, score_vae, train_vae

    c = Checks("Model checks (tiny networks on the CPU)")
    dev = get_device("cpu")
    cfg = VAEConfig(emb_dim=16, hidden=32, latent=4, epochs=1, batch_size=16, score_samples=2)
    rows = items[(items["split"] == "train") & (items["cls"] == "benign")].sample(48, random_state=seed).reset_index(drop=True)
    enc = encode_items(rows)
    data = to_tensors(rows, enc, np.ones(len(rows), dtype=bool), dev)
    torch.manual_seed(seed)
    model = URLVAE(cfg).eval()
    lens = data.lengths[:8]
    ids = trim(data.ids[:8], lens)
    mu, logvar = model.encode(ids, lens)

    z = mu
    logits = model.decode(ids, z)
    c.add("decoder gives one logit vector of 97 per character position", tuple(logits.shape) == (8, ids.shape[1], VOCAB_SIZE), str(tuple(logits.shape)))
    c.add("the start token id is outside the character vocabulary", START_ID == VOCAB_SIZE)

    with torch.no_grad():
        per_char = model.rec_nll(ids, z) / lens
    c.add("an untrained model costs about log(97) = 4.57 nats per character", bool((per_char > 4.0).all() and (per_char < 5.2).all()),
          f"{per_char.min():.2f} to {per_char.max():.2f}")

    j = int(lens.argmin())
    short_ids, short_len = data.ids[j:j + 1][:, :int(lens[j])].long(), lens[j:j + 1]
    with torch.no_grad():
        alone = model.rec_nll(short_ids, model.encode(short_ids, short_len)[0])
        in_batch = model.rec_nll(ids, mu)[j:j + 1]
    c.add("padding does not change a URL's loss (alone or inside a longer batch)", bool(torch.allclose(alone, in_batch, atol=1e-4)),
          f"{alone.item():.5f} vs {in_batch.item():.5f}")

    k = int(lens.argmax())
    pos = min(5, int(lens[k]) - 2)
    changed = ids.clone()
    changed[k, pos] = 2 + (int(changed[k, pos]) - 1) % 95
    with torch.no_grad():
        a, b = model.decode(ids, z), model.decode(changed, z)
    c.add("the decoder is causal: changing character t does not change the predictions up to t", bool(torch.allclose(a[k, :pos + 1], b[k, :pos + 1], atol=1e-5)))
    c.add("the decoder does read earlier characters: it changes the prediction after t", not bool(torch.allclose(a[k, pos + 1], b[k, pos + 1], atol=1e-6)))

    c.add("KL to the prior is 0 for N(0, I) and positive otherwise",
          float(kl_to_prior(torch.zeros(3, 4), torch.zeros(3, 4)).abs().max()) == 0 and bool((kl_to_prior(torch.ones(3, 4), torch.zeros(3, 4)) > 0).all()))

    sc = score_vae(model, data, dev, seed=seed)
    c.add("scores are finite and positive, one row per URL", len(sc) == len(rows) and bool(np.isfinite(sc[["nll_per_char", "neg_elbo_per_char"]].to_numpy()).all()) and bool((sc[["nll_per_char", "neg_elbo_per_char"]] > 0).all().all()))
    with torch.no_grad():
        direct = (model.rec_nll(ids, mu) / lens).numpy()
    c.add("nll_per_char equals the reconstruction loss at z = mu divided by the URL length", bool(np.allclose(sc["nll_per_char"].to_numpy()[:8], direct, atol=1e-4)))
    sc2 = score_vae(model, data, dev, seed=seed)
    c.add("scoring the same data twice gives the same scores (fixed noise)", bool(np.allclose(sc.to_numpy(), sc2.to_numpy())))
    c.add("the negative ELBO is at least the NLL at z = mu on average (KL is added)", float(sc["neg_elbo_per_char"].mean()) > float(sc["nll_per_char"].mean()) - 1e-9)

    fit = VAEConfig(emb_dim=16, hidden=48, latent=4, epochs=25, batch_size=16, lr=5e-3, anneal_epochs=2, word_dropout=0.0, score_samples=1)
    tiny = to_tensors(rows, enc, (np.arange(len(rows)) < 32), dev)
    held = to_tensors(rows, enc, (np.arange(len(rows)) >= 32), dev)
    trained, hist = train_vae(tiny, held, fit, seed, dev, log=lambda *_: None)
    c.add("training lowers the reconstruction cost per character", float(hist["train_rec_per_char"].iloc[-1]) < 0.85 * float(hist["train_rec_per_char"].iloc[0]),
          f"{hist['train_rec_per_char'].iloc[0]:.2f} to {hist['train_rec_per_char'].iloc[-1]:.2f} nats per character")
    c.add("the history has one row per epoch with a best epoch marked and the KL term logged", len(hist) == fit.epochs and int(hist["best"].sum()) == 1 and bool(hist["train_kl"].notna().all()))

    sup_cfg = SupervisedConfig.matching(cfg, epochs=1)
    clf = URLClassifier(sup_cfg)
    n_params = lambda m: sum(p.numel() for p in m.parameters())
    c.add("the supervised reference has the VAE's encoder (same sizes) plus one linear layer",
          n_params(clf.encoder) == n_params(model.encoder) and n_params(clf.head) == clf.encoder.out_dim + 1)
    probs = score_classifier(clf, data, dev)
    c.add("reference scores are probabilities between 0 and 1", bool(((probs >= 0) & (probs <= 1)).all()) and len(probs) == len(rows))

    from pathlib import Path
    # batching: bucketed batches cover every row once and carry far less padding than random batches
    import torch as _torch
    from src.nn_common import epoch_batches
    ln = items.loc[items["split"] == "train", "url_len"].clip(upper=256).to_numpy()[:20000]
    def mean_longest(bs):
        return float(np.mean([ln[b].max() for b in bs]))
    g1, g2, g3 = (_torch.Generator().manual_seed(seed) for _ in range(3))
    rand, buck, buck2 = epoch_batches(ln, 256, g1, 0), epoch_batches(ln, 256, g2, 20), epoch_batches(ln, 256, g3, 20)
    c.add("random and bucketed batches each use every row exactly once",
          all(np.array_equal(np.sort(np.concatenate(b)), np.arange(len(ln))) for b in (rand, buck)))
    c.add("bucketed batches have the same batch count as random batches", len(rand) == len(buck))
    c.add("bucketed batches are mostly padding-free (mean longest URL in a batch is far below that of random batches)",
          mean_longest(buck) < 0.6 * mean_longest(rand), f"{mean_longest(buck):.0f} against {mean_longest(rand):.0f} characters")
    c.add("the same seed gives the same batches", all(np.array_equal(x, y) for x, y in zip(buck, buck2)))

    # final runs reuse the settings of the development run
    from src.runner import final_configs
    from src.url_vae import save_vae
    with tempfile.TemporaryDirectory() as tmp:
        d_vae, d_ref = final_configs(tmp)
        c.add("with no saved run the final settings are the defaults", d_vae == VAEConfig() and d_ref == SupervisedConfig.matching(VAEConfig()))
        save_vae(Path(tmp) / "url_vae" / "seed0", trained, hist, seed)
        f_vae, f_ref = final_configs(tmp)
        c.add("the final VAE settings are the ones saved with the development model", f_vae == fit)
        c.add("the final reference takes its encoder sizes from the VAE", (f_ref.emb_dim, f_ref.hidden, f_ref.bidirectional) == (fit.emb_dim, fit.hidden, fit.bidirectional))

    # run records: a finished seed is skipped, and a changed setting, other data or a missing record trains it again
    from src.runner import RECORD_FILE, quick_subset, run_url_vae
    sub = quick_subset(items, per_group=30, seed=seed)
    rec_cfg = VAEConfig(emb_dim=8, hidden=16, latent=4, epochs=1, batch_size=32, score_samples=1)

    fake_sources = {"all_generated_urls_1.csv": {"bytes": 1, "sha256": "ab" * 32}}

    def run(tmp, data, config, msgs, **kw):
        run_url_vae(data, Path(tmp) / "res", Path(tmp) / "art", config, [0], "cpu", log=msgs.append, **kw)

    with tempfile.TemporaryDirectory() as tmp:
        first, again, new_cfg, new_data, no_rec = [], [], [], [], []
        run(tmp, sub, rec_cfg, first, sources=fake_sources)
        record = Path(tmp) / "res" / "url_vae" / "nll_per_char" / "seed0" / RECORD_FILE
        c.add("a finished seed writes a run record with its settings and the data version", record.exists() and "data_version" in record.read_text())
        saved = json.loads(record.read_text())
        c.add("the record names the environment and the source checksums", {"python", "torch", "numpy", "pandas", "device"} <= set(saved.get("environment", {})) and saved.get("sources") == fake_sources,
              ", ".join(f"{k} {v}" for k, v in saved.get("environment", {}).items() if k in ("python", "torch")))
        run(tmp, sub, rec_cfg, again)                      # no sources this time: environment and checksums are never compared
        c.add("rerunning the same settings on the same data skips the finished seed", any("skipped" in m for m in again) and not any("training again" in m for m in again))
        run(tmp, sub, VAEConfig(**{**rec_cfg.to_dict(), "epochs": 2}), new_cfg)
        c.add("a changed setting trains the seed again and says which setting", any("training again" in m and "epochs 1 to 2" in m for m in new_cfg))
        other = sub.copy()
        other.loc[other.index[0], "split"] = "test" if other["split"].iloc[0] != "test" else "validation"
        run(tmp, other, VAEConfig(**{**rec_cfg.to_dict(), "epochs": 2}), new_data)
        c.add("a different split of the rows trains the seed again", any("training again" in m and "data version" in m for m in new_data))
        record.unlink()
        run(tmp, other, VAEConfig(**{**rec_cfg.to_dict(), "epochs": 2}), no_rec)
        c.add("scores without a run record are not trusted", any("no run record" in m for m in no_rec))

    # tuning sweep: one change at a time, the lowest benign validation negative ELBO wins, nothing is trained twice
    from src.runner import Tuning, run_tuning
    tune_base = VAEConfig(emb_dim=8, hidden=16, latent=4, epochs=1, batch_size=32, score_samples=1)
    with tempfile.TemporaryDirectory() as tmp:
        first, again = [], []
        tune = run_tuning(sub, tmp, tune_base, latent=(2, 4), beta=(0.1, 1.0), device="cpu", log=first.append)
        names = list(tune.table["candidate"])
        c.add("the sweep tries the base settings and one change at a time", names == ["base", "latent2", "beta0.1"] and isinstance(tune, Tuning), ", ".join(names))
        pick = tune.table.loc[tune.table["val_neg_elbo_per_char"].idxmin(), "candidate"]
        c.add("the winner is the candidate with the lowest benign validation negative ELBO", tune.best == pick and int(tune.table["selected"].sum()) == 1, f"{tune.best}")
        c.add("every candidate is saved with a run record", all((Path(tmp) / "url_vae_tuning" / n / RECORD_FILE).exists() and (Path(tmp) / "url_vae_tuning" / n / "model.pt").exists() for n in names))
        run_tuning(sub, tmp, tune_base, latent=(2, 4), beta=(0.1, 1.0), device="cpu", log=again.append)
        c.add("rerunning the sweep trains nothing again", sum("skipped" in m for m in again) == 3 and not any("latent " in m for m in again))
        run_tuning(sub, tmp, VAEConfig(**{**tune_base.to_dict(), "epochs": 2}), latent=(2, 4), beta=(0.1, 1.0), device="cpu", log=(changed := []).append)
        c.add("a changed base setting trains the candidates again", not any("skipped" in m for m in changed))
        run_tuning(sub, tmp, tune_base, latent=(2, 4), beta=(0.1, 1.0), device="cpu", log=lambda *_: None)
        reuse = []
        run_url_vae(sub, Path(tmp) / "res", Path(tmp) / "art", tune.best_config, [0], "cpu", log=reuse.append, trained=tune.winners("cpu"))
        c.add("the winner is scored as seed 0 without training again", any("no training" in m for m in reuse) and (Path(tmp) / "res" / "url_vae" / "nll_per_char" / "seed0").exists())
        c.add("the sweep used no malicious or test row (its table has validation numbers only)", not any("test" in col or "auroc" in col for col in tune.table.columns))

    # a non-finite gradient stops training instead of being clipped into NaN weights (an absurd learning rate blows the weights up to inf in a step or two)
    def stops(train_fn) -> bool:
        try:
            train_fn()
        except RuntimeError as e:
            return "non-finite" in str(e)
        return False
    nan_vae = VAEConfig(emb_dim=8, hidden=16, latent=4, epochs=2, batch_size=16, lr=1e30)
    nan_sup = SupervisedConfig(emb_dim=8, hidden=16, epochs=2, batch_size=16, lr=1e30)
    mixed = to_tensors(sub, encode_items(sub), np.ones(len(sub), dtype=bool), dev)
    c.add("the VAE trainer stops on a non-finite gradient", stops(lambda: train_vae(tiny, held, nan_vae, seed, dev, log=lambda *_: None)))
    import src.supervised as sup_mod                          # the classifier's loss stays finite for any learning rate, so poison its output instead

    class _NaNClassifier(sup_mod.URLClassifier):
        def forward(self, *a, **k):
            return super().forward(*a, **k) * float("nan")
    real = sup_mod.URLClassifier
    sup_mod.URLClassifier = _NaNClassifier
    try:
        stopped = stops(lambda: train_classifier(mixed, mixed, SupervisedConfig(emb_dim=8, hidden=16, epochs=1, batch_size=16), seed, dev, log=lambda *_: None))
    finally:
        sup_mod.URLClassifier = real
    c.add("the supervised trainer stops on a non-finite gradient", stopped)

    # looking inside the model (src/inspect_vae.py), on the tiny network above
    import src.inspect_vae as iv
    import torch.nn.functional as F
    per_char = iv.char_nll(model, data, dev)
    lens_all = data.lengths.cpu().numpy()
    c.add("char_nll gives one array per URL with one value per real character", len(per_char) == len(rows) and all(len(a) == n for a, n in zip(per_char, lens_all)))
    c.add("the per-character values average to the nll_per_char score",
          bool(np.allclose([a.mean() for a in per_char], sc["nll_per_char"].to_numpy(), atol=1e-4)))
    mu_all = iv.latent_means(model, data, dev)
    c.add("latent_means has one posterior mean per URL, equal to the encoder output",
          mu_all.shape == (len(rows), cfg.latent) and bool(np.allclose(mu_all[:8], mu.detach().numpy(), atol=1e-5)))
    picked = [5, 2, 9]
    c.add("example_nll returns the picked URLs in the order asked for", all(np.allclose(x, per_char[p]) for x, p in zip(iv.example_nll(model, rows, enc, picked, dev), picked)))
    enc_sub = encode_items(sub)
    pos, mu_s = iv.latent_sample(model, sub, enc_sub, dev, per_class=10, seed=seed)
    mu_sub = iv.latent_means(model, mixed, dev)
    c.add("latent_sample takes the same number of URLs of each class, in row order, with their posterior means",
          len(pos) == 20 and int(sub["label"].to_numpy()[pos].sum()) == 10 and bool((np.diff(pos) > 0).all()) and mu_s.shape == (20, cfg.latent)
          and bool(np.allclose(mu_s, mu_sub[pos], atol=1e-5)))
    flat = np.random.default_rng(seed).normal(size=(200, 2)) @ np.random.default_rng(seed + 1).normal(size=(2, 6))
    coords, expl = iv.pca_2d(flat)
    c.add("pca_2d recovers two components that explain all the variance of data that lies in a plane", coords.shape == (200, 2) and float(expl.sum()) > 0.999999)
    spread = np.zeros((100, 5))
    spread[:, 2] = np.random.default_rng(seed).normal(size=100)
    c.add("active_units counts only the dimensions that vary", iv.active_units(np.zeros((100, 5))) == 0 and iv.active_units(spread) == 1)
    one, two = iv.prior_samples(model, 4, 30, dev, seed=3), iv.prior_samples(model, 4, 30, dev, seed=3)
    c.add("prior samples are repeatable for a seed and have the requested length", one == two and all(len(u) == 30 for u in one) and iv.prior_samples(model, 4, 30, dev, seed=4) != one)
    gen = torch.Generator().manual_seed(seed)
    z = torch.randn(5, cfg.latent, generator=gen)
    s_ids, s_logp = iv._sample_ids(model, z, 25, gen, 1.0)
    with torch.no_grad():                                         # the sampler never writes padding, so compare with the same renormalised probabilities
        forced = model.decode(s_ids, z)
        forced[:, :, 0] = float("-inf")
        forced_logp = F.log_softmax(forced, dim=2).gather(2, s_ids.unsqueeze(2)).squeeze(2).sum(dim=1)
    c.add("writing a URL character by character gives the same likelihood as scoring it with teacher forcing",
          bool(torch.allclose(s_logp, forced_logp, atol=1e-3)), f"{s_logp.tolist()[0]:.3f} vs {forced_logp.tolist()[0]:.3f}")
    return c


# ----------------------------------------------------------------------------------------------------------------------
# Analysis (weighted metrics, bootstrap, comparisons)
# ----------------------------------------------------------------------------------------------------------------------
def analysis_checks(items: pd.DataFrame, seed: int = 0) -> Checks:
    """Check src/analysis.py on made-up scores written to a temporary folder (random, informative, reversed, copies)."""
    from src.analysis import Analysis, ScoreView, domain_bootstrap

    c = Checks("Analysis checks (made-up scores in a temporary folder)")
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, 3000)
    s = np.round(rng.normal(size=3000) + y * 0.5, 1)              # rounded so there are ties
    v = ScoreView(y, s)
    c.add("weighted AUROC and AUPRC equal src/evaluate.py with equal weights",
          abs(v.auroc(np.ones(3000)) - auroc(y, s)) < 1e-12 and abs(v.auprc(np.ones(3000)) - auprc(y, s)) < 1e-12)
    w = rng.integers(0, 4, 3000)
    rep = np.repeat(np.arange(3000), w)
    c.add("integer weights equal repeating the rows (AUROC and AUPRC)",
          abs(v.auroc(w.astype(float)) - auroc(y[rep], s[rep])) < 1e-12 and abs(v.auprc(w.astype(float)) - auprc(y[rep], s[rep])) < 1e-12)

    val, test = (items[items["split"] == k] for k in ("validation", "test"))
    labels = test["label"].to_numpy(float)
    runs = {"check_random": rng.normal(size=len(test)), "check_good": labels + rng.normal(size=len(test)) * 1.5,
            "check_reversed": -(labels + rng.normal(size=len(test)) * 1.5)}
    with tempfile.TemporaryDirectory() as tmp:
        for name, t_scores in runs.items():
            v_scores = (val["label"].to_numpy(float) * (0 if name == "check_random" else (-1 if name == "check_reversed" else 1))
                        + rng.normal(size=len(val)) * (1.0 if name == "check_random" else 1.5))
            for sd in (0, 1):
                save_and_evaluate(tmp, items, name, "made_up", sd, validation=(val["item_id"], v_scores + 0.0 * sd),
                                  test=(test["item_id"], t_scores + 0.01 * sd))
        save_and_evaluate(tmp, items, "check_single", "made_up", 0, validation=(val["item_id"], val["label"].to_numpy(float) + rng.normal(size=len(val)) * 1.5),
                          test=(test["item_id"], labels + rng.normal(size=len(test)) * 1.5))
        an = Analysis(tmp, items, n_boot=60)
        one = np.ones(len(an.test))
        c.add("the weighted AUROC equals the AUROC in the metrics table",
              abs(an.view("check_good", "made_up", 0).auroc(one) - an.seed_table().query("model == 'check_good' and seed == 0")["auroc"].iloc[0]) < 1e-12)

        boot = domain_bootstrap(an.test["domain"], {"a": an.view("check_random", "made_up", 0).auroc}, 60, 0)["a"]
        c.add("bootstrap replicates of a random scorer are centred near 0.5", abs(boot.mean() - 0.5) < 0.02 and boot.std() > 0, f"mean {boot.mean():.3f}, sd {boot.std():.4f}")
        ci = an.metrics_ci([("check_random", "made_up")]).iloc[0]
        c.add("the 95% interval of a random scorer's AUROC contains 0.5", ci["auroc_lo"] <= 0.5 <= ci["auroc_hi"], f"[{ci['auroc_lo']:.3f}, {ci['auroc_hi']:.3f}]")

        good = an.compare("check_good", "made_up", "check_random", "made_up")
        c.add("an informative score beats a random one (paired interval above 0), overall and for every seed", bool(good["beats_reference"].all()) and len(good) == 3,
              f"difference {good['difference'].iloc[0]:+.3f}")
        rev = an.compare("check_reversed", "made_up", "check_random", "made_up")
        c.add("a reversed score is below a random one (interval below 0)", bool((rev["ci_high"] < 0).all()))
        same = an.compare("check_good", "made_up", "check_good", "made_up", ref_seed=0).iloc[1]
        c.add("a score compared with itself has difference 0 and interval [0, 0]", same["difference"] == 0 and same["ci_low"] == 0 and same["ci_high"] == 0)

        st = an.strata("check_good", "made_up", "check_random", "made_up")
        sizes = st.assign(least=st[["n_benign", "n_malicious"]].min(axis=1), doms=st[["domains_benign", "domains_malicious"]].min(axis=1))
        c.add("a stratum has an AUROC exactly when both classes have enough items and domains",
              bool((((sizes["least"] >= an.min_per_class) & (sizes["doms"] >= an.min_domains)) == sizes["eligible"]).all()) and bool(st.loc[~st["eligible"], "model_auroc"].isna().all()),
              f"{int(st['eligible'].sum())} of {len(st)} strata eligible (at least {an.min_per_class} items and {an.min_domains} domains per class)")
        few = items.copy()                                       # 150 rows of each class from one single domain: enough rows, too few domains
        for k in (0, 1):
            idx = few.index[(few["split"] == "test") & (few["label"] == k)][:150]
            few.loc[idx, ["scheme", "domain"]] = ["check_few_domains", "one.example"]
        counts = Analysis(tmp, few).strata_counts(["scheme"]).set_index("stratum")
        c.add("a stratum with 150 rows per class from one domain is not eligible",
              int(counts.loc["check_few_domains", "n_benign"]) >= 150 and not bool(counts.loc["check_few_domains", "eligible"]),
              f"{int(counts.loc['check_few_domains', 'domains_benign'])} domain per class")
        crit = an.criteria("check_good", "made_up", floor=("check_random", "made_up"), floor_b=("check_random", "made_up"), reference=("check_missing", "x"))
        c.add("the success criteria table has its five rows and a pass for an informative score", len(crit) == 5 and crit["result"].iloc[:3].eq("pass").all(), ", ".join(crit["result"]))
        single = an.criteria("check_single", "made_up", floor=("check_random", "made_up"), floor_b=("check_random", "made_up"), reference=("check_missing", "x"))
        c.add("with one seed the success criteria have no seed robustness row", len(single) == 4 and "Seed robustness" not in single["criterion"].tolist() and single["result"].iloc[:2].eq("pass").all(),
              ", ".join(single["result"]))
        c.add("the best score is chosen on validation AUROC", an.choose_score("check_good", ("made_up",)) == "made_up")

        from src.inspect_vae import surprise_examples
        from src.plots import roc_points
        view = an.view("check_good", "made_up", 0)
        fpr, tpr = roc_points(view)
        area = float(np.sum(np.diff(fpr[::-1]) * (tpr[::-1][1:] + tpr[::-1][:-1]) / 2))
        c.add("the area under the ROC points equals the AUROC of the metric code", abs(area - view.auroc(np.ones(len(an.test)))) < 1e-9 and fpr[0] == 1 and tpr[0] == 1 and fpr[-1] == 0, f"{area:.4f}")
        short = an.test.assign(url_len=np.where(np.arange(len(an.test)) % 3 == 0, 200, 40))
        ex = surprise_examples(short, view.score, per_group=2, max_len=70)
        by = ex.groupby("group")
        ok_groups = len(ex) == 8 and by.size().eq(2).all() and bool((ex["url_len"] <= 70).all())
        hi_b = short[(short["label"] == 0) & (short["url_len"] <= 70)].assign(score=view.score[(short["label"] == 0).to_numpy() & (short["url_len"] <= 70).to_numpy()])["score"].nlargest(2)
        top = ex[ex["group"].str.startswith("benign, highest")]
        c.add("the example URLs are chosen by score within the length limit, two per group, with the right labels",
              ok_groups and np.allclose(sorted(top["score"]), sorted(hi_b)) and ex[ex["group"].str.startswith("malicious")]["label"].eq(1).all() and ex[ex["group"].str.startswith("benign")]["label"].eq(0).all())
    return c
