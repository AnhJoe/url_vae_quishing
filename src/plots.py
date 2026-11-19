"""Small plotting helpers for the model notebooks, so the notebook cells stay one line each.

    plot_vae_history(histories)       training curves of the URL VAE: reconstruction per character (train and validation), the
                                      validation KL term (a collapse check), and the validation negative ELBO per character
    plot_classifier_history(histories)  training and validation cross-entropy of the supervised reference
    plot_strata(table)                AUROC of the model and of its reference inside each stratum, with the paired difference
                                      and its interval (from Analysis.strata)

Figures of the final results (6_results.ipynb section 9), built on src/analysis.py and src/inspect_vae.py:
    plot_detection(an, entries)       ROC curves and the AUROC with its 95% interval (beats the floor, gap to the reference)
    plot_score_distributions(an, ..)  the headline score of benign and malicious test URLs, overall and by URL scheme
    plot_surprise(examples, nll)      example URLs with every character coloured by how surprising the decoder found it
    plot_latent(coords, ...)          the latent space of the test URLs (first two principal components), coloured by class
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_COLOR, REF_COLOR = "#2a7ab9", "#8a8a8a"
BENIGN_COLOR, MALICIOUS_COLOR = "#2a7ab9", "#d1495b"


def plot_vae_history(histories: dict[int, pd.DataFrame]):
    """Training curves of the URL VAE against the epoch, one color per seed.

    Left: reconstruction negative log-likelihood per character, training (dashed) and benign validation (solid). The
    learning objective is almost all reconstruction, and a validation line that stays near the training line means the
    model is not overfitting. The training value is a running average over the epoch and the validation value is measured
    at the end of it. Middle: KL term on the benign validation URLs (a line near 0 means posterior collapse). Right:
    negative ELBO per character on the benign validation URLs, the criterion that picks the epoch kept.
    """
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    for seed, h in histories.items():
        line, = ax[0].plot(h["epoch"], h["train_rec_per_char"], linestyle="--", alpha=0.7)
        ax[0].plot(h["epoch"], h["val_nll_per_char"], color=line.get_color(), label=f"seed {seed}")
        ax[1].plot(h["epoch"], h["val_kl"], color=line.get_color(), label=f"seed {seed}")
        ax[2].plot(h["epoch"], h["val_neg_elbo_per_char"], color=line.get_color(), label=f"seed {seed}")
    ax[0].set(title="Reconstruction NLL per character (dashed: train, solid: validation)", xlabel="epoch")
    ax[1].set(title="KL term, benign validation (nats per URL)", xlabel="epoch")
    ax[2].set(title="Negative ELBO per character, benign validation", xlabel="epoch")
    ax[0].title.set_fontsize(9)
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def plot_classifier_history(histories: dict[int, pd.DataFrame]):
    """Training and validation binary cross-entropy against the epoch, for every seed."""
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for seed, h in histories.items():
        line, = ax.plot(h["epoch"], h["train_bce"], linestyle="--", alpha=0.7)
        ax.plot(h["epoch"], h["val_bce"], color=line.get_color(), label=f"seed {seed}")
    ax.set(title="Cross-entropy (dashed: train, solid: validation)", xlabel="epoch")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def plot_strata(table: pd.DataFrame, model_label: str = "URL VAE", ref_label: str = "floor A"):
    """AUROC inside each stratum for the model and its reference, with the paired difference and its 95% interval.

    The label of a stratum carries its number of benign and malicious test URLs. Strata with too few URLs or domains have no
    AUROC and are named under the figure.
    """
    ok = table["eligible"].astype(bool)
    t = table[ok].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.2, 0.62 * len(t) + 1.5))
    y = np.arange(len(t))
    ax.barh(y - 0.2, t["model_auroc"], height=0.38, color=MODEL_COLOR, label=model_label)
    ax.barh(y + 0.2, t["reference_auroc"], height=0.38, color=REF_COLOR, label=ref_label)
    ax.axvline(0.5, color="black", linewidth=0.8, linestyle=":")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c}: {s}\n{nb:,} benign, {nm:,} malicious" for c, s, nb, nm in
                        zip(t["stratum_column"], t["stratum"], t["n_benign"], t["n_malicious"])], fontsize=8)
    for yi, d, lo, hi in zip(y, t["difference"], t["ci_low"], t["ci_high"]):
        ax.text(1.02, yi, f"{d:+.3f}  [{lo:+.3f}, {hi:+.3f}]", transform=ax.get_yaxis_transform(), va="center", fontsize=8)
    ax.text(1.02, -0.9, "paired difference, 95% interval", transform=ax.get_yaxis_transform(), fontsize=8, color="#555555")
    ax.invert_yaxis()
    ax.set(xlabel="test AUROC", xlim=(0.3, 1.0), title="AUROC within strata")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    if (~ok).any():
        left_out = ", ".join(f"{c}: {s}" for c, s in zip(table.loc[~ok, "stratum_column"], table.loc[~ok, "stratum"]))
        fig.text(0.01, 0.01, f"Not shown (too few URLs or domains for an AUROC): {left_out}", fontsize=7.5, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


# ----------------------------------------------------------------------------------------------------------------------
# Figures of the final results
# ----------------------------------------------------------------------------------------------------------------------
def roc_points(view) -> tuple[np.ndarray, np.ndarray]:
    """False- and true-positive rate at every distinct score of a ScoreView (malicious is positive, flag a score and above)."""
    wp, wn = view.groups(np.ones(len(view.y)))
    fpr = np.r_[1.0, 1.0 - np.cumsum(wn) / wn.sum()]
    tpr = np.r_[1.0, 1.0 - np.cumsum(wp) / wp.sum()]
    return fpr, tpr


def plot_detection(an, entries):
    """Left: ROC curves of the test set. Right: AUROC with its 95% interval (domain bootstrap), and the AUROC of every seed
    for a model that has several.

    `entries` is a list of (label, model, score, colour). A model with several seeds draws one thin curve per seed. The
    circles on the curves mark the two operating points of the report (the threshold that lets 1% and 5% of validation
    benign URLs through) for the first seed. The interval is the one of the mean over seeds, so it measures the test data,
    and the dots next to it show the seeds.
    """
    ci = an.metrics_ci([(m, s) for _, m, s, _ in entries]).set_index(["model", "score"])
    many = any(len(an.seeds(m, s)) > 1 for _, m, s, _ in entries)          # draw the seed dots only when some model has several
    one = np.ones(len(an.test))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
    for label, model, score, colour in entries:
        seeds = an.seeds(model, score)
        for k, seed in enumerate(seeds):
            view = an.view(model, score, seed)
            fpr, tpr = roc_points(view)
            ax[0].plot(fpr, tpr, color=colour, linewidth=1.6 if len(seeds) == 1 else 1.0, alpha=0.9 if k == 0 else 0.5,
                       label=label if k == 0 else None)
        view = an.view(model, score, seeds[0])
        wp, wn = view.groups(one)
        for level in (0.01, 0.05):
            cut = view.cut(an.thresholds(model, score, seeds[0])[level])
            ax[0].plot(view.recall_of(wn, cut), view.recall_of(wp, cut), "o", color=colour, markersize=4.5,
                       markeredgecolor="white", markeredgewidth=0.6)
    ax[0].plot([0, 1], [0, 1], color="black", linewidth=0.8, linestyle=":")
    ax[0].set(xlabel="false-positive rate (benign URLs flagged)", ylabel="true-positive rate (malicious URLs flagged)",
              title="ROC curves on the test set", xlim=(0, 1), ylim=(0, 1.01))
    ax[0].legend(frameon=False, loc="lower right", fontsize=8)
    y = np.arange(len(entries))
    for i, (label, model, score, colour) in enumerate(entries):
        r = ci.loc[(model, score)]
        ax[1].plot([r["auroc_lo"], r["auroc_hi"]], [i, i], color=colour, linewidth=3, solid_capstyle="round")
        ax[1].plot(r["auroc"], i, "D", color="white", markeredgecolor=colour, markersize=7, zorder=3)
        if len(an.seeds(model, score)) > 1:
            for seed in an.seeds(model, score):
                ax[1].plot(an.view(model, score, seed).auroc(one), i + 0.22, "o", color=colour, markersize=3.5, alpha=0.8)
    ax[1].axvline(0.5, color="black", linewidth=0.8, linestyle=":")
    ax[1].set_yticks(y)
    ax[1].set_yticklabels([e[0] for e in entries])
    ax[1].invert_yaxis()
    ax[1].set(xlabel="test AUROC", title="AUROC: mean over seeds (diamond), 95% interval (bar), seeds (dots)" if many
              else "AUROC (diamond) with its 95% interval (bar)")
    ax[1].title.set_fontsize(9)
    fig.tight_layout()
    return fig


def plot_score_distributions(an, model: str, score: str, seed: int = 0, n_schemes: int = 2, bins: int = 60):
    """The score of benign and malicious test URLs: all of them, then inside the `n_schemes` largest URL schemes.

    A VAE that works puts the malicious curve to the right of the benign one in every panel. If the two curves overlap in
    all of them, a high reconstruction quality does not make the anomalies stand out (Bouman and Heskes, 2025). The AUROC of
    each panel is in its title.
    """
    view = an.view(model, score, seed)
    test, s = an.test, view.score
    schemes = test["scheme"].value_counts().index[:n_schemes].tolist()
    panels = [("all test URLs", np.ones(len(test), dtype=bool))] + [(f"scheme: {c}", (test["scheme"] == c).to_numpy()) for c in schemes]
    lo, hi = np.quantile(s, [0.005, 0.995])
    edges = np.linspace(lo, hi, bins + 1)
    fig, ax = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 3.4), sharey=False)
    for a, (title, mask) in zip(np.atleast_1d(ax), panels):
        for label, colour, name in ((0, BENIGN_COLOR, "benign"), (1, MALICIOUS_COLOR, "malicious")):
            a.hist(s[mask & (test["label"].to_numpy() == label)].clip(lo, hi), bins=edges, density=True, histtype="stepfilled",
                   alpha=0.45, color=colour, label=name)
        a.set(title=f"{title} (AUROC {view.subset(mask).auroc(np.ones(int(mask.sum()))):.3f})", xlabel=score.replace("_", " "), yticks=[])
    np.atleast_1d(ax)[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def plot_surprise(examples: pd.DataFrame, nll: list, vmax: float = 6.0, max_chars: int = 70):
    """Example URLs with each character coloured by its negative log-likelihood (nats) under the decoder, given z = mu.

    `examples` has `url`, `group` and `score` columns (from inspect_vae.surprise_examples) and `nll` holds the per-character
    values of the same rows. A dark character is one the decoder found surprising. log(97) = 4.6 nats is the cost of a
    uniform guess over the 97 symbols, so a character above that is worse than not knowing.
    """
    n = len(examples)
    width = min(max_chars, int(examples["url"].str.len().max()))
    cmap = plt.get_cmap("Reds")
    fig, ax = plt.subplots(figsize=(0.14 * width + 1.2, 0.5 * n + 0.45 * examples["group"].nunique() + 1.4))
    y, last = 0.0, None
    for row, values in zip(examples.itertuples(), nll):
        if row.group != last:
            y += 0.5 if last is None else 1.0
            ax.text(0, y, row.group, fontsize=9, fontweight="bold", va="center")
            y += 1.0
            last = row.group
        for j, (ch, v) in enumerate(zip(row.url[:max_chars], values[:max_chars])):
            ax.add_patch(plt.Rectangle((j, y - 0.4), 1, 0.8, facecolor=cmap(min(v / vmax, 1.0)), edgecolor="none"))
            ax.text(j + 0.5, y, ch if " " <= ch <= "~" else "?", family="monospace", fontsize=8, ha="center", va="center", color="white" if v / vmax > 0.6 else "black")
        ax.text(width + 1.0, y, f"{values.mean():.2f} nats per character", fontsize=8, va="center", color="#555555")
        y += 1.0
    ax.set(xlim=(0, width + 12), ylim=(y, -0.2))
    ax.axis("off")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
    fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02, label="negative log-likelihood of the character (nats)")
    return fig


def plot_latent(coords: np.ndarray, labels: np.ndarray, explained, n_active: int, latent: int, seed: int = 0):
    """The test URLs in the first two principal components of the posterior mean, coloured by class."""
    order = np.random.default_rng(seed).permutation(len(coords))                 # so neither class is drawn on top of the other
    c, y = coords[order], np.asarray(labels)[order]
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.scatter(c[:, 0], c[:, 1], s=3, alpha=0.5, c=np.where(y == 1, MALICIOUS_COLOR, BENIGN_COLOR), linewidths=0)
    ax.scatter([], [], s=20, c=BENIGN_COLOR, label="benign")
    ax.scatter([], [], s=20, c=MALICIOUS_COLOR, label="malicious")
    ax.legend(frameon=False, fontsize=8)
    ax.set(xlabel=f"component 1 ({explained[0]:.0%} of variance)", ylabel=f"component 2 ({explained[1]:.0%})",
           title=f"Latent space of {len(coords):,} test URLs: {n_active} of {latent} dimensions active")
    ax.title.set_fontsize(10)
    fig.tight_layout()
    return fig
