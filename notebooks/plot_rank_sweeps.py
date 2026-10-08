"""Appendix figure: rank sweeps as small multiples, every metric and arm.

The main-text table reports three metrics at the ranks the argument needs.
This figure carries the rest of the evidence to the appendix, where the whole
sweep can be shown at once: a table with five metrics times five ranks times
four arms times four settings is unreadable, but the story is a shape -- one
family of curves rises while another falls, and only in one metric.

Rows are metrics, columns are (dataset, mode). Each panel plots the metric
against the rank, one line per arm, with the seed spread as a band. The
log-score row is the only one that separates the families; the four rows below
it are the control that rules out a general degradation, which is what makes
objective mismatch the remaining explanation.

The figure caption is maintained with the paper's appendix source.

    python notebooks/plot_rank_sweeps.py --out paper_draft/figures
"""

import argparse
import collections
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

ENTITY_PROJECT = "precip/spectral-experiments"
EXCLUDE_TAGS = {"shakeout", "exclude-from-results", "bs4-no-accum-obsolete"}

SETTINGS = [("etth1", "M", 96 * 7), ("etth1", "S", 96),
            ("etth2", "M", 96 * 7), ("etth2", "S", 96)]

# (label, colour, marker, predicate on the facet dict)
ARMS = [
    ("LorD, KS", "#d62728", "o",
     lambda f: f["method"] == "phys_lora" and f["loss"] == "kernel_score"),
    ("LorD, NLL", "#ff9896", "s",
     lambda f: f["method"] == "phys_lora" and f["loss"] == "nll"),
    ("SCORE, NLL", "#1f77b4", "o",
     lambda f: f["method"].startswith("whitened") and f["stage2"] == "nll"
     and f["transform"] == "fourier"),
    ("SCORE, KS", "#9edae5", "s",
     lambda f: f["method"].startswith("whitened") and f["stage2"] == "kernel_score"
     and f["transform"] == "fourier"),
]

# Panel order, following `make_results_tables.COLUMNS` (the variogram score is
# not swept here).
METRICS = [("mse", "MSE"), ("crps", "CRPS"),
           ("energy_score", "energy score"), ("kernel_score", "kernel score"),
           ("nll_per_d", "log-score / $d$")]


def _tag(tags, prefix):
    return next((t[len(prefix):] for t in tags if t.startswith(prefix)), None)


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def collect():
    """{(dataset, features, arm_label, rank): {metric: [values over seeds]}}"""
    api = wandb.Api(timeout=90)
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for dataset, features, d in SETTINGS:
        for run in api.runs(ENTITY_PROJECT,
                            filters={"tags": f"dataset:{dataset}"}, per_page=300):
            tags = list(run.tags)
            if EXCLUDE_TAGS & set(tags) or _tag(tags, "features:") != features:
                continue
            if "evaluation/test_crps" not in run.summary:
                continue
            # Only the default stage-1 arm: the stage-1 ablation is a separate
            # question and would double every line without adding to this one.
            stage1 = _tag(tags, "stage1:")
            if stage1 is not None and stage1 != "marginal_kernel_score":
                continue
            facet = {"method": _tag(tags, "method:") or "",
                     "loss": _tag(tags, "loss:") or "",
                     "stage2": _tag(tags, "stage2:") or "",
                     "transform": _tag(tags, "transform:") or "fourier"}
            rank = int(_tag(tags, "rank:") or 0)
            for label, _, _, pred in ARMS:
                if not pred(facet):
                    continue
                key = (dataset, features, label, rank)
                for metric, _ in METRICS:
                    src = "nll" if metric == "nll_per_d" else metric
                    value = _num(run.summary.get(f"evaluation/test_{src}"))
                    if value is None:
                        continue
                    out[key][metric].append(value / d if metric == "nll_per_d" else value)
                break
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="notebooks/figures/paper")
    parser.add_argument("--name", default="rank_sweeps")
    args = parser.parse_args()

    data = collect()
    n_rows, n_cols = len(METRICS), len(SETTINGS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.1 * n_cols, 2.05 * n_rows),
                             sharex=True, squeeze=False)

    for j, (dataset, features, _) in enumerate(SETTINGS):
        # "ETTh1", not "ETTH1": only the leading acronym is capitalized.
        pretty = dataset[:3].upper() + dataset[3:]
        title = f"{pretty} {'multivariate' if features == 'M' else 'univariate'}"
        axes[0][j].set_title(title, fontsize=10)
        for i, (metric, mlabel) in enumerate(METRICS):
            ax = axes[i][j]
            drew = False
            for label, colour, marker, _ in ARMS:
                ranks, means, los, his = [], [], [], []
                for rank in (0, 1, 2, 4, 8, 16):
                    vals = data.get((dataset, features, label, rank), {}).get(metric)
                    if not vals:
                        continue
                    ranks.append(max(rank, 0.5))  # 0 plotted at 0.5 on a log axis
                    means.append(float(np.mean(vals)))
                    los.append(float(np.min(vals)))
                    his.append(float(np.max(vals)))
                if not ranks:
                    continue
                drew = True
                ax.plot(ranks, means, marker=marker, ms=3.5, lw=1.4,
                        color=colour, label=label)
                ax.fill_between(ranks, los, his, color=colour, alpha=0.18, lw=0)
            ax.set_xscale("log", base=2)
            ax.set_xticks([0.5, 1, 2, 4, 8, 16])
            ax.set_xticklabels(["0", "1", "2", "4", "8", "16"], fontsize=8)
            ax.tick_params(axis="y", labelsize=8)
            ax.grid(alpha=0.25, lw=0.5)
            if metric == "nll_per_d" and drew:
                # The failure mode spans orders of magnitude; a symlog axis keeps
                # the well-behaved curves readable next to it.
                ax.set_yscale("symlog", linthresh=1.0, linscale=0.6)
            if j == 0:
                ax.set_ylabel(mlabel, fontsize=9)
            if i == n_rows - 1:
                ax.set_xlabel("rank $r$", fontsize=9)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(ARMS),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.005))
    fig.tight_layout(rect=(0, 0, 1, 0.965))

    os.makedirs(args.out, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(args.out, f"{args.name}.{ext}")
        fig.savefig(path, dpi=200 if ext == "png" else None, bbox_inches="tight")
        print(f"written {path}")


if __name__ == "__main__":
    main()
