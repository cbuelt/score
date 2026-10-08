"""Builds the paper's per-experiment results tables straight from wandb.

Runs are identified by the facet tags that ``src/train/train.py`` attaches
(``dataset:``, ``method:``, ``transform:``, ``rank:``, ``stage1:``, ``stage2:``,
``loss:``, ``features:``, ``seed:``), so a table never depends on parsing run
names. Runs carrying an exclusion tag are dropped, so excluded runs stay
visible in W&B without entering a table.

    python notebooks/make_results_tables.py era5
    python notebooks/make_results_tables.py etth1 --features M --out tables/

Output is a LaTeX table in the style of the ERA5 table plus the same numbers
as CSV. Configurations that are planned but have no finished run yet appear as
empty rows, so the table shows what is still missing instead of hiding it.
"""

import argparse
import csv
import json
import math
import os
import re
import statistics
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import wandb

ENTITY_PROJECT = "precip/spectral-experiments"

# Tags that disqualify a run from every table (smoke tests, superseded batches).
EXCLUDE_TAGS = {"shakeout", "exclude-from-results", "bs4-no-accum-obsolete"}

# Tags every run must carry to enter a table (set from --require-tag), e.g.
# `main2026` for the main-track grid (see `PLANNED_MAIN`).
REQUIRE_TAGS: set = set()

# Target dimension d per (dataset, features). Needed only to report the
# log-score per coordinate; `features` is None for the non-time-series sets.
#   era5      160 x 220 single-channel field
#   depth_bts 480 x 640 depth map
#   ETT       pred_len 96 x (7 series | 1 series)
DIMENSION: Dict[Tuple[str, Optional[str]], int] = {
    ("era5", None): 160 * 220,
    ("depth_bts", None): 480 * 640,
    ("depth_bts_crop", None): 426 * 560,   # Eigen crop of the same benchmark
    ("euppbench", None): 115,   # European surface stations, R2F task
    ("etth1", "M"): 96 * 7,
    ("etth1", "S"): 96,
    ("etth2", "M"): 96 * 7,
    ("etth2", "S"): 96,
    ("ettm1", "M"): 96 * 7,
    ("ettm1", "S"): 96,
    ("ettm2", "M"): 96 * 7,
    ("ettm2", "S"): 96,
    # ETTh at the reported horizons T96 and T192 (seeds 1-5). The key is the
    # task *name* rather than a (dataset, mode) pair -- the tree nests them one
    # level deeper
    # (`<dataset>/<mode>/T<pred_len>/<slug>/`) and they are read with
    # `--from-dir-exact`, which takes that directory and leaves the positional
    # argument as a pure label. `features` stays None because the mode is
    # already in the name; printing it again would duplicate it in the caption.
    # `_S_` tasks have d = pred_len, `_M_` tasks d = pred_len x 7 channels.
    ("etth1_S_T96", None): 96,
    ("etth1_S_T192", None): 192,
    ("etth2_S_T96", None): 96,
    ("etth2_S_T192", None): 192,
    ("etth1_M_T96", None): 96 * 7,
    ("etth1_M_T192", None): 192 * 7,
    ("etth2_M_T96", None): 96 * 7,
    ("etth2_M_T192", None): 192 * 7,
    # ETTm, same layout and the same horizon pair (15-minute data, so T96 is
    # 24 h).
    ("ettm1_S_T96", None): 96,
    ("ettm1_S_T192", None): 192,
    ("ettm2_S_T96", None): 96,
    ("ettm2_S_T192", None): 192,
    ("ettm1_M_T96", None): 96 * 7,
    ("ettm1_M_T192", None): 192 * 7,
    ("ettm2_M_T96", None): 96 * 7,
    ("ettm2_M_T192", None): 192 * 7,
    # Illness (ILI; seeds 1-5, horizons 24 and 36), multivariate only. Same
    # `--from-dir-exact` treatment; these are `features: M` tasks, so
    # d = pred_len x 7 channels.
    ("illness_M_T24", None): 24 * 7,
    ("illness_M_T36", None): 36 * 7,
    # Weather (seeds 1-5): 21 channels at a 10-minute cadence, multivariate
    # only, the reported horizons T96 and T192.
    ("weather_M_T96", None): 96 * 21,
    ("weather_M_T192", None): 192 * 21,
    # Electricity (seeds 1-5): 321 channels, hourly, multivariate only, the
    # reported horizons T96 and T192.
    ("electricity_M_T96", None): 96 * 321,
    ("electricity_M_T192", None): 192 * 321,
    # Traffic (seeds 1-5): 862 channels, hourly, multivariate only, the
    # reported horizons T96 and T192.
    ("traffic_M_T96", None): 96 * 862,
    ("traffic_M_T192", None): 192 * 862,
}

# Column order, and the single place it is defined: every table, figure panel
# and exported CSV in `notebooks/` reads its metric order from here (the one
# exception is `runtime_analysis.ipynb`, whose 2x3 grid is laid out by effect
# size instead -- see its `GRID_METRICS`).
#
# It runs from the metric that sees the least of the covariance to the one that
# sees the most: MSE ignores it entirely, CRPS and ES are marginal/energy
# summaries, VS and KS are the dependence-sensitive scores, and the log score
# closes the row as the only one with its own unit and the only one a run can
# be absent from.
#
# The variogram score is skipped where its O(d^2) pair sweep was too expensive
# to run (depth_bts); the printer drops all-empty columns anyway.
COLUMNS = [
    ("mse", "MSE"),
    ("crps", "CRPS"),
    ("energy_score", "ES"),
    ("variogram_score", "VS"),
    ("kernel_score", "KS"),
    ("nll_per_d", "NLL/$d$"),
]

# Methods that do not consume a `rank`. Some of their runs carry an inert
# `rank:1` tag inherited from method/default.yaml while the trained model is
# plain diagonal; the printed label must not claim a low-rank baseline that
# does not exist. `phys_cholesky` (Chol) is here for the same reason: its
# covariance is full, so a rank is meaningless for it.
RANK_BLIND_METHODS = {"phys_diag", "spectral_diag", "phys_cholesky", "generative"}

# Stage-1 loss that needs no row annotation because it is the default; mirrors
# _DEFAULT_STAGE1_LOSS in src/train/train.py, which drives the run naming.
_DEFAULT_STAGE1_LOSS_TABLE = "marginal_kernel_score"

#: Decimal places for the single-dataset table, when `--digits N` is given:
#: `{"n": N}` puts every column at N places, `{}` keeps the per-metric defaults.
#: A mutable container rather than a bare int so `main` can set it without
#: `global`.
#:
#: The per-metric defaults are not arbitrary -- 5 places on CRPS and the kernel
#: score, 3 on the log score -- they are set to the spread of each column. On
#: EUPPBench the kernel-score arms differ in the 4th decimal, so N = 3 prints
#: several of them identically. That is a presentation choice, not a bug, but it
#: means a reader cannot rank those rows off the rounded table.
DIGITS_OVERRIDE: Dict[str, int] = {}

# Which block of the table a run belongs to, from its method tag.
BLOCKS = [
    ("Physical space", {"phys_diag", "phys_lora", "phys_cholesky", "deterministic",
                        "generative"}),
    ("Transformed domain, unnormalized", {"spectral_diag"}),
    ("Marginally standardized (ours)", None),  # everything else (whitened_*, SCORE)
]

# Planned configurations per dataset, as facet tuples
# (method, transform, rank, stage1, stage2, loss) -- the same shape as
# Config.key(). Configurations that are planned but have no finished run yet
# appear as EMPTY rows in the per-dataset tables, so the table shows what is
# still missing instead of hiding it. Keep these lists in sync with the
# training launch scripts.
_MKS = "marginal_kernel_score"
PLANNED: Dict[str, list] = {
    "era5": [
        # SB baseline, fitted with the fair energy score
        ("generative", None, None, None, None, "energy_score"),
        # physical space (no deterministic row on ERA5)
        ("phys_diag", None, None, None, None, "nll"),
        ("phys_diag", None, None, None, None, "kernel_score"),
        *[("phys_lora", None, str(r), None, None, ls)
          for r in (1, 2, 4, 8, 16) for ls in ("kernel_score", "nll")],
        # transformed, unnormalized
        ("spectral_diag", None, None, None, None, "kernel_score"),
        ("spectral_diag", None, None, None, None, "nll"),
        # SCORE: diagonal core + low-rank sweep, both transforms/losses
        *[(m, tr, r, _MKS, s2, None)
          for m, tr in (("whitened_spectral_diag", "fourier"),
                        ("whitened_dct", "dct"))
          for r in (None, "2", "4", "8", "16")
          for s2 in ("kernel_score", "nll")],
        # stage-1 CRPS variants
        ("whitened_spectral_diag", "fourier", None, "crps", "nll", None),
        ("whitened_dct", "dct", "2", "crps", "nll", None),
    ],
    "depth_bts": [
        # Same grid as ETT/ERA5: both families, both losses, ranks 1/2/4/8/16.
        ("deterministic", None, None, None, None, "mse"),
        ("generative", None, None, None, None, "energy_score"),
        ("phys_diag", None, None, None, None, "kernel_score"),
        ("phys_diag", None, None, None, None, "nll"),
        *[("phys_lora", None, str(r), None, None, ls)
          for r in (1, 2, 4, 8, 16) for ls in ("kernel_score", "nll")],
        ("spectral_diag", None, None, None, None, "kernel_score"),
        ("spectral_diag", None, None, None, None, "nll"),
        *[(m, tr, r, _MKS, s2, None)
          for m, tr in (("whitened_spectral_diag", "fourier"),
                        ("whitened_dct", "dct"))
          for r in (None, "1", "2", "4", "8", "16")
          for s2 in ("kernel_score", "nll")],
    ],
    # Station post-processing: five seeds (1-5) of the full LorD rank ladder
    # under both losses, against the Det, Diag, Chol and SB baselines.
    "euppbench": [
        ("deterministic", None, None, None, None, "mse"),
        # SB is fitted with the (fair) kernel score here, not the energy
        # score: same estimator as the reported metric, so it compares
        # directly with the Gaussian families.
        ("generative", None, None, None, None, "kernel_score"),
        ("phys_diag", None, None, None, None, "kernel_score"),
        ("phys_diag", None, None, None, None, "nll"),
        *[("phys_lora", None, str(r), None, None, ls)
          for r in (1, 2, 4, 8, 16) for ls in ("kernel_score", "nll")],
        # Unstructured reference arm: only tractable here and on the univariate
        # ETT horizons (V = 115 / 96), not on the gridded tasks.
        ("phys_cholesky", None, None, None, None, "nll"),
        ("phys_cholesky", None, None, None, None, "kernel_score"),
        # Stage 2 scores the predictive covariance Sigma_P, not the bare
        # correlation core: on the localized graph Fourier basis the
        # transformed standardized target moves with the stage-2 parameter,
        # so a score of the core alone is not a proper score of the predictive
        # distribution. `dense_kernel_score` scores Sigma_P directly.
        ("whitened_graph_matern", "graph", None, "crps",
         "dense_kernel_score", None),
    ],
}


# Main-track grid (runs tagged main2026): SCORE with a Fourier diagonal core,
# CRPS in stage 1 and the kernel score in stage 2, against Det, Diag and
# LorD r = 1/2/4/16 under KS and NLL. Same grid on ERA5 and the Eigen-crop
# depth benchmark. Selected instead of PLANNED whenever
# --require-tag main2026 is given.
_MAIN_GRID = [
    ("deterministic", None, None, None, None, "mse"),
    ("phys_diag", None, None, None, None, "kernel_score"),
    ("phys_diag", None, None, None, None, "nll"),
    *[("phys_lora", None, str(r), None, None, ls)
      for r in (1, 2, 4, 16) for ls in ("kernel_score", "nll")],
    ("whitened_spectral_diag", "fourier", None, "crps", "kernel_score", None),
]
PLANNED_MAIN: Dict[str, list] = {"era5": _MAIN_GRID, "depth_bts_crop": _MAIN_GRID}


def _planned_cfg(facets: tuple) -> "Config":
    """Build a Config from a planned facet tuple via synthetic tags."""
    method, transform, rank, stage1, stage2, loss = facets
    tags = [f"method:{method}"]
    if transform:
        tags.append(f"transform:{transform}")
    if rank:
        tags.append(f"rank:{rank}")
    if stage1:
        tags.append(f"stage1:{stage1}")
    if stage2:
        tags.append(f"stage2:{stage2}")
    if loss:
        tags.append(f"loss:{loss}")
    return Config(tags)


def merge_planned(grouped: Dict[tuple, list], dataset: str) -> None:
    """Add empty rows for planned configurations without a finished run.

    The sentinel entry carries the Config and no metrics, so ``aggregate``
    reports ``(None, None, 0)`` for every column and the printer renders an
    empty cell -- a visible gap rather than a silently missing row.
    """
    registry = PLANNED_MAIN if "main2026" in REQUIRE_TAGS else PLANNED
    for facets in registry.get(dataset, []):
        cfg = _planned_cfg(facets)
        grouped.setdefault(cfg.key(),
                           [{"cfg": cfg, "seed": None, "run": None,
                             "metrics": {}}])


# Outer grouping of the loss-wise grid: the objective the core was fitted with,
# in the order the table reads. A row is only comparable to the other rows of
# its own group, which is why this is the outer level and not a column.
LOSS_GROUPS = [
    ("mse", "Point forecast only"),
    ("energy_score", "Sample-based, fitted with the energy score"),
    ("kernel_score", "Cores fitted with the kernel score"),
    ("nll", "Cores fitted with the log score"),
]


#: Stage-2 losses that estimate the *same* functional as ``kernel_score`` and
#: therefore share its objective group: ``dense_kernel_score`` is the identical
#: analytic formula evaluated against the predictive covariance rather than
#: against a family's structured surrogate, ``sample_kernel_score`` the fair
#: ensemble estimator of it. Rows fitted with these are comparable with the
#: plain kernel-score rows, so the estimator belongs in the score label and not
#: in the grouping -- a row whose objective fell outside LOSS_GROUPS would be
#: dropped from the by-objective tables without a word.
_KERNEL_SCORE_VARIANTS = {"dense_kernel_score": "dense",
                          "sample_kernel_score": "sample"}


def _tag(tags: Sequence[str], prefix: str) -> Optional[str]:
    for t in tags:
        if t.startswith(prefix):
            return t[len(prefix):]
    return None


def _num(value) -> Optional[float]:
    """Coerce a summary entry to a finite float, else None.

    The deterministic method reports an infinite log-score, and wandb stores
    that as the string ``"Infinity"``; both must drop out of the average
    rather than poison it.
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


class Config:
    """One table row: the facet tuple shared by a configuration's seeds."""

    def __init__(self, tags: Sequence[str]):
        self.method = _tag(tags, "method:") or "?"
        self.transform = _tag(tags, "transform:")
        self.rank = _tag(tags, "rank:")
        self.stage1 = _tag(tags, "stage1:")
        self.stage2 = _tag(tags, "stage2:")
        self.loss = _tag(tags, "loss:")

    def key(self) -> tuple:
        # Rank-blind methods ignore their inert `rank:` tag also in the
        # grouping: a phys_diag run tagged rank:1 and an untagged one are the
        # same trained model and must share one row.
        rank = None if self.method in RANK_BLIND_METHODS else self.rank
        return (self.method, self.transform, rank, self.stage1,
                self.stage2, self.loss)

    def block(self) -> str:
        for name, methods in BLOCKS:
            if methods is None or self.method in methods:
                return name
        return BLOCKS[-1][0]

    def core_label(self) -> str:
        if self.method == "deterministic":
            return "Deterministic"
        if self.method == "generative":
            # Nonparametric: no covariance at all, so no core to qualify.
            return "Generative sampler"
        if self.method == "phys_lora":
            # The rank is the whole point of this method, so it names the row
            # directly instead of being appended to a "Diagonal" base.
            return f"Low-rank $r={self.rank or 1}$"
        if self.method == "phys_cholesky":
            # Unstructured Sigma = L Lᵀ: the reference arm the structured
            # families approximate, so it names itself rather than being a
            # qualified "Diagonal". Only defined on the low-dimensional tasks.
            return "Full covariance"
        if self.method in RANK_BLIND_METHODS:
            # These methods swallow `rank` through **kwargs and build a
            # single-channel head regardless, so runs whose config carried an
            # inert `rank: 1` are plain diagonal models. Honouring the tag here
            # would advertise a low-rank baseline that was never trained.
            return {"phys_diag": "Diagonal",
                    "spectral_diag": "Fourier diagonal"}[self.method]
        base = {"phys_diag": "Diagonal",
                "spectral_diag": "Fourier diagonal"}.get(self.method)
        if base is None:  # whitened_* (SCORE)
            pretty = {"dct": "DCT", "dst": "DST",
                      "graph": r"Graph Mat\'ern"}.get(self.transform)
            base = "Fourier diagonal" if self.transform in (None, "fourier") \
                else f"{pretty or self.transform.upper()} diagonal"
        if self.rank:
            base += f" $+$ low-rank $r={self.rank}$"
        return base

    def compact_label(self) -> str:
        """Short row label for the wide grid table, where width is binding."""
        if self.method == "deterministic":
            return "Deterministic"
        if self.method == "generative":
            return "Generative"
        if self.method == "phys_lora":
            return f"Low-rank $r{{=}}{self.rank or 1}$"
        if self.method == "phys_cholesky":
            return "Full cov."
        if self.method in RANK_BLIND_METHODS:
            return {"phys_diag": "Diagonal", "spectral_diag": "Fourier"}[self.method]
        pretty = {"dct": "DCT", "dst": "DST",
                  "graph": r"Graph Mat\'ern"}.get(self.transform)
        base = "Fourier" if self.transform in (None, "fourier") \
            else pretty or self.transform.upper()
        return f"{base} $+\\,r{{=}}{self.rank}$" if self.rank else base

    def objective(self) -> str:
        """The score the *core* was fitted with.

        For staged runs that is the stage-2 loss: stage 1 only sees the
        marginals, so it cannot be what a core is judged on. Comparisons are
        only meaningful within one objective.
        """
        obj = self.stage2 or self.loss or "?"
        return "kernel_score" if obj in _KERNEL_SCORE_VARIANTS else obj

    def score_label(self) -> str:
        """The score identifying the row, mirroring the run-name convention."""
        if self.stage2:
            variant = _KERNEL_SCORE_VARIANTS.get(self.stage2)
            name = ("KS" if variant else
                    {"nll": "NLL", "kernel_score": "KS"}.get(self.stage2,
                                                             self.stage2))
            # Qualifiers that separate two rows sharing a core label. They go in
            # the label, not only in the block heading, because the CSV has no
            # block column: ("KS", core) has to be unique on its own there.
            # The raw stage-2 name would also be LaTeX-unsafe (underscores).
            quals = []
            if variant:
                quals.append(variant)
            # The stage-1 loss only distinguishes rows where it deviates.
            if self.stage1 and self.stage1 != "marginal_kernel_score":
                quals.append(f"s1 {self.stage1}")
            return f"{name} ({', '.join(quals)})" if quals else name
        return {"nll": "NLL", "kernel_score": "KS", "mse": "MSE",
                "crps": "CRPS", "energy_score": "ES"}.get(
                    self.loss or "", self.loss or "---")


def _run_features(run, tags: Sequence[str]) -> Optional[str]:
    """The forecasting mode of a run: from the tag, else from its config.

    Older runs have no ``features:`` tag, and filtering on the tag alone would
    silently drop them from *both* mode tables. Every run carries
    ``datamodule.features`` in its logged config, so that is the fallback.
    """
    tagged = _tag(tags, "features:")
    if tagged is not None:
        return tagged
    dm = run.config.get("datamodule")
    return dm.get("features") if isinstance(dm, dict) else None


#: Loss suffixes a config slug can end in, longest first so that
#: ``_kernel_score`` is not misread as loss ``score``. The order is
#: load-bearing for the kernel-score variants too: the match is the first hit,
#: so ``dense_kernel_score`` has to precede ``kernel_score`` or a slug would
#: split as loss ``kernel_score`` with ``_dense`` still glued to the method
#: name -- which then misses _METHOD_TRANSFORM and renders under an unrelated
#: core label instead of raising.
_SLUG_LOSSES = ("dense_kernel_score", "sample_kernel_score", "kernel_score",
                "energy_score", "crps", "nll", "mse")

#: Transform implied by a SCORE (`whitened_*`) method name; mirrors the
#: run-name convention used by src/train/train.py and the checkpoint tree.
_METHOD_TRANSFORM = {
    "whitened_spectral_diag": "fourier",
    "whitened_dct": "dct",
    "whitened_dst": "dst",
    "whitened_graph_matern": "graph",
}


def _slug_to_tags(slug: str) -> List[str]:
    """Facet tags recovered from a checkpoint-tree config slug.

    The checkpoint tree names configurations by the run-name convention
    (``whitened_dct_lora2_s1crps_nll``); runs stored there carry their scores
    in ``metrics.json`` instead of W&B tags, so the facets are parsed back out
    of the directory name.
    """
    rest = slug
    loss = next((l for l in _SLUG_LOSSES if rest.endswith("_" + l)), None)
    if loss:
        rest = rest[: -len(loss) - 1]
    stage1 = None
    if rest.endswith("_s1crps"):
        stage1, rest = "crps", rest[: -len("_s1crps")]
    rank = None
    m = re.search(r"_(?:lora|r)(\d+)$", rest)
    if m:
        rank, rest = m.group(1), rest[: m.start()]
    method = rest
    tags = [f"method:{method}"]
    transform = _METHOD_TRANSFORM.get(method)
    if transform:
        tags.append(f"transform:{transform}")
    if rank:
        tags.append(f"rank:{rank}")
    if method.startswith("whitened"):
        # Two-stage runs: the trailing loss is the stage-2 loss and stage 1
        # defaults to the marginal kernel score unless the slug says s1crps.
        tags.append(f"stage1:{stage1 or _DEFAULT_STAGE1_LOSS_TABLE}")
        if loss:
            tags.append(f"stage2:{loss}")
    else:
        if stage1:
            tags.append(f"stage1:{stage1}")
        if loss:
            tags.append(f"loss:{loss}")
    return tags


def fetch_dir(root: str, dataset: str) -> Dict[tuple, List[dict]]:
    """Finished runs from a checkpoint tree, grouped like :func:`fetch`.

    Walks ``<root>/<dataset>/<slug>/seed<N>/metrics.json``. This is the data
    source for experiments whose runs are not facet-tagged in W&B (e.g.
    station post-processing); ``metrics.json`` carries the same test scores
    the W&B summary would.
    """
    grouped: Dict[tuple, List[dict]] = defaultdict(list)
    n = unfinished = 0
    base = os.path.join(root, dataset)
    for slug in sorted(os.listdir(base)):
        sdir = os.path.join(base, slug)
        if not os.path.isdir(sdir):
            continue
        tags = _slug_to_tags(slug)
        for seed_dir in sorted(os.listdir(sdir)):
            mfile = os.path.join(sdir, seed_dir, "metrics.json")
            if not seed_dir.startswith("seed"):
                continue
            if not os.path.isfile(mfile):
                unfinished += 1
                continue
            with open(mfile) as fh:
                test = json.load(fh).get("test", {})
            cfg = Config(tags)
            grouped[cfg.key()].append(
                {"cfg": cfg, "seed": seed_dir[4:], "run": None,
                 "metrics": {k: _num(test.get(k))
                             for k in ("nll", "crps", "energy_score",
                                       "kernel_score", "variogram_score",
                                       "mse", "mae")}})
            n += 1
    print(f"[{dataset}] {n} runs in {len(grouped)} configurations "
          f"(from {base}, {unfinished} seed dirs without metrics.json)",
          file=sys.stderr)
    return grouped


def fetch(dataset: str, features: Optional[str]) -> Dict[tuple, List[dict]]:
    """Finished runs of one dataset, grouped by configuration."""
    api = wandb.Api(timeout=60)
    filters: Dict[str, object] = {"tags": f"dataset:{dataset}"}
    if REQUIRE_TAGS:
        filters = {"$and": [filters] + [{"tags": t} for t in sorted(REQUIRE_TAGS)]}
    runs = api.runs(ENTITY_PROJECT, filters=filters, per_page=200)

    grouped: Dict[tuple, List[dict]] = defaultdict(list)
    skipped_excluded = skipped_unfinished = 0
    for run in runs:
        tags = list(run.tags)
        if EXCLUDE_TAGS & set(tags) or REQUIRE_TAGS - set(tags):
            skipped_excluded += 1
            continue
        if features is not None and _run_features(run, tags) != features:
            continue
        if "evaluation/test_crps" not in run.summary:
            skipped_unfinished += 1
            continue
        cfg = Config(tags)
        grouped[cfg.key()].append(
            {"cfg": cfg, "seed": _tag(tags, "seed:"), "run": run,
             "metrics": {k: _num(run.summary.get(f"evaluation/test_{k}"))
                         for k in ("nll", "crps", "energy_score",
                                   "kernel_score", "variogram_score",
                                   "mse", "mae")}}
        )
    print(f"[{dataset}{'/' + features if features else ''}] "
          f"{sum(len(v) for v in grouped.values())} runs in "
          f"{len(grouped)} configurations "
          f"(excluded {skipped_excluded}, unfinished {skipped_unfinished})",
          file=sys.stderr)
    return grouped


#: Largest per-coordinate log score that is a *score* rather than a symptom.
#: A Gaussian density evaluated on its own scale gives O(1) nats per
#: coordinate (roughly -3.3 to +1.6 across the tasks here), so anything past
#: this bound means the covariance went numerically singular and the number is
#: an artefact of that, not a measurement to average.
#:
#: `_num` already drops the non-finite case (a Dirac's log score, a diverged
#: fit's `inf`). It is not enough on its own: a diverged fit can also report a
#: *finite* but astronomically large value that passes every finiteness check.
NLL_PER_D_LIMIT = 1e3

#: Whether one degenerate seed condemns its configuration's whole log-score cell.
#:
#: It has to: the seeds of a diverging configuration are often all far outside
#: the healthy range while only some of them clear NLL_PER_D_LIMIT, so dropping
#: *per seed* would report a clean-looking mean from the "surviving" seeds,
#: which is worse than reporting nothing. Seeds of one configuration differ only
#: by initialization, so a single diverging seed says the configuration's
#: density is unstable, not that that seed was unlucky.
NLL_DROP_WHOLE_CONFIG = True


_SEED_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
               6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def _seed_word(grouped: Dict[tuple, List[dict]]) -> str:
    """The modal seed count of a table, spelled out for the caption.

    Counted rather than hard-coded, because the experiments differ in their
    number of seeds. The mode rather than the max, because a table normally
    has one seed count and a handful of thinner rows, which the
    dagger/ddagger footnote already marks.
    """
    counts = [len(e) for e in grouped.values() if e and e[0].get("metrics")]
    if not counts:
        return "three"
    n = statistics.mode(counts)
    return _SEED_WORDS.get(n, str(n))


def aggregate(entries: List[dict], d: int) -> Dict[str, Tuple[Optional[float], Optional[float], int]]:
    """Mean, standard deviation and seed count per metric.

    Entries whose metric is missing are skipped rather than counted, so the
    reported ``n`` is the number of seeds actually behind a cell. The log score
    additionally drops values past ``NLL_PER_D_LIMIT``; a cell that loses every
    seed that way prints empty, like one that never had a finite value.
    """
    out = {}
    for key, _ in COLUMNS:
        source = "nll" if key == "nll_per_d" else key
        values = [e["metrics"][source] for e in entries
                  if e["metrics"].get(source) is not None]
        if key == "nll_per_d":
            values = [v / d for v in values]
            kept = [v for v in values if abs(v) <= NLL_PER_D_LIMIT]
            if len(kept) < len(values):
                cfg = entries[0]["cfg"]
                bad = len(values) - len(kept)
                if NLL_DROP_WHOLE_CONFIG:
                    kept = []
                print(f"  dropped the log score of {cfg.core_label()} / "
                      f"{cfg.score_label()}: {bad} of {len(values)} seeds have "
                      f"|nll/d| > {NLL_PER_D_LIMIT:g} (singular covariance, not "
                      f"a score)", file=sys.stderr)
            values = kept
        if not values:
            out[key] = (None, None, 0)
        else:
            sd = statistics.stdev(values) if len(values) > 1 else None
            out[key] = (statistics.mean(values), sd, len(values))
    return out


#: Columns printed with an explicit sign, because they genuinely take both.
#: Only the log score does: the energy score cannot be negative. The log score
#: per coordinate is about -1.3 on ERA5 and +1.5 on EUPPBench, and
#: a column mixing the two without a sign reads as if every entry were positive.
#: Listed explicitly rather than inferred from the precision, so `--digits 3`
#: does not put a `+` on CRPS and MSE, which cannot be negative.
_SIGNED_COLUMNS = {"nll_per_d"}


def _fmt(mean: Optional[float], sd: Optional[float], n: int, digits: int,
         signed: bool = False) -> str:
    if mean is None:
        return ""
    body = f"{mean:+.{digits}f}" if signed else f"{mean:.{digits}f}"
    if sd is not None:
        body += f" \\pm {sd:.{digits}f}"
    marker = {1: "^{\\dagger}", 2: "^{\\ddagger}"}.get(n, "")
    return f"${body}{marker}$"


def to_latex(grouped: Dict[tuple, List[dict]], dataset: str,
             features: Optional[str], d: int) -> str:
    rows_by_block: Dict[str, List[tuple]] = defaultdict(list)
    used_columns = set()
    for entries in grouped.values():
        cfg = entries[0]["cfg"]
        stats = aggregate(entries, d)
        for key, _ in COLUMNS:
            if stats[key][0] is not None:
                used_columns.add(key)
        rows_by_block[cfg.block()].append((cfg, stats))

    columns = [(k, h) for k, h in COLUMNS if k in used_columns]
    digits = {"nll_per_d": 3, "crps": 5, "energy_score": 3,
              "kernel_score": 5, "variogram_score": 5, "mse": 5}
    if "n" in DIGITS_OVERRIDE:
        digits = {k: DIGITS_OVERRIDE["n"] for k in digits}

    label = dataset + (f"-{features}" if features else "")
    head = " & ".join(h + r" $\downarrow$" for _, h in columns)
    # Raw dataset tags are not caption-safe (the underscore in depth_bts is a
    # LaTeX error outside math mode); print the paper name instead.
    pretty = {"era5": "ERA5", "depth_bts": "NYU Depth v2 (BTS)",
              "depth_bts_crop": "NYU Depth v2 (BTS, Eigen crop)",
              "depth_bts_crop": "NYU Depth v2 (BTS, Eigen crop)",
              "euppbench": "EUPPBench station post-processing",
              "etth1_S_T96": r"ETTh1, univariate, $T = 96$",
              "etth1_S_T192": r"ETTh1, univariate, $T = 192$",
              "etth1_M_T96": r"ETTh1, multivariate, $T = 96$",
              "etth1_M_T192": r"ETTh1, multivariate, $T = 192$",
              "etth2_S_T96": r"ETTh2, univariate, $T = 96$",
              "etth2_S_T192": r"ETTh2, univariate, $T = 192$",
              "etth2_M_T96": r"ETTh2, multivariate, $T = 96$",
              "etth2_M_T192": r"ETTh2, multivariate, $T = 192$",
              "ettm1_S_T96": r"ETTm1, univariate, $T = 96$",
              "ettm1_S_T192": r"ETTm1, univariate, $T = 192$",
              "ettm1_M_T96": r"ETTm1, multivariate, $T = 96$",
              "ettm1_M_T192": r"ETTm1, multivariate, $T = 192$",
              "ettm2_S_T96": r"ETTm2, univariate, $T = 96$",
              "ettm2_S_T192": r"ETTm2, univariate, $T = 192$",
              "ettm2_M_T96": r"ETTm2, multivariate, $T = 96$",
              "ettm2_M_T192": r"ETTm2, multivariate, $T = 192$",
              "illness_M_T24": r"National illness, multivariate, $T = 24$",
              "illness_M_T36": r"National illness, multivariate, $T = 36$",
              "weather_M_T96": r"Weather, multivariate, $T = 96$",
              "weather_M_T192": r"Weather, multivariate, $T = 192$",
              "electricity_M_T96": r"Electricity, multivariate, $T = 96$",
              "electricity_M_T192": r"Electricity, multivariate, $T = 192$",
              "traffic_M_T96": r"Traffic, multivariate, $T = 96$",
              "traffic_M_T192": r"Traffic, multivariate, $T = 192$"}.get(
        dataset, dataset[:3].upper() + dataset[3:])
    # Counted from the table rather than guessed from the dataset name, since
    # the experiments differ in their number of seeds.
    nseeds = _seed_word(grouped)
    lines = [
        r"\begin{table*}[ht]", r"\centering",
        rf"\caption{{Test scores on {pretty} ($d={d:,}$)."
        r" The log-score is reported per coordinate, all scores are"
        r" lower-is-better, and we report mean and standard deviation over"
        rf" {nseeds} seeds unless marked otherwise. The CRPS column is constant"
        r" across the marginally standardized variants by"
        r" Lemma~\ref{lem:corr-core}\,(i), which fixes the marginals in the"
        r" first stage.}",
        rf"\label{{tab:{label}}}", r"\small",
        r"\adjustbox{max width=\linewidth}{%",
        r"\begin{tabular}{ll" + "r" * len(columns) + "}",
        r"\toprule",
        rf"Core & Score & {head} \\",
        r"\midrule",
    ]
    for block_name, _ in BLOCKS:
        rows = rows_by_block.get(block_name)
        if not rows:
            continue
        rows.sort(key=lambda r: (r[0].method, r[0].transform or "",
                                 int(r[0].rank or 0), r[0].stage1 or "",
                                 r[0].stage2 or r[0].loss or ""))
        lines.append(rf"\multicolumn{{{2 + len(columns)}}}{{l}}{{\emph{{{block_name}}}}}\\")
        for cfg, stats in rows:
            cells = [_fmt(*stats[k], digits[k], k in _SIGNED_COLUMNS)
                     for k, _ in columns]
            lines.append(f"{cfg.core_label()} & {cfg.score_label()} & "
                         + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines.pop()
    lines += [
        r"\bottomrule", r"\end{tabular}}", r"\\[2pt]",
        r"{\footnotesize $^{\dagger}$ single seed, $^{\ddagger}$ two seeds.}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


def to_csv(grouped: Dict[tuple, List[dict]], d: int, path: str) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["core", "score", "n_seeds"]
                        + [c for c, _ in COLUMNS]
                        + [f"{c}_std" for c, _ in COLUMNS])
        for entries in sorted(grouped.values(),
                              key=lambda e: (e[0]["cfg"].block(),
                                             e[0]["cfg"].core_label())):
            cfg = entries[0]["cfg"]
            stats = aggregate(entries, d)
            # Seed count from the aggregate, not len(entries): planned-but-
            # unfinished rows carry one metric-less sentinel entry.
            n_seeds = max(stats[c][2] for c, _ in COLUMNS)
            writer.writerow(
                [cfg.core_label(), cfg.score_label(), n_seeds]
                + [stats[c][0] for c, _ in COLUMNS]
                + [stats[c][1] for c, _ in COLUMNS]
            )


def to_latex_two_modes(dataset: str, grouped_by_mode: Dict[str, Dict[tuple, List[dict]]],
                       dims: Dict[str, int], metrics: Sequence[str]) -> str:
    """Both time-series modes side by side, one column group each.

    The point of running univariate and multivariate is the contrast between
    them, so they belong in one table -- a reader should not have to hold two
    tables in mind to see that the transform matters in one mode and not the
    other.
    """
    modes = [m for m in ("M", "S") if m in grouped_by_mode]
    titles = {"M": "Multivariate", "S": "Univariate"}
    digits = {"nll_per_d": 3, "crps": 5, "energy_score": 3,
              "kernel_score": 5, "variogram_score": 5, "mse": 5}
    short = {"nll_per_d": r"NLL/$d$", "crps": "CRPS", "energy_score": "ES",
             "variogram_score": "VS", "mse": "MSE"}

    # Row order follows the first mode; a configuration missing in the other
    # mode simply leaves its cells empty rather than shifting the rows.
    rows_by_block: Dict[str, List[tuple]] = defaultdict(list)
    for key, entries in grouped_by_mode[modes[0]].items():
        rows_by_block[entries[0]["cfg"].block()].append((key, entries[0]["cfg"]))

    ncols = 2 + len(modes) * len(metrics)
    group_head = " & ".join(
        rf"\multicolumn{{{len(metrics)}}}{{c}}{{{titles[m]} ($d={dims[m]:,}$)}}"
        for m in modes)
    metric_head = " & ".join(
        " & ".join(short[k] + r" $\downarrow$" for k in metrics) for _ in modes)

    lines = [
        r"\begin{table*}[ht]", r"\centering",
        rf"\caption{{Test scores on {dataset}, both forecasting modes. The"
        r" log-score is reported per coordinate, all scores are"
        r" lower-is-better, and we report the mean over three seeds. The CRPS"
        r" column is constant across the marginally standardized variants by"
        r" Lemma~\ref{lem:corr-core}\,(i), which fixes the marginals in the"
        r" first stage.}",
        rf"\label{{tab:{dataset}}}", r"\small",
        r"\adjustbox{max width=\linewidth}{%",
        r"\begin{tabular}{ll" + "r" * (len(modes) * len(metrics)) + "}",
        r"\toprule",
        rf"Core & Score & {group_head} \\",
        rf"\cmidrule(lr){{3-{2 + len(metrics)}}}"
        rf"\cmidrule(lr){{{3 + len(metrics)}-{2 + 2 * len(metrics)}}}",
        rf" & & {metric_head} \\",
        r"\midrule",
    ]
    for block_name, _ in BLOCKS:
        rows = rows_by_block.get(block_name)
        if not rows:
            continue
        rows.sort(key=lambda r: (r[1].method, r[1].transform or "",
                                 int(r[1].rank or 0), r[1].stage1 or "",
                                 r[1].stage2 or r[1].loss or ""))
        lines.append(rf"\multicolumn{{{ncols}}}{{l}}{{\emph{{{block_name}}}}}\\")
        for key, cfg in rows:
            cells = []
            for mode in modes:
                entries = grouped_by_mode[mode].get(key)
                stats = aggregate(entries, dims[mode]) if entries else None
                for k in metrics:
                    if stats is None or stats[k][0] is None:
                        cells.append("")
                    else:
                        cells.append(f"${stats[k][0]:+.{digits[k]}f}$" if k == "nll_per_d"
                                     else f"${stats[k][0]:.{digits[k]}f}$")
            lines.append(f"{cfg.core_label()} & {cfg.score_label()} & "
                         + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    return "\n".join(lines)


def to_latex_grid(groups: List[tuple], metrics: Sequence[str], label: str,
                  caption: str, mark_best: bool = True) -> str:
    """One table over several (dataset, mode) column groups.

    ``groups`` is a list of ``(title, d, grouped)``. Used for the time-series
    main table, where differences between data sets and modes are only
    visible side by side. Row order follows the first
    group; a configuration missing elsewhere leaves empty cells.
    """
    # Fewer decimals than the per-setting tables: at four column groups the
    # width is the binding constraint, and the standard deviations live in the
    # appendix CSVs anyway.
    digits = {"nll_per_d": 2, "crps": 4, "energy_score": 2,
              "kernel_score": 4, "variogram_score": 4, "mse": 4}
    short = {"nll_per_d": r"NLL/$d$", "crps": "CRPS", "energy_score": "ES",
             "kernel_score": "KS", "variogram_score": "VS", "mse": "MSE"}
    # Precision at which two entries count as tied for the bold, independent of
    # how many decimals a column prints.
    TIE_DIGITS = 2
    per = len(metrics)
    ncols = 2 + len(groups) * per
    # Twelve numeric columns still fit across a two-column page, but only at
    # footnotesize and with the default column padding cut back.
    size = r"\footnotesize" if len(groups) * per > 8 else r"\small"

    rows_by_block: Dict[str, List[tuple]] = defaultdict(list)
    for key, entries in groups[0][2].items():
        rows_by_block[entries[0]["cfg"].block()].append((key, entries[0]["cfg"]))

    # Ties are judged at two decimals, deliberately coarser than most columns
    # print. The claim the marking makes is therefore "indistinguishable from
    # the best at a hundredth", not "lowest at the printed precision" -- so a
    # CRPS column whose entries all agree to 0.25 marks all of them, baselines
    # included, rather than picking a winner the seed spread cannot support.
    best: Dict[tuple, float] = {}
    if mark_best:
        for gi, (_, d, grouped) in enumerate(groups):
            for mi, metric in enumerate(metrics):
                vals = [aggregate(entries, d)[metric][0]
                        for entries in grouped.values()]
                vals = [round(v, TIE_DIGITS) for v in vals if v is not None]
                if vals:
                    best[(gi, mi)] = min(vals)

    if per == 1:
        head = " & ".join(t for t, _, _ in groups)
        sub = None
    else:
        head = " & ".join(rf"\multicolumn{{{per}}}{{c}}{{{t}}}" for t, _, _ in groups)
        sub = " & ".join(" & ".join(short[k] for k in metrics) for _ in groups)

    lines = [
        r"\begin{table*}[ht]", r"\centering",
        rf"\caption{{{caption}}}", rf"\label{{tab:{label}}}", size,
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\adjustbox{max width=\linewidth}{%",
        r"\begin{tabular}{ll" + "r" * (len(groups) * per) + "}",
        r"\toprule",
        rf"Core & Score & {head} \\",
    ]
    if sub is not None:
        lines.append("".join(
            rf"\cmidrule(lr){{{3 + i * per}-{2 + (i + 1) * per}}}"
            for i in range(len(groups))))
        lines.append(rf" & & {sub} \\")
    lines.append(r"\midrule")

    for block_name, _ in BLOCKS:
        rows = rows_by_block.get(block_name)
        if not rows:
            continue
        rows.sort(key=lambda r: (r[1].method, r[1].transform or "",
                                 int(r[1].rank or 0), r[1].stage1 or "",
                                 r[1].stage2 or r[1].loss or ""))
        lines.append(rf"\multicolumn{{{ncols}}}{{l}}{{\emph{{{block_name}}}}}\\")
        for key, cfg in rows:
            cells = []
            for gi, (_, d, grouped) in enumerate(groups):
                entries = grouped.get(key)
                stats = aggregate(entries, d) if entries else None
                for mi, k in enumerate(metrics):
                    if stats is None or stats[k][0] is None:
                        cells.append("")
                        continue
                    mean = stats[k][0]
                    sign = "+" if k == "nll_per_d" else ""
                    body = f"{mean:{sign}.{digits[k]}f}"
                    ref = best.get((gi, mi))
                    won = (ref is not None
                           and round(mean, TIE_DIGITS) <= ref + 1e-9)
                    body = f"\\mathbf{{{body}}}" if won else body
                    # Same convention as the per-setting tables: a cell backed
                    # by fewer than three seeds says so, so that an unreplicated
                    # entry cannot pass as a three-seed mean -- least of all one
                    # that carries the column's bold.
                    body += {1: "^{\\dagger}", 2: "^{\\ddagger}"}.get(stats[k][2], "")
                    cells.append(f"${body}$")
            lines.append(f"{cfg.compact_label()} & {cfg.score_label()} & "
                         + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}}", r"\\[2pt]",
              r"{\footnotesize $^{\dagger}$ single seed, "
              r"$^{\ddagger}$ two seeds.}", r"\end{table*}"]
    return "\n".join(lines)


#: Ranks kept in the headline table. The full sweep lives in the appendix, both
#: as a figure and as the loss-wise table; three points are enough to show that
#: SCORE does not need a large rank to win.
MAIN_RANKS = {0, 2, 16}

#: Transforms kept in the headline table. dst/uhaar/dct1/dct4/random are
#: single-seed probes and belong to the basis ablation, not the main result.
MAIN_TRANSFORMS = {None, "fourier", "dct"}


def _significance_bold(entries_by_key, d, metric, digits) -> set:
    """Keys whose value is within one seed deviation of the column's best.

    Returns the empty set when the whole column fits inside that band: a
    column with no separable winner should carry no bold at all, rather than
    awarding one to whichever run happened to land lowest. The deviation is
    taken from the best row itself, falling back to the column median when that
    row has too few seeds to have one.
    """
    cand = []
    for key, (entries, dd) in entries_by_key.items():
        mean, sd, n = aggregate(entries, dd)[metric]
        if mean is not None and n >= 3:
            cand.append((mean, sd, key))
    if len(cand) < 2:
        return set()
    cand.sort()
    best, best_sd, _ = cand[0]
    if best_sd is None or best_sd == 0:
        sds = [s for _, s, _ in cand if s]
        best_sd = statistics.median(sds) if sds else 0.0
    band = best + best_sd
    if cand[-1][0] <= band:          # everything within one deviation
        return set()
    return {k for v, _, k in cand if v <= band}


def to_latex_main(groups: List[tuple], metrics: Sequence[str], label: str,
                  caption: str) -> str:
    """Headline table: log-score-fitted cores only, few ranks, few columns.

    The kernel-score arm is left out on purpose. Loss-matched, its cores sit
    within one to two seed deviations of each other, so twelve numbers would
    carry what one sentence of text carries. The log score is where the
    differences are clear, and the variogram score corroborates them without
    being any method's training objective.
    """
    digits = {"nll_per_d": 2, "crps": 4, "energy_score": 2,
              "kernel_score": 4, "variogram_score": 4, "mse": 4}
    short = {"nll_per_d": r"NLL/$d$", "crps": "CRPS", "energy_score": "ES",
             "kernel_score": "KS", "variogram_score": "VS", "mse": "MSE"}
    per = len(metrics)
    ncols = 1 + len(groups) * per

    def keep(cfg: Config) -> bool:
        if cfg.method in ("deterministic", "generative"):
            return True                      # baseline reference rows
        if cfg.objective() != "nll":
            return False
        if cfg.stage2 and cfg.stage1 and cfg.stage1 != _DEFAULT_STAGE1_LOSS_TABLE:
            return False                     # stage-1 variants: appendix
        if int(cfg.rank or 0) not in MAIN_RANKS:
            return False
        return cfg.transform in MAIN_TRANSFORMS

    cfg_by_key: Dict[tuple, Config] = {}
    for _, _, grouped in groups:
        for key, entries in grouped.items():
            cfg_by_key.setdefault(key, entries[0]["cfg"])
    rows_by: Dict[str, List[tuple]] = defaultdict(list)
    for key, cfg in cfg_by_key.items():
        if keep(cfg):
            rows_by[cfg.block()].append((key, cfg))

    bold: Dict[tuple, set] = {}
    # Best physical baseline per column, underlined. Bold alone says "nobody
    # beat this"; the underline gives the eye the reference point, so the gap
    # between the strongest physical low-rank core and SCORE is
    # read off directly instead of computed.
    base: Dict[tuple, Optional[tuple]] = {}
    phys_block = BLOCKS[0][0]
    for gi, (_, d, grouped) in enumerate(groups):
        for mi, metric in enumerate(metrics):
            sub = {key: (grouped[key], d) for key in cfg_by_key
                   if key in grouped and keep(cfg_by_key[key])}
            bold[(gi, mi)] = _significance_bold(sub, d, metric, digits)
            cand = []
            for key, (entries, dd) in sub.items():
                cfg = cfg_by_key[key]
                if cfg.block() != phys_block or cfg.method == "deterministic":
                    continue
                mean, _, n = aggregate(entries, dd)[metric]
                if mean is not None and n >= 3:
                    cand.append((mean, key))
            base[(gi, mi)] = min(cand)[1] if cand else None

    lines = [
        r"\begin{table*}[ht]", r"\centering",
        rf"\caption{{{caption}}}", rf"\label{{tab:{label}}}", r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\adjustbox{max width=\linewidth}{%",
        r"\begin{tabular}{l" + "r" * (len(groups) * per) + "}",
        r"\toprule",
        "Core & " + " & ".join(
            rf"\multicolumn{{{per}}}{{c}}{{{t}}}" for t, _, _ in groups) + r" \\",
        "".join(rf"\cmidrule(lr){{{2 + i * per}-{1 + (i + 1) * per}}}"
                for i in range(len(groups))),
        " & " + " & ".join(" & ".join(short[k] for k in metrics)
                           for _ in groups) + r" \\",
        r"\midrule",
    ]

    first = True
    for block_name, _ in BLOCKS:
        rows = rows_by.get(block_name)
        if not rows:
            continue
        if not first:
            # A thin rule reads as a group boundary without the vertical space
            # of a \midrule, which at this row count would dominate the table.
            lines.append(rf"\cmidrule(l){{1-{ncols}}}")
        first = False
        rows.sort(key=lambda r: (r[1].method != "deterministic",
                                 r[1].method, r[1].transform or "",
                                 int(r[1].rank or 0)))
        lines.append(rf"\multicolumn{{{ncols}}}{{l}}{{\emph{{{block_name}}}}}\\")
        for key, cfg in rows:
            cells = []
            for gi, (_, d, grouped) in enumerate(groups):
                entries = grouped.get(key)
                stats = aggregate(entries, d) if entries else None
                for mi, k in enumerate(metrics):
                    if stats is None or stats[k][0] is None:
                        cells.append("")
                        continue
                    mean, _, n = stats[k]
                    sign = "+" if k == "nll_per_d" else ""
                    body = f"{mean:{sign}.{digits[k]}f}"
                    if key in bold[(gi, mi)]:
                        body = f"\\mathbf{{{body}}}"
                    if key == base[(gi, mi)]:
                        body = f"\\underline{{{body}}}"
                    body += {1: "^{\\dagger}", 2: "^{\\ddagger}"}.get(n, "")
                    cells.append(f"${body}$")
            lines.append(rf"\quad {cfg.compact_label()} & "
                         + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\\[2pt]",
              r"{\footnotesize $^{\dagger}$ single seed, "
              r"$^{\ddagger}$ two seeds; only three-seed entries compete for "
              r"the bold.}", r"\end{table*}"]
    return "\n".join(lines)


def to_latex_ablation(groups: List[tuple], metrics: Sequence[str], label: str,
                      caption: str) -> str:
    """Two-factor ablation: fitting objective x parameterization family.

    One row per (objective, family), holding the best arm of that family. The
    arm is selected on the first metric and its remaining values are reported
    as they are, so a row describes one model rather than a per-column
    envelope. Reading down a block isolates the parameterization at a fixed
    objective; reading across blocks isolates the objective.
    """
    digits = {"nll_per_d": 2, "crps": 4, "energy_score": 2,
              "kernel_score": 4, "variogram_score": 4, "mse": 4}
    short = {"nll_per_d": r"NLL/$d$", "crps": "CRPS", "energy_score": "ES",
             "kernel_score": "KS", "variogram_score": "VS", "mse": "MSE"}
    # Short labels throughout: this table has to fit one column of a two-column
    # page, and at six rows the long block names alone overflow it by ~90pt.
    # The caption carries what the abbreviations mean.
    label_of = {"Physical space": "Physical",
                "Transformed domain, unnormalized": "Transformed",
                "Marginally standardized (ours)": "Standardized"}
    per = len(metrics)
    ncols = 1 + len(groups) * per
    groups = [(t.replace(" multi", " M").replace(" uni", " S"), d, g)
              for t, d, g in groups]

    def keep(cfg: Config) -> bool:
        if cfg.method == "deterministic":
            return False                     # no covariance to ablate
        if cfg.stage2 and cfg.stage1 and cfg.stage1 != _DEFAULT_STAGE1_LOSS_TABLE:
            return False                     # stage-1 variants: appendix
        return cfg.transform in MAIN_TRANSFORMS

    # best[(li, block, gi)] -> (key, stats) chosen on metrics[0]
    best: Dict[tuple, tuple] = {}
    for li, (obj, _) in enumerate(LOSS_GROUPS):
        for block_name, _m in BLOCKS:
            for gi, (_, d, grouped) in enumerate(groups):
                cand = []
                for key, entries in grouped.items():
                    cfg = entries[0]["cfg"]
                    if cfg.objective() != obj or cfg.block() != block_name \
                            or not keep(cfg):
                        continue
                    stats = aggregate(entries, d)
                    mean, _, n = stats[metrics[0]]
                    if mean is not None and n >= 3:
                        cand.append((mean, key, stats))
                if cand:
                    best[(li, block_name, gi)] = min(cand, key=lambda c: c[0])[1:]

    lines = [
        r"\begin{table}[t]", r"\centering",
        rf"\caption{{{caption}}}", rf"\label{{tab:{label}}}", r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\adjustbox{max width=\linewidth}{%",
        r"\begin{tabular}{l" + "r" * (len(groups) * per) + "}",
        r"\toprule",
        " & " + " & ".join(
            rf"\multicolumn{{{per}}}{{c}}{{{t}}}" for t, _, _ in groups) + r" \\",
        "".join(rf"\cmidrule(lr){{{2 + i * per}-{1 + (i + 1) * per}}}"
                for i in range(len(groups))),
    ]
    if per > 1:
        lines.append(" & " + " & ".join(" & ".join(short[k] for k in metrics)
                                        for _ in groups) + r" \\")
    else:
        lines.append(rf" & \multicolumn{{{len(groups)}}}{{c}}"
                     rf"{{{short[metrics[0]]}}} \\")
    lines.append(r"\midrule")

    for li, (obj, obj_title) in enumerate(LOSS_GROUPS):
        if obj == "mse" or not any(
                (li, b, gi) in best for b, _ in BLOCKS
                for gi in range(len(groups))):
            continue
        if li and lines[-1] != r"\midrule":
            lines.append(r"\addlinespace[2pt]")
        lines.append(rf"\multicolumn{{{ncols}}}{{l}}{{\emph{{{obj_title}}}}}\\")
        # Winner per column inside this objective group, so the bold answers
        # "which parameterization at a fixed objective" and nothing wider.
        for block_name, _m in BLOCKS:
            if not any((li, block_name, gi) in best
                       for gi in range(len(groups))):
                continue
            cells = []
            for gi in range(len(groups)):
                for mi, k in enumerate(metrics):
                    hit = best.get((li, block_name, gi))
                    if hit is None or hit[1][k][0] is None:
                        cells.append("")
                        continue
                    mean = hit[1][k][0]
                    col = [best[(li, b, gi)][1][k][0]
                           for b, _ in BLOCKS if (li, b, gi) in best
                           and best[(li, b, gi)][1][k][0] is not None]
                    sign = "+" if k == "nll_per_d" else ""
                    body = f"{mean:{sign}.{digits[k]}f}"
                    if col and round(mean, digits[k]) <= round(min(col), digits[k]):
                        body = f"\\mathbf{{{body}}}"
                    cells.append(f"${body}$")
            lines.append(rf"\quad {label_of.get(block_name, block_name)} & "
                         + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


def to_latex_loss_grid(groups: List[tuple], metrics: Sequence[str], label: str,
                       caption: str) -> str:
    """Grid table grouped by fitting objective first, method family second.

    The interleaved layout of :func:`to_latex_grid` puts KS- and NLL-fitted rows
    next to each other, which invites reading a column minimum across
    objectives. That comparison is not sound: a model fitted with the kernel
    score is not competing for the log score. Here the objective is the outer
    grouping, so a column reads top-to-bottom within one group only, and the
    bold is computed per group -- not per column.
    """
    digits = {"nll_per_d": 2, "crps": 4, "energy_score": 2,
              "kernel_score": 4, "variogram_score": 4, "mse": 4}
    short = {"nll_per_d": r"NLL/$d$", "crps": "CRPS", "energy_score": "ES",
             "kernel_score": "KS", "variogram_score": "VS", "mse": "MSE"}
    per = len(metrics)
    ncols = 1 + len(groups) * per

    # Union over column groups, not just the first: a configuration that ran on
    # ETTh2 but not on ETTh1 must still get a row rather than vanish.
    cfg_by_key: Dict[tuple, Config] = {}
    for _, _, grouped in groups:
        for key, entries in grouped.items():
            cfg_by_key.setdefault(key, entries[0]["cfg"])

    # Single-seed probes (one exploratory run, one column group -- the basis
    # probes) go into their own trailing block. Interleaved with the real
    # rows they drag nine empty cells each through the middle of the table.
    def is_probe(key) -> bool:
        # Seed count = max over ALL metrics: the deterministic baseline has an
        # infinite log-score (n = 0 on that column) but three seeds elsewhere,
        # and must not be classified as a probe.
        counts = []
        for _, d, grouped in groups:
            if key in grouped:
                stats = aggregate(grouped[key], d)
                counts.append(max(stats[m][2] for m in metrics))
        return bool(counts) and max(counts) <= 1

    probes: List[tuple] = []
    rows_by: Dict[tuple, List[tuple]] = defaultdict(list)
    for key, cfg in cfg_by_key.items():
        if is_probe(key):
            probes.append((key, cfg))
        else:
            rows_by[(cfg.objective(), cfg.block())].append((key, cfg))

    # Best per (loss group, column). Only configurations with all three seeds
    # compete: an unreplicated run must not carry the bold of a main table.
    best: Dict[tuple, float] = {}
    for li, (obj, _) in enumerate(LOSS_GROUPS):
        for gi, (_, d, grouped) in enumerate(groups):
            for mi, metric in enumerate(metrics):
                vals = []
                for key, entries in grouped.items():
                    if entries[0]["cfg"].objective() != obj:
                        continue
                    mean, _, n = aggregate(entries, d)[metric]
                    if mean is not None and n >= 3:
                        vals.append(round(mean, digits[metric]))
                # A single-row group has no competition; bolding its every
                # entry (e.g. Det) would be pure noise.
                if len(vals) >= 2:
                    best[(li, gi, mi)] = min(vals)

    size = r"\footnotesize" if len(groups) * per > 8 else r"\small"
    lines = [
        r"\begin{table*}[ht]", r"\centering",
        rf"\caption{{{caption}}}", rf"\label{{tab:{label}}}", size,
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\adjustbox{max width=\linewidth, max totalheight=.78\textheight}{%",
        r"\begin{tabular}{l" + "r" * (len(groups) * per) + "}",
        r"\toprule",
        "Core & " + " & ".join(
            rf"\multicolumn{{{per}}}{{c}}{{{t}}}" for t, _, _ in groups) + r" \\",
        "".join(rf"\cmidrule(lr){{{2 + i * per}-{1 + (i + 1) * per}}}"
                for i in range(len(groups))),
        " & " + " & ".join(" & ".join(short[k] for k in metrics)
                           for _ in groups) + r" \\",
        r"\midrule",
    ]

    def render_row(key, cfg, li) -> str:
        lab = cfg.compact_label()
        if cfg.stage2 and cfg.stage1 \
                and cfg.stage1 != _DEFAULT_STAGE1_LOSS_TABLE:
            lab += r", s1 CRPS"
        cells = []
        for gi, (_, d, grouped) in enumerate(groups):
            entries = grouped.get(key)
            stats = aggregate(entries, d) if entries else None
            for mi, k in enumerate(metrics):
                if stats is None or stats[k][0] is None:
                    cells.append("")
                    continue
                mean, _, n = stats[k]
                sign = "+" if k == "nll_per_d" else ""
                body = f"{mean:{sign}.{digits[k]}f}"
                ref = best.get((li, gi, mi)) if li is not None else None
                if n >= 3 and ref is not None \
                        and round(mean, digits[k]) <= ref + 1e-9:
                    body = f"\\mathbf{{{body}}}"
                body += {1: "^{\\dagger}", 2: "^{\\ddagger}"}.get(n, "")
                cells.append(f"${body}$")
        return f"\\quad {lab} & " + " & ".join(cells) + r" \\"

    for li, (obj, obj_title) in enumerate(LOSS_GROUPS):
        if not any(rows_by.get((obj, b)) for b, _ in BLOCKS):
            continue
        lines.append(rf"\multicolumn{{{ncols}}}{{l}}{{\textbf{{{obj_title}}}}}\\")
        first_block = True
        for block_name, _ in BLOCKS:
            rows = rows_by.get((obj, block_name))
            if not rows:
                continue
            if not first_block:
                # Thin rule between method families: at this row count the
                # italic sub-headers alone do not separate visually.
                lines.append(rf"\cmidrule(l){{1-{ncols}}}")
            first_block = False
            rows.sort(key=lambda r: (r[1].method, r[1].transform or "",
                                     int(r[1].rank or 0), r[1].stage1 or ""))
            lines.append(rf"\multicolumn{{{ncols}}}{{l}}"
                         rf"{{\quad\emph{{{block_name}}}}}\\")
            for key, cfg in rows:
                lines.append(render_row(key, cfg, li))
        lines.append(r"\midrule")
    if probes:
        # Exploratory one-off runs, collected at the end instead of scattered
        # through the blocks above; they never compete for a bold.
        probes.sort(key=lambda r: (r[1].objective(), r[1].method,
                                   r[1].transform or "", int(r[1].rank or 0)))
        obj_name = {"kernel_score": "kernel score", "nll": "log score",
                    "mse": "MSE"}
        lines.append(r"\multicolumn{%d}{l}{\textbf{Single-seed transform "
                     r"probes}}\\" % ncols)
        for key, cfg in probes:
            row = render_row(key, cfg, None)
            lines.append(row.replace(
                "\\quad ", f"\\quad ", 1).replace(
                " & ", f" ({obj_name.get(cfg.objective(), cfg.objective())}) & ", 1))
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}}", r"\\[2pt]",
              r"{\footnotesize $^{\dagger}$ single seed, "
              r"$^{\ddagger}$ two seeds; only three-seed entries compete for "
              r"the bold.}", r"\end{table*}"]
    return "\n".join(lines)


def to_latex_full(dataset: str, grouped_by_mode: Dict[str, Dict[tuple, List[dict]]],
                  dims: Dict[str, int]) -> str:
    """Complete per-dataset reference: one float per forecasting mode.

    Every configuration, every metric, mean +- standard deviation, rows
    grouped by fitting objective as in the loss-wise grid. Nothing is bolded -- this is the archival table the
    compact tables resolve to, and the seed spreads are its point, not a
    ranking. The two modes are separate floats because a float cannot break
    across pages and either panel alone already fills one: ~50 configurations
    at footnotesize is a page, both panels stacked silently clip.
    """
    digits = {"nll_per_d": 3, "crps": 5, "energy_score": 3,
              "kernel_score": 5, "variogram_score": 5, "mse": 5}
    titles = {"M": "Multivariate", "S": "Univariate"}
    modes = [m for m in ("M", "S") if m in grouped_by_mode]
    pretty = dataset[:3].upper() + dataset[3:]

    # Drop columns that are empty in both modes (same guard as the printer's;
    # depth has no variogram, ETT has all six).
    used: set = set()
    for m in modes:
        for entries in grouped_by_mode[m].values():
            st = aggregate(entries, dims[m])
            used |= {k for k, _ in COLUMNS if st[k][0] is not None}
    columns = [(k, h) for k, h in COLUMNS if k in used]
    ncols = 1 + len(columns)

    head = " & ".join(h + r" $\downarrow$" for _, h in columns)
    obj_name = {"kernel_score": "kernel score", "nll": "log score",
                "mse": "MSE"}
    floats: List[str] = []
    for mode in modes:
        grouped, d = grouped_by_mode[mode], dims[mode]
        caption = (
            rf"Complete results on {pretty}, {titles[mode].lower()} mode"
            rf" ($d={d:,}$): every configuration and metric, as mean $\pm$"
            rf" standard deviation over {_seed_word(grouped)} seeds unless"
            r" marked otherwise."
            r" Lower is better and the log-score is reported per coordinate."
            r" Rows are grouped by the score the core was fitted with, since"
            r" only rows within one group compete. CRPS and MSE depend only on"
            r" the marginals and the mean, which the first stage fixes for"
            r" every marginally standardized core"
            r" (Lemma~\ref{lem:corr-core}\,(i)). Exploratory single-seed"
            r" probes close the table with the fitting objective in"
            r" parentheses. Nothing is bolded: this is the reference table"
            r" behind the compact tables, and the seed spreads are its point.")
        lines = [
            r"\begin{table*}[p]", r"\centering",
            rf"\caption{{{caption}}}",
            rf"\label{{tab:{dataset}-full-{mode}}}",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{3pt}",
            r"\adjustbox{max width=\linewidth, max totalheight=.78\textheight}{%",
            r"\begin{tabular}{l" + "r" * len(columns) + "}",
            r"\toprule",
            rf"Core & {head} \\",
            r"\midrule",
        ]

        def is_probe(entries) -> bool:
            stats = aggregate(entries, d)
            return max(stats[k][2] for k, _ in columns) == 1

        probes: List[tuple] = []
        rows_by: Dict[tuple, List[tuple]] = defaultdict(list)
        for key, entries in grouped.items():
            cfg = entries[0]["cfg"]
            if is_probe(entries):
                probes.append((key, cfg))
            else:
                rows_by[(cfg.objective(), cfg.block())].append((key, cfg))

        def render_row(key, cfg, indent=r"\qquad", note="") -> str:
            lab = cfg.compact_label()
            if cfg.stage2 and cfg.stage1 \
                    and cfg.stage1 != _DEFAULT_STAGE1_LOSS_TABLE:
                lab += r", s1 CRPS"
            stats = aggregate(grouped[key], d)
            cells = [_fmt(*stats[k], digits[k], k in _SIGNED_COLUMNS)
                     for k, _ in columns]
            return f"{indent} {lab}{note} & " + " & ".join(cells) + r" \\"

        for obj, obj_title in LOSS_GROUPS:
            if not any(rows_by.get((obj, b)) for b, _ in BLOCKS):
                continue
            lines.append(rf"\multicolumn{{{ncols}}}{{l}}"
                         rf"{{\quad\textit{{{obj_title}}}}}\\")
            for block_name, _ in BLOCKS:
                rows = rows_by.get((obj, block_name))
                if not rows:
                    continue
                rows.sort(key=lambda r: (r[1].method, r[1].transform or "",
                                         int(r[1].rank or 0), r[1].stage1 or ""))
                lines.append(rf"\multicolumn{{{ncols}}}{{l}}"
                             rf"{{\quad\quad\emph{{{block_name}}}}}\\")
                for key, cfg in rows:
                    lines.append(render_row(key, cfg))
        if probes:
            probes.sort(key=lambda r: (r[1].objective(), r[1].method,
                                       r[1].transform or "",
                                       int(r[1].rank or 0)))
            lines.append(rf"\multicolumn{{{ncols}}}{{l}}"
                         rf"{{\quad\textit{{Single-seed transform probes}}}}\\")
            for key, cfg in probes:
                note = (" ("
                        f"{obj_name.get(cfg.objective(), cfg.objective())})")
                lines.append(render_row(key, cfg, note=note))

        lines += [r"\bottomrule", r"\end{tabular}}", r"\\[2pt]",
                  r"{\footnotesize $^{\dagger}$ single seed,"
                  r" $^{\ddagger}$ two seeds.}",
                  r"\end{table*}"]
        floats.append("\n".join(lines))
    return "\n\n".join(floats)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="dataset tag, e.g. era5, depth_bts, etth1")
    parser.add_argument("--features", choices=["M", "S"], default=None,
                        help="time series only: multivariate (M) or univariate (S)")
    parser.add_argument("--both-modes", action="store_true",
                        help="time series only: one table with M and S side by side")
    parser.add_argument("--full", action="store_true",
                        help="time series only: complete per-dataset reference "
                             "-- both modes as stacked panels, all metrics "
                             "with std, rows grouped by fitting objective")
    parser.add_argument("--grid", default=None,
                        help="comma-separated datasets: one table over all "
                             "(dataset, mode) column groups, e.g. etth1,etth2")
    parser.add_argument("--ablation", action="store_true",
                        help="with --grid: two-factor ablation, objective x "
                             "parameterization family, best arm per cell")
    parser.add_argument("--main", action="store_true",
                        help="with --grid: headline table -- log-score-fitted "
                             "cores only, ranks 0/2/16, significance bolding")
    parser.add_argument("--by-loss", action="store_true",
                        help="with --grid: group rows by fitting objective "
                             "first, so columns are read within one objective")
    parser.add_argument("--metrics", default="nll_per_d,crps",
                        help="columns for --both-modes (comma separated)")
    parser.add_argument("--from-dir", default=None, metavar="ROOT",
                        help="read <ROOT>/<dataset>/<slug>/seed*/metrics.json "
                             "from a shared checkpoint tree instead of W&B "
                             "(for runs without facet tags)")
    parser.add_argument("--require-tag", action="append", default=[],
                        metavar="TAG",
                        help="only runs carrying this W&B tag (repeatable), "
                             "e.g. --require-tag main2026")
    parser.add_argument("--from-dir-exact", default=None, metavar="DIR",
                        help="read <DIR>/<slug>/seed*/metrics.json, i.e. the "
                             "slug directory itself rather than <ROOT>/<dataset>. "
                             "Use it where the tree nests deeper than the "
                             "dataset name -- the univariate T-horizon campaign "
                             "lives at <dataset>/univariate/T<pred_len>/ -- and "
                             "the positional argument is then a pure label, "
                             "keyed in DIMENSION and used for the caption, the "
                             "\\label and the output file stem")
    parser.add_argument("--digits", type=int, default=None, metavar="N",
                        help="round every column of the single-dataset table to N "
                             "decimals; default is per-metric (5 for CRPS and the "
                             "kernel score, 3 for the log and energy scores), set "
                             "to each column's own spread. The CSV is unaffected "
                             "and always carries full precision.")
    parser.add_argument("--out", default=None,
                        help="directory for <dataset>.tex / .csv; default stdout only")
    args = parser.parse_args()
    REQUIRE_TAGS.update(args.require_tag)
    if args.digits is not None:
        DIGITS_OVERRIDE["n"] = args.digits

    if args.grid:
        specs = []
        for ds in [d.strip() for d in args.grid.split(",")]:
            for mode in ("M", "S"):
                if (ds, mode) not in DIMENSION:
                    continue
                # "ETTh1", not "ETTH1": only the leading acronym is capitalized.
                pretty = ds[:3].upper() + ds[3:]
                title = f"{pretty} {'multi' if mode == 'M' else 'uni'}"
                specs.append((title, DIMENSION[(ds, mode)], fetch(ds, mode)))
        metrics = [m.strip() for m in args.metrics.split(",")]
        caption = (r"Test scores on the ETT hourly benchmarks, both forecasting"
                   r" modes. The log-score is reported per coordinate and lower"
                   r" is better; means over three seeds unless marked"
                   r" otherwise. The log-score and the energy score both see"
                   r" the full predictive covariance. MSE sees only the mean,"
                   r" which the standardized models fix in the first stage; it"
                   r" is therefore constant down each standardized block and"
                   r" reads as a check that modeling the covariance costs no"
                   r" point accuracy, not as a ranking among cores. Per"
                   r" coordinate CRPS is omitted for the same reason --- the"
                   r" first stage fixes the marginals"
                   r" (Lemma~\ref{lem:corr-core}\,(i)) --- and is reported in"
                   r" the per-setting tables in the appendix. Bold marks every"
                   r" entry that matches the best in its column to two"
                   r" decimals; margins below that are within the"
                   r" seed-to-seed spread and should not be read as a ranking.")
        if args.ablation:
            caption = (
                r"Ablation of the two design choices on the ETT hourly"
                r" benchmarks, both forecasting modes: the score the core is"
                r" fitted with, and the covariance parameterization. Each row"
                r" holds the best three-seed arm of its family, selected on the"
                r" log score; lower is better and the log-score is per"
                r" coordinate. Reading down a block varies the"
                r" parameterization at a fixed objective, reading across blocks"
                r" varies the objective. The standardized core is best within"
                r" its own objective block in all eight cells. The finer"
                r" orderings have exceptions: under the kernel score on ETTh2"
                r" univariate the unnormalized core falls behind the physical"
                r" one, and on ETTh2 multivariate fitting with the log score is"
                r" worse than fitting with the kernel score for both"
                r" transform-domain families. \textbf{Bold} is the best"
                r" parameterization within its own objective block. M and S"
                r" denote the multivariate and univariate forecasting mode, and"
                r" `Standardized' is the marginally standardized core proposed"
                r" here. Stage-1 variants and the single-seed basis probes are"
                r" excluded and reported in the appendix.")
            latex = to_latex_ablation(specs, metrics, "etth-ablation", caption)
            out_name = "etth_ablation.tex"
        elif args.main:
            caption = (
                r"Test scores on the ETT hourly benchmarks, both forecasting"
                r" modes; lower is better, the log-score per coordinate, means"
                r" over three seeds unless marked otherwise. All cores shown are"
                r" fitted with the log score, so the columns compare like with"
                r" like; the deterministic row is fitted with MSE and serves as"
                r" a point-forecast reference. The variogram score"
                r" \citep{scheuerer2015variogram} is not a training objective of"
                r" any method here. \textbf{Bold} marks every entry within one"
                r" seed standard deviation of the best in its column; a column"
                r" with no bold has no separable winner. \underline{Underlined}"
                r" is the strongest physical baseline of that column, so the"
                r" margin over the best low-rank core in the data space can be"
                r" read off directly. Cores fitted with the"
                r" kernel score, the remaining ranks and the stage-1 variants"
                r" are in the appendix.")
            latex = to_latex_main(specs, metrics, "etth-main", caption)
            out_name = "etth_main.tex"
        elif args.by_loss:
            caption = (
                r"Complete ETT hourly results: every configuration, both"
                r" datasets and both forecasting modes. Lower is better; the"
                r" log-score is reported per coordinate and all entries are"
                r" means over three seeds unless marked otherwise. Rows are"
                r" grouped by the score the core was fitted with, since only"
                r" rows within one group compete -- a core fitted with the"
                r" kernel score is not optimizing the log-score column, and"
                r" those cells are diagnostics rather than contenders. Bold is"
                r" accordingly the best entry per column \emph{within its"
                r" group}. The variogram score"
                r" \citep{scheuerer2015variogram} is a training objective of no"
                r" method here. CRPS and MSE depend only on the marginals and"
                r" the mean, which the first stage fixes for every"
                r" standardized core (Lemma~\ref{lem:corr-core}\,(i)); they are"
                r" therefore constant down each standardized block by"
                r" construction and are given in the per-setting CSVs together"
                r" with the seed standard deviations. Exploratory single-seed"
                r" transform probes (a DST pilot under CRPS stage 1, unbalanced"
                r" Haar, rank-10 pilots; all"
                r" ETTh1 univariate) are collected in the final block with"
                r" their fitting objective in parentheses; they never compete"
                r" for a bold.")
            latex = to_latex_loss_grid(specs, metrics, "etth-grid-loss", caption)
            out_name = "etth_grid_by_loss.tex"
        else:
            latex = to_latex_grid(specs, metrics, "etth-grid", caption)
            out_name = "etth_grid.tex"
        print(latex)
        if args.out:
            os.makedirs(args.out, exist_ok=True)
            with open(os.path.join(args.out, out_name), "w") as fh:
                fh.write(latex + "\n")
            print(f"\nwritten to {args.out}/{out_name}", file=sys.stderr)
        return 0

    if args.full:
        dims = {m: DIMENSION[(args.dataset, m)] for m in ("M", "S")
                if (args.dataset, m) in DIMENSION}
        if len(dims) < 2:
            parser.error(f"{args.dataset} has no two modes in DIMENSION")
        grouped_by_mode = {m: fetch(args.dataset, m) for m in dims}
        latex = to_latex_full(args.dataset, grouped_by_mode, dims)
        print(latex)
        if args.out:
            os.makedirs(args.out, exist_ok=True)
            with open(os.path.join(args.out, f"{args.dataset}_full.tex"), "w") as fh:
                fh.write(latex + "\n")
            print(f"\nwritten to {args.out}/{args.dataset}_full.tex", file=sys.stderr)
        return 0

    if args.both_modes:
        dims = {m: DIMENSION[(args.dataset, m)] for m in ("M", "S")
                if (args.dataset, m) in DIMENSION}
        if len(dims) < 2:
            parser.error(f"{args.dataset} has no two modes in DIMENSION")
        grouped_by_mode = {m: fetch(args.dataset, m) for m in dims}
        metrics = [m.strip() for m in args.metrics.split(",")]
        latex = to_latex_two_modes(args.dataset, grouped_by_mode, dims, metrics)
        print(latex)
        if args.out:
            os.makedirs(args.out, exist_ok=True)
            with open(os.path.join(args.out, f"{args.dataset}_both.tex"), "w") as fh:
                fh.write(latex + "\n")
            print(f"\nwritten to {args.out}/{args.dataset}_both.tex", file=sys.stderr)
        return 0

    d = DIMENSION.get((args.dataset, args.features))
    if d is None:
        parser.error(f"no dimension known for {(args.dataset, args.features)}; "
                     f"add it to DIMENSION")

    if args.from_dir_exact:
        grouped = fetch_dir(os.path.dirname(args.from_dir_exact.rstrip("/")),
                            os.path.basename(args.from_dir_exact.rstrip("/")))
    elif args.from_dir:
        grouped = fetch_dir(args.from_dir, args.dataset)
    else:
        grouped = fetch(args.dataset, args.features)
    merge_planned(grouped, args.dataset)
    if not grouped:
        print("no finished runs matched", file=sys.stderr)
        return 1

    latex = to_latex(grouped, args.dataset, args.features, d)
    print(latex)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        stem = args.dataset + (f"_{args.features}" if args.features else "")
        with open(os.path.join(args.out, f"{stem}.tex"), "w") as fh:
            fh.write(latex + "\n")
        to_csv(grouped, d, os.path.join(args.out, f"{stem}.csv"))
        print(f"\nwritten to {args.out}/{stem}.{{tex,csv}}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
