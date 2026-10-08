"""Turns the runtime benchmark's jsonl into the cost table.

Reads the rows written by ``src/train/benchmark.py`` (one per dataset x method x
seed x stage), aggregates over seeds and prints a markdown table plus the LaTeX
version and a CSV.

    python notebooks/make_runtime_table.py
    python notebooks/make_runtime_table.py --dataset euppbench --out tables/

Two rates are reported because neither alone is comparable everywhere. **s/step**
is the honest per-iteration cost and is what the arms of one task are ranked by --
they share a batch size, so the step is the same unit of work. **samples/s** is
the only number that survives a change of batch size, so it is what carries
across tasks. The *relative* column is the ratio to the **deterministic** arm of
the same task, which is the cleanest statement of what predicting a covariance
at all costs: the point predictor is the same backbone with a mean-only head, so
the ratio divides out the trunk, the resolution and the card and leaves the price
of the distributional head. (The diagonal arm would be the wrong denominator:
it is already a covariance model, so ratios against it would understate every
family by the cost of the diagonal head.)
"""

import argparse
import csv
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_IN = Path("logs/benchmark/runtime.jsonl")

# Row order: the deterministic reference first (it is the denominator), then the
# diagonal, then the structured families by increasing generality, then the
# nonparametric arm.
METHOD_ORDER = [
    "deterministic",
    "phys_diag",
    "phys_lora",
    "whitened_spectral_diag",
    "whitened_matern",
    "whitened_dct",
    "whitened_dst",
    "phys_cholesky",
    "generative",
]

#: Denominator of every relative column. The point predictor: same backbone,
#: mean-only head, so the ratio isolates the cost of predicting a covariance.
BASELINE_METHOD = "deterministic"


def label(row: Dict[str, Any]) -> str:
    """Row label, in the paper's method names (Det, SB, Diag, LorD, Chol, SCORE)."""
    method, rank = row["method"], row.get("rank") or 0
    if method == "phys_diag":
        return "Diag"
    if method == "phys_lora":
        return f"LorD r={rank or 1}"
    if method == "phys_cholesky":
        return "Chol"
    if method == "generative":
        m = row.get("n_train_samples") or 0
        return f"SB (M={m})" if m else "SB"
    if method == "deterministic":
        return "Det"
    if method.startswith("whitened_"):
        pretty = {"whitened_spectral_diag": "SCORE (Fourier core)",
                  "whitened_matern": "SCORE (graph Matern core)",
                  "whitened_dct": "SCORE (DCT core)",
                  "whitened_dst": "SCORE (DST core)"}
        return pretty.get(method, f"SCORE ({method[len('whitened_'):]})")
    return method


def _key(row: Dict[str, Any]) -> Tuple:
    """Identity of one measured arm: task first, then configuration.

    ``pred_len`` is part of the *task*, not a detail of the arm: the horizon
    sets the event dimension (V = pred_len x C) and therefore the cost of every
    covariance head -- phys_lora r=16 on multivariate ETTh1 is 37 ms at T=96 and
    63 ms at T=720. Pooling horizons would give one arm a meaningless mean and
    an error bar made of the gap between them.
    """
    return (row["dataset"], row.get("features"), row.get("pred_len") or 0,
            row["method"], row.get("rank") or 0)


#: Statuses that carry numbers. `over_budget` rows are real measurements that
#: simply exceeded the memory budget of the target card; only an `oom` has nothing to
#: report -- and that absence is itself a result, which is why `read_all` keeps
#: it and only `load` drops it.
MEASURED_STATUS = ("ok", "over_budget")


def read_all(path: Path, dataset: Optional[str] = None) -> List[Dict[str, Any]]:
    """Every row in the file, measured or not, optionally one dataset."""
    if not path.exists():
        sys.exit(f"No results at {path}. Run runs/benchmark.sh first.")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if dataset:
        rows = [r for r in rows if r["dataset"] == dataset]
    return rows


def load(path: Path, dataset: Optional[str]) -> List[Dict[str, Any]]:
    rows = [r for r in read_all(path, dataset)
            if r.get("status") in MEASURED_STATUS]
    if not rows:
        sys.exit("No successful rows matched.")
    return rows


def check_one_gpu(rows: List[Dict[str, Any]]) -> None:
    """Warn loudly when the rows did not all come from one card.

    A cost table is only one table if it was measured on one machine. Absolute
    times obviously do not transfer between card types, and the ratio to the
    deterministic arm -- the column meant to be hardware-robust -- only mostly does:
    it divides out the card for arms of similar arithmetic intensity, which the
    arms here are not (a trunk-bound station-graph run and a head-bound T=720
    low-rank run scale differently with bandwidth-vs-FLOPs). So this is a
    warning on stderr rather than a note in a docstring nobody reads.
    """
    gpus = sorted({str(r.get("gpu")) for r in rows if r.get("gpu")})
    if len(gpus) > 1:
        print(
            "WARNING: rows come from more than one GPU -- "
            + ", ".join(gpus)
            + ".\n         Per-step times are not comparable across them and "
            "`rel_to_det` only\n         partly is. Re-measure on one card, or "
            "split the table by GPU.",
            file=sys.stderr,
        )


def aggregate(rows: List[Dict[str, Any]]) -> Dict[Tuple, Dict[str, Any]]:
    """Collapse seeds into one entry per arm.

    Aggregating the per-seed *medians* rather than pooling every step: the
    within-seed spread is measurement noise on one model, the across-seed spread
    is what a rerun would actually see, and only the latter belongs in an error
    bar.
    """
    by_arm: Dict[Tuple, Dict[int, List[Dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        by_arm[_key(r)][r.get("stage") or 0].append(r)

    out: Dict[Tuple, Dict[str, Any]] = {}
    for arm, by_stage in by_arm.items():
        entry: Dict[str, Any] = {
            "dataset": arm[0], "features": arm[1], "method": arm[3], "rank": arm[4],
            "stages": {},
        }
        for stage, rs in sorted(by_stage.items()):
            med = [r["step_ms_median"] for r in rs]
            entry["stages"][stage] = {
                "step_ms": statistics.fmean(med),
                # Sample sd (n-1), not population: the seeds are a sample of the
                # runs that could have been made, and this is quoted next to a
                # mean as its spread; `stdev` needs n >= 2.
                "step_ms_sd": statistics.stdev(med) if len(med) > 1 else 0.0,
                "samples_per_s": statistics.fmean([r["samples_per_s"] for r in rs]),
                "forward_ms": statistics.fmean([r["forward_ms_median"] for r in rs]),
                "peak_gb": max(r["peak_alloc_gb"] for r in rs),
                "wall_ms": statistics.fmean([r["wall_ms_per_step"] for r in rs]),
                "n_seeds": len(med),
            }
            entry.setdefault("batch_size", rs[0]["batch_size"])
            entry.setdefault("params", rs[0]["params"])
            entry.setdefault("n_train_samples", rs[0].get("n_train_samples"))
            entry.setdefault("steps_per_epoch", rs[0].get("steps_per_epoch"))
            entry.setdefault("max_epochs", rs[0].get("max_epochs"))
            entry.setdefault("stage_epochs", (rs[0].get("stage1_max_epochs"),
                                              rs[0].get("stage2_max_epochs")))
            # Provenance travels with the numbers: a cost is only meaningful
            # next to the card and the build that produced it.
            for key in ("loss", "event_dim", "gpu", "torch", "datamodule_cls",
                        "pred_len"):
                entry.setdefault(key, rs[0].get(key))
            # Present only on staged runs; the epoch-weighted per-step cost.
            eff = [r["effective_step_ms"] for r in rs if r.get("effective_step_ms")]
            if eff:
                entry["effective_step_ms"] = statistics.fmean(eff)
        out[arm] = entry
    return out


def _headline(entry: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """(step_ms, sd, samples/s, peak GiB) for the arm as a whole.

    For a staged method the headline is the epoch-weighted effective cost, not
    either stage: stage 2 is cheaper (frozen trunk) *and* shorter, so quoting
    stage 1 overstates the method and quoting stage 2 understates it. The
    per-stage numbers stay in the table underneath.
    """
    stages = entry["stages"]
    if 0 in stages:
        s = stages[0]
        return s["step_ms"], s["step_ms_sd"], s["samples_per_s"], s["peak_gb"]
    eff = entry.get("effective_step_ms") or stages[1]["step_ms"]
    sd = max(s["step_ms_sd"] for s in stages.values())
    peak = max(s["peak_gb"] for s in stages.values())
    # samples/s from the effective step so it stays consistent with it.
    return eff, sd, stages[1]["samples_per_s"] * stages[1]["step_ms"] / eff, peak


def train_hours(entry: Dict[str, Any]) -> Optional[float]:
    """Compute-only cost of one full training run, in GPU hours.

    The point of a per-step number is what it implies for a campaign, and the
    implication is not proportional to the step time alone: a staged method
    spends its epochs at two different prices. Excludes validation, checkpointing
    and the input pipeline, so it is a lower bound -- comparable across arms
    because the excluded parts are identical for all of them.
    """
    spe, stages = entry.get("steps_per_epoch"), entry["stages"]
    if not spe:
        return None
    if 0 in stages:
        epochs = entry.get("max_epochs") or 0
        return stages[0]["step_ms"] * spe * epochs / 3.6e6
    e1, e2 = entry.get("stage_epochs") or (0, 0)
    if not (e1 and e2):
        return None
    return (stages[1]["step_ms"] * e1 + stages[2]["step_ms"] * e2) * spe / 3.6e6


# ---------------------------------------------------------------------------
# Export to the results directory
# ---------------------------------------------------------------------------

#: Root of the results folder (``paths.results_dir``).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.paths import configured_path  # noqa: E402

SHARED_ROOT = configured_path("results_dir")

#: Collected cost measurements of one experiment, a sibling of its ``tables/``
#: and ``calibration/`` -- the third derived artifact, and like them not
#: produced by ``sync_checkpoints.py``.
RUNTIME_DIRNAME = "runtime_analysis"

#: (experiment, dataset) per datamodule class, mirroring
#: ``tools/sync_checkpoints.py::identity`` so a cost file sits under the same
#: dataset name as the checkpoints and the calibration diagnostics it describes.
_EXPERIMENT = {
    "TimeSeriesDataModule": ("time_series", None),   # dataset from data_path
    "DepthDataModule": ("depth", None),      # dataset from the datamodule config
    "WeatherBenchModule": ("surface_temperature", "era5"),
    "EUPPBenchModule": ("station_postprocessing", "euppbench"),
}

_MODE = {"M": "multivariate", "S": "univariate", "MS": "multivariate"}

#: Depth: the benchmark row's ``dataset`` is the *datamodule config name*
#: (`benchmark.py` records `choices["datamodule"]`), while the checkpoint tree is
#: named by the resolved `datamodule.dataset`. The configs are also different
#: tasks: `depth` is the 128x160 `depth_reg` set (d = 20,480) and `depth_bts*`
#: the BTS/Eigen version (d = 238,560 cropped), whose covariance heads cost an
#: order of magnitude more, so each must map to its own folder.
_DEPTH_DATASET = {"depth": "nyu", "depth_bts": "nyu_bts",
                  "depth_bts_crop": "nyu_bts"}

#: Time series: the same mismatch. `sync_checkpoints.identity` names a
#: time-series folder after the CSV (`Path(data_path).stem.lower()`), while a
#: benchmark row carries the datamodule *config* name. The two coincide for
#: every data set except illness (`illness.yaml` reads `national_illness.csv`);
#: unmapped, its cost rows would not sit next to its scores and would never
#: reach the cost-accuracy panels of `runtime_analysis.ipynb`. Add an entry here
#: for any new data set whose CSV stem differs from its config name.
_SERIES_DATASET = {"illness": "national_illness"}

#: The forecast horizons the paper reports, per time-series data set (tree
#: name), as in `runtime_analysis.ipynb`. The benchmark log is append-only and
#: may hold rows for other horizons; only the aggregate CSVs are filtered --
#: `runtime.jsonl` stays the complete raw record.
REPORTED_HORIZONS = {"national_illness": (24, 36)}
DEFAULT_HORIZONS = (96, 192)


def identity(row: Dict[str, Any]) -> Tuple[str, str, Optional[str], Optional[int]]:
    """``(experiment, dataset, mode, horizon)`` of a benchmark row.

    Deliberately the same mapping ``tools/sync_checkpoints.py`` uses, so the
    runtime file for an arm lands beside that arm's checkpoints and calibration
    diagnostics instead of inventing a second naming scheme for the same models
    -- including the ``T<pred_len>`` level, since a 96- and a 720-step run of one
    configuration are different models with different event dimensions.
    """
    dm = row.get("datamodule_cls") or row.get("dataset_cls") or ""
    if dm not in _EXPERIMENT:
        raise ValueError(
            f"unknown datamodule {dm!r}; extend _EXPERIMENT to match "
            "tools/sync_checkpoints.py"
        )
    experiment, dataset = _EXPERIMENT[dm]
    if dataset is None:
        dataset = str(row["dataset"])
        if experiment == "depth":
            dataset = _DEPTH_DATASET.get(dataset, dataset)
        elif experiment == "time_series":
            dataset = _SERIES_DATASET.get(dataset, dataset)
    is_series = experiment == "time_series"
    mode = _MODE.get(str(row.get("features") or "M")) if is_series else None
    horizon = int(row["pred_len"]) if is_series and row.get("pred_len") else None
    return experiment, dataset, mode, horizon


#: Columns of an exported runtime file, in order. One row per (arm, stage);
#: staged arms carry stages 1, 2 *and* `mean`, so a reader never has to know the
#: epoch budgets to quote the method's cost.
EXPORT_FIELDS = [
    "method", "label", "rank", "n_train_samples", "loss", "stage",
    "pred_len", "batch_size", "event_dim", "steps_per_epoch", "epochs",
    "step_ms", "step_ms_sd", "steps_per_s", "samples_per_s", "ms_per_sample",
    "forward_ms", "backward_ms", "peak_gb", "gb_per_sample",
    "epoch_s", "train_hours", "rel_to_det", "n_seeds", "gpu", "torch",
]


def export_rows(entry: Dict[str, Any], base_ms: Optional[float]) -> List[Dict[str, Any]]:
    """One export row per stage, plus the epoch-weighted `mean` for staged arms.

    The `mean` row is always written for a two-stage method even though it is
    derivable, because the derivation needs the stage epoch budgets and those
    live in a config the reader of a CSV does not have. A single-stage arm gets
    one row, `stage = single`, whose numbers are already the method's cost.
    """
    stages = entry["stages"]
    spe = entry.get("steps_per_epoch") or 0
    e1, e2 = entry.get("stage_epochs") or (0, 0)
    out = []

    def row(stage_name, ms, sd, sps, fwd, bwd, peak, epochs, n_seeds):
        return {
            "method": entry["method"], "label": label(entry), "rank": entry["rank"],
            "n_train_samples": entry.get("n_train_samples"),
            "loss": entry.get("loss"), "stage": stage_name,
            "pred_len": entry.get("pred_len"),
            "batch_size": entry["batch_size"], "event_dim": entry.get("event_dim"),
            "steps_per_epoch": spe, "epochs": epochs,
            "step_ms": round(ms, 4),
            "step_ms_sd": round(sd, 4),
            "steps_per_s": round(1e3 / ms, 4),
            "samples_per_s": round(sps, 3),
            "ms_per_sample": round(ms / entry["batch_size"], 5),
            "forward_ms": round(fwd, 4) if fwd is not None else None,
            "backward_ms": round(bwd, 4) if bwd is not None else None,
            "peak_gb": round(peak, 3),
            "gb_per_sample": round(peak / entry["batch_size"], 4),
            "epoch_s": round(ms * spe / 1e3, 2) if spe else None,
            "train_hours": round(ms * spe * epochs / 3.6e6, 3) if spe and epochs else None,
            "rel_to_det": round(ms / base_ms, 4) if base_ms else None,
            "n_seeds": n_seeds, "gpu": entry.get("gpu"), "torch": entry.get("torch"),
        }

    if 0 in stages:
        st = stages[0]
        out.append(row("single", st["step_ms"], st["step_ms_sd"], st["samples_per_s"],
                       st["forward_ms"], st["step_ms"] - st["forward_ms"],
                       st["peak_gb"], entry.get("max_epochs"), st["n_seeds"]))
        return out

    for stage in sorted(stages):
        st = stages[stage]
        out.append(row(str(stage), st["step_ms"], st["step_ms_sd"], st["samples_per_s"],
                       st["forward_ms"], st["step_ms"] - st["forward_ms"],
                       st["peak_gb"], e1 if stage == 1 else e2, st["n_seeds"]))
    eff = entry.get("effective_step_ms")
    if eff:
        # Weighted like the run itself: stage 2 is cheaper AND shorter, so an
        # unweighted mean of the two stages would flatter the method.
        w = lambda k: (stages[1][k] * e1 + stages[2][k] * e2) / (e1 + e2)
        out.append(row("mean", eff, max(s["step_ms_sd"] for s in stages.values()),
                       w("samples_per_s"), w("forward_ms"), eff - w("forward_ms"),
                       max(s["peak_gb"] for s in stages.values()),
                       e1 + e2, min(s["n_seeds"] for s in stages.values())))
    return out


def export_shared(agg: Dict[Tuple, Dict[str, Any]], root: Path) -> List[Path]:
    """Write one CSV per (experiment, dataset, mode) into the results directory.

    Path mirrors ``calibration/``: ``<experiment>/runtime_analysis/<dataset>/
    [<mode>/]runtime.csv``. One file per dataset, as opposed to calibration's
    per-arm files, because a cost table is read as a whole -- the comparison
    between arms *is* the artifact.
    """
    groups: Dict[Tuple, List[Tuple]] = defaultdict(list)
    for arm, entry in agg.items():
        groups[identity(entry)].append(arm)

    written = []
    for (experiment, dataset, mode, horizon), arms in sorted(groups.items(), key=str):
        if (experiment == "time_series" and horizon
                and horizon not in REPORTED_HORIZONS.get(dataset, DEFAULT_HORIZONS)):
            print(f"skipped {dataset}/{mode}/T{horizon}: not a reported horizon",
                  file=sys.stderr)
            continue
        arms.sort(key=_sort_key)
        base = next((agg[a] for a in arms if a[3] == BASELINE_METHOD), None)
        base_ms = _headline(base)[0] if base else None
        out_dir = root / experiment / RUNTIME_DIRNAME / dataset
        if mode:
            out_dir = out_dir / mode
        if horizon:
            out_dir = out_dir / f"T{horizon}"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "runtime.csv"
        rows = [r for arm in arms for r in export_rows(agg[arm], base_ms)]
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=EXPORT_FIELDS)
            w.writeheader()
            w.writerows(rows)
        written.append(path)
    return written


def export_raw(rows: List[Dict[str, Any]], root: Path) -> List[Path]:
    """Copy the measurements themselves into each experiment's ``runtime_analysis/``.

    ``export_shared`` writes the aggregate; this writes the record it was
    derived from -- every seed and every stage separately, plus the rows an
    aggregate cannot represent at all: an ``oom`` arm has no numbers, so it
    vanishes from the CSV, and "this configuration does not fit on the card" is
    a result worth keeping rather than an absence.

    Split per experiment rather than left as one file at the root, so that an
    experiment folder is readable on its own. Rows whose datamodule
    ``identity`` does not know are skipped with a warning rather than dropping
    the whole export.
    """
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    unknown: Dict[str, int] = defaultdict(int)
    for row in rows:
        try:
            experiment = identity(row)[0]
        except ValueError:
            unknown[str(row.get("datamodule_cls"))] += 1
            continue
        groups[experiment].append(row)
    for dm, n in sorted(unknown.items()):
        print(f"WARNING: {n} rows with unknown datamodule {dm!r} not exported",
              file=sys.stderr)

    written = []
    for experiment, rs in sorted(groups.items()):
        out_dir = root / experiment / RUNTIME_DIRNAME
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "runtime.jsonl"
        # Sorted so the file has a stable order across re-exports and a diff
        # shows what actually changed rather than the order the sweep finished.
        rs = sorted(rs, key=lambda r: (str(r.get("dataset")), str(r.get("features")),
                                       r.get("pred_len") or 0, str(r.get("method")),
                                       r.get("rank") or 0, r.get("seed") or 0,
                                       r.get("stage") or 0))
        with open(path, "w") as fh:
            for r in rs:
                fh.write(json.dumps(r, sort_keys=True) + "\n")
        written.append(path)
    return written


def _sort_key(arm: Tuple) -> Tuple:
    method = arm[3]
    order = METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER)
    return (arm[0], str(arm[1]), arm[2], order, arm[4])


def to_markdown(agg: Dict[Tuple, Dict[str, Any]]) -> str:
    lines: List[str] = []
    by_task: Dict[Tuple, List[Tuple]] = defaultdict(list)
    for arm in agg:
        by_task[(arm[0], arm[1], arm[2])].append(arm)

    for (dataset, features, pred_len), arms in sorted(by_task.items(),
                                                      key=lambda kv: str(kv[0])):
        arms.sort(key=_sort_key)
        base = next(
            (agg[a] for a in arms if a[3] == BASELINE_METHOD), None
        )
        base_ms = _headline(base)[0] if base else None
        bs = agg[arms[0]]["batch_size"]
        title = dataset + (f" ({features})" if features else "")
        title += f", T={pred_len}" if pred_len else ""
        seeds = max((st["n_seeds"] for a in arms for st in agg[a]["stages"].values()),
                    default=0)
        lines.append(
            f"\n### {title} -- batch size {bs}, mean +/- sd over {seeds} seeds\n"
        )
        lines.append(
            "| Method | ms/step | steps/s | samples/s | vs. Det | fwd share "
            "| peak GiB | full run (GPU h) |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for arm in arms:
            e = agg[arm]
            ms, sd, sps, peak = _headline(e)
            rel = f"{ms / base_ms:.3f}x" if base_ms else "--"
            stages = e["stages"]
            fwd = sum(s["forward_ms"] for s in stages.values())
            tot = sum(s["step_ms"] for s in stages.values())
            hours = train_hours(e)
            row = (
                f"| {label(e)} | {ms:.3f} +/- {sd:.3f} | {1e3 / ms:.3f} | "
                f"{sps:.3f} | {rel} | {100 * fwd / tot:.0f}% | {peak:.3f} | "
                f"{hours:.3f} |" if hours else
                f"| {label(e)} | {ms:.3f} +/- {sd:.3f} | {1e3 / ms:.3f} | "
                f"{sps:.3f} | {rel} | {100 * fwd / tot:.0f}% | {peak:.3f} | -- |"
            )
            lines.append(row)
            if 0 not in stages:
                for stage, s in sorted(stages.items()):
                    lines.append(
                        f"| &nbsp;&nbsp;stage {stage} | {s['step_ms']:.3f} | "
                        f"{1e3 / s['step_ms']:.3f} | {s['samples_per_s']:.3f} | "
                        f"{s['step_ms'] / base_ms:.3f}x | "
                        f"{100 * s['forward_ms'] / s['step_ms']:.0f}% | "
                        f"{s['peak_gb']:.3f} | |"
                    )
    return "\n".join(lines)


def to_latex(agg: Dict[Tuple, Dict[str, Any]], dataset: str, features,
             pred_len: int = 0) -> str:
    arms = sorted(
        [a for a in agg if a[0] == dataset and a[1] == features and a[2] == pred_len],
        key=_sort_key,
    )
    if not arms:
        return ""
    base = next((agg[a] for a in arms if a[3] == BASELINE_METHOD), None)
    base_ms = _headline(base)[0] if base else None
    bs = agg[arms[0]]["batch_size"]
    # The horizon is part of the task, not a footnote: it sets the event
    # dimension, so a T=96 and a T=720 table are about different models.
    task = dataset + (f" ({features})" if features else "")
    task += f", horizon $T={pred_len}$" if pred_len else ""
    slug = dataset + (f"-{features}" if features else "") + \
        (f"-T{pred_len}" if pred_len else "")
    out = [
        r"\begin{table}[t]", r"  \centering",
        rf"  \caption{{Training cost on {task}, at a shared batch size of {bs}. "
        r"Per-step time is the median over the measured steps, averaged across "
        r"seeds; the error is the sample standard deviation across them. SCORE "
        r"is staged, so its headline is the epoch-weighted mean of the "
        r"two stage costs. \emph{vs.~Det} is the ratio to Det, the deterministic "
        r"arm of the same task -- the same backbone with a mean-only head, so "
        r"the ratio is the price of predicting a covariance at all.}}",
        rf"  \label{{tab:runtime-{slug}}}",
        r"  \begin{tabular}{lrrrr}", r"    \toprule",
        r"    Method & ms/step & samples/s & vs.~Det & peak (GiB) \\",
        r"    \midrule",
    ]
    for arm in arms:
        e = agg[arm]
        ms, sd, sps, peak = _headline(e)
        rel = rf"{ms / base_ms:.3f}$\times$" if base_ms else "--"
        name = label(e).replace("Matern", r"Mat\'ern")
        out.append(
            rf"    {name} & ${ms:.3f} \pm {sd:.3f}$ & {sps:.3f} & "
            rf"{rel} & {peak:.3f} \\"
        )
    out += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(out)


def to_csv(agg: Dict[Tuple, Dict[str, Any]], path: Path) -> None:
    fields = ["dataset", "features", "pred_len", "method", "rank",
              "batch_size", "stage",
              "step_ms", "step_ms_sd", "steps_per_s", "samples_per_s",
              "forward_ms", "wall_ms", "peak_gb", "n_seeds", "params"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for arm in sorted(agg, key=_sort_key):
            e = agg[arm]
            for stage, s in sorted(e["stages"].items()):
                w.writerow({
                    "dataset": e["dataset"], "features": e["features"],
                    "pred_len": e.get("pred_len"),
                    "method": e["method"], "rank": e["rank"],
                    "batch_size": e["batch_size"], "stage": stage or "",
                    "step_ms": round(s["step_ms"], 4),
                    "step_ms_sd": round(s["step_ms_sd"], 4),
                    "steps_per_s": round(1e3 / s["step_ms"], 4),
                    "samples_per_s": round(s["samples_per_s"], 3),
                    "forward_ms": round(s["forward_ms"], 4),
                    "wall_ms": round(s["wall_ms"], 4),
                    "peak_gb": round(s["peak_gb"], 3),
                    "n_seeds": s["n_seeds"], "params": e["params"],
                })


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_IN)
    ap.add_argument("--dataset", default=None, help="restrict to one dataset")
    ap.add_argument("--out", type=Path, default=None, help="directory for .tex/.csv")
    ap.add_argument(
        "--export", action="store_true",
        help="also write <experiment>/runtime_analysis/runtime.jsonl (the raw "
             "measurements) and <dataset>/[<mode>/][T<pred_len>/]runtime.csv "
             "(the aggregate) into the shared results tree",
    )
    ap.add_argument("--shared-root", type=Path, default=SHARED_ROOT)
    args = ap.parse_args()

    raw = read_all(args.input, args.dataset)
    rows = [r for r in raw if r.get("status") in MEASURED_STATUS]
    if not rows:
        sys.exit("No successful rows matched.")
    check_one_gpu(rows)
    agg = aggregate(rows)
    print(to_markdown(agg))

    if args.export:
        # umask 002 so group members can overwrite the exported files (on a
        # setgid directory the group itself is inherited).
        os.umask(0o002)
        for path in export_raw(raw, args.shared_root) + \
                export_shared(agg, args.shared_root):
            print(f"wrote {path}")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        to_csv(agg, args.out / "runtime.csv")
        for dataset, features, pred_len in sorted({(a[0], a[1], a[2]) for a in agg},
                                                  key=str):
            tex = to_latex(agg, dataset, features, pred_len)
            if tex:
                suffix = f"_{features}" if features else ""
                suffix += f"_T{pred_len}" if pred_len else ""
                (args.out / f"runtime_{dataset}{suffix}.tex").write_text(tex + "\n")
        print(f"\nWrote {args.out}/runtime.csv and per-dataset .tex files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
