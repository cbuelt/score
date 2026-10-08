# --- Project root bootstrap (must run before any `src` import) ---
# Mirrors src/train/train.py: walk up to the `.project-root` marker, put the repo
# root on sys.path so `import src...` works when run as
# `python src/evaluation/epoch_budget.py`, and export PROJECT_ROOT for Hydra's
# ${oc.env:PROJECT_ROOT} interpolation.
import os
import sys
from pathlib import Path

for _parent in Path(__file__).resolve().parents:
    if (_parent / ".project-root").exists():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        os.environ.setdefault("PROJECT_ROOT", str(_parent))
        break
# -----------------------------------------------------------------

import copy
import json
import math
import platform
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf, open_dict

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

__doc__ = """Epoch-budget ablation: the core, or just the epochs?

SCORE splits its budget between fitting the marginals (stage 1)
and fitting the correlation core on top of them (stage 2), so every gain it
reports is confounded with how the budget was spent -- the same epochs poured
entirely into the marginals might have done as well. This script trains, at each
of several *equal* total budgets E, the two arms that separate those:

    marginal   Diag on the marginal CRPS for the full E epochs
    ours       SCORE on the same backbone, 3E/4 marginals (stage 1) + E/4 core (stage 2)

Identical total epochs, identical backbone, identical optimizer and identical
stage-1 objective, so the difference in test score is the core and not the
training length.

One invocation is one cell: (experiment, arm, frac, seed). The budget is derived
from the experiment's own ``trainer.max_epochs``, which makes the ``frac=1.0``
rung the main training configuration by construction, and therefore a
reproducibility check on the whole pipeline.

Rows are appended to one jsonl (``epoch_budget.out``), keyed by the cell, and a
cell already present is skipped, so re-running an interrupted sweep resumes it.
``notebooks/epoch_budget.ipynb`` aggregates the file over seeds and draws the
figures.

Usage::

    python src/evaluation/epoch_budget.py experiments=train/euppbench \\
        method=diag method.loss=crps \\
        epoch_budget.arm=marginal epoch_budget.frac=0.6 seed=1

    python src/evaluation/epoch_budget.py -m \\
        experiments=train/euppbench_score \\
        epoch_budget.arm=ours epoch_budget.frac=0.6 seed=1,2,3,4,5

    bash runs/epoch_budget.sh graph          # the whole ladder, both arms
"""

#: Bumped whenever the row's key set or the meaning of a field changes. The skip
#: check is keyed on it, so a bump re-runs every cell rather than mixing two
#: definitions of the same column in one file.
SCHEMA_VERSION = 1

#: Split of a total budget between the two stages, as (stage 1, stage 2). The
#: repo standard, and what every experiment config already carries: 75/25 for the
#: time series (experiments/train/settings.yaml), 30/10 for EUPPBench.
STAGE_SPLIT = (3, 1)

#: Fewest epochs stage 2 may be given. Its warmup is subtractive and is clamped
#: to `stage2 - 1`, so at 1 epoch the warmup covers the whole stage and the LR
#: never anneals -- a rung trained on a different recipe from every other rung.
#: This is what rules out `frac=0.1` on EUPPBench (E = 4 -> 3+1).
MIN_STAGE2 = 2

#: The two arms. ``marginal`` (Diag on CRPS) requires a single-stage experiment,
#: ``ours`` (SCORE, two-stage fit) a staged one; see `apply_arm` for why that is
#: checked rather than inferred.
ARMS = ("marginal", "ours")

#: Test metrics kept in a row, in the key order of a synced run's metrics.json
#: (which also carries `mae`). This is a *storage* order, not the printed one --
#: figures and tables order their metrics by make_results_tables.COLUMNS -- and
#: it is frozen because the finished ladder's rows are already written with it.
#: A metric the experiment did not compute is stored as None, never dropped, so
#: a reader can tell "not measured" from "not in this schema".
TEST_METRICS = (
    "nll", "crps", "energy_score", "kernel_score", "variogram_score", "mse", "mae",
)


# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------


def resolve_budget(max_epochs: int, frac: float) -> Tuple[int, int, int]:
    """``(E, stage1, stage2)`` for a rung at fraction ``frac`` of ``max_epochs``.

    The rung is a fraction of the experiment's configured budget rather than an
    absolute epoch count, so the ladder is anchored to the main configuration: at
    ``frac=1.0`` the split reproduces the config's own ``stage{1,2}_max_epochs``
    exactly (40 -> 30+10, 100 -> 75+25).

    The split is rounded half-up when the budget does not divide by
    ``sum(STAGE_SPLIT)``, which a ladder at 0.1 spacing forces: five of the ten
    ETT rungs (E = 10, 30, 50, 70, 90) are not multiples of 4. The rounding
    moves at most one epoch, so the realised ratio deviates from 3:1 by at most
    ``1/E`` -- 80:20 at the E = 10 rung, 76.7:23.3 at E = 30, under a point from
    E = 50 up. Both `stage1_epochs` and `stage2_epochs` are recorded per row, so
    what a rung actually trained is never inferred from the fraction. The
    deviation is acceptable because it is a *within-rung* property: both arms of
    a rung always get the same total E, which is what the comparison rests on.

    Raises when the budget cannot carry a two-stage fit at all: stage 2 needs at
    least 2 epochs, or the subtractive warmup (clamped to `stage2 - 1`) consumes
    the whole stage and its LR never anneals. On EUPPBench that rules out
    ``frac=0.1`` (E = 4 -> 3+1), which is why the ladder there starts at 0.2.
    """
    budget = int(round(frac * max_epochs))
    if budget <= 0:
        raise ValueError(
            f"frac={frac} of max_epochs={max_epochs} is {budget} epochs; the "
            "rung has to train something."
        )
    stage1, stage2 = _split(budget)
    if stage2 < MIN_STAGE2:
        raise ValueError(
            f"frac={frac} of max_epochs={max_epochs} is {budget} epochs, so "
            f"stage 2 gets {stage2} -- too few to anneal (its warmup alone "
            "would cover the stage). Start the ladder one rung higher: on a "
            f"{max_epochs}-epoch budget that means frac >= "
            f"{_min_frac(max_epochs):.2f}."
        )
    return budget, stage1, stage2


def _split(budget: int) -> Tuple[int, int]:
    """``(stage1, stage2)`` of a budget at ``STAGE_SPLIT``, rounded half-up."""
    total_parts = sum(STAGE_SPLIT)
    stage1 = (STAGE_SPLIT[0] * budget + total_parts // 2) // total_parts
    return stage1, budget - stage1


def _min_frac(max_epochs: int) -> float:
    """Smallest tenth-spaced rung whose stage 2 can still anneal.

    Shares `_split` with `resolve_budget` rather than calling it: this runs while
    building that function's error message, so calling back into it would recurse
    until the stack ran out.
    """
    for tenth in range(1, 11):
        if _split(int(round(tenth / 10 * max_epochs)))[1] >= MIN_STAGE2:
            return tenth / 10
    return 1.0


def _warmup_of(bundle: DictConfig) -> int:
    """Configured linear-warmup epochs of an optimizer bundle, 0 when unset."""
    scheduler = bundle.get("scheduler") if bundle is not None else None
    return int((scheduler or {}).get("warmup_epochs") or 0)


def _clamp_warmup(cfg: DictConfig, key: str, epochs: int, label: str) -> int:
    """Keep a short fit's LR schedule able to anneal; return the warmup used.

    Warmup is *subtractive* -- it is spent inside the fit's budget -- so at the
    bottom of the ladder a 3-epoch warmup can cover a whole fit and the LR never
    anneals at all. ``train._log_warmup_budget`` only warns about that, so left
    alone it would hand one rung a different recipe from every other rung and
    then call the difference an effect of the budget.

    ``key`` is the config key the fit reads its bundle from, and it differs by
    arm: a single-stage run reads ``cfg.optimizer``, while a stage of a staged
    run reads ``cfg.optimizer_stage{n}`` falling back to ``cfg.optimizer`` (the
    fallback ``train._maybe_partial`` implements). Writing to the wrong one is
    a silent no-op, hence the explicit parameter.

    The clamp *deep-copies* the resolved bundle into ``key`` before lowering
    ``warmup_epochs``. Copying matters for the staged arm: the EUPPBench
    experiment composes no per-stage bundles, so both stages read
    ``cfg.optimizer`` and editing that in place would move stage 1's schedule
    too -- while a bare ``optimizer@optimizer_stage2=optimizer`` override would
    bring the group's default lr (1e-3) instead of the experiment's (2e-4).
    """
    bundle = cfg.get(key) or cfg.get("optimizer")
    warmup = _warmup_of(bundle)
    if warmup < epochs:
        return warmup
    clamped = max(1, epochs - 1)
    log.warning(
        f"{label}: clamping warmup_epochs {warmup} -> {clamped} so the "
        f"{epochs}-epoch budget has room to anneal."
    )
    with open_dict(cfg):
        cfg[key] = copy.deepcopy(bundle)
        cfg[key].scheduler.warmup_epochs = clamped
    return clamped


def apply_arm(cfg: DictConfig, arm: str, frac: float) -> Dict[str, Any]:
    """Set the epoch keys this arm is driven by; return the plan for the row.

    The arm is *checked* against the experiment rather than inferred from it,
    because the two arms are driven by different keys and each key is silently
    inert for the other arm: a staged run takes its ceilings from
    ``stage{1,2}_max_epochs`` and ignores ``trainer.max_epochs`` entirely
    (``train._build_stage_trainer``), while a single-stage run does the reverse.
    Getting that backwards trains a full-budget model and files it as a rung.
    """
    if arm not in ARMS:
        raise ValueError(f"epoch_budget.arm must be one of {ARMS}, got {arm!r}")
    staged = bool(cfg.get("staged", False))
    if arm == "ours" and not staged:
        raise ValueError(
            "epoch_budget.arm=ours needs a staged experiment, but this one has "
            "`staged: false`, so it has no stage 2 and its stage budgets would "
            "be ignored. Use one of the *_score experiments, or arm=marginal."
        )
    if arm == "marginal" and staged:
        raise ValueError(
            "epoch_budget.arm=marginal needs a single-stage experiment "
            "(staged: false); a staged one would ignore trainer.max_epochs and "
            "train the config's own stage budgets. Use the single-stage "
            "experiment with `method=diag method.loss=crps`, or arm=ours."
        )

    # The budget is a fraction of the *experiment's* ceiling. For a staged run
    # that ceiling is documentation only (the stages carry the real budgets), but
    # every experiment keeps it equal to their sum, which is what anchors the two
    # arms of a rung to the same number.
    max_epochs = int(cfg.trainer.max_epochs)
    budget, stage1, stage2 = resolve_budget(max_epochs, float(frac))

    # `warmup_stage1` is the warmup of the first (for `marginal`, the only) fit,
    # `warmup_stage2` the core fit's, None on the single-stage arm. Recorded
    # because the clamp below can lower them on the shortest rungs.
    plan: Dict[str, Any] = {
        "frac": float(frac),
        "budget": budget,
        "max_epochs": max_epochs,
        "stage1_epochs": None,
        "stage2_epochs": None,
        "warmup_stage1": None,
        "warmup_stage2": None,
    }
    with open_dict(cfg):
        if arm == "ours":
            cfg.stage1_max_epochs = stage1
            cfg.stage2_max_epochs = stage2
            # Kept consistent so anything reading the total (the run banner, the
            # logged config, this row) sees the rung and not the full budget.
            cfg.trainer.max_epochs = budget
            plan["stage1_epochs"] = stage1
            plan["stage2_epochs"] = stage2
        else:
            cfg.trainer.max_epochs = budget
            # Inert on this path, but left at the config's values a reader would
            # see a single-stage row claiming stage budgets it never ran.
            cfg.stage1_max_epochs = None
            cfg.stage2_max_epochs = None

    if arm == "ours":
        plan["warmup_stage1"] = _clamp_warmup(cfg, "optimizer_stage1", stage1, "Stage 1")
        plan["warmup_stage2"] = _clamp_warmup(cfg, "optimizer_stage2", stage2, "Stage 2")
    else:
        plan["warmup_stage1"] = _clamp_warmup(cfg, "optimizer", budget, "Training")
    return plan


# ---------------------------------------------------------------------------
# Identity (the results-directory path components of a run)
# ---------------------------------------------------------------------------

#: (experiment, dataset) per datamodule class. The same naming as the results
#: directory ``<experiment>/<dataset>/[<mode>/][T<horizon>/]<slug>/seed<N>/``
#: (and as ``make_runtime_table.identity``), so an ablation row carries the same
#: dataset name as the checkpoints and calibration diagnostics of the models it
#: describes.
_EXPERIMENT = {
    "TimeSeriesDataModule": ("time_series", None),   # dataset from data_path
    "DepthDataModule": ("depth", None),              # dataset from the dm config
    "WeatherBenchModule": ("surface_temperature", "era5"),
    "EUPPBenchModule": ("station_postprocessing", "euppbench"),
}

_MODE = {"M": "multivariate", "S": "univariate", "MS": "multivariate"}

_DEPTH_DATASET = {"nyu_bts_crop": "nyu_bts"}

#: method class name -> slug family, and the two slug defaults that are omitted
#: rather than spelled out. SCORE runs use the slug family ``whitened``. Kept in
#: step with the results-directory naming so a row's `slug` joins to a
#: checkpoint folder.
_FAMILY = {
    "Deterministic": "deterministic",
    "Diag": "phys_diag",
    "LorD": "phys_lora",
    "Cholesky": "phys_cholesky",
    "SampleBased": "generative",
    "SpectralDiag": "spectral_diag",
    "SCORE": "whitened",
}
_DEFAULT_STAGE1 = "marginal_kernel_score"
_DEFAULT_CORE = "free"


def _transform_kind(tr) -> str:
    """Basis name used in the slug of a SCORE run (``fourier`` -> ``spectral_diag``)."""
    if not isinstance(tr, dict):
        return "spectral_diag"
    kind = tr.get("kind")
    if kind is not None:
        return "spectral_diag" if kind == "fourier" else str(kind)
    target = str(tr.get("_target_", "")).rsplit(".", 1)[-1]
    if target == "make_graph_transform":
        return "graph"
    raise ValueError(f"cannot name transform {target or tr!r}")


def identity(cfg: DictConfig) -> Dict[str, Any]:
    """``experiment``/``dataset``/``mode``/``horizon``/``slug`` of a run.

    In the results directory the path identifies a run, so these are the
    components that place a result, derived here from the *resolved* config.
    The slug is ``<family>[_<basis>][_<core>][_r<rank>|_lora<rank>][_s1<stage-1
    loss>]_<loss>``, where the basis/core parts apply to SCORE only and the
    defaults (``free`` core, ``marginal_kernel_score`` stage-1 loss) are
    omitted. ``slug`` is carried even though this study never stores a
    checkpoint (the epoch budget is not part of a slug, so the rungs of one arm
    would collapse onto one path and overwrite each other *and* the main run),
    because it is what joins a row to the arm's entry in the score tables.
    """
    # Only these two nodes are resolved, not the whole config: `paths.output_dir`
    # interpolates `${hydra:runtime.output_dir}`, which needs a live Hydra run and
    # would make this unusable from a test. Interpolations *inside* these nodes
    # still resolve against the root (e.g. the graph transform's
    # `meta_path: ${datamodule.data_path}`), since the nodes stay attached to it.
    dm = OmegaConf.to_container(cfg.datamodule, resolve=True)
    method = OmegaConf.to_container(cfg.method, resolve=True)
    dm_cls = str(dm["_target_"]).rsplit(".", 1)[-1]
    m_cls = str(method["_target_"]).rsplit(".", 1)[-1]
    if dm_cls not in _EXPERIMENT:
        raise ValueError(
            f"unknown datamodule {dm_cls!r}; extend _EXPERIMENT"
        )
    experiment, dataset = _EXPERIMENT[dm_cls]
    mode = horizon = None
    if experiment == "time_series":
        dataset = Path(dm["data_path"]).stem.lower()
        mode = _MODE.get(str(dm.get("features", "M")))
        horizon = int(dm["pred_len"])
    elif experiment == "depth":
        dataset = str(dm.get("dataset", "nyu"))
        dataset = _DEPTH_DATASET.get(dataset, dataset)

    family = _FAMILY.get(m_cls)
    if family is None:
        raise ValueError(f"unknown method {m_cls!r}")
    staged = bool(cfg.get("staged", False)) and "stage2_loss" in method
    if family == "whitened":
        base = f"whitened_{_transform_kind(method.get('transform'))}"
        core = method.get("core_parameterization") or _DEFAULT_CORE
        if core != _DEFAULT_CORE:
            base += f"_{core}"
    else:
        base = family
    rank = int(method.get("rank") or 0)
    if rank > 0 and family in ("phys_lora", "whitened"):
        base += f"_r{rank}" if "lora" in base else f"_lora{rank}"
    if staged and method.get("stage1_loss", _DEFAULT_STAGE1) != _DEFAULT_STAGE1:
        base += f"_s1{str(method['stage1_loss']).replace('_', '')}"
    loss = method.get("stage2_loss") if staged else method.get("loss")

    return {
        "experiment": experiment,
        "dataset": dataset,
        "mode": mode,
        "horizon": horizon,
        "slug": f"{base}_{loss}",
        "method_cls": m_cls,
        "loss": None if staged else loss,
        "stage1_loss": method.get("stage1_loss") if staged else None,
        "stage2_loss": method.get("stage2_loss") if staged else None,
    }


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

#: Fields whose tuple identifies a cell of the study. Everything else in a row is
#: either a measurement or provenance.
KEY_FIELDS = (
    "schema_version", "experiment", "dataset", "mode", "horizon",
    "arm", "budget", "seed",
)


def row_key(row: Dict[str, Any]) -> Tuple:
    return tuple(row.get(f) for f in KEY_FIELDS)


def needs_refresh(
    out_path: Path, key: Tuple, overwrite: bool = False
) -> Tuple[bool, str]:
    """Whether this cell has to be trained, and why.

    Keyed on the cell *and* the schema version rather than on the file existing,
    as in the calibration script: long sweeps are often interrupted and re-run,
    and the re-run must skip what is already there without being told to. A row
    written under an older ``SCHEMA_VERSION`` has a different key and so is
    re-trained rather than mixed in.

    An unreadable line is ignored rather than fatal -- a sweep killed mid-write
    leaves a partial last line, and refusing to start again over it would make
    the kill unrecoverable.
    """
    if overwrite:
        return True, "epoch_budget.overwrite=true"
    if not out_path.exists():
        return True, "no results file yet"
    with open(out_path) as fh:
        for n, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                if row_key(json.loads(line)) == key:
                    return False, "already present"
            except json.JSONDecodeError:
                log.warning(f"{out_path}:{n} is not valid json; ignoring it")
    return True, "cell not in results file"


def _write_row(row: Dict[str, Any], cfg: DictConfig) -> None:
    """Append to the results file and drop a copy in the run directory."""
    shared = Path(cfg.epoch_budget.out)
    payload = json.dumps(row) + "\n"
    # umask 002 so the files stay group-writable when the results file lives in
    # a directory shared by a group (as in the calibration script).
    previous = os.umask(0o002)
    try:
        shared.parent.mkdir(parents=True, exist_ok=True)
        with open(shared, "a") as fh:
            fh.write(payload)
        run_dir = Path(cfg.paths.output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "epoch_budget.jsonl", "w") as fh:
            fh.write(payload)
    finally:
        os.umask(previous)
    log.info(f"Appended 1 row to {shared}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _finite(value: Any) -> Optional[float]:
    """A JSON-safe float: non-finite scores (a Dirac's nll) become null."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _best_epoch(objects: Dict[str, Any]) -> Optional[int]:
    """Epoch of the checkpoint that was evaluated, from its filename.

    The checkpoints are named ``epoch_{epoch:03d}`` with
    ``auto_insert_metric_name: False`` (configs/callbacks/*.yaml), so the
    selected epoch is recoverable from the path -- worth keeping, because "the
    budget was 60 but the best epoch was 12" is the difference between a rung
    that is short and one that has converged.
    """
    trainer = objects.get("trainer")
    callback = getattr(trainer, "checkpoint_callback", None) if trainer else None
    path = getattr(callback, "best_model_path", "") or ""
    stem = Path(path).stem
    digits = stem.rsplit("_", 1)[-1] if "_" in stem else ""
    return int(digits) if digits.isdigit() else None


def _param_count(objects: Dict[str, Any]) -> Optional[int]:
    """Total backbone parameters of the fitted model."""
    model = objects.get("model")
    backbone = getattr(model, "backbone", None) if model is not None else None
    if backbone is None:
        return None
    return int(sum(p.numel() for p in backbone.parameters()))


def run_cell(cfg: DictConfig) -> Optional[Dict[str, Any]]:
    """Train one (arm, frac, seed) cell and return its row, or None if skipped."""
    # Imported here, not at module scope: `train` pulls in Lightning, the
    # backbones and every method, which is a few seconds of import even to find
    # out the cell is already done.
    from src.train.train import train

    arm = str(cfg.epoch_budget.arm)
    plan = apply_arm(cfg, arm, cfg.epoch_budget.frac)
    ident = identity(cfg)
    row = {
        "schema_version": SCHEMA_VERSION,
        **{k: ident[k] for k in ("experiment", "dataset", "mode", "horizon", "slug")},
        "arm": arm,
        **plan,
        "seed": int(cfg.seed) if cfg.get("seed") is not None else None,
    }
    key = row_key(row)
    out_path = Path(cfg.epoch_budget.out)
    refresh, reason = needs_refresh(out_path, key, bool(cfg.epoch_budget.overwrite))
    label = (f"{ident['dataset']} {arm} frac={plan['frac']} "
             f"E={plan['budget']} seed={row['seed']}")
    if not refresh:
        log.info(f"skipping {label} ({reason})")
        return None
    log.info(f"training {label} ({reason})")

    started = time.perf_counter()
    metrics, objects = train(cfg)
    wall_s = time.perf_counter() - started

    row.update(
        method=ident["method_cls"],
        loss=ident["loss"],
        stage1_loss=ident["stage1_loss"],
        stage2_loss=ident["stage2_loss"],
        test={m: _finite(metrics.get(m)) for m in TEST_METRICS},
        # Whatever the last stage's callbacks logged, as a coarse check that the
        # fit itself was healthy -- a rung whose val loss never moved is visible
        # here without reopening the run.
        training={
            str(k): _finite(v) for k, v in metrics.items()
            if str(k).startswith("training/")
        },
        best_epoch=_best_epoch(objects),
        # Backbone parameter count, so the obvious objection to the comparison
        # -- "the staged arm is just a bigger model" -- is answerable from the
        # row. The two arms share the trunk and the mu/sigma head (both at the
        # backbone's default `shared_num_blocks`); the staged arm additionally
        # carries the `other` branch that emits the core, which the control has
        # nothing to do with.
        params=_param_count(objects),
        # Equal epochs are not equal wall-clock: stage 2 runs on a frozen trunk,
        # so the staged arm buys its core at well under a stage-1 epoch each.
        # Recording this lets the same rows be read as score-vs-GPU-seconds.
        wall_s=round(wall_s, 1),
        gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available()
             else platform.processor()),
        torch_version=torch.__version__,
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    scores = ", ".join(
        f"{m}={row['test'][m]:.4f}" for m in TEST_METRICS
        if row["test"].get(m) is not None
    )
    log.info(f"done {label} in {wall_s / 60:.1f} min: {scores}")
    return row


@hydra.main(version_base="1.3", config_path="../../configs", config_name="epoch_budget")
def main(cfg: DictConfig) -> None:
    try:
        row = run_cell(cfg)
    finally:
        # Staged training finishes its own W&B run only from train.py's main, so
        # a multirun sweep through this entrypoint would otherwise carry one run
        # into the next cell.
        if wandb.run is not None:
            wandb.finish()
    if row is not None:
        _write_row(row, cfg)


if __name__ == "__main__":
    main()
