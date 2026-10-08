# --- Project root bootstrap (must run before any `src` import) ---
# Mirrors src/train/train.py: walk up to the `.project-root` marker, put the repo
# root on sys.path so `import src...` works when run as `python src/train/benchmark.py`,
# and export PROJECT_ROOT for Hydra's ${oc.env:PROJECT_ROOT} interpolation.
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

import json
import platform
import statistics
import time
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import torch
from hydra.core.hydra_config import HydraConfig
from lightning import LightningDataModule
from omegaconf import DictConfig, OmegaConf

from src.methods import PredictiveMethod
from src.models.gaussian_predictive import (
    LightningPredictiveModel,
    StagedLightningModel,
    _split_weight_decay,
)
from src.train.train import _resolve_gammas
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


# ---------------------------------------------------------------------------
# Batch handling
# ---------------------------------------------------------------------------


def _pin_batch_size(cfg: DictConfig, batch_size: Optional[int]) -> int:
    """Force every arm onto one batch size, and report the value used.

    A per-step time is only comparable across methods at a *fixed* batch size,
    so the benchmark overrides whatever the dataset config carries. ``null``
    keeps the dataset default, which is the right choice only when a single
    dataset is being profiled and no arm is memory-bound.
    """
    if batch_size is None:
        return int(cfg.datamodule.batch_size)
    OmegaConf.update(cfg, "datamodule.batch_size", int(batch_size), merge=True)
    # Kept in lockstep so a datamodule that sizes buffers from the eval loader
    # (e.g. the gridded sets) does not allocate for a different shape.
    if "eval_batch_size" in cfg.datamodule:
        OmegaConf.update(cfg, "datamodule.eval_batch_size", int(batch_size), merge=True)
    return int(batch_size)


def _to_device(batch: Any, device: str) -> Any:
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, (list, tuple)):
        return type(batch)(_to_device(b, device) for b in batch)
    if isinstance(batch, dict):
        return {k: _to_device(v, device) for k, v in batch.items()}
    return batch


def _batch_n(batch: Any) -> int:
    x = batch[0] if isinstance(batch, (tuple, list)) and batch else batch
    return int(x.shape[0]) if torch.is_tensor(x) and x.dim() else 0


class _Batches:
    """Endless supply of full-size batches from a train dataloader.

    Restarts the loader when exhausted and skips any trailing partial batch:
    a short batch is cheaper to compute, and letting one into the sample would
    bias whichever arm happened to draw more of them.
    """

    def __init__(self, datamodule: LightningDataModule, batch_size: int, device: str):
        self.datamodule = datamodule
        self.batch_size = batch_size
        self.device = device
        self._it = iter(datamodule.train_dataloader())

    def next(self) -> Any:
        for _ in range(2):
            for batch in self._it:
                if _batch_n(batch) == self.batch_size:
                    return _to_device(batch, self.device)
            self._it = iter(self.datamodule.train_dataloader())
        raise RuntimeError(
            f"No batch of size {self.batch_size} in two passes over the train "
            "loader -- is benchmark.batch_size larger than the training split?"
        )


# ---------------------------------------------------------------------------
# Object construction (mirrors src/train/train.py)
# ---------------------------------------------------------------------------


def build_model(
    cfg: DictConfig, datamodule: LightningDataModule
) -> Tuple[LightningPredictiveModel, PredictiveMethod, bool]:
    """Build method + backbone + Lightning module exactly as ``train.py`` does.

    Kept a straight mirror of ``train``/``train_staged`` rather than a
    simplified stand-in: the point of the benchmark is the cost of the real
    configuration, so any divergence here (a different head width, a missing
    ``other_zero_init``) would be measuring a model nobody trains.
    """
    gamma_train, gamma_eval = _resolve_gammas(cfg, datamodule)
    method: PredictiveMethod = hydra.utils.instantiate(
        cfg.method,
        out_channels=datamodule.get_channels(),
        event_dim=datamodule.get_total_dimension(),
        gamma=gamma_train,
        gamma_eval=gamma_eval,
    )
    backbone: torch.nn.Module = hydra.utils.instantiate(
        cfg.backbone,
        in_channels=datamodule.get_input_channels(),
        out_channels=method.out_channels,
        group_widths=method.group_widths,
        group_denorm=method.group_denorm,
        other_zero_init=method.other_zero_init,
        noise=method.noise,
    )
    staged = bool(cfg.get("staged", False))
    if staged:
        model: LightningPredictiveModel = StagedLightningModel(
            backbone=backbone,
            predictive_method=method,
            stage1_loss=cfg.method.get("stage1_loss", "crps"),
            stage2_loss=cfg.method.get("stage2_loss", "kernel_score"),
            freeze_backbone=cfg.method.get("freeze_backbone", True),
        )
    else:
        model = LightningPredictiveModel(backbone=backbone, predictive_method=method)
    return model, method, staged


def build_optimizer(cfg: DictConfig, model: LightningPredictiveModel):
    """The configured optimizer over the currently-trainable parameters.

    Uses the same weight-decay split as ``configure_optimizers``. The LR
    scheduler is deliberately omitted: it needs an attached Trainer to size its
    schedule, and its per-step cost is a few microseconds of Python -- far below
    the resolution of anything measured here.
    """
    optimizer_cfg = hydra.utils.instantiate(cfg.optimizer.optimizer)
    params = [p for p in model.backbone.parameters() if p.requires_grad]
    weight_decay = optimizer_cfg.keywords.get("weight_decay", 0.0)
    return optimizer_cfg(_split_weight_decay(params, weight_decay))


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def _summarize(times_ms: List[float], batch_size: int) -> Dict[str, float]:
    """Median/mean/IQR of a per-step sample, plus the derived rates.

    The median is the headline: a step sample is right-skewed (an allocator
    growth, a dataloader hiccup that lands inside the event window) and a mean
    tracks those outliers while a median does not. Both are reported so a large
    mean-median gap is visible rather than hidden.
    """
    ts = sorted(times_ms)
    n = len(ts)
    median = statistics.median(ts)
    return {
        "step_ms_median": median,
        "step_ms_mean": statistics.fmean(ts),
        # Population sd here, deliberately unlike the across-seed spread: these
        # are *all* the steps that were measured for this seed, so it describes
        # that sample rather than estimating a wider one.
        "step_ms_std": statistics.pstdev(ts) if n > 1 else 0.0,
        "step_ms_iqr": ts[(3 * n) // 4] - ts[n // 4] if n >= 4 else 0.0,
        "step_ms_min": ts[0],
        "steps_per_s": 1e3 / median,
        "samples_per_s": batch_size * 1e3 / median,
        "ms_per_sample": median / batch_size,
        "n_steps": float(n),
    }


def time_stage(
    model: LightningPredictiveModel,
    optimizer: torch.optim.Optimizer,
    batches: _Batches,
    warmup_steps: int,
    measure_steps: int,
    device: str,
) -> Dict[str, float]:
    """Time ``forward -> loss -> backward -> optimizer.step`` over a batch loop.

    Three CUDA events per step split the cost into the forward (including the
    loss) and everything after it, so a method whose *score* is the expensive
    part is distinguishable from one whose head is. Events rather than
    ``perf_counter`` because the GPU runs asynchronously: a wall clock around
    the step would measure kernel *launch*, not execution.

    The batch is moved to the device before the first event is recorded, so the
    host-to-device copy sits outside the measured window -- it is a property of
    the dataset, identical across arms, and including it would dilute the
    difference the benchmark exists to show. The wall-clock number reported
    alongside does include it (and the dataloader), and is the honest "what does
    an epoch cost" figure.
    """
    model.train()
    model.predictive_method.training = True

    def one_step(record: bool):
        batch = batches.next()
        x, y = batch
        if record:
            e0, e1, e2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
            e0.record()
        optimizer.zero_grad(set_to_none=True)
        loss = model.train_loss(model(x), y)
        if record:
            e1.record()
        loss.backward()
        optimizer.step()
        if record:
            e2.record()
            return e0, e1, e2
        return None

    # Released before the warmup, not after: `max_memory_reserved` reports the
    # allocator's high-water mark, and blocks the *previous* stage cached are
    # still reserved. Without this, stage 2 of a staged run inherits stage 1's
    # reservation and reports a footprint it does not have -- which is exactly
    # the number the batch-size pin is chosen from.
    torch.cuda.empty_cache()

    # Warmup: allocator growth, cudnn autotune and the first-call compile of any
    # fused kernel all land here rather than in the sample.
    for _ in range(warmup_steps):
        one_step(record=False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    events = []
    wall_t0 = time.perf_counter()
    for _ in range(measure_steps):
        events.append(one_step(record=True))
    torch.cuda.synchronize()
    wall_s = time.perf_counter() - wall_t0

    total = [e0.elapsed_time(e2) for e0, _, e2 in events]
    fwd = [e0.elapsed_time(e1) for e0, e1, _ in events]
    out = _summarize(total, batches.batch_size)
    out["forward_ms_median"] = statistics.median(fwd)
    out["backward_ms_median"] = out["step_ms_median"] - out["forward_ms_median"]
    # Includes the dataloader and the H2D copy; the gap to `step_ms_median`
    # is the input pipeline, and a large gap means the arm is data-bound and
    # its compute number is not what limits a real run.
    out["wall_ms_per_step"] = wall_s * 1e3 / measure_steps
    out["peak_alloc_gb"] = torch.cuda.max_memory_allocated(device) / 2**30
    out["peak_reserved_gb"] = torch.cuda.max_memory_reserved(device) / 2**30
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _arm_id(cfg: DictConfig) -> Dict[str, Any]:
    """Facets identifying the arm, using the Hydra choices rather than names."""
    choices = HydraConfig.get().runtime.choices
    method_cfg = cfg.get("method", {})
    return {
        "dataset": choices.get("datamodule", "?"),
        # The class, not the config-group name: results are grouped by
        # datamodule type, and a group name like `depth` does not determine it.
        "datamodule_cls": str(cfg.datamodule._target_).rsplit(".", 1)[-1],
        "backbone": choices.get("backbone", "?"),
        # Forecast horizon: part of a time-series run's identity (the paper
        # reports several horizons T), and it sets the event dimension, so it
        # must not be collapsed away when rows from two horizons meet in one file.
        "pred_len": (int(cfg.datamodule["pred_len"])
                     if "pred_len" in cfg.datamodule else None),
        "method": choices.get("method", "?"),
        "features": str(cfg.get("datamodule", {}).get("features", "")) or None,
        "loss": str(method_cfg.get("loss", "")),
        "stage1_loss": str(method_cfg.get("stage1_loss", "")) or None,
        "stage2_loss": str(method_cfg.get("stage2_loss", "")) or None,
        "rank": int(method_cfg.get("rank") or 0),
        "n_train_samples": int(method_cfg.get("n_train_samples") or 0),
    }


def _weighted_effective(rows: List[Dict[str, Any]], cfg: DictConfig) -> Optional[float]:
    """Epoch-weighted per-step time for a staged run.

    Stage 2 is both cheaper (the trunk is frozen, so no backward through it) and
    shorter than stage 1, so quoting either stage alone misstates what the method
    costs end to end. The weights are the configured stage budgets, which is what
    a real run spends.
    """
    by_stage = {r["stage"]: r for r in rows}
    if 1 not in by_stage or 2 not in by_stage:
        return None
    e1 = float(cfg.get("stage1_max_epochs", 0) or 0)
    e2 = float(cfg.get("stage2_max_epochs", 0) or 0)
    if e1 + e2 <= 0:
        return None
    return (
        by_stage[1]["step_ms_median"] * e1 + by_stage[2]["step_ms_median"] * e2
    ) / (e1 + e2)


def benchmark(cfg: DictConfig) -> List[Dict[str, Any]]:
    """Time one training step of the configured arm for every benchmark seed.

    Returns one row per seed (and per stage for two-stage runs) with the
    per-step timing statistics, peak memory and the arm's identifying facets.
    """
    device = str(cfg.benchmark.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("benchmark.device is cuda but no GPU is visible.")
    torch.backends.cudnn.benchmark = bool(cfg.benchmark.cudnn_benchmark)

    batch_size = _pin_batch_size(cfg, cfg.benchmark.get("batch_size"))
    log.info(f"Benchmarking at pinned batch_size={batch_size}")

    # The datamodule is built once and shared across seeds. Only the *model* is
    # re-seeded and rebuilt: seeds are here to cover initialization and kernel
    # scheduling variance, and re-reading a multi-GB split per seed would spend
    # the whole budget on I/O for a factor the timing does not depend on.
    L.seed_everything(int(cfg.get("data_seed", 42)), workers=True)
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup("fit")

    arm = _arm_id(cfg)
    arm["batch_size"] = batch_size
    arm["event_dim"] = int(datamodule.get_total_dimension())
    # Epoch geometry, so a per-step time converts into total GPU hours for a
    # full training run. Both depend on the pinned batch size,
    # so they are recorded per run rather than looked up later.
    arm["steps_per_epoch"] = len(datamodule.train_dataloader())
    arm["max_epochs"] = int(cfg.trainer.max_epochs)
    arm["stage1_max_epochs"] = int(cfg.get("stage1_max_epochs", 0) or 0)
    arm["stage2_max_epochs"] = int(cfg.get("stage2_max_epochs", 0) or 0)
    arm["gpu"] = torch.cuda.get_device_name(0) if device.startswith("cuda") else platform.processor()
    arm["torch"] = torch.__version__

    rows: List[Dict[str, Any]] = []
    for seed in cfg.benchmark.seeds:
        L.seed_everything(int(seed), workers=True)
        model, method, staged = build_model(cfg, datamodule)
        model.to(device)
        batches = _Batches(datamodule, batch_size, device)
        n_params = sum(p.numel() for p in model.backbone.parameters())

        # Stage 2 must be timed on the *same* module that ran stage 1: entering
        # it flips requires_grad on the trunk, which is the whole reason it is
        # cheaper, so a freshly built model would not reproduce it.
        stages = [1, 2] if staged else [None]
        seed_rows: List[Dict[str, Any]] = []
        for stage in stages:
            if stage is not None:
                model.set_stage(stage)
            optimizer = build_optimizer(cfg, model)
            trainable = sum(
                p.numel() for p in model.backbone.parameters() if p.requires_grad
            )
            label = f"seed={seed}" + (f" stage={stage}" if stage else "")
            try:
                stats = time_stage(
                    model,
                    optimizer,
                    batches,
                    int(cfg.benchmark.warmup_steps),
                    int(cfg.benchmark.measure_steps),
                    device,
                )
                status = "ok"
                budget = float(cfg.benchmark.get("memory_budget_gb") or 0)
                if budget and stats["peak_reserved_gb"] > budget:
                    # Not an error: the measurement is valid, it just does not
                    # fit the configured memory budget of the target GPU, so the
                    # pinned batch size needs to come down for *every* arm.
                    status = "over_budget"
                    log.warning(
                        f"{label}: peak reserved "
                        f"{stats['peak_reserved_gb']:.1f} GiB exceeds the "
                        f"{budget:.0f} GiB budget -- lower benchmark.batch_size."
                    )
            except torch.cuda.OutOfMemoryError:
                # Recorded rather than raised: an OOM at the pinned batch size is
                # itself a result (it is what forces the batch size down for every
                # arm), and the remaining seeds/stages still have something to say.
                log.error(f"OOM at batch_size={batch_size} ({label})")
                torch.cuda.empty_cache()
                stats, status = {}, "oom"
            row = {
                **arm,
                "seed": int(seed),
                "stage": stage,
                "staged": staged,
                "status": status,
                "params": n_params,
                "trainable_params": trainable,
                **stats,
            }
            seed_rows.append(row)
            if status != "oom":
                log.info(
                    f"{arm['dataset']}/{arm['method']} {label}: "
                    f"{stats['step_ms_median']:.2f} ms/step "
                    f"({stats['steps_per_s']:.2f} steps/s, "
                    f"{stats['samples_per_s']:.1f} samples/s, "
                    f"peak {stats['peak_alloc_gb']:.2f} GiB)"
                )
        if staged and all(r["status"] != "oom" for r in seed_rows):
            eff = _weighted_effective(seed_rows, cfg)
            for r in seed_rows:
                r["effective_step_ms"] = eff
        rows.extend(seed_rows)

        del model, method, batches
        torch.cuda.empty_cache()

    return rows


def _write(rows: List[Dict[str, Any]], cfg: DictConfig) -> None:
    """Append to the results file ``benchmark.out`` and drop a copy in the run directory."""
    shared = Path(cfg.benchmark.out)
    shared.parent.mkdir(parents=True, exist_ok=True)
    run_dir = Path(cfg.paths.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(r) + "\n" for r in rows)
    with open(shared, "a") as fh:
        fh.write(payload)
    with open(run_dir / "runtime.jsonl", "w") as fh:
        fh.write(payload)
    log.info(f"Wrote {len(rows)} rows to {shared}")


def _report(rows: List[Dict[str, Any]]) -> None:
    """Per-arm summary over seeds: median step time, its spread, and the rates."""
    ok = [r for r in rows if r["status"] != "oom"]
    if not ok:
        log.error("No successful measurements.")
        return
    groups: Dict[Any, List[Dict[str, Any]]] = {}
    for r in ok:
        groups.setdefault((r["dataset"], r["method"], r["stage"]), []).append(r)
    print()
    print(f"{'arm':38s} {'s/step':>10s} {'steps/s':>9s} {'samples/s':>11s} {'peak GiB':>9s}")
    eff = [r["effective_step_ms"] for r in ok if r.get("effective_step_ms")]
    for (dataset, method, stage), rs in groups.items():
        med = [r["step_ms_median"] for r in rs]
        name = f"{dataset}/{method}" + (f" (stage {stage})" if stage else "")
        # Sample sd across seeds, matching what the table reports.
        sd = statistics.stdev(med) if len(med) > 1 else 0.0
        print(
            f"{name:38s} {statistics.fmean(med) / 1e3:10.4f} "
            f"{1e3 / statistics.fmean(med):9.2f} "
            f"{statistics.fmean([r['samples_per_s'] for r in rs]):11.1f} "
            f"{max(r['peak_alloc_gb'] for r in rs):9.2f}"
            + (f"   +/- {sd / 1e3:.4f} s across {len(med)} seeds" if len(med) > 1 else "")
        )
    if eff:
        # The epoch-weighted cost of a staged run. Neither stage alone is the
        # method's price: stage 2 is cheaper *and* shorter, so this is the number
        # that belongs next to the single-stage arms.
        print(f"{'  -> effective (epoch-weighted)':38s} "
              f"{statistics.fmean(eff) / 1e3:10.4f} "
              f"{1e3 / statistics.fmean(eff):9.2f}")


def _already_measured(cfg: DictConfig) -> bool:
    """Whether `benchmark.out` already holds this arm for every configured seed.

    Matched on the facets `make_runtime_table.py` aggregates by -- dataset,
    features, horizon, method, rank -- plus the batch size, loss and member
    count, so a changed pin or objective is measured again rather than skipped.
    An `oom` row counts as measured: "does not fit" is the result for that arm.
    """
    out = Path(cfg.benchmark.out)
    if not out.is_file():
        return False
    arm = _arm_id(cfg)
    arm["batch_size"] = _pin_batch_size(cfg, cfg.benchmark.get("batch_size"))
    facets = ("dataset", "features", "pred_len", "method", "rank", "loss",
              "n_train_samples", "batch_size")
    seeds = set()
    for line in out.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if all(row.get(k) == arm[k] for k in facets):
                seeds.add(row.get("seed"))
    return {int(s) for s in cfg.benchmark.seeds} <= seeds


@hydra.main(version_base="1.3", config_path="../../configs", config_name="benchmark")
def main(cfg: DictConfig) -> None:
    if cfg.benchmark.get("skip_measured", False) and _already_measured(cfg):
        log.info(f"Skipping {_arm_id(cfg)}: already in {cfg.benchmark.out} "
                 "for every seed (benchmark.skip_measured=true)")
        return
    rows = benchmark(cfg)
    _write(rows, cfg)
    _report(rows)


if __name__ == "__main__":
    main()
