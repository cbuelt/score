"""Runtime-timing callback for scientific cost comparison across methods.

Tracks two complementary quantities and logs both to whatever logger the Trainer
uses (e.g. W&B):

- **Per-epoch wall time** (`time/train_epoch_s`): includes data loading, the forward/
  backward, and any per-epoch overhead. This is the practical "what does a run cost"
  number. Robust without synchronization because epochs are long.

- **Per-step compute time** (`time/train_step_ms`): the isolated forward+backward cost
  of one training step, measured with CUDA events so async GPU execution is captured
  correctly (a naive ``time.time()`` would only measure kernel launch). Warmup steps
  and the first epoch are excluded; the warmup-free **median** and **IQR** are logged
  as run-summary values at the end of training, which is the metric to report when
  comparing methods.

Designed as a ``lightning.pytorch.Callback`` so timing is orthogonal to the model and
opt-in per run via the Hydra callbacks config group.
"""

from __future__ import annotations

import statistics
import time
from typing import Any, List, Optional

import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.utilities import rank_zero_only


class TimingCallback(Callback):
    """Logs per-epoch wall time and warmup-free per-step compute time.

    :param warmup_steps: Number of training steps to discard at the start of each
        kept epoch before recording per-step times (allocator / cudnn autotune warmup).
    :param skip_first_epoch: If True, exclude the entire first training epoch from the
        per-step statistics (its steps are systematically slower).
    :param flush_interval: How many recorded steps to buffer before synchronizing the
        CUDA events and reading their elapsed time, to keep sync overhead low.
    """

    def __init__(
        self,
        warmup_steps: int = 50,
        skip_first_epoch: bool = True,
        flush_interval: int = 50,
    ) -> None:
        super().__init__()
        self.warmup_steps = warmup_steps
        self.skip_first_epoch = skip_first_epoch
        self.flush_interval = flush_interval

        # Per-epoch wall-clock timing
        self._epoch_t0: Optional[float] = None

        # Per-step timing state
        self._use_cuda = torch.cuda.is_available()
        self._step_in_epoch = 0
        # Kept per-step times in milliseconds (warmup-free).
        self._step_times_ms: List[float] = []
        # Throughput samples (samples/s) aligned with kept steps.
        self._samples_per_s: List[float] = []
        # Pending CUDA-event pairs awaiting a synchronized read: (start, end, batch_n).
        self._pending: List[tuple] = []
        # CPU-path start time for the current step.
        self._cpu_t0: Optional[float] = None
        self._cur_start_event: Optional[torch.cuda.Event] = None
        self._cur_batch_size: int = 0

    # --- per-epoch wall time -------------------------------------------------

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._epoch_t0 = time.perf_counter()
        self._step_in_epoch = 0

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self._epoch_t0 is None:
            return
        epoch_s = time.perf_counter() - self._epoch_t0
        # Logged from inside `on_train_epoch_end`, where Lightning only permits
        # `on_epoch=True`. There is one call per epoch, so the epoch-level
        # "aggregation" is over a single value and the logged scalar is exact.
        pl_module.log("time/train_epoch_s", epoch_s, on_step=False, on_epoch=True,
                      sync_dist=False, rank_zero_only=True)
        self._flush(force=True)

    # --- per-step compute time ----------------------------------------------

    def _epoch_is_kept(self, trainer: Trainer) -> bool:
        return not (self.skip_first_epoch and trainer.current_epoch == 0)

    def on_train_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int
    ) -> None:
        if not self._epoch_is_kept(trainer) or self._step_in_epoch < self.warmup_steps:
            self._cur_start_event = None
            self._cpu_t0 = None
            return

        # Derive batch size from the input tensor for throughput.
        self._cur_batch_size = _infer_batch_size(batch)

        if self._use_cuda:
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            self._cur_start_event = start
        else:
            self._cpu_t0 = time.perf_counter()

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._step_in_epoch += 1

        if self._use_cuda:
            if self._cur_start_event is None:
                return
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self._pending.append((self._cur_start_event, end, self._cur_batch_size))
            self._cur_start_event = None
            if len(self._pending) >= self.flush_interval:
                self._flush()
        else:
            if self._cpu_t0 is None:
                return
            step_ms = (time.perf_counter() - self._cpu_t0) * 1e3
            self._record_step(pl_module, step_ms, self._cur_batch_size)
            self._cpu_t0 = None

    def _flush(self, force: bool = False) -> None:
        """Synchronize buffered CUDA events and record their elapsed times."""
        if not self._use_cuda or not self._pending:
            return
        # Synchronizing the last end event guarantees all earlier ones are done too.
        self._pending[-1][1].synchronize()
        for start, end, batch_n in self._pending:
            step_ms = start.elapsed_time(end)  # milliseconds
            self._record_step(None, step_ms, batch_n)
        self._pending.clear()

    def _record_step(self, pl_module: Optional[LightningModule], step_ms: float, batch_n: int) -> None:
        self._step_times_ms.append(step_ms)
        if batch_n > 0 and step_ms > 0:
            self._samples_per_s.append(batch_n / (step_ms / 1e3))
        # Log the live curve only on the CPU path, where we hold pl_module and are not
        # inside a deferred flush. (CUDA per-step curve points are summarized at the end
        # to avoid logging from outside a Lightning step context.)
        if pl_module is not None:
            pl_module.log("time/train_step_ms", step_ms, on_step=True, on_epoch=False,
                          sync_dist=False, rank_zero_only=True)

    # --- summary -------------------------------------------------------------

    @rank_zero_only
    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._flush(force=True)
        if not self._step_times_ms:
            return

        times = sorted(self._step_times_ms)
        median = statistics.median(times)
        n = len(times)
        q1 = times[n // 4]
        q3 = times[(3 * n) // 4]

        summary = {
            "time/train_step_ms_median": median,
            "time/train_step_ms_iqr": q3 - q1,
            "time/train_step_ms_n": float(n),
        }
        if self._samples_per_s:
            summary["time/samples_per_s_median"] = statistics.median(self._samples_per_s)

        logger = getattr(pl_module, "logger", None)
        experiment = getattr(logger, "experiment", None)
        # Prefer writing to the W&B run summary so values are sortable across runs.
        if experiment is not None and hasattr(experiment, "summary"):
            try:
                experiment.summary.update(summary)
                return
            except Exception:
                pass
        # Fallback: log as scalars.
        for k, v in summary.items():
            pl_module.log(k, v, on_step=False, on_epoch=False, rank_zero_only=True)


def _infer_batch_size(batch: Any) -> int:
    """Best-effort batch size from a ``(x, y)`` batch or a bare tensor."""
    if isinstance(batch, (tuple, list)) and len(batch) > 0:
        batch = batch[0]
    if isinstance(batch, torch.Tensor) and batch.dim() > 0:
        return batch.shape[0]
    return 0
