import math
from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import OneCycleLR as _TorchOneCycleLR
from torch.optim.lr_scheduler import _LRScheduler

# Every scheduler here takes the same ``(optimizer, warmup_steps, max_steps, ...)``
# contract and is stepped per optimizer step, so they are interchangeable in
# `configure_optimizers`. Configs express the step counts in *epochs* -- any
# `<name>_epochs` scheduler keyword is converted to `<name>_steps` by
# `src.models.gaussian_predictive._build_optimizer_and_scheduler` once the
# trainer knows steps-per-epoch.


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """Linear warmup followed by cosine annealing, stepped per optimizer step.

    For the first ``warmup_steps`` steps the LR rises linearly from
    ``warmup_start_lr`` to each param group's ``base_lr``; thereafter it decays
    along a half-cosine from ``base_lr`` to ``eta_min`` over the remaining
    ``max_steps - warmup_steps`` steps. The LR is computed in closed form from
    ``self.last_epoch`` (the step counter) so it is correct on checkpoint resume,
    unlike the recurrence-based variants.

    :param optimizer: wrapped optimizer.
    :param warmup_steps: number of linear-warmup steps.
    :param max_steps: total number of scheduler steps (warmup + cosine).
    :param warmup_start_lr: LR at the very first step.
    :param eta_min: final (and floor) LR after annealing.
    :param last_epoch: the index of the last step; -1 starts fresh.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        max_steps: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        step = self.last_epoch
        if step < self.warmup_steps:
            # Linear warmup. With warmup_steps>=1, step in [0, warmup_steps-1]
            # maps the LR from warmup_start_lr (step 0) up toward base_lr.
            frac = step / max(self.warmup_steps, 1)
            return [
                self.warmup_start_lr + frac * (base_lr - self.warmup_start_lr)
                for base_lr in self.base_lrs
            ]
        # Cosine annealing over the post-warmup span. Clamp progress to [0, 1] so
        # any steps beyond max_steps simply hold eta_min.
        span = max(self.max_steps - self.warmup_steps, 1)
        progress = min((step - self.warmup_steps) / span, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [
            self.eta_min + (base_lr - self.eta_min) * cosine
            for base_lr in self.base_lrs
        ]


class StepwiseExponentialLR(_LRScheduler):
    """Optional warmup, then a staircase exponential decay of the LR.

    Reproduces PatchTST's default ``lradj='type3'`` schedule, which is the one
    the ETTh and weather benchmarks are tuned under: the LR is held at
    ``base_lr`` for the first few epochs and then multiplied by ``gamma`` once
    per epoch. Upstream applies it in ``adjust_learning_rate`` at each epoch
    boundary; here the decay is a function of the *step* counter (the boundaries
    come in as ``decay_start_steps`` / ``decay_period_steps``), so the schedule
    is stepped per optimizer step like the others and is correct on resume.

    Unlike cosine annealing this never anneals to a terminal LR -- it just keeps
    shrinking geometrically -- so there is no "annealed endpoint" to lose and it
    composes cleanly with early stopping. That is exactly why upstream pairs it
    with a patience.

    ``max_steps`` is accepted for a uniform scheduler contract but unused: the
    decay depends only on how far into training a step is, not on the total.

    :param optimizer: wrapped optimizer.
    :param warmup_steps: number of linear-warmup steps (0 disables warmup, which
        is what ``type3`` does).
    :param max_steps: total scheduler steps; unused, see above.
    :param gamma: multiplicative decay applied once per decay period (upstream
        ``type3`` uses 0.9).
    :param decay_start_steps: steps held at ``base_lr`` before the decay starts.
        Upstream's ``type3`` starts decaying after epoch 3, i.e.
        ``decay_start_epochs: 3``.
    :param decay_period_steps: steps between two decay applications (upstream
        decays every epoch, i.e. ``decay_period_epochs: 1``).
    :param warmup_start_lr: LR at the very first step.
    :param eta_min: floor the decayed LR never falls below.
    :param last_epoch: the index of the last step; -1 starts fresh.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        max_steps: int,
        gamma: float = 0.9,
        decay_start_steps: int = 0,
        decay_period_steps: int = 1,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.gamma = gamma
        self.decay_start_steps = decay_start_steps
        # A sub-epoch period would round to 0 steps and decay every step; one
        # step is the smallest meaningful period.
        self.decay_period_steps = max(decay_period_steps, 1)
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        step = self.last_epoch
        if step < self.warmup_steps:
            frac = step / max(self.warmup_steps, 1)
            return [
                self.warmup_start_lr + frac * (base_lr - self.warmup_start_lr)
                for base_lr in self.base_lrs
            ]
        # Number of completed decay periods since the decay began. Integer
        # division makes this a staircase: the LR is constant within an epoch and
        # drops at the boundary, matching upstream's per-epoch adjustment.
        periods = max(step - self.decay_start_steps, 0) // self.decay_period_steps
        factor = self.gamma**periods
        return [max(base_lr * factor, self.eta_min) for base_lr in self.base_lrs]


class OneCycleLR(_TorchOneCycleLR):
    """:class:`torch.optim.lr_scheduler.OneCycleLR` under this repo's contract.

    Reproduces PatchTST's ``lradj='TST'`` schedule (used by the ETTm, traffic and
    electricity benchmarks), which builds a stock ``OneCycleLR`` over the whole
    epoch budget and steps it per batch. The only change here is the
    parameterization: upstream's ``pct_start`` is expressed as ``warmup_epochs``
    so the ramp-up length is configured in the same units as every other
    scheduler in this module, and ``max_lr`` is read from the optimizer's param
    groups (which carry the configured ``lr``) instead of being passed twice.

    ``pct_start = warmup_steps / max_steps``, so upstream's ``pct_start=0.2``
    over a 100-epoch budget is ``warmup_epochs: 20``. Note that the LR only
    reaches its terminal ``max_lr / final_div_factor`` at the *last* step: unlike
    :class:`StepwiseExponentialLR`, cutting the run short with early stopping
    also cuts the anneal short. Upstream pairs this schedule with a patience
    anyway; keep that in mind when reading a run that stopped early.

    :param optimizer: wrapped optimizer; its param groups' ``lr`` is the peak LR.
    :param warmup_steps: steps of the ramp-up phase, i.e. ``pct_start * max_steps``.
    :param max_steps: total scheduler steps (``total_steps`` upstream).
    :param div_factor: initial LR is ``max_lr / div_factor``.
    :param final_div_factor: divides the *initial* LR, not the peak, so the
        terminal LR is ``max_lr / (div_factor * final_div_factor)``.
    :param cycle_momentum: cycle the optimizer's momentum (Adam/AdamW ``betas[0]``)
        inversely to the LR. True matches upstream's default.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        max_steps: int,
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        anneal_strategy: str = "cos",
        cycle_momentum: bool = True,
        three_phase: bool = False,
        last_epoch: int = -1,
    ) -> None:
        if not 0 < warmup_steps < max_steps:
            raise ValueError(
                "OneCycleLR needs a ramp-up strictly inside the budget, i.e. "
                f"0 < warmup_steps < max_steps; got warmup_steps={warmup_steps}, "
                f"max_steps={max_steps}. Set 0 < warmup_epochs < max_epochs."
            )
        # Read the peak LRs before super().__init__ overwrites each group's `lr`
        # with the ramp-up start (max_lr / div_factor).
        max_lrs = [group["lr"] for group in optimizer.param_groups]
        super().__init__(
            optimizer,
            max_lr=max_lrs,
            total_steps=max_steps,
            pct_start=warmup_steps / max_steps,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            anneal_strategy=anneal_strategy,
            cycle_momentum=cycle_momentum,
            three_phase=three_phase,
            last_epoch=last_epoch,
        )
