"""Tests pinning the PatchTST LR schedules to upstream's own formulas.

The time-series experiments are meant to reproduce
https://github.com/yuqinie98/PatchTST, whose optimization recipe is per-dataset:
its ETTh/weather scripts run the default ``lradj='type3'`` staircase decay, its
ETTm/traffic/electricity scripts run ``lradj='TST'`` (a stock ``OneCycleLR``
stepped per batch). Both are reimplemented here under this repo's
``(optimizer, warmup_steps, max_steps, ...)`` scheduler contract, so the thing
worth testing is not that they run but that they produce upstream's *numbers* --
a silent drift here would look like a modelling result.

The references below are transcribed from upstream ``utils/tools.py`` and
``exp/exp_main.py``, not from our implementation.
"""

from functools import partial
from types import SimpleNamespace

import pytest
import torch

from src.models.gaussian_predictive import _build_optimizer_and_scheduler
from src.utils.lr_scheduler import (
    LinearWarmupCosineAnnealingLR,
    OneCycleLR,
    StepwiseExponentialLR,
)

BASE_LR = 1e-4
STEPS_PER_EPOCH = 64
EPOCHS = 100
MAX_STEPS = STEPS_PER_EPOCH * EPOCHS


def _optimizer(lr: float = BASE_LR) -> torch.optim.Optimizer:
    return torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=lr)


def _lrs_per_step(optimizer, scheduler, n_steps: int) -> list:
    """The LR actually in force at each of the first ``n_steps`` steps."""
    lrs = []
    for _ in range(n_steps):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return lrs


def _type3_reference(epoch: int) -> float:
    """Upstream ``lradj='type3'`` LR during 0-based epoch ``epoch``.

    Upstream holds the LR in ``optimizer.param_groups`` and rewrites it in
    ``adjust_learning_rate(optimizer, scheduler, epoch + 1, args)`` *after* each
    epoch, with

        lr_adjust = {epoch: lr if epoch < 3 else lr * (0.9 ** ((epoch - 3) // 1))}

    Unrolling the off-by-one: the call made at the end of 0-based epoch ``e``
    passes ``e + 1`` and sets the LR used during epoch ``e + 1``. So epochs 0-3
    all run at the base LR and epoch ``e >= 3`` runs at ``0.9 ** (e - 3)``.
    """
    return BASE_LR * 0.9 ** max(0, epoch - 3)


def test_stepwise_exponential_matches_patchtst_type3():
    """Our type3 reproduces upstream's per-epoch LR exactly, as a staircase."""
    optimizer = _optimizer()
    scheduler = StepwiseExponentialLR(
        optimizer,
        warmup_steps=0,
        max_steps=MAX_STEPS,
        gamma=0.9,
        decay_start_steps=3 * STEPS_PER_EPOCH,
        decay_period_steps=STEPS_PER_EPOCH,
    )
    lrs = _lrs_per_step(optimizer, scheduler, MAX_STEPS)

    for epoch in range(EPOCHS):
        in_epoch = lrs[epoch * STEPS_PER_EPOCH : (epoch + 1) * STEPS_PER_EPOCH]
        # Upstream adjusts only at epoch boundaries, so the LR is flat within one
        # epoch -- this is a staircase, not a smooth per-step decay.
        assert in_epoch == [in_epoch[0]] * STEPS_PER_EPOCH, f"epoch {epoch} not flat"
        assert in_epoch[0] == pytest.approx(_type3_reference(epoch), rel=1e-12)


def test_stepwise_exponential_decay_starts_after_the_flat_epochs():
    """The first decay lands at epoch 4, not epoch 3 (upstream's off-by-one)."""
    optimizer = _optimizer()
    scheduler = StepwiseExponentialLR(
        optimizer,
        warmup_steps=0,
        max_steps=MAX_STEPS,
        gamma=0.9,
        decay_start_steps=3 * STEPS_PER_EPOCH,
        decay_period_steps=STEPS_PER_EPOCH,
    )
    lrs = _lrs_per_step(optimizer, scheduler, 5 * STEPS_PER_EPOCH)
    at_epoch = [lrs[e * STEPS_PER_EPOCH] for e in range(5)]
    assert at_epoch[:4] == [BASE_LR] * 4
    assert at_epoch[4] == pytest.approx(BASE_LR * 0.9, rel=1e-12)


def test_stepwise_exponential_floors_at_eta_min():
    """``eta_min`` bounds the decay; without it type3 runs away to ~0."""
    optimizer = _optimizer()
    floor = BASE_LR / 100
    scheduler = StepwiseExponentialLR(
        optimizer,
        warmup_steps=0,
        max_steps=MAX_STEPS,
        gamma=0.9,
        decay_start_steps=0,
        decay_period_steps=STEPS_PER_EPOCH,
        eta_min=floor,
    )
    lrs = _lrs_per_step(optimizer, scheduler, MAX_STEPS)
    assert min(lrs) == pytest.approx(floor, rel=1e-12)


def test_stepwise_exponential_warms_up_first():
    """type3 itself has no warmup, but the option composes with the decay."""
    optimizer = _optimizer()
    warmup = 2 * STEPS_PER_EPOCH
    scheduler = StepwiseExponentialLR(
        optimizer,
        warmup_steps=warmup,
        max_steps=MAX_STEPS,
        gamma=0.9,
        decay_start_steps=warmup,
        decay_period_steps=STEPS_PER_EPOCH,
    )
    lrs = _lrs_per_step(optimizer, scheduler, warmup + STEPS_PER_EPOCH)
    assert lrs[0] == pytest.approx(0.0)
    assert lrs[:warmup] == sorted(lrs[:warmup])  # monotone ramp
    assert lrs[warmup] == pytest.approx(BASE_LR, rel=1e-12)


def test_one_cycle_matches_upstream_construction():
    """Re-parameterizing ``pct_start`` as ``warmup_steps`` changes no LR.

    Upstream builds ``OneCycleLR(optimizer, steps_per_epoch=..., epochs=...,
    pct_start=..., max_lr=args.learning_rate)``; ours takes the ramp length in
    steps and reads ``max_lr`` off the optimizer. The two must coincide step for
    step, including the ``max_lr / div_factor`` starting point.
    """
    pct_start = 0.2
    ours_opt = _optimizer()
    ours = OneCycleLR(
        ours_opt,
        warmup_steps=int(pct_start * MAX_STEPS),
        max_steps=MAX_STEPS,
    )
    ref_opt = _optimizer()
    ref = torch.optim.lr_scheduler.OneCycleLR(
        ref_opt,
        max_lr=BASE_LR,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS,
        pct_start=pct_start,
    )
    assert _lrs_per_step(ours_opt, ours, MAX_STEPS) == _lrs_per_step(
        ref_opt, ref, MAX_STEPS
    )


def test_one_cycle_peaks_at_the_end_of_warmup():
    """The peak is the configured lr and sits exactly at ``warmup_steps``."""
    optimizer = _optimizer()
    warmup_steps = 20 * STEPS_PER_EPOCH
    scheduler = OneCycleLR(optimizer, warmup_steps=warmup_steps, max_steps=MAX_STEPS)
    lrs = _lrs_per_step(optimizer, scheduler, MAX_STEPS)
    assert lrs[0] == pytest.approx(BASE_LR / 25.0, rel=1e-9)  # div_factor
    # torch ends the ramp-up at `pct_start * total_steps - 1`, so the peak is in
    # force at the last warmup step rather than the first annealing one. Pinned
    # here because the off-by-one is torch's convention, not ours -- see
    # test_one_cycle_matches_upstream_construction for the exact equivalence.
    assert lrs.index(max(lrs)) == warmup_steps - 1
    assert max(lrs) == pytest.approx(BASE_LR, rel=1e-9)
    # torch's final_div_factor divides the *initial* LR (max_lr / div_factor),
    # not max_lr, so the floor is max_lr / (div_factor * final_div_factor). It is
    # only reached at the very last step -- which is why an early stop truncates
    # this schedule's anneal.
    assert lrs[-1] == pytest.approx(BASE_LR / (25.0 * 1e4), rel=1e-2)


@pytest.mark.parametrize("warmup_steps", [0, MAX_STEPS, MAX_STEPS + 1])
def test_one_cycle_rejects_a_ramp_outside_the_budget(warmup_steps):
    """A degenerate pct_start is a config error, not a silent clamp."""
    with pytest.raises(ValueError, match="warmup_steps"):
        OneCycleLR(_optimizer(), warmup_steps=warmup_steps, max_steps=MAX_STEPS)


def test_schedulers_share_the_builder_contract():
    """All three take ``(optimizer, warmup_steps, max_steps)`` and are steppable.

    ``_build_optimizer_and_scheduler`` calls every scheduler this way, so a new
    one that drops an argument would only fail at training time.
    """
    for factory in (
        lambda opt: LinearWarmupCosineAnnealingLR(
            opt, warmup_steps=10, max_steps=MAX_STEPS
        ),
        lambda opt: StepwiseExponentialLR(opt, warmup_steps=10, max_steps=MAX_STEPS),
        lambda opt: OneCycleLR(opt, warmup_steps=10, max_steps=MAX_STEPS),
    ):
        optimizer = _optimizer()
        scheduler = factory(optimizer)
        lrs = _lrs_per_step(optimizer, scheduler, 32)
        assert all(lr >= 0.0 for lr in lrs)


def _stub_module(scheduler_partial, weight_decay: float = 0.0):
    """The minimal surface ``_build_optimizer_and_scheduler`` reads.

    It needs the two partials plus an attached trainer to size the schedule; a
    namespace avoids standing up a Lightning Trainer for what is a pure
    epochs-to-steps calculation.
    """
    return SimpleNamespace(
        optimizer_cfg=partial(
            torch.optim.AdamW, lr=BASE_LR, weight_decay=weight_decay
        ),
        scheduler_cfg=scheduler_partial,
        trainer=SimpleNamespace(
            estimated_stepping_batches=MAX_STEPS, max_epochs=EPOCHS
        ),
    )


def test_builder_converts_every_epoch_keyword_to_steps():
    """``<name>_epochs`` config keys reach the scheduler as ``<name>_steps``.

    Configs are written in epochs because steps-per-epoch is only known once the
    trainer is attached. StepwiseExponentialLR is the case that exercises more
    than ``warmup_epochs``: a decay period left in epochs would decay once per
    *step* and collapse the LR within the first epoch.
    """
    module = _stub_module(
        partial(
            StepwiseExponentialLR,
            warmup_epochs=0,
            gamma=0.9,
            decay_start_epochs=3,
            decay_period_epochs=1,
        )
    )
    scheduler = _build_optimizer_and_scheduler(
        module, [torch.nn.Parameter(torch.zeros(2, 2))]
    )["lr_scheduler"]["scheduler"]
    assert scheduler.warmup_steps == 0
    assert scheduler.decay_start_steps == 3 * STEPS_PER_EPOCH
    assert scheduler.decay_period_steps == STEPS_PER_EPOCH
    assert scheduler.max_steps == MAX_STEPS


def test_builder_sizes_one_cycle_pct_start_from_warmup_epochs():
    """``warmup_epochs`` is upstream's ``pct_start`` in epoch units."""
    module = _stub_module(partial(OneCycleLR, warmup_epochs=20))
    scheduler = _build_optimizer_and_scheduler(
        module, [torch.nn.Parameter(torch.zeros(2, 2))]
    )["lr_scheduler"]["scheduler"]
    assert scheduler.total_steps == MAX_STEPS
    assert scheduler._schedule_phases[0]["end_step"] == pytest.approx(
        0.2 * MAX_STEPS - 1
    )


def test_builder_supplies_warmup_steps_when_the_config_omits_it():
    """A partial without ``warmup_epochs`` still satisfies the contract.

    patchtst_type3 does not warm up, and every scheduler here takes
    ``warmup_steps`` positionally-by-keyword, so the builder must default it
    rather than let the call fail with a missing argument.
    """
    module = _stub_module(partial(StepwiseExponentialLR, gamma=0.9))
    scheduler = _build_optimizer_and_scheduler(
        module, [torch.nn.Parameter(torch.zeros(2, 2))]
    )["lr_scheduler"]["scheduler"]
    assert scheduler.warmup_steps == 0
