"""Tests for the epoch-budget ablation's plumbing.

The study's whole claim rests on the two arms of a rung getting the *same* total
epoch budget, and each arm is driven by keys that are silently inert for the
other: a staged run takes its ceilings from ``stage{1,2}_max_epochs`` and ignores
``trainer.max_epochs``, a single-stage run does the reverse. Nothing about a
wrongly-driven run looks wrong -- it trains, scores and files a row -- so the
checks live here instead.

Everything is pure config/JSON logic: no model is built and no epoch is run.
"""

import json
from typing import List

import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict

from src.evaluation.epoch_budget import (
    ARMS,
    SCHEMA_VERSION,
    apply_arm,
    identity,
    needs_refresh,
    resolve_budget,
    row_key,
)

#: (max_epochs, frac) -> (budget, stage1, stage2) for the whole ladder, spelled
#: out rather than recomputed: these are the ten cells the study trains, and the
#: 1.0 rung must reproduce the split the experiment configs already carry
#: (euppbench 30+10, time series 75+25) or the top of the ladder is not the
#: main training configuration it is claimed to be.
LADDER = {
    # EUPPBench (40 epochs): 0.1 is refused, see
    # `test_shortest_euppbench_rung_is_refused`.
    (40, 0.2): (8, 6, 2),
    (40, 0.3): (12, 9, 3),
    (40, 0.4): (16, 12, 4),
    (40, 0.5): (20, 15, 5),
    (40, 0.6): (24, 18, 6),
    (40, 0.7): (28, 21, 7),
    (40, 0.8): (32, 24, 8),
    (40, 0.9): (36, 27, 9),
    (40, 1.0): (40, 30, 10),
    # Time series (100 epochs). Half of these are not multiples of 4, so their
    # split is rounded half-up and deviates from 3:1 by one epoch.
    (100, 0.1): (10, 8, 2),
    (100, 0.2): (20, 15, 5),
    (100, 0.3): (30, 23, 7),
    (100, 0.4): (40, 30, 10),
    (100, 0.5): (50, 38, 12),
    (100, 0.6): (60, 45, 15),
    (100, 0.7): (70, 53, 17),
    (100, 0.8): (80, 60, 20),
    (100, 0.9): (90, 68, 22),
    (100, 1.0): (100, 75, 25),
}

#: Rungs whose budget divides by 4 and so realise the 3:1 split exactly.
EXACT = {k for k, (budget, _s1, _s2) in LADDER.items() if budget % 4 == 0}

EUPP = "train/euppbench"
EUPP_WHITENED = "train/euppbench_score"
ETT = "train/time_series/univariate/etth1"
ETT_WHITENED = "train/time_series/univariate/etth1_score"


def _cfg(experiment: str, overrides: List[str] = ()) -> DictConfig:
    """The composed ablation config for one experiment.

    Composed for real rather than hand-built, so these tests also cover the
    config tree itself -- that ``configs/epoch_budget.yaml`` still composes the
    training experiments, and that the epoch keys it edits exist where the
    driver looks for them.
    """
    with initialize(version_base="1.3", config_path="../../configs"):
        return compose(
            config_name="epoch_budget",
            overrides=[
                f"experiments={experiment}",
                "epoch_budget.arm=marginal",
                "epoch_budget.frac=1.0",
                *overrides,
            ],
        )


@pytest.fixture(autouse=True)
def _clear_hydra():
    yield
    GlobalHydra.instance().clear()


# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key,expected", sorted(LADDER.items()))
def test_resolve_budget_matches_the_ladder(key, expected):
    max_epochs, frac = key
    assert resolve_budget(max_epochs, frac) == expected


@pytest.mark.parametrize("key", sorted(LADDER))
def test_stages_sum_to_the_budget(key):
    budget, stage1, stage2 = resolve_budget(*key)
    assert stage1 + stage2 == budget


@pytest.mark.parametrize("key", sorted(EXACT))
def test_split_is_exact_where_the_budget_allows(key):
    """A multiple of 4 must give exactly 3:1 -- no rounding drift."""
    _budget, stage1, stage2 = resolve_budget(*key)
    assert stage1 == 3 * stage2


@pytest.mark.parametrize("key", sorted(set(LADDER) - EXACT))
def test_rounded_split_moves_at_most_one_epoch(key):
    """The rounded rungs must sit within one epoch of the exact 3:1 split.

    This is the tolerance the 0.1-spaced ladder buys: `resolve_budget` rounds
    a budget that is not a multiple of 4 rather than refusing it, so the thing
    worth pinning is that it rounds to the *nearest* split rather than drifting.
    """
    budget, stage1, stage2 = resolve_budget(*key)
    assert abs(stage1 - 0.75 * budget) <= 1.0
    assert abs(stage2 - 0.25 * budget) <= 1.0
    assert stage1 + stage2 == budget


def test_full_rung_reproduces_the_config_split():
    """frac=1.0 must be the main training configuration, not merely close to it."""
    for experiment, expected in ((EUPP_WHITENED, (30, 10)), (ETT_WHITENED, (75, 25))):
        cfg = _cfg(experiment)
        budget, stage1, stage2 = resolve_budget(int(cfg.trainer.max_epochs), 1.0)
        assert (stage1, stage2) == expected
        assert (int(cfg.stage1_max_epochs), int(cfg.stage2_max_epochs)) == expected
        assert budget == int(cfg.trainer.max_epochs)


def test_shortest_euppbench_rung_is_refused():
    """0.1 of EUPPBench's 40 epochs is 4 -> 3+1, and 1 epoch cannot anneal.

    Its warmup is subtractive and clamped to `stage2 - 1`, so a 1-epoch stage 2
    is pure warmup: that rung would run a different recipe from every other one
    and the difference would read as an effect of the budget. Refused rather
    than silently trained, which is why the EUPPBench ladder starts at 0.2.
    """
    with pytest.raises(ValueError, match="too few to anneal"):
        resolve_budget(40, 0.1)
    # And the message points at the first rung that does work.
    with pytest.raises(ValueError, match=r"frac >= 0\.20"):
        resolve_budget(40, 0.1)


def test_tenth_rungs_all_resolve():
    """Every rung the sweep submits must resolve, per data set."""
    for max_epochs, first in ((100, 1), (40, 2)):
        for tenth in range(first, 11):
            budget, stage1, stage2 = resolve_budget(max_epochs, tenth / 10)
            assert stage2 >= 2 and stage1 + stage2 == budget


def test_empty_budget_raises():
    with pytest.raises(ValueError, match="has to train something"):
        resolve_budget(40, 0.0)


# ---------------------------------------------------------------------------
# Driving the arms
# ---------------------------------------------------------------------------


def test_ours_sets_the_stage_ceilings():
    cfg = _cfg(EUPP_WHITENED)
    plan = apply_arm(cfg, "ours", 0.6)
    assert plan["budget"] == 24
    assert (plan["stage1_epochs"], plan["stage2_epochs"]) == (18, 6)
    # The keys a staged run is actually driven by.
    assert int(cfg.stage1_max_epochs) == 18
    assert int(cfg.stage2_max_epochs) == 6
    # And the (inert but reported) total, so nothing downstream reads 40.
    assert int(cfg.trainer.max_epochs) == 24


def test_marginal_sets_the_trainer_ceiling_and_clears_the_stage_keys():
    cfg = _cfg(EUPP)
    plan = apply_arm(cfg, "marginal", 0.4)
    assert plan["budget"] == 16
    assert int(cfg.trainer.max_epochs) == 16
    assert plan["stage1_epochs"] is None and plan["stage2_epochs"] is None
    # Left at the config's 30/10 a single-stage row would claim stage budgets it
    # never ran (and `sync_checkpoints` reads this key to decide "staged").
    assert cfg.stage1_max_epochs is None
    assert cfg.stage2_max_epochs is None


def test_arm_must_match_the_experiment():
    with pytest.raises(ValueError, match="needs a staged experiment"):
        apply_arm(_cfg(EUPP), "ours", 1.0)
    with pytest.raises(ValueError, match="needs a single-stage experiment"):
        apply_arm(_cfg(EUPP_WHITENED), "marginal", 1.0)


def test_unknown_arm_raises():
    with pytest.raises(ValueError, match="must be one of"):
        apply_arm(_cfg(EUPP), "stage1-only", 1.0)


@pytest.mark.parametrize("arm", ARMS)
def test_arms_agree_on_the_total(arm):
    experiment = {"marginal": ETT, "ours": ETT_WHITENED}[arm]
    plan = apply_arm(_cfg(experiment), arm, 0.6)
    assert plan["budget"] == 60
    if arm == "ours":
        assert plan["stage1_epochs"] + plan["stage2_epochs"] == 60


# ---------------------------------------------------------------------------
# The warmup clamp
# ---------------------------------------------------------------------------


def test_warmup_clamped_only_where_it_would_swallow_the_stage():
    """The shortest EUPPBench rung gives stage 2 two epochs; warmup is three.

    Left alone the LR would never anneal there (train._log_warmup_budget only
    warns), so that one rung would run a different recipe from every other rung
    and the difference would be read as an effect of the budget.
    """
    cfg = _cfg(EUPP_WHITENED)
    plan = apply_arm(cfg, "ours", 0.2)
    assert (plan["stage1_epochs"], plan["stage2_epochs"]) == (6, 2)
    assert plan["warmup_stage1"] == 3        # 6 epochs: untouched
    assert plan["warmup_stage2"] == 1        # 2 epochs: clamped

    # One rung up there is room, so nothing is clamped.
    plan = apply_arm(_cfg(EUPP_WHITENED), "ours", 0.4)
    assert (plan["warmup_stage1"], plan["warmup_stage2"]) == (3, 3)


def test_clamp_does_not_disturb_the_other_stage_or_the_lr():
    """EUPPBench composes no per-stage bundles: both stages read cfg.optimizer.

    So the clamp has to deep-copy that bundle into the stage it is narrowing.
    Editing it in place would move stage 1's schedule too, and re-composing the
    group from scratch would bring its default lr (1e-3) instead of the
    experiment's (2e-4).
    """
    cfg = _cfg(EUPP_WHITENED)
    lr = float(cfg.optimizer.optimizer.lr)
    apply_arm(cfg, "ours", 0.2)
    assert float(cfg.optimizer_stage2.optimizer.lr) == lr
    assert int(cfg.optimizer_stage2.scheduler.warmup_epochs) == 1
    # Stage 1 still resolves to the shared bundle, unnarrowed.
    assert int(cfg.optimizer.scheduler.warmup_epochs) == 3


def test_marginal_clamp_targets_the_bundle_the_single_stage_path_reads():
    """A single-stage run reads cfg.optimizer, never cfg.optimizer_stage1.

    Writing the clamp to ``optimizer_stage1`` would be a silent no-op here. On
    the configured ladder this arm never needs clamping (its shortest rung is 8
    epochs against a 3-epoch warmup), so the situation is constructed with a
    longer warmup rather than a shorter budget: `resolve_budget` refuses a budget
    whose stage 2 would fall under `MIN_STAGE2`, and it refuses it for *both*
    arms -- a rung only exists if both sides of it can be trained.
    """
    cfg = _cfg(EUPP)
    with open_dict(cfg):
        cfg.trainer.max_epochs = 16
        cfg.optimizer.scheduler.warmup_epochs = 9
    plan = apply_arm(cfg, "marginal", 0.5)
    assert plan["budget"] == 8
    assert plan["warmup_stage1"] == 7          # clamped from 9 to budget - 1
    assert int(cfg.optimizer.scheduler.warmup_epochs) == 7


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def test_identity_of_the_station_arms():
    ours = identity(_cfg(EUPP_WHITENED))
    assert ours["experiment"] == "station_postprocessing"
    assert ours["dataset"] == "euppbench"
    assert ours["mode"] is None and ours["horizon"] is None
    # The slug the main runs' checkpoints are filed under, so a row joins to them.
    assert ours["slug"] == "whitened_graph_matern_s1crps_dense_kernel_score"
    assert ours["stage2_loss"] == "dense_kernel_score"
    assert ours["loss"] is None

    control = identity(_cfg(EUPP, ["method=diag", "method.loss=crps"]))
    assert control["slug"] == "phys_diag_crps"
    assert control["loss"] == "crps"
    assert control["stage1_loss"] is None


def test_identity_of_the_univariate_arms():
    ours = identity(_cfg(ETT_WHITENED))
    assert (ours["experiment"], ours["dataset"]) == ("time_series", "etth1")
    assert ours["mode"] == "univariate"
    # The horizon is part of the identity: it sets the event dimension.
    assert ours["horizon"] == 96
    # `_s1crps` is present because the sync script's default stage-1 loss is
    # `marginal_kernel_score`, not `crps` -- the tree carries these slugs
    # spelled out (whitened_spectral_diag_lora2_s1crps_kernel_score et al.).
    assert ours["slug"] == "whitened_spectral_diag_s1crps_kernel_score"

    longer = identity(_cfg(ETT_WHITENED, ["datamodule.pred_len=720"]))
    assert longer["horizon"] == 720


def test_identity_rejects_an_unknown_method():
    cfg = _cfg(EUPP)
    with open_dict(cfg):
        cfg.method._target_ = "src.methods.score.NotAMethod"
    with pytest.raises(ValueError, match="unknown method"):
        identity(cfg)


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


def _row(**overrides):
    row = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "station_postprocessing",
        "dataset": "euppbench",
        "mode": None,
        "horizon": None,
        "arm": "ours",
        "budget": 24,
        "seed": 3,
    }
    row.update(overrides)
    return row


def test_missing_file_needs_a_run(tmp_path):
    refresh, reason = needs_refresh(tmp_path / "nope.jsonl", row_key(_row()))
    assert refresh and "no results file" in reason


def test_present_cell_is_skipped(tmp_path):
    path = tmp_path / "scores.jsonl"
    path.write_text(json.dumps(_row()) + "\n")
    refresh, reason = needs_refresh(path, row_key(_row()))
    assert not refresh and reason == "already present"


def test_a_different_cell_still_needs_a_run(tmp_path):
    path = tmp_path / "scores.jsonl"
    path.write_text(json.dumps(_row()) + "\n")
    for other in (_row(seed=4), _row(budget=32), _row(arm="marginal"),
                  _row(dataset="etth1")):
        refresh, _ = needs_refresh(path, row_key(other))
        assert refresh, other


def test_stale_schema_is_retrained(tmp_path):
    path = tmp_path / "scores.jsonl"
    path.write_text(json.dumps(_row(schema_version=SCHEMA_VERSION - 1)) + "\n")
    refresh, _ = needs_refresh(path, row_key(_row()))
    assert refresh


def test_overwrite_forces_a_run(tmp_path):
    path = tmp_path / "scores.jsonl"
    path.write_text(json.dumps(_row()) + "\n")
    refresh, reason = needs_refresh(path, row_key(_row()), overwrite=True)
    assert refresh and "overwrite" in reason


def test_partial_last_line_does_not_block_a_resume(tmp_path):
    """A sweep killed mid-write leaves a truncated line; it must not be fatal."""
    path = tmp_path / "scores.jsonl"
    path.write_text(json.dumps(_row(seed=1)) + "\n" + '{"experiment": "stat')
    refresh, _ = needs_refresh(path, row_key(_row(seed=2)))
    assert refresh
    refresh, _ = needs_refresh(path, row_key(_row(seed=1)))
    assert not refresh
