"""Tests for the wandb run naming and facet tags built from the Hydra config.

The run name encodes the method facets that distinguish otherwise identically
named runs (transform basis, low-rank order, core activation, stage-2 loss);
the tags mirror them one facet per tag so runs are filterable in the wandb UI.
"""

from typing import List

import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from src.train.train import (
    _DEFAULT_STAGE1_LOSS,
    _apply_wandb_naming,
    _build_run_name,
    _build_wandb_tags,
)


def _cfg(overrides: List[str]) -> DictConfig:
    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(
            config_name="train",
            return_hydra_config=True,
            overrides=["experiments=train/depth_score", *overrides],
        )
        HydraConfig().set_config(cfg)
    return cfg


@pytest.fixture(autouse=True)
def _clear_hydra():
    yield
    GlobalHydra.instance().clear()


def test_default_whitened_name_and_tags():
    cfg = _cfg([])
    # Fourier default: no transform/lora/activation parts; staged runs are
    # named by the stage-2 (core) loss.
    assert _build_run_name(cfg) == "depth_whitened_spectral_diag_kernel_score"
    tags = _build_wandb_tags(cfg)
    assert "dataset:depth" in tags
    assert "method:whitened_spectral_diag" in tags
    assert "stage1:crps" in tags
    assert "stage2:kernel_score" in tags
    assert "transform:fourier" in tags
    assert "act:softmax" in tags
    assert "seed:42" in tags
    # rank 0 is not tagged (only genuine low-rank runs carry a rank tag).
    assert not any(t.startswith("rank:") for t in tags)


def test_dct_method_choice_not_duplicated():
    cfg = _cfg(["method=score_dct", "method.stage2_loss=nll"])
    # "dct" is already part of the method choice name, so no transform part is
    # appended; the tag still carries the parsed kind.
    assert _build_run_name(cfg) == "depth_whitened_dct_nll"
    tags = _build_wandb_tags(cfg)
    assert "transform:dct" in tags
    assert "stage2:nll" in tags


def test_lora_and_activation_in_name():
    cfg = _cfg(
        [
            "method.rank=2",
            "method.core_activation=softplus",
            "method.stage2_loss=nll",
            "seed=44",
        ]
    )
    assert _build_run_name(cfg) == "depth_whitened_spectral_diag_lora2_softplus_nll"
    tags = _build_wandb_tags(cfg)
    assert "rank:2" in tags
    assert "act:softplus" in tags
    assert "seed:44" in tags


def test_dct_lora_combination():
    cfg = _cfg(["method=score_dct", "method.rank=1", "method.stage2_loss=nll"])
    assert _build_run_name(cfg) == "depth_whitened_dct_lora1_nll"
    tags = _build_wandb_tags(cfg)
    assert "transform:dct" in tags
    assert "rank:1" in tags


def test_default_stage1_loss_tracks_the_config():
    """The naming default must be whatever the method config actually ships.

    When the two drift apart every staged run silently picks up a spurious
    `s1<loss>` part, so pin them together rather than to a literal.
    """
    cfg = _cfg([])
    assert _DEFAULT_STAGE1_LOSS == cfg.method.stage1_loss == "crps"


def test_default_stage1_loss_is_not_named():
    cfg = _cfg(["method.stage2_loss=nll"])
    # The default stage-1 loss carries no information in the name, but is
    # tagged so both sides of the ablation stay selectable.
    assert _build_run_name(cfg) == "depth_whitened_spectral_diag_nll"
    assert "stage1:crps" in _build_wandb_tags(cfg)


def test_deviating_stage1_loss_in_name():
    cfg = _cfg(
        ["method.stage1_loss=marginal_kernel_score", "method.stage2_loss=nll"]
    )
    # Without the s1 part this run would be named exactly like the one above.
    assert (
        _build_run_name(cfg)
        == "depth_whitened_spectral_diag_s1marginal_kernel_score_nll"
    )
    assert "stage1:marginal_kernel_score" in _build_wandb_tags(cfg)


def test_stage1_loss_combines_with_transform_and_rank():
    cfg = _cfg(
        [
            "method=score_dct",
            "method.rank=2",
            "method.stage1_loss=marginal_kernel_score",
            "method.stage2_loss=nll",
        ]
    )
    assert (
        _build_run_name(cfg) == "depth_whitened_dct_lora2_s1marginal_kernel_score_nll"
    )


def _ts_cfg(experiment: str, overrides: List[str]) -> DictConfig:
    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(
            config_name="train",
            return_hydra_config=True,
            overrides=[f"experiments={experiment}", *overrides],
        )
        HydraConfig().set_config(cfg)
    return cfg


def test_multivariate_time_series_keeps_bare_dataset_name():
    cfg = _ts_cfg("train/time_series/multivariate/etth1_score", [])
    # features=M is the datamodule default, so it adds nothing to the name.
    assert not _build_run_name(cfg).startswith("etth1_uni")
    assert _build_run_name(cfg).startswith("etth1_whitened")
    assert "features:M" in _build_wandb_tags(cfg)


def test_univariate_time_series_is_distinguishable():
    uni = _ts_cfg("train/time_series/univariate/etth1_score", [])
    multi = _ts_cfg("train/time_series/multivariate/etth1_score", [])
    # Same datamodule choice for both, so only the features facet separates
    # them -- without it the two runs would collide in wandb.
    assert _build_run_name(uni).startswith("etth1_uni_")
    assert _build_run_name(uni) != _build_run_name(multi)
    assert "features:S" in _build_wandb_tags(uni)


def test_apply_sets_name_and_merges_tags():
    cfg = _cfg(["logger.wandb.tags=[campaign-x]"])
    _apply_wandb_naming(cfg)
    assert cfg.logger.wandb.name == "depth_whitened_spectral_diag_kernel_score"
    tags = list(cfg.logger.wandb.tags)
    # User tags stay in front, generated facet tags follow, no duplicates.
    assert tags[0] == "campaign-x"
    assert "method:whitened_spectral_diag" in tags
    assert len(tags) == len(set(tags))


def test_apply_respects_explicit_name():
    cfg = _cfg(["logger.wandb.name=my-custom-run"])
    _apply_wandb_naming(cfg)
    assert cfg.logger.wandb.name == "my-custom-run"
    assert "dataset:depth" in list(cfg.logger.wandb.tags)
