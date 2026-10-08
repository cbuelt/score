"""The EUPPBench datamodule contract and its NaN-preserving batches."""

import numpy as np
import pytest
import torch

from src.datamodules import EUPPBenchModule
from src.datamodules.datasets import EUPPBench

from tests.conftest import (
    EUPP_MEMBERS,
    EUPP_N_FEATURES,
    EUPP_N_STATIONS,
    EUPP_SAMPLES,
)


def _module(data_dir, **kwargs):
    return EUPPBenchModule(data_path=data_dir, batch_size=4, eval_batch_size=4,
                           num_workers=0, pin_memory=False, **kwargs)


def test_sizes_resolve_before_setup(euppbench_data_dir):
    """`train.py` sizes the method and backbone before `setup()` ever runs.

    If these needed the arrays, instantiation would read ~1 GB from disk in
    every DDP process just to learn three integers.
    """
    module = _module(euppbench_data_dir)
    assert module.get_input_channels() == EUPP_N_FEATURES
    assert module.get_channels() == 1
    assert module.get_total_dimension() == EUPP_N_STATIONS
    assert module.train_dataset is None  # nothing was built


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_batch_layout(euppbench_data_dir, split):
    """Batches are the repo-wide `(x, y)` tuple with the station axis last."""
    dataset = EUPPBench(euppbench_data_dir, split)
    x, y = dataset[0]

    assert len(dataset) == EUPP_SAMPLES[split]
    assert x.shape == (EUPP_N_FEATURES, EUPP_MEMBERS[split], EUPP_N_STATIONS)
    assert y.shape == (1, EUPP_N_STATIONS)
    assert x.dtype == torch.float32 and y.dtype == torch.float32
    assert torch.isfinite(x).all(), "predictors must never be missing"


def test_missing_targets_survive_the_loader(euppbench_data_dir):
    """NaNs must reach the loss, which masks them -- not be filled in here.

    Imputing a target would fabricate exactly the spatial dependence the
    multivariate scores measure, so the dataset deliberately passes NaN through.
    """
    module = _module(euppbench_data_dir)
    module.setup("fit")
    _, y = next(iter(module.train_dataloader()))
    assert torch.isnan(y).any()
    assert torch.isnan(y[:, 0, 0]).all(), "the never-observed station stays NaN"


def test_ensemble_size_differs_between_fit_and_test(euppbench_data_dir):
    """Train/val come from reforecasts, test from forecasts, with more members."""
    module = _module(euppbench_data_dir)
    module.setup("fit")
    module.setup("test")
    x_train, _ = next(iter(module.train_dataloader()))
    x_test, _ = next(iter(module.test_dataloader()))
    assert x_train.shape[2] == EUPP_MEMBERS["train"]
    assert x_test.shape[2] == EUPP_MEMBERS["test"]
    assert x_test.shape[2] != x_train.shape[2]


def test_setup_is_idempotent(euppbench_data_dir):
    """Lightning may call `setup` more than once; it must not rebuild."""
    module = _module(euppbench_data_dir)
    module.setup("fit")
    first = module.train_dataset
    module.setup("fit")
    assert module.train_dataset is first


def test_graph_geometry_is_shared_with_the_backbone(euppbench_data_dir):
    """The distance matrix the model and the variogram weights read is one object."""
    module = _module(euppbench_data_dir)
    dist = module.get_distances()
    assert dist.shape == (EUPP_N_STATIONS, EUPP_N_STATIONS)
    assert np.allclose(dist, dist.T)
    assert np.allclose(np.diag(dist), 0.0)

    _, coords = module.get_graph()
    assert coords.shape == (EUPP_N_STATIONS, 2)


def test_missing_build_is_reported_actionably(tmp_path):
    """The dataset is built offline, so a missing build must say how to make it."""
    with pytest.raises(FileNotFoundError, match="build_euppbench"):
        EUPPBench(str(tmp_path), "train")
