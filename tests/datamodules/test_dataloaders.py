"""Tests for ``src.datamodules.dataloaders.DepthDataModule``.

A LightningDataModule has a defined lifecycle: you construct it, call
``setup()`` to build the datasets, then ask it for ``train_dataloader()`` etc.
These tests cover that contract and the split / batching behaviour.

Note the use of ``pytest.raises`` below: it asserts that a block of code raises
a particular exception. The test passes only if the error actually occurs --
this is how you test failure paths, not just happy paths.
"""

import pytest
import torch
from torch.utils.data import DataLoader

from src.datamodules.dataloaders import DepthDataModule

N_SAMPLES = 6
HEIGHT = 8
WIDTH = 10


def make_kwargs(depth_data_dir, **overrides):
    """Build the ``DepthDataModule`` kwargs for the synthetic depth dataset.

    Defaults mirror the values the tests rely on (``preload=False`` so we don't
    pull everything into memory); ``overrides`` lets each test tweak only the
    fields it cares about.
    """
    params = dict(data_path=depth_data_dir, dataset="nyu", preload=False)
    params.update(overrides)
    return params


def test_requesting_loader_before_setup_raises(depth_data_dir):
    """Datasets are built in ``setup()``; asking earlier must fail loudly."""
    dm = DepthDataModule(**make_kwargs(depth_data_dir))
    with pytest.raises(RuntimeError):
        dm.train_dataloader()


def test_setup_splits_train_and_val(depth_data_dir):
    """``val_split`` carves a validation set out of the training data.

    With 6 samples and a 0.5 split we expect a 3/3 partition, and the two
    splits must together account for every original sample.
    """
    dm = DepthDataModule(**make_kwargs(depth_data_dir, val_split=0.5, batch_size=2))
    dm.setup("fit")

    assert len(dm.train_dataset) + len(dm.val_dataset) == N_SAMPLES
    assert len(dm.val_dataset) == 3


def test_dataloaders_yield_correctly_shaped_batches(depth_data_dir):
    """The train loader should return (image, depth) batches of the right shape."""
    dm = DepthDataModule(**make_kwargs(depth_data_dir, val_split=0.5, batch_size=2))
    dm.setup("fit")

    loader = dm.train_dataloader()
    assert isinstance(loader, DataLoader)

    x, y = next(iter(loader))
    assert x.shape == (2, 3, HEIGHT, WIDTH)  # (batch, C, H, W)
    assert y.shape == (2, 1, HEIGHT, WIDTH)
    assert x.dtype == torch.float32


def test_test_loader_uses_configured_batch_size(depth_data_dir):
    """Without ``eval_batch_size``, the test loader uses ``batch_size``."""
    dm = DepthDataModule(**make_kwargs(depth_data_dir, val_split=0.5, batch_size=2))
    dm.setup("fit")

    assert dm.test_dataloader(test=True).batch_size == 2
    assert dm.test_dataloader(test=False).batch_size == 2


def test_eval_batch_size_overrides_test_loader(depth_data_dir):
    """``eval_batch_size`` sets the test loader's batch size; train keeps its own."""
    dm = DepthDataModule(
        **make_kwargs(
            depth_data_dir, val_split=0.5, batch_size=2, eval_batch_size=4
        )
    )
    dm.setup("fit")

    # Eval/test path picks up the override; the non-test path keeps batch_size.
    assert dm.test_dataloader(test=True).batch_size == 4
    assert dm.test_dataloader(test=False).batch_size == 2
    assert dm.train_dataloader().batch_size == 2


def test_split_is_reproducible_with_seed(depth_data_dir):
    """Same ``split_seed`` -> same train/val partition, so runs are reproducible."""
    sizes = []
    for _ in range(2):
        dm = DepthDataModule(
            **make_kwargs(depth_data_dir, val_split=0.5, split_seed=42)
        )
        dm.setup("fit")
        # Compare the actual indices chosen for the validation subset.
        sizes.append(tuple(dm.val_dataset.indices))
    assert sizes[0] == sizes[1]


def test_split_ignores_global_seed(depth_data_dir):
    """The partition must come from ``split_seed`` only, not the global RNG.

    This is what lets a multi-seed sweep vary model init / shuffling while every
    run trains on exactly the same data: seeding torch differently (as
    ``L.seed_everything(cfg.seed)`` does) must not move the split.
    """
    indices = []
    for global_seed in (0, 1234):
        torch.manual_seed(global_seed)
        dm = DepthDataModule(
            **make_kwargs(depth_data_dir, val_split=0.5, split_seed=42)
        )
        dm.setup("fit")
        indices.append(tuple(dm.val_dataset.indices))
    assert indices[0] == indices[1]


def test_different_split_seeds_give_different_splits(depth_data_dir):
    """``split_seed`` is actually wired to the partition (guards the test above).

    Without this, a split that ignored *every* seed would still pass
    ``test_split_ignores_global_seed``.
    """
    indices = []
    for split_seed in (0, 1234):
        dm = DepthDataModule(
            **make_kwargs(depth_data_dir, val_split=0.5, split_seed=split_seed)
        )
        dm.setup("fit")
        indices.append(tuple(dm.val_dataset.indices))
    assert indices[0] != indices[1]


def test_repeated_setup_keeps_the_same_split(depth_data_dir):
    """``setup()`` runs more than once (fit, then validate/test) -- it must not reshuffle."""
    dm = DepthDataModule(**make_kwargs(depth_data_dir, val_split=0.5, split_seed=42))
    dm.setup("fit")
    first = tuple(dm.val_dataset.indices)
    dm.setup("validate")
    assert tuple(dm.val_dataset.indices) == first


def test_channel_accessors_resolve_before_setup(depth_data_dir):
    """``get_input_channels`` (RGB=3) and ``get_channels`` (depth=1) are static.

    They must size the backbone before any dataset is built, matching the loader
    batch shapes (input ``(B, 3, H, W)``, target ``(B, 1, H, W)``).
    """
    dm = DepthDataModule(**make_kwargs(depth_data_dir))
    assert dm.train_dataset is None  # setup() not called
    assert dm.get_input_channels() == 3
    assert dm.get_channels() == 1
