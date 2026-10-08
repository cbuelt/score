"""Tests for the WeatherBench dataset and datamodule.

Split across two subjects:

* ``src.datamodules.datasets.WeatherBench`` -- the ``torch`` Dataset that reads
  the ERA5 zarr, normalises, downscales, and packs the input/target tensors.
* ``src.datamodules.dataloaders.WeatherBenchModule`` -- the LightningDataModule
  that builds the three splits in ``setup()`` and hands out DataLoaders.

All tests run against a *tiny synthetic* ``era5.zarr`` written by the
``weatherbench_data_dir`` fixture (see conftest.py), never the real store. We
always use ``preload=False`` and a ``downscaling_factor`` so nothing large is
pulled into RAM -- these tests are fast and machine-independent. A separate,
``@pytest.mark.realdata`` smoke test at the bottom is skipped unless the real
store is mounted.

The synthetic geometry (must match the conftest fixture):
  full grid 8x10, dynamic vars carry ``time``, static vars do not, and each
  variable has mean ``100*(i+1)`` / std ``10*(i+1)`` in ``era5_statistics.npy``.
  The fixture writes ``WB_DAYS_PER_YEAR=3`` daily UTC12 timesteps per year,
  mirroring the real store under ``.../utc12/``, which is already subsampled to
  one timestep per day. The dataset does no hour filtering of its own, so every
  stored timestep is a sample and consecutive samples are one day apart.
"""

import math
import os

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.datamodules.dataloaders import WeatherBenchModule
from src.utils.paths import configured_path
from src.datamodules.datasets import (
    WB_DYNAMIC,
    WB_INPUT,
    WB_STATIC,
    WB_TARGET,
    WeatherBench,
)

# Full synthetic grid (mirrors WB_LAT/WB_LON/WB_DAYS_PER_YEAR in conftest.py).
FULL_LAT = 8
FULL_LON = 10
DAYS_PER_YEAR = 3
LAST_T_STEPS = 2
LEAD_TIME = 1
DOWNSCALE = 2

# Number of UTC12 (daily) samples per split in the synthetic store. The fixture
# writes WB_DAYS_PER_YEAR=3 daily samples/year, and `get_split`'s year slices
# pick up train=3 years, val=2 years, test=2 years of the fixture's WB_YEARS.
N_TIME = {"train": 9, "val": 6, "test": 6}

# Expected channel counts derived from the variable groupings.
C_IN = len(WB_DYNAMIC) * LAST_T_STEPS + len(WB_STATIC)
C_TARGET = len(WB_TARGET)

# Downscaled spatial size: slicing ``[::factor]`` keeps ceil(n / factor) points.
H_DS = math.ceil(FULL_LAT / DOWNSCALE)
W_DS = math.ceil(FULL_LON / DOWNSCALE)


def make_dataset(data_dir, **overrides):
    """Build a ``WeatherBench`` with the tests' defaults (lazy, downscaled)."""
    params = dict(
        data_path=data_dir,
        var="train",
        normalize=True,
        preload=False,
        downscaling_factor=DOWNSCALE,
        last_t_steps=LAST_T_STEPS,
        lead_time=LEAD_TIME,
    )
    params.update(overrides)
    return WeatherBench(**params)


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("var", ["train", "val", "test"])
def test_len_is_time_minus_context(weatherbench_data_dir, var):
    """``len`` is the number of usable windows over the UTC12 (daily) samples.

    Each window needs ``last_t_steps`` input days plus ``lead_time`` days to
    reach the forecast target, so ``len == n_days - last_t_steps - lead_time + 1``
    and the last ``last_t_steps + lead_time - 1`` days cannot start a window.
    """
    ds = make_dataset(weatherbench_data_dir, var=var)
    assert len(ds) == N_TIME[var] - LAST_T_STEPS - LEAD_TIME + 1


@pytest.mark.parametrize("lead_time", [1, 2, 3])
def test_len_shrinks_with_lead_time(weatherbench_data_dir, lead_time):
    """A longer ``lead_time`` extends each window, so fewer windows fit."""
    ds = make_dataset(weatherbench_data_dir, var="train", lead_time=lead_time)
    assert len(ds) == N_TIME["train"] - LAST_T_STEPS - lead_time + 1


def test_every_stored_timestep_is_a_daily_sample(weatherbench_data_dir):
    """The store is pre-subsampled to UTC12, so the dataset filters no hours.

    This is what makes ``lead_time`` a plain day offset: consecutive timesteps
    are one day apart, so ``idx + lead_time`` lands ``lead_time`` days ahead. If
    a future store ever carried a finer cadence, ``__getitem__`` would silently
    reinterpret ``lead_time`` in 6-hour units instead -- hence the assertion on
    the cadence rather than only on the hour.
    """
    ds = make_dataset(weatherbench_data_dir, var="train")
    times = ds.dataset.time
    assert (times.dt.hour.values == 12).all()
    assert len(times) == N_TIME["train"]

    # One day apart within a fixture year (years are deliberately far apart, so
    # only the within-year steps are consecutive days).
    steps = np.diff(times.values).astype("timedelta64[D]").astype(int)
    assert (steps[: DAYS_PER_YEAR - 1] == 1).all()


def test_getitem_shapes_and_dtype(weatherbench_data_dir):
    """A sample is (input, target) with channels-first, downscaled spatial dims.

    Input packs ``len(WB_DYNAMIC)*last_t_steps`` dynamic channels followed by
    ``len(WB_STATIC)`` static channels; the target is the ``WB_TARGET`` field.
    Guards the reshape/concatenate logic in ``__getitem__``.
    """
    ds = make_dataset(weatherbench_data_dir)
    x, y = ds[0]

    assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
    assert x.shape == (C_IN, H_DS, W_DS)
    assert y.shape == (C_TARGET, H_DS, W_DS)
    assert x.dtype == torch.float32 and y.dtype == torch.float32


@pytest.mark.parametrize("lead_time", [1, 2])
def test_target_is_delta_over_lead_time(weatherbench_data_dir, lead_time):
    """Target == field(day0 + lead_time) - field(day0) for ``WB_TARGET``.

    ``day0`` is the last input day (local window index ``last_t_steps - 1``).
    We reconstruct the expected residual straight from the normalized daily
    (UTC12) series to pin the day offset the target uses.
    """
    ds = make_dataset(weatherbench_data_dir, lead_time=lead_time, normalize=True)

    # Normalized UTC12 series for the single target variable. ``ds.dataset`` is
    # already UTC12-only and downscaled, so no further slicing is needed.
    tgt_name = WB_TARGET[0]
    var_idx = WB_INPUT.index(tgt_name)
    mean = 100.0 * (var_idx + 1)
    std = 10.0 * (var_idx + 1)
    series = ds.dataset[tgt_name].values  # (time, lat, lon)
    series = (series - mean) / std

    idx = 0
    day0 = idx + LAST_T_STEPS - 1
    expected = series[day0 + lead_time] - series[day0]

    _, y = ds[idx]
    assert torch.allclose(y[0], torch.from_numpy(expected).float(), atol=1e-4)


def test_preload_and_lazy_yield_identical_samples(weatherbench_data_dir):
    """``preload=True`` (all in RAM) and ``preload=False`` (lazy) must agree.

    Preload is a memory/speed trade-off, not a behavioural one.
    """
    eager = make_dataset(weatherbench_data_dir, preload=True)
    lazy = make_dataset(weatherbench_data_dir, preload=False)

    assert len(eager) == len(lazy)
    xe, ye = eager[0]
    xl, yl = lazy[0]
    assert torch.equal(xe, xl)
    assert torch.equal(ye, yl)


def test_normalize_matches_raw_standardisation(weatherbench_data_dir):
    """``normalize=True`` yields ``(raw - mean) / std`` per variable.

    We compare the first dynamic input channel between a normalized and an
    unnormalized dataset. That channel is variable ``WB_INPUT[0]`` at ``t=0``;
    its stats are ``mean=100, std=10`` in the synthetic fixture.
    """
    raw = make_dataset(weatherbench_data_dir, normalize=False)
    norm = make_dataset(weatherbench_data_dir, normalize=True)

    x_raw, _ = raw[0]
    x_norm, _ = norm[0]

    var_idx = WB_INPUT.index(WB_DYNAMIC[0])  # first dynamic channel's source var
    mean = 100.0 * (var_idx + 1)
    std = 10.0 * (var_idx + 1)

    expected = (x_raw[0] - mean) / std
    assert torch.allclose(x_norm[0], expected, atol=1e-4)


def test_downscaling_changes_grid_dimensions(weatherbench_data_dir):
    """``downscaling_factor`` shrinks the grid to ``ceil(n / factor)`` per axis."""
    full = make_dataset(weatherbench_data_dir, downscaling_factor=1)
    down = make_dataset(weatherbench_data_dir, downscaling_factor=DOWNSCALE)

    assert full.get_dimensions() == (FULL_LAT, FULL_LON)
    assert down.get_dimensions() == (H_DS, W_DS)


def test_output_channels_is_number_of_targets(weatherbench_data_dir):
    """``get_output_channels`` reports the number of predicted variables."""
    ds = make_dataset(weatherbench_data_dir)
    assert ds.get_output_channels() == len(WB_TARGET)


def test_input_channels_matches_packed_sample(weatherbench_data_dir):
    """``get_input_channels`` equals what ``__getitem__`` actually packs.

    It must count the dynamic vars over ``last_t_steps`` plus the static vars,
    i.e. it depends on ``last_t_steps`` -- not just the raw variable count.
    """
    ds = make_dataset(weatherbench_data_dir)
    x, _ = ds[0]
    assert WeatherBench.get_input_channels(LAST_T_STEPS) == C_IN
    assert WeatherBench.get_input_channels(LAST_T_STEPS) == x.shape[0]


@pytest.mark.parametrize("last_t_steps", [1, 2, 4])
def test_input_channels_scales_with_context(last_t_steps):
    """More context frames add ``len(WB_DYNAMIC)`` channels each (pure function)."""
    expected = len(WB_DYNAMIC) * last_t_steps + len(WB_STATIC)
    assert WeatherBench.get_input_channels(last_t_steps) == expected


@pytest.mark.parametrize(
    "var, expected",
    [
        ("train", (slice("1990", "2014"))),
        ("val", (slice("2015", "2019"))),
        ("test", (slice("2020", "2022"))),
    ],
)
def test_get_split_year_ranges(var, expected):
    """Each split maps to its documented year range (pure function, no data)."""
    assert WeatherBench.get_split(var) == expected


def test_get_split_rejects_unknown_var():
    """An unrecognised split name must fail loudly rather than silently."""
    with pytest.raises(AssertionError):
        WeatherBench.get_split("nonsense")


# ---------------------------------------------------------------------------
# Datamodule tests
# ---------------------------------------------------------------------------

BATCH_SIZE = 2
EVAL_BATCH_SIZE = 3


def make_module(data_dir, **overrides):
    """Build a ``WeatherBenchModule`` with the tests' (light) defaults."""
    params = dict(
        data_path=data_dir,
        batch_size=BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        num_workers=0,
        normalize=True,
        preload=False,
        downscaling_factor=DOWNSCALE,
        last_t_steps=LAST_T_STEPS,
        lead_time=LEAD_TIME,
    )
    params.update(overrides)
    return WeatherBenchModule(**params)


def test_requesting_loader_before_setup_raises(weatherbench_data_dir):
    """Datasets are built in ``setup()``; asking earlier must fail loudly."""
    dm = make_module(weatherbench_data_dir)
    with pytest.raises(RuntimeError):
        dm.train_dataloader()


def test_setup_builds_all_datasets(weatherbench_data_dir):
    """A full ``setup()`` (stage=None) constructs train, val, and test splits."""
    dm = make_module(weatherbench_data_dir)
    assert dm.train_dataset is None and dm.val_dataset is None and dm.test_dataset is None

    dm.setup()

    assert dm.train_dataset is not None
    assert dm.val_dataset is not None
    assert dm.test_dataset is not None


def test_dataloaders_yield_correctly_shaped_batches(weatherbench_data_dir):
    """Each loader is a DataLoader whose batches have the expected shapes."""
    dm = make_module(weatherbench_data_dir)
    dm.setup()

    for loader in (dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()):
        assert isinstance(loader, DataLoader)
        x, y = next(iter(loader))
        assert x.shape[1:] == (C_IN, H_DS, W_DS)
        assert y.shape[1:] == (C_TARGET, H_DS, W_DS)
        assert x.dtype == torch.float32


def test_loader_batch_sizes(weatherbench_data_dir):
    """Train uses ``batch_size``; val/test use ``eval_batch_size``."""
    dm = make_module(weatherbench_data_dir)
    dm.setup()

    assert dm.train_dataloader().batch_size == BATCH_SIZE
    assert dm.val_dataloader().batch_size == EVAL_BATCH_SIZE
    assert dm.test_dataloader().batch_size == EVAL_BATCH_SIZE


def test_module_threads_lead_time_into_datasets(weatherbench_data_dir):
    """The module passes ``lead_time`` down, changing each split's length."""
    dm = make_module(weatherbench_data_dir, lead_time=2)
    dm.setup()

    assert dm.train_dataset.lead_time == 2
    assert len(dm.train_dataset) == N_TIME["train"] - LAST_T_STEPS - 2 + 1


def test_get_channels_resolves_before_setup(weatherbench_data_dir):
    """``get_channels`` is static: it works without building any dataset."""
    dm = make_module(weatherbench_data_dir)
    assert dm.train_dataset is None  # setup() not called
    assert dm.get_channels() == len(WB_TARGET)


def test_get_input_channels_resolves_before_setup(weatherbench_data_dir):
    """``get_input_channels`` is static and honours the module's ``last_t_steps``."""
    dm = make_module(weatherbench_data_dir)
    assert dm.train_dataset is None  # setup() not called
    assert dm.get_input_channels() == C_IN
    # It must track last_t_steps, delegating to the dataset helper.
    dm3 = make_module(weatherbench_data_dir, last_t_steps=3)
    assert dm3.get_input_channels() == len(WB_DYNAMIC) * 3 + len(WB_STATIC)


def test_get_total_dimension_resolves_before_setup(weatherbench_data_dir):
    """``get_total_dimension`` == C_target * H' * W', via an on-demand probe."""
    dm = make_module(weatherbench_data_dir)
    assert dm.train_dataset is None  # resolved without setup()
    assert dm.get_total_dimension() == C_TARGET * H_DS * W_DS


# ---------------------------------------------------------------------------
# Optional real-data smoke test
# ---------------------------------------------------------------------------

# The configured store (paths.era5_dir), i.e. the one configs/datamodule/era5.yaml
# uses: the daily UTC12 store, which the UTC12 assertion below relies on.
REAL_DATA_PATH = os.path.join(configured_path("era5_dir"), "")
_HAS_REAL_DATA = os.path.exists(os.path.join(REAL_DATA_PATH, "era5.zarr"))


@pytest.mark.realdata
@pytest.mark.skipif(not _HAS_REAL_DATA, reason="real ERA5 store not available")
def test_real_era5_loads_and_has_expected_shapes():
    """The real store opens (lazily, heavily downscaled) and yields sane samples.

    ``preload=False`` avoids pulling the multi-year arrays into RAM, and a large
    ``downscaling_factor`` keeps the per-sample read cheap.
    """
    ds = WeatherBench(
        data_path=REAL_DATA_PATH,
        var="test",
        normalize=True,
        preload=False,
        downscaling_factor=8,
        last_t_steps=LAST_T_STEPS,
        lead_time=LEAD_TIME,
    )

    assert len(ds) > 0
    # Only the daily UTC12 timestep is kept.
    assert (ds.dataset.time.dt.hour.values == 12).all()

    x, y = ds[0]
    assert x.shape[0] == C_IN
    assert y.shape[0] == C_TARGET
    assert x.shape[1:] == y.shape[1:]  # input and target share the spatial grid
    assert x.dtype == torch.float32
    assert np.isfinite(x.numpy()).all()
