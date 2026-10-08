"""Tests for the PatchTST-style time-series dataset and datamodule.

Two subjects:

* ``src.datamodules.datasets.TimeSeriesDataset`` -- the ``torch`` Dataset that
  reads a forecasting CSV, standardises (train-range stats only), and packs the
  channels-first ``(x, y)`` look-back/horizon windows.
* ``src.datamodules.dataloaders.TimeSeriesDataModule`` -- the LightningDataModule
  that builds the three splits in ``setup()`` and hands out DataLoaders with the
  per-flag shuffle/drop_last behaviour PatchTST's ``data_provider`` uses.

Most tests run against a tiny synthetic CSV written by the ``timeseries_csv``
fixture (see conftest.py), sized for the ``custom`` (70/10/20 ratio) split. The
fixed-month ETT splits need ~14k rows, so they are checked via the pure
``TimeSeriesDataset._borders`` function directly. A ``@pytest.mark.realdata``
smoke test at the bottom is skipped unless the real CSVs are mounted.
"""

import os

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.datamodules.dataloaders import TimeSeriesDataModule
from src.datamodules.datasets import TimeSeriesDataset
from src.utils.paths import configured_path

SEQ_LEN = 16
PRED_LEN = 8
DATA_SPLIT = "custom"


def make_dataset(csv_path, **overrides):
    """Build a ``TimeSeriesDataset`` with the tests' (small-window) defaults."""
    params = dict(
        data_path=csv_path,
        split="train",
        data_split=DATA_SPLIT,
        features="M",
        target="OT",
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        scale=True,
    )
    params.update(overrides)
    return TimeSeriesDataset(**params)


def make_module(csv_path, **overrides):
    """Build a ``TimeSeriesDataModule`` with the tests' (small) defaults."""
    params = dict(
        data_path=csv_path,
        data_split=DATA_SPLIT,
        features="M",
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        batch_size=BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        num_workers=0,
    )
    params.update(overrides)
    return TimeSeriesDataModule(**params)


def raw_frame(csv_path):
    """The CSV as a DataFrame, reordered to ['date'] + others + ['OT']."""
    df = pd.read_csv(csv_path)
    cols = [c for c in df.columns if c not in ("date", "OT")]
    return df[["date"] + cols + ["OT"]]


# ---------------------------------------------------------------------------
# Split-boundary logic (pure function, no data)
# ---------------------------------------------------------------------------

def test_borders_ett_hour():
    """ETT hourly: 12 months train, then +4 and +4 months (``12*30*24`` steps)."""
    s = SEQ_LEN
    b1, b2 = TimeSeriesDataset._borders("ett_hour", n=99999, seq_len=s)
    assert b2 == [8640, 11520, 14400]
    assert b1 == [0, 8640 - s, 11520 - s]


def test_borders_ett_minute():
    """ETT 15-minute: identical to hourly but every term x4."""
    s = SEQ_LEN
    b1, b2 = TimeSeriesDataset._borders("ett_minute", n=99999, seq_len=s)
    assert b2 == [34560, 46080, 57600]
    assert b1 == [0, 34560 - s, 46080 - s]


def test_borders_custom_ratio():
    """Custom: 70% train / 10% val / 20% test, each starting ``seq_len`` early."""
    n = 200
    s = SEQ_LEN
    b1, b2 = TimeSeriesDataset._borders("custom", n=n, seq_len=s)
    num_train = int(n * 0.7)  # 140
    num_test = int(n * 0.2)  # 40
    num_vali = n - num_train - num_test  # 20
    assert b2 == [num_train, num_train + num_vali, n]
    assert b1 == [0, num_train - s, n - num_test - s]


def test_unknown_data_split_rejected(timeseries_csv):
    """An unrecognised split strategy must fail loudly."""
    with pytest.raises(AssertionError):
        make_dataset(timeseries_csv, data_split="nonsense")


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_len_is_window_count(timeseries_csv, split):
    """``len == (b2 - b1) - seq_len - pred_len + 1`` for the split's window."""
    n = len(raw_frame(timeseries_csv))
    b1s, b2s = TimeSeriesDataset._borders(DATA_SPLIT, n, SEQ_LEN)
    i = {"train": 0, "val": 1, "test": 2}[split]
    ds = make_dataset(timeseries_csv, split=split)
    assert len(ds) == (b2s[i] - b1s[i]) - SEQ_LEN - PRED_LEN + 1


@pytest.mark.parametrize(
    "features, c_in, c_out",
    [("M", 5, 5), ("MS", 5, 1), ("S", 1, 1)],
)
def test_getitem_shapes_and_dtype(timeseries_csv, features, c_in, c_out):
    """Channels-first ``x=(C_in, seq_len)``, ``y=(C_out, pred_len)``, float32.

    The synthetic CSV has 4 feature columns + ``OT`` = 5 value columns, so
    ``M`` -> in/out 5, ``MS`` -> in 5 / out 1, ``S`` -> in/out 1.
    """
    ds = make_dataset(timeseries_csv, features=features)
    x, y = ds[0]
    assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
    assert x.shape == (c_in, SEQ_LEN)
    assert y.shape == (c_out, PRED_LEN)
    assert x.dtype == torch.float32 and y.dtype == torch.float32


def test_ms_target_is_ot_column(timeseries_csv):
    """``MS`` output is exactly the ``OT`` (last) channel of the ``M`` output."""
    ds_m = make_dataset(timeseries_csv, features="M")
    ds_ms = make_dataset(timeseries_csv, features="MS")
    _, y_m = ds_m[3]
    _, y_ms = ds_ms[3]
    assert y_ms.shape[0] == 1
    assert torch.equal(y_ms[0], y_m[-1])  # OT is the last channel


def test_scaling_fits_on_train_only(timeseries_csv):
    """Val/test are standardised with the *train* mean/std, not their own.

    Reconstructs the expected standardisation from the raw CSV: fit on the train
    slice ``[0:num_train]`` and apply everywhere.
    """
    df = raw_frame(timeseries_csv)
    values = df[df.columns[1:]].values.astype(np.float64)  # all value columns (M)
    n = len(df)
    b1s, b2s = TimeSeriesDataset._borders(DATA_SPLIT, n, SEQ_LEN)

    train_slice = values[b1s[0] : b2s[0]]
    mean = train_slice.mean(axis=0)
    std = train_slice.std(axis=0)
    std[std == 0.0] = 1.0
    scaled = ((values - mean) / std).astype(np.float32)

    for i, split in enumerate(("train", "val", "test")):
        ds = make_dataset(timeseries_csv, split=split)
        expected = scaled[b1s[i] : b2s[i]]
        assert np.allclose(ds.data_x, expected, atol=1e-4)


def test_scale_false_returns_raw(timeseries_csv):
    """``scale=False`` leaves the values untouched (just cast to float32)."""
    df = raw_frame(timeseries_csv)
    values = df[df.columns[1:]].values.astype(np.float32)
    n = len(df)
    b1s, b2s = TimeSeriesDataset._borders(DATA_SPLIT, n, SEQ_LEN)

    ds = make_dataset(timeseries_csv, split="test", scale=False)
    assert np.allclose(ds.data_x, values[b1s[2] : b2s[2]], atol=1e-5)


def test_features_s_uses_target_only(timeseries_csv):
    """``S`` input/output is the (scaled) ``OT`` column only."""
    df = raw_frame(timeseries_csv)
    ot = df[["OT"]].values.astype(np.float64)
    n = len(df)
    b1s, b2s = TimeSeriesDataset._borders(DATA_SPLIT, n, SEQ_LEN)
    train_ot = ot[b1s[0] : b2s[0]]
    scaled_ot = ((ot - train_ot.mean()) / train_ot.std()).astype(np.float32)

    ds = make_dataset(timeseries_csv, split="train", features="S")
    assert ds.data_x.shape[1] == 1
    x, y = ds[0]
    assert torch.allclose(x[0], torch.from_numpy(scaled_ot[0:SEQ_LEN, 0]), atol=1e-4)
    assert torch.allclose(
        y[0], torch.from_numpy(scaled_ot[SEQ_LEN : SEQ_LEN + PRED_LEN, 0]), atol=1e-4
    )


def test_return_marks_yields_four_tuple(timeseries_csv):
    """``return_marks=True`` adds calendar marks; hourly ``timeenc=0`` -> 4 feats."""
    ds = make_dataset(timeseries_csv, features="M", return_marks=True, freq="h")
    out = ds[0]
    assert len(out) == 4
    x, y, x_mark, y_mark = out
    assert x.shape == (5, SEQ_LEN)
    assert y.shape == (5, PRED_LEN)  # label_len defaults to 0
    assert x_mark.shape == (4, SEQ_LEN)  # month, day, weekday, hour
    assert y_mark.shape == (4, PRED_LEN)
    assert x_mark.dtype == torch.float32


def test_label_len_extends_marks_target_window(timeseries_csv):
    """With ``return_marks`` the target window includes the ``label_len`` warm-up."""
    label_len = 4
    ds = make_dataset(
        timeseries_csv, features="M", return_marks=True, label_len=label_len
    )
    _, y, _, y_mark = ds[0]
    assert y.shape == (5, label_len + PRED_LEN)
    assert y_mark.shape == (4, label_len + PRED_LEN)


def test_inverse_transform_roundtrips(timeseries_csv):
    """``inverse_transform`` undoes the fitted standardisation."""
    ds = make_dataset(timeseries_csv, split="train", features="M")
    restored = ds.inverse_transform(ds.data_x)
    df = raw_frame(timeseries_csv)
    values = df[df.columns[1:]].values.astype(np.float32)
    n = len(df)
    b1s, b2s = TimeSeriesDataset._borders(DATA_SPLIT, n, SEQ_LEN)
    assert np.allclose(restored, values[b1s[0] : b2s[0]], atol=1e-3)


def test_n_columns_reads_header_only(timeseries_csv):
    """``n_columns`` counts value columns (excludes ``date``) from the header."""
    assert TimeSeriesDataset.n_columns(timeseries_csv) == 5


# ---------------------------------------------------------------------------
# Datamodule tests
# ---------------------------------------------------------------------------

BATCH_SIZE = 4
EVAL_BATCH_SIZE = 3


def test_requesting_loader_before_setup_raises(timeseries_csv):
    """Datasets are built in ``setup()``; asking earlier must fail loudly."""
    dm = make_module(timeseries_csv)
    with pytest.raises(RuntimeError):
        dm.train_dataloader()


def test_setup_builds_all_datasets(timeseries_csv):
    """A full ``setup()`` (stage=None) constructs train, val, and test splits."""
    dm = make_module(timeseries_csv)
    assert dm.train_dataset is None and dm.val_dataset is None and dm.test_dataset is None
    dm.setup()
    assert dm.train_dataset is not None
    assert dm.val_dataset is not None
    assert dm.test_dataset is not None


def test_dataloaders_yield_correctly_shaped_batches(timeseries_csv):
    """Each loader is a DataLoader whose batches are channels-first windows."""
    dm = make_module(timeseries_csv, features="M")
    dm.setup()
    for loader in (dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()):
        assert isinstance(loader, DataLoader)
        x, y = next(iter(loader))
        assert x.shape[1:] == (5, SEQ_LEN)
        assert y.shape[1:] == (5, PRED_LEN)
        assert x.dtype == torch.float32


def test_loader_batch_sizes(timeseries_csv):
    """Train uses ``batch_size``; val/test use ``eval_batch_size``."""
    dm = make_module(timeseries_csv)
    dm.setup()
    assert dm.train_dataloader().batch_size == BATCH_SIZE
    assert dm.val_dataloader().batch_size == EVAL_BATCH_SIZE
    assert dm.test_dataloader().batch_size == EVAL_BATCH_SIZE


def test_loader_shuffle_and_drop_last_flags(timeseries_csv):
    """Train shuffles and drops the partial batch; test does neither."""
    dm = make_module(timeseries_csv)
    dm.setup()
    # DataLoader stores drop_last directly; shuffling is inferred from sampler.
    from torch.utils.data import RandomSampler, SequentialSampler

    train = dm.train_dataloader()
    test = dm.test_dataloader()
    assert train.drop_last is True
    assert test.drop_last is False
    assert isinstance(train.sampler, RandomSampler)  # shuffle=True
    assert isinstance(test.sampler, SequentialSampler)  # shuffle=False


def test_drop_last_can_be_disabled(timeseries_csv):
    """``drop_last=False`` keeps the trailing partial batch on the train loader."""
    dm = make_module(timeseries_csv, drop_last=False)
    dm.setup()
    assert dm.train_dataloader().drop_last is False
    assert dm.test_dataloader().drop_last is False


def test_evaluation_loaders_ignore_drop_last(timeseries_csv):
    """``drop_last`` is a *training* flag: val and test always keep the tail.

    PatchTST's ``data_provider`` drops it on every split, which silently
    truncates the test set (97 of ETTh1's 2 785 windows at the default batch
    size) and makes a reported test score an average over an arbitrary prefix.
    """
    dm = make_module(timeseries_csv, drop_last=True)
    dm.setup()
    assert dm.val_dataloader().drop_last is False
    assert dm.test_dataloader().drop_last is False
    # Every window is emitted, not just the whole batches.
    n_seen = sum(x.shape[0] for x, _ in dm.test_dataloader())
    assert n_seen == len(dm.test_dataset)


@pytest.mark.parametrize(
    "features, c_in, c_out",
    [("M", 5, 5), ("MS", 5, 1), ("S", 1, 1)],
)
def test_sizing_contract_resolves_before_setup(timeseries_csv, features, c_in, c_out):
    """``get_input_channels``/``get_channels``/``get_total_dimension`` are static."""
    dm = make_module(timeseries_csv, features=features)
    assert dm.train_dataset is None  # setup() not called
    assert dm.get_input_channels() == c_in
    assert dm.get_channels() == c_out
    assert dm.get_total_dimension() == c_out * PRED_LEN


# ---------------------------------------------------------------------------
# Optional real-data smoke tests
# ---------------------------------------------------------------------------

REAL_BASE = str(configured_path("time_series_dir"))
REAL_ETTH1 = os.path.join(REAL_BASE, "ETT-small", "ETTh1.csv")
REAL_WEATHER = os.path.join(REAL_BASE, "weather", "weather.csv")


@pytest.mark.realdata
@pytest.mark.skipif(not os.path.exists(REAL_ETTH1), reason="real ETT data not available")
def test_real_etth1_ett_hour_split():
    """Real ETTh1 loads with the fixed-month ``ett_hour`` split and sane shapes."""
    ds = TimeSeriesDataset(
        data_path=REAL_ETTH1,
        split="test",
        data_split="ett_hour",
        features="M",
        seq_len=96,
        pred_len=96,
    )
    assert len(ds) > 0
    assert TimeSeriesDataset.n_columns(REAL_ETTH1) == 7  # 6 features + OT
    x, y = ds[0]
    assert x.shape == (7, 96)
    assert y.shape == (7, 96)
    assert x.dtype == torch.float32
    assert np.isfinite(x.numpy()).all()


@pytest.mark.realdata
@pytest.mark.skipif(not os.path.exists(REAL_WEATHER), reason="real weather data not available")
def test_real_weather_custom_split_is_wide():
    """Real weather loads with the ``custom`` ratio split (genuinely multivariate)."""
    n_vars = TimeSeriesDataset.n_columns(REAL_WEATHER)
    assert n_vars == 21
    ds = TimeSeriesDataset(
        data_path=REAL_WEATHER,
        split="val",
        data_split="custom",
        features="M",
        seq_len=96,
        pred_len=96,
    )
    assert len(ds) > 0
    x, y = ds[0]
    assert x.shape == (n_vars, 96)
    assert y.shape == (n_vars, 96)
