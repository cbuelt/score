"""Shared pytest configuration and fixtures.

pytest automatically discovers a file named ``conftest.py`` and makes every
fixture defined here available to all test files in this directory (and
sub-directories) *without* needing to import anything. It is the standard place
to put setup code shared across many tests.

Two things are set up here:

1. ``sys.path`` is extended so that ``import src...`` works when pytest is run
   from the project root, regardless of how the package is installed.
2. A ``depth_data_dir`` fixture that writes a *tiny* synthetic dataset in the
   exact on-disk format the real depth ``.h5`` files use, into a temporary
   directory. This lets the tests exercise the real loading code without
   depending on the large, machine-specific data files.
"""

import os
import sys

import numpy as np
import pytest

# Make the project root importable (so ``import src.datamodules`` works) no
# matter what the current working directory is when pytest runs.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# Geometry of the real depth files: image is (N, H, W, 3), depth is (N, H, W, 1),
# both stored as uint8. We use a handful of small frames here to keep tests fast.
N_SAMPLES = 6
HEIGHT = 8
WIDTH = 10


def _write_depth_h5(path: str, n_samples: int = N_SAMPLES) -> None:
    """Create a synthetic depth ``.h5`` file matching the real layout.

    The real files contain two HDF5 datasets, ``image`` (..., 3 channels) and
    ``depth`` (..., 1 channel), both ``uint8`` with shape ``(N, H, W, C)`` and
    *unnamed* ("phony") dimensions -- which is why ``DepthDataset`` renames the
    ``phony_dim_*`` axes. We reproduce that exactly using h5py.
    """
    import h5py

    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(n_samples, HEIGHT, WIDTH, 3), dtype=np.uint8)
    depth = rng.integers(0, 256, size=(n_samples, HEIGHT, WIDTH, 1), dtype=np.uint8)

    with h5py.File(path, "w") as f:
        f.create_dataset("image", data=image)
        f.create_dataset("depth", data=depth)


@pytest.fixture
def depth_data_dir(tmp_path):
    """Return a temp directory containing synthetic ``nyu_{train,test}.h5`` files.

    ``tmp_path`` is a built-in pytest fixture giving a unique temporary
    directory (a ``pathlib.Path``) that is automatically cleaned up after the
    test. Returning a trailing-slash string because ``DepthDataset`` builds its
    path via simple string concatenation (``data_path + filename``).
    """
    _write_depth_h5(os.path.join(tmp_path, "nyu_train.h5"))
    _write_depth_h5(os.path.join(tmp_path, "nyu_test.h5"))
    return str(tmp_path) + os.sep


# ---------------------------------------------------------------------------
# DINOv2 hub-source stub (DepthAnything backbone)
# ---------------------------------------------------------------------------
# ``DepthAnything`` loads its encoder via ``torch.hub.load(<dir>, "dinov2_vits14",
# source="local")`` from a machine-specific DINOv2 checkout. This stub writes a
# minimal ``hubconf.py`` providing that entrypoint, so tests can instantiate the
# backbone (and run its forward at small sizes) without the real checkout.

_DINOV2_STUB_HUBCONF = '''\
"""Minimal DINOv2 hub stub for tests (see tests/conftest.py)."""
import torch
from torch import nn

EMBED_DIM = 32
PATCH_SIZE = 14


class _Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # DepthAnything reads the embedding dim from ``blocks[0].attn.qkv``.
        self.qkv = nn.Linear(dim, dim * 3)


class _Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = _Attention(dim)


class _StubDinoV2(nn.Module):
    patch_size = PATCH_SIZE

    def __init__(self, dim=EMBED_DIM):
        super().__init__()
        self.blocks = nn.ModuleList([_Block(dim)])
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)

    def get_intermediate_layers(self, x, n=1, return_class_token=False):
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, N, D)
        cls_token = tokens.mean(dim=1)
        layer = (tokens, cls_token) if return_class_token else tokens
        return tuple(layer for _ in range(n))

    def forward(self, x):
        return self.get_intermediate_layers(x)[0]


def dinov2_vits14(pretrained=False, **kwargs):
    return _StubDinoV2()
'''


@pytest.fixture
def dinov2_stub_dir(tmp_path):
    """Return a directory usable as ``dinov2_source_path`` (stub ``hubconf.py``)."""
    stub_dir = tmp_path / "dinov2_stub"
    stub_dir.mkdir()
    (stub_dir / "hubconf.py").write_text(_DINOV2_STUB_HUBCONF)
    return str(stub_dir)


# ---------------------------------------------------------------------------
# WeatherBench (ERA5) synthetic fixture
# ---------------------------------------------------------------------------
# Real geometry is 160x220 with ~12k daily timesteps spanning 1990-2022. We
# reproduce the *layout* (a zarr store + a statistics .npy) at a tiny scale so
# the loading code runs without touching the huge machine-specific store.

WB_LAT = 8
WB_LON = 10
# A handful of consecutive days per split-boundary year, at the store's daily
# UTC12 cadence -- the real store lives under `.../utc12/` and is already
# subsampled to one timestep per day, so `WeatherBench` does no hour filtering of
# its own and every timestep here is a sample. ``WB_DAYS_PER_YEAR`` per year is
# enough to exercise the ``last_t_steps`` / ``lead_time`` windows.
#
# The years straddle `get_split`'s boundaries ("1990".."2014", "2015".."2019",
# "2020".."2022") and each slice picks up whole years, so every split is
# non-empty: 3 years of train, 2 of val, 2 of test.
WB_HOUR = 12
WB_DAYS_PER_YEAR = 3
WB_YEARS = [1990, 2000, 2014, 2015, 2019, 2020, 2022]


def _write_weatherbench_zarr(dir_path: str) -> None:
    """Write a synthetic ``era5.zarr`` + ``era5_statistics.npy`` under ``dir_path``.

    Mirrors the real store: the 7 ``WB_INPUT`` variables, where the 5 dynamic
    ones carry a ``time`` axis and the 2 static ones (``WB_STATIC``) do not.
    Values are deterministic (seeded) and each variable gets a distinct
    mean/scale so normalization is observable in the tests.
    """
    import numpy as np
    import pandas as pd
    import xarray as xr

    from src.datamodules.datasets import WB_INPUT, WB_STATIC

    # A real, in-range time coordinate spanning all splits (splits select by
    # year), at the store's daily UTC12 cadence over a few consecutive days per
    # year.
    times = pd.to_datetime(
        [
            f"{year}-06-{1 + day:02d}T{WB_HOUR:02d}:00:00"
            for year in WB_YEARS
            for day in range(WB_DAYS_PER_YEAR)
        ]
    )
    n_time = len(times)

    rng = np.random.default_rng(0)
    lat = np.linspace(90.0, -90.0, WB_LAT)
    lon = np.linspace(0.0, 360.0, WB_LON, endpoint=False)

    data_vars = {}
    mean = np.empty(len(WB_INPUT), dtype=np.float64)
    std = np.empty(len(WB_INPUT), dtype=np.float64)
    for i, name in enumerate(WB_INPUT):
        # Distinct offset/scale per variable so a mis-indexed normalization or a
        # swapped channel would change the values a test checks.
        offset = 100.0 * (i + 1)
        scale = 10.0 * (i + 1)
        if name in WB_STATIC:
            values = offset + scale * rng.standard_normal((WB_LAT, WB_LON))
            data_vars[name] = (("latitude", "longitude"), values)
        else:
            values = offset + scale * rng.standard_normal((n_time, WB_LAT, WB_LON))
            data_vars[name] = (("time", "latitude", "longitude"), values)
        mean[i] = offset
        std[i] = scale

    ds = xr.Dataset(
        data_vars,
        coords={"time": times, "latitude": lat, "longitude": lon},
    )
    ds.to_zarr(os.path.join(dir_path, "era5.zarr"), consolidated=False, mode="w")

    # era5_statistics.npy is shape (n_vars, 2): columns are mean and std. The
    # loader does ``np.split(stats, 2, axis=-1)`` then ``.squeeze()``.
    statistics = np.stack([mean, std], axis=-1)
    np.save(os.path.join(dir_path, "era5_statistics.npy"), statistics)


@pytest.fixture
def weatherbench_data_dir(tmp_path):
    """Return a temp dir holding a tiny synthetic ``era5.zarr`` + statistics.

    Lets the WeatherBench dataset/datamodule tests run against the real loading
    code (``xr.open_zarr``, normalization, downscaling) without depending on the
    large machine-specific store. Returns a trailing-``os.sep`` string
    because ``WeatherBench`` joins the path with ``os.path.join`` (a trailing
    separator is harmless), matching the depth fixture's convention.
    """
    _write_weatherbench_zarr(str(tmp_path))
    return str(tmp_path) + os.sep


# ---------------------------------------------------------------------------
# Time-series (PatchTST-style) synthetic fixture
# ---------------------------------------------------------------------------
# The real benchmark CSVs (ETT, weather, electricity, ...) all share the layout
# ``date`` first column + feature columns + ``OT`` target last. We write a tiny
# CSV in that exact shape so ``TimeSeriesDataset`` runs against real loading
# code without touching the large, machine-specific data files.
#
# Sized for the ``custom`` (70/10/20 ratio) split: with ``TS_ROWS`` rows every
# split is comfortably non-empty for small ``seq_len``/``pred_len``. The
# fixed-month ETT splits need ~14k rows, so those are tested via the pure
# ``TimeSeriesDataset._borders`` function instead (no data needed).

TS_ROWS = 200
TS_N_FEATURES = 4  # feature columns in addition to the ``OT`` target
TS_FREQ = "h"


def _write_timeseries_csv(
    path: str,
    n_rows: int = TS_ROWS,
    n_features: int = TS_N_FEATURES,
    freq: str = TS_FREQ,
) -> None:
    """Write a synthetic forecasting CSV (``date`` + ``n_features`` cols + ``OT``).

    Values are deterministic (seeded); each column gets a distinct offset/scale
    so a mis-fit scaler or a swapped column would change values a test checks.
    """
    import pandas as pd

    rng = np.random.default_rng(0)
    dates = pd.date_range("2016-07-01", periods=n_rows, freq=freq)
    data = {"date": dates.strftime("%Y-%m-%d %H:%M:%S")}
    for j in range(n_features):
        offset = 10.0 * (j + 1)
        scale = float(j + 1)
        data[f"f{j}"] = offset + scale * rng.standard_normal(n_rows)
    # Target column, written last to match the standard layout.
    data["OT"] = 100.0 + 5.0 * rng.standard_normal(n_rows)
    pd.DataFrame(data).to_csv(path, index=False)


@pytest.fixture
def timeseries_csv(tmp_path):
    """Return the path to a tiny synthetic forecasting CSV (custom-split sized)."""
    path = os.path.join(tmp_path, "series.csv")
    _write_timeseries_csv(path)
    return path


# ---------------------------------------------------------------------------
# EUPPBench station graph
# ---------------------------------------------------------------------------

EUPP_N_STATIONS = 8
EUPP_N_FEATURES = 35
EUPP_MEMBERS = {"train": 3, "val": 3, "test": 5}
EUPP_SAMPLES = {"train": 20, "val": 8, "test": 10}


def _write_euppbench(dir_path: str) -> None:
    """Write a miniature replica of ``src/data/build_euppbench.py``'s output.

    Same file names, shapes and dtypes as the real dataset, at a size that runs
    in milliseconds: 8 stations on a small lat/lon grid, 3 members for the
    reforecast-like splits and 5 for the forecast-like test split (the real data
    also changes ensemble size between fit and test, and the model has to
    absorb that).

    A few targets are NaN, including one station that is never observed, so the
    masked-score paths are exercised rather than silently skipped.
    """
    rng = np.random.default_rng(0)
    os.makedirs(dir_path, exist_ok=True)

    for split in ("train", "val", "test"):
        n, m = EUPP_SAMPLES[split], EUPP_MEMBERS[split]
        x = rng.standard_normal((n, EUPP_N_FEATURES, m, EUPP_N_STATIONS), dtype=np.float32)
        y = rng.standard_normal((n, 1, EUPP_N_STATIONS)).astype(np.float32)
        y[:, 0, 0] = np.nan  # a station with no observations at all
        y[0, 0, 3] = np.nan  # and a scattered gap
        np.save(os.path.join(dir_path, f"{split}_x.npy"), x)
        np.save(os.path.join(dir_path, f"{split}_y.npy"), y)

    # Stations spread over roughly 1 degree so that a 100 km threshold keeps a
    # sparse-but-connected graph rather than all-or-nothing.
    lat = 50.0 + 0.5 * np.arange(EUPP_N_STATIONS)
    lon = 8.0 + 0.3 * np.arange(EUPP_N_STATIONS)
    dlat = (lat[:, None] - lat[None, :]) * 111.0
    dlon = (lon[:, None] - lon[None, :]) * 111.0 * np.cos(np.radians(50.0))
    dist = np.sqrt(dlat**2 + dlon**2)

    np.savez(
        os.path.join(dir_path, "meta.npz"),
        station_id=np.arange(EUPP_N_STATIONS, dtype=np.int64),
        station_name=np.array([f"s{i}" for i in range(EUPP_N_STATIONS)]),
        lat=lat,
        lon=lon,
        alt=np.zeros(EUPP_N_STATIONS),
        dist=dist,
        feature_names=np.array([f"f{i}" for i in range(EUPP_N_FEATURES)]),
        mean=np.zeros(EUPP_N_FEATURES - 2, dtype=np.float32),
        std=np.ones(EUPP_N_FEATURES - 2, dtype=np.float32),
        dropped_station_id=np.array([], dtype=np.int64),
        lead_time=np.array("24h"),
        min_obs_frac=np.array(0.5),
    )


@pytest.fixture
def euppbench_data_dir(tmp_path):
    """Return the path to a tiny synthetic EUPPBench station dataset."""
    path = os.path.join(tmp_path, "euppbench_24h")
    _write_euppbench(path)
    return path
