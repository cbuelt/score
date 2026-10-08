# PyTorch datasets for ERA5 (WeatherBench2), monocular depth (NYU Depth v2),
# long-horizon time series (PatchTST benchmarks) and EUPPBench station forecasts.

import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset

from src.datamodules.timefeatures import time_features

WB_INPUT = [
    "2m_temperature",
    "temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "geopotential",
    "land_sea_mask",
    "geopotential_at_surface",
]
WB_STATIC = ["land_sea_mask", "geopotential_at_surface"]
WB_DYNAMIC = [
    "2m_temperature",
    "temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "geopotential",
]
WB_TARGET = ["2m_temperature"]


class WeatherBench(Dataset):
    """
    ERA5 surface-temperature dataset from WeatherBench2, written by ``src/data/download_era5.py``.
    """

    def __init__(
        self,
        data_path: str = "data/era5/",
        var: str = "train",
        normalize=True,
        downscaling_factor: int = 1,
        last_t_steps: int = 2,
        lead_time: int = 1,
        preload: bool = True,
        residual: bool = True,
        statistics_path: Optional[str] = None,
    ):
        """Initialize WeatherBench2 dataset.

        Only the daily UTC12 timestep is used (one sample per day), so
        consecutive timesteps are exactly one day apart. This makes ``lead_time``
        a straightforward day offset for the forecast target.

        Args:
            data_path (str, optional): Data path. Defaults to "data/era5/".
            var (str, optional): Train/test/val selection. Defaults to "train".
            normalize (bool, optional): Normalization. Defaults to True.
            downscaling_factor (int, optional): Grid downscaling factor. Defaults to 1.
            last_t_steps (int, optional): Last t timesteps (days) to use as input. Defaults to 2.
            lead_time (int, optional): Forecast horizon in days ahead of the last
                input day. ``lead_time=1`` is a +1 day forecast, ``lead_time=2`` a
                +2 day forecast. Defaults to 1.
            preload (bool, optional): Whether to preload the whole dataset into RAM. Defaults to True.
            statistics_path (str, optional): File with the per-variable
                normalization mean/std, shape ``(len(WB_INPUT), 2)``. Defaults
                to ``<data_path>/era5_statistics.npy`` as written by the download
                script (whole 1990-2022 period); pass a file computed on the
                training years only with ``src/data/era5_statistics.py``
                (``era5_statistics_train1990-2014.npy``) to keep the val/test
                years out of the normalization.
        """
        self.var = var
        self.normalize = normalize
        self.last_t_steps = last_t_steps
        self.lead_time = lead_time
        self.downscaling_factor = downscaling_factor

        time_slice = self.get_split(self.var)
        zarr_path = os.path.join(data_path, "era5.zarr")
        self.dataset = xr.open_zarr(zarr_path, consolidated=False)[WB_INPUT].sel(
            time=time_slice
        )
        # The store is already subsampled to a daily UTC12 cadence, so every
        # timestep is a sample (one per day, consecutive steps one day apart).
        self.mean, self.std = self.get_statistics(data_path, statistics_path)
        self.dataset = self.dataset.isel(
            latitude=slice(None, None, downscaling_factor),
            longitude=slice(None, None, downscaling_factor),
        )
        if preload:
            self.data_array = self.dataset.to_array().load()
        else:
            self.data_array = None

        self.n_vars = len(WB_INPUT)

    def get_coords(self):
        """``(latitude, longitude)`` in degrees for this split's grid.

        Read off the already-downscaled ``xarray`` object, so it reflects
        ``downscaling_factor`` rather than the store's native resolution.
        """
        return (
            self.dataset["latitude"].values.astype("float64"),
            self.dataset["longitude"].values.astype("float64"),
        )

    @staticmethod
    def get_split(var: str) -> slice:
        # 33 years of daily UTC12 data (1990-2022), split by year with no gaps:
        # 25y train / 5y val / 3y test (the last three years held out as test).
        if var == "train":
            return slice("1990", "2014")
        elif var == "val":
            return slice("2015", "2019")
        elif var == "test":
            return slice("2020", "2022")
        else:
            raise AssertionError(f"{var} is not in [train, val, test, ood]")

    @staticmethod
    def get_statistics(data_path: str, statistics_path: Optional[str] = None) -> Tuple:
        """Load data statistics from path.

        Args:
            data_path (str): Path to data folder
            statistics_path (str, optional): explicit statistics file; defaults
                to ``<data_path>/era5_statistics.npy``.

        Returns:
            Tuple: Tuple of mean and standard deviation of data.
        """
        if statistics_path is None:
            statistics_path = os.path.join(data_path, "era5_statistics.npy")
        statistics = np.load(statistics_path)
        mean, std = np.split(statistics, 2, axis=-1)
        return mean.squeeze(), std.squeeze()

    @staticmethod
    def get_output_channels() -> int:
        return len(WB_TARGET)

    @staticmethod
    def get_input_channels(last_t_steps: int = 2) -> int:
        """Number of channels the input tensor carries (see ``__getitem__``).

        ``__getitem__`` concatenates the dynamic variables over ``last_t_steps``
        timesteps with the static variables at ``t=0``:
        ``len(WB_DYNAMIC) * last_t_steps + len(WB_STATIC)``. Kept in sync with the
        stacking in ``__getitem__`` so the model input head can be sized
        statically (no data loaded)."""
        return len(WB_DYNAMIC) * last_t_steps + len(WB_STATIC)

    def __len__(self) -> int:
        """Returns the length of the dataset

        Each window spans ``last_t_steps`` input days plus ``lead_time`` days to
        reach the forecast target, so the last ``last_t_steps + lead_time - 1``
        days cannot start a window.

        Returns:
            int: Length of the dataset
        """
        return len(self.dataset.time) - self.last_t_steps - self.lead_time + 1

    def __getitem__(self, idx: int) -> tuple:
        """Returns the idx-th element of the dataset

        Args:
            idx (int): Index of the element to be returned

        Returns:
            tuple: Tuple containing the input and output tensors
        """
        # Window: last_t_steps input days, then lead_time days on to the target.
        # The last input day ("day 0") is at local index last_t_steps - 1, and
        # the forecast target is lead_time days later.
        window_len = self.last_t_steps + self.lead_time
        if self.data_array is not None:
            sample = self.data_array[:, idx : idx + window_len].values.copy()
        else:
            sample = (
                self.dataset.isel(time=slice(idx, idx + window_len))
                .to_array()
                .values.copy()
            )

        # Shape: [variables, time, lat, lon]
        if self.normalize:
            sample = (sample - self.mean[:, None, None, None]) / self.std[
                :, None, None, None
            ]

        # Input: [dynamic (all but WB_STATIC) over last_t_steps] + [static (first time)]
        input_dynamic = sample[np.isin(WB_INPUT, WB_DYNAMIC)][:, : self.last_t_steps]
        input_dynamic = input_dynamic.reshape(
            -1, *input_dynamic.shape[2:]
        )  # flatten time and var dims

        static = sample[np.isin(WB_INPUT, WB_STATIC), 0]  # static at t=0
        input_tensor = torch.from_numpy(
            np.concatenate([input_dynamic, static], axis=0)
        ).float()

        # Target: delta between the +lead_time forecast day and day 0 (the last
        # input day) for WB_TARGET.
        target_idx = np.isin(WB_INPUT, WB_TARGET)
        day0 = self.last_t_steps - 1
        target_t = sample[target_idx, day0 + self.lead_time]
        target_t_prev = sample[target_idx, day0]
        target_tensor = torch.from_numpy(target_t - target_t_prev).float()

        return input_tensor, target_tensor

    def get_grid(self) -> Tuple:
        """Returns the grid of the dataset.

        Returns:
            Tuple: Tuple containing x (lat,lon) and t coordinate values
        """
        lat = self.dataset.latitude.values
        lon = self.dataset.longitude.values
        # One (daily UTC12) timestep per sample; the prediction time is the
        # forecast target, i.e. day 0 shifted by lead_time days. Day 0 of sample
        # idx is at time index idx + last_t_steps - 1.
        day0_start = self.last_t_steps - 1
        t = self.dataset.time[day0_start : day0_start + len(self)] + pd.Timedelta(
            days=self.lead_time
        )  # Returns prediction time
        return ((lat, lon), t)

    def get_dimensions(self) -> Tuple:
        """Get dimensions of the coordinates.

        Returns:
            Tuple: Size of the data-dimensions (lat,lon).
        """
        x = self.dataset.sizes["latitude"]
        y = self.dataset.sizes["longitude"]
        return (x, y)


class DepthDataset(Dataset):
    def __init__(
        self,
        data_path: str = "data/nyu/",
        dataset: str = "nyu",
        split: str = "train",
        preload: bool = True,
    ):
        """Initialize the depth-regression dataset.

        Args:
            data_path (str, optional): Path to the data folder.
            dataset (str, optional): Dataset name (e.g. "nyu"). Defaults to "nyu".
            split (str, optional): Split name (e.g. "train"). Defaults to "train".
            preload (bool, optional): Whether to load the whole dataset into RAM
                up front. If False, samples are read lazily from disk on access,
                which is slower per sample but avoids holding the full arrays in
                memory. Defaults to True.
        """
        super().__init__()
        self.path = data_path
        self.filename = dataset + "_" + split + ".h5"
        self.dataset = xr.open_dataset(data_path + self.filename, phony_dims="access")

        def _to_ncxy(var):
            # (n, x, y, [1,] c) with an optional singleton between y and c: the
            # original NYU benchmark file stores the image 5-dimensional, the
            # files written by src/data/build_nyu_bts.py are 4-dimensional.
            dims = list(var.dims)
            names = {dims[0]: "n", dims[1]: "x", dims[2]: "y", dims[-1]: "c"}
            var = var.rename(names)
            if len(dims) == 5:
                var = var.squeeze(dims[3])
            return var.transpose("n", "c", "x", "y")

        self.depth = _to_ncxy(self.dataset["depth"])
        self.image = _to_ncxy(self.dataset["image"])

        # Two supported target encodings, distinguished by dtype (available
        # without loading): uint8 = the original NYU benchmark file (values / 255)
        # and uint16 = build_nyu_bts.py (millimeters, capped at 10 m, / 10000).
        # Anything else is rejected rather than silently mis-scaled.
        if self.image.dtype != np.uint8:
            raise TypeError(
                f"{self.filename}: variable 'image' has dtype "
                f"{self.image.dtype}, expected uint8 (values are normalized "
                "by /255)."
            )
        if self.depth.dtype == np.uint8:
            self._depth_scale = 255.0
        elif self.depth.dtype == np.uint16:
            self._depth_scale = 10000.0
        else:
            raise TypeError(
                f"{self.filename}: variable 'depth' has dtype "
                f"{self.depth.dtype}, expected uint8 (/255) or uint16 "
                "millimeters (/10000)."
            )

        if preload:
            self.depth = self.depth.load()
            self.image = self.image.load()

    def __len__(self) -> int:
        """Returns the length of the dataset

        Returns:
            int: Length of the dataset
        """
        return self.image.shape[0]

    def get_image_size(self) -> torch.Size:
        return self.depth.shape[2:]

    def __getitem__(self, idx: int) -> tuple:
        x = self.image[idx].values
        y = self.depth[idx].values
        x_tensor = torch.tensor(x / 255, dtype=torch.float32)
        y_tensor = torch.tensor(y / self._depth_scale, dtype=torch.float32)
        return x_tensor, y_tensor


# The three border strategies PatchTST uses. ``ett_hour``/``ett_minute`` are the
# fixed-month splits from ``Dataset_ETT_hour``/``Dataset_ETT_minute`` (minute =
# hour x4 for the 15-minute cadence); ``custom`` is the 70/10/20 ratio split from
# ``Dataset_Custom`` used by every non-ETT benchmark.
TS_SPLITS = ("ett_hour", "ett_minute", "custom")
TS_SPLIT_INDEX = {"train": 0, "val": 1, "test": 2}


class _StandardScaler:
    """Minimal per-column standardiser (numpy only).

    Mirrors ``sklearn.preprocessing.StandardScaler`` closely enough for the
    forecasting datasets -- population std (``ddof=0``), and zero-variance
    columns get a scale of ``1`` so constant channels pass through unchanged.
    sklearn is not a dependency of this project, hence this small stand-in;
    PatchTST fits the scaler on the training range only and transforms the whole
    series, which is what :class:`TimeSeriesDataset` does with it.
    """

    def fit(self, data: np.ndarray) -> "_StandardScaler":
        self.mean_ = data.mean(axis=0)
        std = data.std(axis=0)
        std[std == 0.0] = 1.0
        self.scale_ = std
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean_) / self.scale_

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * self.scale_ + self.mean_


class TimeSeriesDataset(Dataset):
    """PatchTST-style long-horizon forecasting dataset.

    A single class covering the ETT (fixed-month split) and "custom" (70/10/20
    ratio split) benchmarks, for both univariate (``features="S"``) and
    multivariate (``features="M"`` / ``"MS"``) targets. It reproduces the
    behaviour of PatchTST's ``data_provider`` Dataset classes -- the only thing
    that differs between them is the split-boundary computation, selected here by
    the ``data_split`` argument (see :meth:`_borders`).

    Each CSV is assumed to have a ``date`` first column and the target
    (``OT`` by default) last, matching the standard Informer/Autoformer layout.

    A sample is, by default, a channels-first 2-tuple ``(x, y)`` with
    ``x = (C_in, seq_len)`` and ``y = (C_out, pred_len)`` (this repo's
    convention). Setting ``return_marks=True`` additionally returns the calendar
    time features and uses the faithful ``[s_end - label_len : s_end + pred_len]``
    target window -- but PatchTST's own model ignores marks and ``label_len``, so
    the default 2-tuple loses nothing for reproducing it.
    """

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        data_split: str = "custom",
        features: str = "M",
        target: str = "OT",
        seq_len: int = 336,
        pred_len: int = 96,
        label_len: int = 0,
        scale: bool = True,
        timeenc: int = 0,
        freq: str = "h",
        return_marks: bool = False,
        preload: bool = True,
    ):
        """Initialise the forecasting dataset.

        Args:
            data_path: Full path to the CSV file (this collapses PatchTST's
                ``root_path`` + ``data_path`` into one, matching this repo's
                single-``data_path`` convention).
            split: Which split to expose -- ``"train"`` | ``"val"`` | ``"test"``.
            data_split: Border strategy -- ``"ett_hour"`` | ``"ett_minute"`` |
                ``"custom"``. Picks how train/val/test boundaries are computed.
            features: ``"M"`` (multivariate in, multivariate out), ``"MS"``
                (multivariate in, univariate target out), or ``"S"`` (univariate
                in and out, target only).
            target: Name of the target column. Defaults to ``"OT"``.
            seq_len: Input (look-back) window length.
            pred_len: Forecast horizon length.
            label_len: Decoder warm-up length. Kept for fidelity; only affects
                the ``return_marks=True`` target window (PatchTST's own model
                does not use it).
            scale: Standardise using stats fit on the training range only.
            timeenc: ``0`` -> calendar features, ``1`` -> :func:`time_features`.
                Only consulted when ``return_marks=True``.
            freq: pandas offset alias driving the time-feature encoding.
            return_marks: If ``True`` return the faithful 4-tuple
                ``(x, y, x_mark, y_mark)`` instead of ``(x, y)``.
            preload: Accepted for API symmetry with the other datamodules; the
                CSV is always read into memory as numpy arrays, so this is a
                no-op here.
        """
        super().__init__()
        assert data_split in TS_SPLITS, (
            f"data_split must be one of {TS_SPLITS}, got {data_split!r}"
        )
        assert split in TS_SPLIT_INDEX, (
            f"split must be one of {tuple(TS_SPLIT_INDEX)}, got {split!r}"
        )
        assert features in ("M", "MS", "S"), (
            f"features must be one of ('M', 'MS', 'S'), got {features!r}"
        )
        self.data_path = data_path
        self.split = split
        self.data_split = data_split
        self.features = features
        self.target = target
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.return_marks = return_marks

        self.scaler = _StandardScaler()
        self.__read_data__()

    @staticmethod
    def _borders(data_split: str, n: int, seq_len: int) -> Tuple[List[int], List[int]]:
        """Left/right index of each split (train, val, test).

        Faithful to PatchTST: ETT uses fixed months (``12*30*24`` hourly steps
        train, then 4 + 4 months for val/test; minute = hour x4), custom uses a
        70% train / 10% val / 20% test ratio. Each split's window is
        ``[b1[i], b2[i]]`` and the scaler is always fit on the *train* window
        ``[b1[0], b2[0]]``.
        """
        if data_split == "ett_hour":
            b1 = [0, 12 * 30 * 24 - seq_len, 12 * 30 * 24 + 4 * 30 * 24 - seq_len]
            b2 = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        elif data_split == "ett_minute":
            b1 = [
                0,
                12 * 30 * 24 * 4 - seq_len,
                12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - seq_len,
            ]
            b2 = [
                12 * 30 * 24 * 4,
                12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
                12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,
            ]
        else:  # "custom": 70/10/20 ratio split
            num_train = int(n * 0.7)
            num_test = int(n * 0.2)
            num_vali = n - num_train - num_test
            b1 = [0, num_train - seq_len, n - num_test - seq_len]
            b2 = [num_train, num_train + num_vali, n]
        return b1, b2

    @staticmethod
    def n_columns(data_path: str) -> int:
        """Number of value columns (all columns except ``date``).

        Read from the CSV header only (no data), so the datamodule can size the
        model before any split is built.
        """
        return len(pd.read_csv(data_path, nrows=0).columns) - 1

    def __read_data__(self) -> None:
        df_raw = pd.read_csv(self.data_path)

        # Reorder to the canonical layout ['date'] + others + [target] so the
        # target is always the last column (matches PatchTST's Dataset_Custom;
        # a no-op for the already-ordered ETT files).
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove("date")
        df_raw = df_raw[["date"] + cols + [self.target]]

        b1s, b2s = self._borders(self.data_split, len(df_raw), self.seq_len)
        i = TS_SPLIT_INDEX[self.split]
        b1, b2 = b1s[i], b2s[i]

        if self.features in ("M", "MS"):
            df_data = df_raw[df_raw.columns[1:]]  # everything except 'date'
        else:  # "S": target only
            df_data = df_raw[[self.target]]

        if self.scale:
            train_values = df_data.values[b1s[0] : b2s[0]]
            self.scaler.fit(train_values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        data = np.asarray(data, dtype=np.float32)

        self.data_x = data[b1:b2]
        self.data_y = data[b1:b2]

        if self.return_marks:
            self.data_stamp = self._build_time_marks(df_raw[["date"]][b1:b2])
        else:
            self.data_stamp = None

    def _build_time_marks(self, df_stamp: pd.DataFrame) -> np.ndarray:
        """Calendar / time-feature encoding for the split's timestamps.

        ``timeenc=0`` mirrors PatchTST's per-field calendar columns (with the
        extra ``minute // 15`` bucket for the ``ett_minute`` cadence);
        ``timeenc=1`` delegates to :func:`time_features`.
        """
        dates = pd.to_datetime(df_stamp["date"].values)
        if self.timeenc == 0:
            marks = [
                dates.month.to_numpy(),
                dates.day.to_numpy(),
                dates.weekday.to_numpy(),
                dates.hour.to_numpy(),
            ]
            if self.data_split == "ett_minute":
                marks.append(dates.minute.to_numpy() // 15)
            data_stamp = np.stack(marks, axis=-1)
        else:  # timeenc == 1
            data_stamp = time_features(dates, freq=self.freq).transpose(1, 0)
        return np.asarray(data_stamp, dtype=np.float32)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Undo the standardisation (delegates to the fitted scaler)."""
        return self.scaler.inverse_transform(data)

    def __len__(self) -> int:
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index: int) -> tuple:
        s_begin = index
        s_end = s_begin + self.seq_len

        seq_x = self.data_x[s_begin:s_end]  # (seq_len, C_in)
        x = torch.from_numpy(seq_x).float().transpose(0, 1)  # (C_in, seq_len)

        if self.return_marks:
            # Faithful PatchTST target window: label_len warm-up + pred_len.
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len
            seq_y = self.data_y[r_begin:r_end]
            if self.features == "MS":
                seq_y = seq_y[:, -1:]
            y = torch.from_numpy(seq_y).float().transpose(0, 1)
            x_mark = torch.from_numpy(self.data_stamp[s_begin:s_end]).float().transpose(0, 1)
            y_mark = torch.from_numpy(self.data_stamp[r_begin:r_end]).float().transpose(0, 1)
            return x, y, x_mark, y_mark

        seq_y = self.data_y[s_end : s_end + self.pred_len]  # (pred_len, C)
        if self.features == "MS":
            seq_y = seq_y[:, -1:]  # target column only
        y = torch.from_numpy(seq_y).float().transpose(0, 1)  # (C_out, pred_len)
        return x, y


class EUPPBench(Dataset):
    """One split of the EUPPBench station subset, built by ``src/data/build_euppbench.py``.

    Samples are ``(x, y)`` with

        x: (F, M, N)  standardized ensemble predictors
        y: (C=1, N)   station observations in degrees Celsius, NaN where missing

    where ``F`` is the predictor count, ``M`` the ensemble size (11 for the
    reforecast train/val splits, 51 for the forecast test split) and ``N`` the
    number of stations. The station axis is the event axis, so the flattened
    event dimension is ``V = C * N`` and every score in the repo works on it
    unchanged.

    **NaNs are deliberately preserved.** Observations are missing for a few
    percent of station-days and no day has every station reporting, so dropping
    incomplete samples would drop the dataset. The predictive methods mask them
    out of the loss instead (see
    :meth:`~src.methods.base.PredictiveMethod.apply_mask`); the predictors
    themselves contain no NaN.

    The arrays are memory-mapped by default: the test split alone is ~600 MB and
    only one batch is touched at a time, so paying for a full read into RAM (and
    replicating it per dataloader worker) is wasted.
    """

    SPLITS = ("train", "val", "test")

    def __init__(self, data_path: str, split: str, preload: bool = False):
        if split not in self.SPLITS:
            raise ValueError(f"Unknown split {split!r}; expected one of {self.SPLITS}")
        paths = {a: os.path.join(data_path, f"{split}_{a}.npy") for a in ("x", "y")}
        missing = [p for p in paths.values() if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                f"{missing[0]} not found. Build the dataset first with:\n"
                f"    python src/data/build_euppbench.py --lead_time <lead>"
            )
        mode = None if preload else "r"
        self.x = np.load(paths["x"], mmap_mode=mode)
        self.y = np.load(paths["y"], mmap_mode=mode)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # `.copy()` is not optional on the memory-mapped path: a slice of a
        # read-only mmap would otherwise back the tensor directly, which torch
        # flags as non-writable (and which would keep the mapping alive for as
        # long as the batch does).
        x = torch.from_numpy(np.array(self.x[index], dtype=np.float32, copy=True))
        y = torch.from_numpy(np.array(self.y[index], dtype=np.float32, copy=True))
        return x, y
