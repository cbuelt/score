# Lightning datamodules for the four task families: monocular depth, ERA5
# (WeatherBench2), long-horizon time series, and EUPPBench station forecasts.

import os
from pathlib import Path
from typing import Optional, Tuple
from math import prod

import lightning as L
import numpy as np
import torch
from torch.utils.data import random_split

from src.datamodules.datasets import (
    DepthDataset,
    EUPPBench,
    TimeSeriesDataset,
    WeatherBench,
)


class DepthDataModule(L.LightningDataModule):
    """LightningDataModule for monocular depth: train/val split and test set.

    The train/val partition is drawn from ``split_seed`` alone, never from the
    global (``L.seed_everything``) RNG, so a sweep over run seeds trains
    different models on the *same* data. See :meth:`setup`.
    """

    def __init__(
        self,
        data_path: Path,
        dataset: str,
        batch_size: int = 32,
        num_workers: int = 0,
        val_split: float = 0.1,
        split_seed: int = 42,
        pin_memory: bool = False,
        preload: bool = False,
        eval_batch_size: Optional[int] = None,
        persistent_workers: bool = True,
    ):
        """Initialize the depth datamodule.

        Args:
            data_path: Path to the depth data folder.
            dataset: Dataset name (e.g. ``"nyu"``).
            batch_size: Training batch size.
            num_workers: DataLoader worker processes.
            val_split: Fraction of the training file held out for validation.
            split_seed: Seed for the train/val partition *only*. Deliberately
                separate from the global run seed (``cfg.seed``) so that running
                several seeds varies model init / shuffling but never the data;
                bind it to ``cfg.data_seed`` in the config.
            pin_memory: Pin host memory in the DataLoaders.
            preload: Whether to load the arrays into RAM up front.
            eval_batch_size: Batch size for the test loader (falls back to
                ``batch_size``).
            persistent_workers: Keep the train/val workers alive between epochs
                instead of respawning them each epoch. Ignored when
                ``num_workers=0``. With ``preload=False`` the workers re-open the
                HDF5 file on every respawn, so this saves real work.
        """
        super().__init__()
        self.data_path = data_path
        self.dataset = dataset
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.split_seed = split_seed
        self.pin_memory = pin_memory
        self.preload = preload
        self.persistent_workers = persistent_workers

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def get_channels(self) -> int:
        """Number of channels in the target, taken from the dataset class.

        Resolved statically (no data loaded) so it can size the model head and
        predictive method before `setup()`.
        """
        return 1

    def get_input_channels(self) -> int:
        """Number of channels in the model input (RGB image).

        Resolved statically (no data loaded), mirroring :meth:`get_channels`, so
        it can size the backbone input stem before `setup()`.
        """
        return 3

    def get_total_dimension(self) -> int:
        """Total target event dimension ``d = C * H * W`` of a single sample.

        Used to scale the kernel bandwidth as ``gamma = sqrt(gamma_* * d)``, which
        makes sampling-based scores roughly dataset-independent under pixel-wise
        standardization. The spatial size is not known statically, so it is read
        from one (lazily loaded) target sample without materialising the full set.
        """
        probe = DepthDataset(
            data_path=self.data_path,
            dataset=self.dataset,
            split="test",
            preload=False,
        )
        shape = probe.get_image_size()
        return int(prod(shape))

    def setup(self, stage: Optional[str] = None):
        """Prepare datasets for different stages (idempotent).

        The train/val partition is drawn from a generator seeded fresh from
        ``split_seed`` on every call, so it does not depend on the global RNG
        state, on how often ``setup()`` runs, or on the DDP rank -- the same
        ``split_seed`` always yields the same partition.
        """
        if stage in (None, "fit", "validate") and self.train_dataset is None:
            full_train = DepthDataset(
                data_path=self.data_path,
                dataset=self.dataset,
                split="train",
                preload=self.preload,
            )
            if self.val_split > 0 and len(full_train) > 1:
                val_size = max(1, int(len(full_train) * self.val_split))
                val_size = min(val_size, len(full_train) - 1)
                train_size = len(full_train) - val_size
                self.train_dataset, self.val_dataset = random_split(
                    full_train,
                    [train_size, val_size],
                    generator=torch.Generator().manual_seed(self.split_seed),
                )
            else:
                self.train_dataset = full_train
                self.val_dataset = None

        if self.test_dataset is None:
            self.test_dataset = DepthDataset(
                data_path=self.data_path,
                dataset=self.dataset,
                split="test",
                preload=self.preload,
            )

    def _build_loader(
        self,
        dataset,
        shuffle: bool = False,
        test: bool = False,
        persistent: bool = False,
    ):
        if dataset is None:
            raise RuntimeError("Requested dataloader before calling setup().")
        batch_size = self.batch_size
        if test and self.eval_batch_size is not None:
            batch_size = self.eval_batch_size
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # DataLoader rejects persistent_workers with num_workers=0, so the
            # flag is only honoured once there are workers to keep alive.
            persistent_workers=persistent and self.num_workers > 0,
        )

    def train_dataloader(self):
        return self._build_loader(
            self.train_dataset, shuffle=True, persistent=self.persistent_workers
        )

    def val_dataloader(self):
        dataset = (
            self.val_dataset if self.val_dataset is not None else self.train_dataset
        )
        return self._build_loader(
            dataset, shuffle=False, persistent=self.persistent_workers
        )

    def test_dataloader(self, test=True):
        # Not persistent: the test loader runs once, so live workers afterwards
        # would only hold memory.
        return self._build_loader(self.test_dataset, shuffle=False, test=test)


class WeatherBenchModule(L.LightningDataModule):
    """Lightning datamodule wrapper for the ERA5 (WeatherBench2) dataset.

    The split needs no seed: it is the fixed year range per split
    (:meth:`WeatherBench.get_split`), so it is identical across runs regardless
    of the global run seed. Only shuffling and model training consume the global
    RNG.
    """

    def __init__(
        self,
        batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 0,
        data_path: str = "data/era5/",
        normalize=True,
        preload: bool = True,
        downscaling_factor: int = 1,
        last_t_steps: int = 2,
        lead_time: int = 1,
        residual: bool = True,
        persistent_workers: bool = True,
        pin_memory: bool = True,
        statistics_path: Optional[str] = None,
    ):
        """Initialize Datamodule.

        Args:
            batch_size (int, optional): Training batch size. Defaults to 32.
            eval_batch_size (int, optional): Val/test batch size. Defaults to 32.
            num_workers (int, optional): Num workers for dataloader. Defaults to 0.
            data_path (str, optional): Defaults to "data/era5/".
            normalize (bool, optional): Whether to normalize the data. Defaults to True.
            preload (bool, optional): Whether to preload each split into RAM. Defaults to True.
            downscaling_factor (int, optional): Grid downscaling factor. Defaults to 1.
            last_t_steps (int, optional): Last t timesteps (days) to use as input. Defaults to 2.
            lead_time (int, optional): Forecast horizon in days ahead. ``lead_time=1``
                is a +1 day forecast, ``lead_time=2`` a +2 day forecast. Defaults to 1.
            persistent_workers (bool, optional): Keep the train/val workers alive
                between epochs instead of respawning them each epoch. Ignored when
                ``num_workers=0``. Note that ``preload=True`` does *not* make one
                worker sufficient: it removes the zarr read, but ``__getitem__``
                still copies, normalizes and concatenates the window, which is
                ~130 ms/sample at ``downscaling_factor=1`` -- so the loader stays
                CPU-bound and wants several workers. Defaults to True.

        Note:
            The datasets are constructed lazily in :meth:`setup`, not here, so
            that no zarr is opened (and, with ``preload=True``, no data is pulled
            into RAM) merely by constructing the module -- which matters under
            DDP where every process instantiates it.
        """
        super().__init__()
        self.data_path = data_path
        self.normalize = normalize
        self.train_batch_size = batch_size
        self.num_workers = num_workers
        self.val_batch_size = eval_batch_size
        self.preload = preload
        self.downscaling_factor = downscaling_factor
        self.last_t_steps = last_t_steps
        self.lead_time = lead_time
        self.residual = residual
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        # Normalization statistics file; None -> <data_path>/era5_statistics.npy
        # (whole-store moments). See WeatherBench.get_statistics.
        self.statistics_path = statistics_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _make_dataset(self, var: str) -> WeatherBench:
        """Construct a single ``WeatherBench`` split with the module's config."""
        return WeatherBench(
            data_path=self.data_path,
            var=var,
            normalize=self.normalize,
            preload=self.preload,
            downscaling_factor=self.downscaling_factor,
            last_t_steps=self.last_t_steps,
            lead_time=self.lead_time,
            residual=self.residual,
            statistics_path=self.statistics_path,
        )

    def setup(self, stage: Optional[str] = None):
        """Build the datasets for the requested stage (idempotent)."""
        if stage in (None, "fit", "validate"):
            if self.train_dataset is None:
                self.train_dataset = self._make_dataset("train")
            if self.val_dataset is None:
                self.val_dataset = self._make_dataset("val")

        if stage in (None, "test", "predict") and self.test_dataset is None:
            self.test_dataset = self._make_dataset("test")

    def get_grid_coords(self):
        """``(latitude, longitude)`` of the test grid in degrees, after downscaling.

        The presence of this method is what tells
        :mod:`src.evaluation.calibration` that the two spatial axes are a lat-lon
        grid rather than square pixels, so a pixel lag has to be converted to a
        physical distance before directions can be compared. Requires
        ``setup("test")``.
        """
        if self.test_dataset is None:
            raise RuntimeError("get_grid_coords() before setup('test')")
        return self.test_dataset.get_coords()

    def get_channels(self) -> int:
        """Number of channels in the target, taken from the dataset class.

        Resolved statically (no data loaded) so it can size the model head and
        predictive method before `setup()`.
        """
        return WeatherBench.get_output_channels()

    def get_input_channels(self) -> int:
        """Number of channels in the model input, taken from the dataset class.

        Delegates to :meth:`WeatherBench.get_input_channels` with the module's
        ``last_t_steps`` so it stays in sync with what ``__getitem__`` stacks
        (dynamic vars over time + static vars). Resolved statically (no data
        loaded), mirroring :meth:`get_channels`.
        """
        return WeatherBench.get_input_channels(self.last_t_steps)

    def get_total_dimension(self) -> int:
        """Total target event dimension ``d = C * H * W`` of a single sample.

        Used to scale the kernel bandwidth as ``gamma = sqrt(gamma_* * d)``, which
        makes sampling-based scores roughly dataset-independent under pixel-wise
        standardization. The spatial size is not known statically, so it is read
        from one (lazily opened) probe dataset without materialising the full set
        or requiring `setup()` to have run.
        """
        probe = WeatherBench(
            data_path=self.data_path,
            var="train",
            normalize=self.normalize,
            preload=False,
            downscaling_factor=self.downscaling_factor,
            last_t_steps=self.last_t_steps,
            lead_time=self.lead_time,
            residual=self.residual,
            statistics_path=self.statistics_path,
        )
        shape = probe.get_dimensions()
        out_channels = self.get_channels()
        dims = out_channels * prod(shape)
        return dims

    def _build_loader(
        self, dataset, shuffle: bool, batch_size: int, persistent: bool = False
    ):
        if dataset is None:
            raise RuntimeError("Requested dataloader before calling setup().")
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=persistent and self.num_workers > 0,
        )

    def train_dataloader(self):
        return self._build_loader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.train_batch_size,
            persistent=self.persistent_workers,
        )

    def val_dataloader(self):
        return self._build_loader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.val_batch_size,
            persistent=self.persistent_workers,
        )

    def test_dataloader(self):
        # Not persistent: the test loader runs once, so live workers afterwards
        # would only hold memory (batches here are ~0.8 GB at df=1).
        return self._build_loader(
            self.test_dataset, shuffle=False, batch_size=self.val_batch_size
        )


class TimeSeriesDataModule(L.LightningDataModule):
    """LightningDataModule for PatchTST-style long-horizon forecasting.

    Wraps :class:`~src.datamodules.datasets.TimeSeriesDataset`, reproducing the
    behaviour of PatchTST's ``data_provider``: one dataset class covering both
    the ETT (fixed-month) and custom (70/10/20 ratio) benchmarks, for univariate
    (``features="S"``) and multivariate (``features="M"`` / ``"MS"``) targets.
    Per-flag loader options match ``data_provider`` for training -- shuffle and
    drop the last partial batch.

    The evaluation loaders deliberately depart from ``data_provider``: val and
    test always pass ``drop_last=False``, whatever ``drop_last`` is set to.
    PatchTST drops the trailing partial batch on every split, which silently
    discards the tail of the test set -- at the default batch size, 97 of
    ETTh1's 2 785 windows (3.5%) and 76 of weather's 10 444 (0.7%) -- and makes
    a reported test score an average over an arbitrary prefix whose length
    depends on the batch size. Only ``train_dataloader`` honours the flag,
    because there partial batches affect optimization and PatchTST parity is
    wanted.

    The split needs no seed: the boundaries are a deterministic function of the
    series length (:meth:`TimeSeriesDataset._borders`) -- fixed months for ETT,
    a 70/10/20 ratio otherwise -- and the windows are chronological, so every run
    sees the same data regardless of the global run seed. Only shuffling and
    model training consume the global RNG.
    """

    def __init__(
        self,
        data_path: str,
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
        batch_size: int = 128,
        eval_batch_size: int = 128,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = True,
        shuffle_val: bool = False,
        preload: bool = True,
        persistent_workers: bool = True,
    ):
        """Initialise the datamodule.

        Args:
            data_path: Full path to the CSV file.
            data_split: Border strategy -- ``"ett_hour"`` | ``"ett_minute"`` |
                ``"custom"``.
            features: ``"M"`` | ``"MS"`` | ``"S"`` (see
                :class:`TimeSeriesDataset`).
            target: Target column name (default ``"OT"``).
            seq_len: Look-back window length.
            pred_len: Forecast horizon length.
            label_len: Decoder warm-up length (only used with ``return_marks``).
            scale: Standardise using train-range statistics only.
            timeenc: ``0`` calendar / ``1`` :func:`time_features` (marks only).
            freq: pandas offset alias for the time-feature encoding.
            return_marks: Return the faithful 4-tuple instead of ``(x, y)``.
            batch_size: Training batch size.
            eval_batch_size: Val/test batch size.
            num_workers: DataLoader worker processes.
            pin_memory: Pin host memory in the DataLoaders.
            drop_last: Drop the last partial *training* batch (matches
                PatchTST). The val and test loaders ignore it and always keep
                the trailing partial batch -- see the class docstring.
            shuffle_val: Shuffle the validation loader (PatchTST shuffles val).
            preload: Forwarded to the dataset (no-op for CSV-backed series).
            persistent_workers: Keep the train/val workers alive between epochs
                instead of respawning them each epoch. Ignored when
                ``num_workers=0``. This matters more here than for the spatial
                datamodules: the series live in RAM and ``__getitem__`` is a
                slice (a few microseconds), so epochs are tens of batches and
                per-epoch worker respawn would otherwise dominate the actual
                data work. Defaults to True.

        Note:
            Datasets are built lazily in :meth:`setup`, not here, so merely
            constructing the module opens no files -- which matters under DDP
            where every process instantiates it.
        """
        super().__init__()
        self.data_path = data_path
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
        self.train_batch_size = batch_size
        self.val_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.shuffle_val = shuffle_val
        self.preload = preload
        self.persistent_workers = persistent_workers

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _make_dataset(self, split: str) -> TimeSeriesDataset:
        """Construct one :class:`TimeSeriesDataset` split with the config."""
        return TimeSeriesDataset(
            data_path=self.data_path,
            split=split,
            data_split=self.data_split,
            features=self.features,
            target=self.target,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            label_len=self.label_len,
            scale=self.scale,
            timeenc=self.timeenc,
            freq=self.freq,
            return_marks=self.return_marks,
            preload=self.preload,
        )

    def setup(self, stage: Optional[str] = None):
        """Build the datasets for the requested stage (idempotent)."""
        if stage in (None, "fit", "validate"):
            if self.train_dataset is None:
                self.train_dataset = self._make_dataset("train")
            if self.val_dataset is None:
                self.val_dataset = self._make_dataset("val")

        if stage in (None, "test", "predict") and self.test_dataset is None:
            self.test_dataset = self._make_dataset("test")

    def get_input_channels(self) -> int:
        """Number of channels in the model input.

        ``"S"`` -> 1 (target only); ``"M"`` / ``"MS"`` -> all value columns.
        Resolved statically from the CSV header (no split built), so it can size
        the backbone before :meth:`setup`.
        """
        if self.features == "S":
            return 1
        return TimeSeriesDataset.n_columns(self.data_path)

    def get_channels(self) -> int:
        """Number of channels in the target/output.

        ``"M"`` -> all value columns; ``"MS"`` / ``"S"`` -> 1 (target only).
        Resolved statically, mirroring :meth:`get_input_channels`.
        """
        if self.features == "M":
            return TimeSeriesDataset.n_columns(self.data_path)
        return 1

    def get_total_dimension(self) -> int:
        """Total target event dimension ``d = C_out * pred_len`` of a sample.

        Used to scale the kernel bandwidth as ``gamma = sqrt(gamma_* * d)``.
        Both factors are known from config, so (unlike the spatial datamodules)
        no probe dataset is needed.
        """
        return self.get_channels() * self.pred_len

    def _build_loader(
        self,
        dataset,
        shuffle: bool,
        batch_size: int,
        drop_last: bool,
        persistent: bool = False,
    ):
        if dataset is None:
            raise RuntimeError("Requested dataloader before calling setup().")
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            # DataLoader rejects persistent_workers with num_workers=0, so the
            # flag is only honoured once there are workers to keep alive.
            persistent_workers=persistent and self.num_workers > 0,
        )

    def train_dataloader(self):
        return self._build_loader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.train_batch_size,
            drop_last=self.drop_last,
            persistent=self.persistent_workers,
        )

    def val_dataloader(self):
        # `drop_last=False`, not `self.drop_last`: an evaluation pass must see
        # every window (see the class docstring).
        return self._build_loader(
            self.val_dataset,
            shuffle=self.shuffle_val,
            batch_size=self.val_batch_size,
            drop_last=False,
            persistent=self.persistent_workers,
        )

    def test_dataloader(self):
        # Not persistent: the test loader runs once, so live workers afterwards
        # would only hold memory. `drop_last=False` as for val.
        return self._build_loader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.val_batch_size,
            drop_last=False,
        )


class EUPPBenchModule(L.LightningDataModule):
    """LightningDataModule for the EUPPBench station graph post-processing task.

    The splits are fixed calendar ranges baked into the preprocessed arrays (see
    ``src/data/build_euppbench.py``), so no split seed is needed -- they are
    identical across runs regardless of the global run seed.

    Train and validation come from the 11-member reforecasts; the test split is
    the 51-member operational forecasts. The differing ensemble size is
    deliberate and is what the backbone's permutation-invariant aggregation has
    to absorb -- it is the "reforecast to forecast" task of Feik et al. (2024).

    Beyond the standard datamodule contract this exposes :meth:`get_distances`,
    which the graph backbone uses to build its edge set and the variogram score
    uses for inverse-distance pair weights, so both read the same station
    geometry.
    """

    def __init__(
        self,
        data_path: str,
        batch_size: int = 8,
        eval_batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        preload: bool = False,
        persistent_workers: bool = True,
    ):
        """Initialize the datamodule.

        Args:
            data_path: Directory written by ``src/data/build_euppbench.py``,
                e.g. ``data/euppbench_24h``.
            batch_size: Training batch size. Feik et al. (2024) use 8; see the
                memory note in the backbone -- the multigraph edge count scales with
                ``batch_size * n_members**2``.
            eval_batch_size: Val/test batch size.
            preload: Read the arrays into RAM instead of memory-mapping them.
                Off by default: the test split alone is ~600 MB and only one
                batch is live at a time.
        """
        super().__init__()
        self.data_path = data_path
        self.train_batch_size = batch_size
        self.val_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.preload = preload
        self.persistent_workers = persistent_workers

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self._meta = None

    @property
    def meta(self) -> dict:
        """Lazily-loaded ``meta.npz`` contents (station geometry, feature names).

        Read on first access rather than in ``__init__`` so that constructing
        the module touches no disk -- which matters under DDP, where every
        process instantiates it.
        """
        if self._meta is None:
            path = os.path.join(self.data_path, "meta.npz")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"{path} not found. Build the dataset first with:\n"
                    f"    python src/data/build_euppbench.py --lead_time <lead>"
                )
            with np.load(path, allow_pickle=False) as f:
                self._meta = {k: f[k] for k in f.files}
        return self._meta

    def setup(self, stage: Optional[str] = None):
        """Build the datasets for the requested stage (idempotent)."""
        if stage in (None, "fit", "validate"):
            if self.train_dataset is None:
                self.train_dataset = EUPPBench(self.data_path, "train", self.preload)
            if self.val_dataset is None:
                self.val_dataset = EUPPBench(self.data_path, "val", self.preload)

        if stage in (None, "test", "predict") and self.test_dataset is None:
            self.test_dataset = EUPPBench(self.data_path, "test", self.preload)

    def get_input_channels(self) -> int:
        """Number of predictor channels ``F`` fed to the backbone.

        Read from the metadata rather than the arrays, so it resolves before
        `setup()` (the backbone and method are sized from it at instantiation).
        """
        return int(len(self.meta["feature_names"]))

    def get_channels(self) -> int:
        """Number of target channels ``C``. One: 2-m temperature."""
        return 1

    def get_total_dimension(self) -> int:
        """Total target event dimension ``d = C * N`` (stations).

        Used to scale the kernel bandwidth as ``gamma = sqrt(gamma_* * d)``.
        """
        return self.get_channels() * int(len(self.meta["station_id"]))

    def get_distances(self) -> np.ndarray:
        """Pairwise geodesic station distance matrix in km, shape ``(N, N)``."""
        return self.meta["dist"]

    def get_graph(self) -> Tuple[np.ndarray, np.ndarray]:
        """``(dist, coords)`` with ``coords`` the ``(N, 2)`` lat/lon array."""
        return self.meta["dist"], np.stack([self.meta["lat"], self.meta["lon"]], axis=-1)

    def _build_loader(self, dataset, shuffle: bool, batch_size: int, persistent: bool = False):
        if dataset is None:
            raise RuntimeError("Requested dataloader before calling setup().")
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=persistent and self.num_workers > 0,
        )

    def train_dataloader(self):
        return self._build_loader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.train_batch_size,
            persistent=self.persistent_workers,
        )

    def val_dataloader(self):
        return self._build_loader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.val_batch_size,
            persistent=self.persistent_workers,
        )

    def test_dataloader(self):
        return self._build_loader(
            self.test_dataset, shuffle=False, batch_size=self.val_batch_size
        )
