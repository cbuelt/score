"""Build the EUPPBench station subset for the graph post-processing model.

The model and data pipeline follow Feik, Lerch & Stühmer (2024), *Graph Neural
Networks and Spatial Information Learning for Post-Processing Ensemble Weather
Forecasts* (arXiv:2407.11050), referred to below as "the reference".

Reads the raw EUPPBench zarr stores (station subset of the ESSD benchmark,
Demaeyer et al. 2023) and writes dense per-split arrays plus a station metadata
file, for one lead time. This is the offline counterpart of
``utils/data.py`` in https://github.com/hits-mli/gnn-post-processing, but it
writes ``.npy`` arrays rather than pandas pickles: the model consumes a dense
``(samples, features, members, stations)`` tensor, so materializing an
intermediate long-format dataframe (which the reference does, at ~10x the
memory) buys nothing, and pickles do not survive a pandas major version.

**Splits.** The dataset ships two disjoint families, and the "reforecast to
forecast" (R2F) task of the reference uses both:

* *reforecasts* -- 209 reference dates x 20 hindcast years x 11 members. Each
  reference date carries the same calendar day in each of the 20 preceding
  years, so the sample axis is ``(time, year)`` flattened and keyed by the
  *pseudo-time* ``time - (21 - year) * 365 d`` (the reference implementation's
  convention). Split into train (< 2010) and validation ([2010, 2014)).
* *forecasts* -- 730 daily runs 2017-2018 with 51 members. The test set.

The resulting sizes (2611 / 836 / 730) reproduce Table A.1 of the reference.

**Station filtering.** Observations are missing for a substantial fraction of
the training window, and the missingness is concentrated: four stations have no
usable observation at all in 1997-2009, so their per-station head would never
receive a supervised gradient. ``--min_obs_frac`` drops stations whose
train-window observed fraction falls below the threshold; the default 0.5 keeps
115 of 122. Note this does *not* remove the need for masked losses -- even after
filtering, only ~14% of training days have every kept station reporting.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COUNTRIES = ["austria", "belgium", "france", "germany", "netherlands"]

# The 33 predictors of the reference implementation, in a fixed order that both
# the arrays and `meta.npz`'s `feature_names` follow.
#
# Two variables are omitted, both because of the data:
#   * `cin` -- excluded by the reference ("omitted in the final analysis due to
#     missing data", Table A.2 footnote).
#   * `r` (relative humidity at 850 hPa) -- present in the *forecast* pressure
#     stores but absent from every *reforecast* one, so it cannot be a training
#     predictor.
# The first four are station-level coordinates rather than data variables and
# are broadcast over the sample/member axes below.
STATIC_FEATURES = [
    "model_orography",
    "station_altitude",
    "station_latitude",
    "station_longitude",
]
DYNAMIC_FEATURES = [
    "cape", "sd", "stl1", "swvl1", "t2m", "tcc", "tcw", "tcwv",
    "u10", "u100", "v10", "v100", "vis",
    "cp6", "mn2t6", "mx2t6", "p10fg6", "slhf6", "sshf6",
    "ssr6", "ssrd6", "str6", "strd6", "tp6",
    "z", "q", "u", "v", "t",
]
# Appended after standardization, as in the reference: seasonality is already
# bounded in [-1, 1] and its scale is meaningful.
CYCLIC_FEATURES = ["cos_doy", "sin_doy"]
FEATURE_NAMES = STATIC_FEATURES + DYNAMIC_FEATURES + CYCLIC_FEATURES

FAMILIES = ["surface", "surface_postprocessed", "pressure_500", "pressure_700", "pressure_850"]

TRAIN_END = pd.Timestamp("2010-01-01")
VAL_END = pd.Timestamp("2014-01-01")

# Hindcast year offsets run 1..20 and count *backwards* from the reference date,
# so `year=20` is one year before it. Matching the reference implementation.
YEAR_ORIGIN = 21


def _open(path: Path) -> xr.Dataset:
    return xr.open_dataset(path, engine="zarr", chunks=None, decode_timedelta=True)


def _load_family(data_path: Path, kind: str, country: str, lead: pd.Timedelta) -> xr.Dataset:
    """Merge the five predictor families of one country at one lead time.

    ``kind`` is ``"forecasts"`` or ``"reforecasts"``. ``valid_time`` is dropped
    before merging because its dimensionality differs between families, and the
    degenerate level dims (``surface``, ``depthBelowLandLayer``,
    ``isobaricInhPa``) are squeezed away.

    The lead time is selected *before* the merge, not after. The
    ``surface_postprocessed`` family holds 6-hourly accumulations and so starts
    at step 6 (20 steps) while the instantaneous families start at step 0
    (21 steps); merging first would align them on the union of steps and
    back-fill the accumulated variables with NaN at step 0.
    """
    parts = []
    for family in FAMILIES:
        ds = _open(data_path / f"stations_ensemble_{kind}_{family}_{country}.zarr")
        ds = ds.drop_vars("valid_time").squeeze(drop=True).sel(step=lead)
        parts.append(ds.reset_coords("step", drop=True))
    return xr.merge(parts, join="exact", compat="no_conflicts")


def _load_obs(data_path: Path, kind: str, country: str, lead: pd.Timedelta) -> xr.DataArray:
    """The t2m observations matching one predictor family, as a bare DataArray.

    The observation stores name their station coordinates differently from the
    forecast stores (``latitude`` vs ``station_latitude`` etc.); returning only
    the data variable avoids a silent coordinate conflict on concat.
    """
    ds = _open(data_path / f"stations_{kind}_observations_surface_{country}.zarr")
    da = ds.squeeze(drop=True).sel(step=lead).t2m
    return da.reset_coords(drop=True)


def _concat_countries(parts: list, dim: str = "station_id"):
    """Concatenate per-country objects, harmonizing the ``station_id`` dtype.

    The stores disagree: int32 for France/Germany, int64 for the rest. Without
    the cast, ``xr.concat`` promotes and silently reorders.
    """
    parts = [p.assign_coords(station_id=p.station_id.astype("int64")) for p in parts]
    return xr.concat(parts, dim=dim)


def _pseudo_time(times: np.ndarray, years: np.ndarray) -> pd.DatetimeIndex:
    """Map a reforecast ``(time, year)`` grid to one flat calendar axis.

    Each reference date is re-dated back by ``(21 - year) * 365`` days, giving
    the hindcast's own (approximate) calendar date. Returns the flattened
    ``(n_time * n_year,)`` axis in C order.
    """
    times = pd.to_datetime(times)
    offsets = pd.to_timedelta((YEAR_ORIGIN - years) * 365, unit="D")
    return pd.DatetimeIndex(
        (times.values[:, None] - offsets.values[None, :]).reshape(-1)
    )


def _stack_features(ds: xr.Dataset, sample_dims: list, n_members: int) -> np.ndarray:
    """Assemble the predictor tensor ``(S, F_raw, M, N)`` from a merged dataset.

    ``sample_dims`` names the dataset dims that make up the sample axis
    (``["time"]`` for forecasts, ``["time", "year"]`` for reforecasts); they are
    flattened in order. Station-level coordinates are broadcast over the sample
    and member axes so they appear as ordinary features, matching the reference
    implementation (which gets the same effect via ``to_dataframe``).
    """
    order = sample_dims + ["number", "station_id"]
    n_station = ds.sizes["station_id"]
    n_sample = int(np.prod([ds.sizes[d] for d in sample_dims]))

    out = np.empty((n_sample, len(FEATURE_NAMES) - len(CYCLIC_FEATURES), n_members, n_station),
                   dtype=np.float32)
    for i, name in enumerate(STATIC_FEATURES):
        # (N,) -> broadcast over samples and members.
        out[:, i] = ds[name].values.astype(np.float32)[None, None, :]
    for j, name in enumerate(DYNAMIC_FEATURES, start=len(STATIC_FEATURES)):
        arr = ds[name].transpose(*order).values.astype(np.float32)
        out[:, j] = arr.reshape(n_sample, n_members, n_station)
    return out


def _cyclic(times: pd.DatetimeIndex, n_members: int, n_station: int) -> np.ndarray:
    """``(S, 2, M, N)`` cosine/sine day-of-year features."""
    doy = 2 * np.pi * times.dayofyear.to_numpy(dtype=np.float32) / 365.0
    both = np.stack([np.cos(doy), np.sin(doy)], axis=1)  # (S, 2)
    return np.broadcast_to(both[:, :, None, None], (len(times), 2, n_members, n_station))


def _distance_matrix(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Pairwise geodesic distance in km on the WGS-84 ellipsoid.

    Uses ``geopy`` (as the reference does) rather than a haversine
    approximation: the great-circle formula is off by up to ~0.3% at these
    latitudes, which is enough to move edges across the ``max_dist`` threshold.
    """
    from geopy.distance import geodesic

    n = len(lat)
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = geodesic((lat[i], lon[i]), (lat[j], lon[j])).km
            dist[i, j] = dist[j, i] = d
    return dist


def build(data_path: Path, out_dir: Path, lead_time: str, min_obs_frac: float) -> None:
    lead = pd.Timedelta(lead_time)
    print(f"[INFO] lead time {lead_time} ({lead}), min_obs_frac {min_obs_frac}")

    # ---- load & concatenate -------------------------------------------------
    fc, rfc, obs_f, obs_rf = [], [], [], []
    for country in COUNTRIES:
        print(f"[INFO] loading {country}")
        fc.append(_load_family(data_path, "forecasts", country, lead))
        rfc.append(_load_family(data_path, "reforecasts", country, lead))
        obs_f.append(_load_obs(data_path, "forecasts", country, lead))
        obs_rf.append(_load_obs(data_path, "reforecasts", country, lead))

    forecasts = _concat_countries(fc)
    reforecasts = _concat_countries(rfc)
    targets_f = _concat_countries(obs_f)
    targets_rf = _concat_countries(obs_rf)

    if not np.array_equal(forecasts.station_id.values, reforecasts.station_id.values):
        raise RuntimeError("Station ids differ between forecasts and reforecasts.")
    if not np.array_equal(forecasts.station_id.values, targets_f.station_id.values):
        raise RuntimeError("Station ids differ between forecasts and observations.")

    n_station = forecasts.sizes["station_id"]
    print(f"[INFO] {n_station} stations, "
          f"{reforecasts.sizes['time']}x{reforecasts.sizes['year']} reforecast samples "
          f"({reforecasts.sizes['number']} members), "
          f"{forecasts.sizes['time']} forecast samples ({forecasts.sizes['number']} members)")

    # ---- reforecast sample axis --------------------------------------------
    pseudo = _pseudo_time(reforecasts.time.values, reforecasts.year.values)
    order = np.argsort(pseudo.values, kind="stable")
    pseudo = pseudo[order]

    # Targets in degrees Celsius, matching the reference and its tables.
    y_rf = (targets_rf.transpose("time", "year", "station_id").values
            .reshape(-1, n_station).astype(np.float32) - 273.15)[order]
    y_f = (targets_f.transpose("time", "station_id").values.astype(np.float32) - 273.15)

    times_f = pd.to_datetime(forecasts.time.values)
    train_sel = pseudo < TRAIN_END
    val_sel = (pseudo >= TRAIN_END) & (pseudo < VAL_END)

    # ---- station filter (decided on the training window only) ---------------
    observed = np.isfinite(y_rf[train_sel]).mean(axis=0)
    keep = observed >= min_obs_frac
    dropped = np.where(~keep)[0]
    names = np.array([str(s) for s in forecasts.station_name.values])
    if len(dropped):
        print(f"[INFO] dropping {len(dropped)} of {n_station} stations "
              f"(train-observed fraction < {min_obs_frac}):")
        for i in dropped[np.argsort(observed[dropped])]:
            print(f"         {forecasts.station_id.values[i]:>6}  {names[i]:<22} "
                  f"observed {100 * observed[i]:5.1f}%")
    n_keep = int(keep.sum())
    dropped_ids = forecasts.station_id.values[dropped]

    forecasts = forecasts.isel(station_id=keep)
    reforecasts = reforecasts.isel(station_id=keep)
    y_rf = y_rf[:, keep]
    y_f = y_f[:, keep]

    # ---- predictors ---------------------------------------------------------
    n_mem_rf = reforecasts.sizes["number"]
    n_mem_f = forecasts.sizes["number"]
    print("[INFO] stacking predictors")
    x_rf = _stack_features(reforecasts, ["time", "year"], n_mem_rf)[order]
    x_f = _stack_features(forecasts, ["time"], n_mem_f)

    if not np.isfinite(x_rf).all() or not np.isfinite(x_f).all():
        raise RuntimeError("Predictors contain non-finite values; expected none.")

    # Standardize with training statistics only, per feature, pooled over
    # samples/members/stations (the reference's StandardScaler convention).
    train_x = x_rf[train_sel]
    mean = train_x.mean(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    std = train_x.std(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    std[std == 0] = 1.0
    for arr in (x_rf, x_f):
        arr -= mean[None, :, None, None]
        arr /= std[None, :, None, None]

    def with_cyclic(x, times, n_members):
        return np.concatenate([x, _cyclic(times, n_members, n_keep)], axis=1)

    splits = {
        "train": (with_cyclic(x_rf[train_sel], pseudo[train_sel], n_mem_rf),
                  y_rf[train_sel], pseudo[train_sel]),
        "val": (with_cyclic(x_rf[val_sel], pseudo[val_sel], n_mem_rf),
                y_rf[val_sel], pseudo[val_sel]),
        "test": (with_cyclic(x_f, times_f, n_mem_f), y_f, times_f),
    }

    # ---- write --------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    split_times = {}
    for name, (x, y, times) in splits.items():
        nan = np.isnan(y).mean()
        print(f"[INFO] {name:<5} x{x.shape}  y{y.shape}  target NaN {100 * nan:.2f}%")
        # Plain .npy rather than a .npz archive: `np.load` cannot memory-map a
        # zip, and these arrays are large enough (the test split alone is
        # ~600 MB) that the loader wants to page them in rather than hold them.
        np.save(out_dir / f"{name}_x.npy", np.ascontiguousarray(x))
        np.save(out_dir / f"{name}_y.npy", np.ascontiguousarray(y[:, None, :]))  # (S, C=1, N)
        split_times[f"{name}_time"] = times.values.astype("datetime64[ns]")

    lat = forecasts.station_latitude.values.astype(np.float64)
    lon = forecasts.station_longitude.values.astype(np.float64)
    print("[INFO] computing geodesic distance matrix")
    np.savez(
        out_dir / "meta.npz",
        station_id=forecasts.station_id.values,
        station_name=names[keep],
        lat=lat,
        lon=lon,
        alt=forecasts.station_altitude.values.astype(np.float64),
        dist=_distance_matrix(lat, lon),
        feature_names=np.array(FEATURE_NAMES),
        mean=mean,
        std=std,
        dropped_station_id=dropped_ids,
        lead_time=np.array(lead_time),
        min_obs_frac=np.array(min_obs_frac),
        **split_times,
    )
    print(f"[INFO] wrote {out_dir} ({n_keep} stations, {len(FEATURE_NAMES)} features)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", type=Path,
                        default=Path("data/EUPPBench"),
                        help="Raw EUPPBench zarr stores (station subset).")
    parser.add_argument("--out_dir", type=Path, default=None,
                        help="Defaults to <repo>/data/euppbench_<lead_time>.")
    parser.add_argument("--lead_time", type=str, default="24h",
                        help="Any step present in the stores, e.g. 24h, 72h, 120h.")
    parser.add_argument("--min_obs_frac", type=float, default=0.5,
                        help="Drop stations observed less than this fraction of training days.")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if the output directory is already complete.")
    args = parser.parse_args()

    out_dir = args.out_dir or ROOT / "data" / f"euppbench_{args.lead_time}"
    expected = ["meta.npz"] + [f"{s}_{a}.npy" for s in ("train", "val", "test") for a in ("x", "y")]
    if not args.force and all((out_dir / f).exists() for f in expected):
        print(f"[INFO] {out_dir} already complete; use --force to rebuild.")
        return
    build(args.data_path, out_dir, args.lead_time, args.min_obs_frac)


if __name__ == "__main__":
    main()
