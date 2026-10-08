"""Per-variable normalization statistics for the ERA5 (WeatherBench) store.

Writes an ``(len(WB_INPUT), 2)`` array of (mean, std) in the order of
``src.datamodules.datasets.WB_INPUT`` -- the order in which ``WeatherBench``
applies them -- computed over the training years only, so the val/test years
never enter the normalization. The land-sea mask is left at (0, 1).

Note that the ``era5_statistics.npy`` written by ``src/data/download_era5.py``
uses the download script's variable order (2m_temperature, 10m_u, 10m_v,
temperature, ...), which is not ``WB_INPUT``'s, and covers the whole 1990-2022
period; point the ``statistics_path`` option of ``configs/datamodule/era5.yaml``
at the file written here instead.

Run (CPU, a few minutes, ~8 GB of dask working memory):
    python src/data/era5_statistics.py --data_path data/era5/ \
        --years 1990 2014 --out era5_statistics_train1990-2014.npy
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr

for _parent in Path(__file__).resolve().parents:
    if (_parent / ".project-root").exists():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break

from src.datamodules.datasets import WB_INPUT  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data_path", default="data/era5/")
    ap.add_argument("--years", nargs=2, default=["1990", "2014"], metavar=("FIRST", "LAST"))
    ap.add_argument("--out", default=None, help="output .npy (default: <data_path>/era5_statistics_train<FIRST>-<LAST>.npy)")
    args = ap.parse_args()

    ds = xr.open_zarr(os.path.join(args.data_path, "era5.zarr"), consolidated=False)[WB_INPUT]
    train = ds.sel(time=slice(args.years[0], args.years[1]))
    stats = np.zeros((len(WB_INPUT), 2))
    for i, var in enumerate(WB_INPUT):
        if var == "land_sea_mask":
            stats[i] = [0.0, 1.0]
            continue
        a = train[var].astype("float64")
        stats[i, 0] = float(a.mean().compute())
        stats[i, 1] = float(a.std().compute())
        print(f"{var:28s} mean={stats[i, 0]:.6g} std={stats[i, 1]:.6g}", flush=True)

    out = args.out or os.path.join(
        args.data_path, f"era5_statistics_train{args.years[0]}-{args.years[1]}.npy"
    )
    np.save(out, stats)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
