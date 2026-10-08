"""Download the ERA5 data of the surface-temperature task from WeatherBench2.

Selects one daily observation at 12 UTC, the European domain at 0.25 degrees
(35-75 N, 12.5 W-42.5 E, a 160 x 220 grid) and the seven input variables, and
writes ``<out>/era5.zarr`` plus per-variable statistics ``<out>/era5_statistics.npy``.
The training-period statistics used in the paper are computed afterwards with
``src/data/era5_statistics.py``.

    python src/data/download_era5.py --out data/era5/
"""

import argparse
import os

import gcsfs  # noqa: F401  (gs:// backend for xarray)
import numpy as np
import pandas as pd
import xarray as xr

def main():
    parser = argparse.ArgumentParser(description="Download the ERA5 subset from WeatherBench2.")
    parser.add_argument("--out", default="data/era5/", help="output directory")
    parser.add_argument("--first_year", default="1990")
    parser.add_argument("--last_year", default="2022")
    args = parser.parse_args()

    date_range = pd.date_range(f"{args.first_year}-01-01T12", f"{args.last_year}-12-31T12", freq="24h")
    path = os.path.join(args.out, "")
    os.makedirs(path, exist_ok=True)
    # Europe grid
    lat_range = np.arange(35, 75, 0.25)
    lon_range = np.append(np.arange(347.5, 360, 0.25), np.arange(0, 42.5, 0.25))

    era5_path = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"

    # Same order as src.datamodules.datasets.WB_INPUT: the statistics file is
    # applied to the input channels by position.
    variables = [
        "2m_temperature",
        "temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "geopotential",
        "land_sea_mask",
        "geopotential_at_surface",
    ]
    filename = "era5"

    # Open file from weatherbench and filter
    data = xr.open_zarr(era5_path)
    data_reduced = (
        data[variables]
        .sel(time=date_range, method="nearest")
        .sel(latitude=lat_range, longitude=lon_range, method="nearest")
    )
    data_reduced["geopotential"] = data_reduced["geopotential"].sel(level=500)
    data_reduced["temperature"] = data_reduced["temperature"].sel(level=850)
    # Rechunk and save
    data_reduced.chunk("auto").to_zarr(
        path + filename + ".zarr", zarr_format=2, consolidated=False
    )

    # Save statistics
    statistics = np.zeros((len(variables), 2))
    for i, var in enumerate(variables):
        if var == "land_sea_mask":
            statistics[i, 0] = 0
            statistics[i, 1] = 1
        else:
            statistics[i, 0] = data_reduced[var].mean().compute()
            statistics[i, 1] = data_reduced[var].std().compute()

    np.save(path + "era5_statistics.npy", statistics)

    data_reduced.close()
    data.close()


if __name__ == "__main__":
    main()