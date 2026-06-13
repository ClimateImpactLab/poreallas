# Running this on notebooks.cilresearch.org with pangeo/pangeo-notebook:2026.04.29
#
# This script loads and parses ERA5 data. It is run on the cluster
# because it loads from a petabyte-scale dataset co-located with this cluster.
# Data regridding also uses a compiled library which can be difficult to install on
# some platforms, but is readily available on the cluster.
#
# If you run this on a remote cluster you'll need to ensure the process has access to the
# required data in the ./data directory. Be sure to download the processed data
# to your ./data/parsed/ directory, also.

import datetime
import os
import uuid

import dask
from dotenv import load_dotenv
import xarray as xr
import xesmf as xe
from dask_gateway import GatewayCluster

load_dotenv()

START_YEAR = 1993  # Need at least 1993-2016 to match Copernicus CDS hindcast period for seasonal forecasts.
STOP_YEAR = 2025
TARGET_REGRID_URI = "s51_hcm.nc"
OUT_ZARR = os.environ["POREALLAS_ERA5_URI"]


JUPYTER_IMAGE = os.environ.get("JUPYTER_IMAGE")
UID = str(uuid.uuid4())
START_TIME = datetime.datetime.now(datetime.UTC).isoformat()

print(
    f"""
        {JUPYTER_IMAGE=}
        {START_TIME=}
        {UID=}
    """
)

dask.config.set({"distributed.comm.timeouts.connect": "60s"})
cluster = GatewayCluster(worker_image=JUPYTER_IMAGE, scheduler_image=JUPYTER_IMAGE)
client = cluster.get_client()
print(client.dashboard_link)
cluster.scale(50)

# Get ERA5 from Google
# https://console.cloud.google.com/marketplace/product/bigquery-public-data/arco-era5
# https://github.com/google-research/arco-era5/
ds = xr.open_zarr(
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
    chunks=None,
    storage_options=dict(token="anon"),
)
ar_full_37_1h = ds.sel(
    time=slice(ds.attrs["valid_time_start"], ds.attrs["valid_time_stop"])
)
# This is multiple TiB.

# Want climatology so last 30 years-ish.
# Chunk so doesn't read all data in at once.
clipped_window = (
    ar_full_37_1h["2m_temperature"]
    .sel(time=slice(str(START_YEAR), str(STOP_YEAR)))
    .chunk({"time": "auto", "latitude": -1, "longitude": -1})
)
# This is ~1 TiB.

annual_tas = (
    clipped_window.groupby("time.year")
    .mean("time")
    .to_dataset()
    .chunk({"year": "auto", "latitude": -1, "longitude": -1})
    .rename_vars({"2m_temperature": "tas"})
)


clipped_window_daily = clipped_window.resample(time="D").mean()
clipped_window_daily = clipped_window_daily.chunk(
    {"time": "auto", "latitude": -1, "longitude": -1}
)

# Using the S51 seasonal monthly seasonal hindcast ensemble mean from copernicus as the target grid for our regrid...
# Selecting so only have coords for latitude and longitude for regridding.
target = xr.open_dataset(TARGET_REGRID_URI).isel(
    {"forecast_reference_time": 0, "forecastMonth": 0}, drop=True
)
regridder = xe.Regridder(clipped_window_daily, target, method="bilinear", periodic=True)
clipped_window_daily_regrid = regridder(clipped_window_daily)

clipped_window_daily_regrid.name = "tas"
clipped_window_daily_regrid.attrs |= clipped_window_daily.attrs

# Seems to be an xarray bug? This only runs if we first compute() like this:
clipped_window_daily_regrid.to_dataset().chunk(
    {"time": "auto", "latitude": -1, "longitude": -1}
).compute().to_zarr(OUT_ZARR, consolidated=False)
print(f"Output written to {OUT_ZARR}")


cluster.scale(0)
cluster.shutdown()
