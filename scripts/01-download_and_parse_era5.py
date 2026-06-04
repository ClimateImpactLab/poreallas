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
import xarray as xr
import xesmf as xe
from dask_gateway import GatewayCluster

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

# Want climatology so last 30 years.
# Chunk so doesn't read all data in at once.
clipped_window = (
    ar_full_37_1h["2m_temperature"]
    .sel(time=slice("1995", "2025"))
    .chunk({"time": "auto", "latitude": -1, "longitude": -1})
)
# This is ~1 TiB.

# Monthly climatology for the study period. Used to adjust S51 forecasts.
monthly_climatology = (
    clipped_window.groupby("time.month")
    .mean(dim="time")
    .chunk({"month": "auto", "latitude": -1, "longitude": -1})
    .rename({"2m_temperature": "tas"})
    .compute()
)

# What if we only used a single month?
# clipped_window.sel(time=clipped_window["time"].dt.month == 6)
# This gets us to ~93 GiB.


annual_tas = (
    clipped_window.groupby("time.year")
    .mean("time")
    .to_dataset()
    .chunk({"year": "auto", "latitude": -1, "longitude": -1})
)
annual_tas = annual_tas.rename(
    {"2m_temperature": "tas"}
).compute()  # For some reason we need to compute here, otherwise this version of xarray fails to write to file.
# annual_tas.to_netcdf("era5_tas_1995_2025.nc", mode="w")

# If we were to regrid the above ERA5 data...
# annual_tas = xr.open_dataset("era5_tas_1995_2025.nc")

# Using the S51 seasonal monthly seasonal hindcast ensemble mean from copernicus as the target grid for our regrid...
# Selecting so only have coords for latitude and longitude for regridding.
target = xr.open_dataset("s51_hcm.nc").isel(
    {"forecast_reference_time": 0, "forecastMonth": 0}, drop=True
)
regridder = xe.Regridder(annual_tas, target, method="bilinear", periodic=True)
annual_tas_regrid = regridder(annual_tas)
monthly_climatology_regrid = regridder(monthly_climatology)

annual_tas_regrid.to_netcdf("era5_annual_tas_1995_2025_regrid.nc", mode="w")
monthly_climatology_regrid.to_netcdf(
    "era5_monthly_tas_climatology_1995_2025_regrid.nc", mode="w"
)

cluster.scale(0)
cluster.shutdown()
