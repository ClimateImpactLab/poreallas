# Parse ERA5 data store to prepare for analysis.
#
# Run on notebooks.cilresearch.org with container image pangeo/pangeo-notebook:2026.06.04.
#
# This script loads and parses ERA5 data. It is run on the cluster
# because it loads from a petabyte-scale dataset co-located with this cluster.
# Data regridding also uses a compiled library which can be difficult to install on
# some platforms, but is readily available on the cluster.
#
# More information on the ERA5 data store hosted on GCP:
# https://console.cloud.google.com/marketplace/product/bigquery-public-data/arco-era5
# https://github.com/google-research/arco-era5/

import datetime
import os
import uuid

import dask
from dask_gateway import GatewayCluster  # type: ignore[ty:unresolved-import]
from dotenv import load_dotenv
import xarray as xr
import xesmf as xe  # type: ignore[ty:unresolved-import]

load_dotenv()

OUT_ZARR = os.environ["POREALLAS_PARSED_ERA5_URI"]
START_YEAR = 1981
STOP_YEAR = 2025
TARGET_REGRID_URI = "s51_hcm.nc"
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


def open_regrid_target(uri: str) -> xr.Dataset:
    """Open/clean a dataset to use as a regridding target"""
    # Using the S51 seasonal monthly seasonal hindcast ensemble mean from copernicus as the target grid for our regrid...
    # Selecting so only have coords for latitude and longitude for regridding.
    target = xr.open_dataset(uri).isel(
        {"forecast_reference_time": 0, "forecastMonth": 0}, drop=True
    )
    return target


def open_era5(
    start_year: int | str,
    stop_year: int | str,
    uri="gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
) -> xr.Dataset:
    """Opens and parses Googe's ARCO ERA5 store, returning rechunked daily tas dataset

    This can be very heavy and data-IO intensive.
    """
    ds = xr.open_zarr(
        uri,
        chunks=None,
        storage_options=dict(token="anon"),
    )

    # Grab only valid periods
    ar_full_37_1h = ds.sel(
        time=slice(ds.attrs["valid_time_start"], ds.attrs["valid_time_stop"])
    )

    # This is huge so only get what we need. It also needs to be chunked so
    # it isn't read all into memory at once.
    clipped_window = (
        ar_full_37_1h["2m_temperature"]
        .sel(time=slice(str(start_year), str(stop_year)))
        .chunk({"time": "auto", "latitude": -1, "longitude": -1})
    )

    # Collect the subdaily data into daily means and rechunk again.
    daily = clipped_window.resample(time="D").mean()
    clipped_window_daily = daily.chunk(
        {"time": "auto", "latitude": -1, "longitude": -1}
    )

    # We made it a DataArray but let's make it "tas" in a Dataset.
    clipped_window_daily.name = "tas"
    out_ds = clipped_window_daily.to_dataset()

    # Add metadata from the full-sized data.
    out_ds.attrs |= ds.attrs
    return out_ds


dask.config.set({"distributed.comm.timeouts.connect": "60s"})
cluster = GatewayCluster(worker_image=JUPYTER_IMAGE, scheduler_image=JUPYTER_IMAGE)
client = cluster.get_client()
print(client.dashboard_link)
cluster.scale(50)

regrid_target = open_regrid_target(TARGET_REGRID_URI)

era5 = open_era5(
    start_year=START_YEAR,
    stop_year=STOP_YEAR,
)

# Cannot have leap years in QDM bias adjustment so convert to a no-leapyear calendar.
era5 = era5.convert_calendar("noleap", dim="time")

regridder = xe.Regridder(era5, regrid_target, method="bilinear", periodic=True)
era5_regrid = regridder(era5)
era5_regrid.attrs |= era5.attrs

# Metadata on units is required later in the workflow.
era5_regrid["tas"].attrs["units"] = "K"

# Add additional general metadata.
era5_regrid.attrs |= {
    "poreallas_created_at": START_TIME,
    "poreallas_uid": UID,
    "poreallas_description": "Parsed ERA5 climate fields",
}
era5_regrid["tas"].attrs |= {
    "poreallas_created_at": START_TIME,
    "poreallas_uid": UID,
    "poreallas_description": "Parsed ERA5 tas field",
}

# All of time needs to be in a single chunk for QDM bias adjustment.
# This generally gets you ~110 MiB chunks.
era5_regrid = era5_regrid.chunk({"time": -1, "latitude": 30, "longitude": 60})

era5_regrid.to_zarr(OUT_ZARR, consolidated=True)
print(f"Output written to {OUT_ZARR}")

cluster.scale(0)
cluster.shutdown()
