# Clean archived GMFD NetCDF files.
#
# Run on notebooks.cilresearch.org with container image pangeo/pangeo-notebook:2026.06.04.
# The raw GMFD files this script is parsing are available in our internal cloud storage.
# This is run on the cluster 1) in order to access the raw dataset; 2) Data regridding also
# uses a compiled library which can be difficult to install on some platforms, but is readily
# available on the cluster.

import datetime
import os
import uuid

import dask
from dask_gateway import GatewayCluster
from dotenv import load_dotenv
import xarray as xr
import xesmf as xe

load_dotenv()


OUT_ZARR = os.environ["POREALLAS_PARSED_GMFD_URI"]
START_YEAR = 1981
STOP_YEAR = 2010
TARGET_REGRID_URI = "s51_hcm.nc"
GMFD_FILE_PATTERN = (
    "/gcs/impactlab-data/climate/source_data/GMFD/tas/tas_0p25_daily_{year}-{year}.nc"
)
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


def open_gmfd(file_pattern: str, start_year: int, stop_year: int) -> xr.Dataset:
    """Open the GMFD dataset for a range of years"""
    # We have these stored for years 1950 - 2010. Pattern for file name:
    target_paths = [
        file_pattern.format(year=yr) for yr in range(start_year, stop_year + 1)
    ]
    gmfd = xr.open_mfdataset(target_paths)
    gmfd.attrs["source_uris"] = str(target_paths)
    return gmfd


dask.config.set({"distributed.comm.timeouts.connect": "60s"})
cluster = GatewayCluster(worker_image=JUPYTER_IMAGE, scheduler_image=JUPYTER_IMAGE)
client = cluster.get_client()
print(client.dashboard_link)
cluster.scale(50)

regrid_target = open_regrid_target(TARGET_REGRID_URI)

gmfd = open_gmfd(
    file_pattern=GMFD_FILE_PATTERN,
    start_year=START_YEAR,
    stop_year=STOP_YEAR,
)
# Cannot have leap years in QDM bias adjustment.
gmfd = gmfd.convert_calendar("noleap", dim="time")

regridder = xe.Regridder(gmfd, regrid_target, method="bilinear", periodic=True)
gmfd_regrid = regridder(gmfd)
gmfd_regrid.attrs |= gmfd.attrs

# Metadata on units is required later in the workflow.
gmfd_regrid["tas"].attrs["units"] = "K"

# Add additional general metadata.
gmfd_regrid.attrs |= {
    "poreallas_created_at": START_TIME,
    "poreallas_uid": UID,
    "poreallas_description": "Parsed GMFD climate fields",
}
gmfd_regrid["tas"].attrs |= {
    "poreallas_created_at": START_TIME,
    "poreallas_uid": UID,
    "poreallas_description": "Parsed GMFD tas field",
}

# All of time needs to be in a single chunk for QDM bias adjustment.
gmfd_regrid = gmfd_regrid.chunk({"time": -1, "latitude": 30, "longitude": "auto"})

gmfd_regrid.to_zarr(OUT_ZARR, consolidated=True)
print(f"Output written to {OUT_ZARR}")

cluster.scale(0)
cluster.shutdown()
