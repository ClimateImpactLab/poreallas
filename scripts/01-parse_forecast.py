# Parse ECMWF S51 forcast/hindcast files to create daily temperature data ready
# for the analysis workflow.

import datetime
import os
import uuid

from dotenv import load_dotenv
import xarray as xr


load_dotenv()

OUT_ZARR = os.environ["POREALLAS_PARSED_FORECAST_URI"]
RAW_FORECAST_FILE_PATTERN = "./data/raw/s51_hist_tasmin_tasmax/{var}-{year}-05.nc"
START_YEAR = 1981
STOP_YEAR = 2026
UID = str(uuid.uuid4())
START_TIME = datetime.datetime.now(datetime.UTC).isoformat()

print(
    f"""
        {START_TIME=}
        {UID=}
    """
)

# # We are intentionally merging hindcast and forecast datasets even though they
# have different ensemble sizes (the "number" dim). The hindcast gets stuffed
# with NaNs. This is needed when training QDM on the hindcast data and applying
# to forecast data, otherwise the forecast ensemble size gets cut to the hindcast
# size.
target_tasmax_paths = [
    RAW_FORECAST_FILE_PATTERN.format(year=yr, var="tasmax")
    for yr in range(START_YEAR, STOP_YEAR + 1)
]
target_tasmin_paths = [
    RAW_FORECAST_FILE_PATTERN.format(year=yr, var="tasmin")
    for yr in range(START_YEAR, STOP_YEAR + 1)
]

mfdataset_kwargs = dict(
    join="outer",
    coords="different",
    compat="no_conflicts",
)
s51 = (
    xr.open_mfdataset(target_tasmax_paths, **mfdataset_kwargs)["mx2t24"]
    + xr.open_mfdataset(target_tasmin_paths, **mfdataset_kwargs)["mn2t24"]
) / 2
s51.name = "tas"
s51 = s51.to_dataset()


# Make "valid_time" the "time" dim and main time dim and reducing it to a single
# dimension so xsdba's QDM can run this against the `ref` data. Version of
# xsdba we're running with seems to require a "time" dim to train QDM. So, this
# *needs* to be named "time".
s51 = s51.set_coords("valid_time").rename({"valid_time": "time"})
# Collapse multidimension time-like dims into to one dimension we can swap in "time" for.
s51 = (
    s51.stack({"_time_placeholder": ["forecast_reference_time", "forecast_period"]})
    .swap_dims({"_time_placeholder": "time"})
    .squeeze(drop=True)
    .drop_vars("_time_placeholder")
)

# Need matching calendars. Removing leap days makes QDM easier.
s51 = s51.convert_calendar("noleap", dim="time")

# Add additional general metadata.
s51.attrs |= {
    "poreallas_created_at": START_TIME,
    "poreallas_uid": UID,
    "poreallas_description": "Parsed ECMWF S51 ensemble fields",
}
s51["tas"].attrs |= {
    "poreallas_created_at": START_TIME,
    "poreallas_uid": UID,
    "poreallas_description": "Parsed ECMWF S51 ensemble tas fields",
}

# Rechunking because all of "time", or whatever we're grouping QDM on, needs to be in one chunk.
s51 = s51.chunk({"number": -1, "time": -1, "latitude": 30, "longitude": "auto"})

s51.to_zarr(OUT_ZARR, consolidated=True)
print(f"Output written to {OUT_ZARR}")
