"""
Estimate forecast ensemble daily-average air temperature from daily 24-hour
maximum and minimum air temperature and bias-adjust by comparing monthly
hindcast and reanalysis.

Write output to POREALLAS_TAS_FORECAST_URI
"""

import datetime
import os
import uuid

from dotenv import load_dotenv
import xarray as xr


load_dotenv()

OUT_ZARR = os.environ["POREALLAS_TAS_FORECAST_URI"]
ERA5_URI = os.environ["POREALLAS_ERA5_URI"]
S51_TASMAX_URI = "./data/raw/s51_tasmax.nc"
S51_TASMIN_URI = "./data/raw/s51_tasmin.nc"
HC_URI = "./data/raw/s51_hcm.nc"
# It would be great if we could grab this from CDS/ECMWF file metadata but we can't, so be careful.
HINDCAST_PERIOD = slice("1993", "2016")
UID = str(uuid.uuid4())
DATETIME_NOW = datetime.datetime.now(datetime.timezone.utc).isoformat()


reanalysis = xr.open_dataset(ERA5_URI)

hc = xr.open_dataset(HC_URI)


# Estimate forecast daily tas from daily tasmax and tasmin average.
ds_tas = xr.merge(
    [
        xr.open_dataset(S51_TASMAX_URI),
        xr.open_dataset(S51_TASMIN_URI),
    ],
    compat="no_conflicts",
)
# Estimate daily tas from daily tasmax and daily tasmin.
ds_tas["tas"] = (ds_tas["mx2t24"] + ds_tas["mn2t24"]) / 2


# Add a new month variable to represent the calendar month for each valid period of the forecast.
# Squeeze it so it only has the needed "ForecastMonth" dim so we can swap them out in place.
# Need this "month" so can align with reanalysis to estimate forecast bias.
hc["month"] = (
    hc["forecast_reference_time"].dt.month + hc["forecastMonth"]
).squeeze() - 1
hc = hc.swap_dims({"forecastMonth": "month"})


# Estimate forecast monthly bias: difference between monthly hindcasts and
# average of reanalysis for all 12 months in forecast's handcast period.
months_climatology = reanalysis.sel(time=HINDCAST_PERIOD).groupby("time.month").mean()
bias = (hc["t2m"] - months_climatology["tas"]).squeeze()


# Make "valid_time" the "time" dim and main time dim rather than
# "forecast_period", dropping everything else we don't need.
ds_tas = ds_tas.set_coords("valid_time").rename({"valid_time": "time"})
ds_tas = ds_tas.swap_dims({"forecast_period": "time"}).squeeze(drop=True)

# Only use months from the forecast-ensemble that are bias-adjustable. If a month is not here, it's likely incomplete.
ds_tas = ds_tas.where(ds_tas["time.month"].isin(bias["month"]), drop=True)
# Apply monthly bias adjustment to forecast ensemble.
adjusted = ds_tas["tas"] - bias.sel(month=ds_tas["time.month"])

# Each month also uniquely corresponds to a forecast lead time, so the bias
# adjustment accounts for the combined bias from the forecast ensemble systematic bias in simulating
# 1) a particular month of a year
# 2) lead time from when the forecast was initialized -- or, how long the model has been running

forecast_ensemble = xr.Dataset({"tas": adjusted})

forecast_ensemble.attrs |= {
    "poreallas_created_at": DATETIME_NOW,
    "poreallas_uid": UID,
    "poreallas_description": "Bias-adjusted forecast ensemble daily air temperature",
    "poreallas_reanalysis_uri": ERA5_URI,
    "poreallas_forecast_tasmax_uri": S51_TASMAX_URI,
    "poreallas_forecast_tasmin_uri": S51_TASMIN_URI,
    "poreallas_hindcast_uri": HC_URI,
}

forecast_ensemble = forecast_ensemble.chunk("auto")

forecast_ensemble.to_zarr(OUT_ZARR, consolidated=False)
print(f"s51 data written to {OUT_ZARR}")
