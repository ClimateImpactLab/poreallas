"""
Project mortality effects for forecast ensemble and baseline period
"""

import datetime
import os
import uuid

from dotenv import load_dotenv
import isku
import numpy as np
import xarray as xr

from poreallas.extract import make_climtas, make_tas_monthly_histogram
from poreallas.project import mortality_effect_model, calculate_beta


load_dotenv()


TAS_FORECAST_URI = os.environ["POREALLAS_TAS_FORECAST_URI"]
ERA5_URI = os.environ["POREALLAS_ERA5_URI"]
GAMMA_URI = os.environ["POREALLAS_GAMMA_URI"]
SOCIOECONOMICS_URI = os.environ["POREALLAS_SOCIOECONOMICS_URI"]
REGIONS_URI = os.environ["POREALLAS_REGIONS_URI"]

# Output
EFFECTS_URI = os.getenv("POREALLAS_EFFECTS_URI")


def trim_ragged_months(
    ds: xr.Dataset, *, required_proportion: float = 0.9, datetime_dim="time"
) -> xr.Dataset:
    """
    Drop months without required number of days, returning a copy of input dataset.

    Forecast ensemble is for a fixed number of days so we expect to usually
    trim off the last month of the forecast if it is ragged and missing days beyond
    a threshold.
    """
    ds = ds.copy()

    n_initial = ds[datetime_dim].size
    number_obs = ds[datetime_dim].resample(time="ME").count()
    days_in_month = number_obs[datetime_dim].dt.days_in_month
    min_req = np.round(days_in_month * required_proportion)
    qualifying_months = number_obs.where(number_obs >= min_req, drop=True)[
        datetime_dim
    ].dt.month
    ds = ds.where(ds[datetime_dim].dt.month.isin(qualifying_months), drop=True)

    n_current = ds[datetime_dim].size
    n_initial_months = number_obs[datetime_dim].size
    n_qualifying_months = qualifying_months[datetime_dim].size

    print(
        f"continuing with {n_qualifying_months} of {n_initial_months} forecast months after removing incomplete months"
    )
    print(
        f"continuing with {n_current} of {n_initial} forecast periods after removing incomplete months"
    )

    assert (n_qualifying_months - n_initial_months) < 2, (
        "More than one incomplete month was removed from the forecast while checking for incomplete months. Something unexpected is happening."
    )
    return ds


def read_regions(uri: str) -> isku.GridWeightingRegions:
    _region_weights = xr.load_dataset(uri)[
        ["lat", "lon", "region", "weight"]
    ]  # Load only what we need.
    # Apparently in this version of xarray the `.load()` method type-hints it'll return a DataArray instead of a Dataset.
    # It is a Dataset (I checked). So telling ty to ignore it.
    # # TODO: send bug upstream?
    regions = isku.GridWeightingRegions(_region_weights)  # ty: ignore[invalid-argument-type]
    return regions


def main():
    reanalysis = xr.load_dataset(ERA5_URI)
    forecast_ensemble = xr.load_dataset(TAS_FORECAST_URI)
    regions = read_regions(REGIONS_URI)
    socioeconomics = xr.load_dataset(SOCIOECONOMICS_URI)
    gammas = xr.load_dataset(GAMMA_URI)

    # Last months of forecast often missing significant number of days. Remove these ragged months.
    forecast_ensemble = trim_ragged_months(forecast_ensemble)

    # Transform gridded data, extracting regional data needed for projections.
    histogram_hist_tas = isku.extract_regions(
        reanalysis,
        template=make_tas_monthly_histogram,
        regions=regions,
    )
    histogram_forecast_tas = isku.extract_regions(
        forecast_ensemble,
        template=make_tas_monthly_histogram,
        regions=regions,
    )
    # Using the same static beta for forecast and reanalysis projection requires the histogram tas_bin for these data need to be equal, too. So we're  calculating it here.
    xr.testing.assert_allclose(
        histogram_hist_tas["tas_bin"],
        histogram_forecast_tas["tas_bin"],
    )

    climtas = isku.extract_regions(
        reanalysis,
        template=make_climtas,
        regions=regions,
    ).sel(year=2025, drop=True)

    # log(GDPpc + 1), the + 1 prevents warning about undefined behavior when log is fed a GDPpc of 0.
    loggdppc = np.log(socioeconomics["gdppc"].sel(year=2023, drop=True) + 1)

    # Calculate a fixed response function, i.e. beta.
    # Single, static response function with no adaptation is used for both projections.
    # Stick everything together and make sure it aligns and matches. Rechunk all together. Also drop any regions with NaNs.
    beta_input = (
        xr.Dataset(
            {
                "tas_bin": histogram_forecast_tas["tas_bin"],
                "climtas": climtas["climtas"],
                "loggdppc": loggdppc,
                "gamma": gammas["gamma_sampled"],
            }
        )
        .dropna(dim="region")
        .chunk(
            {
                "region": "auto",  # "auto" is a sensible default.
                "tas_bin": -1,  # This also needs to be all in memory.
                "age_cohort": 1,  # We're doing all age_cohorts at once but could be done one-by-one.
                "degree": -1,  # For gammas and polynomial calculations. Should all be in memory.
                "sample": 1,  # Dimension for gamma draws. Having them in memory or not is not strictly required.
            },
        )
        .unify_chunks()
    )
    fixed_beta = calculate_beta(beta_input).astype("float32").compute()
    fixed_beta["beta"].attrs = {
        "units": "deaths per 100,000 people",
        "long_name": "Temperature mortality rate",
    }
    # Do beta allowing only hot deaths by 0-ing out everthing on the cold side of the minimum-mortality temperature.
    fixed_beta["beta_hotonly"] = fixed_beta["beta"].where(
        fixed_beta["tas_bin"] > fixed_beta["mmt"], other=0
    )
    fixed_beta["beta_hotonly"].attrs["long_name"] = "Hot temperature mortality rate"

    # Do beta, allowing only COLD deaths by 0-ing out everything on the warm side of the minimum-mortality temperature.
    fixed_beta["beta_coldonly"] = fixed_beta["beta"].where(
        fixed_beta["tas_bin"] < fixed_beta["mmt"], other=0
    )
    fixed_beta["beta_coldonly"].attrs["long_name"] = "Cold temperature mortality rate"

    # Project mortality.
    # Start with forecast ensemble.
    forecast_input = (
        xr.Dataset(
            {
                "histogram_tas": histogram_forecast_tas["histogram_tas"],
                "beta": fixed_beta["beta"],
            }
        )
        .dropna(dim="region")
        .chunk(
            {
                "region": "auto",  # "auto" is a sensible default.
                "time": -1,
                "tas_bin": -1,
                "age_cohort": 1,
                "number": 1,
                "sample": 1,
            },
        )
        .unify_chunks()
    )
    projected_forecast = isku.project(
        forecast_input, model=mortality_effect_model
    ).compute()
    projected_forecast["effect"].attrs = {
        "units": "deaths per 100,000 people",
        "long_name": "Temperature mortality",
    }

    # Now hot-only projection
    forecast_input = (
        xr.Dataset(
            {
                "histogram_tas": histogram_forecast_tas["histogram_tas"],
                "beta": fixed_beta["beta_hotonly"],
            }
        )
        .dropna(dim="region")
        .chunk(
            {
                "region": "auto",  # "auto" is a sensible default.
                "time": -1,
                "tas_bin": -1,
                "age_cohort": 1,
                "number": 1,
            },
        )
        .unify_chunks()
    )
    projected_forecast_hotonly = isku.project(
        forecast_input, model=mortality_effect_model
    ).compute()
    projected_forecast_hotonly["effect"].attrs = {
        "units": "deaths per 100,000 people",
        "long_name": "Hot temperature mortality",
    }

    # Now cold-only projection
    forecast_input = (
        xr.Dataset(
            {
                "histogram_tas": histogram_forecast_tas["histogram_tas"],
                "beta": fixed_beta["beta_coldonly"],
            }
        )
        .dropna(dim="region")
        .chunk(
            {
                "region": "auto",  # "auto" is a sensible default.
                "time": -1,
                "tas_bin": -1,
                "age_cohort": 1,
                "number": 1,
            },
        )
        .unify_chunks()
    )
    projected_forecast_coldonly = isku.project(
        forecast_input, model=mortality_effect_model
    ).compute()
    projected_forecast_coldonly["effect"].attrs = {
        "units": "deaths per 100,000 people",
        "long_name": "Cold temperature mortality",
    }

    # Now do the baseline period.
    # Stick everything together and make sure it aligns and matches. Rechunk all together. Also drop any regions with NaNs.
    hist_input = (
        xr.Dataset(
            {
                "histogram_tas": histogram_hist_tas["histogram_tas"],
                "beta": fixed_beta["beta"],
            }
        )
        .dropna(dim="region")
        .chunk(
            {
                "region": "auto",  # "auto" is a sensible default.
                "time": -1,
                "tas_bin": -1,
                "age_cohort": 1,
            },
        )
        .unify_chunks()
    )
    projected_hist = isku.project(hist_input, model=mortality_effect_model).compute()
    projected_hist["effect"].attrs = {
        "units": "deaths per 100,000 people",
        "long_name": "Temperature mortality",
    }
    # Now hot-only projection
    hist_input = (
        xr.Dataset(
            {
                "histogram_tas": histogram_hist_tas["histogram_tas"],
                "beta": fixed_beta["beta_hotonly"],
            }
        )
        .dropna(dim="region")
        .chunk(
            {
                "region": "auto",  # "auto" is a sensible default.
                "time": -1,
                "tas_bin": -1,
                "age_cohort": 1,
            },
        )
        .unify_chunks()
    )
    projected_hist_hotonly = isku.project(
        hist_input, model=mortality_effect_model
    ).compute()
    projected_hist_hotonly["effect"].attrs = {
        "units": "deaths per 100,000 people",
        "long_name": "Hot temperature mortality",
    }

    # Now cold-only projection
    hist_input = (
        xr.Dataset(
            {
                "histogram_tas": histogram_hist_tas["histogram_tas"],
                "beta": fixed_beta["beta_coldonly"],
            }
        )
        .dropna(dim="region")
        .chunk(
            {
                "region": "auto",  # "auto" is a sensible default.
                "time": -1,
                "tas_bin": -1,
                "age_cohort": 1,
            },
        )
        .unify_chunks()
    )
    projected_hist_coldonly = isku.project(
        hist_input, model=mortality_effect_model
    ).compute()
    projected_hist_coldonly["effect"].attrs = {
        "units": "deaths per 100,000 people",
        "long_name": "Cold temperature mortality",
    }

    # Collect everything and write to storage.
    _out = {
        "forecast": projected_forecast,
        "baseline": projected_hist,
        "forecast_hotonly": projected_forecast_hotonly,
        "baseline_hotonly": projected_hist_hotonly,
        "forecast_coldonly": projected_forecast_coldonly,
        "baseline_coldonly": projected_hist_coldonly,
    }
    _out_dt = xr.DataTree.from_dict(_out)

    # Add metadata
    _uid = str(uuid.uuid4())
    _datetime_now = datetime.datetime.now(datetime.timezone.utc).isoformat()

    _out_dt.attrs |= {
        "poreallas_created_at": _datetime_now,
        "poreallas_uid": _uid,
        "poreallas_description": "Projected temperature mortality effects",
    }

    _out_dt["forecast"].attrs |= {
        "poreallas_created_at": _datetime_now,
        "poreallas_uid": _uid,
        "poreallas_description": "Forecast ensemble projected temperature mortality effects",
        "poreallas_temperature_uri": TAS_FORECAST_URI,
        "poreallas_socioeconomics_uri": SOCIOECONOMICS_URI,
        "poreallas_model_parameters_uri": GAMMA_URI,
        "poreallas_regions_uri": REGIONS_URI,
    }

    _out_dt["baseline"].attrs |= {
        "poreallas_created_at": _datetime_now,
        "poreallas_uid": _uid,
        "poreallas_description": "Baseline projected temperature mortality effects",
        "poreallas_temperature_uri": ERA5_URI,
        "poreallas_socioeconomics_uri": SOCIOECONOMICS_URI,
        "poreallas_model_parameters_uri": GAMMA_URI,
        "poreallas_regions_uri": REGIONS_URI,
    }

    _out_dt["forecast_hotonly"].attrs |= {
        "poreallas_created_at": _datetime_now,
        "poreallas_uid": _uid,
        "poreallas_description": "Forecast ensemble projected hot temperature mortality effects",
        "poreallas_temperature_uri": TAS_FORECAST_URI,
        "poreallas_socioeconomics_uri": SOCIOECONOMICS_URI,
        "poreallas_model_parameters_uri": GAMMA_URI,
        "poreallas_regions_uri": REGIONS_URI,
    }

    _out_dt["baseline_hotonly"].attrs |= {
        "poreallas_created_at": _datetime_now,
        "poreallas_uid": _uid,
        "poreallas_description": "Baseline projected hot temperature mortality effects",
        "poreallas_temperature_uri": ERA5_URI,
        "poreallas_socioeconomics_uri": SOCIOECONOMICS_URI,
        "poreallas_model_parameters_uri": GAMMA_URI,
        "poreallas_regions_uri": REGIONS_URI,
    }

    _out_dt["forecast_coldonly"].attrs |= {
        "poreallas_created_at": _datetime_now,
        "poreallas_uid": _uid,
        "poreallas_description": "Forecast ensemble projected cold temperature mortality effects",
        "poreallas_temperature_uri": TAS_FORECAST_URI,
        "poreallas_socioeconomics_uri": SOCIOECONOMICS_URI,
        "poreallas_model_parameters_uri": GAMMA_URI,
        "poreallas_regions_uri": REGIONS_URI,
    }

    _out_dt["baseline_coldonly"].attrs |= {
        "poreallas_created_at": _datetime_now,
        "poreallas_uid": _uid,
        "poreallas_description": "Baseline projected cold temperature mortality effects",
        "poreallas_temperature_uri": ERA5_URI,
        "poreallas_socioeconomics_uri": SOCIOECONOMICS_URI,
        "poreallas_model_parameters_uri": GAMMA_URI,
        "poreallas_regions_uri": REGIONS_URI,
    }

    if EFFECTS_URI is not None:
        _out_dt.to_zarr(EFFECTS_URI, consolidated=True)
        print(f"Effects written to {EFFECTS_URI}")


if __name__ == "__main__":
    main()
