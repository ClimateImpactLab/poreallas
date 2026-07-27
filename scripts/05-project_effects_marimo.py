import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(
        """
        # Project mortality effects

        Project mortality effects for forecast ensemble and baseline period.
        """
    )
    return


@app.cell
def _():
    import datetime
    import os
    import uuid

    from dotenv import load_dotenv
    import isku
    import numpy as np
    import xarray as xr

    from poreallas.extract import make_climtas, make_tas_monthly_histogram
    from poreallas.project import calculate_beta, mortality_effect_model

    load_dotenv()
    return (
        calculate_beta,
        datetime,
        isku,
        make_climtas,
        make_tas_monthly_histogram,
        mortality_effect_model,
        np,
        os,
        uuid,
        xr,
    )


@app.cell
def _(os):
    TAS_FORECAST_URI = os.environ["POREALLAS_TAS_FORECAST_URI"]
    ERA5_URI = os.environ["POREALLAS_ERA5_URI"]
    GAMMA_URI = os.environ["POREALLAS_GAMMA_URI"]
    SOCIOECONOMICS_URI = os.environ["POREALLAS_SOCIOECONOMICS_URI"]
    REGIONS_URI = os.environ["POREALLAS_REGIONS_URI"]
    EFFECTS_URI = "gs://poreallas-public-20260605/v20260702/effects_net_hot_cold.zarr"#os.getenv("POREALLAS_EFFECTS_URI")
    return (
        EFFECTS_URI,
        ERA5_URI,
        GAMMA_URI,
        REGIONS_URI,
        SOCIOECONOMICS_URI,
        TAS_FORECAST_URI,
    )


@app.cell
def _(TAS_FORECAST_URI):
    TAS_FORECAST_URI
    return


@app.cell
def _(isku, np, xr):
    # def make_daily_tas_avg(ds: xr.Dataset) -> xr.Dataset:
    #     _tas = xr.DataArray(units.convert_units_to(ds["tas"], "degC"))

    #     _tas_daily_avg = _tas.groupby("time.day").mean(dim="time")

    #     return _tas_daily_avg.to_dataset().astype("float32")


    def read_reanalysis(uri: str) -> xr.Dataset:
        _ds = xr.load_dataset(uri, storage_options={"token": "anon"},)

        # Clean up longitude. The data goes from longitude 0 to 360. It needs to go -180 to 180 in ascending order.
        _ds["longitude"] = (_ds["longitude"] + 180) % 360 - 180
        _ds = _ds.sortby("longitude")
        _ds = _ds.rename({"longitude": "lon", "latitude": "lat"})
        _ds = _ds.chunk("auto")

        return _ds

    def read_forecast_ensemble(uri: str) -> xr.Dataset:
        _ds = xr.load_dataset(uri, storage_options={"token": "anon"},)

        # Clean up longitude. The data goes from longitude 0 to 360. It needs to go -180 to 180 in ascending order.
        _ds["longitude"] = (_ds["longitude"] + 180) % 360 - 180
        _ds = _ds.sortby("longitude")
        _ds = _ds.rename({"latitude": "lat", "longitude": "lon"})
        _ds = _ds.chunk("auto")

        # TODO: We prob don't want this here. Should be in earlier cleaning. Here for backwards compatibility.
        # Drop months without required number of obs. Forecast ensemble is for a fixed number of days so we expect to usually trim off the last month of the forecast if it is ragged and missing days beyond a threshold.
        _dt_dim = "time"
        _n_initial = _ds[_dt_dim].size
        _number_obs = _ds[_dt_dim].resample(time="ME").count()
        _days_in_month = _number_obs[_dt_dim].dt.days_in_month
        required_percent = 0.9
        _min_req = np.round(_days_in_month * required_percent)
        _qualifying_months = _number_obs.where(_number_obs >= _min_req, drop=True)[
            "time"
        ].dt.month
        _ds = _ds.where(_ds[_dt_dim].dt.month.isin(_qualifying_months), drop=True)

        _n_current = _ds[_dt_dim].size
        _n_initial_months = _number_obs[_dt_dim].size
        _n_qualifying_months = _qualifying_months["time"].size

        print(
            f"continuing with {_n_qualifying_months} of {_n_initial_months} forecast months after removing incomplete months"
        )
        print(
            f"continuing with {_n_current} of {_n_initial} forecast periods after removing incomplete months"
        )

        assert (_n_qualifying_months - _n_initial_months) < 2, (
            "More than one incomplete month was removed from the forecast while checking for incomplete months. Something unexpected is happening."
        )

        return _ds

    def read_regions(uri: str) -> isku.GridWeightingRegions:
        _region_weights = xr.load_dataset(uri, storage_options={"token": "anon"},)[
            ["lat", "lon", "region", "weight"]
        ]  # Load only what we need.
        # Apparently in this version of xarray the `.load()` method type-hints it'll return a DataArray instead of a Dataset.
        # It is a Dataset (I checked). So telling ty to ignore it.
        # # TODO: send bug upstream?
        regions = isku.GridWeightingRegions(_region_weights)  # ty: ignore[invalid-argument-type]
        return regions

    def read_gammas(uri: str) -> xr.Dataset:
        return xr.load_dataset(uri, storage_options={"token": "anon"},)

    def read_socioeconomics(uri: str) -> xr.Dataset:
        return xr.load_dataset(uri, storage_options={"token": "anon"},)

    return read_forecast_ensemble, read_reanalysis, read_regions


@app.cell
def _(
    ERA5_URI,
    REGIONS_URI,
    TAS_FORECAST_URI,
    read_forecast_ensemble,
    read_reanalysis,
    read_regions,
):
    reanalysis = read_reanalysis(ERA5_URI)
    forecast_ensemble = read_forecast_ensemble(TAS_FORECAST_URI)
    regions = read_regions(REGIONS_URI)
    # socioeconomics = read_socioeconomics(SOCIOECONOMICS_URI)
    # gammas = read_gammas(GAMMA_URI)
    return forecast_ensemble, reanalysis, regions


@app.cell
def _(ERA5_URI, TAS_FORECAST_URI, forecast_ensemble, reanalysis, xr):
    bias_adjust = True # bias adjust to GMFD
    save_adjusted = True # save the bias adjusted reanalysis and forecast data to gcp
    if bias_adjust:
        # here we want to delta shift both era5 and the forecast to GMFD I think
        era5file = "../tas_era5_1deg_monthly_climo_1980-2010.nc"
        gmfdfile = "../tas_gmfd_1deg_monthly_climo_1980-2010.nc"
        era5 = xr.open_dataset(era5file)
        gmfd = xr.open_dataset(gmfdfile)
        # need to convert lons to -180 - 180
        era5["longitude"] = (era5["longitude"] + 180) % 360 - 180
        gmfd["longitude"] = (gmfd["longitude"] + 180) % 360 - 180

        bias = gmfd - era5
        bias = bias.rename({"latitude":"lat","longitude": "lon"})

        # bias = bias.assign_coords({"month":reanalysis.time.dt.month})
        # want to match GMFD so if ERA5 is warmer, need to make it cooler. 
        # This means delta for gmfd-era5 is negative and can add it to era5 to cool era5 down. 
        # So bias = gmfd - era5, and then adjusted = forecast + bias. And era5 + bias
        reanalysis_adjusted = reanalysis + bias.sel(month=reanalysis.time.dt.month)

        forecast_ensemble_adjusted = forecast_ensemble + bias.sel(month=forecast_ensemble.time.dt.month)

        if save_adjusted:
            reanalysis_adjusted.attrs.update(
                {"Description": "ERA5 adjusted to GMFD using a monthly climatology difference from ERA5-GMFD over 1980-2010.",
                "reanalysis data": ERA5_URI,
                "forecast_ensemble_data": TAS_FORECAST_URI}
            )

            reanalysis_adjusted.to_zarr("reanalysis_bias_adjusted_to_GMFD.zarr")

            forecast_ensemble_adjusted.attrs.update(
                {"Description": "Seasonal forecast ensemble adjusted to GMFD using a monthly climatology difference from ERA5-GMFD over 1980-2010.",
                "reanalysis data": ERA5_URI,
                "forecast_ensemble_data": TAS_FORECAST_URI,
                 "Bias adjustment": "TODO add this step to poreallas. Bias adjustment delta calculated on notebooks.cilresearch.org. https://notebooks.cilresearch.org/user/kemccusker/lab/tree/ClimateImpactLab/bias_adjust_ERA5_to_GMFD_monthly_for_ENSO_work.ipynb"
                }
            )

            forecast_ensemble_adjusted.to_zarr("forecast_ensemble_bias_adjusted_to_GMFD.zarr")
    return bias_adjust, forecast_ensemble_adjusted, reanalysis_adjusted


@app.cell
def _(forecast_ensemble_adjusted):
    forecast_ensemble_adjusted.tas.isnull().any().values
    return


@app.cell
def _(xr):
    testds = xr.open_zarr("forecast_ensemble_bias_adjusted_to_GMFD.zarr")
    return (testds,)


@app.cell
def _(testds):
    testds.tas.isnull().any().values
    return


@app.cell
def _(
    bias_adjust,
    forecast_ensemble,
    forecast_ensemble_adjusted,
    isku,
    make_climtas,
    make_tas_monthly_histogram,
    np,
    reanalysis,
    reanalysis_adjusted,
    regions,
    socioeconomics,
    xr,
):

    if bias_adjust:
        # Transform _adjusted_ gridded data, extracting regional data needed for projections.
        histogram_hist_tas = isku.extract_regions(
            reanalysis_adjusted,
            template=make_tas_monthly_histogram,
            regions=regions,
        )
        histogram_forecast_tas = isku.extract_regions(
            forecast_ensemble_adjusted,
            template=make_tas_monthly_histogram,
            regions=regions,
        )
    else:    
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

    # calculate covariates
    #   average recent climate
    climtas = isku.extract_regions(
        reanalysis,
        template=make_climtas,
        regions=regions,
    ).sel(year=2025, drop=True)
    #   income
    loggdppc = np.log(socioeconomics["gdppc"].sel(year=2023, drop=True))
    return climtas, histogram_forecast_tas, histogram_hist_tas, loggdppc


@app.cell
def _(calculate_beta, climtas, gammas, histogram_forecast_tas, loggdppc, xr):
    # Calculate a fixed response function, i.e. beta.
    # Single, static response function with no adaptation is used for both projections (does not vary in time).
    # Stick everything together and make sure it aligns and matches. Rechunk all together. Also drop any regions with NaNs.
    beta_input = (
        xr.Dataset(
            {
                "tas_bin": histogram_forecast_tas["tas_bin"],
                "climtas": climtas["climtas"],
                "loggdppc": loggdppc,
                "gamma": gammas["gamma_mean"],
            }
        )
        .dropna(dim="region") # TODO: check these nan values. should they be nan?
        .chunk(
            {
                "region": "auto",  # "auto" is a sensible default.
                "tas_bin": -1,  # This also needs to be all in memory.
                "age_cohort": 1,  # We're doing all age_cohorts at once but could be done one-by-one.
                "degree": -1,  # For gammas and polynomial calculations. Should all be in memory.
            },
        )
        .unify_chunks()
    )
    fixed_beta = calculate_beta(beta_input).astype("float32").compute()
    fixed_beta["beta"].attrs = {
        "units": "deaths per 100,000 people",
        "long_name": "Temperature mortality rate",
    }
    # Do beta, allowing only hot deaths by 0-ing out everything on the cold side of the minimum-mortality temperature.
    fixed_beta["beta_hotonly"] = fixed_beta["beta"].where(
        fixed_beta["tas_bin"] > fixed_beta["mmt"], other=0
    )
    fixed_beta["beta_hotonly"].attrs["long_name"] = "Hot temperature mortality rate"

    # Do beta, allowing only COLD deaths by 0-ing out everything on the warm side of the minimum-mortality temperature.
    fixed_beta["beta_coldonly"] = fixed_beta["beta"].where(
        fixed_beta["tas_bin"] < fixed_beta["mmt"], other=0
    )
    fixed_beta["beta_coldonly"].attrs["long_name"] = "Cold temperature mortality rate"
    return (fixed_beta,)


@app.cell
def _(fixed_beta, histogram_forecast_tas, isku, mortality_effect_model, xr):
    # Project mortality.
    # Start with forecast ensemble.
    _forecast_input = (
        xr.Dataset(
            {
                "histogram_tas": histogram_forecast_tas["histogram_tas"],
                "beta": fixed_beta["beta"],
            }
        )
        .dropna(dim="region") # TODO check the nans are expected
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
    projected_forecast = isku.project(
        _forecast_input, model=mortality_effect_model
    ).compute()
    projected_forecast["effect"].attrs = {
        "units": "deaths per 100,000 people",
        "long_name": "Temperature mortality",
    }
    return (projected_forecast,)


@app.cell
def _(fixed_beta, histogram_forecast_tas, isku, mortality_effect_model, xr):
    # Now hot-only projection
    _forecast_input = (
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
        _forecast_input, model=mortality_effect_model
    ).compute()
    projected_forecast_hotonly["effect"].attrs = {
        "units": "deaths per 100,000 people",
        "long_name": "Hot temperature mortality",
    }
    return (projected_forecast_hotonly,)


@app.cell
def _(fixed_beta, histogram_forecast_tas, isku, mortality_effect_model, xr):
    # Now cold-only projection
    _forecast_input = (
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
        _forecast_input, model=mortality_effect_model
    ).compute()
    projected_forecast_coldonly["effect"].attrs = {
        "units": "deaths per 100,000 people",
        "long_name": "Cold temperature mortality",
    }
    return (projected_forecast_coldonly,)


@app.cell
def _(fixed_beta, histogram_hist_tas, isku, mortality_effect_model, xr):
    # Now do the baseline period.
    # Stick everything together and make sure it aligns and matches. Rechunk all together. Also drop any regions with NaNs.
    _hist_input = (
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
    projected_hist = isku.project(_hist_input, model=mortality_effect_model).compute()
    projected_hist["effect"].attrs = {
        "units": "deaths per 100,000 people",
        "long_name": "Temperature mortality",
    }
    return (projected_hist,)


@app.cell
def _(fixed_beta, histogram_hist_tas, isku, mortality_effect_model, xr):
    # Now hot-only projection
    _hist_input = (
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
        _hist_input, model=mortality_effect_model
    ).compute()
    projected_hist_hotonly["effect"].attrs = {
        "units": "deaths per 100,000 people",
        "long_name": "Hot temperature mortality",
    }
    return (projected_hist_hotonly,)


@app.cell
def _(fixed_beta, histogram_hist_tas, isku, mortality_effect_model, xr):
    # Now cold-only projection
    _hist_input = (
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
        _hist_input, model=mortality_effect_model
    ).compute()
    projected_hist_coldonly["effect"].attrs = {
        "units": "deaths per 100,000 people",
        "long_name": "Cold temperature mortality",
    }
    return (projected_hist_coldonly,)


@app.cell
def _(EFFECTS_URI):
    #EFFECTS_URI_LOCAL = "local_output/v20260702/effects_net_hot_cold.zarr"
    EFFECTS_URI_LOCAL = "local_output/v20260702/effects_net_hot_cold_gmfd_biasadjusted.zarr"

    print(EFFECTS_URI)
    print(EFFECTS_URI_LOCAL)
    return (EFFECTS_URI_LOCAL,)


@app.cell
def _(
    EFFECTS_URI,
    EFFECTS_URI_LOCAL,
    ERA5_URI,
    GAMMA_URI,
    REGIONS_URI,
    SOCIOECONOMICS_URI,
    TAS_FORECAST_URI,
    datetime,
    projected_forecast,
    projected_forecast_coldonly,
    projected_forecast_hotonly,
    projected_hist,
    projected_hist_coldonly,
    projected_hist_hotonly,
    uuid,
    xr,
):

    # Collect everything and write to storage.
    _out = {
        "forecast": projected_forecast,
        "baseline": projected_hist,
        "forecast_hotonly": projected_forecast_hotonly,
        "baseline_hotonly": projected_hist_hotonly,
        "forecast_coldonly": projected_forecast_coldonly,
        "baseline_coldonly": projected_hist_coldonly,}
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
    if EFFECTS_URI_LOCAL is not None:
        _out_dt.to_zarr(EFFECTS_URI_LOCAL, consolidated=False)
        print(f"Effects written to {EFFECTS_URI_LOCAL}")
    if EFFECTS_URI is not None:
        try:
            _out_dt.to_zarr(EFFECTS_URI, consolidated=False)
            print(f"Effects written to {EFFECTS_URI}")
        except Exception as e:
            print("Caught Exception. You probably don't have permissions to write to the bucket")

    _out_dt
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
