import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import datetime
    import os
    import uuid

    import geopandas
    import isku
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import xarray as xr

    from poreallas.extract import make_climtas, make_tas_monthly_histogram
    from poreallas.project import mortality_effect_model, calculate_beta

    return (
        calculate_beta,
        datetime,
        geopandas,
        isku,
        make_climtas,
        make_tas_monthly_histogram,
        mo,
        mortality_effect_model,
        np,
        os,
        plt,
        sns,
        uuid,
        xr,
    )


@app.cell
def _(os):
    # Input
    TAS_FORECAST_URI = os.getenv("POREALLAS_TAS_FORECAST_URI")
    ERA5_URI = os.getenv("POREALLAS_ERA5_URI")
    GAMMA_URI = os.getenv("POREALLAS_GAMMA_URI")
    SOCIOECONOMICS_URI = os.getenv("POREALLAS_SOCIOECONOMICS_URI")
    REGIONS_URI = os.getenv("POREALLAS_REGIONS_URI")
    IMPACT_REGION_POLYGONS = os.getenv("POREALLAS_REGIONS_POLYGONS_URI")

    # Output
    EFFECTS_URI = os.getenv("POREALLAS_EFFECTS_URI")
    return (
        EFFECTS_URI,
        ERA5_URI,
        GAMMA_URI,
        IMPACT_REGION_POLYGONS,
        REGIONS_URI,
        SOCIOECONOMICS_URI,
        TAS_FORECAST_URI,
    )


@app.cell
def _(ERA5_URI, xr):
    _ds = xr.open_dataset(ERA5_URI)

    # Clean up longitude. The data goes from longitude 0 to 360. It needs to go -180 to 180 in ascending order.
    _ds["longitude"] = (_ds["longitude"] + 180) % 360 - 180
    _ds = _ds.sortby("longitude")
    _ds = _ds.rename({"longitude": "lon", "latitude": "lat"})
    _ds = _ds.chunk("auto")

    reanalysis = _ds
    reanalysis
    return (reanalysis,)


@app.cell
def _(TAS_FORECAST_URI, np, xr):
    _ds = xr.open_dataset(TAS_FORECAST_URI).set_coords("valid_time")

    # Clean up longitude. The data goes from longitude 0 to 360. It needs to go -180 to 180 in ascending order.
    _ds["longitude"] = (_ds["longitude"] + 180) % 360 - 180
    _ds = _ds.sortby("longitude")

    _ds = _ds.rename(
        {
            "valid_time": "time",
            "latitude": "lat",
            "longitude": "lon",
        }
    )
    _ds = _ds.swap_dims({"forecast_period": "time"}).squeeze(drop=True)
    _ds = _ds.chunk("auto")

    # Drop months without required number of obs. Forecast ensemble is for a fixed number of days so we expect to usually trim off the last month of the forecast if it is ragged and missing days beyond a threshold.
    _dt_dim = "time"
    _n_initial = _ds[_dt_dim].size
    _number_obs = _ds[_dt_dim].resample(time="ME").count()
    _days_in_month = _number_obs["time"].dt.days_in_month
    required_percent = 0.9
    _min_req = np.round(_days_in_month * required_percent)
    _qualifying_months = _number_obs.where(_number_obs >= _min_req, drop=True)[
        "time"
    ].dt.month
    _ds = _ds.where(_ds["time"].dt.month.isin(_qualifying_months), drop=True)

    _n_current = _ds[_dt_dim].size
    _n_initial_months = _number_obs["time"].size
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

    forecast_ensemble = _ds
    forecast_ensemble
    return (forecast_ensemble,)


@app.cell
def _(forecast_ensemble, plt, sns):
    # More sanity checks

    # Arbitrary gridpoint.
    target_region = dict(lat=100, lon=100)

    with sns.axes_style("whitegrid"):
        forecast_ensemble["tas"].isel(**target_region).squeeze(drop=True).plot.scatter(
            x="time", marker=".", alpha=0.3, edgecolors="none"
        )
        forecast_ensemble["tas"].isel(**target_region).squeeze(drop=True).mean(
            dim="number"
        ).plot.line(x="time", color="C1")
        plt.show()
    return


@app.cell
def _(REGIONS_URI, isku, xr):
    _region_weights = xr.open_dataset(REGIONS_URI)[
        ["lat", "lon", "region", "weight"]
    ]  # Only what we need.
    regions = isku.GridWeightingRegions(_region_weights.load())
    return (regions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fix response function: calculate $\beta$
    """)
    return


@app.cell
def _(
    forecast_ensemble,
    isku,
    make_tas_monthly_histogram,
    reanalysis,
    regions,
    xr,
):
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
    # But, we otherwise, we don't need the complete histgram_tas to calculate beta.
    xr.testing.assert_allclose(
        histogram_hist_tas["tas_bin"], histogram_forecast_tas["tas_bin"]
    )
    return histogram_forecast_tas, histogram_hist_tas


@app.cell
def _(
    GAMMA_URI,
    SOCIOECONOMICS_URI,
    calculate_beta,
    histogram_forecast_tas,
    isku,
    make_climtas,
    np,
    reanalysis,
    regions,
    xr,
):
    # Using same, static beta in mortality projections so calculating it once here.
    # Calculating it ahead of time can help with diagnostics.

    _climtas = isku.extract_regions(
        reanalysis,
        template=make_climtas,
        regions=regions,
    ).sel(year=2025, drop=True)

    _socioecon = xr.open_dataset(SOCIOECONOMICS_URI)
    _loggdppc = np.log(_socioecon["gdppc"].sel(year=2023, drop=True))

    _gammas = xr.open_dataset(GAMMA_URI)

    # Stick everything together and make sure it aligns and matches. Rechunk all together. Also drop any regions with NaNs.
    beta_input = (
        xr.Dataset(
            {
                "tas_bin": histogram_forecast_tas["tas_bin"],
                "climtas": _climtas["climtas"],
                "loggdppc": _loggdppc,
                "gamma": _gammas["gamma_mean"],
            }
        )
        .dropna(dim="region")
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

    fixed_beta
    return (fixed_beta,)


@app.cell
def _(fixed_beta, plt, sns):
    # Sanity check. It should be u-shaped.
    with sns.axes_style("whitegrid"):
        fixed_beta["beta"].sel(region="USA.14.608").plot(hue="age_cohort")
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Project mortality
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Effect for seasonal forecast ensemble
    """)
    return


@app.cell
def _(fixed_beta, histogram_forecast_tas, xr):
    # Stick everything together and make sure it aligns and matches. Rechunk all together. Also drop any regions with NaNs.
    forecast_input_ds = (
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
            },
        )
        .unify_chunks()
    )

    forecast_input_ds
    return (forecast_input_ds,)


@app.cell
def _(forecast_input_ds):
    # Quick sanity check

    forecast_input_ds["histogram_tas"].sel(region="USA.14.608").sum(dim="number").plot(
        y="tas_bin", x="time"
    )
    # You can see there are only a few days in December, as the forecast is run for 215 days.
    return


@app.cell
def _(forecast_input_ds, isku, mortality_effect_model):
    projected_forecast = isku.project(forecast_input_ds, model=mortality_effect_model)

    # Not required, just for fun.
    projected_forecast["effect"].attrs = {
        "units": "deaths per 100,000 people",
        "long_name": "Temperature mortality",
    }

    projected_forecast = projected_forecast.compute()
    projected_forecast
    return (projected_forecast,)


@app.cell
def _(plt, projected_forecast, sns):
    # Quick plot for an arbitrary region for diagnostics.
    with sns.axes_style("whitegrid"):
        projected_forecast["effect"].mean(dim="number").sel(region="USA.14.608").plot(
            hue="age_cohort"
        )
    plt.gca()
    return


@app.cell
def _(IMPACT_REGION_POLYGONS, geopandas, plt, projected_forecast):
    # Quick map for diagnostics
    # Grab polygons for our regions.
    _polygons = (
        geopandas.read_parquet(IMPACT_REGION_POLYGONS)
        .rename(columns={"hierid": "region"})
        .set_index("region")
        .set_crs(epsg=4326)  # Assuming the data is WGS-82.
    )
    # Join projected effects on region names.
    _polygons = _polygons.merge(
        projected_forecast["effect"].mean(dim="number").to_dataframe().reset_index(),
        on="region",
    )

    # Subset what we want to plot
    _plot_data = _polygons[
        (_polygons["age_cohort"] == "age65plus") & (_polygons["time"].dt.month == 5)
    ].to_crs("ESRI:54030")  # Convert to Robinson projection.

    ax = _plot_data.plot(
        column="effect",
        legend=True,
        figsize=(15, 10),
        cmap="viridis",
        legend_kwds={
            "label": f"{projected_forecast['effect'].attrs['long_name']} [{projected_forecast['effect'].attrs['units']}]",
            "orientation": "horizontal",
        },
    )
    ax.set_axis_off()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Effect for reanalysis climatology
    """)
    return


@app.cell
def _(fixed_beta, histogram_hist_tas, xr):
    # Stick everything together and make sure it aligns and matches. Rechunk all together. Also drop any regions with NaNs.
    hist_input_ds = (
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

    hist_input_ds
    return (hist_input_ds,)


@app.cell
def _(hist_input_ds):
    # Quick sanity check for histogram tas over the "historical" period.

    hist_input_ds["histogram_tas"].sel(region="USA.14.608").plot(y="tas_bin", x="time")
    return


@app.cell
def _(hist_input_ds, isku, mortality_effect_model):
    projected_hist = isku.project(hist_input_ds, model=mortality_effect_model)

    # Not required, just for fun.
    projected_hist["effect"].attrs = {
        "units": "deaths per 100,000 people",
        "long_name": "Temperature mortality",
    }

    projected_hist = projected_hist.compute()
    return (projected_hist,)


@app.cell
def _(plt, projected_hist, sns):
    # Quick plot for an arbitrary region for diagnostics.
    with sns.axes_style("whitegrid"):
        projected_hist["effect"].sel(region="USA.14.608").plot(hue="age_cohort")
    plt.gca()
    return


@app.cell
def _(plt, projected_hist, sns):
    # Monthly mean over most recent 10 year period of hist projection.
    with sns.axes_style("whitegrid"):
        projected_hist["effect"].sel(
            region="USA.14.608", time=slice("2015", "2025")
        ).groupby("time.month").mean().plot(hue="age_cohort")
    plt.gca()
    return


@app.cell
def _(plt, projected_forecast, projected_hist, sns):
    # Monthly mean over most recent 10 year period of hist projection.
    _target_region = "USA.14.608"
    _climatology = (
        projected_hist["effect"]
        .sel(region=_target_region, time=slice("2015", "2025"))
        .groupby("time.month")
        .mean()
    )
    _forecast = (
        projected_forecast["effect"]
        .mean(dim="number")
        .sel(region=_target_region)
        .groupby("time.month")
    )

    with sns.axes_style("whitegrid"):
        (_forecast - _climatology).plot(hue="age_cohort")
    plt.gca()
    return


@app.cell
def _(
    EFFECTS_URI,
    ERA5_URI,
    GAMMA_URI,
    REGIONS_URI,
    SOCIOECONOMICS_URI,
    TAS_FORECAST_URI,
    datetime,
    projected_forecast,
    projected_hist,
    uuid,
    xr,
):
    # Write output to storage.

    _out = {
        "forecast": projected_forecast,
        "baseline": projected_hist,
    }
    _out_dt = xr.DataTree.from_dict(_out)

    # Add metadata
    _uuid = str(uuid.uuid4())
    _datetime_now = datetime.datetime.now(datetime.timezone.utc).isoformat()

    _out_dt.attrs |= {
        "poreallas_created_at": _datetime_now,
        "poreallas_uuid": _uuid,
        "poreallas_description": "Projected temperature mortality effects",
    }

    _out_dt["forecast"].attrs |= {
        "poreallas_created_at": _datetime_now,
        "poreallas_uuid": _uuid,
        "poreallas_description": "Forecast ensemble projected temperature mortality effects",
        "poreallas_temperature_uri": TAS_FORECAST_URI,
        "poreallas_socioeconomics_uri": SOCIOECONOMICS_URI,
        "poreallas_model_parameters_uri": GAMMA_URI,
        "poreallas_regions_uri": REGIONS_URI,
    }

    _out_dt["baseline"].attrs |= {
        "poreallas_created_at": _datetime_now,
        "poreallas_uuid": _uuid,
        "poreallas_description": "Baseline projected temperature mortality effects",
        "poreallas_temperature_uri": ERA5_URI,
        "poreallas_socioeconomics_uri": SOCIOECONOMICS_URI,
        "poreallas_model_parameters_uri": GAMMA_URI,
        "poreallas_regions_uri": REGIONS_URI,
    }

    if EFFECTS_URI is not None:
        _out_dt.to_zarr(EFFECTS_URI, consolidated=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Outstanding:

    [x] ERA5 baseline?

    [x] Guard against incomplete months.

    [x] Gap between ERA5 and forecast.

        (thurs -> Mon (friday, latest) ^)

    [ ] Higher temporal resolution hindcast from forecast ensemble?

    [x] What period are hindcasts?
            - According to https://confluence.ecmwf.int/display/CKB/C3S+seasonal+forecast+product+descriptions. "In general, the common hindcast period, 1993 - 2016, is used as the reference period for C3S data and graphical products, regardless of the hindcast period available for each individual component system (unless stated otherwise)."

    [x] Grid weights and regions (aka segment weights). Population weighting.

    [/] Proportion of population in each age cohort.

    [x] GDPpc.
    """)
    return


if __name__ == "__main__":
    app.run()
