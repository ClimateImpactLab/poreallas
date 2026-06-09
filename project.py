import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    import geopandas
    import isku
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import xarray as xr

    from poreallas.extract import (
        FuzzyGridWeightingExtractor,
        make_climtas,
        make_tas_monthly_histogram,
    )
    from poreallas.project import mortality_effect_model

    return (
        FuzzyGridWeightingExtractor,
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
        xr,
    )


@app.cell
def _(os):
    TAS_FORECAST_URI = os.getenv("POREALLAS_TAS_FORECAST_URI")
    ERA5_URI = os.getenv("POREALLAS_ERA5_URI")
    GAMMA_URI = os.getenv("POREALLAS_GAMMA_URI")
    REGIONS_URI = os.getenv("POREALLAS_REGIONS_URI")
    IMPACT_REGION_POLYGONS = os.getenv("POREALLAS_REGIONS_POLYGONS_URI")

    # Or using this because I have it on hand:
    # NOTE: You'll need to update this path to match your system.
    # Download and unpack https://zenodo.org/records/6416119/files/data.zip?download=1.
    # Beware, it's ~35 GiB.
    # This is the file at ./data/input/raw/data/2_projection/2_econ_vars/SSP3.nc4 within the downloaded data.
    SOCIOECONOMICS_URI = os.getenv("POREALLAS_SOCIOECONOMICS_URI")
    return (
        ERA5_URI,
        GAMMA_URI,
        IMPACT_REGION_POLYGONS,
        REGIONS_URI,
        SOCIOECONOMICS_URI,
        TAS_FORECAST_URI,
    )


@app.cell
def _(FuzzyGridWeightingExtractor, REGIONS_URI, xr):
    regions = FuzzyGridWeightingExtractor(
        xr.open_dataset(REGIONS_URI).load(), tolerance=0.5
    )
    return (regions,)


@app.cell
def _(ERA5_URI, isku, make_climtas, regions, xr):
    _ds = xr.open_dataset(ERA5_URI)

    # Clean up longitude. The data goes from longitude 0 to 360. It needs to go -180 to 180 in ascending order.
    _ds["longitude"] = (_ds["longitude"] + 180) % 360 - 180
    _ds = _ds.sortby("longitude")
    _ds = _ds.rename({"longitude": "lon", "latitude": "lat"})

    climtas = isku.extract_regions(
        _ds,
        template=make_climtas,
        regions=regions,
    ).sel(year=2025, drop=True)
    return (climtas,)


@app.cell
def _(SOCIOECONOMICS_URI, np, xr):
    socioecon = xr.open_dataset(SOCIOECONOMICS_URI)

    # TODO: Maybe this should be in a cleaning script.
    # Socioecon projection data is backfilled.
    # Though the socioeconomic projection start in 2010, projections used 2015
    # GDPpc values to backfill to 1981. Extend GDPpc data range further back so
    # 13-year half-Bartlett smoothing can begin in 1981 without NaNs.
    _kernel_length = 13
    loggdppc = np.log(
        socioecon["gdppc"]
        .sel(year=slice(2015, 2100))
        .reindex({"year": range(1981 - _kernel_length, 2100)}, method="backfill")
    )
    _w = np.arange(_kernel_length)
    _weight = xr.DataArray(_w / _w.sum(), dims=["window"])
    loggdppc = (
        loggdppc.rolling(year=_kernel_length).construct(year="window").dot(_weight)
    )

    # But we just need something for this projection prototype, so let's grab 2015.
    loggdppc = loggdppc.sel(model="high", year=2015, drop=True)
    loggdppc
    return (loggdppc,)


@app.cell
def _(GAMMA_URI, xr):
    gammas = xr.open_dataset(GAMMA_URI)
    return (gammas,)


@app.cell
def _(TAS_FORECAST_URI, plt, sns, xr):
    # DEBUG more sanity checks

    _ds = xr.open_dataset(TAS_FORECAST_URI).set_coords("valid_time")

    # Clean up longitude. The data goes from longitude 0 to 360. It needs to go -180 to 180 in ascending order.
    _ds["longitude"] = (_ds["longitude"] + 180) % 360 - 180
    _ds = _ds.sortby("longitude")

    with sns.axes_style("whitegrid"):
        _ds.set_coords("valid_time")["tas"].isel(latitude=100, longitude=100).squeeze(
            drop=True
        ).plot.scatter(x="valid_time", marker=".", alpha=0.3, edgecolors="none")
        _ds.set_coords("valid_time")["tas"].isel(latitude=100, longitude=100).squeeze(
            drop=True
        ).mean(dim="number").plot.line(x="valid_time", color="C1")
        plt.show()
    return


@app.cell
def _(TAS_FORECAST_URI, isku, make_tas_monthly_histogram, regions, xr):
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

    histogram_forecast_tas = isku.extract_regions(
        _ds,
        template=make_tas_monthly_histogram,
        regions=regions,
    )
    return (histogram_forecast_tas,)


@app.cell
def _():
    # # # # DEBUG Remove incomplete months.
    # # # # TODO Maybe should be part of earlier cleanup step.

    # # # histogram_forecast_tas.sum(dim="tas_bin")
    # # # assert (histogram_forecast_tas["time"].dt.days_in_month == histogram_forecast_tas.sum(dim="tas_bin")).all().compute()

    # _ds = xr.open_dataset(TAS_FORECAST_URI).set_coords("valid_time")

    # # Clean up longitude. The data goes from longitude 0 to 360. It needs to go -180 to 180 in ascending order.
    # _ds["longitude"] = (_ds["longitude"] + 180) % 360 - 180
    # _ds = _ds.sortby("longitude")

    # _ds = _ds.rename(
    #     {
    #         "valid_time": "time",
    #         "latitude": "lat",
    #         "longitude": "lon",
    #     }
    # )
    # _ds = _ds.swap_dims({"forecast_period": "time"}).squeeze(drop=True)
    # _ds = _ds.chunk("auto")

    # # # number_obs = _ds["time"].resample(time="ME").count()
    # # # days_in_month = number_obs["time"].dt.days_in_month
    # # # complete_months = number_obs == days_in_month
    # # # complete_months["time"]
    # # # _ds["time"].dt.month (number_obs == days_in_month).sel(time.dt.month = _ds["time"].dt.month)

    # # # _ds["time"].dt.days_in_month
    # # # _ds["time"].dt.month

    # # def _map_fn(x, dt_dim):
    # #     number_obs = x[dt_dim].count()
    # #     days_in_month = x[dt_dim].dt.days_in_month
    # #     complete_months = number_obs == days_in_month
    # #     if not all(complete_months):
    # #         return xr.full_like(x, fill_value=np.nan)
    # #     return x

    # #     # print(x["time"].count())
    # #     # if (x.notnull()["time"].count() == x["time"].dt.days_in_month).all():
    # #     #     return x
    # #     # return None

    # # _obs_in_month = _ds.resample(time="ME").map(_map_fn, args=("time",))#.dropna(dim="time")
    # # _obs_in_month

    # _ds.sel(time=slice("2026-05", "2026-11"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Forecast
    """)
    return


@app.cell
def _(climtas, gammas, histogram_forecast_tas, loggdppc, xr):
    # Stick everything together and make sure it aligns and matches. Rechunk all together. Also drop any regions with NaNs.
    forecast_input_ds = (
        xr.Dataset(
            {
                "histogram_tas": histogram_forecast_tas["histogram_tas"],
                "climtas": climtas["climtas"],
                "loggdppc": loggdppc,
                "gamma": gammas["gamma_mean"],
            }
        )
        .dropna(dim="region")
        .chunk(
            {
                "region": "auto",  # "auto" is a sensible default.
                "time": -1,  # This needs to be all in memory, thus -1.
                "tas_bin": -1,  # This also needs to be all in memory.
                "age_cohort": 1,  # We're doing all age_cohorts at once but could be done one-by-one.
                "degree": -1,  # For gammas and polynomial calculations. Should all be in memory.
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

    # Taking the average of the forecast ensemble members.
    projected_forecast["effect"] = projected_forecast["effect"].mean(dim="number")

    # Not required, just for fun.
    projected_forecast["effect"].attrs = {
        "units": "deaths per 100,000 people",
        "long_name": "Temperature mortality",
    }

    projected_forecast = projected_forecast.compute()
    return (projected_forecast,)


@app.cell
def _(projected_forecast):
    projected_forecast
    return


@app.cell
def _(plt, projected_forecast, sns):
    # Quick plot for an arbitrary region for diagnostics.
    with sns.axes_style("whitegrid"):
        projected_forecast["effect"].sel(region="USA.14.608").plot(hue="age_cohort")
    plt.gca()
    # WARNING: Not sure why we're getting values for December...
    # Ahh, because it's 215 days from the beginning of May this year...
    # TODO: Guard against incomplete months.
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
        projected_forecast["effect"].to_dataframe().reset_index(),
        on="region",
    )

    # Subset what we want to plot
    _plot_data = _polygons[
        (_polygons["age_cohort"] == "age3") & (_polygons["time"].dt.month == 5)
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
    ### Hist
    """)
    return


@app.cell
def _(ERA5_URI, isku, make_tas_monthly_histogram, regions, xr):
    _ds = xr.open_dataset(ERA5_URI)

    # Clean up longitude. The data goes from longitude 0 to 360. It needs to go -180 to 180 in ascending order.
    _ds["longitude"] = (_ds["longitude"] + 180) % 360 - 180
    _ds = _ds.sortby("longitude")
    _ds = _ds.rename({"longitude": "lon", "latitude": "lat"})
    _ds = _ds.chunk("auto")

    histogram_hist_tas = isku.extract_regions(
        _ds,
        template=make_tas_monthly_histogram,
        regions=regions,
    )
    return (histogram_hist_tas,)


@app.cell
def _(climtas, gammas, histogram_hist_tas, loggdppc, xr):
    # Stick everything together and make sure it aligns and matches. Rechunk all together. Also drop any regions with NaNs.
    hist_input_ds = (
        xr.Dataset(
            {
                "histogram_tas": histogram_hist_tas["histogram_tas"],
                "climtas": climtas["climtas"],
                "loggdppc": loggdppc,
                "gamma": gammas["gamma_mean"],
            }
        )
        .dropna(dim="region")
        .chunk(
            {
                "region": "auto",  # "auto" is a sensible default.
                "time": -1,  # This needs to be all in memory, thus -1.
                "tas_bin": -1,  # This also needs to be all in memory.
                "age_cohort": 1,  # We're doing all age_cohorts at once but could be done one-by-one.
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
        projected_forecast["effect"].sel(region=_target_region).groupby("time.month")
    )

    with sns.axes_style("whitegrid"):
        (_forecast - _climatology).plot(hue="age_cohort")
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Outstanding:

    [x] ERA5 baseline?

    [ ] Guard against incomplete months.

    [ ] Gap between ERA5 and forecast.

        (thurs -> Mon (friday, latest) ^)

    [ ] Higher temporal resolution hindcast from forecast ensemble? (what period are hindcasts?)

    [ ] tasmin/tasmax vs hr instantaneous average.

    [ ] Grid weights and regions (aka segment weights). Population weighting.

    [ ] Proportion of population in each age cohort.

    [ ] GDPpc.

    [ ] Uniform dollar year.
    """)
    return


if __name__ == "__main__":
    app.run()
