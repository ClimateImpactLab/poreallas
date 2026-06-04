import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():

    import geopandas
    import isku
    import marimo as mo
    import matplotlib.pyplot as plt
    import numba
    import numpy as np
    import seaborn as sns
    import xarray as xr
    from xclim.core import units
    from xhistogram.xarray import histogram

    return geopandas, histogram, isku, mo, np, numba, plt, sns, units, xr


@app.cell
def _():
    TAS_FORECAST_URI = "./data/parsed/s51_tas.zarr/"
    CLIMTAS_URI = "./data/parsed/climtas.zarr/"
    ERA5_URI = "./data/era5_annual_tas_1995_2025_regrid.nc"
    GAMMA_URI = "./data/parsed/gamma.zarr/"
    REGIONS_URI = "./data/parsed/segment_weights.zarr/"
    IMPACT_REGION_POLYGONS = "./data/parsed/impact_region.parquet"

    # Or using this because I have it on hand:
    # NOTE: You'll need to update this path to match your system.
    # Download and unpack https://zenodo.org/records/6416119/files/data.zip?download=1.
    # Beware, it's ~35 GiB.
    # This is the file at ./data/input/raw/data/2_projection/2_econ_vars/SSP3.nc4 within the downloaded data.
    SOCIOECONOMICS_URI = "./data/raw/SSP3.nc4"

    return (
        ERA5_URI,
        GAMMA_URI,
        IMPACT_REGION_POLYGONS,
        REGIONS_URI,
        SOCIOECONOMICS_URI,
        TAS_FORECAST_URI,
    )


@app.cell
def _(isku, np, units, xr):
    def _make_annual_tas(ds: xr.Dataset) -> xr.Dataset:
        """
        Compute tas variable in degC. Should be annual.
        """
        tas = units.convert_units_to(ds["tas"], "degC")
        ## TODO: If the data needs to be annualized... Might need this.
        # return tas.groupby("time.year").mean("time").to_dataset()
        return tas.to_dataset()

    def _make_30hbartlett_climtas(ds: xr.Dataset) -> xr.Dataset:
        """
        From annaual 'tas' compute 30-year half-Bartlett kernel average.

        Output variable is "climtas". This assumes input's "tas" has "year"
        time dim.
        """
        kernel_length = 30
        w = np.arange(kernel_length)
        weight = xr.DataArray(w / w.sum(), dims=["window"])
        da = ds["tas"].rolling(year=30).construct(year="window").dot(weight)
        return da.to_dataset(name="climtas").astype("float32")

    make_climtas = isku.build_extraction_template(
        pre=_make_annual_tas,
        post=_make_30hbartlett_climtas,
    )

    # Need this because the regions/segment weights don't properly align with the ECMWRF grids. So, fuzzy match for now.
    class FuzzyGridWeightingExtractor(isku.RegionExtractor):
        """
        Weight a grid and extract regions when the region and weight position don't exactly match lat, lon on the grid.

        Follows the isku.RegionExtractor protocol.
        Uses Nearest-neighbor search within `tolerance` to extract points from a grid. Suggest using a reasonable `tolerance` that does not extend beyond one grid cell width. Otherwise, the algorithm will happily walk across the globe to find a matching grid point.
        """

        def __init__(self, weights: xr.Dataset, tolerance: float):
            # Check everything we need is there.
            target_variables = ("lat", "lon", "weight", "region")
            missing_variables = [
                v for v in target_variables if v not in weights.variables
            ]
            if missing_variables:
                raise ValueError(
                    f"input weights is missing required {missing_variables} variable(s)"
                )

            self._data = weights
            self.tolerance = tolerance

        def extract_regions(self, ds: xr.Dataset) -> xr.Dataset:
            region_sel = ds.sel(
                lat=self._data["lat"],
                lon=self._data["lon"],
                method="nearest",
                tolerance=self.tolerance,
            )
            out = (
                (region_sel * self._data["weight"]).groupby(self._data["region"]).sum()
            )
            return out

    return FuzzyGridWeightingExtractor, make_climtas


@app.cell
def _(FuzzyGridWeightingExtractor, REGIONS_URI, xr):
    regions = FuzzyGridWeightingExtractor(
        xr.open_zarr(REGIONS_URI).load(), tolerance=0.5
    )
    return (regions,)


@app.cell
def _(ERA5_URI, isku, make_climtas, regions, xr):
    _ds = xr.open_dataset(ERA5_URI)

    # Clean up longitude. The data goes from longitude 0 to 360. It needs to go -180 to 180 in ascending order.
    _ds["longitude"] = (_ds["longitude"] + 180) % 360 - 180
    _ds = _ds.sortby("longitude")
    # Add missing unit information. TODO: This should really be in cleaning or something.
    _ds["tas"].attrs["units"] = "K"
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
    gammas = xr.open_zarr(GAMMA_URI)
    return (gammas,)


@app.cell
def _():
    # climtas = xr.open_zarr(CLIMTAS_URI, chunks={})
    return


@app.cell
def _(histogram, isku, np, units, xr):
    def _make_monthly_tas_histogram_across_ensembles(ds: xr.Dataset) -> xr.Dataset:
        _tas = units.convert_units_to(ds["tas"], "degC")

        _bins = np.arange(-105, 66)  # Range we get histogram count for. NOTE: in degC!
        # _tas_annual_histogram = _tas.resample(time="1ME").map(
        _tas_annual_histogram = _tas.resample(
            time="1MS"
        ).map(
            histogram,
            bins=[_bins],
            dim=[
                "time",
                "number",
            ],  # So histograms count across each month ("time"), and 51 ensemble members ("number"). Also saves space.
        )
        return _tas_annual_histogram.to_dataset().astype("float32")

    make_tas_monthly_histogram = isku.build_extraction_template(
        pre=_make_monthly_tas_histogram_across_ensembles,
        post=lambda ds: ds.astype(
            "float32"
        ),  # Save space. Don't need float64 precision.
    )
    return (make_tas_monthly_histogram,)


@app.cell
def _(TAS_FORECAST_URI, plt, sns, xr):
    # DEBUG: THIS CELL IS ALL DIAGNOSTICS.

    _ds = xr.open_zarr(TAS_FORECAST_URI, chunks={}).set_coords("valid_time")

    # Clean up longitude. The data goes from longitude 0 to 360. It needs to go -180 to 180 in ascending order.
    _ds["longitude"] = (_ds["longitude"] + 180) % 360 - 180
    _ds = _ds.sortby("longitude")

    # _ds = _ds.rename(
    #     {
    #         "valid_time": "time",
    #         "latitude": "lat",
    #         "longitude": "lon",
    #     }
    # )
    # _ds = _ds.swap_dims({"forecast_period": "time"}).squeeze(drop=True)

    # _ds.isel(latitude=100, longitude=100)["tas"].squeeze(drop=True).plot(hue="number", x="forecast_period")

    # _ds["valid_time"]
    with sns.axes_style("whitegrid"):
        _ds.set_coords("valid_time")["tas"].isel(latitude=100, longitude=100).squeeze(
            drop=True
        ).plot.scatter(x="valid_time", marker=".", alpha=0.3, edgecolors="none")
        _ds.set_coords("valid_time")["tas"].isel(latitude=100, longitude=100).squeeze(
            drop=True
        ).mean(dim="number").plot.line(x="valid_time", color="C1")
        plt.show()


@app.cell
def _(TAS_FORECAST_URI, isku, make_tas_monthly_histogram, regions, xr):
    _ds = xr.open_zarr(TAS_FORECAST_URI, chunks={}).set_coords("valid_time")

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

    histogram_tas = isku.extract_regions(
        _ds,
        template=make_tas_monthly_histogram,
        regions=regions,
    )
    return (histogram_tas,)


@app.cell
def _(isku, np, numba, xr):
    @numba.njit()
    def _maximum_accumulate(x):
        rmax = x[0]
        y = np.empty_like(x)
        for i, val in enumerate(x):
            rmax = max(rmax, val)
            y[i] = rmax
        return y

    @numba.guvectorize(["void(float64[:], int64, float64[:])"], "(n),()->(n)")
    def _uclip_gufunc(x, idx_min, result):
        # Doing this if/else because can't return early in guvectorize funcs.
        if len(x) < 3:
            result[:] = x
        else:
            n = len(x)
            # WARNING: Throw error if min_idx_max is greater than n.
            # WARNING: Throw error if min_idx_min is greater than min_idx_max.

            # Get right side of minimum idx, ascending from minimum.
            rs = x[idx_min:n]
            # Left size, but reversed to still ascend from minimum.
            ls = x[0 : idx_min + 1][::-1]
            n_ls = len(ls)

            # Take accumulative maximum for each side and stick it in outgoing array,
            # with left hand side reversed back.
            result[0:idx_min] = _maximum_accumulate(ls)[1:n_ls][::-1]
            result[idx_min:n] = _maximum_accumulate(rs)

    def uclip(x, dim, idx_min):
        """Performs U Clipping of an unclipped response function for all regions
        simultaneously, centered around each region's Minimum Mortality Temperature (MMT).

        Parameters
        ----------
        da : DataArray
            xarray.DataArray of unclipped response functions
        dim : str
            Dimension name along which clipping will be applied. Clipping will be applied
            along dimensions 'dim' for all other dimensions independently. This is usually
            the "dose" climate or weather variable, e.g., daily-average air
            temperature ('tas' or 'tas_bin').
        idx_min : int
            Index of value to use as the middle base of the "U". Found along the
            dimension 'dim' of 'da'. In these mortality projections, this is often the
            index of the Minimum Mortality Temperature (MMT).

        Returns
        -------
        clipped : xarray DataArray of the regions response functioned centered on idx_min.
        """
        # TODO: Check that `idx_min` can be found along the `dim` of `x`.
        return xr.apply_ufunc(
            _uclip_gufunc,
            x,
            idx_min,
            input_core_dims=[[dim], []],
            output_core_dims=[[dim]],
            output_dtypes=["float64"],
            dask="parallelized",
        )

    def _no_processing(ds: xr.Dataset) -> xr.Dataset:
        return ds

    def minimum_arg(x: xr.DataArray, *, dim="tas_bin", lmmt=10.0, ummt=30.0):
        """
        Get minimum and its associated dim label within an inclusive range along dim
        """
        # Find minimum within inclusive range, get the dim label for the minimum's position.
        # Need to .compute() mmt_argmin because can't yet do vectorized indexing with dask arrays. See https://github.com/dask/dask/issues/8958
        min_arg = (
            x.where((x[dim] >= lmmt) & (x[dim] <= ummt)).argmin(dim=dim).compute()
        )  # This forces a compute on dask arrays. It's a problem. TODO.
        # Get the actual minima values.
        min_value = x.isel({dim: min_arg})
        return min_value, min_arg

    def _add_degree_coord(da: xr.DataArray, max_degrees: float) -> xr.DataArray:
        """
        Raises array to 1 ... max_degrees power, concatenating all together in a new "degree" coordinate.
        """
        if max_degrees < 2:
            # TODO: Test what actually happens for this edge case.
            # Raising an error because we're avoiding calculating da^1 because da is sometimes really big,
            # not sure the code handles this case very well and it's likely a mistake so just raising an
            # error for now.
            raise ValueError("'max_degree' arg must be >= 2")

        degree_idx = list(range(1, max_degrees + 1))
        out = xr.concat(
            [da]
            + [
                da**i for i in degree_idx[1:]
            ],  # Avoids computing ds^1 to not add tasks to dask graph when dask-backed data.
            dim=xr.DataArray(degree_idx, dims="degree", name="degree"),
        )
        return out

    def _calculate_beta(ds: xr.Dataset) -> xr.DataArray:
        """
        Helper function to calculate beta from gamma and covariates using a scalable Einstein notation tensordot.
        """
        # The ds["gamma"] has a "covarname" dimension with one element for each of the model's covariates.
        # Coefficient for predictor tas (tas histogram bin labels).
        gamma_1 = ds["gamma"].sel(covarname="1", drop=True)
        # coefficient for climtas covariate and coefficient log of GDP per capita covariate.
        gamma_covar = ds["gamma"].sel(covarname=["climtas", "loggdppc"])
        # The covariates to pair with the coefficients, selecting for the baseline year.
        covar = xr.concat(
            [ds["climtas"], ds["loggdppc"]],
            dim=xr.DataArray(
                ["climtas", "loggdppc"], dims="covarname", name="covarname"
            ),
        )

        # Remember, annual histograms as input use histogram bin labels ("tas_bin") as "tas".
        # Creates a "degree" coordinate and populates it with tas^1, tas^2, tas^3, etc. equal to degrees in polynomial.
        tas = _add_degree_coord(ds["tas_bin"], max_degrees=gamma_1["degree"].size)
        # Do it this way so we don't need to repeat the same math for each degree of the polynomial below.

        #  γ_1 * tas + γ_climtas * climtas * tas + γ_loggdppc * loggdppc * tas
        # term for each of the polynomial degrees (∵ "degree" is a coordinate for variables that vary by degree).
        beta0 = xr.dot(gamma_1, tas, dim=["degree"], optimize=True)
        beta1 = xr.dot(
            gamma_covar, covar, tas, dim=["covarname", "degree"], optimize=True
        )

        return beta0 + beta1

    def _calculate_shifted_baseline_beta(
        ds: xr.Dataset,
        *,
        tas_bin_dim: str = "tas_bin",
        lmmt: float = 10.0,
        ummt: float = 30.0,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """
        Helper to calculate baseline beta and the index of the MMT from gamma and covariates.
        """
        ds = ds.copy()  # So we don't change accidentally change the input data.
        beta = _calculate_beta(ds)

        # Find idx with lowest beta & minimum mortality temperature (MMT) within °C
        # range in the baseline period.
        mmt_beta, mmt_idx = minimum_arg(
            beta,
            dim=tas_bin_dim,
            lmmt=lmmt,
            ummt=ummt,
        )
        # Shift beta so MMT is 0.
        beta -= mmt_beta

        return beta, mmt_idx

    def _mortality_effects_model(ds: xr.Dataset) -> xr.Dataset:
        # dot product of betas and t_bins for effect.
        # Divide by number of esemble members (51) so it's the average forecast effect.
        # TODO: Should make ensemble_n a variable or something rather than hard-coded.
        effect = (ds["histogram_tas"] * ds["beta"]).sum(dim="tas_bin") / 51
        return xr.Dataset({"effect": effect.astype("float32")})

    def _noadapt_beta_from_gamma(ds: xr.Dataset) -> xr.Dataset:
        """
        Calculates mortality impact polynomial model's beta coefficients from gamma coefficients for the no-adaptation scenario.

        Returns a copy of `ds` with new "beta" variable.
        """
        # Subset all the covariate variables to the baseline year because no adaptation is allowed.
        beta, mmt_idx = _calculate_shifted_baseline_beta(ds)

        # Just in case, clip negative values to zero. Sometimes called "level clipping".
        beta = beta.clip(min=0)

        # u-clip. Makes the response function shaped like a big U, centered on the MMT.
        beta = uclip(
            beta.chunk({"tas_bin": -1}),  # Core dim must be in single chunk.
            dim="tas_bin",
            idx_min=mmt_idx,
        )

        # Returns new dataset with beta added as new variable. Not modifying
        # original ds. Also ensure original data is passed through to projection.
        return ds.assign(beta=beta)

    mortality_effect_model = isku.build_projection_template(
        pre=_noadapt_beta_from_gamma,
        project=_mortality_effects_model,
        post=_no_processing,
    )
    return (mortality_effect_model,)


@app.cell
def _(climtas, gammas, histogram_tas, loggdppc, xr):
    # Stick everything together and make sure it aligns and matches. Rechunk all together. Also drop any regions with NaNs.
    input_ds = (
        xr.Dataset(
            {
                "histogram_tas": histogram_tas["histogram_tas"],
                "climtas": climtas["climtas"],
                "loggdppc": loggdppc,
                "gamma": gammas["gamma_mean"],
            }
        )
        .dropna(dim="region")
        .chunk(
            {
                "region": "auto",  # "auto" is a sensible default, but if memory trouble, set this to 1000, 500, 100.
                "time": -1,  # This needs to be all in memory, thus -1.
                "tas_bin": -1,  # This also needs to be all in memory.
                "age_cohort": 1,  # We're doing all age_cohorts at once but could be done one-by-one.
                "degree": -1,  # For gammas and polynomial calculations. Should all be in memory.
            },
        )
        .unify_chunks()
    )

    input_ds
    return (input_ds,)


@app.cell
def _(input_ds):
    # Quick sanity check

    input_ds["histogram_tas"].sel(region="USA.14.608").plot(y="tas_bin", x="time")

    # You can see there are only a few days in December, as the forecast is run for 215 days.


@app.cell
def _(input_ds, isku, mortality_effect_model):
    projected = isku.project(input_ds, model=mortality_effect_model)

    # Not required, just for fun.
    projected["effect"].attrs = {
        "units": "deaths per 100,000 people",
        "long_name": "Temperature mortality",
    }

    projected = projected.compute()
    return (projected,)


@app.cell
def _(plt, projected, sns):
    # Quick plot for an arbitrary region for diagnostics.
    with sns.axes_style("whitegrid"):
        projected["effect"].sel(region="USA.14.608").plot(hue="age_cohort")
    plt.gca()

    # WARNING: Not sure why we're getting values for December...
    # Ahh, because it's 215 days from the beginning of May this year...
    # TODO: Guard against incomplete months.


@app.cell
def _(IMPACT_REGION_POLYGONS, geopandas, plt, projected):
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
        projected["effect"].to_dataframe().reset_index(),
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
            "label": f"{projected['effect'].attrs['long_name']} [{projected['effect'].attrs['units']}]",
            "orientation": "horizontal",
        },
    )
    ax.set_axis_off()
    plt.gca()


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Outstanding:

    [ ] ERA5 baseline?

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


if __name__ == "__main__":
    app.run()
