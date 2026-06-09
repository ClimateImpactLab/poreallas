"""
Logic for projecting mortality effects and impacts

"""

import isku
import numba
import numpy as np
import xarray as xr


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


def _add_degree_coord(da: xr.DataArray, max_degrees: int) -> xr.DataArray:
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
        dim=xr.DataArray(["climtas", "loggdppc"], dims="covarname", name="covarname"),
    )

    # Remember, annual histograms as input use histogram bin labels ("tas_bin") as "tas".
    # Creates a "degree" coordinate and populates it with tas^1, tas^2, tas^3, etc. equal to degrees in polynomial.
    tas = _add_degree_coord(ds["tas_bin"], max_degrees=gamma_1["degree"].size)
    # Do it this way so we don't need to repeat the same math for each degree of the polynomial below.

    #  γ_1 * tas + γ_climtas * climtas * tas + γ_loggdppc * loggdppc * tas
    # term for each of the polynomial degrees (∵ "degree" is a coordinate for variables that vary by degree).
    beta0 = xr.dot(gamma_1, tas, dim=["degree"], optimize=True)
    beta1 = xr.dot(gamma_covar, covar, tas, dim=["covarname", "degree"], optimize=True)

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
    effect = (ds["histogram_tas"] * ds["beta"]).sum(dim="tas_bin")
    return xr.Dataset({"effect": effect.astype("float32")})


def _noadapt_beta_from_gamma(ds: xr.Dataset) -> xr.Dataset:
    """
    Calculates mortality impact polynomial model's beta coefficients from gamma coefficients for the no-adaptation scenario.

    Returns a copy of `ds` with new "beta" variable.
    """
    ds = ds.copy()

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

    beta.attrs = {
        "units": "deaths per 100,000 people",
        "long_name": "Temperature mortality rate",
    }

    # Returns new dataset with beta added as new variable. Not modifying
    # original ds. Also ensure original data is passed through to projection.
    return ds.assign(beta=beta)


mortality_effect_model = isku.build_projection_template(
    pre=_noadapt_beta_from_gamma,
    project=_mortality_effects_model,
    post=_no_processing,
)
