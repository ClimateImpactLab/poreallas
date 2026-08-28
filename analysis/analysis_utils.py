### Utilities to compute age-weighted mortality impacts relative to a base period
# ### Emily Zuetell
### July 7, 2026

import xarray as xr
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def get_baseline_period(effect_xr, years=30):
    # Baseline
    _max_year = effect_xr["/baseline"]["time"].max().dt.year.item()
    # Subtracting 29 even though we want 30 year baseline because the time slice is an inclusive range.
    _min_year = _max_year - (years - 1)
    baseline_period = slice(str(_min_year), str(_max_year))
    return baseline_period


def pop_weight_sum(
    da, socioeconomics, rate=True, age_weight=True, cohort="age65plus", impact=True
):
    if age_weight:
        # Weighted by cohort_population
        pop_weight = xr.concat(
            [
                socioeconomics["pop0to4"],
                socioeconomics["pop5to64"],
                socioeconomics["pop65plus"],
            ],
            dim=pd.Index(["age0to4", "age5to64", "age65plus"], name="age_cohort"),
        )
        # Total Mortality = rate*pop/100,000
        age_weighted_total = da * pop_weight / 100000

        if rate:
            # Mortality Rate = Total Age-Weighted Mortality/Total Pop (deaths/100k)
            age_weighted_total = age_weighted_total * 100000 / socioeconomics["pop"]
        # Sum across age cohorts for each Impact Region
        regional_sum = age_weighted_total.sum(dim="age_cohort")
        regional_sum.name = "age_weighted_impact" if impact else "age_weighted_effect"
    else:
        # Return individual cohort
        regional_sum = da.sel(age_cohort=cohort)

        if not rate:
            # Total Mortality
            _cohortstem = cohort[3:]
            _col = f"pop{_cohortstem}"
            regional_sum = da.sel(age_cohort=cohort) * socioeconomics[_col]

        regional_sum.name = f"{cohort}_impact" if impact else f"{cohort}_effect"

    return regional_sum


def compute_impact(
    projected,
    socioeconomics,
    baseline_period,
    chunks={"number": -1, "sample": -1, "region": "auto"},
    ensemble=False,
    hotonly=False,
    rate=False,
    age_weight=True,
    cohort="age65plus",
):
    """
    Calculates the difference between forecast and baseline period mortality effects (deaths/100k or total deaths)
    (monthly climatology), then applies population-weighted for age cohorts.

    Parameters
    ----------
    projected : xarray.DataTree
        Container with "/baseline", "/forecast", "/baseline_hotonly", and
        "/forecast_hotonly" groups, each holding an "effect" DataArray.
    socioeconomics : xarray.Dataset or DataArray
        Population and demographic data used for weighting in pop_weight_sum.
    baseline_period : slice or array-like
        Time selector defining the baseline period to average over.
    ensemble : bool, default False
        If True, keep the ensemble ("number") dimension when computing the
        forecast monthly climatology. If False, average over "number" first.
    hotonly : bool, default False
        If True, use the hot-only effect groups instead of the net effect groups.
    rate : bool, default False
        Passed to pop_weight_sum; if True, return mortality rates instead of population-weighted totals.
    age_weight : bool, default True
        Passed to pop_weight_sum; if True, apply age-based weighting.
    cohort : str, default 'age65plus'
        Age cohort to use if NOT weighting.

    Returns
    -------
    xarray.DataArray or Dataset
        Population-weighted regional sum (or rate) of the impact, as returned
        by pop_weight_sum.
    """

    if hotonly == "hotonly":
        # Hotonly
        _baseline = (
            projected["/baseline_hotonly"]["effect"]
            .chunk({"region": "auto"})
            .sel(time=baseline_period)
            .groupby("time.month")
            .mean()
        )
        if ensemble:
            _forecast = (
                projected["/forecast_hotonly"]["effect"]
                .chunk(chunks)
                .groupby("time.month")
                .mean()
            )
        else:
            _forecast = (
                projected["/forecast_hotonly"]["effect"]
                .chunk(chunks)
                .mean(dim="number")
                .groupby("time.month")
                .mean()
            )
    elif hotonly == "coldonly":
        # Coldonly
        _baseline = (
            projected["/baseline_coldonly"]["effect"]
            .chunk({"region": "auto"})
            .sel(time=baseline_period)
            .groupby("time.month")
            .mean()
        )
        if ensemble:
            _forecast = (
                projected["/forecast_coldonly"]["effect"]
                .chunk(chunks)
                .groupby("time.month")
                .mean()
            )
        else:
            _forecast = (
                projected["/forecast_coldonly"]["effect"]
                .chunk(chunks)
                .mean(dim="number")
                .groupby("time.month")
                .mean()
            )
    else:
        ## Net
        _baseline = (
            projected["/baseline"]["effect"]
            .chunk({"region": "auto"})
            .sel(time=baseline_period)
            .groupby("time.month")
            .mean()
        )
        if ensemble:
            # Maintain ensmble dimension
            _forecast = (
                projected["/forecast"]["effect"]
                .chunk(chunks)
                .groupby("time.month")
                .mean()
            )
        else:
            _forecast = (
                projected["/forecast"]["effect"]
                .chunk(chunks)
                .mean(dim="number")
                .groupby("time.month")
                .mean()
            )

    # Compute Impact
    impact = _forecast - _baseline
    # Add metadata
    impact.name = "impact"
    impact.attrs["long_name"] = "Temperature mortality impact"
    impact.attrs["units"] = "Deaths per 100,000 people" if rate else "Deaths"
    impact.attrs["hotonly"] = hotonly

    # Apply population weighting
    regional_sum = pop_weight_sum(
        impact,
        socioeconomics,
        rate=rate,
        age_weight=age_weight,
        cohort=cohort,
        impact=True,
    )
    return regional_sum

def compute_global_impact(impact, socioeconomics, rate=False):
    if rate:
        pop = socioeconomics["population"].sel(region=impact.region)
        return (impact * pop).sum(dim="region") / pop.sum(dim="region")
    return impact.sum(dim="region")

### Analysis Functions ###


def xarray_to_gpd(data, polygons, crs="ESRI:54030"):
    _polygons_data = polygons.merge(
        data.to_dataframe(name=data.name or "value").reset_index(),
        on="region",
    )

    _polygons_data = _polygons_data.cx[:, -60:90].to_crs(crs)

    return _polygons_data


def compute_stats(da, dim="number", polygon=None):
    """
    From an xarray, return statistics along dimension, 'dim'

    Return an xarray or optionally return a geodataframe (if polygon gdf is passed)
    """

    mean = da.mean(dim=dim)
    std = da.std(dim=dim)
    min = da.min(dim=dim)
    max = da.max(dim=dim)
    p = da.quantile([0.10, 0.17, 0.5, 0.83, 0.9], dim=dim)

    p10 = p.sel(quantile=0.10, drop=True)
    p17 = p.sel(quantile=0.17, drop=True)
    p50 = p.sel(quantile=0.50, drop=True)
    p83 = p.sel(quantile=0.83, drop=True)
    p90 = p.sel(quantile=0.90, drop=True)
    likely_range = p83 - p17

    ds_out = xr.Dataset(
        {
            "median": p50,
            "p17": p17,
            "p83": p83,
            "likely_range_IPCC": likely_range,
            "mean": mean,
            "std": std,
            "min": min,
            "max": max,
            "p10": p10,
            "p90": p90,
        }
    )
    if polygon is not None:
        _polygons_num = polygon.merge(
            ds_out.to_dataframe().reset_index(),
            on="region",
        )
        return _polygons_num
    return ds_out


###Output Functions ###
def dataset_to_dataframe(ds):
    if len(ds.dims) == 0:
        return pd.DataFrame({k: [v.values.item()] for k, v in ds.data_vars.items()})
    return ds.to_dataframe().reset_index()

def make_csv(
    effect,
    socioeconomics,
    polygon,
    baseline_period,
    ensemble=True,
    dims = ["number", "sample"],
    months = [8, 9, 10, 11, 12, 1],
    hotonly="net",
    rate=False,
    age_weight=True,
    filename_template="2608_{hotonly}_{scope}_{rate_l}_{stat_scope}stats.csv",
    output_scope = ["regional_monthly", "regional_6mo", "global_monthly", "global_6mo"],
):
    """
    Compute regional and global impact stats from an effects datatree and
    write them out as CSVs.

    Computes impact from `effect`, then generates monthly, 6-month, and
    global summary statistics (region-level via `polygon`, global via
    population-weighted averaging when `rate=True` or summation when
    `rate=False`). All stats are rounded to the nearest integer before
    being written to CSV.

    Parameters
    ----------
    effect : xr.DataArray or xr.Dataset
        Effect data used to compute impact.
    socioeconomics : xr.Dataset
        Socioeconomic data, including population, used for impact
        computation and population-weighted global rates.
    polygon : GeoDataFrame or similar
        Impact region polygons used to group regional stats.
    baseline_period : tuple or list
        Start/end period defining the baseline for impact computation.
    ensemble : bool, default True
        Whether to compute impact across an ensemble of runs or return ensemble mean.
    hotonly : str, default "net"
        Filter for hot-only effects; used in output filenames.
    rate : bool, default False
        If True, compute population-weighted rates instead of totals.
    age_weight : bool, default True
        Whether to apply age weighting in the impact computation.
    filename_template : str, optional
        Template string for output filenames. Supports the placeholders
        {hotonly}, {scope}, {rate_l}, and {stat_scope}.

    Returns
    -------
    None
        Writes regional monthly, regional 6-month, global monthly, and
        global 6-month stats CSVs to disk.
    """

    rate_l = "rate" if rate else "total"

    # Compute Impact from Effect
    impact = compute_impact(
        effect,
        socioeconomics,
        ensemble=ensemble,
        baseline_period=baseline_period,
        hotonly=hotonly,
        rate=rate,
        age_weight=age_weight,
    )
    impact = impact.sel(month = months)  # Only use first 6 months

    ### Stats by Impact Region ###
    # Monthly Stats
    stat_cols = [
        "median",
        "p17",
        "p83",
        "likely_range_IPCC",
        "mean",
        "std",
        "min",
        "max",
        "p10",
        "p90",
    ]
    if "regional_monthly" in output_scope:
        _polygons_impact = compute_stats(impact, dim=dims, polygon=polygon)
        wide = _polygons_impact.pivot(
            index=["region", "ISO"], columns="month", values=stat_cols
        )
        wide.columns = [f"month {m} {stat}" for stat, m in wide.columns]
        stat_col_names = wide.columns.difference(["region", "ISO"])
        wide[stat_col_names] = wide[stat_col_names].round(0).astype("Int64")
        wide = wide.reset_index()
        wide.to_csv(
        filename_template.format(hotonly=hotonly, rate_l=rate_l, scope="all", stat_scope=""),
        index=False,)
    if "regional_6mo" in output_scope:
        # 6-month stats
        mo6 = impact.sum(dim="month")
        _polygons_mo6 = compute_stats(mo6, dim=dims, polygon=polygon)
        mo6_out = _polygons_mo6[
            [
                "region",
                "ISO",
                "median",
                "p17",
                "p83",
                "likely_range_IPCC",
                "mean",
                "std",
                "min",
                "max",
                "p10",
                "p90",
            ]
        ]
        stat_col_names = mo6_out.columns.difference(["region", "ISO"])
        mo6_out[stat_col_names] = mo6_out[stat_col_names].round(0).astype("Int64")
        mo6_out.to_csv(
        filename_template.format(hotonly=hotonly, rate_l=rate_l, scope="6mo", stat_scope=""),
        index=False,
)
    if "global_monthly" in output_scope or "global_6mo" in output_scope:
        ### Global Stats ###
        global_impact = compute_global_impact(impact, socioeconomics, rate=rate)

        if "global_monthly" in output_scope:
            global_monthly = dataset_to_dataframe(compute_stats(global_impact, dim=dims))
            stat_col_names = global_monthly.columns.difference(["region", "ISO"])
            global_monthly[stat_col_names] = global_monthly[stat_col_names].round(0).astype("Int64")
            global_monthly.to_csv(
            filename_template.format(hotonly=hotonly, rate_l=rate_l, scope="global", stat_scope=""),
            index=False,
        )
        if "global_6mo" in output_scope:
            global_mo6 = dataset_to_dataframe(compute_stats(global_impact.sum(dim="month"), dim=dims))
            stat_col_names = global_mo6.columns.difference(["region", "ISO"])
            global_mo6[stat_col_names] = global_mo6[stat_col_names].round(0).astype("Int64")
            global_mo6.to_csv(
            filename_template.format(hotonly=hotonly, rate_l=rate_l, scope="6mo_global", stat_scope=""),
            index=False,
        )
    return


def build_stats_text(da, dim=None, fmt="{:.2f}"):
    stats = {
        "mean": float(da.mean(dim=dim)),
        "std": float(da.std(dim=dim)),
        "min": float(da.min(dim=dim)),
        "max": float(da.max(dim=dim)),
    }
    return "\n".join(f"{k}: {fmt.format(v)}" for k, v in stats.items())


def add_stats_annotation(text, ax, loc="upper left"):
    loc_map = {
        "upper right": (0.975, 1.05, "top", "right"),
        "upper left": (0.005, 1.05, "top", "left"),
        "lower right": (0.975, 0.025, "bottom", "right"),
        "lower left": (0.025, 0.025, "bottom", "left"),
    }
    x, y, va, ha = loc_map[loc]

    return ax.annotate(
        text,
        xy=(x, y),
        xycoords="axes fraction",
        ha=ha,
        va=va,
        fontsize=9,
    )


##### Plotting Functions #####
from functools import lru_cache
import regionmask


# Get and store land data
@lru_cache(maxsize=None)
def _get_land(crs):
    land = gpd.read_file(geodatasets.get_path("naturalearth land"))
    return land.cx[:, -60:90].to_crs(crs)


@lru_cache(maxsize=None)
def _get_land_mask(lon_key, lat_key):
    dummy = xr.DataArray(
        np.zeros((len(lat_key), len(lon_key))),
        coords={"lat": list(lat_key), "lon": list(lon_key)},
        dims=["lat", "lon"],
    )
    return regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(dummy)


def land_only(da):
    da = da.rename({"longitude": "lon", "latitude": "lat"})
    mask = _get_land_mask(tuple(da.lon.values), tuple(da.lat.values))
    return da.where(mask.notnull() & (da.lat > -60))


def get_step(target_range, min_bins=2, max_bins=8):
    magnitude = 10 ** np.floor(np.log10(target_range))
    for _ in range(3):
        for step in [1, 2, 5, 10]:
            candidate = step * magnitude
            n_bins = target_range / candidate
            if min_bins <= n_bins <= max_bins:
                return candidate
        magnitude /= 10
    return magnitude * 10


def nice_step(target):
    """Snap an arbitrary step size to the nearest round value (1, 2, 5, or 10 x 10**n)."""
    magnitude = 10 ** np.floor(np.log10(target))
    options = np.array([1, 2, 5, 10]) * magnitude
    return options[np.argmin(np.abs(options - target))]


def round_bounds(bounds, step):
    """Remove floating-point drift so bounds land exactly on multiples of step/2."""
    return np.round(bounds * 2 / step) * step / 2


def get_ticks(bounds, step=None, max_ticks=10, symmetric=False):
    if symmetric:
        absmax = bounds.max()
        half = step * np.arange(0, int(np.floor(absmax / step)) + 1)
        stride = int(np.ceil((2 * len(half) - 1) / max_ticks))
        half = half[::stride]
        return np.unique(np.concatenate([-half, half]))
    else:
        stride = int(np.ceil(len(bounds) / max_ticks))
        return bounds[::stride]


def make_cmap(bounds, cm="bwr"):
    cmap_base = plt.get_cmap(cm)
    colors = cmap_base(np.linspace(0, 1, len(bounds) - 1))

    diverging_cmaps = {
        "bwr",
        "seismic",
        "coolwarm",
        "RdBu",
        "RdYlBu",
        "PiYG",
        "PRGn",
        "BrBG",
    }
    base_name = cm[:-2] if cm.endswith("_r") else cm
    if base_name in diverging_cmaps:
        center_idx = len(colors) // 2
        colors[center_idx] = [0.95, 0.95, 0.95, 1]

    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    return cmap, norm, sm


def build_colormap(gdf=None, col=None, cm="bwr", vmin=None, vmax=None, n_colors=None):
    # Build discrete colormap, branching on diverging vs sequential
    diverging_cmaps = {
        "bwr",
        "seismic",
        "coolwarm",
        "RdBu",
        "RdYlBu",
        "PiYG",
        "PRGn",
        "BrBG",
    }
    base_name = cm[:-2] if cm.endswith("_r") else cm

    if base_name in diverging_cmaps:
        # Symmetric around zero, with a white center band
        if vmin is not None or vmax is not None:
            absmax = max(abs(vmin), abs(vmax))
        else:
            absmax = math.ceil(gdf[col].abs().quantile(0.95))

        step = nice_step((2 * absmax) / n_colors) if n_colors else get_step(2 * absmax)
        bounds = np.arange(
            -np.ceil(absmax / step) * step - step / 2,
            np.ceil(absmax / step) * step + step,
            step,
        )
        bounds = round_bounds(bounds, step)
        cmap, norm, sm = make_cmap(bounds, cm=cm)
    else:
        # Sequential, uses vmin/vmax (or data min/max) directly, no center-forcing
        if vmin is not None and vmax is not None:
            lo, hi = vmin, vmax
        else:
            lo = vmin if vmin is not None else gdf[col].min()
            hi = vmax if vmax is not None else gdf[col].max()

        step = nice_step((hi - lo) / n_colors) if n_colors else get_step(hi - lo)
        bounds = np.arange(
            np.floor(lo / step) * step, np.ceil(hi / step) * step + step, step
        )
        bounds = round_bounds(bounds, step)
        cmap = plt.get_cmap(cm, len(bounds) - 1)
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    ticks = get_ticks(bounds, step=step, symmetric=(base_name in diverging_cmaps))
    return cmap, norm, sm, ticks, step


import math
import geodatasets

land = gpd.read_file(geodatasets.get_path("naturalearth land"))


def plot_single(
    gdf,
    col,
    sup_title="",
    save_title="",
    cm="bwr",
    cbar_label=None,
    vmin=None,
    vmax=None,
    edgecolor=None,
    linewidth=0,
    ax=None,
    cbar_location="right",
    colorbar=True,
    n_colors=None,
    annotation=None,
    target_crs="ESRI:54030",
):

    if gdf.crs is None:
        raise ValueError("gdf has no CRS set")
    if gdf.crs.to_string() != target_crs:
        gdf = gdf.to_crs(target_crs)

    land = _get_land(target_crs)

    cmap, norm, sm, ticks, step = build_colormap(
        gdf, col, cm=cm, vmin=vmin, vmax=vmax, n_colors=n_colors
    )

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    # Plot background land
    land.plot(color="lightgray", edgecolor="lightgray", ax=ax)
    # Plot data
    gdf.plot(
        column=col,
        legend=False,
        ax=ax,
        cmap=cmap,
        norm=norm,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    ax.set_axis_off()

    if colorbar:
        orientation = "horizontal" if cbar_location in ("bottom", "top") else "vertical"
        cb = fig.colorbar(
            sm,
            ax=ax,
            location=cbar_location,
            orientation=orientation,
            shrink=0.6,
            ticks=ticks,
            label=cbar_label,
        )
        decimals = max(0, int(np.ceil(-np.log10(step)))) if step < 1 else 0
        cb.set_ticklabels([f"{t:.{decimals}f}" for t in ticks])
        for label in cb.ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")

    if annotation:
        add_stats_annotation(annotation, ax)

    if standalone and save_title:
        fig.savefig(save_title, dpi=600, bbox_inches="tight")
    return ax


def plot_monthly(
    gdf,
    col,
    sup_title="",
    save_title="",
    cm="bwr",
    cbar_label=None,
    vmin=None,
    vmax=None,
    edgecolor=None,
    n_colors=None,
    linewidth=0,
    month_order=None,
):
    if vmin is None:
        vmin = -gdf[col].abs().quantile(0.95)
    if vmax is None:
        vmax = gdf[col].abs().quantile(0.95)

    if month_order is None:
        months = sorted(gdf["month"].unique())
    else:
        present = set(gdf["month"].unique())
        months = [m for m in month_order if m in present]

    fig, axes = plt.subplots(2, 3, figsize=(16, 6))

    for ax, month in zip(axes.flat, months):
        group = gdf[gdf["month"] == month]
        plot_single(
            group,
            col,
            cm=cm,
            vmin=vmin,
            vmax=vmax,
            edgecolor=edgecolor,
            linewidth=linewidth,
            ax=ax,
            colorbar=False,
            n_colors=n_colors,
        )
        ax.set_title(f"Month {month}")

    for ax in axes.flat[len(months) :]:
        ax.set_axis_off()

    cmap, norm, sm, ticks, step = build_colormap(
        gdf, col, cm=cm, vmin=vmin, vmax=vmax, n_colors=n_colors
    )
    fig.colorbar(
        sm,
        ax=axes.ravel().tolist(),
        location="right",
        shrink=0.6,
        ticks=ticks,
        label=cbar_label,
    )

    fig.suptitle(sup_title, fontsize=14)
    fig.savefig(save_title, dpi=600, bbox_inches="tight")
    return fig


def plot_aggregate(
    gdf,
    col,
    agg="sum",
    rate=True,
    sup_title="",
    save_title="",
    cm="bwr",
    cbar_label=None,
    vmin=None,
    vmax=None,
    edgecolor=None,
    linewidth=0,
    ax=None,
    cbar_location="right",
    colorbar=True,
    n_colors=None,
    annotation=None,
    target_crs="ESRI:54030",
):

    annual = gdf.groupby(["region", "geometry"], as_index=False)[col].agg(agg)
    annual = gpd.GeoDataFrame(annual, geometry="geometry", crs=gdf.crs)

    ax = plot_single(
        annual,
        col,
        cm=cm,
        ax=ax,
        cbar_label=cbar_label,
        vmin=vmin,
        vmax=vmax,
        edgecolor=edgecolor,
        linewidth=linewidth,
        cbar_location=cbar_location,
        colorbar=colorbar,
        n_colors=n_colors,
        annotation=annotation,
        target_crs=target_crs,
    )

    fig = ax.figure
    fig.suptitle(sup_title, fontsize=14)
    fig.savefig(save_title, dpi=600, bbox_inches="tight")
    return fig
