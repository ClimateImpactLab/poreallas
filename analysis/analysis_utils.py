
### Utilities to compute age-weighted mortality impacts relative to a base period
# ### Emily Zuetell
### July 7, 2026

import xarray as xr
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def get_baseline_period(effect_xr, years = 30):
    #Baseline
    _max_year = effect_xr["/baseline"]["time"].max().dt.year.item()
    # Subtracting 29 even though we want 30 year baseline because the time slice is an inclusive range.
    _min_year = _max_year - (years-1)
    baseline_period = slice(str(_min_year), str(_max_year))
    return baseline_period


def pop_weight_sum(da, socioeconomics, rate=True, age_weight=True, cohort = 'age65plus', impact = True):
    if age_weight:
        # Weighted by cohort_population
        pop_weight = xr.concat(
            [socioeconomics['pop0to4'], socioeconomics['pop5to64'], socioeconomics['pop65plus']],
            dim=pd.Index(['age0to4', 'age5to64', 'age65plus'], name='age_cohort')
        )
        # Total Mortality = rate*pop/100,000
        age_weighted_total = da * pop_weight / 100000

        if rate:
        # Mortality Rate = Total Age-Weighted Mortality/Total Pop (deaths/100k)
            age_weighted_total = age_weighted_total * 100000 / socioeconomics['pop']
        #Sum across age cohorts for each Impact Region
        regional_sum = age_weighted_total.sum(dim='age_cohort')
        regional_sum.name = "age_weighted_impact" if impact else "age_weighted_effect"
    else:
        # Return individual cohort
        regional_sum = da.sel(age_cohort=cohort)

        if not rate:
        # Total Mortality
            _cohortstem = cohort[3:]
            _col = f"pop{_cohortstem}"
            regional_sum = da.sel(age_cohort=cohort)*socioeconomics[_col]

        regional_sum.name = f"{cohort}_impact" if impact else f"{cohort}_effect"
        
    return regional_sum

def compute_impact(projected, socioeconomics, baseline_period, ensemble = False, hotonly = False, rate = False, age_weight = True, cohort = 'age65plus'):
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

    if hotonly:
        #Hotonly
        _baseline = (
            projected["/baseline_hotonly"]["effect"]
            .sel(time=baseline_period)
            .groupby("time.month")
            .mean()
        )
        if ensemble:
            _forecast = (
                projected["/forecast_hotonly"]["effect"].groupby("time.month").mean()
            )
        else:
            _forecast = (
                projected["/forecast_hotonly"]["effect"].mean(dim="number").groupby("time.month").mean()
            )
    else:
        ## Net
        _baseline = (
            projected["/baseline"]["effect"]
            .sel(time=baseline_period)
            .groupby("time.month")
            .mean()
        )
        if ensemble:
        #Maintain ensmble dimension
            _forecast = (
                projected["/forecast"]["effect"].groupby("time.month").mean()
            )
        else:
            _forecast = (
                projected["/forecast"]["effect"].mean(dim="number").groupby("time.month").mean()
            )

    # Compute Impact
    impact = _forecast - _baseline
    # Add metadata
    impact.name = "impact"
    impact.attrs["long_name"] = "Temperature mortality impact"
    impact.attrs["units"] = "Deaths per 100,000 people" if rate else "Deaths"
    impact.attrs['hotonly'] = hotonly

    # Apply population weighting
    regional_sum = pop_weight_sum(impact, socioeconomics, rate=rate, age_weight=age_weight, cohort = cohort, impact = True)
    return regional_sum

### Analysis Functions ###

def xarray_to_gpd(data, polygons, crs = 'ESRI:54030'):
    _polygons_data = polygons.merge(
        data.to_dataframe().reset_index(),
        on="region",
        )

    _polygons_data = _polygons_data.cx[:, -60:90].to_crs(crs)

    return _polygons_data

def compute_stats(da, dim = 'number', polygon = None):
    """
    From an xarray, return statistics along dimension, 'dim'

    Return an xarray or optionally return a geodataframe (if polygon gdf is passed)
    """

    mean = da.mean(dim=dim)
    std = da.std(dim = dim)
    p10 = da.quantile(0.10, dim= dim)
    p90 = da.quantile(0.90, dim=dim)
    prange = p90 - p10

    ds_out = xr.Dataset({
        "mean": mean,
        "std": std,
        "p10": p10.drop_vars("quantile"),
        "p90": p90.drop_vars("quantile"),
        "range": prange.drop_vars("quantile", errors="ignore"),
    })
    if polygon is not None:
        _polygons_num = polygon.merge(ds_out
            .to_dataframe()
            .reset_index(),
            on="region",)
        return _polygons_num
    return ds_out

##### Plotting Functions #####
from functools import lru_cache

#Get and store land data
@lru_cache(maxsize=None)
def _get_land(crs):
    land = gpd.read_file(geodatasets.get_path('naturalearth land'))
    return land.cx[:, -60:90].to_crs(crs)

#Monthly Effects#
def get_step(absmax, min_bins=5, max_bins=10):
    magnitude = 10 ** np.floor(np.log10(absmax))
    for _ in range(3):  
        for step in [1, 2, 2.5, 5, 10]:
            candidate = step * magnitude
            n_bins = absmax / candidate
            if min_bins <= n_bins <= max_bins:
                return candidate
        magnitude /= 10
    return magnitude * 10

def get_ticks(bounds, max_ticks=10, symmetric=False):
    if symmetric:
        half = np.sort(bounds[bounds >= 0])
        stride = int(np.ceil((2 * len(half) - 1) / max_ticks))
        half = half[::stride]
        return np.unique(np.concatenate([-half, half]))
    else:
        stride = int(np.ceil(len(bounds) / max_ticks))
        return bounds[::stride]

def make_cmap(bounds, cm='bwr'):
    cmap_base = plt.get_cmap(cm)
    colors = cmap_base(np.linspace(0, 1, len(bounds) - 1))

    diverging_cmaps = {'bwr', 'seismic', 'coolwarm', 'RdBu', 'RdYlBu', 'PiYG', 'PRGn', 'BrBG'}
    base_name = cm[:-2] if cm.endswith('_r') else cm
    if base_name in diverging_cmaps:
        center_idx = len(colors) // 2
        colors[center_idx] = [0.95, 0.95, 0.95, 1]

    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    return cmap, norm, sm

def build_colormap(gdf=None, col=None, cm='bwr', vmin=None, vmax=None):
    # Build discrete colormap, branching on diverging vs sequential
    diverging_cmaps = {'bwr', 'seismic', 'coolwarm', 'RdBu', 'RdYlBu', 'PiYG', 'PRGn', 'BrBG'}
    base_name = cm[:-2] if cm.endswith('_r') else cm

    if base_name in diverging_cmaps:
        # Symmetric around zero, with a white center band
        if vmin is not None or vmax is not None:
            absmax = max(abs(vmin), abs(vmax))
        else:
            absmax = math.ceil(gdf[col].abs().quantile(0.95))

        step = get_step(2 * absmax)
        bounds = np.arange(-np.ceil(absmax / step) * step - step / 2, np.ceil(absmax / step) * step + step, step)
        cmap, norm, sm = make_cmap(bounds, cm=cm)
    else:
        # Sequential, uses vmin/vmax (or data min/max) directly, no center-forcing
        if vmin is not None and vmax is not None:
            lo, hi = vmin, vmax
        else:
            lo = vmin if vmin is not None else gdf[col].min()
            hi = vmax if vmax is not None else gdf[col].max()

        step = get_step(hi - lo)
        bounds = np.arange(np.floor(lo / step) * step, np.ceil(hi / step) * step + step, step)
        cmap = plt.get_cmap(cm, len(bounds) - 1)
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    ticks = get_ticks(bounds, symmetric=(base_name in diverging_cmaps))
    return cmap, norm, sm, ticks

import math
import geodatasets
land = gpd.read_file(geodatasets.get_path('naturalearth land'))

def plot_single(gdf, col, sup_title="", save_title="", cm='bwr', 
                cbar_label=None, vmin=None, vmax=None, edgecolor=None, linewidth=0, 
                ax=None, cbar_location='right', colorbar=True,
                target_crs = 'ESRI:54030'):

    if gdf.crs is None:
        raise ValueError("gdf has no CRS set")
    if gdf.crs.to_string() != target_crs:
        gdf = gdf.to_crs(target_crs)

    land = _get_land(target_crs)
        
    cmap, norm, sm, ticks = build_colormap(gdf, col, cm=cm, vmin=vmin, vmax=vmax)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    #Plot background land
    land.plot(color='lightgray', edgecolor='lightgray', ax=ax)
    #Plot data
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
        orientation = 'horizontal' if cbar_location in ('bottom', 'top') else 'vertical'
        cb = fig.colorbar(sm, ax=ax, location=cbar_location, orientation=orientation, shrink=0.6, ticks=ticks, label=cbar_label)
        cb.set_ticklabels([f"{0 if t == 0 else t:.0f}" for t in ticks])
        for label in cb.ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha('right')

    if standalone and save_title:
        fig.savefig(save_title, dpi=600, bbox_inches="tight")
    return ax

def plot_monthly(gdf, col, sup_title="", save_title="", cm='bwr', cbar_label=None, vmin=None, vmax=None, edgecolor=None, linewidth=0):
    if vmin is None:
        vmin = -gdf[col].abs().quantile(0.95)
    if vmax is None:
        vmax = gdf[col].abs().quantile(0.95)

    months = sorted(gdf['month'].unique())
    fig, axes = plt.subplots(2, 3, figsize=(16, 6))

    for ax, month in zip(axes.flat, months):
        group = gdf[gdf['month'] == month]
        plot_single(
            group, col, cm=cm, vmin=vmin, vmax=vmax,
            edgecolor=edgecolor, linewidth=linewidth, ax=ax, colorbar=False,
        )
        ax.set_title(f'Month {month}')

    for ax in axes.flat[len(months):]:
        ax.set_axis_off()

    _, _, sm, ticks = build_colormap(gdf, col, cm=cm, vmin=vmin, vmax=vmax)
    fig.colorbar(sm, ax=axes.ravel().tolist(), location='right', shrink=0.6, ticks=ticks, label=cbar_label)

    fig.suptitle(sup_title, fontsize=14)
    fig.savefig(save_title, dpi=600, bbox_inches="tight")
    return fig


def plot_aggregate(gdf, col, agg='sum', rate=True, sup_title="", save_title="", cm='bwr',
                   cbar_label=None, vmin=None, vmax=None, edgecolor=None, linewidth=0, 
                    ax=None, cbar_location='right', colorbar=True,
                    target_crs = 'ESRI:54030'):
    
    annual = gdf.groupby(['region', 'geometry'], as_index=False)[col].agg(agg)
    annual = gpd.GeoDataFrame(annual, geometry='geometry', crs=gdf.crs)

    ax = plot_single(annual, col,cm=cm, ax=ax, cbar_label=cbar_label, 
                      vmin=vmin, vmax=vmax, edgecolor=edgecolor, linewidth=linewidth, 
                    cbar_location=cbar_location, colorbar=colorbar,
                    target_crs = target_crs)

    fig = ax.figure
    fig.suptitle(sup_title, fontsize=14)
    fig.savefig(save_title, dpi=600, bbox_inches="tight")
    return fig