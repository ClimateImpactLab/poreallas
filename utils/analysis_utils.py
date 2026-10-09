### Utilities to compute age-weighted mortality impacts relative to a base period
# ### Emily Zuetell
### July 7, 2026

from dataclasses import dataclass
from functools import cache

import cil_regionalization as cilreg
import geodatasets
import geopandas as gpd
import numpy as np
import pandas as pd
import regionmask
import xarray as xr
from cil_regionalization.config import SourceUnitPolicies
from scipy import stats


@dataclass
class ImpactConfig:  # Class for impact computation
    version: str
    baseline_period: slice
    polygons_path: str = None  # path to polygons parquet file
    socioeconomics_path: str = None  # path to socioeconomics zarr store
    regions_path: str = None  # segment weights
    socioeconomics: object = None  # xr.Dataset; loaded in __post_init__ if None
    polygons: object = None  # geopandas.GeoDataFrame; loaded in __post_init__ if None
    dims: list = None  # Dims preserved for uncertainty
    months: list = None  # Forecast Months (6-months) #### todo get rid of defaults
    hotonly: str = "net"  # "hotonly", "coldonly", or "net"
    rate: bool = False  # "False" = Total Deaths, "True" = Mortality Rate
    age_weight: bool = True  # Age-cohort weighting
    cohort: str = "age65plus"

    def __post_init__(self):
        if self.polygons is None:  # Load polygons gpd
            self.polygons = (
                gpd.read_parquet(self.polygons_path)
                .rename(columns={"hierid": "region"})
                .set_index("region")
                .set_crs(epsg=4326)  # Assuming the data is WGS-84.
            )

        if self.socioeconomics is None:  # Load socioeconomics xarray
            self.socioeconomics = xr.open_zarr(self.socioeconomics_path).sel(year=2026)[
                ["pop0to4", "pop5to64", "pop65plus", "pop", "gdppc", "iso3"]
            ]


def pop_weight_sum(da, config, impact=True):
    """
    Parameters
    ----------
    da : xr.DataArray
        Mortality effect array (deaths per 100,000) with an 'age_cohort'
        dimension when config.age_weight is False.
    config : ImpactConfig
        Provides age_weight, rate, cohort, and socioeconomics.
    impact : bool, default True
        If True, names the output "*_impact"; if False, names it "*_effect".
        Distinguishes a computed impact (relative to baseline) from a raw
        effect value.

    Returns
    -------
    xr.DataArray
        Population-weighted mortality total or rate, named according to
        `impact` and the weighting strategy used.
    """

    def _pop_weight_total():
        if config.age_weight:
            # Age Weight = Mortality Rate * Pop Share * Total Pop/100k
            age_weighted_total = da * pop_weight * total_pop / 100000
            # Sum over age_cohort for each region (sum(pop_weight = 1))
            age_sum = age_weighted_total.sum(dim="age_cohort")
            age_sum.name = "age_weighted_impact" if impact else "age_weighted_effect"
        else:
            cohort_rate = da.sel(age_cohort=config.cohort)
            # Population of specified age cohort
            pop_col = f"pop{config.cohort[3:]}"
            # Total deaths = rate*pop/100k
            age_sum = cohort_rate * config.socioeconomics[pop_col] / 100000
            age_sum.name = (
                f"{config.cohort}_impact" if impact else f"{config.cohort}_effect"
            )
        return age_sum

    def _pop_weight_rate():
        if config.age_weight:
            # Age-weighted rate = sum(rate * popshare)
            age_weighted_rate = da * pop_weight
            age_sum = age_weighted_rate.sum(dim="age_cohort")
            age_sum.name = "age_weighted_impact" if impact else "age_weighted_effect"
        else:
            # Return selected age_cohort
            age_sum = da.sel(age_cohort=config.cohort)
            age_sum.name = (
                f"{config.cohort}_impact" if impact else f"{config.cohort}_effect"
            )
        return age_sum

    if config.age_weight:
        total_pop = config.socioeconomics["pop"]
        # Compute pop share (cohort_pop/total_pop)
        pop_weight = (
            xr.concat(
                [
                    config.socioeconomics["pop0to4"],
                    config.socioeconomics["pop5to64"],
                    config.socioeconomics["pop65plus"],
                ],
                dim=pd.Index(["age0to4", "age5to64", "age65plus"], name="age_cohort"),
            )
            / total_pop
        )

    return _pop_weight_rate() if config.rate else _pop_weight_total()


def compute_impact(
    projected,
    config,
    chunks=None,
    ensemble=False,
    hotonly=None,
):
    """
    Parameters
    ----------
    projected : xarray.DataTree
        Datatree with "/baseline", "/forecast", and their "_hotonly"/
        "_coldonly" leaves, each holding an "effect" DataArray (deaths/100k) with
        a "time" dimension.
    config : ImpactConfig
        Provides baseline_period, rate, and the default hotonly value.
    chunks : dict, default {"number": -1, "sample": -1, "region": "auto"}
        Chunk sizes applied to the forecast effect array.
    ensemble : bool, default False
        If True, keep the "number" (ensemble) dimension in the forecast
        climatology. If False, average over "number" before computing
        impact.
    hotonly : str, optional
        Overrides config.hotonly for this call only, without mutating the
        config. Must be one of "net", "hotonly", or "coldonly" if provided.

    Returns
    -------
    xr.DataArray
        Population-weighted impact, as returned by pop_weight_sum.

    Raises
    ------
    ValueError
        If the resolved hotonly value is not one of "net", "hotonly", or "coldonly".
    """
    if chunks is None:
        chunks = {"number": -1, "sample": -1, "region": "auto"}

    # Per-call override takes priority over the config default (helpful for multi-option plotting)
    hotonly = hotonly if hotonly is not None else config.hotonly

    # Check for valid chunks based on dims present
    # Find datatree leaves
    effect_dims = {
        dim
        for node in projected.subtree
        if "effect" in node.data_vars
        for dim in node["effect"].dims
    }
    if not effect_dims:
        raise ValueError(
            "No 'effect' variable found in any node of the input DataTree."
        )

    valid_chunks = {k: v for k, v in chunks.items() if k in effect_dims}

    # Check validity of "hotonly" item
    valid_terms = ["net", "hotonly", "coldonly"]
    if hotonly not in valid_terms:
        raise ValueError(f"Invalid term: {hotonly!r}. Must be one of {valid_terms}")

    # Select leaf according to "hotonly" flag
    group = {"net": "", "hotonly": "_hotonly", "coldonly": "_coldonly"}[hotonly]

    # Monthly mean over baseline period
    _baseline = (
        projected[f"/baseline{group}"]["effect"]
        .chunk({"region": "auto"})
        .sel(
            time=config.baseline_period
        )  # Select configured baseline period [Should typically be 30yrs]
        .groupby("time.month")  # Group all years by month
        .mean()  # Monthly average
    )
    _forecast = projected[f"/forecast{group}"]["effect"].chunk(valid_chunks)

    if not ensemble:
        # Return ensemble mean
        _forecast = _forecast.mean(dim="number")
    # Build "month" dimension (redundant because forecast only includes 1 year)
    _forecast = _forecast.groupby("time.month").mean()
    # Compute Impact (Forecast Effect - Baseline Effect)
    impact = _forecast - _baseline
    # Save attributes
    impact.name = "impact"
    impact.attrs["long_name"] = "Temperature mortality impact"
    impact.attrs["units"] = "Deaths per 100,000 people" if config.rate else "Deaths"
    impact.attrs["hotonly"] = hotonly

    return pop_weight_sum(impact, config)


def get_baseline_period(effect_xr, years=30):
    # Baseline
    _max_year = effect_xr["/baseline"]["time"].max().dt.year.item()
    # Subtracting 29 even though we want 30 year baseline because the time slice is an inclusive range.
    _min_year = _max_year - (years - 1)
    baseline_period = slice(str(_min_year), str(_max_year))
    return baseline_period


def compute_global_impact(impact, socioeconomics, rate, group_dim="region"):
    # Compute the global impact region by summing across spatial "group_dim"
    if rate:
        if group_dim == "region":
            # Pop-weight each region
            pop = socioeconomics["pop"].sel(region=impact.region)
        else:
            raise ValueError("Pop-Weighting only available for region group")
            # TODO Aggregate population to other group levels
        return (impact * pop).sum(dim=group_dim) / pop.sum(dim=group_dim)
    return impact.sum(dim=group_dim)


### Analysis Functions ###
def xarray_to_gpd(data, polygons, crs="ESRI:54030"):
    # Merge xarray with dim "region" with Impact Region Geopandas
    _polygons_data = polygons.merge(
        data.to_dataframe(name=data.name or "value").reset_index(),
        on="region",
    )
    # Clip Antarctica
    _polygons_data = _polygons_data.cx[:, -60:90].to_crs(crs)

    return _polygons_data


def crossed_sem(da, a="number", b="sample"):
    na, nb = da.sizes[a], da.sizes[b]
    grand = da.mean((a, b))
    m_a = da.mean(b)
    m_b = da.mean(a)

    resid = da - m_a - m_b + grand
    var_e = (resid**2).sum((a, b)) / ((na - 1) * (nb - 1))

    var_mean = m_a.var(a, ddof=1) / na + m_b.var(b, ddof=1) / nb - var_e / (na * nb)
    return np.sqrt(var_mean.clip(min=0))


def compute_stats(da, dim=("number", "sample")):
    """
    From an xarray, return statistics over the two crossed dimensions in 'dim' as an xr.Dataset.
    """
    mean = da.mean(dim=dim)
    std = da.std(dim=dim)
    sem = crossed_sem(da, a=dim[0], b=dim[1])
    min = da.min(dim=dim)
    max = da.max(dim=dim)
    p = da.quantile([0.10, 0.17, 0.5, 0.83, 0.9], dim=dim)

    p10 = p.sel(quantile=0.10, drop=True)
    p17 = p.sel(quantile=0.17, drop=True)
    p50 = p.sel(quantile=0.50, drop=True)
    p83 = p.sel(quantile=0.83, drop=True)
    p90 = p.sel(quantile=0.90, drop=True)
    likely_range = p83 - p17

    return xr.Dataset(
        {
            "median": p50,
            "p17": p17,
            "p83": p83,
            "likely_range_IPCC": likely_range,
            "mean": mean,
            "std": std,
            "sem": sem,
            "min": min,
            "max": max,
            "p10": p10,
            "p90": p90,
        }
    )


def crossed_ttest(d, member_dim="number", sample_dim="sample"):
    """Crossed-design test of mean(d) = 0 over member_dim x sample_dim,
    vectorized over any remaining dims (e.g. region). Returns t, df, p as DataArrays."""
    M, N = d.sizes[member_dim], d.sizes[sample_dim]
    both = [member_dim, sample_dim]
    # grand mean
    g = d.mean(both)
    # member means
    rm = d.mean(sample_dim)
    # sample means
    cn = d.mean(member_dim)
    # MN * var(g)
    ms_a = N * rm.var(member_dim, ddof=1)
    ms_c = M * cn.var(sample_dim, ddof=1)
    resid = d - rm - cn + g
    ms_e = resid.var(both, ddof=M + N - 1)
    # Check for negative ms_a, ms_c values
    ms_a, ms_c = np.maximum(ms_a, ms_e), np.maximum(ms_c, ms_e)
    # MN * var
    num = ms_a + ms_c - ms_e
    # t-statistic
    # g/sqrt(var)
    t = g / np.sqrt(num / (M * N))
    # effective degrees of freedom
    df = min(M - 1, N - 1)
    # p-value (two-sided)
    p = xr.apply_ufunc(lambda t, df: 2 * stats.t.sf(np.abs(t), df), t, df)
    return p


def merge_polygon_stats(stats_ds, polygon, merge_key="region"):
    """Merge a stats Dataset onto a polygon GeoDataFrame on merge_key."""
    return polygon.merge(stats_ds.to_dataframe().reset_index(), on=merge_key)


### Regionalization Functions ###
def redistribute_adm1(impact, config, chunk_size=4):

    value_name = impact.name or "value"
    # List of impact regions
    col_regions = {(h,) for h in impact["region"].values}

    # Total Deaths: "per_source", "kind = extensive"
    # Death Rates: "per_destination", "kind = intensive"
    kind = "intensive" if config.rate else "extensive"
    # Get weights from pre-defined files
    weights = cilreg.fetch_weights(
        "gadm41-adm1-per-destination" if config.rate else "gadm41-adm1-per-source"
    )
    # store chunked results
    results = []
    # Check for ensemble members
    numbers = impact["number"].values if "number" in impact.dims else [None]
    # Process for each ensemble member chunk to manage memory (Other option could be dask-pandas)
    for start in range(0, len(numbers), chunk_size):
        chunk = (
            impact.isel(number=slice(start, start + chunk_size))
            if "number" in impact.dims
            else impact
        )
        # xarray to dataframe
        df_chunk = xarray_to_gpd(chunk, config.polygons)
        # Align region names with weights file
        df_chunk = df_chunk.rename(columns={"region": "hierid", value_name: "value"})
        # keep only region, value, and dim columns
        keep_cols = ["hierid", "value"] + list(config.dims)
        if "month" in df_chunk.columns:
            keep_cols.append("month")
        df_chunk = df_chunk[keep_cols]
        # Apply weights from cilreg
        out = cilreg.apply_weights(
            weights,  # pre-computed from input file
            df_chunk,
            kind=kind,
            weight="pop",
            value_col="value",
            data_version="world-combo-201710",
            restrict_to_sources=col_regions,
            allow_partial_coverage=True,
            policies=SourceUnitPolicies(
                on_unmatched="skip",
                on_zero_weight="skip",  # handle zero pop
                on_absent_from_data="skip",  # handel Antarctica ISO
            ),
        ).frame
        results.append(out)
        del df_chunk, out  # free memory before next iteration

    adm1 = pd.concat(results, ignore_index=True)
    adm1["GID_1"] = adm1["GID_1"].astype(str)
    # Manage National-level ADM1 (Typically missing an GID_1, fill with GID_0)
    mask = adm1["GID_1"] == ""
    adm1.loc[mask, "GID_1"] = adm1.loc[mask, "GID_0"]

    index_cols = ["GID_1"] + list(config.dims)
    if "month" in adm1.columns:
        index_cols.append("month")
    # Add back GID_0 lookup for alignment (if it gets left in the dataframe you get GID_0xGID_1)
    gid0_lookup = adm1.drop_duplicates("GID_1").set_index("GID_1")["GID_0"]

    # Transform dataframe back to xarray
    da = adm1.set_index(index_cols)["value"].to_xarray()
    da = da.assign_coords(
        GID_0=("GID_1", gid0_lookup.reindex(da["GID_1"].values).values)
    )

    return da


def aggregate_by_iso(ds, polygon, operation="sum"):
    # Add ISO (from shapefile gpd) to xarray
    # align ISO and Impact region dataframes
    iso_map = polygon["ISO"].reindex(ds["region"].values)
    ds = ds.assign_coords(ISO=("region", iso_map.to_numpy(dtype=object)))

    grouped = ds.swap_dims({"region": "ISO"}).groupby("ISO")
    if operation == "sum":
        # Sum total deaths
        ds = grouped.sum(dim="ISO")
    else:
        raise ValueError(f"Unsupported operation: {operation!r}")
    # Dissolve Impact Region geometry to ISO level
    polygon = polygon.copy()
    polygon["geometry"] = polygon["geometry"].buffer(0)
    polygon = polygon.dissolve(by="ISO", aggfunc="first")

    return ds, polygon


def aggregate_impact(impact, config, group_level):  # Needs validation
    if group_level == "IR":
        # Add ISO (from shapefile gpd) to xarray to match data across spatial resolutions
        # align ISO and Impact region dataframes
        iso_map = config.polygons["ISO"].reindex(impact["region"].values)
        impact = impact.assign_coords(ISO=("region", iso_map.to_numpy(dtype=object)))
        return impact, "region", ["region", "ISO"]

    if group_level == "ISO":
        if config.rate:
            # Pop-weighting total deaths required
            impact, pop = xr.align(impact, config.socioeconomics["pop"], join="exact")
            pop_total = impact * pop
            # Population by ISO
            pop_ISO, _ = aggregate_by_iso(
                config.socioeconomics["pop"], config.polygons, operation="sum"
            )
            # Total Deaths by ISO
            impact_rate_ISO, _ = aggregate_by_iso(
                pop_total, config.polygons, operation="sum"
            )
            return impact_rate_ISO / pop_ISO, "ISO", ["ISO"]
        else:
            # Sum total deaths
            impact, _ = aggregate_by_iso(impact, config.polygons, operation="sum")
            return impact, "ISO", ["ISO"]

    if group_level == "ADM1":
        impact = redistribute_adm1(impact, config)
        return impact, "GID_1", ["GID_0", "GID_1"]

    raise ValueError(f"Unsupported group_level: {group_level!r}")


###Output Functions ###
def _baseline_tag(baseline_period):
    start_year = baseline_period.start[:4]
    stop_year = baseline_period.stop[:4]
    return f"{start_year}-{stop_year}"


def dataset_to_dataframe(ds):
    if len(ds.dims) == 0:
        return pd.DataFrame({k: [v.values.item()] for k, v in ds.data_vars.items()})
    return ds.to_dataframe().reset_index()


def round_output(df):
    """
    Round each value to the nearest 1 (abs < 100), nearest 10 (abs < 1000),
    or nearest 100 (abs >= 1000), then cast to integer.
    """
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include="number").columns
    result = df.copy()

    abs_vals = df[numeric_cols].abs()
    # Identify magnitude
    unit = np.select(
        [abs_vals < 100, abs_vals < 1000, abs_vals >= 1000],
        [1, 10, 100],
        default=np.nan,
    )
    rounded = (df[numeric_cols] / unit).round(0) * unit
    result[numeric_cols] = rounded.astype("Int64")
    return result


def make_csv(
    effect,
    config: ImpactConfig,
    group_level="IR",
    filename_template="{version}_{hotonly}_{scope}_{rate_l}_{stat_scope}_{group_level}_{baseline}_{cleaned}.csv",
    output_scope=None,
):
    if output_scope is None:
        output_scope = [
            "regional_monthly",
            "regional_6mo",
            "global_monthly",
            "global_6mo",
        ]

    rate_l = "rate" if config.rate else "total"
    baseline_tag = _baseline_tag(config.baseline_period)

    impact = compute_impact(
        effect.chunk({dim: -1 for dim in config.dims}), config, ensemble=True
    )
    impact = impact.sel(month=config.months)

    # Aggregate Impact Regions to group_level
    impact, merge_key, base_cols = aggregate_impact(impact, config, group_level)

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
    # Step through outputs listed in 'output_scope"
    if "regional_monthly" in output_scope:
        _polygons_impact = dataset_to_dataframe(compute_stats(impact, dim=config.dims))
        wide = _polygons_impact.pivot(
            index=base_cols,
            columns="month",
            values=stat_cols,
        )
        wide.columns = [f"month {m} {stat}" for stat, m in wide.columns]
        wide = wide.reset_index()
        wide_rounded = round_output(wide)
        wide.to_csv(
            filename_template.format(
                version=config.version,
                hotonly=config.hotonly,
                rate_l=rate_l,
                scope="monthly",
                stat_scope="",
                group_level=group_level,
                baseline=baseline_tag,
                cleaned="raw",
            ),
            index=False,
        )
        wide_rounded.to_csv(
            filename_template.format(
                version=config.version,
                hotonly=config.hotonly,
                rate_l=rate_l,
                scope="monthly",
                stat_scope="",
                group_level=group_level,
                baseline=baseline_tag,
                cleaned="rounded",
            ),
            index=False,
        )

    if "regional_6mo" in output_scope:
        mo6 = impact.sum(dim="month")
        _polygons_mo6 = dataset_to_dataframe(compute_stats(mo6, dim=config.dims))
        mo6_out = _polygons_mo6[base_cols + stat_cols]
        mo6_out_rounded = round_output(mo6_out)
        mo6_out.to_csv(
            filename_template.format(
                version=config.version,
                hotonly=config.hotonly,
                rate_l=rate_l,
                scope="6mo",
                stat_scope="",
                group_level=group_level,
                baseline=baseline_tag,
                cleaned="raw",
            ),
            index=False,
        )
        mo6_out_rounded.to_csv(
            filename_template.format(
                version=config.version,
                hotonly=config.hotonly,
                rate_l=rate_l,
                scope="6mo",
                stat_scope="",
                group_level=group_level,
                baseline=baseline_tag,
                cleaned="rounded",
            ),
            index=False,
        )

    if "global_monthly" in output_scope or "global_6mo" in output_scope:
        # Compute global total before getting stats
        global_impact = compute_global_impact(
            impact, config.socioeconomics, rate=config.rate, group_dim=merge_key
        )

        if "global_monthly" in output_scope:
            global_monthly = dataset_to_dataframe(
                compute_stats(global_impact, dim=config.dims)
            )
            global_monthly_rounded = round_output(global_monthly)
            global_monthly.to_csv(
                filename_template.format(
                    version=config.version,
                    hotonly=config.hotonly,
                    rate_l=rate_l,
                    scope="monthly_global",
                    stat_scope="",
                    group_level=group_level,
                    baseline=baseline_tag,
                    cleaned="raw",
                ),
                index=False,
            )
            global_monthly_rounded.to_csv(
                filename_template.format(
                    version=config.version,
                    hotonly=config.hotonly,
                    rate_l=rate_l,
                    scope="monthly_global",
                    stat_scope="",
                    group_level=group_level,
                    baseline=baseline_tag,
                    cleaned="rounded",
                ),
                index=False,
            )
        if "global_6mo" in output_scope:
            global_mo6 = dataset_to_dataframe(
                compute_stats(global_impact.sum(dim="month"), dim=config.dims)
            )
            global_mo6_rounded = round_output(global_mo6)
            global_mo6.to_csv(
                filename_template.format(
                    version=config.version,
                    hotonly=config.hotonly,
                    rate_l=rate_l,
                    scope="6mo_global",
                    stat_scope="",
                    group_level=group_level,
                    baseline=baseline_tag,
                    cleaned="raw",
                ),
                index=False,
            )
            global_mo6_rounded.to_csv(
                filename_template.format(
                    version=config.version,
                    hotonly=config.hotonly,
                    rate_l=rate_l,
                    scope="6mo_global",
                    stat_scope="",
                    group_level=group_level,
                    baseline=baseline_tag,
                    cleaned="rounded",
                ),
                index=False,
            )


##### Land Only #####
# Get and store land data
@cache
def _get_land(crs):
    land = gpd.read_file(geodatasets.get_path("naturalearth land"))
    return land.cx[:, -60:90].to_crs(crs)


@cache
def _get_land_mask(lon_key, lat_key):
    dummy = xr.DataArray(
        np.zeros((len(lat_key), len(lon_key))),
        coords={"lat": list(lat_key), "lon": list(lon_key)},
        dims=["lat", "lon"],
    )
    return regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(dummy)


def compute_area_weighted_mean(ds, lat_name="lat", lon_name="lon"):
    # Compute area weighted-mean of gridded data
    weights = np.cos(np.deg2rad(ds[lat_name]))
    weights.name = "weights"
    return ds.weighted(weights).mean((lat_name, lon_name))


def land_only(da, lat_name="lat", lon_name="lon"):
    # Clip gridded data to land mask
    da = da.rename({lon_name: "lon", lat_name: "lat"})
    mask = _get_land_mask(tuple(da.lon.values), tuple(da.lat.values))
    return da.where(mask.notnull() & (da.lat > -60))
