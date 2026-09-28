
import math

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs

from analysis_utils import _get_land

DIVERGING_CMAPS = {
    "bwr",
    "seismic",
    "coolwarm",
    "RdBu",
    "RdYlBu",
    "PiYG",
    "PRGn",
    "BrBG",
}

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

    base_name = cm[:-2] if cm.endswith("_r") else cm
    if base_name in DIVERGING_CMAPS:
        center_idx = len(colors) // 2
        colors[center_idx] = [0.95, 0.95, 0.95, 1]

    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    return cmap, norm, sm


def build_colormap(gdf=None, col=None, cm="bwr", vmin=None, vmax=None, n_colors=None):
    # Build discrete colormap, branching on diverging vs sequential
    base_name = cm[:-2] if cm.endswith("_r") else cm

    if base_name in DIVERGING_CMAPS:
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

    ticks = get_ticks(bounds, step=step, symmetric=(base_name in DIVERGING_CMAPS))
    return cmap, norm, sm, ticks, step


def plot_single(
    gdf,
    col,
    sup_title="",
    save_title="",
    cm="bwr",
    cbar_label=None,
    vmin=None,
    vmax=None,
    alpha=1,
    edgecolor=None,
    linewidth=0,
    ax=None,
    cbar_location="right",
    colorbar=True,
    cbar_shrink=0.6, 
    cbar_aspect=20, 
    cbar_pad=0.02,
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
        alpha=alpha,
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
            shrink=cbar_shrink,
            aspect = cbar_aspect,
            pad = cbar_pad,
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

    fig, axes = plt.subplots(2, 3, figsize=(16, 6),gridspec_kw={"wspace": 0.02, "hspace": 0.02}, constrained_layout=True)

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

    for ax in axes.flat[len(months):]:
        ax.set_axis_off()

    cmap, norm, sm, ticks, step = build_colormap(
        gdf, col, cm=cm, vmin=vmin, vmax=vmax, n_colors=n_colors
    )
    fig.colorbar(
        sm,
        ax=axes.ravel().tolist(),
        location="right",
        shrink=0.6,
        pad=0.02,
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