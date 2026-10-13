import numpy as np
import xarray as xr
from scipy import stats

from utils.analysis_utils import crossed_ttest, land_only


def compute_agreement_mask(anom, threshold=0.8, member_dim="number"):
    ens_mean = anom.mean(member_dim)
    agree = (np.sign(anom) == np.sign(ens_mean)).mean(member_dim)
    return land_only((agree <= threshold).astype(float))


def ttest_mask(anom, alpha=0.05, member_dim="number"):
    # One-sample t-test: is the ensemble mean significantly different from zero?
    n = anom.sizes[member_dim]
    mean = anom.mean(member_dim)
    std = anom.std(member_dim, ddof=1)
    t = mean / (std / np.sqrt(n))
    # Two-sided p-value; apply_ufunc keeps lat/lon coords
    p = xr.apply_ufunc(
        lambda x: 2 * stats.t.sf(np.abs(x), n - 1),
        t,
        dask="parallelized",
        output_dtypes=[float],
    )
    mask = (p >= alpha).astype(float)
    return land_only(mask)


def region_ttest_mask(anom, alpha=0.05):
    # Per-region t-test; 1 where the signal is NOT significant (same convention as ttest_mask)
    p = crossed_ttest(anom.compute())
    return (p >= alpha).astype(float)
