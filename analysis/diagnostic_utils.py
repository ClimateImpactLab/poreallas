import os
from dotenv import load_dotenv

import geopandas
import marimo as mo
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from xclim.core import units


load_dotenv()

DATA_DIR = os.environ["DATA_DIR"]
TAS_FORECAST_URI = os.environ["POREALLAS_TAS_FORECAST_URI"]
ERA5_URI = os.environ["POREALLAS_ERA5_URI"]
GAMMA_URI = os.environ["POREALLAS_GAMMA_URI"]
SOCIOECONOMICS_URI = os.environ["POREALLAS_SOCIOECONOMICS_URI"]
REGIONS_URI = os.environ["POREALLAS_REGIONS_URI"]
BETAS_PATH = os.environ["BETAS_PATH"]

def weighted_cdf(data, bins, weights):
   
    edges = np.concatenate([bins, [bins[-1] + np.diff(bins)[-1]]])
    counts, edges = np.histogram(data, bins=edges, density=False)

    weighted = counts * weights * (31/counts.sum())
    cdf = np.cumsum(weighted)

    return bins, cdf

def compute_cumulative_effect(forecast_local, reanalysis_local, region_filter, months, monthly = False, hotonly = False):
    betas_mmt = xr.open_zarr(os.path.join(DATA_DIR, BETAS_PATH))

    cdf_data = {}
    max_cdf = 0
    if monthly:
        for month in months:
            forecast_local_month = forecast_local.sel(region = region_filter, time=forecast_local.time.dt.month == month)
            reanalysis_local_month = reanalysis_local.sel(region = region_filter, time=reanalysis_local.time.dt.month == month)


            ref_vals = betas_mmt['mmt'].sel(region=region_filter).sel(age_cohort='age65plus') 
            betas = betas_mmt['beta_hotonly'] if hotonly else betas_mmt['beta']
            da_temp_bins = betas.sel(region=region_filter).sel(age_cohort='age65plus')
            da_temp_bins['tas_bin'] = da_temp_bins['tas_bin'] 

            bins = da_temp_bins['tas_bin'].values
            weights = betas.sel(region = region_filter).sel(age_cohort = 'age65plus').values

            era5_centers, era5_cdf = weighted_cdf(reanalysis_local_month.values.flatten(), bins, weights)
            fc_centers, fc_cdf = weighted_cdf(forecast_local_month.values.flatten(), bins, weights)

            cdf_data[month] = (era5_centers, era5_cdf, fc_centers, fc_cdf)
            max_cdf = max(max_cdf, era5_cdf.max(), fc_cdf.max())
    else:
        forecast_local_month = forecast_local.sel(region = region_filter)
        reanalysis_local_month = reanalysis_local.sel(region = region_filter)


        ref_vals = betas_mmt['mmt'].sel(region=region_filter).sel(age_cohort='age65plus') 
        betas = betas_mmt['beta_hotonly'] if hotonly else betas_mmt['beta']
        da_temp_bins = betas.sel(region=region_filter).sel(age_cohort='age65plus')
        da_temp_bins['tas_bin'] = da_temp_bins['tas_bin'] 

        bins = da_temp_bins['tas_bin'].values
        weights = betas.sel(region = region_filter).sel(age_cohort = 'age65plus').values

        era5_centers, era5_cdf = weighted_cdf(reanalysis_local_month.values.flatten(), bins, weights)
        fc_centers, fc_cdf = weighted_cdf(forecast_local_month.values.flatten(), bins, weights)

        cdf_data = (era5_centers, era5_cdf, fc_centers, fc_cdf)
        max_cdf = max(max_cdf, era5_cdf.max(), fc_cdf.max())
    return cdf_data, max_cdf