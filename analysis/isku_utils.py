import os
from dotenv import load_dotenv

import geopandas
import marimo as mo
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr
from xclim.core import units

import isku

load_dotenv()

TAS_FORECAST_URI = os.environ["POREALLAS_TAS_FORECAST_URI"]
ERA5_URI = os.environ["POREALLAS_ERA5_URI"]
GAMMA_URI = os.environ["POREALLAS_GAMMA_URI"]
SOCIOECONOMICS_URI = os.environ["POREALLAS_SOCIOECONOMICS_URI"]
REGIONS_URI = os.environ["POREALLAS_REGIONS_URI"]

def _do_nothing(ds: xr.Dataset) -> xr.Dataset:
    return ds

do_nothing_func = isku.build_extraction_template(
    pre=_do_nothing,
    post=lambda ds: ds.astype("float32"),  # Save space. Don't need float64.
)

def read_regions(uri: str) -> isku.GridWeightingRegions:
    _region_weights = xr.load_dataset(uri)[
        ["lat", "lon", "region", "weight"]
    ]  # Load only what we need.
    # Apparently in this version of xarray the `.load()` method type-hints it'll return a DataArray instead of a Dataset.
    # It is a Dataset (I checked). So telling ty to ignore it.
    # # TODO: send bug upstream?
    regions = isku.GridWeightingRegions(_region_weights)  # ty: ignore[invalid-argument-type]
    return regions

def grid_to_ir(data, savefile = None):
    regions = read_regions(REGIONS_URI)
    data_ir = isku.extract_regions(
        data,
        template=do_nothing_func,
        regions=regions,
    )
    data_ir = data_ir.chunk({dim: "auto" for dim in data_ir.dims})
    if savefile is not None:
        data_ir.to_zarr(f"{savefile}.zarr")
    return data_ir
