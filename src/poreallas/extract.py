"""
Logic for region extraction and transformation

"""

import isku
import numpy as np
import xarray as xr
from xclim.core import units
from xhistogram.xarray import histogram


def _make_annual_tas(ds: xr.Dataset) -> xr.Dataset:
    """
    Compute tas variable in degC. Should be annual.
    """
    tas = xr.DataArray(units.convert_units_to(ds["tas"], "degC"))
    ## TODO: If the data needs to be annualized... Might need this.
    return tas.groupby("time.year").mean("time").to_dataset()


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


def _make_monthly_tas_histogram(ds: xr.Dataset) -> xr.Dataset:
    _tas = xr.DataArray(units.convert_units_to(ds["tas"], "degC"))

    _bins = np.arange(-105, 66)  # Range we get histogram count for. NOTE: in degC!
    _tas_annual_histogram = _tas.resample(time="1MS").map(
        histogram, bins=[_bins], dim=["time"]
    )
    return _tas_annual_histogram.to_dataset().astype("float32")


make_tas_monthly_histogram = isku.build_extraction_template(
    pre=_make_monthly_tas_histogram,
    post=lambda ds: ds.astype("float32"),  # Save space. Don't need float64.
)
