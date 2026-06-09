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
        missing_variables = [v for v in target_variables if v not in weights.variables]
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
        out = (region_sel * self._data["weight"]).groupby(self._data["region"]).sum()
        return out


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
