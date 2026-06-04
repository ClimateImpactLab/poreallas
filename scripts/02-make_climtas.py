import isku
import numpy as np
import xarray as xr
from xclim.core import units


def _make_annual_tas(ds: xr.Dataset) -> xr.Dataset:
    """
    Compute tas variable in degC. Should be annual.
    """
    tas = units.convert_units_to(ds["tas"], "degC")
    ## If the data needs to be annualized...
    # return tas.groupby("time.year").mean("time").to_dataset()
    return tas.to_dataset()


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


class FuzzyGridWeightingExtractor(isku.RegionExtractor):
    """
    Weight a grid and extract regions when the region and weight position don't exactly match lat, lon on the grid.

    Follows the isku.RegionExtractor protocol.
    Uses Nearest-neighbor search within `tolerance` to extract points from a grid. Suggest using a reasonable `tolerance` that does not extend beyond one grid cell width. Otherwise the algorithm will happily walk across the globe to find a matching grid point.
    """

    def __init__(self, weights: xr.Dataset, tolerance: float):
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


ds = xr.open_dataset("./data/era5_annual_tas_1995_2025_regrid.nc")

# Clean up longitude. The data goes from longitude 0 to 360. It needs to go -180 to 180 in ascending order.
ds["longitude"] = (ds["longitude"] + 180) % 360 - 180
ds = ds.sortby("longitude")
# Add missing unit information.
ds["tas"].attrs["units"] = "K"
ds = ds.rename({"longitude": "lon", "latitude": "lat"})

segment_weights_raw = xr.open_zarr("./data/parsed/segment_weights.zarr")[
    ["lat", "lon", "weight", "region"]
]
segment_weights = FuzzyGridWeightingExtractor(segment_weights_raw.load(), tolerance=0.5)

climtas = isku.extract_regions(
    ds,
    template=make_climtas,
    regions=segment_weights,
)

out_path = "./data/parsed/climtas.zarr"
climtas.sel(year=2025, drop=True).to_zarr(out_path)
print(f"File written to {out_path}")
