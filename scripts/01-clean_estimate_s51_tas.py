import os

from dotenv import load_dotenv
import xarray as xr

load_dotenv()

OUT_ZARR = os.getenv("POREALLAS_TAS_FORECAST_URI")


ds_tas = xr.merge(
    [
        xr.open_dataset("./data/raw/s51_tasmax.nc"),
        xr.open_dataset("./data/raw/s51_tasmin.nc"),
    ],
    compat="no_conflicts",
)

# Estimate daily tas from daily tasmax and daily tasmin.
ds_tas["tas"] = (ds_tas["mx2t24"] + ds_tas["mn2t24"]) / 2

ds_tas[["tas"]].to_zarr(OUT_ZARR, consolidated=False)
print(f"s51 data written to {OUT_ZARR}")

# # Need to know min, max of daily tas if we're doing this via histograms so we know the range for histogram bins.
# global_tas_domain = (
#     ds_tas["tas"].min().item(),
#     ds_tas["tas"].max().item(),
# )

# # Plot showing ensemble values vs ensemble mean for some arbitrary grid point.
# ds_tas.set_coords("valid_time")["tas"].isel(latitude=100, longitude=100).squeeze(
#     drop=True
# ).plot.scatter(x="valid_time", marker=".", alpha=0.3, edgecolors="none")
# ds_tas.set_coords("valid_time")["tas"].isel(latitude=100, longitude=100).squeeze(
#     drop=True
# ).mean(dim="number").plot.line(x="valid_time", color="C1")
# plt.show()
