import marimo

__generated_with = "0.23.8"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Calculate `tas` climatologies for ERA5 and GMFD for use in bias adjusting the 7-month forecast for ENSO mortality projections

    This notebook needs to be run on notebooks.cilresearch.org for data access.

    K. McCusker

    July 2026
    """)
    return


@app.cell
def _():
    import xarray as xr
    import pandas as pd
    import numpy as np
    import xesmf as xe
    from datetime import datetime

    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    return ccrs, cfeature, np, plt, xe, xr


@app.cell
def _():
    gpath = "/gcs/impactlab-data/climate/source_data/GMFD/tas/tas_0p25_daily_{year}-{year}.nc" #1950-2010
    epathzarr = "gs://poreallas-public-20260605/v20260612/era5_daily_tas_regrid.zarr" # 1993-2025. 
    epath = "/gcs/impactlab-data/climate/source_data/ERA-5/tas/daily/netcdf/v1.1/tas_daily_{year}-{year}.nc" # 1979-2020
    return epath, epathzarr, gpath


@app.cell
def _(epathzarr, xr):
    # This is the ERA5 data used in the projection step for the baseline
    era5zarr = xr.open_dataset(epathzarr)

    # Use for regridding to 1x1deg. 
    target_grid = era5zarr.isel(time=0, drop=True)
    return (target_grid,)


@app.cell
def _(epath, gpath, xr):
    # these are the .25deg grids
    gmfd_grid = xr.open_dataset(gpath.format(year="1990")).isel(time=0).rename({"lat":"latitude","lon":"longitude"})
    era5_grid = xr.open_dataset(epath.format(year="1990")).isel(time=0)
    era5_grid
    return era5_grid, gmfd_grid


@app.cell
def _(target_grid):
    target_grid
    return


@app.cell
def _(era5_grid, gmfd_grid, target_grid, xe):
    # set up the regridders to regrid GMFD and ERA5 .25deg to the ERA5 1deg grid
    gregridder = xe.Regridder(gmfd_grid, target_grid, method="bilinear", periodic=True)
    eregridder = xe.Regridder(era5_grid, target_grid, method="bilinear", periodic=True)
    return eregridder, gregridder


@app.cell
def _(epath, eregridder, gpath, gregridder, np, xr):
    # Regrid ERA5 and GMFD for each year
    # This could be distributed instead of looped if want to speed it up
    YEARS = np.arange(1980,2011)

    gmfd_regrid_lst = []
    era5_regrid_lst = []
    for year in YEARS:
        print(year)

        gmfd = xr.open_dataset(gpath.format(year=year)).rename({"lat":"latitude","lon":"longitude"})
        era5 = xr.open_dataset(epath.format(year=year))

        gmfd_regrid_lst.append(gregridder(gmfd))
        era5_regrid_lst.append(eregridder(era5))
    return era5_regrid_lst, gmfd_regrid_lst


@app.cell
def _(era5_regrid_lst, gmfd_regrid_lst, xr):
    # combine each year into one dataset
    era5_regrid = xr.concat(era5_regrid_lst, dim="time")
    gmfd_regrid = xr.concat(gmfd_regrid_lst, dim="time")
    return era5_regrid, gmfd_regrid


@app.cell
def _(era5_regrid, gmfd_regrid):
    # calculate the monthly mean timeseries and then the climatological monthly mean
    era5_regrid_climo = era5_regrid.tas.resample(time="MS").mean().groupby("time.month").mean()
    gmfd_regrid_climo = gmfd_regrid.tas.resample(time="MS").mean().groupby("time.month").mean()
    return era5_regrid_climo, gmfd_regrid_climo


@app.cell
def _(era5_regrid_climo, gmfd_regrid_climo):
    # save climos to file
    era5_regrid_climo.to_netcdf("tas_era5_1deg_monthly_climo_1980-2010.nc")
    gmfd_regrid_climo.to_netcdf("tas_gmfd_1deg_monthly_climo_1980-2010.nc")
    return


@app.cell
def _(xr):
    # check the data:

    era5_climo = xr.open_dataset("tas_era5_1deg_monthly_climo_1980-2010.nc")
    gmfd_climo = xr.open_dataset("tas_gmfd_1deg_monthly_climo_1980-2010.nc")
    return era5_climo, gmfd_climo


@app.cell
def _(era5_climo, gmfd_climo):
    # Look at the difference in climos
    monthly_diff = (gmfd_climo - era5_climo)
    return (monthly_diff,)


@app.cell
def _(ccrs, cfeature, monthly_diff, plt):
    fg = monthly_diff["tas"].isel(month=[5,6,7,8,9]).plot(
        col="month",
        transform=ccrs.PlateCarree(),  
        subplot_kws={"projection": ccrs.Robinson()},#central_longitude=-95, central_latitude=45)},
        cbar_kwargs={"orientation": "horizontal", "shrink": 0.8, "aspect": 40},
        robust=True,
        vmin=-5,
        vmax=5,
        cmap="RdBu_r",
    )
    plt.suptitle("tas (degC), GMFD - ERA5 (1980 - 2010 mean)")

    # lets add a coastline to each axis
    # great reason to use FacetGrid.map
    fg.map(lambda: plt.gca().coastlines())
    fg.map(lambda: plt.gca().add_feature(cfeature.OCEAN, facecolor='lightgray', zorder=1))

    plt.savefig("tas_enso_monthly_tas_difference_gmfd_v_era5_1deg.png",dpi=300,)
    return


if __name__ == "__main__":
    app.run()
