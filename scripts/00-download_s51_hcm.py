# See https://www.ecmwf.int/en/forecasts/documentation-and-support/seasonal
# https://cds.climate.copernicus.eu/datasets/seasonal-original-single-levels
# https://iri.columbia.edu/our-expertise/climate/forecasts/seasonal-climate-forecasts/

# Daily ERA5:
# https://cds.climate.copernicus.eu/datasets/derived-era5-single-levels-daily-statistics
# # ARCO ERA5:
# https://github.com/google-research/arco-era5
# Seasonal forecast daily + subdaily
# https://cds.climate.copernicus.eu/datasets/seasonal-original-single-levels

import cdsapi

# Download ensemble hindcast climate mean. See https://cds.climate.copernicus.eu/datasets/seasonal-monthly-single-levels?tab=download
dataset = "seasonal-monthly-single-levels"
request = {
    "originating_centre": "ecmwf",
    "system": "51",
    "variable": ["2m_temperature"],
    "product_type": [
        "hindcast_climate_mean",
    ],
    "year": ["2026"],
    "month": ["05"],
    "leadtime_month": ["1", "2", "3", "4", "5", "6"],
    "data_format": "netcdf",
}
client = cdsapi.Client()
client.retrieve(dataset, request, "./data/raw/s51_hcm.nc")

# # Download ensemble mean.
# dataset = "seasonal-monthly-single-levels"
# request = {
#     "originating_centre": "ecmwf",
#     "system": "51",
#     "variable": ["2m_temperature"],
#     "product_type": [
#         "ensemble_mean",
#     ],
#     "year": ["2026"],
#     "month": ["05"],
#     "leadtime_month": ["1", "2", "3", "4", "5", "6"],
#     "data_format": "netcdf",
# }
# client.retrieve(dataset, request, "./data/raw/download_em.nc")

# ds_em = xr.open_dataset("download_em.nc")
# ds_hcm = xr.open_dataset("download_hcm.nc")

# # Plot comparison between the two series.
# ds_em["t2m"].isel(latitude=100, longitude=100).squeeze(drop=True).plot.line(
#     x="forecastMonth"
# )
# ds_hcm["t2m"].isel(latitude=100, longitude=100).squeeze(drop=True).plot.line(
#     x="forecastMonth", color="C1"
# )
# plt.show()


# # Plot difference like
# (ds_em["t2m"] - ds_hcm["t2m"]).plot(col="forecastMonth")
# plt.show()
