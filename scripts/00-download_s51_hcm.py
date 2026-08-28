# Download historical hindcast ensemble mean
#
# This file is used as a target for regridding when other datasets are parsed,
# as this data is smaller than a full forecast ensemble. It does not need to
# be updated if you are projecting with a forecast ensemble from a different
# month/year as long as the structure of the data remains the same. Regridding
# with this data as a target should produce a regridded dataset that can be
# combined with the forecast ensemble.

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
