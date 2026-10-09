## Download max/min daily temp from
## https://cds.climate.copernicus.eu/datasets/derived-era5-single-levels-daily-statistics?tab=download
## 6-day lag

import cdsapi

YEAR = "2026"
MONTHS = ["01", "02", "03", "04", "05", "06", "07", "08", "09"]
DAYS = [
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "09",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
    "21",
    "22",
    "23",
    "24",
    "25",
    "26",
    "27",
    "28",
    "29",
    "30",
    "31",
]

dataset = "derived-era5-single-levels-daily-statistics"
request = {
    "product_type": "reanalysis",
    "variable": ["2m_temperature"],
    "year": YEAR,
    "month": MONTHS,
    "day": DAYS,
    "daily_statistic": "daily_maximum",
    "time_zone": "utc+00:00",
    "frequency": "6_hourly",
}

client = cdsapi.Client()
client.retrieve(dataset, request, "./data/raw/26_era5_daily_max.nc")

request = {
    "product_type": "reanalysis",
    "variable": ["2m_temperature"],
    "year": YEAR,
    "month": MONTHS,
    "day": DAYS,
    "daily_statistic": "daily_minimum",
    "time_zone": "utc+00:00",
    "frequency": "6_hourly",
}

client = cdsapi.Client()
client.retrieve(dataset, request, "./data/raw/26_era5_daily_min.nc")
