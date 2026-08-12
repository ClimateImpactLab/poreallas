# Created QDM bias adjustment of the parsed ERA5 dataset using the parsed GMFD dataset.

import datetime
import os
import uuid

from dotenv import load_dotenv
import xarray as xr
from xsdba.adjustment import QuantileDeltaMapping

load_dotenv()

ERA5_URI = os.environ["POREALLAS_PARSED_ERA5_URI"]
GMFD_URI = os.environ["POREALLAS_PARSED_GMFD_URI"]
OUT_ZARR = "/home/emily_zuetell/projects/poreallas/data/era5_adj_corrected.zarr"

HISTREF_START_YEAR = 1981
HISTREF_STOP_YEAR = 1997
SIM_START_YEAR = 1996
SIM_STOP_YEAR = 2025
QDM_N_QUANTILES = 10
UID = str(uuid.uuid4())
START_TIME = datetime.datetime.now(datetime.UTC).isoformat()

gmfd = xr.open_dataset(GMFD_URI, engine = 'zarr', chunks = {}, backend_kwargs = {"storage_options": {"token": "anon"}})
# Fill extreme values
gmfd = gmfd.sortby("latitude").chunk({"latitude": -1, "longitude": 30, "time": -1})
gmfd = gmfd.where(gmfd["tas"] < 1000).interpolate_na(dim="latitude", method="linear").compute()

era5 = xr.open_dataset(ERA5_URI, engine = 'zarr', chunks = {}, backend_kwargs = {"storage_options": {"token": "anon"}})

ref = gmfd.sel(time=slice(str(HISTREF_START_YEAR), str(HISTREF_STOP_YEAR)))
hist = era5.sel(time=slice(str(HISTREF_START_YEAR), str(HISTREF_STOP_YEAR)))
sim = era5.sel(time=slice(str(SIM_START_YEAR), str(SIM_STOP_YEAR)))

# # "time" dim cannot be chunked for QDM.
ref = ref.chunk({"time": -1})
hist = hist.chunk({"time": -1})
sim = sim.chunk({"time": -1})

qdm = QuantileDeltaMapping.train(
    ref["tas"], hist["tas"], nquantiles=QDM_N_QUANTILES, kind="+", group="time.month"
)

sim_adj = qdm.adjust(sim["tas"])

sim_adj.name = "tas"
sim_adj = sim_adj.to_dataset()

# Add additional general metadata.
sim_adj.attrs |= {
    "poreallas_created_at": START_TIME,
    "poreallas_uid": UID,
    "poreallas_description": "QDM bias-adjusted ERA5 climate fields",
}
sim_adj["tas"].attrs |= {
    "poreallas_created_at": START_TIME,
    "poreallas_uid": UID,
    "poreallas_description": "QDM bias-adjusted ERA5 tas field",
    "poreallas_adjustment_method": "QDM",
    "poreallas_histref_start_year": HISTREF_START_YEAR,
    "poreallas_histref_stop_year": HISTREF_STOP_YEAR,
    "poreallas_sim_start_year": SIM_START_YEAR,
    "poreallas_sim_stop_year": SIM_STOP_YEAR,
    "poreallas_qdm_nquantiles": QDM_N_QUANTILES,
    "poreallas_ref_uri": GMFD_URI,
    "poreallas_hist_uri": ERA5_URI,
    "poreallas_sim_uri": ERA5_URI,
}

sim_adj = sim_adj.chunk("auto").compute()

sim_adj.to_zarr(OUT_ZARR, consolidated=True)
print(f"Output written to {OUT_ZARR}")
