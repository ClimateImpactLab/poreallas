# Create QDM bias adjustment of the parsed forecast dataset using the parsed GMFD dataset.

import datetime
import os
import uuid

from dotenv import load_dotenv
import xarray as xr
from xsdba.adjustment import QuantileDeltaMapping, TrainAdjust
from xsdba.base import Grouper

load_dotenv()

FORECAST_URI = os.environ["POREALLAS_PARSED_FORECAST_URI"]
GMFD_URI = os.environ["POREALLAS_PARSED_GMFD_URI"]
OUT_ZARR = os.environ["POREALLAS_TAS_FORECAST_URI"]
HISTREF_START_YEAR = 1981
HISTREF_STOP_YEAR = 1997
SIM_START_YEAR = 2007
SIM_STOP_YEAR = 2026
QDM_N_QUANTILES = 10
FORECAST_LENGTH = 215  # ECMWF S51 is 215 days.
UID = str(uuid.uuid4())
START_TIME = datetime.datetime.now(datetime.UTC).isoformat()


def adjust_month(
    *,
    ref: xr.DataArray,
    hist: xr.DataArray,
    sim: xr.DataArray,
    target_month: int,
    nquantiles: int,
) -> tuple[TrainAdjust, xr.DataArray]:
    """
    Train and apply QDM for a particular `time.month`
    """
    ref = ref.where(ref["time.month"] == target_month, drop=True)
    hist = hist.where(hist["time.month"] == target_month, drop=True)
    sim = sim.where(sim["time.month"] == target_month, drop=True)

    qdm = QuantileDeltaMapping.train(
        ref,
        hist,
        nquantiles=nquantiles,
        kind="+",
        group=Grouper("time", add_dims=["number"]),
    )
    adj = qdm.adjust(sim)
    return qdm, adj


def adjust_months(
    *,
    ref: xr.DataArray,
    hist: xr.DataArray,
    sim: xr.DataArray,
    nquantiles: int,
) -> xr.DataArray:
    """
    Train and apply quantile delta mapping (QDM) for all `time.month` in a simulation.

    We need a custom algorithm for this because our forecast ensembles run for <
    365 days yet this QDM implementation does not allow us to group by "time.month"
    when it does not have all 12 months. So we train and apply QDM to each of the
    months in the simulation dataset and then concatenate them back together along
    the time dimension. The concatenated data is then sorted by the time dimension
    to return the data to chronological order.

    Parameters
    ----------
    ref :
        Reference dataset to compare against a historical simulation to train a QDM.
    hist :
        Historical simulation dataset to be compared against ref when training the QDM.
    sim :
        Simulation, or forecast ensemble to be adjusted by the trained QDM.
    nquantiles :
        Number of quantiles to use in the quantile mapping.

    Returns
    -------
    combined :
        Simulated, bias-adjusted by a QDM trained on a historical and reference dataset.
    """
    adjusted = []
    for m in set(sim["time.month"].data):
        _, adj = adjust_month(
            ref=ref,
            hist=hist,
            sim=sim,
            target_month=m,
            nquantiles=nquantiles,
        )
        adjusted.append(adj)

    combined = xr.concat(adjusted, dim="time").sortby("time")
    return combined


gmfd = xr.open_zarr(GMFD_URI)
forecast = xr.open_zarr(FORECAST_URI)

# Outline the datasets we need for the adjustment, grabbing the windows in time needed.
ref = gmfd.sel(time=slice(str(HISTREF_START_YEAR), str(HISTREF_STOP_YEAR)))
hist = forecast.sel(time=slice(str(HISTREF_START_YEAR), str(HISTREF_STOP_YEAR)))
sim = forecast.sel(time=slice(str(SIM_START_YEAR), str(SIM_STOP_YEAR)))

# Only use `ref` with corresponding values in time in `hist`.
# Need this because forecast ensemble are not a complete time series and
# 'ref', 'hist' need to match through time.
ref = ref.where(ref["time"].isin(hist["time"]), drop=True)

# Rechunking because all of "time", or whatever we're grouping QDM on, needs to be in one chunk.
ref = ref.chunk({"time": -1})
hist = hist.chunk({"number": -1, "time": -1, "latitude": "30", "longitude": "auto"})
sim = sim.chunk({"number": -1, "time": -1, "latitude": "30", "longitude": "auto"})

# Train QDM and adjust the forecast ensemble, for the months in the forecast ensemble.
sim_adj = adjust_months(
    ref=ref["tas"],
    hist=hist["tas"],
    sim=sim["tas"],
    nquantiles=QDM_N_QUANTILES,
)

# QDM requires forecasts over several years to estimate distribution functions
# but we're only interested in the most recent forecast (of `FORECAST_LENGTH`
# days; e.g. 215 days for ECMWF S51). So we pop off and use those last
# FORECAST_LENGTH days because that's the real period of interest for the
# mortality projection.
# NOTE: If you turn this off for debugging, the output data can be orders of
# magnitude larger!
sim_adj = sim_adj.isel(time=slice(-int(FORECAST_LENGTH), None))

sim_adj.name = "tas"
sim_adj = sim_adj.to_dataset()

# Add additional general metadata.
sim_adj.attrs |= {
    "poreallas_created_at": START_TIME,
    "poreallas_uid": UID,
    "poreallas_description": "QDM bias-adjusted forecast ensemble fields",
}
sim_adj["tas"].attrs |= {
    "poreallas_created_at": START_TIME,
    "poreallas_uid": UID,
    "poreallas_description": "QDM bias-adjusted forecast ensemble tas fields",
    "poreallas_adjustment_method": "QDM",
    "poreallas_histref_start_year": HISTREF_START_YEAR,
    "poreallas_histref_stop_year": HISTREF_STOP_YEAR,
    "poreallas_sim_start_year": SIM_START_YEAR,
    "poreallas_sim_stop_year": SIM_STOP_YEAR,
    "poreallas_qdm_nquantiles": QDM_N_QUANTILES,
    "poreallas_ref_uri": GMFD_URI,
    "poreallas_hist_uri": FORECAST_URI,
    "poreallas_sim_uri": FORECAST_URI,
}

sim_adj = sim_adj.chunk("auto")

sim_adj.to_zarr(OUT_ZARR, consolidated=True)
print(f"Output written to {OUT_ZARR}")
