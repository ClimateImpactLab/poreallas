import os

from dotenv import load_dotenv
import pandas as pd

load_dotenv()

IN_WEIGHTS_FILE = "./data/raw/s51_segment_weights.parquet"
OUT_ZARR = os.getenv("POREALLAS_REGIONS_URI")


sw = pd.read_parquet(IN_WEIGHTS_FILE)

sw = sw.to_xarray().rename_vars(
    {"cell_lon": "lon", "cell_lat": "lat", "hierid": "region", "popwt": "weight"}
)

# Have trouble interpreting cell_ix as data array when writing to zarr store so dropping.
sw = sw.drop_vars(["cell_ix", "cell_iy"])

sw.to_zarr(OUT_ZARR, consolidated=False)
print(f"Written to {OUT_ZARR}")
