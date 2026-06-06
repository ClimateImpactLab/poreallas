import os

from dotenv import load_dotenv
import pandas as pd

load_dotenv()

IN_WEIGHTS_FILE = (
    "./data/raw/agglomerated-world-new_BCSD_grid_segment_weights_area_pop.csv"
)
OUT_ZARR = os.getenv("POREALLAS_REGIONS_URI")



sw = pd.read_csv(
    IN_WEIGHTS_FILE,
)
sw["pix_cent_x"] = (sw["pix_cent_x"] + 180) % 360 - 180
sw = sw.to_xarray().rename_vars(
    {"pix_cent_x": "lon", "pix_cent_y": "lat", "hierid": "region", "popwt": "weight"}
)
sw.to_zarr(OUT_ZARR, consolidated=False)
print(f"Written to {OUT_ZARR}")
