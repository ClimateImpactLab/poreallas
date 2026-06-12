import os

from dotenv import load_dotenv
import pandas as pd

load_dotenv()

# From Box account folder https://app.box.com/folder/377812758051 at path Methodology_and_Diagnostics/data/OUTPUT/ir_combined_SSP2_IIASA_v4.csv on 2026-06-09.
# it uses Penn World Tables as observed data through 2023, then hands off to SSP from 2024 onwards.
IN_CSV_URI = "./data/raw/ir_combined_SSP2_IIASA_v4.csv"
OUT_ZARR_URI = os.getenv("POREALLAS_SOCIOECONOMICS_URI")

df = (
    pd.read_csv(IN_CSV_URI)
    .rename(columns={"hierid": "region"})
    .set_index(["region", "year"])
)
ds = df.to_xarray()

# Somewhat picked at random, as long as the zarr doesn't require folks to load ~1 GiB data at once.
ds = ds.chunk({"region": 1000, "year": 365})

ds.to_zarr(OUT_ZARR_URI, consolidated=False)
print(f"Socioeconomic data written to {OUT_ZARR_URI}")
