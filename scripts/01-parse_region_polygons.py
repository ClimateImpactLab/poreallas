import os

from dotenv import load_dotenv
import geopandas as gpd

load_dotenv()

# NOTE: You'll need to update the input data path to match your system or copy the required data to the
# indicated directory.
# Download and unpack https://zenodo.org/records/6416119/files/data.zip?download=1.
# Beware, it's ~35 GiB.
# This is the file at ./data/2_projection/1_regions/ir_shp/impact-region.shp
# Note that this is input data is a shape file so you will need to copy all of
# the files within the this .shp file's directory.
OUT_PARQUET = os.getenv("POREALLAS_REGIONS_POLYGONS_URI")

gdf = gpd.read_file("./data/raw/ir_shp/impact-region.shp")
gdf.to_parquet(OUT_PARQUET)
print(f"Regions written to {OUT_PARQUET}")
