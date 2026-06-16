# poreallas

Mortality rate projection using seasonal forecast ensembles. Loosely based on Carleton et al 2022 (https://doi.org/10.3386/w27599).

> [!WARNING]
> This is an incomplete toy prototype. This is not suitable for a production environment.

## Running

Fork/clone this repository.

You will need to have [uv](https://docs.astral.sh/uv/) installed and configured on your system to replicate this environment for analysis and development.

### Configuration

Key configurations are set through environment variables or a .env file (see `example.env`).

The current configurations are:

* POREALLAS_TAS_FORECAST_URI: URI to the cleaned ECMWF S51 ensemble air temperature Zarr Store.
* POREALLAS_ERA5_URI: URI to the Zarr Store of cleaned daily ERA5 dataset used for historical climate and impacts analysis.
* POREALLAS_GAMMA_URI: URI to the Zarr Store of "gamma" parameters used when calculating to calculate a mortality response function.
* POREALLAS_REGIONS_URI: URI to the Zarr Store of region and grid weights or "segment weights".
* POREALLAS_REGIONS_POLYGONS_URI: URI to geoparquet file with polygons for each region. Used for mapping.
* POREALLAS_SOCIOECONOMICS_URI: URI to file with each region's GDP per capita (gdppc).
* POREALLAS_EFFECTS_URI: Optional URI to write Zarr store of projected mortality effects. Will not write output if unset.

These are used to run the prototype in `scripts/` for downloads, parsing/cleaning, and projecting.

Each of these variables can point to data in cloud storage or local storage. 

### Projecting

Projections can be run using the non-interactive script in `scripts/`. You will need access to the parsed input data described above before running a projection.

Run the projection script with

```shell
uv run scripts/05-project_effects.py
```

from the root of this repository.

### Data and parsing

If you do not already have access to parsed input data you will need to download and clean input data, running the scripts in `./scripts/` in ordered sequence. This creates and populates input data in the `./data/` directory. Note this requires downloading and processing a significant amount of data. Some steps will require access to a daskhub cluster. This will be noted in script comments and documentation.

Data downloads and processing for the prototype were run in the first week of June, 2026.

Data downloads from Copernicus CDS (https://cds.climate.copernicus.eu/) require an ECMWF account. You will need to configure `cdsapi` with you account credentials (see https://github.com/ecmwf/cdsapi).


## Support

This is open-source software made available under the terms of the Apache License 2.0.

This repository is available online at https://github.com/brews/poreallas.

Please file issues in the project's [issue tracker](https://github.com/brews/poreallas/issues).
