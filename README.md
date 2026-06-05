# poreallas

Mortality rate projection using seasonal forecast ensembles.

> [!WARNING]
> This is an incomplete toy prototype. This is not suitable for a production environment.

## Running

Fork/clone this repository.

You will need to have [uv](https://docs.astral.sh/uv/) installed and configured on your system to replicate this environment for analysis and development.

### Data and parsing

If you do not already have access to parsed input data you will need to download and clean input data, running the scripts in `./scripts/` in ordered sequence. This creates and populates input data in the `./data/` directory. Note this requires downloading and processing a significant amount of data. Some steps will require access to a daskhub cluster. This will be noted in script comments and documentation.

Data downloads and processing for the prototype were run in the first week of June, 2026.

Data downloads from Copernicus CDS (https://cds.climate.copernicus.eu/) require an ECMWF account. You will need to configure `cdsapi` with you account credentials (see https://github.com/ecmwf/cdsapi).

### Projecting

Projections can be run in the example projection notebook in `./project.py`. You will need access to the parsed input data described above before running the projection notebook.

Setup and run the example projection notebook by running

```shell
uv run marimo edit project.py
```

from the root of this repository.

## Support

This is open-source software made available under the terms of the Apache License 2.0.

This repository is available online at https://github.com/brews/poreallas.

Please file issues in the project's [issue tracker](https://github.com/brews/poreallas/issues).
