# Local data and configuration

No data settings are needed for the figure examples in the [README](README.md).

To run analyses on trajectories, set `OPENSKY26_DATA_DIR` to the directory
containing `flights.delta` and `raw/extended/`. It defaults to `data/` in this
project.

Optional overrides are documented in `.env.example`:

```bash
cp .env.example .env
# Edit .env with your local paths.
set -a
. ./.env
set +a
```

Only source a file you trust. Python does not load `.env` automatically.
Local `.env` files are ignored by Git. Keep provider credentials outside the
repository, in the providers' own configuration.

`OPENSKY26_PIPELINE_CONFIG` overrides the supplied
`src/opensky26_plateau/pipeline/config.yaml`. Relative configuration and data
paths resolve from this project directory.

The cohort preparation script documents its arguments with:

```bash
uv run --locked python investigations/reference-availability/experiments/01-bds40-availability/scripts/00_prepare_cohort.py --help
```

Preparation requires a new destination. `--dry-run` checks the supplied inputs
without downloading data; an actual run needs OpenSky and weather-provider access.
Installing dependencies alone does not provide that access.
