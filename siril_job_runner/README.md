# Siril Job Runner

Automated Siril image processing pipeline with JSON job file configuration.

## Features

- Job file-based configuration for reproducible processing
- Auto-detection of calibration requirements from FITS headers
- Temperature tolerance matching for darks/bias
- Multi-night light frame support
- Automatic master calibration building and caching
- Support for LRGB, SHO, and HOO workflows
- HDR support: separate stacking by exposure for manual blending

## Usage

```bash
# Validate a job file (check calibration availability)
uv run python run_job.py examples/example_lrgb_job.json --validate

# Dry run (show what would happen)
uv run python run_job.py examples/example_lrgb_job.json --dry-run

# Run full pipeline
uv run python run_job.py jobs/M42.json

# Run specific stage
uv run python run_job.py jobs/M42.json --stage preprocess
```

### sirilpy (Siril Integration)

sirilpy is required for actual processing but is not available on PyPI. It comes bundled with Siril 1.2+. Make sure Siril is running with scripting enabled before running jobs.

### Siril Catalog Installation (Recommended)

Install local catalogs for faster processing and offline capability.

In Siril GUI: **Scripts > Python Scripts > Core > Siril_Catalog_Installer**

1. **Astrometry Catalog** (~1.5GB): Select "Astrometry Catalog", click Install
2. **SPCC Catalog** (~5-10GB): Select "SPCC Catalog", enter latitude/longitude, select "Visible from latitude", click Install

Bend, Oregon: Latitude 44.06, Longitude -121.32

Without local SPCC catalog, each job spends ~60s querying Gaia DR3 online. With local catalog, this drops to a few seconds.

## Directory Structure

The system expects this directory layout:

```
E:\Astro\RC51_ASI2600\
├── calibration\
│   ├── masters\
│   │   ├── biases\
│   │   │   └── bias_2024_01_15.fit
│   │   ├── darks\
│   │   │   └── dark_300s_-10C_2024_01_15.fit
│   │   └── flats\
│   │       └── flat_L_2024_01_15.fit
│   └── raw\
│       ├── biases\
│       │   └── 2024_01_15\
│       ├── darks\
│       │   └── 2024_01_15_-10C\
│       │       ├── 180\
│       │       └── 300\
│       └── flats\
│           └── 2024_01_15\
│               ├── L\
│               ├── R\
│               ├── G\
│               └── B\
├── M42\
│   └── 2024_01_15\
│       ├── L180\
│       ├── L30\
│       ├── R60\
│       ├── G60\
│       └── B60\
└── jobs\
    └── M42_Jan2024.json
```

**Key conventions:**
- Dates use underscores: `2024_01_15`
- Light folders: `{filter}{exposure}` (e.g., `L180`, `R60`)
- Bias: No temperature (readout noise is temperature-independent)
- Darks: `{date}_{temp}/{exposure}/` (thermal noise is temperature-dependent)
- Masters are cached in `calibration/masters/`

## Job File Format

```json
{
  "name": "M42_Jan2024",
  "type": "LRGB",
  "calibration": {
    "bias": "2024_01_15",
    "darks": "2024_01_15",
    "flats": "2024_01_20"
  },
  "lights": {
    "L": ["M42/2024_01_15/L180", "M42/2024_01_20/L180"],
    "R": ["M42/2024_01_15/R60"],
    "G": ["M42/2024_01_15/G60"],
    "B": ["M42/2024_01_15/B60"]
  },
  "output": "M42/processed",
  "options": {
    "fwhm_filter": 1.8,
    "temp_tolerance": 2
  }
}
```

### HDR Workflow

For bright objects requiring multiple exposures:

```json
{
  "lights": {
    "L": ["M42/2024_01_15/L180", "M42/2024_01_15/L30"],
    "R": ["M42/2024_01_15/R180", "M42/2024_01_15/R30"]
  }
}
```

This produces separate stacks per exposure:
- `stacks/stack_L_180s.fit`
- `stacks/stack_L_30s.fit`

Auto-composition is skipped; blend manually in PixInsight or other HDR tools.

See `examples/` for more job file examples.
