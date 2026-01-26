# Code Structure

Orientation guide for the Siril Job Runner codebase.

## Repository Layout

```
sirilScripts/
├── docs/                     # Documentation
├── examples/                 # Example job files
├── jobs/                     # User job files
├── logs/                     # Processing logs
├── siril_job_runner/         # Main package
├── tests/                    # pytest tests
├── xisf_to_fits/             # Separate XISF converter tool
├── run_job.py                # CLI entry point
└── settings.template.json    # User settings template
```

## Core Concepts

- Job files are JSON configurations defining what to process (target, filters, calibration dates, options)
- `Config` dataclass in `config.py` holds all processing parameters; users override via `settings.json` or job options
- `SirilWrapper` in `siril_wrapper.py` wraps pysiril; all Siril operations go through this interface

## Processing Pipeline

```
run_job.py → JobRunner
                │
                ├── 1. Validation: scan FITS headers, check calibration
                │
                ├── 2. Calibration: build missing masters (bias, dark, flat)
                │
                ├── 3. Preprocessing: calibrate → register → stack (per filter+exposure)
                │
                └── 4. Composition: combine stacks → stretch → enhance → save outputs
```

## Module Groups

| Group | Purpose | Key Files |
|-------|---------|-----------|
| Orchestration | Pipeline control | `job_runner.py`, `job_config.py` |
| Calibration | Master frame building | `calibration.py` |
| Preprocessing | Frame processing | `preprocessing.py`, `preprocessing_pipeline.py` |
| Composition | Image combination | `composition.py`, `compose_broadband.py`, `compose_narrowband.py` |
| VeraLux | Stretch and enhancement | `veralux_*.py` (stretch, denoise, sharpen, saturate, star recomposition) |
| Siril Interface | Siril command wrappers | `siril_wrapper.py`, `siril_*.py` |
| Utilities | Shared helpers | `models.py`, `logger.py`, `fits_utils.py` |

## Key Entry Points

- `run_job.py` - CLI, parses args, loads job, runs pipeline
- `JobRunner` in `job_runner.py` - orchestrates all stages
- `compose_and_stretch()` in `composition.py` - routes to broadband/narrowband handlers

## Configuration

All defaults live in `Config` dataclass (`config.py`). Override precedence:

```
defaults → settings.json → job file options
```

See `config.py` for available options; `job_schema.json` for job file format.
