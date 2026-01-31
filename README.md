# autoSiril

Automation tools for astrophotography image processing.

## Projects

### Siril Job Runner

Automated Siril processing pipeline with JSON job file configuration.

```bash
# Run a processing job
uv run python run_job.py jobs/M42.json
```

Features:
- Job file-based configuration for reproducible processing
- Auto-detection of calibration requirements from FITS headers
- Temperature tolerance matching for darks/bias
- Multi-night light frame support
- Support for LRGB, SHO, and HOO workflows

See [siril_job_runner/README.md](siril_job_runner/README.md) for full documentation.

### XISF to FITS Converter

Batch convert XISF files to FITS format.

```bash
# Convert all XISF files, excluding processed folders
uv run python -m xisf_to_fits /path/to/images -e process
```

Features:
- Recursive directory scanning
- Exclude patterns for processed/calibration folders
- Progress bar with ETA
- Verification of converted files

See [xisf_to_fits/README.md](xisf_to_fits/README.md) for full documentation.

## Installation

Requires [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone https://github.com/yourusername/autoSiril.git
cd autoSiril
uv sync
```

## Development

```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov
```
