# XISF to FITS Converter

Batch convert XISF files to FITS format. Recursively finds all `.xisf` files and converts them to `.fits`.

## Usage

```bash
# Convert all XISF files in directory tree
uv run python -m xisf_to_fits /path/to/images

# Dry run - see what would be converted
uv run python -m xisf_to_fits /path/to/images --dry-run

# Overwrite existing FITS files
uv run python -m xisf_to_fits /path/to/images --overwrite

# Exclude processed files (e.g., PixInsight output folders)
uv run python -m xisf_to_fits /path/to/images -e process

# Exclude multiple patterns
uv run python -m xisf_to_fits /path/to/images -e process -e calibrated

# Only convert light frames
uv run python -m xisf_to_fits /path/to/images --pattern "Light_*.xisf"

# Output to separate directory
uv run python -m xisf_to_fits /path/to/images --output-dir /path/to/output

# Skip verification (faster)
uv run python -m xisf_to_fits /path/to/images --no-verify --no-stats
```

## Options

| Option | Description |
|--------|-------------|
| `--dry-run, -d` | Show what would be converted without doing it |
| `--overwrite, -o` | Overwrite existing FITS files |
| `--exclude, -e` | Exclude paths containing string (can use multiple times) |
| `--pattern, -p` | Filename pattern filter (e.g., `Light_*.xisf`) |
| `--output-dir, -O` | Output directory (default: same as input) |
| `--no-recursive, -n` | Do not search subdirectories |
| `--no-verify` | Skip verification of output files |
| `--no-stats` | Skip pixel statistics verification |
| `--jobs, -j` | Number of parallel workers (default: 1) |

## Notes

- Original XISF files are never deleted
- Conversion is idempotent - safe to re-run
- Progress bar shows current file and ETA
- Failed files are listed in summary at end
