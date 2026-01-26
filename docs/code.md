# Code Structure

High-level overview of the Siril Job Runner codebase.

## Repository Layout

```
sirilScripts/
├── docs/
│   ├── architecture.md      # Processing workflow documentation
│   └── code.md              # This file - code structure overview
├── examples/
│   ├── example_lrgb_job.json
│   ├── example_lrgb_hdr_job.json
│   ├── example_sho_job.json
│   └── example_hoo_job.json
├── jobs/                     # User job files
├── logs/                     # Processing logs
├── siril_job_runner/
│   ├── __init__.py
│   ├── calibration.py        # Calibration master building
│   ├── calibration_paths.py  # Path resolution
│   ├── compose_broadband.py  # LRGB/RGB composition
│   ├── compose_narrowband.py # SHO/HOO composition
│   ├── composition.py        # Composition orchestration
│   ├── config.py             # Centralized configuration
│   ├── fits_utils.py         # FITS header parsing
│   ├── hdr.py                # HDR blending
│   ├── job_config.py         # Job file loading
│   ├── job_runner.py         # Pipeline orchestration
│   ├── job_schema.json       # Job file schema
│   ├── job_validation.py     # Job validation
│   ├── logger.py             # Logging utilities
│   ├── models.py             # Shared data models
│   ├── palettes.py           # Narrowband palettes
│   ├── preprocessing.py      # Frame preprocessing
│   ├── preprocessing_pipeline.py
│   ├── preprocessing_utils.py
│   ├── protocols.py          # Interface protocols
│   ├── sequence_*.py         # Sequence file parsing/analysis
│   ├── siril_*.py            # Siril operation wrappers
│   ├── veralux_*.py          # VeraLux processing modules
│   └── README.md             # Module documentation
├── tests/
│   ├── conftest.py
│   └── test_*.py
├── xisf_to_fits/             # XISF converter (separate tool)
├── run_job.py                # CLI entry point
├── settings.template.json    # Settings template
├── pyproject.toml
└── README.md
```

## Module Organization

```
siril_job_runner/
├── Core Orchestration
│   ├── job_runner.py        # Main pipeline orchestration
│   ├── job_config.py        # Job file loading and parsing
│   ├── job_validation.py    # Job validation logic
│   └── job_schema.json      # JSON schema for job files
│
├── Configuration
│   ├── config.py            # Centralized Config dataclass
│   └── models.py            # Shared data models
│
├── Calibration
│   ├── calibration.py       # CalibrationManager class
│   └── calibration_paths.py # Path resolution for masters/raw
│
├── Preprocessing
│   ├── preprocessing.py     # High-level preprocessing entry
│   ├── preprocessing_pipeline.py  # Pipeline steps
│   └── preprocessing_utils.py     # Sequence file utilities
│
├── Composition
│   ├── composition.py       # Composer class, entry point
│   ├── compose_broadband.py # LRGB/RGB composition
│   ├── compose_narrowband.py # SHO/HOO composition
│   ├── compose_helpers.py   # Shared composition utilities
│   ├── hdr.py               # HDR blending
│   ├── palettes.py          # Narrowband palette definitions
│   └── stack_discovery.py   # Stack file discovery
│
├── Siril Interface
│   ├── siril_wrapper.py     # SirilWrapper class
│   ├── siril_color.py       # Color operations (SPCC, SCNR)
│   ├── siril_file_ops.py    # File operations (load, save)
│   ├── siril_registration.py # Registration operations
│   ├── siril_stretch.py     # Stretch operations
│   └── protocols.py         # SirilInterface protocol
│
├── VeraLux Processing
│   ├── veralux_stretch.py   # Main stretch orchestration
│   ├── veralux_core.py      # HyperMetric stretch algorithm
│   ├── veralux_silentium.py # Noise suppression (SWT)
│   ├── veralux_revela.py    # Detail enhancement (ATWT)
│   ├── veralux_vectra.py    # Smart saturation (LCH)
│   ├── veralux_starcomposer.py # Star recomposition
│   ├── veralux_colorspace.py   # Color space conversions
│   └── veralux_wavelet.py   # Wavelet transforms
│
├── Frame Analysis
│   ├── fits_utils.py        # FITS header parsing
│   ├── frame_analysis.py    # Frame quality analysis
│   ├── psf_analysis.py      # PSF analysis for deconvolution
│   ├── sequence_parse.py    # .seq file parsing
│   ├── sequence_stats.py    # Sequence statistics
│   ├── sequence_threshold.py # FWHM threshold computation
│   └── sequence_analysis.py # High-level sequence analysis
│
├── Utilities
│   ├── logger.py            # JobLogger class
│   └── stretch_helpers.py   # Stretch utility functions
│
└── __init__.py
```

## Key Data Structures

### Job Configuration

```python
@dataclass
class JobConfig:
    name: str                          # Job name for logging
    job_type: str                      # LRGB, RGB, SHO, HOO, LSHO, LHOO
    calibration_bias: str              # Date string for bias
    calibration_darks: str             # Date string for darks
    calibration_flats: str             # Date string for flats
    lights: dict[str, list[str]]       # filter -> list of directories
    output: str                        # Output directory path
    config: Config                     # Processing configuration
```

### Processing Configuration

```python
@dataclass
class Config:
    # All configurable values with defaults
    # Users can override any field via settings.json or job options

    # Stretch settings
    stretch_method: str = "veralux"
    stretch_compare: bool = True
    veralux_target_median: float = 0.10

    # Star removal
    starnet_enabled: bool = True

    # Color calibration
    spcc_enabled: bool = True

    # Narrowband
    palette: str = "SHO"
    narrowband_star_source: str = "auto"

    # ... many more options
```

### Frame Information

```python
@dataclass
class FrameInfo:
    path: Path           # Full path to FITS file
    exposure: float      # Exposure time in seconds
    temperature: float   # Sensor temperature in Celsius
    filter_name: str     # Filter name (L, R, G, B, H, O, S)
    gain: Optional[int]  # Camera gain setting
```

### Stack Information

```python
@dataclass
class StackInfo:
    path: Path           # Path to stacked FITS file
    filter_name: str     # Filter name
    exposure: int        # Exposure time in seconds
```

### Calibration Status

```python
@dataclass
class CalibrationStatus:
    exists: bool                  # Master already exists
    can_build: bool               # Raw frames available to build
    master_path: Optional[Path]   # Path to master (if exists)
    raw_path: Optional[Path]      # Path to raw frames
    message: str                  # Status description
```

### Validation Result

```python
@dataclass
class ValidationResult:
    valid: bool                        # Overall validation status
    frames: list[FrameInfo]            # All discovered light frames
    requirements: list                 # Calibration requirements
    missing_calibration: list[str]     # Missing and unbuildable
    buildable_calibration: list[str]   # Missing but buildable
    message: str                       # Validation summary
```

### Composition Result

```python
@dataclass
class CompositionResult:
    linear_path: Path              # Unstretched composed image
    linear_pcc_path: Optional[Path] # Color-calibrated linear (SPCC)
    auto_fit: Path                 # Auto-stretched .fit
    auto_tif: Path                 # Auto-stretched .tif
    auto_jpg: Path                 # Auto-stretched .jpg
    stacks_dir: Path               # Directory containing stacks
```

## Processing Flow

### 1. Job Loading

```
run_job.py
    └── job_config.load_job()
        ├── Parse JSON file
        ├── Validate against schema
        ├── Merge settings.json + job options
        └── Return JobConfig
```

### 2. Validation

```
JobRunner.validate()
    └── job_validation.validate_job()
        ├── Scan all light directories for FITS files
        ├── Extract headers (exposure, temp, filter)
        ├── Build calibration requirements
        ├── Check master availability
        └── Return ValidationResult
```

### 3. Calibration

```
JobRunner.run_calibration()
    └── CalibrationManager
        ├── build_bias_master()
        ├── build_dark_master() for each (exposure, temp)
        └── build_flat_master() for each filter
```

### 4. Preprocessing

```
JobRunner.run_preprocessing()
    └── preprocessing.preprocess_with_exposure_groups()
        ├── Group frames by (filter, exposure)
        └── For each group:
            └── preprocessing_pipeline.run_pipeline()
                ├── Create sequence file
                ├── Calibrate (bias/dark + flat)
                ├── Pre-stack background extraction
                ├── Register (2-pass)
                ├── Compute FWHM threshold
                ├── Apply registration with filtering
                └── Stack
```

### 5. Composition

```
JobRunner.run_composition()
    └── composition.compose_and_stretch()
        ├── Discover stacks
        ├── HDR blend if multiple exposures
        └── Route by job type:
            ├── LRGB/RGB → compose_broadband.py
            └── SHO/HOO/etc → compose_narrowband.py
```

### 6. Broadband Composition (LRGB/RGB)

```
compose_broadband.compose_lrgb()
    ├── Cross-register stacks
    ├── Post-stack background extraction
    ├── Compose RGB (rgbcomp R G B)
    ├── SPCC color calibration
    ├── Optional deconvolution
    ├── StarNet on linear RGB (if enabled)
    ├── For each stretch method (autostretch, veralux):
    │   ├── Stretch
    │   ├── Combine LRGB (rgbcomp -lum)
    │   ├── SCNR color removal
    │   ├── Background neutralization
    │   ├── Saturation
    │   ├── VeraLux enhancements (Silentium/Revela/Vectra)
    │   └── Save outputs
    └── StarComposer recomposition (if StarNet enabled)
```

### 7. Narrowband Composition (SHO/HOO)

```
compose_narrowband.compose_narrowband()
    ├── Cross-register stacks
    ├── Post-stack background extraction
    ├── Channel balancing (linear_match to H)
    ├── StarNet on each channel (if enabled)
    ├── Extract stars from L or H
    ├── For each stretch method:
    │   ├── Stretch all channels
    │   ├── Apply channel scale expressions
    │   ├── Apply palette formulas → RGB
    │   ├── Add L as luminance (if LSHO/LHOO)
    │   ├── SCNR color removal
    │   ├── Background neutralization
    │   ├── Composite stars back
    │   ├── Saturation
    │   └── Save outputs
```

## Siril Interface

The `SirilInterface` protocol defines the contract for Siril operations. `SirilWrapper` implements this by wrapping pysiril.

```python
class SirilInterface(Protocol):
    # Navigation
    def cd(self, path: str) -> bool: ...

    # File operations
    def load(self, filename: str) -> bool: ...
    def save(self, filename: str) -> bool: ...

    # Calibration
    def calibrate(self, seq: str, ...) -> bool: ...
    def stack(self, seq: str, ...) -> bool: ...

    # Registration
    def register(self, seq: str, ...) -> bool: ...
    def seqapplyreg(self, seq: str, ...) -> bool: ...

    # Color
    def spcc(self, ...) -> bool: ...
    def rgbcomp(self, ...) -> bool: ...

    # Processing
    def autostretch(self, ...) -> bool: ...
    def starnet(self, ...) -> bool: ...
    # ... etc
```

## VeraLux Processing

VeraLux modules operate on numpy arrays, not Siril directly:

```python
# Load image into numpy
img = fits_to_array(path)

# Apply VeraLux stretch
stretched = veralux_stretch(img, config)

# Apply enhancements
if config.veralux_silentium_enabled:
    stretched = silentium_denoise(stretched, config)
if config.veralux_revela_enabled:
    stretched = revela_enhance(stretched, config)
if config.veralux_vectra_enabled:
    stretched = vectra_saturate(stretched, config)

# Save back to FITS
array_to_fits(stretched, output_path)
```

## Configuration Precedence

```
DEFAULTS (config.py)
    ↓ override
settings.json (repository root)
    ↓ override
job.json options field
    ↓
Final Config
```

Any field in `Config` can be overridden at any level.
