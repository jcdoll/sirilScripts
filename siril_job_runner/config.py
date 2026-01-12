"""
Centralized configuration defaults for Siril job processing.

All magic values and defaults should be defined here.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ClippingThresholds:
    """Thresholds for detecting clipped pixels."""

    # 16-bit thresholds
    low_16bit: int = 100
    high_16bit: int = 65000

    # 8-bit thresholds
    low_8bit: int = 5
    high_8bit: int = 250

    # Normalized float thresholds
    low_float: float = 0.01
    high_float: float = 0.99


@dataclass(frozen=True)
class StretchDefaults:
    """Default parameters for auto-stretch pipeline."""

    mtf_low: float = 0.20
    mtf_mid: float = 0.5
    mtf_high: float = 1.0
    saturation_amount: float = 1.0
    saturation_threshold: float = 0.0


@dataclass(frozen=True)
class ProcessingDefaults:
    """Default parameters for preprocessing."""

    fwhm_filter: float = 1.8
    temp_tolerance: float = 2.0
    linear_match_low: float = 0.0
    linear_match_high: float = 0.92


@dataclass(frozen=True)
class StackingDefaults:
    """Default parameters for stacking."""

    rejection: str = "rej"
    weighting: str = "w"
    sigma_low: str = "3"
    sigma_high: str = "3"
    norm: str = "addscale"


@dataclass(frozen=True)
class HDRDefaults:
    """HDR blending parameters.

    Algorithm uses brightness-weighted blending via Siril's PixelMath:
    - Pixels below low_threshold: use long exposure (better S/N in shadows)
    - Pixels above high_threshold: use short exposure (avoids clipping)
    - Between thresholds: smooth linear blend

    For 3+ exposures, blend iteratively from longest to shortest.
    """

    low_threshold: float = 0.7  # Start blending above this (normalized 0-1)
    high_threshold: float = 0.9  # Fully use short exposure above this


@dataclass(frozen=True)
class JobDefaults:
    """Default job-level options."""

    denoise: bool = False
    palette: str = "HOO"
    clipping_warning_threshold: float = 0.01


@dataclass(frozen=True)
class CalibrationDefaults:
    """Calibration directory and file naming conventions."""

    # Directory names
    base_dir: str = "calibration"
    masters_dir: str = "masters"
    raw_dir: str = "raw"

    # Subdirectory names
    bias_subdir: str = "biases"
    dark_subdir: str = "darks"
    flat_subdir: str = "flats"

    # File prefixes
    bias_prefix: str = "bias_"
    dark_prefix: str = "dark_"
    flat_prefix: str = "flat_"

    # Extensions and suffixes
    fit_extension: str = ".fit"
    fit_glob: str = "*.fit*"
    temp_suffix: str = "C"
    exposure_suffix: str = "s"

    # Siril conventions
    process_dir: str = "./process"
    calibrated_prefix: str = "pp_"

    # Stacking for calibration frames
    rejection: str = "rej"
    sigma: str = "3"
    flat_norm: str = "-norm=mul"
    no_norm: str = "-nonorm"


# Module-level instances for import
CLIPPING = ClippingThresholds()
STRETCH = StretchDefaults()
PROCESSING = ProcessingDefaults()
STACKING = StackingDefaults()
HDR = HDRDefaults()
CALIBRATION = CalibrationDefaults()
