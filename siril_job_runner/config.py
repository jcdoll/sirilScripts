"""
Centralized configuration for Siril job processing.

All configurable values are defined here in a single Config dataclass.
Users can override ANY value via job JSON files or settings.json.

Override precedence: DEFAULTS <- settings.json <- job.json options
"""

from dataclasses import asdict, dataclass, fields, replace
from typing import Optional


@dataclass
class Config:
    """
    All configuration values. User can override any field.

    To add a new option: add a field here with a default value.
    It will automatically be available for override in job files.
    """

    # Clipping detection thresholds
    clipping_low_16bit: int = 100
    clipping_high_16bit: int = 65000
    clipping_low_8bit: int = 5
    clipping_high_8bit: int = 250
    clipping_low_float: float = 0.01
    clipping_high_float: float = 0.99
    clipping_warning_threshold: float = 0.01
    float_normalized_threshold: float = 1.5  # Threshold to detect normalized float data

    # Fallback values
    default_temperature: float = 0.0  # Used when temperature not in FITS header

    # Stretch method: "autostretch", "modasinh", "ght", "autoghs"
    stretch_method: str = "autostretch"

    # Compare all stretch methods (saves each with suffix for comparison)
    stretch_compare: bool = True

    # Autostretch parameters
    autostretch_linked: bool = True  # False for narrowband
    autostretch_shadowclip: float = -2.8  # Shadows clipping in sigma from peak
    autostretch_targetbg: float = 0.10  # Target background brightness (lower=darker)

    # GHT / modasinh / autoghs parameters
    # modasinh ignores ght_B; autoghs ignores ght_SP (calculates it)
    ght_D: float = 3.0  # Stretch strength (0-10)
    ght_B: float = 0.0  # Focal point, GHT only (-5 to 15)
    ght_LP: float = 0.0  # Shadow protection
    ght_SP: float = 0.0  # Symmetry point (ght/modasinh only, autoghs ignores)
    ght_HP: float = 0.85  # Highlight protection - preserves stars
    autoghs_shadowsclip: float = 0.0  # k*sigma from median for SP calculation

    # Saturation (runs after all stretch methods)
    saturation_amount: float = 0.15  # 1.0 = +100%, 0.15 = subtle boost
    saturation_background_factor: float = 1.0  # Threshold factor (0 disables)

    # Processing parameters
    temp_tolerance: float = 2.0
    linear_match_low: float = 0.0
    linear_match_high: float = 0.92

    # FWHM adaptive filtering (after registration)
    # Decision tree:
    #   1. Bimodal (GMM+dip): threshold at midpoint between modes
    #   2. Skewed (long tail): threshold at median + k*MAD (aggressive)
    #   3. Broad symmetric (high CV): threshold at percentile (permissive)
    #   4. Tight symmetric (low CV): keep all images
    fwhm_min_images: int = 6  # minimum images for statistical filtering
    fwhm_bic_threshold: float = 10.0  # delta-BIC for bimodality (>10 = very strong)
    fwhm_dip_alpha: float = 0.05  # significance level for Hartigan dip test
    fwhm_bimodal_sigma: float = 3.0  # sigmas above lower mode (fallback if > midpoint)
    fwhm_skew_threshold: float = 1.0  # skewness above which distribution is "skewed"
    fwhm_skew_mad_factor: float = 2.0  # MAD multiplier for skewed threshold
    fwhm_cv_threshold: float = 0.15  # CV below which all images kept (15%)
    fwhm_broad_percentile: float = 95.0  # percentile for broad symmetric case

    # Background extraction (post-stack subsky)
    # Disable for narrowband (H/O/S) which has inherently low sky background
    subsky_enabled: bool = True
    subsky_rbf: bool = True  # Use RBF interpolation (preferred) vs polynomial degree
    subsky_degree: int = 1  # Polynomial degree if subsky_rbf is False
    subsky_samples: int = 20
    subsky_tolerance: float = 1.0
    subsky_smooth: float = 0.5

    # Deconvolution (sharpening via Richardson-Lucy)
    # For LRGB: runs on L stack and RGB composite
    # For RGB/SHO/HOO: runs on combined image
    # Docs: https://siril.readthedocs.io/en/stable/processing/deconvolution.html
    # Tips: https://siril.readthedocs.io/en/stable/processing/deconvolution.html#deconvolution-usage-tips
    deconv_enabled: bool = False
    deconv_psf_method: str = (
        "stars"  # "stars" (from detected stars) or "blind" (estimate)
    )
    deconv_save_psf: bool = True  # Save PSF images for inspection
    deconv_iterations: int = 10
    deconv_regularization: str = (
        "tv"  # "tv" (Total Variation) or "fh" (Frobenius Hessian)
    )
    deconv_alpha: float = 3000.0  # Regularization strength (Siril default: 3000)

    # Color cast removal (SCNR - Subtractive Chromatic Noise Reduction)
    # Applied after color calibration, before stretch
    # For green cast: use rmgreen directly
    # For magenta cast: use negative-rmgreen-negative technique
    color_removal_mode: str = "none"  # "none", "green", "magenta"
    rmgreen_type: int = (
        0  # 0=average neutral, 1=maximum neutral, 2=max mask, 3=additive
    )
    rmgreen_amount: float = 1.0  # Strength 0-1, only for types 2/3
    rmgreen_preserve_lightness: bool = True

    # Light frame stacking
    stack_rejection: str = "rej"
    stack_weighting: str = "w"
    stack_sigma_low: str = "3"
    stack_sigma_high: str = "3"
    stack_norm: str = "addscale"

    # HDR blending (brightness-weighted via PixelMath)
    hdr_low_threshold: float = 0.7
    hdr_high_threshold: float = 0.9

    # Calibration directory structure
    calibration_base_dir: str = "calibration"
    calibration_masters_dir: str = "masters"
    calibration_raw_dir: str = "raw"
    bias_subdir: str = "biases"
    dark_subdir: str = "darks"
    flat_subdir: str = "flats"

    # Calibration file naming
    bias_prefix: str = "bias_"
    dark_prefix: str = "dark_"
    flat_prefix: str = "flat_"
    fit_extension: str = ".fit"
    fit_glob: str = "*.fit*"
    temp_suffix: str = "C"
    exposure_suffix: str = "s"

    # Siril processing conventions
    process_dir: str = "./process"
    calibrated_prefix: str = "pp_"

    # Calibration frame stacking
    cal_rejection: str = "rej"
    cal_sigma: str = "3"
    cal_flat_norm: str = "-norm=mul"
    cal_no_norm: str = "-nonorm"

    # Job options
    denoise: bool = False
    palette: str = "HOO"
    dark_temp_override: Optional[float] = None
    force_reprocess: bool = False  # Force re-stacking even if cached

    # Color calibration (SPCC)
    # Use spcc_list command in Siril to see available sensors/filters
    # Note: SPCC only applies to broadband LRGB, not narrowband SHO
    spcc_enabled: bool = True
    spcc_sensor: str = "Sony_IMX571"
    spcc_red_filter: str = "Optolong_Red"
    spcc_green_filter: str = "Optolong_Green"
    spcc_blue_filter: str = "Optolong_Blue"


# Default configuration instance
DEFAULTS = Config()


def get_valid_options() -> set[str]:
    """Get all valid option names."""
    return {f.name for f in fields(Config)}


def with_overrides(overrides: dict) -> Config:
    """
    Create a Config with overrides applied.

    Validates that all override keys are valid option names.
    Raises ValueError for unknown options.
    """
    if not overrides:
        return DEFAULTS

    valid = get_valid_options()
    invalid = set(overrides.keys()) - valid
    if invalid:
        raise ValueError(f"Unknown config options: {sorted(invalid)}")

    return replace(DEFAULTS, **overrides)


def merge_overrides(*override_dicts: dict) -> dict:
    """
    Merge multiple override dicts (later dicts take precedence).

    Useful for: DEFAULTS <- settings.json <- job.json
    """
    result = {}
    for d in override_dicts:
        if d:
            result.update(d)
    return result


def config_to_dict(config: Config) -> dict:
    """Convert Config to dict (for serialization)."""
    return asdict(config)
