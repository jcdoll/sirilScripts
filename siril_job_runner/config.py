"""
Centralized configuration for Siril job processing.

All configurable values are defined here in a single Config dataclass.
Users can override ANY value via job JSON files or settings.json.

Override precedence: DEFAULTS <- settings.json <- job.json options
"""

from dataclasses import dataclass, fields, replace
from enum import StrEnum
from typing import Optional


class StretchMethod(StrEnum):
    AUTOSTRETCH = "autostretch"
    VERALUX = "veralux"


class ColorRemovalMode(StrEnum):
    NONE = "none"
    GREEN = "green"
    MAGENTA = "magenta"


class SubskyMethod(StrEnum):
    NONE = "none"
    RBF = "rbf"
    POLY = "poly"


class StarColor(StrEnum):
    MONO = "mono"
    NATIVE = "native"


class BlendMode(StrEnum):
    SCREEN = "screen"
    LINEAR_ADD = "linear_add"


class PSFMethod(StrEnum):
    STARS = "stars"
    BLIND = "blind"


class Regularization(StrEnum):
    TV = "tv"
    FH = "fh"


class Channel(StrEnum):
    L = "L"
    R = "R"
    G = "G"
    B = "B"
    H = "H"
    S = "S"
    O = "O"  # noqa: E741 - standard OIII filter name


class StarSource(StrEnum):
    AUTO = "auto"
    L = "L"
    R = "R"
    G = "G"
    B = "B"
    H = "H"
    S = "S"
    O = "O"  # noqa: E741 - standard OIII filter name


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

    stretch_method: StretchMethod = StretchMethod.VERALUX

    # Compare stretch methods (saves both autostretch and veralux for comparison)
    stretch_compare: bool = True

    # Autostretch parameters
    autostretch_linked: bool = True  # False for narrowband
    autostretch_shadowclip: float = -2.8  # Shadows clipping in sigma from peak
    autostretch_targetbg: float = 0.10  # Target background brightness (match veralux)
    # MTF fine-tuning (applied after autostretch)
    autostretch_mtf_low: float = 0.0  # Black point [0-1]
    autostretch_mtf_mid: float = (
        0.6  # Midtone [0-1], higher = darker/compress highlights
    )
    autostretch_mtf_high: float = 1.0  # White point [0-1]

    # VeraLux HyperMetric Stretch parameters
    # Based on https://gitlab.com/free-astro/siril-scripts/-/blob/main/VeraLux/
    veralux_target_median: float = 0.10  # Target background median after stretch
    veralux_b: float = 6.0  # Highlight protection / curve knee (higher preserves stars)
    veralux_log_d_min: float = 0.0  # Min log_d for binary search (D = 10^log_d)
    veralux_log_d_max: float = 7.0  # Max log_d for binary search
    # Sensor profile for luminance weights (rec709, imx571, imx455, imx533, imx585, hoo, sho)
    veralux_sensor_profile: str = "imx571"
    # Hybrid mode parameters (RTU defaults: grip=1.0, shadow=0.0 means no hybrid)
    veralux_color_grip: float = 1.0  # Vector vs scalar blend 0.4-1.0 (1.0=pure vector)
    veralux_shadow_convergence: float = 0.0  # Shadow damping 0-3 (0=off)

    # VeraLux Revela - Detail enhancement (ATWT wavelets)
    veralux_revela_enabled: bool = True
    veralux_revela_texture: float = 50.0  # Fine detail boost 0-100
    veralux_revela_structure: float = 50.0  # Medium structure boost 0-100
    veralux_revela_shadow_auth: float = 25.0  # Shadow protection 0-100
    veralux_revela_protect_stars: bool = True

    # VeraLux Vectra - Smart saturation (LCH color space)
    veralux_vectra_enabled: bool = False
    veralux_vectra_saturation: float = 25.0  # Global saturation boost 0-100
    veralux_vectra_shadow_auth: float = 0.0  # Background protection 0-100
    veralux_vectra_protect_stars: bool = True
    veralux_vectra_red: Optional[float] = None  # Per-vector override
    veralux_vectra_yellow: Optional[float] = None
    veralux_vectra_green: Optional[float] = None
    veralux_vectra_cyan: Optional[float] = None
    veralux_vectra_blue: Optional[float] = None
    veralux_vectra_magenta: Optional[float] = None

    # VeraLux Silentium - Noise suppression (SWT wavelets)
    veralux_silentium_enabled: bool = True
    veralux_silentium_intensity: float = 25.0  # Luminance noise reduction 0-100
    veralux_silentium_detail_guard: float = 50.0  # Detail protection 0-100
    veralux_silentium_chroma: float = 30.0  # Chroma noise reduction 0-100
    veralux_silentium_shadow_smooth: float = 10.0  # Extra shadow smoothing 0-100

    # Star removal (Siril StarNet integration)
    # When enabled, runs starnet on linear data to create starless + starmask
    # If starcomposer is also enabled, recomposes with controlled star intensity
    starnet_enabled: bool = True
    starnet_stretch: bool = (
        True  # Apply internal MTF stretch (required for linear input)
    )
    starnet_upscale: bool = False  # 2x upscale for small stars (4x slower)
    starnet_stride: Optional[int] = (
        None  # Tile stride (default 256, dev recommends not changing)
    )

    # VeraLux StarComposer - Star compositing (requires starnet_enabled)
    # Recomposes stars onto starless image with hyperbolic stretch control
    veralux_starcomposer_log_d: float = 1.0  # Star intensity 0-2
    veralux_starcomposer_hardness: float = 6.0  # Profile hardness 1-100
    veralux_starcomposer_color_grip: float = 0.5  # Vector vs scalar 0-1
    veralux_starcomposer_shadow_conv: float = 0.0  # Shadow convergence 0-3
    veralux_starcomposer_blend_mode: BlendMode = BlendMode.SCREEN

    # Narrowband star options (used when starnet_enabled and narrowband job)
    narrowband_star_source: StarSource = StarSource.AUTO
    narrowband_star_color: StarColor = StarColor.MONO

    # Saturation (runs after all stretch methods)
    saturation_amount: float = 0.25  # 1.0 = +100%, override in job as needed
    saturation_background_factor: float = 1.0  # Threshold factor (0 disables)

    # Cross-channel registration (for composing stacks)
    # 2-pass: Siril auto-selects best reference (ignores setref)
    # 1-pass: Uses our setref (L for LRGB/LSHO, R for RGB, H for SHO/HOO)
    cross_reg_twopass: bool = True

    # Processing parameters
    temp_tolerance: float = 2.0
    linear_match_low: float = 0.0
    linear_match_high: float = 0.92
    linear_match_reference: Channel = Channel.R  # Reference for linear matching
    # Diagnostic outputs (for debugging color issues)
    diagnostic_previews: bool = (
        True  # Save stretched previews of individual stacks and RGB
    )

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

    # Background extraction - Pre-stack (seqsubsky on individual subs)
    # Applied before registration/stacking. RBF handles complex gradients,
    # poly degree 1 handles simple linear gradients.
    # Disable for narrowband (H/O/S) which has inherently low sky background.
    pre_stack_subsky_method: SubskyMethod = SubskyMethod.RBF
    pre_stack_subsky_degree: int = 1  # Polynomial degree if method="poly"
    pre_stack_subsky_samples: int = 20
    pre_stack_subsky_tolerance: float = 1.0
    pre_stack_subsky_smooth: float = 0.5  # RBF smoothing (0-1, higher=smoother)

    # Background extraction - Post-stack (subsky on each channel stack)
    # Applied after cross-registration, before RGB composition. Can clean up
    # residual gradients that seqsubsky missed due to stacking combining gradients.
    post_stack_subsky_method: SubskyMethod = SubskyMethod.POLY
    post_stack_subsky_degree: int = 2  # Polynomial degree if method="poly"
    post_stack_subsky_samples: int = 20
    post_stack_subsky_tolerance: float = 1.0
    post_stack_subsky_smooth: float = 0.5  # RBF smoothing (0-1, higher=smoother)

    # Background color neutralization (after stretch)
    # Removes color cast from background by linear matching G,B to R.
    # Separate from gradient extraction (subsky) which only removes brightness gradients.
    # Broadband disabled by default because SPCC handles color calibration.
    # Narrowband enabled by default because no SPCC equivalent exists.
    broadband_neutralization: bool = False
    narrowband_neutralization: bool = True
    # Bounds for linear_match - only pixels in this range are used for matching
    # For stretched data, background is ~0.1, so use low bounds to target background only
    broadband_neutralization_low: float = 0.0
    broadband_neutralization_high: float = 0.25

    # Narrowband channel balancing (for SHO/HOO/LSHO/LHOO)
    # Equalizes background levels between channels without affecting nebula signal.
    # Workflow: subsky each channel, then linear_match S and O to H.
    # Uses low high bounds to only match background/mid-tones, preserving emission ratios.
    narrowband_balance_enabled: bool = True
    narrowband_balance_reference: Channel = Channel.H  # Match other channels to this
    narrowband_balance_low: float = 0.0  # Ignore pixels below this (clip artifacts)
    narrowband_balance_high: float = (
        0.20  # Ignore pixels above this (stretched bg ~0.10)
    )

    # Deconvolution (sharpening via Richardson-Lucy)
    # For LRGB: runs on L stack and RGB composite
    # For RGB/SHO/HOO: runs on combined image
    # Docs: https://siril.readthedocs.io/en/stable/processing/deconvolution.html
    # Tips: https://siril.readthedocs.io/en/stable/processing/deconvolution.html#deconvolution-usage-tips
    deconv_enabled: bool = True
    deconv_psf_method: PSFMethod = PSFMethod.STARS
    deconv_save_psf: bool = True  # Save PSF images for inspection
    deconv_iterations: int = 10
    deconv_regularization: Regularization = Regularization.TV
    deconv_alpha: float = 3000.0  # Regularization strength (Siril default: 3000)

    # Color cast removal (SCNR - Subtractive Chromatic Noise Reduction)
    # Applied after color calibration, before stretch
    # For green cast: use rmgreen directly
    # For magenta cast: use negative-rmgreen-negative technique
    color_removal_mode: ColorRemovalMode = ColorRemovalMode.NONE
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

    # Narrowband palette for SHO/HOO composition
    # Available: HOO, SHO, SHO_FORAXX, SHO_DYNAMIC
    # Job "type" (LRGB, RGB, SHO, HOO) determines broadband vs narrowband
    # This "palette" option selects the color mapping for narrowband jobs
    palette: str = "SHO"

    # Optional per-channel formula overrides (PixelMath syntax)
    # Use channel names directly: "0.5*H + 0.5*O"
    # These override the selected palette's formulas
    palette_r_override: Optional[str] = None
    palette_g_override: Optional[str] = None
    palette_b_override: Optional[str] = None

    # Channel scale expressions (applied after stretch, before palette formulas)
    # PixelMath expressions to scale input channels before palette application.
    # Use cross-terms for signal-dependent scaling that neutralization can't undo.
    # Example: "O * (1 + 2 * H)" boosts O proportionally to H signal.
    # Reference: https://thecoldestnights.com/2020/06/pixinsight-dynamic-narrowband-combinations-with-pixelmath/
    palette_h_scale_expr: Optional[str] = None
    palette_o_scale_expr: Optional[str] = None
    palette_s_scale_expr: Optional[str] = None

    # LinearFit to weakest channel (for dynamic palettes)
    # Fits stronger channels to the weakest (typically O) in LINEAR space before stretch.
    # This balances peak intensities so the dynamic formula can produce both blue and gold.
    # Reference: https://thecoldestnights.com/2020/06/pixinsight-dynamic-narrowband-combinations-with-pixelmath/
    # See also: https://jonrista.com/the-astrophotographers-guide/pixinsights/narrow-band-combinations-with-pixelmath-hoo/
    palette_linearfit_to_weakest: bool = False

    dark_temp_override: Optional[float] = None
    force_reprocess: bool = False  # Force re-stacking even if cached

    # Output filename suffix (appended to all output basenames)
    # Useful for comparing different processing runs in the same output folder
    output_suffix: Optional[str] = None

    # Color calibration (SPCC)
    # Use spcc_list command in Siril to see available sensors/filters
    # Note: SPCC only applies to broadband LRGB, not narrowband SHO
    spcc_enabled: bool = True
    spcc_sensor: str = "Sony IMX411/455/461/533/571"
    spcc_red_filter: str = "Optolong Red"
    spcc_green_filter: str = "Optolong Green"
    spcc_blue_filter: str = "Optolong Blue"
    spcc_whiteref: str = "Average Spiral Galaxy"
    spcc_bgtol_upper: float = 2.0
    spcc_bgtol_lower: float = 2.8  # Specified as positive, means -2.8 sigma
    spcc_obsheight: int = 1000  # Observation height in meters


# Default configuration instance
DEFAULTS = Config()


def get_valid_options() -> set[str]:
    """Get all valid option names."""
    return {f.name for f in fields(Config)}


def with_overrides(overrides: dict) -> Config:
    """
    Create a Config with overrides applied.

    Validates that all override keys are valid option names.
    Coerces string values to StrEnum types for enum-typed fields.
    Raises ValueError for unknown options or invalid enum values.
    """
    if not overrides:
        return DEFAULTS

    valid = get_valid_options()
    invalid = set(overrides.keys()) - valid
    if invalid:
        raise ValueError(f"Unknown config options: {sorted(invalid)}")

    # Coerce string values to StrEnum types where applicable
    type_hints = {f.name: f.type for f in fields(Config)}
    coerced = {}
    for key, value in overrides.items():
        field_type = type_hints.get(key)
        if (
            isinstance(value, str)
            and isinstance(field_type, type)
            and issubclass(field_type, StrEnum)
        ):
            coerced[key] = field_type(value)
        else:
            coerced[key] = value

    return replace(DEFAULTS, **coerced)


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
