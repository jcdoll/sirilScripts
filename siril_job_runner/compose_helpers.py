"""
Shared helper functions for composition modules.

Color removal and other utilities used by broadband and narrowband composition.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from .config import ColorRemovalMode, Config
from .psf_analysis import analyze_psf, format_psf_stats

if TYPE_CHECKING:
    from .protocols import SirilInterface


def save_diagnostic_preview(
    siril: "SirilInterface",
    name: str,
    output_dir: Path,
    log_fn: Callable[[str], None],
) -> None:
    """Save a diagnostic preview JPG of the currently loaded image."""
    siril.load(name)
    siril.autostretch()
    output_path = output_dir / f"diag_{name}.jpg"
    siril.savejpg(str(output_path.with_suffix("")), 90)
    log_fn(f"Diagnostic preview: {output_path.name}")
    # Reload original unstretched data so subsequent operations work on linear data
    siril.load(name)


def apply_color_removal(
    siril: "SirilInterface",
    config: Config,
    log_fn: Callable[[str], None],
) -> bool:
    """Apply color cast removal based on config mode."""
    if config.color_removal_mode == ColorRemovalMode.NONE:
        return True

    if config.color_removal_mode == ColorRemovalMode.GREEN:
        log_fn("Removing green cast (SCNR)...")
        return siril.rmgreen(
            type=config.rmgreen_type,
            amount=config.rmgreen_amount,
            preserve_lightness=config.rmgreen_preserve_lightness,
        )
    elif config.color_removal_mode == ColorRemovalMode.MAGENTA:
        log_fn("Removing magenta cast (negative-SCNR-negative)...")
        if not siril.negative():
            return False
        if not siril.rmgreen(
            type=config.rmgreen_type,
            amount=config.rmgreen_amount,
            preserve_lightness=config.rmgreen_preserve_lightness,
        ):
            return False
        return siril.negative()
    else:
        log_fn(f"Unknown color_removal_mode: {config.color_removal_mode}")
        return True


def apply_spcc_step(
    siril: "SirilInterface",
    config: Config,
    output_dir: Path,
    type_name: str,
    log_fn: Callable[[str], None],
) -> tuple[Optional[Path], str]:
    """Apply SPCC and return (pcc_path, stretch_source)."""
    linear_pcc_path = None
    stretch_source = type_name

    if config.spcc_enabled:
        log_fn("Plate solving for SPCC...")
        if not siril.platesolve():
            log_fn("Plate solve failed, SPCC requires solved image - skipping")
        else:
            log_fn("Plate solve succeeded, running SPCC...")
            # Siril quoting rule: arguments with spaces need quotes around the
            # entire argument including the -flag part, e.g. "-monosensor=Sony IMX571"
            if siril.spcc(
                sensor=config.spcc_sensor,
                red_filter=config.spcc_red_filter,
                green_filter=config.spcc_green_filter,
                blue_filter=config.spcc_blue_filter,
                whiteref=config.spcc_whiteref,
                bgtol_upper=config.spcc_bgtol_upper,
                bgtol_lower=config.spcc_bgtol_lower,
                obsheight=config.spcc_obsheight,
            ):
                linear_pcc_path = output_dir / f"{type_name}_linear_spcc.fit"
                siril.save(str(linear_pcc_path))
                stretch_source = str(linear_pcc_path)
                log_fn(f"Saved color-calibrated: {linear_pcc_path.name}")
            else:
                log_fn("SPCC failed, using uncalibrated image")
    else:
        log_fn("SPCC disabled, skipping color calibration")

    return linear_pcc_path, stretch_source


def neutralize_rgb_background(
    siril: "SirilInterface",
    image_name: str,
    config: Config,
    log_fn: Callable[[str], None],
    low: Optional[float] = None,
    high: Optional[float] = None,
) -> bool:
    """
    Neutralize RGB background color by linear matching G,B to R.

    Removes color cast from background after stretch. Works for both broadband
    and narrowband by using appropriate bounds for linear_match.

    Note: This only handles COLOR neutralization, not gradient removal.
    Use subsky separately for gradient extraction.

    Steps:
    1. Split RGB into separate channels
    2. Linear match G and B to R (only background pixels via bounds)
    3. Recombine channels

    Args:
        siril: Siril interface
        image_name: Name of RGB image to neutralize
        config: Config (unused, kept for API compatibility)
        log_fn: Logging function
        low: Low bound for linear_match (default: narrowband_balance_low)
        high: High bound for linear_match (default: narrowband_balance_high)

    Returns:
        True if successful, False otherwise
    """
    log_fn("Neutralizing RGB background color...")

    # Step 1: Split into R, G, B channels
    siril.load(image_name)
    if not siril.split("_nb_r", "_nb_g", "_nb_b"):
        log_fn("  split failed, skipping neutralization")
        return False
    log_fn("  split into R, G, B channels")

    # Step 2: Linear match G and B to R
    # Use provided bounds, or fall back to narrowband defaults
    if low is None:
        low = config.narrowband_balance_low
    if high is None:
        high = config.narrowband_balance_high
    log_fn(f"  linear matching G, B to R (bounds: {low:.2f}-{high:.2f})")

    siril.load("_nb_g")
    if not siril.linear_match(ref="_nb_r", low=low, high=high):
        log_fn("  linear_match G failed")
        return False
    siril.save("_nb_g")

    siril.load("_nb_b")
    if not siril.linear_match(ref="_nb_r", low=low, high=high):
        log_fn("  linear_match B failed")
        return False
    siril.save("_nb_b")

    # Step 4: Recombine channels
    if not siril.rgbcomp(r="_nb_r", g="_nb_g", b="_nb_b", out=image_name):
        log_fn("  rgbcomp failed")
        return False
    log_fn("  recombined RGB")

    return True


def apply_deconvolution(
    siril: "SirilInterface",
    source_name: str,
    output_name: str,
    config: Config,
    output_dir: Path,
    log_fn: Callable[[str], None],
    psf_suffix: str,
) -> str:
    """
    Apply Richardson-Lucy deconvolution to an image.

    Args:
        siril: Siril interface
        source_name: Name of source image (Siril working name)
        output_name: Name for deconvolved output
        config: Configuration with deconvolution settings
        output_dir: Directory for PSF output file
        log_fn: Logging function
        psf_suffix: Suffix for PSF filename (e.g., "rgb", "L", "sho")

    Returns:
        Name of result image (output_name if successful, source_name if failed)
    """
    siril.load(source_name)

    psf_path = (
        str(output_dir / f"psf_{psf_suffix}.fit") if config.deconv_save_psf else None
    )

    if siril.makepsf(
        method=config.deconv_psf_method,
        symmetric=True,
        save_psf=psf_path,
    ):
        if psf_path:
            log_fn(f"PSF saved: psf_{psf_suffix}.fit")
            psf_stats = analyze_psf(Path(psf_path))
            if psf_stats:
                for line in format_psf_stats(psf_stats):
                    log_fn(f"  {line}")

        if siril.rl(
            iters=config.deconv_iterations,
            regularization=config.deconv_regularization,
            alpha=config.deconv_alpha,
        ):
            siril.save(output_name)
            return output_name

        log_fn("Deconvolution failed, using original")
    else:
        log_fn("PSF creation failed, skipping deconvolution")

    return source_name
