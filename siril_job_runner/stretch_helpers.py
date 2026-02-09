"""
Stretch and finalization helpers for image composition.

Clean standalone functions for stretching, saturation, enhancements, and saving.
Used by both broadband and narrowband composition modules.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from .config import Config, StretchMethod

if TYPE_CHECKING:
    from .protocols import SirilInterface


def apply_stretch(
    siril: "SirilInterface",
    method: str,
    source_path: Path,
    config: Config,
    log_fn: Callable[[str], None],
) -> bool:
    """
    Apply a single stretch method to currently loaded image.

    Args:
        siril: Siril interface
        method: "autostretch" or "veralux"
        source_path: Path to source image (needed for veralux stats)
        config: Configuration
        log_fn: Logging function

    Returns:
        True if successful, False otherwise
    """
    from . import veralux_stretch

    cfg = config

    if method == StretchMethod.AUTOSTRETCH:
        mode = "linked" if cfg.autostretch_linked else "unlinked"
        log_fn(f"Stretching ({method}, {mode}, targetbg={cfg.autostretch_targetbg})...")
        success = siril.autostretch(
            linked=cfg.autostretch_linked,
            shadowclip=cfg.autostretch_shadowclip,
            targetbg=cfg.autostretch_targetbg,
        )
        if success:
            # Apply second MTF for additional stretch control
            # mtf_mid < 0.5 brightens, > 0.5 darkens, 0.5 is neutral
            low, mid, high = (
                cfg.autostretch_mtf_low,
                cfg.autostretch_mtf_mid,
                cfg.autostretch_mtf_high,
            )
            is_neutral = low == 0.0 and mid == 0.5 and high == 1.0
            if not is_neutral:
                log_fn(f"  MTF: low={low}, mid={mid}, high={high}")
                siril.mtf(low, mid, high)
        return success
    elif method == StretchMethod.VERALUX:
        log_fn(
            f"Stretching ({method}, target_median={cfg.veralux_target_median}, "
            f"b={cfg.veralux_b})..."
        )
        success, _log_d = veralux_stretch.apply_stretch(
            siril=siril,
            image_path=source_path,
            config=cfg,
            log_fn=log_fn,
        )
        return success
    else:
        log_fn(f"Unknown stretch method: {method}")
        return False


def apply_saturation(siril: "SirilInterface", config: Config) -> None:
    """
    Apply saturation to currently loaded image.

    Skips if Vectra is enabled (Vectra handles saturation via LCH).

    Args:
        siril: Siril interface
        config: Configuration with saturation settings
    """
    if config.veralux_vectra_enabled:
        return  # Vectra handles saturation
    siril.satu(config.saturation_amount, config.saturation_background_factor)


def apply_enhancements(
    siril: "SirilInterface",
    image_path: Path,
    config: Config,
    log_fn: Callable[[str], None],
) -> None:
    """
    Apply VeraLux enhancement pipeline to a stretched image.

    Order: Silentium (denoise) -> Revela (detail) -> Vectra (saturation)

    Args:
        siril: Siril interface
        image_path: Path to image file to enhance
        config: Configuration with enhancement settings
        log_fn: Logging function
    """
    from . import veralux_revela, veralux_silentium, veralux_vectra

    cfg = config

    if cfg.veralux_silentium_enabled:
        log_fn("Applying VeraLux Silentium (noise reduction)...")
        success, _stats = veralux_silentium.apply_silentium(
            siril, image_path, cfg, log_fn
        )
        if not success:
            log_fn("Silentium failed, continuing...")

    if cfg.veralux_revela_enabled:
        log_fn("Applying VeraLux Revela (detail enhancement)...")
        success, _stats = veralux_revela.apply_revela(siril, image_path, cfg, log_fn)
        if not success:
            log_fn("Revela failed, continuing...")

    if cfg.veralux_vectra_enabled:
        log_fn("Applying VeraLux Vectra (smart saturation)...")
        success, _stats = veralux_vectra.apply_vectra(siril, image_path, cfg, log_fn)
        if not success:
            log_fn("Vectra failed, continuing...")


def save_all_formats(
    siril: "SirilInterface",
    output_dir: Path,
    basename: str,
    log_fn: Callable[[str], None],
    config: Optional[Config] = None,
) -> dict[str, Path]:
    """
    Save currently loaded image in FIT, TIF, and JPG formats.

    Args:
        siril: Siril interface
        output_dir: Directory to save files
        basename: Base filename (without extension)
        log_fn: Logging function
        config: Optional config for output_suffix

    Returns:
        Dict with 'fit', 'tif', 'jpg' paths
    """
    if config and config.output_suffix:
        basename = f"{basename}_{config.output_suffix}"

    fit_path = output_dir / f"{basename}.fit"
    tif_path = output_dir / f"{basename}.tif"
    jpg_path = output_dir / f"{basename}.jpg"

    siril.save(str(output_dir / basename))
    siril.load(str(fit_path))
    siril.savetif(str(output_dir / basename), astro=True, deflate=True)
    siril.savejpg(str(output_dir / basename), quality=90)

    log_fn(f"Saved: {basename}.fit, {basename}.tif, {basename}.jpg")

    return {"fit": fit_path, "tif": tif_path, "jpg": jpg_path}
