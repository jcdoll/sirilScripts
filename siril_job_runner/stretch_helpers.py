"""
Stretch and finalization helpers for image composition.

Clean standalone functions for stretching, saturation, enhancements, and saving.
Used by both broadband and narrowband composition modules.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from .config import Config

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

    if method == "autostretch":
        mode = "linked" if cfg.autostretch_linked else "unlinked"
        log_fn(f"Stretching ({method}, {mode}, targetbg={cfg.autostretch_targetbg})...")
        success = siril.autostretch(
            linked=cfg.autostretch_linked,
            shadowclip=cfg.autostretch_shadowclip,
            targetbg=cfg.autostretch_targetbg,
        )
        if success:
            log_fn(
                f"  MTF: low={cfg.autostretch_mtf_low}, "
                f"mid={cfg.autostretch_mtf_mid}, high={cfg.autostretch_mtf_high}"
            )
            siril.mtf(
                cfg.autostretch_mtf_low,
                cfg.autostretch_mtf_mid,
                cfg.autostretch_mtf_high,
            )
        return success
    elif method == "veralux":
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


def finalize_image(
    siril: "SirilInterface",
    output_dir: Path,
    basename: str,
    config: Config,
    log_fn: Callable[[str], None],
    working_dir: Optional[Path] = None,
) -> dict[str, Path]:
    """
    Finalize image: apply saturation, enhancements, and save all formats.

    Args:
        siril: Siril interface
        output_dir: Directory for final output files
        basename: Base filename (without extension)
        config: Configuration
        log_fn: Logging function
        working_dir: Working directory for temp files (defaults to output_dir)

    Returns:
        Dict with 'fit', 'tif', 'jpg' paths
    """
    if working_dir is None:
        working_dir = output_dir

    # Apply saturation (skipped if Vectra enabled)
    apply_saturation(siril, config)

    # Save initial file for enhancements
    fit_path = output_dir / f"{basename}.fit"
    siril.save(str(output_dir / basename))

    # Apply enhancements if any are enabled
    has_enhancements = (
        config.veralux_silentium_enabled
        or config.veralux_revela_enabled
        or config.veralux_vectra_enabled
    )
    if has_enhancements:
        apply_enhancements(siril, fit_path, config, log_fn)
        siril.save(str(output_dir / basename))

    # Save all formats
    return save_all_formats(siril, output_dir, basename, log_fn, config)


def stretch_and_finalize(
    siril: "SirilInterface",
    input_path: Path,
    output_dir: Path,
    basename: str,
    config: Config,
    log_fn: Callable[[str], None],
    working_dir: Optional[Path] = None,
) -> dict[str, Path]:
    """
    Apply stretch (with compare mode support) and finalize image.

    If config.stretch_compare is True, runs both autostretch and veralux,
    saving each with appropriate suffix. Otherwise runs config.stretch_method only.

    Args:
        siril: Siril interface
        input_path: Path to linear input image
        output_dir: Directory for final output files
        basename: Base filename (without extension)
        config: Configuration
        log_fn: Logging function
        working_dir: Working directory for temp files (defaults to output_dir)

    Returns:
        Dict with 'fit', 'tif', 'jpg' paths for primary output
    """
    if working_dir is None:
        working_dir = output_dir

    cfg = config
    primary_paths = None

    if cfg.stretch_compare:
        methods = ["autostretch", "veralux"]
        log_fn("Comparing stretch methods (autostretch vs veralux)...")
    else:
        methods = [cfg.stretch_method]

    for method in methods:
        # Load fresh linear data for each method
        siril.load(str(input_path))

        # Apply stretch
        success = apply_stretch(siril, method, input_path, cfg, log_fn)
        if not success:
            log_fn(f"Stretch failed ({method}), falling back to autostretch")
            siril.load(str(input_path))
            siril.autostretch()

        # Determine output name
        output_name = f"{basename}_{method}" if cfg.stretch_compare else basename

        # Finalize (saturation + enhancements + save)
        paths = finalize_image(siril, output_dir, output_name, cfg, log_fn, working_dir)

        if primary_paths is None:
            primary_paths = paths

    return primary_paths
