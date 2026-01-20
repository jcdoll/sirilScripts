"""
Shared helper functions for composition modules.

Color removal and other utilities used by broadband and narrowband composition.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .config import Config
from .psf_analysis import analyze_psf, format_psf_stats

if TYPE_CHECKING:
    from .protocols import SirilInterface


def apply_color_removal_step(
    siril: "SirilInterface",
    config: Config,
    stretch_source: str,
    log_fn: callable,
) -> str:
    """Apply color cast removal and return the (possibly updated) stretch source."""
    if config.color_removal_mode == "none":
        return stretch_source

    siril.load(stretch_source)
    success = apply_color_removal(siril, config, log_fn)
    if success:
        siril.save(stretch_source)
    else:
        log_fn("Color removal failed, continuing without")
    return stretch_source


def apply_color_removal(
    siril: "SirilInterface",
    config: Config,
    log_fn: callable,
) -> bool:
    """Apply color cast removal based on config mode."""
    if config.color_removal_mode == "none":
        return True

    if config.color_removal_mode == "green":
        log_fn("Removing green cast (SCNR)...")
        return siril.rmgreen(
            type=config.rmgreen_type,
            amount=config.rmgreen_amount,
            preserve_lightness=config.rmgreen_preserve_lightness,
        )
    elif config.color_removal_mode == "magenta":
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
    log_fn: callable,
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


def apply_rgb_deconvolution(
    siril: "SirilInterface",
    config: Config,
    output_dir: Path,
    stretch_source: str,
    type_name: str,
    log_fn: callable,
) -> str:
    """Apply deconvolution to RGB composite and return updated stretch_source."""
    if not config.deconv_enabled:
        return stretch_source

    log_fn("Deconvolving RGB composite...")
    siril.load(stretch_source)
    psf_path = (
        str(output_dir / f"psf_{type_name}.fit") if config.deconv_save_psf else None
    )
    if siril.makepsf(
        method=config.deconv_psf_method,
        symmetric=True,
        save_psf=psf_path,
    ):
        if psf_path:
            log_fn(f"PSF saved: psf_{type_name}.fit")
            psf_stats = analyze_psf(Path(psf_path))
            if psf_stats:
                for line in format_psf_stats(psf_stats):
                    log_fn(f"  {line}")
        if siril.rl(
            iters=config.deconv_iterations,
            regularization=config.deconv_regularization,
            alpha=config.deconv_alpha,
        ):
            deconv_path = output_dir / f"{type_name}_deconv.fit"
            siril.save(str(deconv_path))
            log_fn(f"Saved deconvolved: {deconv_path.name}")
            return str(deconv_path)
        else:
            log_fn("RGB deconvolution failed, using original")
    else:
        log_fn("RGB PSF creation failed, skipping deconvolution")

    return stretch_source
