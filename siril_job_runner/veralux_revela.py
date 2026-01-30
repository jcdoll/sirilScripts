"""
VeraLux Revela implementation - Detail enhancement.

Based on VeraLux by Riccardo Paterniti.
https://gitlab.com/free-astro/siril-scripts/-/blob/main/VeraLux/VeraLux_Revela.py

Uses a trous wavelet transform (ATWT) to enhance fine details (texture)
and medium-scale structures while protecting shadows and stars.
"""

from pathlib import Path
from typing import Callable

import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter

from siril_job_runner.config import Config
from siril_job_runner.protocols import SirilInterface
from siril_job_runner.veralux_core import (
    atrous_decomposition,
    atrous_reconstruction,
    lab_to_rgb,
    rgb_to_lab,
)


def _compute_signal_mask(
    L: np.ndarray, shadow_auth: float
) -> np.ndarray:
    """
    Compute signal mask using MAD-based adaptive thresholding.

    Reference formula:
        threshold_sigma = (shadow_auth * 0.12) - 3.0
        noise_floor = median + threshold_sigma * sigma
        signal_mask = clip((L - noise_floor) / (2 * sigma), 0, 1)

    Args:
        L: Luminance channel normalized to [0, 1]
        shadow_auth: Shadow authority slider (0-100)

    Returns:
        Signal mask in [0, 1] where higher = more signal
    """
    # Dynamic threshold based on shadow_auth
    # Maps 0-100 to -3.0 to +9.0
    threshold_sigma = (shadow_auth * 0.12) - 3.0

    median = np.median(L)
    mad = np.median(np.abs(L - median))
    sigma = 1.4826 * mad

    noise_floor = median + threshold_sigma * sigma

    # Signal mask: smooth ramp from noise floor
    signal_mask = np.clip((L - noise_floor) / (2.0 * sigma + 1e-10), 0, 1)

    # Smooth the mask
    signal_mask = gaussian_filter(signal_mask, sigma=1.5)

    return signal_mask.astype(np.float64)


def _compute_star_mask_energy(
    planes: list[np.ndarray], strength: float = 0.8
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute star protection masks using wavelet energy detection.

    Reference formula:
        e_fine = |plane[0]| + |plane[1]|
        e_mid = |plane[2]| + |plane[3]|
        energy = e_fine + 0.5 * e_mid
        threshold = median_energy + 4.0 * sigma_energy
        star_map = clip((energy - threshold) / (2 * sigma_energy), 0, 1)

    Args:
        planes: Wavelet detail planes from ATWT decomposition
        strength: Protection strength (0-1)

    Returns:
        Tuple of (texture_protection, structure_protection) masks
    """
    # Compute energy from fine and mid scales
    e_fine = np.abs(planes[0]) + np.abs(planes[1])
    e_mid = np.abs(planes[2]) + np.abs(planes[3])
    energy = e_fine + 0.5 * e_mid

    # MAD-based threshold for star detection
    med_energy = np.median(energy)
    mad_energy = np.median(np.abs(energy - med_energy))
    sigma_energy = 1.4826 * mad_energy

    threshold = med_energy + 4.0 * sigma_energy

    # Star map with smooth ramp
    star_map = np.clip(
        (energy - threshold) / (2.0 * sigma_energy + 1e-10), 0, 1
    )

    # Smooth and clip for clean edges
    star_map = gaussian_filter(star_map, sigma=2.5)
    star_map = np.clip(star_map * 1.5, 0, 1)
    star_map = gaussian_filter(star_map, sigma=2.5)

    # Protection mask for texture (where 1 = full enhancement allowed)
    star_protection_tex = 1.0 - (star_map * strength)

    # Tighter protection for structure (squared mask)
    star_map_structure = np.clip(star_map ** 2, 0, 1)
    star_protection_str = 1.0 - (star_map_structure * strength)

    return star_protection_tex.astype(np.float64), star_protection_str.astype(np.float64)


def enhance_details(
    data: np.ndarray,
    texture: float = 50.0,
    structure: float = 50.0,
    shadow_auth: float = 25.0,
    protect_stars: bool = True,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Enhance image details using ATWT wavelet decomposition.

    Reference gain formulas:
        t_gain = 1.0 + (texture/100 * 1.5)
        s_gain = 1.0 + (structure/100 * 1.0)

    Args:
        data: RGB image data, shape (3, H, W), values in [0, 1]
        texture: Fine detail enhancement (0-100), affects scales 0-1
        structure: Medium structure enhancement (0-100), affects scales 2-4
        shadow_auth: Shadow protection strength (0-100)
        protect_stars: If True, reduce enhancement in star regions

    Returns:
        Tuple of (enhanced_data, stats_dict)
    """
    # Convert to Lab color space
    lab = rgb_to_lab(data)
    L = lab[0]  # L channel is 0-100 range

    # Normalize L for mask computation
    L_norm = L / 100.0

    # Wavelet decomposition (6 scales)
    planes, residual = atrous_decomposition(L, n_scales=6)

    # Compute signal mask from normalized luminance
    signal_mask = _compute_signal_mask(L_norm, shadow_auth)

    # Compute star protection masks from wavelet planes
    if protect_stars:
        star_prot_tex, star_prot_str = _compute_star_mask_energy(planes, strength=0.8)
    else:
        star_prot_tex = np.ones_like(L)
        star_prot_str = np.ones_like(L)

    # Active masks: combine signal and star protection
    active_mask_tex = signal_mask * star_prot_tex
    active_mask_str = signal_mask * star_prot_str

    # Compute gains (reference formulas)
    texture_amt = texture / 100.0
    structure_amt = structure / 100.0

    t_gain = 1.0 + (texture_amt * 1.5)
    s_gain = 1.0 + (structure_amt * 1.0)

    # Apply texture enhancement to fine scales (0, 1)
    texture_scales = [0, 1]
    for i in texture_scales:
        boost = 1.0 + (t_gain - 1.0) * active_mask_tex
        planes[i] = planes[i] * boost

    # Apply structure enhancement to medium scales (2, 3, 4)
    structure_scales = [2, 3, 4]
    for i in structure_scales:
        boost = 1.0 + (s_gain - 1.0) * active_mask_str
        planes[i] = planes[i] * boost

    # Reconstruct enhanced luminance
    L_enhanced = atrous_reconstruction(planes, residual)
    L_enhanced = np.clip(L_enhanced, 0, 100)

    # Recombine with original chrominance
    lab_enhanced = np.stack([L_enhanced, lab[1], lab[2]], axis=0)
    rgb_enhanced = lab_to_rgb(lab_enhanced)

    stats = {
        "texture_gain": t_gain,
        "structure_gain": s_gain,
        "signal_coverage": float(np.mean(signal_mask > 0.5)),
        "star_coverage": float(np.mean(star_prot_tex < 0.5)) if protect_stars else 0.0,
    }

    return rgb_enhanced, stats


def apply_revela(
    siril: SirilInterface,
    image_path: Path,
    config: Config,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[bool, dict[str, float]]:
    """
    Apply VeraLux Revela detail enhancement to an image.

    Loads the image, applies ATWT-based enhancement, and saves back.

    Args:
        siril: SirilWrapper instance (used for loading context)
        image_path: Path to the image to enhance
        config: Configuration with revela parameters
        log_fn: Optional logging function

    Returns:
        Tuple of (success, stats_dict)
    """

    def log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    log(
        f"Revela: texture={config.veralux_revela_texture}, "
        f"structure={config.veralux_revela_structure}, "
        f"shadow_auth={config.veralux_revela_shadow_auth}"
    )

    with fits.open(image_path) as hdul:
        data = hdul[0].data.astype(np.float64)
        header = hdul[0].header.copy()

    if data.max() > 1.5:
        data = data / 65535.0

    if data.ndim != 3 or data.shape[0] != 3:
        log("Revela: Image must be RGB (3, H, W)")
        return False, {}

    enhanced, stats = enhance_details(
        data,
        texture=config.veralux_revela_texture,
        structure=config.veralux_revela_structure,
        shadow_auth=config.veralux_revela_shadow_auth,
        protect_stars=config.veralux_revela_protect_stars,
    )

    out_data = np.clip(enhanced * 65535, 0, 65535).astype(np.uint16)
    hdu = fits.PrimaryHDU(out_data, header=header)
    hdu.writeto(image_path, overwrite=True)

    log(
        f"Revela applied: texture_gain={stats['texture_gain']:.2f}, "
        f"structure_gain={stats['structure_gain']:.2f}"
    )

    if not siril.load(str(image_path)):
        return False, stats

    return True, stats
