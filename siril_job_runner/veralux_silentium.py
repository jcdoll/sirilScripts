"""
VeraLux Silentium implementation - Noise suppression.

Based on VeraLux by Riccardo Paterniti.
https://gitlab.com/free-astro/siril-scripts/-/blob/main/VeraLux/VeraLux_Silentium.py

Uses Stationary Wavelet Transform (SWT) with soft thresholding for noise reduction.
Includes photometric gating with signal probability and Sobel edge detection.
"""

from pathlib import Path
from typing import Callable

import numpy as np
import pywt
from astropy.io import fits
from scipy.ndimage import maximum_filter
from scipy.signal import convolve2d

from siril_job_runner.config import Config
from siril_job_runner.protocols import SirilInterface
from siril_job_runner.veralux_core import lab_to_rgb, rgb_to_lab


def _soft_threshold(coeffs: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    """
    Apply soft thresholding to wavelet coefficients.

    Args:
        coeffs: Wavelet coefficients
        threshold: Threshold array (same shape as coeffs)

    Returns:
        Thresholded coefficients
    """
    return np.sign(coeffs) * np.maximum(np.abs(coeffs) - threshold, 0)


def _compute_signal_probability(channel: np.ndarray) -> np.ndarray:
    """
    Compute signal probability map using MAD statistics.

    Linear ramp from noise (0.0) to safe signal (1.0) across 2.5-sigma range.

    Args:
        channel: 2D image array

    Returns:
        Signal probability map in [0, 1]
    """
    med = float(np.median(channel))
    mad = float(np.median(np.abs(channel - med)))
    sigma = 1.4826 * mad if mad > 0 else 1e-6

    low_thr = med + (1.0 * sigma)
    high_thr = med + (3.5 * sigma)
    diff = high_thr - low_thr
    if diff < 1e-9:
        diff = 1e-9

    signal_map = (channel - low_thr) / diff
    signal_map = np.clip(signal_map, 0.0, 1.0)
    return signal_map.astype(np.float32)


def _auto_stretch_proxy(channel: np.ndarray) -> np.ndarray:
    """
    Simple auto-stretch for edge detection preprocessing.

    Args:
        channel: 2D image array

    Returns:
        Stretched array in [0, 1]
    """
    med = np.median(channel)
    mad = np.median(np.abs(channel - med))
    sigma = 1.4826 * mad if mad > 0 else 1e-6

    low = med - 2.5 * sigma
    high = med + 5.0 * sigma
    stretched = (channel - low) / (high - low + 1e-9)
    return np.clip(stretched, 0.0, 1.0).astype(np.float32)


def _compute_edge_map(L: np.ndarray) -> np.ndarray:
    """
    Compute edge map using Sobel gradients on auto-stretched luminance.

    Args:
        L: Luminance channel (2D array)

    Returns:
        Edge map in [0, 1]
    """
    L_str = _auto_stretch_proxy(L)

    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    gx = convolve2d(L_str, kx, mode="same", boundary="symm")
    gy = convolve2d(L_str, ky, mode="same", boundary="symm")
    mag = np.sqrt(gx * gx + gy * gy)

    mx = np.percentile(mag, 98)
    if mx <= 1e-9:
        mx = 1e-9
    mag = mag / mx
    mag = np.clip(mag, 0.0, 1.0)

    # Morphological dilation to protect edge neighborhoods
    mag = maximum_filter(mag, size=2)
    return mag.astype(np.float32)


def _estimate_noise_map(
    channel: np.ndarray, block_size: int = 64
) -> tuple[np.ndarray, float]:
    """
    Estimate local noise using block-based MAD on background pixels.

    Args:
        channel: 2D image array
        block_size: Size of blocks for local estimation

    Returns:
        Tuple of (noise_map, global_sigma)
    """
    h, w = channel.shape
    sigma_map = np.zeros_like(channel, dtype=np.float32)

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            y2 = min(y + block_size, h)
            x2 = min(x + block_size, w)
            patch = channel[y:y2, x:x2]

            # Use only background pixels (below median)
            q = np.quantile(patch, 0.5)
            bg = patch[patch <= q]
            if bg.size < 16:
                bg = patch

            med = np.median(bg)
            mad = np.median(np.abs(bg - med))
            sigma = 1.4826 * mad if mad > 0 else np.std(bg)
            sigma_map[y:y2, x:x2] = sigma

    # Fill any zero values with median sigma
    mask_zero = sigma_map <= 0
    if np.any(mask_zero):
        median_sigma = np.median(sigma_map[~mask_zero]) if np.any(~mask_zero) else 1e-6
        sigma_map[mask_zero] = median_sigma

    global_sigma = float(np.median(sigma_map))
    return sigma_map, global_sigma


def _pad_for_swt(
    arr: np.ndarray, max_levels: int
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Pad array to be compatible with SWT at given levels.

    Args:
        arr: 2D array to pad
        max_levels: Number of SWT levels

    Returns:
        Tuple of (padded_array, (pad_h_before, pad_h_after, pad_w_before, pad_w_after))
    """
    h, w = arr.shape
    factor = 2**max_levels

    new_h = int(np.ceil(h / factor) * factor)
    new_w = int(np.ceil(w / factor) * factor)

    pad_h = new_h - h
    pad_w = new_w - w

    if pad_h > 0 or pad_w > 0:
        padded = np.pad(arr, ((0, pad_h), (0, pad_w)), mode="reflect")
    else:
        padded = arr

    return padded, (0, pad_h, 0, pad_w)


def _multiscale_denoise(
    channel: np.ndarray,
    sigma_map: np.ndarray,
    edge_map: np.ndarray,
    signal_map: np.ndarray,
    intensity: float,
    detail_guard: float,
    shadow_smooth: float = 0.0,
    is_chroma: bool = False,
) -> np.ndarray:
    """
    Perform SWT-based denoising with photometric gating.

    Args:
        channel: 2D image channel to denoise
        sigma_map: Local noise estimate map
        edge_map: Edge detection map
        signal_map: Signal probability map
        intensity: Normalized intensity (0-2 range, from slider/50)
        detail_guard: Normalized detail guard (0-1 range, from slider/100)
        shadow_smooth: Normalized shadow smoothness (0-1 range, from slider/100)
        is_chroma: Whether this is a chroma channel

    Returns:
        Denoised channel
    """
    max_levels = 4
    wavelet = "db2"
    layer_weights = [0.60, 0.80, 1.0, 1.0]

    original_shape = channel.shape
    padded_channel, _ = _pad_for_swt(channel, max_levels)
    pad_sigma, _ = _pad_for_swt(sigma_map, max_levels)
    pad_edge, _ = _pad_for_swt(edge_map, max_levels)
    pad_signal, _ = _pad_for_swt(signal_map, max_levels)

    # SWT decomposition
    coeffs = pywt.swt2(padded_channel, wavelet=wavelet, level=max_levels)

    # Base threshold multiplier: 4.5 * intensity
    base_sigma_mult = 4.5 * intensity

    new_coeffs = []

    for i, coeff_tuple in enumerate(coeffs):
        # Reference structure: (cA, (cH, cV, cD))
        cA = coeff_tuple[0]
        cH, cV, cD = coeff_tuple[1]

        # Layer weight for frequency damping
        w_layer = layer_weights[i] if i < len(layer_weights) else 1.0

        # Level index (max_levels at coarsest, 1 at finest)
        level_idx = max_levels - i
        scale = 2**level_idx

        # Chroma degradation factor
        lvl_degrade = 1.0 / (2 ** (4 - level_idx)) if is_chroma else 1.0

        # Detail guard map: robust_edge = edge * signal_probability
        # K=40 gives strong protection
        robust_edge = pad_edge * pad_signal
        K = 40.0
        guard_map = 1.0 + (K * detail_guard * robust_edge)

        # Shadow smoothness boost: only active where signal is low
        # Gate closes completely when signal > 25% (signal*4 >= 1)
        shadow_gate_active = np.clip(pad_signal * 4.0, 0.0, 1.0)
        inv_signal = 1.0 - shadow_gate_active
        boost_map = 1.0
        if shadow_smooth > 0.01:
            boost_map = 1.0 + (3.0 * shadow_smooth * inv_signal)

        # Compute threshold
        # Start with sigma * base multiplier
        thr = pad_sigma * base_sigma_mult
        # Divide by sqrt(scale) - finer scales get LOWER threshold
        thr = thr / (scale**0.5)
        # Apply chroma degradation
        thr = thr * lvl_degrade
        # Apply layer weight
        thr = thr * w_layer
        # Divide by guard map - edges get much lower threshold (more protection)
        thr = thr / guard_map
        # Multiply by shadow boost - backgrounds get higher threshold (more smoothing)
        thr = thr * boost_map

        # Apply soft thresholding to detail coefficients
        cH_dn = _soft_threshold(cH, thr)
        cV_dn = _soft_threshold(cV, thr)
        cD_dn = _soft_threshold(cD, thr)

        new_coeffs.append((cA, (cH_dn, cV_dn, cD_dn)))

    # Inverse SWT
    channel_dn = pywt.iswt2(new_coeffs, wavelet=wavelet)

    # Remove padding
    channel_dn = channel_dn[: original_shape[0], : original_shape[1]]

    return channel_dn.astype(np.float32)


def denoise_channel(
    channel: np.ndarray,
    intensity: float = 25.0,
    detail_guard: float = 50.0,
    shadow_smooth: float = 10.0,
    n_levels: int = 4,
) -> np.ndarray:
    """
    Denoise a single channel using SWT with soft thresholding.

    This is a simplified interface for single-channel denoising.

    Args:
        channel: 2D image channel
        intensity: Noise reduction intensity (0-100 slider units)
        detail_guard: Detail protection strength (0-100 slider units)
        shadow_smooth: Extra smoothing for shadows (0-100 slider units)
        n_levels: Number of SWT decomposition levels (ignored, always 4)

    Returns:
        Denoised channel
    """
    # Compute maps
    sigma_map, _ = _estimate_noise_map(channel)
    edge_map = _compute_edge_map(channel)
    signal_map = _compute_signal_probability(channel)

    # Convert slider units to normalized values
    intensity_norm = intensity / 50.0  # 0-100 -> 0-2
    guard_norm = detail_guard / 100.0  # 0-100 -> 0-1
    shadow_norm = shadow_smooth / 100.0  # 0-100 -> 0-1

    return _multiscale_denoise(
        channel=channel,
        sigma_map=sigma_map,
        edge_map=edge_map,
        signal_map=signal_map,
        intensity=intensity_norm,
        detail_guard=guard_norm,
        shadow_smooth=shadow_norm,
        is_chroma=False,
    )


def denoise_image(
    data: np.ndarray,
    intensity: float = 25.0,
    detail_guard: float = 50.0,
    chroma: float = 30.0,
    shadow_smooth: float = 10.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Denoise RGB image with separate luminance and chroma processing.

    Args:
        data: RGB image data, shape (3, H, W), values in [0, 1]
        intensity: Luminance noise reduction (0-100 slider units)
        detail_guard: Detail protection strength (0-100 slider units)
        chroma: Chroma noise reduction (0-100 slider units)
        shadow_smooth: Extra shadow smoothing (0-100 slider units)

    Returns:
        Tuple of (denoised_data, stats_dict)
    """
    # Convert to LAB
    lab = rgb_to_lab(data)
    L, a, b = lab[0], lab[1], lab[2]

    # Compute maps from luminance
    sigma_map_L, L_noise = _estimate_noise_map(L)
    edge_map = _compute_edge_map(L)
    signal_map = _compute_signal_probability(L)

    # Estimate chroma noise
    _, a_noise = _estimate_noise_map(a)
    _, b_noise = _estimate_noise_map(b)

    # Convert slider units to normalized values
    intensity_norm = intensity / 50.0
    guard_norm = detail_guard / 100.0
    shadow_norm = shadow_smooth / 100.0
    chroma_intensity_norm = chroma / 50.0

    # Denoise luminance
    L_denoised = _multiscale_denoise(
        channel=L,
        sigma_map=sigma_map_L,
        edge_map=edge_map,
        signal_map=signal_map,
        intensity=intensity_norm,
        detail_guard=guard_norm,
        shadow_smooth=shadow_norm,
        is_chroma=False,
    )

    # Compute chroma noise maps
    sigma_map_a, _ = _estimate_noise_map(a)
    sigma_map_b, _ = _estimate_noise_map(b)

    # Denoise chroma channels with reduced guard (edges less critical in color)
    a_denoised = _multiscale_denoise(
        channel=a,
        sigma_map=sigma_map_a,
        edge_map=edge_map,
        signal_map=signal_map,
        intensity=chroma_intensity_norm,
        detail_guard=guard_norm * 0.5,
        shadow_smooth=0.0,
        is_chroma=True,
    )

    b_denoised = _multiscale_denoise(
        channel=b,
        sigma_map=sigma_map_b,
        edge_map=edge_map,
        signal_map=signal_map,
        intensity=chroma_intensity_norm,
        detail_guard=guard_norm * 0.5,
        shadow_smooth=0.0,
        is_chroma=True,
    )

    # Clip luminance to valid range
    L_denoised = np.clip(L_denoised, 0, 100)

    # Reconstruct LAB and convert back to RGB
    lab_denoised = np.stack([L_denoised, a_denoised, b_denoised], axis=0)
    rgb_denoised = lab_to_rgb(lab_denoised)

    stats = {
        "L_noise_estimate": L_noise,
        "a_noise_estimate": a_noise,
        "b_noise_estimate": b_noise,
        "intensity": intensity,
        "chroma_intensity": chroma,
    }

    return rgb_denoised, stats


def apply_silentium(
    siril: SirilInterface,
    image_path: Path,
    config: Config,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[bool, dict[str, float]]:
    """
    Apply VeraLux Silentium noise reduction to an image.

    Loads the image, applies SWT-based denoising, and saves back.

    Args:
        siril: SirilWrapper instance (used for loading context)
        image_path: Path to the image to denoise
        config: Configuration with silentium parameters
        log_fn: Optional logging function

    Returns:
        Tuple of (success, stats_dict)
    """

    def log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    log(
        f"Silentium: intensity={config.veralux_silentium_intensity}, "
        f"detail_guard={config.veralux_silentium_detail_guard}, "
        f"chroma={config.veralux_silentium_chroma}"
    )

    with fits.open(image_path) as hdul:
        data = hdul[0].data.astype(np.float64)
        header = hdul[0].header.copy()

    if data.max() > 1.5:
        data = data / 65535.0

    if data.ndim != 3 or data.shape[0] != 3:
        log("Silentium: Image must be RGB (3, H, W)")
        return False, {}

    denoised, stats = denoise_image(
        data,
        intensity=config.veralux_silentium_intensity,
        detail_guard=config.veralux_silentium_detail_guard,
        chroma=config.veralux_silentium_chroma,
        shadow_smooth=config.veralux_silentium_shadow_smooth,
    )

    out_data = np.clip(denoised * 65535, 0, 65535).astype(np.uint16)
    hdu = fits.PrimaryHDU(out_data, header=header)
    hdu.writeto(image_path, overwrite=True)

    log(
        f"Silentium applied: L_noise={stats['L_noise_estimate']:.4f}, "
        f"chroma_noise={stats['a_noise_estimate']:.4f}"
    )

    if not siril.load(str(image_path)):
        return False, stats

    return True, stats
