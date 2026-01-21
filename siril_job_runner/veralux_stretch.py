"""
VeraLux HyperMetric Stretch implementation.

Based on VeraLux by Riccardo Paterniti.
https://gitlab.com/free-astro/siril-scripts/-/blob/main/VeraLux/VeraLux_HyperMetric_Stretch.py

This module implements the core stretch algorithm matching the reference exactly.
"""

import math
from pathlib import Path
from typing import Callable

from siril_job_runner.config import Config
from siril_job_runner.siril_wrapper import SirilWrapper

# Reference defaults from VeraLux
DEFAULT_CONVERGENCE_POWER = 3.5
DEFAULT_PROTECT_B = 6.0
DEFAULT_TARGET_BG = 0.20
REC709_WEIGHTS = (0.2126, 0.7152, 0.0722)
RTU_PEDESTAL = 0.001
RTU_SOFT_CEIL_PERCENTILE = 99.0


def hyperbolic_stretch_value(
    value: float, D: float, b: float, SP: float = 0.0
) -> float:
    """
    Apply hyperbolic stretch to a single value.

    Formula: (asinh(D*(value-SP)+b) - asinh(b)) / (asinh(D*(1-SP)+b) - asinh(b))
    """
    D = max(D, 0.1)
    b = max(b, 0.1)

    term1 = math.asinh(D * (value - SP) + b)
    term2 = math.asinh(b)
    norm_factor = math.asinh(D * (1.0 - SP) + b) - term2

    if norm_factor == 0:
        norm_factor = 1e-6

    return (term1 - term2) / norm_factor


def solve_log_d(
    median_in: float,
    target_median: float,
    b: float,
    log_d_min: float = 0.0,
    log_d_max: float = 7.0,
    max_iterations: int = 40,
    tolerance: float = 0.0001,
) -> float:
    """Binary search to find log_d that achieves target median."""
    if median_in < 1e-9:
        return 2.0

    low_log = log_d_min
    high_log = log_d_max
    best_log_d = 2.0

    for _ in range(max_iterations):
        mid_log = (low_log + high_log) / 2.0
        mid_D = 10.0**mid_log

        test_val = hyperbolic_stretch_value(median_in, mid_D, b)

        if abs(test_val - target_median) < tolerance:
            best_log_d = mid_log
            break

        if test_val < target_median:
            low_log = mid_log
        else:
            high_log = mid_log

        best_log_d = mid_log

    return best_log_d


def build_pixelmath_formula(D: float, b: float) -> str:
    """Build Siril PixelMath formula for hyperbolic stretch."""
    norm = math.asinh(D + b) - math.asinh(b)
    asinh_b = math.asinh(b)
    return f"(asinh({D}*$T+{b})-{asinh_b})/{norm}"


def _normalize_input(data):
    """
    Normalize input to 0-1 range based on dtype.

    Reference implementation: detects dtype and divides by max representable value.
    """
    import numpy as np

    data = np.nan_to_num(data, nan=0.0, posinf=None, neginf=0.0)
    input_dtype = data.dtype
    img_float = data.astype(np.float64)

    if np.issubdtype(input_dtype, np.integer):
        if input_dtype == np.uint8:
            return img_float / 255.0
        elif input_dtype == np.uint16:
            return img_float / 65535.0
        elif input_dtype == np.int16:
            return img_float / 32767.0
        else:
            return img_float / 4294967295.0

    elif np.issubdtype(input_dtype, np.floating):
        current_max = float(np.max(data))
        if current_max <= 1.1:
            return img_float
        if current_max < 100000.0:
            return img_float / 65535.0
        return img_float / 4294967295.0

    return img_float


def _calculate_anchor(data, percentile: float = 0.5, offset: float = 0.00025):
    """
    Calculate anchor (black point) using percentile method.

    Reference: For RGB, compute 0.5th percentile on EACH channel,
    take minimum of those floors, subtract 0.00025 offset.
    """
    import numpy as np

    if data.ndim == 3:
        floors = [np.percentile(data[i], percentile) for i in range(data.shape[0])]
        anchor = min(floors) - offset
    else:
        anchor = np.percentile(data, percentile) - offset

    return max(anchor, 0.0)


def _hyperbolic_stretch_array(data, D: float, b: float):
    """Apply hyperbolic stretch to a numpy array."""
    import numpy as np

    term1 = np.arcsinh(D * data + b)
    term2 = np.arcsinh(b)
    norm_factor = np.arcsinh(D + b) - term2
    return (term1 - term2) / norm_factor


def _apply_mtf(data, m):
    """
    Apply MTF (Midtone Transfer Function).

    Reference formula: y = (m - 1) * x / ((2*m - 1) * x - m)
    """
    import numpy as np

    term1 = (m - 1.0) * data
    term2 = (2.0 * m - 1.0) * data - m
    with np.errstate(divide="ignore", invalid="ignore"):
        res = term1 / term2
    return np.nan_to_num(res, nan=0.0, posinf=1.0, neginf=0.0)


def _soft_clip_channel(c, thresh: float = 0.98, roll: float = 2.0):
    """
    Apply soft-clip to values above threshold.

    Reference formula: y = thresh + (1 - thresh) * (1 - (1 - t)^rolloff)
    where t = (c - thresh) / (1 - thresh)
    """
    import numpy as np

    mask = c > thresh
    result = c.copy()
    if np.any(mask):
        t = np.clip((c[mask] - thresh) / (1.0 - thresh + 1e-9), 0.0, 1.0)
        result[mask] = thresh + (1.0 - thresh) * (1.0 - np.power(1.0 - t, roll))
    return np.clip(result, 0.0, 1.0)


def _check_valid_physical_max(L_raw, abs_max):
    """
    Smart Max: Check if max pixel has similar neighbors (real star) or is isolated (hot pixel).

    Reference: If max neighbor < 20% of abs_max, it's likely a hot pixel.
    """
    import numpy as np

    if abs_max <= 0.001:
        return True

    idx_max = np.argmax(L_raw)
    y_max, x_max = np.unravel_index(idx_max, L_raw.shape)
    y0, y1 = max(0, y_max - 1), min(L_raw.shape[0], y_max + 2)
    x0, x1 = max(0, x_max - 1), min(L_raw.shape[1], x_max + 2)
    window = L_raw[y0:y1, x0:x1]
    neighbors = window[window < abs_max]

    if neighbors.size > 0:
        max_neighbor = np.max(neighbors)
        if max_neighbor < (abs_max * 0.20):
            return False

    return True


def _adaptive_output_scaling(data, target_bg: float = 0.20):
    """
    Adaptive output scaling from reference.

    Reference steps:
    1. Extract luminance
    2. Find global floor: max(min_L, median_L - 2.7*std_L)
    3. Smart Max: check if max is hot pixel
    4. Compute soft_ceil as max of per-channel 99.9th percentiles (RGB)
    5. Compute scale factors
    6. Apply scaling with pedestal
    7. Apply MTF to reach target background
    """
    import numpy as np

    r_weight, g_weight, b_weight = REC709_WEIGHTS
    is_rgb = data.ndim == 3 and data.shape[0] == 3

    # Extract luminance
    if is_rgb:
        R, G, B = data[0], data[1], data[2]
        L_raw = r_weight * R + g_weight * G + b_weight * B
    else:
        L_raw = data

    # Find global floor
    median_L = float(np.median(L_raw))
    std_L = float(np.std(L_raw))
    min_L = float(np.min(L_raw))
    global_floor = max(min_L, median_L - 2.7 * std_L)

    # Smart Max: check if physical max is valid (not hot pixel)
    abs_max = float(np.max(L_raw))
    valid_physical_max = _check_valid_physical_max(L_raw, abs_max)

    # Compute soft ceiling (max of per-channel percentiles for RGB)
    if is_rgb:
        soft_ceil = max(
            np.percentile(R, RTU_SOFT_CEIL_PERCENTILE),
            np.percentile(G, RTU_SOFT_CEIL_PERCENTILE),
            np.percentile(B, RTU_SOFT_CEIL_PERCENTILE),
        )
    else:
        soft_ceil = np.percentile(L_raw, RTU_SOFT_CEIL_PERCENTILE)

    # Ensure valid ranges
    if soft_ceil <= global_floor:
        soft_ceil = global_floor + 1e-6
    if abs_max <= soft_ceil:
        abs_max = soft_ceil + 1e-6

    # Compute scale factors
    scale_contrast = (0.98 - RTU_PEDESTAL) / (soft_ceil - global_floor + 1e-9)

    if valid_physical_max:
        scale_physical_limit = (1.0 - RTU_PEDESTAL) / (abs_max - global_floor + 1e-9)
        final_scale = min(scale_contrast, scale_physical_limit)
    else:
        final_scale = scale_contrast

    # Apply scaling with pedestal
    def expand_channel(c):
        return np.clip((c - global_floor) * final_scale + RTU_PEDESTAL, 0.0, 1.0)

    if is_rgb:
        result = np.zeros_like(data)
        result[0] = expand_channel(R)
        result[1] = expand_channel(G)
        result[2] = expand_channel(B)
        L = r_weight * result[0] + g_weight * result[1] + b_weight * result[2]
    else:
        result = expand_channel(L_raw)
        L = result

    # Apply MTF to reach target background
    current_bg = float(np.median(L))
    if current_bg > 0.0 and current_bg < 1.0 and abs(current_bg - target_bg) > 1e-3:
        # Solve for MTF parameter m
        m = (current_bg * (target_bg - 1.0)) / (
            current_bg * (2.0 * target_bg - 1.0) - target_bg
        )
        if is_rgb:
            for i in range(3):
                result[i] = _apply_mtf(result[i], m)
        else:
            result = _apply_mtf(result, m)

    return result


def apply_stretch(
    siril: SirilWrapper,
    image_path: Path,
    config: Config,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[bool, float]:
    """
    Apply VeraLux HyperMetric stretch matching reference implementation.

    For RGB images:
    1. Normalize input based on dtype
    2. Calculate anchor (black point)
    3. Subtract anchor from data
    4. Extract luminance from anchored data
    5. Compute color ratios from anchored data
    6. Stretch luminance
    7. Apply color convergence (power=3.5)
    8. Reconstruct RGB
    9. Apply adaptive output scaling (with MTF)
    10. Apply soft-clip

    For mono images: applies stretch directly.
    """
    import numpy as np
    from astropy.io import fits

    def log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    # Load image with astropy
    with fits.open(image_path) as hdul:
        raw_data = hdul[0].data
        header = hdul[0].header.copy()

    # Normalize to 0-1 based on dtype (reference implementation)
    data = _normalize_input(raw_data)

    # Check if RGB (3, H, W) or mono (H, W)
    is_rgb = data.ndim == 3 and data.shape[0] == 3

    # Get parameters from config or use reference defaults
    protect_b = getattr(config, "veralux_b", DEFAULT_PROTECT_B)
    target_bg = getattr(config, "veralux_target_median", DEFAULT_TARGET_BG)
    convergence_power = getattr(
        config, "veralux_convergence_power", DEFAULT_CONVERGENCE_POWER
    )

    if is_rgb:
        log("RGB image detected - using vector preservation")
        r_weight, g_weight, b_weight = REC709_WEIGHTS

        # Step 1: Calculate anchor (black point) from RGB channels
        anchor = _calculate_anchor(data)
        log(f"Anchor (black point): {anchor:.6f}")

        # Step 2: Subtract anchor from all channels
        img_anchored = np.maximum(data - anchor, 0.0)

        # Step 3: Extract luminance from ANCHORED data
        luminance = (
            r_weight * img_anchored[0]
            + g_weight * img_anchored[1]
            + b_weight * img_anchored[2]
        )

        # Step 4: Compute color ratios from ANCHORED data
        lum_safe = np.maximum(luminance, 1e-10)
        r_ratio = img_anchored[0] / lum_safe
        g_ratio = img_anchored[1] / lum_safe
        b_ratio = img_anchored[2] / lum_safe

        # Calculate stretch parameters from anchored luminance
        median_in = float(np.median(luminance))
    else:
        log("Mono image detected - direct stretch")
        anchor = _calculate_anchor(data)
        log(f"Anchor (black point): {anchor:.6f}")
        data_anchored = np.maximum(data - anchor, 0.0)
        median_in = float(np.median(data_anchored))
        luminance = data_anchored

    log(f"Input median (anchored): {median_in:.6f}, target: {target_bg}")

    # Solve for optimal log_d
    log_d = solve_log_d(
        median_in=median_in,
        target_median=target_bg,
        b=protect_b,
        log_d_min=getattr(config, "veralux_log_d_min", 0.0),
        log_d_max=getattr(config, "veralux_log_d_max", 7.0),
    )

    D = 10.0**log_d
    log(f"Calculated log_d={log_d:.2f} (D={D:.1f})")

    if is_rgb:
        # Step 5: Stretch luminance only
        lum_stretched = _hyperbolic_stretch_array(luminance, D, protect_b)

        # Step 6: Color convergence (reference uses power=3.5)
        # This blends color ratios toward neutral (1.0) for bright pixels
        k = np.power(lum_stretched, convergence_power)

        r_final = r_ratio * (1.0 - k) + 1.0 * k
        g_final = g_ratio * (1.0 - k) + 1.0 * k
        b_final = b_ratio * (1.0 - k) + 1.0 * k

        # Step 7: Reconstruct RGB using converged ratios
        stretched = np.zeros_like(data)
        stretched[0] = lum_stretched * r_final
        stretched[1] = lum_stretched * g_final
        stretched[2] = lum_stretched * b_final

        # Step 8: Ready-to-Use mode - apply adaptive output scaling with MTF
        # This matches the original VeraLux RTU mode which applies pedestal,
        # contrast scaling, and MTF tone mapping to reach target background.
        # Scientific mode would skip this step.
        stretched = _adaptive_output_scaling(stretched, target_bg)

        # Step 9: Apply soft-clip to each channel for highlight protection
        stretched[0] = _soft_clip_channel(stretched[0])
        stretched[1] = _soft_clip_channel(stretched[1])
        stretched[2] = _soft_clip_channel(stretched[2])
    else:
        # Direct stretch for mono
        stretched = _hyperbolic_stretch_array(luminance, D, protect_b)
        stretched = _adaptive_output_scaling(stretched, target_bg)
        stretched = _soft_clip_channel(stretched)

    # Clip to valid range
    stretched = np.clip(stretched, 0, 1)

    # Convert back to 16-bit for saving
    out_data = (stretched * 65535).astype(np.uint16)

    # Save stretched image to a new output file (avoid Windows file lock issues)
    output_path = image_path.with_name(image_path.stem + "_stretched.fit")
    hdu = fits.PrimaryHDU(out_data, header=header)
    hdu.writeto(output_path, overwrite=True)

    # Reload stretched image into Siril
    if not siril.load(str(output_path)):
        return False, log_d

    log(f"Stretch complete: convergence_power={convergence_power}")
    return True, log_d
