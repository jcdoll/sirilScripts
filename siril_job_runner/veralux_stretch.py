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
RTU_PEDESTAL = 0.001
RTU_SOFT_CEIL_PERCENTILE = 99.0

# Sensor profiles: (R_weight, G_weight, B_weight)
# Based on hardware-specific Quantum Efficiency measurements
# Reference: VeraLux Database v2.2 - Siril SPCC Derived
SENSOR_PROFILES: dict[str, tuple[float, float, float]] = {
    # Standard
    "rec709": (0.2126, 0.7152, 0.0722),  # ITU-R BT.709 (default)
    # Sony Modern BSI
    "imx571": (0.2944, 0.5021, 0.2035),  # Sony IMX571 (ASI2600/QHY268)
    "imx455": (0.2987, 0.5001, 0.2013),  # Sony IMX455 (ASI6200/QHY600)
    "imx410": (0.3015, 0.5050, 0.1935),  # Sony IMX410 (ASI2400)
    "imx269": (0.3040, 0.5010, 0.1950),  # Sony IMX269 (Altair/ToupTek)
    "imx294": (0.3068, 0.5008, 0.1925),  # Sony IMX294 (ASI294)
    # Sony Medium Format / Square
    "imx533": (0.2910, 0.5072, 0.2018),  # Sony IMX533 (ASI533)
    "imx676": (0.2880, 0.5100, 0.2020),  # Sony IMX676 (ASI676)
    # Sony Planetary / Guiding (STARVIS 2)
    "imx585": (0.3431, 0.4822, 0.1747),  # Sony IMX585 (ASI585)
    "imx662": (0.3430, 0.4821, 0.1749),  # Sony IMX662 (ASI662)
    "imx678": (0.3426, 0.4825, 0.1750),  # Sony IMX678 (ASI678)
    "imx462": (0.3333, 0.4866, 0.1801),  # Sony IMX462 (ASI462)
    "imx715": (0.3410, 0.4840, 0.1750),  # Sony IMX715 (ASI715)
    "imx482": (0.3150, 0.4950, 0.1900),  # Sony IMX482 (ASI482)
    "imx183": (0.2967, 0.4983, 0.2050),  # Sony IMX183 (ASI183)
    "imx178": (0.2346, 0.5206, 0.2448),  # Sony IMX178 (ASI178)
    "imx224": (0.3402, 0.4765, 0.1833),  # Sony IMX224 (ASI224)
    # Narrowband
    "hoo": (0.5000, 0.2500, 0.2500),  # HOO narrowband
    "sho": (0.3333, 0.3400, 0.3267),  # SHO narrowband
}


def get_sensor_weights(profile: str) -> tuple[float, float, float]:
    """
    Get RGB luminance weights for a sensor profile.

    Args:
        profile: Sensor profile name (case-insensitive)

    Returns:
        Tuple of (R_weight, G_weight, B_weight)
    """
    key = profile.lower().replace("-", "").replace("_", "")
    return SENSOR_PROFILES.get(key, SENSOR_PROFILES["rec709"])


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


def estimate_star_pressure(L_anchored) -> float:
    """
    Estimate star pressure (stellar dominance) in the image.

    Reference formula:
        p999 = 99.9th percentile
        p9999 = 99.99th percentile
        bright_frac = count(L > p999) / sample_size
        p_term = clip(p9999 / (p999 + eps), 1, 5) normalized to [0, 1]
        f_term = clip(bright_frac * 200, 0, 1)
        star_pressure = 0.7 * p_term + 0.3 * f_term

    Returns [0, 1]: 0 = no stars, 1 = extreme stellar dominance

    Args:
        L_anchored: Anchored luminance array

    Returns:
        Star pressure value in [0, 1]
    """
    import numpy as np

    # Sample up to 300k pixels for efficiency
    total_pixels = L_anchored.size
    stride = max(1, total_pixels // 300000)
    sampled = L_anchored.flat[::stride]

    # Remove zeros
    valid = sampled[sampled > 1e-7]
    if len(valid) < 100:
        return 0.0

    # Compute percentiles
    p999 = np.percentile(valid, 99.9)
    p9999 = np.percentile(valid, 99.99)

    # Bright fraction
    bright_count = np.sum(valid > p999)
    bright_frac = bright_count / len(valid)

    # p_term: ratio of extreme to very bright, normalized
    epsilon = 1e-9
    ratio = p9999 / (p999 + epsilon)
    p_term = np.clip((ratio - 1.0) / 4.0, 0.0, 1.0)  # maps [1, 5] to [0, 1]

    # f_term: fraction of bright pixels
    f_term = np.clip(bright_frac * 200.0, 0.0, 1.0)

    # Combined star pressure
    star_pressure = 0.7 * p_term + 0.3 * f_term

    return float(np.clip(star_pressure, 0.0, 1.0))


def solve_log_d(
    median_in: float,
    target_median: float,
    b: float,
    log_d_min: float = 0.0,
    log_d_max: float = 7.0,
    max_iterations: int = 40,
    tolerance: float = 0.0001,
) -> float:
    """
    Binary search to find log_d that achieves target median.

    Reference: Simple binary search without star pressure damping.
    Star pressure adjustment is applied to target_median BEFORE calling this.

    Args:
        median_in: Input median value
        target_median: Target median after stretch
        b: Protect b parameter
        log_d_min: Minimum log_d search bound
        log_d_max: Maximum log_d search bound
        max_iterations: Maximum binary search iterations
        tolerance: Convergence tolerance

    Returns:
        Optimal log_d value
    """
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


def _calculate_anchor_percentile(
    data, percentile: float = 0.5, offset: float = 0.00025
):
    """
    Calculate anchor (black point) using percentile method.

    Reference: For RGB, compute 0.5th percentile on EACH channel (with stride sampling),
    take minimum of those floors, subtract 0.00025 offset.
    """
    import numpy as np

    if data.ndim == 3 and data.shape[0] == 3:
        # Reference: stride = max(1, data_norm.size // 500000)
        stride = max(1, data.size // 500000)
        floors = []
        for c in range(3):
            floors.append(np.percentile(data[c].flatten()[::stride], percentile))
        anchor = min(floors) - offset
    elif data.ndim == 3 and data.shape[0] == 1:
        # Mono as (1, H, W)
        stride = max(1, data.size // 200000)
        anchor = np.percentile(data[0].flatten()[::stride], percentile) - offset
    else:
        # Mono as (H, W)
        stride = max(1, data.size // 200000)
        anchor = np.percentile(data.flatten()[::stride], percentile) - offset

    return max(anchor, 0.0)


def _calculate_anchor_morphological(
    data, weights: tuple[float, float, float], peak_threshold: float = 0.06
):
    """
    Calculate anchor using adaptive morphological histogram method.

    Reference:
    1. Build luminance proxy
    2. Sample up to 2M pixels
    3. Create histogram with 65536 bins
    4. Smooth with 50-wide box filter (np.convolve)
    5. Handle low-signal data (search_start logic)
    6. Find peak, then find left-side point where histogram < 6% of peak
    7. Fallback to 0.5 percentile of sample if no candidate

    Args:
        data: RGB data, shape (3, H, W)
        weights: RGB luminance weights
        peak_threshold: Fraction of peak to use as threshold (default 6%)

    Returns:
        Anchor value (black point)
    """
    import numpy as np

    r_w, g_w, b_w = weights

    # Build luminance proxy
    # Reference handles both (3,H,W) and (3,N) as RGB
    if (data.ndim == 3 and data.shape[0] == 3) or (
        data.ndim == 2 and data.shape[0] == 3
    ):
        L = r_w * data[0] + g_w * data[1] + b_w * data[2]
    elif data.ndim == 3 and data.shape[0] == 1:
        L = data[0]
    else:
        L = data

    # Reference: stride = max(1, base.size // 2000000)
    stride = max(1, L.size // 2000000)
    sample = L.flatten()[::stride]

    # Build histogram
    n_bins = 65536
    hist, bin_edges = np.histogram(sample, bins=n_bins, range=(0.0, 1.0))

    # Reference: box filter with 50-wide window (NOT Gaussian)
    hist_smooth = np.convolve(hist, np.ones(50) / 50, mode="same")

    # Reference: search_start logic for low-signal data
    # For very low-signal linear data, the histogram peak can sit below bin 100.
    search_start = 100
    if np.max(hist_smooth[:search_start]) > 0:
        search_start = 0
    if search_start >= len(hist_smooth):
        search_start = 0

    # Find peak starting from search_start
    peak_idx = int(np.argmax(hist_smooth[search_start:]) + search_start)
    peak_val = float(hist_smooth[peak_idx])

    # Reference: threshold = 6% of peak
    target_val = peak_val * peak_threshold

    # Reference: find anchor using np.where
    left_side = hist_smooth[:peak_idx]
    candidates = np.where(left_side < target_val)[0]

    if len(candidates) > 0:
        anchor_idx = candidates[-1]
        anchor = bin_edges[anchor_idx]
    else:
        # Reference fallback: 0.5 percentile of the SAME sample
        anchor = np.percentile(sample, 0.5)

    return max(0.0, anchor)


def _calculate_anchor(
    data, weights: tuple[float, float, float], use_morphological: bool = True
):
    """
    Calculate anchor (black point) using best available method.

    Args:
        data: Image data
        weights: RGB luminance weights
        use_morphological: If True, try morphological method first

    Returns:
        Anchor value (black point)
    """
    if use_morphological and data.ndim == 3 and data.shape[0] == 3:
        return _calculate_anchor_morphological(data, weights)
    return _calculate_anchor_percentile(data)


def _hyperbolic_stretch_array(data, D: float, b: float, SP: float = 0.0):
    """
    Apply hyperbolic stretch to a numpy array.

    Reference formula: (asinh(D*(data-SP)+b) - asinh(b)) / (asinh(D*(1-SP)+b) - asinh(b))
    """
    import numpy as np

    D = max(D, 0.1)
    b = max(b, 0.1)

    term1 = np.arcsinh(D * (data - SP) + b)
    term2 = np.arcsinh(b)
    norm_factor = np.arcsinh(D * (1.0 - SP) + b) - term2

    if norm_factor == 0:
        norm_factor = 1e-6

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


def _adaptive_output_scaling(
    data, weights: tuple[float, float, float], target_bg: float = 0.20
):
    """
    Adaptive output scaling from reference.

    Reference steps:
    1. Extract luminance using sensor weights
    2. Find global floor: max(min_L, median_L - 2.7*std_L)
    3. Smart Max: check if max is hot pixel
    4. Compute soft_ceil as max of per-channel 99.9th percentiles (RGB)
    5. Compute scale factors
    6. Apply scaling with pedestal
    7. Apply MTF to reach target background

    Args:
        data: Image data
        weights: RGB luminance weights
        target_bg: Target background median
    """
    import numpy as np

    r_weight, g_weight, b_weight = weights
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
    # Reference: uses strided sampling for performance
    if is_rgb:
        stride = max(1, R.size // 500000)
        soft_ceil = max(
            np.percentile(R.flatten()[::stride], RTU_SOFT_CEIL_PERCENTILE),
            np.percentile(G.flatten()[::stride], RTU_SOFT_CEIL_PERCENTILE),
            np.percentile(B.flatten()[::stride], RTU_SOFT_CEIL_PERCENTILE),
        )
    else:
        stride = max(1, L_raw.size // 200000)
        soft_ceil = np.percentile(L_raw.flatten()[::stride], RTU_SOFT_CEIL_PERCENTILE)

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
            # Linked MTF: apply to luminance, scale channels proportionally
            # This preserves color ratios (matches reference documentation "Linked MTF")
            L_before = r_weight * result[0] + g_weight * result[1] + b_weight * result[2]
            L_after = _apply_mtf(L_before, m)
            scale = L_after / (L_before + 1e-9)
            for i in range(3):
                result[i] = result[i] * scale
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
    2. Calculate anchor (black point) using morphological or percentile method
    3. Subtract anchor from data
    4. Extract luminance from anchored data using sensor weights
    5. Estimate star pressure for solver damping
    6. Compute color ratios from anchored data
    7. Stretch luminance
    8. Apply color convergence (power=3.5)
    9. Reconstruct RGB
    10. Apply adaptive output scaling (with MTF)
    11. Apply soft-clip

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
    color_grip = getattr(config, "veralux_color_grip", 1.0)
    shadow_convergence = getattr(config, "veralux_shadow_convergence", 0.0)

    # Get sensor profile weights
    sensor_profile = getattr(config, "veralux_sensor_profile", "rec709")
    weights = get_sensor_weights(sensor_profile)
    r_weight, g_weight, b_weight = weights
    log(
        f"Sensor profile: {sensor_profile} (R={r_weight:.4f}, G={g_weight:.4f}, B={b_weight:.4f})"
    )

    if is_rgb:
        log("RGB image detected - using vector preservation")

        # Step 1: Calculate anchor (black point) using percentile method (reference default)
        anchor = _calculate_anchor(data, weights, use_morphological=False)
        log(f"Anchor (black point): {anchor:.6f}")

        # Step 2: Subtract anchor from all channels
        img_anchored = np.maximum(data - anchor, 0.0)

        # Step 3: Extract luminance from ANCHORED data using sensor weights
        luminance = (
            r_weight * img_anchored[0]
            + g_weight * img_anchored[1]
            + b_weight * img_anchored[2]
        )

        # Step 4: Estimate star pressure for solver damping
        star_pressure = estimate_star_pressure(luminance)
        log(f"Star pressure: {star_pressure:.3f}")

        # Step 5: Compute color ratios from ANCHORED data
        # Reference (line 1099): epsilon = 1e-9; L_safe = L_anchored + epsilon
        epsilon = 1e-9
        lum_safe = luminance + epsilon
        r_ratio = img_anchored[0] / lum_safe
        g_ratio = img_anchored[1] / lum_safe
        b_ratio = img_anchored[2] / lum_safe

        # Calculate stretch parameters from anchored luminance
        median_in = float(np.median(luminance))
    else:
        log("Mono image detected - direct stretch")
        anchor = _calculate_anchor_percentile(data)
        log(f"Anchor (black point): {anchor:.6f}")
        data_anchored = np.maximum(data - anchor, 0.0)
        median_in = float(np.median(data_anchored))
        luminance = data_anchored
        star_pressure = estimate_star_pressure(luminance)
        log(f"Star pressure: {star_pressure:.3f}")

    log(f"Input median (anchored): {median_in:.6f}, target: {target_bg}")

    # Apply star pressure damping to target BEFORE solver (reference: lines 1534-1536)
    # Only applies when star_pressure > 0.6 to avoid star-driven over-compression
    effective_target = target_bg
    if star_pressure > 0.6:
        effective_target = target_bg * (1.0 - 0.15 * star_pressure)
        log(f"Star pressure damping: target adjusted to {effective_target:.4f}")

    # Solve for optimal log_d
    log_d = solve_log_d(
        median_in=median_in,
        target_median=effective_target,
        b=protect_b,
        log_d_min=getattr(config, "veralux_log_d_min", 0.0),
        log_d_max=getattr(config, "veralux_log_d_max", 7.0),
    )

    D = 10.0**log_d
    log(f"Calculated log_d={log_d:.2f} (D={D:.1f})")

    if is_rgb:
        # Step 6: Stretch luminance only
        lum_stretched = _hyperbolic_stretch_array(luminance, D, protect_b)
        # Reference line 1107: clip BEFORE color convergence
        lum_stretched = np.clip(lum_stretched, 0.0, 1.0)

        # Step 7: Color convergence (reference uses power=3.5)
        # This blends color ratios toward neutral (1.0) for bright pixels
        k = np.power(lum_stretched, convergence_power)

        r_final = r_ratio * (1.0 - k) + 1.0 * k
        g_final = g_ratio * (1.0 - k) + 1.0 * k
        b_final = b_ratio * (1.0 - k) + 1.0 * k

        # Step 8: Reconstruct RGB using converged ratios
        stretched = np.zeros_like(data)
        stretched[0] = lum_stretched * r_final
        stretched[1] = lum_stretched * g_final
        stretched[2] = lum_stretched * b_final

        # Step 8b: Hybrid mode (reference lines 1135-1152)
        # Blends linked stretch with per-channel scalar stretch
        needs_hybrid = (color_grip < 1.0) or (shadow_convergence > 0.01)
        if needs_hybrid:
            log(f"Hybrid mode: grip={color_grip}, shadow_conv={shadow_convergence}")
            # Compute per-channel scalar stretch
            scalar = np.zeros_like(stretched)
            scalar[0] = _hyperbolic_stretch_array(img_anchored[0], D, protect_b)
            scalar[1] = _hyperbolic_stretch_array(img_anchored[1], D, protect_b)
            scalar[2] = _hyperbolic_stretch_array(img_anchored[2], D, protect_b)
            scalar = np.clip(scalar, 0.0, 1.0)

            # Build grip map
            grip_map = np.full_like(lum_stretched, color_grip)

            # Apply shadow convergence damping
            if shadow_convergence > 0.01:
                damping = np.power(lum_stretched, shadow_convergence)
                grip_map = grip_map * damping

            # Blend: stretched = linked * grip + scalar * (1 - grip)
            stretched = (stretched * grip_map) + (scalar * (1.0 - grip_map))

        # Step 9: Apply pedestal (reference line 1156)
        stretched = stretched * (1.0 - 0.005) + 0.005
        stretched = np.clip(stretched, 0.0, 1.0)

        # Step 10: Ready-to-Use mode - apply adaptive output scaling with MTF
        stretched = _adaptive_output_scaling(stretched, weights, target_bg)

        # Step 11: Apply soft-clip to each channel for highlight protection
        stretched[0] = _soft_clip_channel(stretched[0])
        stretched[1] = _soft_clip_channel(stretched[1])
        stretched[2] = _soft_clip_channel(stretched[2])
    else:
        # Direct stretch for mono
        stretched = _hyperbolic_stretch_array(luminance, D, protect_b)
        # Reference line 1107: clip after stretch
        stretched = np.clip(stretched, 0.0, 1.0)
        # Apply pedestal (reference line 1156)
        stretched = stretched * (1.0 - 0.005) + 0.005
        stretched = np.clip(stretched, 0.0, 1.0)
        stretched = _adaptive_output_scaling(stretched, weights, target_bg)
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
