"""
PSF analysis module for deconvolution diagnostics.

Analyzes saved PSF kernels and reports statistics.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from astropy.io import fits


@dataclass
class PSFStats:
    """Statistics from PSF kernel analysis."""

    dimensions: tuple[int, int]
    peak_value: float
    peak_location: tuple[int, int]
    total_flux: float
    fwhm_x: float
    fwhm_y: float
    ellipticity: float
    ee50_radius: float  # 50% encircled energy radius
    ee90_radius: float  # 90% encircled energy radius


def _compute_fwhm_1d(profile: np.ndarray) -> float:
    """Compute FWHM from a 1D profile."""
    half_max = profile.max() / 2.0
    above_half = profile >= half_max
    if not above_half.any():
        return 0.0
    indices = np.where(above_half)[0]
    return float(indices[-1] - indices[0] + 1)


def _compute_encircled_energy(
    data: np.ndarray, center: tuple[int, int]
) -> tuple[float, float]:
    """
    Compute radii containing 50% and 90% of total energy.

    Returns (ee50_radius, ee90_radius) in pixels.
    """
    cy, cx = center
    ny, nx = data.shape
    total = data.sum()
    if total <= 0:
        return 0.0, 0.0

    # Create distance array from center
    y, x = np.ogrid[:ny, :nx]
    distances = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    # Sort pixels by distance
    max_radius = max(ny, nx) / 2
    radii = np.linspace(0, max_radius, 50)

    ee50_radius = 0.0
    ee90_radius = 0.0

    for r in radii:
        mask = distances <= r
        enclosed = data[mask].sum() / total
        if enclosed >= 0.5 and ee50_radius == 0.0:
            ee50_radius = r
        if enclosed >= 0.9 and ee90_radius == 0.0:
            ee90_radius = r
            break

    return ee50_radius, ee90_radius


def analyze_psf(path: Path) -> Optional[PSFStats]:
    """
    Load PSF FITS file and compute statistics.

    Args:
        path: Path to PSF FITS file

    Returns:
        PSFStats with computed parameters, or None if analysis fails
    """
    if not path.exists():
        return None

    try:
        with fits.open(path) as hdu:
            data = hdu[0].data.astype(np.float64)
    except Exception:
        return None

    if data is None or data.size == 0:
        return None

    # Handle 3D data (color PSF) by taking first channel
    if data.ndim == 3:
        data = data[0]

    ny, nx = data.shape
    dimensions = (ny, nx)

    # Peak
    peak_value = float(data.max())
    peak_idx = np.unravel_index(np.argmax(data), data.shape)
    peak_location = (int(peak_idx[0]), int(peak_idx[1]))

    # Total flux
    total_flux = float(data.sum())

    # FWHM in X and Y through peak
    cy, cx = peak_location
    profile_y = data[:, cx]
    profile_x = data[cy, :]
    fwhm_y = _compute_fwhm_1d(profile_y)
    fwhm_x = _compute_fwhm_1d(profile_x)

    # Ellipticity
    ellipticity = fwhm_x / fwhm_y if fwhm_y > 0 else 1.0

    # Encircled energy
    ee50_radius, ee90_radius = _compute_encircled_energy(data, peak_location)

    return PSFStats(
        dimensions=dimensions,
        peak_value=peak_value,
        peak_location=peak_location,
        total_flux=total_flux,
        fwhm_x=fwhm_x,
        fwhm_y=fwhm_y,
        ellipticity=ellipticity,
        ee50_radius=ee50_radius,
        ee90_radius=ee90_radius,
    )


def format_psf_stats(stats: PSFStats) -> list[str]:
    """Format PSF stats as log lines."""
    return [
        f"Dimensions: {stats.dimensions[0]}x{stats.dimensions[1]}, "
        f"Peak: {stats.peak_value:.2f} at {stats.peak_location}",
        f"FWHM: {stats.fwhm_x:.1f}x{stats.fwhm_y:.1f} px, "
        f"Ellipticity: {stats.ellipticity:.2f}",
        f"Encircled energy: 50% at r={stats.ee50_radius:.1f}px, "
        f"90% at r={stats.ee90_radius:.1f}px",
    ]
