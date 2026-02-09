"""Tests for psf_analysis module."""

from pathlib import Path

import numpy as np
from astropy.io import fits

from siril_job_runner.psf_analysis import (
    _compute_encircled_energy,
    _compute_fwhm_1d,
    analyze_psf,
    format_psf_stats,
)


class TestComputeFWHM:
    """Tests for 1D FWHM computation."""

    def test_gaussian_fwhm(self):
        """FWHM of a Gaussian should be approximately 2.355 * sigma."""
        x = np.arange(100)
        sigma = 5.0
        profile = np.exp(-0.5 * ((x - 50) / sigma) ** 2)
        fwhm = _compute_fwhm_1d(profile)
        expected = 2.355 * sigma
        assert abs(fwhm - expected) < 2.0  # Within 2 pixels

    def test_zero_profile(self):
        """All-zero profile: half_max=0, all pixels >= 0, returns full width."""
        profile = np.zeros(50)
        assert _compute_fwhm_1d(profile) == 50.0

    def test_single_peak(self):
        """Single bright pixel should have FWHM of 1."""
        profile = np.zeros(50)
        profile[25] = 1.0
        assert _compute_fwhm_1d(profile) == 1.0


class TestEncircledEnergy:
    """Tests for encircled energy computation."""

    def test_point_source(self):
        """Point source should have small encircled energy radii."""
        data = np.zeros((51, 51))
        data[25, 25] = 1.0
        ee50, ee90 = _compute_encircled_energy(data, (25, 25))
        assert ee50 < 5.0
        assert ee90 < 5.0

    def test_extended_source(self):
        """Extended source should have larger radii."""
        y, x = np.ogrid[:51, :51]
        data = np.exp(-((x - 25) ** 2 + (y - 25) ** 2) / (2 * 8**2))
        ee50, ee90 = _compute_encircled_energy(data, (25, 25))
        assert ee90 > ee50
        assert ee50 > 0

    def test_zero_image(self):
        """Zero image should return 0 radii."""
        data = np.zeros((20, 20))
        ee50, ee90 = _compute_encircled_energy(data, (10, 10))
        assert ee50 == 0.0
        assert ee90 == 0.0


class TestAnalyzePSF:
    """Tests for full PSF analysis."""

    def test_analyze_gaussian_psf(self, temp_dir):
        """Should analyze a synthetic Gaussian PSF."""
        y, x = np.ogrid[:31, :31]
        data = np.exp(-((x - 15) ** 2 + (y - 15) ** 2) / (2 * 3**2))
        data = data.astype(np.float32)

        path = temp_dir / "psf.fit"
        hdu = fits.PrimaryHDU(data)
        hdu.writeto(path)

        stats = analyze_psf(path)
        assert stats is not None
        assert stats.dimensions == (31, 31)
        assert stats.peak_value > 0.9
        assert stats.peak_location == (15, 15)
        assert stats.fwhm_x > 0
        assert stats.fwhm_y > 0

    def test_analyze_nonexistent_file(self):
        """Should return None for nonexistent file."""
        assert analyze_psf(Path("/nonexistent.fit")) is None

    def test_ellipticity_symmetric(self, temp_dir):
        """Symmetric PSF should have ellipticity near 1.0."""
        y, x = np.ogrid[:31, :31]
        data = np.exp(-((x - 15) ** 2 + (y - 15) ** 2) / (2 * 4**2))
        data = data.astype(np.float32)

        path = temp_dir / "psf.fit"
        hdu = fits.PrimaryHDU(data)
        hdu.writeto(path)

        stats = analyze_psf(path)
        assert abs(stats.ellipticity - 1.0) < 0.1


class TestFormatPSFStats:
    """Tests for PSF stats formatting."""

    def test_format_returns_lines(self, temp_dir):
        """Should return list of formatted strings."""
        y, x = np.ogrid[:31, :31]
        data = np.exp(-((x - 15) ** 2 + (y - 15) ** 2) / (2 * 3**2))
        data = data.astype(np.float32)

        path = temp_dir / "psf.fit"
        hdu = fits.PrimaryHDU(data)
        hdu.writeto(path)

        stats = analyze_psf(path)
        lines = format_psf_stats(stats)
        assert len(lines) == 3
        assert "Dimensions" in lines[0]
        assert "FWHM" in lines[1]
        assert "Encircled" in lines[2]
