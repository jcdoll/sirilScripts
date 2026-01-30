"""Tests for veralux_revela module."""

import numpy as np

from siril_job_runner.veralux_revela import (
    _compute_signal_mask,
    _compute_star_mask_energy,
    enhance_details,
)
from siril_job_runner.veralux_core import atrous_decomposition


class TestSignalMask:
    """Tests for signal mask computation."""

    def test_signal_mask_shape(self):
        """Output should match input shape."""
        L = np.random.rand(64, 64)
        mask = _compute_signal_mask(L, shadow_auth=25.0)
        assert mask.shape == L.shape

    def test_signal_mask_range(self):
        """Signal mask should be in [0, 1]."""
        L = np.random.rand(64, 64)
        mask = _compute_signal_mask(L, shadow_auth=50.0)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_signal_mask_bright_areas_higher(self):
        """Bright areas should have higher signal mask values."""
        L = np.zeros((100, 100))
        L[:50, :] = 0.1  # Dark
        L[50:, :] = 0.8  # Bright

        mask = _compute_signal_mask(L, shadow_auth=25.0)

        dark_avg = np.mean(mask[:40, :])  # Avoid edge blur
        bright_avg = np.mean(mask[60:, :])
        assert bright_avg > dark_avg

    def test_signal_mask_shadow_auth_effect(self):
        """Higher shadow_auth should protect more shadow areas."""
        L = np.random.rand(64, 64) * 0.5  # Mid-range values

        mask_low = _compute_signal_mask(L, shadow_auth=10.0)
        mask_high = _compute_signal_mask(L, shadow_auth=90.0)

        # Higher shadow_auth = more restrictive = lower average mask
        assert np.mean(mask_high) < np.mean(mask_low)


class TestStarMaskEnergy:
    """Tests for energy-based star mask computation."""

    def test_star_mask_returns_two_masks(self):
        """Should return texture and structure protection masks."""
        L = np.random.rand(64, 64) * 50  # Lab L range
        planes, _ = atrous_decomposition(L, n_scales=6)

        tex_prot, str_prot = _compute_star_mask_energy(planes)

        assert tex_prot.shape == L.shape
        assert str_prot.shape == L.shape

    def test_star_mask_range(self):
        """Protection masks should be in [0, 1]."""
        L = np.random.rand(64, 64) * 50
        planes, _ = atrous_decomposition(L, n_scales=6)

        tex_prot, str_prot = _compute_star_mask_energy(planes)

        assert tex_prot.min() >= 0.0
        assert tex_prot.max() <= 1.0
        assert str_prot.min() >= 0.0
        assert str_prot.max() <= 1.0

    def test_star_mask_detects_point_sources(self):
        """Point sources should have lower protection (more masking)."""
        L = np.ones((100, 100)) * 20.0
        L[50, 50] = 100.0  # Bright point source

        planes, _ = atrous_decomposition(L, n_scales=6)
        tex_prot, _ = _compute_star_mask_energy(planes)

        # Star region should have lower protection value
        center_prot = tex_prot[45:55, 45:55].min()
        edge_prot = tex_prot[:10, :10].mean()
        assert center_prot < edge_prot


class TestEnhanceDetails:
    """Tests for the main enhancement function."""

    def test_enhance_returns_correct_shape(self):
        """Output should have same shape as input."""
        data = np.random.rand(3, 64, 64) * 0.5 + 0.25
        enhanced, stats = enhance_details(data)
        assert enhanced.shape == data.shape

    def test_enhance_returns_valid_range(self):
        """Output should be clipped to [0, 1]."""
        data = np.random.rand(3, 64, 64) * 0.5 + 0.25
        enhanced, stats = enhance_details(data, texture=100, structure=100)
        assert enhanced.min() >= 0.0
        assert enhanced.max() <= 1.0

    def test_enhance_zero_params_near_identity(self):
        """Zero enhancement should be close to identity."""
        np.random.seed(42)
        data = np.random.rand(3, 32, 32) * 0.5 + 0.25
        enhanced, stats = enhance_details(data, texture=0, structure=0)
        np.testing.assert_array_almost_equal(data, enhanced, decimal=2)

    def test_enhance_returns_stats(self):
        """Should return enhancement statistics."""
        data = np.random.rand(3, 32, 32)
        _, stats = enhance_details(data)

        assert "texture_gain" in stats
        assert "structure_gain" in stats
        assert "signal_coverage" in stats
        assert "star_coverage" in stats

    def test_enhance_texture_affects_gain(self):
        """Higher texture should increase texture gain."""
        np.random.seed(42)
        data = np.random.rand(3, 64, 64) * 0.5 + 0.25

        _, stats_low = enhance_details(data, texture=10, structure=0)
        _, stats_high = enhance_details(data, texture=90, structure=0)

        assert stats_high["texture_gain"] > stats_low["texture_gain"]

    def test_enhance_structure_affects_gain(self):
        """Higher structure should increase structure gain."""
        np.random.seed(42)
        data = np.random.rand(3, 64, 64) * 0.5 + 0.25

        _, stats_low = enhance_details(data, texture=0, structure=10)
        _, stats_high = enhance_details(data, texture=0, structure=90)

        assert stats_high["structure_gain"] > stats_low["structure_gain"]

    def test_enhance_with_star_protection(self):
        """Star protection should affect star coverage stat."""
        data = np.zeros((3, 64, 64)) + 0.1
        data[:, 32, 32] = 1.0

        _, stats_on = enhance_details(data, protect_stars=True)
        _, stats_off = enhance_details(data, protect_stars=False)

        # With protection off, star_coverage should be 0
        assert stats_off["star_coverage"] == 0.0

    def test_gain_formula_texture(self):
        """Texture gain should follow reference formula: 1 + (t/100 * 1.5)."""
        data = np.random.rand(3, 32, 32) * 0.5 + 0.25
        _, stats = enhance_details(data, texture=50, structure=0)

        expected_gain = 1.0 + (50 / 100.0 * 1.5)  # 1.75
        assert abs(stats["texture_gain"] - expected_gain) < 0.01

    def test_gain_formula_structure(self):
        """Structure gain should follow reference formula: 1 + (s/100 * 1.0)."""
        data = np.random.rand(3, 32, 32) * 0.5 + 0.25
        _, stats = enhance_details(data, texture=0, structure=50)

        expected_gain = 1.0 + (50 / 100.0 * 1.0)  # 1.5
        assert abs(stats["structure_gain"] - expected_gain) < 0.01
