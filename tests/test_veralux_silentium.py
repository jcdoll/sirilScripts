"""Tests for veralux_silentium module."""

import numpy as np

from siril_job_runner.veralux_silentium import (
    _compute_edge_map,
    _compute_signal_probability,
    _estimate_noise_map,
    _soft_threshold,
    denoise_channel,
    denoise_image,
)


class TestSoftThreshold:
    """Tests for soft thresholding function."""

    def test_soft_threshold_zeros_below(self):
        """Values below threshold should become zero."""
        data = np.array([0.5, 1.5, 2.5])
        threshold = np.array([1.0, 1.0, 1.0])
        result = _soft_threshold(data, threshold)
        assert result[0] == 0.0

    def test_soft_threshold_shrinks_above(self):
        """Values above threshold should be shrunk."""
        data = np.array([2.0])
        threshold = np.array([1.0])
        result = _soft_threshold(data, threshold)
        np.testing.assert_almost_equal(result[0], 1.0)

    def test_soft_threshold_preserves_sign(self):
        """Sign should be preserved after thresholding."""
        data = np.array([-2.0, 2.0])
        threshold = np.array([1.0, 1.0])
        result = _soft_threshold(data, threshold)
        assert result[0] < 0
        assert result[1] > 0


class TestNoiseEstimate:
    """Tests for noise estimation."""

    def test_noise_estimate_positive(self):
        """Noise estimate should be positive."""
        data = np.random.rand(64, 64)
        sigma_map, global_sigma = _estimate_noise_map(data)
        assert global_sigma > 0
        assert sigma_map.min() >= 0

    def test_noise_estimate_scales_with_noise(self):
        """Higher noise should give higher estimate."""
        np.random.seed(42)
        clean = np.random.rand(64, 64) * 0.1
        noisy = clean + np.random.randn(64, 64) * 0.1

        _, sigma_clean = _estimate_noise_map(clean)
        _, sigma_noisy = _estimate_noise_map(noisy)

        assert sigma_noisy > sigma_clean


class TestEdgeMap:
    """Tests for edge detection map."""

    def test_edge_map_shape(self):
        """Output should match input shape."""
        L = np.random.rand(64, 64)
        edge_map = _compute_edge_map(L)
        assert edge_map.shape == L.shape

    def test_edge_map_range(self):
        """Edge map should be in [0, 1]."""
        L = np.random.rand(64, 64)
        edge_map = _compute_edge_map(L)
        assert edge_map.min() >= 0.0
        assert edge_map.max() <= 1.0

    def test_edge_map_high_at_edges(self):
        """Edges should have higher values."""
        L = np.zeros((64, 64), dtype=np.float32)
        L[20:44, 20:44] = 1.0

        edge_map = _compute_edge_map(L)
        center = edge_map[32, 32]
        edge = edge_map[20, 32]
        assert edge > center


class TestSignalProbability:
    """Tests for signal probability map."""

    def test_signal_probability_shape(self):
        """Output should match input shape."""
        channel = np.random.rand(64, 64)
        signal_map = _compute_signal_probability(channel)
        assert signal_map.shape == channel.shape

    def test_signal_probability_range(self):
        """Signal probability should be in [0, 1]."""
        channel = np.random.rand(64, 64)
        signal_map = _compute_signal_probability(channel)
        assert signal_map.min() >= 0.0
        assert signal_map.max() <= 1.0

    def test_signal_probability_high_for_bright(self):
        """Bright areas should have higher signal probability."""
        # Create image with background noise and bright signal
        np.random.seed(42)
        channel = np.random.rand(64, 64).astype(np.float32) * 0.1  # Background noise
        channel[20:44, 20:44] = 0.8  # Bright signal region

        signal_map = _compute_signal_probability(channel)
        bg_prob = signal_map[:20, :20].mean()  # Corner background
        sig_prob = signal_map[25:40, 25:40].mean()  # Center signal
        assert sig_prob > bg_prob


class TestDenoiseChannel:
    """Tests for single channel denoising."""

    def test_denoise_returns_correct_shape(self):
        """Output should match input shape."""
        channel = np.random.rand(64, 64).astype(np.float32)
        result = denoise_channel(channel, intensity=25)
        assert result.shape == channel.shape

    def test_denoise_reduces_noise(self):
        """Denoising should reduce noise variance."""
        np.random.seed(42)
        clean = np.ones((64, 64), dtype=np.float32) * 0.5
        noisy = clean + np.random.randn(64, 64).astype(np.float32) * 0.05

        denoised = denoise_channel(noisy, intensity=50, detail_guard=0)

        noise_before = np.std(noisy - clean)
        noise_after = np.std(denoised - clean)

        assert noise_after < noise_before

    def test_denoise_zero_intensity_near_identity(self):
        """Zero intensity should be close to identity."""
        np.random.seed(42)
        channel = (np.random.rand(64, 64) * 0.5 + 0.25).astype(np.float32)
        result = denoise_channel(channel, intensity=0)
        np.testing.assert_array_almost_equal(channel, result, decimal=2)

    def test_denoise_handles_non_power_of_two(self):
        """Should handle images with non-power-of-two dimensions."""
        channel = np.random.rand(67, 53).astype(np.float32)
        result = denoise_channel(channel, intensity=25)
        assert result.shape == channel.shape


class TestDenoiseImage:
    """Tests for full RGB denoising."""

    def test_denoise_image_returns_correct_shape(self):
        """Output should match input shape."""
        data = (np.random.rand(3, 64, 64) * 0.5 + 0.25).astype(np.float32)
        result, stats = denoise_image(data)
        assert result.shape == data.shape

    def test_denoise_image_returns_valid_range(self):
        """Output should be in [0, 1]."""
        data = (np.random.rand(3, 64, 64) * 0.5 + 0.25).astype(np.float32)
        result, stats = denoise_image(data, intensity=50)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_denoise_image_returns_stats(self):
        """Should return noise estimates in stats."""
        data = np.random.rand(3, 64, 64).astype(np.float32)
        _, stats = denoise_image(data)

        assert "L_noise_estimate" in stats
        assert "a_noise_estimate" in stats
        assert "b_noise_estimate" in stats
        assert "intensity" in stats

    def test_denoise_image_chroma_processing(self):
        """Chroma channels should be processed separately."""
        np.random.seed(42)
        data = (np.random.rand(3, 64, 64) * 0.5 + 0.25).astype(np.float32)

        _, stats_low = denoise_image(data, chroma=10)
        _, stats_high = denoise_image(data, chroma=90)

        assert stats_high["chroma_intensity"] > stats_low["chroma_intensity"]
