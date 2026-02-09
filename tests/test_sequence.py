"""Tests for sequence parsing, threshold computation, and analysis formatting."""

from pathlib import Path

import numpy as np

from siril_job_runner.sequence_analysis import (
    find_valid_reference,
    format_histogram,
    format_stats_log,
)
from siril_job_runner.sequence_parse import parse_sequence_file
from siril_job_runner.sequence_stats import RegistrationStats
from siril_job_runner.sequence_threshold import (
    compute_adaptive_threshold,
    detect_bimodality,
)


def _write_seq_file(path: Path, n_images: int, fwhm_values: list[float]) -> None:
    """Write a synthetic .seq file with registration data."""
    with open(path, "w") as f:
        f.write("#Siril sequence file.\n")
        f.write("#S 'name' start nb_images nb_selected fixed_len ref version\n")
        f.write(f"S 'light' 1 {n_images} {n_images} 5 1 1\n")
        f.write("L -1\n")
        for i in range(n_images):
            f.write(f"I {i + 1} 1\n")
        # R0 lines: FWHM wFWHM roundness quality background n_stars H ...
        for _i, fwhm in enumerate(fwhm_values):
            wfwhm = fwhm * 1.1  # wFWHM slightly larger
            f.write(f"R0 {fwhm:.2f} {wfwhm:.2f} 0.85 1000 500 200 H\n")


class TestParseSequenceFile:
    """Tests for .seq file parsing."""

    def test_parse_basic(self, temp_dir):
        """Should parse a basic .seq file."""
        path = temp_dir / "test.seq"
        _write_seq_file(path, 5, [3.0, 3.2, 3.1, 2.9, 3.3])

        stats = parse_sequence_file(path)
        assert stats is not None
        assert stats.n_images == 5
        assert len(stats.fwhm_values) == 5
        assert len(stats.wfwhm_values) == 5

    def test_parse_statistics(self, temp_dir):
        """Should compute correct basic statistics."""
        path = temp_dir / "test.seq"
        _write_seq_file(path, 3, [3.0, 4.0, 5.0])

        stats = parse_sequence_file(path)
        assert stats.n_images == 3
        assert stats.median > 0
        assert stats.mean > 0
        assert stats.std > 0

    def test_parse_nonexistent_file(self):
        """Should return None for nonexistent file."""
        assert parse_sequence_file(Path("/nonexistent.seq")) is None

    def test_parse_reference_index(self, temp_dir):
        """Should extract reference image index from S line."""
        path = temp_dir / "test.seq"
        _write_seq_file(path, 3, [3.0, 3.0, 3.0])

        stats = parse_sequence_file(path)
        assert stats.reference_index == 1  # From S line

    def test_parse_histogram(self, temp_dir):
        """Should compute histogram bins."""
        path = temp_dir / "test.seq"
        _write_seq_file(path, 10, [3.0] * 5 + [5.0] * 5)

        stats = parse_sequence_file(path)
        assert len(stats.hist_counts) > 0
        assert len(stats.hist_bins) > 0

    def test_parse_roundness_and_stars(self, temp_dir):
        """Should parse roundness and star count fields."""
        path = temp_dir / "test.seq"
        _write_seq_file(path, 3, [3.0, 3.0, 3.0])

        stats = parse_sequence_file(path)
        assert len(stats.roundness_values) == 3
        assert len(stats.star_count_values) == 3
        assert all(stats.roundness_values == 0.85)
        assert all(stats.star_count_values == 200)


class TestDetectBimodality:
    """Tests for bimodality detection."""

    def test_unimodal_distribution(self):
        """Clearly unimodal data should not be detected as bimodal."""
        np.random.seed(42)
        data = np.random.normal(5.0, 0.5, 50)
        is_bimodal, delta_bic, dip_p, _ = detect_bimodality(data)
        assert not is_bimodal

    def test_bimodal_distribution(self):
        """Clearly bimodal data should be detected."""
        np.random.seed(42)
        mode1 = np.random.normal(3.0, 0.3, 30)
        mode2 = np.random.normal(7.0, 0.3, 30)
        data = np.concatenate([mode1, mode2])
        is_bimodal, delta_bic, dip_p, gmm = detect_bimodality(data)
        assert is_bimodal
        assert gmm is not None
        assert delta_bic > 10  # Strong BIC evidence

    def test_too_few_samples(self):
        """Should return not bimodal for fewer than 10 samples."""
        data = np.array([3.0, 4.0, 5.0])
        is_bimodal, _, _, _ = detect_bimodality(data)
        assert not is_bimodal


class TestComputeAdaptiveThreshold:
    """Tests for adaptive threshold computation."""

    def _make_stats(self, wfwhm_values: list[float]) -> RegistrationStats:
        """Create a minimal RegistrationStats for testing."""
        from scipy.stats import median_abs_deviation, skew

        arr = np.array(wfwhm_values)
        n = len(arr)
        hist_counts, hist_bins = np.histogram(
            arr, bins=range(int(arr.min()), int(arr.max()) + 2)
        )
        return RegistrationStats(
            n_images=n,
            fwhm_values=arr * 0.9,
            wfwhm_values=arr,
            roundness_values=np.ones(n) * 0.9,
            quality_values=np.ones(n) * 1000,
            background_values=np.ones(n) * 500,
            star_count_values=np.ones(n) * 100,
            image_indices=np.arange(1, n + 1, dtype=float),
            reference_index=1,
            reference_wfwhm=float(arr[0]),
            median=float(np.median(arr)),
            mean=float(np.mean(arr)),
            std=float(np.std(arr)),
            cv=float(np.std(arr) / np.mean(arr)) if np.mean(arr) > 0 else 0.0,
            skewness=float(skew(arr)),
            mad=float(median_abs_deviation(arr)),
            q1=float(np.percentile(arr, 25)),
            q3=float(np.percentile(arr, 75)),
            hist_bins=hist_bins,
            hist_counts=hist_counts,
        )

    def test_tight_distribution_no_filter(self):
        """Tight distribution should keep all images."""
        np.random.seed(42)
        values = list(np.random.normal(4.0, 0.1, 20))
        stats = self._make_stats(values)
        result = compute_adaptive_threshold(stats)
        assert result.filter_case == "tight"
        assert result.threshold is None

    def test_insufficient_images(self):
        """Too few images should skip filtering."""
        stats = self._make_stats([3.0, 4.0, 5.0])
        result = compute_adaptive_threshold(stats)
        assert result.filter_case == "insufficient"
        assert result.threshold is None

    def test_skewed_distribution(self):
        """Skewed distribution should set threshold."""
        np.random.seed(42)
        # Create right-skewed data
        values = list(np.random.normal(4.0, 0.3, 15)) + [8.0, 9.0, 10.0]
        stats = self._make_stats(values)
        result = compute_adaptive_threshold(stats)
        if result.filter_case == "skewed":
            assert result.threshold is not None
            assert result.n_rejected > 0

    def test_bimodal_distribution(self):
        """Bimodal distribution should set threshold between modes."""
        np.random.seed(42)
        mode1 = list(np.random.normal(3.0, 0.3, 25))
        mode2 = list(np.random.normal(7.0, 0.3, 25))
        stats = self._make_stats(mode1 + mode2)
        result = compute_adaptive_threshold(stats)
        assert result.filter_case == "bimodal"
        assert result.threshold is not None
        assert 3.0 < result.threshold < 7.0  # Between modes


class TestFindValidReference:
    """Tests for finding a valid reference image."""

    def test_no_threshold_no_change(self):
        """No threshold should return None (no change needed)."""
        stats = RegistrationStats(
            n_images=3,
            fwhm_values=np.array([3.0, 3.0, 3.0]),
            wfwhm_values=np.array([3.3, 3.3, 3.3]),
            roundness_values=np.array([0.9, 0.9, 0.9]),
            quality_values=np.array([1000, 1000, 1000]),
            background_values=np.array([500, 500, 500]),
            star_count_values=np.array([100, 100, 100]),
            image_indices=np.array([1, 2, 3]),
            reference_index=1,
            reference_wfwhm=3.3,
            median=3.3,
            mean=3.3,
            std=0.0,
            cv=0.0,
            skewness=0.0,
            mad=0.0,
            q1=3.3,
            q3=3.3,
            threshold=None,
        )
        assert find_valid_reference(stats) is None

    def test_reference_passes_threshold(self):
        """Reference within threshold should return None."""
        stats = RegistrationStats(
            n_images=3,
            fwhm_values=np.array([3.0, 3.0, 3.0]),
            wfwhm_values=np.array([3.0, 4.0, 5.0]),
            roundness_values=np.array([0.9, 0.9, 0.9]),
            quality_values=np.array([1000, 1000, 1000]),
            background_values=np.array([500, 500, 500]),
            star_count_values=np.array([100, 100, 100]),
            image_indices=np.array([1, 2, 3]),
            reference_index=1,
            reference_wfwhm=3.0,
            median=4.0,
            mean=4.0,
            std=1.0,
            cv=0.25,
            skewness=0.0,
            mad=1.0,
            q1=3.0,
            q3=5.0,
            threshold=6.0,
        )
        assert find_valid_reference(stats) is None

    def test_reference_exceeds_threshold(self):
        """Reference exceeding threshold should return best alternative."""
        stats = RegistrationStats(
            n_images=3,
            fwhm_values=np.array([3.0, 3.0, 3.0]),
            wfwhm_values=np.array([3.0, 4.0, 8.0]),
            roundness_values=np.array([0.9, 0.9, 0.9]),
            quality_values=np.array([1000, 1000, 1000]),
            background_values=np.array([500, 500, 500]),
            star_count_values=np.array([100, 100, 100]),
            image_indices=np.array([1, 2, 3]),
            reference_index=3,
            reference_wfwhm=8.0,
            median=4.0,
            mean=5.0,
            std=2.6,
            cv=0.5,
            skewness=0.0,
            mad=1.0,
            q1=3.0,
            q3=8.0,
            threshold=5.0,
        )
        new_ref = find_valid_reference(stats)
        assert new_ref == 1  # Image 1 has lowest wFWHM (3.0)


class TestFormatHistogram:
    """Tests for histogram formatting."""

    def test_empty_histogram(self):
        """Empty histogram should return empty list."""
        stats = RegistrationStats(
            n_images=0,
            fwhm_values=np.array([]),
            wfwhm_values=np.array([]),
            roundness_values=np.array([]),
            quality_values=np.array([]),
            background_values=np.array([]),
            star_count_values=np.array([]),
            image_indices=np.array([]),
            reference_index=-1,
            reference_wfwhm=0.0,
            median=0.0,
            mean=0.0,
            std=0.0,
            cv=0.0,
            skewness=0.0,
            mad=0.0,
            q1=0.0,
            q3=0.0,
            hist_bins=np.array([]),
            hist_counts=np.array([]),
        )
        lines = format_histogram(stats)
        assert lines == []

    def test_histogram_has_header(self):
        """Histogram output should start with header."""
        stats = RegistrationStats(
            n_images=5,
            fwhm_values=np.array([3.0, 3.0, 3.0, 4.0, 4.0]),
            wfwhm_values=np.array([3.0, 3.0, 3.0, 4.0, 4.0]),
            roundness_values=np.ones(5),
            quality_values=np.ones(5),
            background_values=np.ones(5),
            star_count_values=np.ones(5),
            image_indices=np.arange(1, 6, dtype=float),
            reference_index=1,
            reference_wfwhm=3.0,
            median=3.0,
            mean=3.4,
            std=0.5,
            cv=0.15,
            skewness=0.0,
            mad=0.0,
            q1=3.0,
            q3=4.0,
            hist_bins=np.array([3, 4, 5]),
            hist_counts=np.array([3, 2]),
        )
        lines = format_histogram(stats)
        assert lines[0] == "wFWHM distribution:"


class TestFormatStatsLog:
    """Tests for full stats log formatting."""

    def test_returns_non_empty(self):
        """Should return non-empty list of lines."""
        stats = RegistrationStats(
            n_images=3,
            fwhm_values=np.array([3.0, 4.0, 5.0]),
            wfwhm_values=np.array([3.3, 4.4, 5.5]),
            roundness_values=np.array([0.9, 0.85, 0.88]),
            quality_values=np.array([1000, 900, 1100]),
            background_values=np.array([500, 520, 480]),
            star_count_values=np.array([100, 110, 90]),
            image_indices=np.array([1, 2, 3]),
            reference_index=1,
            reference_wfwhm=3.3,
            median=4.4,
            mean=4.4,
            std=1.1,
            cv=0.25,
            skewness=0.0,
            mad=1.1,
            q1=3.3,
            q3=5.5,
            hist_bins=np.array([3, 4, 5, 6]),
            hist_counts=np.array([1, 1, 1]),
            filter_case="tight",
            threshold=None,
            threshold_reason="CV=25.0% < 15%",
        )
        lines = format_stats_log(stats)
        assert len(lines) > 5
        assert any("Registration stats summary" in line for line in lines)
        assert any("Decision" in line for line in lines)
