"""
Sequence file analysis for adaptive FWHM filtering.

Parses Siril .seq files after registration to extract FWHM statistics
and compute adaptive thresholds based on distribution shape.

Adaptive Filtering Decision Tree:
    1. Bimodal (GMM+dip test): Two distinct populations of images.
       -> Threshold at midpoint between modes to reject the worse mode.

    2. Skewed (long tail): Single population with outliers on high end.
       -> Threshold at median + k*MAD (aggressive tail cutting).

    3. Broad symmetric (high CV, low skew): Wide but symmetric distribution.
       -> Threshold at high percentile (permissive, just cut extremes).

    4. Tight symmetric (low CV): All images similar quality.
       -> Keep all images, no filtering needed.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from diptest import diptest
from scipy.stats import median_abs_deviation, skew
from sklearn.mixture import GaussianMixture

from .config import DEFAULTS, Config


@dataclass
class RegistrationStats:
    """FWHM statistics from a registered sequence."""

    n_images: int
    fwhm_values: np.ndarray
    wfwhm_values: np.ndarray
    roundness_values: np.ndarray

    # Basic statistics
    median: float
    mean: float
    std: float
    cv: float  # coefficient of variation (std/mean)
    skewness: float  # distribution skewness (>0 = right tail)
    mad: float  # median absolute deviation
    q1: float
    q3: float

    # Bimodality analysis
    is_bimodal: bool = False
    delta_bic: float = 0.0
    dip_pvalue: float = 1.0
    gmm_means: Optional[np.ndarray] = None
    gmm_stds: Optional[np.ndarray] = None
    gmm_weights: Optional[np.ndarray] = None

    # Computed threshold
    threshold: Optional[float] = None
    threshold_reason: str = ""
    filter_case: str = ""  # bimodal, skewed, broad, tight
    n_rejected: int = 0

    # Histogram data
    hist_bins: np.ndarray = field(default_factory=lambda: np.array([]))
    hist_counts: np.ndarray = field(default_factory=lambda: np.array([]))


def parse_sequence_file(seq_path: Path) -> Optional[RegistrationStats]:
    """
    Parse a Siril .seq file and extract registration data.

    The .seq file contains R0 lines with format:
    R0 FWHM wFWHM roundness quality metric n_stars transform_type ...

    Args:
        seq_path: Path to .seq file

    Returns:
        RegistrationStats or None if parsing fails
    """
    if not seq_path.exists():
        return None

    fwhm_list = []
    wfwhm_list = []
    roundness_list = []

    with open(seq_path) as f:
        for line in f:
            if line.startswith("R0 "):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        fwhm = float(parts[1])
                        wfwhm = float(parts[2])
                        roundness = float(parts[3])

                        # Skip reference image (has 0 0 nan) and invalid entries
                        if fwhm > 0 and not np.isnan(roundness):
                            fwhm_list.append(fwhm)
                            wfwhm_list.append(wfwhm)
                            roundness_list.append(roundness)
                    except (ValueError, IndexError):
                        continue

    if len(fwhm_list) == 0:
        return None

    fwhm = np.array(fwhm_list)
    wfwhm = np.array(wfwhm_list)
    roundness = np.array(roundness_list)

    # Use wFWHM for all analysis since Siril's seqapplyreg filters by wFWHM
    # Compute histogram bins (1px wide, from floor to ceil)
    bin_min = max(0, int(np.floor(wfwhm.min())))
    bin_max = int(np.ceil(wfwhm.max())) + 1
    hist_counts, hist_bins = np.histogram(wfwhm, bins=range(bin_min, bin_max + 1))

    return RegistrationStats(
        n_images=len(wfwhm),
        fwhm_values=fwhm,
        wfwhm_values=wfwhm,
        roundness_values=roundness,
        median=float(np.median(wfwhm)),
        mean=float(np.mean(wfwhm)),
        std=float(np.std(wfwhm)),
        cv=float(np.std(wfwhm) / np.mean(wfwhm)) if np.mean(wfwhm) > 0 else 0.0,
        skewness=float(skew(wfwhm)),
        mad=float(median_abs_deviation(wfwhm)),
        q1=float(np.percentile(wfwhm, 25)),
        q3=float(np.percentile(wfwhm, 75)),
        hist_bins=hist_bins,
        hist_counts=hist_counts,
    )


def detect_bimodality(
    fwhm: np.ndarray,
    bic_threshold: float = 10.0,
    dip_alpha: float = 0.05,
) -> tuple[bool, float, float, Optional[GaussianMixture]]:
    """
    Detect bimodality using GMM + BIC with dip test confirmation.

    Uses two statistical tests that must both pass:
    1. BIC comparison: 2-component GMM must fit significantly better than 1-component
    2. Dip test: Distribution must have a significant "dip" indicating two modes

    Args:
        fwhm: Array of FWHM values
        bic_threshold: delta-BIC threshold (>10 = very strong evidence)
        dip_alpha: significance level for dip test

    Returns:
        (is_bimodal, delta_bic, dip_pvalue, gmm2_model)
    """
    if len(fwhm) < 10:
        return False, 0.0, 1.0, None

    fwhm_reshaped = fwhm.reshape(-1, 1)

    # Fit 1-component and 2-component GMMs
    gmm1 = GaussianMixture(n_components=1, random_state=42)
    gmm2 = GaussianMixture(n_components=2, random_state=42)

    gmm1.fit(fwhm_reshaped)
    gmm2.fit(fwhm_reshaped)

    # Compare BIC (lower is better, so positive delta means 2-component is better)
    delta_bic = gmm1.bic(fwhm_reshaped) - gmm2.bic(fwhm_reshaped)

    # Hartigan's dip test for unimodality
    dip_stat, dip_pvalue = diptest(fwhm)

    # Bimodal if both tests agree
    is_bimodal = (delta_bic > bic_threshold) and (dip_pvalue < dip_alpha)

    return is_bimodal, delta_bic, dip_pvalue, gmm2 if is_bimodal else None


def compute_adaptive_threshold(
    stats: RegistrationStats,
    config: Config = DEFAULTS,
) -> RegistrationStats:
    """
    Compute adaptive FWHM threshold based on distribution shape.

    Decision tree:
        1. Bimodal (GMM+dip): threshold at midpoint between modes
           - Cleanly separates two distinct image quality populations

        2. Skewed (skewness > threshold): threshold at median + k*MAD
           - Aggressively cuts the long tail of poor quality images

        3. Broad symmetric (high CV, low skew): threshold at high percentile
           - Permissive filtering, only removes extreme outliers

        4. Tight symmetric (low CV): keep all images
           - All images are similar quality, no filtering needed

    Args:
        stats: RegistrationStats from parse_sequence_file
        config: Configuration with threshold parameters

    Returns:
        Updated RegistrationStats with threshold and analysis
    """
    # Use wFWHM since that's what Siril's seqapplyreg filters on
    fwhm = stats.wfwhm_values

    # Case 0: Not enough images for statistical filtering
    if stats.n_images < config.fwhm_min_images:
        stats.threshold = None
        stats.filter_case = "insufficient"
        stats.threshold_reason = (
            f"Too few images ({stats.n_images} < {config.fwhm_min_images})"
        )
        return stats

    # Check for bimodality
    is_bimodal, delta_bic, dip_pvalue, gmm2 = detect_bimodality(
        fwhm,
        bic_threshold=config.fwhm_bic_threshold,
        dip_alpha=config.fwhm_dip_alpha,
    )

    stats.is_bimodal = is_bimodal
    stats.delta_bic = delta_bic
    stats.dip_pvalue = dip_pvalue

    # Case 1: Bimodal distribution - two distinct populations
    if is_bimodal and gmm2 is not None:
        means = gmm2.means_.flatten()
        stds = np.sqrt(gmm2.covariances_.flatten())
        weights = gmm2.weights_

        stats.gmm_means = means
        stats.gmm_stds = stds
        stats.gmm_weights = weights

        # Find lower and upper modes
        lower_idx = np.argmin(means)
        upper_idx = np.argmax(means)
        lower_mean = means[lower_idx]
        lower_std = stds[lower_idx]
        upper_mean = means[upper_idx]

        # Threshold = min of (lower_mean + k*sigma) or midpoint
        # This ensures we cleanly separate the two modes
        sigma_threshold = lower_mean + config.fwhm_bimodal_sigma * lower_std
        midpoint_threshold = (lower_mean + upper_mean) / 2

        if sigma_threshold < midpoint_threshold:
            threshold = sigma_threshold
            stats.threshold_reason = (
                f"{lower_mean:.2f} + {config.fwhm_bimodal_sigma}*{lower_std:.2f}"
            )
        else:
            threshold = midpoint_threshold
            stats.threshold_reason = f"midpoint({lower_mean:.2f}, {upper_mean:.2f})"

        stats.threshold = float(threshold)
        stats.filter_case = "bimodal"

    # Case 2: Skewed distribution - long tail on right
    elif stats.skewness > config.fwhm_skew_threshold:
        # Use median + k*MAD for robust tail cutting
        threshold = stats.median + config.fwhm_skew_mad_factor * stats.mad
        stats.threshold = float(threshold)
        stats.filter_case = "skewed"
        stats.threshold_reason = (
            f"median({stats.median:.2f}) + "
            f"{config.fwhm_skew_mad_factor}*MAD({stats.mad:.2f})"
        )

    # Case 3: Broad symmetric distribution - high variance but not skewed
    elif stats.cv > config.fwhm_cv_threshold:
        threshold = np.percentile(fwhm, config.fwhm_broad_percentile)
        stats.threshold = float(threshold)
        stats.filter_case = "broad"
        stats.threshold_reason = f"P{config.fwhm_broad_percentile:.0f}"

    # Case 4: Tight symmetric distribution - keep all
    else:
        stats.threshold = None
        stats.filter_case = "tight"
        stats.threshold_reason = f"CV={stats.cv:.1%} < {config.fwhm_cv_threshold:.0%}"

    # Count how many would be rejected
    if stats.threshold is not None:
        stats.n_rejected = int(np.sum(fwhm > stats.threshold))

    return stats


def format_histogram(stats: RegistrationStats, width: int = 30) -> list[str]:
    """
    Format FWHM distribution as text histogram.

    Args:
        stats: RegistrationStats with histogram data
        width: Maximum bar width in characters

    Returns:
        List of histogram lines
    """
    if len(stats.hist_counts) == 0:
        return []

    lines = ["wFWHM distribution:"]
    max_count = max(stats.hist_counts) if len(stats.hist_counts) > 0 else 1

    for i, count in enumerate(stats.hist_counts):
        if count == 0:
            continue

        bin_start = stats.hist_bins[i]
        bin_end = stats.hist_bins[i + 1]
        pct = 100 * count / stats.n_images

        # Scale bar to width
        bar_len = int(width * count / max_count) if max_count > 0 else 0
        bar = "#" * bar_len

        # Mark threshold position
        threshold_marker = ""
        if stats.threshold is not None:
            if bin_start <= stats.threshold < bin_end:
                threshold_marker = " <-- threshold"
            elif bin_start >= stats.threshold and i > 0:
                if stats.hist_bins[i - 1] < stats.threshold:
                    threshold_marker = " [rejected]"
            elif bin_start >= stats.threshold:
                threshold_marker = " [rejected]"

        lines.append(
            f"  {bin_start:4.0f}-{bin_end:4.0f}px: {bar:<{width}} {count:3d} ({pct:5.1f}%){threshold_marker}"
        )

    return lines


def format_stats_log(stats: RegistrationStats) -> list[str]:
    """
    Format registration stats for logging.

    Returns list of log lines including histogram and threshold decision.
    """
    lines = []

    # Histogram first for visual overview
    lines.extend(format_histogram(stats))
    lines.append("")  # blank line

    # Basic statistics
    lines.append(
        f"Stats: N={stats.n_images}, median={stats.median:.2f}px, "
        f"std={stats.std:.2f}, CV={stats.cv:.1%}, skew={stats.skewness:.2f}"
    )

    # Bimodality test results (if we have enough images)
    if stats.n_images >= 10:
        bimodal_str = "BIMODAL" if stats.is_bimodal else "unimodal"
        lines.append(
            f"Bimodality: dBIC={stats.delta_bic:.1f}, "
            f"dip_p={stats.dip_pvalue:.3f} -> {bimodal_str}"
        )

    # GMM mode details if bimodal
    if stats.is_bimodal and stats.gmm_means is not None:
        sorted_idx = np.argsort(stats.gmm_means)
        mode_strs = [
            f"{stats.gmm_means[i]:.2f}px (w={stats.gmm_weights[i]:.2f})"
            for i in sorted_idx
        ]
        lines.append(f"GMM modes: {', '.join(mode_strs)}")

    # Threshold decision
    case_descriptions = {
        "bimodal": "Two populations detected",
        "skewed": "Long tail detected",
        "broad": "Broad symmetric distribution",
        "tight": "Tight distribution, keeping all",
        "insufficient": "Too few images for filtering",
    }
    case_desc = case_descriptions.get(stats.filter_case, stats.filter_case)

    if stats.threshold is not None:
        lines.append(
            f"Decision: {case_desc} -> threshold={stats.threshold:.2f}px "
            f"({stats.threshold_reason})"
        )
        lines.append(f"Result: Rejecting {stats.n_rejected}/{stats.n_images} images")
    else:
        lines.append(
            f"Decision: {case_desc} -> no filtering ({stats.threshold_reason})"
        )

    return lines
