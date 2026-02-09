"""
Preprocessing pipeline execution.

Contains the core pipeline steps for calibration, registration, and stacking.
"""

from pathlib import Path
from typing import Callable, Optional

from .config import Config, SubskyMethod
from .preprocessing_utils import create_sequence_file
from .protocols import SirilInterface
from .sequence_analysis import (
    compute_adaptive_threshold,
    find_valid_reference,
    format_stats_log,
    parse_sequence_file,
)


def compute_fwhm_threshold(
    siril: SirilInterface,
    process_dir: Path,
    seq_name: str,
    config: Config,
    log_fn: Optional[Callable[[str], None]] = None,
    log_detail_fn: Optional[Callable[[str], None]] = None,
) -> Optional[float]:
    """
    Compute adaptive FWHM threshold from registration data.

    Parses the .seq file after registration and uses GMM + dip test
    to detect bimodality and compute appropriate threshold.

    Also checks if the reference image would be filtered out and
    sets a new reference if needed.

    Returns:
        FWHM threshold in pixels, or None if no filtering needed
    """
    seq_path = process_dir / f"{seq_name}.seq"

    stats = parse_sequence_file(seq_path)
    if stats is None:
        if log_detail_fn:
            log_detail_fn("Could not parse sequence file for FWHM analysis")
        return None

    stats = compute_adaptive_threshold(stats, config)

    # Log the analysis
    if log_detail_fn:
        for line in format_stats_log(stats):
            log_detail_fn(line)

    # Check if reference image would be filtered out
    if stats.threshold is not None:
        new_ref = find_valid_reference(stats)
        if new_ref is not None:
            if log_fn:
                log_fn(
                    f"Reference image (wFWHM={stats.reference_wfwhm:.2f}px) "
                    f"exceeds threshold, switching to image {new_ref}"
                )
            if not siril.setref(seq_name, new_ref) and log_fn:
                log_fn("Warning: Failed to set new reference image")

    return stats.threshold


def run_pipeline(
    siril: SirilInterface,
    num_frames: int,
    process_dir: Path,
    stacks_dir: Path,
    stack_name: str,
    bias_master: Path,
    dark_master: Path,
    flat_master: Path,
    config: Config,
    log_fn: Optional[Callable[[str], None]] = None,
    log_detail_fn: Optional[Callable[[str], None]] = None,
) -> Path:
    """
    Run the preprocessing pipeline.

    Args:
        siril: Siril interface
        num_frames: Number of frames to process
        process_dir: Working directory for processing
        stacks_dir: Output directory for stacked result
        stack_name: Name for the output stack
        bias_master: Path to bias master
        dark_master: Path to dark master
        flat_master: Path to flat master
        config: Configuration
        log_fn: Logging function for main steps
        log_detail_fn: Logging function for details

    Returns:
        Path to the stacked result
    """
    # Validate required calibration files
    # Flat is always required
    if not flat_master.exists():
        raise FileNotFoundError(f"Calibration master not found: flat at {flat_master}")
    # Need either dark (preferred) or bias
    if dark_master and dark_master.exists():
        pass  # Dark contains bias, we're good
    elif bias_master and bias_master.exists():
        pass  # No dark, but have bias
    else:
        raise FileNotFoundError(
            f"Need either dark or bias master. dark={dark_master}, bias={bias_master}"
        )

    seq_path = process_dir / "light.seq"
    if log_fn:
        log_fn("Creating sequence file...")
    create_sequence_file(seq_path, num_frames, "light")

    if log_fn:
        log_fn("Calibrating...")
    cfg = config
    if not siril.cd(str(process_dir)):
        raise RuntimeError(f"Failed to cd to process directory: {process_dir}")

    # Smart calibration: dark already contains bias, so don't pass both.
    # Only use bias separately if no dark is available.
    if dark_master and dark_master.exists():
        # Dark contains bias - use dark + flat only
        if not siril.calibrate(
            "light",
            dark=str(dark_master),
            flat=str(flat_master),
        ):
            raise RuntimeError("Failed to calibrate light sequence")
    else:
        # No dark - use bias + flat
        if not siril.calibrate(
            "light",
            bias=str(bias_master),
            flat=str(flat_master),
        ):
            raise RuntimeError("Failed to calibrate light sequence")

    # Pre-stack background extraction (seqsubsky on individual subs)
    # Handles gradients on individual subs before they get combined
    if cfg.pre_stack_subsky_method != SubskyMethod.NONE:
        method_desc = (
            "RBF"
            if cfg.pre_stack_subsky_method == SubskyMethod.RBF
            else f"polynomial degree {cfg.pre_stack_subsky_degree}"
        )
        if log_fn:
            log_fn(f"Background extraction (pre-stack, {method_desc})...")
        if not siril.seqsubsky(
            "pp_light",
            method=cfg.pre_stack_subsky_method,
            degree=cfg.pre_stack_subsky_degree,
            samples=cfg.pre_stack_subsky_samples,
            tolerance=cfg.pre_stack_subsky_tolerance,
            smooth=cfg.pre_stack_subsky_smooth,
        ):
            if log_fn:
                log_fn("Pre-stack background extraction failed, continuing without")
            seq_to_register = "pp_light"
        else:
            seq_to_register = "bkg_pp_light"
    else:
        if log_fn:
            log_fn("Pre-stack background extraction disabled")
        seq_to_register = "pp_light"

    if log_fn:
        log_fn("Registering (2-pass)...")
    if not siril.register(seq_to_register, twopass=True):
        raise RuntimeError(f"Failed to register {seq_to_register} sequence")

    # Analyze FWHM distribution and compute adaptive threshold
    fwhm_threshold = compute_fwhm_threshold(
        siril, process_dir, seq_to_register, config, log_fn, log_detail_fn
    )

    if log_fn:
        log_fn("Applying registration...")
    if not siril.seqapplyreg(seq_to_register, filter_fwhm=fwhm_threshold):
        raise RuntimeError(f"Failed to apply registration to {seq_to_register}")

    if log_fn:
        log_fn("Stacking...")
    stack_path = stacks_dir / f"{stack_name}.fit"
    registered_seq = f"r_{seq_to_register}"
    if not siril.stack(
        registered_seq,
        cfg.stack_rejection,
        cfg.stack_weighting,
        cfg.stack_sigma_low,
        cfg.stack_sigma_high,
        norm=cfg.stack_norm,
        fastnorm=True,
        out=str(stack_path),
    ):
        raise RuntimeError(f"Failed to stack {registered_seq} to {stack_path}")

    if not stack_path.exists():
        raise FileNotFoundError(f"Stack output not created: {stack_path}")

    if log_fn:
        log_fn(f"Complete -> {stack_path.name}")
    return stack_path
