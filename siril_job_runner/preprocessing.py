"""
Preprocessing module for Siril job processing.

Handles per-channel preprocessing: convert, calibrate, subsky, register, stack.
Supports stacking by exposure for HDR workflows.
"""

import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

from .config import DEFAULTS, Config
from .logger import JobLogger
from .models import FrameInfo, StackGroup
from .protocols import SirilInterface
from .sequence_analysis import (
    compute_adaptive_threshold,
    format_stats_log,
    parse_sequence_file,
)


def link_or_copy(src: Path, dest: Path) -> None:
    """Hard link if possible, otherwise copy."""
    try:
        os.link(src, dest)
    except OSError:
        # Cross-device link or unsupported filesystem, fall back to copy
        shutil.copy2(src, dest)


def create_sequence_file(seq_path: Path, num_images: int, seq_name: str) -> None:
    """
    Create a Siril .seq file directly.

    Format matches pysiril's CreateSeqFile output:
    - Header comments
    - S line: sequence metadata (fixed_len=5 for 5-digit numbering)
    - L line: layer count (-1 = auto)
    - I lines: one per image (index, included flag)
    """
    with open(seq_path, "w", newline="") as f:
        f.write(
            "#Siril sequence file. "
            "Contains list of files (images), selection, and registration data\n"
        )
        f.write(
            "#S 'sequence_name' start_index nb_images nb_selected "
            "fixed_len reference_image version\n"
        )
        # S 'name' start nb_images nb_selected fixed_len ref_image version
        # fixed_len=5 means 5-digit numbering (00001, 00002, etc.)
        f.write(f"S '{seq_name}' 1 {num_images} {num_images} 5 -1 1\n")
        f.write("L -1\n")
        for i in range(1, num_images + 1):
            f.write(f"I {i} 1\n")


def group_frames_by_filter_exposure(frames: list[FrameInfo]) -> list[StackGroup]:
    """
    Group frames by (filter, exposure) for separate stacking.

    Returns list of StackGroup, sorted by filter then exposure.
    """
    groups: dict[tuple[str, float], list[FrameInfo]] = defaultdict(list)

    for frame in frames:
        key = (frame.filter_name, frame.exposure)
        groups[key].append(frame)

    result = []
    for (filter_name, exposure), frame_list in sorted(groups.items()):
        result.append(
            StackGroup(
                filter_name=filter_name,
                exposure=exposure,
                frames=frame_list,
            )
        )

    return result


class Preprocessor:
    """Handles preprocessing of light frames."""

    def __init__(
        self,
        siril: SirilInterface,
        config: Config = DEFAULTS,
        logger: Optional[JobLogger] = None,
    ):
        self.siril = siril
        self.config = config
        self.logger = logger

    def _log(self, message: str) -> None:
        if self.logger:
            self.logger.substep(message)

    def _log_detail(self, message: str) -> None:
        if self.logger:
            self.logger.detail(message)

    def _compute_fwhm_threshold(
        self, process_dir: Path, seq_name: str
    ) -> Optional[float]:
        """
        Compute adaptive FWHM threshold from registration data.

        Parses the .seq file after registration and uses GMM + dip test
        to detect bimodality and compute appropriate threshold.

        Returns:
            FWHM threshold in pixels, or None if no filtering needed
        """
        seq_path = process_dir / f"{seq_name}.seq"

        stats = parse_sequence_file(seq_path)
        if stats is None:
            self._log_detail("Could not parse sequence file for FWHM analysis")
            return None

        stats = compute_adaptive_threshold(stats, self.config)

        # Log the analysis
        for line in format_stats_log(stats):
            self._log_detail(line)

        return stats.threshold

    def _clean_process_dir(self, process_dir: Path) -> None:
        """Remove old Siril output files from process directory, preserving source/."""
        patterns = ["*.fits", "*.fit", "*.seq", "*.csv"]
        removed = 0
        for pattern in patterns:
            for f in process_dir.glob(pattern):
                if f.parent == process_dir:
                    f.unlink()
                    removed += 1

        for d in process_dir.iterdir():
            if d.is_dir() and d.name != "source":
                shutil.rmtree(d)
                removed += 1

        if removed > 0:
            self._log_detail(
                f"Cleaned {removed} old files/dirs from {process_dir.name}"
            )

    def _is_stack_cached(
        self,
        stack_path: Path,
        frames: list,
        bias_master: Path,
        dark_master: Path,
        flat_master: Path,
    ) -> bool:
        """Check if stack exists and is newer than all sources."""
        if not stack_path.exists():
            return False

        stack_mtime = stack_path.stat().st_mtime

        # Check all source frames
        for frame in frames:
            if frame.path.stat().st_mtime > stack_mtime:
                return False

        # Check calibration masters
        for master in [bias_master, dark_master, flat_master]:
            if master.stat().st_mtime > stack_mtime:
                return False

        return True

    def process_stack_group(
        self,
        group: StackGroup,
        output_dir: Path,
        bias_master: Path,
        dark_master: Path,
        flat_master: Path,
        force: bool = False,
    ) -> Path:
        """
        Process a single stack group (filter + exposure).

        Args:
            force: If True, reprocess even if cached stack exists.

        Returns path to the stacked result.
        """
        stacks_dir = output_dir / "stacks"
        stacks_dir.mkdir(parents=True, exist_ok=True)
        stack_path = stacks_dir / f"{group.stack_name}.fit"

        # Check cache first
        if not force and self._is_stack_cached(
            stack_path, group.frames, bias_master, dark_master, flat_master
        ):
            if self.logger:
                self.logger.step(
                    f"Using cached {group.filter_name} @ {group.exposure_str} "
                    f"({len(group.frames)} frames)"
                )
            self._log(f"Stack is up-to-date: {stack_path.name}")
            return stack_path

        if self.logger:
            self.logger.step(
                f"Preprocessing {group.filter_name} @ {group.exposure_str} "
                f"({len(group.frames)} frames)"
            )

        process_dir = (
            output_dir / "process" / f"{group.filter_name}_{group.exposure_str}"
        )
        process_dir.mkdir(parents=True, exist_ok=True)

        self._clean_process_dir(process_dir)

        num_frames = self._prepare_frames(group, process_dir)

        stack_path = self._run_pipeline(
            num_frames=num_frames,
            process_dir=process_dir,
            stacks_dir=stacks_dir,
            stack_name=group.stack_name,
            bias_master=bias_master,
            dark_master=dark_master,
            flat_master=flat_master,
        )

        return stack_path

    def _prepare_frames(self, group: StackGroup, process_dir: Path) -> int:
        """Link frames to working directory with sequential naming."""
        self._log("Preparing frames...")

        linked_count = 0
        for i, frame in enumerate(group.frames, 1):
            src = frame.path
            if not src.exists():
                raise FileNotFoundError(f"Source frame not found: {src}")

            dest = process_dir / f"light{i:05d}.fit"
            if not dest.exists():
                link_or_copy(src, dest)

            if not dest.exists():
                raise IOError(f"Failed to link/copy frame to: {dest}")
            linked_count += 1

        fit_files = list(process_dir.glob("light*.fit"))
        if not fit_files:
            raise FileNotFoundError(f"No light frames found in {process_dir}")

        self._log_detail(
            f"Prepared {linked_count} frames ({len(fit_files)} files in {process_dir})"
        )
        return linked_count

    def _run_pipeline(
        self,
        num_frames: int,
        process_dir: Path,
        stacks_dir: Path,
        stack_name: str,
        bias_master: Path,
        dark_master: Path,
        flat_master: Path,
    ) -> Path:
        """Run the preprocessing pipeline."""
        for name, path in [
            ("bias", bias_master),
            ("dark", dark_master),
            ("flat", flat_master),
        ]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Calibration master not found: {name} at {path}"
                )

        seq_path = process_dir / "light.seq"
        self._log("Creating sequence file...")
        create_sequence_file(seq_path, num_frames, "light")

        self._log("Calibrating...")
        cfg = self.config
        if not self.siril.cd(str(process_dir)):
            raise RuntimeError(f"Failed to cd to process directory: {process_dir}")
        if not self.siril.calibrate(
            "light",
            bias=str(bias_master),
            dark=str(dark_master),
            flat=str(flat_master),
        ):
            raise RuntimeError("Failed to calibrate light sequence")

        self._log("Registering (2-pass)...")
        if not self.siril.register("pp_light", twopass=True):
            raise RuntimeError("Failed to register pp_light sequence")

        # Analyze FWHM distribution and compute adaptive threshold
        fwhm_threshold = self._compute_fwhm_threshold(process_dir, "pp_light")

        self._log("Applying registration...")
        if not self.siril.seqapplyreg("pp_light", filter_fwhm=fwhm_threshold):
            raise RuntimeError("Failed to apply registration to pp_light")

        self._log("Stacking...")
        stack_path = stacks_dir / f"{stack_name}.fit"
        if not self.siril.stack(
            "r_pp_light",
            cfg.stack_rejection,
            cfg.stack_weighting,
            cfg.stack_sigma_low,
            cfg.stack_sigma_high,
            norm=cfg.stack_norm,
            fastnorm=True,
            out=str(stack_path),
        ):
            raise RuntimeError(f"Failed to stack r_pp_light to {stack_path}")

        if not stack_path.exists():
            raise FileNotFoundError(f"Stack output not created: {stack_path}")

        # Background extraction on stacked image (optional)
        if cfg.subsky_enabled:
            self._log("Background extraction...")
            if not self.siril.load(str(stack_path)):
                raise RuntimeError(f"Failed to load stack: {stack_path}")
            if self.siril.subsky(
                rbf=cfg.subsky_rbf,
                degree=cfg.subsky_degree,
                samples=cfg.subsky_samples,
                tolerance=cfg.subsky_tolerance,
                smooth=cfg.subsky_smooth,
            ):
                if not self.siril.save(str(stack_path)):
                    raise RuntimeError(
                        f"Failed to save stack after subsky: {stack_path}"
                    )
            else:
                self._log("Background extraction failed, continuing without")
        else:
            self._log("Background extraction disabled")

        self._log(f"Complete -> {stack_path.name}")
        return stack_path


def preprocess_with_exposure_groups(
    siril: SirilInterface,
    frames: list[FrameInfo],
    output_dir: Path,
    get_calibration: Callable,
    config: Config = DEFAULTS,
    logger: Optional[JobLogger] = None,
) -> dict[str, Path]:
    """
    Preprocess frames, grouping by filter and exposure.

    Args:
        siril: Siril interface
        frames: List of FrameInfo (already scanned)
        output_dir: Output directory
        get_calibration: Callable(filter, exposure, temp) -> (bias, dark, flat) paths
        config: Configuration
        logger: Optional logger

    Returns:
        Dict of stack_name -> stacked result path
        e.g., {"stack_L_180s": Path(...), "stack_L_30s": Path(...)}
    """
    preprocessor = Preprocessor(siril, config, logger)

    groups = group_frames_by_filter_exposure(frames)

    if logger:
        logger.step(f"Processing {len(groups)} stack groups")

    results = {}
    for group in groups:
        temp = group.frames[0].temperature if group.frames else 0.0

        bias, dark, flat = get_calibration(group.filter_name, group.exposure, temp)

        if not all([bias, dark, flat]):
            if logger:
                logger.error(
                    f"Missing calibration for {group.filter_name} @ {group.exposure_str}"
                )
            continue

        stack_path = preprocessor.process_stack_group(
            group=group,
            output_dir=output_dir,
            bias_master=bias,
            dark_master=dark,
            flat_master=flat,
            force=config.force_reprocess,
        )
        results[group.stack_name] = stack_path

    return results
