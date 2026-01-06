"""
Preprocessing module for Siril job processing.

Handles per-channel preprocessing: convert, calibrate, subsky, register, stack.
Supports stacking by exposure for HDR workflows.
"""

import os
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .fits_utils import FrameInfo
from .logger import JobLogger
from .protocols import SirilInterface


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


@dataclass
class StackGroup:
    """A group of frames to stack together (same filter + exposure)."""

    filter_name: str
    exposure: float
    frames: list[FrameInfo]

    @property
    def exposure_str(self) -> str:
        return f"{int(self.exposure)}s"

    @property
    def stack_name(self) -> str:
        """Name for the output stack file."""
        return f"stack_{self.filter_name}_{self.exposure_str}"


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
        logger: Optional[JobLogger] = None,
        fwhm_filter: float = 1.8,
    ):
        self.siril = siril
        self.logger = logger
        self.fwhm_filter = fwhm_filter

    def _log(self, message: str) -> None:
        if self.logger:
            self.logger.substep(message)

    def _log_detail(self, message: str) -> None:
        if self.logger:
            self.logger.detail(message)

    def _clean_process_dir(self, process_dir: Path) -> None:
        """Remove old Siril output files from process directory, preserving source/."""
        # Patterns for Siril-generated files
        patterns = ["*.fits", "*.fit", "*.seq", "*.csv"]
        removed = 0
        for pattern in patterns:
            for f in process_dir.glob(pattern):
                # Only remove files directly in process_dir, not in subdirs
                if f.parent == process_dir:
                    f.unlink()
                    removed += 1

        # Remove any leftover subdirectories except source/
        for d in process_dir.iterdir():
            if d.is_dir() and d.name != "source":
                shutil.rmtree(d)
                removed += 1

        if removed > 0:
            self._log_detail(
                f"Cleaned {removed} old files/dirs from {process_dir.name}"
            )

    def process_stack_group(
        self,
        group: StackGroup,
        output_dir: Path,
        bias_master: Path,
        dark_master: Path,
        flat_master: Path,
    ) -> Path:
        """
        Process a single stack group (filter + exposure).

        Returns path to the stacked result.
        """
        if self.logger:
            self.logger.step(
                f"Preprocessing {group.filter_name} @ {group.exposure_str} "
                f"({len(group.frames)} frames)"
            )

        # Create working directories
        process_dir = (
            output_dir / "process" / f"{group.filter_name}_{group.exposure_str}"
        )
        stacks_dir = output_dir / "stacks"
        process_dir.mkdir(parents=True, exist_ok=True)
        stacks_dir.mkdir(parents=True, exist_ok=True)

        # Clean any leftover files from previous runs
        self._clean_process_dir(process_dir)

        # Link frames to working directory
        num_frames = self._prepare_frames(group, process_dir)

        # Run preprocessing pipeline
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

            # Use 5-digit sequential naming to match .seq format (fixed_len=5)
            # No underscore between name and number: light00001.fit
            dest = process_dir / f"light{i:05d}.fit"
            if not dest.exists():
                link_or_copy(src, dest)

            # Verify link/copy succeeded
            if not dest.exists():
                raise IOError(f"Failed to link/copy frame to: {dest}")
            linked_count += 1

        # Verify we have files in the process directory
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

        # Validate calibration files exist
        for name, path in [
            ("bias", bias_master),
            ("dark", dark_master),
            ("flat", flat_master),
        ]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Calibration master not found: {name} at {path}"
                )

        # Step 1: Create sequence file (replaces Siril's convert command)
        # Files are already named light_00001.fit, light_00002.fit, etc.
        seq_path = process_dir / "light.seq"
        self._log("Creating sequence file...")
        create_sequence_file(seq_path, num_frames, "light")

        # Step 2: Calibrate
        self._log("Calibrating...")
        if not self.siril.cd(str(process_dir)):
            raise RuntimeError(f"Failed to cd to process directory: {process_dir}")
        if not self.siril.calibrate(
            "light",
            bias=str(bias_master),
            dark=str(dark_master),
            flat=str(flat_master),
        ):
            raise RuntimeError("Failed to calibrate light sequence")

        # Step 3: Background extraction
        self._log("Background extraction...")
        if not self.siril.seqsubsky("pp_light", 1):
            raise RuntimeError("Failed to run background extraction on pp_light")

        # Step 4: Registration
        self._log("Registering (2-pass)...")
        if not self.siril.register("bkg_pp_light", twopass=True):
            raise RuntimeError("Failed to register bkg_pp_light sequence")

        # Step 5: Apply registration with FWHM filter
        self._log("Applying registration...")
        if not self.siril.seqapplyreg(
            "bkg_pp_light",
            filter_fwhm=f"{self.fwhm_filter}k",
        ):
            raise RuntimeError("Failed to apply registration to bkg_pp_light")

        # Step 6: Stack
        self._log("Stacking...")
        stack_path = stacks_dir / f"{stack_name}.fit"
        if not self.siril.stack(
            "r_bkg_pp_light",
            "rej",
            "w",
            "3",
            "3",
            norm="addscale",
            fastnorm=True,
            out=str(stack_path),
        ):
            raise RuntimeError(f"Failed to stack r_bkg_pp_light to {stack_path}")

        # Verify output exists
        if not stack_path.exists():
            raise FileNotFoundError(f"Stack output not created: {stack_path}")

        self._log(f"Complete -> {stack_path.name}")
        return stack_path


def preprocess_with_exposure_groups(
    siril: SirilInterface,
    frames: list[FrameInfo],
    output_dir: Path,
    get_calibration: callable,
    logger: Optional[JobLogger] = None,
    fwhm_filter: float = 1.8,
) -> dict[str, Path]:
    """
    Preprocess frames, grouping by filter and exposure.

    Args:
        siril: Siril interface
        frames: List of FrameInfo (already scanned)
        output_dir: Output directory
        get_calibration: Callable(filter, exposure, temp) -> (bias, dark, flat) paths
        logger: Optional logger
        fwhm_filter: FWHM filter factor

    Returns:
        Dict of stack_name -> stacked result path
        e.g., {"stack_L_180s": Path(...), "stack_L_30s": Path(...)}
    """
    preprocessor = Preprocessor(siril, logger, fwhm_filter)

    # Group frames by filter + exposure
    groups = group_frames_by_filter_exposure(frames)

    if logger:
        logger.step(f"Processing {len(groups)} stack groups")

    results = {}
    for group in groups:
        # Get representative temperature (use first frame)
        temp = group.frames[0].temperature if group.frames else 0.0

        # Get calibration paths for this group
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
        )
        results[group.stack_name] = stack_path

    return results


# Keep old function for backwards compatibility but mark deprecated
def preprocess_all_filters(
    siril: SirilInterface,
    filters_config: dict[str, list[Path]],
    output_dir: Path,
    calibration_paths: dict[str, dict[str, Path]],
    logger: Optional[JobLogger] = None,
    fwhm_filter: float = 1.8,
) -> dict[str, Path]:
    """
    DEPRECATED: Use preprocess_with_exposure_groups instead.

    This function doesn't support HDR/multi-exposure workflows.
    """
    if logger:
        logger.warning("Using deprecated preprocess_all_filters - HDR not supported")

    # Import here to avoid circular import
    from .fits_utils import scan_multiple_directories

    # Scan frames
    all_frames = []
    for _filter_name, light_dirs in filters_config.items():
        frames = scan_multiple_directories([Path(d) for d in light_dirs])
        all_frames.extend(frames)

    def get_cal(filter_name, exposure, temp):
        cal = calibration_paths.get(filter_name, {})
        return cal.get("bias"), cal.get("dark"), cal.get("flat")

    return preprocess_with_exposure_groups(
        siril=siril,
        frames=all_frames,
        output_dir=output_dir,
        get_calibration=get_cal,
        logger=logger,
        fwhm_filter=fwhm_filter,
    )
