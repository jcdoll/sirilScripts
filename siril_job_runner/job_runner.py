"""
Job runner - orchestrates the full processing pipeline.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .calibration import CalibrationDates, CalibrationManager
from .composition import CompositionResult, compose_and_stretch
from .fits_utils import FrameInfo, scan_multiple_directories
from .frame_analysis import (
    build_date_summary_table,
    build_requirements_table,
    format_date_summary_table,
    get_unique_filters,
)
from .job_config import load_job
from .logger import JobLogger, print_completion_summary
from .preprocessing import preprocess_with_exposure_groups
from .protocols import SirilInterface


@dataclass
class ValidationResult:
    """Result of job validation."""

    valid: bool
    frames: list[FrameInfo]
    requirements: list
    missing_calibration: list[str]
    buildable_calibration: list[str]
    message: str


class JobRunner:
    """Orchestrates the full job processing pipeline."""

    def __init__(
        self,
        job_path: Path,
        base_path: Path,
        siril: Optional[SirilInterface] = None,
        dry_run: bool = False,
    ):
        self.job_path = Path(job_path)
        self.base_path = Path(base_path)
        self.siril = siril
        self.dry_run = dry_run

        # Load job config
        self.config = load_job(self.job_path)

        # Set up paths
        self.output_dir = self.base_path / self.config.output
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up logger
        self.logger = JobLogger(self.output_dir, self.config.name)

        # Set up calibration manager
        self.cal_manager = CalibrationManager(
            base_path=self.base_path,
            dates=CalibrationDates(
                bias=self.config.calibration_bias,
                darks=self.config.calibration_darks,
                flats=self.config.calibration_flats,
            ),
            temp_tolerance=self.config.options.temp_tolerance,
            logger=self.logger,
        )

        # Cache for built calibration masters
        self._bias_master: Optional[Path] = None
        self._dark_masters: dict[tuple[float, float], Path] = {}
        self._flat_masters: dict[str, Path] = {}

    def _get_dark_temp(self, actual_temp: float) -> float:
        """Get temperature to use for dark matching (override if set)."""
        if self.config.options.dark_temp_override is not None:
            return self.config.options.dark_temp_override
        return actual_temp

    def validate(self) -> ValidationResult:
        """
        Validate the job - scan frames and check calibration.

        Returns ValidationResult with details.
        """
        self.logger.step(f"Validating job: {self.config.name}")

        # Scan all light frames
        all_frames = []
        for filter_name, dirs in self.config.lights.items():
            full_dirs = [self.base_path / d for d in dirs]
            frames = scan_multiple_directories(full_dirs)
            all_frames.extend(frames)
            self.logger.substep(
                f"{filter_name}: {len(frames)} frames from {len(dirs)} dirs"
            )

        if not all_frames:
            return ValidationResult(
                valid=False,
                frames=[],
                requirements=[],
                missing_calibration=[],
                buildable_calibration=[],
                message="No light frames found",
            )

        # Check saturation levels by exposure
        from collections import defaultdict

        from .fits_utils import check_clipping

        by_exposure: dict[float, list] = defaultdict(list)
        for frame in all_frames:
            by_exposure[frame.exposure].append(frame)

        self.logger.step("Clipping check:")
        for exp in sorted(by_exposure.keys(), reverse=True):
            frames_at_exp = by_exposure[exp]
            # Sample frames to check clipping
            sample = frames_at_exp[: min(5, len(frames_at_exp))]
            low_pcts = []
            high_pcts = []
            for frame in sample:
                info = check_clipping(frame.path)
                if info:
                    low_pcts.append(info.clipped_low_percent)
                    high_pcts.append(info.clipped_high_percent)
            if low_pcts and high_pcts:
                avg_low = sum(low_pcts) / len(low_pcts)
                avg_high = sum(high_pcts) / len(high_pcts)
                self.logger.substep(
                    f"{int(exp)}s: {avg_low:.3f}% black, {avg_high:.3f}% white ({len(frames_at_exp)} frames)"
                )

        # Show date summary table
        date_summary = build_date_summary_table(all_frames)
        all_filters = sorted(get_unique_filters(all_frames))
        # Reorder filters to put common ones first: L, R, G, B, H, S, O
        filter_order = ["L", "R", "G", "B", "H", "S", "O"]
        ordered_filters = [f for f in filter_order if f in all_filters]
        ordered_filters += [f for f in all_filters if f not in filter_order]

        self.logger.step("Frames by date:")
        for line in format_date_summary_table(date_summary, ordered_filters):
            self.logger.info(line)

        # Build requirements table
        requirements = build_requirements_table(all_frames)
        self.logger.step("Requirements:")
        for req in requirements:
            self.logger.substep(
                f"{req.filter_name}: {req.exposure_str} @ {req.temp_str} ({req.count} frames)"
            )

        # Check calibration availability
        missing = []
        buildable = []

        # Check bias (single, no temperature dependency)
        status = self.cal_manager.check_bias()
        if not status.exists and not status.can_build:
            missing.append("bias")
        elif not status.exists and status.can_build:
            buildable.append("bias")

        # Check darks for each exposure/temp combo (with override if set)
        dark_combos = {
            (req.exposure, self._get_dark_temp(req.temperature)) for req in requirements
        }
        for exp, temp in dark_combos:
            status = self.cal_manager.check_dark(exp, temp)
            if not status.exists and not status.can_build:
                missing.append(f"dark_{int(exp)}s_{int(temp)}C")
            elif not status.exists and status.can_build:
                buildable.append(f"dark_{int(exp)}s_{int(temp)}C")

        # Check flats for each filter
        filters = {req.filter_name for req in requirements}
        for filter_name in filters:
            status = self.cal_manager.check_flat(filter_name)
            if not status.exists and not status.can_build:
                missing.append(f"flat_{filter_name}")
            elif not status.exists and status.can_build:
                buildable.append(f"flat_{filter_name}")

        # Report status
        self.logger.step("Calibration status:")
        for name in buildable:
            self.logger.substep(f"[BUILD] {name}")
        for name in missing:
            self.logger.substep(f"[MISSING] {name}")

        valid = len(missing) == 0
        message = (
            "Validation passed"
            if valid
            else f"Missing {len(missing)} calibration files"
        )

        return ValidationResult(
            valid=valid,
            frames=all_frames,
            requirements=requirements,
            missing_calibration=missing,
            buildable_calibration=buildable,
            message=message,
        )

    def run_calibration(self, validation: ValidationResult) -> None:
        """
        Build any missing calibration masters.

        Caches results in instance variables for later lookup.
        """
        if self.dry_run:
            self.logger.step("[DRY RUN] Would build calibration masters")
            return

        self.logger.step("Building calibration masters...")

        # Get unique requirements (with temp override if set)
        dark_combos = {
            (req.exposure, self._get_dark_temp(req.temperature))
            for req in validation.requirements
        }
        filters = {req.filter_name for req in validation.requirements}

        # Build bias master (single, no temperature dependency)
        self._bias_master = self.cal_manager.build_bias_master(self.siril)

        # Build dark masters
        for exp, temp in dark_combos:
            path = self.cal_manager.build_dark_master(exp, temp, self.siril)
            self._dark_masters[(exp, temp)] = path

        # Build flat masters
        for filter_name in filters:
            path = self.cal_manager.build_flat_master(
                filter_name, self._bias_master, self.siril
            )
            self._flat_masters[filter_name] = path

    def get_calibration(
        self, filter_name: str, exposure: float, temp: float
    ) -> tuple[Optional[Path], Optional[Path], Optional[Path]]:
        """
        Get calibration paths for a specific filter/exposure/temp combo.

        Returns (bias_path, dark_path, flat_path).
        """
        # Bias is universal (no temperature dependency)
        bias = self._bias_master

        # Find dark with matching exposure (use override temp if set)
        dark_temp = self._get_dark_temp(temp)
        dark = None
        for (cached_exp, cached_temp), path in self._dark_masters.items():
            if (
                cached_exp == exposure
                and abs(cached_temp - dark_temp) <= self.config.options.temp_tolerance
            ):
                dark = path
                break

        # Get flat for filter
        flat = self._flat_masters.get(filter_name)

        return bias, dark, flat

    def run_preprocessing(self, frames: list[FrameInfo]) -> dict[str, Path]:
        """Run preprocessing for all frames, grouped by filter+exposure."""
        if self.dry_run:
            self.logger.step("[DRY RUN] Would preprocess all frames")
            return {}

        self.logger.step("Preprocessing...")

        return preprocess_with_exposure_groups(
            siril=self.siril,
            frames=frames,
            output_dir=self.output_dir,
            get_calibration=self.get_calibration,
            logger=self.logger,
            fwhm_filter=self.config.options.fwhm_filter,
        )

    def run_composition(self) -> CompositionResult:
        """
        Run composition and stretching.

        Discovers stacks from output_dir/stacks/ directory.
        HDR mode is handled automatically - multiple exposures per filter are
        blended using brightness-weighted HDR before composition.
        """
        if self.dry_run:
            self.logger.step("[DRY RUN] Would compose and stretch")
            return None

        return compose_and_stretch(
            siril=self.siril,
            output_dir=self.output_dir,
            job_type=self.config.job_type,
            palette=self.config.options.palette,
            hdr_low_threshold=self.config.options.hdr_low_threshold,
            hdr_high_threshold=self.config.options.hdr_high_threshold,
            logger=self.logger,
        )

    def run(self) -> Optional[CompositionResult]:
        """Run the full pipeline."""
        self.logger.info(f"Starting job: {self.config.name}")
        self.logger.info(f"Type: {self.config.job_type}")

        # Stage 0: Validation
        validation = self.validate()
        if not validation.valid:
            self.logger.error(validation.message)
            raise ValueError(validation.message)

        # Stage 1: Calibration
        self.run_calibration(validation)

        # Stage 2: Preprocessing (now uses frames from validation)
        self.run_preprocessing(validation.frames)

        # Stage 3 & 4: Composition and Stretching (discovers stacks from output dir)
        result = self.run_composition()

        self.logger.info(f"Job complete: {self.config.name}")

        # Print completion summary with user instructions
        if result:
            print_completion_summary(
                job_name=self.config.name,
                job_type=self.config.job_type,
                linear_path=result.linear_path,
                auto_fit=result.auto_fit,
                auto_tif=result.auto_tif,
                auto_jpg=result.auto_jpg,
                stacks_dir=result.stacks_dir,
            )

        return result

    def close(self):
        """Close the logger."""
        self.logger.close()
