"""
Job runner - orchestrates the full processing pipeline.
"""

from pathlib import Path
from typing import Optional

from .calibration import CalibrationManager
from .composition import compose_and_stretch
from .job_config import load_job
from .job_validation import validate_job
from .logger import JobLogger, print_completion_summary
from .models import CalibrationDates, CompositionResult, FrameInfo, ValidationResult
from .preprocessing import preprocess_with_exposure_groups
from .protocols import SirilInterface


class JobRunner:
    """Orchestrates the full job processing pipeline."""

    def __init__(
        self,
        job_path: Path,
        base_path: Path,
        siril: Optional[SirilInterface] = None,
        dry_run: bool = False,
        force: bool = False,
    ):
        self.job_path = Path(job_path)
        self.base_path = Path(base_path)
        self.siril = siril
        self.dry_run = dry_run

        # Load job config
        self.config = load_job(self.job_path)

        # CLI --force overrides job file setting
        if force:
            from dataclasses import replace

            self.config.config = replace(self.config.config, force_reprocess=True)

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
            config=self.config.config,
            logger=self.logger,
        )

        # Cache for built calibration masters
        self._bias_master: Optional[Path] = None
        self._dark_masters: dict[tuple[float, float], Path] = {}
        self._flat_masters: dict[str, Path] = {}

    def _get_dark_temp(self, actual_temp: float) -> float:
        """Get temperature to use for dark matching (override if set)."""
        if self.config.config.dark_temp_override is not None:
            return self.config.config.dark_temp_override
        return actual_temp

    def validate(self) -> ValidationResult:
        """Validate the job - scan frames and check calibration."""
        return validate_job(
            job_name=self.config.name,
            lights=self.config.lights,
            base_path=self.base_path,
            cal_manager=self.cal_manager,
            config=self.config.config,
            logger=self.logger,
            get_dark_temp=self._get_dark_temp,
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
                and abs(cached_temp - dark_temp) <= self.config.config.temp_tolerance
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
            config=self.config.config,
            logger=self.logger,
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
            config=self.config.config,
            logger=self.logger,
        )

    def run(self) -> Optional[CompositionResult]:
        """Run the full pipeline."""
        self.logger.info(f"Starting job: {self.config.name}")
        self.logger.info(f"Type: {self.config.job_type}")
        self.logger.log_config(self.config.config)

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
                linear_pcc_path=result.linear_pcc_path,
            )

        return result

    def close(self):
        """Close the logger."""
        self.logger.close()
