"""
Calibration file management for Siril job processing.

Handles finding, building, and caching master calibration files.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import DEFAULTS, Config
from .fits_utils import temperatures_match
from .logger import JobLogger
from .protocols import SirilInterface


@dataclass
class CalibrationDates:
    """Dates for each type of calibration data."""

    bias: str
    darks: str
    flats: str


@dataclass
class CalibrationStatus:
    """Status of a calibration file."""

    exists: bool
    can_build: bool
    master_path: Optional[Path]
    raw_path: Optional[Path]
    message: str


class CalibrationManager:
    """Manages calibration file finding and building."""

    def __init__(
        self,
        base_path: Path,
        dates: CalibrationDates,
        config: Config = DEFAULTS,
        logger: Optional[JobLogger] = None,
    ):
        self.base_path = Path(base_path)
        self.dates = dates
        self.config = config
        self.logger = logger
        self.calibration_dir = self.base_path / config.calibration_base_dir
        self.masters_dir = self.calibration_dir / config.calibration_masters_dir
        self.raw_dir = self.calibration_dir / config.calibration_raw_dir

    def _log(self, message: str) -> None:
        if self.logger:
            self.logger.substep(message)

    # Path resolution

    def get_bias_master_path(self) -> Path:
        """Get expected path for bias master."""
        cfg = self.config
        filename = f"{cfg.bias_prefix}{self.dates.bias}{cfg.fit_extension}"
        return self.masters_dir / cfg.bias_subdir / filename

    def get_dark_master_path(self, exposure: float, temp: float) -> Path:
        """Get expected path for dark master."""
        cfg = self.config
        exp_str = f"{int(exposure)}{cfg.exposure_suffix}"
        temp_str = f"{int(round(temp))}{cfg.temp_suffix}"
        filename = f"{cfg.dark_prefix}{exp_str}_{temp_str}_{self.dates.darks}{cfg.fit_extension}"
        return self.masters_dir / cfg.dark_subdir / filename

    def get_flat_master_path(self, filter_name: str) -> Path:
        """Get expected path for flat master."""
        cfg = self.config
        filename = (
            f"{cfg.flat_prefix}{filter_name}_{self.dates.flats}{cfg.fit_extension}"
        )
        return self.masters_dir / cfg.flat_subdir / filename

    def get_bias_raw_path(self) -> Path:
        """Get expected path for raw bias frames."""
        return self.raw_dir / self.config.bias_subdir / self.dates.bias

    def get_dark_raw_path(self, exposure: float, temp: float) -> Path:
        """Get expected path for raw dark frames."""
        cfg = self.config
        exp_str = f"{int(exposure)}"
        temp_str = f"{int(round(temp))}{cfg.temp_suffix}"
        # Structure: darks/{date}_{temp}/{exposure}/
        return (
            self.raw_dir / cfg.dark_subdir / f"{self.dates.darks}_{temp_str}" / exp_str
        )

    def get_flat_raw_path(self, filter_name: str) -> Path:
        """Get expected path for raw flat frames."""
        return self.raw_dir / self.config.flat_subdir / self.dates.flats / filter_name

    # Status checking

    def check_bias(self) -> CalibrationStatus:
        """Check if bias master exists or can be built."""
        master_path = self.get_bias_master_path()
        raw_path = self.get_bias_raw_path()

        if master_path.exists():
            return CalibrationStatus(
                exists=True,
                can_build=True,
                master_path=master_path,
                raw_path=None,
                message="Master exists",
            )

        if raw_path.exists() and any(raw_path.glob(self.config.fit_glob)):
            return CalibrationStatus(
                exists=False,
                can_build=True,
                master_path=master_path,
                raw_path=raw_path,
                message="Can build from raw",
            )

        return CalibrationStatus(
            exists=False,
            can_build=False,
            master_path=master_path,
            raw_path=raw_path,
            message=f"No master or raw frames at {raw_path}",
        )

    def check_dark(self, exposure: float, temp: float) -> CalibrationStatus:
        """Check if dark master exists or can be built (with temperature tolerance)."""
        master_path = self.get_dark_master_path(exposure, temp)
        raw_path = self.get_dark_raw_path(exposure, temp)

        if master_path.exists():
            return CalibrationStatus(
                exists=True,
                can_build=True,
                master_path=master_path,
                raw_path=None,
                message="Master exists",
            )

        if raw_path.exists() and any(raw_path.glob(self.config.fit_glob)):
            return CalibrationStatus(
                exists=False,
                can_build=True,
                master_path=master_path,
                raw_path=raw_path,
                message="Can build from raw",
            )

        # Try temperature tolerance matching for existing masters
        matching_master = self.find_matching_dark(exposure, temp)
        if matching_master:
            return CalibrationStatus(
                exists=True,
                can_build=True,
                master_path=matching_master,
                raw_path=None,
                message=f"Using tolerance-matched master: {matching_master.name}",
            )

        # Try tolerance matching for raw frames
        matching_raw = self._find_matching_dark_raw(exposure, temp)
        if matching_raw:
            return CalibrationStatus(
                exists=False,
                can_build=True,
                master_path=master_path,
                raw_path=matching_raw,
                message=f"Can build from tolerance-matched raw: {matching_raw}",
            )

        return CalibrationStatus(
            exists=False,
            can_build=False,
            master_path=master_path,
            raw_path=raw_path,
            message=f"No master or raw frames at {raw_path}",
        )

    def _find_matching_dark_raw(self, exposure: float, temp: float) -> Optional[Path]:
        """Find raw dark frames within temperature tolerance."""
        cfg = self.config
        darks_raw_dir = self.raw_dir / cfg.dark_subdir
        if not darks_raw_dir.exists():
            return None

        exp_str = f"{int(exposure)}"
        for temp_dir in darks_raw_dir.iterdir():
            if not temp_dir.is_dir():
                continue
            # Parse temp from dir name like "2025_01_23_-10C"
            parts = temp_dir.name.split("_")
            if len(parts) >= 2:
                temp_part = parts[-1]  # Last part should be temp like "-10C"
                try:
                    dir_temp = float(temp_part.replace(cfg.temp_suffix, ""))
                    if temperatures_match(temp, dir_temp, cfg.temp_tolerance):
                        exp_path = temp_dir / exp_str
                        if exp_path.exists() and any(exp_path.glob(cfg.fit_glob)):
                            return exp_path
                except ValueError:
                    continue
        return None

    def check_flat(self, filter_name: str) -> CalibrationStatus:
        """Check if flat master exists or can be built."""
        master_path = self.get_flat_master_path(filter_name)
        raw_path = self.get_flat_raw_path(filter_name)

        if master_path.exists():
            return CalibrationStatus(
                exists=True,
                can_build=True,
                master_path=master_path,
                raw_path=None,
                message="Master exists",
            )

        if raw_path.exists() and any(raw_path.glob(self.config.fit_glob)):
            return CalibrationStatus(
                exists=False,
                can_build=True,
                master_path=master_path,
                raw_path=raw_path,
                message="Can build from raw",
            )

        return CalibrationStatus(
            exists=False,
            can_build=False,
            master_path=master_path,
            raw_path=raw_path,
            message=f"No master or raw frames at {raw_path}",
        )

    # Building masters

    def build_bias_master(self, siril: SirilInterface) -> Path:
        """Build bias master from raw frames."""
        status = self.check_bias()
        if status.exists:
            return status.master_path

        if not status.can_build:
            raise ValueError(status.message)

        self._log(f"Building bias master: {status.master_path.name}")

        status.master_path.parent.mkdir(parents=True, exist_ok=True)

        raw_files = list(status.raw_path.glob(self.config.fit_glob))
        if not raw_files:
            raise FileNotFoundError(f"No bias frames found in {status.raw_path}")

        cfg = self.config
        seq_name = "bias"
        if not siril.cd(str(status.raw_path)):
            raise RuntimeError(f"Failed to cd to bias raw path: {status.raw_path}")
        if not siril.convert(seq_name, out=cfg.process_dir):
            raise RuntimeError(f"Failed to convert bias frames in {status.raw_path}")
        if not siril.cd(str(status.raw_path / "process")):
            raise RuntimeError("Failed to cd to bias process path")
        if not siril.stack(
            seq_name,
            cfg.cal_rejection,
            cfg.cal_sigma,
            cfg.cal_sigma,
            cfg.cal_no_norm,
            out=str(status.master_path),
        ):
            raise RuntimeError("Failed to stack bias frames")

        if not status.master_path.exists():
            raise FileNotFoundError(f"Bias master not created: {status.master_path}")

        return status.master_path

    def build_dark_master(
        self, exposure: float, temp: float, siril: SirilInterface
    ) -> Path:
        """Build dark master from raw frames."""
        status = self.check_dark(exposure, temp)
        if status.exists:
            return status.master_path

        if not status.can_build:
            raise ValueError(status.message)

        self._log(f"Building dark master: {status.master_path.name}")

        status.master_path.parent.mkdir(parents=True, exist_ok=True)

        raw_files = list(status.raw_path.glob(self.config.fit_glob))
        if not raw_files:
            raise FileNotFoundError(f"No dark frames found in {status.raw_path}")

        cfg = self.config
        seq_name = "dark"
        if not siril.cd(str(status.raw_path)):
            raise RuntimeError(f"Failed to cd to dark raw path: {status.raw_path}")
        if not siril.convert(seq_name, out=cfg.process_dir):
            raise RuntimeError(f"Failed to convert dark frames in {status.raw_path}")
        if not siril.cd(str(status.raw_path / "process")):
            raise RuntimeError("Failed to cd to dark process path")
        if not siril.stack(
            seq_name,
            cfg.cal_rejection,
            cfg.cal_sigma,
            cfg.cal_sigma,
            cfg.cal_no_norm,
            out=str(status.master_path),
        ):
            raise RuntimeError("Failed to stack dark frames")

        if not status.master_path.exists():
            raise FileNotFoundError(f"Dark master not created: {status.master_path}")

        return status.master_path

    def build_flat_master(
        self, filter_name: str, bias_path: Path, siril: SirilInterface
    ) -> Path:
        """Build flat master from raw frames (calibrated with bias)."""
        status = self.check_flat(filter_name)
        if status.exists:
            return status.master_path

        if not status.can_build:
            raise ValueError(status.message)

        self._log(f"Building flat master: {status.master_path.name}")

        status.master_path.parent.mkdir(parents=True, exist_ok=True)

        raw_files = list(status.raw_path.glob(self.config.fit_glob))
        if not raw_files:
            raise FileNotFoundError(f"No flat frames found in {status.raw_path}")
        if not bias_path.exists():
            raise FileNotFoundError(f"Bias master not found: {bias_path}")

        cfg = self.config
        if not siril.cd(str(status.raw_path)):
            raise RuntimeError(f"Failed to cd to flat raw path: {status.raw_path}")
        if not siril.convert(filter_name, out=cfg.process_dir):
            raise RuntimeError(f"Failed to convert flat frames in {status.raw_path}")
        if not siril.cd(str(status.raw_path / "process")):
            raise RuntimeError("Failed to cd to flat process path")
        if not siril.calibrate(filter_name, bias=str(bias_path)):
            raise RuntimeError("Failed to calibrate flat frames with bias")
        calibrated_seq = f"{cfg.calibrated_prefix}{filter_name}"
        if not siril.stack(
            calibrated_seq,
            cfg.cal_rejection,
            cfg.cal_sigma,
            cfg.cal_sigma,
            cfg.cal_flat_norm,
            out=str(status.master_path),
        ):
            raise RuntimeError("Failed to stack flat frames")

        if not status.master_path.exists():
            raise FileNotFoundError(f"Flat master not created: {status.master_path}")

        return status.master_path

    def find_matching_dark(self, exposure: float, temp: float) -> Optional[Path]:
        """
        Find a dark master matching exposure and temperature (with tolerance).

        Returns the master path if found, None otherwise.
        """
        master_path = self.get_dark_master_path(exposure, temp)
        if master_path.exists():
            return master_path

        cfg = self.config
        darks_dir = self.masters_dir / cfg.dark_subdir
        if not darks_dir.exists():
            return None

        exp_str = f"{int(exposure)}{cfg.exposure_suffix}"
        glob_pattern = (
            f"{cfg.dark_prefix}{exp_str}_*_{self.dates.darks}{cfg.fit_extension}"
        )
        for master in darks_dir.glob(glob_pattern):
            parts = master.stem.split("_")
            if len(parts) >= 3:
                temp_part = parts[2]  # e.g., "-10C"
                try:
                    master_temp = float(temp_part.replace(cfg.temp_suffix, ""))
                    if temperatures_match(temp, master_temp, cfg.temp_tolerance):
                        return master
                except ValueError:
                    continue

        return None
