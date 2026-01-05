"""
Calibration file management for Siril job processing.

Handles finding, building, and caching master calibration files.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
        temp_tolerance: float = 2.0,
        logger: Optional[JobLogger] = None,
    ):
        self.base_path = Path(base_path)
        self.dates = dates
        self.temp_tolerance = temp_tolerance
        self.logger = logger
        self.calibration_dir = self.base_path / "calibration"
        self.masters_dir = self.calibration_dir / "masters"
        self.raw_dir = self.calibration_dir / "raw"

    def _log(self, message: str) -> None:
        if self.logger:
            self.logger.substep(message)

    # Path resolution

    def get_bias_master_path(self) -> Path:
        """Get expected path for bias master."""
        return self.masters_dir / "biases" / f"bias_{self.dates.bias}.fit"

    def get_dark_master_path(self, exposure: float, temp: float) -> Path:
        """Get expected path for dark master."""
        exp_str = f"{int(exposure)}s"
        temp_str = f"{int(round(temp))}C"
        return self.masters_dir / "darks" / f"dark_{exp_str}_{temp_str}_{self.dates.darks}.fit"

    def get_flat_master_path(self, filter_name: str) -> Path:
        """Get expected path for flat master."""
        return self.masters_dir / "flats" / f"flat_{filter_name}_{self.dates.flats}.fit"

    def get_bias_raw_path(self) -> Path:
        """Get expected path for raw bias frames."""
        return self.raw_dir / "biases" / self.dates.bias

    def get_dark_raw_path(self, exposure: float, temp: float) -> Path:
        """Get expected path for raw dark frames."""
        exp_str = f"{int(exposure)}"
        temp_str = f"{int(round(temp))}C"
        # Structure: darks/{date}_{temp}/{exposure}/
        return self.raw_dir / "darks" / f"{self.dates.darks}_{temp_str}" / exp_str

    def get_flat_raw_path(self, filter_name: str) -> Path:
        """Get expected path for raw flat frames."""
        return self.raw_dir / "flats" / self.dates.flats / filter_name

    # Status checking

    def check_bias(self) -> CalibrationStatus:
        """Check if bias master exists or can be built."""
        master_path = self.get_bias_master_path()
        raw_path = self.get_bias_raw_path()

        if master_path.exists():
            return CalibrationStatus(
                exists=True, can_build=True, master_path=master_path,
                raw_path=None, message="Master exists"
            )

        if raw_path.exists() and any(raw_path.glob("*.fit*")):
            return CalibrationStatus(
                exists=False, can_build=True, master_path=master_path,
                raw_path=raw_path, message="Can build from raw"
            )

        return CalibrationStatus(
            exists=False, can_build=False, master_path=master_path,
            raw_path=raw_path, message=f"No master or raw frames at {raw_path}"
        )

    def check_dark(self, exposure: float, temp: float) -> CalibrationStatus:
        """Check if dark master exists or can be built (with temperature tolerance)."""
        # First try exact match
        master_path = self.get_dark_master_path(exposure, temp)
        raw_path = self.get_dark_raw_path(exposure, temp)

        if master_path.exists():
            return CalibrationStatus(
                exists=True, can_build=True, master_path=master_path,
                raw_path=None, message="Master exists"
            )

        if raw_path.exists() and any(raw_path.glob("*.fit*")):
            return CalibrationStatus(
                exists=False, can_build=True, master_path=master_path,
                raw_path=raw_path, message="Can build from raw"
            )

        # Try temperature tolerance matching for existing masters
        matching_master = self.find_matching_dark(exposure, temp)
        if matching_master:
            return CalibrationStatus(
                exists=True, can_build=True, master_path=matching_master,
                raw_path=None, message=f"Using tolerance-matched master: {matching_master.name}"
            )

        # Try tolerance matching for raw frames
        matching_raw = self._find_matching_dark_raw(exposure, temp)
        if matching_raw:
            return CalibrationStatus(
                exists=False, can_build=True, master_path=master_path,
                raw_path=matching_raw, message=f"Can build from tolerance-matched raw: {matching_raw}"
            )

        return CalibrationStatus(
            exists=False, can_build=False, master_path=master_path,
            raw_path=raw_path, message=f"No master or raw frames at {raw_path}"
        )

    def _find_matching_dark_raw(self, exposure: float, temp: float) -> Optional[Path]:
        """Find raw dark frames within temperature tolerance."""
        darks_raw_dir = self.raw_dir / "darks"
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
                    dir_temp = float(temp_part.replace("C", ""))
                    if temperatures_match(temp, dir_temp, self.temp_tolerance):
                        exp_path = temp_dir / exp_str
                        if exp_path.exists() and any(exp_path.glob("*.fit*")):
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
                exists=True, can_build=True, master_path=master_path,
                raw_path=None, message="Master exists"
            )

        if raw_path.exists() and any(raw_path.glob("*.fit*")):
            return CalibrationStatus(
                exists=False, can_build=True, master_path=master_path,
                raw_path=raw_path, message="Can build from raw"
            )

        return CalibrationStatus(
            exists=False, can_build=False, master_path=master_path,
            raw_path=raw_path, message=f"No master or raw frames at {raw_path}"
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

        # Ensure output directory exists
        status.master_path.parent.mkdir(parents=True, exist_ok=True)

        # Build using siril commands
        siril.cd(str(status.raw_path))
        siril.convert("bias", out="./process")
        siril.cd(str(status.raw_path / "process"))
        siril.stack("bias", "rej", "3", "3", "-nonorm", out=str(status.master_path))

        return status.master_path

    def build_dark_master(self, exposure: float, temp: float, siril: SirilInterface) -> Path:
        """Build dark master from raw frames."""
        status = self.check_dark(exposure, temp)
        if status.exists:
            return status.master_path

        if not status.can_build:
            raise ValueError(status.message)

        self._log(f"Building dark master: {status.master_path.name}")

        status.master_path.parent.mkdir(parents=True, exist_ok=True)

        siril.cd(str(status.raw_path))
        siril.convert("dark", out="./process")
        siril.cd(str(status.raw_path / "process"))
        siril.stack("dark", "rej", "3", "3", "-nonorm", out=str(status.master_path))

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

        siril.cd(str(status.raw_path))
        siril.convert(filter_name, out="./process")
        siril.cd(str(status.raw_path / "process"))
        siril.calibrate(filter_name, bias=str(bias_path))
        siril.stack(f"pp_{filter_name}", "rej", "3", "3", "-norm=mul", out=str(status.master_path))

        return status.master_path

    def find_matching_dark(self, exposure: float, temp: float) -> Optional[Path]:
        """
        Find a dark master matching exposure and temperature (with tolerance).

        Returns the master path if found, None otherwise.
        """
        # First try exact match
        master_path = self.get_dark_master_path(exposure, temp)
        if master_path.exists():
            return master_path

        # Try temperature tolerance matching
        darks_dir = self.masters_dir / "darks"
        if not darks_dir.exists():
            return None

        exp_str = f"{int(exposure)}s"
        for master in darks_dir.glob(f"dark_{exp_str}_*_{self.dates.darks}.fit"):
            # Parse temperature from filename
            parts = master.stem.split("_")
            if len(parts) >= 3:
                temp_part = parts[2]  # e.g., "-10C"
                try:
                    master_temp = float(temp_part.replace("C", ""))
                    if temperatures_match(temp, master_temp, self.temp_tolerance):
                        return master
                except ValueError:
                    continue

        return None
