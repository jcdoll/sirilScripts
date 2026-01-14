"""
Shared data models for Siril job processing.

All dataclasses are defined here to prevent circular imports
and centralize data structure definitions.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Stack-related models


@dataclass
class StackInfo:
    """Information about a discovered stack file."""

    path: Path
    filter_name: str
    exposure: int  # seconds

    @property
    def name(self) -> str:
        """Stack name without extension."""
        return self.path.stem


# FITS frame models


@dataclass
class FrameInfo:
    """Information extracted from a FITS frame header."""

    path: Path
    exposure: float  # seconds
    temperature: float  # Celsius
    filter_name: str
    gain: Optional[int] = None

    @property
    def exposure_str(self) -> str:
        """Exposure as string for matching (e.g., '300s')."""
        return f"{int(self.exposure)}s"

    @property
    def temp_str(self) -> str:
        """Temperature as string for matching (e.g., '-10C')."""
        return f"{int(round(self.temperature))}C"


@dataclass
class ClippingInfo:
    """Information about clipping in an image."""

    path: Path
    total_pixels: int
    clipped_low: int  # Pixels clipped to black (near 0)
    clipped_high: int  # Pixels clipped to white (near max)
    bit_depth: int

    @property
    def clipped_low_percent(self) -> float:
        """Percentage of pixels clipped to black."""
        if self.total_pixels == 0:
            return 0.0
        return 100.0 * self.clipped_low / self.total_pixels

    @property
    def clipped_high_percent(self) -> float:
        """Percentage of pixels clipped to white."""
        if self.total_pixels == 0:
            return 0.0
        return 100.0 * self.clipped_high / self.total_pixels


# Preprocessing models


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


# Calibration models


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


# Composition models


@dataclass
class CompositionResult:
    """Result of composition stage."""

    linear_path: Path  # Unstretched composed image (for VeraLux)
    linear_pcc_path: Optional[Path]  # Color-calibrated linear (if PCC succeeded)
    auto_fit: Path  # Auto-stretched .fit
    auto_tif: Path  # Auto-stretched .tif
    auto_jpg: Path  # Auto-stretched .jpg
    stacks_dir: Path  # Directory containing linear stacks


# Job validation models


@dataclass
class ValidationResult:
    """Result of job validation."""

    valid: bool
    frames: list[FrameInfo]
    requirements: list
    missing_calibration: list[str]
    buildable_calibration: list[str]
    message: str
