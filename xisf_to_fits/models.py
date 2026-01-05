"""Data models for XISF to FITS conversion."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ConversionResult:
    """Result of a single file conversion."""
    input_path: Path
    output_path: Optional[Path] = None
    success: bool = False
    skipped: bool = False
    skip_reason: str = ""
    error: str = ""
    warnings: list = field(default_factory=list)
    input_shape: tuple = ()
    output_shape: tuple = ()
    dtype: str = ""
    pixel_stats: dict = field(default_factory=dict)


@dataclass
class ConversionConfig:
    """Configuration for conversion process."""
    overwrite: bool = False
    verify: bool = True
    output_dir: Optional[Path] = None
    preserve_structure: bool = True
    atomic_write: bool = True
    check_stats: bool = True
    nan_handling: str = "warn"  # "warn", "zero", "error"
