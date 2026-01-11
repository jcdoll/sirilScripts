"""Data models for XISF to FITS conversion."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ConversionResult:
    """Result of a single file conversion."""

    input_path: Path
    output_path: Path | None = None
    success: bool = False
    skipped: bool = False
    skip_reason: str = ""
    error: str = ""
    warnings: list = field(default_factory=list)


@dataclass
class ConversionConfig:
    """Configuration for conversion process."""

    overwrite: bool = False
    output_dir: Path | None = None
    preserve_structure: bool = True
