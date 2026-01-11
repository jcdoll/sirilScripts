"""Core XISF to FITS conversion logic using Siril."""

import logging
from pathlib import Path
from typing import Protocol

from .models import ConversionConfig, ConversionResult

__version__ = "2.0.0"


class SirilInterface(Protocol):
    """Protocol for Siril scripting interface."""

    def cd(self, path: str) -> bool: ...
    def load(self, path: str) -> bool: ...
    def save(self, path: str) -> bool: ...


def get_output_path(
    input_path: Path, config: ConversionConfig, root_dir: Path | None = None
) -> Path:
    """Determine output path for a given input file."""
    if config.output_dir is None:
        return input_path.with_suffix(".fit")

    if config.preserve_structure and root_dir:
        rel_path = input_path.relative_to(root_dir)
        output_path = config.output_dir / rel_path.with_suffix(".fit")
    else:
        output_path = config.output_dir / input_path.with_suffix(".fit").name

    return output_path


def convert_xisf_to_fits(
    xisf_path: Path,
    siril: SirilInterface,
    config: ConversionConfig,
    root_dir: Path | None = None,
) -> ConversionResult:
    """
    Convert a single XISF file to FITS format using Siril.

    Args:
        xisf_path: Path to the XISF file
        siril: Siril interface instance
        config: Conversion configuration
        root_dir: Root directory for preserving structure

    Returns:
        ConversionResult with success/failure info
    """
    result = ConversionResult(input_path=xisf_path)

    try:
        fit_path = get_output_path(xisf_path, config, root_dir)
        result.output_path = fit_path

        if fit_path.exists() and not config.overwrite:
            result.skipped = True
            result.skip_reason = "output exists"
            return result

        fit_path.parent.mkdir(parents=True, exist_ok=True)

        # Siril save command adds .fit extension automatically
        fit_stem_str = str(fit_path.with_suffix("")).replace("\\", "/")

        # Change to parent directory, load XISF, save as FITS
        parent_dir = str(xisf_path.parent).replace("\\", "/")
        if not siril.cd(parent_dir):
            result.error = f"Failed to cd to {parent_dir}"
            return result

        if not siril.load(xisf_path.name):
            result.error = f"Failed to load {xisf_path.name}"
            return result

        if not siril.save(fit_stem_str):
            result.error = f"Failed to save {fit_path}"
            return result

        if fit_path.exists():
            result.success = True
        else:
            result.error = "Output file not created"

        return result

    except Exception as e:
        result.error = str(e)
        logging.debug(f"Exception details for {xisf_path}:", exc_info=True)
        return result
