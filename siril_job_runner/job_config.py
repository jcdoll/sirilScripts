"""
Job configuration loading and validation.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .config import HDR, PROCESSING

try:
    import jsonschema
except ImportError:
    jsonschema = None


@dataclass
class JobOptions:
    """Optional job parameters."""

    fwhm_filter: float = PROCESSING.fwhm_filter
    temp_tolerance: float = PROCESSING.temp_tolerance
    denoise: bool = False
    palette: str = "HOO"
    dark_temp_override: Optional[float] = None
    clipping_warning_threshold: float = 0.01
    hdr_low_threshold: float = HDR.low_threshold
    hdr_high_threshold: float = HDR.high_threshold


@dataclass
class JobConfig:
    """Parsed job configuration."""

    name: str
    job_type: str  # "type" is reserved
    calibration_bias: str
    calibration_darks: str
    calibration_flats: str
    lights: dict[str, list[str]]  # filter -> list of directories
    output: str
    options: JobOptions = field(default_factory=JobOptions)

    @classmethod
    def from_dict(cls, data: dict) -> "JobConfig":
        """Create JobConfig from dictionary."""
        # Normalize lights to always be lists
        lights = {}
        for filter_name, paths in data["lights"].items():
            if isinstance(paths, str):
                lights[filter_name] = [paths]
            else:
                lights[filter_name] = list(paths)

        # Parse options - only override fields present in data
        options_data = data.get("options", {})
        options_kwargs = {}
        for key in [
            "fwhm_filter",
            "temp_tolerance",
            "denoise",
            "palette",
            "dark_temp_override",
            "clipping_warning_threshold",
            "hdr_low_threshold",
            "hdr_high_threshold",
        ]:
            if key in options_data:
                options_kwargs[key] = options_data[key]
        options = JobOptions(**options_kwargs)

        return cls(
            name=data["name"],
            job_type=data["type"],
            calibration_bias=data["calibration"]["bias"],
            calibration_darks=data["calibration"]["darks"],
            calibration_flats=data["calibration"]["flats"],
            lights=lights,
            output=data["output"],
            options=options,
        )

    @classmethod
    def from_file(cls, path: Path) -> "JobConfig":
        """Load and validate job configuration from file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Job file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Validate against schema if jsonschema available
        if jsonschema is not None:
            schema_path = Path(__file__).parent / "job_schema.json"
            if schema_path.exists():
                with open(schema_path, "r", encoding="utf-8") as f:
                    schema = json.load(f)
                jsonschema.validate(data, schema)

        return cls.from_dict(data)

    def get_filters(self) -> list[str]:
        """Get list of filter names."""
        return list(self.lights.keys())

    def get_light_directories(self, filter_name: str) -> list[str]:
        """Get light directories for a filter."""
        return self.lights.get(filter_name, [])


def load_job(path: Path) -> JobConfig:
    """Load a job configuration file."""
    return JobConfig.from_file(path)


def load_settings(repo_root: Path) -> dict:
    """
    Load user settings from settings.json.

    Returns empty dict if file doesn't exist.
    """
    settings_path = Path(repo_root) / "settings.json"
    if settings_path.exists():
        with open(settings_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def validate_job_file(path: Path) -> tuple[bool, Optional[str]]:
    """
    Validate a job file without loading it fully.

    Returns (is_valid, error_message).
    """
    try:
        load_job(path)
        return True, None
    except FileNotFoundError as e:
        return False, str(e)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    except jsonschema.ValidationError as e:
        return False, f"Schema validation failed: {e.message}"
    except KeyError as e:
        return False, f"Missing required field: {e}"
    except Exception as e:
        return False, f"Unexpected error: {e}"
