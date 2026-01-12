"""
Job configuration loading and validation.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .config import Config, merge_overrides, with_overrides

try:
    import jsonschema
except ImportError:
    jsonschema = None


@dataclass
class JobConfig:
    """Parsed job configuration."""

    name: str
    job_type: str  # "type" is reserved in Python
    calibration_bias: str
    calibration_darks: str
    calibration_flats: str
    lights: dict[str, list[str]]  # filter -> list of directories
    output: str
    config: Config = field(default_factory=lambda: Config())

    @classmethod
    def from_dict(cls, data: dict, settings: Optional[dict] = None) -> "JobConfig":
        """
        Create JobConfig from dictionary.

        Args:
            data: Job definition dict (from JSON)
            settings: Optional user settings (from settings.json)
        """
        # Normalize lights to always be lists
        lights = {}
        for filter_name, paths in data["lights"].items():
            if isinstance(paths, str):
                lights[filter_name] = [paths]
            else:
                lights[filter_name] = list(paths)

        # Merge settings and job options, then create Config
        settings_options = settings.get("options", {}) if settings else {}
        job_options = data.get("options", {})
        merged = merge_overrides(settings_options, job_options)
        config = with_overrides(merged)

        return cls(
            name=data["name"],
            job_type=data["type"],
            calibration_bias=data["calibration"]["bias"],
            calibration_darks=data["calibration"]["darks"],
            calibration_flats=data["calibration"]["flats"],
            lights=lights,
            output=data["output"],
            config=config,
        )

    @classmethod
    def from_file(cls, path: Path, settings: Optional[dict] = None) -> "JobConfig":
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

        return cls.from_dict(data, settings)

    def get_filters(self) -> list[str]:
        """Get list of filter names."""
        return list(self.lights.keys())

    def get_light_directories(self, filter_name: str) -> list[str]:
        """Get light directories for a filter."""
        return self.lights.get(filter_name, [])


def load_job(path: Path, settings: Optional[dict] = None) -> JobConfig:
    """Load a job configuration file."""
    return JobConfig.from_file(path, settings)


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
    except ValueError as e:
        return False, str(e)  # Unknown config option
    except Exception as e:
        return False, f"Unexpected error: {e}"
