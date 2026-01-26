"""Tests for job_config module."""

import json
import tempfile
from pathlib import Path

import pytest

from siril_job_runner.config import DEFAULTS
from siril_job_runner.job_config import JobConfig, load_job, validate_job_file


@pytest.fixture
def valid_job_dict():
    """Valid job configuration dict."""
    return {
        "name": "TestJob",
        "type": "LRGB",
        "calibration": {
            "bias": "2024_01_15",
            "darks": "2024_01_15",
            "flats": "2024_01_20",
        },
        "lights": {
            "L": ["M42/L"],
            "R": "M42/R",  # Single string
        },
        "output": "M42/processed",
        "options": {
            "fwhm_bic_threshold": 15.0,
            "temp_tolerance": 3,
        },
    }


@pytest.fixture
def job_file(valid_job_dict):
    """Create temporary job file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(valid_job_dict, f)
        return Path(f.name)


def test_job_config_from_dict(valid_job_dict):
    """Test creating JobConfig from dict."""
    config = JobConfig.from_dict(valid_job_dict)

    assert config.name == "TestJob"
    assert config.job_type == "LRGB"
    assert config.calibration_bias == "2024_01_15"
    assert config.calibration_darks == "2024_01_15"
    assert config.calibration_flats == "2024_01_20"
    assert config.output == "M42/processed"


def test_job_config_normalizes_lights(valid_job_dict):
    """Test that single string lights are normalized to lists."""
    config = JobConfig.from_dict(valid_job_dict)

    # Both should be lists
    assert config.lights["L"] == ["M42/L"]
    assert config.lights["R"] == ["M42/R"]


def test_job_config_options(valid_job_dict):
    """Test options parsing."""
    job = JobConfig.from_dict(valid_job_dict)

    assert job.config.fwhm_bic_threshold == 15.0
    assert job.config.temp_tolerance == 3
    assert job.config.denoise is False  # Default


def test_job_config_default_options():
    """Test default options when not specified."""
    minimal = {
        "name": "Test",
        "type": "LRGB",
        "calibration": {
            "bias": "2024-01-01",
            "darks": "2024-01-01",
            "flats": "2024-01-01",
        },
        "lights": {"L": ["L/"]},
        "output": "out",
    }
    job = JobConfig.from_dict(minimal)

    assert job.config.fwhm_bic_threshold == 10.0
    assert job.config.temp_tolerance == 2.0
    assert job.config.denoise is False
    assert job.config.palette == "SHO"


def test_load_job_from_file(job_file):
    """Test loading job from file."""
    config = load_job(job_file)
    assert config.name == "TestJob"


def test_validate_job_file_valid(job_file):
    """Test validation of valid job file."""
    is_valid, error = validate_job_file(job_file)
    assert is_valid
    assert error is None


def test_validate_job_file_not_found():
    """Test validation of non-existent file."""
    is_valid, error = validate_job_file(Path("/nonexistent/job.json"))
    assert not is_valid
    assert "not found" in error.lower()


def test_validate_job_file_invalid_json():
    """Test validation of invalid JSON."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("not valid json")
        path = Path(f.name)

    is_valid, error = validate_job_file(path)
    assert not is_valid
    assert "json" in error.lower()


def test_get_filters(valid_job_dict):
    """Test get_filters method."""
    config = JobConfig.from_dict(valid_job_dict)
    filters = config.get_filters()
    assert set(filters) == {"L", "R"}


def test_get_light_directories(valid_job_dict):
    """Test get_light_directories method."""
    config = JobConfig.from_dict(valid_job_dict)
    dirs = config.get_light_directories("L")
    assert dirs == ["M42/L"]


def test_config_defaults():
    """Test Config defaults."""
    assert DEFAULTS.fwhm_bic_threshold == 10.0
    assert DEFAULTS.fwhm_bimodal_sigma == 3.0
    assert DEFAULTS.temp_tolerance == 2.0
    assert DEFAULTS.denoise is False
    assert DEFAULTS.palette == "SHO"
    # HDR defaults
    assert DEFAULTS.hdr_low_threshold == 0.7
    assert DEFAULTS.hdr_high_threshold == 0.9
    # Stretch defaults
    assert DEFAULTS.stretch_method == "veralux"
    assert DEFAULTS.stretch_compare is True
    assert DEFAULTS.autostretch_targetbg == 0.10
    assert DEFAULTS.veralux_target_median == 0.10
    assert DEFAULTS.veralux_b == 6.0
    assert DEFAULTS.veralux_log_d_min == 0.0
    assert DEFAULTS.veralux_log_d_max == 7.0
    assert DEFAULTS.saturation_amount == 0.25


def test_config_override_any_value():
    """Test that any config value can be overridden via job options."""
    job_dict = {
        "name": "Test",
        "type": "LRGB",
        "calibration": {
            "bias": "2024-01-01",
            "darks": "2024-01-01",
            "flats": "2024-01-01",
        },
        "lights": {"L": ["L/"]},
        "output": "out",
        "options": {
            # Override various config values
            "hdr_low_threshold": 0.5,
            "hdr_high_threshold": 0.8,
            "autostretch_targetbg": 0.15,
            "stack_sigma_low": "2.5",
            "clipping_high_16bit": 60000,
        },
    }
    job = JobConfig.from_dict(job_dict)

    assert job.config.hdr_low_threshold == 0.5
    assert job.config.hdr_high_threshold == 0.8
    assert job.config.autostretch_targetbg == 0.15
    assert job.config.stack_sigma_low == "2.5"
    assert job.config.clipping_high_16bit == 60000
    # Unchanged values stay at defaults
    assert job.config.fwhm_bic_threshold == 10.0


def test_config_unknown_option_raises():
    """Test that unknown options raise ValueError."""
    from siril_job_runner.config import with_overrides

    with pytest.raises(ValueError, match="Unknown config options"):
        with_overrides({"invalid_option": 123})


def test_config_settings_and_job_merge():
    """Test that settings.json and job options merge correctly."""
    settings = {"options": {"fwhm_bic_threshold": 15.0, "temp_tolerance": 3.0}}
    job_dict = {
        "name": "Test",
        "type": "LRGB",
        "calibration": {
            "bias": "2024-01-01",
            "darks": "2024-01-01",
            "flats": "2024-01-01",
        },
        "lights": {"L": ["L/"]},
        "output": "out",
        "options": {"temp_tolerance": 5.0},  # Override settings value
    }
    job = JobConfig.from_dict(job_dict, settings)

    # fwhm_bic_threshold from settings
    assert job.config.fwhm_bic_threshold == 15.0
    # temp_tolerance from job (overrides settings)
    assert job.config.temp_tolerance == 5.0
