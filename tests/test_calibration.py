"""Tests for calibration module."""

import tempfile
from pathlib import Path

import pytest

from siril_job_runner.calibration import (
    CalibrationDates,
    CalibrationManager,
    CalibrationStatus,
)


@pytest.fixture
def temp_base():
    """Create temporary base directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def dates():
    """Standard calibration dates for testing."""
    return CalibrationDates(
        bias="2024_01_15",
        darks="2024_01_15",
        flats="2024_01_20",
    )


@pytest.fixture
def cal_manager(temp_base, dates):
    """Create CalibrationManager with temp directory."""
    return CalibrationManager(temp_base, dates)


# Path resolution tests


def test_bias_master_path(cal_manager):
    """Test bias master path generation."""
    path = cal_manager.get_bias_master_path()
    assert path.name == "bias_2024_01_15.fit"
    assert "masters" in str(path)
    assert "biases" in str(path)


def test_dark_master_path(cal_manager):
    """Test dark master path generation."""
    path = cal_manager.get_dark_master_path(300.0, -10.0)
    assert path.name == "dark_300s_-10C_2024_01_15.fit"
    assert "darks" in str(path)


def test_flat_master_path(cal_manager):
    """Test flat master path generation."""
    path = cal_manager.get_flat_master_path("L")
    assert path.name == "flat_L_2024_01_20.fit"
    assert "flats" in str(path)


def test_bias_raw_path(cal_manager):
    """Test bias raw path generation."""
    path = cal_manager.get_bias_raw_path()
    assert path.name == "2024_01_15"
    assert "raw" in str(path)
    assert "biases" in str(path)


def test_dark_raw_path(cal_manager):
    """Test dark raw path generation."""
    path = cal_manager.get_dark_raw_path(300.0, -10.0)
    # Structure: darks/{date}_{temp}/{exposure}/
    assert path.name == "300"
    assert "2024_01_15_-10C" in str(path)


def test_flat_raw_path(cal_manager):
    """Test flat raw path generation."""
    path = cal_manager.get_flat_raw_path("Ha")
    assert "2024_01_20" in str(path)
    assert path.name == "Ha"


# Status checking tests


def test_check_bias_not_exists(cal_manager):
    """Test bias check when nothing exists."""
    status = cal_manager.check_bias()
    assert not status.exists
    assert not status.can_build


def test_check_bias_master_exists(cal_manager, temp_base):
    """Test bias check when master exists."""
    # Create master file
    master_path = cal_manager.get_bias_master_path()
    master_path.parent.mkdir(parents=True, exist_ok=True)
    master_path.touch()

    status = cal_manager.check_bias()
    assert status.exists
    assert status.can_build
    assert status.master_path == master_path


def test_check_bias_can_build(cal_manager):
    """Test bias check when raw frames exist."""
    # Create raw directory with files
    raw_path = cal_manager.get_bias_raw_path()
    raw_path.mkdir(parents=True, exist_ok=True)
    (raw_path / "bias_001.fit").touch()

    status = cal_manager.check_bias()
    assert not status.exists
    assert status.can_build
    assert status.raw_path == raw_path


def test_check_dark_not_exists(cal_manager):
    """Test dark check when nothing exists."""
    status = cal_manager.check_dark(300.0, -10.0)
    assert not status.exists
    assert not status.can_build


def test_check_dark_master_exists(cal_manager):
    """Test dark check when master exists."""
    master_path = cal_manager.get_dark_master_path(300.0, -10.0)
    master_path.parent.mkdir(parents=True, exist_ok=True)
    master_path.touch()

    status = cal_manager.check_dark(300.0, -10.0)
    assert status.exists
    assert status.can_build


def test_check_dark_can_build(cal_manager):
    """Test dark check when raw frames exist."""
    raw_path = cal_manager.get_dark_raw_path(300.0, -10.0)
    raw_path.mkdir(parents=True, exist_ok=True)
    (raw_path / "dark_001.fit").touch()

    status = cal_manager.check_dark(300.0, -10.0)
    assert not status.exists
    assert status.can_build
    assert status.raw_path == raw_path


def test_check_flat_not_exists(cal_manager):
    """Test flat check when nothing exists."""
    status = cal_manager.check_flat("L")
    assert not status.exists
    assert not status.can_build


def test_check_flat_master_exists(cal_manager):
    """Test flat check when master exists."""
    master_path = cal_manager.get_flat_master_path("L")
    master_path.parent.mkdir(parents=True, exist_ok=True)
    master_path.touch()

    status = cal_manager.check_flat("L")
    assert status.exists
    assert status.can_build


# Temperature tolerance tests (only for darks)


def test_find_matching_dark_exact(cal_manager):
    """Test finding dark with exact temperature match."""
    master_path = cal_manager.get_dark_master_path(300.0, -10.0)
    master_path.parent.mkdir(parents=True, exist_ok=True)
    master_path.touch()

    result = cal_manager.find_matching_dark(300.0, -10.0)
    assert result == master_path


def test_find_matching_dark_within_tolerance(cal_manager):
    """Test finding dark within temperature tolerance."""
    # Create master at -10C
    master_path = cal_manager.get_dark_master_path(300.0, -10.0)
    master_path.parent.mkdir(parents=True, exist_ok=True)
    master_path.touch()

    # Should find it when looking for -9C (within 2C tolerance)
    result = cal_manager.find_matching_dark(300.0, -9.0)
    assert result == master_path


def test_find_matching_dark_not_found(cal_manager):
    """Test finding dark when none exists."""
    result = cal_manager.find_matching_dark(300.0, -10.0)
    assert result is None


def test_temperature_rounding(cal_manager):
    """Test that temperature is rounded in dark paths."""
    path = cal_manager.get_dark_master_path(300.0, -9.6)
    assert "-10C" in path.name


# CalibrationDates tests


def test_calibration_dates():
    """Test CalibrationDates dataclass."""
    dates = CalibrationDates(
        bias="2024_01_15",
        darks="2024_01_16",
        flats="2024_01_17",
    )
    assert dates.bias == "2024_01_15"
    assert dates.darks == "2024_01_16"
    assert dates.flats == "2024_01_17"


# CalibrationStatus tests


def test_calibration_status():
    """Test CalibrationStatus dataclass."""
    status = CalibrationStatus(
        exists=True,
        can_build=True,
        master_path=Path("/test/master.fit"),
        raw_path=None,
        message="Master exists",
    )
    assert status.exists
    assert status.can_build
    assert status.master_path == Path("/test/master.fit")
