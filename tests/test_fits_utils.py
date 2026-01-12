"""Tests for fits_utils module."""

from pathlib import Path

from siril_job_runner.fits_utils import (
    FrameInfo,
    _get_header_value,
    temperatures_match,
)
from siril_job_runner.frame_analysis import (
    RequirementsEntry,
    build_requirements_table,
    get_unique_exposures,
    get_unique_filters,
    get_unique_temperatures,
)

# Test FrameInfo


def test_frame_info_exposure_str():
    """Test exposure string formatting."""
    frame = FrameInfo(
        path=Path("test.fit"),
        exposure=300.0,
        temperature=-10.0,
        filter_name="L",
    )
    assert frame.exposure_str == "300s"


def test_frame_info_temp_str():
    """Test temperature string formatting."""
    frame = FrameInfo(
        path=Path("test.fit"),
        exposure=300.0,
        temperature=-10.0,
        filter_name="L",
    )
    assert frame.temp_str == "-10C"


def test_frame_info_temp_str_rounds():
    """Test temperature string rounds correctly."""
    frame = FrameInfo(
        path=Path("test.fit"),
        exposure=300.0,
        temperature=-9.6,
        filter_name="L",
    )
    assert frame.temp_str == "-10C"


# Test header value extraction


def test_get_header_value_first_match():
    """Test that first matching keyword is used."""
    header = {"EXPTIME": 300, "EXPOSURE": 600}
    result = _get_header_value(header, ["EXPTIME", "EXPOSURE"])
    assert result == 300


def test_get_header_value_fallback():
    """Test fallback to second keyword."""
    header = {"EXPOSURE": 600}
    result = _get_header_value(header, ["EXPTIME", "EXPOSURE"])
    assert result == 600


def test_get_header_value_default():
    """Test default when no keyword found."""
    header = {}
    result = _get_header_value(header, ["EXPTIME"], default=0)
    assert result == 0


# Test requirements table


def test_build_requirements_table_groups():
    """Test that frames are grouped correctly."""
    frames = [
        FrameInfo(Path("1.fit"), 300.0, -10.0, "L"),
        FrameInfo(Path("2.fit"), 300.0, -10.0, "L"),
        FrameInfo(Path("3.fit"), 300.0, -10.0, "L"),
        FrameInfo(Path("4.fit"), 60.0, -10.0, "R"),
    ]

    table = build_requirements_table(frames)

    assert len(table) == 2

    l_entry = next(e for e in table if e.filter_name == "L")
    assert l_entry.count == 3
    assert l_entry.exposure == 300.0

    r_entry = next(e for e in table if e.filter_name == "R")
    assert r_entry.count == 1
    assert r_entry.exposure == 60.0


def test_build_requirements_table_temperature_grouping():
    """Test that similar temperatures are grouped."""
    frames = [
        FrameInfo(Path("1.fit"), 300.0, -10.2, "L"),
        FrameInfo(Path("2.fit"), 300.0, -9.8, "L"),  # Rounds to -10
    ]

    table = build_requirements_table(frames)

    assert len(table) == 1
    assert table[0].count == 2


def test_build_requirements_empty():
    """Test empty frame list."""
    table = build_requirements_table([])
    assert table == []


# Test unique value extraction


def test_get_unique_exposures():
    """Test unique exposure extraction."""
    frames = [
        FrameInfo(Path("1.fit"), 300.0, -10.0, "L"),
        FrameInfo(Path("2.fit"), 300.0, -10.0, "R"),
        FrameInfo(Path("3.fit"), 60.0, -10.0, "R"),
    ]

    exposures = get_unique_exposures(frames)
    assert exposures == {300.0, 60.0}


def test_get_unique_temperatures():
    """Test unique temperature extraction."""
    frames = [
        FrameInfo(Path("1.fit"), 300.0, -10.0, "L"),
        FrameInfo(Path("2.fit"), 300.0, -10.2, "R"),  # Rounds to -10
        FrameInfo(Path("3.fit"), 60.0, -5.0, "R"),
    ]

    temps = get_unique_temperatures(frames)
    assert temps == {-10, -5}


def test_get_unique_filters():
    """Test unique filter extraction."""
    frames = [
        FrameInfo(Path("1.fit"), 300.0, -10.0, "L"),
        FrameInfo(Path("2.fit"), 300.0, -10.0, "R"),
        FrameInfo(Path("3.fit"), 60.0, -10.0, "L"),
    ]

    filters = get_unique_filters(frames)
    assert filters == {"L", "R"}


# Test temperature matching


def test_temperatures_match_exact():
    """Test exact temperature match."""
    assert temperatures_match(-10.0, -10.0)


def test_temperatures_match_within_tolerance():
    """Test temperature match within tolerance."""
    assert temperatures_match(-10.0, -8.5, tolerance=2.0)
    assert temperatures_match(-10.0, -11.5, tolerance=2.0)


def test_temperatures_no_match_outside_tolerance():
    """Test temperature no match outside tolerance."""
    assert not temperatures_match(-10.0, -7.0, tolerance=2.0)


def test_temperatures_match_custom_tolerance():
    """Test custom tolerance."""
    assert temperatures_match(-10.0, -5.0, tolerance=5.0)
    assert not temperatures_match(-10.0, -5.0, tolerance=4.0)


# Test RequirementsEntry


def test_requirements_entry_strings():
    """Test RequirementsEntry string formatting."""
    entry = RequirementsEntry(
        filter_name="L",
        exposure=300.0,
        temperature=-10.0,
        count=45,
    )

    assert entry.exposure_str == "300s"
    assert entry.temp_str == "-10C"
