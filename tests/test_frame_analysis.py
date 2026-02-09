"""Tests for frame_analysis module (date parsing and summary tables)."""

from pathlib import Path

from siril_job_runner.frame_analysis import (
    _extract_date_from_path,
    build_date_summary_table,
    format_date_summary_table,
)
from siril_job_runner.models import FrameInfo


class TestExtractDateFromPath:
    """Tests for date extraction from file paths."""

    def test_standard_path(self):
        """Should extract date from standard path structure."""
        path = Path("/data/M42/2025_01_15/L180/frame.fit")
        assert _extract_date_from_path(path) == "2025_01_15"

    def test_different_date(self):
        """Should extract different dates."""
        path = Path("/data/target/2024_08_28/H300/frame.fit")
        assert _extract_date_from_path(path) == "2024_08_28"

    def test_no_date_in_path(self):
        """Should return None if no date pattern found."""
        path = Path("/data/some/random/path/frame.fit")
        assert _extract_date_from_path(path) is None

    def test_multiple_date_parts(self):
        """Should find date even with multiple path components."""
        path = Path("/base/2025_01_15/subdir/file.fit")
        assert _extract_date_from_path(path) == "2025_01_15"


class TestBuildDateSummaryTable:
    """Tests for date summary table building."""

    def test_single_date_single_filter(self):
        """Should create entry for single date with one filter."""
        frames = [
            FrameInfo(Path("/M42/2025_01_15/L180/f1.fit"), 180.0, -10.0, "L"),
            FrameInfo(Path("/M42/2025_01_15/L180/f2.fit"), 180.0, -10.0, "L"),
        ]
        entries = build_date_summary_table(frames)
        assert len(entries) == 1
        assert entries[0].date == "2025_01_15"
        assert entries[0].filter_counts["L"] == "2"

    def test_multiple_dates(self):
        """Should create separate entries for different dates."""
        frames = [
            FrameInfo(Path("/M42/2025_01_15/L180/f1.fit"), 180.0, -10.0, "L"),
            FrameInfo(Path("/M42/2025_01_20/L180/f2.fit"), 180.0, -10.0, "L"),
        ]
        entries = build_date_summary_table(frames)
        assert len(entries) == 2
        dates = [e.date for e in entries]
        assert "2025_01_15" in dates
        assert "2025_01_20" in dates

    def test_multiple_filters_same_date(self):
        """Should show counts per filter for each date."""
        frames = [
            FrameInfo(Path("/M42/2025_01_15/L180/f1.fit"), 180.0, -10.0, "L"),
            FrameInfo(Path("/M42/2025_01_15/R180/f2.fit"), 180.0, -10.0, "R"),
            FrameInfo(Path("/M42/2025_01_15/R180/f3.fit"), 180.0, -10.0, "R"),
        ]
        entries = build_date_summary_table(frames)
        assert len(entries) == 1
        assert entries[0].filter_counts["L"] == "1"
        assert entries[0].filter_counts["R"] == "2"

    def test_empty_frames(self):
        """Should return empty list for no frames."""
        entries = build_date_summary_table([])
        assert entries == []

    def test_temperature_most_common(self):
        """Should report most common temperature for each date."""
        frames = [
            FrameInfo(Path("/M42/2025_01_15/L/f1.fit"), 180.0, -10.0, "L"),
            FrameInfo(Path("/M42/2025_01_15/L/f2.fit"), 180.0, -10.0, "L"),
            FrameInfo(Path("/M42/2025_01_15/L/f3.fit"), 180.0, -5.0, "L"),
        ]
        entries = build_date_summary_table(frames)
        assert entries[0].temperature == -10  # -10 appears twice


class TestFormatDateSummaryTable:
    """Tests for date summary table formatting."""

    def test_empty_entries(self):
        """Should return empty list for no entries."""
        lines = format_date_summary_table([], ["L", "R"])
        assert lines == []

    def test_output_has_header_and_separators(self):
        """Should include header row and separator lines."""
        frames = [
            FrameInfo(Path("/M42/2025_01_15/L180/f1.fit"), 180.0, -10.0, "L"),
        ]
        entries = build_date_summary_table(frames)
        lines = format_date_summary_table(entries, ["L"])
        assert len(lines) >= 4  # separator, header, separator, data row, separator
        assert any("Date" in line for line in lines)

    def test_output_contains_date(self):
        """Should contain date values in output."""
        frames = [
            FrameInfo(Path("/M42/2025_01_15/L180/f1.fit"), 180.0, -10.0, "L"),
        ]
        entries = build_date_summary_table(frames)
        lines = format_date_summary_table(entries, ["L"])
        assert any("2025_01_15" in line for line in lines)
