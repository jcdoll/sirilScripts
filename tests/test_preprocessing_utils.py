"""Tests for preprocessing_utils module."""

from pathlib import Path

from siril_job_runner.models import FrameInfo
from siril_job_runner.preprocessing_utils import (
    create_sequence_file,
    group_frames_by_filter_exposure,
)


class TestCreateSequenceFile:
    """Tests for Siril .seq file creation."""

    def test_creates_file(self, temp_dir):
        """Should create a .seq file."""
        seq_path = temp_dir / "test.seq"
        create_sequence_file(seq_path, 10, "light")
        assert seq_path.exists()

    def test_header_comments(self, temp_dir):
        """Should include Siril header comments."""
        seq_path = temp_dir / "test.seq"
        create_sequence_file(seq_path, 5, "light")
        content = seq_path.read_text()
        assert "#Siril sequence file" in content

    def test_s_line_format(self, temp_dir):
        """S line should have correct format with sequence name and counts."""
        seq_path = temp_dir / "test.seq"
        create_sequence_file(seq_path, 12, "myseq")
        content = seq_path.read_text()
        assert "S 'myseq' 1 12 12 5 -1 1" in content

    def test_i_lines_count(self, temp_dir):
        """Should have one I line per image."""
        seq_path = temp_dir / "test.seq"
        create_sequence_file(seq_path, 7, "light")
        content = seq_path.read_text()
        i_lines = [line for line in content.splitlines() if line.startswith("I ")]
        assert len(i_lines) == 7

    def test_i_lines_sequential(self, temp_dir):
        """I lines should have sequential indices starting at 1."""
        seq_path = temp_dir / "test.seq"
        create_sequence_file(seq_path, 3, "light")
        content = seq_path.read_text()
        assert "I 1 1" in content
        assert "I 2 1" in content
        assert "I 3 1" in content

    def test_layer_line(self, temp_dir):
        """Should include L -1 line for auto layer count."""
        seq_path = temp_dir / "test.seq"
        create_sequence_file(seq_path, 1, "light")
        content = seq_path.read_text()
        assert "L -1" in content


class TestGroupFramesByFilterExposure:
    """Tests for frame grouping."""

    def test_single_group(self):
        """All same filter+exposure should be one group."""
        frames = [
            FrameInfo(Path("1.fit"), 300.0, -10.0, "L"),
            FrameInfo(Path("2.fit"), 300.0, -10.0, "L"),
            FrameInfo(Path("3.fit"), 300.0, -10.0, "L"),
        ]
        groups = group_frames_by_filter_exposure(frames)
        assert len(groups) == 1
        assert groups[0].filter_name == "L"
        assert groups[0].exposure == 300.0
        assert len(groups[0].frames) == 3

    def test_multiple_filters(self):
        """Different filters should create separate groups."""
        frames = [
            FrameInfo(Path("1.fit"), 180.0, -10.0, "L"),
            FrameInfo(Path("2.fit"), 180.0, -10.0, "R"),
            FrameInfo(Path("3.fit"), 180.0, -10.0, "G"),
        ]
        groups = group_frames_by_filter_exposure(frames)
        assert len(groups) == 3
        filter_names = {g.filter_name for g in groups}
        assert filter_names == {"L", "R", "G"}

    def test_multiple_exposures_same_filter(self):
        """Different exposures for same filter should create separate groups."""
        frames = [
            FrameInfo(Path("1.fit"), 180.0, -10.0, "L"),
            FrameInfo(Path("2.fit"), 30.0, -10.0, "L"),
        ]
        groups = group_frames_by_filter_exposure(frames)
        assert len(groups) == 2
        exposures = {g.exposure for g in groups}
        assert exposures == {180.0, 30.0}

    def test_sorted_by_filter_then_exposure(self):
        """Groups should be sorted by filter name then exposure."""
        frames = [
            FrameInfo(Path("1.fit"), 300.0, -10.0, "R"),
            FrameInfo(Path("2.fit"), 60.0, -10.0, "L"),
            FrameInfo(Path("3.fit"), 300.0, -10.0, "L"),
        ]
        groups = group_frames_by_filter_exposure(frames)
        result = [(g.filter_name, g.exposure) for g in groups]
        assert result == [("L", 60.0), ("L", 300.0), ("R", 300.0)]

    def test_empty_input(self):
        """Empty frame list should return empty group list."""
        groups = group_frames_by_filter_exposure([])
        assert groups == []
