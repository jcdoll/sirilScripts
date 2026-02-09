"""Tests for stack_discovery module."""

from pathlib import Path

from siril_job_runner.stack_discovery import discover_stacks, is_hdr_mode


def test_discover_stacks_single_exposure(temp_dir):
    """Should discover single-exposure stacks."""
    (temp_dir / "stack_L_180s.fit").touch()
    (temp_dir / "stack_R_180s.fit").touch()
    (temp_dir / "stack_G_180s.fit").touch()
    (temp_dir / "stack_B_180s.fit").touch()

    stacks = discover_stacks(temp_dir)
    assert set(stacks.keys()) == {"L", "R", "G", "B"}
    for channel, stack_list in stacks.items():
        assert len(stack_list) == 1
        assert stack_list[0].filter_name == channel
        assert stack_list[0].exposure == 180


def test_discover_stacks_hdr(temp_dir):
    """Should discover multiple exposures per filter for HDR."""
    (temp_dir / "stack_L_180s.fit").touch()
    (temp_dir / "stack_L_30s.fit").touch()

    stacks = discover_stacks(temp_dir)
    assert "L" in stacks
    assert len(stacks["L"]) == 2
    exposures = [s.exposure for s in stacks["L"]]
    assert sorted(exposures) == [30, 180]


def test_discover_stacks_sorted_by_exposure(temp_dir):
    """Stacks within a filter should be sorted by exposure."""
    (temp_dir / "stack_H_300s.fit").touch()
    (temp_dir / "stack_H_60s.fit").touch()
    (temp_dir / "stack_H_10s.fit").touch()

    stacks = discover_stacks(temp_dir)
    exposures = [s.exposure for s in stacks["H"]]
    assert exposures == [10, 60, 300]


def test_discover_stacks_ignores_non_matching(temp_dir):
    """Should ignore files that don't match the naming pattern."""
    (temp_dir / "stack_L_180s.fit").touch()
    (temp_dir / "other_file.fit").touch()
    (temp_dir / "rgb_linear.fit").touch()

    stacks = discover_stacks(temp_dir)
    assert set(stacks.keys()) == {"L"}


def test_discover_stacks_empty_dir(temp_dir):
    """Should return empty dict for empty directory."""
    stacks = discover_stacks(temp_dir)
    assert stacks == {}


def test_discover_stacks_nonexistent_dir():
    """Should return empty dict for nonexistent directory."""
    stacks = discover_stacks(Path("/nonexistent"))
    assert stacks == {}


def test_is_hdr_mode_single_exposures():
    """Single exposure per filter is not HDR."""
    from siril_job_runner.models import StackInfo

    stacks = {
        "L": [StackInfo(Path("L.fit"), "L", 180)],
        "R": [StackInfo(Path("R.fit"), "R", 180)],
    }
    assert not is_hdr_mode(stacks)


def test_is_hdr_mode_multiple_exposures():
    """Multiple exposures for any filter is HDR."""
    from siril_job_runner.models import StackInfo

    stacks = {
        "L": [
            StackInfo(Path("L180.fit"), "L", 180),
            StackInfo(Path("L30.fit"), "L", 30),
        ],
        "R": [StackInfo(Path("R.fit"), "R", 180)],
    }
    assert is_hdr_mode(stacks)
