"""Tests for models module (uncovered properties)."""

from pathlib import Path

from siril_job_runner.models import ClippingInfo, StackGroup, StackInfo


class TestStackInfo:
    """Tests for StackInfo properties."""

    def test_name_property(self):
        """Name should be stem of path (without extension)."""
        info = StackInfo(
            path=Path("/data/stacks/stack_L_180s.fit"), filter_name="L", exposure=180
        )
        assert info.name == "stack_L_180s"


class TestStackGroup:
    """Tests for StackGroup properties."""

    def test_exposure_str(self):
        """Should format exposure as integer seconds."""
        group = StackGroup(filter_name="L", exposure=180.0, frames=[])
        assert group.exposure_str == "180s"

    def test_stack_name(self):
        """Should combine filter and exposure."""
        group = StackGroup(filter_name="R", exposure=60.0, frames=[])
        assert group.stack_name == "stack_R_60s"


class TestClippingInfo:
    """Tests for ClippingInfo properties."""

    def test_clipped_low_percent(self):
        """Should compute percentage of low-clipped pixels."""
        info = ClippingInfo(
            path=Path("test.fit"),
            total_pixels=10000,
            clipped_low=100,
            clipped_high=50,
            bit_depth=16,
        )
        assert info.clipped_low_percent == 1.0

    def test_clipped_high_percent(self):
        """Should compute percentage of high-clipped pixels."""
        info = ClippingInfo(
            path=Path("test.fit"),
            total_pixels=10000,
            clipped_low=100,
            clipped_high=50,
            bit_depth=16,
        )
        assert info.clipped_high_percent == 0.5

    def test_zero_total_pixels(self):
        """Should return 0 for zero total pixels (avoid division by zero)."""
        info = ClippingInfo(
            path=Path("test.fit"),
            total_pixels=0,
            clipped_low=0,
            clipped_high=0,
            bit_depth=16,
        )
        assert info.clipped_low_percent == 0.0
        assert info.clipped_high_percent == 0.0
