"""Tests for palettes module."""

import pytest

from siril_job_runner.palettes import (
    PALETTES,
    build_effective_palette,
    formula_to_pixelmath,
    get_palette,
)


class TestGetPalette:
    """Tests for palette lookup."""

    def test_get_known_palette(self):
        """Should return palette for known name."""
        palette = get_palette("SHO")
        assert palette.name == "SHO (Hubble)"
        assert palette.required == {"S", "H", "O"}

    def test_get_hoo_palette(self):
        """HOO palette should require H and O only."""
        palette = get_palette("HOO")
        assert palette.required == {"H", "O"}
        assert palette.r == "H"
        assert palette.g == "O"
        assert palette.b == "O"

    def test_get_unknown_palette_raises(self):
        """Should raise KeyError for unknown palette."""
        with pytest.raises(KeyError, match="Unknown palette"):
            get_palette("NONEXISTENT")

    def test_error_message_lists_available(self):
        """Error message should list available palettes."""
        with pytest.raises(KeyError) as exc_info:
            get_palette("BOGUS")
        assert "SHO" in str(exc_info.value)
        assert "HOO" in str(exc_info.value)


class TestPaletteIsSimple:
    """Tests for Palette.is_simple()."""

    def test_sho_is_simple(self):
        """SHO (direct mapping) should be simple."""
        palette = get_palette("SHO")
        assert palette.is_simple()

    def test_hoo_is_simple(self):
        """HOO (direct mapping, O used twice) should be simple."""
        palette = get_palette("HOO")
        assert palette.is_simple()

    def test_sho_foraxx_is_not_simple(self):
        """SHO Foraxx (blended G channel) should not be simple."""
        palette = get_palette("SHO_FORAXX")
        assert not palette.is_simple()

    def test_sho_dynamic_is_not_simple(self):
        """SHO Dynamic (all channels blended) should not be simple."""
        palette = get_palette("SHO_DYNAMIC")
        assert not palette.is_simple()


class TestFormulaToPixelmath:
    """Tests for formula to PixelMath conversion."""

    def test_simple_channel(self):
        """Single channel should become $X$."""
        result = formula_to_pixelmath("H")
        assert result == "$H$"

    def test_weighted_sum(self):
        """Weighted sum should convert channels and normalize spacing."""
        result = formula_to_pixelmath("0.5*H + 0.5*O")
        assert "$H$" in result
        assert "$O$" in result
        assert "+" in result

    def test_preserves_numbers(self):
        """Numbers should not be wrapped in $ markers."""
        result = formula_to_pixelmath("0.8*S + 0.2*H")
        assert "0.8" in result
        assert "0.2" in result
        assert "$S$" in result
        assert "$H$" in result

    def test_complex_formula(self):
        """Complex formula with power operator should convert correctly."""
        result = formula_to_pixelmath("(O^(1-O))*S + (1-(O^(1-O)))*H")
        assert "$O$" in result
        assert "$S$" in result
        assert "$H$" in result
        # Numbers should be preserved
        assert "1" in result

    def test_all_channels(self):
        """All channel names (S, H, O, L) should be converted."""
        result = formula_to_pixelmath("S + H + O + L")
        assert "$S$" in result
        assert "$H$" in result
        assert "$O$" in result
        assert "$L$" in result


class TestBuildEffectivePalette:
    """Tests for palette override building."""

    def test_no_overrides_returns_same(self):
        """No overrides should return palette with same name."""
        base = get_palette("SHO")
        result = build_effective_palette(base)
        assert result.name == base.name
        assert result.r == base.r
        assert result.g == base.g
        assert result.b == base.b

    def test_r_override(self):
        """R override should replace R formula."""
        base = get_palette("SHO")
        result = build_effective_palette(base, r_override="0.5*S + 0.5*H")
        assert result.r == "0.5*S + 0.5*H"
        assert result.g == base.g
        assert result.b == base.b

    def test_override_marks_custom_name(self):
        """Any override should append (custom) to name."""
        base = get_palette("SHO")
        result = build_effective_palette(base, g_override="O")
        assert "(custom)" in result.name

    def test_preserves_required_channels(self):
        """Override should preserve required channel set."""
        base = get_palette("SHO")
        result = build_effective_palette(base, r_override="H")
        assert result.required == base.required

    def test_preserves_dynamic_flag(self):
        """Override should preserve dynamic flag."""
        base = get_palette("SHO_FORAXX_DYNAMIC")
        result = build_effective_palette(base, b_override="S")
        assert result.dynamic is True


class TestPaletteRegistry:
    """Tests for the palette registry."""

    def test_all_palettes_have_required_fields(self):
        """All registered palettes should have non-empty required channels."""
        for name, palette in PALETTES.items():
            assert len(palette.required) > 0, f"{name} has empty required set"
            assert palette.r, f"{name} has empty R formula"
            assert palette.g, f"{name} has empty G formula"
            assert palette.b, f"{name} has empty B formula"

    def test_all_palettes_accessible_by_name(self):
        """All registered palettes should be accessible via get_palette."""
        for name in PALETTES:
            palette = get_palette(name)
            assert palette is not None
