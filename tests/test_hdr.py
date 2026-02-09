"""Tests for hdr module."""

from siril_job_runner.hdr import build_blend_formula


class TestBuildBlendFormula:
    """Tests for HDR blend formula construction."""

    def test_formula_contains_variables(self):
        """Formula should reference both long and short variables."""
        formula = build_blend_formula("long", "short", 0.7, 0.9)
        assert "$long$" in formula
        assert "$short$" in formula

    def test_formula_contains_thresholds(self):
        """Formula should contain threshold values."""
        formula = build_blend_formula("long", "short", 0.7, 0.9)
        assert "0.7" in formula
        assert "0.9" in formula

    def test_different_thresholds(self):
        """Different thresholds should produce different formulas."""
        f1 = build_blend_formula("long", "short", 0.5, 0.8)
        f2 = build_blend_formula("long", "short", 0.7, 0.9)
        assert f1 != f2

    def test_custom_variable_names(self):
        """Should work with custom variable names."""
        formula = build_blend_formula("ref", "exp1", 0.6, 0.85)
        assert "$ref$" in formula
        assert "$exp1$" in formula

    def test_formula_is_valid_pixelmath(self):
        """Formula should be valid PixelMath syntax (balanced parens, iif)."""
        formula = build_blend_formula("long", "short", 0.7, 0.9)
        assert formula.count("(") == formula.count(")")
        assert "iif" in formula
