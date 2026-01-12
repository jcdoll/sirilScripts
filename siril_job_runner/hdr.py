"""
HDR (High Dynamic Range) blending for multi-exposure astrophotography.

This module implements brightness-weighted HDR compositing using Siril's PixelMath.
It blends multiple exposures of the same target to extend dynamic range, using
longer exposures for shadow detail and shorter exposures to recover clipped highlights.

Algorithm
---------
For each pixel, the blend weight is determined by the pixel's brightness in the
longer exposure:

    weight = clamp((long - low_threshold) / (high_threshold - low_threshold), 0, 1)
    result = long * (1 - weight) + short * weight

Where:
    - low_threshold (default 0.7): Below this, use 100% long exposure
    - high_threshold (default 0.9): Above this, use 100% short exposure
    - Between thresholds: smooth linear blend

For 3+ exposures, blending is done iteratively from longest to shortest:
    1. Blend 180s + 30s -> intermediate
    2. Blend intermediate + 10s -> final HDR

This approach:
    - Preserves maximum S/N in faint regions (long exposure dominates)
    - Recovers clipped highlights (short exposure fills in)
    - Smooth transitions avoid visible seams

Usage
-----
The HDRBlender class handles the full workflow:

    1. Register all exposure stacks to the longest exposure (best S/N for detection)
    2. Linear match backgrounds so all exposures are on same scale
    3. Blend using PixelMath formulas
    4. Output single HDR-blended stack per channel

References
----------
- Siril PixelMath: https://siril.readthedocs.io/en/stable/processing/pixelmath.html
- HDR discussion: https://discuss.pixls.us/t/hdr-processing-in-siril/20492
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .composition import StackInfo
from .config import HDR, PROCESSING
from .logger import JobLogger
from .protocols import SirilInterface


@dataclass
class HDROptions:
    """Options for HDR blending."""

    low_threshold: float = HDR.low_threshold
    high_threshold: float = HDR.high_threshold


def build_blend_formula(
    long_var: str,
    short_var: str,
    low_threshold: float,
    high_threshold: float,
) -> str:
    """
    Build PixelMath formula for brightness-weighted HDR blend.

    The formula computes a weight based on the long exposure's pixel value,
    then blends between long and short exposures.

    Args:
        long_var: Variable name for long exposure (e.g., "long")
        short_var: Variable name for short exposure (e.g., "short")
        low_threshold: Below this, use 100% long exposure
        high_threshold: Above this, use 100% short exposure

    Returns:
        PixelMath expression string
    """
    # Siril PixelMath uses iif(condition, true_val, false_val)
    # We need to compute: w = clamp((long - low) / (high - low), 0, 1)
    # Then: result = long * (1 - w) + short * w

    # The range for the linear ramp
    ramp_range = high_threshold - low_threshold

    # Build the weight calculation using nested iif:
    # w = 0 if long < low_threshold
    # w = 1 if long > high_threshold
    # w = (long - low) / range otherwise
    weight_expr = (
        f"iif(${long_var}$ < {low_threshold}, 0, "
        f"iif(${long_var}$ > {high_threshold}, 1, "
        f"(${long_var}$ - {low_threshold}) / {ramp_range}))"
    )

    # result = long * (1 - w) + short * w
    # Rewritten to avoid computing weight twice:
    # We compute weight inline in both terms
    formula = f"${long_var}$ * (1 - {weight_expr}) + ${short_var}$ * {weight_expr}"

    return formula


class HDRBlender:
    """
    Handles HDR blending of multiple exposures per channel.

    Workflow:
        1. For each channel with multiple exposures:
           a. Register all to longest exposure
           b. Linear match backgrounds
           c. Iteratively blend from longest to shortest
        2. Output single HDR stack per channel
    """

    def __init__(
        self,
        siril: SirilInterface,
        work_dir: Path,
        options: Optional[HDROptions] = None,
        logger: Optional[JobLogger] = None,
    ):
        self.siril = siril
        self.work_dir = Path(work_dir)
        self.options = options or HDROptions()
        self.logger = logger

    def _log(self, message: str) -> None:
        if self.logger:
            self.logger.substep(message)

    def _log_step(self, message: str) -> None:
        if self.logger:
            self.logger.step(message)

    def blend_channel(
        self,
        channel: str,
        stacks: list[StackInfo],
        output_path: Path,
    ) -> Path:
        """
        Blend multiple exposures of a single channel into HDR result.

        Args:
            channel: Channel/filter name (e.g., "L", "R")
            stacks: List of StackInfo for this channel, sorted by exposure
            output_path: Where to save the blended result

        Returns:
            Path to the blended HDR stack
        """
        if len(stacks) < 2:
            # Nothing to blend, just return the single stack
            if stacks:
                return stacks[0].path
            raise ValueError(f"No stacks provided for channel {channel}")

        self._log_step(f"HDR blending {channel} ({len(stacks)} exposures)")

        # Sort by exposure (longest first for blending order)
        sorted_stacks = sorted(stacks, key=lambda s: s.exposure, reverse=True)

        # Create HDR working directory
        hdr_dir = self.work_dir / "hdr" / channel
        hdr_dir.mkdir(parents=True, exist_ok=True)
        self.siril.cd(str(hdr_dir))

        # Step 1: Load and save all stacks with consistent names
        # Longest exposure is our reference
        ref_stack = sorted_stacks[0]
        self._log(f"Reference: {ref_stack.name} ({ref_stack.exposure}s)")

        self.siril.load(str(ref_stack.path))
        self.siril.save("ref")

        # Load other exposures and linear match to reference
        for i, stack in enumerate(sorted_stacks[1:], 1):
            self._log(f"Loading {stack.name} ({stack.exposure}s)")
            self.siril.load(str(stack.path))

            # Register to reference
            # Note: stacks should already be registered from preprocessing
            # but we linear match to ensure consistent background levels
            self.siril.linear_match(
                "ref", PROCESSING.linear_match_low, PROCESSING.linear_match_high
            )
            self.siril.save(f"exp{i}")

        # Step 2: Iterative blending from longest to shortest
        # Start with ref (longest), blend with next shortest, etc.
        current = "ref"

        for i, _stack in enumerate(sorted_stacks[1:], 1):
            short_name = f"exp{i}"
            blend_name = f"blend{i}"

            self._log(
                f"Blending {current} with {short_name} "
                f"(thresholds: {self.options.low_threshold}-{self.options.high_threshold})"
            )

            # Build and execute blend formula
            formula = build_blend_formula(
                long_var=current,
                short_var=short_name,
                low_threshold=self.options.low_threshold,
                high_threshold=self.options.high_threshold,
            )

            self.siril.pm(formula, rescale=True)
            self.siril.save(blend_name)
            current = blend_name

        # Step 3: Save final result
        self.siril.load(current)
        self.siril.save(str(output_path))

        if not output_path.exists():
            # Siril may have added .fit extension
            if output_path.with_suffix(".fit").exists():
                return output_path.with_suffix(".fit")
            raise FileNotFoundError(f"HDR blend output not created: {output_path}")

        self._log(f"HDR complete: {output_path.name}")
        return output_path

    def blend_all_channels(
        self,
        stacks_by_channel: dict[str, list[StackInfo]],
        output_dir: Path,
    ) -> dict[str, Path]:
        """
        Blend all channels that have multiple exposures.

        Args:
            stacks_by_channel: Dict mapping channel name to list of StackInfo
            output_dir: Directory for output stacks

        Returns:
            Dict mapping channel name to HDR-blended stack path
            (or original path if only single exposure)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        for channel, stacks in stacks_by_channel.items():
            if len(stacks) > 1:
                # Multiple exposures - blend
                output_path = output_dir / f"stack_{channel}_HDR.fit"
                results[channel] = self.blend_channel(channel, stacks, output_path)
            elif len(stacks) == 1:
                # Single exposure - use as-is
                results[channel] = stacks[0].path
            # Skip channels with no stacks

        return results


def blend_hdr_stacks(
    siril: SirilInterface,
    stacks: dict[str, list[StackInfo]],
    work_dir: Path,
    output_dir: Path,
    options: Optional[HDROptions] = None,
    logger: Optional[JobLogger] = None,
) -> dict[str, Path]:
    """
    Convenience function to blend HDR stacks.

    Args:
        siril: Siril interface
        stacks: Dict from discover_stacks() - channel name to list of StackInfo
        work_dir: Working directory for intermediate files
        output_dir: Output directory for final HDR stacks
        options: HDR options (thresholds)
        logger: Optional logger

    Returns:
        Dict mapping channel name to HDR-blended (or single) stack path
    """
    blender = HDRBlender(siril, work_dir, options, logger)
    return blender.blend_all_channels(stacks, output_dir)
