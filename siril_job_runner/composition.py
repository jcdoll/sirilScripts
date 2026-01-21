"""
Composition module for Siril job processing.

Handles LRGB composition, narrowband palette mixing, and stretching.
Based on LRGB_pre.ssf and LRGB_compose.ssf workflows.
"""

import shutil
from pathlib import Path
from typing import Optional

from .compose_broadband import compose_lrgb, compose_rgb
from .compose_narrowband import compose_narrowband
from .config import DEFAULTS, Config
from .fits_utils import check_color_balance
from .hdr import HDRBlender
from .logger import JobLogger
from .models import CompositionResult, StackInfo
from .protocols import SirilInterface
from .stack_discovery import PALETTES, discover_stacks, is_hdr_mode
from .stretch_pipeline import StretchPipeline


def _cross_register_stacks(
    siril: SirilInterface,
    stacks: dict[str, list[StackInfo]],
    output_dir: Path,
    logger: Optional[JobLogger] = None,
) -> dict[str, list[StackInfo]]:
    """
    Cross-register all exposure stacks to a common reference before HDR blending.

    This ensures all channels and exposures are spatially aligned before blending.
    Uses the longest L exposure as reference (highest SNR, most stars).

    Args:
        siril: Siril interface
        stacks: Dict mapping channel to list of StackInfo (multiple exposures)
        output_dir: Output directory for registered stacks
        logger: Optional logger

    Returns:
        Updated stacks dict with paths to registered versions
    """

    def log(msg: str) -> None:
        if logger:
            logger.substep(msg)

    def log_step(msg: str) -> None:
        if logger:
            logger.step(msg)

    log_step("Cross-registering all stacks for HDR")

    # Collect all stacks into a flat list with index mapping
    all_stacks: list[tuple[str, StackInfo]] = []
    for channel, stack_list in sorted(stacks.items()):
        for stack in sorted(stack_list, key=lambda s: -s.exposure):
            all_stacks.append((channel, stack))

    if len(all_stacks) < 2:
        log("Only one stack, skipping registration")
        return stacks

    # Find reference: longest L exposure, or longest overall if no L
    ref_idx = 0
    ref_exposure = 0
    for i, (channel, stack) in enumerate(all_stacks):
        if channel == "L" and stack.exposure > ref_exposure:
            ref_idx = i
            ref_exposure = stack.exposure
    # If no L found, use longest exposure overall
    if ref_exposure == 0:
        for i, (_channel, stack) in enumerate(all_stacks):
            if stack.exposure > ref_exposure:
                ref_idx = i
                ref_exposure = stack.exposure

    ref_channel, ref_stack = all_stacks[ref_idx]
    log(f"Reference: {ref_stack.name} ({ref_channel} {ref_stack.exposure}s)")

    # Create working directory
    work_dir = output_dir / "stacks" / "hdr_register"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Copy all stacks with sequential numbered names
    log("Preparing stacks for registration...")
    for i, (channel, stack) in enumerate(all_stacks):
        src_path = stack.path
        dst_path = work_dir / f"stack_{i + 1:05d}.fit"
        shutil.copy2(src_path, dst_path)
        log(f"  {i + 1}: {stack.name} ({channel} {stack.exposure}s)")

    # Run Siril registration
    siril.cd(str(work_dir))
    siril.convert("stack", out="./registered")
    siril.cd(str(work_dir / "registered"))

    # Set reference (1-based index)
    siril.setref("stack", ref_idx + 1)
    log(f"Set reference to image {ref_idx + 1}")

    # Register with 2-pass for better accuracy
    siril.register("stack", twopass=True)
    siril.seqapplyreg("stack", framing="min")

    # Build result dict with registered paths
    registered_dir = work_dir / "registered"
    result: dict[str, list[StackInfo]] = {}

    for i, (channel, stack) in enumerate(all_stacks):
        reg_path = registered_dir / f"r_stack_{i + 1:05d}.fit"
        if not reg_path.exists():
            log(f"WARNING: Registration failed for {stack.name}, using original")
            reg_path = stack.path

        if channel not in result:
            result[channel] = []
        result[channel].append(
            StackInfo(
                path=reg_path,
                filter_name=stack.filter_name,
                exposure=stack.exposure,
            )
        )

    log("Cross-registration complete")
    return result


# Re-export for backwards compatibility
__all__ = [
    "PALETTES",
    "discover_stacks",
    "is_hdr_mode",
    "Composer",
    "compose_and_stretch",
]


class Composer:
    """
    Handles image composition and stretching.

    Workflow based on LRGB_pre.ssf:
    1. Register all stacks to each other
    2. Linear match all channels to reference (R)
    3. Optional: Deconvolve L channel
    4. rgbcomp R G B -> rgb
    5. rgbcomp -lum=L rgb -> lrgb (for LRGB)
    6. autostretch + mtf + satu for auto output
    """

    def __init__(
        self,
        siril: SirilInterface,
        output_dir: Path,
        config: Config = DEFAULTS,
        logger: Optional[JobLogger] = None,
    ):
        self.siril = siril
        self.output_dir = Path(output_dir)
        self.config = config
        self.logger = logger
        self.stacks_dir = self.output_dir / "stacks"
        self._stretch_pipeline = StretchPipeline(siril, output_dir, config, self._log)

    def _log(self, message: str) -> None:
        if self.logger:
            self.logger.substep(message)

    def _log_step(self, message: str) -> None:
        if self.logger:
            self.logger.step(message)

    def _log_color_balance(self, image_path: Path) -> None:
        """Check and log color balance for an RGB image."""
        balance = check_color_balance(image_path)
        if balance is None:
            return

        self._log(
            f"Color balance: R={balance.r_median:.1f}, "
            f"G={balance.g_median:.1f}, B={balance.b_median:.1f}"
        )

        if balance.is_imbalanced:
            self._log(
                f"WARNING: Color imbalance detected! "
                f"{balance.dominant_channel} dominates by {balance.dominance_ratio:.1f}x"
            )

    def compose_lrgb(
        self,
        stacks: dict[str, list[StackInfo]],
        is_hdr: bool = False,
    ) -> CompositionResult:
        """Compose LRGB image from stacked channels."""
        return compose_lrgb(
            siril=self.siril,
            stacks=stacks,
            stacks_dir=self.stacks_dir,
            output_dir=self.output_dir,
            config=self.config,
            stretch_pipeline=self._stretch_pipeline,
            log_fn=self._log,
            log_step_fn=self._log_step,
            log_color_balance_fn=self._log_color_balance,
            is_hdr=is_hdr,
        )

    def compose_rgb(
        self,
        stacks: dict[str, list[StackInfo]],
    ) -> CompositionResult:
        """Compose RGB image (no luminance channel)."""
        return compose_rgb(
            siril=self.siril,
            stacks=stacks,
            stacks_dir=self.stacks_dir,
            output_dir=self.output_dir,
            config=self.config,
            stretch_pipeline=self._stretch_pipeline,
            log_fn=self._log,
            log_step_fn=self._log_step,
            log_color_balance_fn=self._log_color_balance,
        )

    def compose_narrowband(
        self,
        stacks: dict[str, list[StackInfo]],
        palette: str = "HOO",
    ) -> CompositionResult:
        """Compose narrowband image using palette mapping."""
        return compose_narrowband(
            siril=self.siril,
            stacks=stacks,
            stacks_dir=self.stacks_dir,
            output_dir=self.output_dir,
            config=self.config,
            stretch_pipeline=self._stretch_pipeline,
            palette=palette,
            log_fn=self._log,
            log_step_fn=self._log_step,
            log_color_balance_fn=self._log_color_balance,
        )


def compose_and_stretch(
    siril: SirilInterface,
    output_dir: Path,
    job_type: str,
    palette: str = "HOO",
    config: Config = DEFAULTS,
    logger: Optional[JobLogger] = None,
) -> CompositionResult:
    """
    Discover stacks and compose based on job type.

    Automatically handles HDR mode by blending multiple exposures per channel
    before composition.

    Args:
        siril: Siril interface
        output_dir: Output directory (contains stacks/ subdirectory)
        job_type: "LRGB", "RGB", "SHO", or "HOO"
        palette: Narrowband palette (for SHO/HOO)
        config: Configuration with processing parameters
        logger: Optional logger

    Returns:
        CompositionResult with paths to all outputs
    """
    stacks_dir = Path(output_dir) / "stacks"
    stacks = discover_stacks(stacks_dir)

    if not stacks:
        raise FileNotFoundError(f"No stacks found in {stacks_dir}")

    if logger:
        logger.step("Discovered stacks:")
        for _filter_name, stack_list in sorted(stacks.items()):
            for s in stack_list:
                logger.substep(f"{s.name}.fit")

    # Check for HDR mode and blend if needed
    hdr_mode = is_hdr_mode(stacks)
    if hdr_mode:
        if logger:
            logger.step("Multiple exposures detected - HDR blending")

        # Cross-register ALL stacks before HDR blending
        # This ensures channels are aligned before blending different exposures
        stacks = _cross_register_stacks(siril, stacks, output_dir, logger)

        blender = HDRBlender(siril, output_dir, config, logger)

        # Blend all channels (now using pre-registered stacks)
        hdr_stacks_dir = stacks_dir / "hdr"
        hdr_stacks_dir.mkdir(parents=True, exist_ok=True)

        blended_paths = blender.blend_all_channels(stacks, hdr_stacks_dir)

        # Convert back to single-exposure stacks dict for composition
        # Use exposure=0 to indicate HDR blend (not a real exposure time)
        stacks = {}
        for channel, path in blended_paths.items():
            stacks[channel] = [StackInfo(path=path, filter_name=channel, exposure=0)]

        # Use HDR directory for composition instead of original stacks directory
        stacks_dir = hdr_stacks_dir

        if logger:
            logger.step("HDR blending complete")

    composer = Composer(siril, output_dir, config, logger)
    composer.stacks_dir = (
        stacks_dir  # Override with correct directory (HDR or original)
    )

    if job_type == "LRGB":
        return composer.compose_lrgb(stacks, is_hdr=hdr_mode)
    elif job_type == "RGB":
        return composer.compose_rgb(stacks)
    elif job_type in ("SHO", "HOO"):
        return composer.compose_narrowband(stacks, palette=palette or job_type)
    else:
        raise ValueError(f"Unknown job type: {job_type}")
