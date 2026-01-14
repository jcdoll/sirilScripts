"""
Composition module for Siril job processing.

Handles LRGB composition, narrowband palette mixing, and stretching.
Based on LRGB_pre.ssf and LRGB_compose.ssf workflows.
"""

import re
from pathlib import Path
from typing import Optional

from .config import DEFAULTS, Config
from .fits_utils import check_color_balance
from .hdr import HDRBlender
from .logger import JobLogger
from .models import CompositionResult, StackInfo
from .protocols import SirilInterface

# Narrowband palette definitions (channel mappings)
PALETTES = {
    "HOO": {"R": "H", "G": "O", "B": "O"},
    "SHO": {"R": "S", "G": "H", "B": "O"},
}


def discover_stacks(stacks_dir: Path) -> dict[str, list[StackInfo]]:
    """
    Discover stacks in the stacks directory.

    Parses filenames like `stack_L_180s.fit` to extract filter and exposure.

    Returns:
        Dict mapping filter name to list of StackInfo (multiple if HDR)
    """
    pattern = re.compile(r"^stack_([A-Z]+)_(\d+)s\.fit$")
    result: dict[str, list[StackInfo]] = {}

    for path in stacks_dir.glob("stack_*_*s.fit"):
        match = pattern.match(path.name)
        if match:
            filter_name = match.group(1)
            exposure = int(match.group(2))
            info = StackInfo(path=path, filter_name=filter_name, exposure=exposure)

            if filter_name not in result:
                result[filter_name] = []
            result[filter_name].append(info)

    # Sort each filter's stacks by exposure
    for filter_name in result:
        result[filter_name].sort(key=lambda s: s.exposure)

    return result


def is_hdr_mode(stacks: dict[str, list[StackInfo]]) -> bool:
    """Check if any filter has multiple exposures (HDR mode)."""
    return any(len(stack_list) > 1 for stack_list in stacks.values())


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
        deconvolve_l: bool = True,
    ) -> CompositionResult:
        """
        Compose LRGB image from stacked channels.

        Args:
            stacks: Dict from discover_stacks() - filter name to list of StackInfo
            deconvolve_l: Whether to deconvolve L channel

        Returns CompositionResult with paths to all outputs.
        """
        self._log_step("Composing LRGB")

        # Verify all required channels (single exposure each)
        for ch in ["L", "R", "G", "B"]:
            if ch not in stacks:
                raise ValueError(f"Missing required channel: {ch}")
            if len(stacks[ch]) != 1:
                raise ValueError(
                    f"Expected single exposure for {ch}, got {len(stacks[ch])}"
                )

        self.siril.cd(str(self.stacks_dir))

        # Step 1: Register stacks to each other
        # After convert, files are numbered alphabetically: B=1, G=2, L=3, R=4
        self._log("Cross-registering stacks...")
        self.siril.convert("stack", out="./registered")
        self.siril.cd(str(self.stacks_dir / "registered"))
        self.siril.register("stack", twopass=True)
        self.siril.seqapplyreg("stack", framing="min")

        # Step 2: Linear match to reference (R)
        # Alphabetical order: B=00001, G=00002, L=00003, R=00004
        self._log("Linear matching to R reference...")
        self.siril.load("r_stack_00004")  # R
        self.siril.save("R")

        cfg = self.config
        self.siril.load("r_stack_00001")  # B
        self.siril.linear_match("R", cfg.linear_match_low, cfg.linear_match_high)
        self.siril.save("B")

        self.siril.load("r_stack_00002")  # G
        self.siril.linear_match("R", cfg.linear_match_low, cfg.linear_match_high)
        self.siril.save("G")

        self.siril.load("r_stack_00003")  # L
        self.siril.linear_match("R", cfg.linear_match_low, cfg.linear_match_high)
        self.siril.save("L")

        # Step 3: Optional deconvolution on L
        l_name = "L"
        if deconvolve_l:
            self._log("Deconvolving L channel...")
            self.siril.load("L")
            self.siril.makepsf("blind")
            self.siril.rl()
            self.siril.save("L_deconv")
            l_name = "L_deconv"

        # Step 4: Compose RGB
        self._log("Creating RGB composite...")
        self.siril.rgbcomp(r="R", g="G", b="B", out="rgb")

        # Step 5: Add luminance
        self._log("Adding luminance channel...")
        self.siril.rgbcomp(lum=l_name, rgb="rgb", out="lrgb")

        # Save linear (unstretched) result
        linear_path = self.output_dir / "lrgb_linear.fit"
        self.siril.load("lrgb")
        self.siril.save(str(linear_path))
        self._log(f"Saved linear: {linear_path.name}")
        self._log_color_balance(linear_path)

        # Step 6: Spectrophotometric Color Calibration (optional)
        # SPCC uses actual sensor QE and filter curves for accurate calibration
        linear_pcc_path = None
        stretch_source = "lrgb"

        if cfg.spcc_enabled:
            self._log("Color calibration (SPCC)...")
            if self.siril.spcc(
                sensor=cfg.spcc_sensor,
                red_filter=cfg.spcc_red_filter,
                green_filter=cfg.spcc_green_filter,
                blue_filter=cfg.spcc_blue_filter,
            ):
                linear_pcc_path = self.output_dir / "lrgb_linear_spcc.fit"
                self.siril.save(str(linear_pcc_path))
                stretch_source = str(linear_pcc_path)
                self._log(f"Saved color-calibrated: {linear_pcc_path.name}")
            else:
                self._log("SPCC failed, using uncalibrated image")
        else:
            self._log("SPCC disabled, skipping color calibration")

        # Step 7: Auto-stretch and save
        auto_paths = self._auto_stretch(stretch_source, "lrgb_auto")

        return CompositionResult(
            linear_path=linear_path,
            linear_pcc_path=linear_pcc_path,
            auto_fit=auto_paths["fit"],
            auto_tif=auto_paths["tif"],
            auto_jpg=auto_paths["jpg"],
            stacks_dir=self.stacks_dir,
        )

    def compose_rgb(
        self,
        stacks: dict[str, list[StackInfo]],
    ) -> CompositionResult:
        """
        Compose RGB image (no luminance channel).

        For cases where L channel is not available.
        """
        self._log_step("Composing RGB")

        # Verify required channels
        for ch in ["R", "G", "B"]:
            if ch not in stacks:
                raise ValueError(f"Missing required channel: {ch}")
            if len(stacks[ch]) != 1:
                raise ValueError(
                    f"Expected single exposure for {ch}, got {len(stacks[ch])}"
                )

        self.siril.cd(str(self.stacks_dir))

        # Register stacks
        # Alphabetical: B=00001, G=00002, R=00003
        self._log("Cross-registering stacks...")
        self.siril.convert("stack", out="./registered")
        self.siril.cd(str(self.stacks_dir / "registered"))
        self.siril.register("stack", twopass=True)
        self.siril.seqapplyreg("stack", framing="min")

        # Linear match to R
        self._log("Linear matching to R reference...")
        self.siril.load("r_stack_00003")  # R
        self.siril.save("R")

        cfg = self.config
        self.siril.load("r_stack_00001")  # B
        self.siril.linear_match("R", cfg.linear_match_low, cfg.linear_match_high)
        self.siril.save("B")

        self.siril.load("r_stack_00002")  # G
        self.siril.linear_match("R", cfg.linear_match_low, cfg.linear_match_high)
        self.siril.save("G")

        # Compose RGB
        self._log("Creating RGB composite...")
        self.siril.rgbcomp(r="R", g="G", b="B", out="rgb")

        # Save linear result
        linear_path = self.output_dir / "rgb_linear.fit"
        self.siril.load("rgb")
        self.siril.save(str(linear_path))
        self._log(f"Saved linear: {linear_path.name}")
        self._log_color_balance(linear_path)

        # Spectrophotometric Color Calibration (optional)
        # SPCC uses actual sensor QE and filter curves for accurate calibration
        linear_pcc_path = None
        stretch_source = "rgb"

        if cfg.spcc_enabled:
            self._log("Color calibration (SPCC)...")
            if self.siril.spcc(
                sensor=cfg.spcc_sensor,
                red_filter=cfg.spcc_red_filter,
                green_filter=cfg.spcc_green_filter,
                blue_filter=cfg.spcc_blue_filter,
            ):
                linear_pcc_path = self.output_dir / "rgb_linear_spcc.fit"
                self.siril.save(str(linear_pcc_path))
                stretch_source = str(linear_pcc_path)
                self._log(f"Saved color-calibrated: {linear_pcc_path.name}")
            else:
                self._log("SPCC failed, using uncalibrated image")
        else:
            self._log("SPCC disabled, skipping color calibration")

        # Auto-stretch
        auto_paths = self._auto_stretch(stretch_source, "rgb_auto")

        return CompositionResult(
            linear_path=linear_path,
            linear_pcc_path=linear_pcc_path,
            auto_fit=auto_paths["fit"],
            auto_tif=auto_paths["tif"],
            auto_jpg=auto_paths["jpg"],
            stacks_dir=self.stacks_dir,
        )

    def compose_narrowband(
        self,
        stacks: dict[str, list[StackInfo]],
        palette: str = "HOO",
    ) -> CompositionResult:
        """
        Compose narrowband image using palette mapping.

        Palettes:
        - HOO: H->R, O->G, O->B
        - SHO: S->R, H->G, O->B
        """
        self._log_step(f"Composing narrowband ({palette})")

        if palette not in PALETTES:
            raise ValueError(
                f"Unknown palette: {palette}. Available: {list(PALETTES.keys())}"
            )

        mapping = PALETTES[palette]
        required = set(mapping.values())

        for ch in required:
            if ch not in stacks:
                raise ValueError(f"Missing required channel: {ch}")
            if len(stacks[ch]) != 1:
                raise ValueError(
                    f"Expected single exposure for {ch}, got {len(stacks[ch])}"
                )

        self.siril.cd(str(self.stacks_dir))

        # Register stacks
        self._log("Cross-registering stacks...")
        self.siril.convert("stack", out="./registered")
        self.siril.cd(str(self.stacks_dir / "registered"))
        self.siril.register("stack", twopass=True)
        self.siril.seqapplyreg("stack", framing="min")

        # For HOO: H=00001, O=00002
        # For SHO: H=00001, O=00002, S=00003
        # Determine numbering based on what channels exist
        channels_sorted = sorted(stacks.keys())
        channel_to_num = {ch: f"{i + 1:05d}" for i, ch in enumerate(channels_sorted)}

        # Note: No linear matching for narrowband
        # Linear matching destroys the relative intensities between channels
        # which are needed for proper palette mapping. Each narrowband filter
        # captures different emission lines with different intrinsic brightness.
        self._log("Saving registered channels (no linear match for narrowband)...")
        for ch in required:
            self.siril.load(f"r_stack_{channel_to_num[ch]}")
            self.siril.save(ch)

        # Map channels according to palette
        self._log(f"Applying {palette} palette...")
        r_src = mapping["R"]
        g_src = mapping["G"]
        b_src = mapping["B"]

        self.siril.rgbcomp(r=r_src, g=g_src, b=b_src, out="narrowband")

        # Save linear result
        type_name = palette.lower()
        linear_path = self.output_dir / f"{type_name}_linear.fit"
        self.siril.load("narrowband")
        self.siril.save(str(linear_path))
        self._log(f"Saved linear: {linear_path.name}")
        self._log_color_balance(linear_path)

        # Note: PCC not applicable for narrowband - uses synthetic colors
        linear_pcc_path = None

        # Auto-stretch
        auto_paths = self._auto_stretch("narrowband", f"{type_name}_auto")

        return CompositionResult(
            linear_path=linear_path,
            linear_pcc_path=linear_pcc_path,
            auto_fit=auto_paths["fit"],
            auto_tif=auto_paths["tif"],
            auto_jpg=auto_paths["jpg"],
            stacks_dir=self.stacks_dir,
        )

    def _auto_stretch(self, input_name: str, output_name: str) -> dict[str, Path]:
        """
        Apply auto-stretch pipeline and save in multiple formats.

        Pipeline from LRGB_compose.ssf:
        - autostretch
        - mtf 0.20 0.5 1.0
        - satu 1 0
        """
        self._log("Auto-stretching...")

        cfg = self.config
        self.siril.load(input_name)
        self.siril.autostretch(linked=True)
        self.siril.mtf(cfg.mtf_low, cfg.mtf_mid, cfg.mtf_high)
        self.siril.satu(cfg.saturation_amount, cfg.saturation_threshold)

        # Save in multiple formats
        # Note: Siril auto-adds extensions, so pass path without extension
        self.siril.cd(str(self.output_dir))

        fit_path = self.output_dir / f"{output_name}.fit"
        tif_path = self.output_dir / f"{output_name}.tif"
        jpg_path = self.output_dir / f"{output_name}.jpg"

        self.siril.save(str(self.output_dir / output_name))
        self.siril.savetif(str(self.output_dir / output_name), astro=True, deflate=True)
        self.siril.savejpg(str(self.output_dir / output_name), 90)

        self._log(f"Saved: {fit_path.name}, {tif_path.name}, {jpg_path.name}")

        return {"fit": fit_path, "tif": tif_path, "jpg": jpg_path}


def compose_and_stretch(
    siril: SirilInterface,
    output_dir: Path,
    job_type: str,
    palette: str = "HOO",
    deconvolve_l: bool = True,
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
        deconvolve_l: Whether to deconvolve L channel (LRGB only)
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
    if is_hdr_mode(stacks):
        if logger:
            logger.step("Multiple exposures detected - HDR blending")

        blender = HDRBlender(siril, output_dir, config, logger)

        # Blend all channels
        hdr_stacks_dir = stacks_dir / "hdr"
        hdr_stacks_dir.mkdir(parents=True, exist_ok=True)

        blended_paths = blender.blend_all_channels(stacks, hdr_stacks_dir)

        # Convert back to single-exposure stacks dict for composition
        # Use exposure=0 to indicate HDR blend (not a real exposure time)
        stacks = {}
        for channel, path in blended_paths.items():
            stacks[channel] = [StackInfo(path=path, filter_name=channel, exposure=0)]

        if logger:
            logger.step("HDR blending complete")

    composer = Composer(siril, output_dir, config, logger)

    if job_type == "LRGB":
        return composer.compose_lrgb(stacks, deconvolve_l=deconvolve_l)
    elif job_type == "RGB":
        return composer.compose_rgb(stacks)
    elif job_type in ("SHO", "HOO"):
        return composer.compose_narrowband(stacks, palette=palette or job_type)
    else:
        raise ValueError(f"Unknown job type: {job_type}")
