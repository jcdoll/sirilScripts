"""
Broadband composition for LRGB and RGB imaging.

Handles composition of L, R, G, B filter stacks into color images.

LRGB Workflow (see docs/lrgb_workflow.md for details):

LINEAR PHASE:
 1. Cross-register all stacks (R, G, B, L)
 2. Compose RGB (NO linear matching between channels)
 3. Color calibration (SPCC) on RGB
 4. Deconvolution on RGB and L (optional)

STRETCH PHASE (always - baseline with original stars):
 5. Stretch RGB and L
 6. Combine LRGB
 7. Color removal (SCNR)
 8. Output baseline images (lrgb_autostretch, lrgb_veralux)

STARNET PHASE (conditional - if starnet_enabled):
 9. Star removal on RGB -> RGB_starless + RGB_stars
10. Star removal on L -> L_starless
11. Stretch starless images
12. Combine LRGB from starless
13. Output starless images
14. Recompose stars (if starcomposer_enabled)
15. Output starcomposer images

Key principles:
- Background extraction is done pre-stack on individual subs (seqsubsky degree 1)
- SPCC must run on RGB before adding L
- Baseline outputs always have original stars (no StarNet artifacts)
- StarNet branch is optional and conditional on starnet_enabled config
"""

from pathlib import Path
from typing import TYPE_CHECKING, Callable

from .compose_helpers import (
    apply_color_removal,
    apply_deconvolution,
    apply_spcc_step,
    neutralize_rgb_background,
    save_diagnostic_preview,
)
from .config import ColorRemovalMode, Config, SubskyMethod
from .models import CompositionResult, StackInfo
from .siril_file_ops import link_or_copy
from .stretch_helpers import (
    apply_enhancements,
    apply_saturation,
    apply_stretch,
    save_all_formats,
)

if TYPE_CHECKING:
    from .protocols import SirilInterface


# Mapping from channel name to alphabetical index (for registered sequences)
# LRGB: B=00001, G=00002, L=00003, R=00004
LRGB_CHANNEL_INDEX = {"B": "00001", "G": "00002", "L": "00003", "R": "00004"}
# RGB: B=00001, G=00002, R=00003
RGB_CHANNEL_INDEX = {"B": "00001", "G": "00002", "R": "00003"}


def _apply_post_stretch(
    siril: "SirilInterface",
    image_name: str,
    working_dir: Path,
    config: Config,
    log_fn: Callable[[str], None],
) -> None:
    """
    Apply post-stretch processing to a broadband image.

    This is the standard pipeline after any stretch method (autostretch or veralux).
    The stretch method only affects tone mapping; post-stretch processing is always
    the same regardless of which stretch was used.

    Steps:
        1. Color cast removal (SCNR green/magenta)
        2. Background color neutralization (linear match G,B to R)
        3. Saturation adjustment
        4. VeraLux enhancements (Silentium denoise, Revela detail, Vectra saturation)
    """
    if config.color_removal_mode != ColorRemovalMode.NONE:
        siril.load(image_name)
        apply_color_removal(siril, config, log_fn)
        siril.save(image_name)

    if config.broadband_neutralization:
        neutralize_rgb_background(
            siril,
            image_name,
            config,
            log_fn,
            low=config.broadband_neutralization_low,
            high=config.broadband_neutralization_high,
        )

    siril.load(image_name)
    apply_saturation(siril, config)
    siril.save(image_name)

    _apply_veralux_processing(siril, image_name, working_dir, config, log_fn)


def _apply_veralux_processing(
    siril: "SirilInterface",
    source_name: str,
    working_dir: Path,
    config: Config,
    log_fn: Callable[[str], None],
) -> None:
    """Apply VeraLux post-processing (Silentium, Revela, Vectra) to stretched image."""
    # Save current state to file for veralux processing
    siril.load(source_name)
    image_path = working_dir / f"{source_name}.fit"
    siril.save(str(working_dir / source_name))

    # Apply enhancements using shared helper
    apply_enhancements(siril, image_path, config, log_fn)

    # Reload processed image
    siril.load(str(image_path))
    siril.save(source_name)


def _add_stars_back(
    siril: "SirilInterface",
    starless_name: str,
    stars_name: str,
    output_name: str,
    working_dir: Path,
    config: Config,
    log_fn: Callable[[str], None],
) -> None:
    """Add stars back to starless image using StarComposer (screen blend fallback)."""
    from .veralux_starcomposer import apply_starcomposer

    log_fn("Applying StarComposer for star recomposition...")

    # Save files for starcomposer
    starless_path = working_dir / f"{starless_name}.fit"
    stars_path = working_dir / f"{stars_name}.fit"
    siril.load(starless_name)
    siril.save(str(starless_path))

    success, result_path = apply_starcomposer(
        siril, starless_path, stars_path, config, log_fn
    )
    if not success:
        log_fn("StarComposer failed, falling back to screen blend")
        siril.load(starless_name)
        siril.pm(f"1 - (1 - ${starless_name}$) * (1 - ${stars_name}$)")
        siril.save(output_name)
        return

    # Load result and save with output name
    siril.load(str(result_path))
    siril.save(output_name)


def _stretch_and_combine_lrgb(
    siril: "SirilInterface",
    rgb_source: str,
    l_source: str,
    output_prefix: str,
    working_dir: Path,
    output_dir: Path,
    config: Config,
    log_fn: Callable[[str], None],
    use_veralux: bool,
) -> str:
    """Stretch RGB and L, combine to LRGB, apply SCNR and saturation."""
    suffix = "veralux" if use_veralux else "auto"
    method = "veralux" if use_veralux else "autostretch"
    rgb_stretched = f"rgb_{output_prefix}_{suffix}"
    l_stretched = f"L_{output_prefix}_{suffix}"
    lrgb_combined = f"lrgb_{output_prefix}_{suffix}"

    # Stretch RGB
    siril.load(rgb_source)
    if use_veralux:
        temp_path = working_dir / f"{rgb_source}_temp.fit"
        siril.save(str(working_dir / f"{rgb_source}_temp"))
        siril.load(l_source)  # Release file lock (Windows)
        success = apply_stretch(siril, method, temp_path, config, log_fn)
        if not success:
            log_fn("  RGB veralux stretch failed, using autostretch")
            siril.load(rgb_source)
            siril.autostretch()
    else:
        apply_stretch(siril, method, working_dir / f"{rgb_source}.fit", config, log_fn)
    siril.save(rgb_stretched)

    # Stretch L
    siril.load(l_source)
    if use_veralux:
        temp_path = working_dir / f"{l_source}_temp.fit"
        siril.save(str(working_dir / f"{l_source}_temp"))
        siril.load(rgb_source)  # Release file lock (Windows)
        success = apply_stretch(siril, method, temp_path, config, log_fn)
        if not success:
            log_fn("  L veralux stretch failed, using autostretch")
            siril.load(l_source)
            siril.autostretch()
    else:
        apply_stretch(siril, method, working_dir / f"{l_source}.fit", config, log_fn)
    siril.save(l_stretched)

    # Combine LRGB
    siril.rgbcomp(lum=l_stretched, rgb=rgb_stretched, out=lrgb_combined)

    _apply_post_stretch(siril, lrgb_combined, working_dir, config, log_fn)

    return lrgb_combined


def _save_outputs(
    siril: "SirilInterface",
    source_name: str,
    output_dir: Path,
    output_basename: str,
    log_fn: Callable[[str], None],
    config: Config,
) -> None:
    """Save FIT, TIF, and JPG outputs."""
    siril.load(source_name)
    save_all_formats(siril, output_dir, output_basename, log_fn, config)


def compose_lrgb(
    siril: "SirilInterface",
    stacks: dict[str, list[StackInfo]],
    stacks_dir: Path,
    output_dir: Path,
    config: Config,
    log_fn: Callable[[str], None],
    log_step_fn: Callable[[str], None],
    log_color_balance_fn: Callable,
    is_hdr: bool = False,
) -> CompositionResult:
    """
    Compose LRGB image from stacked channels.

    See module docstring for full workflow description.

    Args:
        is_hdr: If True, skip cross-registration (HDR stacks are pre-aligned)
    """
    log_step_fn("Composing LRGB")

    # Verify all required channels (single exposure each)
    for ch in ["L", "R", "G", "B"]:
        if ch not in stacks:
            raise ValueError(f"Missing required channel: {ch}")
        if len(stacks[ch]) != 1:
            raise ValueError(
                f"Expected single exposure for {ch}, got {len(stacks[ch])}"
            )

    cfg = config

    # =========================================================================
    # LINEAR PHASE
    # =========================================================================

    # Step 1: Create working directory with numbered symlinks for Siril's convert
    # This handles both HDR (stack_00001.fit) and non-HDR (stack_B_180s.fit) naming
    work_dir = stacks_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    log_fn("Preparing stacks for registration...")
    for ch, idx in LRGB_CHANNEL_INDEX.items():
        src_path = stacks[ch][0].path
        link_path = work_dir / f"stack_{idx}.fit"
        link_or_copy(src_path, link_path)
        log_fn(f"  {ch}: {src_path.name} -> stack_{idx}.fit")

    siril.cd(str(work_dir))

    if is_hdr:
        # HDR stacks are pre-aligned from the same imaging session.
        # Skip cross-registration to avoid issues with different star profiles
        # between HDR-blended channels.
        log_fn("HDR mode: skipping cross-registration (stacks pre-aligned)")
        working_dir = work_dir
        # Load stacks directly with channel names
        for ch, idx in LRGB_CHANNEL_INDEX.items():
            siril.load(f"stack_{idx}")
            siril.save(ch)
            log_fn(f"  {ch}: loaded")
    else:
        # Step 2: Cross-register all stacks
        log_fn("Cross-registering stacks...")
        siril.convert("stack", out="./registered")
        siril.cd(str(work_dir / "registered"))

        if cfg.cross_reg_twopass:
            # 2-pass: Siril auto-selects best reference (ignores setref)
            log_fn("Using 2-pass registration (Siril auto-selects reference)")
            siril.register("stack", twopass=True)
        else:
            # 1-pass: Use L as reference (highest SNR, most stars)
            # L=00003 is image 3 in LRGB sequence (B=1, G=2, L=3, R=4)
            siril.setref("stack", 3)
            log_fn("Using 1-pass registration with L as reference (image 3)")
            siril.register("stack", twopass=False)

        siril.seqapplyreg("stack", framing="min")
        working_dir = work_dir / "registered"

        # Step 3: Save channels with standard names
        log_fn("Saving registered channels...")
        for ch, idx in LRGB_CHANNEL_INDEX.items():
            siril.load(f"r_stack_{idx}")
            siril.save(ch)
            log_fn(f"  {ch}: saved")

    # Post-stack background extraction (on registered channel stacks)
    if cfg.post_stack_subsky_method != SubskyMethod.NONE:
        method_desc = (
            "RBF"
            if cfg.post_stack_subsky_method == SubskyMethod.RBF
            else f"polynomial degree {cfg.post_stack_subsky_degree}"
        )
        log_fn(f"Post-stack background extraction ({method_desc})...")
        for ch in ["R", "G", "B", "L"]:
            siril.load(ch)
            if siril.subsky(
                rbf=(cfg.post_stack_subsky_method == SubskyMethod.RBF),
                degree=cfg.post_stack_subsky_degree,
                samples=cfg.post_stack_subsky_samples,
                tolerance=cfg.post_stack_subsky_tolerance,
                smooth=cfg.post_stack_subsky_smooth,
            ):
                siril.save(ch)
                log_fn(f"  {ch}: background extracted")
            else:
                log_fn(f"  {ch}: subsky failed, using original")

    # Diagnostic previews for individual stacks
    if cfg.diagnostic_previews:
        log_fn("Saving diagnostic previews...")
        for ch in ["R", "G", "B", "L"]:
            save_diagnostic_preview(siril, ch, output_dir, log_fn)

    # Step 3: Compose RGB
    log_fn("Creating RGB composite...")
    siril.rgbcomp(r="R", g="G", b="B", out="rgb")

    # Save linear RGB before calibration
    rgb_linear_path = output_dir / "rgb_linear.fit"
    siril.load("rgb")
    siril.save(str(rgb_linear_path))
    log_fn(f"Saved linear RGB: {rgb_linear_path.name}")
    log_color_balance_fn(rgb_linear_path)

    if cfg.diagnostic_previews:
        save_diagnostic_preview(siril, "rgb", output_dir, log_fn)

    # Step 4: SPCC on RGB
    log_fn("Color calibration (SPCC) on RGB...")
    rgb_spcc_path, rgb_calibrated = apply_spcc_step(
        siril, cfg, output_dir, "rgb", log_fn
    )
    siril.load(rgb_calibrated)
    siril.save("rgb_spcc")

    if cfg.diagnostic_previews:
        save_diagnostic_preview(siril, "rgb_spcc", output_dir, log_fn)

    # Step 5: Deconvolution (optional, on linear data)
    rgb_source = "rgb_spcc"
    l_source = "L"

    if cfg.deconv_enabled:
        log_fn("Deconvolving RGB...")
        rgb_source = apply_deconvolution(
            siril, "rgb_spcc", "rgb_deconv", cfg, output_dir, log_fn, "rgb"
        )
        log_fn("Deconvolving L...")
        l_source = apply_deconvolution(
            siril, "L", "L_deconv", cfg, output_dir, log_fn, "L"
        )

    # working_dir was set above based on is_hdr mode

    # =========================================================================
    # BRANCH 1: BASELINE WITH ORIGINAL STARS (always)
    # =========================================================================
    log_fn("Creating baseline outputs with original stars...")

    # --- AUTOSTRETCH VERSION ---
    log_fn("Creating AUTOSTRETCH version (original stars)...")
    lrgb_auto = _stretch_and_combine_lrgb(
        siril,
        rgb_source,
        l_source,
        "baseline",
        working_dir,
        output_dir,
        cfg,
        log_fn,
        use_veralux=False,
    )
    _save_outputs(siril, lrgb_auto, output_dir, "lrgb_autostretch", log_fn, cfg)

    # --- VERALUX VERSION ---
    log_fn("Creating VERALUX version (original stars)...")
    lrgb_veralux = _stretch_and_combine_lrgb(
        siril,
        rgb_source,
        l_source,
        "baseline",
        working_dir,
        output_dir,
        cfg,
        log_fn,
        use_veralux=True,
    )
    _save_outputs(siril, lrgb_veralux, output_dir, "lrgb_veralux", log_fn, cfg)

    # =========================================================================
    # BRANCH 2: STARNET OUTPUTS (conditional on starnet_enabled)
    # =========================================================================
    if cfg.starnet_enabled:
        log_fn("StarNet enabled - creating starless outputs...")

        # Step 6-7: Star removal on RGB and L (linear phase)
        log_fn("Removing stars from RGB (linear)...")
        siril.load(rgb_source)
        if siril.starnet(
            stretch=cfg.starnet_stretch,
            upscale=cfg.starnet_upscale,
            stride=cfg.starnet_stride,
        ):
            siril.save("rgb_starless")
            # StarNet creates starmask file in current working directory
            stars_path = working_dir / f"starmask_{rgb_source}.fit"
            if stars_path.exists():
                siril.load(str(stars_path))
                siril.save("rgb_stars")
                # Save stars checkpoint to output
                siril.save(str(output_dir / "rgb_stars"))
                log_fn("  Saved rgb_stars.fit checkpoint")
            log_fn("  RGB starless and stars saved")
        else:
            log_fn("  StarNet failed on RGB, skipping starless outputs")
            # Return early without starless outputs
            return CompositionResult(
                linear_path=output_dir / "lrgb_autostretch.fit",
                linear_pcc_path=rgb_spcc_path,
                auto_fit=output_dir / "lrgb_veralux.fit",
                auto_tif=output_dir / "lrgb_veralux.tif",
                auto_jpg=output_dir / "lrgb_veralux.jpg",
                stacks_dir=stacks_dir,
            )

        log_fn("Removing stars from L (linear)...")
        siril.load(l_source)
        if siril.starnet(
            stretch=cfg.starnet_stretch,
            upscale=cfg.starnet_upscale,
            stride=cfg.starnet_stride,
        ):
            siril.save("L_starless")
            log_fn("  L starless saved")
        else:
            log_fn("  StarNet failed on L, using original L for starless output")
            siril.load(l_source)
            siril.save("L_starless")

        # Prepare stars (clip background)
        log_fn("Preparing RGB stars (clip background)...")
        siril.load("rgb_stars")
        siril.pm("max($rgb_stars$ - 0.001, 0)")
        siril.save("rgb_stars_stretched")

        # --- AUTOSTRETCH STARLESS ---
        log_fn("Creating AUTOSTRETCH STARLESS version...")
        lrgb_auto_starless = _stretch_and_combine_lrgb(
            siril,
            "rgb_starless",
            "L_starless",
            "starless",
            working_dir,
            output_dir,
            cfg,
            log_fn,
            use_veralux=False,
        )
        _save_outputs(
            siril,
            lrgb_auto_starless,
            output_dir,
            "lrgb_autostretch_starless",
            log_fn,
            cfg,
        )

        # --- VERALUX STARLESS ---
        log_fn("Creating VERALUX STARLESS version...")
        lrgb_veralux_starless = _stretch_and_combine_lrgb(
            siril,
            "rgb_starless",
            "L_starless",
            "starless",
            working_dir,
            output_dir,
            cfg,
            log_fn,
            use_veralux=True,
        )
        _save_outputs(
            siril,
            lrgb_veralux_starless,
            output_dir,
            "lrgb_veralux_starless",
            log_fn,
            cfg,
        )

        # --- STARCOMPOSER OUTPUTS ---
        log_fn("Creating STARCOMPOSER versions...")

        # Autostretch starcomposer
        _add_stars_back(
            siril,
            lrgb_auto_starless,
            "rgb_stars_stretched",
            "lrgb_autostretch_starcomposer",
            working_dir,
            cfg,
            log_fn,
        )
        _save_outputs(
            siril,
            "lrgb_autostretch_starcomposer",
            output_dir,
            "lrgb_autostretch_starcomposer",
            log_fn,
            cfg,
        )

        # Veralux starcomposer
        _add_stars_back(
            siril,
            lrgb_veralux_starless,
            "rgb_stars_stretched",
            "lrgb_veralux_starcomposer",
            working_dir,
            cfg,
            log_fn,
        )
        _save_outputs(
            siril,
            "lrgb_veralux_starcomposer",
            output_dir,
            "lrgb_veralux_starcomposer",
            log_fn,
            cfg,
        )
    else:
        log_fn("StarNet disabled - skipping starless outputs")

    # =========================================================================
    # Return result (use veralux baseline as primary)
    # =========================================================================
    return CompositionResult(
        linear_path=output_dir / "lrgb_autostretch.fit",
        linear_pcc_path=rgb_spcc_path,
        auto_fit=output_dir / "lrgb_veralux.fit",
        auto_tif=output_dir / "lrgb_veralux.tif",
        auto_jpg=output_dir / "lrgb_veralux.jpg",
        stacks_dir=stacks_dir,
    )


def compose_rgb(
    siril: "SirilInterface",
    stacks: dict[str, list[StackInfo]],
    stacks_dir: Path,
    output_dir: Path,
    config: Config,
    log_fn: Callable[[str], None],
    log_step_fn: Callable[[str], None],
    log_color_balance_fn: Callable,
) -> CompositionResult:
    """
    Compose RGB image (no luminance channel).

    Similar workflow to LRGB but without the L channel processing.
    """
    log_step_fn("Composing RGB")

    # Verify required channels
    for ch in ["R", "G", "B"]:
        if ch not in stacks:
            raise ValueError(f"Missing required channel: {ch}")
        if len(stacks[ch]) != 1:
            raise ValueError(
                f"Expected single exposure for {ch}, got {len(stacks[ch])}"
            )

    cfg = config

    # =========================================================================
    # LINEAR PHASE
    # =========================================================================

    # Step 1: Create working directory with numbered symlinks for Siril's convert
    work_dir = stacks_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    log_fn("Preparing stacks for registration...")
    for ch, idx in RGB_CHANNEL_INDEX.items():
        src_path = stacks[ch][0].path
        link_path = work_dir / f"stack_{idx}.fit"
        link_or_copy(src_path, link_path)
        log_fn(f"  {ch}: {src_path.name} -> stack_{idx}.fit")

    siril.cd(str(work_dir))

    # Step 2: Cross-register stacks
    log_fn("Cross-registering stacks...")
    siril.convert("stack", out="./registered")
    siril.cd(str(work_dir / "registered"))

    if cfg.cross_reg_twopass:
        # 2-pass: Siril auto-selects best reference (ignores setref)
        log_fn("Using 2-pass registration (Siril auto-selects reference)")
        siril.register("stack", twopass=True)
    else:
        # 1-pass: Use R as reference (better star profiles than B)
        # R=00003 is image 3 in RGB sequence (B=1, G=2, R=3)
        siril.setref("stack", 3)
        log_fn("Using 1-pass registration with R as reference (image 3)")
        siril.register("stack", twopass=False)

    siril.seqapplyreg("stack", framing="min")

    # Step 3: Save channels with standard names
    log_fn("Saving registered channels...")
    for ch, idx in RGB_CHANNEL_INDEX.items():
        siril.load(f"r_stack_{idx}")
        siril.save(ch)
        log_fn(f"  {ch}: saved")

    # Post-stack background extraction (on registered channel stacks)
    if cfg.post_stack_subsky_method != SubskyMethod.NONE:
        method_desc = (
            "RBF"
            if cfg.post_stack_subsky_method == SubskyMethod.RBF
            else f"polynomial degree {cfg.post_stack_subsky_degree}"
        )
        log_fn(f"Post-stack background extraction ({method_desc})...")
        for ch in ["R", "G", "B"]:
            siril.load(ch)
            if siril.subsky(
                rbf=(cfg.post_stack_subsky_method == SubskyMethod.RBF),
                degree=cfg.post_stack_subsky_degree,
                samples=cfg.post_stack_subsky_samples,
                tolerance=cfg.post_stack_subsky_tolerance,
                smooth=cfg.post_stack_subsky_smooth,
            ):
                siril.save(ch)
                log_fn(f"  {ch}: background extracted")
            else:
                log_fn(f"  {ch}: subsky failed, using original")

    if cfg.diagnostic_previews:
        log_fn("Saving diagnostic previews...")
        for ch in ["R", "G", "B"]:
            save_diagnostic_preview(siril, ch, output_dir, log_fn)

    # Step 3: Compose RGB
    log_fn("Creating RGB composite...")
    siril.rgbcomp(r="R", g="G", b="B", out="rgb")

    linear_path = output_dir / "rgb_linear.fit"
    siril.load("rgb")
    siril.save(str(linear_path))
    log_fn(f"Saved linear: {linear_path.name}")
    log_color_balance_fn(linear_path)

    # Step 4: SPCC on RGB
    linear_pcc_path, rgb_calibrated = apply_spcc_step(
        siril, cfg, output_dir, "rgb", log_fn
    )
    siril.load(rgb_calibrated)
    siril.save("rgb_spcc")

    # Step 5: Deconvolution (optional)
    rgb_source = "rgb_spcc"
    if cfg.deconv_enabled:
        log_fn("Deconvolving RGB...")
        rgb_source = apply_deconvolution(
            siril, "rgb_spcc", "rgb_deconv", cfg, output_dir, log_fn, "rgb"
        )

    working_dir = work_dir / "registered"

    # =========================================================================
    # BRANCH 1: BASELINE WITH ORIGINAL STARS (always)
    # =========================================================================
    log_fn("Creating baseline outputs with original stars...")

    # --- AUTOSTRETCH VERSION ---
    log_fn("Creating AUTOSTRETCH version (original stars)...")
    siril.load(rgb_source)
    apply_stretch(siril, "autostretch", working_dir / f"{rgb_source}.fit", cfg, log_fn)
    siril.save("rgb_auto")
    _apply_post_stretch(siril, "rgb_auto", working_dir, cfg, log_fn)
    _save_outputs(siril, "rgb_auto", output_dir, "rgb_autostretch", log_fn, cfg)

    # --- VERALUX VERSION ---
    log_fn("Creating VERALUX version (original stars)...")
    siril.load(rgb_source)
    temp_path = working_dir / f"{rgb_source}_temp.fit"
    siril.save(str(working_dir / f"{rgb_source}_temp"))
    siril.load("R")  # Release file lock (Windows)
    success = apply_stretch(siril, "veralux", temp_path, cfg, log_fn)
    if not success:
        log_fn("  Veralux stretch failed, using autostretch")
        siril.load(rgb_source)
        siril.autostretch()
    siril.save("rgb_veralux")
    _apply_post_stretch(siril, "rgb_veralux", working_dir, cfg, log_fn)
    _save_outputs(siril, "rgb_veralux", output_dir, "rgb_veralux", log_fn, cfg)

    # =========================================================================
    # BRANCH 2: STARNET OUTPUTS (conditional on starnet_enabled)
    # =========================================================================
    if cfg.starnet_enabled:
        log_fn("StarNet enabled - creating starless outputs...")

        # Star removal
        log_fn("Removing stars from RGB (linear)...")
        siril.load(rgb_source)
        if siril.starnet(
            stretch=cfg.starnet_stretch,
            upscale=cfg.starnet_upscale,
            stride=cfg.starnet_stride,
        ):
            siril.save("rgb_starless")
            # StarNet creates starmask file in current working directory
            stars_path = working_dir / f"starmask_{rgb_source}.fit"
            if stars_path.exists():
                siril.load(str(stars_path))
                siril.save("rgb_stars")
                siril.save(str(output_dir / "rgb_stars"))
        else:
            log_fn("StarNet failed, skipping starless outputs")
            return CompositionResult(
                linear_path=linear_path,
                linear_pcc_path=linear_pcc_path,
                auto_fit=output_dir / "rgb_veralux.fit",
                auto_tif=output_dir / "rgb_veralux.tif",
                auto_jpg=output_dir / "rgb_veralux.jpg",
                stacks_dir=stacks_dir,
            )

        # Prepare stars
        siril.load("rgb_stars")
        siril.pm("max($rgb_stars$ - 0.001, 0)")
        siril.save("rgb_stars_stretched")

        # --- AUTOSTRETCH STARLESS ---
        log_fn("Creating AUTOSTRETCH STARLESS version...")
        siril.load("rgb_starless")
        siril.autostretch()
        siril.save("rgb_auto_starless")
        _apply_post_stretch(siril, "rgb_auto_starless", working_dir, cfg, log_fn)
        _save_outputs(
            siril,
            "rgb_auto_starless",
            output_dir,
            "rgb_autostretch_starless",
            log_fn,
            cfg,
        )

        # --- VERALUX STARLESS ---
        log_fn("Creating VERALUX STARLESS version...")
        siril.load("rgb_starless")
        temp_path = working_dir / "rgb_starless_temp.fit"
        siril.save(str(working_dir / "rgb_starless_temp"))
        siril.load("R")  # Release file lock (Windows)
        success = apply_stretch(siril, "veralux", temp_path, cfg, log_fn)
        if not success:
            siril.load("rgb_starless")
            siril.autostretch()
        siril.save("rgb_veralux_starless")
        _apply_post_stretch(siril, "rgb_veralux_starless", working_dir, cfg, log_fn)
        _save_outputs(
            siril,
            "rgb_veralux_starless",
            output_dir,
            "rgb_veralux_starless",
            log_fn,
            cfg,
        )

        # --- STARCOMPOSER ---
        log_fn("Creating STARCOMPOSER versions...")
        _add_stars_back(
            siril,
            "rgb_auto_starless",
            "rgb_stars_stretched",
            "rgb_autostretch_starcomposer",
            working_dir,
            cfg,
            log_fn,
        )
        _save_outputs(
            siril,
            "rgb_autostretch_starcomposer",
            output_dir,
            "rgb_autostretch_starcomposer",
            log_fn,
            cfg,
        )

        _add_stars_back(
            siril,
            "rgb_veralux_starless",
            "rgb_stars_stretched",
            "rgb_veralux_starcomposer",
            working_dir,
            cfg,
            log_fn,
        )
        _save_outputs(
            siril,
            "rgb_veralux_starcomposer",
            output_dir,
            "rgb_veralux_starcomposer",
            log_fn,
            cfg,
        )
    else:
        log_fn("StarNet disabled - skipping starless outputs")

    return CompositionResult(
        linear_path=linear_path,
        linear_pcc_path=linear_pcc_path,
        auto_fit=output_dir / "rgb_veralux.fit",
        auto_tif=output_dir / "rgb_veralux.tif",
        auto_jpg=output_dir / "rgb_veralux.jpg",
        stacks_dir=stacks_dir,
    )
