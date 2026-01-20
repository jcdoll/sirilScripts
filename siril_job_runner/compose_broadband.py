"""
Broadband composition for LRGB and RGB imaging.

Handles composition of L, R, G, B filter stacks into color images.

LRGB Workflow (see docs/lrgb_workflow.md for details):

LINEAR PHASE:
 1. Cross-register all stacks (R, G, B, L)
 2. Background extraction (subsky) on EACH channel
 3. Compose RGB (NO linear matching between channels)
 4. Color calibration (SPCC) on RGB
 5. Deconvolution on RGB and L (optional)

STRETCH PHASE (always - baseline with original stars):
 6. Stretch RGB and L
 7. Combine LRGB
 8. Color removal (SCNR)
 9. Output baseline images (lrgb_autostretch, lrgb_veralux)

STARNET PHASE (conditional - if starnet_enabled):
10. Star removal on RGB -> RGB_starless + RGB_stars
11. Star removal on L -> L_starless
12. Stretch starless images
13. Combine LRGB from starless
14. Output starless images
15. Recompose stars (if starcomposer_enabled)
16. Output starcomposer images

Key principles:
- Background extraction per-channel eliminates need for linear matching
- SPCC must run on RGB before adding L
- Baseline outputs always have original stars (no StarNet artifacts)
- StarNet branch is optional and conditional on starnet_enabled config
"""

from pathlib import Path
from typing import TYPE_CHECKING

from .compose_helpers import apply_spcc_step
from .config import Config
from .models import CompositionResult, StackInfo
from .psf_analysis import analyze_psf, format_psf_stats

if TYPE_CHECKING:
    from .protocols import SirilInterface


# Mapping from channel name to alphabetical index (for registered sequences)
# LRGB: B=00001, G=00002, L=00003, R=00004
LRGB_CHANNEL_INDEX = {"B": "00001", "G": "00002", "L": "00003", "R": "00004"}
# RGB: B=00001, G=00002, R=00003
RGB_CHANNEL_INDEX = {"B": "00001", "G": "00002", "R": "00003"}


def _save_diagnostic_preview(
    siril: "SirilInterface",
    name: str,
    output_dir: Path,
    log_fn: callable,
) -> None:
    """Save a stretched preview image for diagnostics."""
    siril.load(name)
    siril.autostretch()
    preview_stem = output_dir / f"diag_{name}"
    siril.savejpg(str(preview_stem))
    log_fn(f"Diagnostic preview: diag_{name}.jpg")


def _apply_deconvolution(
    siril: "SirilInterface",
    source_name: str,
    output_name: str,
    config: Config,
    output_dir: Path,
    log_fn: callable,
    psf_suffix: str,
) -> str:
    """Apply deconvolution to an image. Returns the name of the result."""
    cfg = config
    siril.load(source_name)

    psf_path = (
        str(output_dir / f"psf_{psf_suffix}.fit") if cfg.deconv_save_psf else None
    )

    if siril.makepsf(
        method=cfg.deconv_psf_method,
        symmetric=True,
        save_psf=psf_path,
    ):
        if psf_path:
            log_fn(f"PSF saved: psf_{psf_suffix}.fit")
            psf_stats = analyze_psf(Path(psf_path))
            if psf_stats:
                for line in format_psf_stats(psf_stats):
                    log_fn(f"  {line}")

        if siril.rl(
            iters=cfg.deconv_iterations,
            regularization=cfg.deconv_regularization,
            alpha=cfg.deconv_alpha,
        ):
            siril.save(output_name)
            return output_name

        log_fn("Deconvolution failed, using original")
    else:
        log_fn("PSF creation failed, skipping deconvolution")

    return source_name


def _apply_veralux_processing(
    siril: "SirilInterface",
    source_name: str,
    working_dir: Path,
    config: Config,
    log_fn: callable,
) -> None:
    """Apply VeraLux post-processing (Silentium, Revela, Vectra) to stretched image."""
    from .veralux_revela import apply_revela
    from .veralux_silentium import apply_silentium
    from .veralux_vectra import apply_vectra

    # Save current state to file for veralux processing
    # Siril's save command adds .fit automatically, so pass path WITHOUT extension
    siril.load(source_name)
    image_path = working_dir / f"{source_name}.fit"
    siril.save(str(working_dir / source_name))  # Siril adds .fit -> saves to image_path

    if config.veralux_silentium_enabled:
        log_fn("Applying VeraLux Silentium (noise reduction)...")
        success, stats = apply_silentium(siril, image_path, config, log_fn)
        if success and stats:
            log_fn(
                f"Silentium applied: L_noise={stats.get('L_noise', 0):.4f}, "
                f"chroma_noise={stats.get('chroma_noise', 0):.4f}"
            )

    if config.veralux_revela_enabled:
        log_fn("Applying VeraLux Revela (detail enhancement)...")
        success, stats = apply_revela(siril, image_path, config, log_fn)
        if success and stats:
            log_fn(
                f"Revela applied: texture_boost={stats.get('texture_boost', 0):.2f}, "
                f"structure_boost={stats.get('structure_boost', 0):.2f}"
            )

    if config.veralux_vectra_enabled:
        log_fn("Applying VeraLux Vectra (smart saturation)...")
        success, stats = apply_vectra(siril, image_path, config, log_fn)
        if success and stats:
            log_fn(
                f"Vectra applied: base_boost={stats.get('base_boost', 0):.2f}, "
                f"mean_boost={stats.get('mean_boost', 0):.2f}"
            )

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
    log_fn: callable,
) -> None:
    """Add stars back to starless image using screen blend or StarComposer."""
    if config.veralux_starcomposer_enabled:
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
    else:
        # Simple screen blend: result = 1 - (1 - starless) * (1 - stars)
        log_fn("Adding stars back with screen blend...")
        siril.load(starless_name)
        siril.pm(f"1 - (1 - ${starless_name}$) * (1 - ${stars_name}$)")
        siril.save(output_name)


def _stretch_and_combine_lrgb(
    siril: "SirilInterface",
    rgb_source: str,
    l_source: str,
    output_prefix: str,
    working_dir: Path,
    output_dir: Path,
    config: Config,
    log_fn: callable,
    use_veralux: bool,
) -> str:
    """Stretch RGB and L, combine to LRGB, apply SCNR. Returns the combined image name."""
    from . import veralux_stretch

    suffix = "veralux" if use_veralux else "auto"
    rgb_stretched = f"rgb_{output_prefix}_{suffix}"
    l_stretched = f"L_{output_prefix}_{suffix}"
    lrgb_combined = f"lrgb_{output_prefix}_{suffix}"

    if use_veralux:
        # Stretch RGB with veralux
        siril.load(rgb_source)
        temp_path = working_dir / f"{rgb_source}_temp.fit"
        siril.save(str(working_dir / f"{rgb_source}_temp"))
        # Load a different image to release file lock on temp file (Windows issue)
        siril.load(l_source)
        success, log_d = veralux_stretch.apply_stretch(siril, temp_path, config, log_fn)
        if success:
            log_fn(f"  RGB veralux stretch applied (log_d={log_d:.2f})")
            siril.save(rgb_stretched)
        else:
            log_fn("  RGB veralux stretch failed, using autostretch")
            siril.load(rgb_source)
            siril.autostretch()
            siril.save(rgb_stretched)

        # Stretch L with veralux
        siril.load(l_source)
        temp_path = working_dir / f"{l_source}_temp.fit"
        siril.save(str(working_dir / f"{l_source}_temp"))
        # Load a different image to release file lock on temp file (Windows issue)
        siril.load(rgb_source)
        success, log_d = veralux_stretch.apply_stretch(siril, temp_path, config, log_fn)
        if success:
            log_fn(f"  L veralux stretch applied (log_d={log_d:.2f})")
            siril.save(l_stretched)
        else:
            log_fn("  L veralux stretch failed, using autostretch")
            siril.load(l_source)
            siril.autostretch()
            siril.save(l_stretched)
    else:
        # Stretch with Siril autostretch
        siril.load(rgb_source)
        siril.autostretch()
        siril.save(rgb_stretched)

        siril.load(l_source)
        siril.autostretch()
        siril.save(l_stretched)

    # Combine LRGB
    siril.rgbcomp(lum=l_stretched, rgb=rgb_stretched, out=lrgb_combined)

    # SCNR
    if config.color_removal_mode == "green":
        siril.load(lrgb_combined)
        siril.rmgreen(type=0, amount=1.0)
        siril.save(lrgb_combined)

    # Apply veralux post-processing for veralux branch
    if use_veralux:
        _apply_veralux_processing(siril, lrgb_combined, working_dir, config, log_fn)

    return lrgb_combined


def _save_outputs(
    siril: "SirilInterface",
    source_name: str,
    output_dir: Path,
    output_basename: str,
    log_fn: callable,
) -> None:
    """Save FIT, TIF, and JPG outputs."""
    siril.load(source_name)
    fit_path = output_dir / f"{output_basename}.fit"
    siril.save(str(fit_path))
    siril.savetif(str(output_dir / output_basename))
    siril.savejpg(str(output_dir / output_basename))
    log_fn(f"Saved: {output_basename}.fit, .tif, .jpg")


def compose_lrgb(
    siril: "SirilInterface",
    stacks: dict[str, list[StackInfo]],
    stacks_dir: Path,
    output_dir: Path,
    config: Config,
    stretch_pipeline,  # Not used directly anymore, but kept for API compatibility
    log_fn: callable,
    log_step_fn: callable,
    log_color_balance_fn: callable,
) -> CompositionResult:
    """
    Compose LRGB image from stacked channels.

    See module docstring for full workflow description.
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
    siril.cd(str(stacks_dir))

    # =========================================================================
    # LINEAR PHASE
    # =========================================================================

    # Step 1: Cross-register all stacks
    log_fn("Cross-registering stacks...")
    siril.convert("stack", out="./registered")
    siril.cd(str(stacks_dir / "registered"))
    siril.register("stack", twopass=True)
    siril.seqapplyreg("stack", framing="min")

    # Step 2: Save channels and apply background extraction
    log_fn("Background extraction on each channel...")
    for ch, idx in LRGB_CHANNEL_INDEX.items():
        siril.load(f"r_stack_{idx}")
        siril.subsky(
            rbf=True,
            samples=cfg.subsky_samples,
            tolerance=cfg.subsky_tolerance,
            smooth=cfg.subsky_smooth,
        )
        siril.save(ch)
        log_fn(f"  {ch}: subsky applied")

    # Diagnostic previews for individual stacks
    if cfg.diagnostic_previews:
        log_fn("Saving diagnostic previews...")
        for ch in ["R", "G", "B", "L"]:
            _save_diagnostic_preview(siril, ch, output_dir, log_fn)

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
        _save_diagnostic_preview(siril, "rgb", output_dir, log_fn)

    # Step 4: SPCC on RGB
    log_fn("Color calibration (SPCC) on RGB...")
    rgb_spcc_path, rgb_calibrated = apply_spcc_step(
        siril, cfg, output_dir, "rgb", log_fn
    )
    siril.load(rgb_calibrated)
    siril.save("rgb_spcc")

    if cfg.diagnostic_previews:
        _save_diagnostic_preview(siril, "rgb_spcc", output_dir, log_fn)

    # Step 5: Deconvolution (optional, on linear data)
    rgb_source = "rgb_spcc"
    l_source = "L"

    if cfg.deconv_enabled:
        log_fn("Deconvolving RGB...")
        rgb_source = _apply_deconvolution(
            siril, "rgb_spcc", "rgb_deconv", cfg, output_dir, log_fn, "rgb"
        )
        log_fn("Deconvolving L...")
        l_source = _apply_deconvolution(
            siril, "L", "L_deconv", cfg, output_dir, log_fn, "L"
        )

    working_dir = stacks_dir / "registered"

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
    _save_outputs(siril, lrgb_auto, output_dir, "lrgb_autostretch", log_fn)

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
    _save_outputs(siril, lrgb_veralux, output_dir, "lrgb_veralux", log_fn)

    # =========================================================================
    # BRANCH 2: STARNET OUTPUTS (conditional on starnet_enabled)
    # =========================================================================
    if cfg.starnet_enabled:
        log_fn("StarNet enabled - creating starless outputs...")

        # Step 6-7: Star removal on RGB and L (linear phase)
        log_fn("Removing stars from RGB (linear)...")
        siril.load(rgb_source)
        if siril.starnet():
            siril.save("rgb_starless")
            # StarNet creates starmask file
            stars_path = stacks_dir / "registered" / f"starmask_{rgb_source}.fit"
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
        if siril.starnet():
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
            siril, lrgb_auto_starless, output_dir, "lrgb_autostretch_starless", log_fn
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
            siril, lrgb_veralux_starless, output_dir, "lrgb_veralux_starless", log_fn
        )

        # --- STARCOMPOSER OUTPUTS (if enabled) ---
        if cfg.veralux_starcomposer_enabled:
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
    stretch_pipeline,
    log_fn: callable,
    log_step_fn: callable,
    log_color_balance_fn: callable,
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
    siril.cd(str(stacks_dir))

    # =========================================================================
    # LINEAR PHASE
    # =========================================================================

    # Step 1: Cross-register stacks
    log_fn("Cross-registering stacks...")
    siril.convert("stack", out="./registered")
    siril.cd(str(stacks_dir / "registered"))
    siril.register("stack", twopass=True)
    siril.seqapplyreg("stack", framing="min")

    # Step 2: Save channels with background extraction
    log_fn("Background extraction on each channel...")
    for ch, idx in RGB_CHANNEL_INDEX.items():
        siril.load(f"r_stack_{idx}")
        siril.subsky(
            rbf=True,
            samples=cfg.subsky_samples,
            tolerance=cfg.subsky_tolerance,
            smooth=cfg.subsky_smooth,
        )
        siril.save(ch)
        log_fn(f"  {ch}: subsky applied")

    if cfg.diagnostic_previews:
        log_fn("Saving diagnostic previews...")
        for ch in ["R", "G", "B"]:
            _save_diagnostic_preview(siril, ch, output_dir, log_fn)

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
        rgb_source = _apply_deconvolution(
            siril, "rgb_spcc", "rgb_deconv", cfg, output_dir, log_fn, "rgb"
        )

    working_dir = stacks_dir / "registered"

    # =========================================================================
    # BRANCH 1: BASELINE WITH ORIGINAL STARS (always)
    # =========================================================================
    log_fn("Creating baseline outputs with original stars...")
    from . import veralux_stretch

    # --- AUTOSTRETCH VERSION ---
    log_fn("Creating AUTOSTRETCH version (original stars)...")
    siril.load(rgb_source)
    siril.autostretch()
    siril.save("rgb_auto")
    if cfg.color_removal_mode == "green":
        siril.load("rgb_auto")
        siril.rmgreen(type=0, amount=1.0)
        siril.save("rgb_auto")
    _save_outputs(siril, "rgb_auto", output_dir, "rgb_autostretch", log_fn)

    # --- VERALUX VERSION ---
    log_fn("Creating VERALUX version (original stars)...")
    siril.load(rgb_source)
    temp_path = working_dir / f"{rgb_source}_temp.fit"
    siril.save(str(working_dir / f"{rgb_source}_temp"))
    # Load a different image to release file lock on temp file (Windows issue)
    siril.load("R")
    success, log_d = veralux_stretch.apply_stretch(siril, temp_path, cfg, log_fn)
    if success:
        log_fn(f"  Veralux stretch applied (log_d={log_d:.2f})")
        siril.save("rgb_veralux")
    else:
        log_fn("  Veralux stretch failed, using autostretch")
        siril.load(rgb_source)
        siril.autostretch()
        siril.save("rgb_veralux")
    if cfg.color_removal_mode == "green":
        siril.load("rgb_veralux")
        siril.rmgreen(type=0, amount=1.0)
        siril.save("rgb_veralux")
    _apply_veralux_processing(siril, "rgb_veralux", working_dir, cfg, log_fn)
    _save_outputs(siril, "rgb_veralux", output_dir, "rgb_veralux", log_fn)

    # =========================================================================
    # BRANCH 2: STARNET OUTPUTS (conditional on starnet_enabled)
    # =========================================================================
    if cfg.starnet_enabled:
        log_fn("StarNet enabled - creating starless outputs...")

        # Star removal
        log_fn("Removing stars from RGB (linear)...")
        siril.load(rgb_source)
        if siril.starnet():
            siril.save("rgb_starless")
            stars_path = stacks_dir / "registered" / f"starmask_{rgb_source}.fit"
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
        if cfg.color_removal_mode == "green":
            siril.load("rgb_auto_starless")
            siril.rmgreen(type=0, amount=1.0)
            siril.save("rgb_auto_starless")
        _save_outputs(
            siril, "rgb_auto_starless", output_dir, "rgb_autostretch_starless", log_fn
        )

        # --- VERALUX STARLESS ---
        log_fn("Creating VERALUX STARLESS version...")
        siril.load("rgb_starless")
        temp_path = working_dir / "rgb_starless_temp.fit"
        siril.save(str(working_dir / "rgb_starless_temp"))
        # Load a different image to release file lock on temp file (Windows issue)
        siril.load("R")
        success, log_d = veralux_stretch.apply_stretch(siril, temp_path, cfg, log_fn)
        if success:
            siril.save("rgb_veralux_starless")
        else:
            siril.load("rgb_starless")
            siril.autostretch()
            siril.save("rgb_veralux_starless")
        if cfg.color_removal_mode == "green":
            siril.load("rgb_veralux_starless")
            siril.rmgreen(type=0, amount=1.0)
            siril.save("rgb_veralux_starless")
        _apply_veralux_processing(
            siril, "rgb_veralux_starless", working_dir, cfg, log_fn
        )
        _save_outputs(
            siril, "rgb_veralux_starless", output_dir, "rgb_veralux_starless", log_fn
        )

        # --- STARCOMPOSER ---
        if cfg.veralux_starcomposer_enabled:
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
