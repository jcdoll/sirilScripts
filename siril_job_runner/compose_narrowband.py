"""
Narrowband composition for SHO/HOO imaging.

Handles composition of H, S, O filter stacks into palette-mapped images.
Supports optional L channel for LSHO/LHOO variants.

Processing flow:
    1. Cross-register all channel stacks
    2. Post-stack background extraction (subsky)
    3. Channel balancing (linear_match to H)
    4. StarNet on linear data (if enabled) -> {ch}_linear_starless + stars
    5. For each stretch method (autostretch, veralux):
       a. Stretch channels (starless if available)
       b. Apply palette formulas -> narrowband RGB
       c. Add L as luminance (if LSHO/LHOO)
       d. Color removal (SCNR)
       e. Background neutralization
       f. Save starless output (if starnet enabled)
       g. Composite stars back (if starnet enabled)
       h. Apply saturation
       i. Save final output
"""

from pathlib import Path
from typing import TYPE_CHECKING, Callable

from .compose_helpers import (
    apply_color_removal,
    apply_deconvolution,
    neutralize_rgb_background,
    save_diagnostic_preview,
)
from .config import (
    ColorRemovalMode,
    Config,
    StarColor,
    StarSource,
    StretchMethod,
    SubskyMethod,
)
from .models import CompositionResult, StackInfo
from .palettes import build_effective_palette, formula_to_pixelmath, get_palette
from .siril_file_ops import link_or_copy
from .stretch_helpers import (
    apply_enhancements,
    apply_saturation,
    apply_stretch,
    save_all_formats,
)

if TYPE_CHECKING:
    from .protocols import SirilInterface


def _separate_stars_linear(
    siril: "SirilInterface",
    cfg: Config,
    has_luminance: bool,
    all_channels: list[str],
    registered_dir: Path,
    log_fn: Callable[[str], None],
) -> str | None:
    """
    Separate stars from all linear channels using StarNet.

    Runs on {ch}_linear files with stretch=True (internal MTF stretch).
    Creates {ch}_linear_starless for each channel and stars_source from the
    designated star channel.

    Returns the channel name used as star source (for later compositing),
    or None if star separation failed.
    """
    log_fn("  Star separation enabled, running StarNet on linear data...")

    # Determine star source: L if available, else H, or configured channel
    if cfg.narrowband_star_source == StarSource.AUTO:
        star_source_ch = "L" if has_luminance else "H"
    else:
        star_source_ch = cfg.narrowband_star_source

    # Run StarNet on all linear channels
    for ch in all_channels:
        siril.load(f"{ch}_linear")
        if siril.starnet(stretch=True):  # Linear data, needs internal stretch
            siril.save(f"{ch}_linear_starless")
            # StarNet creates starmask_<filename>.fit in the working directory
            starmask_path = registered_dir / f"starmask_{ch}_linear.fit"
            if starmask_path.exists() and ch == star_source_ch:
                siril.load(str(starmask_path))
                siril.save("stars_source")
                log_fn(f"    {ch}: starless saved, stars extracted (source)")
            else:
                log_fn(f"    {ch}: starless saved")
        else:
            log_fn(f"    {ch}: StarNet failed, using original")
            siril.load(f"{ch}_linear")
            siril.save(f"{ch}_linear_starless")

    return star_source_ch


def _apply_scale_expressions(
    siril: "SirilInterface",
    cfg: Config,
    narrowband_channels: list[str],
    log_fn: Callable[[str], None],
) -> None:
    """
    Apply channel scale expressions before palette formulas.

    Scale expressions allow non-linear channel manipulation (e.g., O + k*O^3)
    to boost signal in a way that survives linear background neutralization.
    Operates on stretched channels saved with their base names (H, O, S).
    """
    scale_exprs = {
        "H": cfg.palette_h_scale_expr,
        "O": cfg.palette_o_scale_expr,
        "S": cfg.palette_s_scale_expr,
    }
    for ch in narrowband_channels:
        expr = scale_exprs.get(ch)
        if expr:
            siril.load(ch)
            pm_expr = formula_to_pixelmath(expr)
            siril.pm(pm_expr)
            siril.save(ch)
            log_fn(f"  {ch}: scaled with {expr}")


def _composite_stars(
    siril: "SirilInterface",
    cfg: Config,
    stars_image: str,
    log_fn: Callable[[str], None],
) -> None:
    """
    Composite stars back onto the processed narrowband image.

    Stars are blended using screen blend (1 - (1-a)*(1-b)) which is the
    mathematically correct way to add light sources without harsh clipping.

    Args:
        stars_image: Name of the stars image to composite (e.g. "stars_prepared")
    """
    log_fn("  Compositing stars back...")
    siril.load("narrowband")
    siril.save("narrowband_starless")

    # Prepare stars: mono (grayscale) creates white stars
    siril.load(stars_image)
    if cfg.narrowband_star_color == StarColor.MONO:
        siril.save("_stars_mono")
        siril.rgbcomp(
            r="_stars_mono",
            g="_stars_mono",
            b="_stars_mono",
            out="_stars_rgb",
        )
        log_fn("    Using stars as grayscale (white)")
    else:
        siril.save("_stars_rgb")
        log_fn("    Using stars with native color")

    # Screen blend: 1 - (1-a)*(1-b)
    siril.load("narrowband_starless")
    siril.pm("1-(1-$narrowband_starless$)*(1-$_stars_rgb$)")
    siril.save("narrowband")
    log_fn("    Stars composited (screen blend)")


def compose_narrowband(
    siril: "SirilInterface",
    stacks: dict[str, list[StackInfo]],
    stacks_dir: Path,
    output_dir: Path,
    config: Config,
    job_type: str,
    log_fn: Callable[[str], None],
    log_step_fn: Callable[[str], None],
    log_color_balance_fn: Callable,
) -> CompositionResult:
    """
    Compose narrowband image using palette mapping.

    Job types determine channel requirements:
    - HOO: H, O channels
    - SHO: S, H, O channels
    - LHOO: L, H, O channels (L as luminance)
    - LSHO: L, S, H, O channels (L as luminance)

    Palette (from config) determines color mapping:
    - HOO: H->R, O->G, O->B (direct)
    - SHO: S->R, H->G, O->B (direct)
    - SHO_FORAXX: S->R, 0.5*H+0.5*O->G, O->B (blended)

    When starnet_enabled=True, outputs per stretch method:
    - {type}_stars.fit/tif/jpg (once, before stretch loop)
    - {type}_starless_{method}.fit/tif/jpg
    - {type}_auto_{method}.fit/tif/jpg (with stars composited)

    When starnet_enabled=False:
    - {type}_auto_{method}.fit/tif/jpg only
    """
    # Determine if job uses luminance channel
    has_luminance = job_type.startswith("L")

    # Get palette from config and apply any overrides
    palette = get_palette(config.palette)
    palette = build_effective_palette(
        palette,
        r_override=config.palette_r_override,
        g_override=config.palette_g_override,
        b_override=config.palette_b_override,
    )

    log_step_fn(f"Composing narrowband ({job_type}, palette: {palette.name})")

    # Required channels = palette requirements + L if job type includes it
    required = set(palette.required)
    if has_luminance:
        required.add("L")

    for ch in required:
        if ch not in stacks:
            raise ValueError(f"Missing required channel: {ch}")
        if len(stacks[ch]) != 1:
            raise ValueError(
                f"Expected single exposure for {ch}, got {len(stacks[ch])}"
            )

    # Build channel index mapping from sorted required channels
    channels_sorted = sorted(required)
    channel_to_num = {ch: f"{i + 1:05d}" for i, ch in enumerate(channels_sorted)}

    # Create working directory with numbered files for Siril's convert
    work_dir = stacks_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    log_fn("Preparing stacks for registration...")
    for ch, idx in channel_to_num.items():
        src_path = stacks[ch][0].path
        link_path = work_dir / f"stack_{idx}.fit"
        link_or_copy(src_path, link_path)
        log_fn(f"  {ch}: {src_path.name} -> stack_{idx}.fit")

    siril.cd(str(work_dir))

    # Register stacks
    log_fn("Cross-registering stacks...")
    siril.convert("stack", out="./registered")
    siril.cd(str(work_dir / "registered"))

    cfg = config
    if cfg.cross_reg_twopass:
        log_fn("Using 2-pass registration (Siril auto-selects reference)")
        siril.register("stack", twopass=True)
    else:
        ref_ch = "L" if has_luminance else "H"
        ref_idx = int(channel_to_num[ref_ch])
        siril.setref("stack", ref_idx)
        log_fn(
            f"Using 1-pass registration with {ref_ch} as reference (image {ref_idx})"
        )
        siril.register("stack", twopass=False)

    siril.seqapplyreg("stack", framing="min")

    # Save registered channels
    log_fn("Saving registered channels...")
    for ch, idx in channel_to_num.items():
        siril.load(f"r_stack_{idx}")
        siril.save(ch)
        log_fn(f"  {ch}: saved")

    # Post-stack background extraction
    if cfg.post_stack_subsky_method != SubskyMethod.NONE:
        method_desc = (
            "RBF"
            if cfg.post_stack_subsky_method == SubskyMethod.RBF
            else f"polynomial degree {cfg.post_stack_subsky_degree}"
        )
        log_fn(f"Post-stack background extraction ({method_desc})...")
        for ch in required:
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

    # Narrowband channel balancing (linear_match to equalize backgrounds)
    if cfg.narrowband_balance_enabled:
        ref_ch = cfg.narrowband_balance_reference
        if ref_ch in required:
            channels_to_match = [ch for ch in required if ch != ref_ch and ch != "L"]
            if channels_to_match:
                log_fn(
                    f"Balancing channels to {ref_ch} "
                    f"(bounds: {cfg.narrowband_balance_low:.2f}-{cfg.narrowband_balance_high:.2f})..."
                )
                siril.load(ref_ch)
                for ch in channels_to_match:
                    siril.load(ch)
                    if siril.linear_match(
                        ref=ref_ch,
                        low=cfg.narrowband_balance_low,
                        high=cfg.narrowband_balance_high,
                    ):
                        siril.save(ch)
                        log_fn(f"  {ch}: matched to {ref_ch}")
                    else:
                        log_fn(f"  {ch}: linear_match failed, using original")
        else:
            log_fn(
                f"WARNING: Reference channel {ref_ch} not in stacks, skipping balance"
            )

    # Diagnostic previews for individual stacks (linear)
    if cfg.diagnostic_previews:
        log_fn("Saving diagnostic previews...")
        for ch in required:
            save_diagnostic_preview(siril, ch, output_dir, log_fn)

    # Save linear channels for stretch comparison and reprocessing
    narrowband_channels = [ch for ch in required if ch != "L"]
    all_channels = list(narrowband_channels)
    if has_luminance:
        all_channels.append("L")
    registered_dir = work_dir / "registered"

    log_fn("Saving linear channels...")
    for ch in all_channels:
        siril.load(ch)
        siril.save(f"{ch}_linear")

    # LinearFit to weakest channel (optional, for dynamic palettes)
    if cfg.palette_linearfit_to_weakest:
        channel_max = {}
        for ch in narrowband_channels:
            ch_path = registered_dir / f"{ch}_linear.fit"
            stats = siril.get_image_stats(ch_path)
            channel_max[ch] = stats["max"]
            log_fn(f"  {ch}: max={stats['max']:.4f}")

        ref_ch = min(channel_max, key=channel_max.get)
        channels_to_fit = [ch for ch in narrowband_channels if ch != ref_ch]

        if channels_to_fit:
            log_fn(
                f"LinearFit to weakest channel ({ref_ch}, max={channel_max[ref_ch]:.4f})..."
            )
            for ch in channels_to_fit:
                siril.load(f"{ch}_linear")
                if siril.linear_match(ref=f"{ref_ch}_linear"):
                    siril.save(f"{ch}_linear")
                    log_fn(f"  {ch}: fitted to {ref_ch}")
                else:
                    log_fn(f"  {ch}: linear_match failed, using original")

    # Helper to apply palette to current channels
    def apply_palette_combination():
        """Apply palette formulas to currently loaded/saved channels."""
        if palette.is_simple():
            siril.rgbcomp(r=palette.r, g=palette.g, b=palette.b, out="narrowband_rgb")
        else:
            r_pm = formula_to_pixelmath(palette.r)
            g_pm = formula_to_pixelmath(palette.g)
            b_pm = formula_to_pixelmath(palette.b)
            siril.pm(r_pm)
            siril.save("_pm_r")
            siril.pm(g_pm)
            siril.save("_pm_g")
            siril.pm(b_pm)
            siril.save("_pm_b")
            siril.rgbcomp(r="_pm_r", g="_pm_g", b="_pm_b", out="narrowband_rgb")

        if has_luminance:
            siril.rgbcomp(lum="L", rgb="narrowband_rgb", out="narrowband")
        else:
            siril.load("narrowband_rgb")
            siril.save("narrowband")

    # Log palette info
    log_fn(
        f"Applying {palette.name} palette: R={palette.r}, G={palette.g}, B={palette.b}"
    )
    if not palette.is_simple():
        r_pm = formula_to_pixelmath(palette.r)
        g_pm = formula_to_pixelmath(palette.g)
        b_pm = formula_to_pixelmath(palette.b)
        log_fn(f"  PixelMath: R={r_pm}, G={g_pm}, B={b_pm}")

    type_name = job_type.lower()

    # Optional deconvolution on linear channels
    if cfg.deconv_enabled:
        log_fn("Deconvolving linear channels...")
        for ch in narrowband_channels:
            deconv_result = apply_deconvolution(
                siril,
                f"{ch}_linear",
                f"{ch}_linear_deconv",
                cfg,
                output_dir,
                log_fn,
                ch,
            )
            if deconv_result == f"{ch}_linear_deconv":
                siril.load(f"{ch}_linear_deconv")
                siril.save(f"{ch}_linear")
                log_fn(f"  {ch}: deconvolved")

    # ==========================================================================
    # STAR SEPARATION (on linear data, before stretch)
    # ==========================================================================
    star_source_ch = None
    if cfg.starnet_enabled:
        star_source_ch = _separate_stars_linear(
            siril, cfg, has_luminance, all_channels, registered_dir, log_fn
        )
        # Save stars output
        if star_source_ch:
            siril.load("stars_source")
            stars_name = f"{type_name}_stars"
            save_all_formats(siril, output_dir, stars_name, log_fn)

    # ==========================================================================
    # UNIFIED PROCESSING LOOP
    # For each stretch method: stretch -> palette -> neutralize -> saturation
    # ==========================================================================
    methods = (
        [StretchMethod.AUTOSTRETCH, StretchMethod.VERALUX]
        if cfg.stretch_compare
        else [cfg.stretch_method]
    )
    if cfg.stretch_compare:
        log_fn("Comparing stretch methods (autostretch vs veralux)...")

    primary_paths = None
    linear_path = output_dir / f"{type_name}_linear.fit"

    for method in methods:
        suffix = f"_{method}" if cfg.stretch_compare else ""
        log_fn(f"Processing with {method} stretch...")

        # Step 1: Stretch each channel (use starless if available)
        for ch in all_channels:
            if cfg.starnet_enabled and star_source_ch:
                siril.load(f"{ch}_linear_starless")
                ch_path = registered_dir / f"{ch}_linear_starless.fit"
            else:
                siril.load(f"{ch}_linear")
                ch_path = registered_dir / f"{ch}_linear.fit"
            apply_stretch(siril, method, ch_path, cfg, log_fn)
            siril.save(ch)
            log_fn(f"  {ch}: stretched ({method})")

        # Step 2: Apply channel scale expressions
        _apply_scale_expressions(siril, cfg, narrowband_channels, log_fn)

        # Step 3: Apply palette formulas
        apply_palette_combination()

        # Save linear composite (first iteration only)
        if primary_paths is None:
            siril.load("narrowband")
            siril.save(str(linear_path))
            log_fn(f"Saved linear: {linear_path.name}")
            log_color_balance_fn(linear_path)

        # Step 4: Color removal (SCNR)
        if cfg.color_removal_mode != ColorRemovalMode.NONE:
            siril.load("narrowband")
            apply_color_removal(siril, cfg, log_fn)
            siril.save("narrowband")

        # Step 5: Neutralize RGB background
        if cfg.narrowband_neutralization:
            neutralize_rgb_background(siril, "narrowband", cfg, log_fn)

        # Step 6: Save starless output (if star separation enabled)
        if cfg.starnet_enabled and star_source_ch:
            siril.load("narrowband")
            starless_name = f"{type_name}_starless{suffix}"
            save_all_formats(siril, output_dir, starless_name, log_fn, cfg)

        # Step 7: Composite stars back (if separated)
        if cfg.starnet_enabled and star_source_ch:
            # Prepare stars: clip background noise
            siril.load("stars_source")
            siril.pm("max($stars_source$ - 0.001, 0)")
            siril.save("stars_prepared")
            _composite_stars(siril, cfg, "stars_prepared", log_fn)

        # Step 8: Apply saturation
        siril.load("narrowband")
        apply_saturation(siril, cfg)

        # Step 9: Apply enhancements (Silentium/Revela/Vectra) if enabled
        base_name = f"{type_name}_auto{suffix}"
        siril.save(str(output_dir / base_name))
        image_path = output_dir / f"{base_name}.fit"
        apply_enhancements(siril, image_path, cfg, log_fn)

        # Step 10: Save outputs
        paths = save_all_formats(siril, output_dir, base_name, log_fn, cfg)

        if primary_paths is None:
            primary_paths = paths

    return CompositionResult(
        linear_path=linear_path,
        linear_pcc_path=None,  # No PCC for narrowband
        auto_fit=primary_paths["fit"],
        auto_tif=primary_paths["tif"],
        auto_jpg=primary_paths["jpg"],
        stacks_dir=stacks_dir,
    )
