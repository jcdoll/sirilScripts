# Siril Job Runner Architecture

## Overview

Automated image processing pipeline for astrophotography using Siril. Supports broadband (LRGB, RGB) and narrowband (SHO, HOO, LSHO, LHOO) workflows.

## Job Types

| Type | Channels | Description |
|------|----------|-------------|
| RGB | R, G, B | Broadband color without luminance |
| LRGB | L, R, G, B | Broadband color with luminance |
| HOO | H, O | Narrowband (Ha + OIII) |
| SHO | S, H, O | Narrowband (SII + Ha + OIII) |
| LHOO | L, H, O | Narrowband with luminance |
| LSHO | L, S, H, O | Narrowband with luminance |

## Processing Stages

All workflows share common early stages:

1. **Calibration**: Build/verify master bias, darks, flats
2. **Preprocessing**: Calibrate lights, register, stack per (filter, exposure)
3. **Composition**: Type-specific processing (see below)

## Broadband Workflows (RGB, LRGB)

Key principles:
- SPCC runs on RGB only, not LRGB (L channel would dominate the calculation)
- No linear_match between R, G, B channels (destroys real color information)
- Background extraction (subsky) normalizes channel backgrounds instead
- StarNet runs on linear data before stretch for cleaner star removal

### RGB Flow

```
LINEAR PHASE
1. Cross-register R, G, B stacks
2. Post-stack background extraction (subsky)
3. Compose RGB (rgbcomp)
4. Color calibration (SPCC)
5. Deconvolution (optional)

BASELINE OUTPUTS (always produced)
6. Stretch RGB (autostretch)
7. Color removal (SCNR), neutralization, saturation
8. Save rgb_autostretch

9. Stretch RGB (veralux)
10. Color removal (SCNR), neutralization, saturation
11. VeraLux enhancements (Silentium/Revela/Vectra if enabled)
12. Save rgb_veralux

STARNET OUTPUTS (if starnet_enabled)
13. StarNet on linear RGB -> rgb_starless + rgb_stars
14. Stretch rgb_starless (autostretch)
15. Color removal, neutralization, saturation
16. Save rgb_autostretch_starless

17. Stretch rgb_starless (veralux)
18. Color removal, neutralization, saturation, enhancements
19. Save rgb_veralux_starless

20. StarComposer -> rgb_*_starcomposer
```

### LRGB Flow

```
LINEAR PHASE
1. Cross-register L, R, G, B stacks
2. Post-stack background extraction (subsky)
3. Compose RGB (rgbcomp, no L yet)
4. Color calibration (SPCC) on RGB only
5. Deconvolution on RGB and L (optional)

BASELINE OUTPUTS (always produced)
6. Stretch RGB and L (autostretch)
7. Combine LRGB (rgbcomp -lum)
8. Color removal (SCNR), neutralization, saturation
9. Save lrgb_autostretch

10. Stretch RGB and L (veralux)
11. Combine LRGB
12. Color removal, neutralization, saturation, enhancements
13. Save lrgb_veralux

STARNET OUTPUTS (if starnet_enabled)
14. StarNet on linear RGB -> rgb_starless + rgb_stars
15. StarNet on linear L -> L_starless
16. Stretch rgb_starless and L_starless (autostretch)
17. Combine LRGB from starless
18. Color removal, neutralization, saturation
19. Save lrgb_autostretch_starless

20. Stretch rgb_starless and L_starless (veralux)
21. Combine LRGB from starless
22. Color removal, neutralization, saturation, enhancements
23. Save lrgb_veralux_starless

24. StarComposer -> lrgb_*_starcomposer
```

### Broadband Output Files

| Output | StarNet | Description |
|--------|---------|-------------|
| `{type}_autostretch` | No | Baseline with original stars |
| `{type}_veralux` | No | Baseline with original stars |
| `{type}_autostretch_starless` | Yes | Stars removed |
| `{type}_veralux_starless` | Yes | Stars removed |
| `{type}_autostretch_starcomposer` | Yes | Controlled star recomposition |
| `{type}_veralux_starcomposer` | Yes | Controlled star recomposition |
| `rgb_stars` | Yes | Extracted star mask |

## Narrowband Workflows (HOO, SHO, LHOO, LSHO)

Key principles:
- No SPCC (narrowband filters don't match photometric databases)
- Channel balancing via linear_match to H (equalizes background levels)
- Palette formulas map channels to RGB (artistic choice, not photometric)
- StarNet runs on each channel independently before stretch

All narrowband types follow the same flow, differing only in required channels.

### Narrowband Flow

```
LINEAR PHASE
1. Cross-register channel stacks (H, O, S, L as needed)
2. Post-stack background extraction (subsky)
3. Channel balancing (linear_match S, O to H)
4. Deconvolution on narrowband channels (optional)

STAR SEPARATION (if starnet_enabled)
5. StarNet on each {ch}_linear -> {ch}_linear_starless + starmask
6. Extract stars from designated channel (L if available, else H)
7. Save {type}_stars

PROCESSING LOOP (for each stretch method)
8. Stretch channels (starless if starnet enabled)
9. Apply channel scale expressions (if configured)
10. Apply palette formulas -> narrowband_rgb
11. Add L as luminance (if LSHO/LHOO) -> narrowband
12. Color removal (SCNR)
13. Background neutralization
14. Save {type}_starless_{method} (if starnet enabled)
15. Composite stars back (if starnet enabled)
16. Apply saturation
17. Save {type}_auto_{method}
```

### Narrowband Palettes

| Palette | R | G | B | Notes |
|---------|---|---|---|-------|
| HOO | H | O | O | Standard HOO mapping |
| SHO | S | H | O | Hubble palette |
| SHO_FORAXX | S | 0.5*H + 0.5*O | O | Foraxx blend |
| SHO_DYNAMIC | 0.8*S + 0.2*H | 0.7*H + 0.15*S + 0.15*O | 0.8*O + 0.2*H | Dynamic mix |
| SHO_GOLD | 0.8*H + 0.2*S | 0.5*H + 0.5*O | O | Warm gold tones |
| SHO_WARM | 0.75*S + 0.25*H | H | O | Warm palette |
| SHO_BLUEGOLD | 0.8*H + 0.2*S | 0.7*O + 0.3*H | O | Blue-gold bicolor |
| SHO_FORAXX_DYNAMIC | Dynamic | Dynamic | O | Per-pixel adaptive |
| HOO_FORAXX_DYNAMIC | H | Dynamic | O | Per-pixel adaptive |

Custom formulas can override any channel via `palette_r_override`, `palette_g_override`, `palette_b_override`.

Channel scale expressions can be applied before palette formulas via `palette_h_scale_expr`, `palette_o_scale_expr`, `palette_s_scale_expr`.

### Narrowband Output Files

| Output | StarNet | Description |
|--------|---------|-------------|
| `{type}_stars` | Yes | Extracted stars (once) |
| `{type}_starless_autostretch` | Yes | Starless with autostretch |
| `{type}_starless_veralux` | Yes | Starless with veralux |
| `{type}_auto_autostretch` | Any | Final output (stars composited if starnet) |
| `{type}_auto_veralux` | Any | Final output (stars composited if starnet) |

## Key Differences: Broadband vs Narrowband

| Aspect | Broadband | Narrowband |
|--------|-----------|------------|
| Color calibration | SPCC (photometric) | None (palette mapping) |
| StarNet timing | After SPCC, before stretch | Before stretch |
| StarNet input | RGB composite + L | Individual channels |
| Star compositing | Always (StarComposer) | Always (screen blend) |
| Baseline outputs | Yes (with original stars) | No (always process stars) |
| Channel balancing | None (SPCC handles it) | linear_match to H |

## StarNet Behavior

StarNet runs on LINEAR data with internal MTF stretch (`stretch=True`).

Broadband:
- StarNet runs on RGB composite and L channel separately
- Stars extracted from RGB
- Baseline outputs preserve original stars (no StarNet artifacts)
- Starless outputs use processed channels

Narrowband:
- StarNet runs on each channel independently (H, O, S, L)
- Stars extracted from L (if available) or H
- All outputs go through star separation/recomposition
- Stars composited with screen blend: `1 - (1-starless)*(1-stars)`

## VeraLux Processing

The VeraLux system provides advanced image processing capabilities beyond Siril's built-in autostretch.

### VeraLux Stretch

HyperMetric stretch with target background median. Uses binary search to find optimal D parameter for the hyperbolic transfer function. Provides more control over highlight preservation than autostretch.

Key parameters:
- `veralux_target_median`: Target background brightness (default: 0.10)
- `veralux_b`: Highlight protection / curve knee (default: 6.0)

### VeraLux Silentium (Noise Suppression)

Wavelet-based noise reduction using SWT (Stationary Wavelet Transform). Operates in luminance and chrominance channels independently.

Key parameters:
- `veralux_silentium_enabled`: Enable noise suppression (default: false)
- `veralux_silentium_intensity`: Luminance noise reduction 0-100 (default: 25.0)
- `veralux_silentium_chroma`: Chrominance noise reduction 0-100 (default: 30.0)

### VeraLux Revela (Detail Enhancement)

Wavelet-based detail enhancement using ATWT (A Trous Wavelet Transform). Boosts fine detail (texture) and medium-scale structure independently.

Key parameters:
- `veralux_revela_enabled`: Enable detail enhancement (default: false)
- `veralux_revela_texture`: Fine detail boost 0-100 (default: 50.0)
- `veralux_revela_structure`: Medium structure boost 0-100 (default: 50.0)

### VeraLux Vectra (Smart Saturation)

Saturation enhancement in LCH color space with per-vector control. Allows independent saturation adjustment for different hue ranges.

Key parameters:
- `veralux_vectra_enabled`: Enable smart saturation (default: false)
- `veralux_vectra_saturation`: Global saturation boost 0-100 (default: 25.0)
- `veralux_vectra_red/yellow/green/cyan/blue/magenta`: Per-vector overrides

### VeraLux StarComposer (Star Recomposition)

Controlled star recomposition onto starless images. Uses hyperbolic stretch to control star intensity and profile.

Key parameters:
- `veralux_starcomposer_log_d`: Star intensity 0-2 (default: 1.0)
- `veralux_starcomposer_hardness`: Profile hardness 1-100 (default: 6.0)
- `veralux_starcomposer_blend_mode`: "screen" or "linear_add" (default: "screen")

## Configuration System

All configurable values are defined in `siril_job_runner/config.py` in a single `Config` dataclass. Users can override any value via:

1. `settings.json` in the repository root (user defaults)
2. `options` field in job JSON files (per-job overrides)

Override precedence: `DEFAULTS <- settings.json <- job.json options`

### Key Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `starnet_enabled` | true | Enable star removal |
| `stretch_compare` | true | Output both autostretch and veralux |
| `stretch_method` | veralux | Stretch method if compare disabled |
| `narrowband_star_source` | auto | Channel for star extraction (L or H) |
| `narrowband_star_color` | mono | Star color mode (mono = white) |
| `palette` | SHO | Narrowband palette selection |
| `spcc_enabled` | true | Spectrophotometric color calibration |
| `deconv_enabled` | false | Richardson-Lucy deconvolution |
| `temp_tolerance` | 2.0 | Temperature matching tolerance (Celsius) |
| `pre_stack_subsky_method` | rbf | Pre-stack background extraction method |
| `post_stack_subsky_method` | poly | Post-stack background extraction method |
| `broadband_neutralization` | false | Background color neutralization for broadband |
| `narrowband_neutralization` | true | Background color neutralization for narrowband |

See `siril_job_runner/config.py` for the complete list of options.
