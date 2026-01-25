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

| Palette | R | G | B |
|---------|---|---|---|
| HOO | H | O | O |
| SHO | S | H | O |
| SHO_FORAXX | S | 0.5*H + 0.5*O | O |

Custom formulas can override any channel via `palette_r_override`, etc.

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

**Broadband:**
- StarNet runs on RGB composite and L channel separately
- Stars extracted from RGB
- Baseline outputs preserve original stars (no StarNet artifacts)
- Starless outputs use processed channels

**Narrowband:**
- StarNet runs on each channel independently (H, O, S, L)
- Stars extracted from L (if available) or H
- All outputs go through star separation/recomposition
- Stars composited with screen blend: `1 - (1-starless)*(1-stars)`

## Configuration Reference

Key options affecting composition:

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
