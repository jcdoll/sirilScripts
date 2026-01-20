# LRGB Composition Workflow

Best practices for combining monochrome LRGB filter data into color images.

Based on the [PixInsight Workflow Diagram](https://stargazerslounge.com/topic/430351-pixinsight-workflow-diagram/).

## Workflow Overview

The workflow is divided into two phases: LINEAR and NON-LINEAR. The key insight
is that star removal happens in the linear phase, and LRGB combination happens
on stretched (non-linear) starless images.

```
LINEAR PHASE:
 1. Stack each channel separately (R, G, B, L)
 2. Background extraction (gradient correction) on EACH channel
 3. Combine RGB (no linear matching between channels)
 4. Background neutralization on RGB
 5. Color calibration (SPCC) on RGB
 6. Deconvolution on RGB and L (optional)
 7. Noise reduction on RGB and L (optional)
 8. Star removal on RGB -> RGB_starless + RGB_stars
 9. Star removal on L -> L_starless

STRETCH PHASE:
10. Stretch RGB_starless (autostretch or GHS/VeraLux)
11. Stretch L_starless (same method)
12. Stretch RGB_stars

NON-LINEAR PHASE:
13. Combine LRGB from stretched starless images
14. SCNR (green removal)
15. Non-linear enhancements (saturation, curves, etc.)
16. Add stars back
17. Final output
```

## Key Principles

### Background Extraction Per-Channel

Apply gradient correction (subsky/DBE) to each individual channel (R, G, B, L)
BEFORE combining. This normalizes the backgrounds and eliminates the need for
linear matching between color channels.

### No Linear Matching for Broadband LRGB

Do NOT use `linear_match` between R, G, B channels for broadband imaging.

**What linear_match does:**
It finds a linear scaling factor to make one channel's histogram match a reference
channel's histogram. This is histogram equalization, not color calibration.

**Why it's wrong for broadband:**
- Broadband R, G, B channels represent actual colors from the target
- Galaxies like M31 genuinely have more red/yellow than blue (old stellar populations)
- Scaling B up 60% to match R's histogram destroys real color information
- You're saying "make blue as bright as red" when the target IS redder than blue

**When linear_match IS appropriate:**
- Narrowband imaging (Ha, OIII, SII) where emission lines have vastly different intensities
- There's no "correct" color ratio for narrowband - it's artistic mapping
- Equalizing histograms is a desired creative choice

**The correct workflow for broadband:**
1. Background extraction (subsky) normalizes each channel's background to ~0
2. Compose RGB without linear matching
3. Let SPCC handle color calibration using star photometry

See: [Linear Match - Siril Documentation](https://siril.readthedocs.io/en/latest/processing/lmatch.html)

### SPCC on RGB, Not LRGB

Spectrophotometric Color Calibration must run on the RGB composite BEFORE
luminance is added. Running SPCC on LRGB fails because the luminance channel
dominates the calculation.

### Star Removal Before Stretching

Remove stars in the LINEAR phase, before stretching. This allows:
- Cleaner processing of nebulosity without star interference
- Separate control over star appearance
- Better noise reduction on starless images
- More accurate LRGB combination

### LRGB Combination on Non-Linear Data

LRGBCombination requires non-linear (stretched) input. Both the RGB and L
images must be stretched before combining. The combination is done on
STARLESS images to avoid star color issues.

### Stars Added Back at End

Stars are processed separately and added back after all enhancements are
complete. This gives full control over star size, color, and intensity.

## Implementation in Siril

### Phase 1: Linear Processing

#### Step 1: Cross-Register Stacks

```
cd /path/to/stacks
convert stack -out=./registered
cd registered
register stack -2pass
seqapplyreg stack -framing=min
```

#### Step 2: Background Extraction Per-Channel

```
# For each channel (R, G, B, L):
load R
subsky -rbf -samples=20 -tolerance=1.0 -smooth=0.5
save R

load G
subsky -rbf -samples=20 -tolerance=1.0 -smooth=0.5
save G

load B
subsky -rbf -samples=20 -tolerance=1.0 -smooth=0.5
save B

load L
subsky -rbf -samples=20 -tolerance=1.0 -smooth=0.5
save L
```

#### Step 3: Compose RGB (No Linear Matching)

```
rgbcomp R G B -out=rgb
```

#### Step 4: SPCC on RGB

```
load rgb
spcc -monosensor=... -rfilter=... -gfilter=... -bfilter=...
save rgb_spcc
```

#### Step 5: Deconvolution (Optional)

```
# On RGB
load rgb_spcc
makepsf -stars -symmetric
rl -iters=10 -reg=tv
save rgb_deconv

# On L
load L
makepsf -stars -symmetric
rl -iters=10 -reg=tv
save L_deconv
```

#### Step 6: Noise Reduction (Optional)

Apply noise reduction while data is still linear for best results.

#### Step 7: Star Removal

```
# On RGB - outputs starless and stars
load rgb_spcc  # or rgb_deconv
starnet
save rgb_starless
# stars saved automatically as rgb_spcc_stars or similar

# On L
load L  # or L_deconv
starnet
save L_starless
```

### Phase 2: Stretching

#### Step 8: Stretch Starless Images

Both RGB_starless and L_starless must be stretched with the same method:

```
# Stretch RGB starless
load rgb_starless
autostretch  # or GHS/VeraLux
save rgb_starless_stretched

# Stretch L starless
load L_starless
autostretch  # or GHS/VeraLux
save L_starless_stretched

# Stretch stars separately
load rgb_stars
autostretch
save rgb_stars_stretched
```

### Phase 3: Non-Linear Processing

#### Step 9: Combine LRGB

Use rgbcomp -lum on non-linear (stretched) starless images:

```
load rgb_starless_stretched
rgbcomp -lum=L_starless_stretched rgb_starless_stretched -out=lrgb_starless
```

#### Step 10: SCNR (Green Removal)

```
load lrgb_starless
scnr 0 1
save lrgb_scnr
```

#### Step 11: Non-Linear Enhancements

Apply saturation, curves, local histogram equalization, etc.

#### Step 12: Add Stars Back

```
# Use PixelMath or screen blend to add stars
load lrgb_enhanced
pm $lrgb_enhanced$ + $rgb_stars_stretched$ * 0.5  # adjust intensity
save lrgb_final
```

## Common Mistakes

1. **Linear matching R, G, B to each other** - Destroys color ratios
2. **Running SPCC on LRGB** - L channel dominates, wrong colors
3. **LRGB combination on linear data** - Algorithms expect non-linear input
4. **Star removal after stretching** - Harder to process, worse results
5. **Adding stars before enhancements** - Stars affected by processing
6. **Skipping per-channel background extraction** - Causes color imbalance

## Stretch Methods

Choose one stretch method and apply it consistently to RGB and L:

| Method | Description | Use Case |
|--------|-------------|----------|
| autostretch | MTF-based, automatic | Quick results |
| GHS | Generalized Hyperbolic Stretch | Fine control |
| VeraLux | HyperMetric Stretch | Color preservation |
| ArcSinh | Asinh-based stretch | HDR-like results |

### VeraLux HyperMetric Stretch

VeraLux uses vector preservation to maintain color relationships during stretching.
Based on [VeraLux by Riccardo Paterniti](https://gitlab.com/free-astro/siril-scripts/-/tree/main/VeraLux).

**The Problem with Scalar Stretching:**

Traditional stretch methods (autostretch, GHS) apply the same transfer function to
each R, G, B channel independently. This destroys color ratios - a star with
R=2000, G=1000, B=500 might become R=60000, G=58000, B=55000, shifting from orange
toward white.

**Vector Preservation Algorithm (Full Reference Implementation):**

For RGB images, VeraLux treats each pixel as a vector in 3D color space. The
algorithm has multiple steps that must all be followed:

```
1. Normalize input to 0-1 range
2. Calculate anchor (black point): min(0.5th percentile of each channel) - 0.00025
3. Subtract anchor from ALL channels: img_anchored = max(data - anchor, 0)
4. Compute luminance from ANCHORED data:
   L = 0.2126*R + 0.7152*G + 0.0722*B  (Rec.709 weights)
5. Compute color ratios from ANCHORED data:
   r_ratio = R_anchored / L, g_ratio = G_anchored / L, b_ratio = B_anchored / L
6. Stretch luminance with hyperbolic formula:
   L_str = (asinh(D*L + b) - asinh(b)) / (asinh(D + b) - asinh(b))
7. Apply color convergence (power=3.5):
   k = L_str^3.5
   r_final = r_ratio * (1 - k) + 1.0 * k
   g_final = g_ratio * (1 - k) + 1.0 * k
   b_final = b_ratio * (1 - k) + 1.0 * k
8. Reconstruct RGB:
   R_out = L_str * r_final
   G_out = L_str * g_final
   B_out = L_str * b_final
9. Adaptive output scaling:
   - Find floor: max(min_L, median_L - 2.7*std_L)
   - Smart Max: check if abs_max is hot pixel (neighbors < 20% of max)
   - soft_ceil = max of 99.9th percentile per channel (RGB)
   - Compute contrast_scale: (0.98 - 0.001) / (soft_ceil - floor)
   - Compute physical_scale: (1.0 - 0.001) / (abs_max - floor) [if valid max]
   - Use min(contrast_scale, physical_scale)
   - Apply: result = (channel - floor) * scale + 0.001 (pedestal)
   - Apply MTF: y = (m-1)*x / ((2m-1)*x - m) where m is solved for target
   - Soft-clip: y = thresh + (1-thresh) * (1 - (1-t)^rolloff) where t = (x-thresh)/(1-thresh)
```

**Critical Details:**

- Anchor subtraction happens FIRST, before computing luminance or ratios
- Ratios are computed from anchored data, not original data
- Color convergence (step 7) prevents overflow by blending ratios toward 1.0
  (white) as brightness increases - this mimics physical sensor saturation
- convergence_power=3.5 is the reference default (higher = more saturation retained)
- Adaptive output scaling normalizes the dynamic range after stretch

**Hyperbolic Stretch Formula:**

```
output = (asinh(D * input + b) - asinh(b)) / (asinh(D + b) - asinh(b))
```

Where:
- `D` = stretch intensity (10^log_d, solved to achieve target median)
- `b` = highlight protection / curve knee (default 6.0, higher preserves stars)

**Reference Parameter Defaults:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| log_d | 2.0 | Stretch intensity exponent |
| protect_b | 6.0 | Knee sharpness |
| convergence_power | 3.5 | Color saturation recovery rate |
| target_bg | 0.20 | Target background median |

**For Mono Images:**

The stretch applies directly since there's no color to preserve. Same anchor
subtraction and adaptive output scaling still apply.

**SHO/Narrowband Workflow:**

For narrowband palettes (SHO, HOO), use palette-specific luminance weights instead
of Rec.709. The weights define how each channel contributes to perceived brightness
in the synthetic color image.

## References

- [PixInsight Workflow Diagram 2025](https://stargazerslounge.com/topic/430351-pixinsight-workflow-diagram/)
- [PixInsight Forum - LRGB on linear or stretched](https://pixinsight.com/forum/index.php?threads/lrgb-comb-on-linear-or-stretched-data.18885/)
- [VeraLux HMS Workflow](https://www.martinkaessler.com/veralux-hms-revolutionary-stretching-method/)
- [Siril SPCC Documentation](https://siril.readthedocs.io/en/latest/processing/color-calibration/spcc.html)
- [Siril RGB Composition Tutorial](https://siril.org/tutorials/rgb_composition/)
- [Siril Linear Match Documentation](https://siril.readthedocs.io/en/latest/processing/lmatch.html)
- [Siril RGB Compositing Documentation](https://siril.readthedocs.io/en/latest/processing/rgbcomp.html)
