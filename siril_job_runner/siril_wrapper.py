"""
Wrapper for pysiril to provide a convenient method-based interface.

pysiril uses Execute() for all commands. This wrapper provides
typed methods that build command strings internally.
"""

from typing import Optional


class SirilWrapper:
    """
    Wrapper around pysiril's Execute() method.

    Provides convenient methods for common Siril commands.
    All methods internally call Execute() with the appropriate command string.
    """

    def __init__(self, siril):
        """
        Initialize wrapper with a pysiril Siril instance.

        Args:
            siril: pysiril.siril.Siril instance (already Open()'d)
        """
        self._siril = siril

    def execute(self, command: str) -> bool:
        """Execute a raw Siril command string."""
        return self._siril.Execute(command)

    # Directory operations

    def cd(self, path: str) -> bool:
        """Change working directory."""
        # Normalize path separators for Siril
        path = path.replace("\\", "/")
        return self.execute(f"cd {path}")

    # File operations

    def load(self, path: str) -> bool:
        """Load an image file."""
        path = path.replace("\\", "/")
        return self.execute(f"load {path}")

    def save(self, path: str) -> bool:
        """Save current image."""
        path = path.replace("\\", "/")
        return self.execute(f"save {path}")

    def savetif(self, path: str, astro: bool = False, deflate: bool = False) -> bool:
        """Save as TIFF."""
        path = path.replace("\\", "/")
        opts = []
        if astro:
            opts.append("-astro")
        if deflate:
            opts.append("-deflate")
        opts_str = " ".join(opts)
        return self.execute(f"savetif {path} {opts_str}".strip())

    def savejpg(self, path: str, quality: int = 90) -> bool:
        """Save as JPEG."""
        path = path.replace("\\", "/")
        return self.execute(f"savejpg {path} {quality}")

    def close(self) -> bool:
        """Close current image."""
        return self.execute("close")

    # Conversion and sequences

    def convert(self, name: str, out: Optional[str] = None) -> bool:
        """Convert files to Siril sequence."""
        cmd = f"convert {name}"
        if out:
            out = out.replace("\\", "/")
            cmd += f" -out={out}"
        return self.execute(cmd)

    # Calibration

    def calibrate(
        self,
        name: str,
        bias: Optional[str] = None,
        dark: Optional[str] = None,
        flat: Optional[str] = None,
    ) -> bool:
        """Calibrate a sequence."""
        cmd = f"calibrate {name}"
        if bias:
            cmd += f" -bias={bias.replace(chr(92), '/')}"
        if dark:
            cmd += f" -dark={dark.replace(chr(92), '/')}"
        if flat:
            cmd += f" -flat={flat.replace(chr(92), '/')}"
        return self.execute(cmd)

    # Background extraction

    def subsky(
        self,
        rbf: bool = True,
        degree: int = 1,
        samples: int = 20,
        tolerance: float = 1.0,
        smooth: float = 0.5,
    ) -> bool:
        """Background extraction on loaded image."""
        method = "-rbf" if rbf else str(degree)
        cmd = f"subsky {method}"
        cmd += f" -samples={samples} -tolerance={tolerance} -smooth={smooth}"
        return self.execute(cmd)

    # Registration

    def register(self, name: str, twopass: bool = False) -> bool:
        """Register a sequence."""
        cmd = f"register {name}"
        if twopass:
            cmd += " -2pass"
        return self.execute(cmd)

    def seqapplyreg(
        self,
        name: str,
        framing: Optional[str] = None,
        filter_fwhm: Optional[float] = None,
    ) -> bool:
        """
        Apply registration to sequence.

        Args:
            name: Sequence name
            framing: Framing mode (e.g., "min", "max", "current")
            filter_fwhm: Absolute FWHM threshold in pixels (filters wFWHM)
        """
        cmd = f"seqapplyreg {name}"
        if framing:
            cmd += f" -framing={framing}"
        if filter_fwhm is not None:
            cmd += f" -filter-wfwhm={filter_fwhm:.2f}"
        return self.execute(cmd)

    # Stacking

    def stack(
        self,
        name: str,
        rejection: str = "rej",
        weight: str = "w",
        sigma_low: str = "3",
        sigma_high: str = "3",
        norm: Optional[str] = None,
        fastnorm: bool = False,
        out: Optional[str] = None,
    ) -> bool:
        """Stack a sequence."""
        cmd = f"stack {name} {rejection} {weight} {sigma_low} {sigma_high}"
        if norm:
            cmd += f" -norm={norm}"
        if fastnorm:
            cmd += " -fastnorm"
        if out:
            out = out.replace("\\", "/")
            cmd += f" -out={out}"
        return self.execute(cmd)

    # Linear matching

    def linear_match(self, ref: str, low: float = 0, high: float = 0.92) -> bool:
        """Linear match current image to reference."""
        return self.execute(f"linear_match {ref} {low} {high}")

    # Composition

    def rgbcomp(
        self,
        r: Optional[str] = None,
        g: Optional[str] = None,
        b: Optional[str] = None,
        lum: Optional[str] = None,
        rgb: Optional[str] = None,
        out: Optional[str] = None,
    ) -> bool:
        """
        RGB composition.

        Either provide r, g, b for RGB composition,
        or lum + rgb to add luminance to existing RGB.
        """
        if lum and rgb:
            # Adding luminance to RGB
            cmd = f"rgbcomp -lum={lum} {rgb}"
        elif r and g and b:
            # Creating RGB from channels
            cmd = f"rgbcomp {r} {g} {b}"
        else:
            raise ValueError("Either provide r,g,b or lum+rgb")

        if out:
            cmd += f" -out={out}"
        return self.execute(cmd)

    # Deconvolution

    def makepsf(self, method: str = "blind") -> bool:
        """Create PSF for deconvolution."""
        return self.execute(f"makepsf {method}")

    def rl(self) -> bool:
        """Richardson-Lucy deconvolution."""
        return self.execute("rl")

    # Stretching

    def autostretch(self, linked: bool = True) -> bool:
        """Auto-stretch image."""
        cmd = "autostretch"
        if linked:
            cmd += " -linked"
        return self.execute(cmd)

    def mtf(self, low: float, mid: float, high: float) -> bool:
        """Midtone transfer function."""
        return self.execute(f"mtf {low} {mid} {high}")

    # Color adjustments

    def satu(self, amount: float, threshold: float = 0) -> bool:
        """Adjust saturation."""
        return self.execute(f"satu {amount} {threshold}")

    def rmgreen(self) -> bool:
        """Remove green cast."""
        return self.execute("rmgreen")

    # Header manipulation

    def update_key(self, key: str, value: str, comment: str = "") -> bool:
        """
        Update a FITS header keyword value.

        Args:
            key: FITS keyword name (e.g., 'EQUINOX')
            value: New value for the keyword
            comment: Optional comment for the keyword
        """
        if comment:
            return self.execute(f"update_key {key} {value} {comment}")
        return self.execute(f"update_key {key} {value}")

    # Astrometry and color calibration

    def platesolve(self) -> bool:
        """
        Plate solve the loaded image.

        Uses FITS header metadata (focal length, pixel size, coordinates)
        if available. Returns True if solve succeeds.
        """
        return self.execute("platesolve")

    def pcc(self, catalog: str = "nomad") -> bool:
        """
        Photometric Color Calibration on loaded image.

        Requires image to be plate-solved first.

        Args:
            catalog: Star catalog to use (nomad, apass, gaia, localgaia)
        """
        return self.execute(f"pcc -catalog={catalog}")

    def spcc(
        self,
        sensor: str,
        red_filter: str,
        green_filter: str,
        blue_filter: str,
    ) -> bool:
        """
        Spectrophotometric Color Calibration on loaded image.

        Uses actual sensor QE and filter transmission curves for accurate
        color calibration. Preferable to PCC when filter profiles are known.

        Args:
            sensor: Mono sensor name (e.g., "Sony_IMX571")
            red_filter: Red filter name (e.g., "Optolong_Red")
            green_filter: Green filter name (e.g., "Optolong_Green")
            blue_filter: Blue filter name (e.g., "Optolong_Blue")

        Note: Use underscores instead of spaces in names to avoid quoting issues.
              Run 'spcc_list monosensor' or 'spcc_list redfilter' in Siril
              to see available options.
        """
        cmd = (
            f"spcc -monosensor={sensor} "
            f"-rfilter={red_filter} "
            f"-gfilter={green_filter} "
            f"-bfilter={blue_filter}"
        )
        return self.execute(cmd)

    # Star removal

    def starnet(self, stretch: bool = False) -> bool:
        """Run StarNet for star removal."""
        cmd = "starnet"
        if stretch:
            cmd += " -stretch"
        return self.execute(cmd)

    # Pixel math

    def pm(
        self,
        expression: str,
        rescale: bool = False,
        rescale_low: float = 0.0,
        rescale_high: float = 1.0,
    ) -> bool:
        """
        Pixel math expression.

        Args:
            expression: PixelMath formula (image vars wrapped in $)
            rescale: Whether to rescale output
            rescale_low: Low rescale bound (0-1)
            rescale_high: High rescale bound (0-1)
        """
        cmd = f'pm "{expression}"'
        if rescale:
            cmd += f" -rescale {rescale_low} {rescale_high}"
        return self.execute(cmd)
