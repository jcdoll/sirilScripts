"""
Siril color and calibration commands.

Provides mixin class for color adjustments, calibration, and star operations.
"""


class SirilColorMixin:
    """Mixin for Siril color adjustments and calibration operations."""

    def execute(self, command: str) -> bool:
        """Execute a raw Siril command string. Must be implemented by subclass."""
        raise NotImplementedError

    # Color adjustments

    def satu(self, amount: float, threshold: float = 0) -> bool:
        """Adjust saturation."""
        return self.execute(f"satu {amount} {threshold}")

    def rmgreen(
        self,
        type: int = 0,
        amount: float = 1.0,
        preserve_lightness: bool = True,
    ) -> bool:
        """
        Remove green cast (SCNR - Subtractive Chromatic Noise Reduction).

        Args:
            type: Algorithm type
                0 = average neutral (default)
                1 = maximum neutral
                2 = maximum mask
                3 = additive mask
            amount: Strength 0-1, only used for types 2 and 3
            preserve_lightness: Preserve lightness (default True)
        """
        cmd = "rmgreen"
        if not preserve_lightness:
            cmd += " -nopreserve"
        cmd += f" {type}"
        if type in (2, 3):
            cmd += f" {amount}"
        return self.execute(cmd)

    def negative(self) -> bool:
        """Apply negative transformation (invert colors)."""
        return self.execute("negative")

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
        whiteref: str = "Average Spiral Galaxy",
        bgtol_upper: float = 2.0,
        bgtol_lower: float = 2.8,
        obsheight: int = 1000,
    ) -> bool:
        """
        Spectrophotometric Color Calibration on loaded image.

        Uses actual sensor QE and filter transmission curves for accurate
        color calibration. Preferable to PCC when filter profiles are known.

        Args:
            sensor: Mono sensor name (e.g., "Sony IMX571")
            red_filter: Red filter name (e.g., "Baader R")
            green_filter: Green filter name (e.g., "Baader G")
            blue_filter: Blue filter name (e.g., "Baader B")
            whiteref: White reference target (e.g., "Average Spiral Galaxy")
            bgtol_upper: Background tolerance upper bound in sigma (default 2.0)
            bgtol_lower: Background tolerance lower bound in sigma (default 2.8, meaning -2.8)
            obsheight: Observation height in meters for atmospheric correction (default 1000)

        Note: Names with spaces are automatically quoted.
              Run 'spcc_list monosensor' or 'spcc_list redfilter' in Siril
              to see available options.
        """
        # Siril quoting rule: arguments with spaces need quotes around the
        # entire argument including the -flag part, e.g. "-monosensor=Sony IMX571"
        cmd = (
            f'spcc "-monosensor={sensor}" '
            f'"-rfilter={red_filter}" '
            f'"-gfilter={green_filter}" '
            f'"-bfilter={blue_filter}" '
            f'"-whiteref={whiteref}" '
            f"-bgtol={bgtol_upper},{bgtol_lower} "
            f"-atmos -obsheight={obsheight}"
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
