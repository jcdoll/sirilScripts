"""
Protocol definitions for Siril interfaces.
"""

from typing import Optional, Protocol


class SirilInterface(Protocol):
    """Protocol for Siril scripting interface."""

    # Directory operations
    def cd(self, path: str) -> bool: ...

    # File operations
    def load(self, path: str) -> bool: ...
    def save(self, path: str) -> bool: ...
    def savetif(
        self, path: str, astro: bool = False, deflate: bool = False
    ) -> bool: ...
    def savejpg(self, path: str, quality: int = 90) -> bool: ...
    def close(self) -> bool: ...

    # Conversion and sequences
    def convert(self, name: str, out: Optional[str] = None) -> bool: ...

    # Calibration
    def calibrate(
        self,
        name: str,
        bias: Optional[str] = None,
        dark: Optional[str] = None,
        flat: Optional[str] = None,
    ) -> bool: ...

    # Background extraction
    def seqsubsky(self, name: str, degree: int = 1) -> bool: ...

    # Registration
    def register(self, name: str, twopass: bool = False) -> bool: ...
    def seqapplyreg(
        self,
        name: str,
        framing: Optional[str] = None,
        filter_fwhm: Optional[str] = None,
    ) -> bool: ...

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
    ) -> bool: ...

    # Linear matching
    def linear_match(self, ref: str, low: float = 0, high: float = 0.92) -> bool: ...

    # Composition
    def rgbcomp(
        self,
        r: Optional[str] = None,
        g: Optional[str] = None,
        b: Optional[str] = None,
        lum: Optional[str] = None,
        rgb: Optional[str] = None,
        out: Optional[str] = None,
    ) -> bool: ...

    # Deconvolution
    def makepsf(self, method: str = "blind") -> bool: ...
    def rl(self) -> bool: ...

    # Stretching
    def autostretch(self, linked: bool = True) -> bool: ...
    def mtf(self, low: float, mid: float, high: float) -> bool: ...

    # Color adjustments
    def satu(self, amount: float, threshold: float = 0) -> bool: ...
    def rmgreen(self) -> bool: ...

    # Star removal
    def starnet(self, stretch: bool = False) -> bool: ...

    # Pixel math
    def pm(
        self,
        expression: str,
        rescale: bool = False,
        rescale_low: float = 0.0,
        rescale_high: float = 1.0,
    ) -> bool: ...
