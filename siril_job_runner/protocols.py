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
    def subsky(
        self,
        rbf: bool = True,
        degree: int = 1,
        samples: int = 20,
        tolerance: float = 1.0,
        smooth: float = 0.5,
    ) -> bool: ...
    def seqsubsky(
        self,
        name: str,
        method: str = "rbf",
        degree: int = 1,
        samples: int = 20,
        tolerance: float = 1.0,
        smooth: float = 0.5,
        dither: bool = True,
        prefix: Optional[str] = None,
    ) -> bool: ...

    # Registration
    def register(self, name: str, twopass: bool = False) -> bool: ...
    def setref(self, name: str, index: int) -> bool: ...
    def seqapplyreg(
        self,
        name: str,
        framing: Optional[str] = None,
        filter_fwhm: Optional[float] = None,
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
    def makepsf(
        self,
        method: str = "stars",
        symmetric: bool = True,
        save_psf: Optional[str] = None,
    ) -> bool: ...
    def rl(
        self,
        iters: int = 10,
        regularization: str = "tv",
        alpha: float = 0.001,
    ) -> bool: ...

    # Stretching
    def autostretch(
        self,
        linked: bool = True,
        shadowclip: float = -2.8,
        targetbg: float = 0.10,
    ) -> bool: ...
    def mtf(self, low: float, mid: float, high: float) -> bool: ...
    def modasinh(
        self,
        D: float,
        LP: float = 0.0,
        SP: float = 0.0,
        HP: float = 1.0,
    ) -> bool: ...
    def ght(
        self,
        D: float,
        B: float = 0.0,
        LP: float = 0.0,
        SP: float = 0.0,
        HP: float = 1.0,
    ) -> bool: ...
    def autoghs(
        self,
        shadowsclip: float = 0.0,
        D: float = 3.0,
        B: float = 0.0,
        LP: float = 0.0,
        HP: float = 1.0,
        linked: bool = True,
    ) -> bool: ...

    # Color adjustments
    def satu(self, amount: float, threshold: float = 0) -> bool: ...
    def rmgreen(
        self,
        type: int = 0,
        amount: float = 1.0,
        preserve_lightness: bool = True,
    ) -> bool: ...
    def negative(self) -> bool: ...

    # Header manipulation
    def update_key(self, key: str, value: str, comment: str = "") -> bool: ...

    # Astrometry and color calibration
    def platesolve(self) -> bool: ...
    def pcc(self, catalog: str = "nomad") -> bool: ...
    def spcc(
        self,
        sensor: str,
        red_filter: str,
        green_filter: str,
        blue_filter: str,
    ) -> bool: ...

    # Star removal
    def starnet(
        self, stretch: bool = True, upscale: bool = False, stride: int | None = None
    ) -> bool: ...

    # Pixel math
    def pm(
        self,
        expression: str,
        rescale: bool = False,
        rescale_low: float = 0.0,
        rescale_high: float = 1.0,
    ) -> bool: ...
