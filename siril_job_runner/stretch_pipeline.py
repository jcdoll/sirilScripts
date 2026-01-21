"""
Stretch pipeline for image composition.

Handles stretching, enhancements, star processing, and output saving.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from . import (
    veralux_revela,
    veralux_silentium,
    veralux_starcomposer,
    veralux_stretch,
    veralux_vectra,
)
from .config import Config
from .protocols import SirilInterface

if TYPE_CHECKING:
    pass


class StretchPipeline:
    """Handles stretching, enhancements, and star processing."""

    def __init__(
        self,
        siril: SirilInterface,
        output_dir: Path,
        config: Config,
        log_fn: Optional[Callable[[str], None]] = None,
    ):
        self.siril = siril
        self.output_dir = Path(output_dir)
        self.config = config
        self._log_fn = log_fn

    def _log(self, message: str) -> None:
        if self._log_fn:
            self._log_fn(message)

    def apply_stretch(self, method: str, source_path: Optional[str] = None) -> None:
        """
        Apply a single stretch method to currently loaded image.

        Args:
            method: Stretch method name
            source_path: Path to source image (for veralux stats calculation)
        """
        cfg = self.config

        if method == "autostretch":
            mode = "linked" if cfg.autostretch_linked else "unlinked"
            self._log(
                f"Stretching (autostretch, {mode}, targetbg={cfg.autostretch_targetbg})..."
            )
            self.siril.autostretch(
                linked=cfg.autostretch_linked,
                shadowclip=cfg.autostretch_shadowclip,
                targetbg=cfg.autostretch_targetbg,
            )
        elif method == "veralux":
            if not source_path:
                raise ValueError("veralux stretch requires source_path")
            self._log(
                f"Stretching (veralux, target_median={cfg.veralux_target_median}, "
                f"b={cfg.veralux_b})..."
            )
            success, _log_d = veralux_stretch.apply_stretch(
                siril=self.siril,
                image_path=Path(source_path),
                config=cfg,
                log_fn=self._log,
            )
            if not success:
                raise RuntimeError("VeraLux stretch failed")
        else:
            raise ValueError(f"Unknown stretch method: {method}")

    def apply_enhancements(self, image_path: Path) -> None:
        """
        Apply VeraLux enhancement pipeline to a stretched image.

        Order: Silentium (denoise) -> Revela (detail) -> Vectra (saturation)
        """
        cfg = self.config

        # 1. Noise reduction (optional, on stretched image)
        if cfg.veralux_silentium_enabled:
            self._log("Applying VeraLux Silentium (noise reduction)...")
            success, _stats = veralux_silentium.apply_silentium(
                self.siril, image_path, cfg, self._log
            )
            if not success:
                self._log("Silentium failed, continuing...")

        # 2. Detail enhancement
        if cfg.veralux_revela_enabled:
            self._log("Applying VeraLux Revela (detail enhancement)...")
            success, _stats = veralux_revela.apply_revela(
                self.siril, image_path, cfg, self._log
            )
            if not success:
                self._log("Revela failed, continuing...")

        # 3. Smart saturation (replaces basic satu command if enabled)
        if cfg.veralux_vectra_enabled:
            self._log("Applying VeraLux Vectra (smart saturation)...")
            success, _stats = veralux_vectra.apply_vectra(
                self.siril, image_path, cfg, self._log
            )
            if not success:
                self._log("Vectra failed, continuing...")

    def apply_star_processing(self, image_path: Path) -> dict[str, Optional[Path]]:
        """
        Apply star removal and optional recomposition.

        If starnet_enabled: runs StarNet to create starless + stars
        If starcomposer_enabled: recomposes with controlled star intensity

        Args:
            image_path: Path to the image to process (the "full" image)

        Returns:
            Dict with paths: {full, starless, stars, starcomposer}
            Values are None if that output was not created.
        """
        cfg = self.config
        result = {
            "full": image_path,
            "starless": None,
            "stars": None,
            "starcomposer": None,
        }

        if not cfg.starnet_enabled:
            return result

        self._log("Running StarNet for star removal...")

        # Load image and run starnet
        # StarNet modifies loaded image to starless and creates *_starmask.fit
        if not self.siril.load(str(image_path)):
            self._log("Failed to load image for StarNet")
            return result

        cfg = self.config
        if not self.siril.starnet(
            stretch=cfg.starnet_stretch,
            upscale=cfg.starnet_upscale,
            stride=cfg.starnet_stride,
        ):
            self._log("StarNet failed, continuing with original image")
            return result

        # Save starless image
        starless_path = image_path.parent / f"{image_path.stem}_starless.fit"
        self.siril.save(str(starless_path.with_suffix("")))
        self._log(f"Saved starless: {starless_path.name}")
        result["starless"] = starless_path

        # StarNet creates starmask - rename to _stars for clarity
        starmask_path = image_path.parent / f"starmask_{image_path.stem}.fit"
        stars_path = image_path.parent / f"{image_path.stem}_stars.fit"

        if starmask_path.exists():
            if stars_path.exists():
                stars_path.unlink()
            starmask_path.rename(stars_path)
            self._log(f"Saved stars: {stars_path.name}")
            result["stars"] = stars_path
        else:
            self._log(f"Warning: Star mask not found at {starmask_path.name}")

        # If starcomposer enabled, recompose with controlled star intensity
        if cfg.veralux_starcomposer_enabled and result["stars"]:
            self._log("Applying StarComposer for star recomposition...")
            success, composed_path = veralux_starcomposer.apply_starcomposer(
                siril=self.siril,
                starless_path=starless_path,
                starmask_path=result["stars"],
                config=cfg,
                log_fn=self._log,
            )
            if success:
                # Rename to _starcomposer suffix
                starcomposer_path = (
                    image_path.parent / f"{image_path.stem}_starcomposer.fit"
                )
                if starcomposer_path.exists():
                    starcomposer_path.unlink()
                composed_path.rename(starcomposer_path)
                self._log(f"Saved starcomposer: {starcomposer_path.name}")
                result["starcomposer"] = starcomposer_path
            else:
                self._log("StarComposer failed")

        return result

    def _save_all_formats(self, fit_path: Path) -> dict[str, Path]:
        """Save a FIT file in all formats (FIT already saved, add TIF/JPG)."""
        self.siril.load(str(fit_path))
        self.siril.savetif(str(fit_path.with_suffix("")), astro=True, deflate=True)
        self.siril.savejpg(str(fit_path.with_suffix("")), 90)
        tif_path = fit_path.with_suffix(".tif")
        jpg_path = fit_path.with_suffix(".jpg")
        self._log(f"Saved: {fit_path.name}, {tif_path.name}, {jpg_path.name}")
        return {"fit": fit_path, "tif": tif_path, "jpg": jpg_path}

    def save_stretched(self, output_name: str) -> dict[str, Path]:
        """Save currently loaded (stretched) image in multiple formats."""
        cfg = self.config
        # Only apply basic satu if Vectra is not enabled
        if not cfg.veralux_vectra_enabled:
            self.siril.satu(cfg.saturation_amount, cfg.saturation_background_factor)
        self.siril.cd(str(self.output_dir))

        fit_path = self.output_dir / f"{output_name}.fit"

        # Save initial stretched FIT
        self.siril.save(str(self.output_dir / output_name))

        # Apply VeraLux enhancements if any are enabled
        has_enhancements = (
            cfg.veralux_silentium_enabled
            or cfg.veralux_revela_enabled
            or cfg.veralux_vectra_enabled
        )
        if has_enhancements:
            self.apply_enhancements(fit_path)
            # Re-save after enhancements
            self.siril.save(str(self.output_dir / output_name))

        # Apply star processing (starnet + optional starcomposer)
        star_results = self.apply_star_processing(fit_path)

        # Save all outputs in all formats
        # Full image (no star processing) - always exists
        full_paths = self._save_all_formats(star_results["full"])

        # Starless - if star processing was applied
        if star_results["starless"]:
            self._save_all_formats(star_results["starless"])

        # Stars - if star processing was applied
        if star_results["stars"]:
            self._save_all_formats(star_results["stars"])

        # Starcomposer - if starcomposer was applied
        if star_results["starcomposer"]:
            self._save_all_formats(star_results["starcomposer"])

        # Return paths for the "full" image as the primary output
        return full_paths

    def auto_stretch(self, input_name: str, output_name: str) -> dict[str, Path]:
        """
        Apply stretch and saturation, save in multiple formats.

        If stretch_compare is enabled, applies all methods and saves each.

        Args:
            input_name: Path to linear image (absolute or relative to current dir)
            output_name: Base name for output files
        """
        cfg = self.config

        if cfg.stretch_compare:
            # Compare mode: apply both stretch methods
            methods = ["autostretch", "veralux"]
            self._log("Comparing stretch methods (autostretch vs veralux)...")
            primary_paths = None

            for method in methods:
                self.siril.load(input_name)
                self.apply_stretch(method, source_path=input_name)
                suffix = f"{output_name}_{method}"
                paths = self.save_stretched(suffix)
                # Use first method as primary return
                if primary_paths is None:
                    primary_paths = paths

            return primary_paths
        else:
            # Single method mode
            self.siril.load(input_name)
            self.apply_stretch(cfg.stretch_method, source_path=input_name)
            return self.save_stretched(output_name)
