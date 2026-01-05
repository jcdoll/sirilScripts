"""Core XISF to FITS conversion logic."""

import logging
import os
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from astropy.io import fits
from xisf import XISF

from .models import ConversionConfig, ConversionResult

# Suppress noisy astropy FITS warnings (card length, HIERARCH keywords, etc.)
warnings.filterwarnings('ignore', category=fits.verify.VerifyWarning)

__version__ = "1.0.0"


def check_disk_space(path: Path, required_bytes: int) -> bool:
    """Check if sufficient disk space is available."""
    try:
        stat = shutil.disk_usage(path)
        return stat.free >= required_bytes * 2
    except OSError:
        return True


def compute_pixel_stats(data: np.ndarray) -> dict:
    """Compute pixel statistics for verification."""
    finite_data = data[np.isfinite(data)]
    if len(finite_data) == 0:
        return {"min": None, "max": None, "mean": None, "std": None, "finite_count": 0}

    return {
        "min": float(np.min(finite_data)),
        "max": float(np.max(finite_data)),
        "mean": float(np.mean(finite_data)),
        "std": float(np.std(finite_data)),
        "finite_count": len(finite_data),
        "total_count": data.size,
        "nan_count": int(np.sum(np.isnan(data))),
        "inf_count": int(np.sum(np.isinf(data))),
    }


def stats_match(stats1: dict, stats2: dict, rtol: float = 1e-5) -> bool:
    """Check if two stat dictionaries match within tolerance."""
    if stats1.get("finite_count") != stats2.get("finite_count"):
        return False
    if stats1.get("nan_count") != stats2.get("nan_count"):
        return False
    if stats1.get("inf_count") != stats2.get("inf_count"):
        return False

    for key in ("min", "max"):
        v1, v2 = stats1.get(key), stats2.get(key)
        if v1 is None and v2 is None:
            continue
        if v1 is None or v2 is None:
            return False
        if not np.isclose(v1, v2, rtol=1e-6):
            return False

    v1, v2 = stats1.get("mean"), stats2.get("mean")
    if v1 is not None and v2 is not None:
        if not np.isclose(v1, v2, rtol=rtol):
            return False

    return True


def get_output_path(input_path: Path, config: ConversionConfig,
                    root_dir: Optional[Path] = None) -> Path:
    """Determine output path for a given input file."""
    if config.output_dir is None:
        return input_path.with_suffix('.fits')

    if config.preserve_structure and root_dir:
        rel_path = input_path.relative_to(root_dir)
        output_path = config.output_dir / rel_path.with_suffix('.fits')
    else:
        output_path = config.output_dir / input_path.with_suffix('.fits').name

    return output_path


def convert_xisf_to_fits(xisf_path: Path, config: ConversionConfig,
                         root_dir: Optional[Path] = None) -> ConversionResult:
    """Convert a single XISF file to FITS format."""
    result = ConversionResult(input_path=xisf_path)

    try:
        fits_path = get_output_path(xisf_path, config, root_dir)
        result.output_path = fits_path

        if fits_path.exists() and not config.overwrite:
            result.skipped = True
            result.skip_reason = "output exists"
            return result

        fits_path.parent.mkdir(parents=True, exist_ok=True)

        input_size = xisf_path.stat().st_size
        if not check_disk_space(fits_path.parent, input_size * 3):
            result.error = "Insufficient disk space"
            return result

        xisf = XISF(str(xisf_path))
        meta = xisf.get_images_metadata()[0]
        geometry = meta.get('geometry')

        if geometry is None:
            result.error = "No geometry metadata found in XISF file"
            return result

        meta_width, meta_height, meta_channels = geometry
        im_data = xisf.read_image(0)
        result.input_shape = im_data.shape
        result.dtype = str(im_data.dtype)

        if im_data.ndim == 2:
            arr_height, arr_width = im_data.shape
            arr_channels = 1
            if meta_channels != 1:
                result.error = f"Dimension mismatch: metadata says {meta_channels} channels but array is 2D"
                return result
        elif im_data.ndim == 3:
            arr_height, arr_width, arr_channels = im_data.shape
        else:
            result.error = f"Unexpected {im_data.ndim}D array, expected 2D or 3D"
            return result

        if arr_width != meta_width or arr_height != meta_height or arr_channels != meta_channels:
            result.error = f"Dimension mismatch: metadata vs array"
            return result

        nan_count = np.sum(np.isnan(im_data))
        inf_count = np.sum(np.isinf(im_data))

        if nan_count > 0 or inf_count > 0:
            msg = f"Image contains {nan_count} NaN and {inf_count} Inf values"
            if config.nan_handling == "error":
                result.error = msg
                return result
            elif config.nan_handling == "warn":
                result.warnings.append(msg)
            elif config.nan_handling == "zero":
                result.warnings.append(f"{msg} - replaced with zeros")
                im_data = np.nan_to_num(im_data, nan=0.0, posinf=0.0, neginf=0.0)

        if arr_channels not in (1, 3, 4):
            result.warnings.append(f"Unusual channel count: {arr_channels}")

        if config.check_stats:
            input_stats = compute_pixel_stats(im_data)
            result.pixel_stats["input"] = input_stats

        if im_data.ndim == 3:
            im_data = np.moveaxis(im_data, -1, 0)

        result.output_shape = im_data.shape

        hdu = fits.PrimaryHDU(im_data)

        if 'FITSKeywords' in meta:
            skip_keywords = {'SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'NAXIS3', 'EXTEND'}
            for keyword, values_list in meta['FITSKeywords'].items():
                if keyword.upper() in skip_keywords:
                    continue
                if values_list:
                    val = values_list[0].get('value', '')
                    comment = values_list[0].get('comment', '')
                    try:
                        hdu.header[keyword] = (val, comment)
                    except (ValueError, KeyError) as e:
                        result.warnings.append(f"Could not set keyword {keyword}: {e}")

        hdu.header['HISTORY'] = f'Converted from XISF by xisf2fits v{__version__}'
        hdu.header['XISFSRC'] = (str(xisf_path.name), 'Original XISF filename')

        if config.atomic_write:
            fd, temp_path = tempfile.mkstemp(suffix='.fits.tmp', dir=fits_path.parent)
            os.close(fd)

            try:
                hdu.writeto(temp_path, overwrite=True)

                if config.verify:
                    with fits.open(temp_path) as verify_hdu:
                        verify_data = verify_hdu[0].data
                        if verify_data.shape != result.output_shape:
                            raise ValueError(f"Verification failed: shape mismatch")

                        if config.check_stats:
                            if verify_data.ndim == 3:
                                verify_data_orig_order = np.moveaxis(verify_data, 0, -1)
                            else:
                                verify_data_orig_order = verify_data

                            output_stats = compute_pixel_stats(verify_data_orig_order)
                            result.pixel_stats["output"] = output_stats

                            if not stats_match(input_stats, output_stats):
                                raise ValueError(f"Verification failed: pixel stats mismatch")

                os.replace(temp_path, fits_path)

            except Exception:
                if os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except OSError:
                        pass
                raise
        else:
            hdu.writeto(str(fits_path), overwrite=config.overwrite)

            if config.verify:
                with fits.open(str(fits_path)) as verify_hdu:
                    if verify_hdu[0].data.shape != result.output_shape:
                        result.error = "Verification failed: shape mismatch after write"
                        return result

        result.success = True
        return result

    except Exception as e:
        result.error = str(e)
        logging.debug(f"Exception details for {xisf_path}:", exc_info=True)
        return result
