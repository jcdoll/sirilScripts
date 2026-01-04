#!/usr/bin/env python3
"""
Batch XISF to FITS converter - Production Grade

Recursively finds all .xisf files and converts them to .fits.
Supports parallel processing, atomic writes, verification, and resume.

Requirements:
    pip install xisf astropy numpy tqdm

Author: Claude
License: MIT
"""

import argparse
import hashlib
import logging
import os
import shutil
import signal
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from astropy.io import fits
from xisf import XISF

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

__version__ = "1.0.0"

# Global flag for graceful shutdown
_shutdown_requested = False


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    global _shutdown_requested
    _shutdown_requested = True
    logging.warning("Shutdown requested, finishing current file...")


@dataclass
class ConversionResult:
    """Result of a single file conversion."""
    input_path: Path
    output_path: Optional[Path] = None
    success: bool = False
    skipped: bool = False
    skip_reason: str = ""
    error: str = ""
    warnings: list = field(default_factory=list)
    input_shape: tuple = ()
    output_shape: tuple = ()
    dtype: str = ""
    pixel_stats: dict = field(default_factory=dict)


@dataclass 
class ConversionConfig:
    """Configuration for conversion process."""
    overwrite: bool = False
    verify: bool = True
    output_dir: Optional[Path] = None
    preserve_structure: bool = True
    atomic_write: bool = True
    check_stats: bool = True
    nan_handling: str = "warn"  # "warn", "zero", "error"


def setup_logging(verbose: bool, log_file: Optional[Path] = None) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def check_disk_space(path: Path, required_bytes: int) -> bool:
    """Check if sufficient disk space is available."""
    try:
        stat = shutil.disk_usage(path)
        # Require 2x the space for safety (temp file + final file)
        return stat.free >= required_bytes * 2
    except OSError:
        return True  # If we can't check, proceed anyway


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


def stats_match(stats1: dict, stats2: dict, rtol: float = 1e-6) -> bool:
    """Check if two stat dictionaries match within tolerance."""
    if stats1.get("finite_count") != stats2.get("finite_count"):
        return False
    if stats1.get("nan_count") != stats2.get("nan_count"):
        return False
    if stats1.get("inf_count") != stats2.get("inf_count"):
        return False
    
    for key in ("min", "max", "mean"):
        v1, v2 = stats1.get(key), stats2.get(key)
        if v1 is None and v2 is None:
            continue
        if v1 is None or v2 is None:
            return False
        if not np.isclose(v1, v2, rtol=rtol):
            return False
    return True


def get_output_path(input_path: Path, config: ConversionConfig, 
                    root_dir: Optional[Path] = None) -> Path:
    """Determine output path for a given input file."""
    if config.output_dir is None:
        # In-place conversion
        return input_path.with_suffix('.fits')
    
    if config.preserve_structure and root_dir:
        # Preserve directory structure in output
        rel_path = input_path.relative_to(root_dir)
        output_path = config.output_dir / rel_path.with_suffix('.fits')
    else:
        # Flat output directory
        output_path = config.output_dir / input_path.with_suffix('.fits').name
    
    return output_path


def convert_xisf_to_fits(xisf_path: Path, config: ConversionConfig,
                         root_dir: Optional[Path] = None) -> ConversionResult:
    """
    Convert a single XISF file to FITS format.
    
    Args:
        xisf_path: Path to input .xisf file
        config: Conversion configuration
        root_dir: Root directory for structure preservation
        
    Returns:
        ConversionResult with details of the conversion
    """
    result = ConversionResult(input_path=xisf_path)
    
    try:
        fits_path = get_output_path(xisf_path, config, root_dir)
        result.output_path = fits_path
        
        # Check if output exists
        if fits_path.exists() and not config.overwrite:
            result.skipped = True
            result.skip_reason = "output exists"
            return result
        
        # Create output directory if needed
        fits_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check disk space (estimate based on input file size)
        input_size = xisf_path.stat().st_size
        if not check_disk_space(fits_path.parent, input_size * 3):
            result.error = "Insufficient disk space"
            return result
        
        # Read XISF file
        xisf = XISF(str(xisf_path))
        
        # Get metadata - contains geometry as (width, height, channels)
        meta = xisf.get_images_metadata()[0]
        geometry = meta.get('geometry')
        
        if geometry is None:
            result.error = "No geometry metadata found in XISF file"
            return result
        
        meta_width, meta_height, meta_channels = geometry
        
        # Read image data
        # xisf library returns (height, width, channels) by default (channels-last)
        im_data = xisf.read_image(0)
        result.input_shape = im_data.shape
        result.dtype = str(im_data.dtype)
        
        # Validate array dimensions
        if im_data.ndim == 2:
            arr_height, arr_width = im_data.shape
            arr_channels = 1
            
            if meta_channels != 1:
                result.error = (
                    f"Dimension mismatch: metadata says {meta_channels} channels "
                    f"but array is 2D (mono)"
                )
                return result
                
        elif im_data.ndim == 3:
            arr_height, arr_width, arr_channels = im_data.shape
        else:
            result.error = f"Unexpected {im_data.ndim}D array, expected 2D or 3D"
            return result
        
        # Verify dimensions match metadata
        if (arr_width != meta_width or arr_height != meta_height or 
            arr_channels != meta_channels):
            result.error = (
                f"Dimension mismatch: metadata=(w={meta_width}, h={meta_height}, "
                f"c={meta_channels}) but array=(h={arr_height}, w={arr_width}, "
                f"c={arr_channels})"
            )
            return result
        
        # Check for NaN/Inf values
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
        
        # Warn on unusual channel counts
        if arr_channels not in (1, 3, 4):
            result.warnings.append(f"Unusual channel count: {arr_channels}")
        
        # Compute input statistics for verification
        if config.check_stats:
            input_stats = compute_pixel_stats(im_data)
            result.pixel_stats["input"] = input_stats
        
        # Handle axis order for FITS
        # XISF: (height, width, channels) -> FITS: (channels, height, width)
        if im_data.ndim == 3:
            im_data = np.moveaxis(im_data, -1, 0)
            
            if im_data.shape != (arr_channels, arr_height, arr_width):
                result.error = (
                    f"Axis transpose failed: expected "
                    f"{(arr_channels, arr_height, arr_width)}, got {im_data.shape}"
                )
                return result
        
        result.output_shape = im_data.shape
        
        # Create FITS HDU
        hdu = fits.PrimaryHDU(im_data)
        
        # Transfer FITS keywords from XISF metadata
        if 'FITSKeywords' in meta:
            skip_keywords = {'SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 
                           'NAXIS2', 'NAXIS3', 'EXTEND'}
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
        
        # Add conversion metadata
        hdu.header['HISTORY'] = f'Converted from XISF by xisf2fits_batch v{__version__}'
        hdu.header['XISFSRC'] = (str(xisf_path.name), 'Original XISF filename')
        
        # Write FITS file (atomic or direct)
        if config.atomic_write:
            # Write to temp file first, then rename
            fd, temp_path = tempfile.mkstemp(
                suffix='.fits.tmp', 
                dir=fits_path.parent
            )
            os.close(fd)
            
            try:
                hdu.writeto(temp_path, overwrite=True)
                
                # Verify temp file before renaming
                if config.verify:
                    with fits.open(temp_path) as verify_hdu:
                        verify_data = verify_hdu[0].data
                        if verify_data.shape != result.output_shape:
                            raise ValueError(
                                f"Verification failed: shape mismatch "
                                f"{verify_data.shape} vs {result.output_shape}"
                            )
                        
                        if config.check_stats:
                            # For verification, transpose back to compare
                            if verify_data.ndim == 3:
                                verify_data_orig_order = np.moveaxis(verify_data, 0, -1)
                            else:
                                verify_data_orig_order = verify_data
                            
                            output_stats = compute_pixel_stats(verify_data_orig_order)
                            result.pixel_stats["output"] = output_stats
                            
                            if not stats_match(input_stats, output_stats):
                                raise ValueError(
                                    f"Verification failed: pixel stats mismatch\n"
                                    f"  Input:  {input_stats}\n"
                                    f"  Output: {output_stats}"
                                )
                
                # Atomic rename
                os.replace(temp_path, fits_path)
                
            except Exception:
                # Clean up temp file on failure
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise
        else:
            # Direct write
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


def find_xisf_files(root_dir: Path, recursive: bool = True,
                    pattern: Optional[str] = None) -> list[Path]:
    """Find all .xisf files in directory tree."""
    glob_pattern = '**/*.xisf' if recursive else '*.xisf'
    files = list(root_dir.glob(glob_pattern))
    
    if pattern:
        import fnmatch
        files = [f for f in files if fnmatch.fnmatch(f.name, pattern)]
    
    return sorted(files)


def convert_single_file_wrapper(args: tuple) -> ConversionResult:
    """Wrapper for multiprocessing - unpacks arguments."""
    xisf_path, config, root_dir = args
    return convert_xisf_to_fits(xisf_path, config, root_dir)


def run_batch_conversion(
    xisf_files: list[Path],
    config: ConversionConfig,
    root_dir: Path,
    workers: int = 1,
    show_progress: bool = True
) -> list[ConversionResult]:
    """
    Run batch conversion with optional parallelism.
    
    Args:
        xisf_files: List of XISF files to convert
        config: Conversion configuration
        root_dir: Root directory for structure preservation
        workers: Number of parallel workers (1 = sequential)
        show_progress: Show progress bar if tqdm available
        
    Returns:
        List of ConversionResult objects
    """
    results = []
    
    # Prepare arguments for each file
    work_items = [(f, config, root_dir) for f in xisf_files]
    
    # Progress bar setup
    if show_progress and TQDM_AVAILABLE:
        pbar = tqdm(total=len(xisf_files), desc="Converting", unit="file")
    else:
        pbar = None
    
    try:
        if workers == 1:
            # Sequential processing
            for xisf_path, cfg, root in work_items:
                if _shutdown_requested:
                    logging.warning("Shutdown requested, stopping...")
                    break
                    
                result = convert_xisf_to_fits(xisf_path, cfg, root)
                results.append(result)
                
                if pbar:
                    pbar.update(1)
                    
                # Log result
                if result.success:
                    logging.debug(f"Converted: {xisf_path}")
                elif result.skipped:
                    logging.debug(f"Skipped ({result.skip_reason}): {xisf_path}")
                else:
                    logging.error(f"Failed: {xisf_path} - {result.error}")
                
                for warning in result.warnings:
                    logging.warning(f"  {xisf_path}: {warning}")
        else:
            # Parallel processing
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(convert_single_file_wrapper, item): item[0] 
                    for item in work_items
                }
                
                for future in as_completed(futures):
                    if _shutdown_requested:
                        logging.warning("Shutdown requested, cancelling remaining...")
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
                    
                    xisf_path = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        result = ConversionResult(
                            input_path=xisf_path,
                            error=f"Worker exception: {e}"
                        )
                        results.append(result)
                    
                    if pbar:
                        pbar.update(1)
                    
                    if result.success:
                        logging.debug(f"Converted: {xisf_path}")
                    elif result.skipped:
                        logging.debug(f"Skipped ({result.skip_reason}): {xisf_path}")
                    else:
                        logging.error(f"Failed: {xisf_path} - {result.error}")
    finally:
        if pbar:
            pbar.close()
    
    return results


def print_summary(results: list[ConversionResult]) -> dict:
    """Print and return conversion summary."""
    summary = {
        "total": len(results),
        "converted": sum(1 for r in results if r.success),
        "skipped": sum(1 for r in results if r.skipped),
        "failed": sum(1 for r in results if not r.success and not r.skipped),
        "warnings": sum(len(r.warnings) for r in results),
    }
    
    print("\n" + "=" * 50)
    print("CONVERSION SUMMARY")
    print("=" * 50)
    print(f"  Total files:  {summary['total']}")
    print(f"  Converted:    {summary['converted']}")
    print(f"  Skipped:      {summary['skipped']}")
    print(f"  Failed:       {summary['failed']}")
    print(f"  Warnings:     {summary['warnings']}")
    
    # Show failed files
    failed = [r for r in results if not r.success and not r.skipped]
    if failed:
        print("\nFailed files:")
        for r in failed[:10]:  # Limit output
            print(f"  {r.input_path}: {r.error}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Batch convert XISF files to FITS format (Production Grade)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert all XISF files in directory tree
    python xisf2fits_batch.py /path/to/images

    # Parallel conversion with 4 workers
    python xisf2fits_batch.py /path/to/images -j 4

    # Output to separate directory, preserving structure
    python xisf2fits_batch.py /path/to/images --output-dir /path/to/output

    # Dry run
    python xisf2fits_batch.py /path/to/images --dry-run

    # Skip verification (faster but less safe)
    python xisf2fits_batch.py /path/to/images --no-verify

    # Replace NaN/Inf with zeros instead of warning
    python xisf2fits_batch.py /path/to/images --nan-handling zero
        """
    )
    
    parser.add_argument('directory', type=Path,
                        help='Root directory to search for XISF files')
    parser.add_argument('--output-dir', '-O', type=Path, default=None,
                        help='Output directory (default: convert in place)')
    parser.add_argument('--overwrite', '-o', action='store_true',
                        help='Overwrite existing FITS files')
    parser.add_argument('--no-recursive', '-n', action='store_true',
                        help='Do not search subdirectories')
    parser.add_argument('--pattern', '-p', type=str, default=None,
                        help='Filename pattern filter (e.g., "*_light_*.xisf")')
    parser.add_argument('--dry-run', '-d', action='store_true',
                        help='Show what would be converted without doing it')
    parser.add_argument('--jobs', '-j', type=int, default=1,
                        help='Number of parallel workers (default: 1)')
    parser.add_argument('--no-verify', action='store_true',
                        help='Skip verification of output files')
    parser.add_argument('--no-atomic', action='store_true',
                        help='Disable atomic writes (faster but less safe)')
    parser.add_argument('--no-stats', action='store_true',
                        help='Skip pixel statistics verification')
    parser.add_argument('--nan-handling', choices=['warn', 'zero', 'error'],
                        default='warn',
                        help='How to handle NaN/Inf values (default: warn)')
    parser.add_argument('--flat-output', action='store_true',
                        help='Do not preserve directory structure in output')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--log-file', '-l', type=Path, default=None,
                        help='Write log to file')
    parser.add_argument('--version', action='version', 
                        version=f'%(prog)s {__version__}')

    args = parser.parse_args()
    
    # Setup
    setup_logging(args.verbose, args.log_file)
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):  # Not available on Windows
        signal.signal(signal.SIGTERM, signal_handler)
    
    # Validate inputs
    if not args.directory.exists():
        logging.error(f"Directory not found: {args.directory}")
        sys.exit(1)
    
    if args.output_dir and not args.output_dir.exists():
        try:
            args.output_dir.mkdir(parents=True)
            logging.info(f"Created output directory: {args.output_dir}")
        except OSError as e:
            logging.error(f"Cannot create output directory: {e}")
            sys.exit(1)
    
    if args.jobs < 1:
        logging.error("Jobs must be at least 1")
        sys.exit(1)
    
    if args.jobs > 1 and not TQDM_AVAILABLE:
        logging.warning("Install tqdm for progress bars: pip install tqdm")
    
    # Find files
    recursive = not args.no_recursive
    xisf_files = find_xisf_files(args.directory, recursive=recursive, 
                                  pattern=args.pattern)
    
    if not xisf_files:
        print(f"No .xisf files found in {args.directory}")
        sys.exit(0)

    print(f"Found {len(xisf_files)} XISF file(s) in {args.directory}")

    # Build config
    config = ConversionConfig(
        overwrite=args.overwrite,
        verify=not args.no_verify,
        output_dir=args.output_dir,
        preserve_structure=not args.flat_output,
        atomic_write=not args.no_atomic,
        check_stats=not args.no_stats,
        nan_handling=args.nan_handling,
    )
    
    # Dry run
    if args.dry_run:
        new_count = 0
        exists_count = 0
        total_size = 0

        print("\nDry run - files that would be converted:\n")
        for xisf_path in xisf_files:
            output_path = get_output_path(xisf_path, config, args.directory)
            file_size = xisf_path.stat().st_size
            total_size += file_size

            if output_path.exists():
                exists_count += 1
                status = "[exists - skip]" if not config.overwrite else "[exists - overwrite]"
            else:
                new_count += 1
                status = "[new]"

            print(f"  {xisf_path.name}")
            print(f"    -> {output_path} {status}")

        print(f"\n{'='*50}")
        print("DRY RUN SUMMARY")
        print(f"{'='*50}")
        print(f"  Total files:     {len(xisf_files)}")
        print(f"  To convert:      {new_count if not config.overwrite else len(xisf_files)}")
        print(f"  Already exist:   {exists_count}")
        print(f"  Total size:      {total_size / (1024*1024):.1f} MB")
        if exists_count > 0 and not config.overwrite:
            print(f"\n  Use --overwrite to reconvert existing files")
        sys.exit(0)
    
    # Run conversion
    results = run_batch_conversion(
        xisf_files=xisf_files,
        config=config,
        root_dir=args.directory,
        workers=args.jobs,
        show_progress=True
    )
    
    # Summary
    summary = print_summary(results)
    
    # Exit code
    if summary["failed"] > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == '__main__':
    main()