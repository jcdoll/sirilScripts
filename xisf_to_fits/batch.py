"""Batch processing for XISF to FITS conversion."""

import fnmatch
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from .converter import convert_xisf_to_fits
from .models import ConversionConfig, ConversionResult

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Global flag for graceful shutdown
_shutdown_requested = False


def request_shutdown():
    """Request graceful shutdown of batch processing."""
    global _shutdown_requested
    _shutdown_requested = True
    logging.warning("Shutdown requested, finishing current file...")


def find_xisf_files(root_dir: Path, recursive: bool = True,
                    pattern: Optional[str] = None,
                    exclude: Optional[list[str]] = None) -> list[Path]:
    """Find all .xisf files in directory tree."""
    glob_pattern = '**/*.xisf' if recursive else '*.xisf'
    files = list(root_dir.glob(glob_pattern))

    if pattern:
        files = [f for f in files if fnmatch.fnmatch(f.name, pattern)]

    if exclude:
        files = [f for f in files if not any(ex in str(f) for ex in exclude)]

    return sorted(files)


def _convert_single_file_wrapper(args: tuple) -> ConversionResult:
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
    """Run batch conversion with optional parallelism."""
    global _shutdown_requested
    results = []
    work_items = [(f, config, root_dir) for f in xisf_files]

    if show_progress and TQDM_AVAILABLE:
        pbar = tqdm(total=len(xisf_files), desc="Converting", unit="file",
                    bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')
    else:
        pbar = None

    try:
        if workers == 1:
            for i, (xisf_path, cfg, root) in enumerate(work_items):
                if _shutdown_requested:
                    logging.warning("Shutdown requested, stopping...")
                    break

                if pbar:
                    pbar.set_postfix_str(xisf_path.name[:40])
                else:
                    pct = (i / len(work_items)) * 100
                    print(f"[{i+1}/{len(work_items)}] ({pct:.0f}%) {xisf_path.name}")

                result = convert_xisf_to_fits(xisf_path, cfg, root)
                results.append(result)

                if pbar:
                    pbar.update(1)

                if result.success:
                    logging.debug(f"Converted: {xisf_path}")
                elif result.skipped:
                    if not pbar:
                        print(f"  -> Skipped ({result.skip_reason})")
                    logging.debug(f"Skipped ({result.skip_reason}): {xisf_path}")
                else:
                    if pbar:
                        pbar.write(f"FAILED: {xisf_path.name} - {result.error}")
                    else:
                        print(f"  -> FAILED: {result.error}")
                    logging.error(f"Failed: {xisf_path} - {result.error}")

                for warning in result.warnings:
                    if pbar:
                        pbar.write(f"WARNING: {xisf_path.name}: {warning}")
                    logging.warning(f"  {xisf_path}: {warning}")
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(_convert_single_file_wrapper, item): item[0]
                    for item in work_items
                }

                completed = 0
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

                    completed += 1
                    if pbar:
                        pbar.set_postfix_str(xisf_path.name[:40])
                        pbar.update(1)
                    else:
                        pct = (completed / len(work_items)) * 100
                        status = "OK" if result.success else ("skip" if result.skipped else "FAIL")
                        print(f"[{completed}/{len(work_items)}] ({pct:.0f}%) {xisf_path.name} [{status}]")

                    if result.success:
                        logging.debug(f"Converted: {xisf_path}")
                    elif result.skipped:
                        logging.debug(f"Skipped ({result.skip_reason}): {xisf_path}")
                    else:
                        if pbar:
                            pbar.write(f"FAILED: {xisf_path.name} - {result.error}")
                        logging.error(f"Failed: {xisf_path} - {result.error}")
    finally:
        if pbar:
            pbar.close()
            print()

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

    failed = [r for r in results if not r.success and not r.skipped]
    if failed:
        print("\nFailed files:")
        for r in failed[:10]:
            print(f"  {r.input_path}: {r.error}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")

    return summary
