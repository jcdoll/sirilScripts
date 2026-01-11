"""Command-line interface for XISF to FITS conversion."""

import argparse
import contextlib
import logging
import signal
import sys
from pathlib import Path

from .batch import find_xisf_files, print_summary, request_shutdown, run_batch_conversion
from .converter import __version__, get_output_path
from .models import ConversionConfig


def setup_logging(verbose: bool, log_file: Path | None = None) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def _signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    request_shutdown()


def get_siril():
    """
    Get Siril interface.

    Returns SirilWrapper around pysiril if available, otherwise None.
    """
    try:
        from pysiril.siril import Siril

        siril = Siril()
        siril.Open()
        return siril
    except ImportError:
        print("ERROR: pysiril not available.")
        print("pysiril comes bundled with Siril 1.2+")
        print("Make sure Siril is installed and running with scripting enabled.")
        return None
    except Exception as e:
        print(f"ERROR: Could not connect to Siril: {e}")
        print("Make sure Siril is running with scripting enabled.")
        return None


class SirilWrapper:
    """Simple wrapper to provide method interface for pysiril."""

    def __init__(self, siril):
        self._siril = siril

    def _quote(self, path: str) -> str:
        """Quote path if it contains spaces."""
        path = path.replace("\\", "/")
        if " " in path:
            return f'"{path}"'
        return path

    def cd(self, path: str) -> bool:
        return self._siril.Execute(f"cd {self._quote(path)}")

    def load(self, path: str) -> bool:
        return self._siril.Execute(f"load {self._quote(path)}")

    def save(self, path: str) -> bool:
        return self._siril.Execute(f"save {self._quote(path)}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch convert XISF files to FITS format using Siril",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert all XISF files in directory tree
    xisf2fits /path/to/images

    # Dry run - see what would be converted
    xisf2fits /path/to/images --dry-run

    # Exclude processed folders
    xisf2fits /path/to/images -e process -e calibrated

    # Only convert light frames
    xisf2fits /path/to/images --pattern "Light_*.xisf"

Requirements:
    Siril must be running with scripting enabled before running this tool.
        """,
    )

    parser.add_argument("directory", type=Path, help="Root directory to search for XISF files")
    parser.add_argument(
        "--output-dir",
        "-O",
        type=Path,
        default=None,
        help="Output directory (default: convert in place)",
    )
    parser.add_argument(
        "--overwrite", "-o", action="store_true", help="Overwrite existing FITS files"
    )
    parser.add_argument(
        "--no-recursive", "-n", action="store_true", help="Do not search subdirectories"
    )
    parser.add_argument(
        "--pattern",
        "-p",
        type=str,
        default=None,
        help='Filename pattern filter (e.g., "Light_*.xisf")',
    )
    parser.add_argument(
        "--exclude",
        "-e",
        type=str,
        action="append",
        default=None,
        help="Exclude paths containing string (repeatable)",
    )
    parser.add_argument(
        "--dry-run",
        "-d",
        action="store_true",
        help="Show what would be converted without doing it",
    )
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help="Do not preserve directory structure in output",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--log-file", "-l", type=Path, default=None, help="Write log to file")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    args = parser.parse_args()

    setup_logging(args.verbose, args.log_file)
    signal.signal(signal.SIGINT, _signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _signal_handler)

    if not args.directory.exists():
        logging.error(f"Directory not found: {args.directory}")
        sys.exit(1)

    if args.output_dir and not args.output_dir.exists():
        try:
            args.output_dir.mkdir(parents=True)
        except OSError as e:
            logging.error(f"Cannot create output directory: {e}")
            sys.exit(1)

    recursive = not args.no_recursive
    xisf_files = find_xisf_files(
        args.directory, recursive=recursive, pattern=args.pattern, exclude=args.exclude
    )

    if not xisf_files:
        print(f"No .xisf files found in {args.directory}")
        sys.exit(0)

    print(f"Found {len(xisf_files)} XISF file(s) in {args.directory}")

    config = ConversionConfig(
        overwrite=args.overwrite,
        output_dir=args.output_dir,
        preserve_structure=not args.flat_output,
    )

    # Log active options
    options = []
    if config.overwrite:
        options.append("overwrite")
    if config.output_dir:
        options.append(f"output-dir={config.output_dir}")
    if args.pattern:
        options.append(f"pattern={args.pattern}")
    if args.exclude:
        options.append(f"exclude={args.exclude}")
    print(f"Options: {', '.join(options) if options else 'defaults'}")

    if args.dry_run:
        _run_dry_run(xisf_files, config, args.directory)
        sys.exit(0)

    # Connect to Siril
    raw_siril = get_siril()
    if raw_siril is None:
        sys.exit(1)

    siril = SirilWrapper(raw_siril)

    try:
        results = run_batch_conversion(
            xisf_files=xisf_files,
            siril=siril,
            config=config,
            root_dir=args.directory,
            show_progress=True,
        )

        summary = print_summary(results)

        if summary["failed"] > 0:
            sys.exit(1)
        sys.exit(0)
    finally:
        # Close Siril connection
        with contextlib.suppress(Exception):
            raw_siril.Close()


def _run_dry_run(xisf_files: list[Path], config: ConversionConfig, root_dir: Path):
    """Execute dry run - show what would be converted."""
    new_count = 0
    exists_count = 0
    total_size = 0

    print("\nDry run - files that would be converted:\n")
    for xisf_path in xisf_files:
        output_path = get_output_path(xisf_path, config, root_dir)
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

    print(f"\n{'=' * 50}")
    print("DRY RUN SUMMARY")
    print(f"{'=' * 50}")
    print(f"  Total files:     {len(xisf_files)}")
    print(f"  To convert:      {new_count if not config.overwrite else len(xisf_files)}")
    print(f"  Already exist:   {exists_count}")
    print(f"  Total size:      {total_size / (1024 * 1024):.1f} MB")
    if exists_count > 0 and not config.overwrite:
        print("\n  Use --overwrite to reconvert existing files")


if __name__ == "__main__":
    main()
