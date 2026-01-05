"""Command-line interface for XISF to FITS conversion."""

import argparse
import logging
import signal
import sys
from pathlib import Path
from typing import Optional

from .batch import find_xisf_files, print_summary, request_shutdown, run_batch_conversion
from .converter import __version__, get_output_path
from .models import ConversionConfig


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


def _signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    request_shutdown()


def main():
    parser = argparse.ArgumentParser(
        description='Batch convert XISF files to FITS format',
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

    # Skip verification (faster)
    xisf2fits /path/to/images --no-verify --no-stats
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
                        help='Filename pattern filter (e.g., "Light_*.xisf")')
    parser.add_argument('--exclude', '-e', type=str, action='append', default=None,
                        help='Exclude paths containing string (repeatable)')
    parser.add_argument('--dry-run', '-d', action='store_true',
                        help='Show what would be converted without doing it')
    parser.add_argument('--jobs', '-j', type=int, default=1,
                        help='Number of parallel workers (default: 1)')
    parser.add_argument('--no-verify', action='store_true',
                        help='Skip verification of output files')
    parser.add_argument('--no-atomic', action='store_true',
                        help='Disable atomic writes')
    parser.add_argument('--no-stats', action='store_true',
                        help='Skip pixel statistics verification')
    parser.add_argument('--nan-handling', choices=['warn', 'zero', 'error'],
                        default='warn', help='How to handle NaN/Inf values')
    parser.add_argument('--flat-output', action='store_true',
                        help='Do not preserve directory structure in output')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--log-file', '-l', type=Path, default=None,
                        help='Write log to file')
    parser.add_argument('--version', action='version',
                        version=f'%(prog)s {__version__}')

    args = parser.parse_args()

    setup_logging(args.verbose, args.log_file)
    signal.signal(signal.SIGINT, _signal_handler)
    if hasattr(signal, 'SIGTERM'):
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

    if args.jobs < 1:
        logging.error("Jobs must be at least 1")
        sys.exit(1)

    recursive = not args.no_recursive
    xisf_files = find_xisf_files(args.directory, recursive=recursive,
                                  pattern=args.pattern, exclude=args.exclude)

    if not xisf_files:
        print(f"No .xisf files found in {args.directory}")
        sys.exit(0)

    print(f"Found {len(xisf_files)} XISF file(s) in {args.directory}")

    config = ConversionConfig(
        overwrite=args.overwrite,
        verify=not args.no_verify,
        output_dir=args.output_dir,
        preserve_structure=not args.flat_output,
        atomic_write=not args.no_atomic and sys.platform != 'win32',
        check_stats=not args.no_stats,
        nan_handling=args.nan_handling,
    )

    # Log active options
    options = []
    if config.overwrite:
        options.append("overwrite")
    if not config.verify:
        options.append("no-verify")
    if not config.check_stats:
        options.append("no-stats")
    if config.output_dir:
        options.append(f"output-dir={config.output_dir}")
    if args.pattern:
        options.append(f"pattern={args.pattern}")
    if args.exclude:
        options.append(f"exclude={args.exclude}")
    if args.jobs > 1:
        options.append(f"jobs={args.jobs}")
    print(f"Options: {', '.join(options) if options else 'defaults'}")

    if args.dry_run:
        _run_dry_run(xisf_files, config, args.directory)
        sys.exit(0)

    results = run_batch_conversion(
        xisf_files=xisf_files,
        config=config,
        root_dir=args.directory,
        workers=args.jobs,
        show_progress=True
    )

    summary = print_summary(results)

    if summary["failed"] > 0:
        sys.exit(1)
    sys.exit(0)


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

    print(f"\n{'='*50}")
    print("DRY RUN SUMMARY")
    print(f"{'='*50}")
    print(f"  Total files:     {len(xisf_files)}")
    print(f"  To convert:      {new_count if not config.overwrite else len(xisf_files)}")
    print(f"  Already exist:   {exists_count}")
    print(f"  Total size:      {total_size / (1024*1024):.1f} MB")
    if exists_count > 0 and not config.overwrite:
        print(f"\n  Use --overwrite to reconvert existing files")


if __name__ == '__main__':
    main()
