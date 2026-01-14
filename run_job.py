#!/usr/bin/env python3
"""
CLI entry point for Siril job processing.

Usage:
    python run_job.py jobs/M42_Jan2024.json
    python run_job.py jobs/M42_Jan2024.json --validate
    python run_job.py jobs/M42_Jan2024.json --dry-run
    python run_job.py jobs/M42_Jan2024.json --stage preprocess
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Fix Windows console encoding for pysiril output (contains Greek letters like σ)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


class TeeWriter:
    """Write to both stdout and a file."""

    def __init__(self, file_path: Path):
        self.file = open(file_path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from siril_job_runner.job_config import load_settings, validate_job_file
from siril_job_runner.job_runner import JobRunner


def get_siril_interface():
    """
    Get Siril interface.

    Returns SirilWrapper around pysiril if available, otherwise None.
    """
    try:
        from pysiril.siril import Siril

        from siril_job_runner.siril_wrapper import SirilWrapper

        siril = Siril()
        siril.Open()
        return SirilWrapper(siril), siril  # Return wrapper and raw for cleanup
    except ImportError:
        print("WARNING: pysiril not available.")
        print("Install with: uv pip install git+https://gitlab.com/free-astro/pysiril.git")
        print("Running in validation-only mode.")
        return None, None
    except Exception as e:
        print(f"WARNING: Could not connect to Siril: {e}")
        print("Make sure Siril is installed and accessible.")
        return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Run Siril image processing jobs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_job.py jobs/M42.json              # Run full pipeline
    python run_job.py jobs/M42.json --validate   # Validate only
    python run_job.py jobs/M42.json --dry-run    # Show what would happen
    python run_job.py jobs/M42.json --stage calibrate
        """,
    )

    parser.add_argument(
        "job_file",
        type=Path,
        help="Path to job JSON file",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate job file and check calibration, then exit",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without executing",
    )

    parser.add_argument(
        "--stage",
        choices=["calibrate", "preprocess", "compose"],
        help="Run only a specific stage",
    )

    parser.add_argument(
        "--base-path",
        type=Path,
        default=None,
        help="Base path for data (default: parent of job file)",
    )

    parser.add_argument(
        "--log",
        action="store_true",
        help="Write output to timestamped log file in logs/ folder",
    )

    args = parser.parse_args()

    # Set up logging to file if requested
    tee = None
    if args.log:
        logs_dir = Path(__file__).parent / "logs"
        logs_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_name = args.job_file.stem
        log_path = logs_dir / f"{job_name}_{timestamp}.log"
        tee = TeeWriter(log_path)
        sys.stdout = tee
        sys.stderr = tee
        print(f"Logging to: {log_path}")

    # Validate job file exists
    if not args.job_file.exists():
        print(f"ERROR: Job file not found: {args.job_file}")
        sys.exit(1)

    # Quick validation check
    is_valid, error = validate_job_file(args.job_file)
    if not is_valid:
        print(f"ERROR: Invalid job file: {error}")
        sys.exit(1)

    # Determine base path
    repo_root = Path(__file__).parent
    if args.base_path:
        base_path = args.base_path
    else:
        # Try to load from settings.json
        settings = load_settings(repo_root)
        if "base_path" in settings:
            base_path = Path(settings["base_path"])
        else:
            print("ERROR: No base_path specified.")
            print("Either:")
            print("  1. Use --base-path argument")
            print("  2. Create settings.json with base_path (copy from settings.template.json)")
            sys.exit(1)

    # Get Siril interface
    siril = None
    siril_raw = None
    if not args.validate and not args.dry_run:
        siril, siril_raw = get_siril_interface()
        if siril is None and not args.validate:
            print("Cannot run without Siril connection. Use --validate or --dry-run.")
            sys.exit(1)

    # Create job runner
    try:
        runner = JobRunner(
            job_path=args.job_file,
            base_path=base_path,
            siril=siril,
            dry_run=args.dry_run,
        )

        if args.validate:
            # Validation only
            result = runner.validate()
            print()
            if result.valid:
                print("Validation PASSED")
                print(f"  {len(result.frames)} light frames found")
                print(f"  {len(result.buildable_calibration)} calibration masters to build")
            else:
                print("Validation FAILED")
                print(f"  Missing: {', '.join(result.missing_calibration)}")
            runner.close()
            sys.exit(0 if result.valid else 1)

        elif args.stage:
            # Run specific stage
            validation = runner.validate()
            if not validation.valid:
                print(f"ERROR: {validation.message}")
                runner.close()
                sys.exit(1)

            if args.stage == "calibrate":
                runner.run_calibration(validation)
            elif args.stage == "preprocess":
                cal_paths = runner.run_calibration(validation)
                runner.run_preprocessing(cal_paths)
            elif args.stage == "compose":
                # Need stacks to exist
                print("ERROR: Compose stage requires preprocessing to be complete")
                sys.exit(1)

            runner.close()

        else:
            # Full pipeline
            result = runner.run()
            if result:
                print()
                print("Outputs:")
                print(f"  Linear: {result.linear_path}")
                if result.linear_pcc_path:
                    print(f"  Linear (PCC): {result.linear_pcc_path}")
                print(f"  Auto FIT: {result.auto_fit}")
                print(f"  Auto TIF: {result.auto_tif}")
                print(f"  Auto JPG: {result.auto_jpg}")
                print(f"  Stacks: {result.stacks_dir}")
            runner.close()

    except Exception as e:
        print(f"ERROR: {e}")
        if 'runner' in locals():
            runner.close()
        sys.exit(1)

    finally:
        # Close Siril connection (use raw pysiril object)
        if siril_raw is not None:
            try:
                siril_raw.Close()
            except Exception:
                pass

        # Close log file
        if tee is not None:
            sys.stdout = tee.stdout
            sys.stderr = tee.stdout
            tee.close()


if __name__ == "__main__":
    main()
