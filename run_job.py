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
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from siril_job_runner.job_runner import JobRunner
from siril_job_runner.job_config import validate_job_file, load_settings


def get_siril_interface():
    """
    Get Siril interface.

    Returns sirilpy interface if available, otherwise None.
    """
    try:
        from sirilpy import Siril
        siril = Siril()
        siril.Open()
        return siril
    except ImportError:
        print("WARNING: sirilpy not available. Install with: pip install sirilpy")
        print("Running in validation-only mode.")
        return None
    except Exception as e:
        print(f"WARNING: Could not connect to Siril: {e}")
        print("Make sure Siril is running with scripting enabled.")
        return None


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

    args = parser.parse_args()

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
    if not args.validate and not args.dry_run:
        siril = get_siril_interface()
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
            outputs = runner.run()
            print()
            print("Outputs:")
            for name, path in outputs.items():
                print(f"  {name}: {path}")
            runner.close()

    except Exception as e:
        print(f"ERROR: {e}")
        if 'runner' in locals():
            runner.close()
        sys.exit(1)

    finally:
        # Close Siril connection
        if siril is not None:
            try:
                siril.Close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
