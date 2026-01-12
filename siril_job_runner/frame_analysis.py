"""
Frame analysis and reporting utilities.

Provides summary tables and analysis of frame collections.
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .fits_utils import FrameInfo


@dataclass
class RequirementsEntry:
    """Entry in the requirements table."""

    filter_name: str
    exposure: float
    temperature: float
    count: int

    @property
    def exposure_str(self) -> str:
        return f"{int(self.exposure)}s"

    @property
    def temp_str(self) -> str:
        return f"{int(round(self.temperature))}C"


def build_requirements_table(frames: list[FrameInfo]) -> list[RequirementsEntry]:
    """
    Build a requirements table from a list of frames.

    Groups frames by (filter, exposure, temperature) and counts them.
    """
    counts: dict[tuple[str, float, float], int] = defaultdict(int)

    for frame in frames:
        # Round temperature to nearest integer for grouping
        rounded_temp = round(frame.temperature)
        key = (frame.filter_name, frame.exposure, rounded_temp)
        counts[key] += 1

    entries = []
    for (filter_name, exposure, temp), count in sorted(counts.items()):
        entries.append(
            RequirementsEntry(
                filter_name=filter_name,
                exposure=exposure,
                temperature=temp,
                count=count,
            )
        )

    return entries


def get_unique_exposures(frames: list[FrameInfo]) -> set[float]:
    """Get unique exposure times from frames."""
    return {frame.exposure for frame in frames}


def get_unique_temperatures(frames: list[FrameInfo]) -> set[float]:
    """Get unique temperatures from frames (rounded to int)."""
    return {round(frame.temperature) for frame in frames}


def get_unique_filters(frames: list[FrameInfo]) -> set[str]:
    """Get unique filter names from frames."""
    return {frame.filter_name for frame in frames}


@dataclass
class DateSummaryEntry:
    """Summary of frames for a single date."""

    date: str
    temperature: int  # Most common temp for this date
    filter_counts: dict[str, str]  # filter -> count string (e.g., "11+60*")


def _extract_date_from_path(path: Path) -> Optional[str]:
    """
    Extract date from frame path.

    Expects structure like: .../M42/2025_01_15/L180/file.fit
    Returns the date part (e.g., "2025_01_15").
    """
    parts = path.parts
    # Look for a part matching date pattern YYYY_MM_DD
    for part in parts:
        if len(part) == 10 and part[4] == "_" and part[7] == "_":
            try:
                int(part[:4])  # year
                int(part[5:7])  # month
                int(part[8:10])  # day
                return part
            except ValueError:
                continue
    return None


def build_date_summary_table(frames: list[FrameInfo]) -> list[DateSummaryEntry]:
    """
    Build a summary table grouped by date.

    Returns entries sorted by date, with frame counts per filter.
    Shows exposure breakdown when multiple exposures exist.
    """
    # Group frames by date
    by_date: dict[str, list[FrameInfo]] = defaultdict(list)
    for frame in frames:
        date = _extract_date_from_path(frame.path)
        if date:
            by_date[date].append(frame)

    # Build summary for each date
    entries = []
    for date in sorted(by_date.keys()):
        date_frames = by_date[date]

        # Get most common temperature for this date
        temps = [round(f.temperature) for f in date_frames]
        temp = max(set(temps), key=temps.count)

        # Group by filter, then by exposure
        filter_data: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for frame in date_frames:
            exp = int(frame.exposure)
            filter_data[frame.filter_name][exp] += 1

        # Format filter counts
        filter_counts = {}
        for filter_name, exp_counts in filter_data.items():
            if len(exp_counts) == 1:
                # Single exposure
                exp, count = list(exp_counts.items())[0]
                filter_counts[filter_name] = str(count)
            else:
                # Multiple exposures - show as "long+short*"
                sorted_exps = sorted(exp_counts.items(), reverse=True)
                parts = []
                for i, (_exp, count) in enumerate(sorted_exps):
                    if i == 0:
                        parts.append(str(count))
                    else:
                        parts.append(f"{count}*")
                filter_counts[filter_name] = "+".join(parts)

        entries.append(
            DateSummaryEntry(
                date=date,
                temperature=temp,
                filter_counts=filter_counts,
            )
        )

    return entries


def format_date_summary_table(
    entries: list[DateSummaryEntry], filters: list[str]
) -> list[str]:
    """
    Format date summary as ASCII table lines.

    Args:
        entries: List of DateSummaryEntry
        filters: List of filter names to show as columns (in order)

    Returns:
        List of formatted lines for printing
    """
    if not entries:
        return []

    # Build header
    header = f"| {'Date':<10} | {'Temp':<5} |"
    for f in filters:
        header += f" {f:<7} |"

    separator = "|" + "-" * 12 + "|" + "-" * 7 + "|"
    for _ in filters:
        separator += "-" * 9 + "|"

    lines = [separator, header, separator]

    # Build rows
    for entry in entries:
        row = f"| {entry.date:<10} | {entry.temperature:>4}C |"
        for f in filters:
            count = entry.filter_counts.get(f, "-")
            row += f" {count:<7} |"
        lines.append(row)

    lines.append(separator)
    return lines
