"""
Protocol definitions for Siril interfaces.
"""

from typing import Protocol


class SirilInterface(Protocol):
    """Protocol for Siril scripting interface."""

    def cd(self, path: str) -> None: ...
    def convert(self, name: str, out: str) -> None: ...
    def stack(self, name: str, *args) -> None: ...
    def calibrate(self, name: str, *args, **kwargs) -> None: ...
