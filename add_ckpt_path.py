"""Locate the repository code independently of checkpoint storage."""
import sys
from pathlib import Path


def add_path_to_dust3r(ckpt):
    # Retain the historical helper name for upstream callers.
    paths = [Path(ckpt).resolve().parent, Path(__file__).resolve().parent / "src"]
    for directory in paths:
        value = str(directory)
        if value not in sys.path:
            sys.path.insert(0, value)
