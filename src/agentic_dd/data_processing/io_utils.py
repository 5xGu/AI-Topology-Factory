"""
Functions for in- and out operations of the system. Originally introduced to uniformly handle a large dataset, its use for Discuss Data
is not clear. Some functionality belongs to the upload process, other might be useful for intra-system organization. 
"""

import os
import re
import shutil
import unicodedata
from pathlib import Path
from typing import List

def clean_name(name: str) -> str:
    """Normalizes and sanitizes filenames."""
    name = unicodedata.normalize("NFKD", name)
    replacements = {
        "–": "-", "—": "-", "−": "-",
        "“": '"', "”": '"', "’": "'", "‘": "'",
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    
    name = re.sub(r"[\r\n\t]+", " ", name)
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9._-]", "", name)
    name = re.sub(r"_+", "_", name)
    name = re.sub(r"-+", "-", name)
    return name.strip("._-")

def unique_path(path: Path) -> Path:
    """Returns a unique path by appending _1, _2, etc., if path exists."""
    if not path.exists():
        return path
    counter = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1

def write_json(data, fp: Path):
    """Writes data to a JSON file."""
    fp = fp.with_suffix(".json")
    try:
        import json
        fp.write_text(json.dumps(data, indent=4, default=str))
    except (TypeError, OSError) as e:
        raise ValueError(f"Failed to write JSON file '{fp}': {e}") from e

def ensure_directory(path: Path):
    """Ensures a directory exists."""
    path.mkdir(parents=True, exist_ok=True)