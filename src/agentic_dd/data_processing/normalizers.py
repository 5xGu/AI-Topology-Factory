'''
Defines per-format a normalization process, which at the end returns an AI suited output format. Some caution has to be payed to the Docling service whereever
it is used, as its documentation as of early September 2026 was neither complete nor correct, and the services behaviour was inconsistent with unclear 
conditionality.

According to https://docs.hpc.gwdg.de/services/ai-services/arcana/getting-started/index.html (Arcanas uses Docling for parsing) supported formats:
    Text (.txt)
    Markdown (.md)
    Word (.docx, .dotx, .docm, .dotm)
    Powerpoint(.pptx, .potx, .ppsx, .pptm, .potm, .ppsm)
    PDF (.pdf)
    HTML (.html, .htm, .xhtml)

    -> Elsewhere they write all formats are supported. Sometimes they convert themselves, sometimes not.... This is noticably different from the
    original Docling repos documentation.
'''

import logging
import os
import shutil
from pathlib import Path
from typing import Callable, Dict, Optional

import pandas as pd

from parsers import call_docling
from io_utils import write_json, ensure_directory

logger = logging.getLogger(__name__)

# Type alias for a normalizer function
NormalizerFunc = Callable[[str, Path, str, str], None]

def _none(fp: str, out_dir: Path, root_dir: str, api_key: str) -> None:
    """Copies file without processing."""
    rel_path = Path(fp).relative_to(root_dir)
    dest = out_dir / rel_path #TODO put into own function in io_utils
    ensure_directory(dest.parent)
    shutil.copy2(fp, dest)

def _pdf(fp: str, out_dir: Path, root_dir: str, api_url: str, api_key: str) -> None:
    """Parses PDF to JSON via Docling."""
    data = call_docling(fp, api_url, api_key)
    rel_path = Path(fp).relative_to(root_dir)
    dest = out_dir / rel_path
    ensure_directory(dest.parent)
    write_json(data, dest)

def _docx(fp: str, out_dir: Path, root_dir: str, api_url: str, api_key: str) -> None:
    """Parses DOCX to JSON via Docling."""
    data = call_docling(fp, api_url, api_key)
    rel_path = Path(fp).relative_to(root_dir)
    dest = out_dir / rel_path
    ensure_directory(dest.parent)
    write_json(data, dest)

def _xlsx(fp: str, out_dir: Path, root_dir: str, api_url: str, api_key: str) -> None:
    """Converts XLSX to CSV."""
    rel_path = Path(fp).relative_to(root_dir)
    dest_base = out_dir / rel_path.with_suffix('') # remove extension to add sheet name
    ensure_directory(dest_base.parent)
    
    xls = pd.ExcelFile(fp)
    for sheet in xls.sheet_names:
        df = pd.read_excel(fp, sheet_name=sheet)
        # Append sheet name to filename
        sheet_dest = dest_base.with_name(f"{dest_base.name}_{sheet}.csv")
        df.to_csv(sheet_dest, index=False)

def _sav(fp: str, out_dir: Path, root_dir: str, api_url: str, api_key: str) -> None:
    """Converts SAV to CSV and metadata to JSON."""
    import pyreadstat
    df, meta = pyreadstat.read_sav(fp)
    
    rel_path = Path(fp).relative_to(root_dir)
    csv_dest = out_dir / rel_path.with_suffix('.csv')
    json_dest = out_dir / rel_path.with_name(f"{rel_path.stem}_metadata.json")
    
    ensure_directory(csv_dest.parent)
    df.to_csv(csv_dest, index=False)
    write_json(vars(meta), json_dest)


# Registry of normalizers
FORMAT_TO_NORMALIZER: Dict[str, Callable] = {
    '.pdf': _pdf,
    '.docx': _docx,
    '.html': _docx, # Assuming docling handles html similarly
    '.xlsx': _xlsx,
    '.txt': _none,
    '.csv': _none,
    '.json': _none,
    '.sav': _sav,
}

def get_normalizer(ext: str) -> Optional[Callable]:
    return FORMAT_TO_NORMALIZER.get(ext.lower())