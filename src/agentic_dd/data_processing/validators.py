import gzip
import json
import csv
import zipfile
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Callable, Dict, Optional

class ValidationStatus(Enum):
    OK = "ok"
    FAILED = "failed"
    ERROR = "error"

@dataclass
class ValidationResult:
    status: ValidationStatus
    validator: str
    message: str

    @property
    def valid(self) -> bool:
        return self.status == ValidationStatus.OK

# --- Validator Functions ---

def check_archives():
    #TODO implement archive handling, reject all in individual methods below
    raise NotImplementedError

def heuristic_detection(file_path: str, sniff_bytes: int = 2048) -> ValidationResult:
    ERROR_MARKERS = (b'<html', b'<!doctype html', b'"error"', b'internal server error',
                      b'traceback (most recent call last)', b'502 bad gateway', b'503 service')
    validator = 'heuristic_detection'
    try:
        with open(file_path, 'rb') as f:
            head = f.read(sniff_bytes)
    except OSError as e:
        return ValidationResult(ValidationStatus.ERROR, validator, f'Could not read file: {e}')

    head_lower = head.lower()
    found = [m for m in ERROR_MARKERS if m in head_lower]
    if found:
        return ValidationResult(ValidationStatus.FAILED, validator, f'Error marker found: {found[0]!r}')
    return ValidationResult(ValidationStatus.OK, validator, "Heuristic check OK")

def check_zip_based(file_path: str, expected_members: Tuple[str, ...] = ()) -> ValidationResult:
    validator = "check_zip_based"
    try:
        with zipfile.ZipFile(file_path) as zf:
            bad = zf.testzip()
            if bad is not None:
                return ValidationResult(ValidationStatus.FAILED, validator, f"Corrupt member in zip: {bad}")
            names = zf.namelist()
            for member in expected_members:
                if member not in names:
                    return ValidationResult(ValidationStatus.FAILED, validator, f'Missing expected member: {member}')
    except zipfile.BadZipFile as e:
        return ValidationResult(ValidationStatus.FAILED, validator, f"Bad zip file: {e}")
    except OSError as e:
        return ValidationResult(ValidationStatus.ERROR, validator, f"Could not read file: {e}")
    return ValidationResult(ValidationStatus.OK, validator, "Zip structure OK")

def check_ole(file_path: str) -> ValidationResult:
    validator = "check_ole"
    MAGIC = b'\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1'
    try:
        with open(file_path, 'rb') as f:
            head = f.read(8)
    except OSError as e:
        return ValidationResult(ValidationStatus.ERROR, validator, f"Could not read file: {e}")
    if head != MAGIC:
        return ValidationResult(ValidationStatus.FAILED, validator, "Invalid OLE2 signature")
    return ValidationResult(ValidationStatus.OK, validator, "OLE2 signature OK")

def check_7z(file_path: str) -> ValidationResult:
    validator = "check_7z"
    MAGIC = b'7z\xBC\xAF\x27\x1C'
    try:
        with open(file_path, 'rb') as f:
            head = f.read(6)
    except OSError as e:
        return ValidationResult(ValidationStatus.ERROR, validator, f"Could not read file: {e}")
    if head != MAGIC:
        return ValidationResult(ValidationStatus.FAILED, validator, "Invalid 7z signature")
    return ValidationResult(ValidationStatus.OK, validator, "7z signature OK")

def check_gzip(file_path: str) -> ValidationResult:
    validator = "check_gzip"
    try:
        with gzip.open(file_path, 'rb') as f:
            f.read(1)
    except OSError as e:
        return ValidationResult(ValidationStatus.FAILED, validator, f"Invalid gzip file: {e}")
    return ValidationResult(ValidationStatus.OK, validator, "Gzip check OK")

def check_sav(file_path: str) -> ValidationResult:
    validator = "check_sav"
    try:
        with open(file_path, "rb") as f:
            head = f.read(4)
    except OSError as e:
        return ValidationResult(ValidationStatus.ERROR, validator, f"Could not read file: {e}")
    if head not in (b"$FL2", b"$FL3"):
        return ValidationResult(ValidationStatus.FAILED, validator, "Invalid SAV signature")
    return ValidationResult(ValidationStatus.OK, validator, "SAV signature OK")

def check_dta(file_path: str) -> ValidationResult:
    validator = "check_dta"
    try:
        with open(file_path, "rb") as f:
            head = f.read(11)
    except OSError as e:
        return ValidationResult(ValidationStatus.ERROR, validator, f"Could not read file: {e}")
    if (head.startswith(b"<stata_dta>") or (head and head[0] in (114, 115, 117, 118, 119, 120))):
        return ValidationResult(ValidationStatus.OK, validator, "DTA signature OK")
    return ValidationResult(ValidationStatus.FAILED, validator, "Invalid DTA signature")

def check_rds(file_path: str) -> ValidationResult:
    validator = "check_rds"
    try:
        with open(file_path, "rb") as f:
            head = f.read(5)
    except OSError as e:
        return ValidationResult(ValidationStatus.ERROR, validator, f"Could not read file: {e}")
    if head[:2] == b"\x1f\x8b":
        return ValidationResult(ValidationStatus.OK, validator, "RDS (gzip) signature OK")
    if head[:4] in (b"RDX2", b"RDX3", b"RDA2"):
        return ValidationResult(ValidationStatus.OK, validator, "RDS (uncompressed) signature OK")
    return ValidationResult(ValidationStatus.FAILED, validator, "Invalid RDS signature")

def check_csv(file_path: str, sample_lines: int = 50) -> ValidationResult:
    validator = "check_csv"
    try:
        with open(file_path, 'r', newline='', encoding='utf-8', errors='strict') as f:
            reader = csv.reader(f)
            rows = []
            for i, row in enumerate(reader):
                rows.append(row)
                if i + 1 >= sample_lines:
                    break
    except (OSError, UnicodeDecodeError, csv.Error) as e:
        return ValidationResult(ValidationStatus.FAILED, validator, f"Invalid CSV: {e}")
    if not rows:
        return ValidationResult(ValidationStatus.FAILED, validator, "Empty CSV file")
    expected_cols = len(rows[0])
    for idx, row in enumerate(rows):
        if len(row) != expected_cols:
            return ValidationResult(ValidationStatus.FAILED, validator, f"Inconsistent column count at row {idx}")
    return ValidationResult(ValidationStatus.OK, validator, "CSV check OK")

def check_json(file_path: str) -> ValidationResult:
    validator = "check_json"
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json.load(f)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        return ValidationResult(ValidationStatus.FAILED, validator, f"Invalid JSON: {e}")
    return ValidationResult(ValidationStatus.OK, validator, "JSON check OK")

def check_plaintext(file_path: str, sniff_bytes: int = 4096) -> ValidationResult:
    validator = "check_plaintext"
    try:
        with open(file_path, 'rb') as f:
            head = f.read(sniff_bytes)
    except OSError as e:
        return ValidationResult(ValidationStatus.ERROR, validator, f"Could not read file: {e}")
    if b'\x00' in head:
        return ValidationResult(ValidationStatus.FAILED, validator, "Null byte found, likely binary file")
    try:
        head.decode('utf-8')
    except UnicodeDecodeError as e:
        return ValidationResult(ValidationStatus.FAILED, validator, f"Not valid UTF-8 text: {e}")
    return ValidationResult(ValidationStatus.OK, validator, "Plaintext check OK")

# --- Registry ---

from functools import partial

EXT_TO_VALIDATOR: Dict[str, Callable[[str], ValidationResult]] = {
    '.pdf': heuristic_detection,
    '.md': heuristic_detection,
    '.html': heuristic_detection,
    '.doc': check_ole,
    '.xls': check_ole,
    '.sav': check_sav,
    '.dta': check_dta,
    '.rds': check_rds,
    '.csv': check_csv,
    '.json': check_json,
    '.txt': check_plaintext,
    '.r': check_plaintext,
    '.docx': partial(check_zip_based, expected_members=('[Content_Types].xml',)),
    '.xlsx': partial(check_zip_based, expected_members=('[Content_Types].xml',)),
    '.ods': partial(check_zip_based, expected_members=('mimetype',)),
    '.qdpx': partial(check_zip_based, expected_members=('project.qde',)),
}

def get_validator(ext: str) -> Callable[[str], ValidationResult]:
    """Returns the validator function for a given extension."""
    ext = ext.lower()
    if ext not in EXT_TO_VALIDATOR:
        raise NotImplementedError(f'Extension {ext} currently not supported')
    return EXT_TO_VALIDATOR[ext]