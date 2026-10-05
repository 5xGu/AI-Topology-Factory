import os
import logging
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from collections import defaultdict

from validators import get_validator, ValidationResult
from archives import extract_safely, ArchiveError, _is_archive
from io_utils import clean_name, unique_path, ensure_directory
from normalizers import get_normalizer

# Setup logging
def setup_logger(log_dir: str, name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(Path(log_dir) / name)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

def is_cached(fp: str, out_dir: Path, root_dir: Path, logger: logging.Logger) -> bool:
    """Checks if a processed version exists and is newer than the source."""
    try:
        rel_path = Path(fp).relative_to(root_dir) #TODO this relative_to crops up often, I should unify it into one custom function for controllability
    except ValueError:
        return False
    
    # We check if any normalized format exists for this file
    # This logic depends on how you name outputs. Assuming same name, different ext.
    cached_path = out_dir / rel_path.with_suffix('') 
    
    if not cached_path.parent.exists():
        return False

    # Check against known output extensions
    # In a real app, pass the list of target extensions dynamically
    target_exts = ['.csv', '.json', '.md']  #TODO make configurable, best case, the entire file format mappings
    
    source_mtime = Path(fp).stat().st_mtime
    
    for ext in target_exts:
        candidate = cached_path.with_suffix(ext)
        if candidate.exists() and candidate.stat().st_mtime >= source_mtime:
            logger.info(f"Up-to-date file parse exists: {candidate}")
            return True
            
    return False

def process_file(
    file_path: str, 
    root_dir: str, 
    out_dir: str, 
    api_url: str, 
    api_key: str, 
    logger: logging.Logger
):
    """Validates and Normalizes a single file."""
    ext = Path(file_path).suffix.lower()
    
    # 1. Check Cache
    if is_cached(file_path, Path(out_dir), Path(root_dir), logger):
        return

    # 2. Validate
    try:
        validator_func = get_validator(ext)
        result: ValidationResult = validator_func(file_path)
        if not result.valid:
            logger.warning(f"Validation failed for {file_path}: {result.message}")
            return
    except NotImplementedError:
        logger.error(f"No validator for {ext} ({file_path})")
        return
    except Exception as e:
        logger.exception(f"Validation error for {file_path}")
        return

    # 3. Normalize
    normalizer_func = get_normalizer(ext)
    if normalizer_func is None:
        logger.warning(f"No normalizer for {ext} ({file_path})")
        return

    try:
        normalizer_func(file_path, Path(out_dir), root_dir, api_url, api_key)
        logger.info(f"Processed {file_path}")
    except Exception as e:
        logger.exception(f"Error normalizing {file_path}")

def main():
    load_dotenv(find_dotenv())
    
    log_dir = os.getenv("LOG_DIR") #TODO this needs to be refactored according to new setup
    logger = setup_logger(log_dir, 'parser.log')

    data_root = os.getenv("DATA_ROOT_DIR")
    parsed_dir = os.getenv("PARSED_DATA_DIR")
    api_url = os.getenv("DOCLING_BASE_URL")
    api_key = os.getenv("API_KEY")

    # 1. Discover and Clean Files
    # In a real refactor, this discovery logic might be its own module 'discovery.py'
    # For brevity, I'm putting a simplified version here.
    
    all_files = defaultdict(list)
    
    # Rename directories first
    for dirpath, dirnames, _ in os.walk(data_root, topdown=False):
        current = Path(dirpath)
        cleaned = clean_name(current.name)
        if cleaned != current.name:
            new_path = current.with_name(cleaned)
            if new_path.exists():
                new_path = unique_path(new_path)
            current.rename(new_path)

    # Discover files
    for dirpath, _, filenames in os.walk(data_root):
        for fn in filenames:
            fp = Path(dirpath) / fn
            
            # Handle Archives (Simplified for refactor)
            if _is_archive(fn):
                # Extract logic would go here, similar to original FileHandler
                # For now, we skip or handle separately to keep this example focused
                continue

            # Rename files
            cleaned_name = clean_name(fp.stem) + fp.suffix
            new_fp = fp.with_name(cleaned_name)
            if new_fp != fp:
                new_fp = unique_path(new_fp)
                fp.rename(new_fp)
            
            ext = new_fp.suffix.lower()
            all_files[ext].append(str(new_fp))

    # 2. Process Files
    for ext, files in all_files.items():
        logger.info(f"Processing {len(files)} files of type {ext}")
        for fp in files:
            process_file(fp, data_root, parsed_dir, api_url, api_key, logger)

if __name__ == "__main__":
    main()