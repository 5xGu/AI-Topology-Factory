'''
Shared file for reading .csv/.json input, used by agent/worker single-file and direct/branch mulit-file read.
'''
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

SUPPORTED_SUFFIXES = {".csv", ".json"}


def strip_images(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: strip_images(v) for k, v in data.items() if k != "images"}
    if isinstance(data, list):
        return [strip_images(item) for item in data]
    return data


def read_one_file(fp: str, tool_call_budget: Optional[int] = 20000) -> Any:
    ''' Reads a .csv or .json file and returns its contents directly. If a file is larger than the budget, use
    csv_schema/csv_sample or json_schema_outline instead '''
    p = Path(fp)
    try:
        if p.suffix == ".csv":
            contents = pd.read_csv(p).to_dict(orient="records")
        elif p.suffix == ".json":
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and 'images' in data:
                del data['images']  # remove images for the moment, they need additional support
            contents = data
        else:
            return f"Unsupported file type: {p.suffix}"
    except Exception as e:
        return f"Error reading '{fp}': {type(e).__name__}: {e}"

    if tool_call_budget is None:
        return contents

    from agentic_dd.agentlib.tokenization import real_token_count  # deferred: tokenization imports read_files from here
    text = contents if isinstance(contents, str) else json.dumps(contents)
    token_count = real_token_count(text, "qwen3.8-27b")
    effective_count = token_count if token_count is not None else len(text) // 3
    if effective_count > tool_call_budget:
        return f"[File too large for direct reading: {len(text)} chars! use csv_schema/csv_sample or json_schema_outline instead]"
    return contents

def read_files(filepaths: List[str], tool_call_budget: Optional[int] = 20000) -> Dict[str, Any]:
    return {fp: read_one_file(fp,  tool_call_budget) for fp in filepaths}