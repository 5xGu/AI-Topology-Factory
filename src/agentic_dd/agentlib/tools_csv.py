from langchain_core.tools import tool
import pandas as pd
from agentic_dd.extraction.csvExtraction import get_schema, get_sample, inspect_column, get_heuristic_column_stats, get_full_column , get_row
from typing import Any

_df_cache: dict[str, pd.DataFrame] = {}

def _get_df(path: str) -> pd.DataFrame:
    if path not in _df_cache:
        _df_cache[path] = pd.read_csv(path)
    return _df_cache[path]

@tool
def csv_schema(path: str) -> dict:
    """Get row count and per-column dtype/null/cardinality summary for a CSV file."""
    return get_schema(_get_df(path))

@tool
def csv_sample(path: str, n: int = 10, method: str = "head") -> list[dict]:
    """Sample n rows from a CSV file ('head' or 'random')."""
    return get_sample(_get_df(path), n=n, method=method)

@tool
def csv_column_stats(path: str, column: str) -> dict:
    """Get heuristic statistics (numeric or categorical) for one column of a CSV file."""
    return get_heuristic_column_stats(_get_df(path), column)

@tool
def csv_inspect_column(path: str, column: str, n: int = 10, method: str = "head") -> list:
    """Inspect n values of a specific column."""
    return inspect_column(_get_df(path), column, n=n, method=method)

@tool
def csv_full_column(path: str, column: str):
    ''' Retrieve a full column from a csv file '''
    return get_full_column(_get_df(path), column)

@tool
def csv_row(path: str, index: Any):
    ''' Retrieve a row from a csv file, based on a specified index '''
    return get_row(_get_df(path), index)

# -------- Register tools -------- #
from agentic_dd.agentlib.tools import register_tool

register_tool('csv_schema', csv_schema)
register_tool('csv_sample', csv_sample)
register_tool('csv_column_stats', csv_column_stats)
register_tool('csv_inspect_column', csv_inspect_column)
register_tool('csv_full_column', csv_full_column)
register_tool('csv_row', csv_row)