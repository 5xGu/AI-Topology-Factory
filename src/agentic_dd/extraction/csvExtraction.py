'''
Read-only, descriptive profiling methods for csv files through pandas Dataframes.

Langgraph provides a csv agent, but it calls a python agent to generate code for pandas.
We implement here a more stable and understandable implementation that is abstracted as tools.

Moreover, virtually all of our .csv data is categorially encoded rather than numerical. A priori analysis
of the .csvs themselves is thus restricted in scope, and we offer count-based methods to guide decisions about
more detailed inspection of files.

We allow an agent to write and perform more elaborate queries in a safe and monitored execution environment. This
includes statistics over ordinal data, which we here defer. #TODO
'''
from __future__ import annotations
from typing import Any, Literal
import pandas as pd
import numpy as np


MAX_SAMPLE_ROWS = 50
DEFAULT_SAMPLE_ROWS = 10


def _check_cols(df: pd.DataFrame, *cols: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Unknown column(col): {missing}")

def get_schema(df: pd.DataFrame) -> dict[str, Any]:
    """ Row count and per-column dtype/null/cardinality report for a high-level overview of the data. """
    n = len(df)
    return {
        "row_count": n,
        "columns": [
            {
                "name": c,
                "dtype": str(df[c].dtype),
                "missing_values": int(df[c].isna().sum()),
                "distinct_values": int(df[c].nunique(dropna=True)),
                "likely_identifier": bool(n and df[c].nunique(dropna=True) / n > 0.98), # prevent confusion with high entropy + 
            }
            for c in df.columns
        ],
    }


def get_sample(df: pd.DataFrame, n: int = DEFAULT_SAMPLE_ROWS, method: Literal["head", "random"] = "head") -> list[dict]:
    ''' Limited sample of whole rows '''
    n = min(n, MAX_SAMPLE_ROWS)
    sample = df.head(n) if method == "head" else df.sample(min(n, len(df)))
    return sample.to_dict(orient="records")

def get_full_column(df: pd.DataFrame, column: str):
    _check_cols(df, column)
    return df[column]

def inspect_column(df: pd.DataFrame, column: str, n: int = DEFAULT_SAMPLE_ROWS, method: Literal["head", "random"] = "head"):
    ''' Inspect n (at most 50) values of a colum beginning at index 0, or randomly sampled '''
    _check_cols(df, column)
    n = min(n, MAX_SAMPLE_ROWS)
    col = df[column] # do not .dropna() for an acurate represention of the column data
    values = col.head(n) if method == "head" else col.sample(min(n, len(col)))
    return values.tolist()

def get_row(df, index: any):
    ''' returns a single row by positional or label index, or none if index was not found '''
    if isinstance(index, int):
        try: 
            return df.iloc[index]
        except IndexError: return None
    else:
        try:
            return df.loc[index]
        except KeyError: return None

def get_heuristic_column_stats(df: pd.DataFrame, column: str) -> dict[str, Any]:
    """ 
    Returns small statistics report for a column to guide decisions about further processing.
    If a column contains more than 10 unique values and the column is numeric, it is assumed the variable
    is continous. The report then contains numerical signals, like min, max, and percentiles.

    Otherwise the variable is assumed descrete and a categorical summary is returned, including the number
    of distinct values, the mode, column entropy and top values.
    """

    _check_cols(df, column)
    s = df[column]
    base = {"column": column, "dtype": str(s.dtype), "missing_values": int(s.isna().sum())}

    CONTINUITY_THRESHOLD = 10 # as a heuristic we assume that more than 10 unique values indicate a continous variable
    if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > CONTINUITY_THRESHOLD:
        desc = s.describe()
        base["numeric"] = {
            "min": desc.get("min"), "max": desc.get("max"),
            "mean": desc.get("mean"), "std": desc.get("std"),
            "p25": desc.get("25%"), "p50": desc.get("50%"), "p75": desc.get("75%"),
        }
    else:
        counts = s.value_counts(dropna=True)
        total = counts.sum()
        top = counts.head(10)
        base["categorical"] = {     
            "distinct_count": int(counts.shape[0]),
            "mode": counts.index[0] if not counts.empty else None,
            "entropy_bits": _entropy(counts) if not counts.empty else None,
            "normalized_entropy": _normalized_entropy(counts) if not counts.empty else None,
            "top_values": [
                {"value": k, "count": int(v), "proportion": float(v / total)}
                for k, v in top.items()
            ],
        }
    return base

### Frequency based methods to inform further processing '''

def _entropy(counts: pd.Series) -> float:
    ''' 
    Shannon entropy (in bits), to inform about the rough distribution of a column 
    Return values are located in an interval delimitated by:
        0 ~ column is dominated by a single value
        log_2(distinct_values) ~ column values are uniformly distributed
    '''
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())

def _normalized_entropy(counts: pd.Series) -> float | None:
    '''
    Normalize the Shannon entropy per column to compare across columns. 
    Returns None for columns with a single value, and values between 0 and 1 otherwise.
        0 ~ one dominant value
        1 ~ values follow uniform distribution
    '''
    distinct = counts.shape[0]
    if distinct <= 1:
        return None
    return _entropy(counts) / np.log2(distinct)


#test
'''
def main():
    import pandas as pd
    import json
    fp = ''
    df= pd.read_csv(fp)

    profile = {
    "schema": get_schema(df),
    "sample": get_sample(df, n=3),
    "column stats": get_heuristic_column_stats(df, "participated_1_wave"),
    'A': get_full_column(df, 'participated_1_wave'),
    'B': inspect_column(df, 'participated_1_wave'),
    'C': get_row(df, '4125')
    }


    with open("csv_test.profile.json", "w") as f:
        json.dump(profile, f, indent=2, default=str)
if __name__ == '__main__':
    main()
'''