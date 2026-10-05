'''
Tokenization support for lookup in a pre-calculated .csv file or live tokenization.
If a pre-calculated file is to be used, a file name look up is performed below in get_token_count, relative to the data root directory.
For this to work, the naming schemes for input files must be mirrored in the tokenization statistics csv.

It is crucial to tokenize files in the same memory representation used in the system, to avoid major differences and consequent system failure or
deviant behaviour.
'''
import os
from agentic_dd.agentlib.file_io import read_files
from agentic_dd.agentlib.shared_methods import _build_sample_text
from agentic_dd.agentlib.SupportedModels import SupportedModels
from typing import Any, Dict, List, Optional
import pandas as pd
from pathlib import Path

_TOKENIZER_CACHE: Dict[str, Any] = {}
_TOKEN_COUNTS_CACHE: Optional[Dict[str, Dict[str, int]]] = None
DEFAULT_LOCAL_MODEL = "qwen2.5:7b"
FALLBACK_CONTEXT_WINDOW = 4096
QWEN_TOKENIZER_PROXY = "qwen3.8-27b"


def _resolve_tokenizer_model(model: str) -> Optional[str]:
    ''' Maps a runtime model name to the HF-tokenizer model name to use,
    including the qwen proxy fallback already used for precomputed counts, so
    live tokenization and precomputed counts never disagree on which tokenizer
    represents a given model '''
    if SupportedModels.is_supported(model):
        return model
    if model.split(":")[0].lower().startswith("qwen"):
        return QWEN_TOKENIZER_PROXY
    return None


def _get_tokenizer(hf_model: str):
    if hf_model in _TOKENIZER_CACHE:
        cached = _TOKENIZER_CACHE[hf_model]
        return cached if cached is not False else None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            SupportedModels.get_huggingface_tokenizer(hf_model),
            **SupportedModels.get_special_tokenizer_kwargs().get(hf_model, {}),
        )
        _TOKENIZER_CACHE[hf_model] = tok
        return tok
    except Exception as e:
        print(f"[chunking] failed to load tokenizer for {hf_model!r}: {e!r}; "
              f"falling back to heuristic count for this model going forward")
        _TOKENIZER_CACHE[hf_model] = False
        return None


def real_token_count_for_files(filepaths: List[str], model: str) -> Optional[int]:
    try:
        files = read_files(filepaths)
    except Exception:
        return None
    text = _build_sample_text(files)
    return real_token_count(text, model)


def real_token_count(text: str, model: str) -> Optional[int]:
    ''' token count via the model's actual HF tokenizer. Returns
    None if no tokenizer is resolvable/loadable, so callers can fall back
    to the heuristic estimate rather than crash the run. '''
    tok_model = _resolve_tokenizer_model(model)
    if tok_model is None:
        return None
    tokenizer = _get_tokenizer(tok_model)
    if tokenizer is None:
        return None
    try:
        return len(tokenizer.encode(text))
    except Exception:
        return None


def _load_token_counts() -> pd.DataFrame:
    global _TOKEN_COUNTS_CACHE
    if _TOKEN_COUNTS_CACHE is None:
        path = os.getenv("TOKEN_COUNTS_FILE")
        if not path or not Path(path).exists():
            _TOKEN_COUNTS_CACHE = pd.DataFrame()
        else:
            _TOKEN_COUNTS_CACHE = pd.read_csv(path)
    return _TOKEN_COUNTS_CACHE


def get_token_count(filepath: str, model: str) -> Optional[int]:
    df = _load_token_counts()
    root = os.getenv("SUB_DATA_ROOT")

    candidates = []
    if root:
        try:
            candidates.append(str(Path(filepath).resolve().relative_to(Path(root).resolve())))
        except ValueError:
            print(f"VALUE ERROR IN GET TOKEN COUNT FOR: {filepath} \n {model}")

    is_qwen = model.split(":")[0].lower().startswith("qwen")
    if is_qwen:
        model = "qwen3.8-27b"

    for key in candidates:
        try:
            f_toks = df.loc[df["key"] == key, model].iloc[0]
            return f_toks
        except IndexError:
            return None
    return None


def get_total_token_count(filepaths: List[str], model: str) -> Optional[int]:
    ''' Sums precomputed token counts across ALL files in a (possibly
    multi-file) sample. Returns None if ANY file lacks a precomputed entry, so
    callers fall back to the estimate for the whole concatenated text
    rather than under-counting by mixing a partial sum with missing
    files treated as 0 '''
    total = 0
    for fp in filepaths:
        if Path(fp).suffix.lower() not in {".csv", ".json"}:
            continue
        count = get_token_count(fp, model)
        if count is None:
            print(f"NO TOKEN COUNT FOUND FOR fp: {fp} \n model: {model}")
            return None
        total += count
    return total


def live_tokenization() -> Optional[int]:
    ''' Performs tokenization during the system's run. Not yet implemented;
    see module docstring for why this needs sandboxing before use. '''
    raise NotImplementedError