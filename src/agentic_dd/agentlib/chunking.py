'''
Deterministic chunk-splitting node type ("chunk_split") and its dispatching condition ("chunk_fanout_decision"). 
The LLM nodes that consume chunk_split's output are plain topology-declared instances of the generic "llm" node type.
'''
from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple
from langgraph.types import Send

from agentic_dd.agentlib.state import GraphState
from agentic_dd.agentlib.nodes_factory import register_node_type, find_latest_node_output
from agentic_dd.topologies.conditions import register_condition
from agentic_dd.agentlib.shared_methods import _build_sample_text
from agentic_dd.agentlib.tokenization import get_total_token_count, DEFAULT_LOCAL_MODEL, FALLBACK_CONTEXT_WINDOW
from agentic_dd.agentlib.SupportedModels import CONTEXT_WINDOWS, LOCAL_CONTEXT_WINDOWS


class ChunkPlan:
    def __init__(self, needs_chunking: bool, num_chunks: int, chunk_size_chars: Optional[int] = None):
        self.needs_chunking = needs_chunking
        self.num_chunks = num_chunks
        self.chunk_size_chars = chunk_size_chars


def compute_chunk_plan(text: str, filepaths: List[str], model: str, max_tokens: int) -> ChunkPlan:
    precomputed = get_total_token_count(filepaths, model)
    char_based_estimate = len(text) // 3
    if precomputed is None:
        total_tokens = char_based_estimate
    else:
        total_tokens = max(precomputed, char_based_estimate)

    if total_tokens <= max_tokens:
        return ChunkPlan(needs_chunking=False, num_chunks=1)
    num_chunks = math.ceil(total_tokens / max_tokens)
    chunk_size_chars = math.ceil(len(text) / num_chunks * 1.15)
    return ChunkPlan(needs_chunking=True, num_chunks=num_chunks, chunk_size_chars=chunk_size_chars)


def split_into_chunks(text: str, chunk_size_chars: int, overlap: int = 200) -> List[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size_chars, chunk_overlap=overlap)
    return splitter.split_text(text)


def resolve_max_input_tokens(provider: str, model: str, reserved_for_completion: int = 1500, safety_margin: int = 500) -> int:
    windows = LOCAL_CONTEXT_WINDOWS if provider == "local" else CONTEXT_WINDOWS
    window = windows.get(model)
    if window is None:
        print(f"[chunking] no known context window for {provider}:{model!r}, "
              f"falling back to conservative default {FALLBACK_CONTEXT_WINDOW}")
        window = FALLBACK_CONTEXT_WINDOW
    return max(500, window - reserved_for_completion - safety_margin)


def resolve_chunk_settings(params: Dict[str, Any]) -> Tuple[str, str, int]:
    model = params.get("model", DEFAULT_LOCAL_MODEL)
    provider = params.get("provider", "local")
    max_tokens = params.get("max_tokens") or resolve_max_input_tokens(provider, model)
    return model, provider, max_tokens


def build_chunk_split_compute(name: str, params: Dict[str, Any]):
    ''' 
    Params: 
        - source_node (str, default "read_file"): the node whose {path: content} output supplies the sample text; 
        - model/provider/max_tokens: chunk-sizing settings.
        
    Chumnking always runs, because it is a technical necessity, next to a strategic decision.
    It always returns a List[str] of length >= 1, and the fan-out decision is made downstream by
    "chunk_fanout_decision" based on this output's length.
    The chunk plan is computed exactly once per sample. 
    '''
    source_node = params.get("source_node", "read_file")
    model, provider, max_tokens = resolve_chunk_settings(params)

    def _compute(state: GraphState, config) -> List[str]:
        files = find_latest_node_output(state.get("results", []), source_node) or {}
        text = _build_sample_text(files)
        plan = compute_chunk_plan(text, state["input"], model, max_tokens)
        if not plan.needs_chunking:
            return [text]
        return split_into_chunks(text, plan.chunk_size_chars)

    return _compute


register_node_type("chunk_split")(build_chunk_split_compute)


@register_condition("chunk_fanout_decision")
def chunk_fanout_decision(state: GraphState, config):
    ''' 
    Reads chunk_split's already-computed output:
        - len > 1: fan out one Send pair per chunk to chunk_summary_llm/chunk_metadata_llm, with a fresh 
        per-branch `results` and `chunk_index`
        - len == 1: route via plain edge target names to the non-chunked output
        nodes, which read chunk_split's single-chunk text via `text_source`. 
    '''
    chunks = find_latest_node_output(state.get("results", []), "chunk_split") or []
    if len(chunks) > 1:
        sends: List[Send] = []
        for i, chunk in enumerate(chunks):
            payload = {**state, "results": [], "input": [chunk], "chunk_index": i}
            sends.append(Send("chunk_summary_llm", payload))
            sends.append(Send("chunk_metadata_llm", payload))
        return sends
    return ["summary_out_LLM", "metadata_out_LLM"]