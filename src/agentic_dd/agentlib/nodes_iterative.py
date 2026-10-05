'''
Iterative (incremental synthesis + reflection) loop node/condition types.
Loop progress (files read/remaining, iteration count) is derived entirely
from the shared `results` log via the node-name params below, never stored
as dedicated AgentState fields 
'''
from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
from langchain_core.runnables import RunnableConfig

from agentic_dd.agentlib.state import GraphState
from agentic_dd.agentlib.file_io import read_one_file
from agentic_dd.agentlib.mlflow_utils import tag_resolved_model
from agentic_dd.agentlib.llm import ModelParams, get_model_for
from agentic_dd.agentlib.prompts import to_prompt_spec, compose_messages
from agentic_dd.agentlib.structuredOutputs import IncrementalSynthesis, ReflectionVerdict, MetadataResponse
from agentic_dd.agentlib.nodes_factory import (
    register_node_type, find_node_output, find_latest_node_output, find_all_node_outputs,
)
from agentic_dd.topologies.conditions import register_condition_type
from agentic_dd.agentlib.chunking import resolve_max_input_tokens
from agentic_dd.agentlib.tokenization import real_token_count

ROLE_PRIORITY = {"description": 0, "metadata": 1, "data": 2}


def _infer_file_role(fp: str) -> str:
    ''' 
    Cheap, deterministic role inference from directory naming.
    '''
    parent = Path(fp).parent.name.lower()
    if "description" in parent:
        return "description"
    if "metadata" in parent:
        return "metadata"
    return "data"


def triage_progress(
    results: List[Dict[str, Any]], triage_node: str, read_node: str,
) -> Tuple[List[str], List[str]]:
    ''' 
    Single source of truth for "which files have been read, which
    remain": file_triage's full ordering, sliced by how many read_node
    entries exist so far. 
    
    Never stored as dedicated state, always
    recomputed from the results log and is shared by every function below that
    needs it, so they cannot silently diverge on the answer. 
    '''
    ordered = find_node_output(results, triage_node) or []
    read_count = len(find_all_node_outputs(results, read_node))
    return ordered[:read_count], ordered[read_count:]


def _safe_truncate(text: str, provider: str, model: str, budget_fraction: float = 0.75, path: Optional[str] = None) -> str:
    max_tokens = resolve_max_input_tokens(provider, model)
    budget = int(max_tokens * budget_fraction)
    count = real_token_count(text, model)
    if count is not None and count <= budget:
        return text
    keep_chars = int(len(text) * (budget / count)) if count else budget * 3
    if path:
        try:
            mlflow.set_tag(f"file_truncated::{path}", "true")
        except Exception:
            pass
    return text[:keep_chars] + "\n\n[... TRUNCATED: source exceeded per-iteration token budget ...]"


########## file_triage 

def build_file_triage_compute(name: str, params: Dict[str, Any]):
    ''' 
    Params: 
        - ordering ("heuristic" | "reverse" | "random", defaults to "heuristic"), 
        - ordering_seed (int, default 0; "random" only).
        
    An unknown ordering value fails at topology-build time. 
    '''
    ordering = params.get("ordering", "heuristic")
    ordering_seed = params.get("ordering_seed", 0)
    if ordering not in ("heuristic", "reverse", "random"):
        raise ValueError(f"file_triage node '{name}': unknown ordering strategy {ordering!r}")

    def _compute(state: GraphState, config: RunnableConfig) -> List[str]:
        mlflow.set_tag("file_triage_ordering", ordering)
        if ordering == "heuristic":
            return sorted(state["input"], key=lambda fp: ROLE_PRIORITY.get(_infer_file_role(fp), 2))
        if ordering == "reverse":
            return sorted(state["input"], key=lambda fp: ROLE_PRIORITY.get(_infer_file_role(fp), 2), reverse=True)
        ordered = list(state["input"])
        random.Random(ordering_seed).shuffle(ordered)
        return ordered

    return _compute


register_node_type("file_triage")(build_file_triage_compute)


########## read_next_source

def build_read_next_source_compute(name: str, params: Dict[str, Any]):
    ''' 
    Params: 
        - triage_node (str, default "file_triage"). 
    
    Raises when no files are left to read.
    '''
    triage_node = params.get("triage_node", "file_triage")

    def _compute(state: GraphState, config: RunnableConfig) -> Dict[str, Any]:
        results = state.get("results", [])
        _files_read, files_remaining = triage_progress(results, triage_node, name)
        if not files_remaining:
            raise ValueError(
                f"read_next_source '{name}': no files remaining; this indicates a routing bug "
                f"upstream (the dispatching condition should have routed to finalize_incremental)"
            )
        fp = files_remaining[0]
        content = read_one_file(fp)
        return {"path": fp, "content": content}

    return _compute


register_node_type("read_next_source")(build_read_next_source_compute)


########## incremental_synthesis

def build_incremental_synthesis_compute(name: str, params: Dict[str, Any]):
    ''' 
    Params: 
        - prompt_spec (dict), 
        - model_params (dict), 
        - read_node (default "read_next_source"), 
        - truncate_budget_fraction (default 0.75).

    Overridable per-invocation via config["configurable"], keyed by this node's own name.
    Looks up its OWN prior output via `name`, meaning that it is not supported for  a topology to aliase this node's writes
    elsewhere. 
    '''
    default_model_params = ModelParams(**(params.get("model_params") or {}))
    default_spec = to_prompt_spec(params.get("prompt_spec") or {"template_name": "incremental_synthesis_default"})
    read_node = params.get("read_node", "read_next_source")
    truncate_budget_fraction = params.get("truncate_budget_fraction", 0.75)

    def _compute(state: GraphState, config: RunnableConfig) -> Any:
        configurable = (config or {}).get("configurable", {}) or {}
        override_model_params = (configurable.get("model_params") or {}).get(name)
        model_params = ModelParams(**override_model_params) if override_model_params else default_model_params
        override_prompt = (configurable.get("prompt_overrides") or {}).get(name)
        spec = to_prompt_spec(override_prompt) if override_prompt else default_spec

        results = state.get("results", [])
        prior = find_latest_node_output(results, name) or {}
        prior_parsed = prior.get("parsed") or {}
        prior_summary = prior_parsed.get("updated_summary")
        prior_metadata = prior_parsed.get("updated_metadata")

        new_source = find_latest_node_output(results, read_node) or {}
        new_content = new_source.get("content")
        new_text = new_content if isinstance(new_content, str) else json.dumps(new_content)
        path = new_source.get("path")
        new_text = _safe_truncate(new_text, model_params.provider, model_params.model,
                                   budget_fraction=truncate_budget_fraction, path=path)

        context_text = (
            f"Current working summary:\n{prior_summary or '(none yet -- this is the first source)'}\n\n"
            f"Current working metadata:\n{prior_metadata or '(none yet)'}\n\n"
            f"New source ({path or 'unknown'}):\n{new_text}"
        )

        tag_resolved_model(name, model_params.provider, model_params.model)
        llm = get_model_for(model_params).with_structured_output(IncrementalSynthesis, include_raw=True)
        resp = llm.invoke(compose_messages(spec, {"input": context_text}))
        if resp.get("parsing_error"):
            mlflow.log_metric(f"{name}_parse_errors", 1)
            fallback = IncrementalSynthesis(updated_summary=prior_summary or "", updated_metadata=MetadataResponse())
            return {"parsed": fallback.model_dump(), "parsing_error": str(resp["parsing_error"])}
        return {"parsed": resp["parsed"].model_dump(), "parsing_error": None}

    return _compute


register_node_type("incremental_synthesis")(build_incremental_synthesis_compute)


########## reflection_gate

def build_reflection_gate_compute(name: str, params: Dict[str, Any]):
    ''' 
    Params: 
        - prompt_spec, 
        - model_params, 
        - synthesis_node (defaults to "incremental_synthesis_llm"),
        - triage_node (default "file_triage"),
        - read_node (default "read_next_source"). 
    '''
    default_model_params = ModelParams(**(params.get("model_params") or {}))
    default_spec = to_prompt_spec(params.get("prompt_spec") or {"template_name": "reflection_gate_default"})
    synthesis_node = params.get("synthesis_node", "incremental_synthesis_llm")
    triage_node = params.get("triage_node", "file_triage")
    read_node = params.get("read_node", "read_next_source")

    def _compute(state: GraphState, config: RunnableConfig) -> Any:
        configurable = (config or {}).get("configurable", {}) or {}
        override_model_params = (configurable.get("model_params") or {}).get(name)
        model_params = ModelParams(**override_model_params) if override_model_params else default_model_params
        override_prompt = (configurable.get("prompt_overrides") or {}).get(name)
        spec = to_prompt_spec(override_prompt) if override_prompt else default_spec

        results = state.get("results", [])
        latest = find_latest_node_output(results, synthesis_node) or {}
        parsed = latest.get("parsed") or {}
        working_summary = parsed.get("updated_summary", "")
        working_metadata = parsed.get("updated_metadata", {})
        _files_read, files_remaining = triage_progress(results, triage_node, read_node)

        context_text = (
            f"Working summary so far:\n{working_summary}\n\n"
            f"Working metadata so far:\n{working_metadata}\n\n"
            f"Files not yet examined: {files_remaining}"
        )

        tag_resolved_model(name, model_params.provider, model_params.model)
        llm = get_model_for(model_params).with_structured_output(ReflectionVerdict, include_raw=True)
        resp = llm.invoke(compose_messages(spec, {"input": context_text}))
        if resp.get("parsing_error"):
            verdict = ReflectionVerdict(sufficient=True, conflicts_detected=False,
                                         reasoning=f"[parse failed, defaulting to accept: {resp['parsing_error']}]")
        else:
            verdict = resp["parsed"]
        return verdict.model_dump()

    return _compute


register_node_type("reflection_gate")(build_reflection_gate_compute)



########## finalize_incremental

def build_finalize_incremental_compute(name: str, params: Dict[str, Any]):
    ''' 
    Params: 
        - synthesis_node, 
        - triage_node, 
        - read_node, 
        - reflection_node
    
    Bridges the loop's final working draft into the shape the "extract_named_outputs" aggregator step
    reads. 
    
    NOTE: this node instance MUST be declared in the topology with 
        "writes": ["summary_out_LLM", "metadata_out_LLM"] 
    because this compute always returns a dict keyed by exactly those two names.
    '''
    synthesis_node = params.get("synthesis_node", "incremental_synthesis_llm")
    triage_node = params.get("triage_node", "file_triage")
    read_node = params.get("read_node", "read_next_source")
    reflection_node = params.get("reflection_node", "reflection_gate")

    def _compute(state: GraphState, config: RunnableConfig) -> Dict[str, Any]:
        results = state.get("results", [])
        latest = find_latest_node_output(results, synthesis_node) or {}
        parsed = latest.get("parsed") or {}

        files_read, _files_remaining = triage_progress(results, triage_node, read_node)
        iteration_count = len(find_all_node_outputs(results, reflection_node))
        mlflow.set_tag("iterative_iterations_used", iteration_count)
        mlflow.set_tag("iterative_files_read", len(files_read))

        return {
            "summary_out_LLM": {"content": parsed.get("updated_summary", "")},
            "metadata_out_LLM": {
                "parsed": parsed.get("updated_metadata", {}),
                "parsing_error": latest.get("parsing_error"),
            },
        }

    return _compute


register_node_type("finalize_incremental")(build_finalize_incremental_compute)


########## iterative_revision_gate (condition)

def build_iterative_revision_gate(name: str, params: Dict[str, Any]):
    ''' 
    Params: 
        - max_iterations (int, default 5), 
        - reflection_node, 
        - triage_node, 
        - read_node. 
        
    The iteration cap is enforced here, independent of the graph recursion limit.
    '''
    max_iterations = params.get("max_iterations", 5)
    reflection_node = params.get("reflection_node", "reflection_gate")
    triage_node = params.get("triage_node", "file_triage")
    read_node = params.get("read_node", "read_next_source")

    def _condition(state: GraphState, config: RunnableConfig) -> str:
        results = state.get("results", [])
        latest = find_latest_node_output(results, reflection_node) or {}
        sufficient = latest.get("sufficient", True)
        conflicts_detected = latest.get("conflicts_detected", False)

        _files_read, files_remaining = triage_progress(results, triage_node, read_node)
        iteration_count = len(find_all_node_outputs(results, reflection_node))

        if not files_remaining or iteration_count >= max_iterations:
            return "finalize_incremental"
        if sufficient and not conflicts_detected:
            return "finalize_incremental"
        return "read_next_source"

    return _condition


register_condition_type("iterative_revision_gate")(build_iterative_revision_gate)