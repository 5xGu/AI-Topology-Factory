'''
Aggregation pipeline: a sequence of named "steps" (AGGREGATOR_REGISTRY), run by the "aggregator" node type. 
Each step has signature
    fn(state: AgentState, acc: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]
where `acc` is the dict accumulated by all earlier steps in this run's
pipeline and `params` is this step's own slice of the aggregator node's
params["step_params"]. 

The aggregator's own logged output is the final `acc` dict as a whole.
'''
from __future__ import annotations
from typing import Callable, Dict, Any, List, Optional

import mlflow

from agentic_dd.agentlib.state import GraphState
from agentic_dd.agentlib.llm import get_model_for, ModelParams
from agentic_dd.agentlib.nodes_factory import find_node_output, find_latest_node_output, register_node_type
from agentic_dd.agentlib.prompts import to_prompt_spec, compose_messages
from agentic_dd.agentlib.chunking import split_into_chunks, resolve_max_input_tokens
from agentic_dd.agentlib.structuredOutputs import MetadataResponse
from agentic_dd.agentlib.mlflow_utils import tag_resolved_model

AggregatorStepFn = Callable[[GraphState, Dict[str, Any], Dict[str, Any]], Dict[str, Any]]
AGGREGATOR_REGISTRY: Dict[str, AggregatorStepFn] = {}


def register_aggregator(name: str):
    def deco(fn: AggregatorStepFn) -> AggregatorStepFn:
        AGGREGATOR_REGISTRY[name] = fn
        return fn
    return deco


def _stringify(output: Any) -> str:
    if isinstance(output, dict) and "content" in output:
        return str(output["content"])
    return str(output)


@register_aggregator("concat")
def concat_aggregator(state: GraphState, acc: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    merged = "\n\n".join(
        f"[{r['node']}] {_stringify(r.get('output'))}"
        for r in state.get("results", []) if not r.get("error")
    )
    return {"final_output": merged}


@register_aggregator("extract_named_outputs")
def extract_named_outputs(state: GraphState, acc: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    ''' 
    Pulls the two fixed output-LLM results into named output-dict keys.
    Params: 
        - summary_node (default "summary_out_LLM"), 
        - metadata_node (default "metadata_out_LLM")
    
    These are overridable since a topology may alias them.
    '''
    summary_node = params.get("summary_node", "summary_out_LLM")
    metadata_node = params.get("metadata_node", "metadata_out_LLM")
    results = state.get("results", [])
    update: Dict[str, Any] = {}
    summary = find_node_output(results, summary_node)
    if summary:
        update["summary_output"] = summary.get("content")
        update["final_output"] = summary.get("content")
    metadata = find_node_output(results, metadata_node)
    if metadata:
        update["metadata_output"] = metadata.get("parsed")
    return update


def _summarize_text(text: str, params: ModelParams) -> str:
    llm = get_model_for(params)
    resp = llm.invoke(compose_messages(to_prompt_spec({"template_name": "summarizer_default"}), {"input": text}))
    return resp.content


@register_aggregator("hierarchical_reduce_recursive")
def hierarchical_reduce_recursive(state: GraphState, acc: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    ''' 
    Params: 
        - model_params (dict), 
        - max_tokens (int, optional, is resolved from model_params if absent), 
        - max_depth (int, default 5), 
        - source_node (default "chunk_summary_llm") 
    '''
    reduce_params = ModelParams(**(params.get("model_params") or {}))
    max_tokens = params.get("max_tokens") or resolve_max_input_tokens(reduce_params.provider, reduce_params.model)
    max_depth = params.get("max_depth", 5)
    source_node = params.get("source_node", "chunk_summary_llm")

    chunk_results = [r for r in state.get("results", []) if r.get("node") == source_node and not r.get("error")]
    ordered = sorted(chunk_results, key=lambda r: r.get("chunk_index", 0))
    summaries = [_stringify(r["output"]) for r in ordered]
    joined = "\n\n".join(summaries)

    depth = 0
    while len(joined) // 4 > max_tokens and len(summaries) > 1 and depth < max_depth:
        sub_chunks = split_into_chunks(joined, chunk_size_chars=max_tokens * 4)
        summaries = [_summarize_text(c, reduce_params) for c in sub_chunks]
        joined = "\n\n".join(summaries)
        depth += 1

    return {"final_output": joined, "summary_output": joined, "summarization_depth": depth}


@register_aggregator("metadata_merge")
def metadata_merge_aggregator(state: GraphState, acc: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    ''' 
    Field-wise merge of MetadataResponse dicts across chunks: 
        - union for list fields, 
        - first-non-null for scalar fields. 
    
    Params: 
        - source_nodes (list, default ["metadata_out_LLM", "chunk_metadata_llm"])
        
    Accepts either the direct or chunked metadata node by default.
    '''
    source_nodes = set(params.get("source_nodes") or ["metadata_out_LLM", "chunk_metadata_llm"])
    chunk_results = [r for r in state.get("results", []) if r.get("node") in source_nodes and not r.get("error")]
    merged: Dict[str, Any] = {}
    for r in chunk_results:
        parsed = (r.get("output") or {}).get("parsed")
        if not parsed:
            continue
        for field, value in parsed.items():
            if isinstance(value, list):
                merged[field] = list({*(merged.get(field) or []), *(value or [])})
            elif merged.get(field) is None:
                merged[field] = value
    return {"metadata_output": merged}


@register_aggregator("llm_merge")
def llm_merge_aggregator(state: GraphState, acc: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    ''' 
    Params: 
        - model_params (dict), 
        - prompt_spec (dict, default merge_default template). 
    '''
    model_params = ModelParams(**(params.get("model_params") or {}))
    spec = to_prompt_spec(params.get("prompt_spec") or {"template_name": "merge_default"})
    tag_resolved_model("llm_merge", model_params.provider, model_params.model)
    llm = get_model_for(model_params)
    context = {"input": state["input"], "prior_results": state.get("results")}
    resp = llm.invoke(compose_messages(spec, context))
    return {"final_output": resp.content}


@register_aggregator("critic")
def critic_aggregator(state: GraphState, acc: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    ''' 
    Params: 
        - model_params (dict), 
        - prompt_spec (dict, default critic_default template). 
    
    Reads `final_output` from `acc`.
    '''
    model_params = ModelParams(**(params.get("model_params") or {}))
    spec = to_prompt_spec(params.get("prompt_spec") or {"template_name": "critic_default"})
    tag_resolved_model("critic", model_params.provider, model_params.model)
    llm = get_model_for(model_params)
    context = {"input": state["input"], "prior_results": [{"node": "final_output", "output": acc.get("final_output", "")}]}
    resp = llm.invoke(compose_messages(spec, context))
    return {"critique": resp.content}


@register_aggregator("auto_direct")
def auto_direct_aggregator(state: GraphState, acc: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    ''' 
    Dispatches to hierarchical_reduce_recursive + metadata_merge if
    chunk fan-out occurred, else extract_named_outputs. 
    
    Params are passed through unchanged to whichever sub-step(s) run.
    '''
    node_names = {r.get("node") for r in state.get("results", [])}
    if "chunk_summary_llm" in node_names or "chunk_metadata_llm" in node_names:
        update = hierarchical_reduce_recursive(state, acc, params)
        update.update(metadata_merge_aggregator(state, {**acc, **update}, params))
        return update
    return extract_named_outputs(state, acc, params)


@register_aggregator("agent_structured_split")
def agent_structured_split(state: GraphState, acc: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    ''' 
    Post-hoc parse of a free-text agent/worker final_output into
    summary_output/metadata_output, so agent-route runs remain comparable
    to direct-route runs under the same evaluators. 
    
    Params: 
        - parser_model_params, 
        - metadata_model_params
    '''
    text = acc.get("final_output") or ""
    if not text:
        return {}

    parser_params = ModelParams(**(params.get("parser_model_params") or {}))
    parser_llm = get_model_for(parser_params)
    summary_resp = parser_llm.invoke(compose_messages(
        to_prompt_spec({"template_name": "summarizer_default"}), {"input": text}))

    metadata_params = ModelParams(**(params.get("metadata_model_params") or {}))
    metadata_llm = get_model_for(metadata_params).with_structured_output(MetadataResponse, include_raw=True)
    metadata_resp = metadata_llm.invoke(compose_messages(
        to_prompt_spec({"template_name": "metadata_extraction_expert"}), {"input": text}))

    update: Dict[str, Any] = {"summary_output": summary_resp.content}
    if not metadata_resp.get("parsing_error"):
        update["metadata_output"] = metadata_resp["parsed"].model_dump()
    return update


DEFAULT_PIPELINES: Dict[str, List[str]] = {
    "direct": ["auto_direct"],
    "spawn": ["concat", "llm_merge", "agent_structured_split"],
    "agent": ["concat", "agent_structured_split"],
}


def build_aggregator_compute(name: str, params: Dict[str, Any]):
    ''' 
    Params: 
        - pipeline (Optional[List[str]] ; explicit override, skips
        router lookup entirely and isrequired for topologies with no router node,
        - router_node (default "router"),
        - default_pipelines (Optional[Dict[str, List[str]]], overrides the built-in DEFAULT_PIPELINES), 
        - step_params (Dict[str, Dict[str, Any]] per-step params, keyed by step name). 
        
    All step names referenced by an explicit `pipeline` OR by any route in the effective
    default_pipelines table are validated against AGGREGATOR_REGISTRY here,
    at build time. 
    '''
    explicit_pipeline: Optional[List[str]] = params.get("pipeline")
    router_node = params.get("router_node", "router")
    default_pipelines = params.get("default_pipelines") or DEFAULT_PIPELINES
    step_params: Dict[str, Dict[str, Any]] = params.get("step_params") or {}

    all_declared_steps = set(explicit_pipeline or [])
    for steps in default_pipelines.values():
        all_declared_steps.update(steps)
    unknown = all_declared_steps - set(AGGREGATOR_REGISTRY.keys())
    if unknown:
        raise ValueError(f"aggregator node '{name}': unknown aggregation step(s) {sorted(unknown)}")

    def _compute(state: GraphState, config) -> Dict[str, Any]:
        results = state.get("results", [])
        if explicit_pipeline is not None:
            pipeline = explicit_pipeline
        else:
            router_output = find_latest_node_output(results, router_node)
            if router_output is None:
                raise ValueError(
                    f"aggregator node '{name}': no explicit 'pipeline' param set and no "
                    f"'{router_node}' output found in results; either set params.pipeline "
                    f"explicitly or ensure a router node runs before this aggregator"
                )
            route = router_output.get("route")
            pipeline = default_pipelines.get(route)
            if pipeline is None:
                mlflow.set_tag("aggregator_unknown_route", str(route))
                pipeline = ["concat"]

        acc: Dict[str, Any] = {}
        for step_name in pipeline:
            fn = AGGREGATOR_REGISTRY[step_name]
            acc.update(fn(state, acc, step_params.get(step_name, {})))
        return acc

    return _compute


register_node_type("aggregator")(build_aggregator_compute)