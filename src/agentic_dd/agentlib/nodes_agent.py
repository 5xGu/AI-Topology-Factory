'''
Tool-calling compute for the generic "llm" node type (see
agentic_dd.agentlib.nodes_factory.build_llm_compute): used whenever a
node's topology params include a non-empty "tools" list. This is the
entire mechanism by which "agent"/"worker"-style nodes are expressed.
'''
from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path
import mlflow
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from agentic_dd.agentlib.state import GraphState
from agentic_dd.agentlib.agent_subgraph import build_agent_subgraph
from agentic_dd.agentlib.prompts import to_prompt_spec, PROMPT_REGISTRY
from agentic_dd.agentlib.tools import resolve_tools
from agentic_dd.agentlib.delegation import build_delegate_tool
from agentic_dd.agentlib.llm import ModelParams


def _format_file_listing(paths: List[str]) -> str:
    lines = []
    for fp in paths:
        try:
            size = Path(fp).stat().st_size
        except OSError:
            size = None
        lines.append(f"{fp}" + (f" ({size} bytes)" if size is not None else ""))
    return "\n".join(lines)


def _build_human_message(state: GraphState) -> HumanMessage:
    ''' 
    A spawned worker gets a single delegated instruction (state["task"]);
    anything else gets a listing of the run's input files (state["input"])
    '''
    task = state.get("task")
    if task is not None:
        return HumanMessage(content=task)
    return HumanMessage(content=_format_file_listing(state.get("input") or []))


def build_tool_calling_llm_compute(name: str, params: Dict[str, Any]):
    ''' 
    Params: 
        - tools (List[str], may include "delegate"), 
        - prompt_spec (dict),
        - model_params (dict), 
        - max_depth (int, default 2; only meaningful together with the "delegate" tool), 
        - recursion_limit (int, default 50),
        - selective_file_reading (bool). 
    
    model_params and max_depth are overridable per-invocation via config["configurable"], keyed by this node's name.
    '''
    default_model_params = ModelParams(**(params.get("model_params") or {}))
    default_spec = to_prompt_spec(params.get("prompt_spec") or {})
    tool_names = params.get("tools") or []
    default_max_depth = params.get("max_depth", 2)
    recursion_limit = params.get("recursion_limit", 50)
    selective_file_reading = params.get("selective_file_reading", False)

    def _apply_addendum(spec):
        if not selective_file_reading:
            return spec
        addendum = PROMPT_REGISTRY["selective_file_reading_addendum"]
        return spec.model_copy(update={
            "instructions": f"{spec.instructions}\n\n{addendum}" if spec.instructions else addendum
        })

    def _compute(state: GraphState, config: RunnableConfig) -> Any:
        configurable = (config or {}).get("configurable", {}) or {}
        override_model_params = (configurable.get("model_params") or {}).get(name)
        model_params = ModelParams(**override_model_params) if override_model_params else default_model_params
        override_prompt = (configurable.get("prompt_overrides") or {}).get(name)
        spec = to_prompt_spec(override_prompt) if override_prompt else default_spec
        spec = _apply_addendum(spec)
        max_depth = (configurable.get("max_depth") or {}).get(name, default_max_depth)

        tools = resolve_tools([t for t in tool_names if t != "delegate"], default=[])
        if "delegate" in tool_names:
            delegate_tool = build_delegate_tool(depth=0, max_depth=max_depth, model_params=model_params)
            if delegate_tool:
                tools = tools + [delegate_tool]

        with mlflow.start_span(name=name, span_type="AGENT") as span:
            span.set_inputs({"tools": [t.name for t in tools]})
            compiled = build_agent_subgraph(tools=tools, prompt_spec=spec, name=name, model_params=model_params)
            result = compiled.invoke(
                {"messages": [_build_human_message(state)], "input": state.get("input")},
                {"recursion_limit": recursion_limit},
            )
            output = result["messages"][-1].content
            span.set_outputs({"output": output[:2000]})

        return output

    return _compute