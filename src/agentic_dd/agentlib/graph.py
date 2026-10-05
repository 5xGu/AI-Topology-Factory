'''
Builds and compiles a LangGraph StateGraph[AgentState] entirely from a
topology JSON file. This module owns two responsibilities:

  1. Deciding which node/condition-defining modules get imported (and
     therefore which `type` names exist to be referenced by a topology
     file).
  2. Mechanically translating an already-validated TopologySpec into
     LangGraph wiring.

All node/condition BEHAVIOR lives in the imported modules; all topology
STRUCTURE validation lives in agentic_dd.topologies.topologies. Neither
concern belongs here.

Invocation contract for callers of the compiled graph: AgentState carries
only strict per-run data (`input`, `results`, `chunk_index`, `task`), e.g.:

    compiled.invoke(
        {"input": [...]},
        {"configurable": {
            "mlflow_run_id": "...",
            "model_params": {"summary_out_LLM": {"model": "...", ...}},
            "prompt_overrides": {"summary_out_LLM": {...}},
            "max_depth": {"agent": 3},
        }},
    )
'''
from __future__ import annotations
from typing import Optional, Dict, Any, List
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import BaseCheckpointSaver

import agentic_dd.agentlib  # noqa: F401, triggers node/condition/tool/prompt registration, see agentlib/__init__.py
from agentic_dd.agentlib.state import GraphState
from agentic_dd.agentlib.nodes_factory import instantiate_node
from agentic_dd.topologies.conditions import instantiate_condition
from agentic_dd.topologies import (
    load_topology,
    validate_loaded_topology,
    validate_registries,
    TopologySpec, 
    EdgeSpec, 
    END as TOPOLOGY_END,
)
from agentic_dd.logging import topology_logger

# NOTE: placeholder path
DEFAULT_TOPOLOGY_PATH = "example_topology.json"


def _resolve_target(name: str) -> str:
    ''' 
    Maps the topology file's "END" sentinel string onto LangGraph's own
    END object
    '''
    return END if name == TOPOLOGY_END else name


def _build_node_map(spec: TopologySpec) -> Dict[str, Any]:
    return {
        node.name: instantiate_node(node.name, node.type, node.params, reads=node.reads, writes=node.writes)
        for node in spec.nodes
    }


def _build_condition_map(spec: TopologySpec) -> Dict[str, Any]:
    return {
        cond.name: instantiate_condition(cond.name, cond.type, cond.params, reads=cond.reads)
        for cond in spec.conditions
    }


def _edges_by_source(spec: TopologySpec) -> Dict[str, List[EdgeSpec]]:
    by_source: Dict[str, List[EdgeSpec]] = {}
    for e in spec.edges:
        by_source.setdefault(e.source, []).append(e)
    return by_source


def _wire_edges(builder: StateGraph, spec: TopologySpec, condition_fns: Dict[str, Any]) -> None:
    for source, edges in _edges_by_source(spec).items():
        cond_edges = [e for e in edges if e.condition is not None]
        plain_edges = [e for e in edges if e.target is not None]
        # topologies.py's structural validation already guarantees these are
        # mutually exclusive per source, and at most one conditional edge
        # exists per source; re-asserted here for simple safety
        assert not (cond_edges and plain_edges)
        assert len(cond_edges) <= 1

        if cond_edges:
            edge = cond_edges[0]
            condition_fn = condition_fns[edge.condition]
            targets = [_resolve_target(t) for t in (edge.condition_targets or [])]
            builder.add_conditional_edges(source, condition_fn, targets)
        else:
            for edge in plain_edges:
                builder.add_edge(source, _resolve_target(edge.target))


def build_graph(topology_file_path: Optional[str] = None, checkpointer: Optional[BaseCheckpointSaver] = None):
    path = topology_file_path or DEFAULT_TOPOLOGY_PATH
    raw = load_topology(path)
    spec = validate_loaded_topology(raw)
    validate_registries(spec)

    builder = StateGraph(GraphState)

    node_fns = _build_node_map(spec)
    for name, fn in node_fns.items():
        builder.add_node(name, fn)

    condition_fns = _build_condition_map(spec)

    builder.add_edge(START, spec.entry)
    _wire_edges(builder, spec, condition_fns)

    topology_logger.info(
        f"Compiled graph from topology '{path}': {len(spec.nodes)} node(s), {len(spec.edges)} edge(s)"
    )
    return builder.compile(checkpointer=checkpointer)