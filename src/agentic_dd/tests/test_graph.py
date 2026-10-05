'''
Tests compile functionality only, i.e., never calls a compute/condition function.
'''
import json
import pytest

from agentic_dd.agentlib.graph import build_graph
from agentic_dd.agentlib.nodes_factory import list_available_node_types
from agentic_dd.topologies import list_available_condition_types


def _write_topology(tmp_path, data):
    path = tmp_path / "topology.json"
    path.write_text(json.dumps(data))
    return str(path)


@pytest.mark.parametrize("node_type", list_available_node_types())
def test_minimal_single_node_topology_compiles(tmp_path, node_type):
    data = {
        "entry": node_type,
        "nodes": [{"name": node_type, "type": node_type}],
        "edges": [{"source": node_type, "target": "END"}],
    }
    assert build_graph(_write_topology(tmp_path, data)) is not None


@pytest.mark.parametrize("condition_type", list_available_condition_types())
def test_minimal_single_condition_topology_compiles(tmp_path, condition_type):
    data = {
        "entry": "entry_node",
        "nodes": [{"name": "entry_node", "type": "read_file"}],
        "conditions": [{"name": condition_type, "type": condition_type}],
        "edges": [{"source": "entry_node", "condition": condition_type, "condition_targets": ["END"]}],
    }
    assert build_graph(_write_topology(tmp_path, data)) is not None


# NOTE: embedded rather than loaded from a shared fixture file, to prevent accidental drift, but requires explicit update if significant
# differences emerge.

DIRECT_AGENT_SPAWN_TOPOLOGY = {
    "entry": "router",
    "nodes": [
        {"name": "router", "type": "router", "params": {
            "route": "free", "allowed_routes": ["direct", "agent", "spawn"], "fallback_route": "direct",
            "model_params": {"provider": "local", "model": "qwen2.5:7b", "temperature": 0.0},
        }},
        {"name": "read_file", "type": "read_file"},
        {"name": "chunk_split", "type": "chunk_split",
         "params": {"source_node": "read_file", "provider": "local", "model": "qwen2.5:7b"},
         "reads": [{"node": "read_file", "mode": "last"}]},
        {"name": "chunk_summary_llm", "type": "llm",
         "params": {"prompt_spec": {"template_name": "summarizer_default"}}, "reads": []},
        {"name": "chunk_metadata_llm", "type": "llm",
         "params": {"prompt_spec": {"template_name": "metadata_extraction_expert"},
                    "structured_output": "MetadataResponse"}, "reads": []},
        {"name": "summary_out_LLM", "type": "llm",
         "params": {"prompt_spec": {"template_name": "summarizer_default"}, "text_source": "chunk_split"},
         "reads": [{"node": "chunk_split", "mode": "last"}]},
        {"name": "metadata_out_LLM", "type": "llm",
         "params": {"prompt_spec": {"template_name": "metadata_extraction_expert"},
                    "structured_output": "MetadataResponse", "text_source": "chunk_split"},
         "reads": [{"node": "chunk_split", "mode": "last"}]},
        {"name": "aggregator", "type": "aggregator"},
        {"name": "agent", "type": "llm",
         "params": {"tools": ["read_file", "delegate"],
                    "prompt_spec": {"template_name": "agent_default"},
                    "model_params": {"provider": "api", "model": "qwen3.8-27b", "temperature": 0.0},
                    "max_depth": 2, "recursion_limit": 50, "selective_file_reading": True}},
        {"name": "worker", "type": "llm",
         "params": {"tools": ["read_file"], "prompt_spec": {"template_name": "worker_default"},
                    "model_params": {"provider": "local", "model": "qwen2.5:7b", "temperature": 0.0}}},
    ],
    "conditions": ["route_decision", "chunk_fanout_decision"],
    "edges": [
        {"source": "router", "condition": "route_decision", "condition_targets": ["read_file", "agent", "worker"]},
        {"source": "read_file", "target": "chunk_split"},
        {"source": "chunk_split", "condition": "chunk_fanout_decision",
         "condition_targets": ["chunk_summary_llm", "chunk_metadata_llm", "summary_out_LLM", "metadata_out_LLM"]},
        {"source": "chunk_summary_llm", "target": "aggregator"},
        {"source": "chunk_metadata_llm", "target": "aggregator"},
        {"source": "summary_out_LLM", "target": "aggregator"},
        {"source": "metadata_out_LLM", "target": "aggregator"},
        {"source": "agent", "target": "aggregator"},
        {"source": "worker", "target": "aggregator"},
        {"source": "aggregator", "target": "END"},
    ],
}

ITERATIVE_SYNTHESIS_TOPOLOGY = {
    "entry": "file_triage",
    "nodes": [
        {"name": "file_triage", "type": "file_triage", "params": {"ordering": "heuristic"}, "reads": []},
        {"name": "read_next_source", "type": "read_next_source",
         "params": {"triage_node": "file_triage"}, "reads": [{"node": "file_triage", "mode": "last"}]},
        {"name": "incremental_synthesis_llm", "type": "incremental_synthesis",
         "params": {"read_node": "read_next_source",
                    "model_params": {"provider": "local", "model": "qwen2.5:7b", "temperature": 0.0},
                    "truncate_budget_fraction": 0.75},
         "reads": [{"node": "read_next_source", "mode": "last"}]},
        {"name": "reflection_gate", "type": "reflection_gate",
         "params": {"synthesis_node": "incremental_synthesis_llm", "triage_node": "file_triage",
                    "read_node": "read_next_source",
                    "model_params": {"provider": "local", "model": "qwen2.5:7b", "temperature": 0.0}},
         "reads": [{"node": "incremental_synthesis_llm", "mode": "last"},
                   {"node": "file_triage", "mode": "last"},
                   {"node": "read_next_source", "mode": "all"}]},
        {"name": "finalize_incremental", "type": "finalize_incremental",
         "params": {"synthesis_node": "incremental_synthesis_llm", "triage_node": "file_triage",
                    "read_node": "read_next_source", "reflection_node": "reflection_gate"},
         "writes": ["summary_out_LLM", "metadata_out_LLM"],
         "reads": [{"node": "incremental_synthesis_llm", "mode": "last"},
                   {"node": "file_triage", "mode": "last"},
                   {"node": "read_next_source", "mode": "all"},
                   {"node": "reflection_gate", "mode": "all"}]},
        {"name": "aggregator", "type": "aggregator", "params": {"pipeline": ["extract_named_outputs"]},
         "reads": [{"node": "summary_out_LLM", "mode": "last"}, {"node": "metadata_out_LLM", "mode": "last"}]},
    ],
    "conditions": [
        {"name": "iterative_revision_gate", "type": "iterative_revision_gate",
         "params": {"max_iterations": 5, "reflection_node": "reflection_gate",
                    "triage_node": "file_triage", "read_node": "read_next_source"},
         "reads": [{"node": "file_triage", "mode": "last"},
                   {"node": "read_next_source", "mode": "all"},
                   {"node": "reflection_gate", "mode": "all"}]},
    ],
    "edges": [
        {"source": "file_triage", "target": "read_next_source"},
        {"source": "read_next_source", "target": "incremental_synthesis_llm"},
        {"source": "incremental_synthesis_llm", "target": "reflection_gate"},
        {"source": "reflection_gate", "condition": "iterative_revision_gate",
         "condition_targets": ["read_next_source", "finalize_incremental"]},
        {"source": "finalize_incremental", "target": "aggregator"},
        {"source": "aggregator", "target": "END"},
    ],
}


def test_direct_agent_spawn_topology_compiles(tmp_path):
    assert build_graph(_write_topology(tmp_path, DIRECT_AGENT_SPAWN_TOPOLOGY)) is not None


def test_iterative_synthesis_topology_compiles(tmp_path):
    assert build_graph(_write_topology(tmp_path, ITERATIVE_SYNTHESIS_TOPOLOGY)) is not None