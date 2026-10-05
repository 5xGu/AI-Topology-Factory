'''
This fails loduly when a registered node or condition catalog changes. 
This can only be resolved by updating the EXPECTED_* set below.
'''
from agentic_dd.agentlib.nodes_factory import NODE_TYPE_REGISTRY
from agentic_dd.topologies.conditions import CONDITION_TYPE_REGISTRY

EXPECTED_NODE_TYPES = {
    "llm", "read_file", "markdown_from_json", "md_schema", "json_schema", "csv_schema",
    "chunk_split", "router", "aggregator",
    "file_triage", "read_next_source", "incremental_synthesis", "reflection_gate", "finalize_incremental",
}

EXPECTED_CONDITION_TYPES = {
    "route_decision", "chunk_fanout_decision", "iterative_revision_gate",
}


def test_node_type_catalog_pin():
    assert set(NODE_TYPE_REGISTRY.keys()) == EXPECTED_NODE_TYPES


def test_condition_type_catalog_pin():
    assert set(CONDITION_TYPE_REGISTRY.keys()) == EXPECTED_CONDITION_TYPES