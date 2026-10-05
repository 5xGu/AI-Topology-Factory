import pytest
from pydantic import ValidationError
from agentic_dd.topologies import validate_loaded_topology


def _minimal_valid_dict():
    return {
        "entry": "n1",
        "nodes": [{"name": "n1", "type": "t1"}],
        "edges": [{"source": "n1", "target": "END"}],
    }


def test_minimal_topology_validates():
    spec = validate_loaded_topology(_minimal_valid_dict())
    assert spec.entry == "n1"


def test_bare_string_node_shorthand():
    d = _minimal_valid_dict()
    d["nodes"] = ["n1"]
    spec = validate_loaded_topology(d)
    assert spec.nodes[0].name == "n1" and spec.nodes[0].type == "n1"


def test_end_reserved_name_rejected_as_node():
    d = {"entry": "END", "nodes": [{"name": "END", "type": "t1"}], "edges": []}
    with pytest.raises(ValidationError):
        validate_loaded_topology(d)


def test_duplicate_node_names_rejected():
    d = _minimal_valid_dict()
    d["nodes"] = [{"name": "n1", "type": "t1"}, {"name": "n1", "type": "t2"}]
    with pytest.raises(ValidationError):
        validate_loaded_topology(d)


def test_entry_must_be_declared_node():
    d = _minimal_valid_dict()
    d["entry"] = "nope"
    with pytest.raises(ValidationError):
        validate_loaded_topology(d)


def test_every_node_must_have_outgoing_edge():
    d = _minimal_valid_dict()
    d["nodes"].append({"name": "n2", "type": "t1"})  # no edge from n2
    with pytest.raises(ValidationError):
        validate_loaded_topology(d)


def test_edge_must_set_exactly_one_of_target_or_condition():
    d = _minimal_valid_dict()
    d["edges"] = [{"source": "n1"}]
    with pytest.raises(ValidationError):
        validate_loaded_topology(d)


def test_edge_cannot_set_both_target_and_condition():
    d = _minimal_valid_dict()
    d["conditions"] = [{"name": "c1", "type": "ct1"}]
    d["edges"] = [{"source": "n1", "target": "END", "condition": "c1", "condition_targets": ["END"]}]
    with pytest.raises(ValidationError):
        validate_loaded_topology(d)


def test_mixing_conditional_and_plain_edges_same_source_rejected():
    d = {
        "entry": "n1",
        "nodes": [{"name": "n1", "type": "t1"}, {"name": "n2", "type": "t1"}],
        "conditions": [{"name": "c1", "type": "ct1"}],
        "edges": [
            {"source": "n1", "target": "n2"},
            {"source": "n1", "condition": "c1", "condition_targets": ["END"]},
            {"source": "n2", "target": "END"},
        ],
    }
    with pytest.raises(ValidationError):
        validate_loaded_topology(d)


def test_multiple_conditional_edges_same_source_rejected():
    d = {
        "entry": "n1",
        "nodes": [{"name": "n1", "type": "t1"}],
        "conditions": [{"name": "c1", "type": "ct1"}, {"name": "c2", "type": "ct2"}],
        "edges": [
            {"source": "n1", "condition": "c1", "condition_targets": ["END"]},
            {"source": "n1", "condition": "c2", "condition_targets": ["END"]},
        ],
    }
    with pytest.raises(ValidationError):
        validate_loaded_topology(d)


def test_unknown_edge_target_rejected():
    d = _minimal_valid_dict()
    d["edges"] = [{"source": "n1", "target": "ghost"}]
    with pytest.raises(ValidationError):
        validate_loaded_topology(d)


def test_unknown_condition_reference_rejected():
    d = _minimal_valid_dict()
    d["edges"] = [{"source": "n1", "condition": "ghost_cond", "condition_targets": ["END"]}]
    with pytest.raises(ValidationError):
        validate_loaded_topology(d)


def test_read_referencing_unknown_write_target_rejected():
    d = _minimal_valid_dict()
    d["nodes"] = [{"name": "n1", "type": "t1", "reads": [{"node": "ghost", "mode": "last"}]}]
    with pytest.raises(ValidationError):
        validate_loaded_topology(d)


def test_read_wildcard_always_allowed():
    d = _minimal_valid_dict()
    d["nodes"] = [{"name": "n1", "type": "t1", "reads": [{"node": "*", "mode": "all"}]}]
    spec = validate_loaded_topology(d)
    assert spec.nodes[0].reads[0].node == "*"


def test_duplicate_write_targets_within_one_node_rejected():
    d = _minimal_valid_dict()
    d["nodes"] = [{"name": "n1", "type": "t1", "writes": ["a", "a"]}]
    with pytest.raises(ValidationError):
        validate_loaded_topology(d)


def test_overlapping_write_targets_across_nodes_warns_not_raises():
    ''' F1: deliberate aliasing (e.g. finalize_incremental vs. summary_out_LLM)
    is a supported pattern, not an error -- see topologies.py's logged
    warning. This only asserts it does not raise. '''
    d = {
        "entry": "n1",
        "nodes": [
            {"name": "n1", "type": "t1", "writes": ["shared"]},
            {"name": "n2", "type": "t1", "writes": ["shared"]},
        ],
        "edges": [{"source": "n1", "target": "END"}, {"source": "n2", "target": "END"}],
    }
    spec = validate_loaded_topology(d)
    assert {n.name for n in spec.nodes} == {"n1", "n2"}