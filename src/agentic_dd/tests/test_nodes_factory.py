import pytest
from agentic_dd.agentlib.nodes_factory import (
    select_reads, ReadSpec, build_node, instantiate_node, register_node_type, NODE_TYPE_REGISTRY,
)


def _r(node, output):
    return {"node": node, "output": output, "error": None}


# --------------------------- select_reads --------------------------- #

def test_select_reads_none_is_unrestricted():
    results = [_r("a", 1), _r("b", 2)]
    assert select_reads(results, None) == results


def test_select_reads_last_mode():
    results = [_r("a", 1), _r("a", 2), _r("b", 3)]
    assert select_reads(results, [ReadSpec(node="a", mode="last")]) == [_r("a", 2)]


def test_select_reads_all_mode():
    results = [_r("a", 1), _r("a", 2), _r("b", 3)]
    assert select_reads(results, [ReadSpec(node="a", mode="all")]) == [_r("a", 1), _r("a", 2)]


def test_select_reads_wildcard_last():
    results = [_r("a", 1), _r("a", 2), _r("b", 3)]
    assert select_reads(results, [ReadSpec(node="*", mode="last")]) == [_r("a", 2), _r("b", 3)]


def test_select_reads_wildcard_all():
    results = [_r("a", 1), _r("b", 2)]
    assert select_reads(results, [ReadSpec(node="*", mode="all")]) == results


def test_select_reads_own_targets_always_visible_even_if_not_in_reads():
    results = [_r("a", 1), _r("self", 99)]
    assert select_reads(results, [], own_targets=["self"]) == [_r("self", 99)]


def test_select_reads_empty_reads_list_restricts_fully_without_own_targets():
    assert select_reads([_r("a", 1)], [], own_targets=None) == []


# --------------------------- build_node --------------------------- #

def test_build_node_default_single_write():
    node = build_node("mynode", lambda state, config: {"value": 42})
    result = node({"results": []}, {})
    assert result == {"results": [{"node": "mynode", "output": {"value": 42}, "error": None}]}


def test_build_node_multi_write_success():
    node = build_node("mynode", lambda state, config: {"a": 1, "b": 2}, writes=["a", "b"])
    result = node({"results": []}, {})
    outs = {r["node"]: r["output"] for r in result["results"]}
    assert outs == {"a": 1, "b": 2}


def test_build_node_multi_write_missing_key_raises():
    node = build_node("mynode", lambda state, config: {"a": 1}, writes=["a", "b"])
    with pytest.raises(ValueError):
        node({"results": []}, {})


def test_build_node_multi_write_non_dict_return_raises():
    node = build_node("mynode", lambda state, config: "not a dict", writes=["a", "b"])
    with pytest.raises(ValueError):
        node({"results": []}, {})


def test_build_node_compute_error_produces_error_entries_for_all_targets():
    def _compute(state, config):
        raise RuntimeError("boom")
    node = build_node("mynode", _compute, writes=["a", "b"])
    result = node({"results": []}, {})
    outs = {r["node"]: r for r in result["results"]}
    assert outs["a"]["error"] is not None and outs["a"]["output"] is None
    assert outs["b"]["error"] is not None and outs["b"]["output"] is None


def test_build_node_passes_through_chunk_index():
    node = build_node("mynode", lambda state, config: "x")
    result = node({"results": [], "chunk_index": 3}, {})
    assert result["results"][0]["chunk_index"] == 3


def test_build_node_filters_visible_results_per_reads():
    captured = {}

    def _compute(state, config):
        captured["seen"] = state.get("results")
        return "ok"

    node = build_node("consumer", _compute, reads=[ReadSpec(node="a", mode="last")])
    state = {"results": [_r("a", 1), _r("b", 2)]}
    node(state, {})
    assert captured["seen"] == [_r("a", 1)]


# --------------------------- registration --------------------------- #

def test_instantiate_node_unknown_type_raises():
    with pytest.raises(ValueError):
        instantiate_node("x", "definitely_not_a_registered_node_type__test", {})


def test_register_node_type_duplicate_raises():
    type_name = "__test_dup_node_type__"
    register_node_type(type_name)(lambda name, params: (lambda s, c: None))
    try:
        with pytest.raises(ValueError):
            register_node_type(type_name)(lambda name, params: (lambda s, c: None))
    finally:
        NODE_TYPE_REGISTRY.pop(type_name, None)