import pytest
from agentic_dd.agentlib.nodes_factory import ReadSpec
from agentic_dd.topologies.conditions import (
    build_condition, instantiate_condition, register_condition_type, CONDITION_TYPE_REGISTRY
)


def _r(node, output):
    return {"node": node, "output": output, "error": None}


def test_build_condition_returns_raw_routing_value():
    cond = build_condition("mycond", lambda state, config: "target_node")
    assert cond({"results": []}, {}) == "target_node"


def test_build_condition_filters_reads():
    captured = {}

    def _compute(state, config):
        captured["seen"] = state.get("results")
        return "somewhere"

    cond = build_condition("mycond", _compute, reads=[ReadSpec(node="a", mode="last")])
    results = [_r("a", 1), _r("b", 2)]
    out = cond({"results": results}, {})
    assert out == "somewhere"
    assert captured["seen"] == [_r("a", 1)]


def test_instantiate_condition_unknown_type_raises():
    with pytest.raises(ValueError):
        instantiate_condition("x", "definitely_not_a_registered_condition_type__test", {})


def test_register_condition_type_duplicate_raises():
    type_name = "__test_dup_condition_type__"
    register_condition_type(type_name)(lambda name, params: (lambda s, c: "x"))
    try:
        with pytest.raises(ValueError):
            register_condition_type(type_name)(lambda name, params: (lambda s, c: "x"))
    finally:
        CONDITION_TYPE_REGISTRY.pop(type_name, None)