from __future__ import annotations
from typing import List
from langgraph.types import Send

from agentic_dd.agentlib.state import GraphState
from agentic_dd.agentlib.nodes_factory import find_latest_node_output


def spawn_workers(state: GraphState) -> List[Send]:
    router_output = find_latest_node_output(state.get("results", []), "router") or {}
    plan = router_output.get("plan") or []
    return [Send("worker", {**state, "task": item["task"]}) for item in plan]