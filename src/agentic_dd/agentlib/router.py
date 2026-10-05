'''
Router node type ("router") and its dispatching condition ("route_decision").
'''
from __future__ import annotations
from typing import Dict, Any, List
from langchain_core.runnables import RunnableConfig

from agentic_dd.agentlib.state import GraphState
from agentic_dd.agentlib.llm import get_model_for, ModelParams
from agentic_dd.agentlib.mlflow_utils import tag_resolved_model
from agentic_dd.agentlib.prompts import PromptSpec, to_prompt_spec, compose_messages
from agentic_dd.agentlib.structuredOutputs import RouterDecision
from agentic_dd.agentlib.nodes_factory import register_node_type, find_latest_node_output
from agentic_dd.agentlib.spawn import spawn_workers
from agentic_dd.topologies.conditions import register_condition


def build_router_compute(name: str, params: Dict[str, Any]):
    '''
    Params:
        - route (Optional[str]; a fixed route, or None/"free" for LLM-decided),
        - allowed_routes (List[str]; required when route is None/"free" to validate the LLM's choice),
        - fallback_route (str, defaults to "direct"; used if the LLM's route isn't allowed, or "spawn" is chosen with an empty plan),
        - model_params (dict),
        - prompt_spec (dict, defaults to {"template_name": "free_default"}).

    allowed_routes and the global tool name list are injected into prompt variables.
    '''
    fixed_route = params.get("route")
    allowed_routes: List[str] = params.get("allowed_routes") or []
    fallback_route: str = params.get("fallback_route", "direct")
    default_model_params = ModelParams(**(params.get("model_params") or {}))
    default_spec = to_prompt_spec(params.get("prompt_spec") or {"template_name": "free_default"})

    def _compute(state: GraphState, config: RunnableConfig) -> Any:
        if fixed_route and fixed_route != "free":
            return {"route": fixed_route, "reasoning": "fixed by topology params", "confidence": 1.0}

        configurable = (config or {}).get("configurable", {}) or {}
        override_model_params = (configurable.get("model_params") or {}).get(name)
        model_params = ModelParams(**override_model_params) if override_model_params else default_model_params
        override_prompt = (configurable.get("prompt_overrides") or {}).get(name)
        spec = to_prompt_spec(override_prompt) if override_prompt else default_spec

        decision = _routing_llm_call(state, model_params, spec, allowed_routes)
        route = decision.route
        reasoning = decision.reasoning
        plan = [item.model_dump() for item in (decision.plan or [])]

        if route not in allowed_routes:
            reasoning = f"{reasoning} [fallback: '{route}' not in allowed_routes, using '{fallback_route}']"
            route = fallback_route
            plan = []
        elif route == "spawn" and not plan:
            reasoning = f"{reasoning} [fallback: empty plan, using '{fallback_route}']"
            route = fallback_route

        output: Dict[str, Any] = {"route": route, "reasoning": reasoning, "confidence": decision.confidence}
        if route == "spawn":
            output["plan"] = plan
        return output

    return _compute


register_node_type("router")(build_router_compute)


def _routing_llm_call(state: GraphState, model_params: ModelParams, spec: PromptSpec, allowed_routes: List[str]) -> RouterDecision:
    from agentic_dd.agentlib.tools import TOOL_REGISTRY  # read at call time, after all tool registration has completed

    tag_resolved_model("router", model_params.provider, model_params.model)
    routing_llm = get_model_for(model_params).with_structured_output(RouterDecision, include_raw=True)

    rendered_spec = spec.model_copy(update={"variables": {
        **spec.variables,
        "allowed_routes": ", ".join(allowed_routes),
        "tool_names": ", ".join(sorted(TOOL_REGISTRY.keys())),
    }})
    messages = compose_messages(rendered_spec, {"input": state["input"]})
    resp = routing_llm.invoke(messages)
    if resp.get("parsing_error") or resp.get("parsed") is None:
        return RouterDecision(
            route="direct",
            reasoning=f"[router parse failed, defaulting to direct: {resp.get('parsing_error')}]",
            confidence=0.0,
        )
    return resp["parsed"]


@register_condition("route_decision")
def route_decision(state: GraphState, config: RunnableConfig):
    '''
    Reads the router's own logged output (results entry named "router"),
    not a dedicated state field.

    "spawn" fans out via Send; every other route is returned as a bare node-name string, checked against the
    edge's condition_targets at topology-validation time.
    '''
    router_output = find_latest_node_output(state.get("results", []), "router") or {}
    route = router_output.get("route")
    if route is None:
        raise ValueError("route_decision: no 'router' output found in results; router must run before this condition")
    if route == "spawn":
        return spawn_workers(state)
    return route