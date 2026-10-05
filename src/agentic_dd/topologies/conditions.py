'''
Condition contract: a condition function has signature
    (state: AgentState, config: RunnableConfig) -> Union[str, List[str], Send, List[Send]]
and returns the raw routing value directly.

Conditions are pure control flow. The decision they cause is implicit
in which node LangGraph dispatches to next, not a logged output, and thus they do not write
to a results channel.
'''
from __future__ import annotations
from typing import Callable, Dict, Any, List, Optional, Union
from langgraph.types import Send
from langchain_core.runnables import RunnableConfig

from agentic_dd.agentlib.state import GraphState
from agentic_dd.agentlib.nodes_factory import ReadSpec, select_reads

RouteValue = Union[str, List[str], Send, List[Send]]
ConditionFn = Callable[[GraphState, RunnableConfig], RouteValue]
ConditionBuilder = Callable[[str, Dict[str, Any]], ConditionFn]

CONDITION_TYPE_REGISTRY: Dict[str, ConditionBuilder] = {}


def register_condition_type(type_name: str):
    def _decorator(builder: ConditionBuilder) -> ConditionBuilder:
        if type_name in CONDITION_TYPE_REGISTRY:
            raise ValueError(f"Condition type conflict with silent overwrite: '{type_name}' is already registered")
        CONDITION_TYPE_REGISTRY[type_name] = builder
        return builder
    return _decorator


def register_condition(name: Optional[str] = None):
    ''' 
    Sugar for register_condition_type, for bespoke, parameter-free
    conditions (the wrapped function IS the condition function). 
    
    Same for node registration.
    '''
    def _decorator(fn: Callable) -> Callable:
        type_name = name or fn.__name__

        def _builder(_name: str, _params: Dict[str, Any]) -> ConditionFn:
            return fn

        register_condition_type(type_name)(_builder)
        return fn
    return _decorator


def list_available_condition_types() -> List[str]:
    return sorted(CONDITION_TYPE_REGISTRY.keys())


def build_condition(
    name: str,
    compute: ConditionFn,
    reads: Optional[List[ReadSpec]] = None,
) -> Callable:
    ''' 
    Wraps a condition compute function for graph wiring: 
    - filters visible `results` per `reads`,
    - then returns the raw routing value unchanged. 
    '''
    def _condition(state: GraphState, config: RunnableConfig) -> RouteValue:
        visible_results = select_reads(state.get("results"), reads, own_targets=None)
        filtered_state: GraphState = {**state, "results": visible_results}
        return compute(filtered_state, config)

    return _condition


def instantiate_condition(
    name: str,
    type_name: str,
    params: Optional[Dict[str, Any]] = None,
    reads: Optional[List[ReadSpec]] = None,
) -> Callable:
    if type_name not in CONDITION_TYPE_REGISTRY:
        raise ValueError(f"Unknown condition type '{type_name}' (referenced by condition '{name}')")
    builder = CONDITION_TYPE_REGISTRY[type_name]
    compute = builder(name, params or {})
    return build_condition(name, compute, reads=reads)