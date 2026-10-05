'''
Factory for all nodes, regardless of type or function. Enforces uniform tracking and a uniformly handled node contract across all node types:
All nodes write ther results to the same results communication channel in the same basic schema:
    
    {"node": name, "output": Any, "error": Optional[str]}

This means that all results are essentially stored like a log, which is partionied by provence and keyed by result producer name.

The factory wraps a compute function object into a node object and therefore strictly separates these layers from each other.

Compute functions take the graph's state and a runnable config object. These are the only two inputs allowed to nodes by LangGraph. They return the
raw output value, a dictionary where each key corresponds to a write target name and each value is a raw output value, or raise errors. They never
wrap their results themselves into the node contract, as compute functions are not nodes.


'''
from __future__ import annotations
from typing import Callable, Dict, Any, List, Optional, Literal, Iterable, Type
from pydantic import BaseModel
import mlflow
from mlflow.entities import SpanType
from langchain_core.runnables import RunnableConfig

from agentic_dd.agentlib.state import GraphState
from agentic_dd.agentlib.mlflow_utils import ensure_active_run
from agentic_dd.logging import default_logger, ai_default_logger


ComputeFn = Callable[[GraphState, RunnableConfig], Any]
NodeBuilder = Callable[[str, Dict[str, Any]], ComputeFn]

NODE_TYPE_REGISTRY: Dict[str, NodeBuilder] = {}


########## Read from results operation 

class ReadSpec(BaseModel):
    node: str                              # eiterh a node name, a write-target alias, or "*"
    mode: Literal["all", "last"] = "last"


def select_reads(
    results: Optional[List[Dict[str, Any]]], 
    reads: Optional[List[ReadSpec]], 
    own_targets: Optional[Iterable[str]] = None
    ) -> List[Dict[str, Any]]:

    ''' 
    
    Filters a results log down to exactly what a node declared
    it may see. 
    
    - reads = None means unrestricted access to results
    
    A node's own write-target names are always fully visible
    regardless of `reads`
    
    '''
    results = results or []
    if reads is None:
        return list(results)

    own = set(own_targets or [])
    last_seen: Dict[str, int] = {}
    for idx, item in enumerate(results):
        node = item.get("node")
        if node is not None:
            last_seen[node] = idx

    keep: set = set()
    for idx, item in enumerate(results):
        if item.get("node") in own:
            keep.add(idx)

    for spec in reads:
        if spec.node == "*":
            if spec.mode == "all":
                keep.update(range(len(results)))
            else:
                keep.update(last_seen.values())
        else:
            if spec.mode == "all":
                keep.update(idx for idx, item in enumerate(results) if item.get("node") == spec.node)
            elif spec.node in last_seen:
                keep.add(last_seen[spec.node])

    return [results[i] for i in sorted(keep)]



########## Node Types 

def register_node_type(type_name: str):
    ''' 
    Registers a builder: 
        (name, params) -> ComputeFn 
    under a type name. 
    Is used by topology files via {"type": type_name, ...} 
    '''
    def _decorator(builder: NodeBuilder) -> NodeBuilder:
        if type_name in NODE_TYPE_REGISTRY:
            message = f"Node type conflict with silent overwrite: '{type_name}' is already registered"
            default_logger.error(message)
            raise ValueError(message)
        NODE_TYPE_REGISTRY[type_name] = builder
        default_logger.debug(f"Registered node type '{type_name}'")
        return builder
    return _decorator


def register_node(name: Optional[str] = None):
    ''' 
    Syntactical sugar for register_node_type. 
    Use for bespoke, parameter-free node types. 
    '''
    def _decorator(compute: Callable) -> Callable:
        type_name = name or compute.__name__

        def _builder(_name: str, _params: Dict[str, Any]) -> ComputeFn:
            return compute

        register_node_type(type_name)(_builder)
        return compute
    return _decorator

def list_available_node_types() -> List[str]:
    return sorted(NODE_TYPE_REGISTRY.keys())



########## Node Factory

def build_node(
    name: str,
    compute: ComputeFn,
    reads: Optional[List[ReadSpec]] = None,
    writes: Optional[List[str]] = None,
) -> Callable:
    ''' 
    Wraps a compute function into a node, which is used within the graph.
    Every node:
        - filters visible `results` via `reads`,     
        - runs `compute` and stores the raw return value in one ore more 
        `results` log entries depending on the `writes` parameter.
    '''
    targets = writes or [name]

    def _node(state: GraphState, config: RunnableConfig) -> Dict[str, Any]:
        configurable = (config or {}).get("configurable", {}) or {}
        run_id = configurable.get("mlflow_run_id")
        visible_results = select_reads(state.get("results"), reads, own_targets=targets)
        filtered_state: GraphState = {**state, "results": visible_results}

        default_logger.debug(f"[{name}] running (visible_results={len(visible_results)})")

        error: Optional[str] = None
        output: Any = None
        with ensure_active_run(run_id):
            with mlflow.start_span(name=name, span_type=SpanType.CHAIN) as span:
                span.set_inputs({"input": state.get("input")})
                try:
                    output = compute(filtered_state, config)
                except Exception as e:
                    default_logger.error(f"[{name}] compute raised an exception", exc_info=True)
                    import traceback; traceback.print_exc()
                    error = repr(e)
                span.set_outputs({"output": str(output)[:2000], "error": error})

        if error is None:
            default_logger.debug(f"[{name}] completed successfully")

        chunk_index = state.get("chunk_index")

        def _entry(node_name: str, node_output: Any, node_error: Optional[str]) -> Dict[str, Any]:
            entry = {"node": node_name, "output": node_output, "error": node_error}
            if chunk_index is not None:
                entry["chunk_index"] = chunk_index
            return entry

        if error is not None:
            results_out = [_entry(t, None, error) for t in targets]
        elif len(targets) > 1:
            if not isinstance(output, dict) or (set(targets) - set(output.keys())):
                raise ValueError(
                    f"Node '{name}' declares writes={targets} but its compute function did not "
                    f"return a dict containing all of those keys (got: {output!r})"
                )
            results_out = [_entry(t, output[t], None) for t in targets]
        else:
            results_out = [_entry(targets[0], output, None)]

        return {"results": results_out}

    return _node

def instantiate_node(
    name: str,
    type_name: str,
    params: Optional[Dict[str, Any]] = None,
    reads: Optional[List[ReadSpec]] = None,
    writes: Optional[List[str]] = None,
) -> Callable:
    ''' 
    Instantiates nodes by:
        - Looking up the type_name in NODE_TYPE_REGISTRY
        - building a compute function from (name, params)
        - wrapping the compute function into a node
        
    Raises immediately at topology-build time if a type name is
    not registered or if params fail the type's validation.
    
    '''
    if type_name not in NODE_TYPE_REGISTRY:
        message = f"Unknown node type '{type_name}' (referenced by node '{name}')"
        default_logger.error(message)
        raise ValueError(message)
    builder = NODE_TYPE_REGISTRY[type_name]
    compute = builder(name, params or {})
    default_logger.debug(f"Instantiated node '{name}' (type='{type_name}')")
    return build_node(name, compute, reads=reads, writes=writes)



########## Structured Output Registry for generic LLM type

STRUCTURED_OUTPUT_REGISTRY: Dict[str, Type[BaseModel]] = {}


def register_structured_output(name: str):
    def _decorator(cls: Type[BaseModel]) -> Type[BaseModel]:
        if name in STRUCTURED_OUTPUT_REGISTRY:
            message = f"Structured output name conflict with silent overwrite: '{name}' is already registered"
            default_logger.error(message)
            raise ValueError(message)
        STRUCTURED_OUTPUT_REGISTRY[name] = cls
        default_logger.debug(f"Registered structured output '{name}'")
        return cls
    return _decorator

def find_structured_output(name: str) -> Type[BaseModel]:
    if name not in STRUCTURED_OUTPUT_REGISTRY:
        message = f"Unknown structured output '{name}'"
        default_logger.error(message)
        raise ValueError(message)
    return STRUCTURED_OUTPUT_REGISTRY[name]



########## Generic LLM type

def build_plain_llm_compute(name: str, params: Dict[str, Any]) -> ComputeFn:
    ''' 
    Params:
        prompt_spec (dict, -> PromptSpec), 
        model_params (dict, -> ModelParams), 
        structured_output (Optional[str], looked up in STRUCTURED_OUTPUT_REGISTRY). 
        text_source (Optional[str]): if set, names an upstream node whose latest output becomes this node's sole primary text content
    
    Invocation-time overrides of model_params or prompts are read from
        config['configurable']
    and keyed by this node's own `name` parameters. They take precedence over 
    parameters specified in the topology.
    
    '''
    from agentic_dd.agentlib.llm import ModelParams, get_model_for
    from agentic_dd.agentlib.prompts import to_prompt_spec, compose_messages
    from agentic_dd.agentlib.mlflow_utils import tag_resolved_model

    default_model_params = ModelParams(**(params.get("model_params") or {}))
    default_spec = to_prompt_spec(params.get("prompt_spec") or {})
    structured_output_name = params.get("structured_output")
    structured_output_cls = find_structured_output(structured_output_name) if structured_output_name else None
    text_source = params.get("text_source")

    def _compute(state: GraphState, config: RunnableConfig) -> Any:
        configurable = (config or {}).get("configurable", {}) or {}
        override_model_params = (configurable.get("model_params") or {}).get(name)
        override_prompt = (configurable.get("prompt_overrides") or {}).get(name)

        model_params = ModelParams(**override_model_params) if override_model_params else default_model_params
        spec = to_prompt_spec(override_prompt) if override_prompt else default_spec

        if text_source:
            source_output = find_latest_node_output(state.get("results", []), text_source)
            primary_text = "\n\n".join(str(s) for s in source_output) if isinstance(source_output, list) else source_output
            context = {"input": primary_text, "prior_results": None}
        else:
            context = {"input": state["input"], "prior_results": state.get("results")}

        tag_resolved_model(name, model_params.provider, model_params.model)
        llm = get_model_for(model_params)
        if structured_output_cls:
            llm = llm.with_structured_output(structured_output_cls, include_raw=True)

        ai_default_logger.debug(f"[{name}] invoking LLM")
        try:
            resp = llm.invoke(compose_messages(spec, context))
        except Exception as e:
            if "maximum context length" in str(e):
                ai_default_logger.warning(f"[{name}] context length exceeded")
                mlflow.set_tag(f"context_overflow::{name}", "true")
            raise

        if structured_output_cls:
            parsing_error = resp.get("parsing_error")
            if parsing_error:
                ai_default_logger.warning(f"[{name}] structured output parse failed: {parsing_error}")
                mlflow.log_metric(f"{name}_parse_errors", 1)
                return {"parsed": structured_output_cls().model_dump(), "parsing_error": str(parsing_error)}
            return {"parsed": resp["parsed"].model_dump(), "parsing_error": None}
        return {"content": resp.content}

    return _compute


def build_llm_compute(name: str, params: Dict[str, Any]) -> ComputeFn:
    ''' 
    The generic "llm" node type, with a single pass upon call.

    "Agent" and "worker" nodes are instantiated by passing non-empty params['tools']. This is driven by
    the intuition that Agents are functionally LLMs with tool-use.
    '''
    if params.get("tools"):
        from agentic_dd.agentlib.nodes_agent import build_tool_calling_llm_compute
        return build_tool_calling_llm_compute(name, params)
    return build_plain_llm_compute(name, params)


register_node_type("llm")(build_llm_compute)



########## Results log lookup-helpers


def find_node_output(results: list, node_name: str) -> Any:
    ''' First-match. Correct for nodes that run once per sample. '''
    for item in results or []:
        if isinstance(item, dict) and item.get("node") == node_name:
            return item.get("output")
    return None

def find_latest_node_output(results: list, node_name: str) -> Any:
    ''' Most-recent match. Needed for nodes that run multiple times per
    sample (chunk fan-out, or the iterative synthesis/reflection loop). '''
    latest = None
    for item in results or []:
        if isinstance(item, dict) and item.get("node") == node_name:
            latest = item.get("output")
    return latest

def find_all_node_outputs(results: list, node_name: str) -> List[Any]:
    return [item.get("output") for item in results or []
            if isinstance(item, dict) and item.get("node") == node_name]