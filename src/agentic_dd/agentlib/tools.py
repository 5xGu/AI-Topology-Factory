from __future__ import annotations
from typing import Optional, List, Any, Dict, Callable
import warnings
from agentic_dd.agentlib.file_io import read_one_file
from langchain_core.tools import tool, StructuredTool

TOOL_REGISTRY: Dict[str, Callable] = {}


def register_tool(name: str, tool_obj: Callable) -> None:
    TOOL_REGISTRY[name] = tool_obj


def register_tools_from_instance(instance: Any, tool_to_method: Dict[str, str]) -> None:
    for tool_name, method_name in tool_to_method.items():
        method = getattr(instance, method_name)
        register_tool(
            tool_name,
            StructuredTool.from_function(
                func=method, name=tool_name,
                description=(method.__doc__ or "") or f"Call {method_name}.",
            ),
        )


def resolve_tools(names: Optional[List[str]], default: List[Any]) -> List[Any]:
    if not names:
        return default
    resolved = []
    for n in names:
        if n in TOOL_REGISTRY:
            resolved.append(TOOL_REGISTRY[n])
        else:
            warnings.warn(f"Unknown tool name requested: {n!r}")
    return resolved


@tool
def read_file(fp: str, tool_call_budget: int = 800000) -> Any:
    ''' Reads a .csv or .json file and returns its contents directly. '''
    result = read_one_file(fp, tool_call_budget=tool_call_budget)
    if isinstance(result, str) and result.startswith("[File too large for direct reading"):
        try:
            import mlflow
            mlflow.set_tag(f"file_rejected_for_size::{fp}", "true")
        except Exception:
            pass
    return result


register_tool("read_file", read_file)