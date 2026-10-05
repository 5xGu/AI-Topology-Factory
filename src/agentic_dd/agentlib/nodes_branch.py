from __future__ import annotations
from typing import Dict, Any, Optional
from pathlib import Path

from agentic_dd.agentlib.nodes_factory import register_node, find_node_output
from agentic_dd.agentlib.file_io import read_files


@register_node("read_file")
def read_file(state, config=None) -> Any:
    return read_files(state["input"], tool_call_budget=None)


@register_node("markdown_from_json")
def markdown_from_json(state, config=None) -> Any:
    md_source = find_node_output(state.get("results", []), "read_file")
    if isinstance(md_source, dict) and "markdown" in md_source:
        return md_source["markdown"]
    return None


@register_node("md_schema")
def md_schema(state, config=None) -> Any:
    # Requires "markdown_from_json" to have run earlier in the topology.
    from agentic_dd.extraction.markdownExtraction import MarkdownDocument
    md_text = find_node_output(state.get("results", []), "markdown_from_json")
    if md_text is None:
        raise ValueError("md_schema requires 'markdown_from_json' to run earlier in the topology")
    return MarkdownDocument(md_text).outline_json()


@register_node("json_schema")
def json_schema(state, config=None) -> Any:
    from agentic_dd.extraction.jsonExtraction import schema, path_to_string

    files = find_node_output(state.get("results", []), "read_file")
    if files is None:
        files = read_file(state, config)

    json_files = {p: c for p, c in files.items() if Path(p).suffix == ".json"}
    if not json_files:
        return []
    entries = list(schema(json_files))
    return [{"path": path_to_string(e.path), "type": e.type, "level": e.lvl} for e in entries]


@register_node("csv_schema")
def csv_schema(state, config=None) -> Any:
    from agentic_dd.extraction.csvExtraction import get_schema
    import pandas as pd

    files = find_node_output(state.get("results", []), "read_file")
    if files is None:
        files = read_file(state, config)

    schemas: Dict[str, Any] = {}
    for path, content in files.items():
        if Path(path).suffix != ".csv":
            continue
        try:
            df = content if isinstance(content, pd.DataFrame) else pd.DataFrame(content)
            schemas[path] = get_schema(df)
        except Exception:
            return "Couldn't read file, no csv schema was generated."
    return schemas   # empty if no CSV files present in this sample