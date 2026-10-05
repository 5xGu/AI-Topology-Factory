'''
Ensures every node type, condition type, tool, and prompt template is
registered as a side effect of importing this package, per the
convention documented in agentic_dd/__init__.py. 

Any import touching a submodule of agentic_dd.agentlib triggers this file first, per ordinary
Python package-loading semantics, so this is the place that determines which node/condition/tool/prompt names 
exist in a given process.

Evaluators (agentic_dd.agentlib.evaluators) are not imported
here, as they belong to the experiment/analysis layer, not the graph itself. 
They are invoked explicitly by whatever script computes metrics over a
completed run's final state, but not by the graph during a run.

nodes_agent.py is also not imported here, because it registers no
node/condition type of its own. Itt is a plain helper module, lazily
imported by nodes_factory.build_llm_compute only when a node's own params
include a non-empty "tools" list.
'''
# node types (+ colocated condition types); nodes_factory first, since
# every other node/condition module depends on it.
import agentic_dd.agentlib.nodes_factory      # noqa: F401  llm type
import agentic_dd.agentlib.nodes_branch       # noqa: F401  read_file, markdown_from_json, md_schema, json_schema, csv_schema
import agentic_dd.agentlib.nodes_iterative    # noqa: F401  file_triage, read_next_source, incremental_synthesis, reflection_gate, finalize_incremental; condition: iterative_revision_gate
import agentic_dd.agentlib.chunking           # noqa: F401  chunk_split; condition: chunk_fanout_decision
import agentic_dd.agentlib.router             # noqa: F401  router; condition: route_decision
import agentic_dd.agentlib.aggregation        # noqa: F401  aggregator

# tool registry population (consumed by "llm"-type nodes with a non-empty
# params["tools"], see nodes_agent.build_tool_calling_llm_compute)
from agentic_dd.agentlib.tools import TOOL_REGISTRY, register_tool  # noqa: F401
import agentic_dd.agentlib.tools_md           # noqa: F401
import agentic_dd.agentlib.tools_csv          # noqa: F401
import agentic_dd.agentlib.tools_json         # noqa: F401
import agentic_dd.agentlib.tools_llm_extract  # noqa: F401

# prompt template registry population
import agentic_dd.agentlib.prompt_collection  # noqa: F401