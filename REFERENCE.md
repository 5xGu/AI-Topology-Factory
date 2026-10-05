## AI generated Reference

Exhaustive listing of everything a topology author or contributor can reach for.

### Topology JSON schema

A topology file has four top-level keys: `entry`, `nodes`, `conditions` (optional,
default `[]`), `edges`.

**Node** (`NodeSpec`):
```json
{"name": "my_node", "type": "llm", "params": {...}, "reads": [...], "writes": [...]}
```
- `name`: unique identifier used everywhere else in the file (edges, `reads`, `writes`).
- `type`: must be a registered node type (see "Node types" below).
- `params`: type-specific, see below. Defaults to `{}`.
- `reads` (optional): list of `{"node": <name-or-"*">, "mode": "last"|"all"}`. Omit
  entirely for unrestricted access to the whole `results` log so far. `"*"` matches any
  node. `"last"` (default) returns only the most recent matching entry; `"all"` returns
  every one (needed for nodes that run more than once, e.g. inside a loop or chunk fan-out).
  A node's own past output (per its `writes`) is always visible regardless of `reads`.
- `writes` (optional): list of names to log this node's output under. Defaults to
  `[name]`. More than one name means this node's compute function must return a dict
  keyed by each of those names (see `finalize_incremental` in the iterative example).
- **Shorthand:** a bare string in the `nodes` list, e.g. `"read_file"`, is equivalent to
  `{"name": "read_file", "type": "read_file"}`.

**Condition** (`ConditionSpec`): identical shape to `NodeSpec` minus `writes`
(conditions never write to `results`). Referenced by name from an edge's `condition` field.

**Edge** (`EdgeSpec`):
```json
{"source": "node_a", "target": "node_b"}
{"source": "node_a", "condition": "my_condition", "condition_targets": ["node_b", "node_c"]}
```
- Exactly one of `target` / `condition` must be set, never both, never neither.
- `"END"` is a reserved sentinel usable as a `target` or inside `condition_targets`, to
  terminate the graph.
- A given `source` may have any number of plain (`target`-only) edges (unconditional
  fan-out), **or** exactly one conditional edge — never a mix of both, and never more
  than one conditional edge per source.
- Every declared node must appear as a `source` in at least one edge (including a
  terminal `{"source": "x", "target": "END"}`) — there is no implicit termination.

Both structural rules above (and more: unique names, `reads` pointing at real
write-targets, `entry` referencing a declared node) are validated with no dependency on
which node/condition types happen to be registered. A second, separate pass then checks
that every referenced `type` actually exists in the running process.

### Node types

| `type` | Params | Notes |
|---|---|---|
| `llm` | `prompt_spec` (dict, e.g. `{"template_name": "..."}`), `model_params` (dict: `provider`, `model`, `temperature`), `structured_output` (optional, name registered in `STRUCTURED_OUTPUT_REGISTRY`, see below), `text_source` (optional, name of an upstream node whose output becomes this node's sole input text, bypassing `state["input"]`) | If `params.tools` is a non-empty list, this becomes a tool-calling node instead of a plain single-shot call — see below. |
| `llm` *(with tools)* | all of the above, plus: `tools` (list of tool names, see "Tools" below; `"delegate"` is a special recognized name), `max_depth` (default 2; only relevant with `"delegate"`), `recursion_limit` (default 50), `selective_file_reading` (bool) | This is how "agent"/"worker"-style nodes are expressed — there is no separate `agent`/`worker` type. |
| `read_file` | none | Reads every path in `state["input"]`, returns `{path: content}`. |
| `markdown_from_json` | none | Requires a prior `read_file`-shaped output containing a `"markdown"` key. |
| `md_schema` | none | Requires `markdown_from_json` to have run earlier. |
| `json_schema` | none | Filters `read_file`'s output to `.json` files and returns their structural outline. |
| `csv_schema` | none | Filters `read_file`'s output to `.csv` files and returns per-column profiles. |
| `chunk_split` | `source_node` (default `"read_file"`), `model`, `provider`, `max_tokens` (all optional, sensible defaults) | Always runs, always returns a list of chunk texts (length 1 if no splitting needed). Pair with the `chunk_fanout_decision` condition. |
| `router` | `route` (fixed route name, or `"free"`/omitted for LLM-decided), `allowed_routes` (list, required for `"free"` mode), `fallback_route` (default `"direct"`), `model_params`, `prompt_spec` (default `free_default` template) | Logs `{"route", "reasoning", "confidence", "plan"?}`. |
| `aggregator` | `pipeline` (optional explicit list of step names — required if no router node exists in the topology), `router_node` (default `"router"`), `default_pipelines` (optional override of the built-in route→pipeline table), `step_params` (dict, keyed by step name — see "Aggregation steps" below) | Logs one dict combining every pipeline step's output. |
| `file_triage` | `ordering` (`"heuristic"` \| `"reverse"` \| `"random"`, default `"heuristic"`), `ordering_seed` | Orders `state["input"]` by directory-name role inference (`description` < `metadata` < everything else). |
| `read_next_source` | `triage_node` (default `"file_triage"`) | Pops and reads the next file in triage order. |
| `incremental_synthesis` | `prompt_spec`, `model_params`, `read_node` (default `"read_next_source"`), `truncate_budget_fraction` (default 0.75) | Self-referential: looks up its own prior output by its own instance `name` — do not alias its `writes` elsewhere. |
| `reflection_gate` | `prompt_spec`, `model_params`, `synthesis_node`, `triage_node`, `read_node` (defaults matching the names above) | Decides sufficiency/conflict via structured LLM output. |
| `finalize_incremental` | `synthesis_node`, `triage_node`, `read_node`, `reflection_node` (defaults as above) | **Must** be declared with `"writes": ["summary_out_LLM", "metadata_out_LLM"]` — its compute function always returns exactly those two keys. |

### Condition types

| `type` | Params | Notes |
|---|---|---|
| `route_decision` | none | Reads the `router` node's logged output; dispatches to a fixed target, or fans out via `Send` for the `spawn` route. |
| `chunk_fanout_decision` | none | Reads `chunk_split`'s logged output; fans out to `chunk_summary_llm`/`chunk_metadata_llm` if more than one chunk, else routes to `summary_out_LLM`/`metadata_out_LLM`. |
| `iterative_revision_gate` | `max_iterations` (default 5), `reflection_node`, `triage_node`, `read_node` (defaults matching the node table above) | Enforces the loop's iteration cap independently of LangGraph's own `recursion_limit`. |

### Structured outputs (`params.structured_output`)

Registered names usable by any `llm`-type node: `MetadataResponse`, `RouterDecision`,
`IncrementalSynthesis`, `ReflectionVerdict` (defined in `agentlib/structuredOutputs.py`).

### Aggregation steps (`params.pipeline` / `params.step_params`)

| Step name | Params (under `step_params.<name>`) | Reads from `acc` (earlier steps in the same pipeline) |
|---|---|---|
| `concat` | none | — |
| `extract_named_outputs` | `summary_node`, `metadata_node` (defaults `summary_out_LLM`/`metadata_out_LLM`) | — |
| `hierarchical_reduce_recursive` | `model_params`, `max_tokens`, `max_depth`, `source_node` (default `chunk_summary_llm`) | — |
| `metadata_merge` | `source_nodes` (list, default `["metadata_out_LLM", "chunk_metadata_llm"]`) | — |
| `llm_merge` | `model_params`, `prompt_spec` | — |
| `critic` | `model_params`, `prompt_spec` | `final_output` |
| `auto_direct` | (passes through to whichever of the above it dispatches to) | — |
| `agent_structured_split` | `parser_model_params`, `metadata_model_params` | `final_output` |

Built-in route→pipeline table (`DEFAULT_PIPELINES`, used when `params.pipeline` is
omitted and a router is present): `direct` → `["auto_direct"]`; `spawn` →
`["concat", "llm_merge", "agent_structured_split"]`; `agent` →
`["concat", "agent_structured_split"]`.

### Tools (`params.tools` on a tool-calling `llm` node)

`read_file`, `md_outline`, `md_get_section`, `md_search`, `md_tables`, `md_images`,
`csv_schema`, `csv_sample`, `csv_column_stats`, `csv_inspect_column`, `csv_full_column`,
`csv_row`, `json_schema_outline`, `schema_to_list`, `get_json_entity_by_path`,
`is_docling_json`, `get_path_children`, `get_path_descendants`, `get_entities_of_type`,
`get_keys_for_level`, `llm_extract`.

`"delegate"` is a special recognized name (not in the tool registry itself): it grants
the node the ability to spin up a fresh sub-agent, up to `params.max_depth` deep.

> Tools currently have no per-call parameterization mechanism of their own (unlike
> nodes) — e.g. `llm_extract` always uses a single fixed, hardcoded `ModelParams`. See
> the note in `agentlib/tools_llm_extract.py`.

### Prompt templates (`params.prompt_spec.template_name`)

`summarizer_default`, `metadata_default`, `worker_default`, `agent_default`,
`subagent_default`, `critic_default`, `merge_default`, `assistant_default`,
`free_default`, `selective_file_reading_addendum`, `llm_extractor_default`,
`metadata_extraction_expert` (plus `_0shot`/`_1shot`/`_3shot` variants),
`summarizer_dataset_abstract`, `summarizer_constrained`, `summarizer_cot`,
`summary_eval`, `incremental_synthesis_default`, `reflection_gate_default`.

### CLI (`agentic_dd.main`)

| Flag | Required | Default | Notes |
|---|---|---|---|
| `--topology` | yes | — | No `default_topology.json` currently exists in this repo, so this is effectively required in practice. |
| `--input` | yes | — | One or more file paths and/or directories (directories expanded non-recursively to `.csv`/`.json` files). |
| `--configurable` | no | none | Path to a JSON file of invocation-time overrides (`model_params`, `prompt_overrides`, `max_depth`), keyed by node name. |
| `--experiment-name` | no | `agentic_dd` | MLflow experiment name. |
| `--run-name` | no | none | MLflow run name. |
| `--recursion-limit` | no | `50` | LangGraph's own recursion cap for this invocation. |
| `--mlflow-tracking-uri` | no | none | Overrides `MLFLOW_TRACKING_URI` if given. |

### Environment variables (`.env`)

| Variable | Required when | Notes |
|---|---|---|
| `BASE_URL` | any node uses `"provider": "api"` | No public default — must be your institutional endpoint. |
| `API_KEY` | same as above | |
| `OLLAMA_BASE_URL` | any node uses `"provider": "local"` | Defaults to `http://localhost:11434`. |
| `FORCE_LOCAL_LLM` | optional | `"1"` forces every LLM call onto `FORCE_LOCAL_MODEL`, bypassing per-node `model_params` — useful for cheap smoke tests. |
| `FORCE_LOCAL_MODEL` | with the above | Default `qwen2.5:7b`. |
| `TOKEN_COUNTS_FILE` | optional | Path to a precomputed token-counts CSV, used by `chunk_split`/chunking-related token estimation. |
| `SUB_DATA_ROOT` | with the above | Root directory input paths are made relative to when looking up precomputed counts. |
| `MLFLOW_TRACKING_URI` | optional | Defaults to a local `./mlruns/` directory if unset. |

### Known limitations / out of scope

- Data preprocessing or evaluation are not implemented currently.
- Tools have no topology-level parameterization; any tool needing configuration (e.g.
  `llm_extract`'s model choice) currently uses a fixed value in Python.
