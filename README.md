# Foreword
Due to import issues of the used gitlab instance, this project was copy/pasted and is detached. I have developed this repository as a part of my student work before.

# Agentic Discuss Data
This repository holds the foundations to develop and integrate an agentic data analysis
system for the [Discuss Data project](https://discuss-data.net/). It is based on the
[Summarizing and Metadata Agent For Discuss Data](https://gitlab.gwdg.de/plattner3/summarizing-and-metadata-agent-for-discuss-data)
repository, developed as part of an initial thesis project. That original repository
holds functionality necessitated by the thesis rather than the Discuss Data project
itself. This fork extends and restricts it to what the Discuss Data project actually needs, but contains some leftovers:
    - `file_triage` (see `agentlib/nodes_iterative.py`), which depends on a specific naming convetion your data might not share;
    - `data_processing/*` 
    
**Scope of this repository:** this is the core library and topology engine. The responsibilities are: 
    - Defining, validating and running (LLM-powered) node graphs of arbitrary topology,
    - MLflow tracking fundaments for node behaviour and graph input and output.

# Introduction
The Discuss Data repository represents a rare effort to enhance the usual data repository archetype through a community driven discussion space, in the hopes that this will lead to a higher engagement with and therefore more value provided by the uplodaded research data. One core condition of this is that the [FAIR](https://www.nature.com/articles/sdata201618) principles are realized to a high degree, for which metadata classification is a central aspect. As a concrete problem, this was often not fulfilled, due to a sparse metadata specification by data uploaders.

To that end, it was decided to employ LLMs to attempt metadata enrichment, and early tests showed the potential of this approach. Soon, it was recognized that an AI implementation could perform many more valuable tasks. For example, it could summarize uploaded datasets and the summary could serve as a discussion foundation for the community. There are however, many open questions and problems regarding the employment of AI systems, most concretely: privacy concerns and the quality of the generated output.

Therefore, it was decided to pursue the implementation of a system structure that allows the instantiation of arbitrary LLM powered systems for arbitrary NLP tasks, to evaluate the feasability of AI applications for the goal described and compare the various instances in regard to performance, costs, etc. 

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

### 1. Install uv

    curl -LsSf https://astral.sh/uv/install.sh | sh

(See uv's docs for any issues, or Windows/pipx alternatives, etc.)

### 2. Install dependencies

From the repository root:

    uv sync

This creates a `.venv/` and installs the runtime dependencies plus the
`dev` group (pytest, pylint) by default.

If you need real (non-heuristic) token counting via HuggingFace
tokenizers, also install the optional `tokenizer` extra:

    uv sync --extra tokenizer

This is recommended before you start using the library in earnest with LLM nodes, due to the input limitations of LLM's, and the potentially required chunking before they can actually process your data. However, you need to mind that tokenization takes up resources, and **that tokenization is not sandboxed natively here. I have lost data to this, so be warned.** Moreoever, you need to mind your data types and their runtime representations. For example, if you use tabular data, e.g. in the `csv` format, and load it using `pandas` as a dataframe, its size can grow by multiple magnitudes. 

#### Ollama

To use local models through Ollama, e.g. 

    curl -fsSL https://ollama.com/install.sh | sh

and 
    uv run ollama pull local_model

### 3. Configure environment variables

Copy `.env.example` to `.env` and fill in the values relevant to your
setup:

    cp .env.example .env

At minimum, if you plan to route any node to an API-hosted model
(`provider: "api"` in a node's `model_params`), `BASE_URL` and `API_KEY`
are required. The code deliberately refuses to fall back to a public
default endpoint. If you plan to use local models (`provider: "local"`),
set `OLLAMA_BASE_URL`. See the full reference below for further parameters.

### 4. Run the test suite

    uv run pytest

This is a good sanity check that your environment is set up correctly before running
anything against real data or a real model endpoint — the whole suite compiles every
registered node/condition type into a minimal graph and asserts it wires up correctly,
without making any network calls.

### 5. Running Topology Instances

The general command to run the system, i.e. instantiate a topology is this:

    uv run python -m agentic_dd.main \
        --topology path/to/topology.json \
        --input path/to/your/data/ \
        --experiment-name orientation-example

Replace the path to the topology to run and the data to use with their respective paths, and provide an experiment name for MLflow. To run a topology that either: 
    - instantiates a `direct` route, i.e. forwards the input to the output layer directly, 
    - instantiates an `agent` to process the input data and then produce the output, 
    - or spawns `workers` to analyse the input and produce the output,
based on the decision of a routing LLM, go to the root directory and run:

    uv run python -m agentic_dd.main \
        --topology src/agentic_dd/topologies/direct_agent_spawn_topology.json \
        --input sample_data/ \
        --experiment-name orientation-example


## Quickstart

Everything the system does is driven by a **topology file**, and potentially a `configurable` overwritting aspects of it. The `topology` is a JSON document
declaring which nodes exist, what each is parameterized with, and how control flows
between them.

A minimal topology that reads a set of input files and produces a one-shot LLM summary:

```json
{
  "entry": "read_file",
  "nodes": [
    "read_file",
    {
      "name": "summary_out_LLM",
      "type": "llm",
      "params": {
        "prompt_spec": {"template_name": "summarizer_default"},
        "model_params": {"provider": "local", "model": "qwen2.5:7b", "temperature": 0.0}
      }
    }
  ],
  "edges": [
    {"source": "read_file", "target": "summary_out_LLM"},
    {"source": "summary_out_LLM", "target": "END"}
  ]
}
```

Save this as `my_topology.json`, then run it:

    uv run python -m agentic_dd.main \
        --topology path/to/my_topology.json \
        --input ./sample_data/ \
        --experiment-name my-quickstart

This will:

1. Load and validate the topology, then compile it into a runnable graph.
2. Resolve `./sample_data/` into its individual `.csv`/`.json` files (you can also pass
   individual file paths directly).
3. Start one MLflow run under the `my-quickstart` experiment.
4. Invoke the graph once against those files.
5. Log the topology file, the full `results` log, and a short run summary as MLflow
   artifacts, plus per-node error tags if anything failed.
6. Print a short JSON summary (the final aggregator output, result/error counts) to
   stdout.

By default, MLflow writes to a local `./mlruns/` directory. Inspect the run with:

    uv run mlflow ui

and open `http://localhost:5000` in a browser. Set `MLFLOW_TRACKING_URI` in `.env`
in accordance with deviating preferences.

**Per-invocation overrides**, without editing the topology file (e.g. sweeping a
model or temperature across runs), go in a separate JSON file, passed via
`--configurable`:

```json
{
  "model_params": {
    "summary_out_LLM": {"provider": "api", "model": "qwen3.8-27b", "temperature": 0.2}
  }
}
```

    uv run python -m agentic_dd.main \
        --topology path/to/my_topology.json \
        --input ./sample_data/ \
        --experiment-name orientation-example \
        --configurable path/to/configurable_overrides.json

This dict is itself logged as an MLflow artifact on every run and it's the one and only source of node behavior that isn't visible in the topology file. For reproducibility, it is tracked explicitly.

Two fuller worked examples exist in the repository: a router-driven topology covering
the direct/agent/spawn routes with unconditional chunking (`direct_agent_spawn_topology.json`),
and a no-router iterative source-by-source synthesis loop (`iterative_synthesis_topology.json`). Reading through both is the fastest way to get an idea of the available predefined node/condition types and parameters in context, and how to structure topology files.

### Sample data

A small example dataset lives in `sample_data/`, used by the quickstart commands above:

```
sample_data/
├── description/
│   └── description.json
├── metadata/
│   └── metadata.csv
└── data/
    └── data.csv
```

A few things worth knowing about it:

- The directory names (`description/`, `metadata/`, `data/`) are not arbitrary — they
  match the role-inference heuristic `file_triage` uses (see `agentlib/nodes_iterative.py`)
  to decide processing order in the iterative-synthesis topology. The same directory
  works unmodified as `--input` for either example topology.
- Its total size is deliberately far below any model's context window, so
  `chunk_split` always produces exactly one chunk — the non-chunked
  `summary_out_LLM`/`metadata_out_LLM` pair runs, not the fan-out path. To exercise
  chunking, supply a larger input file exceeding the resolved model's context window.
- `read_file` parses CSVs via `pandas` into a list of row-dicts (not a flat key/value
  mapping) — this is existing behavior in `agentlib/file_io.py`, not specific to this
  example, but relevant when interpreting a generated summary of `metadata.csv`.
  **You need to mind the consequences of in memory data representations for operations with LLMs**.


# System Design Principles
Two preliminaries: the system uses LangGraph as its core library, which uses the common graph related concepts of nodes and edges. Programmatically, nodes are nothing else but functions, and edges are nothing else but control flow. Importantly, control flow and data flow are not necessarily the same, i.e. the execution order of nodes is not tied to the data processed. Instead, all nodes write and read data from a general space, rather than a direct communication between them happening. This is known as a [Blackboard System](https://en.wikipedia.org/wiki/Blackboard_system), and the state of the graph represents this shared space.

### The `results` log
The implementation is backed by LangGraph, which means the system is stateful, and represented as a graph. In LangGraph, two ways of communication between nodes exist, but the primary concept is to use the state of the graph as a communication channel. Because it is just a dictionary (e.g. TypedDict or dataclass), nodes can be assigned communication channels to read from and write into beforehand.

It is easy to conceptualize these communication channels as subspaces on the shared blackboard, which nodes attend to, rather than them attending to the overall blackboard. A true, mechanicmal separation into subspaces would induce a lot of issues, specifically for tracking and aggregation. Moreoever, a fully flexible topology specification would come at increasing costs of maintenance because each node would require communication channels to be explicitly defined in the underlying python code. An automatic solution can be conceptualized, but could not solve tracking and aggregation complexities.

Therefore, rather than enforcing a mechanical separation of communication channels, all nodes communicate through the same `results` channel. Specifically, they write into this channel their own name, the results of their operation, and potential errors that occured. Essentially, this is a logical partioning pattern instance, where the partioning is name-based. Dedicated communication relations between nodes have to be defined by specifying read operations for a node in relation to this channel.

A node (or condition) can declare a `reads` list directly in the topology file, naming exactly which other nodes' output it may see, and how much history (`"last"` vs. `"all"`, for nodes that run more than once, see chunk fan-out or the iterative loop). The graph runtime filters what a node actually sees before its own logic ever runs, so this is enforced structurally, rather than a naming convention a node follows. Leaving `reads` unset means is equivalent to `all`, and the right default for e.g. an aggregator that must survey everything.

A node's own output is written under its own name by default, but it can declare `writes` to log itself under a different (or multiple) name(s) instead. This lets two structurally different branches (e.g. a direct LLM call vs. an entire iterative synthesis loop) both present themselves identically to a shared downstream consumer.

## Configuration vs. Registration vs. Runtime Information
Regarding the information the user must specify for this systems, there are three distinct levels of it. The first is configuration information. This belongs to the topology of the graph, as well as the initalization parameters of functions or models. The second type of information is relevant when the registration of nodes for the graph happens, which is at import time of modules that use one of the `registration decorators`. Before registration, validation of the configuration information occurs, which is relevant to prevent topology errors from propagating and potentially being masked as other errors. Finally, the `state` object holds all information relevant at runtime, such as input files for the graph, or information gathered during runtime.


### Configuration vs. registration vs. runtime information

There are three distinct kinds of information in this system, and keeping them
separate is its central design commitment:

- **Configuration** — a topology file defines which nodes/conditions exist, what type each
  is an instance of, what parameters it's built with, and how edges connect them. This
  is the *central* place node behavior is defined. The configurable mentioned above is meant to allow 
  explicitly-tracked per-invocation overrides, not a definition of the topology.
- **Registration** — which node/condition *types* exist at all in a given process, a
  side effect of importing the modules that define them (via `register_node_type` /
  `register_node` / `register_condition_type` / `register_condition`). A topology
  referencing an unregistered type fails loudly at build time. Nodes and functions are strictly separated by the registration process, which wraps functions into nodes. Being a node means:
    - that you are registered in a node registry
    - that you follow the node contract
    - that you are uniformly tracked in MLflow
- **Runtime state** — the graph's actual `GraphState`: strictly the data flowing
  through a run (the input file list, the shared `results` log, and a couple of small
  per-branch markers like `chunk_index`). No configuration, no model choice, no prompt
  text lives here — if it can be decided before the run starts, it belongs in the
  topology file (or, for the narrow case of invocation-time overrides, in
  `config["configurable"]`), never in state.

A topology is validated in two separate passes: 
    - **structural** validation (are names unique, do edges reference declared nodes, does every node have an outgoing edge, are reads pointing at real write-targets) requires no registered types at all and is entirely self-contained within the file; 
    - **registry** validation (do the referenced `type`s actually exist) runs afterward, once node-defining modules have been imported.


### MLflow tracking

There is one MLflow run per invocation of the graph started outside the graph itself (see `agentic_dd.main.run_experiment`). Its run ID is threaded through every node via `config["configurable"]["mlflow_run_id"]`.Nodes reattach to the existing one by ID if none is active in their current thread (relevant for concurrent chunk fan-out branches), and open per-node spans within it for tracing. Because `mlflow_run_id` is not part of the data that the graph produces or consumes, but rather a structural aspect, answering *how* a set of input parameteres was mapped onto the output, it lives in `config`, and not in `GraphState`.


# Subgraph limitations
Above it was mentioned that LangGraph does not allow changes to a graph after compile time. This has a further consequence: a subgraph cannot be made to run independently of its parent graph. Consequently, if a subgraph does not terminate, it can block the parent graph, or if it returns metrics incompletely, e.g. due to an API timeout as a consequence of an infinite LLM loop, the metrics for the parent graph will be necessarily incomplete, and therefore comparability might be impeded.

## Project layout

```
src/agentic_dd/
  agentlib/       node/condition implementations, the node/condition factories,
                  chunking, aggregation, prompts, tool/node definition
  topologies/     topology schema (NodeSpec/ConditionSpec/EdgeSpec/TopologySpec),
                  loading + validation, the condition registry
  extraction/     read-only markdown/JSON/CSV parsing utilities, consumed both by
                  node types (agentlib.nodes_branch) and by agent-facing tools
  logging/        shared loggers
  main.py         CLI entry point: load topology -> build graph -> run -> track in MLflow
tests/            topology-compile tests covering registered node/condition type
sample_data/      minimal example input, see "Sample data" above
```

## Extending the system

To add a new node, define:
    - `(name, params) -> compute_fn` and register it with `@register_node_type("my_type")` for node types,
    - `(name, params) -> compute_fn` and register it with `@register_node("my_name")` for parameter-free, bespoke node,

in a module imported by `agentic_dd/agentlib/__init__.py` Conditions follow the exact same pattern via `register_condition_type` / `register_condition` in `agentic_dd/topologies/conditions.py`. 
