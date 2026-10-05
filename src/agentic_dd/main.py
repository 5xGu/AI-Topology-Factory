'''
Minimal CLI entry point: load a topology file, build the graph, run it once
against a set of input files, and track the run in MLflow. Meant as an
orientation example for the full lifecycle a user's own experiment scripts
would wire up:

    topology.json --> build_graph() --> compiled.invoke(state, config)

MLflow tracking flows through config["configurable"]["mlflow_run_id"] --
exactly what every node reads via build_node/build_plain_llm_compute/etc.
to reattach to the run started here, including from inside concurrent
chunk-fanout / spawn branches (see agentlib.mlflow_utils.ensure_active_run).
Nodes must never open their own top-level mlflow.start_run() -- reattachment
by run_id, done once here, is the entire mechanism.

Usage:
    uv run python -m agentic_dd.main \
        --topology path/to/topology.json \
        --input path/to/file1.json path/to/a_directory/ \
        --experiment-name my-experiment \
        --configurable path/to/configurable_overrides.json
'''
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow

from agentic_dd.agentlib.graph import build_graph
from agentic_dd.agentlib.nodes_factory import find_node_output
from agentic_dd.agentlib.file_io import SUPPORTED_SUFFIXES


def resolve_input_paths(raw_inputs: List[str]) -> List[str]:
    ''' Accepts a mix of individual file paths and directories; directories
    are expanded (non-recursively) to every contained file with a supported
    suffix (agentlib.file_io.SUPPORTED_SUFFIXES). '''
    resolved: List[str] = []
    for raw in raw_inputs:
        p = Path(raw)
        if p.is_dir():
            found = sorted(
                str(f) for f in p.rglob("*")
                if f.is_file() and f.suffix.lower() in SUPPORTED_SUFFIXES
            )
            if not found:
                print(f"[main] warning: no supported files ({sorted(SUPPORTED_SUFFIXES)}) "
                      f"found in directory {raw!r}", file=sys.stderr)
            resolved.extend(found)
        elif p.is_file():
            resolved.append(str(p))
        else:
            raise FileNotFoundError(f"Input path does not exist: {raw!r}")
    if not resolved:
        raise ValueError("No input files resolved from --input arguments")
    return resolved


def load_configurable_overrides(path: Optional[str]) -> Dict[str, Any]:
    ''' Loads the free-form config["configurable"] dict (model_params /
    prompt_overrides / max_depth overrides, keyed by node name) from a JSON
    file. mlflow_run_id is injected separately by run_experiment, never
    read from this file. '''
    if path is None:
        return {}
    with open(path) as f:
        overrides = json.load(f)
    if not isinstance(overrides, dict):
        raise ValueError(f"--configurable file must contain a JSON object, got {type(overrides).__name__}")
    return overrides


def run_experiment(
    topology_path: str,
    input_paths: List[str],
    configurable: Optional[Dict[str, Any]] = None,
    experiment_name: str = "agentic_dd",
    run_name: Optional[str] = None,
    recursion_limit: int = 50,
) -> Dict[str, Any]:
    ''' 
    Builds the graph from `topology_path` and invokes it once against
    `input_paths`, with MLflow tracking. Returns the final AgentState.

    Logs, for reproducibility: 
        - the topology file itself, 
        - the resolved `configurable` dict (which is not visible in the topology file), 
        - the input file list, 
        - the full `results` log, 
        - the aggregator's output if present. 
    '''
    configurable = dict(configurable or {})

    mlflow.set_experiment(experiment_name)
    graph = build_graph(topology_path)

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        configurable["mlflow_run_id"] = run_id

        mlflow.log_param("topology_path", topology_path)
        mlflow.log_param("num_input_files", len(input_paths))
        mlflow.log_artifact(topology_path, artifact_path="topology")
        mlflow.log_dict(configurable, "configurable.json")
        mlflow.log_dict({"input": input_paths}, "input_files.json")

        initial_state = {"input": input_paths}
        invoke_config = {"configurable": configurable, "recursion_limit": recursion_limit}

        try:
            final_state = graph.invoke(initial_state, invoke_config)
        except Exception:
            mlflow.set_tag("run_outcome", "failed")
            raise

        results = final_state.get("results", [])
        mlflow.log_dict({"results": results}, "results.json")

        errors = [r for r in results if r.get("error")]
        mlflow.log_metric("num_node_errors", len(errors))
        for r in errors:
            mlflow.set_tag(f"error::{r['node']}", str(r["error"])[:250])

        aggregator_output = find_node_output(results, "aggregator")
        if aggregator_output is not None:
            mlflow.log_dict(aggregator_output, "aggregator_output.json")

        mlflow.set_tag("run_outcome", "success" if not errors else "completed_with_errors")

    return final_state


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a topology-defined agentic_dd graph with MLflow tracking.")
    parser.add_argument("--topology", required=True, help="Path to a topology JSON file.")
    parser.add_argument("--input", required=True, nargs="+",
                         help="One or more input file paths and/or directories "
                              "(directories are expanded, non-recursively, to supported files within).")
    parser.add_argument("--configurable", default=None,
                         help="Path to a JSON file with invocation-time overrides "
                              "(model_params / prompt_overrides / max_depth), keyed by node name.")
    parser.add_argument("--experiment-name", default="agentic_dd", help="MLflow experiment name.")
    parser.add_argument("--run-name", default=None, help="MLflow run name (optional).")
    parser.add_argument("--recursion-limit", type=int, default=50, help="LangGraph recursion_limit for this invocation.")
    parser.add_argument("--mlflow-tracking-uri", default=None,
                         help="Overrides MLFLOW_TRACKING_URI if set; otherwise MLflow's own default/env resolution applies.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    input_paths = resolve_input_paths(args.input)
    configurable = load_configurable_overrides(args.configurable)

    final_state = run_experiment(
        topology_path=args.topology,
        input_paths=input_paths,
        configurable=configurable,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        recursion_limit=args.recursion_limit,
    )

    results = final_state.get("results", [])
    aggregator_output = find_node_output(results, "aggregator")
    print(json.dumps({
        "aggregator_output": aggregator_output,
        "num_results": len(results),
        "num_errors": len([r for r in results if r.get("error")]),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()