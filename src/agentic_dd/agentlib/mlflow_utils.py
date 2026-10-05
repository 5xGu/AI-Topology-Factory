'''
Shared MLflow helpers used across node/aggregator implementations.
'''
from __future__ import annotations
from contextlib import contextmanager, nullcontext
from typing import Optional
import mlflow

from agentic_dd.logging import ai_default_logger

@contextmanager
def ensure_active_run(run_id: Optional[str]):
    ''' Reattaches to the parent MLflow run by run_id if no run is active in
    the current thread/process context; else nothing '''
    ctx = mlflow.start_run(run_id=run_id) if run_id and mlflow.active_run() is None else nullcontext()
    with ctx:
        yield


def tag_resolved_model(node_name: str, provider: str, model: str) -> None:
    mlflow.set_tag(f"resolved_model::{node_name}", f"{provider}:{model}")
    ai_default_logger.info(f"[{node_name}] resolved model: provider={provider} model={model}")