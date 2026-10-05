from __future__ import annotations
import operator
from typing import TypedDict, Annotated, List, Dict, Any, Optional


class GraphState(TypedDict, total=False):
    input: List[str]                                          # list of filepaths for the topology driven run
    task: Optional[str]                                        # set only inside a spawned worker's Send branch
    results: Annotated[List[Dict[str, Any]], operator.add]     # shared, append-only node output log
    chunk_index: Optional[int]                                 # set only inside a chunk-fanout Send branch