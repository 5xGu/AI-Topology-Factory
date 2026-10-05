'''
Topology schema: 
    - NodeSpec 
    - ConditionSpec 
    - EdgeSpec 
    - TopologySpec, 

plus structural validation (self-contained within a topology file) and registry
validation (checked against whatever node/condition types have actually
been registered in this process).

A topology file fully determines:
    - which node/condition TYPES are instantiated under which NAMES with which PARAMS, 
    - what each may READ from the shared results log, 
    - what name(s) each node WRITES under, 
    - and how control flows between them. 
    
There is no other source of node/condition behavior configuration outside this file plus per-invocation
config["configurable"] overrides, see agentic_dd.agentlib.nodes_factory.
'''

from __future__ import annotations
import json
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError

from agentic_dd.agentlib.nodes_factory import ReadSpec, NODE_TYPE_REGISTRY
from agentic_dd.topologies.conditions import CONDITION_TYPE_REGISTRY
from agentic_dd.logging import topology_logger

END = "END"



########## schema

class NodeSpec(BaseModel):
    model_config = {"extra": "forbid"}

    name: str
    type: str
    params: Dict[str, Any] = Field(default_factory=dict)
    reads: Optional[List[ReadSpec]] = None    # None = unrestricted (sees full results log)
    writes: Optional[List[str]] = None        # None = writes under `name` only


class ConditionSpec(BaseModel):
    model_config = {"extra": "forbid"}

    name: str
    type: str
    params: Dict[str, Any] = Field(default_factory=dict)
    reads: Optional[List[ReadSpec]] = None    # None = unrestricted; conditions never write


class EdgeSpec(BaseModel):
    ''' 
    - Exactly one of `target`/`condition` must be set. 
    - A plain edge (`target` set) always runs; 
    - Multiple plain EdgeSpecs sharing the same `source` are all taken unconditionally (static fan-out). 
    - A conditional edge (`condition` set) dispatches to exactly one or, via a Send-returning / list-returning condition, several, of its own
    `condition_targets` at runtime;
    - Only one conditional EdgeSpec is permitted per source, and a source may not mix conditional and plain
    edges. 
    - `target`/`condition_targets` entries may be "END" to terminate. 
    '''
    model_config = {"extra": "forbid"}

    source: str
    target: Optional[str] = None
    condition: Optional[str] = None
    condition_targets: Optional[List[str]] = None

    @model_validator(mode="after")
    def _check_shape(self) -> "EdgeSpec":
        has_target = self.target is not None
        has_condition = self.condition is not None
        if has_target == has_condition:
            raise ValueError(
                f"EdgeSpec for source {self.source!r} must set exactly one of `target` or "
                f"`condition` (got target={self.target!r}, condition={self.condition!r})"
            )
        if has_condition and not self.condition_targets:
            raise ValueError(f"EdgeSpec for source {self.source!r} sets `condition` but no `condition_targets`")
        if has_target and self.condition_targets is not None:
            raise ValueError(f"EdgeSpec for source {self.source!r} sets `target` and must not also set `condition_targets`")
        return self


class TopologySpec(BaseModel):
    model_config = {"extra": "forbid"}

    entry: str
    nodes: List[NodeSpec]
    conditions: List[ConditionSpec] = Field(default_factory=list)
    edges: List[EdgeSpec]

    ### --- bare-string shorthand: "name" == {"name": "name", "type": "name"} --- ###

    @field_validator("nodes", mode="before")
    @classmethod
    def _normalize_nodes(cls, v: Any) -> Any:
        if not isinstance(v, list):
            return v
        return [{"name": item, "type": item} if isinstance(item, str) else item for item in v]

    @field_validator("conditions", mode="before")
    @classmethod
    def _normalize_conditions(cls, v: Any) -> Any:
        if v is None:
            return []
        if not isinstance(v, list):
            return v
        return [{"name": item, "type": item} if isinstance(item, str) else item for item in v]

    ### --- structural (closed-world) validation --- ###

    @model_validator(mode="after")
    def _check_structure(self) -> "TopologySpec":
        if not self.nodes:
            raise ValueError("Topology must declare at least one node")

        node_names = [n.name for n in self.nodes]
        if END in node_names:
            raise ValueError(f"'{END}' is a reserved sentinel and may not be used as a node name")
        _reject_duplicates("node", node_names)
        node_name_set = set(node_names)

        condition_names = [c.name for c in self.conditions]
        _reject_duplicates("condition", condition_names)
        condition_name_set = set(condition_names)

        if self.entry not in node_name_set:
            raise ValueError(f"entry '{self.entry}' is not a declared node")

        # write-target map: node name -> effective log name(s) it writes under
        target_owners: Dict[str, List[str]] = {}
        for node in self.nodes:
            targets = node.writes if node.writes is not None else [node.name]
            if len(set(targets)) != len(targets):
                raise ValueError(f"Node '{node.name}' has duplicate entries within its own `writes` list: {targets}")
            for t in targets:
                target_owners.setdefault(t, []).append(node.name)

        for target, owners in target_owners.items():
            if len(owners) > 1:
                topology_logger.warning(
                    f"Multiple nodes write to the same target name '{target}': {owners}. "
                    f"Only safe if these nodes are never both reachable within a single run "
                    f", which is not further validated here."
                )

        valid_log_names = set(target_owners.keys())

        def _check_reads(owner_kind: str, owner_name: str, reads: Optional[List[ReadSpec]]) -> None:
            if reads is None:
                return
            for r in reads:
                if r.node != "*" and r.node not in valid_log_names:
                    raise ValueError(
                        f"{owner_kind} '{owner_name}' declares a read of unknown node/write-target '{r.node}'"
                    )

        for node in self.nodes:
            _check_reads("Node", node.name, node.reads)
        for cond in self.conditions:
            _check_reads("Condition", cond.name, cond.reads)

        # edge referential integrity
        by_source: Dict[str, List[EdgeSpec]] = {}
        for e in self.edges:
            by_source.setdefault(e.source, []).append(e)

            if e.source not in node_name_set:
                raise ValueError(f"Edge source '{e.source}' is not a declared node")

            if e.target is not None and e.target != END and e.target not in node_name_set:
                raise ValueError(f"Edge target '{e.target}' (from source '{e.source}') is not a declared node or '{END}'")

            if e.condition is not None:
                if e.condition not in condition_name_set:
                    raise ValueError(f"Edge condition '{e.condition}' (from source '{e.source}') is not a declared condition")
                for t in e.condition_targets or []:
                    if t != END and t not in node_name_set:
                        raise ValueError(
                            f"condition_targets entry '{t}' (from source '{e.source}', "
                            f"condition '{e.condition}') is not a declared node or '{END}'"
                        )

        # per-source edge-kind exclusivity: no source may mix conditional and plain edges,
        # and at most one conditional edge per source (ambiguous otherwise)
        for source, edges in by_source.items():
            cond_edges = [e for e in edges if e.condition is not None]
            plain_edges = [e for e in edges if e.target is not None]
            if cond_edges and plain_edges:
                raise ValueError(
                    f"Source '{source}' mixes conditional and plain edges; not supported -- "
                    f"express all branches via condition_targets on a single conditional edge"
                )
            if len(cond_edges) > 1:
                raise ValueError(f"Source '{source}' declares multiple conditional edges; only one is supported per source")

        # every node must explicitly route somewhere, including "END"; no implicit termination
        missing_outgoing = node_name_set - set(by_source.keys())
        if missing_outgoing:
            raise ValueError(
                f"The following nodes have no outgoing edge declared (every node must "
                f"explicitly route somewhere, including '{END}'): {sorted(missing_outgoing)}"
            )

        return self


def _reject_duplicates(kind: str, names: List[str]) -> None:
    seen = set()
    dupes = set()
    for n in names:
        if n in seen:
            dupes.add(n)
        seen.add(n)
    if dupes:
        raise ValueError(f"Duplicate {kind} name(s) declared: {sorted(dupes)}")


########## loading / validation entry points

def load_topology(fp: str) -> Dict[str, Any]:
    ''' 
    Parses a topology JSON file into a raw dict, but does not performs validation.
    '''
    try:
        with open(fp) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        topology_logger.error(f"Error occurred loading topology json file '{fp}': {e}")
        raise


def validate_loaded_topology(topology_dict: Dict[str, Any]) -> TopologySpec:
    ''' 
    Structural validation only: 
        - shape, 
        - uniqueness, 
        - referential integrity between nodes/conditions/edges, 
        - reads pointing at declared write-targets, 
        - every node having an outgoing edge. 
    '''
    try:
        return TopologySpec.model_validate(topology_dict)
    except ValidationError as e:
        topology_logger.error(f"Topology failed structural validation: {e}")
        raise


def validate_registries(spec: TopologySpec) -> None:
    ''' 
    Checks every node/condition `type` referenced by an already
    structurally-valid TopologySpec against whatever is currently
    registered in NODE_TYPE_REGISTRY / CONDITION_TYPE_REGISTRY. Must be
    called after all node/condition-defining modules have been imported
    (see agentic_dd.agentlib.graph_build, which owns that import list). 
    '''
    unknown_node_types = {n.type for n in spec.nodes} - set(NODE_TYPE_REGISTRY.keys())
    if unknown_node_types:
        message = (
            f"Topology references unregistered node type(s): {sorted(unknown_node_types)}. "
            f"Ensure the module(s) defining these node types have been imported."
        )
        topology_logger.error(message)
        raise ValueError(message)

    unknown_condition_types = {c.type for c in spec.conditions} - set(CONDITION_TYPE_REGISTRY.keys())
    if unknown_condition_types:
        message = (
            f"Topology references unregistered condition type(s): {sorted(unknown_condition_types)}. "
            f"Ensure the module(s) defining these condition types have been imported."
        )
        topology_logger.error(message)
        raise ValueError(message)