'''
Provides methods to traverse a .json structure through a schema.
Implements exploration through a generator, to avoid issues for large files.
Use by generating a schema, and querying it for entries. Base further exploration
on these by storing them in a stable iterator.
'''


import json
import os
from pathlib import Path
from typing import Dict, Tuple, Any, Iterable, List
from dataclasses import dataclass

# TODO generator? I am assuming this wont be called often consequently on the same file
# If the .json is too large, we will run into issues w/o generator?
@dataclass
class SchemaEntry:
    ''' Schema dataclass for json files representing their structure '''
    path: tuple[str | int, ...]
    parent_path: tuple[str | int, ...]
    lvl: int
    type: str


def load_json(fp: str | Path) -> Any:
    ''' load a json file '''
    with open(fp) as data:    
        jf = json.load(data)
    return jf

def schema(obj: Any, path=()) -> Iterable[SchemaEntry]:
    ''' yield the schema of a .json file through a generator. Use by creating a list of schema
    and querying the original schema only at the end '''
    if isinstance(obj, dict):
        for key, value in obj.items():
            current = path + (key,)
            yield SchemaEntry(
                path=current,
                parent_path=current[:-1],
                lvl=len(current)-1,
                type=type(value).__name__
            )
            yield from schema(value, current)

    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            current = path + (i,)

            if isinstance(value, (dict, list)):
                yield SchemaEntry(
                    path=current,
                    parent_path=current[:-1],
                    lvl=len(current)-1,
                    type=type(value).__name__
                )
                yield from schema(value, current)

def get_by_path(data: Any, path: tuple[str | int, ...]) -> SchemaEntry:
    ''' retrieve an item based on the schema generator key mapping '''
    current = data
    for part in path:
        current = current[part]
    return current

def path_to_string(path: tuple[str | int, ...]) -> str:
    ''' write a schema path as a string '''
    return ".".join(map(str, path))



### Query .json based on schema representation
def children(entries: Iterable[SchemaEntry], parent_path) -> List[SchemaEntry]:
    """Return the direct children of a path."""
    return [
        e for e in entries
        if e.parent_path == parent_path
    ]


def descendants(entries: Iterable[SchemaEntry], parent_path) -> List[SchemaEntry]:
    """Return all descendants of a path."""
    return [
        e for e in entries
        if e.path[:len(parent_path)] == parent_path
    ]


def of_type(entries: Iterable[SchemaEntry], typ: str) -> List[SchemaEntry]:
    """Return all entries of a given type."""
    return [
        e for e in entries
        if e.type == typ
    ]


def level_keys(entries: Iterable[SchemaEntry], lvl: int) -> List[tuple[str | int, ...]]:
    ''' retrieves paths for entities on a given level '''
    return list(
        dict.fromkeys(
            [e.path[:lvl+1] for e in entries if e.lvl == lvl]
        )
    ) 


def is_docling_json(json_dict: dict) -> bool:
    ''' Helper method to associate docling returned jsons with the parsed file '''
    magic_keys = {'filename', 'images', 'markdown'}
    return set(json_dict.keys()) == magic_keys
