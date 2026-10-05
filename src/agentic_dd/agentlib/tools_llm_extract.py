# NOTE TODO This file is outdated, and was not yet refactored/removed @deprecated
'''

from __future__ import annotations
from langchain_core.tools import tool
from agentlib.file_io import read_one_fileg
from agentlib.llm import ModelParams, get_model_for
from agentlib.prompts import PromptSpec, compose_messages
from agentlib.tools import register_tool

#NOTE: Simplification for now, needs to be removed if experiments/system instances are run in parallel 
_EXTRACT_MODEL_PARAMS = ModelParams()

def set_llm_extract_model(params: ModelParams) -> None:
    global _EXTRACT_MODEL_PARAMS
    _EXTRACT_MODEL_PARAMS = params

@tool
def llm_extract(path: str) -> str:
    Uses an LLM to extract key structured facts (topic, scope, source/type,
    entities, dates) from a file. Prefer this over reading the whole file
    directly when you want a condensed, fact-focused extraction rather than
    raw content.
    content = read_one_file(path)
    llm = get_model_for(_EXTRACT_MODEL_PARAMS)
    resp = llm.invoke(compose_messages(
        PromptSpec(template_name="llm_extractor_default"),  # same prompt as the 9b node
        {"input": str(content)},
    ))
    return resp.content

register_tool("llm_extract", llm_extract)
'''