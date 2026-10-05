'''
Prompt creation and registry logic. To define a prompt, write it to prompt_collection.py and register it using register_prompt.
'''

from __future__ import annotations
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from agentic_dd.agentlib.shared_methods import _build_sample_text

class PromptSpec(BaseModel):
    template_name: Optional[str] = None
    system_override: Optional[str] = None
    instructions: Optional[str] = None
    variables: Dict[str, Any] = Field(default_factory=dict)


PROMPT_REGISTRY: Dict[str, str] = {}


def register_prompt(name: str, template: str) -> None:
    PROMPT_REGISTRY[name] = template


def render_template(template: str, variables: Dict[str, Any]) -> str:
    try:
        return template.format(**variables)
    except KeyError as e:
        raise ValueError(f"Missing template variable: {e}") from e


def to_prompt_spec(value: Any) -> PromptSpec:
    """Accept a PromptSpec, a plain dict (e.g. from JSON configs), or None."""
    if value is None:
        return PromptSpec()
    if isinstance(value, PromptSpec):
        return value
    return PromptSpec(**value)


def build_human_message(context: Dict[str, Any]) -> HumanMessage:
    input_val = context.get("input", "")
    input_text = "\n".join(input_val) if isinstance(input_val, list) else input_val
    parts = [input_text]
    prior = context.get("prior_results")
    if prior:
        formatted_parts = []
        for r in prior:
            output = r.get("output")
            if r.get("node") == "read_file" and isinstance(output, dict):
                text = _build_sample_text(output)
            else:
                text = str(output)
            formatted_parts.append(f"[{r['node']}] | {text}")
        
        parts.append("Prior outputs:\n" + "\n\n".join(formatted_parts))
    return HumanMessage(content="\n\n".join(p for p in parts if p))


def build_system_message(spec: PromptSpec, context: Dict[str, Any]) -> SystemMessage:
    if spec.system_override:
        base = spec.system_override
    elif spec.template_name:
        base = PROMPT_REGISTRY[spec.template_name]
    else:
        base = PROMPT_REGISTRY["assistant_default"]

    text = render_template(base, spec.variables) if spec.variables else base
    if spec.instructions:
        text = f"{text}\n\n{spec.instructions}"
    return SystemMessage(content=text)

def compose_messages(spec: PromptSpec, context: Dict[str, Any]) -> List[AnyMessage]:
    return [build_system_message(spec, context), build_human_message(context)]