from __future__ import annotations
from typing import Optional
import mlflow
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

from agentic_dd.agentlib.agent_subgraph import build_agent_subgraph
from agentic_dd.agentlib.prompts import PromptSpec
from agentic_dd.agentlib.llm import ModelParams


def build_delegate_tool(depth: int, max_depth: int, model_params: Optional[ModelParams] = None):
    ''' 
    depth/max_depth are plain Python closure values scoped to this
    specific delegate chain and are never contained in the graph state (see agent_subgraph.SubagentState). 
    '''
    if depth >= max_depth:
        return None

    @tool
    def delegate_to_subagent(task: str) -> str:
        """Delegate a task to a fresh sub-agent."""
        with mlflow.start_span(name="delegate", span_type="AGENT") as span:
            span.set_inputs({"from_depth": depth, "to_depth": depth + 1, "task": task})
            sub_delegate = build_delegate_tool(depth + 1, max_depth, model_params=model_params)
            sub_tools = [sub_delegate] if sub_delegate else []
            sub_compiled = build_agent_subgraph(
                tools=sub_tools, prompt_spec=PromptSpec(template_name="subagent_default"),
                name=f"subagent_depth_{depth + 1}",
                model_params=model_params,
            )
            result = sub_compiled.invoke({
                "messages": [HumanMessage(content=task)],
                "input": task,
            })
            output = result["messages"][-1].content
            span.set_outputs({"output": output[:2000]})
        return output

    return delegate_to_subagent