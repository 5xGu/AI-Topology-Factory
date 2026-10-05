from __future__ import annotations
from typing import List, Dict, Any, Optional, Annotated, TypedDict
from langchain_core.messages import SystemMessage, AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from agentic_dd.agentlib.mlflow_utils import tag_resolved_model
from agentic_dd.agentlib.llm import ModelParams, get_model_for
from agentic_dd.agentlib.prompts import PromptSpec, build_system_message


class SubagentState(TypedDict, total=False):
    ''' 
    Local, minimal state for the internal tool-calling loop.
    It is distinct from AgentState and depth/max_depth are pure Python closure values
    (see delegation.build_delegate_tool), never graph state, and messages
    have no meaning outside this subgraph. 
    '''
    messages: Annotated[List[AnyMessage], add_messages]
    input: Any


def build_agent_subgraph(tools: List[Any], prompt_spec: PromptSpec, name: str, model_params: Optional[ModelParams] = None):
    params = model_params or ModelParams()
    tag_resolved_model(name, params.provider, params.model)
    llm = get_model_for(params).bind_tools(tools)

    def call_llm(state: SubagentState) -> Dict[str, Any]:
        msgs = state["messages"]
        if not msgs or not isinstance(msgs[0], SystemMessage):
            msgs = [build_system_message(prompt_spec, {})] + list(msgs) #TODO this might be faulty due to how system message deals with its context
        resp = llm.invoke(msgs)
        return {"messages": [resp]}

    builder = StateGraph(SubagentState)
    builder.add_node("llm", call_llm)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "llm")
    builder.add_conditional_edges("llm", tools_condition, {"tools": "tools", "__end__": END})
    builder.add_edge("tools", "llm")
    return builder.compile(name=name)