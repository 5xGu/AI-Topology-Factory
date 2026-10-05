from __future__ import annotations
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from pydantic import BaseModel
import os
from dotenv import find_dotenv, load_dotenv
from agentic_dd.agentlib.rate_limit import acquire_gwdg_slot
from typing import Optional

load_dotenv(find_dotenv())

FORCE_LOCAL_LLM = os.getenv("FORCE_LOCAL_LLM", "0") == "1"
FORCE_LOCAL_MODEL = os.getenv("FORCE_LOCAL_MODEL", "qwen2.5:7b")

if FORCE_LOCAL_LLM:
    ai_default_logger.warning(
        f"FORCE_LOCAL_LLM is set: every node's model_params.provider/model will be "
        f"overridden to local:{FORCE_LOCAL_MODEL}, regardless of topology/configurable settings."
    )


class ModelParams(BaseModel):
    model_config = {"extra": "forbid"}

    provider: str = "api"   # "local" | "api"
    model: str = "qwen3.8-27b"
    temperature: float = 0.0


def get_llm(model: str = "qwen3.8-27b", temperature: float = 0.0, timeout: int = 300, max_retries: int = 5) -> ChatOpenAI:
    base_url = os.getenv("BASE_URL")
    if not base_url:
        raise RuntimeError("BASE_URL is not set.")
        
    llm = ChatOpenAI(
        base_url=base_url,
        api_key=os.getenv("API_KEY"),
        model=model,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries
    )
    return _RateLimitedWrapper(llm)


def get_local_llm(model: str = "qwen2.5:7b", temperature: float = 0.0, timeout: int = 240, num_ctx: Optional[int] = None, num_predict=5000) -> ChatOllama:
    return ChatOllama(
        base_url=os.getenv("OLLAMA_BASE_URL"),
        model=model,
        temperature=temperature,
        timeout=timeout,
        num_ctx=num_ctx,
        num_predict=num_predict
    )


def get_model_for(params: ModelParams):
    if FORCE_LOCAL_LLM:
        return get_local_llm(FORCE_LOCAL_MODEL, params.temperature, num_ctx=27000)
    if params.provider == "api":
        return get_llm(params.model, params.temperature)
    return get_local_llm(params.model, params.temperature, num_ctx=27000)


class _RateLimitedWrapper:
    def __init__(self, inner):
        self._inner = inner

    def invoke(self, *args, **kwargs):
        acquire_gwdg_slot()
        return self._inner.invoke(*args, **kwargs)

    def bind_tools(self, *args, **kwargs):
        return _RateLimitedWrapper(self._inner.bind_tools(*args, **kwargs))

    def with_structured_output(self, *args, **kwargs):
        return _RateLimitedWrapper(self._inner.with_structured_output(*args, **kwargs))

    def __getattr__(self, name):
        return getattr(self._inner, name)