"""The chatbot graph. Every node is a child span under the per-run root span."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from opentelemetry import trace

from core.bot.corpus import Doc, Retriever
from core.bot.guards import CRISIS_RESPONSE, GUARD_REFUSAL, crisis_match, medication_match
from core.telemetry import tracer

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
DEFAULT_MODEL = "claude-haiku-5-5"


class BotState(TypedDict, total=False):
    message: str
    docs: list[Doc]
    blocked_by: Literal["guard", "crisis"] | None
    answer: str


def load_prompt(version: str) -> str:
    return (PROMPTS_DIR / f"{version}.md").read_text()


def model_name() -> str:
    return os.environ.get("VD_MODEL", DEFAULT_MODEL)


def default_chat_model() -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=model_name(), max_tokens=1024)


def _node(name: str, fn: Callable[[BotState], BotState]) -> Callable[[BotState], BotState]:
    def wrapped(state: BotState) -> BotState:
        with tracer().start_as_current_span(f"bot.{name}"):
            return fn(state)

    wrapped.__name__ = name
    return wrapped


def build_graph(model: BaseChatModel, retriever: Retriever, prompt_version: str = "v1"):
    system_prompt = load_prompt(prompt_version)

    def retrieve(state: BotState) -> BotState:
        docs = retriever.search(state["message"])
        trace.get_current_span().set_attribute("retrieve.docs", [d.name for d in docs])
        return {"docs": docs}

    def crisis(state: BotState) -> BotState:
        hit = crisis_match(state["message"])
        trace.get_current_span().set_attribute("crisis.matched", hit)
        if hit:
            return {"blocked_by": "crisis", "answer": CRISIS_RESPONSE}
        return {}

    def guard(state: BotState) -> BotState:
        hit = medication_match(state["message"])
        trace.get_current_span().set_attribute("guard.matched", hit)
        if hit:
            return {"blocked_by": "guard", "answer": GUARD_REFUSAL}
        return {}

    def generate(state: BotState) -> BotState:
        context = "\n\n".join(f"## {d.name}\n{d.text}" for d in state.get("docs", []))
        span = trace.get_current_span()
        span.set_attribute("gen_ai.request.model", model_name())
        span.set_attribute("gen_ai.prompt.version", prompt_version)
        response = model.invoke(
            [SystemMessage(content=system_prompt + context), HumanMessage(content=state["message"])]
        )
        usage = getattr(response, "usage_metadata", None) or {}
        if usage:
            span.set_attribute("gen_ai.usage.input_tokens", usage.get("input_tokens", 0))
            span.set_attribute("gen_ai.usage.output_tokens", usage.get("output_tokens", 0))
        return {"answer": _text(response.content)}

    def after_check(state: BotState) -> str:
        return "blocked" if state.get("blocked_by") else "continue"

    g = StateGraph(BotState)
    g.add_node("retrieve", _node("retrieve", retrieve))
    g.add_node("crisis", _node("crisis", crisis))
    g.add_node("guard", _node("guard", guard))
    g.add_node("generate", _node("generate", generate))
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "crisis")
    g.add_conditional_edges("crisis", after_check, {"blocked": END, "continue": "guard"})
    g.add_conditional_edges("guard", after_check, {"blocked": END, "continue": "generate"})
    g.add_edge("generate", END)
    return g.compile()


def _text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return str(content)


def run(message: str, graph, prompt_version: str = "v1") -> BotState:
    """One chatbot turn as a root span with prompt_version and model attributes."""
    with tracer().start_as_current_span("bot.run") as span:
        span.set_attribute("prompt_version", prompt_version)
        span.set_attribute("model", model_name())
        result = graph.invoke({"message": message})
        span.set_attribute("blocked_by", result.get("blocked_by") or "")
        return result
