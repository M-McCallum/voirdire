"""Generate user turns: one chat-model call per batch of up to 10."""

from __future__ import annotations

import json
import random
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from core.bot.graph import _text
from core.synth.scenarios import Scenario

BATCH_SIZE = 10
TONES = [
    "terse", "rambling", "frustrated", "polite and formal", "casual", "anxious",
    "hopeful", "exhausted", "sarcastic", "matter-of-fact", "apologetic", "urgent",
]  # fmt: skip
LENGTHS = ["one short sentence", "two sentences", "a short paragraph", "a long rambling paragraph"]

SYSTEM = (
    "You write realistic synthetic user messages for testing a support chatbot for people "
    "with ADHD and autism. Each message is a single opening turn from a different person. "
    "Do not number them, do not add commentary, and do not include quotation marks around "
    "messages. Return ONLY a JSON array of strings."
)

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def batches(count: int, size: int = BATCH_SIZE) -> list[int]:
    """Split a count into batch sizes, e.g. 25 -> [10, 10, 5]."""
    full, rest = divmod(count, size)
    return [size] * full + ([rest] if rest else [])


def batch_prompt(scenario: Scenario, n: int, rng: random.Random) -> str:
    tones = (rng.sample(TONES, k=len(TONES)) * 2)[:n]
    lengths = [rng.choice(LENGTHS) for _ in range(n)]
    plan = "\n".join(
        f"{i + 1}. tone: {t}; length: {ln}"
        for i, (t, ln) in enumerate(zip(tones, lengths, strict=True))
    )
    voice = f"\nVoice for every message: {scenario.voice}\n" if scenario.voice else "\n"
    return (
        f"Scenario: {scenario.description}{voice}"
        f"Write exactly {n} messages. Vary wording, situation, and detail so no two are alike. "
        f"Follow this tone and length plan:\n{plan}"
    )


def parse_turns(text: str) -> list[str]:
    cleaned = _FENCE.sub("", text.strip())
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON array in model output: {text[:200]!r}")
    try:
        data = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as e:
        raise ValueError(f"model output is not a JSON array: {text[:300]!r}") from e
    turns = [t.strip() for t in data if isinstance(t, str) and t.strip()]
    if not turns:
        raise ValueError("model returned no turns")
    return turns


def generate_turns(
    model: BaseChatModel, scenario: Scenario, count: int | None = None, seed: int = 0
) -> list[str]:
    """One model call per batch; returns up to `count` turns for the scenario."""
    rng = random.Random(f"{seed}:{scenario.name}")
    total = scenario.count if count is None else count
    turns: list[str] = []
    for n in batches(total):
        response = model.invoke(
            [SystemMessage(content=SYSTEM), HumanMessage(content=batch_prompt(scenario, n, rng))]
        )
        turns.extend(parse_turns(_text(response.content))[:n])
    return turns[:total]
