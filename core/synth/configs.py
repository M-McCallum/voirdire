"""Bot configurations for synth: the baseline and the planted-bad variants from S9.

Each planted-bad configuration breaks one thing on purpose so the labelled set has real
failures for the judges to be measured on. Rows written under a configuration carry its
name in the `config` field of data/conversations.jsonl.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.bot import build_graph
from core.bot.corpus import CORPUS_DIR, Retriever


@dataclass(frozen=True)
class BotConfig:
    name: str
    prompt_version: str = "v1"
    exclude_docs: frozenset[str] = frozenset()
    note: str = ""


BASELINE = BotConfig("baseline", note="prompts/v1.md, full index")
PLANTED_BAD = (
    BotConfig("prompt_v2", prompt_version="v2", note="prompts/v2.md, dosing prohibition weakened"),
    BotConfig(
        "missing_crisis_doc",
        exclude_docs=frozenset({"crisis"}),
        note="index without data/corpus/crisis.md",
    ),
)
CONFIGS = {c.name: c for c in (BASELINE, *PLANTED_BAD)}


def retriever_for(config: BotConfig, corpus_dir=CORPUS_DIR) -> Retriever:
    full = Retriever.from_dir(corpus_dir)
    docs = [d for d in full.docs if d.name not in config.exclude_docs]
    return Retriever(docs) if docs != full.docs else full


def graph_for(config: BotConfig, model, corpus_dir=CORPUS_DIR):
    return build_graph(model, retriever_for(config, corpus_dir), config.prompt_version)
