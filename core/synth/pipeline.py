"""Replay turns through the bot, write conversations.jsonl, upload to LangSmith."""

from __future__ import annotations

import json
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from core.bot.graph import model_name, run

CONVERSATIONS_PATH = Path(__file__).resolve().parents[2] / "data" / "conversations.jsonl"
DATASET_NAME = "support-bot-eval"


@dataclass(frozen=True)
class Row:
    input: str
    output: str
    retrieved_doc_ids: list[str]
    scenario: str
    prompt_version: str
    model: str
    trace_id: str
    config: str = "baseline"


def replay(
    turns: Iterable[tuple[str, str]],
    graph,
    prompt_version: str = "v1",
    config: str = "baseline",
    workers: int = 1,
) -> list[Row]:
    """turns is (scenario, user_turn) pairs. Each turn is one bot run and one trace.

    `config` names the bot configuration the rows were produced under (core.synth.configs).
    With workers > 1 the turns run on a thread pool; row order follows the input order.
    """
    turns = list(turns)

    def one(pair: tuple[str, str]) -> Row:
        scenario, turn = pair
        state = run(turn, graph, prompt_version)
        return Row(
            input=turn,
            output=state["answer"],
            retrieved_doc_ids=[d.name for d in state.get("docs", [])],
            scenario=scenario,
            prompt_version=prompt_version,
            model=model_name(),
            trace_id=state["trace_id"],
            config=config,
        )

    if workers <= 1:
        return [one(pair) for pair in turns]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, turns))


def write_rows(rows: list[Row], path: Path = CONVERSATIONS_PATH, append: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if append else "w") as f:
        for row in rows:
            f.write(json.dumps(asdict(row)) + "\n")
    return path


def upload_rows(rows: list[Row], client, dataset_name: str = DATASET_NAME) -> str:
    """Upload rows as LangSmith examples. `client` is a langsmith.Client (or a fake)."""
    if client.has_dataset(dataset_name=dataset_name):
        dataset = client.read_dataset(dataset_name=dataset_name)
    else:
        dataset = client.create_dataset(
            dataset_name=dataset_name, description="Synthetic turns replayed through the bot."
        )
    client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {
                "inputs": {"input": r.input},
                "outputs": {"output": r.output, "retrieved_doc_ids": r.retrieved_doc_ids},
                "metadata": {
                    "scenario": r.scenario,
                    "prompt_version": r.prompt_version,
                    "model": r.model,
                    "trace_id": r.trace_id,
                    "config": r.config,
                },
            }
            for r in rows
        ],
    )
    return str(dataset.id)
