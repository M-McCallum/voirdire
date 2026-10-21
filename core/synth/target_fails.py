"""Build a fail-rich labelled set: planted-bad replays, then labelling until the target.

Replays the scenario grid through the bot under each planted-bad configuration
(core.synth.configs.PLANTED_BAD) and the adversarial scenario under every configuration,
appending the rows to data/conversations.jsonl tagged with the configuration. Then the S3
labeller runs on a sample that puts those rows first, until every target rule has at least
`target` human fails. Nothing here writes a label; a person types every one.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from rich.console import Console

from core.label import (
    KEYS,
    LABELS_PATH,
    Ask,
    fail_counts,
    label_rows,
    priority_sample,
    read_jsonl,
)
from core.synth.configs import BASELINE, PLANTED_BAD, BotConfig, graph_for
from core.synth.generator import generate_turns
from core.synth.pipeline import CONVERSATIONS_PATH, Row, replay, write_rows
from core.synth.scenarios import Scenario

ADVERSARIAL = "adversarial_dosing"
TARGET_RULES = ("no_dosing_advice", "crisis_handled")


def is_fail_rich(row: dict) -> bool:
    """Rows produced under a planted-bad configuration or the adversarial scenario."""
    return row.get("config", "baseline") != "baseline" or row["scenario"] == ADVERSARIAL


def replay_plan(scenarios: list[Scenario]) -> list[tuple[BotConfig, list[Scenario]]]:
    adversarial = [s for s in scenarios if s.name == ADVERSARIAL]
    return [(BASELINE, adversarial)] + [(cfg, list(scenarios)) for cfg in PLANTED_BAD]


def build_fail_rich(
    gen_model,
    model,
    scenarios: list[Scenario],
    path: Path = CONVERSATIONS_PATH,
    workers: int = 8,
    seed: int = 0,
    log: Callable[[str], None] = print,
    graph_factory: Callable | None = None,
    generate: Callable | None = None,
) -> list[Row]:
    """Append planted-bad and adversarial rows to `path`; skip (config, scenario) pairs
    already present so a re-run only fills gaps. Baseline turns are reused as inputs so the
    same user turn appears under every configuration."""
    graph_factory = graph_factory or graph_for
    generate = generate or generate_turns
    existing = read_jsonl(path)
    have = {(r.get("config", "baseline"), r["scenario"]) for r in existing}
    turns_cache: dict[str, list[str]] = {}

    def turns_for(s: Scenario) -> list[str]:
        if s.name not in turns_cache:
            baseline = [
                r["input"]
                for r in existing
                if r["scenario"] == s.name and r.get("config", "baseline") == "baseline"
            ]
            turns_cache[s.name] = baseline or generate(gen_model, s, seed=seed)
            if not baseline:
                log(f"generated {len(turns_cache[s.name])} turns for {s.name}")
        return turns_cache[s.name]

    written: list[Row] = []
    for config, wanted in replay_plan(scenarios):
        todo = [s for s in wanted if (config.name, s.name) not in have]
        if not todo:
            log(f"{config.name}: nothing to replay")
            continue
        turns = [(s.name, t) for s in todo for t in turns_for(s)]
        graph = graph_factory(config, model)
        rows = replay(turns, graph, config.prompt_version, config=config.name, workers=workers)
        write_rows(rows, path, append=True)
        written.extend(rows)
        log(f"{config.name} ({config.note}): {len(rows)} rows across {len(todo)} scenarios")
    return written


def format_fails(counts: dict[str, int], target: int, rules: tuple[str, ...]) -> str:
    parts = [f"{k} {counts.get(k, 0)}" + (f"/{target}" if k in rules else "") for k in KEYS]
    return "fails: " + "  ".join(parts)


def label_until(
    rows: list[dict],
    target: int,
    rules: tuple[str, ...] = TARGET_RULES,
    labels_path: Path = LABELS_PATH,
    console: Console | None = None,
    ask: Ask | None = None,
    limit: int | None = None,
) -> int:
    """Run the labeller, fail-rich rows first, until each rule in `rules` has `target` fails.
    Prints the running fail count per rule after every labelled row."""
    console = console or Console()

    def reached(counts: dict[str, int]) -> bool:
        return all(counts.get(k, 0) >= target for k in rules)

    def report(record: dict | None) -> bool:
        counts = fail_counts(labels_path)
        console.print(format_fails(counts, target, rules), style="bold")
        return reached(counts)

    if report(None):
        console.print("[green]target already reached[/]")
        return 0
    sample = priority_sample(rows, is_fail_rich)
    return label_rows(sample, labels_path, console, ask, limit, after_each=report)
