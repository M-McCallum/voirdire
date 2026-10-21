"""Human labelling loop: sample conversations, ask y/n per key, append to labels.jsonl.

Labels are typed by a person at the terminal. Nothing in this module decides a label.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Iterable
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CONVERSATIONS_PATH = DATA_DIR / "conversations.jsonl"
LABELS_PATH = DATA_DIR / "labels.jsonl"

KEYS = ("no_dosing_advice", "crisis_routed", "on_topic", "tone_ok")
CHOICES = ("y", "n", "s")
LABELLER = "human"
SAMPLE_SIZE = 100
SEED = 42

# Called with a key name, returns one of CHOICES. The default reads the terminal.
Ask = Callable[[str], str]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def labelled_ids(path: Path = LABELS_PATH) -> set[str]:
    return {r["trace_id"] for r in read_jsonl(path)}


def stratified_sample(rows: Iterable[dict], n: int = SAMPLE_SIZE, seed: int = SEED) -> list[dict]:
    """Proportional allocation per scenario (largest remainder for leftovers), then shuffled.

    Deterministic for a given (rows, n, seed). Shuffling the final list keeps scenarios
    interleaved so the labeller is not primed by a long run of one kind.
    """
    rows = list(rows)
    if not rows:
        return []
    by_scenario: dict[str, list[dict]] = {}
    for r in rows:
        by_scenario.setdefault(r["scenario"], []).append(r)
    total = len(rows)
    n = min(n, total)
    names = sorted(by_scenario)
    quotas = {s: n * len(by_scenario[s]) // total for s in names}
    leftover = n - sum(quotas.values())
    by_remainder = sorted(names, key=lambda s: (-((n * len(by_scenario[s])) % total), s))
    for s in by_remainder[:leftover]:
        quotas[s] += 1

    rng = random.Random(seed)
    sample: list[dict] = []
    for s in names:
        group = list(by_scenario[s])
        rng.shuffle(group)
        sample.extend(group[: quotas[s]])
    rng.shuffle(sample)
    return sample


def terminal_ask(console: Console) -> Ask:
    def ask(key: str) -> str:
        return Prompt.ask(f"  [bold]{key}[/]", choices=list(CHOICES), console=console)

    return ask


def show_row(console: Console, row: dict, done: int, total: int) -> None:
    console.rule(
        f"[bold]{done}/{total} labelled[/]  scenario={row['scenario']}  trace={row['trace_id']}"
    )
    console.print(Panel(row["input"], title="input", border_style="cyan"))
    console.print(Panel(row["output"], title="output", border_style="magenta"))


def label_rows(
    rows: list[dict],
    labels_path: Path = LABELS_PATH,
    console: Console | None = None,
    ask: Ask | None = None,
    limit: int | None = None,
) -> int:
    """Walk the sample, prompting for each row not already in labels_path.

    Each completed row is appended immediately, so a restart resumes where this left off.
    `s` at any key skips the row without writing it; it comes back on the next run.
    `limit` caps how many rows are shown this run. Returns the number of rows labelled.
    """
    console = console or Console()
    ask = ask or terminal_ask(console)
    done_ids = labelled_ids(labels_path)
    total = len(rows)
    done = sum(r["trace_id"] in done_ids for r in rows)
    pending = [r for r in rows if r["trace_id"] not in done_ids]
    if limit is not None:
        pending = pending[:limit]

    labelled = 0
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with labels_path.open("a") as f:
        for row in pending:
            show_row(console, row, done, total)
            labels: dict[str, bool] = {}
            try:
                for key in KEYS:
                    answer = ask(key)
                    if answer == "s":
                        break
                    labels[key] = answer == "y"
            except (KeyboardInterrupt, EOFError):
                console.print("\n[yellow]stopped[/]")
                break
            if len(labels) < len(KEYS):
                console.print("[yellow]skipped[/]")
                continue
            record = {"trace_id": row["trace_id"], "labels": labels, "labeller": LABELLER}
            f.write(json.dumps(record) + "\n")
            f.flush()
            done += 1
            labelled += 1

    remaining = total - done
    console.print(f"[bold]{done}/{total} labelled[/], {remaining} remaining, {labelled} this run")
    return labelled
