import json
from collections import Counter
from pathlib import Path

from rich.console import Console
from typer.testing import CliRunner

from core.cli import app
from core.label import KEYS, label_rows, labelled_ids, stratified_sample

MIX = {
    "benign_question": 60,
    "meds_question": 30,
    "distress": 25,
    "off_topic": 25,
    "jailbreak": 20,
    "pediatric": 20,
}


def rows(mix: dict[str, int] = MIX) -> list[dict]:
    out = []
    for scenario, count in mix.items():
        for i in range(count):
            out.append(
                {
                    "input": f"{scenario} question {i}",
                    "output": f"{scenario} answer {i}",
                    "scenario": scenario,
                    "trace_id": f"{scenario}-{i:03d}",
                }
            )
    return out


def scripted(*answers: str):
    it = iter(answers)

    def ask(key: str) -> str:
        return next(it)

    return ask


def test_stratified_sample_is_proportional_and_deterministic() -> None:
    sample = stratified_sample(rows(), n=100, seed=42)
    assert len(sample) == 100
    assert Counter(r["scenario"] for r in sample) == {
        "benign_question": 33,
        "meds_question": 17,
        "distress": 14,
        "off_topic": 14,
        "jailbreak": 11,
        "pediatric": 11,
    }
    assert len({r["trace_id"] for r in sample}) == 100
    assert [r["trace_id"] for r in stratified_sample(rows(), n=100, seed=42)] == [
        r["trace_id"] for r in sample
    ]
    assert [r["trace_id"] for r in stratified_sample(rows(), n=100, seed=7)] != [
        r["trace_id"] for r in sample
    ]
    # scenarios are interleaved, not blocked
    assert len({r["scenario"] for r in sample[:10]}) > 1


def test_stratified_sample_small_inputs() -> None:
    assert stratified_sample([], n=100) == []
    few = rows({"a": 3, "b": 2})
    assert len(stratified_sample(few, n=100)) == 5


def test_label_rows_appends_and_resumes(tmp_path: Path) -> None:
    labels = tmp_path / "labels.jsonl"
    sample = rows({"a": 3})
    console = Console(file=open(tmp_path / "out.txt", "w"), width=80)

    n = label_rows(sample, labels, console, scripted("y", "n", "y", "y"), limit=1)
    assert n == 1
    records = [json.loads(line) for line in labels.read_text().splitlines()]
    assert records == [
        {
            "trace_id": "a-000",
            "labels": {
                "no_dosing_advice": True,
                "crisis_handled": False,
                "on_topic": True,
                "tone_ok": True,
            },
            "labeller": "human",
        }
    ]
    assert labelled_ids(labels) == {"a-000"}

    # restart: a-000 is skipped, a-001 is skipped with "s", a-002 is labelled
    n = label_rows(sample, labels, console, scripted("y", "s", "n", "n", "n", "n"))
    assert n == 1
    ids = [json.loads(line)["trace_id"] for line in labels.read_text().splitlines()]
    assert ids == ["a-000", "a-002"]

    # restart again: only the skipped row is pending
    n = label_rows(sample, labels, console, scripted(*["y"] * len(KEYS)))
    assert n == 1
    assert labelled_ids(labels) == {"a-000", "a-001", "a-002"}


def test_label_rows_stops_on_eof_without_partial_write(tmp_path: Path) -> None:
    labels = tmp_path / "labels.jsonl"
    console = Console(file=open(tmp_path / "out.txt", "w"), width=80)

    def ask(key: str) -> str:
        if key == "on_topic":
            raise EOFError
        return "y"

    assert label_rows(rows({"a": 2}), labels, console, ask) == 0
    assert not labels.exists() or labels.read_text() == ""


def test_cli_label_reads_terminal_answers(tmp_path: Path) -> None:
    conversations = tmp_path / "conversations.jsonl"
    conversations.write_text("".join(json.dumps(r) + "\n" for r in rows({"a": 2, "b": 2})))
    labels = tmp_path / "labels.jsonl"
    result = CliRunner().invoke(
        app,
        ["label", "--limit", "1", "--conversations", str(conversations), "--labels", str(labels)],
        input="y\nn\ny\ny\n",
    )
    assert result.exit_code == 0, result.output
    assert "input" in result.output and "output" in result.output
    assert "1/4 labelled" in result.output
    (record,) = [json.loads(line) for line in labels.read_text().splitlines()]
    assert record["labeller"] == "human"
    assert list(record["labels"]) == list(KEYS)


def test_cli_label_without_data_exits_1(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["label", "--conversations", str(tmp_path / "missing.jsonl")])
    assert result.exit_code == 1
