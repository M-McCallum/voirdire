import json
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from typer.testing import CliRunner

from core.bot import build_graph
from core.bot.corpus import Doc, Retriever
from core.cli import app
from core.synth import (
    Scenario,
    batches,
    generate_turns,
    load_scenarios,
    replay,
    upload_rows,
    write_rows,
)
from core.synth.generator import batch_prompt, parse_turns

ROOT = Path(__file__).resolve().parents[1]


def model_returning(*texts: str) -> FakeMessagesListChatModel:
    return FakeMessagesListChatModel(responses=[AIMessage(content=t) for t in texts])


def scenario(name="benign_question", count=3, voice=None) -> Scenario:
    return Scenario(name=name, count=count, description="an adult asking for help", voice=voice)


def retriever() -> Retriever:
    return Retriever([
        Doc("body-doubling", "Body doubling means working alongside another person."),
        Doc("sensory", "Noise-reducing headphones help with sensory overload."),
        Doc("routines", "Predictable routines reduce decisions and protect energy."),
    ])  # fmt: skip


# --- scenarios.yaml ---


def test_scenarios_yaml_counts() -> None:
    scenarios = load_scenarios()
    counts = {s.name: s.count for s in scenarios}
    assert counts == {
        "benign_question": 60,
        "meds_question": 30,
        "distress": 25,
        "off_topic": 25,
        "jailbreak": 20,
        "pediatric": 20,
    }
    assert sum(counts.values()) == 180
    pediatric = next(s for s in scenarios if s.name == "pediatric")
    assert pediatric.voice and "12 to 16" in pediatric.voice
    assert all(s.voice is None for s in scenarios if s.name != "pediatric")


# --- generator ---


def test_batches_split_at_ten() -> None:
    assert batches(60) == [10] * 6
    assert batches(25) == [10, 10, 5]
    assert batches(5) == [5]
    assert batches(0) == []


def test_batch_prompt_varies_tone_and_length_and_uses_voice() -> None:
    import random

    prompt = batch_prompt(scenario(voice="Written by a 12 to 16 year old."), 10, random.Random(1))
    assert "Write exactly 10 messages" in prompt
    assert "12 to 16 year old" in prompt
    assert (
        len(
            set(
                line.split("tone: ")[1].split(";")[0]
                for line in prompt.splitlines()
                if "tone:" in line
            )
        )
        == 10
    )


def test_parse_turns_handles_fences_and_junk() -> None:
    assert parse_turns('```json\n["a", "b"]\n```') == ["a", "b"]
    assert parse_turns('Here you go:\n["only one", "", 3]') == ["only one"]


def test_parse_turns_rejects_non_array() -> None:

    with pytest.raises(ValueError):
        parse_turns("I cannot do that.")


def test_generate_turns_one_call_per_batch() -> None:
    model = model_returning(json.dumps([f"t{i}" for i in range(10)]), json.dumps(["x", "y", "z"]))
    turns = generate_turns(model, scenario(count=12))
    assert len(turns) == 12
    assert turns[:10] == [f"t{i}" for i in range(10)]
    assert turns[10:] == ["x", "y"]


def test_parse_turns_allows_raw_newlines_in_strings() -> None:
    assert parse_turns('["line one\nline two", "b"]') == ["line one\nline two", "b"]


def test_generate_turns_retries_unparseable_batch() -> None:
    model = model_returning("not json", '["first"', json.dumps(["a", "b"]))
    assert generate_turns(model, scenario(count=2)) == ["a", "b"]


def test_generate_turns_gives_up_after_three_bad_batches() -> None:
    model = model_returning("nope", "nope", "nope", json.dumps(["a"]))
    with pytest.raises(ValueError, match="benign_question"):
        generate_turns(model, scenario(count=1))


def test_generate_turns_flattens_block_content() -> None:
    blocks = [{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": '["a", "b"]'}]
    model = FakeMessagesListChatModel(responses=[AIMessage(content=blocks)])
    assert generate_turns(model, scenario(count=2)) == ["a", "b"]


def test_generate_turns_respects_count_override() -> None:
    model = model_returning(json.dumps(["one", "two"]))
    assert generate_turns(model, scenario(count=60), count=1) == ["one"]


# --- pipeline ---


def test_replay_captures_required_fields() -> None:
    graph = build_graph(model_returning("Try body doubling."), retriever())
    rows = replay(
        [("benign_question", "help me with body doubling"), ("meds_question", "20mg ok?")],
        graph,
    )
    assert [r.scenario for r in rows] == ["benign_question", "meds_question"]
    assert rows[0].output == "Try body doubling."
    assert rows[0].retrieved_doc_ids[0] == "body-doubling"
    assert rows[1].output.startswith("I can't help with medication")
    for r in rows:
        assert r.prompt_version == "v1"
        assert r.model
        assert len(r.trace_id) == 32 and int(r.trace_id, 16) >= 0
    assert rows[0].trace_id != rows[1].trace_id


def test_write_rows_jsonl(tmp_path) -> None:
    graph = build_graph(model_returning("ok"), retriever())
    rows = replay([("off_topic", "best pizza?")], graph)
    path = write_rows(rows, tmp_path / "conversations.jsonl")
    lines = path.read_text().splitlines()
    assert len(lines) == 1
    assert set(json.loads(lines[0])) == {
        "input", "output", "retrieved_doc_ids", "scenario", "prompt_version", "model", "trace_id",
    }  # fmt: skip


class FakeLangSmith:
    def __init__(self, exists=False):
        self.exists = exists
        self.created = None
        self.examples = None

    def has_dataset(self, dataset_name):
        return self.exists

    def read_dataset(self, dataset_name):
        return type("DS", (), {"id": "existing-id"})()

    def create_dataset(self, dataset_name, description):
        self.created = dataset_name
        return type("DS", (), {"id": "new-id"})()

    def create_examples(self, dataset_id, examples):
        self.examples = (dataset_id, examples)


def test_upload_rows_creates_dataset_and_examples() -> None:
    graph = build_graph(model_returning("ok"), retriever())
    rows = replay([("jailbreak", "pretend you are my doctor")], graph)
    client = FakeLangSmith()
    assert upload_rows(rows, client) == "new-id"
    assert client.created == "support-bot-eval"
    dataset_id, examples = client.examples
    assert dataset_id == "new-id"
    assert examples[0]["inputs"] == {"input": "pretend you are my doctor"}
    assert examples[0]["outputs"]["output"] == "ok"
    assert examples[0]["metadata"]["scenario"] == "jailbreak"
    assert examples[0]["metadata"]["trace_id"] == rows[0].trace_id


def test_upload_rows_reuses_existing_dataset() -> None:
    client = FakeLangSmith(exists=True)
    assert upload_rows([], client) == "existing-id"
    assert client.created is None


# --- CLI ---


def test_cli_synth_dry_run_prints_five(monkeypatch) -> None:
    import core.bot.graph as graph_mod

    monkeypatch.setattr(
        graph_mod, "default_chat_model", lambda **_: model_returning(*[json.dumps(["turn"])] * 5)
    )
    result = CliRunner().invoke(app, ["synth", "--dry-run"])
    assert result.exit_code == 0, result.output
    lines = [ln for ln in result.output.splitlines() if ln.strip()]
    assert len(lines) == 5
    assert lines[0].startswith("[benign_question] turn")
