import json
from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import Field
from typer.testing import CliRunner

from core.bot import build_graph, run
from core.bot.corpus import Doc, Retriever, tokenize
from core.bot.graph import _text, load_prompt
from core.bot.guards import CRISIS_RESPONSE, GUARD_REFUSAL, crisis_match, medication_match
from core.cli import app

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus"


class RecordingModel(FakeMessagesListChatModel):
    """Fake chat model that records every message list it was invoked with."""

    calls: list[list[BaseMessage]] = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def fake_model(text: str = "Try body doubling.") -> RecordingModel:
    return RecordingModel(responses=[AIMessage(content=text)])


def retriever() -> Retriever:
    return Retriever([
        Doc("body-doubling", "Body doubling means working alongside another person."),
        Doc("sensory", "Noise-reducing headphones help with sensory overload."),
        Doc("routines", "Predictable routines reduce decisions and protect energy."),
    ])  # fmt: skip


# --- corpus and retrieval ---


def test_corpus_has_eight_files() -> None:
    assert len(list(CORPUS.glob("*.md"))) == 8


def test_tokenize_drops_stopwords_and_stems() -> None:
    assert tokenize("I keep starting the tasks when I need to") == ["start", "task"]


def test_bm25_ranks_relevant_doc_first() -> None:
    docs = retriever().search("how does body doubling work")
    assert docs[0].name == "body-doubling"


def test_bm25_returns_nothing_for_no_overlap() -> None:
    assert retriever().search("zebra quantum") == []


def test_real_corpus_finds_task_initiation_for_morning_freeze() -> None:
    names = [d.name for d in Retriever.from_dir().search("I freeze when I need to start work")]
    assert "02-task-initiation" in names


def test_empty_corpus_rejected() -> None:
    import pytest

    with pytest.raises(ValueError):
        Retriever([])


# --- guards ---


def test_guard_matches_medication_names() -> None:
    assert medication_match("is Adderall good for me")
    assert medication_match("I just started methylphenidate")


def test_guard_matches_dosing_phrasing_without_names() -> None:
    assert medication_match("should I take 20mg in the morning?")
    assert medication_match("can I double my dose")
    assert medication_match("I skipped my pills yesterday")
    assert medication_match("how many tablets is normal")


def test_guard_ignores_ordinary_language() -> None:
    assert not medication_match("I struggle to start tasks")
    assert not medication_match("how many breaks should I take")
    assert not medication_match("how much time should I take off work")


def test_crisis_regex() -> None:
    assert crisis_match("I want to kill myself")
    assert crisis_match("thinking about self-harm again")
    assert crisis_match("I don't want to be alive")
    assert not crisis_match("this task is killing me")


# --- prompt ---


def test_prompt_v1_forbids_dosing_advice() -> None:
    prompt = load_prompt("v1")
    assert "medication" in prompt.lower()
    assert "dos" in prompt.lower()


# --- graph ---


def test_generate_path_calls_model_once_with_prompt_and_context() -> None:
    model = fake_model("Try body doubling.")
    result = run("I can't start my work alone, body doubling?", build_graph(model, retriever()))
    assert result["answer"] == "Try body doubling."
    assert result.get("blocked_by") is None
    assert len(model.calls) == 1
    system, human = model.calls[0]
    assert system.type == "system"
    assert load_prompt("v1") in system.content
    assert "## body-doubling" in system.content
    assert human.type == "human"
    assert human.content == "I can't start my work alone, body doubling?"


def test_guard_skips_generation() -> None:
    model = fake_model("SHOULD NOT BE CALLED")
    result = run("how much ritalin should I take", build_graph(model, retriever()))
    assert result["answer"] == GUARD_REFUSAL
    assert result["blocked_by"] == "guard"
    assert model.calls == []


def test_crisis_beats_guard_and_skips_generation() -> None:
    model = fake_model("SHOULD NOT BE CALLED")
    result = run("I want to overdose on my adderall", build_graph(model, retriever()))
    assert result["answer"] == CRISIS_RESPONSE
    assert result["blocked_by"] == "crisis"
    assert model.calls == []


def test_text_flattens_block_list_content() -> None:
    assert _text("plain") == "plain"
    assert _text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "ab"


# --- telemetry ---


def test_spans_root_with_children() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    run("how much ritalin should I take", build_graph(fake_model(), retriever()))
    run("I can't start my work", build_graph(fake_model(), retriever()))

    spans = exporter.get_finished_spans()
    roots = [s for s in spans if s.name == "bot.run"]
    assert len(roots) == 2
    for root in roots:
        assert root.parent is None
        assert root.attributes["prompt_version"] == "v1"
        assert root.attributes["model"]
    blocked, generated = roots
    assert blocked.attributes["blocked_by"] == "guard"
    assert generated.attributes["blocked_by"] == ""

    def children(root):
        return {s.name for s in spans if s.parent and s.parent.span_id == root.context.span_id}

    assert children(blocked) == {"bot.retrieve", "bot.crisis", "bot.guard"}
    assert children(generated) == {"bot.retrieve", "bot.crisis", "bot.guard", "bot.generate"}


def test_spans_carry_inputs_and_outputs() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    import core.telemetry as telemetry

    with trace.use_span(tracer.start_span("harness"), end_on_exit=True):
        original = telemetry.trace.get_tracer
        telemetry.trace.get_tracer = lambda *_a, **_k: tracer
        try:
            run("I can't start my work", build_graph(fake_model("Try body doubling."), retriever()))
        finally:
            telemetry.trace.get_tracer = original

    spans = {s.name: s for s in exporter.get_finished_spans()}
    root = spans["bot.run"].attributes
    assert root["langsmith.span.kind"] == "chain"
    assert root["openinference.span.kind"] == "CHAIN"
    assert json.loads(root["input.value"]) == {"message": "I can't start my work"}
    assert json.loads(root["output.value"]) == {"answer": "Try body doubling."}
    assert root["gen_ai.prompt"] == root["input.value"]
    assert root["gen_ai.completion"] == root["output.value"]

    gen = spans["bot.generate"].attributes
    assert gen["langsmith.span.kind"] == "llm"
    assert gen["gen_ai.system"] == "anthropic"
    prompt = json.loads(gen["gen_ai.prompt"])
    assert [m["role"] for m in prompt] == ["system", "human"]
    assert load_prompt("v1") in prompt[0]["content"]
    assert json.loads(gen["gen_ai.completion"]) == [
        {"role": "assistant", "content": "Try body doubling."}
    ]

    assert spans["bot.retrieve"].attributes["input.value"] == "I can't start my work"
    assert json.loads(spans["bot.guard"].attributes["output.value"]) == {"matched": False}


def test_setup_tracing_targets_collector_endpoint(monkeypatch) -> None:
    import core.telemetry as telemetry

    captured = {}

    class FakeExporter:
        def __init__(self, endpoint):
            captured["endpoint"] = endpoint

        def export(self, spans):
            return None

        def shutdown(self):
            return None

    monkeypatch.setattr(telemetry, "OTLPSpanExporter", FakeExporter)
    monkeypatch.setattr(telemetry.trace, "set_tracer_provider", lambda p: None)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    telemetry.setup_tracing().shutdown()
    assert captured["endpoint"] == "http://localhost:4318/v1/traces"

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:9999/")
    telemetry.setup_tracing().shutdown()
    assert captured["endpoint"] == "http://collector:9999/v1/traces"


# --- CLI ---


def test_cli_bot_runs_with_mocked_model_and_no_exporter(monkeypatch) -> None:
    import core.bot.graph as graph_mod
    import core.telemetry as telemetry

    class NoopProvider:
        def shutdown(self):
            return None

    monkeypatch.setattr(graph_mod, "default_chat_model", lambda: fake_model("mocked answer"))
    monkeypatch.setattr(telemetry, "setup_tracing", lambda: NoopProvider())
    result = CliRunner().invoke(app, ["bot", "I can't get started in the mornings"])
    assert result.exit_code == 0, result.output
    assert "mocked answer" in result.output
