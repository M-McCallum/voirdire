"""vd command-line interface."""

from pathlib import Path

import typer

app = typer.Typer(help="voirdire: judge the judges.", no_args_is_help=True)
triage_app = typer.Typer(help="Triage: ingest, cluster, push.", no_args_is_help=True)
app.add_typer(triage_app, name="triage")

TOP_LEVEL = ["bot", "synth", "label", "score-judges", "triage", "demo", "report", "ui"]
TRIAGE = ["ingest", "cluster", "push"]


def _not_implemented(name: str) -> None:
    typer.echo(f"{name}: not implemented", err=True)
    raise typer.Exit(code=1)


def _register(parent: typer.Typer, name: str) -> None:
    def command() -> None:
        _not_implemented(name)

    command.__doc__ = f"{name} (not implemented)."
    parent.command(name=name)(command)


for _name in TOP_LEVEL:
    if _name not in ("triage", "bot", "synth", "label"):
        _register(app, _name)


@app.command()
def bot(
    message: str = typer.Argument(..., help="One user turn."),
    prompt_version: str = typer.Option("v1", help="Prompt file in prompts/."),
) -> None:
    """Run one turn of the coping-skills chatbot and print the answer."""
    from dotenv import load_dotenv

    load_dotenv()
    from core.bot import build_graph, run
    from core.bot.corpus import Retriever
    from core.bot.graph import default_chat_model
    from core.telemetry import setup_tracing

    provider = setup_tracing()
    graph = build_graph(default_chat_model(), Retriever.from_dir(), prompt_version)
    try:
        result = run(message, graph, prompt_version)
    finally:
        provider.shutdown()
    typer.echo(result["answer"])


for _name in TRIAGE:
    _register(triage_app, _name)


@app.command()
def synth(
    dry_run: bool = typer.Option(False, "--dry-run", help="Generate 5 turns and print them."),
    prompt_version: str = typer.Option("v1", help="Prompt file in prompts/."),
    dataset: str = typer.Option("support-bot-eval", help="LangSmith dataset name."),
    seed: int = typer.Option(0, help="Seed for tone/length variation."),
) -> None:
    """Generate synthetic user turns, replay them through the bot, upload to LangSmith."""
    from dotenv import load_dotenv

    load_dotenv()
    from core.bot import build_graph
    from core.bot.corpus import Retriever
    from core.bot.graph import default_chat_model
    from core.synth import generate_turns, load_scenarios, replay, upload_rows, write_rows
    from core.telemetry import setup_tracing

    scenarios = load_scenarios()
    gen_model = default_chat_model(max_tokens=8192)  # 10 turns per batch, some long

    if dry_run:
        for scenario in scenarios[:5]:
            (turn,) = generate_turns(gen_model, scenario, count=1, seed=seed)
            typer.echo(f"[{scenario.name}] {turn}")
        return

    turns = [(s.name, t) for s in scenarios for t in generate_turns(gen_model, s, seed=seed)]
    typer.echo(f"generated {len(turns)} turns across {len(scenarios)} scenarios", err=True)

    provider = setup_tracing()
    graph = build_graph(default_chat_model(), Retriever.from_dir(), prompt_version)
    try:
        rows = replay(turns, graph, prompt_version)
    finally:
        provider.shutdown()
    path = write_rows(rows)
    typer.echo(f"wrote {len(rows)} rows to {path}", err=True)

    from langsmith import Client

    dataset_id = upload_rows(rows, Client(), dataset)
    typer.echo(f"uploaded {len(rows)} examples to LangSmith dataset {dataset} ({dataset_id})")


@app.command()
def label(
    limit: int | None = typer.Option(None, help="Stop after showing this many rows this run."),
    sample_size: int = typer.Option(
        100, "--sample-size", help="Rows to sample, stratified by scenario."
    ),
    seed: int = typer.Option(42, help="Sampling seed."),
    conversations: str | None = typer.Option(None, help="Override data/conversations.jsonl."),
    labels: str | None = typer.Option(None, help="Override data/labels.jsonl."),
) -> None:
    """Label a stratified sample by hand: y/n per key, s to skip. Resumes on restart."""
    from core.label import (
        CONVERSATIONS_PATH,
        LABELS_PATH,
        label_rows,
        read_jsonl,
        stratified_sample,
    )

    conv_path = Path(conversations) if conversations else CONVERSATIONS_PATH
    rows = read_jsonl(conv_path)
    if not rows:
        typer.echo(f"no rows in {conv_path}; run `vd synth` first", err=True)
        raise typer.Exit(code=1)
    sample = stratified_sample(rows, n=sample_size, seed=seed)
    label_rows(sample, labels_path=Path(labels) if labels else LABELS_PATH, limit=limit)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
