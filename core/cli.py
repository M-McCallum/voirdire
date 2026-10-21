"""vd command-line interface."""

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
    if _name not in ("triage", "bot"):
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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
