"""vd command-line interface."""

from pathlib import Path
from typing import Annotated

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
    if _name not in ("triage", "bot", "synth", "label", "score-judges"):
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
    target_fails: int | None = typer.Option(
        None,
        "--target-fails",
        help="Replay the grid under the planted-bad configs plus the adversarial scenario, "
        "append to conversations.jsonl, then label (those rows first) until each rule has "
        "this many human fails.",
    ),
    workers: int = typer.Option(8, help="Parallel bot runs for --target-fails."),
    rules: str = typer.Option(
        "no_dosing_advice,crisis_handled", help="Rules the --target-fails count applies to."
    ),
    limit: int | None = typer.Option(None, help="Stop labelling after this many rows."),
    conversations: str | None = typer.Option(None, help="Override data/conversations.jsonl."),
    labels: str | None = typer.Option(None, help="Override data/labels.jsonl."),
    trace: bool | None = typer.Option(
        None,
        "--trace/--no-trace",
        help="Export bot traces to the OTel collector. Default on, off with --target-fails.",
    ),
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

    if target_fails is not None:
        from core.label import LABELS_PATH, read_jsonl
        from core.synth.pipeline import CONVERSATIONS_PATH
        from core.synth.target_fails import build_fail_rich, label_until

        conv_path = Path(conversations) if conversations else CONVERSATIONS_PATH
        # labelling-only rows: trace ids are needed, exported traces are not
        provider = setup_tracing(export=trace if trace is not None else False)
        try:
            written = build_fail_rich(
                gen_model, default_chat_model(), scenarios, conv_path, workers, seed,
                log=lambda m: typer.echo(m, err=True),
            )  # fmt: skip
        finally:
            provider.shutdown()
        typer.echo(f"appended {len(written)} rows to {conv_path}", err=True)
        label_until(
            read_jsonl(conv_path),
            target_fails,
            tuple(r.strip() for r in rules.split(",") if r.strip()),
            Path(labels) if labels else LABELS_PATH,
            limit=limit,
        )
        return

    if dry_run:
        for scenario in scenarios[:5]:
            (turn,) = generate_turns(gen_model, scenario, count=1, seed=seed)
            typer.echo(f"[{scenario.name}] {turn}")
        return

    turns = [(s.name, t) for s in scenarios for t in generate_turns(gen_model, s, seed=seed)]
    typer.echo(f"generated {len(turns)} turns across {len(scenarios)} scenarios", err=True)

    provider = setup_tracing(export=trace if trace is not None else True)
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


@app.command(name="score-judges")
def score_judges(
    judge: Annotated[
        list[str] | None,
        typer.Option("--judge", help="Judge type: deterministic or llm. Default deterministic."),
    ] = None,
    criterion: Annotated[
        list[str] | None,
        typer.Option("--criterion", help="Criterion to score; default is all the judge has."),
    ] = None,
    pack: str = typer.Option("startup", help="Policy pack name in packs/ or a path."),
    dataset: str = typer.Option(
        "support-bot-labelled", help="LangSmith dataset of labelled traces."
    ),
    repetitions: int = typer.Option(3, help="Passes per judge x criterion."),
    out: str = typer.Option("out/scorecards", help="Scorecard directory."),
    offline: bool = typer.Option(
        False, "--offline", help="No LangSmith calls: scorecards and plot only."
    ),
) -> None:
    """Qualify judges against human labels per docs/protocol.md; write scorecards and kappa.png."""
    from dotenv import load_dotenv

    load_dotenv()
    from rich.console import Console
    from rich.table import Table

    from core.packs import load_pack
    from core.score_judges import make_client, score

    console = Console()
    policy = load_pack(pack)
    cards = score(
        judge or ["deterministic"],
        criterion or None,
        None if offline else make_client(),
        policy,
        dataset,
        Path(out),
        repetitions,
        log=lambda m: console.print(m, style="dim"),
        offline=offline,
    )

    table = Table(title=f"judge scorecards (pack: {policy.name})")
    for col in ("judge", "criterion", "n/fail", "kappa [95%]", "exact", "TPR", "TNR",
                "test-retest", "length r", "gate"):  # fmt: skip
        table.add_column(col)

    def f(x: float | None, nd: int = 2) -> str:
        return "n/a" if x is None else f"{x:.{nd}f}"

    for c in cards:
        a, cm, g = c["agreement"], c["confusion"], c["gate"][policy.name]
        lo, hi = a["kappa_ci"]
        table.add_row(
            c["judge"], c["criterion"], f"{c['n']}/{c['n_fail']}",
            f"{f(a['kappa'])} [{f(lo)}, {f(hi)}]", f(a["exact"]), f(cm["tpr"]), f(cm["tnr"]),
            f(c["consistency"]["agreement"]), f(c["bias"]["length_verdict_r"]),
            ("[green]qualified[/]" if g["qualified"] else "[red]not qualified[/]")
            + f": {g['reason']}",
        )  # fmt: skip
    console.print(table)
    for c in cards:
        console.print(f"{c['judge']}/{c['criterion']}: {c['experiment']['url'] or 'offline'}")
    console.print(f"kappa plot: {Path(out) / 'kappa.png'}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
