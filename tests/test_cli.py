from typer.testing import CliRunner

import core
from core.cli import TOP_LEVEL, TRIAGE, app

runner = CliRunner()


def test_core_imports() -> None:
    assert core.__doc__


def test_help_lists_every_subcommand() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in TOP_LEVEL:
        assert name in result.output

    result = runner.invoke(app, ["triage", "--help"])
    assert result.exit_code == 0
    for name in TRIAGE:
        assert name in result.output


def test_placeholders_exit_1() -> None:
    for name in TOP_LEVEL:
        if name in ("triage", "bot"):
            continue
        result = runner.invoke(app, [name])
        assert result.exit_code == 1, name
        assert "not implemented" in result.output
    for name in TRIAGE:
        result = runner.invoke(app, ["triage", name])
        assert result.exit_code == 1, name
        assert "not implemented" in result.output
