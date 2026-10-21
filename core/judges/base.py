"""The Judge protocol. One judge instance answers one yes/no criterion.

A verdict is True when the bot did the right thing, matching the human label keys in
core.label.KEYS. The failing class (verdict False) is the positive class in every
scorecard metric, per docs/protocol.md step 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypedDict, runtime_checkable


class Verdict(TypedDict):
    verdict: bool
    confidence: float | None


@dataclass(frozen=True)
class Fingerprint:
    """What pins a judge, recorded on every scorecard (docs/protocol.md step 8)."""

    model_snapshot: str | None
    prompt_hash: str
    rubric_version: str
    temperature: float | None


@runtime_checkable
class Judge(Protocol):
    name: str
    criterion: str

    def judge(self, input: str, output: str, retrieved: list[str]) -> Verdict: ...

    def fingerprint(self) -> Fingerprint: ...
