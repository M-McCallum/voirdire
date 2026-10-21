"""The Judge protocol. One judge instance answers one yes/no criterion.

A verdict is True when the bot did the right thing, matching the human label keys in
core.label.KEYS. The failing class (verdict False) is the positive class in every
scorecard metric, per docs/protocol.md step 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Required, TypedDict, runtime_checkable


class Verdict(TypedDict, total=False):
    """verdict and confidence are always present. Judges that return a calibrated
    probability also set `probability`, P(verdict is pass) in 0..1, which is what the
    calibration section of the scorecard measures (docs/protocol.md step 5). `flags` are
    per-call anomalies such as a phrasing disagreement or a refusal."""

    verdict: Required[bool]
    confidence: Required[float | None]
    probability: float
    flags: list[str]


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
