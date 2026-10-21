"""Load data/scenarios.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

SCENARIOS_PATH = Path(__file__).resolve().parents[2] / "data" / "scenarios.yaml"


@dataclass(frozen=True)
class Scenario:
    name: str
    count: int
    description: str
    voice: str | None = None


def load_scenarios(path: Path = SCENARIOS_PATH) -> list[Scenario]:
    raw = yaml.safe_load(path.read_text()) or {}
    scenarios = [
        Scenario(
            name=s["name"],
            count=int(s["count"]),
            description=" ".join(s["description"].split()),
            voice=" ".join(s["voice"].split()) if s.get("voice") else None,
        )
        for s in raw.get("scenarios", [])
    ]
    if not scenarios:
        raise ValueError(f"no scenarios in {path}")
    return scenarios
