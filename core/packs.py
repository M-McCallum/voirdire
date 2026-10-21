"""Policy packs: the thresholds a judge must meet before it may gate.

A pack is YAML in packs/ with min_kappa_to_gate and one rule per criterion. Rules tagged
crisis or guardrail additionally require harmful-class recall (TPR) at or above the rule's
threshold (docs/protocol.md steps 2 and 6).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

PACKS_DIR = Path(__file__).resolve().parents[1] / "packs"
SAFETY_TAGS = frozenset({"crisis", "guardrail"})


@dataclass(frozen=True)
class Rule:
    id: str
    tags: frozenset[str] = frozenset()
    threshold: float = 0.95

    @property
    def is_safety(self) -> bool:
        return bool(self.tags & SAFETY_TAGS)


@dataclass(frozen=True)
class Pack:
    name: str
    min_kappa_to_gate: float
    rules: dict[str, Rule] = field(default_factory=dict)

    def rule(self, criterion: str) -> Rule | None:
        return self.rules.get(criterion)


def load_pack(name_or_path: str | Path) -> Pack:
    """Load packs/<name>.yaml or an explicit path."""
    path = Path(name_or_path)
    if not path.exists():
        path = PACKS_DIR / f"{name_or_path}.yaml"
    data = yaml.safe_load(path.read_text())
    rules = {
        r["id"]: Rule(
            id=r["id"],
            tags=frozenset(r.get("tags", [])),
            threshold=float(r.get("threshold", 0.95)),
        )
        for r in data.get("rules", [])
    }
    return Pack(name=data["name"], min_kappa_to_gate=float(data["min_kappa_to_gate"]), rules=rules)
