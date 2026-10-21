"""Judges: one yes/no verdict per criterion, deterministic, LLM-backed, or discriminative."""

from core.judges.base import Fingerprint, Judge, Verdict
from core.judges.deterministic import DeterministicJudge
from core.judges.discriminative import DecisionsJudge, DiscriminativeJudge, JevJudge
from core.judges.llm import LLMJudge, LLMJudgeConfig

JUDGES = {
    "deterministic": DeterministicJudge,
    "llm": LLMJudge,
    "jev": JevJudge,
    "decisions": DecisionsJudge,
}

__all__ = [
    "JUDGES",
    "DecisionsJudge",
    "DeterministicJudge",
    "DiscriminativeJudge",
    "Fingerprint",
    "JevJudge",
    "Judge",
    "LLMJudge",
    "LLMJudgeConfig",
    "Verdict",
]
