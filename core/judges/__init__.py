"""Judges: one yes/no verdict per criterion, deterministic or LLM-backed."""

from core.judges.base import Fingerprint, Judge, Verdict
from core.judges.deterministic import DeterministicJudge
from core.judges.llm import LLMJudge, LLMJudgeConfig

JUDGES = {"deterministic": DeterministicJudge, "llm": LLMJudge}

__all__ = [
    "JUDGES",
    "DeterministicJudge",
    "Fingerprint",
    "Judge",
    "LLMJudge",
    "LLMJudgeConfig",
    "Verdict",
]
