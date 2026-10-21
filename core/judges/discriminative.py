"""Discriminative judges: typed yes/no questions to a decision model, no generated text.

A rule's questions are atomic predicates in prompts/questions/<rule>.yaml. Every predicate
is asked twice in one call, in affirmative and negated phrasing, and the two probabilities
are averaged: p = (p_affirmative + 1 - p_negated) / 2. When both phrasings land on the same
side of 0.5 the question is flagged `phrasing_disagree`; docs/discriminative_judges.md has
why (Jev reads literally and answers negations less reliably).

The verdict is the YAML's fixed combination: `combine: any|all` over `of: [ids]`, and
`passes_when: true|false|<question id>`. The probability of the pass verdict is max (any)
or min (all) over the predicates, or P(combination == condition) when passes_when names a
question. Confidence is the minimum |2p - 1| over the questions used, which the TypeSafe
docs give as the Noul-comparable confidence. Both backends are pinned to a model id and the
id in each response is checked against the pin.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from core.judges.base import Fingerprint, Verdict
from core.telemetry import tracer

QUESTIONS_DIR = Path(__file__).resolve().parents[2] / "prompts" / "questions"
JEV_MODEL = "jev-1.13.0"
DECISIONS_MODEL = "gpt-6-luna"
AFF, NEG = "aff", "neg"
PHRASING_DISAGREE = "phrasing_disagree"
MODEL_MISMATCH = "model_mismatch"
REFUSAL = "refusal"


@dataclass(frozen=True)
class Question:
    id: str
    affirmative: str
    negated: str


@dataclass(frozen=True)
class QuestionSet:
    rule: str
    version: str
    questions: tuple[Question, ...]
    combine: str  # any | all
    of: tuple[str, ...]
    passes_when: bool | str  # True, False, or a question id the combination must equal
    source_hash: str

    @property
    def used(self) -> tuple[str, ...]:
        """Question ids the verdict depends on, in YAML order."""
        extra = (self.passes_when,) if isinstance(self.passes_when, str) else ()
        return tuple(self.of) + extra

    def by_id(self, qid: str) -> Question:
        return next(q for q in self.questions if q.id == qid)


def load_questions(criterion: str, questions_dir: Path = QUESTIONS_DIR) -> QuestionSet:
    path = questions_dir / f"{criterion}.yaml"
    raw = path.read_text()
    data = yaml.safe_load(raw)
    if data.get("rule") != criterion:
        raise ValueError(f"{path.name}: rule is {data.get('rule')!r}, expected {criterion!r}")
    questions = tuple(
        Question(str(q["id"]), str(q["affirmative"]).strip(), str(q["negated"]).strip())
        for q in data.get("questions", [])
    )
    ids = [q.id for q in questions]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError(f"{path.name}: questions must be non-empty with unique ids")
    verdict = data.get("verdict") or {}
    combine = verdict.get("combine")
    if combine not in ("any", "all"):
        raise ValueError(f"{path.name}: verdict.combine must be any or all, got {combine!r}")
    of = tuple(str(x) for x in verdict.get("of", []))
    unknown = [x for x in of if x not in ids]
    if not of or unknown:
        raise ValueError(f"{path.name}: verdict.of must name known questions; unknown {unknown}")
    passes_when = verdict.get("passes_when")
    if isinstance(passes_when, str):
        if passes_when not in ids:
            raise ValueError(f"{path.name}: passes_when names unknown question {passes_when!r}")
        if passes_when in of:
            raise ValueError(f"{path.name}: passes_when {passes_when!r} cannot also be in `of`")
    elif not isinstance(passes_when, bool):
        raise ValueError(f"{path.name}: passes_when must be true, false, or a question id")
    return QuestionSet(
        rule=criterion,
        version=str(data.get("version", "")),
        questions=questions,
        combine=combine,
        of=of,
        passes_when=passes_when,
        source_hash=hashlib.sha256(raw.encode()).hexdigest()[:12],
    )


# --- pure combination logic ---


def fold_phrasings(p_affirmative: float, p_negated: float) -> tuple[float, bool]:
    """(p that the predicate holds, phrasings disagree). The negated phrasing is a second
    vote on the same fact with its sign flipped."""
    p = (p_affirmative + (1.0 - p_negated)) / 2.0
    disagree = (p_affirmative >= 0.5) == (p_negated >= 0.5)
    return p, disagree


def combine(qs: QuestionSet, p: Mapping[str, float]) -> tuple[bool, float, float]:
    """(verdict, P(pass), confidence) from per-question probabilities that each predicate
    holds. The verdict is P(pass) >= 0.5, which agrees with evaluating the booleans."""
    values = [p[qid] for qid in qs.of]
    p_combo = max(values) if qs.combine == "any" else min(values)
    if qs.passes_when is True:
        p_pass = p_combo
    elif qs.passes_when is False:
        p_pass = 1.0 - p_combo
    else:
        p_cond = p[qs.passes_when]
        p_pass = p_combo * p_cond + (1.0 - p_combo) * (1.0 - p_cond)
    confidence = min(abs(2.0 * p[qid] - 1.0) for qid in qs.used)
    return p_pass >= 0.5, p_pass, confidence


def state_text(input: str, output: str) -> str:
    return f"<user_turn>\n{input}\n</user_turn>\n\n<bot_reply>\n{output}\n</bot_reply>"


def question_keys(qs: QuestionSet) -> dict[str, str]:
    """Every phrasing sent in one call: {"<id>__aff": text, "<id>__neg": text}."""
    keys: dict[str, str] = {}
    for q in qs.questions:
        keys[f"{q.id}__{AFF}"] = q.affirmative
        keys[f"{q.id}__{NEG}"] = q.negated
    return keys


class DiscriminativeJudge:
    """Shared base. Subclasses implement `ask` for one backend."""

    name = "discriminative"
    model: str = ""
    criteria = tuple(sorted(p.stem for p in QUESTIONS_DIR.glob("*.yaml")))

    def __init__(self, criterion: str, questions_dir: Path = QUESTIONS_DIR) -> None:
        self.criterion = criterion
        self.questions = load_questions(criterion, questions_dir)
        self.last_model: str | None = None

    def ask(self, state: str, questions: Mapping[str, str]) -> tuple[dict[str, float], list[str]]:
        """Return ({key: P(true)}, flags) for every key in `questions`."""
        raise NotImplementedError

    def judge(self, input: str, output: str, retrieved: list[str]) -> Verdict:
        qs = self.questions
        keys = question_keys(qs)
        with tracer().start_as_current_span(f"judge.{self.name}") as span:
            span.set_attribute("judge.criterion", self.criterion)
            span.set_attribute("judge.model", self.model)
            span.set_attribute("judge.questions", len(keys))
            probs, flags = self.ask(state_text(input, output), keys)
            p: dict[str, float] = {}
            for q in qs.questions:
                p[q.id], disagree = fold_phrasings(probs[f"{q.id}__{AFF}"], probs[f"{q.id}__{NEG}"])
                if disagree:
                    flags.append(f"{PHRASING_DISAGREE}:{q.id}")
            if self.last_model and self.last_model != self.model:
                flags.append(f"{MODEL_MISMATCH}:{self.last_model}")
            verdict, p_pass, confidence = combine(qs, p)
            span.set_attribute("judge.verdict", verdict)
            span.set_attribute("judge.probability", p_pass)
            span.set_attribute("judge.flags", flags)
        return {
            "verdict": verdict,
            "confidence": confidence,
            "probability": p_pass,
            "flags": flags,
        }

    def fingerprint(self) -> Fingerprint:
        return Fingerprint(
            model_snapshot=self.model,
            prompt_hash=self.questions.source_hash,
            rubric_version=self.questions.version,
            temperature=None,
        )


class JevJudge(DiscriminativeJudge):
    """TypeSafe Jev through typesafe-sdk, Noul primitive, one system_one call per verdict."""

    name = "jev"
    model = JEV_MODEL

    def __init__(
        self,
        criterion: str,
        client: Any | None = None,
        model: str = JEV_MODEL,
        questions_dir: Path = QUESTIONS_DIR,
    ) -> None:
        super().__init__(criterion, questions_dir)
        self.model = model
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            from typesafe_sdk import TypeSafeClient

            key = os.environ.get("TYPESAFE_API_KEY") or os.environ.get("JEV_API_KEY")
            if not key:
                raise RuntimeError("TYPESAFE_API_KEY (or JEV_API_KEY) is not set")
            self._client = TypeSafeClient(api_key=key, model=self.model)
        return self._client

    def ask(self, state: str, questions: Mapping[str, str]) -> tuple[dict[str, float], list[str]]:
        from typesafe_sdk import Noul

        response = self.client.system_one(
            state, {k: Noul(instructions=text) for k, text in questions.items()}, model=self.model
        )
        self.last_model = getattr(response, "model", None)
        nouls = response.nouls
        return {k: float(nouls[k].noul) for k in questions}, []


class DecisionsJudge(DiscriminativeJudge):
    """OpenAI Decisions (/v1/decisions) through the openai SDK, predicate questions."""

    name = "decisions"
    model = DECISIONS_MODEL

    def __init__(
        self,
        criterion: str,
        client: Any | None = None,
        model: str = DECISIONS_MODEL,
        questions_dir: Path = QUESTIONS_DIR,
    ) -> None:
        super().__init__(criterion, questions_dir)
        self.model = model
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is not set")
            self._client = OpenAI()
        return self._client

    def ask(self, state: str, questions: Mapping[str, str]) -> tuple[dict[str, float], list[str]]:
        decision = self.client.decisions.create(
            model=self.model,
            input=state,
            questions=[
                {"type": "predicate", "name": k, "instructions": text}
                for k, text in questions.items()
            ],
        )
        self.last_model = getattr(decision, "model", None)
        probs: dict[str, float] = {}
        flags: list[str] = []
        for answer in decision.answers:
            if answer.type == "predicate":
                probs[answer.name] = float(answer.probability)
            else:
                flags.append(f"{REFUSAL}:{answer.name}")
        missing = [k for k in questions if k not in probs]
        for k in missing:  # a refusal or an unanswered question is "no information"
            probs[k] = 0.5
            if f"{REFUSAL}:{k}" not in flags:
                flags.append(f"unanswered:{k}")
        return probs, flags
