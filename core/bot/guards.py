"""Deterministic guards. No model calls here."""

from __future__ import annotations

import re

MEDICATIONS = [
    "adderall", "ritalin", "concerta", "vyvanse", "strattera", "intuniv", "focalin",
    "dexedrine", "qelbree", "wellbutrin", "abilify", "risperdal", "zoloft", "prozac",
    "lexapro", "methylphenidate", "amphetamine", "dexamphetamine", "dextroamphetamine",
    "lisdexamfetamine", "atomoxetine", "guanfacine", "clonidine", "viloxazine",
    "bupropion", "aripiprazole", "risperidone", "sertraline", "fluoxetine",
    "escitalopram", "modafinil", "melatonin",
]  # fmt: skip

MEDICATION_RE = re.compile(r"\b(" + "|".join(MEDICATIONS) + r")s?\b", re.IGNORECASE)
DOSING_RE = re.compile(
    r"\b\d+(\.\d+)?\s?(mg|mcg|milligrams?|micrograms?)\b"
    r"|\b(dose|doses|dosage|dosing|titrat\w*)\b"
    r"|\bhow (much|many)\b.{0,40}\b(pill|pills|tablet|tablets|meds|medication|medicine)\b"
    r"|\b(double|skip|skipped|miss|missed|take|taking)\b.{0,20}"
    r"\b(pill|pills|tablet|tablets|meds|medication|medicine)\b",
    re.IGNORECASE,
)
CRISIS_RE = re.compile(
    r"\b(kill(ing)? myself|suicid\w*|end (my|it all)\b|want(ing)? to die|"
    r"self[- ]?harm\w*|hurt(ing)? myself|cut(ting)? myself|"
    r"(don'?t|do not) want to (be alive|live|wake up)|overdos\w*|better off dead)\b",
    re.IGNORECASE,
)

GUARD_REFUSAL = (
    "I can't help with medication or dosing questions. Please talk to your prescriber "
    "or a pharmacist about that. I'm happy to help with non-medical coping strategies."
)
CRISIS_RESPONSE = (
    "It sounds like you might be going through something really hard right now. "
    "Please reach out for support: call or text 988 (Suicide and Crisis Lifeline, US), "
    "or text HOME to 741741. Outside the US, contact your local emergency number. "
    "You deserve support from a person right now."
)


def medication_match(text: str) -> bool:
    return bool(MEDICATION_RE.search(text) or DOSING_RE.search(text))


def crisis_match(text: str) -> bool:
    return bool(CRISIS_RE.search(text))
