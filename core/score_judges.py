"""Score judges against human labels, following docs/protocol.md.

For each judge x criterion: three passes over the labelled traces as one LangSmith
experiment (evaluate(), num_repetitions=3, caching disabled); then Cohen's kappa with a
clustered bootstrap interval, exact agreement, the confusion matrix with the failing class
as positive, test-retest across passes, and length bias. One scorecard JSON per pair lands
in out/scorecards/, the same JSON is attached as feedback on a sentinel run, and kappa.png
plots every kappa with its interval.

Judges that return a probability (core/judges/discriminative.py) also get a calibration
section: a 10-bin reliability diagram, ECE and Brier, and a selective-prediction table fitted
by Platt scaling on half the labels and reported on the other half. Every judge is run on the
injection fixture (tests/fixtures/injection.json) and the scorecard counts the transcripts
whose verdict the planted instruction flipped.

The metric functions at the top are pure and take plain lists, so they are testable without
LangSmith. The failing class (human label False) is the positive class throughout.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from langsmith.evaluation import evaluate

from core.judges import JUDGES, Judge
from core.judges.base import Verdict
from core.label import CONVERSATIONS_PATH, LABELS_PATH, read_jsonl
from core.packs import Pack

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "out" / "scorecards"
DATASET_NAME = "support-bot-labelled"
REPETITIONS = 3
BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 0
# docs/protocol.md step 3: 0.9 is voirdire's convention, the literature disagrees.
TEST_RETEST_BAR = 0.9
LENGTH_R_BAR = 0.2
# docs/protocol.md step 0: at least 20 of each class. Below this, kappa is not reported.
MIN_FAILS = 20
INSUFFICIENT = "INSUFFICIENT_FAILS"
SCORED = "SCORED"
# docs/protocol.md step 5: ten bins. Selective thresholds 0.5..0.95 on calibrated confidence.
CALIBRATION_BINS = 10
SELECTIVE_THRESHOLDS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))
SPLIT_SEED = 0
INJECTION_PATH = ROOT / "tests" / "fixtures" / "injection.json"

# --- metrics (pure) ---


def cohens_kappa(a: Sequence[bool], b: Sequence[bool]) -> float | None:
    """Cohen's kappa for two binary raters. None when chance agreement is 1 (undefined)."""
    n = len(a)
    if n == 0 or n != len(b):
        return None
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if math.isclose(pe, 1.0):
        return None
    return (po - pe) / (1 - pe)


def exact_agreement(a: Sequence[bool], b: Sequence[bool]) -> float | None:
    if not a:
        return None
    return sum(x == y for x, y in zip(a, b, strict=True)) / len(a)


def confusion(human: Sequence[bool], judge: Sequence[bool]) -> dict:
    """Counts and rates with the failing class (label False) as positive."""
    tp = sum((not h) and (not j) for h, j in zip(human, judge, strict=True))
    fn = sum((not h) and j for h, j in zip(human, judge, strict=True))
    fp = sum(h and (not j) for h, j in zip(human, judge, strict=True))
    tn = sum(h and j for h, j in zip(human, judge, strict=True))
    positives, negatives = tp + fn, fp + tn
    n = positives + negatives
    return {
        "positive_class": "fail",
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "prevalence": positives / n if n else None,
        "tpr": tp / positives if positives else None,
        "tnr": tn / negatives if negatives else None,
    }


def bootstrap_kappa(
    human: Sequence[bool],
    judge: Sequence[bool],
    clusters: Sequence[str],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """95% percentile interval for kappa, resampling whole clusters (conversations).

    Every observation of a drawn cluster comes along, so the three passes of one trace
    never split. Resamples where kappa is undefined are dropped and counted.
    """
    by_cluster: dict[str, list[tuple[bool, bool]]] = {}
    for h, j, c in zip(human, judge, clusters, strict=True):
        by_cluster.setdefault(c, []).append((h, j))
    ids = list(by_cluster)
    rng = random.Random(seed)
    kappas: list[float] = []
    undefined = 0
    for _ in range(resamples):
        pairs = [p for c in rng.choices(ids, k=len(ids)) for p in by_cluster[c]]
        k = cohens_kappa([h for h, _ in pairs], [j for _, j in pairs])
        if k is None:
            undefined += 1
        else:
            kappas.append(k)
    kappas.sort()
    lower = upper = None
    if kappas:
        lower = kappas[int(0.025 * (len(kappas) - 1))]
        upper = kappas[int(math.ceil(0.975 * (len(kappas) - 1)))]
    return {
        "lower": lower,
        "upper": upper,
        "resamples": resamples,
        "undefined_kappa_resamples": undefined,
        "clusters": len(ids),
        "seed": seed,
    }


def majority(votes: Sequence[bool]) -> bool:
    return sum(votes) * 2 > len(votes)


def retest_agreement(passes_by_item: Sequence[Sequence[bool]]) -> dict:
    """Mean pairwise agreement between passes, and the share of items with any disagreement."""
    agreements: list[float] = []
    disagreeing = 0
    for votes in passes_by_item:
        if len(votes) < 2:
            continue
        pairs = [(a, b) for i, a in enumerate(votes) for b in votes[i + 1 :]]
        agreements.append(sum(a == b for a, b in pairs) / len(pairs))
        disagreeing += len(set(votes)) > 1
    n = len(agreements)
    return {
        "agreement": sum(agreements) / n if n else None,
        "any_disagreement_count": disagreeing,
        "any_disagreement_share": disagreeing / n if n else None,
        "items": n,
        "passes": max((len(v) for v in passes_by_item), default=0),
    }


def pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    n = len(x)
    if n < 2:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        return None
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    return sxy / math.sqrt(sxx * syy)


def length_bias(
    human: Sequence[bool],
    judge: Sequence[bool],
    lengths: Sequence[int],
    clusters: Sequence[str] | None = None,
) -> dict:
    """Kappa on the longest third vs the shortest third of outputs, and r(length, verdict).

    Sorting is by (length, cluster) so a trace's passes land in the same third.
    """
    clusters = clusters or [""] * len(lengths)
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i], clusters[i]))
    third = len(order) // 3
    shortest, longest = order[:third], order[len(order) - third :] if third else []

    def sub(idx: list[int]) -> float | None:
        return cohens_kappa([human[i] for i in idx], [judge[i] for i in idx])

    return {
        "kappa_shortest_third": sub(shortest),
        "kappa_longest_third": sub(longest),
        "third_size_calls": third,
        "length_verdict_r": pearson([float(n) for n in lengths], [float(v) for v in judge]),
    }


# --- calibration (pure) ---


def reliability_bins(
    probs: Sequence[float], labels: Sequence[bool], n_bins: int = CALIBRATION_BINS
) -> list[dict]:
    """Equal-width bins over P(pass). Each: lo, hi, n, mean_prob, frac_pass (None if empty)."""
    bins = [{"lo": i / n_bins, "hi": (i + 1) / n_bins, "n": 0, "sum_p": 0.0, "sum_y": 0}
            for i in range(n_bins)]  # fmt: skip
    for p, y in zip(probs, labels, strict=True):
        b = bins[min(int(p * n_bins), n_bins - 1)]
        b["n"] += 1
        b["sum_p"] += p
        b["sum_y"] += int(y)
    out = []
    for b in bins:
        n = b["n"]
        out.append({
            "lo": b["lo"], "hi": b["hi"], "n": n,
            "mean_prob": b["sum_p"] / n if n else None,
            "frac_pass": b["sum_y"] / n if n else None,
        })  # fmt: skip
    return out


def expected_calibration_error(bins: Sequence[dict], n_total: int) -> float | None:
    if not n_total:
        return None
    return sum(b["n"] / n_total * abs(b["frac_pass"] - b["mean_prob"]) for b in bins if b["n"])


def brier_score(probs: Sequence[float], labels: Sequence[bool]) -> float | None:
    if not probs:
        return None
    return sum((p - int(y)) ** 2 for p, y in zip(probs, labels, strict=True)) / len(probs)


def split_halves(
    ids: Sequence[str], labels: Mapping[str, bool], seed: int = SPLIT_SEED
) -> tuple[list[str], list[str]]:
    """(fit, held) halves of the trace ids, stratified by label so fails land in both
    whenever there are at least two. Deterministic for a seed."""
    rng = random.Random(seed)
    fit: list[str] = []
    held: list[str] = []
    for want in (False, True):
        group = sorted(t for t in ids if labels[t] is want)
        rng.shuffle(group)
        fit.extend(group[0::2])
        held.extend(group[1::2])
    return sorted(fit), sorted(held)


def _logit(p: float, eps: float = 1e-4) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def _sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z)) if z > -700 else 0.0


def platt_fit(
    probs: Sequence[float], labels: Sequence[bool], iters: int = 50, l2: float = 1e-3
) -> tuple[float, float]:
    """Platt scaling, sigmoid(a * logit(p) + b), by Newton's method on log loss with Platt's
    smoothed targets so a half with no fails still yields a finite map. Returns (a, b);
    (1, 0) is the identity."""
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if not labels:
        return 1.0, 0.0
    t_pos, t_neg = (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2)
    xs = [_logit(p) for p in probs]
    ts = [t_pos if y else t_neg for y in labels]
    a, b = 1.0, 0.0
    for _ in range(iters):
        g_a = g_b = h_aa = h_ab = h_bb = 0.0
        for x, t in zip(xs, ts, strict=True):
            q = _sigmoid(a * x + b)
            d = q - t
            w = q * (1 - q)
            g_a += d * x
            g_b += d
            h_aa += w * x * x
            h_ab += w * x
            h_bb += w
        g_a += l2 * (a - 1)
        g_b += l2 * b
        h_aa += l2
        h_bb += l2
        det = h_aa * h_bb - h_ab * h_ab
        if det <= 1e-12:
            break
        da = (h_bb * g_a - h_ab * g_b) / det
        db = (h_aa * g_b - h_ab * g_a) / det
        a, b = a - da, b - db
        if abs(da) < 1e-8 and abs(db) < 1e-8:
            break
    return a, b


def platt_apply(p: float, a: float, b: float) -> float:
    return _sigmoid(a * _logit(p) + b)


def selective_table(
    probs: Sequence[float],
    labels: Sequence[bool],
    thresholds: Sequence[float] = SELECTIVE_THRESHOLDS,
) -> list[dict]:
    """For each confidence threshold: coverage (share of items the judge decides rather than
    defers) and the false-negative rate on the failing class among decided items (human
    fail, judge pass). probs are P(pass); confidence is max(p, 1 - p)."""
    rows = []
    n = len(probs)
    for t in thresholds:
        decided = [(p, y) for p, y in zip(probs, labels, strict=True) if max(p, 1 - p) >= t]
        fails = [(p, y) for p, y in decided if not y]
        missed = sum(p >= 0.5 for p, _ in fails)
        rows.append({
            "threshold": t,
            "coverage": len(decided) / n if n else None,
            "n_decided": len(decided),
            "fails_decided": len(fails),
            "fnr_fail": missed / len(fails) if fails else None,
        })  # fmt: skip
    return rows


# --- labelled traces and the LangSmith dataset ---


@dataclass(frozen=True)
class LabelledRow:
    trace_id: str
    input: str
    output: str
    retrieved_doc_ids: list[str]
    scenario: str
    labels: dict[str, bool]
    labeller: str
    prompt_version: str = ""
    model: str = ""
    config: str = "baseline"


def load_labelled(
    conversations_path: Path = CONVERSATIONS_PATH, labels_path: Path = LABELS_PATH
) -> list[LabelledRow]:
    conversations = {r["trace_id"]: r for r in read_jsonl(conversations_path)}
    rows = []
    for rec in read_jsonl(labels_path):
        conv = conversations.get(rec["trace_id"])
        if conv is None:
            continue
        rows.append(
            LabelledRow(
                trace_id=rec["trace_id"],
                input=conv["input"],
                output=conv["output"],
                retrieved_doc_ids=list(conv.get("retrieved_doc_ids", [])),
                scenario=conv.get("scenario", ""),
                labels=dict(rec["labels"]),
                labeller=rec.get("labeller", "human"),
                prompt_version=conv.get("prompt_version", ""),
                model=conv.get("model", ""),
                config=conv.get("config", "baseline"),
            )
        )
    return rows


def fail_counts(rows: Iterable[LabelledRow]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in rows:
        for key, value in r.labels.items():
            counts[key] = counts.get(key, 0) + (value is False)
    return counts


def labels_tag(rows: Iterable[LabelledRow]) -> str:
    """Dataset version tag derived from the label set: labels-<sha256 prefix>."""
    canon = json.dumps(sorted((r.trace_id, sorted(r.labels.items())) for r in rows))
    return "labels-" + hashlib.sha256(canon.encode()).hexdigest()[:8]


def example_payload(row: LabelledRow) -> dict:
    return {
        "inputs": {
            "input": row.input,
            "output": row.output,
            "retrieved_doc_ids": row.retrieved_doc_ids,
        },
        "outputs": dict(row.labels),
        "metadata": {
            "trace_id": row.trace_id,
            "scenario": row.scenario,
            "labeller": row.labeller,
            "prompt_version": row.prompt_version,
            "model": row.model,
            "config": row.config,
        },
    }


def sync_dataset(rows: list[LabelledRow], client, dataset_name: str = DATASET_NAME) -> dict:
    """Upsert the labelled traces as a LangSmith dataset keyed by trace_id, then tag it.

    Inputs are the conversation (the judge's input), outputs are the human labels (the
    reference). Returns {name, id, version_tag, as_of, n, created, updated}.
    """
    if client.has_dataset(dataset_name=dataset_name):
        dataset = client.read_dataset(dataset_name=dataset_name)
    else:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="Human-labelled bot turns. inputs = conversation, outputs = labels.",
        )
    existing = {
        (ex.metadata or {}).get("trace_id"): ex
        for ex in client.list_examples(dataset_id=dataset.id)
    }
    created = [r for r in rows if r.trace_id not in existing]
    updated = [
        r
        for r in rows
        if r.trace_id in existing and (existing[r.trace_id].outputs or {}) != r.labels
    ]
    if created:
        client.create_examples(
            dataset_id=dataset.id, examples=[example_payload(r) for r in created]
        )
    if updated:
        client.update_examples(
            dataset_id=dataset.id,
            updates=[{"id": existing[r.trace_id].id, **example_payload(r)} for r in updated],
        )
    tag = labels_tag(rows)
    version = client.read_dataset_version(dataset_id=dataset.id, tag="latest")
    if tag not in (version.tags or []):
        client.update_dataset_tag(dataset_id=dataset.id, as_of=version.as_of, tag=tag)
    return {
        "name": dataset_name,
        "id": str(dataset.id),
        "version_tag": tag,
        "as_of": version.as_of.isoformat(),
        "n": len(rows),
        "created": len(created),
        "updated": len(updated),
    }


# --- the experiment ---


def local_examples(rows: list[LabelledRow], dataset_name: str = DATASET_NAME) -> list:
    """Example objects built locally, for offline runs. Ids derive from trace ids."""
    from langsmith import schemas as ls

    dataset_id = uuid.uuid5(uuid.NAMESPACE_URL, f"voirdire/{dataset_name}")
    now = datetime.now(UTC)
    return [
        ls.Example(
            id=uuid.uuid5(dataset_id, r.trace_id),
            dataset_id=dataset_id,
            created_at=now,
            **example_payload(r),
        )
        for r in rows
    ]


def make_target(judge: Judge) -> Callable[[dict], dict]:
    def target(inputs: dict) -> dict:
        v = judge.judge(
            inputs["input"], inputs["output"], list(inputs.get("retrieved_doc_ids", []))
        )
        out = {"verdict": v["verdict"], "confidence": v["confidence"]}
        if v.get("probability") is not None:
            out["probability"] = v["probability"]
        if v.get("flags"):
            out["flags"] = list(v["flags"])
        return out

    target.__name__ = f"{judge.name}_{judge.criterion}"
    return target


def make_row_evaluator(criterion: str) -> Callable:
    def agrees_with_human(run, example) -> dict:
        verdict = (run.outputs or {}).get("verdict")
        human = (example.outputs or {}).get(criterion)
        if verdict is None or human is None:
            return {"key": "agrees_with_human", "score": None}
        return {"key": "agrees_with_human", "score": int(verdict == human)}

    return agrees_with_human


def _collect(
    runs, examples, criterion: str
) -> tuple[list[bool], list[bool], dict[str, list[bool]]]:
    """Pooled (human, judge) pairs over every run, plus per-example pass lists."""
    by_id = {str(ex.id): ex for ex in examples}
    human: list[bool] = []
    judge: list[bool] = []
    passes: dict[str, list[bool]] = {}
    for run in runs:
        ex = by_id.get(str(run.reference_example_id))
        verdict = (run.outputs or {}).get("verdict") if run.outputs else None
        if ex is None or verdict is None or criterion not in (ex.outputs or {}):
            continue
        human.append(bool(ex.outputs[criterion]))
        judge.append(bool(verdict))
        passes.setdefault(str(ex.id), []).append(bool(verdict))
    return human, judge, passes


def make_summary_evaluators(criterion: str, min_fails: int = MIN_FAILS) -> list[Callable]:
    """Kappa and test-retest as experiment-level feedback, so they show in Compare."""

    def kappa(runs, examples) -> dict:
        human, judge, _ = _collect(runs, examples, criterion)
        # evaluate() repeats each example once per repetition; count distinct items
        unique = {str(ex.id): ex for ex in examples}.values()
        fails = sum(not (ex.outputs or {}).get(criterion, True) for ex in unique)
        if fails < min_fails:
            return {"key": "kappa", "score": None, "comment": f"{INSUFFICIENT}: {fails}"}
        k = cohens_kappa(human, judge)
        return {"key": "kappa", "score": k, "comment": None if k is not None else "undefined"}

    def test_retest_agreement(runs, examples) -> dict:
        _, _, passes = _collect(runs, examples, criterion)
        return {
            "key": "test_retest",
            "score": retest_agreement(list(passes.values()))["agreement"],
        }

    return [kappa, test_retest_agreement]


@dataclass
class ExperimentOutcome:
    experiment_name: str
    url: str | None
    # trace_id -> verdict per pass, in pass order
    passes: dict[str, list[bool]] = field(default_factory=dict)
    human: dict[str, bool] = field(default_factory=dict)
    lengths: dict[str, int] = field(default_factory=dict)
    # trace_id -> P(pass) per pass, only for judges that return one
    probabilities: dict[str, list[float]] = field(default_factory=dict)
    flag_counts: dict[str, int] = field(default_factory=dict)
    errors: int = 0


def disable_caching() -> None:
    """No LLM response cache and no LangSmith test cache: every pass is a fresh call."""
    from langchain_core.globals import set_llm_cache

    set_llm_cache(None)
    os.environ.pop("LANGSMITH_TEST_CACHE", None)


def run_experiment(
    judge: Judge,
    examples: list,
    client,
    dataset_info: dict,
    repetitions: int = REPETITIONS,
    evaluate_fn: Callable | None = None,
    max_concurrency: int = 4,
    min_fails: int = MIN_FAILS,
    offline: bool = False,
) -> ExperimentOutcome:
    """Run the experiment. Evaluator tracing is off: the row evaluator is an equality
    check, and tracing it would double the trace count for no information. With
    offline=True nothing is uploaded and the experiment exists only in the scorecard."""
    evaluate_fn = evaluate_fn or evaluate
    disable_caching()
    fp = judge.fingerprint()
    criterion = judge.criterion
    prefix = f"judge-{judge.name}-{criterion}"
    results = evaluate_fn(
        make_target(judge),
        data=examples,
        evaluators=[make_row_evaluator(criterion)],
        summary_evaluators=make_summary_evaluators(criterion, min_fails),
        experiment_prefix=prefix,
        num_repetitions=repetitions,
        max_concurrency=max_concurrency,
        client=client,
        upload_results=not offline,
        disable_evaluator_tracing=True,
        metadata={
            "judge": judge.name,
            "criterion": criterion,
            "repetitions": repetitions,
            "caching": "disabled",
            "dataset_version_tag": dataset_info.get("version_tag"),
            **asdict(fp),
        },
        description=f"Judge qualification: {judge.name} on {criterion} (docs/protocol.md)",
    )
    if offline:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        outcome = ExperimentOutcome(experiment_name=f"{prefix}-offline-{stamp}", url=None)
    else:
        try:
            url = results.url
        except Exception:  # not started (upload off) or no tenant info
            url = None
        outcome = ExperimentOutcome(experiment_name=results.experiment_name, url=url)
    for r in results:
        run, example = r["run"], r["example"]
        trace_id = (example.metadata or {}).get("trace_id") or str(example.id)
        verdict = (run.outputs or {}).get("verdict") if run.outputs else None
        if verdict is None or criterion not in (example.outputs or {}):
            outcome.errors += 1
            continue
        outcome.passes.setdefault(trace_id, []).append(bool(verdict))
        outcome.human[trace_id] = bool(example.outputs[criterion])
        outcome.lengths[trace_id] = len(example.inputs.get("output", ""))
        prob = run.outputs.get("probability")
        if prob is not None:
            outcome.probabilities.setdefault(trace_id, []).append(float(prob))
        for flag in run.outputs.get("flags") or []:
            kind = str(flag).split(":", 1)[0]
            outcome.flag_counts[kind] = outcome.flag_counts.get(kind, 0) + 1
    return outcome


# --- calibration section and injection fixture ---


def calibration_section(outcome: ExperimentOutcome) -> dict | None:
    """None for judges without a probability. ECE, Brier and the reliability bins pool
    every pass; the selective table is fitted (Platt) on one half of the traces and
    reported on the other, both halves stratified by human label."""
    ids = sorted(t for t in outcome.passes if t in outcome.probabilities)
    if not ids:
        return None
    probs = [p for t in ids for p in outcome.probabilities[t]]
    labels = [outcome.human[t] for t in ids for _ in outcome.probabilities[t]]
    bins = reliability_bins(probs, labels)
    fit_ids, held_ids = split_halves(ids, outcome.human)
    fit_p = [p for t in fit_ids for p in outcome.probabilities[t]]
    fit_y = [outcome.human[t] for t in fit_ids for _ in outcome.probabilities[t]]
    a, b = platt_fit(fit_p, fit_y)
    held_p = [platt_apply(p, a, b) for t in held_ids for p in outcome.probabilities[t]]
    held_y = [outcome.human[t] for t in held_ids for _ in outcome.probabilities[t]]
    return {
        "n_calls": len(probs),
        "bins": bins,
        "ece": expected_calibration_error(bins, len(probs)),
        "brier": brier_score(probs, labels),
        "selective": {
            "fit": {"method": "platt", "a": a, "b": b, "n": len(fit_ids),
                    "n_fail": sum(not outcome.human[t] for t in fit_ids)},
            "held": {"n": len(held_ids), "n_fail": sum(not outcome.human[t] for t in held_ids)},
            "split_seed": SPLIT_SEED,
            "table": selective_table(held_p, held_y),
        },
        "flag_counts": dict(sorted(outcome.flag_counts.items())),
        "note": "probability is P(pass); ECE over 10 equal-width bins; selective table on "
        "the held-out half after Platt scaling fitted on the other half",
    }  # fmt: skip


def load_injection(path: Path = INJECTION_PATH) -> list[dict]:
    return json.loads(path.read_text())


def injection_flips(judge: Judge, fixture: Sequence[dict]) -> dict:
    """Judge every fixture transcript clean and with the planted instruction appended.
    A flip is a verdict that changed; `clean_wrong` counts clean verdicts that already
    disagree with the fixture's expected label for this criterion."""
    items = []
    for item in fixture:
        expected = item["expected"].get(judge.criterion)
        clean: Verdict = judge.judge(item["input"], item["output"], [])
        injected: Verdict = judge.judge(
            item["input"], item["output"] + "\n\n" + item["injection"], []
        )
        items.append({
            "id": item["id"],
            "expected": expected,
            "clean": clean["verdict"],
            "injected": injected["verdict"],
            "flipped": clean["verdict"] != injected["verdict"],
            "clean_probability": clean.get("probability"),
            "injected_probability": injected.get("probability"),
        })  # fmt: skip
    return {
        "n": len(items),
        "flipped": sum(i["flipped"] for i in items),
        "clean_wrong": sum(
            i["expected"] is not None and i["clean"] != i["expected"] for i in items
        ),
        "items": items,
    }


# --- scorecard ---


def build_scorecard(
    judge: Judge, outcome: ExperimentOutcome, dataset_info: dict, min_fails: int = MIN_FAILS
) -> dict:
    """One scorecard. With fewer than `min_fails` human fails the status is
    INSUFFICIENT_FAILS and every kappa is left out rather than reported as a number."""
    fp = judge.fingerprint()
    ids = sorted(outcome.passes)
    human, verdicts, clusters, lengths = [], [], [], []
    for tid in ids:
        for v in outcome.passes[tid]:
            human.append(outcome.human[tid])
            verdicts.append(v)
            clusters.append(tid)
            lengths.append(outcome.lengths[tid])
    n_pass = max((len(outcome.passes[t]) for t in ids), default=0)
    human_items = [outcome.human[t] for t in ids]
    n_fail = sum(not h for h in human_items)
    enough = n_fail >= min_fails
    status = SCORED if enough else INSUFFICIENT

    if enough:
        boot = bootstrap_kappa(human, verdicts, clusters)
        agreement = {
            "kappa": cohens_kappa(human, verdicts),
            "kappa_ci": [boot["lower"], boot["upper"]],
            "kappa_per_pass": [
                cohens_kappa(
                    [outcome.human[t] for t in ids if len(outcome.passes[t]) > i],
                    [outcome.passes[t][i] for t in ids if len(outcome.passes[t]) > i],
                )
                for i in range(n_pass)
            ],
            "kappa_majority": cohens_kappa(human_items, [majority(outcome.passes[t]) for t in ids]),
            "exact": exact_agreement(human, verdicts),
            "bootstrap": boot,
        }
        bias = {**length_bias(human, verdicts, lengths, clusters), "r_bar": LENGTH_R_BAR}
    else:
        agreement = {
            "kappa": None,
            "kappa_ci": [None, None],
            "kappa_per_pass": [],
            "kappa_majority": None,
            "exact": exact_agreement(human, verdicts),
            "bootstrap": None,
            "skipped": f"{INSUFFICIENT}: {n_fail} labelled fails, {min_fails} needed",
        }
        bias = {
            "kappa_shortest_third": None,
            "kappa_longest_third": None,
            "third_size_calls": None,
            "length_verdict_r": pearson([float(n) for n in lengths], [float(v) for v in verdicts]),
            "r_bar": LENGTH_R_BAR,
        }

    return {
        "judge": judge.name,
        "criterion": judge.criterion,
        "status": status,
        "computed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n": len(ids),
        "n_fail": n_fail,
        "min_fails": min_fails,
        "n_calls": len(verdicts),
        "errors": outcome.errors,
        "judge_config": {
            "model_snapshot": fp.model_snapshot,
            "prompt_hash": fp.prompt_hash,
            "rubric_version": fp.rubric_version,
            "temperature": fp.temperature,
        },
        "dataset": {
            "name": dataset_info.get("name"),
            "version_tag": dataset_info.get("version_tag"),
            "as_of": dataset_info.get("as_of"),
        },
        "experiment": {
            "name": outcome.experiment_name,
            "url": outcome.url,
            "repetitions": n_pass,
            "caching": "disabled",
        },
        "agreement": agreement,
        "confusion": confusion(human, verdicts),
        "consistency": {
            **retest_agreement([outcome.passes[t] for t in ids]),
            "bar": TEST_RETEST_BAR,
            "bar_note": "0.9 is voirdire's convention; cited bars range 0.8 to 0.95",
        },
        "bias": bias,
        "calibration": calibration_section(outcome),
        "notes": [
            "Failing class (human label False) is the positive class for TPR and TNR.",
            "Kappa thresholds are content-analysis conventions, not LLM-judge findings.",
            "Kappa, exact, and the confusion matrix pool every pass; the bootstrap "
            "resamples conversations so a trace's passes stay together.",
            f"Kappa is reported only with at least {min_fails} labelled fails.",
            "Calibration is null for judges without a probability.",
        ],
    }


def scorecard_path(card: dict, out_dir: Path = OUT_DIR) -> Path:
    return out_dir / f"{card['judge']}_{card['criterion']}.json"


def write_scorecard(card: dict, out_dir: Path = OUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = scorecard_path(card, out_dir)
    path.write_text(json.dumps(card, indent=2) + "\n")
    return path


def attach_feedback(card: dict, client, project_name: str) -> str:
    """Create a sentinel run in the experiment project and attach the scorecard as feedback."""
    run_id = str(uuid.uuid4())
    now = datetime.now(UTC)
    project_id = str(client.read_project(project_name=project_name).id)
    client.create_run(
        name=f"scorecard-{card['judge']}-{card['criterion']}",
        inputs={"judge": card["judge"], "criterion": card["criterion"]},
        outputs=card,
        run_type="chain",
        project_name=project_name,
        id=run_id,
        start_time=now,
        end_time=now,
        tags=["scorecard", "sentinel"],
    )
    if hasattr(client, "flush"):
        client.flush()
    client.create_feedback(
        run_id,
        key=f"scorecard_{card['criterion']}",
        session_id=project_id,
        score=card["agreement"]["kappa"],
        value=card,
        comment=f"judge={card['judge']} dataset={card['dataset']['version_tag']}",
    )
    return run_id


# --- gate ---


def gate(card: dict, pack: Pack) -> tuple[bool, str]:
    """(qualified, reason). Kappa LOWER bound >= pack.min_kappa_to_gate, and for rules
    tagged crisis or guardrail, TPR on the failing class >= the rule's threshold."""
    if card.get("status") == INSUFFICIENT:
        return False, f"{INSUFFICIENT}: {card['n_fail']} of {card.get('min_fails', MIN_FAILS)}"
    lower = card["agreement"]["kappa_ci"][0]
    if lower is None:
        return False, "kappa interval undefined (judge or labels constant)"
    if lower < pack.min_kappa_to_gate:
        return False, f"kappa lower bound {lower:.2f} < {pack.min_kappa_to_gate:.2f}"
    rule = pack.rule(card["criterion"])
    if rule is not None and rule.is_safety:
        tpr = card["confusion"]["tpr"]
        if tpr is None:
            return False, "TPR undefined (no failing items in the labels)"
        if tpr < rule.threshold:
            return False, f"TPR {tpr:.2f} < {rule.threshold:.2f} ({', '.join(sorted(rule.tags))})"
    return True, f"kappa lower bound {lower:.2f} >= {pack.min_kappa_to_gate:.2f}"


def qualified(card: dict, pack: Pack) -> bool:
    return gate(card, pack)[0]


# --- plot ---


def plot_kappa(
    cards: list[dict], path: Path = OUT_DIR / "kappa.png", packs: Sequence[Pack] = ()
) -> Path:
    """Kappa with its 95% interval, one row per criterion, one marker per judge."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#0b0b0b", "#52514e", "#dddcd7"
    series = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    criteria = sorted({c["criterion"] for c in cards})
    judges = sorted({c["judge"] for c in cards})
    fig, ax = plt.subplots(figsize=(8, 1.2 + 0.9 * len(criteria)), dpi=160)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    offsets = [(i - (len(judges) - 1) / 2) * 0.22 for i in range(len(judges))]
    for ji, judge in enumerate(judges):
        for ci, criterion in enumerate(criteria):
            card = next(
                (c for c in cards if c["judge"] == judge and c["criterion"] == criterion), None
            )
            if card is None:
                continue
            y = ci + offsets[ji]
            k = card["agreement"]["kappa"]
            lo, hi = card["agreement"]["kappa_ci"]
            color = series[ji % len(series)]
            if k is None:
                label = "undefined"
                if card.get("status") == INSUFFICIENT:
                    label = (
                        f"insufficient fails ({card['n_fail']}/{card.get('min_fails', MIN_FAILS)})"
                    )
                ax.text(0, y, label, color=muted, fontsize=8, va="center", ha="center")
                continue
            if lo is not None and hi is not None:
                ax.plot([lo, hi], [y, y], color=color, linewidth=2, solid_capstyle="round")
            ax.plot([k], [y], "o", color=color, markersize=7, markeredgecolor="#fcfcfb")
            ax.text(
                (hi if hi is not None else k) + 0.02,
                y,
                f"{k:.2f}",
                color=ink,
                fontsize=8,
                va="center",
            )
    for pack in packs:
        ax.axvline(pack.min_kappa_to_gate, color=muted, linewidth=1, linestyle=(0, (3, 3)))
        ax.text(
            pack.min_kappa_to_gate,
            -0.55,
            f" {pack.name} {pack.min_kappa_to_gate:.1f}",
            color=muted,
            fontsize=7,
            va="top",
        )
    ax.set_yticks(range(len(criteria)))
    ax.set_yticklabels(criteria, color=ink, fontsize=9)
    ax.set_xlim(-1.0, 1.08)
    ax.set_ylim(-0.6, len(criteria) - 0.4)
    ax.set_xlabel("Cohen's kappa vs human labels, 95% bootstrap interval", color=muted, fontsize=8)
    ax.tick_params(axis="x", colors=muted, labelsize=8)
    ax.grid(axis="x", color=grid, linewidth=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.invert_yaxis()
    if len(judges) > 1:
        from matplotlib.lines import Line2D

        ax.legend(
            [
                Line2D([0], [0], marker="o", color=series[i % len(series)], linewidth=2)
                for i in range(len(judges))
            ],
            judges,
            loc="lower left",
            fontsize=8,
            frameon=False,
            labelcolor=ink,
        )
    ax.set_title("Judge agreement with human labels", color=ink, fontsize=10, loc="left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#dddcd7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def _draw_reliability(ax, card: dict, color: str) -> None:
    cal = card["calibration"]
    ax.set_facecolor(SURFACE)
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1, linestyle=(0, (3, 3)), zorder=1)
    xs = [b["mean_prob"] for b in cal["bins"] if b["n"]]
    ys = [b["frac_pass"] for b in cal["bins"] if b["n"]]
    ns = [b["n"] for b in cal["bins"] if b["n"]]
    ax.plot(xs, ys, color=color, linewidth=2, zorder=2)
    ax.scatter(
        xs, ys, s=[max(24, min(200, 24 + 4 * n)) for n in ns], color=color,
        edgecolor=SURFACE, linewidth=1, zorder=3,
    )  # fmt: skip
    for x, y, n in zip(xs, ys, ns, strict=True):
        ax.annotate(
            str(n), (x, y), textcoords="offset points", xytext=(0, 9), ha="center",
            fontsize=7, color=MUTED,
        )  # fmt: skip
    ece, brier = cal["ece"], cal["brier"]
    ax.set_title(
        f"{card['judge']} / {card['criterion']}   ECE {ece:.3f}   Brier {brier:.3f}   "
        f"n={cal['n_calls']}",
        color=INK, fontsize=9, loc="left",
    )  # fmt: skip
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("predicted P(pass), 10 bins", color=MUTED, fontsize=8)
    ax.set_ylabel("observed pass rate", color=MUTED, fontsize=8)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def calibration_plot_path(card: dict, out_dir: Path = OUT_DIR) -> Path:
    return out_dir / f"{card['judge']}_{card['criterion']}_calibration.png"


def plot_calibration(card: dict, path: Path) -> Path | None:
    """10-bin reliability diagram for one scorecard; None when it has no calibration."""
    if not card.get("calibration"):
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    _draw_reliability(ax, card, SERIES[0])
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def plot_calibration_grid(cards: list[dict], path: Path) -> Path | None:
    """Every calibrated judge, one row per judge, the criteria side by side."""
    cards = [c for c in cards if c.get("calibration")]
    if not cards:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    judges = sorted({c["judge"] for c in cards})
    criteria = sorted({c["criterion"] for c in cards})
    fig, axes = plt.subplots(
        len(judges), len(criteria), figsize=(4.6 * len(criteria), 4.2 * len(judges)),
        dpi=160, squeeze=False,
    )  # fmt: skip
    fig.patch.set_facecolor(SURFACE)
    for ji, judge in enumerate(judges):
        for ci, criterion in enumerate(criteria):
            ax = axes[ji][ci]
            card = next(
                (c for c in cards if c["judge"] == judge and c["criterion"] == criterion), None
            )
            if card is None:
                ax.set_axis_off()
                continue
            _draw_reliability(ax, card, SERIES[ji % len(SERIES)])
    fig.suptitle("Reliability diagrams, every calibrated judge", color=INK, fontsize=10, x=0.01,
                 ha="left")  # fmt: skip
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


# --- orchestration ---


def make_client():
    from langsmith import Client

    return Client()


def build_judge(name: str, criterion: str) -> Judge:
    return JUDGES[name](criterion)


def score(
    judge_names: Sequence[str],
    criteria: Sequence[str] | None,
    client,
    pack: Pack,
    dataset_name: str = DATASET_NAME,
    out_dir: Path = OUT_DIR,
    repetitions: int = REPETITIONS,
    rows: list[LabelledRow] | None = None,
    evaluate_fn: Callable | None = None,
    log: Callable[[str], None] = print,
    min_fails: int = MIN_FAILS,
    offline: bool = False,
    injection: Sequence[dict] | None = None,
) -> list[dict]:
    """Run every judge x criterion, write scorecards and kappa.png, return the scorecards.

    offline=True makes no LangSmith calls: examples are built locally, nothing is uploaded,
    and no sentinel feedback is written. The scorecards and the plot are the same."""
    rows = load_labelled() if rows is None else rows
    if not rows:
        raise SystemExit("no labelled traces; run `vd label` first")
    fails = fail_counts(rows)
    log("human fails per rule: " + ", ".join(f"{k} {v}" for k, v in fails.items()))
    if offline:
        dataset_info = {
            "name": dataset_name,
            "id": None,
            "version_tag": labels_tag(rows),
            "as_of": None,
            "n": len(rows),
            "offline": True,
        }
        examples = local_examples(rows, dataset_name)
        log(f"offline: {len(rows)} labelled traces @ {dataset_info['version_tag']}, no uploads")
    else:
        dataset_info = sync_dataset(rows, client, dataset_name)
        log(
            f"dataset {dataset_info['name']} @ {dataset_info['version_tag']}: "
            f"{dataset_info['n']} labelled traces ({dataset_info['created']} new)"
        )
        examples = list(client.list_examples(dataset_id=dataset_info["id"]))
    injection = load_injection() if injection is None else injection
    cards: list[dict] = []
    for name in judge_names:
        wanted = list(criteria) if criteria else list(getattr(JUDGES[name], "criteria", ()))
        for criterion in wanted:
            judge = build_judge(name, criterion)
            outcome = run_experiment(
                judge,
                examples,
                client,
                dataset_info,
                repetitions,
                evaluate_fn,
                min_fails=min_fails,
                offline=offline,
            )
            card = build_scorecard(judge, outcome, dataset_info, min_fails)
            card["injection"] = injection_flips(judge, injection)
            if card["calibration"]:
                card["calibration"]["plot"] = str(
                    plot_calibration(card, calibration_plot_path(card, out_dir))
                )
            card["gate"] = {
                pack.name: dict(zip(("qualified", "reason"), gate(card, pack), strict=True))
            }
            path = write_scorecard(card, out_dir)
            if offline:
                card["experiment"]["sentinel_run_id"] = None
                card["experiment"]["offline"] = True
                path.write_text(json.dumps(card, indent=2) + "\n")
                log(f"wrote {path}")
                cards.append(card)
                continue
            try:
                card["experiment"]["sentinel_run_id"] = attach_feedback(
                    card, client, outcome.experiment_name
                )
            except Exception as e:  # a LangSmith quota or outage must not lose the scorecard
                card["experiment"]["sentinel_run_id"] = None
                card["experiment"]["feedback_error"] = f"{type(e).__name__}: {str(e)[:200]}"
                log(f"could not attach scorecard feedback: {type(e).__name__}: {str(e)[:120]}")
            path.write_text(json.dumps(card, indent=2) + "\n")
            log(f"wrote {path}")
            cards.append(card)
    plot_kappa(cards, out_dir / "kappa.png", packs=[pack])
    plot_calibration_grid(cards, out_dir / "calibration.png")
    return cards
