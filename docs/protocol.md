# Judge qualification protocol

What voirdire does before a rule's evaluator is allowed to gate, and why. Each step cites the research it rests on. Written Oct 20, 2026 from ~28 sources; the thin spots are listed at the end so nobody mistakes a convention for a finding.

Copied from the "Judge qualification protocol" tab of the voirdire build plan (https://claude.ai/artifact/Tq9HUBWwqiF1gXgv5YDN23) on 2026-10-21.

## The claim this protocol defends

A judge that has passed this protocol agrees with the people who own the risk, beyond chance, with a known margin of error, consistently across repeated runs, without the known biases, on the kinds of cases that matter, and will be re-checked whenever it or the world changes. An auditor can read the scorecard and see each of those as a number with a source.

## 0. Before any judge: the rule and the labels

**Binary, per failure mode.** One rule asks one yes/no question. Likert scales and composite "quality" scores are where agreement collapses; binary pass/fail per failure mode is what Husain's and Shankar's practitioner work converges on, and what EvalGen found keeps criteria from drifting while graders grade. [EvalGen](https://arxiv.org/abs/2404.12272) · [Husain, validate-evaluator](https://skills.sh/hamelsmu/evals-skills/validate-evaluator)

**Code first.** If a deterministic check can express the rule (a drug name regex, presence of the 988 line, a length bound), it is the judge, and no LLM is involved. Deterministic judges skip steps 2–5. Everything in the lab still runs on them, so the scorecard shows a kappa of 1.0 with its reason.

**Who labels.** Domain experts who own the risk, never the engineers. MedJUDGE's review of 49 healthcare judge studies found a median of three validators and expert–expert kappa ranging from 0.11 to 0.98, which is the point: the human ceiling is measured, not assumed. [MedJUDGE](https://arxiv.org/pdf/2604.25933)

**Balanced, stratified, ~100 per rule as the floor.** Husain's working number is about 100 expert-labelled traces, deliberately balanced toward 50/50 pass/fail, with at least 20 of each. Balance matters because kappa collapses when one class dominates (the prevalence paradox, Feinstein and Cicchetti 1990); a rule that passes 97% of the time in production needs its failures over-sampled into the labelled set. Stratify by scenario so rare cases (distress, jailbreak, pediatric) are present in numbers. Split: 10–20% for prompt iteration, the rest held out for the scorecard. [Krippendorff's alpha guide](https://arxiv.org/abs/2008.00977) · [Husain](https://skills.sh/hamelsmu/evals-skills/validate-evaluator)

**Size from the interval you need, not a fixed N.** "How to Correctly Report LLM-as-a-Judge Evaluations" (Nov 2025) shows that once the test set is large, the width of your confidence interval is governed by the calibration set, so you work backward from the interval width you can act on and allocate labels across both classes. Kim (May 2026) gives the version for a judge with known human correlation: the human labels you need fall with judge–human R², with a floor at n(1 − ρ²). voirdire's `min_labelled` default is 100; the lab prints the achieved interval width and says whether more labels would narrow it usefully. [How to Correctly Report](https://arxiv.org/html/2511.21140v1) · [Kim 2026](https://www.alphaxiv.org/abs/2605.16354.md)

## 1. Human agreement first (the ceiling)

**Two labellers on an overlap set; three for safety rules.** Compute Cohen's kappa between humans (Fleiss for three or more). Rating Roulette found human–human accuracy of 0.83 on MT-Bench collapses to a Krippendorff's alpha of 0.48, so the human ceiling must be reported as a chance-corrected number, and the judge is graded against it, not against 1.0. [Rating Roulette](https://arxiv.org/abs/2510.27106)

**Verdict RULE UNCLEAR** if human agreement is below 0.6 (Landis–Koch "moderate" boundary; Krippendorff's 0.667 minimum is the stricter alternative). The fix is the rule statement and examples, not the judge. MedJUDGE: add raters until kappa changes by less than 0.05, adjudicate structurally, and represent persistent disagreement as uncertainty rather than forcing a label.

**Adjudication record.** Disagreements and their resolution are stored with the label, so the gold set carries its own uncertainty. Validating-without-gold-labels (Mar 2025) shows forced-choice agreement on genuinely indeterminate items can pick a judge up to 34% worse; items the humans could not agree on after adjudication are flagged, not forced. [Validating without gold labels](https://arxiv.org/html/2503.05965v2)

## 2. Agreement: chance-corrected, with an interval

**Headline number is Cohen's kappa, not accuracy.** Reliability without Validity (Jun 2026; 21 judges, nine providers, ~541k judgments) found exact-match agreement overstates every judge, with kappa 34–41 points lower on MT-Bench, and judge rankings moving up to 15 places between benchmarks. Exact match is reported alongside, never alone. [Reliability without Validity](https://arxiv.org/abs/2606.19544)

**Report the confusion matrix with it.** For imbalanced rules, kappa alone misleads in the other direction. The scorecard also shows prevalence, and sensitivity and specificity (TPR and TNR) with the harmful class as positive; Husain's bar is both above 0.90. For safety rules the number that gates is recall on the harmful class, because a judge that misses one crisis disclosure in a hundred is not "substantial agreement," it is a missed crisis.

**Bootstrap the interval.** The two biggest 2026 judge studies report point estimates only, which is a gap voirdire does not copy. 5,000 bootstrap resamples, clustered by conversation, reporting the count of undefined-kappa resamples. The gate compares the kappa **lower bound** to `min_kappa_to_gate`, so a lucky point estimate cannot qualify a judge. [OneUptime, kappa calibration](https://oneuptime.com/blog/post/2026-08-31-calibrate-llm-judge-cohens-kappa/view)

**Thresholds, and where they come from.** Landis–Koch: 0.61–0.80 substantial, above 0.80 almost perfect. Krippendorff: 0.667 for tentative conclusions, 0.80 for reliance. voirdire packs: startup `min_kappa_to_gate` 0.6, regulated 0.8, and regulated additionally requires harmful-class recall at or above the rule's threshold. These are conventions from content analysis, not LLM-judge findings; the protocol says so on the scorecard.

**Correct the production rate.** A judge with known sensitivity and specificity gives a biased pass rate in production. The evidence report corrects it with the standard estimator from How to Correctly Report, θ = (p + q₀ − 1) / (q₀ + q₁ − 1), with the combined interval. A raw judge pass rate never appears in the assurance view without the corrected one beside it.

## 3. Consistency: three runs, swapped, temperature pinned

**Three replicates minimum.** Rating Roulette found a judge giving the identical verdict in all three runs only 61% of the time on one model, and that runs beyond three changed little. Reliability without Validity uses three or more at temperature 0 with caching disabled and reports test–retest, self-consistency against the majority, and flip rate. voirdire runs three, reports test–retest agreement and the share of items with any disagreement. [Same Input, Different Scores](https://www.alphaxiv.org/abs/2603.04417.md)

**Temperature 0 is not determinism.** No model in the Mar 2026 study was fully deterministic at temperature 0, and the effect varies by family. Pin temperature, record it, and still run three.

**Consistency is not validity.** The Sep 2026 study found judges 92–97% self-consistent with human correlation of only 0.28–0.34. Test–retest is a necessary check that never substitutes for step 2. The two cited bars disagree (alpha ≈ 0.8 vs test–retest above 0.95); voirdire uses 0.9 and flags it as a convention. [When Consistency Does Not Mean Reliability](https://arxiv.org/abs/2609.13824)

**Majority vote is the production verdict.** Where a judge runs online at three samples, the majority is the score; Rating Roulette found this improves human agreement. Cost is three calls; the pack decides whether a rule earns it.

## 4. Bias audit

Each bias has a standard test from MT-Bench (2023) and the CALM framework (ICLR 2025, twelve biases, perturbation attacks with robustness and consistency rates). voirdire runs the ones that apply to a single-response pass/fail judge: [MT-Bench](https://arxiv.org/abs/2306.05685) · [CALM](https://arxiv.org/html/2410.02736v1)

| Bias | Test | Report | Gate |
|---|---|---|---|
| Position (pairwise judges only) | Judge AB and BA; count inconsistent as tie | Flip rate, | P(A) − 0.5 |
| Verbosity | Kappa on longest third vs shortest third; Pearson r of length vs verdict | Both | r below 0.2 (MedJUDGE) |
| Self-preference | The chatbot's model family must differ from the judge's, or a paired family swap is run | Family of each, delta if swapped | Different family required in the regulated pack |
| Prompt-format sensitivity | Re-run with the rubric reworded and a different persona (the Sprinklr p0/p1/p2 design) | Kappa variance across configs | Below 0.05 (MedJUDGE, uncalibrated) |
| Authority / style | Perturb a fail case with a fake citation or confident tone | Robustness rate | Report only |

Self-preference deserves its own note: Wataoka (2024) found the bias tracks low perplexity, so judges favour familiar text, not strictly their own outputs; and JudgeBench saw Claude 3.5 drop from 64% to 45% on pairs its own family generated. Using a different model family for judge and bot is the cheap mitigation and the regulated pack requires it. [Self-preference](https://arxiv.org/abs/2410.21819) · [JudgeBench](https://arxiv.org/html/2410.12784v2)

## 5. Calibration, for judges that give a probability

**Verbalised confidence is a different thing from a calibrated probability.** ConfidenceBench (Jul 2026) found frontier models' verbalised confidence can match human calibration on QA (Brier ≈ 0.10) while smaller models are worse than guessing; Tian et al. found judge confidence overstates correctness; and the Jan 2026 faithfulness study found models do not abstain when their own stated confidence says they should. So a verbalised confidence from an LLM judge is recorded but never used for routing. [ConfidenceBench](https://www.alphaxiv.org/abs/2607.20526) · [Overconfidence](https://arxiv.org/abs/2508.06225) · [Faithful to verbal confidence?](https://arxiv.org/abs/2601.07767)

**Discriminative judges are the exception.** Jev (TypeSafe AI, Sep 2026) returns a label and a probability from a typed decision model trained for calibration; an independent test routed the 0.3–0.7 band to humans and reached 99.5% accuracy. Evidence is vendor-adjacent and weeks old, so voirdire measures it: reliability diagram over the labelled set, expected calibration error and Brier score, ten bins. A calibrated probability earns a third routing outcome, **defer to human**, for the uncertain band; the pack sets the band. [Arize on Jev](https://arize.com/?p=32075)

**Classifier judges beat generative ones where they fit.** RewardBench 2 found classifier reward models outscoring generative judges; MiniCheck's 770M checker matched GPT-4 on grounding at roughly 400× lower cost. For classification-shaped rules the lab should try a small discriminative judge before a frontier LLM. [RewardBench 2](https://arxiv.org/html/2506.01937v2) · [MiniCheck](https://arxiv.org/abs/2404.10774)

## 6. Safety rules are different

The refusal-style safety benchmarks are easy for frontier judges (RewardBench 2: 96–97%). Operational safety is not. In the Sprinklr voice study, humans logged three to six times more safety events than the judges in telecom, with pooled judge–human correlation of 0.32–0.61 there against 0.91 in retail; quality metrics were near parity. The Gemini audio-judge study was within 0.07 of human–human on five of eight dimensions but its defect-sensitivity tests were underpowered. [Sprinklr](https://arxiv.org/abs/2608.24314) · [Gemini audio judges](https://arxiv.org/abs/2607.07985)

So, in the regulated pack, for any rule tagged crisis or guardrail:

- The gate is harmful-class recall, not kappa.
- A deterministic check runs alongside the LLM judge and either flag routes to a human.
- A standing human sample continues after qualification (every judge-flagged trace plus a random 5%), and the judge's agreement on that sample is tracked as the drift signal.
- The verdict can be READY TO GATE only with a second labeller on the overlap set.

## 7. Multiple judges are not independent evidence

Agreement Overstates Evidence (Sep 2026): ten judges' errors correlate (mean r 0.21), so ten judges are worth about 3.5 independent ones, and frontier judges across providers share errors; up to 28% of system comparisons flipped. When LLMs Agree (Jul 2026): high-agreement answers were wrong 48% of the time on GPQA. A judge ensemble is not a substitute for human labels, and the scorecard never averages judges. [Agreement Overstates Evidence](https://arxiv.org/abs/2609.22512) · [When LLMs Agree](https://www.alphaxiv.org/abs/2607.08065.md)

## 8. Pin it, then re-validate

**Pin.** Judge model snapshot id, prompt hash, rubric version, temperature, few-shot set, and the dataset version tag the scorecard was computed on; all in the scorecard JSON, all in LangSmith evaluator metadata.

**Re-validate on triggers and on a clock.** Hosted endpoints drift, sometimes silently (2606.19544 is explicit that its results are a five-week snapshot). The Gemini study saw a model swap preserve rank order but shift absolute levels, so calibration in particular must be re-run on any swap. Triggers: judge model or prompt change, rules pack version change, a shift in the scenario mix, any slice failing. Clock: quarterly (MedJUDGE), with the caveat that the cadence is uncalibrated. [Tianpan, judge drift](https://tianpan.co/blog/2026-04-23-llm-judge-drift-evaluator-upgrade-phantom-regression)

**Monitor between re-validations.** Sentinel cases (a fixed set of known pass and fail traces scored on every run, alert at more than 5% deviation); human override rate on the standing sample (review at 15%); old and new judge run in parallel on the held-out set before any upgrade is promoted.

## The scorecard, as a reader sees it

```yaml
rule: crisis_routed
judge: llm/claude-sonnet-5.5@2026-09-12, prompt a41c9e, rubric v3, temp 0
dataset: support-bot-eval@v7  (n=112, fail=51)
human_agreement: kappa 0.81 (2 labellers, overlap 40; 3 unresolved, flagged)
agreement: kappa 0.74 [0.62, 0.84]  exact 0.91  TPR 0.96  TNR 0.86
consistency: test-retest 0.93 (3 runs)  any-disagreement 9/112
bias: length r 0.07  format variance 0.03  judge family != bot family
calibration: n/a (no probability)
safety: harmful-class recall 0.96 >= 0.95 required; deterministic twin active
verdict: READY TO GATE (regulated: kappa lower bound 0.62 < 0.80 -> NOT RELIABLE; recall passes)
next_revalidation: 2027-01-20 or on any trigger
```

The last line shows why the pack matters: the same judge is ready under the startup pack and not under the regulated one. The report says both.

## What is thin, so nobody over-claims

- No LLM-judge paper gives a principled minimum N for a kappa interval; 100 is a practitioner floor. The lab reports achieved interval width instead of claiming the number is enough.
- Acceptable test–retest is contested (0.8 vs 0.95); 0.9 is voirdire's convention.
- Safety evidence is contradictory: refusal benchmarks are easy, deployed safety events are badly under-detected. Hence step 6 is stricter than the numbers alone would justify.
- Calibration of verbalised confidence is studied on QA, not judge rubrics; Jev's evidence is weeks old and vendor-adjacent.
- Drift evidence is anecdotal; the quarterly cadence and the 5% / 15% alert levels are MedJUDGE's and explicitly uncalibrated.
- Weighted kappa and Gwet's AC1 for skewed distributions appear in practitioner posts, not peer-reviewed judge studies; voirdire reports Cohen's plus the confusion matrix rather than switching metrics.
- Three practitioner sources could not be fetched directly (Husain's judge posts, Anthropic's evals guide); their guidance is taken from secondary pages.

## Where each step lives

| Step | Story | Lab verdict it feeds |
|---|---|---|
| 0 labels, balance, stratification | S3, S14 synth | n, interval width |
| 1 human agreement | S15 | RULE UNCLEAR |
| 2 kappa + interval + confusion | S4 | JUDGE NOT RELIABLE |
| 3 three runs, test–retest | S4 | consistency |
| 4 bias audit | S4 (length), S14 (format, family) | bias |
| 5 calibration | S5 | defer band |
| 6 safety rules | S7 routing, S11 pack | recall gate |
| 8 pin and re-validate | S4 metadata, S24, roadmap change management | next_revalidation |
