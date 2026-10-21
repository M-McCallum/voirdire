# Discriminative judges: Jev and OpenAI Decisions

Two hosted decision models that answer typed yes/no questions with a probability instead of generating text. voirdire uses them as judges for `no_dosing_advice` and `crisis_handled` (core/judges/discriminative.py), because docs/protocol.md step 5 says a calibrated probability earns a third routing outcome, "defer to human", and step 5 also says to measure that calibration rather than trust the vendor.

Read on 2026-10-21 from docs.typesafe.ai and developers.openai.com. Anything not on those pages is marked as such; nothing below is from memory.

## Side by side

| | TypeSafe Jev | OpenAI Decisions |
|---|---|---|
| Endpoint | `POST https://api.typesafe.ai/v1/systemone` | `POST /v1/decisions` |
| SDK | `typesafe-sdk` 0.7.5 (2026-10-14), `TypeSafeClient.system_one(state, questions)` | `openai` 3.36.0, `client.decisions.create(model, input, questions)` |
| Auth | `Authorization: Bearer <key>`, env `TYPESAFE_API_KEY` | `Authorization: Bearer <key>`, env `OPENAI_API_KEY` |
| Pinned model | `jev-1.13.0` | `gpt-6-luna` (no dated snapshot exists) |
| Yes/no primitive | `Noul`: returns `noul`, P(true) in 0..1 | `predicate`: returns `probability`, P(true) in 0..1 |
| Price | $0.042 per 1M input tokens, output free | $0.10 per 1M input tokens, only input charged |
| Rate limits | Per org, tokens/s and requests/s, console only | Per org/project, RPM and TPM by usage tier |
| Input | Text or JSON; images behind a danger-zone header | Text and images; no `file_id` |
| Context | 64k per request, 32k per question (262k behind a header) | 1,050,000 window, 922,000 max input |
| ZDR | Enterprise only, via sales | Eligible endpoint "with limitations" |
| HIPAA | Not mentioned anywhere on the docs site | Eligible under an executed BAA |
| Status | Stable release; 1.14.0 is the preview | Public beta, "GA in the coming weeks" |

## TypeSafe Jev

**Auth.** API key in `Authorization: Bearer <key>`. The Python client reads `TYPESAFE_API_KEY` and validates the key before any request (empty, non-ASCII, or whitespace-containing keys raise `TypeSafeError` at client construction). Base URL from `TYPESAFE_BASE_URL`, default `https://api.typesafe.ai`. voirdire also accepts `JEV_API_KEY`, which is the name in `.env.example`.

**Rate limits.** "Limits are set per organization and measured in tokens per second and requests per second." Over the limit returns `429 Too Many Requests`; the SDK retries by default and honours `retry-after`. The numbers are not on the docs site, only in the developer console; third-party posts quote figures from 40 to 80 requests/s and 100k to 250k tokens/s that disagree with each other, so treat the console as the only source. Request rules: body at most 32 MiB, 64 nesting levels, UTF-8 JSON, finite numbers; `529` means overloaded, retry with exponential backoff.

**Pricing.** $42 per billion input tokens ($0.042 per million). Output tokens are free. Long-context requests cost the same per token. Images bill as input tokens at the same rate.

**Pinned version.** `jev-1.13.0` is the current stable release and what `jev-latest` resolves to. `jev-1.14.0` has been available as `jev-preview` since 2026-10-14. Aliases move when a release ships "which can change answers without any code change", so the docs say to send the versioned id once thresholds are tuned. Every response carries a `model` field with the id that actually answered; voirdire checks it against the pin and flags a mismatch. Deprecation policy: not documented.

**Input modalities.** Text or JSON `state`, 64k tokens per request and 32k per question by default; 262k with the `TypeSafe-Danger-Zone-Long-Context: true` header, which the docs call slower and less accurate. Images (PNG or JPEG, 280 to 1,568 px a side, 16 MiB) are "dreamer-ready" and need `TypeSafe-Danger-Zone-Images: true`. No audio or video. Questions are typed: `Noul` (yes/no), `Choice`, `Score`, `Box`; many questions go in one call and are answered "in parallel and in isolation" against the same state.

**Data retention, ZDR, HIPAA.** The legal page links a Data Processing Agreement that "covers how we process customer data on your behalf, including data retention" without stating periods, a Privacy Policy described as covering "our commitment not to train models on user data", and says "we also offer zero data retention (ZDR) for enterprise customers", via sales@typesafe.ai. HIPAA, BAA, SOC 2 and GDPR are not mentioned on the docs site. For a regulated pack this is a blocker until the DPA is read and a ZDR agreement is in hand.

**Documented weaknesses** (the "Jev 1.13 jaggedness" page and the Noul page):

- Literal reading: "answers the question you wrote, not the one you meant". Scoping words, negations and implied conditions are taken at face value.
- Negation and indirection: "instructions carrying double negatives or complex indirection are answered less reliably"; multi-hop questions lose accuracy. Contradictory `criteria` (true mapped to no) perform worse. This is why voirdire asks every predicate in both phrasings and flags disagreement.
- Numbers, dates: does not count reliably, reads dates as text, score outputs "weakly calibrated".
- Long inputs: accuracy falls as irrelevant `state` grows ("context rot").
- Prompt injection: "State is data, and jev-1.13 does not treat it as hostile by default." Injected instructions, misleading framing and text arguing for its own classification can change the answer. The docs recommend pairing Jev with deterministic checks, which is what the protocol's "deterministic twin" does.
- Choice option order bias (first option); compound questions ("hiding several judgments inside one question") degrade. Nouls return no `confidence`; the docs suggest |2p − 1| as the comparable number, which voirdire uses.
- No calibration claim is made anywhere on the site for Noul probabilities. The confidence page says thresholds "depend on your domain and the performance of the model for your use case".

## OpenAI Decisions

**Auth.** `Authorization: Bearer $OPENAI_API_KEY`, `Content-Type: application/json`. The SDK call is `client.decisions.create(...)`; the guide's example lists Python SDK 3.26.0 or later.

**Rate limits.** The Decisions guide does not mention them. The rate-limits page gives the shared Luna pool by usage tier: Build 5,000 RPM / 2M TPM, Launch 10,000 / 10M, Grow 30,000 / 180M. Headers `x-ratelimit-*` and `Retry-After`; a `429 slow_down` fires on ramp rate even inside the limits (increase traffic by at most 50% every 15 minutes past 1M input TPM).

**Pricing.** The guide: $0.10 per 1M input tokens, "only input tokens are charged", no cache or output charge. Regional processing adds 10%; over 272k input tokens the long-context rate (2x input) applies. The general pricing page lists `gpt-6-luna` output at $0.50 per 1M and cached input at $0.01, which applies to Responses, not Decisions.

**Pinned version.** `gpt-6-luna` is "the only model currently available" for Decisions. The model page lists no dated snapshot, so `gpt-6-luna` is both the alias and the pin; a silent model update cannot be excluded, which is why the scorecard records the `model` field from each response. The API is in public beta: "we expect to GA in the coming weeks."

**Input modalities.** `input` is a string or an array of user messages with `input_text` and `input_image` parts (inline base64 data URL or public HTTP URL). `file_id` is not supported. The Luna model page: 1,050,000 context window, 922,000 max input, text and image in, knowledge cutoff 2026-05-18. Question types: `predicate` (returns `probability`), `choice` (choice, probabilities, confidence), `score` (probability-weighted level index). Answers can also come back as `refusal`, which the examples check for but the prose does not describe; voirdire counts a refusal as 0.5 and flags it.

**Data retention, ZDR, HIPAA.** Abuse-monitoring logs are kept up to 30 days by default; API data is not used for training since 2023-03-01. `/v1/decisions` is ZDR-eligible "with limitations": GPU-local prompt caching for up to 24 hours, and a CSAM retention exception for image inputs. ZDR needs OpenAI approval per org or project. The Decisions API "is HIPAA-eligible under an executed BAA, subject to account configuration" (Private Retention or Safety Retention provisioning). Data residency: US and EU (EEA + Switzerland) regional storage and processing; EU processing requires ZDR or a modified-retention arrangement.

**Documented weaknesses.** The guide has no jaggedness page. What it does say:

- Beta, single model, no snapshot, no GA terms.
- "For decisions that depend on earlier answers, the page recommends separate requests": questions in one call are independent.
- Thresholds must be set "from labeled examples" weighing false-positive against false-negative cost; no accuracy or calibration figures are given.
- Score answers are averages and can fall between levels; use `choice` for unordered categories.
- Rate limits, error codes, refusal behaviour and injection robustness are not documented for this endpoint. Since the evidence is an inline string that the model reads as a user message, voirdire treats it as injectable until the fixture says otherwise.

## How voirdire uses them

Both judges share one base (core/judges/discriminative.py). A rule's atomic predicates live in prompts/questions/`<rule>`.yaml; each is asked in affirmative and negated phrasing in one call, the two probabilities are averaged (p = (p_aff + 1 − p_neg) / 2) and a `phrasing_disagree` flag is recorded when the two phrasings land on the same side of 0.5. The verdict is the YAML's fixed any/all combination; confidence is the minimum |2p − 1| over the questions used; the probability of the pass verdict is max (any) or min (all) of the predicate probabilities, or the probability that the combination equals a condition question (crisis: routed exactly when the user was in distress). The scorecard adds a 10-bin reliability diagram, ECE and Brier, a selective-prediction table fitted by Platt scaling on half the labels and reported on the other half, and the count of injection-fixture transcripts each judge flipped on.

## Sources

- https://docs.typesafe.ai/introduction, /introduction/quickstart, /primitives/noul, /confidence, /models, /images, /api/errors, /legal, /model-jaggedness/jev-1.13, /sdk/python/usage/basic, /sdk/python/changelog
- https://github.com/pydantic/pydantic-ai/blob/main/docs/models/typesafe.md (`pydantic-ai-slim[typesafe]`, `typesafe:jev-1.13.0`; not used, the SDK is called directly)
- https://developers.openai.com/api/docs/guides/decisions, /guides/your-data, /guides/rate-limits, /pricing, /models/gpt-6-luna
