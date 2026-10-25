# FinAgenticBench three-channel scoring design

## Core decision

FinAgenticBench does not ask an LLM judge to decide whether arithmetic is correct.
It reports three complementary channels and only then computes an optional,
gated composite:

| Channel | Question answered | Evaluator | Output |
|---|---|---|---|
| Programmatic Verification (PV) | Did the agent compute the required facts correctly? | Python validator over runtime variables | strict pass/fail plus per-variable set/correct diagnostics |
| Response Quality (RQ) | Is the final analysis financially sound, grounded, calibrated and useful? | frozen 1–5 rubric, fixed judge panel | per-dimension raw/capped means and weighted 1–5 score |
| Claim Coverage (CC) | Did the answer explicitly address the case-specific context points? | covered/missed verdict per pre-registered claim | coverage rate per claim and required-claim flags |

The vector `(PV, RQ, CC)` is the primary scientific result. A single composite
is convenient for sorting, but it is secondary and must never hide channel-level
failure.

PV has one canonical financial object per field. A PV item is `correct` or it is
not: there is no alternate acceptance path that turns a failure into a pass.
Numeric comparison uses only the requested display-precision tolerance; semantic
datetime encodings may be normalized, while discrete labels use canonical tokens
or an explicitly source-verbatim representation. Where a convention or
intermediate-rounding path could reasonably be read two ways, the query and
variable description name the one that counts rather than the validator admitting
both. If a run exposes ambiguity that the public case did not resolve, revise the
case and keep the old trajectory tied to the old query.
Retrospective validator repairs are replayed into the channel and reported as a
bug-fix sensitivity, never silently substituted into a preregistered headline
result; the model trajectory itself is immutable.

## Design principles

Deterministic runtime validation is kept separate from prose quality and
per-turn claims. Judged scores are fully anchored integers from a fixed jury,
with per-judge dependency caps, raw and capped scores reported side by side,
disagreement flags and corpus-level inter-judge agreement.

The financial adaptation adds a verified reference packet to the RQ prompt.
The judge sees the independently recomputed reference outputs and the agent's
runtime outputs, but not the validator's PASS/FAIL label. This is necessary
because a judge cannot reliably know an issuer's exact filing value, fiscal
period or as-filed transaction field from general model knowledge. Withholding
the verdict reduces anchoring; providing reference facts makes contextual-
correctness judgments evidence-based.

## Response Quality rubric

The single global rubric is [criteria.json](../criteria.json). Authors cannot
change weights or anchors per case.

| Dimension | Weight | Construct |
|---|---:|---|
| Scope and financial conventions | 1.5 | entity, filing, period, cutoff, unit, accounting/transaction meaning |
| Analytical method | 2.0 | formula, denominator, alignment, adjustments, assumptions |
| Evidence grounding and traceability | 1.5 | support from the correct computed facts and source context |
| Interpretation and calibration | 2.0 | what the result means and what it cannot establish |
| Decision-useful communication | 1.0 | clear, complete, properly qualified delivery |

Each judge independently returns one integer from 1 through 5 for every
dimension with a text-specific justification. Fluency, confidence and length
are not scoring criteria. Level 5 is reserved for a genuinely useful additional
reconciliation, sensitivity or implication—not extra prose.

### Dependency cap

The dimensions form a logical chain. If one judge scores any upstream dimension
at 2 or below, all later dimensions for that judge are capped at 3 before the
headline RQ score is averaged. Raw scores remain available for diagnosis.

Example: `[4, 2, 5, 5, 5]` becomes `[4, 2, 3, 3, 3]`. A fluent conclusion
therefore cannot conceal a materially invalid method.

## Claim coverage

Every explanation-bearing turn declares a small set of short, atomic claims about what the analysis should
establish. Each claim has one of three modes:

- `required`: the question asks for the point, or the answer would be materially
  misleading without it. It must be stated explicitly and contributes to CC.
- `optional`: useful evidence that the agent understood or audited its method,
  but absence is diagnostic only and does not lower the composite.
- `conditional`: a correctness boundary on claims the agent chooses to make.
  Silence satisfies it; an unsupported or over-narrow assertion violates it and
  triggers review.

Required claims at or below 50% judge coverage are flagged, including an exact
jury tie. Optional claims are diagnostic only. Conditional claims are also
flagged when at least half of the jury finds a violation, but they are not
included in the required-coverage average. Reports therefore show required
coverage and conditional violations separately; a response can have 100%
required coverage and still violate a material boundary.

Claims are not a substitute for the global rubric. RQ measures general quality;
CC measures whether this particular financial lesson was demonstrated.

## Composite and gates

For dashboards only, normalize RQ as `(RQ - 1) / 4` and calculate:

```text
composite = 100 × (0.60 × PV + 0.25 × normalized_RQ + 0.15 × required_CC)
```

where PV is 0 or 1 and required_CC is a fraction from 0 to 1.

- A PV failure can score at most 40/100. Correct-sounding prose cannot offset
  a wrong result.
- A majority-missed or tied required claim caps the composite at 75/100.
- Leaderboards must still publish PV pass rate, RQ dimensions, CC rates and
  judge disagreement separately.

The 60/25/15 weights are a pre-registration proposal, not an empirical truth.
They should be frozen only after calibration and reported in sensitivity checks
(70/20/10 and 50/30/20 are generated alongside the primary score). Model rankings that change sharply under
reasonable weights are not robust.

## Statistical unit and publication panels

One model × arm × case is the statistical unit. When a case has repeated model
invocations, FinAgenticBench first averages those repeats and then gives each case equal
weight. It does not pool every trajectory as though repeated calls were new
independent benchmark questions. Reported 95% intervals use a deterministic
hierarchical bootstrap: case units are resampled, followed by repeat observations
within each selected unit. Pairwise model comparisons use only common cases in
the same arm and report `model_b - model_a` with a paired case bootstrap.

Only RQ/CC results from the complete preregistered primary judge panel enter the
headline RQ, CC, and composite. A two-judge subset, an unregistered panel, or different effective RQ and CC panels is preserved as a
separate all-scored sensitivity stratum and sent to manual review. PV remains
panel-independent. Group reports by model, arm, task family, source mode and
financial domain contain all three channels, runtime usage, judge usage, panel
strata and confidence intervals—not just PV.

Agent-runtime usage and judge-panel usage/cost are reported separately. Optional
`input_cost_per_million_usd` and `output_cost_per_million_usd` fields in the
model registry freeze the prices used for an estimate. If either price is absent,
the report marks the call or trajectory unpriced instead of silently treating it
as free; provider invoices remain the authoritative monetary record.

The current report denominator is the set of valid `run.json` artifacts found on
disk; there is no built-in frozen invocation matrix. A publication study must
therefore preserve its intended model × case × run matrix in an external study
protocol and reconcile it with the discovered artifacts before interpreting
missing trajectories. Infrastructure errors and missing scoring channels remain
separate report fields.
PV failure types come only from the replaceable authoritative PV channel, never
from the immutable trajectory. Review routing also covers extreme RQ scores,
PV/RQ discordance, required/conditional claim failures, panel mismatch, and
missing channels.

## How the channels divide a turn

The three channels divide one turn between them. Take an insider-transaction
turn that asks for a trade date, a share count, a reported price and their
product:

- PV checks the four stored values exactly, against a recomputation.
- RQ judges whether the method, the filing semantics and the conclusion are sound.
- CC checks the pre-registered points the answer had to make: that the
  transaction-code letter does not distinguish an open-market purchase from a
  private one, that the transaction date is not the filing date, and what
  shares × reported price does and does not establish.

An answer with correct numbers and no explanation passes PV, scores weakly on RQ
and misses CC. An eloquent answer that treats the filing date as the trade date
fails PV and is composite-gated however much a judge likes its prose.

## Judge protocol and reliability

Publication-grade scoring should use the same fixed panel of at least three
independent foundation-model families for every tested model. Judge identity,
rubric fingerprint, prompt version, temperature and failures must be stored.
Same-lab exclusion should be a paired robustness analysis, not a panel that
silently changes between contestants.

Each RQ or CC channel records its exact criteria/prompt fingerprint, non-secret
requested panel, minimum successful panel size, same-lab policy and
scoring parameters. A resumed channel must match its own recorded settings or be
explicitly recomputed with `--rescore`. There is no experiment-level frozen
protocol or case/reference manifest in the current runner: judged channels load
the current case and reference surfaces, while PV refuses a changed stored query
or output contract. Finish scoring against one repository/data snapshot, or use
a new study id after a case/reference change. Case-level scoring concurrency
changes throughput, not the recorded judging settings.

The tested agent is whatever a study's `--model` names; `experiments/` holds
arms across several families and sizes. Judges are a separate population from the
tested agent, and `DEFAULT_JUDGE_PANEL` in `core/judging.py` is the panel of
record — read it there rather than here. RQ and CC call the same registered
panel separately.
A failed judge is not replaced. At least two successful judges are required.
Requested, effective and failed identities are stored, and batch runs report
per-judge health and parsed-call token usage so systematic endpoint failure and
cost are visible. A syntactically valid JSON response that violates the channel
schema counts as that judge failing; it does not abort the whole batch. No
unregistered judge is selected dynamically.

A same-family judge can remain as one panel member for sensitivity analysis, but
the result should also be reported with that judge excluded. Publication results
should either restore the preregistered three-judge panel or report two-judge
results separately rather than silently pooling them.

Before freezing the benchmark:

1. Create 30–50 deliberately varied answers, including numerically wrong but
   fluent, correct but terse, overclaimed and well-calibrated examples.
2. Have at least two finance-competent humans score them independently.
3. Compare LLM judges with humans and refine anchors before looking at model rankings.
4. Report per-dimension Krippendorff's alpha across the corpus, not per sample;
   use per-sample standard deviation and maximum judge range only to create a
   manual-review queue. Flag a dimension when standard deviation is at least
   1.0 or when two judges differ by at least two anchor levels.
5. Freeze rubric, claims, prompts, weights and gates with a version/fingerprint.

The executable rating-template and comparison workflow is documented in
[scoring_calibration.md](scoring_calibration.md). Until two qualified humans have
actually completed that study and the team has preregistered acceptance
thresholds, FinAgenticBench must describe human calibration as pending.

## Why a channel can decline to score

A missing verdict is not a failed verdict. Each channel records why it declined,
so a fact about the stored trajectory is never counted as a wrong answer, and a
later rerun knows what is worth re-attempting.

Programmatic Verification writes one `status` per run:

| Status | Meaning |
|---|---|
| `scored` | the validator ran and returned a verdict |
| `incomplete_result` | the run never finished, or hit an infrastructure error |
| `no_snapshot` | the trajectory stored no outputs to replay |
| `query_changed` | the stored answer addressed a different question |
| `lossy_snapshot` | an output reached disk only as a repr, so the stored text is not what the validator saw live |
| `stale` | the case now declares outputs this run never held |
| `no_validator` | the case module exposes no `validate()` |
| `raised` | the validator itself crashed |

A validator that raises is recorded and the batch continues: a broken validator
is benchmark code under test, and one of them must not take an experiment down.
Per-variable projection runs the validator too, so it shares that guard — a
`scored` run is the only one that carries a `variable_verification` block.

Response Quality and Claim Coverage record a per-sample skip reason:
`incomplete_result`, `empty_response`, `no_claims` (CC only),
and `judges_below_quorum` — evidence lost to an outage, which a rerun
re-attempts. A panel that clears quorum but lost a judge is aggregated, marked
`partial_panel`, and flagged for review.

## Review-queue flags

Flags select samples for a human to read. They are not scoring parameters and
never move a reported number. Thresholds live once, in `core/scoring.py`,
beside the aggregation that applies them, and each aggregate persists the
`flag_reasons` it actually raised so no reader re-derives them:

- `rq_std:{dimension}` — judge standard deviation at or above 1.0
- `rq_range:{dimension}` — two judges at least two anchor levels apart
- `extreme_low:` / `extreme_high:` — raw overall at or beyond 2.0 / 4.5
- `partial_panel:{n}/{m}_judges_responded`
- `unreferenced_tables:{names}` — a multi-source case that passed without its
  code ever naming one of the tables it was given
- `required_claim_missed:` / `required_claim_tie:` and the conditional pair

Flags read raw scores, never capped ones: the dependency cap is a deliberate
ranking rule, and a sample must not escape review because that rule already
pulled its headline down. An exact split panel is reported as a tie rather than
a miss, because it is neither covered nor missed by majority — but ties still
enter `required_claims_missed_or_tied`, which the composite gate reads.

## Channel file layout

Each channel file repeats the shape of the `run.json` it scored: the case, one
block naming the process, then `conversations[].turns[]` with the same
conversation id and 1-based turn. A verdict, a judge score and the response they
are about therefore sit at the same address in all four files.

Every turn is written. One the channel never reached is `not_scored`, and one it
declined carries its own `status` and `reason`, so a missing verdict and a
declined one cannot look alike. The header counters are derived from the turns
written below and appended last, so they cannot disagree with the body.

## Scoring against an alternative rubric

`--criteria PATH` scores against a different rubric file without editing the
default `criteria.json`, which is what a calibration or A/B sensitivity run
needs. The path is recorded for provenance, but what gates a judged-channel
resume is `criteria_fingerprint`, which covers the rubric's content: an
alternative file with identical anchors resumes, an edited one does not.

## Resuming a judged channel

A stored sample is evidence produced under one protocol. Before resuming, each
judged channel compares the recorded criteria fingerprint, requested panel,
same-lab setting, scoring-protocol fingerprint and parameters against the
current ones. Unchanged, it resumes and never re-pays for a scored sample.
Changed, it refuses — unless `--rescore` is passed, which discards the stored
samples and recomputes them under the new settings, reporting that case as
`rescored` rather than `scored` so the discard reaches the run summary instead
of happening silently. Without that refusal a sensitivity run would keep the old
scores while advertising new settings, and the ablation table would silently
reproduce the headline table. `--rescore` also re-judges a sample whose settings
did not change, which is what the superseded `--force` did; `--force` remains as
a deprecated alias.

## Current implementation

- Rubric and weights: `criteria.json`
- Deterministic aggregation: `core/scoring.py`
- RQ prompt: `prompts/judge_prompt.md`
- CC prompt: `prompts/claim_verify_prompt.md`
- Shared registry/client/panel: `core/models.py`, `core/llm.py`,
  `core/judging.py`
- Batch RQ scorer: `scripts/score_response_quality.py`
- Batch CC verifier: `scripts/score_claim_coverage.py`
- Post-hoc PV scorer: `scripts/score_programmatic_verification.py`
- Human/LLM calibration: `scripts/calibrate_judges.py`
- Experiment report builder: `scripts/run_stats.py`

One case/model/invocation is one run directory containing immutable `run.json`
and the independently replaceable channels `programmatic_verification.json`,
`response_quality/<panel>.json` and `claim_coverage/<panel>.json`. A judged
channel keeps one file per judge panel (the sorted `--judge` names joined by
`+`), so a second panel never overwrites the first and every reader names the
panel it reads. RQ and CC are sample-idempotent within a panel, preserve skip
reasons and never store an API key. PV replays the stored runtime outputs and
refuses a changed query. The optional composite is computed downstream from the
independently persisted channels. `aggregates/<panel>/run_stats.json`,
`corpus_agreement.json`, `review_queue.jsonl` and `*_judge_health.json` are
derived, atomically replaceable study reports, one set per panel (`run_stats
--judge-panel`; without it, every panel in the study). A completed human study
additionally produces `aggregates/<panel>/human_judge_calibration.json`; they
never replace primary run/channel artifacts. Only the run-directory artifact
contract is supported; scorers reject unknown run or channel schemas.

Run statistics currently attach `task_family`, derived `source_mode`, and
multi-label `financial_domains` from the live active registry. They therefore
must be generated against the same repository snapshot as the study; the run
artifact does not freeze those classifications. Financial-domain counts are
multi-label and must not be summed as though they were disjoint cohorts.

The publication protocol retains a multi-judge quorum. `--minimum-judges 1`,
with an explicitly single-model `--judge` panel, exists to debug the scoring pipeline without spending a panel; it is not a
reduced protocol. Each scored sample records the panel it requested, so the
condition is visible per sample. A single-judge result is never reported as a
score and never mixed into a calibrated panel's numbers.
