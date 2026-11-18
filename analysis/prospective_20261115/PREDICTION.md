# Prospective prediction: GPT-6.1 Sol (thinking)

Written 2026-11-15, before any run of the study below was launched. Only the smoke run
`smoke-gpt-6.1-sol-20261115` (one task, one run) existed, and it is not part of the test.

## Run

- Study `gpt-6.1-sol-thinking-20261115`; model section `gpt-6.1-sol-thinking` (API model `gpt-6.1-sol`,
  `openai_responses`, reasoning effort medium, temperature 1.0, max_tokens 65536), the same settings as
  `gpt-6-luna-thinking`.
- All 213 tasks, three runs, tool harness, 30-step budget, lazy injection, no protocol nudges, PV scored inline.
  Code at f1f6582, whose harness and runner are identical to cf52026 (2026-10-09), the last harness change
  before the fourteen studies were completed.
- Unanswered runs and outages are handled as in the fourteen studies.

## Measures (fixed now)

- **Accuracy (A).** Task accuracy: a task passes a run only if every turn passes; averaged over the 639 runs.
- **Convergence (C).** Of Sol's wrong values on set slots, the share equal to the consensus error, where the
  consensus error of a slot is the most common wrong value among the **fourteen existing configurations only**,
  counted when at least two of them gave it. The fourteen are not recomputed with Sol added.
- **Within-slot contrast (D).** On slots where Sol and at least one of the four strong configurations
  erred, Sol's share minus the strong tier's pooled share, the same consensus as in C; 95% interval from
  2,000 task resamples.

## Predictions

**P1 (primary).** C lies inside the 95% prediction interval of the line fitted to the fourteen
configurations, C = −0.3152 + 1.1221 A (residual s.d. 0.0743, n = 14), evaluated at Sol's observed A:

| A | predicted C | 95% prediction interval |
|---|---|---|
| 0.80 | 0.582 | 0.408 to 0.757 |
| 0.84 | 0.627 | 0.449 to 0.806 |
| 0.88 | 0.672 | 0.488 to 0.856 |
| 0.90 | 0.695 | 0.507 to 0.882 |
| 0.92 | 0.717 | 0.527 to 0.908 |
| 0.95 | 0.751 | 0.555 to 0.947 |

For other values of A, the interval is computed by the same formula
(t₁₂ = 2.179, mean A of the fourteen and its sum of squares as in `predict_convergence.py`).

**P2 (secondary, sharper).** D has the sign of A − 0.823 (the mean accuracy of the four strong
configurations), with expected size 6.0 points per 10 points of accuracy difference (slot fixed-effects
coefficient, interval 3.8 to 8.1). P2 is evaluated only if |A − 0.823| ≥ 0.05; otherwise D is reported
without a verdict.

## Decision rules

- A > 0.84 and C inside the interval: the relation holds beyond the accuracy range of the fourteen.
- A > 0.84 and C below the interval: the relation does not extend beyond 84%; the title and abstract are
  restricted to the observed range.
- C above the interval: reported as stronger convergence than predicted.
- A ≤ 0.84: reported as an out-of-sample test of the line only, not as evidence about higher accuracy.

The measures, the fourteen-configuration consensus and these thresholds are not changed after the results are seen.
