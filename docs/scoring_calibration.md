# Human calibration of FinAgenticBench judges

Human calibration is a required pre-publication study, not a score that can be
inferred from benchmark runs. Use 30–50 deliberately varied, blinded answers:
numerically wrong but fluent, correct but terse, overclaimed, conventionally
wrong, and fully calibrated examples. At least two finance-competent humans must
rate every selected answer independently before viewing model identities, LLM
judge scores, or benchmark rankings.

Generate one blank JSONL row per completed trajectory:

```bash
uv run --group dev python -m scripts.calibrate_judges \
  --experiment <study> --judge-panel <panel> --write-template human_ratings.jsonl
```

Make one copy of each row per rater. Replace `rater_id`, every 1–5 RQ score, and
every boolean claim judgment. Keep the original `sample_id`; it binds the case,
tested model, and invocation without relying on row order. For conditional
claims, `true` means the answer respects the boundary (including silence when
the triggering assertion was not made), matching the LLM claim-judge contract.

After both raters finish, produce the frozen comparison report:

```bash
uv run --group dev python -m scripts.calibrate_judges \
  --experiment <study> --judge-panel <panel> --ratings human_ratings.jsonl
```

`--judge-panel` names the LLM panel being calibrated (the sorted `--judge` names joined by `+`); it may be omitted only when the study holds one panel. The command writes `aggregates/<panel>/human_judge_calibration.json`. It reports human
and LLM inter-rater reliability, per-dimension human–LLM mean absolute error and
correlation, claim-majority agreement by claim mode, ties, and missing second
ratings. It deliberately does not declare empirical acceptance thresholds or
set `publication_readiness.ready=true`; the research team must preregister those
thresholds, inspect disagreements, and explicitly approve the frozen rubric and
panel before looking at benchmark rankings. Do not tune prompts on test-model
leaderboard results.
