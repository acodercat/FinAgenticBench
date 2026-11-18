# Verification study: design and predictions

Written 2026-11-16, before any run of the study. Two smoke runs on two tasks (`experiments_verify/smoke-verify-20261116`,
DeepSeek V4 Flash verifying DeepSeek V4 Pro) checked the code and are not part of the study.

## Design

- **Answers.** Run 1 (first in run-id order) of DeepSeek V4 Pro (thinking), Gemini 3.8 Flash (thinking) and GPT-6 Luna
  (thinking), the three most accurate configurations, as stored in `experiments/`. No answering model is called.
- **Verifiers.** DeepSeek V4 Flash (thinking), DeepSeek V4 Pro (thinking), GPT-6 Luna (thinking), GPT-6.1 Sol
  (thinking), each with the settings of its existing study. DeepSeek V4 Pro is run when its endpoint is available
  again, with the same snapshot (`deepseek-v4-pro-0813`); if it cannot be run, the analyses use the other three.
- **Tasks.** `tasks.json`: the 109 tasks with a shared wrong value (`common.modal_shared`) and 40 of the other 104,
  drawn with seed 20261116.
- **Procedure.** `scripts/run_verifier.py`: the verifier works each task as any run does, and every turn's query is
  followed by the values the answering run stored for that turn's outputs, with the instruction to check each by
  recomputing it from the data and to store the value it finds correct. Scored inline by the task's validators.
  Studies: `experiments_verify/<answering>__<verifier>-20261116`.

## Measures (fixed now)

For every output of an answering run on the 149 tasks:

- **Wrong before** (the answering run stored a value the validator rejected): *fixed* when the verifier stored a value
  the validator accepts; *kept* when it stored the same wrong value.
- **Correct before**: *broken* when the verifier's output is not accepted.
- **Shared**: the wrong value was given by at least two configurations of vendors other than the answering
  configuration's (the rule of the paper's agreement analysis).
- **Verifier gives it**: the verifier's own configuration stored this wrong value on this output in at least one of
  its three independent runs (for GPT-6.1 Sol, `experiments_prospective/gpt-6.1-sol-thinking-20261115`).
- **Relation**: the answering configuration itself; same vendor (DeepSeek V4 Flash verifying DeepSeek V4 Pro, GPT-6.1
  Sol verifying GPT-6 Luna); another vendor.

Rates pool outputs over tasks; 95% intervals from 2,000 task resamples.

## Predictions

1. Pooled over verifiers, wrong outputs the verifier itself gives are fixed less often than those it does not
   (difference in fix rate above zero, interval excluding zero).
2. Pooled over verifiers, shared wrong outputs are fixed less often than unshared ones.
3. GPT-6.1 Sol fixes a larger share of shared wrong outputs than each of the other verifiers.
4. Among the relations, a configuration verifying its own answers fixes the smallest share of shared wrong outputs.
5. DeepSeek V4 Flash breaks a larger share of correct outputs than GPT-6.1 Sol.

All five are reported whatever their outcome; the measures, the task list and the classes are not changed after the
results are seen.
