# Experiment artifacts

The runs behind the paper: eight models, fourteen configurations, 213 tasks, three runs each (8,946 runs). The
configuration directories are not in Git; they are in `FinAgenticBench-runs.zip` (link in the
[README](../README.md#data-and-run-artifacts)), which unzips into this directory.

```text
experiments/<configuration>/
  runs/<task>/<model>/<run_id>/
    run.json                                  per-turn query, response, code, stored outputs, usage; no verdict
    transcripts/main.jsonl                    full message log
    programmatic_verification.json            PV verdicts, replayable from run.json
    claim_coverage/<panel>.json               claim coverage, one file per judge panel
    response_quality/<panel>.json             response quality, one file per judge panel
  aggregates/<panel>/
    run_stats.json, corpus_agreement.json, review_queue.jsonl
```

`<configuration>` names the model, its reasoning mode and the date the study started; `<model>` is the API model
identifier the runs were sent to.

A judge panel is named by its sorted `--judge` names joined by `+`. The paper's three claim-coverage judges
each scored alone:

| Panel | Judge | Claim coverage | Response quality |
|---|---|---|---|
| `deepseek-v4-flash-oneapi` | DeepSeek V4 Flash (reasoning off, temperature 0.1) | all 8,946 runs | 8,778 runs |
| `gpt-6-luna-nonthinking` | GPT-6 Luna (reasoning none, temperature 0.1) | all 8,946 runs | 8,778 runs |
| `glm-5.3-flash` | GLM-5.3 Flash (reasoning off, temperature 0.1) | all 8,946 runs | — |

Claim coverage was scored on 2026-10-12 by all three judges. Response quality was scored earlier, on 2026-10-10
(DeepSeek V4 Flash) and 2026-10-11 (GPT-6 Luna), before four tasks were reworded and their 168 runs re-executed on
2026-10-12; those runs received programmatic verification and claim coverage only, which is why 168 runs have no
response-quality file. Response quality was piloted and is not used in the paper. The claim-coverage comparison
is `analysis/final_20261012/scripts/cc_vs_pv.py` (Supplementary Table 17).

Code that reads a judged channel names the panel: `core.results.channel_path(run, "claim_coverage", panel)`
refuses a judged channel without one, `scripts.run_stats --judge-panel <panel>` reports one panel (without it,
every panel in the study), and `scripts.calibrate_judges` requires the panel when a study has several.

Internal endpoint addresses and local paths in the run records were replaced in the released archive
(`https://redacted.invalid`, `/workspace/finagenticbench`); stored outputs, verdicts and scores are unchanged.
