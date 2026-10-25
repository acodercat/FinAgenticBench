# Scripts

Every command is a Python module run from the repository root with `uv run python -m ...`.

| Area | Purpose | Entry points |
| --- | --- | --- |
| Execution | Run tasks against a model and summarise a study | `scripts.run`, `scripts.run_stats` |
| Scoring | Programmatic verification, judge scoring and human calibration of the judges | `scripts.score_programmatic_verification`, `scripts.score_claim_coverage`, `scripts.score_response_quality`, `scripts.calibrate_judges` |
| Task checks | Mechanical task checks, validator probes, the generated catalog and capability coverage, suite statistics, a no-data control | `scripts.audit_case_design`, `scripts.verdict_baseline`, `scripts.catalog`, `scripts.capability_coverage`, `scripts.benchmark_stats`, `scripts.no_data_control` |
| `ingest/fetch/` | Download source-native files | `fetch_*` modules, one per source or acquisition batch |
| `ingest/build/` | Build the normalised and runtime data layers and their manifests | `scripts.ingest.build.build_dataset_layers` and the per-batch builders |
| `ingest/audit/` | Check provenance, source accessibility and data quality | per-batch audit and verification modules |

The numbered `phase*` data commands record the acquisition and transformation batches that produced the data
artifact; they are kept as they ran so the build can be traced.

Common commands:

```bash
uv run python -m scripts.run --help
uv run python -m scripts.score_programmatic_verification --experiment STUDY
uv run python -m scripts.score_claim_coverage --experiment STUDY --judge JUDGE
uv run python -m scripts.run_stats --experiment STUDY [--judge-panel PANEL]
uv run python -m scripts.audit_case_design --all-overlap
uv run python -m scripts.catalog --check
uv run python -m scripts.capability_coverage --check
uv run python -m scripts.verdict_baseline
uv run python -m scripts.ingest.build.build_dataset_layers --check
uv run python -m scripts.ingest.build.build_dataset_layers --table TABLE_ID
```

Programmatic verification writes one file beside each run. Claim coverage and response quality write one file per
judge panel, named by the sorted `--judge` names joined by `+`; `run_stats` reports one panel with `--judge-panel`
and every panel without it. `--table` rebuilds only the named governed tables in a partial data checkout.
