# core

`core/` is the benchmark implementation: the runtime, the agent harnesses, data access, validation and scoring.
Task questions, reference recomputations and validators live under `evals/`, and command-line entry points under
`scripts/`. `core/` never imports from `evals/`, and `security.py` blocks the agent's code from importing either,
so the agent cannot reach the answer key.

## Tasks and execution

| Module | Responsibility |
| --- | --- |
| `types.py` | The task schema: a task is a conversation, a conversation is turns. |
| `evaluator.py` | Loads and checks the registered tasks, builds the isolated Python runtime with only the task's tables and output variables, runs the agent turn by turn, and records the trajectory. |
| `tool_agent.py` | The default agent: executes Python through one native `python` tool call. |
| `tool_gateway.py` | The tool-calling harness's connection to the tested model. |
| `responses_model.py` | The tested model over OpenAI's Responses API. |
| `code_agent.py` | The alternative code harness (CaveAgent, which reads fenced code from the model's text), reporting why a run failed. |
| `stream_repair.py` | Repairs a model's code fences in the stream before the code harness parses them. |
| `security.py` | Blocks file, network, process and introspection access, and imports of the answer-key code, from model-executed code. |
| `errors.py` | The failure classes that decide whether the runner retries, asks again or stops. |
| `failure.py` | Why a run ended without an answer, and whether that was the model's failure or the infrastructure's. |
| `transcripts.py` | Writes and reads the full JSONL conversation of each run. |

## Data access

| Module | Responsibility |
| --- | --- |
| `runtime_catalog.py` | The single registry of tables a task can expose to its runtime. |
| `data.py` | Loaders that map a task's `data_sources` labels to runtime tables and the descriptions the agent sees. |
| `datastore.py` | Lazy, scoped access to the tables a task declares (the default `datasets` handle). |
| `dataset_layers.py` | Builds, checks and loads the data layers with their stored dtypes and hashes. |
| `dataset_assets.py` | Inventory of and controlled access to the source-native files, for rebuilds and audits. |

## Validation and scoring

| Module | Responsibility |
| --- | --- |
| `validation.py` | Shared numeric, rounding, date, Boolean, ordered-output and tie-breaking helpers used by the validators. |
| `results.py` | The run-directory layout and artifact schema, atomic writes, judge-panel paths, and suite and data fingerprints. |
| `models.py` | Loads tested-model and judge configurations from `models.toml` without writing API keys to disk. |
| `llm.py` | Provider-aware client for the judges: retries, usage capture and strict JSON extraction. |
| `judging.py` | Judge-panel selection, same-lab exclusion, concurrency and health accounting. |
| `scoring.py` | Aggregation of the three channels: dependency caps, the composite and inter-judge agreement. |

## Metadata and financial helpers

| Module | Responsibility |
| --- | --- |
| `capabilities.py` | Validates the task-to-capability assignments in `metadata/capabilities.json`. |
| `taxonomy.py` | Validates the task-family and financial-domain assignments in `metadata/case_taxonomy.json`. |
| `footnotes.py` | Parses supported SEC transaction-price footnote ranges. |
| `industry.py` | Canonical labels for the Kenneth French ten-industry portfolios. |

Repository paths are defined once, in `config.py` at the repository root.
