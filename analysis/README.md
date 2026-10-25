# Reproducing the paper

`final_20261012/` rebuilds every table and log behind the paper from the run archive: fourteen configurations,
213 tasks, three runs each, as scored on 2026-10-12. All results use programmatic verification except the
claim-coverage comparison, which reads the three judges' claim-coverage files.

```text
final_20261012/
  scripts/   one script per analysis; each writes a table to tables/ and prints the log kept in logs/
  tables/    the CSV and JSON tables the paper's figures and numbers are drawn from
  logs/      the console output of each script
  inputs/    inputs that cannot be rebuilt from the current runs (see below)
  knowledge_classification_20261012.json   the root-cause classification of the most widely shared errors
  knowledge_added_20261012.json            reader classifications of slots added after the task rewording
  spot_cnd.md                              hand check of the correct-value-printed detector (95% precision)
```

## Order

Unzip `FinAgenticBench-runs.zip` at the repository root first. Run everything from the repository root;
intermediate files go to `analysis/scratch/`, which is ignored by Git.

```bash
S=analysis/final_20261012/scripts
export PYTHONPATH=$S
# 1. Base data set (about 20 minutes): task, turn and variable tables, and error shapes
uv run python $S/extract.py
uv run python $S/magnitude.py
# 2. Per-turn delivery and self-check flags (about 20 minutes; reads run.json and transcripts)
uv run python $S/turn_flags.py
# 3. Group definitions: prints each configuration's task accuracy and tier
uv run python $S/groups.py
# 4. Analyses (any order; each writes one or more tables)
for s in shared_errors thinking_by_error_class propagation agreement_signal retry_vs_switch intuitions \
         tier_axis failure_mix delivery_by_model task_family_by_tier spot_checks sensitivity convergence \
         headline cc_vs_pv; do
  uv run python $S/$s.py | tee analysis/final_20261012/logs/$s.log
done
# 5. Knowledge classification: recompute the slots shared by five or more configurations, compare with the
#    pre-revision set, and build the final classification and its rates by tier and mode
uv run python $S/shared_wrong_slots.py
uv run python $S/knowledge_by_tier.py
uv run python $S/slot_example.py        # Supplementary Table 19: one slot, all 42 runs
```

`final_20261012/scripts/groups.py` is the single definition of the configuration groups: reasoning mode, model pairs, vendors,
the capability tiers (strong at least 78% task accuracy, mid 60–78%, weak below 60%) and the size pairs.

## Inputs

Four tasks were reworded on 12 October 2026 and their 168 runs re-executed, so two inputs record the runs from
before the rewording and cannot be rebuilt from the current archive:

- `inputs/shared_wrong_prerevision_20261011.json`, the shared-wrong slot set the knowledge classification was first
  made on, which `shared_wrong_slots.py` compares against;
- `inputs/final_data_prerevision_20261011.json` (71 MB, shipped in the run archive), used by `sensitivity.py` for
  the scenario on the runs from before the rewording; without it that scenario is skipped.

`inputs/knowledge_classification_prerefix.json` is the reader's classification of those slots.
`knowledge_by_tier.py` merges it with `knowledge_added_20261012.json` and writes
`knowledge_classification_20261012.json`. Two records were reclassified after review; each carries a
`reclassified` field with its former class, the date and the reason.

## Measurement notes

- The per-turn validators of 21 tasks fill every other turn's outputs with the reference answers before validating;
  the stored verdicts carry those projected outputs as set and correct with no value. `extract.py` keeps only the
  outputs each turn declares, which removes 20,034 such rows and leaves 161,359 verdicts.
- The correct-value-printed detector searches only the tool output of the same turn, with digit boundaries, and
  requires at least four significant digits in the reference; it fires on 438 of 10,634 wrong numeric outputs.
  `spot_cnd.md` records the hand check of two samples of twenty hits.
