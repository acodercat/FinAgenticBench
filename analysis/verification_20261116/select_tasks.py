"""Tasks for the verification study: every task with a shared wrong value, and a fixed random sample of the rest.

A task has a shared wrong value when at least one of its slots has one (common.modal_shared: the slot's most common
wrong value, given by at least three of the fourteen configurations). The other tasks give the rate at which a
verifier breaks outputs that were correct. Run from the repository root; writes
analysis/verification_20261116/tasks.json.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, "analysis/final_20261012/scripts")
from common import load, modal_shared  # noqa: E402
from groups import STUDIES  # noqa: E402

N_OTHER = 40
SEED = 20261116

cases, _, vars_ = load()
shared = sorted({task for task, _, _ in modal_shared([v for v in vars_ if v["study"] in STUDIES])})
rest = sorted(set(cases) - set(shared))
other = sorted(random.Random(SEED).sample(rest, N_OTHER))
out = Path("analysis/verification_20261116/tasks.json")
out.write_text(json.dumps({"shared": shared, "other": other, "seed": SEED}, indent=1) + "\n")
print(f"tasks with a shared wrong value {len(shared)}, others sampled {len(other)} of {len(rest)} -> {out}")
