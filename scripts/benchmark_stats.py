"""Report benchmark coverage by task category and difficulty."""

from __future__ import annotations

import json
from collections import Counter
from math import floor

from config import BENCHMARKS_JSON
from core.evaluator import load_specs
from core.taxonomy import load_case_taxonomy, validate_case_taxonomy


DIFFICULTIES = ("easy", "medium", "hard")


def target_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    """Largest-remainder allocation with stable easy/medium/hard tie-breaking."""
    ratio_total = sum(ratios.values())
    if ratio_total <= 0 or any(ratios[name] < 0 for name in DIFFICULTIES):
        raise ValueError("difficulty ratios must be non-negative with a positive sum")
    exact = {name: total * ratios[name] / ratio_total for name in DIFFICULTIES}
    result = {name: floor(exact[name]) for name in DIFFICULTIES}
    remaining = total - sum(result.values())
    ranked = sorted(
        DIFFICULTIES,
        key=lambda name: (exact[name] - result[name], -DIFFICULTIES.index(name)),
        reverse=True,
    )
    for name in ranked[:remaining]:
        result[name] += 1
    return result


def main() -> None:
    registry_path = BENCHMARKS_JSON
    registry = json.loads(registry_path.read_text())
    specs = load_specs(registry_path)
    case_taxonomy = load_case_taxonomy()
    errors = validate_case_taxonomy(case_taxonomy, specs)
    if errors:
        raise ValueError("invalid case taxonomy:\n- " + "\n- ".join(errors))
    actual = Counter(spec["difficulty"] for spec in specs)
    taxonomy = Counter(spec["case_type"] for spec in specs)
    ratios = registry.get("difficulty_ratio")
    targets = target_counts(len(specs), ratios) if ratios else None

    print(f"cases: {len(specs)}")
    print(f"taxonomy: baseline={taxonomy['baseline']} trap={taxonomy['trap']}")
    print("difficulty:")
    for name in DIFFICULTIES:
        target = f" proportional_target={targets[name]}" if targets else ""
        print(f"  {name}: actual={actual[name]}{target}")
    if not ratios:
        print(f"difficulty_policy: {registry.get('difficulty_policy', 'unspecified')}")
    print("task_families:")
    for family, count in sorted(Counter(spec["task_family"] for spec in specs).items()):
        by_difficulty = Counter(
            spec["difficulty"] for spec in specs if spec["task_family"] == family
        )
        by_type = Counter(
            spec["case_type"] for spec in specs if spec["task_family"] == family
        )
        easy_baselines = sum(
            spec["difficulty"] == "easy" and spec["case_type"] == "baseline"
            for spec in specs if spec["task_family"] == family
        )
        mix = ", ".join(f"{name}={by_difficulty[name]}" for name in DIFFICULTIES)
        print(
            f"  {family}: {count} ({mix}; baseline={by_type['baseline']}, "
            f"trap={by_type['trap']}, easy_baseline={easy_baselines})"
        )
    print("source_mode:")
    source_modes = Counter(
        "single" if len(set(spec["data_sources"])) == 1 else "multi"
        for spec in specs
    )
    for mode in ("single", "multi"):
        print(f"  {mode}: {source_modes[mode]}")
    print("financial_domains (multi-label):")
    for domain in case_taxonomy["financial_domains"]:
        count = sum(
            domain in case_taxonomy["cases"][spec["name"]]["financial_domains"]
            for spec in specs
        )
        print(f"  {domain}: {count}")


if __name__ == "__main__":
    main()
