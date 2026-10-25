"""Deterministic aggregation for FinAgenticBench's three evaluation channels."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CLAIM_MODES = {"required", "optional", "conditional"}
COMPOSITE_WEIGHT_SCENARIOS = {
    "primary_60_25_15": {
        "programmatic_verification": 0.60,
        "response_quality": 0.25,
        "claim_coverage": 0.15,
    },
    "pv_heavy_70_20_10": {
        "programmatic_verification": 0.70,
        "response_quality": 0.20,
        "claim_coverage": 0.10,
    },
    "balanced_50_30_20": {
        "programmatic_verification": 0.50,
        "response_quality": 0.30,
        "claim_coverage": 0.20,
    },
}

# Review-queue thresholds. These select samples for a human to look at; they are
# not scoring parameters and never move a reported number. They live here, with
# the aggregation that applies them, so no reader has to re-derive them from a
# second copy and drift from what was actually flagged.
SCORE_STD_FLAG = 1.0
SCORE_RANGE_FLAG = 2
EXTREME_LOW_FLAG = 2.0
EXTREME_HIGH_FLAG = 4.5


def claim_mode(claim: dict[str, Any]) -> str:
    """Validate and return the claim mode required by the current case schema."""
    if "mode" not in claim:
        raise ValueError("claim is missing required mode")
    mode = claim["mode"]
    if mode not in CLAIM_MODES:
        raise ValueError(f"invalid claim mode {mode!r}")
    return mode


@dataclass(frozen=True)
class Dimension:
    id: str
    name: str
    description: str
    weight: float
    anchors: dict[int, str]


@dataclass(frozen=True)
class Criteria:
    dimensions: tuple[Dimension, ...]
    channel_weights: dict[str, float]
    gates: dict[str, float]

    @classmethod
    def from_json(cls, path: Path) -> "Criteria":
        raw = json.loads(path.read_text())
        dimensions = tuple(
            Dimension(
                id=item["id"], name=item["name"], description=item["description"],
                weight=float(item["weight"]),
                anchors={int(level): text for level, text in item["anchors"].items()},
            )
            for item in raw["dimensions"]
        )
        criteria = cls(dimensions, raw["channel_weights"], raw["gates"])
        criteria.validate()
        return criteria

    def validate(self) -> None:
        ids = [dimension.id for dimension in self.dimensions]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("criteria must have unique, non-empty dimensions")
        for dimension in self.dimensions:
            if dimension.weight <= 0 or set(dimension.anchors) != {1, 2, 3, 4, 5}:
                raise ValueError(f"invalid dimension {dimension.id}")
        if set(self.channel_weights) != {
            "programmatic_verification", "response_quality", "claim_coverage"
        } or abs(sum(self.channel_weights.values()) - 1.0) > 1e-9:
            raise ValueError("channel weights must name all three channels and sum to 1")
        trigger = int(self.gates["dependency_trigger"])
        cap = int(self.gates["downstream_cap"])
        if cap <= trigger:
            # A cap at or below the trigger would itself trigger the cap, so the
            # rule would depend on evaluation order rather than on the scores.
            raise ValueError(
                f"downstream_cap ({cap}) must exceed dependency_trigger ({trigger})"
            )

    def fingerprint(self) -> str:
        canonical = json.dumps(
            {
                "dimensions": [dimension.__dict__ for dimension in self.dimensions],
                "channel_weights": self.channel_weights,
                "gates": self.gates,
            },
            sort_keys=True, ensure_ascii=False,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def validate_judge_scores(scores: dict[str, Any], criteria: Criteria) -> dict[str, int]:
    expected = {dimension.id for dimension in criteria.dimensions}
    if set(scores) != expected:
        raise ValueError(f"judge score ids differ: got={set(scores)}, expected={expected}")
    normalized = {}
    for key, value in scores.items():
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
            raise ValueError(f"{key} must be an integer from 1 to 5")
        normalized[key] = value
    return normalized


def validate_quality_judge(judge: dict[str, Any], criteria: Criteria) -> dict[str, Any]:
    scores = validate_judge_scores(judge.get("scores", {}), criteria)
    expected = set(scores)
    justifications = judge.get("justifications", {})
    if set(justifications) != expected or not all(
        isinstance(value, str) and value.strip() for value in justifications.values()
    ):
        raise ValueError("quality judge must justify every dimension")
    return {**judge, "scores": scores, "justifications": justifications}


def validate_claim_judge(
    judge: dict[str, Any], claims: list[dict[str, Any]]
) -> dict[str, Any]:
    expected = {claim["id"] for claim in claims}
    coverage = judge.get("coverage", {})
    justifications = judge.get("justifications", {})
    if set(coverage) != expected or any(not isinstance(value, bool) for value in coverage.values()):
        raise ValueError("claim judge must return one boolean per claim")
    if set(justifications) != expected or not all(
        isinstance(value, str) and value.strip() for value in justifications.values()
    ):
        raise ValueError("claim judge must justify every claim")
    return {**judge, "coverage": coverage, "justifications": justifications}


def apply_dependency_cap(scores: dict[str, int], criteria: Criteria) -> dict[str, int]:
    scores = validate_judge_scores(scores, criteria)
    trigger = int(criteria.gates["dependency_trigger"])
    cap = int(criteria.gates["downstream_cap"])
    capped = dict(scores)
    upstream_failed = False
    for dimension in criteria.dimensions:
        if upstream_failed:
            capped[dimension.id] = min(capped[dimension.id], cap)
        if scores[dimension.id] <= trigger:
            upstream_failed = True
    return capped


def aggregate_quality(judges: list[dict[str, Any]], criteria: Criteria) -> dict[str, Any]:
    if not judges:
        raise ValueError("at least one judge score is required")
    judges = [validate_quality_judge(judge, criteria) for judge in judges]
    raw = [judge["scores"] for judge in judges]
    capped = [apply_dependency_cap(scores, criteria) for scores in raw]
    means, capped_means, medians, stds, ranges = {}, {}, {}, {}, {}
    for dimension in criteria.dimensions:
        values = [scores[dimension.id] for scores in raw]
        capped_values = [scores[dimension.id] for scores in capped]
        means[dimension.id] = sum(values) / len(values)
        capped_means[dimension.id] = sum(capped_values) / len(capped_values)
        medians[dimension.id] = float(statistics.median(values))
        stds[dimension.id] = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
        ranges[dimension.id] = max(values) - min(values)
    total_weight = sum(dimension.weight for dimension in criteria.dimensions)
    overall_raw = sum(d.weight * means[d.id] for d in criteria.dimensions) / total_weight
    overall = sum(d.weight * capped_means[d.id] for d in criteria.dimensions) / total_weight
    # Flags read the raw scores, never the capped ones: the cap is a deliberate
    # ranking rule, and a sample must not escape review because the rule already
    # pulled its headline down.
    reasons = []
    for dimension in criteria.dimensions:
        if stds[dimension.id] >= SCORE_STD_FLAG:
            reasons.append(f"rq_std:{dimension.id}={stds[dimension.id]:.3g}")
        if ranges[dimension.id] >= SCORE_RANGE_FLAG:
            reasons.append(f"rq_range:{dimension.id}={ranges[dimension.id]}")
    if overall_raw <= EXTREME_LOW_FLAG:
        reasons.append(f"extreme_low:overall_raw={overall_raw:.2f}<={EXTREME_LOW_FLAG}")
    if overall_raw >= EXTREME_HIGH_FLAG:
        reasons.append(f"extreme_high:overall_raw={overall_raw:.2f}>={EXTREME_HIGH_FLAG}")
    return {
        "n_judges": len(judges), "score_mean": means,
        "score_mean_capped": capped_means, "score_median": medians,
        "score_std": stds, "score_range": ranges, "overall_score_raw": overall_raw,
        "overall_score": overall, "judges": judges,
        "flag_reasons": reasons,
        "flagged_for_review": bool(reasons),
    }


def aggregate_claims(
    judges: list[dict[str, Any]], claims: list[dict[str, Any]], threshold: float = 0.5
) -> dict[str, Any]:
    if not judges or not claims:
        raise ValueError("claim aggregation requires judges and claims")
    judges = [validate_claim_judge(judge, claims) for judge in judges]
    claim_ids = {claim["id"] for claim in claims}
    rates = {}
    for claim_id in claim_ids:
        votes = [bool(judge["coverage"].get(claim_id, False)) for judge in judges]
        rates[claim_id] = sum(votes) / len(votes)
    # A rate exactly on the threshold is an even panel splitting down the middle:
    # neither covered nor missed by majority. It still needs review and still
    # gates the composite, but calling it "missed" would overstate what the
    # judges actually agreed on, so the two are reported separately.
    def _split(mode: str) -> tuple[list[str], list[str]]:
        below = [claim["id"] for claim in claims
                 if claim_mode(claim) == mode and rates[claim["id"]] < threshold]
        tied = [claim["id"] for claim in claims
                if claim_mode(claim) == mode and rates[claim["id"]] == threshold]
        return below, tied

    missed, missed_tied = _split("required")
    violated, violated_tied = _split("conditional")
    reasons = (
        [f"required_claim_missed:{claim_id}" for claim_id in missed]
        + [f"required_claim_tie:{claim_id}" for claim_id in missed_tied]
        + [f"conditional_claim_violated:{claim_id}" for claim_id in violated]
        + [f"conditional_claim_tie:{claim_id}" for claim_id in violated_tied]
    )
    return {
        "n_judges": len(judges), "coverage_rate": rates,
        "required_claims_missed": missed,
        "required_claims_tied": missed_tied,
        "required_claims_missed_or_tied": missed + missed_tied,
        "conditional_claims_violated": violated,
        "conditional_claims_tied": violated_tied,
        "conditional_claims_violated_or_tied": violated + violated_tied,
        "flag_reasons": reasons,
        "flagged_for_review": bool(reasons), "judges": judges,
    }


def composite_score(
    pv_passed: bool, quality_score: float, claim_coverage: float,
    required_claim_missed: bool, criteria: Criteria,
    channel_weights: dict[str, float] | None = None,
) -> float:
    """Secondary 0-100 composite; channel scores remain the primary report."""
    rq_normalized = (quality_score - 1.0) / 4.0
    weights = channel_weights or criteria.channel_weights
    if set(weights) != {
        "programmatic_verification", "response_quality", "claim_coverage"
    } or abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError("composite weights must name all channels and sum to 1")
    score = 100 * (
        weights["programmatic_verification"] * float(pv_passed)
        + weights["response_quality"] * rq_normalized
        + weights["claim_coverage"] * claim_coverage
    )
    if not pv_passed:
        score = min(score, criteria.gates["programmatic_failure_max_composite"])
    if required_claim_missed:
        score = min(score, criteria.gates["required_claim_majority_missed_max_composite"])
    return round(score, 2)


def pass_hat_k(num_trials: int, success_count: int, k: int) -> float:
    """The probability that k independent trials of one case all succeed.

    Mean accuracy answers "how often does this pass"; this answers "how often
    does it pass every time", which is what an analyst relying on the answer
    needs. Introduced by tau-bench (arXiv 2406.12045) as pass^k, and computed
    the same way: the share of k-subsets of the observed trials that contain
    only successes.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    if num_trials < k:
        raise ValueError(f"{num_trials} trials cannot support pass^{k}")
    if not 0 <= success_count <= num_trials:
        raise ValueError("success_count must lie between zero and num_trials")
    return math.comb(success_count, k) / math.comb(num_trials, k)


def krippendorff_alpha(
    units: list[list[float | int]], metric: str = "interval"
) -> float | None:
    """Krippendorff's alpha for variable-size, fully observed rating units."""
    if metric not in {"interval", "nominal"}:
        raise ValueError("metric must be 'interval' or 'nominal'")
    usable = [list(unit) for unit in units if len(unit) >= 2]
    if not usable:
        return None

    values = sorted({value for unit in usable for value in unit})
    coincidence = {(left, right): 0.0 for left in values for right in values}
    for unit in usable:
        denominator = len(unit) - 1
        for left_index, left in enumerate(unit):
            for right_index, right in enumerate(unit):
                if left_index != right_index:
                    coincidence[left, right] += 1.0 / denominator

    marginals = {
        value: sum(coincidence[value, other] for other in values)
        for value in values
    }
    total = sum(marginals.values())
    if total <= 1:
        return None

    def distance(left, right) -> float:
        if metric == "nominal":
            return float(left != right)
        return float(left - right) ** 2

    observed = sum(
        coincidence[left, right] * distance(left, right)
        for left in values for right in values
    ) / total
    expected = sum(
        (
            marginals[left] * (marginals[right] - float(left == right))
            / (total - 1)
        ) * distance(left, right)
        for left in values for right in values
    ) / total
    if expected == 0:
        return 1.0 if observed == 0 else None
    return 1.0 - observed / expected


def compute_corpus_agreement(
    quality_samples: list[dict[str, Any]], criteria: Criteria
) -> dict[str, Any]:
    """Compute corpus-level agreement; per-case panels are too small alone."""
    dimensions = {}
    for dimension in criteria.dimensions:
        units = [
            [judge["scores"][dimension.id] for judge in sample.get("judges", [])]
            for sample in quality_samples
        ]
        units = [unit for unit in units if len(unit) >= 2]
        dimensions[dimension.id] = {
            "n_units": len(units),
            "n_ratings": sum(len(unit) for unit in units),
            "interval_alpha": krippendorff_alpha(units, "interval"),
            "nominal_alpha": krippendorff_alpha(units, "nominal"),
        }
    return {"n_samples": len(quality_samples), "dimensions": dimensions}
