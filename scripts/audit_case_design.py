"""Run FinAgenticBench's mechanical case-design checks and claim-overlap warnings."""

from __future__ import annotations

import argparse
import importlib
import re
from collections import Counter

from config import BENCHMARKS_JSON
from core.evaluator import load_specs
from core.types import conversations_of


CONTENT_STOPWORDS = {
    "the", "and", "that", "this", "with", "from", "into", "each", "both",
    "same", "using", "use", "report", "reports", "response", "explains",
    "explain", "identifies", "identify", "states", "state", "discusses",
    "requested", "value", "values", "result", "results", "form", "filing",
}


def content_words(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in CONTENT_STOPWORDS
    }


def claim_query_overlap(query: str, claim: str) -> float:
    claim_words = content_words(claim)
    if not claim_words:
        return 0.0
    return len(content_words(query) & claim_words) / len(claim_words)


DATE_ANCHOR = re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b")
NUMERIC_ANCHOR = re.compile(
    r"(?<![A-Za-z0-9])[-+]?(?:\$)?(?:\d{1,3}(?:,\d{3})+|\d+\.\d+|\d+%)(?![A-Za-z0-9])"
)


def result_anchors(text: str) -> set[str]:
    """Extract dates and non-trivial numeric values that can make a claim earned."""
    return set(DATE_ANCHOR.findall(text)) | set(NUMERIC_ANCHOR.findall(text))


TEXT_REPRESENTATION_MARKERS = (
    " date", "quarter", "timestamp", "accession", "identifier", "cusip",
    " code", " token", "verbatim", "full or diagonal",
)


def has_closed_text_representation(description: str) -> bool:
    """Return whether a string Variable exposes a deterministic storage form.

    Per-task tests call this on values known to be strings. Static
    design audit deliberately does not call ``ground_truth()`` merely to infer
    runtime types, so new cases must retain the executable per-case check.
    """
    normalized = " ".join(description.lower().split())
    return any(marker in normalized for marker in TEXT_REPRESENTATION_MARKERS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all-overlap", action="store_true",
        help="print every de-anchored trap/required overlap above 40 percent",
    )
    parser.add_argument(
        "--case", action="append", dest="case_names", metavar="CASE_ID",
        help="audit only this active case; repeat the option for a batch",
    )
    args = parser.parse_args()
    specs = load_specs(BENCHMARKS_JSON)
    if args.case_names:
        requested = set(args.case_names)
        specs = [spec for spec in specs if spec["name"] in requested]
        missing = requested - {spec["name"] for spec in specs}
        if missing:
            raise ValueError(f"unknown cases: {sorted(missing)}")
    errors: list[str] = []
    overlap_candidates: list[tuple[float, str]] = []

    for spec in specs:
        module = importlib.import_module(spec["module"])
        turns = [t for c in conversations_of(spec) for t in c.turns]
        if "requirements" in spec:
            errors.append(f"{spec['name']}: deprecated requirements field")
        if any(re.search(r"\b[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+\b", t.query) for t in turns):
            errors.append(f"{spec['name']}: schema/code identifier in query")
        places = dict(zip(
            (variable.name for variable in module.variables),
            getattr(module, "DECIMALS", [None] * len(module.variables)),
        ))
        for variable in module.variables:
            if not variable.description.startswith("Store "):
                errors.append(f"{spec['name']}:{variable.name}: description lacks Store prefix")
            # The rounding gate and the near-miss probes read the precision out of the
            # description with a digit pattern, so a description that spells the number
            # out leaves that output silently unprobed.
            if places.get(variable.name) not in (None, 0) and not re.search(
                r"round(?:ed)? to \d+ decimal", variable.description
            ):
                errors.append(
                    f"{spec['name']}:{variable.name}: precision must be written in digits"
                )
            if re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(variable.name)}(?![A-Za-z0-9_])",
                " ".join(t.query for t in turns),
            ):
                errors.append(f"{spec['name']}: query contains output {variable.name}")
        # A claim is compared against every query the model has read by the time
        # it answers -- its own turn and the ones before it in the same
        # conversation -- because a claim can be satisfied by paraphrasing any
        # of them. Joining all turns, including later ones, was measuring a
        # surface the model had not seen yet.
        for conversation in conversations_of(spec):
            read_so_far: list[str] = []
            for turn in conversation.turns:
                read_so_far.append(turn.query)
                surface = " ".join(read_so_far)
                for claim in turn.claims:
                    waiver = claim.get("overlap_waiver")
                    if waiver is not None and (not isinstance(waiver, str) or not waiver.strip()):
                        errors.append(f"{spec['name']}:{claim['id']}: empty overlap waiver")
                    # Every case type, not only traps. The gate here was
                    # `case_type != "trap"`, which left 81 of 84 cases
                    # unexamined; a 2026-09-07 review found 233 required claims
                    # above this threshold that the audit had never reported.
                    if claim["mode"] != "required":
                        continue
                    overlap = claim_query_overlap(surface, claim["description"])
                    novel_anchors = result_anchors(claim["description"]) - result_anchors(surface)
                    if overlap > 0.40 and not novel_anchors and not waiver:
                        overlap_candidates.append(
                            (overlap, f"{spec['name']}:{claim['id']}")
                        )

    priority = [item for item in overlap_candidates if item[0] >= 0.80]
    print(
        f"cases={len(specs)} errors={len(errors)} "
        f"overlap_candidates={len(overlap_candidates)} "
        f"priority_overlap_warnings={len(priority)}"
    )
    print(f"case_types={dict(Counter(spec['case_type'] for spec in specs))}")
    for item in errors:
        print(f"ERROR {item}")
    shown = overlap_candidates if args.all_overlap else priority
    for overlap, name in sorted(shown, reverse=True):
        print(f"WARN  {name}: de-anchored query/claim overlap={overlap:.0%}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
