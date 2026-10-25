"""Ask a turn with the data taken away, and see what comes back correct anyway.

The question is whether a scored output is measuring an analysis or a prior. An
output a competent model can produce from the question alone -- because the
population is small, the label is one of a handful, the count is bounded, or the
figure is a familiar published number -- is not evidence that the model read the
frozen tables, and a turn whose outputs are ALL like that measures nothing in
the programmatic channel however hard it looks.

FinAgenticBench is mostly numeric (1,713 numeric outputs against 304 text and 8
boolean), so most of this will come back correctly refused. That is the point:
the instrument is worth having for the residue it finds, and the residue is
where the text and small-integer outputs are.

Method. Each probed turn gets a prompt carrying what an agent that had done the
earlier work would know, and nothing else:

  * the case's task family and the declared data sources, by name only;
  * an explicit statement that no data, files or code execution are available;
  * every earlier turn of the conversation with the CORRECT values it produced,
    taken from ``ground_truth()``, so a later turn is answerable at all;
  * this turn's query;
  * the ``Variable`` description of each output the turn registers.

The reply is compared with ``ground_truth()``: numbers within 5 per cent, text
and dates case-folded exact, booleans strictly. Five per cent is deliberately
loose -- the question is recallability, not precision.

Reading the output. A turn where every scored output came back correct is a
candidate for repair, not a proven defect: some quantities are correctly
derivable from an earlier turn's answers, which is the multi-turn design working
rather than failing. The summary separates turns whose whole output set was
recovered from turns where only some were, and names which outputs those were,
because the repair differs -- a recoverable text label wants a harder
population, a recoverable derived figure wants the earlier answer withheld.

The control feeds earlier turns' correct answers by design, so it cannot
distinguish recall from deduction. It measures reachability without the data,
which is the union of the two.

Usage:
    uv run python -m scripts.no_data_control --list
    uv run python -m scripts.no_data_control --case walmart_fy2025_revenue
    uv run python -m scripts.no_data_control --surface recallable --concurrency 8
    uv run python -m scripts.no_data_control --resume     # continue an interrupted sweep
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Lock
from typing import Any

from config import BENCHMARKS_JSON, MODELS_TOML, PROJECT_ROOT
from core.evaluator import load_specs
from core.llm import LLMClient
from core.models import ModelRegistry
from core.types import conversations_of

DEFAULT_MODELS = ("deepseek-v4-pro",)
DEFAULT_OUT = PROJECT_ROOT / "metadata" / "no_data_control.json"
# Loose on purpose: the question is whether the value is reachable, not whether
# it is reproduced to the digit.
NUMERIC_TOLERANCE = 0.05
# Only so that a truth of exactly 0.0 is not unmatchable; never a band for a
# figure with a magnitude of its own.
ZERO_FLOOR = 1e-9


def _truth(spec) -> dict[str, Any]:
    module = importlib.import_module(spec["module"])
    return dict(zip([v.name for v in module.variables], module.ground_truth()))


def _descriptions(spec) -> dict[str, str]:
    module = importlib.import_module(spec["module"])
    return {v.name: (v.description or "") for v in module.variables}


# A guessable output has few enough possible values that a prior can land on
# one: a label or date drawn from a small population, a boolean, or a count
# small enough to enumerate. The bound is a judgement, not a measurement --
# a first pass over 29 turns of long-precision figures returned nothing at all,
# which is the expected and uninformative result for the other 1,713 outputs.
SMALL_COUNT = 100


def _guessable(value: Any) -> bool:
    if isinstance(value, (bool, str)):
        return True
    if isinstance(value, int):
        return abs(value) <= SMALL_COUNT
    return False


def turns_to_probe(only_case: str | None = None, *, surface: str = "all") -> list[dict]:
    """Every turn with scored outputs, or only those that could be guessed.

    ``surface="recallable"`` keeps a turn when ANY of its outputs is of a
    guessable shape, and still probes the whole turn: the question is whether
    the turn measures something, and a turn of twelve figures and one label
    that comes back 1/13 has been answered honestly.
    """
    if surface not in ("all", "recallable"):
        raise ValueError("surface must be 'all' or 'recallable'")
    rows = []
    for spec in load_specs(BENCHMARKS_JSON):
        if only_case and spec["name"] != only_case:
            continue
        truth = _truth(spec)
        described = _descriptions(spec)
        for conversation in conversations_of(spec):
            history: list[dict] = []
            for index, turn in enumerate(conversation.turns, start=1):
                stores = list(turn.stores or [])
                scored = [name for name in stores if name in truth]
                if surface == "recallable":
                    scored = scored if any(_guessable(truth[n]) for n in scored) else []
                if scored:
                    rows.append({
                        "case": spec["name"],
                        "task_family": spec["task_family"],
                        "data_sources": list(spec["data_sources"]),
                        "conversation": conversation.id,
                        "turn": index,
                        "query": turn.query,
                        "history": list(history),
                        "descriptions": {n: described.get(n, "") for n in scored},
                        "expected": {n: truth[n] for n in scored},
                    })
                history.append({
                    "query": turn.query,
                    "answers": {n: truth[n] for n in stores if n in truth},
                })
    return rows


def build_prompt(row: dict) -> str:
    lines = [
        "You are a financial data analyst. This is a reasoning exercise: you "
        "have NO database, NO files and NO ability to run code. Answer from "
        "the information given and from what you know.",
        "",
        f"Task family: {row['task_family']}.",
        f"Sources the real task would read: {', '.join(row['data_sources'])}.",
    ]
    if row["history"]:
        lines.append("\nWork already completed, with its results:")
        for step in row["history"]:
            lines.append(f"- Asked: {step['query']}")
            if step["answers"]:
                lines.append(
                    f"  Found: {json.dumps(step['answers'], ensure_ascii=False, default=str)}"
                )
    lines += ["", f"Now: {row['query']}", "", "Report exactly these:"]
    for name, description in row["descriptions"].items():
        lines.append(f"- {name}: {description}")
    lines += [
        "",
        "Answer immediately, in one line, as a single JSON object mapping each "
        "name to its value and nothing else. Do not deliberate, show working or "
        "explain: give your best guess for every name, and never refuse.",
    ]
    return "\n".join(lines)


def _same(got: Any, expected: Any) -> bool:
    if got is None:
        return False
    if isinstance(expected, bool):
        if isinstance(got, bool):
            return got is expected
        return str(got).strip().lower() in {str(expected).lower(),
                                            "yes" if expected else "no"}
    if isinstance(expected, (int, float)):
        try:
            target = float(expected)
            # Relative, with an absolute floor only for a value at or near
            # zero. A first version used `max(1.0, abs(expected))`, carried
            # over from a benchmark whose figures were in the tens; on a
            # percentage-point RMSE of 0.115 that is a plus-or-minus 0.05 band,
            # 43 per cent wide, and it counted a guess of 0.094 as a recovery.
            width = max(NUMERIC_TOLERANCE * abs(target), ZERO_FLOOR)
            return abs(float(got) - target) <= width
        except (TypeError, ValueError):
            return False
    return _norm(got) == _norm(expected)


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        loaded = json.loads(path.read_text())
    except json.JSONDecodeError:
        return []
    return loaded if isinstance(loaded, list) else []


def _completed(path: Path) -> set[tuple]:
    """(case, turn, model) already on disk, so a resumed sweep does not re-pay."""
    return {
        (record.get("case"), record.get("turn"), record.get("model"))
        for record in _load(path)
        if record.get("error") is None
    }


def _last_json_object(text: str) -> dict | None:
    """The last balanced ``{...}`` in a reply, parsed, or None.

    Needed because a probe is not a judge call: the judged channels can insist
    on strict JSON from a panel chosen for it, while this asks arbitrary models
    a question they have every reason to answer in prose. Measured on
    deepseek-v4-flash, 31 of 53 strict-mode probes raised "malformed JSON after
    3 attempts" -- and every one was recorded as a clean 0/N, which is the same
    shape as the model failing to guess. A refused call and a refused guess
    must never print alike.
    """
    depth = start = 0
    best = None
    for index, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0:
                best = text[start:index + 1]
    if best is None:
        return None
    try:
        loaded = json.loads(best)
    except json.JSONDecodeError:
        return None
    return loaded if isinstance(loaded, dict) else None


def _ask(client: LLMClient, prompt: str) -> tuple[dict | None, str]:
    try:
        answer = client.complete_json(prompt)
        if isinstance(answer, dict):
            return answer, ""
    except Exception:                               # noqa: BLE001 - fall back below
        pass
    text = ""
    try:
        text = client.complete(prompt).text
        answer = _last_json_object(text)
    except Exception as error:                      # noqa: BLE001 - reported, not raised
        return None, f"{type(error).__name__}: {error}"
    if answer is None:
        # An empty reply is its own diagnosis and deserves its own words. On
        # these endpoints it means the generation stopped at
        # finish_reason=length with the whole token budget spent in
        # `reasoning_content` and `content` never started -- 91 of 147 probes
        # on deepseek-v4-flash, until the prompt above asked for brevity.
        return None, ("empty reply -- no content generated, the token budget "
                      "went to reasoning" if not text.strip()
                      else "no JSON object in the reply")
    return answer, ""


def _summarise(results: list[dict], out: Path) -> int:
    # A call that never returned an answer is not evidence that the model could
    # not guess. It is separated here for the same reason an unparseable agent
    # turn is separated from a wrong one: counted as a refusal it would report
    # the instrument's own failure as a property of the case.
    answered = [r for r in results if not r["error"]]
    failed = [r for r in results if r["error"]]
    whole = [r for r in answered if r["all_recovered"]]
    print(f"\n{len(results)} probes: {len(answered)} answered, {len(failed)} unanswered "
          f"(call or parse failure, excluded from everything below).")
    print(f"\n{len(whole)} turns answered COMPLETELY with no data "
          f"-- these measure nothing programmatically:")
    for record in whole:
        print(f"  {record['case']} T{record['turn']} ({record['model']}) "
              f"{len(record['recovered'])}/{len(record['matched'])}")
    scored = sum(len(r["matched"]) for r in answered)
    hit = sum(len(r["recovered"]) for r in answered)
    share = f"{100 * hit / scored:.1f} %" if scored else "n/a"
    print(f"\nper output: {hit} of {scored} recovered "
          f"({share}) across {len(answered)} answered probes")
    by_output: dict[str, set] = {}
    for record in answered:
        for output in record["recovered"]:
            by_output.setdefault(output, set()).add(record["case"])
    print(f"{len(by_output)} distinct outputs were recovered at least once.")
    if failed:
        print(f"\nrerun the {len(failed)} unanswered with --resume "
              "(errored probes are not treated as done).")
    print(f"wrote {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    parser.add_argument("--case")
    parser.add_argument("--limit", type=int, help="probe only the first N turns")
    parser.add_argument(
        "--surface", choices=("all", "recallable"), default="all",
        help="'recallable' probes only turns holding a label, boolean or small count",
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--rescore", action="store_true",
        help="re-judge the stored replies in --out under today's comparison, no calls",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=32768,
        help="token ceiling for a probe; these models reason before answering",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="skip probes already recorded in --out that did not error",
    )
    args = parser.parse_args()

    rows = turns_to_probe(args.case, surface=args.surface)
    if args.limit:
        rows = rows[: args.limit]
    if args.list:
        for row in rows:
            print(f"{row['case']:52s} T{row['turn']} {len(row['expected']):2d} outputs")
        print(f"\n{len(rows)} turns, {sum(len(r['expected']) for r in rows)} scored outputs")
        return 0

    if args.rescore:
        # The same argument the programmatic-verification channel makes: the
        # model's reply is the evidence and it is immutable, so changing how a
        # reply is judged must not cost another sweep. Recomputes every match
        # from the stored answers and rewrites the file in place.
        stored = _load(args.out)
        if not stored:
            print(f"nothing to rescore in {args.out}")
            return 1
        for record in stored:
            matched = {
                output: _same((record.get("answer") or {}).get(output), value)
                for output, value in record["expected"].items()
            }
            record["matched"] = matched
            record["recovered"] = sorted(k for k, v in matched.items() if v)
            record["all_recovered"] = bool(matched) and all(matched.values())
        args.out.write_text(
            json.dumps(stored, indent=2, ensure_ascii=False, default=str)
        )
        print(f"rescored {len(stored)} stored probes in {args.out}")
        results = stored
        return _summarise(results, args.out)

    registry = ModelRegistry.load(MODELS_TOML)
    # A probe needs a bigger budget than a judge call, not a smaller one. The
    # models here reason at length before answering, and `content` only starts
    # once that is finished: at the 8,192 configured for scoring, 91 of 147
    # probes stopped at finish_reason=length with nothing written. The ceiling
    # is raised for this client only and changes nothing about how the same
    # model is configured for the judged channels.
    clients = {
        name: LLMClient(replace(registry.get(name), max_tokens=args.max_tokens))
        for name in args.models
    }
    done = _completed(args.out) if args.resume else set()
    work = [
        (row, name) for row in rows for name in clients
        if (row["case"], row["turn"], name) not in done
    ]
    if done:
        print(f"resuming: {len(done)} probes already on disk, {len(work)} to go")
    # Errored records are dropped rather than carried: they are about to be
    # retried, and keeping both would leave two rows for one probe.
    results = [r for r in _load(args.out) if not r.get("error")] if args.resume else []
    lock = Lock()

    def probe(item):
        row, name = item
        answer, error = _ask(clients[name], build_prompt(row))
        matched = {
            output: _same(answer.get(output) if answer else None, value)
            for output, value in row["expected"].items()
        }
        return {
            **{k: row[k] for k in ("case", "turn", "expected")},
            "model": name,
            "recovered": sorted(k for k, v in matched.items() if v),
            "matched": matched,
            "all_recovered": bool(matched) and all(matched.values()),
            "answer": answer,
            "error": error or None,
        }

    # One probe is one independent question, so the sweep is embarrassingly
    # parallel and its wall time is entirely the endpoint's latency: serially,
    # 186 turns measured at 1.8 minutes each is five and a half hours.
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        for record in pool.map(probe, work):
            with lock:
                results.append(record)
                share = (
                    "ERR" if record["error"]
                    else f"{sum(record['matched'].values())}/{len(record['matched'])}"
                )
                print(f"{record['case']:52s} T{record['turn']} "
                      f"{record['model']:18s} {share}", flush=True)
                # Written after every probe. A full sweep is hours of calls, and
                # losing all of it to one interruption is the avoidable half of
                # that cost; `--resume` reads it back.
                args.out.write_text(
                    json.dumps(results, indent=2, ensure_ascii=False, default=str)
                )


    return _summarise(results, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
