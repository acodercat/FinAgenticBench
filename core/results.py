"""Stable experiment/run artifact contract shared by runners and scorers.

Each run is a directory. Derived scoring channels therefore cannot be
mistaken for trajectories by a ``*.json`` glob, and one study can retain
multiple models and repeated invocations without overwriting earlier output.

The only supported result layout is::

    experiments/<study>/runs/<case>/<model>/<run_id>/
        run.json
        programmatic_verification.json
        response_quality/<panel>.json
        claim_coverage/<panel>.json
        transcripts/<conversation>.jsonl
    experiments/<study>/aggregates/<panel>/

Programmatic verification has no judge. The two judged channels keep one file
per judge panel, named by ``panel_label`` from the judge names the scorer was
given, so a second panel never overwrites the first and every reader states
which panel it reads.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Iterator

import numpy as np

from config import (
    BENCHMARKS_JSON, CAPABILITIES_PATH, CASE_TAXONOMY_PATH, EVALS_DIR,
    EXPERIMENTS_DIR, DATASETS_DIR, PROJECT_ROOT,
)
from core.types import conversations_of


RESULT_SCHEMA = "finbench-run-v4"
RUNS_DIR = "runs"
RUN_FILE = "run.json"
PV_CHANNEL = "programmatic_verification"
JUDGED_CHANNELS = ("response_quality", "claim_coverage")
CHANNEL_FILES = {PV_CHANNEL: "programmatic_verification.json"}
AGGREGATES_DIR = "aggregates"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id(started_at: datetime | None = None) -> str:
    """Return a sortable invocation id with enough precision for retries."""
    value = started_at or datetime.now()
    return value.strftime("%Y%m%d_%H%M%S_%f")


def sanitize_component(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-_")
    if not safe:
        raise ValueError("identifier does not contain a filesystem-safe character")
    return safe


def _lossless_default(value):
    """The conversions that preserve a value; anything else can only be text."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def json_default(value):
    try:
        return _lossless_default(value)
    except TypeError:
        return str(value)


def unserializable_outputs(values: dict[str, Any]) -> list[str]:
    """Names that `json_default` could only store as a repr.

    Stringifying is the right fallback for writing a trajectory, but it means
    the stored text is not what the validator saw live. Replaying it would
    report a wrong value for what is really a storage limit, so the names are
    recorded and the verification channel refuses them. Derived from
    `_lossless_default` rather than restating it, so the two cannot drift.
    """
    lossy = []
    for name, value in values.items():
        try:
            json.dumps(value, default=_lossless_default)
        except (TypeError, ValueError):
            lossy.append(name)
    return sorted(lossy)


def _string_keys(value: Any) -> Any:
    """*value* with every mapping key JSON can hold: a key it cannot (a tuple from
    a groupby, a numpy integer, a Timestamp) is written as its repr, where
    `json.dumps` would raise; one it can is left for `json` to write. Values are
    left to `json_default`."""
    if isinstance(value, dict):
        return {(k if k is None or isinstance(k, (str, int, float)) else repr(k)): _string_keys(v)
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_string_keys(v) for v in value]
    return value


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    content = json.dumps(
        _string_keys(payload), indent=2, ensure_ascii=False, default=json_default
    ) + "\n"
    atomic_write_text(path, content)


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)   # known before a write that may fail
            temporary.write(content)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def result_path(
    experiment_dir: Path,
    case_name: str,
    api_model: str,
    run_id: str = "run",
) -> Path:
    """Return the trajectory marker for one case/model/invocation."""
    return (
        experiment_dir / RUNS_DIR / sanitize_component(case_name)
        / sanitize_component(api_model) / sanitize_component(run_id) / RUN_FILE
    )


def panel_label(judge_names: list[str] | tuple[str, ...]) -> str:
    """The file and directory name of a judge panel: its sorted judge names."""
    names = sorted({sanitize_component(name) for name in judge_names})
    if not names:
        raise ValueError("a judge panel needs at least one judge")
    return "+".join(names)


def _check_panel(panel: str) -> str:
    if not panel or any(sanitize_component(part) != part for part in panel.split("+")):
        raise ValueError(f"{panel!r} is not a judge panel label")
    return panel


def channel_path(result_file: Path, channel: str, panel: str | None = None) -> Path:
    """Resolve a scoring channel of a current run artifact.

    Programmatic verification sits next to ``run.json``; a judged channel sits in
    ``<channel>/<panel>.json`` and cannot be resolved without its panel.
    """
    if result_file.name != RUN_FILE:
        raise ValueError(f"{result_file}: expected a {RUN_FILE} artifact")
    if channel == PV_CHANNEL:
        if panel is not None:
            raise ValueError("programmatic verification has no judge panel")
        return result_file.parent / CHANNEL_FILES[PV_CHANNEL]
    if channel not in JUDGED_CHANNELS:
        raise ValueError(
            f"unknown channel {channel!r}; expected {sorted((PV_CHANNEL, *JUDGED_CHANNELS))}"
        )
    if panel is None:
        raise ValueError(f"channel {channel!r} is stored per judge panel; name the panel")
    return result_file.parent / channel / f"{_check_panel(panel)}.json"


def channel_panels(result_file: Path, channel: str) -> list[str]:
    """Judge panels that have scored this run on a judged channel."""
    if channel not in JUDGED_CHANNELS:
        raise ValueError(f"{channel!r} is not a judged channel")
    directory = result_file.parent / channel
    return sorted(path.stem for path in directory.glob("*.json")) if directory.is_dir() else []


def experiment_panels(experiment_dir: Path) -> list[str]:
    """Every judge panel that has scored any run of a study, on either judged channel."""
    found: set[str] = set()
    for _, path in iter_result_files(experiment_dir):
        for channel in JUDGED_CHANNELS:
            found.update(channel_panels(path, channel))
    return sorted(found)


def aggregates_dir(experiment_dir: Path, panel: str | None) -> Path:
    """Study-level reports: one directory per judge panel, the root when none."""
    root = experiment_dir / AGGREGATES_DIR
    return root / _check_panel(panel) if panel is not None else root


def result_identity(result_file: Path) -> tuple[str, str, str]:
    """Return ``(case_id, model_id, run_id)`` from a current run path."""
    if result_file.name != RUN_FILE:
        raise ValueError(f"{result_file}: expected a {RUN_FILE} artifact")
    try:
        runs_index = result_file.parts.index(RUNS_DIR)
        case_name, model_id, run_id = result_file.parts[runs_index + 1:runs_index + 4]
    except (ValueError, IndexError) as error:
        raise ValueError(f"{result_file}: malformed run path") from error
    return case_name, model_id, run_id


def iter_result_files(
    experiment_dir: Path,
    case_names: set[str] | None = None,
) -> Iterator[tuple[str, Path]]:
    """Yield every current run artifact in a study."""
    if not experiment_dir.exists():
        raise FileNotFoundError(experiment_dir)
    runs_root = experiment_dir / RUNS_DIR
    if not runs_root.is_dir():
        return
    for path in sorted(runs_root.glob(f"*/*/*/{RUN_FILE}")):
        case_name, _, _ = result_identity(path)
        if not case_names or case_name in case_names:
            yield case_name, path


def latest_result_path(
    experiment_dir: Path,
    case_name: str,
    api_model: str,
) -> Path | None:
    model_dir = (
        experiment_dir / RUNS_DIR / sanitize_component(case_name)
        / sanitize_component(api_model)
    )
    candidates = sorted(model_dir.glob(f"*/{RUN_FILE}")) if model_dir.is_dir() else []
    return candidates[-1] if candidates else None


def resolve_experiment(value: str | Path) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = EXPERIMENTS_DIR / candidate
    resolved = candidate.resolve()
    if not resolved.is_relative_to(EXPERIMENTS_DIR.resolve()):
        raise ValueError("experiment must be inside experiments/")
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def load_result(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != RESULT_SCHEMA
        or not isinstance(payload.get("conversations"), list)
    ):
        raise ValueError(f"{path}: invalid FinAgenticBench result payload")
    return payload


def _turn_counts(payload: dict[str, Any]) -> dict[str, int] | None:
    """Conversation id to turn count, or None when the shape is unusable."""
    conversations = payload.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        return None
    counts: dict[str, int] = {}
    for conversation in conversations:
        if not isinstance(conversation, dict):
            return None
        identifier = conversation.get("id")
        turns = conversation.get("turns")
        if not isinstance(identifier, str) or not isinstance(turns, list) or not turns:
            return None
        if identifier in counts:
            return None
        counts[identifier] = len(turns)
    return counts


def result_is_complete(
    payload: dict[str, Any], expected: dict[str, Any] | None = None
) -> bool:
    """Whether this stored run answered the case, and still answers today's.

    With `expected`, the stored conversation and turn shape must still match the
    case: a case that grew a turn makes the stored run incomplete rather than
    merely stale, because the missing turn was never asked.
    """
    if payload.get("schema") != RESULT_SCHEMA:
        return False
    counts = _turn_counts(payload)
    if counts is None:
        return False
    if expected is not None:
        wanted = {
            conversation.id: len(conversation.turns)
            for conversation in conversations_of(expected)
        }
        if wanted != counts:
            return False
    for conversation in payload["conversations"]:
        for turn in conversation["turns"]:
            if not isinstance(turn, dict) or turn.get("infrastructure_error"):
                return False
            if not isinstance(turn.get("response"), str) or not isinstance(
                turn.get("outputs"), dict
            ):
                return False
    return True


def programmatic_verification(result_file: Path) -> tuple[bool | None, dict | None]:
    """Read the authoritative programmatic-verification channel."""
    path = channel_path(result_file, "programmatic_verification")
    if path.exists():
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None, None
        verdict = payload.get("programmatic_verification")
        if isinstance(verdict, dict) and isinstance(verdict.get("success"), bool):
            return verdict["success"], payload
        # Multi-turn PV stores one verdict per turn.  A case is scored only when
        # every stored turn has a boolean verdict, and passes only when every
        # turn passes.  Keep the legacy top-level shape above for old studies.
        conversations = payload.get("conversations")
        if isinstance(conversations, list) and conversations:
            successes = []
            for conversation in conversations:
                turns = conversation.get("turns") if isinstance(conversation, dict) else None
                if not isinstance(turns, list) or not turns:
                    return None, payload
                for turn in turns:
                    turn_verdict = (
                        turn.get("programmatic_verification")
                        if isinstance(turn, dict) else None
                    )
                    if not (
                        isinstance(turn_verdict, dict)
                        and isinstance(turn_verdict.get("success"), bool)
                    ):
                        return None, payload
                    successes.append(turn_verdict["success"])
            if successes:
                return all(successes), payload
        return None, payload
    return None, None


def _code_and_case_files() -> list[Path]:
    paths = [
        PROJECT_ROOT / "criteria.json",
        CAPABILITIES_PATH,
        CASE_TAXONOMY_PATH,
        BENCHMARKS_JSON,
        PROJECT_ROOT / "pyproject.toml",
        PROJECT_ROOT / "uv.lock",
        PROJECT_ROOT / "scripts" / "run.py",
    ]
    paths.extend(EVALS_DIR.rglob("*.json"))
    paths.extend(EVALS_DIR.rglob("*.py"))
    # Every package module can affect runtime exposure, validation, helper
    # semantics or artifact identity.  A hand-maintained allowlist previously
    # missed dataset_layers.py, footnotes.py and industry.py even though active
    # cases use them directly or transitively.
    paths.extend((PROJECT_ROOT / "core").rglob("*.py"))
    return sorted(set(path for path in paths if path.is_file()))


def _data_identity_files() -> list[Path]:
    """Files that define data identity: everything a run reads, and nothing else.

    ``datasets/dataset_new`` holds exactly that — the active runtime tables and
    their catalog, schema and provenance metadata. The archived previous release,
    staging inputs, normalized layer and source-native archives are deliberately
    outside the fingerprint.
    """
    datasets = DATASETS_DIR
    paths = [
        path for path in datasets.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    ]
    return sorted(set(path for path in paths if path.is_file()))


def _fingerprint_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:20]


def data_fingerprint() -> str:
    return _fingerprint_files(_data_identity_files())


def suite_fingerprint(data_identity: str | None = None) -> str:
    code_identity = _fingerprint_files(_code_and_case_files())
    data_identity = data_identity or data_fingerprint()
    return hashlib.sha256(
        f"{code_identity}\0{data_identity}".encode()
    ).hexdigest()[:20]


def bench_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
        text=True, capture_output=True, check=False,
    )
    return (completed.stdout.strip() or None) if completed.returncode == 0 else None


def experiment_summary(
    experiment_dir: Path,
    *,
    model_id: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    passed = failed = completed = 0
    for _, path in iter_result_files(experiment_dir):
        _, stored_model, stored_run = result_identity(path)
        if model_id and stored_model != sanitize_component(model_id):
            continue
        if run_id and stored_run != run_id:
            continue
        try:
            payload = load_result(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if not result_is_complete(payload):
            continue
        completed += 1
        success, _ = programmatic_verification(path)
        passed += int(success is True)
        failed += int(success is False)
    error_paths = list((experiment_dir / RUNS_DIR).glob("*/*/*/error.json"))
    if model_id:
        error_paths = [
            path for path in error_paths
            if result_identity(path.with_name(RUN_FILE))[1] == sanitize_component(model_id)
        ]
    if run_id:
        error_paths = [path for path in error_paths if path.parent.name == run_id]
    return {
        "completed": completed,
        "pv_scored": passed + failed,
        "passed": passed,
        "failed": failed,
        "infrastructure_errors": len(error_paths),
    }
