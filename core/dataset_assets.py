"""Lossless inventory and controlled access for every FinAgenticBench raw asset.

This module does not pretend that PDFs, XML deltas, spreadsheets and regulatory
bulk archives share one table schema.  It creates a normalized asset catalog for
every dataset source, mirrors that catalog into the runtime discovery layer, and
provides strict, on-demand readers.  Source bytes are never rewritten.
"""

from __future__ import annotations

from io import BytesIO, StringIO
import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any
import zipfile

import pandas as pd
from pypdf import PdfReader

from core.dataset_layers import BUILD_DIR, DATASETS_DIR, _sha256, load_catalog, raw_dir


ASSET_FILENAME = "_assets.csv.gz"
TEXT_ENCODINGS = ("utf-8-sig", "utf-16", "cp1252")
TABULAR_SUFFIXES = {
    ".csv", ".tsv", ".parquet", ".xlsx", ".xls", ".xlsb", ".dta",
}
DOCUMENT_SUFFIXES = {".pdf", ".htm", ".html"}
STRUCTURED_SUFFIXES = {".json", ".jsonl", ".xml"}


def _manifest_hashes(source_dir: Path) -> dict[str, tuple[int | None, str | None]]:
    path = source_dir / "sources.json"
    if not path.is_file():
        return {}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    raw_files = manifest.get("raw_files", {})
    records: list[tuple[str, dict[str, Any]]] = []
    if isinstance(raw_files, dict):
        records = [(name, value) for name, value in raw_files.items() if isinstance(value, dict)]
    elif isinstance(raw_files, list):
        records = [
            (str(value.get("file", "")), value)
            for value in raw_files if isinstance(value, dict)
        ]
    result = {}
    for name, record in records:
        normalized = name.replace("\\", "/")
        if normalized.startswith(f"{source_dir.name}/"):
            normalized = normalized[len(source_dir.name) + 1:]
        if not normalized.startswith("raw/"):
            normalized = f"raw/{normalized}"
        result[normalized] = (record.get("bytes"), record.get("sha256"))
    return result


def _manifest_failures(source_dir: Path) -> list[dict[str, Any]]:
    path = source_dir / "sources.json"
    if not path.is_file():
        return [{"error": "missing sources.json"}]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    failures = manifest.get("failures", [])
    return failures if isinstance(failures, list) else [{"error": str(failures)}]


def _strict_decode(payload: bytes) -> tuple[str | None, str | None]:
    for encoding in TEXT_ENCODINGS:
        try:
            return payload.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return None, None


def _delimiter_and_columns(payload: bytes) -> tuple[str | None, str | None, str | None]:
    text, encoding = _strict_decode(payload)
    if text is None:
        return None, None, None
    first_lines = "\n".join(text.splitlines()[:5])
    if not first_lines:
        return encoding, None, json.dumps([], ensure_ascii=False)
    candidates = ["\t", "|", "^", ",", ";"]
    delimiter = max(candidates, key=lambda item: first_lines.splitlines()[0].count(item))
    if first_lines.splitlines()[0].count(delimiter) == 0:
        return encoding, None, None
    try:
        columns = next(csv.reader(StringIO(first_lines), delimiter=delimiter))
    except (csv.Error, StopIteration):
        return encoding, delimiter, None
    return encoding, delimiter, json.dumps(columns, ensure_ascii=False)


def _suffix(name: str) -> str:
    lower = name.lower()
    for suffix in (".csv.gz", ".jsonl.gz", ".json.gz", ".tar.gz"):
        if lower.endswith(suffix):
            return suffix
    return Path(lower).suffix or "<none>"


def _asset_kind(name: str, *, container: bool = False) -> str:
    if container:
        return "archive_container"
    suffix = _suffix(name)
    if suffix in TABULAR_SUFFIXES or suffix == ".csv.gz":
        return "tabular"
    if suffix in DOCUMENT_SUFFIXES:
        return "document"
    if suffix == ".xml":
        return "structured_xml"
    if suffix in {".json", ".jsonl", ".json.gz", ".jsonl.gz"}:
        return "structured_json"
    if suffix in {".txt", "<none>"}:
        return "text_or_delimited"
    return "binary_or_source_native"


def _detected_format(name: str, payload: bytes | None) -> str:
    """Prefer byte signatures when archive members have missing/misleading suffixes."""
    if payload:
        if payload.startswith(b"%PDF-"):
            return ".pdf"
        if payload.startswith(b"\x89PNG\r\n\x1a\n"):
            return ".png"
        if payload.startswith(b"\xff\xd8\xff"):
            return ".jpg"
        if payload.startswith(b"\xd0\xcf\x11\xe0"):
            return ".xls"
        stripped = payload.lstrip()
        if stripped.startswith(b"{"):
            # EIA bulk text uses one complete JSON object per physical line.
            lines = stripped.splitlines()
            if len(lines) > 1 and lines[1].lstrip().startswith(b"{"):
                return ".jsonl"
    return _suffix(name)


def _read_strategy(kind: str, name: str) -> str:
    suffix = _suffix(name)
    if kind == "archive_container":
        return "discover members; load one member on demand"
    if suffix in {".csv", ".csv.gz", ".tsv", ".txt", "<none>"}:
        return "strict text decode and delimited reader; never replace invalid bytes"
    if suffix == ".parquet":
        return "typed parquet reader"
    if suffix in {".xlsx", ".xls", ".xlsb"}:
        return "workbook reader with explicit sheet selection"
    if suffix == ".dta":
        return "Stata reader preserving labelled source columns"
    if suffix in {".json", ".jsonl", ".json.gz", ".jsonl.gz"}:
        return "JSON parser preserving source keys"
    if suffix == ".xml":
        return "XML parser or strict text reader preserving namespaces"
    if suffix == ".pdf":
        return "PDF text extraction on demand; source bytes remain authoritative"
    return "byte-preserving access only until a source-specific adapter is reviewed"


def _period_hint(name: str) -> str | None:
    match = re.search(r"((?:19|20)\d{2})(?:[-_]?([qQ][1-4]|\d{2}))?", name)
    return match.group(0) if match else None


def _asset_id(source_id: str, raw_path: str, member: str | None) -> str:
    canonical = f"{source_id}\n{raw_path}\n{member or ''}".encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:24]


def _probe_payload(name: str, payload: bytes) -> tuple[str | None, str | None, str | None]:
    if _suffix(name) not in {".csv", ".csv.gz", ".tsv", ".txt", "<none>"}:
        return None, None, None
    return _delimiter_and_columns(payload[:65536])


def _base_row(
    source_id: str, raw_path: str, name: str, *, member: str | None,
    size: int, compressed_size: int | None, container_sha256: str | None,
    crc32: str | None, encoding: str | None, delimiter: str | None,
    columns_json: str | None, container: bool = False,
    detected_format: str | None = None,
) -> dict[str, Any]:
    asset_format = detected_format or _suffix(name)
    kind = _asset_kind(name, container=container)
    if not container and asset_format != _suffix(name):
        kind = _asset_kind(f"asset{asset_format}")
    if not container and asset_format == "<none>" and encoding is None:
        kind = "binary_or_source_native"
    if (
        not container and kind == "text_or_delimited"
        and encoding is not None and delimiter is None
    ):
        kind = "plain_text_document"
    return {
        "asset_id": _asset_id(source_id, raw_path, member),
        "source_id": source_id,
        "raw_path": raw_path,
        "archive_member": member,
        "file_name": Path(name).name,
        "format": asset_format,
        "asset_kind": kind,
        "uncompressed_bytes": size,
        "compressed_bytes": compressed_size,
        "container_sha256": container_sha256,
        "member_crc32": crc32,
        "strict_text_encoding": encoding,
        "delimiter": delimiter,
        "columns_json": columns_json,
        "period_hint_from_filename": _period_hint(name),
        "runtime_status": "discoverable_on_demand",
        "read_strategy": (
            "strict text reader; no tabular delimiter inferred"
            if kind == "plain_text_document"
            else _read_strategy(kind, f"asset{asset_format}")
        ),
    }


def inventory_source(source_dir: Path) -> list[dict[str, Any]]:
    source_id = source_dir.name
    hashes = _manifest_hashes(source_dir)
    rows: list[dict[str, Any]] = []
    raw_root = raw_dir(source_dir.name)
    for path in sorted(p for p in raw_root.rglob("*") if p.is_file()):
        # Asset keys stay "raw/<file>" — the manifests record them that way and the
        # archive itself now lives outside the source directory.
        raw_path = f"raw/{str(path.relative_to(raw_root)).replace(chr(92), '/')}"
        manifest_bytes, manifest_sha = hashes.get(raw_path, (None, None))
        size = path.stat().st_size
        if manifest_bytes is not None and int(manifest_bytes) != size:
            raise ValueError(f"{source_id}/{raw_path}: size differs from sources.json")
        container_sha = manifest_sha or _sha256(path)
        if path.suffix.lower() == ".zip":
            rows.append(_base_row(
                source_id, raw_path, path.name, member=None, size=size,
                compressed_size=size, container_sha256=container_sha, crc32=None,
                encoding=None, delimiter=None, columns_json=None, container=True,
            ))
            with zipfile.ZipFile(path) as archive:
                for info in archive.infolist():
                    if info.is_dir():
                        continue
                    encoding = delimiter = columns_json = None
                    with archive.open(info) as handle:
                        sample = handle.read(65536)
                    detected_format = _detected_format(info.filename, sample)
                    if detected_format in {".csv", ".tsv", ".txt", "<none>"}:
                        encoding, delimiter, columns_json = _probe_payload(
                            info.filename, sample
                        )
                    rows.append(_base_row(
                        source_id, raw_path, info.filename, member=info.filename,
                        size=info.file_size, compressed_size=info.compress_size,
                        container_sha256=container_sha,
                        crc32=f"{info.CRC:08x}", encoding=encoding,
                        delimiter=delimiter, columns_json=columns_json,
                        detected_format=detected_format,
                    ))
        else:
            encoding = delimiter = columns_json = None
            if _suffix(path.name) in {".csv", ".csv.gz", ".tsv", ".txt", "<none>"}:
                if path.name.lower().endswith(".gz"):
                    import gzip
                    with gzip.open(path, "rb") as handle:
                        sample = handle.read(65536)
                else:
                    with path.open("rb") as handle:
                        sample = handle.read(65536)
                encoding, delimiter, columns_json = _probe_payload(path.name, sample)
            rows.append(_base_row(
                source_id, raw_path, path.name, member=None, size=size,
                compressed_size=size, container_sha256=container_sha, crc32=None,
                encoding=encoding, delimiter=delimiter, columns_json=columns_json,
            ))
    return rows


def _expected_asset_keys(source_dir: Path) -> set[tuple[str, str]]:
    """Independently enumerate every raw file and ZIP member for omission checks."""
    keys: set[tuple[str, str]] = set()
    raw_root = raw_dir(source_dir.name)
    for path in sorted(p for p in raw_root.rglob("*") if p.is_file()):
        raw_path = f"raw/{str(path.relative_to(raw_root)).replace(chr(92), '/')}"
        keys.add((raw_path, ""))
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                keys.update(
                    (raw_path, info.filename)
                    for info in archive.infolist() if not info.is_dir()
                )
    return keys


def build_asset_layers(
    datasets_dir: Path = DATASETS_DIR, build_dir: Path | None = None,
) -> dict[str, Any]:
    build_dir = BUILD_DIR if build_dir is None else build_dir
    catalog = load_catalog(datasets_dir / "catalog.json")
    source_metadata: dict[str, Any] = {}
    all_asset_ids: set[str] = set()
    total_assets = 0
    issues: list[dict[str, Any]] = []
    for source_dir in sorted(path for path in datasets_dir.iterdir() if path.is_dir()):
        rows = inventory_source(source_dir)
        frame = pd.DataFrame(rows)
        if frame.empty:
            raise ValueError(f"{source_dir.name}: no raw assets found")
        if frame["asset_id"].duplicated().any():
            raise ValueError(f"{source_dir.name}: duplicate asset IDs")
        overlap = all_asset_ids.intersection(frame["asset_id"])
        if overlap:
            raise ValueError(f"global duplicate asset IDs: {sorted(overlap)[:3]}")
        all_asset_ids.update(frame["asset_id"])

        # The normalized layer is a build intermediate and lives under BUILD_DIR;
        # only the runtime copy ships inside datasets/.
        normalized = build_dir / source_dir.name / "normalized" / ASSET_FILENAME
        runtime = source_dir / "runtime" / ASSET_FILENAME
        normalized.parent.mkdir(parents=True, exist_ok=True)
        runtime.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(
            normalized, index=False, na_rep="",
            compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
        )
        # Validate UTF-8 and row identity before publishing the runtime copy.
        roundtrip = pd.read_csv(
            normalized, dtype="string", keep_default_na=False, low_memory=False,
        )
        if len(roundtrip) != len(frame):
            raise ValueError(f"{source_dir.name}: asset catalog row loss")
        if roundtrip["asset_id"].tolist() != frame["asset_id"].tolist():
            raise ValueError(f"{source_dir.name}: asset catalog encoding/order corruption")
        runtime.write_bytes(normalized.read_bytes())

        table_ids = sorted(
            table_id for table_id, metadata in catalog["tables"].items()
            if metadata["source_id"] == source_dir.name
        )
        info = {
            "source_id": source_dir.name,
            "raw_asset_count": len(frame),
            "raw_file_count": int(frame["archive_member"].isna().sum()),
            "archive_member_count": int(frame["archive_member"].notna().sum()),
            "materialized_table_ids": table_ids,
            "materialization_status": (
                "materialized_tables_and_full_asset_index" if table_ids
                else "full_asset_index_with_lossless_on_demand_access"
            ),
            "normalized_asset_catalog": {
                "path": str(normalized.relative_to(build_dir)),
                "bytes": normalized.stat().st_size,
                "sha256": _sha256(normalized),
            },
            "runtime_asset_catalog": {
                "path": str(runtime.relative_to(datasets_dir)),
                "bytes": runtime.stat().st_size,
                "sha256": _sha256(runtime),
            },
            "asset_kinds": {
                str(key): int(value)
                for key, value in frame["asset_kind"].value_counts().sort_index().items()
            },
            "lossless_policy": (
                "raw bytes are immutable; catalogs index every file/archive member; "
                "decode failures are recorded as unknown rather than replaced or dropped"
            ),
        }
        source_metadata[source_dir.name] = info
        total_assets += len(frame)

        if not table_ids:
            issues.append({
                "source_id": source_dir.name,
                "code": "no_materialized_table",
                "severity": "needs_source_adapter",
                "count": len(frame),
                "detail": (
                    "All raw files and archive members are indexed and available on "
                    "demand, but no source-specific semantic table has been approved."
                ),
            })
        unknown_encoding = frame.loc[
            frame["asset_kind"].eq("text_or_delimited")
            & frame["strict_text_encoding"].isna()
        ]
        if not unknown_encoding.empty:
            issues.append({
                "source_id": source_dir.name,
                "code": "strict_text_encoding_unknown",
                "severity": "needs_source_adapter",
                "count": len(unknown_encoding),
                "examples": unknown_encoding["archive_member"].fillna(
                    unknown_encoding["raw_path"]
                ).head(10).tolist(),
                "detail": "Bytes were preserved; no replacement decoding was attempted.",
            })
        binary = frame.loc[frame["asset_kind"].eq("binary_or_source_native")]
        if not binary.empty:
            issues.append({
                "source_id": source_dir.name,
                "code": "source_native_binary_requires_adapter",
                "severity": "needs_source_adapter",
                "count": len(binary),
                "formats": sorted(binary["format"].dropna().unique().tolist()),
                "detail": "Assets remain byte-preserved and indexed; no lossy conversion was made.",
            })
        documents = frame.loc[frame["asset_kind"].eq("document")]
        if not documents.empty and not table_ids:
            issues.append({
                "source_id": source_dir.name,
                "code": "document_corpus_indexed_not_eagerly_extracted",
                "severity": "intentional_deferred_materialization",
                "count": len(documents),
                "detail": (
                    "Documents are readable through read_document_asset; eager OCR/text "
                    "materialization is deferred to avoid silent extraction loss."
                ),
            })
        manifest_failures = _manifest_failures(source_dir)
        if manifest_failures:
            issues.append({
                "source_id": source_dir.name,
                "code": "source_manifest_declared_failures",
                "severity": "declared_source_gap",
                "count": len(manifest_failures),
                "examples": manifest_failures[:10],
                "detail": "Failures are retained from sources.json and were not hidden by layer generation.",
            })

        schema_path = source_dir / "schema.json"
        schema = json.loads(schema_path.read_text()) if schema_path.is_file() else {
            "schema_version": 1,
            "source_id": source_dir.name,
            "benchmark_cutoff": catalog["benchmark_cutoff"],
            "tables": {},
        }
        schema["assets"] = info
        schema_path.write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n")

    catalog["source_count"] = len(source_metadata)
    catalog["asset_count"] = total_assets
    catalog["sources"] = source_metadata
    catalog["all_source_policy"] = (
        "Every source has normalized/runtime asset catalogs. Materialized tables are "
        "listed separately; non-tabular and unreviewed bulk assets remain lossless, "
        "discoverable and readable on demand instead of being coerced into fake tables."
    )
    issue_report = {
        "schema_version": 1,
        "source_count": len(source_metadata),
        "asset_count": total_assets,
        "issue_count": len(issues),
        "issues": issues,
    }
    issue_path = datasets_dir / "layer_issues.json"
    issue_path.write_text(json.dumps(issue_report, indent=2, ensure_ascii=False) + "\n")
    catalog["issues_path"] = str(issue_path.relative_to(datasets_dir))
    catalog["issue_count"] = len(issues)
    (datasets_dir / "catalog.json").write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False) + "\n"
    )
    return catalog


def load_asset_catalog(source_id: str, datasets_dir: Path = DATASETS_DIR) -> pd.DataFrame:
    source_dir = datasets_dir / source_id
    if not source_dir.is_dir() or source_dir.parent != datasets_dir:
        raise KeyError(f"unknown dataset source: {source_id}")
    return pd.read_csv(
        source_dir / "runtime" / ASSET_FILENAME,
        dtype="string", keep_default_na=False, low_memory=False,
    )


def _asset_row(source_id: str, asset_id: str, datasets_dir: Path) -> pd.Series:
    catalog = load_asset_catalog(source_id, datasets_dir)
    rows = catalog.loc[catalog["asset_id"].eq(asset_id)]
    if len(rows) != 1:
        raise KeyError(f"unknown asset {source_id}/{asset_id}")
    return rows.iloc[0]


def _safe_raw_path(source_id: str, relative: str, datasets_dir: Path) -> Path:
    # Recorded keys are "raw/<file>" relative to the source; the bytes live in
    # the separate raw tree.
    source_raw = raw_dir(source_id).resolve()
    path = (source_raw.parent / source_id / relative).resolve()
    if path != source_raw and source_raw not in path.parents:
        raise ValueError("asset path escapes the source raw directory")
    return path


def _open_asset_bytes(row: pd.Series, datasets_dir: Path) -> bytes:
    path = _safe_raw_path(str(row["source_id"]), str(row["raw_path"]), datasets_dir)
    member = str(row["archive_member"])
    if member:
        with zipfile.ZipFile(path) as archive:
            return archive.read(member)
    return path.read_bytes()


def _clean_discovered_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply only lossless whitespace/empty normalization to an unreviewed table."""
    result = frame.copy()
    for column in result.columns:
        if pd.api.types.is_object_dtype(result[column].dtype) or isinstance(result[column].dtype, pd.StringDtype):
            values = result[column].astype("string").str.strip()
            result[column] = values.mask(values.eq(""), pd.NA)
    return result


def load_tabular_asset(
    source_id: str, asset_id: str, *, sheet_name: str | int | None = 0,
    datasets_dir: Path = DATASETS_DIR, **read_options: Any,
) -> pd.DataFrame:
    """Read one indexed table without lossy decoding or access outside raw/."""
    row = _asset_row(source_id, asset_id, datasets_dir)
    suffix = str(row["format"])
    if str(row["asset_kind"]) not in {"tabular", "text_or_delimited", "structured_json"}:
        raise TypeError(f"asset is not tabular: {row['asset_kind']}")
    path = _safe_raw_path(source_id, str(row["raw_path"]), datasets_dir)
    member = str(row["archive_member"])
    requested_rows = read_options.pop("nrows", None)
    payload: bytes | None = None
    if member:
        with zipfile.ZipFile(path) as archive:
            payload = archive.read(member)

    if suffix == ".parquet":
        frame = pd.read_parquet(BytesIO(payload) if payload is not None else path, **read_options)
    elif suffix in {".xlsx", ".xls", ".xlsb"}:
        engine = {".xlsx": "openpyxl", ".xls": "xlrd", ".xlsb": "pyxlsb"}[suffix]
        frame = pd.read_excel(
            BytesIO(payload) if payload is not None else path,
            sheet_name=sheet_name, engine=engine, nrows=requested_rows,
            **read_options,
        )
    elif suffix == ".dta":
        frame = pd.read_stata(
            BytesIO(payload) if payload is not None else path, **read_options
        )
    elif suffix in {".json", ".jsonl", ".json.gz", ".jsonl.gz"}:
        frame = pd.read_json(
            BytesIO(payload) if payload is not None else path,
            lines=suffix.startswith(".jsonl"), nrows=requested_rows,
            **read_options,
        )
    else:
        encoding = str(row["strict_text_encoding"])
        delimiter = str(row["delimiter"])
        if not encoding:
            raise UnicodeError("no strict supported encoding detected; source bytes were preserved")
        if not delimiter:
            raise ValueError("no reliable delimiter detected; pass a source-specific adapter")
        source: Any = BytesIO(payload) if payload is not None else path
        frame = pd.read_csv(
            source, sep=delimiter, encoding=encoding,
            keep_default_na=False, na_values=[""], low_memory=False,
            nrows=requested_rows,
            **read_options,
        )
    if requested_rows is not None and suffix in {".parquet", ".dta"}:
        frame = frame.head(requested_rows)
    return _clean_discovered_frame(frame)


def read_document_asset(
    source_id: str, asset_id: str, *, max_chars: int = 200_000,
    datasets_dir: Path = DATASETS_DIR,
) -> str:
    """Strictly read one indexed document, bounded for safe interactive use."""
    row = _asset_row(source_id, asset_id, datasets_dir)
    payload = _open_asset_bytes(row, datasets_dir)
    suffix = str(row["format"])
    if suffix == ".pdf":
        parts = []
        used = 0
        for page in PdfReader(BytesIO(payload)).pages:
            text = page.extract_text() or ""
            parts.append(text[: max_chars - used])
            used += len(parts[-1])
            if used >= max_chars:
                break
        return "\n".join(parts)
    text, encoding = _strict_decode(payload)
    if text is None:
        raise UnicodeError("document is not valid UTF-8/UTF-16/CP1252; bytes were not replaced")
    return text[:max_chars]


def validate_asset_layers(
    datasets_dir: Path = DATASETS_DIR, *, require_raw: bool = False,
    build_dir: Path | None = None,
) -> list[str]:
    """Validate published asset catalogs and any locally materialized raw archive.

    The GitHub runtime release intentionally omits source-native raw bytes. When a
    source's ``raw/`` directory is present, its complete file/member coverage is
    still checked. Maintainer release audits can require all raw archives.
    """
    errors: list[str] = []
    catalog = load_catalog(datasets_dir / "catalog.json")
    source_dirs = {path.name for path in datasets_dir.iterdir() if path.is_dir()}
    catalog_sources = set(catalog.get("sources", {}))
    if source_dirs != catalog_sources:
        errors.append(
            f"source coverage differs: missing={sorted(source_dirs-catalog_sources)}, "
            f"extra={sorted(catalog_sources-source_dirs)}"
        )
    issue_path = datasets_dir / catalog.get("issues_path", "layer_issues.json")
    if not issue_path.is_file():
        errors.append("missing layer_issues.json")
    normalized_paths = [
        datasets_dir / info["normalized_asset_catalog"]["path"]
        for info in catalog.get("sources", {}).values()
    ]
    normalized_present = [path.is_file() for path in normalized_paths]
    if any(normalized_present) and not all(normalized_present):
        errors.append("normalized asset catalogs are only partially materialized")
    asset_layer_keys = (
        ("normalized_asset_catalog", "runtime_asset_catalog")
        if all(normalized_present)
        else ("runtime_asset_catalog",)
    )
    observed_assets = 0
    for source_id, info in catalog.get("sources", {}).items():
        for layer_key in asset_layer_keys:
            layer = info[layer_key]
            root = (BUILD_DIR if build_dir is None else build_dir) if layer_key.startswith("normalized") else datasets_dir
            path = root / layer["path"]
            if not path.is_file():
                errors.append(f"{source_id}: missing {layer_key}")
                continue
            if path.stat().st_size != layer["bytes"] or _sha256(path) != layer["sha256"]:
                errors.append(f"{source_id}: changed {layer_key}")
        schema = datasets_dir / source_id / "schema.json"
        if not schema.is_file():
            errors.append(f"{source_id}: missing schema.json")
        asset_catalog = load_asset_catalog(source_id, datasets_dir)
        raw_root = raw_dir(source_id)
        raw_files = [path for path in raw_root.rglob("*") if path.is_file()]
        if require_raw and not raw_files:
            errors.append(f"{source_id}: raw archive is not materialized")
        if raw_files:
            actual_keys = set(zip(
                asset_catalog["raw_path"].astype(str),
                asset_catalog["archive_member"].astype(str),
            ))
            expected_keys = _expected_asset_keys(datasets_dir / source_id)
            if actual_keys != expected_keys:
                errors.append(
                    f"{source_id}: asset coverage differs; "
                    f"missing={len(expected_keys-actual_keys)}, "
                    f"extra={len(actual_keys-expected_keys)}"
                )
        observed_assets += int(info["raw_asset_count"])
    if observed_assets != catalog.get("asset_count"):
        errors.append("global asset count mismatch")
    return errors
