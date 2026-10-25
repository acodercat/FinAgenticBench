"""Verify raw files and manifests produced by ``fetch_expansion``."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from config import PROJECT_ROOT


DATASETS = PROJECT_ROOT / "datasets_build"   # reviewed staging inputs
RAW_DIR = PROJECT_ROOT / "datasets_raw"      # source-native archives


@dataclass(frozen=True)
class Entry:
    path: Path
    expected_bytes: int
    expected_sha256: str


def _verify(entry: Entry) -> list[str]:
    errors: list[str] = []
    if not entry.path.is_file():
        return [f"missing: {entry.path}"]
    if entry.path.stat().st_size != entry.expected_bytes:
        errors.append(f"size mismatch: {entry.path}")

    digest = hashlib.sha256()
    with entry.path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != entry.expected_sha256:
        errors.append(f"SHA-256 mismatch: {entry.path}")

    if entry.path.suffix.lower() in {".zip", ".xlsx"}:
        try:
            with ZipFile(entry.path) as archive:
                bad_member = archive.testzip()
            if bad_member is not None:
                errors.append(f"CRC failure in {bad_member}: {entry.path}")
        except BadZipFile:
            errors.append(f"invalid archive: {entry.path}")
    elif entry.path.suffix.lower() == ".pdf":
        with entry.path.open("rb") as handle:
            if handle.read(5) != b"%PDF-":
                errors.append(f"invalid PDF signature: {entry.path}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    entries: list[Entry] = []
    errors: list[str] = []
    expected_failures = 0
    unexpected_failures = 0
    manifests = 0

    for manifest_path in sorted(DATASETS.glob("*/sources.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw_files = manifest.get("raw_files")
        # The three original normalized datasets use an older dict manifest.
        if not isinstance(raw_files, list):
            continue
        manifests += 1
        dataset_dir = manifest_path.parent
        represented: set[Path] = set()
        for record in raw_files:
            relative = Path(record["file"])
            represented.add(relative)
            entries.append(
                Entry(
                    path=dataset_dir / relative,
                    expected_bytes=record["bytes"],
                    expected_sha256=record["sha256"],
                )
            )

        actual = {
            path.relative_to(dataset_dir)
            for path in (RAW_DIR / dataset_dir.name).glob("**/*")
            if path.is_file() and not path.name.endswith(".part")
        }
        for relative in sorted(actual - represented):
            errors.append(f"unmanifested raw file: {dataset_dir / relative}")
        for relative in sorted(represented - actual):
            errors.append(f"manifest references absent raw file: {dataset_dir / relative}")

        for failure in manifest.get("failures", []):
            if failure.get("expected_market_holiday"):
                expected_failures += 1
            else:
                unexpected_failures += 1

    partials = sorted(RAW_DIR.glob("*/**/*.part"))
    errors.extend(f"unfinished partial: {path}" for path in partials)
    if unexpected_failures:
        errors.append(f"unexpected manifest failures: {unexpected_failures}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(_verify, entries):
            errors.extend(result)

    total_bytes = sum(entry.expected_bytes for entry in entries)
    print(
        f"Verified {len(entries)} files in {manifests} manifests "
        f"({total_bytes / 2**30:.4f} GiB); "
        f"expected gaps={expected_failures}; errors={len(errors)}"
    )
    for error in errors:
        print(error)
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
