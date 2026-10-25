"""Accelerate the known large CFPB archives with validated byte ranges.

The ordinary phase-three downloader remains the authoritative manifest writer.
This helper only completes large ``.part`` files faster; a subsequent
``fetch_phase3`` run independently validates each ZIP and records its hash.
"""

from __future__ import annotations

import argparse
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from zipfile import BadZipFile, ZipFile

import requests

from core.dataset_layers import RAW_DIR
from scripts.ingest.fetch.fetch_expansion import USER_AGENT
from scripts.ingest.fetch.fetch_phase3 import _static_downloads


@dataclass(frozen=True)
class Target:
    url: str
    path: Path
    part: Path
    total: int
    etag: str
    prefix_bytes: int


@dataclass(frozen=True)
class Segment:
    target: Target
    index: int
    start: int
    end: int

    @property
    def path(self) -> Path:
        return self.target.part.with_name(f"{self.target.part.name}.range{self.index:02d}")


PRINT_LOCK = Lock()


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "identity"})
    return session


def _target(url: str, path: Path) -> Target:
    last_error: Exception | None = None
    for attempt in range(8):
        try:
            response = _session().head(url, allow_redirects=True, timeout=120)
            response.raise_for_status()
            total = int(response.headers["Content-Length"])
            etag = response.headers.get("ETag", "")
            break
        except Exception as exc:
            last_error = exc
            if attempt == 7:
                raise
            time.sleep(min(30, 2 ** attempt))
    else:
        assert last_error is not None
        raise last_error
    part = path.with_name(path.name + ".part")
    prefix = part.stat().st_size if part.exists() else 0
    if prefix > total:
        raise ValueError(f"partial is larger than source: {part}")
    return Target(url, path, part, total, etag, prefix)


def _segments(target: Target, per_file: int) -> list[Segment]:
    remaining = target.total - target.prefix_bytes
    if remaining <= 0:
        return []
    width = (remaining + per_file - 1) // per_file
    result = []
    for index in range(per_file):
        start = target.prefix_bytes + index * width
        if start >= target.total:
            break
        result.append(Segment(target, index, start, min(target.total - 1, start + width - 1)))
    return result


def _fetch(segment: Segment) -> Segment:
    path = segment.path
    expected = segment.end - segment.start + 1
    for attempt in range(8):
        have = path.stat().st_size if path.exists() else 0
        if have == expected:
            return segment
        if have > expected:
            path.unlink()
            have = 0
        start = segment.start + have
        headers = {"Range": f"bytes={start}-{segment.end}"}
        if segment.target.etag:
            headers["If-Match"] = segment.target.etag
        try:
            with _session().get(
                segment.target.url, headers=headers, stream=True,
                allow_redirects=True, timeout=(30, 300),
            ) as response:
                response.raise_for_status()
                if response.status_code != 206:
                    raise ValueError(f"origin ignored Range: HTTP {response.status_code}")
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {start}-"):
                    raise ValueError(f"unexpected Content-Range {content_range!r}")
                with path.open("ab") as handle:
                    for block in response.iter_content(4 * 1024 * 1024):
                        if block:
                            handle.write(block)
            if path.stat().st_size != expected:
                raise ValueError(f"short segment {path.stat().st_size} != {expected}")
            with PRINT_LOCK:
                print(
                    f"RANGE {segment.target.path.name} {segment.index + 1} "
                    f"{expected / 2**20:.2f} MiB", flush=True,
                )
            return segment
        except Exception:
            if attempt == 7:
                raise
            time.sleep(min(30, 2 ** attempt))
    raise AssertionError("unreachable")


def _assemble(target: Target, segments: list[Segment]) -> None:
    # Confirm the object did not change while its ranges were in flight.
    current = _target(target.url, target.path)
    if current.total != target.total or current.etag != target.etag:
        raise ValueError(f"source changed during segmented download: {target.path.name}")
    with target.part.open("ab") as output:
        for segment in sorted(segments, key=lambda value: value.start):
            with segment.path.open("rb") as source:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
    if target.part.stat().st_size != target.total:
        raise ValueError(f"assembled size mismatch for {target.path.name}")
    with ZipFile(target.part) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise BadZipFile(f"CRC failure in {bad}")
    target.part.replace(target.path)
    for segment in segments:
        segment.path.unlink(missing_ok=True)
    print(f"OK    {target.path.name} {target.total / 2**20:.2f} MiB", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments-per-file", type=int, default=4)
    parser.add_argument("--workers", type=int, default=28)
    args = parser.parse_args()

    items = [
        item for item in _static_downloads()
        if item.dataset in {"cfpb_credit_card_agreements", "cfpb_prepaid_agreements"}
        and item.name.lower().endswith(".zip")
    ]
    # Start the nightly prepaid snapshot first so its four ranges make progress
    # alongside (rather than only after) the seven remaining card archives.
    items.sort(key=lambda item: item.dataset != "cfpb_prepaid_agreements")
    targets = []
    for item in items:
        path = RAW_DIR / item.dataset / item.name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file():
            continue
        targets.append(_target(item.url, path))

    segments = [segment for target in targets for segment in _segments(target, args.segments_per_file)]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_fetch, segment): segment for segment in segments}
        for future in as_completed(futures):
            future.result()

    for target in targets:
        _assemble(target, _segments(target, args.segments_per_file))


if __name__ == "__main__":
    main()
