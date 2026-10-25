#!/usr/bin/env python3
"""Scan source provenance and the governed dataset catalog."""
import glob
import hashlib
import json
import os
from urllib.parse import urlparse

from config import PROJECT_ROOT


DATASETS_DIR = str(PROJECT_ROOT / "datasets_build")
RAW_DIR = str(PROJECT_ROOT / "datasets_raw")

URL_KEYS = ("url", "download_url", "source_url", "link")
SHA_KEYS = ("sha256", "sha256_hex", "hash")
SIZE_KEYS = ("bytes", "size", "file_size")
TIME_KEYS = ("downloaded_at", "retrieved_at", "fetched_at")
SOURCE_KEYS = ("source", "agency", "publisher", "institution")
LICENSE_KEYS = ("license", "terms", "terms_of_use", "usage_terms")


def first_key(d, keys):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return k
    return None


def extract_entries(data):
    """Accept raw_files as a dict or a list; return (entries, layout)."""
    rf = data.get("raw_files")
    entries = []
    if isinstance(rf, dict):
        for fname, meta in rf.items():
            if isinstance(meta, dict):
                e = dict(meta)
                e.setdefault("file", fname)
                entries.append(e)
        return entries, "dict"
    if isinstance(rf, list):
        for item in rf:
            if isinstance(item, dict):
                entries.append(item)
        return entries, "list"
    return [], "missing"


def scan_sources():
    rows = []
    anomalies = []
    all_domains = {}
    for path in sorted(glob.glob(os.path.join(DATASETS_DIR, "**", "sources.json"), recursive=True)):
        ds = os.path.relpath(path, DATASETS_DIR).replace("/sources.json", "")
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            anomalies.append(f"{ds}: JSON parse error: {e}")
            continue

        entries, layout = extract_entries(data)
        if layout == "missing":
            anomalies.append(f"{ds}: no raw_files field (top-level keys: {sorted(data.keys())})")

        n = len(entries)
        n_url = n_sha = n_size = n_time = 0
        domains = {}
        all_field_names = set()
        for e in entries:
            all_field_names.update(e.keys())
            uk = first_key(e, URL_KEYS)
            sk = first_key(e, SHA_KEYS)
            zk = first_key(e, SIZE_KEYS)
            tk = first_key(e, TIME_KEYS)
            if uk:
                n_url += 1
                dom = urlparse(e[uk]).netloc.lower()
                domains[dom] = domains.get(dom, 0) + 1
                all_domains[dom] = all_domains.get(dom, 0) + 1
            else:
                anomalies.append(f"{ds}: entry {e.get('file', '?')} has no URL")
            if sk:
                n_sha += 1
            else:
                anomalies.append(f"{ds}: entry {e.get('file', '?')} has no sha256")
            if zk:
                n_size += 1
            if tk:
                n_time += 1

        # Licence or terms of use, at the top level or per entry
        has_license = any(k in data for k in LICENSE_KEYS) or any(
            any(k in e for k in LICENSE_KEYS) for e in entries
        )
        top_fields = sorted(data.keys())

        rows.append({
            "dataset": ds,
            "layout": layout,
            "entries": n,
            "url_pct": f"{n_url}/{n}" if n else "-",
            "sha_pct": f"{n_sha}/{n}" if n else "-",
            "size_pct": f"{n_size}/{n}" if n else "-",
            "time_pct": f"{n_time}/{n}" if n else "-",
            "domains": ", ".join(f"{d}({c})" for d, c in sorted(domains.items(), key=lambda x: -x[1])) or "-",
            "license": "Y" if has_license else "-",
            "downloaded_at": data.get("downloaded_at", data.get("retrieved_at", "-")),
            "freeze": data.get("benchmark_freeze_date", data.get("freeze_date", data.get("as_of", "-"))),
            "top_fields": top_fields,
            "entry_fields": sorted(all_field_names),
            "failures": data.get("failures"),
        })
        if isinstance(data.get("failures"), list) and data["failures"]:
            anomalies.append(f"{ds}: non-empty failures list ({len(data['failures'])} entries): "
                             f"{json.dumps(data['failures'][:2], ensure_ascii=False)[:200]}")
        if n > 0 and (n_url < n or n_sha < n):
            anomalies.append(f"{ds}: incomplete coverage url={n_url}/{n} sha256={n_sha}/{n}")
    return rows, anomalies, all_domains


def scan_catalog():
    path = os.path.join(DATASETS_DIR, "catalog.json")
    with open(path) as handle:
        catalog = json.load(handle)
    runtime_bytes = sum(
        table["layers"]["runtime"]["bytes"]
        for table in catalog.get("tables", {}).values()
    )
    return {
        "schema_version": catalog.get("schema_version"),
        "dataset_release": catalog.get("dataset_release"),
        "benchmark_cutoff": catalog.get("benchmark_cutoff"),
        "table_count": catalog.get("table_count"),
        "source_count": catalog.get("source_count"),
        "asset_count": catalog.get("asset_count"),
        "issue_count": catalog.get("issue_count"),
        "runtime_bytes": runtime_bytes,
    }


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def spot_check(ds_list, max_files=2):
    """Verify up to max_files raw entries with a sha256 per dataset against the files on disk."""
    results = []
    for ds in ds_list:
        p = os.path.join(DATASETS_DIR, ds, "sources.json")
        with open(p) as f:
            data = json.load(f)
        entries, _ = extract_entries(data)
        checked = 0
        for e in entries:
            sk = first_key(e, SHA_KEYS)
            rel = e.get("file")
            if not sk or not rel:
                continue
            fp = os.path.join(DATASETS_DIR, ds, rel)
            if not os.path.exists(fp):
                # Some layouts omit the raw/ prefix; try the dataset root and raw/
                alt = os.path.join(RAW_DIR, ds, os.path.basename(rel))
                if os.path.exists(alt):
                    fp = alt
                elif os.path.exists(os.path.join(DATASETS_DIR, ds, os.path.basename(rel))):
                    fp = os.path.join(DATASETS_DIR, ds, os.path.basename(rel))
                else:
                    results.append((ds, rel, e[sk][:16], "FILE_NOT_FOUND", "MISS"))
                    checked += 1
                    continue
            actual = sha256_of(fp)
            ok = actual == e[sk]
            results.append((ds, rel, e[sk][:16], actual[:16], "OK" if ok else "MISMATCH"))
            checked += 1
            if checked >= max_files:
                break
    return results


def main():
    rows, anomalies, all_domains = scan_sources()

    print("=" * 120)
    print("Table 1: sources.json summary")
    print("=" * 120)
    hdr = f"{'dataset':<32} {'files':>4} {'URL':>7} {'SHA256':>7} {'bytes':>7} {'time':>7} {'lic':>3}  source domains"
    print(hdr)
    print("-" * 120)
    for r in rows:
        print(f"{r['dataset']:<32} {r['entries']:>4} {r['url_pct']:>7} {r['sha_pct']:>7} "
              f"{r['size_pct']:>7} {r['time_pct']:>7} {r['license']:>3}  {r['domains'][:60]}")

    print()
    print("=" * 120)
    print("Table 2: field structure of each sources.json (top-level keys / entry fields / layout / download time / freeze date)")
    print("=" * 120)
    for r in rows:
        print(f"[{r['dataset']}] layout={r['layout']} downloaded_at={r['downloaded_at']} freeze={r['freeze']}")
        print(f"  top: {', '.join(r['top_fields'])}")
        print(f"  entry: {', '.join(r['entry_fields'])}")

    print()
    print("=" * 120)
    print(f"Table 3: anomalies and incomplete records ({len(anomalies)})")
    print("=" * 120)
    for a in anomalies:
        print(" -", a)
    if not anomalies:
        print(" (none)")

    print()
    print("=" * 120)
    print(f"Table 4: all source domains ({len(all_domains)} distinct)")
    print("=" * 120)
    for d, c in sorted(all_domains.items(), key=lambda x: -x[1]):
        print(f"  {d:<55} {c} entries")

    print()
    print("=" * 120)
    print("Table 5: governed catalog release contract")
    print("=" * 120)
    for key, value in scan_catalog().items():
        print(f"  {key}: {value}")

    print()
    print("=" * 120)
    print("Table 6: raw-file sha256 spot check (fdic_bankfind / treasury_rates / ken_french)")
    print("=" * 120)
    for ds, rel, expect, actual, status in spot_check(["fdic_bankfind", "treasury_rates", "ken_french"]):
        print(f"  {status:<14} {ds}/{rel}")
        print(f"                 recorded={expect}...  actual={actual}...")


if __name__ == "__main__":
    main()
