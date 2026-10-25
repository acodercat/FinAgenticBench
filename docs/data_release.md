# Data release

The tables the tasks read are distributed as `FinAgenticBench-data.zip` (about 1.3 GB, link in the
[README](../README.md#data-and-run-artifacts)), not in Git. Unzipped at the repository root, it gives
`datasets/dataset_new/`.

## Contents

- `datasets/dataset_new/catalog.json`: the table registry. For every table it records the source, row and column
  counts, identifier and date storage types, the cutoff applied, missing-value and duplicate policies,
  provenance, and the hashes of the runtime files.
- `datasets/dataset_new/<source>/runtime/`: the tables themselves, as gzip-compressed CSV, plus per-source asset
  catalogs.
- `datasets/dataset_new/<source>/sources.json`: source URLs, acquisition dates and the hashes of the downloaded
  files.
- `datasets/dataset_new/<source>/schema.json`: the storage contract of each table.

The archive holds the runtime layer only. The source-native downloads and the intermediate normalized layer are
not distributed; the catalog records their hashes so that a rebuild can be checked against them.

All data end on or before 31 December 2025, and every table is the full population for its declared window. Two
sources cover shorter windows than 2019–2025 because their bulk endpoints reject scripted retrieval:

- `sec_nmfp`: quarterly data sets 2019q1–2022q2 plus the monthly data sets for 2025; the N-MFP3 schedules (daily
  flows, liquidity, seven-day yields) exist only in the 2025 archives.
- `ffiec_y9c`: 2024Q1–2025Q3.

Per-table coverage is in `metadata/dataset_temporal_coverage.json`.

## How tasks read the data

Every loader resolves a table ID through `core/runtime_catalog.py` and reads the `runtime/` path that the catalog
declares, with the stored dtypes. The agent, the reference recomputation and the validators therefore see the same
tables, and a task sees only the tables it declares.

## Licences

The code licence ([MIT](../LICENSE)) does not cover the data. Each of the 44 sources keeps its own terms,
recorded in `metadata/dataset_licenses.json`:

| Basis | Sources |
|---|---|
| Work of the US Government (17 U.S.C. 105), not subject to copyright | 39 sources from the SEC, Federal Reserve Board, Treasury, FDIC, FFIEC, NCUA, BLS, BEA, Census Bureau, CFTC, EIA, FEMA, NOAA, SBA, Department of Labor, CFPB and other federal agencies |
| BIS terms of use: free reproduction with acknowledgement of the source | `bis` |
| FINRA terms of use | `finra_short_volume` |
| Kenneth R. French Data Library: research and educational use, with citation | `ken_french` |
| Federal Reserve Bank of New York terms of use | `nyfed_rates` |
| Federal Reserve Bank of Philadelphia Real-Time Data Set: research use, with citation | `rtdsm` |

If you use these tables, please cite the original sources as well as FinAgenticBench.

## Checking an installation

```bash
uv run python -m scripts.ingest.build.build_dataset_layers --check   # catalog, schemas and runtime hashes
uv run --group dev python -m pytest -m runtime_data                   # tests that read the tables
```

Adding `--skip-hashes` skips the hash comparison for a faster check.

## Rebuilding from source

The data can be rebuilt from the official sources with the scripts under `scripts/ingest/`:

1. `scripts/ingest/fetch/` downloads the source-native files into `datasets_raw/` and writes each source's
   manifest.
2. `scripts/ingest/build/` builds each source's staging tables into `datasets_build/`.
3. `uv run python -m scripts.ingest.build.build_dataset_layers` normalizes the staging tables, writes the runtime
   files, and updates the schemas and the catalog.

A rebuild is compared with the release by `build_dataset_layers --check --require-raw` and
`uv run --group dev python -m pytest -m raw_data`. Official files are revised and their URLs change, so a rebuild
made later can differ from the release; the catalog hashes show where.

`--table TABLE_ID` (repeatable) rebuilds only the named tables. The numbered `phase*` modules record the
acquisition and build batches that produced the release and are kept as they ran.
