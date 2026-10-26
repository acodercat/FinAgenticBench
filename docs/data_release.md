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
not distributed; the catalog records their hashes so that a rebuild can be checked against them. The tables of
three sources are not in the archive either, because their publishers' terms do not grant, or do not state, a
right to redistribute them (see [Sources not in the archive](#sources-not-in-the-archive)); their `sources.json`
and `schema.json` are included so that they can be rebuilt. `NOTICE.md` in the archive carries the notices the
other publishers require.

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

The code licence ([MIT](../LICENSE)) does not cover the data. Each of the 44 sources keeps its own terms. They
were read on 26 October 2026 and are quoted, with their URLs, in `metadata/dataset_licenses.json`:

| Source | Terms | In the archive |
|---|---|---|
| 39 federal sources (SEC, Federal Reserve Board, Treasury, FDIC, FFIEC, NCUA, CFPB, BLS, Census Bureau, CFTC, Department of Labor, EIA, FEMA, FHFA, NOAA, OFR, SBA) | Works of the US Government (17 U.S.C. § 105), not subject to copyright | yes |
| `nyfed_rates` | [New York Fed Terms of Use](https://www.newyorkfed.org/privacy/termsofuse): licence to copy and distribute, with attribution, the same terms passed on, and a prescribed notice for reference-rate data | yes, with the notice in `NOTICE.md` |
| `bis` | [Terms of permitted use of BIS statistics](https://data.bis.org/help/legal): use is unrestricted provided the BIS is cited as the source | yes, cited in `NOTICE.md` |
| `finra_short_volume` | [FINRA Terms of Use](https://www.finra.org/terms-of-use): personal non-commercial use only; distribution requires FINRA's written consent | no |
| `ken_french` | [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html): a copyright line and no terms of use | no |
| `rtdsm` | [Philadelphia Fed Real-Time Data Set](https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/real-time-data-set-for-macroeconomists): research use; redistribution not addressed | no |

If you use these tables, cite the original sources as well as FinAgenticBench; the Philadelphia Fed asks for
Croushore and Stark, *Journal of Econometrics* 105 (2001), 111–130.

## Sources not in the archive

The three sources above that are not in the archive are read by 27 of the 213 tasks (`finra_short_volume` 3,
`ken_french` 20, `rtdsm` 4). Until they are rebuilt, `build_dataset_layers --check` reports them as not
installed, the tests that read them are skipped, and a run of one of those tasks fails with a message naming the
source. To rebuild them from the publishers (the first downloads about 500 daily files, the other two a few):

```bash
# FINRA daily short sale volume, 2024–2025  ->  finra_short_volume.daily
uv run python -m scripts.ingest.fetch.fetch_full_population --only finra_short_volume
uv run python -m scripts.ingest.build.build_full_population_small_sources --only finra_short_volume.daily
uv run python -m scripts.ingest.build.build_dataset_layers --table finra_short_volume.daily

# Kenneth French daily factors and 10-industry portfolios  ->  ken_french.daily_factors, ken_french.daily_industry_returns
uv run python -m scripts.ingest.fetch.fetch_ken_french
uv run python -m scripts.ingest.build.build_dataset_layers --table ken_french.daily_factors --table ken_french.daily_industry_returns

# Philadelphia Fed real-time data set  ->  rtdsm.revisions
uv run python -m scripts.ingest.fetch.fetch_expansion --stage 1 --only rtdsm
uv run python -m scripts.ingest.build.build_expansion_runtime --only rtdsm
uv run python -m scripts.ingest.build.build_dataset_layers --table rtdsm.revisions
```

Each `--table` step cleans the staging table into `datasets/dataset_new/<source>/runtime/` and records its new
hash in the catalog. The publishers revise these files (the French library in particular), so a rebuild can
differ from the tables the paper's runs used; `sources.json` records the hashes of the original downloads, and
the paper's reference values for the affected tasks were computed from the original tables.

## Checking an installation

```bash
uv run python -m scripts.ingest.build.build_dataset_layers --check   # catalog, schemas and runtime hashes
uv run --group dev python -m pytest -m runtime_data                   # tests that read the tables
```

Adding `--skip-hashes` skips the hash comparison for a faster check. A source with no runtime table installed
is reported, not treated as an error; `--require-all-sources` makes it one.

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
