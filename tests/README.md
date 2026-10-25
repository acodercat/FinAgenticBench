# Tests

The suite is flat: `test_<area>.py` for the harness, results, scoring and data layers, and one file per task or
task batch for the benchmark itself.

Two markers separate what needs data:

- `runtime_data`: needs the runtime tables in `datasets/dataset_new/` (`FinAgenticBench-data.zip`). The files are
  listed in `RUNTIME_TEST_FILES` in `conftest.py` and are skipped automatically when the tables are absent.
- `raw_data`: needs the source-native archives that the data build reads; these are not distributed, so these tests
  are for rebuilding the data from source.

```bash
uv run --group dev python -m pytest -m "not runtime_data and not raw_data"   # code only
uv run --group dev python -m pytest -m runtime_data                          # with the data artifact
uv run --group dev python -m pytest                                          # everything available
```
