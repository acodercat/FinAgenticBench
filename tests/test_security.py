"""Model-written code may not leave files that another run on the machine could read, nor
reach the raw data behind the runtime's variables."""

import pytest

from core.security import SECURITY_CHECKER


@pytest.mark.parametrize("code", [
    "df.to_pickle('cache.pkl')",
    "x = pd.read_pickle('/tmp/merged.pkl')",
    "x = pd.read_parquet('x.parquet')",
    "df.to_csv('out.csv')",
    "df.to_csv(path, index=False)",
    "np.save('a.npy', [1])",
    "import pickle",
    "import joblib",
    "from tempfile import mkdtemp",
    "path = '/tmp/NF.pkl'",
    "import datasets",
    "from datasets import dataset_new",
    "import config",
    "base = 'datasets/dataset_new/cftc_cot/runtime'",
])
def test_shared_files_and_the_raw_data_are_blocked(code):
    assert SECURITY_CHECKER.check_code(code), code


@pytest.mark.parametrize("code", [
    "print(df.to_csv())",
    "print(df.to_csv(index=False))",
    "s = df.to_json(orient='records')",
    "table = datasets.load('filings')",
])
def test_text_output_and_the_data_handle_are_allowed(code):
    assert not SECURITY_CHECKER.check_code(code), code
