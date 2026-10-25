"""The lazy data handle: scoping, caching and the description it advertises."""

import pandas as pd
import pytest

from core.datastore import HANDLE_NAME, DataStore, datastore_description, datastore_variable
from core.runtime_catalog import RuntimeTable


def _table(name, description="A small table. Columns: a, b."):
    calls = []

    def loader():
        calls.append(name)
        return pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    return RuntimeTable(name=name, labels=(name,), loader=loader, description=description), calls


def test_load_is_scoped_cached_and_ordered():
    first, first_calls = _table("first_df")
    second, second_calls = _table("second_df")
    store = DataStore([first, second])
    assert store.loaded() == []
    frame = store.load("second_df")
    assert frame.equals(store.load("second_df"))
    assert second_calls == ["second_df"], "a table is read once"
    assert first_calls == []
    assert store.loaded() == ["second_df"]
    with pytest.raises(KeyError, match="loadable tables"):
        store.load("other_df")
    assert "other_df" not in repr(store)


def test_listing_and_description_come_from_the_registry():
    table, _ = _table("only_df", "Deposits by branch. Columns: cert, deposits. Blanks are not zero.")
    store = DataStore([table])
    assert store.tables() == [{"name": "only_df", "summary": "Deposits by branch."}]
    assert store.describe("only_df").endswith("Blanks are not zero.")
    with pytest.raises(KeyError):
        store.describe("missing_df")


def test_handle_variable_lists_every_table_and_only_those():
    tables = [_table("alpha_df")[0], _table("beta_df", "Beta rows. Columns: x.")[0]]
    variable = datastore_variable(tables)
    assert variable.name == HANDLE_NAME
    assert isinstance(variable.value, DataStore)
    description = variable.description
    assert description == datastore_description(tables)
    assert "  - alpha_df: A small table." in description
    assert "  - beta_df: Beta rows." in description
    assert f"{HANDLE_NAME}.load(name)" in description
    assert "Columns: x." not in description, "column detail stays behind describe()"


def test_handle_requires_at_least_one_table():
    with pytest.raises(ValueError):
        datastore_variable([])
