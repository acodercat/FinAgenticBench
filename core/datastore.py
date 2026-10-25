"""Lazy, scoped access to the tables a case declares.

Eager injection registers every declared table as a loaded DataFrame before
the agent runs. For a case whose sources include a 12M-row register that is
only needed in its third turn, that front-loads memory and fills the prompt
with column-level descriptions. The lazy alternative registers one handle,
``datasets``, whose description lists the tables the case may use; the agent
loads a table when it needs it and can ask for the full description of any
listed table. Tables outside the declaration are not reachable through the
handle, so the scoping guarantee is the same as eager injection's.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from cave_agent import Variable

from core.runtime_catalog import RuntimeTable

HANDLE_NAME = "datasets"


class DataStore:
    """The tables one case may load, materialised on first request."""

    def __init__(self, tables: Sequence[RuntimeTable]):
        self._tables = {table.name: table for table in tables}
        self._frames: dict[str, pd.DataFrame] = {}

    def tables(self) -> list[dict[str, str]]:
        """Name and one-line summary of every loadable table."""
        return [{"name": name, "summary": table.summary} for name, table in self._tables.items()]

    def describe(self, name: str) -> str:
        """The full description of one table: columns, units and caveats."""
        return self._table(name).description

    def load(self, name: str) -> pd.DataFrame:
        """The table as a pandas DataFrame, read once and cached."""
        if name not in self._frames:
            self._frames[name] = self._table(name).loader()
        return self._frames[name]

    def loaded(self) -> list[str]:
        """Names of the tables loaded so far, in load order."""
        return list(self._frames)

    def _table(self, name: str) -> RuntimeTable:
        try:
            return self._tables[name]
        except KeyError:
            raise KeyError(
                f"{name!r} is not a table of this case; loadable tables: {sorted(self._tables)}"
            ) from None

    def __repr__(self) -> str:
        return f"DataStore(tables={sorted(self._tables)}, loaded={self.loaded()})"


def datastore_description(tables: Sequence[RuntimeTable]) -> str:
    """The handle description the runtime shows the agent."""
    listing = "\n".join(f"  - {table.name}: {table.summary}" for table in tables)
    return (
        "Handle to this task's data. Loadable tables:\n"
        f"{listing}\n"
        f"{HANDLE_NAME}.load(name) returns the table as a pandas DataFrame (cached); "
        f"{HANDLE_NAME}.describe(name) returns its columns, units and caveats; "
        f"{HANDLE_NAME}.tables() lists the names. Only the tables above can be loaded."
    )


def datastore_variable(tables: Sequence[RuntimeTable]) -> Variable:
    """The single runtime variable that replaces eager DataFrame injection."""
    if not tables:
        raise ValueError("a DataStore needs at least one table")
    return Variable(HANDLE_NAME, DataStore(tables), datastore_description(tables))
