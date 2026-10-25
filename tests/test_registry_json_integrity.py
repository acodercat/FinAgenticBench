"""Reject duplicate keys even when a normal JSON loader silently accepts them."""

import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("relative_path", [
    "benchmarks.json", "metadata/capabilities.json", "metadata/case_taxonomy.json",
])
def test_registry_documents_have_unique_keys(relative_path):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            assert key not in result, f"{relative_path}: duplicate key {key}"
            result[key] = value
        return result

    root = Path(__file__).resolve().parents[1]
    json.loads((root / relative_path).read_text(), object_pairs_hook=unique_object)
