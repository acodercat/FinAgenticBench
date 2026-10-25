"""The control's comparison decides what counts as a recovery.

Both of this file's subjects have already been wrong once in the same sweep, so
they are pinned rather than left to reading:

* the numeric band was `0.05 * max(1.0, abs(expected))`, carried over from a
  benchmark whose figures were in the tens.  On a percentage-point RMSE of
  0.1152 that is a +/-0.05 band, 43 per cent wide, and it scored a guess of
  0.094 as a recovery -- which promoted `treasury_dns_inversion_forecast` T4 to
  a turn "answered completely with no data" that it had not been.
* an empty reply is a failed call, not a model that could not guess.  Recording
  it as 0/N made 91 of 147 probes read as clean refusals.
"""

from __future__ import annotations

from scripts.no_data_control import (
    NUMERIC_TOLERANCE, _guessable, _last_json_object, _same,
)


def test_numeric_band_is_relative_to_the_figure():
    assert _same(0.1152, 0.1152)
    assert _same(0.1152 * (1 + NUMERIC_TOLERANCE / 2), 0.1152)
    # The case that was scored a recovery under the absolute floor.
    assert not _same(0.094, 0.1152)


def test_a_large_figure_still_gets_a_proportional_band():
    assert _same(680.9, 680.985)
    assert not _same(600.0, 680.985)


def test_zero_is_matchable_without_widening_anything_else():
    assert _same(0.0, 0.0)
    assert not _same(0.001, 0.0)


def test_booleans_and_labels_are_not_numeric():
    assert _same(True, True) and not _same(False, True)
    assert _same("yes", True) and _same("no", False)
    assert _same("2024-02-01", "2024-02-01")
    assert _same(" Dongsi ", "dongsi")
    assert not _same("2024-02-02", "2024-02-01")


def test_a_missing_answer_never_counts_as_a_match():
    assert not _same(None, 0.0)
    assert not _same(None, False)
    assert not _same(None, "label")


def test_json_is_recovered_from_a_reply_that_reasons_first():
    text = 'Thinking about it...\n{"a": 1}\nand finally {"b": 2}'
    assert _last_json_object(text) == {"b": 2}
    assert _last_json_object("no object here") is None
    assert _last_json_object("[1, 2]") is None
    assert _last_json_object('{"nested": {"x": 1}}') == {"nested": {"x": 1}}


def test_guessable_shapes_are_labels_booleans_and_small_counts():
    assert _guessable("2024-02-01") and _guessable(True) and _guessable(6)
    assert not _guessable(4168)
    assert not _guessable(680.985)
