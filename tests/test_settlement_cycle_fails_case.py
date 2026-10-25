"""Boundaries for the settlement-cycle fails alignment trap.

Anchors pin the literals. Property tests name no literal, so each can fail on
its own rather than being decided by an equality asserted earlier in the same
function.
"""

from datetime import date, timedelta

import numpy as np

from evals.temporal_path import settlement_cycle_trade_date_fails_alignment as case


def _settlement_days():
    return sorted(case._fails_by_symbol().index.get_level_values("day").unique())


def _next_settlement(day, steps):
    days = [d for d in _settlement_days() if d > day]
    return days[steps - 1]


# --- anchors ---------------------------------------------------------------

def test_anchor_values():
    answers = case.analysis()
    assert answers["recent_peak_trading_day"] == "2025-04-09"
    assert answers["recent_peak_settlement_date"] == "2025-04-10"
    assert answers["recent_peak_failed_shares"] == 515030616
    assert answers["largest_rise_symbol"] == "DMN"
    assert answers["largest_rise_shares"] == 98318799
    assert answers["early_peak_trading_day"] == "2024-01-16"
    assert answers["early_peak_settlement_date"] == "2024-01-18"
    assert answers["early_peak_failed_shares"] == 118542824
    assert answers["settlement_for_2024_05_24"] == "2024-05-29"
    assert answers["settlement_for_2025_01_08"] == "2025-01-09"
    assert answers["settlement_for_2025_10_10"] == "2025-10-14"


# --- the settlement calendar ---------------------------------------------

def test_calendar_reproduces_the_files_settlement_dates():
    days = _settlement_days()
    current, last = date.fromisoformat(days[0]), date.fromisoformat(days[-1])
    expected = []
    while current <= last:
        if case.is_settlement_day(current.isoformat()):
            expected.append(current.isoformat())
        current += timedelta(days=1)
    assert expected == days


def test_bank_holidays_trade_but_do_not_settle():
    trading = set(case._daily_volume().index)
    settling = set(_settlement_days())
    assert sorted(trading - settling) == sorted(
        d for d in case.SETTLEMENT_HOLIDAYS if d in trading
    )
    assert sorted(settling - trading)


# --- the trap: wrong cycles give other answers ------------------------------

def test_wrong_cycles_change_every_turn():
    answers = case.analysis()
    recent, early = answers["recent_peak_trading_day"], answers["early_peak_trading_day"]
    assert case._failed_on(_next_settlement(recent, 2)) != answers["recent_peak_failed_shares"]
    assert case._failed_on(recent) != answers["recent_peak_failed_shares"]
    assert case._failed_on(_next_settlement(early, 1)) != answers["early_peak_failed_shares"]
    assert _next_settlement("2024-05-24", 1) != answers["settlement_for_2024_05_24"]
    trading = sorted(case._daily_volume().index)
    for day, name in (("2025-01-08", "settlement_for_2025_01_08"), ("2025-10-10", "settlement_for_2025_10_10")):
        next_trading_day = trading[trading.index(day) + 1]
        assert next_trading_day != answers[name]


def test_two_day_reading_of_the_rise_differs():
    answers = case.analysis()
    fails = case._fails_by_symbol()
    wrong = _next_settlement(answers["recent_peak_trading_day"], 2)
    before = _next_settlement(answers["recent_peak_trading_day"], 1)
    rise = fails.xs(wrong, level="day").sub(fails.xs(before, level="day"), fill_value=0).max()
    assert int(rise) != answers["largest_rise_shares"]


def test_ranking_by_short_volume_picks_another_2025_day():
    volume = case.load_expansion_table("finra_short_volume")
    short = volume.assign(day=volume.date.astype(str)).groupby("day").short_volume_shares.sum()
    in_2025 = short.loc[(short.index >= case.RECENT_WINDOW[0]) & (short.index <= case.RECENT_WINDOW[1])]
    assert in_2025.idxmax() != case.analysis()["recent_peak_trading_day"]


# --- properties (no literals) ----------------------------------------------

def test_selections_have_clear_margins():
    volume = case._daily_volume()
    for window in (case.RECENT_WINDOW, case.EARLY_WINDOW):
        span = volume.loc[(volume.index >= window[0]) & (volume.index <= window[1])].sort_values()
        assert span.iloc[-1] - span.iloc[-2] > 0.01 * span.iloc[-1]
    answers = case.analysis()
    fails = case._fails_by_symbol()
    settlement = answers["recent_peak_settlement_date"]
    previous = _settlement_days()[_settlement_days().index(settlement) - 1]
    rise = fails.xs(settlement, level="day").sub(fails.xs(previous, level="day"), fill_value=0).sort_values()
    assert rise.iloc[-1] - rise.iloc[-2] > 1_000_000


def test_fails_jump_one_cycle_after_volume_spikes():
    fails = case._fails_by_symbol().unstack("day").fillna(0)
    volume = case.load_expansion_table("finra_short_volume")
    volume = volume.assign(day=volume.date.astype(str), symbol=volume.symbol.astype(str))
    volume = volume.groupby(["symbol", "day"]).total_volume_shares.sum().unstack("day").astype(float)
    days = list(volume.columns)
    symbols = volume.index.intersection(fails.index)
    volume = volume.loc[symbols]
    fails = fails.reindex(index=symbols, columns=days).fillna(0).astype(float)
    median = volume.T.rolling(20, min_periods=10).median().shift(1).T
    spikes = (volume > 10 * median) & (volume > 1e6)

    def peak_offset(first, last):
        total, count = np.zeros(6), 0
        for i, day in enumerate(days):
            if not first <= day <= last or i < 1 or i + 5 >= len(days):
                continue
            for symbol in spikes.index[spikes[day].fillna(False)]:
                row = fails.loc[symbol].to_numpy()
                total += row[i:i + 6] - row[i - 1]
                count += 1
        return int(np.argmax(total / count))

    assert peak_offset("2024-01-03", "2024-05-20") == 2
    assert peak_offset("2024-06-03", "2025-12-20") == 1


# --- validator ---------------------------------------------------------------

def test_validator_accepts_truth_and_rejects_a_change():
    truth = dict(zip((v.name for v in case.variables), case.ground_truth()))
    assert case.validate(truth).success
    for name in truth:
        changed = dict(truth)
        changed[name] = truth[name] + 1 if isinstance(truth[name], int) else truth[name] + "x"
        assert not case.validate(changed).success, name
