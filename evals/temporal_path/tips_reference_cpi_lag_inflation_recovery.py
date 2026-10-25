"""Reading consumer-price inflation out of TIPS reference CPIs.

Treasury indexes inflation-protected securities to the CPI-U, not seasonally
adjusted, with a lag. The reference CPI for the first day of a month is the
index for the third preceding month; for later days it moves linearly toward
the index for the second preceding month, by the share of the month elapsed.
The auction records publish the reference CPI on each security's dated date
and issue date, so two reference values in one month solve for two monthly
indexes, and one value ties two neighbouring months together.

Trap. The auction records list reference CPIs on dates in every month, and the
plausible wrong path reads a reference value dated in July 2025 as the July
2025 index. That path first becomes live in turn 1. On 2025-07-15 the
reference CPI is 321.09758 and on 2025-07-31 it is 321.44339, against a July
index of 323.048; the 12-month rate it gives is 2.33 percent from the 15th or
2.35 from the 31st, against 2.70. The real-time data set offers a second
shortcut: its July 2025 figure, 2.38 percent first released and 2.77 most
recent, is a seasonally adjusted one-month change at an annual rate, and
compounding its twelve monthly changes gives 2.74, still not the unadjusted
12-month rate. At two decimals every one of these lies outside the
rounding-boundary tolerance of 0.006 around 2.70, and the reference values lie
at least 1.60 away from the level, far outside its 0.0006. July 2025 is chosen
because its index was published after most models' training data, so the
level has to be read from the records rather than recalled.

The correct path reads July 2025 and August 2025 from the two October 2025
reference values, and July 2024 the same way from October 2024. September 2025
follows from the single reference value on 2025-11-28 once August 2025 is
known. September 2024 follows from the value on 2024-12-31 once October 2024
is known from the January 2025 pair, and agrees to three decimals with the
route through 2024-11-29 and the August 2024 index (315.30133 against
315.30100).

The Bureau of Labor Statistics published no index for October 2025. The value
the December 2025 reference CPI implies, 325.604, is the September 2025 index
grown for one month at its 12-month rate: 324.800 times the twelfth root of
324.800 over 315.301 gives 325.6044, the substitute Treasury's rules prescribe
when a month's index is missing.

Readings measured so they are not mistaken for oversights:

1. Dated-date references are read from original issues only. A reopening row
   repeats the original security's dated-date reference, and in one row
   (the August 2025 reopening of the 30-year security) the date beside it reads
   2025-08-15 while the value is the 2025-02-15 reference; no output uses
   August 2025 dates, but reading that row as an August reference would solve
   May and June 2025 as 315.054 and 323.935.
2. Index levels are rounded to three decimals, as published, and the 12-month
   rates are taken from the rounded levels. From the unrounded solutions they
   are 2.7049 and 3.0126 percent, the same at two decimals.
3. The seasonally adjusted rates compound each annualized change to a monthly
   factor by its twelfth root. Averaging the annualized changes instead gives
   2.7502 from the most recent values.
"""

import calendar
from functools import lru_cache

import numpy as np
import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


ADJUSTED_SERIES = "cpi_annualized_pct"
ADJUSTED_WINDOW = ("2024:08", "2025:07")
PUBLISHED_DECIMALS = 3

OUTPUTS = (
    ("july_2025_index_level", 3, "the July 2025 index level"),
    ("july_2025_twelve_month_pct", 2, "the 12-month inflation rate to July 2025 in percent"),
    ("adjusted_most_recent_twelve_month_pct", 4, "the compounded seasonally adjusted rate from the most recent values in percent"),
    ("adjusted_first_release_twelve_month_pct", 4, "the compounded seasonally adjusted rate from the first releases in percent"),
    ("revised_month_count", 0, "how many of the twelve monthly figures changed after first release, as an integer"),
    ("september_2025_index_level", 3, "the September 2025 index level"),
    ("september_2025_twelve_month_pct", 2, "the 12-month inflation rate to September 2025 in percent"),
    ("october_2025_index_level", 3, "the October 2025 index level"),
)


def _description(precision, text):
    if precision == 0:
        return f"Store {text}."
    return f"Store {text}, rounded to {precision} decimals."


variables = [Variable(name, None, _description(p, text)) for name, p, text in OUTPUTS]
PRECISION = {name: p for name, p, _ in OUTPUTS}
TURNS = (
    tuple(n for n, _, _ in OUTPUTS[:2]),
    tuple(n for n, _, _ in OUTPUTS[2:5]),
    tuple(n for n, _, _ in OUTPUTS[5:7]),
    tuple(n for n, _, _ in OUTPUTS[7:]),
)


@lru_cache(maxsize=1)
def _reference_cpi():
    """Reference CPI by date, from issue dates and original issues' dated dates."""
    auctions = load_expansion_table("treasury_auctions")
    tips = auctions.loc[auctions.inflation_index_security.astype(str).eq("Yes")]
    originals = tips.loc[tips.reopening.astype(str).eq("No")]
    rows = pd.concat([
        tips[["issue_date", "ref_cpi_on_issue_date"]].set_axis(["day", "value"], axis=1),
        originals[["dated_date", "ref_cpi_on_dated_date"]].set_axis(["day", "value"], axis=1),
    ]).dropna()
    rows = rows.assign(day=rows.day.astype(str), value=rows.value.astype(float)).drop_duplicates()
    if rows.day.duplicated().any():
        raise ValueError("conflicting reference CPI on one date")
    references = dict(zip(rows.day, rows.value))
    return references


def _elapsed(day):
    year, month, dom = (int(part) for part in day.split("-"))
    return (dom - 1) / calendar.monthrange(year, month)[1]


def _solve_pair(first, second):
    """Indexes for the third and second months before two dates in one month."""
    ref = _reference_cpi()
    step = (ref[second] - ref[first]) / (_elapsed(second) - _elapsed(first))
    earlier = ref[first] - _elapsed(first) * step
    return earlier, earlier + step


def _later_from_earlier(day, earlier):
    return earlier + (_reference_cpi()[day] - earlier) / _elapsed(day)


def _earlier_from_later(day, later):
    weight = _elapsed(day)
    return (_reference_cpi()[day] - weight * later) / (1 - weight)


def _published(level):
    return round(level, PUBLISHED_DECIMALS)


def _twelve_month(now, year_ago):
    return (_published(now) / _published(year_ago) - 1) * 100


def _compounded(rates):
    factors = (1 + np.asarray(rates, dtype=float) / 100) ** (1 / 12)
    return float((factors.prod() - 1) * 100)


@lru_cache(maxsize=1)
def analysis():
    july_2024, _ = _solve_pair("2024-10-15", "2024-10-31")
    october_2024, _ = _solve_pair("2025-01-15", "2025-01-31")
    september_2024 = _earlier_from_later("2024-12-31", october_2024)
    july_2025, august_2025 = _solve_pair("2025-10-15", "2025-10-31")
    september_2025 = _later_from_earlier("2025-11-28", august_2025)
    october_2025 = _later_from_earlier("2025-12-31", september_2025)

    realtime = load_expansion_table("philadelphia_fed_rtdsm")
    window = realtime.loc[
        realtime.series.astype(str).eq(ADJUSTED_SERIES)
        & realtime.period.astype(str).between(*ADJUSTED_WINDOW)
    ]
    if len(window) != 12 or window[["first_release", "most_recent"]].isna().any().any():
        raise ValueError("expected twelve complete monthly adjusted changes")

    return {
        "july_2025_index_level": _published(july_2025),
        "july_2025_twelve_month_pct": _twelve_month(july_2025, july_2024),
        "adjusted_most_recent_twelve_month_pct": _compounded(window.most_recent),
        "adjusted_first_release_twelve_month_pct": _compounded(window.first_release),
        "revised_month_count": int(window.first_release.ne(window.most_recent).sum()),
        "september_2025_index_level": _published(september_2025),
        "september_2025_twelve_month_pct": _twelve_month(september_2025, september_2024),
        "october_2025_index_level": _published(october_2025),
    }


@lru_cache(maxsize=1)
def ground_truth():
    answers = analysis()
    return tuple(answers[variable.name] for variable in variables)


def _check(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip(by_name, ground_truth()))
    return validate_ordered_outputs(
        outputs,
        [by_name[n] for n in names],
        [truth[n] for n in names],
        [PRECISION[n] for n in names],
    )


def validate(outputs):
    return _check(outputs, tuple(v.name for v in variables))


def validate_turn_1(outputs):
    return _check(outputs, TURNS[0])


def validate_turn_2(outputs):
    return _check(outputs, TURNS[1])


def validate_turn_3(outputs):
    return _check(outputs, TURNS[2])


def validate_turn_4(outputs):
    return _check(outputs, TURNS[3])


validators = {
    f"validate_turn_{i}": turn_validator(globals()[f"validate_turn_{i}"])
    for i in range(1, 5)
}
