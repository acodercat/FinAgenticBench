"""A SOFR average is compounded on business days, with simple interest across gaps.

The New York Fed publishes 30-, 90- and 180-day SOFR averages each business
day. Each covers the calendar days from the date that many days before the
publication date up to the day before it. A daily rate earns simple interest
for every calendar day until the next business day, so a Friday rate covers
the weekend in one simple step; interest compounds only from one business day
to the next; and the year has 360 days. Where the window opens on a day that is
not a business day, the rate of the preceding business day covers it until the
next one. Rounded half up to five decimals, that calculation reproduces all 748
publication dates from 2023 through 2025 on all three windows, and the SOFR
Index itself steps the same way from one business day to the next.

Every competing reading reproduces none of the 748 on any window. Compounding
every calendar day differs from the reproducing value by a median 0.000313,
0.000333 and 0.000337 percentage points on the three windows; a 365-day year
by 0.000125, 0.000397 and 0.000804; a simple average weighting each calendar
day equally by 0.009102, 0.028897 and 0.058227. Five decimals separate all of
them. Shifting the window a day later or earlier under the reproducing
calculation matches 115 or 111 of the 748 30-day averages. The case reports the
unrounded 2025-12-31 values, 3.7865938 on the 30-day window and 4.1999627 on
the 180-day window, which each competing reading moves.

The gap between the compounded and the simple day-weighted average is interest
on interest. Over 2023 to 2025 it has a median of 0.9102 basis points on the
30-day window and 5.8227 on the 180-day window, and reaches 7.0790, growing
with the window and with the level of rates.

The curve quotes the one-month bill on a 365-day bond-equivalent basis. The
first cut in the target range the table carries falls on 2024-09-19. Before
it, the published 30-day average reads a median 18.626 basis points below the
curve's 1-month point, and 11.232 below once restated to a 365-day year; from
that date on it reads 0.262 above, and 6.313 above once restated. The basis is
worth roughly five days' interest. The change of sign is the difference between
a rate already fixed over the past month and a yield that prices the month
ahead.

The averages lag the policy rate by their window. The 30-day window first
holds only rates fixed on or after the 2024-09-19 cut on 2024-10-21, and the
180-day window on 2025-03-18; on 2024-10-21 the published 30-day average
stands 45.488 basis points below the published 180-day average, which still
holds pre-cut rates. Reading the window start as the calendar day itself,
without carrying a non-business day back to the preceding fixing, gives the
same two dates here.

Nearest cases, recorded for the portfolio checkpoint. The retired
single-source sofr_index_financing_cost took one year's financing cost from
the index ratio and set it against an arithmetic mean of fixings, close to
this case's second turn. What is new here is recovering the averaging
convention from the published averages themselves, which the index does not
settle on its own, and the fourth turn's pass-through lag. The active
treasury_bill_discount_investment_rate_basis converts bill quotes between
bases; here the basis step is one comparison inside a question about
in-arrears averages.

Three conventions the query states because each moves an output:

1. Medians use linear interpolation. The 748 publication dates are an even
   count whose middle values differ on both premium series; the 429 and 319
   dates either side of the first cut are odd, so there the rule is stated and
   does not bind.
2. The first cut is read from the target range the table itself carries, as
   the first date whose upper bound is below the previous reported date's.
3. The premium is taken on the unrounded reproducing average, not the
   published five-decimal figure. The published figure reads 0.9103, 5.8223
   and 7.0787 against the reported 0.9102, 5.8227 and 7.0790.
"""

from functools import lru_cache

import numpy as np
import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


WINDOW = ("2023-01-01", "2025-12-31")
REPORT_DATE = "2025-12-31"
DAY_BASIS = 360
BILL_BASIS = 365
BASIS_POINTS = 100
AVERAGE_COLUMN = {30: "30-Day Average SOFR", 90: "90-Day Average SOFR", 180: "180-Day Average SOFR"}

OUTPUTS = (
    ("publication_date_count", 0, "the count of publication dates carrying all three averages as an integer"),
    ("unrounded_30_day_average_pct", 7, "the unrounded 30-day average for the report date in percent"),
    ("unrounded_180_day_average_pct", 7, "the unrounded 180-day average for the report date in percent"),
    ("median_premium_30_day_bp", 4, "the median 30-day compounded less simple average in basis points"),
    ("median_premium_180_day_bp", 4, "the median 180-day compounded less simple average in basis points"),
    ("max_premium_180_day_bp", 4, "the largest 180-day compounded less simple average in basis points"),
    ("first_cut_date", None, "the first date the target range's upper bound fell, as an ISO YYYY-MM-DD string"),
    ("median_raw_spread_before_bp", 3, "the median raw 30-day average less the 1-month curve point before that date, in basis points"),
    ("median_adjusted_spread_before_bp", 3, "the median 365-day restated average less the curve point before that date, in basis points"),
    ("median_raw_spread_after_bp", 3, "the median raw spread on or after that date, in basis points"),
    ("median_adjusted_spread_after_bp", 3, "the median restated spread on or after that date, in basis points"),
    ("first_clean_30_day_date", None, "the first publication date whose 30-day window holds only rates fixed on or after the cut, as an ISO YYYY-MM-DD string"),
    ("first_clean_180_day_date", None, "the first publication date whose 180-day window holds only such rates, as an ISO YYYY-MM-DD string"),
    ("thirty_less_180_day_on_clean_date_bp", 3, "the published 30-day average less the published 180-day average on the first of those dates, in basis points"),
)


def _description(precision, text):
    if precision in (None, 0):
        return f"Store {text}."
    return f"Store {text}, rounded to {precision} decimals from unrounded source rows."


variables = [Variable(name, None, _description(p, text)) for name, p, text in OUTPUTS]
PRECISION = {name: p for name, p, _ in OUTPUTS}
TURNS = (
    tuple(n for n, _, _ in OUTPUTS[:3]),
    tuple(n for n, _, _ in OUTPUTS[3:6]),
    tuple(n for n, _, _ in OUTPUTS[6:11]),
    tuple(n for n, _, _ in OUTPUTS[11:]),
)


@lru_cache(maxsize=1)
def _rates():
    table = load_expansion_table("nyfed_reference_rates").copy()
    table["date"] = table["Effective Date"].astype(str)
    sofr = table.loc[table["Rate Type"].eq("SOFR")].set_index("date")["Rate (%)"].astype(float).sort_index()
    if sofr.index.duplicated().any():
        raise ValueError("more than one SOFR row on a date")
    averages = table.loc[table["Rate Type"].eq("SOFRAI")].set_index("date").sort_index()
    effr = table.loc[table["Rate Type"].eq("EFFR")].set_index("date").sort_index()
    return sofr, averages, effr


@lru_cache(maxsize=1)
def _calendar():
    """Each calendar day's applicable rate and whether it is a business day."""
    sofr, _, _ = _rates()
    days = pd.date_range(sofr.index.min(), sofr.index.max(), freq="D").strftime("%Y-%m-%d")
    rate = sofr.reindex(days).ffill().to_numpy() / 100
    business = np.isin(days, sofr.index)
    return pd.Index(days), rate, business


def _window(publication, length):
    days, rate, business = _calendar()
    end = days.get_loc(publication)
    start = days.get_loc((pd.Timestamp(publication) - pd.Timedelta(days=length)).strftime("%Y-%m-%d"))
    return rate[start:end], business[start:end]


def compounded_average(publication, length):
    """Business-day compounding, simple interest across the days a rate covers."""
    rate, business = _window(publication, length)
    opens = business.copy()
    opens[0] = True
    starts = np.flatnonzero(opens)
    spans = np.diff(np.append(starts, len(rate)))
    growth = np.prod(1 + rate[starts] * spans / DAY_BASIS)
    return (growth - 1) * DAY_BASIS / length * 100


def simple_average(publication, length):
    rate, _ = _window(publication, length)
    return rate.mean() * 100


@lru_cache(maxsize=1)
def _publications():
    _, averages, _ = _rates()
    columns = list(AVERAGE_COLUMN.values())
    frame = averages.loc[averages.index.to_series().between(*WINDOW), columns].astype(float)
    frame = frame.dropna()
    for length, column in AVERAGE_COLUMN.items():
        rebuilt = [round(compounded_average(d, length) + 1e-12, 5) for d in frame.index]
        if not np.allclose(rebuilt, frame[column].to_numpy(), atol=1e-9, rtol=0):
            raise ValueError(f"the {length}-day convention does not reproduce every publication")
    return frame


def _first_cut():
    _, _, effr = _rates()
    upper = pd.to_numeric(effr["Target Rate To (%)"], errors="coerce")
    upper = upper.loc[upper.index.to_series().between(*WINDOW)]
    falls = upper.loc[upper.diff() < 0]
    if falls.empty:
        raise ValueError("the target range never falls in the window")
    return str(falls.index[0])


@lru_cache(maxsize=1)
def _spreads():
    frame = _publications()
    curve = load_expansion_table("treasury_yield_curve").copy()
    curve["Date"] = curve.Date.astype(str)
    bill = pd.to_numeric(curve.set_index("Date")["1 Mo"], errors="coerce").reindex(frame.index)
    if bill.isna().any():
        raise ValueError("a publication date has no 1-month curve point")
    average = frame[AVERAGE_COLUMN[30]]
    return pd.DataFrame({
        "raw": (average - bill) * BASIS_POINTS,
        "adjusted": (average * BILL_BASIS / DAY_BASIS - bill) * BASIS_POINTS,
    })


def _window_opening_rate_date(publication, length):
    """The fixing date of the rate that covers a window's first calendar day."""
    sofr, _, _ = _rates()
    start = (pd.Timestamp(publication) - pd.Timedelta(days=length)).strftime("%Y-%m-%d")
    return sofr.index[sofr.index <= start][-1]


def _first_clean_publication(cut, length):
    frame = _rates()[1]
    for publication in frame.index[frame.index >= cut]:
        if _window_opening_rate_date(publication, length) >= cut:
            return str(publication)
    raise ValueError(f"no {length}-day window is free of rates fixed before {cut}")


@lru_cache(maxsize=1)
def analysis():
    frame = _publications()
    premium = {
        length: pd.Series([
            (compounded_average(d, length) - simple_average(d, length)) * BASIS_POINTS
            for d in frame.index
        ])
        for length in (30, 180)
    }
    cut = _first_cut()
    spreads = _spreads()
    before = spreads.loc[spreads.index < cut]
    after = spreads.loc[spreads.index >= cut]
    return {
        "publication_date_count": len(frame),
        "unrounded_30_day_average_pct": float(compounded_average(REPORT_DATE, 30)),
        "unrounded_180_day_average_pct": float(compounded_average(REPORT_DATE, 180)),
        "median_premium_30_day_bp": float(premium[30].median()),
        "median_premium_180_day_bp": float(premium[180].median()),
        "max_premium_180_day_bp": float(premium[180].max()),
        "first_cut_date": cut,
        "median_raw_spread_before_bp": float(before.raw.median()),
        "median_adjusted_spread_before_bp": float(before.adjusted.median()),
        "median_raw_spread_after_bp": float(after.raw.median()),
        "median_adjusted_spread_after_bp": float(after.adjusted.median()),
        **_lag(cut),
    }


def _lag(cut):
    _, averages, _ = _rates()
    clean_30 = _first_clean_publication(cut, 30)
    clean_180 = _first_clean_publication(cut, 180)
    gap = float(averages.loc[clean_30, AVERAGE_COLUMN[30]]) - float(averages.loc[clean_30, AVERAGE_COLUMN[180]])
    return {
        "first_clean_30_day_date": clean_30,
        "first_clean_180_day_date": clean_180,
        "thirty_less_180_day_on_clean_date_bp": gap * BASIS_POINTS,
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


def validate_turn_1(outputs):
    return _check(outputs, TURNS[0])


def validate_turn_2(outputs):
    return _check(outputs, TURNS[1])


def validate_turn_3(outputs):
    return _check(outputs, TURNS[2])


def validate(outputs):
    return _check(outputs, tuple(v.name for v in variables))


def validate_turn_4(outputs):
    return _check(outputs, TURNS[3])


validators = {
    f"validate_turn_{i}": turn_validator(globals()[f"validate_turn_{i}"])
    for i in range(1, 5)
}
