"""A BDC cross-section assembled by fiscal-period label mixes reporting dates.

Business development companies do not share a fiscal calendar. Of the 167 filers
in this filing index, 22 close their year somewhere other than 31 December, so
the label "fiscal 2024 Q3" is attached to period ends from 2023-11-30 through
2024-09-30. An analyst who groups by the label rather than by the calendar
period end builds a cross-section whose marks are up to 305 days apart, and for
floating-rate loan books that spans a move in the front-end rate: overnight SOFR
was 5.33% at the earliest of those dates and 4.96% at the latest, the September
2024 cut falling in between.

Three conventions the query states. Only the first, and the negative half of
the second, change a reported number; the rest are stated so that the answer is
well posed rather than because a competing reading would score differently.
Measured alternatives: dropping a whole accession on one absent mark gives
447,864.528 rather than 1,287,095.147 USD millions; excluding negative marks
gives 1,299,091.220. Excluding zero marks, and retaining the earliest rather
than the latest filed accession, both reproduce every reported output exactly,
the latter because the amendment restates an identical 1,132-row schedule:

1. ``fair_value_usd`` is absent on 5.79% of the rows in this label's filings,
   touching 91 of the 135 accessions that carry schedule rows. Dropping a whole
   accession on one missing mark would discard two thirds of the cross-section,
   so a missing mark contributes nothing and the surviving total is a reported
   floor rather than a portfolio valuation.
2. 6,082 rows carry a negative fair value and 5,616 carry zero. Both are real
   reported marks, so neither is filtered.
3. One filer appears twice under the label, a 10-Q and the 10-Q/A filed the next
   day. Keeping the latest filed accession per filer is what makes the
   cross-section one row per filer.

Only 135 of the 163 filers carrying the label have rows in the investment
schedule, so the share counts and the value totals rest on different
denominators; both are reported rather than blended.
"""

from decimal import Decimal
from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


FISCAL_YEAR = 2024
FISCAL_PERIOD = "Q3"
CALENDAR_END = "2024-09-30"
MILLION = Decimal(1_000_000)

OUTPUTS = (
    ("index_filer_count", 0, "the count of filers in the filing index as an integer"),
    ("noncalendar_fiscal_year_end_filers", 0, "the count of those filers closing their year other than 31 December, as an integer"),
    ("label_filing_count", 0, "the count of filings carrying the stated fiscal label as an integer"),
    ("label_distinct_period_ends", 0, "the count of distinct calendar period ends behind that label as an integer"),
    ("label_period_span_days", 0, "the day count from the earliest to the latest of those period ends as an integer"),
    ("label_filer_count", 0, "the count of filers retained under the label as an integer"),
    ("label_dominant_period_end", None, "the period end shared by the most retained filers, as an ISO YYYY-MM-DD string"),
    ("label_off_date_filers", 0, "the count of retained filers reporting a different period end, as an integer"),
    ("label_filers_with_schedule_rows", 0, "the count of retained filers carrying investment-schedule rows as an integer"),
    ("label_fair_value_mn", 3, "the retained filers' reported fair value in USD millions"),
    ("off_date_fair_value_mn", 3, "the portion of that total contributed by filers off the dominant period end, in USD millions"),
    ("calendar_filer_count", 0, "the count of filers in the calendar-date cross-section as an integer"),
    ("calendar_fair_value_mn", 3, "that cross-section's reported fair value in USD millions"),
    ("earliest_period_sofr_pct", 2, "overnight SOFR in effect at the earliest period end behind the label, in percent"),
    ("dominant_period_sofr_pct", 2, "overnight SOFR in effect at the dominant period end, in percent"),
)
variables = [
    Variable(name, None, f"Store {text}" + (f", rounded to {p} decimals from unrounded source rows." if p is not None else "."))
    for name, p, text in OUTPUTS
]
PRECISION = {name: p for name, p, _ in OUTPUTS}
TURNS = (
    tuple(n for n, _, _ in OUTPUTS[:5]),
    tuple(n for n, _, _ in OUTPUTS[5:8]),
    tuple(n for n, _, _ in OUTPUTS[8:11]),
    tuple(n for n, _, _ in OUTPUTS[11:]),
)


def _latest_per_filer(filings):
    """One filing per filer: the latest filed, then the higher accession."""
    ordered = filings.sort_values(["cik", "filed_date", "accession"])
    return ordered.drop_duplicates("cik", keep="last")


def _fair_value(schedule, accessions):
    """Reported marks only; an absent mark contributes nothing."""
    rows = schedule.loc[schedule.accession.isin(accessions), "fair_value_usd"]
    return sum((Decimal(str(x)) for x in rows.dropna()), Decimal(0))


def _sofr_at(rates, as_of):
    window = rates.loc[rates["Effective Date"].astype(str).le(as_of)].sort_values("Effective Date")
    if window.empty:
        raise ValueError(f"no SOFR fixing on or before {as_of}")
    return Decimal(str(window["Rate (%)"].iloc[-1]))


@lru_cache(maxsize=1)
def analysis():
    filings = load_expansion_table("sec_bdc_filings")
    schedule = load_expansion_table("sec_bdc")
    rates = load_expansion_table("nyfed_reference_rates")
    rates = rates.loc[rates["Rate Type"].eq("SOFR")]
    if rates["Rate (%)"].isna().any():
        raise ValueError("SOFR series carries a null rate")

    labelled = filings.loc[
        filings.fiscal_year.eq(FISCAL_YEAR) & filings.fiscal_period.eq(FISCAL_PERIOD)
    ]
    if labelled.empty:
        raise ValueError("no filing carries the stated fiscal label")
    ends = pd.to_datetime(labelled.period.astype(str))
    earliest, latest = ends.min(), ends.max()

    retained = _latest_per_filer(labelled)
    counts = retained.period.value_counts()
    if len(counts) < 2:
        raise ValueError("label carries a single period end, so there is nothing to align")
    # The query breaks a plurality tie by the earlier period end; value_counts
    # would otherwise resolve it by pandas' internal group order.
    dominant = min(str(p) for p in counts.index[counts.eq(counts.iloc[0])])
    off_date = retained.loc[retained.period.ne(dominant)]

    with_rows = retained.loc[retained.accession.isin(set(schedule.accession))]
    label_value = _fair_value(schedule, set(retained.accession))
    off_value = _fair_value(schedule, set(off_date.accession))

    calendar = _latest_per_filer(filings.loc[filings.period.eq(CALENDAR_END)])
    calendar_value = _fair_value(schedule, set(calendar.accession))

    return {
        "index_filer_count": int(filings.cik.nunique()),
        "noncalendar_fiscal_year_end_filers": int(filings.loc[~filings.fiscal_year_end.eq(1231)].cik.nunique()),
        "label_filing_count": len(labelled),
        "label_distinct_period_ends": int(labelled.period.nunique()),
        "label_period_span_days": int((latest - earliest).days),
        "label_filer_count": len(retained),
        "label_dominant_period_end": dominant,
        "label_off_date_filers": len(off_date),
        "label_filers_with_schedule_rows": len(with_rows),
        "label_fair_value_mn": float(label_value / MILLION),
        "off_date_fair_value_mn": float(off_value / MILLION),
        "calendar_filer_count": len(calendar),
        "calendar_fair_value_mn": float(calendar_value / MILLION),
        "earliest_period_sofr_pct": float(_sofr_at(rates, earliest.strftime("%Y-%m-%d"))),
        "dominant_period_sofr_pct": float(_sofr_at(rates, dominant)),
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


def validate_turn_4(outputs):
    return _check(outputs, TURNS[3])


def validate(outputs):
    return _check(outputs, tuple(v.name for v in variables))


validators = {
    f"validate_turn_{i}": turn_validator(globals()[f"validate_turn_{i}"])
    for i in range(1, 5)
}
