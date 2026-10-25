"""A pension obligation is remeasured at current rates every year.

A sponsor discounts its pension benefits at a rate read at each measurement
date from yields on high-quality corporate bonds whose timing matches the
benefit payments. Unlike a lease rate, which is fixed when the lease begins,
the pension rate is reset every year, so it moves with market yields and the
obligation moves with it.

Among the large filers' annual reports, 124 tag a discount rate for their
defined benefit pension plans at both their 2021 and 2022 year ends. One of
them tags a negative 2022 rate, which cannot be a discount rate. For the other
123 the rate rose a median 250 basis points while the ten-year Treasury par
yield on the same dates rose a median 236, and 118 rate changes lie within 50
basis points of the yield change. At the 2022 year end the median rate is 5.20
percent, a median 132 basis points above the ten-year yield, and only 4 stand
below it: high-quality corporate bonds carry a credit spread over Treasuries.

Rising rates shrink the obligation's present value, which the sponsor records
as an actuarial gain. 112 of the filers carry the 2022 gain and the obligation
at both year ends. 20 tag the gain below zero, and every one of those shows a
lower obligation at the end of the year: a sign convention, not a loss. The
gain's size over the opening obligation, per unit of rate change, implies a
median duration of 9.0457 years. All 108 filers that carry plan assets with
the year's return lost money on those assets, yet 71 of them ended the year
with assets less obligation strictly higher, and the median funded ratio rose from 0.9495
to 0.9794.

Readings measured so they are not mistaken for oversights:

0. The tracking count compares each filer's rate change with that same filer's
   own yield change. The earlier wording "within 50 basis points of it",
   following the median yield change, was read by models as a comparison with
   the median; the two readings differ for one filer whose year end is not in
   December, so the query and variable description now name the per-filer
   comparison.
1. Only the primary presentation (presentation index 0) is read. Narrative
   restatements at other indexes round the same fact, as 3,668 against 3,700
   million for one filer's 2022 gain; reading them gives values the tables do
   not carry.
2. Every value is taken as first reported, from the earliest-filed annual
   report carrying it. Reading the latest report instead gives 2 invalid
   rates, 117 tracking, 111 gain filers with 21 negative, a median duration of
   9.0623, and 70 of 106 losing plans improving out of 107.
3. Filers whose rate is not strictly between zero and one at either date are
   left out. Reading the negative rate as its absolute value admits that filer
   and gives 119 tracking, 113 gain filers, 21 negative, 9.0623 years, and 72
   of 109 improving.
4. The comparison yield is the ten-year par yield on the year-end date or the
   latest earlier date. On the twenty-year point 103 rates track and the
   median spread is 106; on the thirty-year point 71 track and it is 123.
5. The implied duration divides the gain's size by the opening obligation and
   by the rate change as a decimal. The signed gain gives 8.7768, the closing
   obligation 12.4858 and the average obligation 10.5787.
6. "Strictly higher" matters: one losing plan ends with an unchanged funded
   position, so "at least as high" gives 72.
"""

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


TAGS = {
    "DefinedBenefitPlanAssumptionsUsedCalculatingBenefitObligationDiscountRate": "rate",
    "DefinedBenefitPlanBenefitObligation": "obligation",
    "DefinedBenefitPlanFairValueOfPlanAssets": "assets",
    "DefinedBenefitPlanActuarialGainLoss": "gain",
    "DefinedBenefitPlanActualReturnOnPlanAssets": "return",
}
INSTANTS = ("rate", "obligation", "assets")
AXIS, MEMBER = "RetirementPlanType", "PensionPlansDefinedBenefit"
PRIMARY_PRESENTATION = 0
START_YEAR, END_YEAR = 2021, 2022
TENOR = "10 Yr"
TRACKING_BAND_BP = 50

OUTPUTS = (
    ("rate_filer_count", 0, "the count of filers tagging a rate at both year ends as an integer"),
    ("invalid_rate_count", 0, "how many tag a value that cannot be a rate stated as a fraction, as an integer"),
    ("median_rate_change_bp", 2, "the median change in the remaining filers' rate in basis points"),
    ("median_yield_change_bp", 2, "the median change in the Treasury yield between the same dates in basis points"),
    ("tracking_count", 0, "how many filers' rate changes lie within the band of that filer's own yield change, as an integer"),
    ("median_end_rate_pct", 2, "the median rate at the later year end in percent"),
    ("below_treasury_count", 0, "how many rates stand below the Treasury yield at the later year end, as an integer"),
    ("median_spread_bp", 2, "the median rate less the Treasury yield at the later year end in basis points"),
    ("gain_filer_count", 0, "the count of filers carrying the gain and both obligations as an integer"),
    ("negative_gain_count", 0, "how many tag the gain below zero, as an integer"),
    ("negative_gain_obligation_fell_count", 0, "how many of those show a lower obligation at the later year end, as an integer"),
    ("median_implied_duration_years", 4, "the median implied duration in years"),
    ("funded_filer_count", 0, "the count of filers also carrying assets at both year ends and the year's return as an integer"),
    ("negative_return_count", 0, "how many report a return below zero, as an integer"),
    ("negative_return_improved_count", 0, "how many of those end the year with assets less obligation higher, as an integer"),
    ("median_start_funded_ratio", 4, "the median assets divided by obligation at the earlier year end"),
    ("median_end_funded_ratio", 4, "the median assets divided by obligation at the later year end"),
)


def _description(precision, text):
    if precision == 0:
        return f"Store {text}."
    return f"Store {text}, rounded to {precision} decimals from unrounded source rows."


variables = [Variable(name, None, _description(p, text)) for name, p, text in OUTPUTS]
PRECISION = {name: p for name, p, _ in OUTPUTS}
TURNS = (
    tuple(n for n, _, _ in OUTPUTS[:3]),
    tuple(n for n, _, _ in OUTPUTS[3:8]),
    tuple(n for n, _, _ in OUTPUTS[8:12]),
    tuple(n for n, _, _ in OUTPUTS[12:]),
)


@lru_cache(maxsize=1)
def _facts():
    """First-reported pension-plan facts per filer, item, kind and calendar year."""
    facts = load_expansion_table("sec_notes_dimensional")
    rows = facts.loc[
        facts.tag.astype(str).isin(list(TAGS))
        & facts.form.astype(str).eq("10-K")
        & facts.axis.astype(str).eq(AXIS)
        & facts.member.astype(str).eq(MEMBER)
        & facts.iprx.eq(PRIMARY_PRESENTATION)
        & facts.value.notna()
    ].copy()
    rows["item"] = rows.tag.astype(str).map(TAGS)
    rows = rows.loc[rows.qtrs.eq(rows.item.isin(INSTANTS).map({True: 0, False: 4}))]
    rows["end_date"] = pd.to_datetime(rows.ddate.astype(str)).dt.strftime("%Y-%m-%d")
    rows["year"] = rows.end_date.str[:4].astype(int)
    rows = rows.sort_values(["filed", "adsh"]).drop_duplicates(["cik", "item", "end_date"], keep="first")
    rows = rows.sort_values("end_date").drop_duplicates(["cik", "item", "year"], keep="last")
    return rows[["cik", "item", "year", "end_date", "value"]]


@lru_cache(maxsize=1)
def _yield_on():
    curve = load_expansion_table("treasury_yield_curve")
    points = curve.assign(date=curve.Date.astype(str))[["date", TENOR]].dropna().sort_values("date")
    dates, values = points.date.to_numpy(), points[TENOR].astype(float).to_numpy()

    def lookup(day):
        position = dates.searchsorted(day, side="right") - 1
        if position < 0:
            raise ValueError(f"no {TENOR} yield published on or before {day}")
        return float(values[position])

    return lookup


def _year(item, year):
    rows = _facts()
    return rows.loc[rows.item.eq(item) & rows.year.eq(year)].set_index("cik")


@lru_cache(maxsize=1)
def analysis():
    lookup = _yield_on()
    start, end = _year("rate", START_YEAR), _year("rate", END_YEAR)
    both = start.index.intersection(end.index)
    rates = pd.DataFrame({
        "start": start.loc[both, "value"].astype(float),
        "end": end.loc[both, "value"].astype(float),
        "start_date": start.loc[both, "end_date"],
        "end_date": end.loc[both, "end_date"],
    })
    valid = rates.start.gt(0) & rates.start.lt(1) & rates.end.gt(0) & rates.end.lt(1)
    rates = rates.loc[valid].copy()
    rates["start_yield"] = [lookup(day) for day in rates.start_date]
    rates["end_yield"] = [lookup(day) for day in rates.end_date]
    rate_change = ((rates.end - rates.start) * 10_000).round(6)
    yield_change = ((rates.end_yield - rates.start_yield) * 100).round(6)
    spread = (rates.end * 10_000 - rates.end_yield * 100).round(6)

    def on_date(item, year, dates):
        frame = _year(item, year)
        frame = frame.loc[frame.index.isin(dates.index)]
        frame = frame.loc[frame.end_date.eq(dates.loc[frame.index])]
        return frame.value.astype(float)

    gain = on_date("gain", END_YEAR, rates.end_date)
    start_obligation = on_date("obligation", START_YEAR, rates.start_date)
    end_obligation = on_date("obligation", END_YEAR, rates.end_date)
    carried = gain.index.intersection(start_obligation.index).intersection(end_obligation.index)
    gain, start_obligation, end_obligation = gain.loc[carried], start_obligation.loc[carried], end_obligation.loc[carried]
    negative = gain.lt(0)
    duration = gain.abs() / start_obligation / (rate_change.loc[carried] / 10_000)

    start_assets = on_date("assets", START_YEAR, rates.start_date)
    end_assets = on_date("assets", END_YEAR, rates.end_date)
    plan_return = on_date("return", END_YEAR, rates.end_date)
    funded = carried.intersection(start_assets.index).intersection(end_assets.index).intersection(plan_return.index)
    start_status = start_assets.loc[funded] - start_obligation.loc[funded]
    end_status = end_assets.loc[funded] - end_obligation.loc[funded]
    losing = plan_return.loc[funded].lt(0)

    return {
        "rate_filer_count": len(both),
        "invalid_rate_count": int((~valid).sum()),
        "median_rate_change_bp": float(rate_change.median()),
        "median_yield_change_bp": float(yield_change.median()),
        "tracking_count": int((rate_change - yield_change).abs().le(TRACKING_BAND_BP).sum()),
        "median_end_rate_pct": float(rates.end.median() * 100),
        "below_treasury_count": int(spread.lt(0).sum()),
        "median_spread_bp": float(spread.median()),
        "gain_filer_count": len(carried),
        "negative_gain_count": int(negative.sum()),
        "negative_gain_obligation_fell_count": int((negative & end_obligation.lt(start_obligation)).sum()),
        "median_implied_duration_years": float(duration.median()),
        "funded_filer_count": len(funded),
        "negative_return_count": int(losing.sum()),
        "negative_return_improved_count": int((losing & end_status.gt(start_status)).sum()),
        "median_start_funded_ratio": float((start_assets.loc[funded] / start_obligation.loc[funded]).median()),
        "median_end_funded_ratio": float((end_assets.loc[funded] / end_obligation.loc[funded]).median()),
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
