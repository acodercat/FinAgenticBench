"""An operating lease liability is a present value at rates fixed in the past.

Under the 2016 lease standard a lessee discounts each operating lease's
remaining payments at the rate implicit in the lease or, usually, its
incremental borrowing rate, both set when the lease begins. Existing leases
are not remeasured when market rates move, so the weighted-average discount
rate a filer reports is an average of rates fixed over many years.

Among the large filers' 2024 annual reports, 372 tag the undiscounted lease
payments, the amount representing interest and the operating lease liability
at the same date, and 359 reconcile exactly: payments less interest equal the
liability. One filer tags its interest as a negative amount. The median
interest is 17.0856 percent of undiscounted payments.

Of the 339 filers that reconcile and tag a discount rate, 166 report a
weighted rate below the five-year Treasury par yield on their balance date;
in 2021 only one of them did. Between 2021 and 2024 the filers' rates rose a
median 100 basis points while the five-year yield on their balance dates rose
312. A rate below the Treasury yield cannot be what the company would pay to
borrow today; it is what it paid when its leases began. 2022 is the first year
in which more than half of these filers stood below the yield.

The right-of-use asset starts from the liability and is adjusted for prepaid
and accrued rent, lease incentives, initial direct costs and impairment. 281
of the 328 filers tagging it carry an asset below the liability, at a median
0.9348 of it.

Readings measured so they are not mistaken for oversights:

1. Every value is taken as first reported: from the earliest-filed annual
   report carrying it at that date. 48 discount-rate values differ between the
   first and the latest report carrying them. Reading the latest report gives
   358 exact reconciliations, 338 rate filers at a median 4.31 percent, 165
   below the yield, 329 paired filers, and 280 of 327 assets below the
   liability.
2. Each figure keeps the sign it is tagged with. Reading the negative interest
   amount as its absolute value lets that filer reconcile, giving 360 rather
   than 359 exact reconciliations.
3. The balance date in a calendar year is the latest date in that year at
   which an annual report tags the item. In 2019, 45 filers tag the liability
   at two dates; for 42 of them the earlier is 2019-01-01, the opening balance
   on adopting the standard, and the later is the fiscal year end.
4. The comparison yield is the five-year par yield published on the balance
   date or, where none was published that day, the latest earlier date. On
   the ten-year point 2024 has 199 rather than 166 filers below.
5. Two rates are tagged on the wrong scale: 32000 for one filer at 2019 and
   3.1 for another at 2021, both as first reported. Leaving them out or
   dividing them by 100 moves no output. One 2022 rate equals the yield on its
   balance date; reading "below" as "at or below" would not change the first
   majority year.
6. The median interest share is taken across all 372 filers tagging the
   schedule; across only the 359 that reconcile it is 17.0735. Reading each
   item at the latest 2024 date where all three are tagged together, or the
   rate and asset at their own latest 2024 dates, gives the same populations.
"""

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table, load_facts
from core.validation import turn_validator, validate_ordered_outputs


CONCEPTS = {
    "LesseeOperatingLeaseLiabilityPaymentsDue": "payments",
    "LesseeOperatingLeaseLiabilityUndiscountedExcessAmount": "interest",
    "OperatingLeaseLiability": "liability",
    "OperatingLeaseWeightedAverageDiscountRatePercent": "rate",
    "OperatingLeaseRightOfUseAsset": "asset",
}
FOCUS_YEAR = 2024
PATH_YEARS = range(2019, 2025)
BASE_YEAR = 2021
TENOR = "5 Yr"

OUTPUTS = (
    ("schedule_filer_count", 0, "the count of filers tagging all three schedule figures as an integer"),
    ("schedule_exact_count", 0, "how many of them reconcile exactly, as an integer"),
    ("negative_interest_count", 0, "how many tag the interest amount below zero, as an integer"),
    ("median_interest_share_pct", 4, "the median interest amount as a percent of undiscounted payments"),
    ("rate_filer_count", 0, "the count of reconciling filers tagging a discount rate as an integer"),
    ("median_discount_rate_pct", 2, "their median weighted-average discount rate in percent"),
    ("below_treasury_count", 0, "how many rates stand below the Treasury yield, as an integer"),
    ("median_spread_to_treasury_bp", 2, "the median rate less the Treasury yield in basis points"),
    ("first_majority_below_year", 0, "the earliest year in which more than half stand below the yield, as an integer"),
    ("below_treasury_2021_count", 0, "how many stood below the yield in 2021, as an integer"),
    ("paired_filer_count", 0, "the count of filers with a rate in both 2021 and 2024 as an integer"),
    ("median_rate_change_bp", 2, "the median change in their rate in basis points"),
    ("median_yield_change_bp", 2, "the median change in the Treasury yield on their balance dates in basis points"),
    ("rate_kept_pace_count", 0, "how many rates rose by at least the yield change, as an integer"),
    ("asset_filer_count", 0, "the count of filers tagging the right-of-use asset as an integer"),
    ("asset_below_liability_count", 0, "how many carry an asset below the liability, as an integer"),
    ("median_asset_to_liability", 4, "the median asset divided by the liability"),
)


def _description(precision, text):
    if precision == 0:
        return f"Store {text}."
    return f"Store {text}, rounded to {precision} decimals from unrounded source rows."


variables = [Variable(name, None, _description(p, text)) for name, p, text in OUTPUTS]
PRECISION = {name: p for name, p, _ in OUTPUTS}
TURNS = (
    tuple(n for n, _, _ in OUTPUTS[:4]),
    tuple(n for n, _, _ in OUTPUTS[4:8]),
    tuple(n for n, _, _ in OUTPUTS[8:14]),
    tuple(n for n, _, _ in OUTPUTS[14:]),
)


@lru_cache(maxsize=1)
def _balances():
    """First-reported 10-K instants per filer, item and calendar year."""
    facts = load_facts()
    rows = facts.loc[
        facts.concept.astype(str).isin(list(CONCEPTS))
        & facts.form.astype(str).eq("10-K")
        & facts.start_date.isna()
        & facts.value.notna()
    ].copy()
    rows["item"] = rows.concept.astype(str).map(CONCEPTS)
    rows = rows.sort_values(["filed_date", "accession"]).drop_duplicates(["cik", "item", "end_date"], keep="first")
    rows["year"] = rows.end_date.str[:4].astype(int)
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
    rows = _balances()
    return rows.loc[rows.item.eq(item) & rows.year.eq(year)].set_index("cik")


def _rates_against_yield(year, ciks):
    rates = _year("rate", year)
    rates = rates.loc[rates.index.isin(ciks)].copy()
    lookup = _yield_on()
    rates["rate_pct"] = (rates.value.astype(float) * 100).round(6)
    rates["yield_pct"] = [lookup(day) for day in rates.end_date]
    return rates


@lru_cache(maxsize=1)
def analysis():
    focus = {item: _year(item, FOCUS_YEAR) for item in ("payments", "interest", "liability", "asset")}
    schedule = pd.concat(
        {item: focus[item][["end_date", "value"]] for item in ("payments", "interest", "liability")}, axis=1
    ).dropna()
    schedule = schedule.loc[
        schedule[("payments", "end_date")].eq(schedule[("liability", "end_date")])
        & schedule[("interest", "end_date")].eq(schedule[("liability", "end_date")])
    ]
    payments = schedule[("payments", "value")].astype(float)
    interest = schedule[("interest", "value")].astype(float)
    liability = schedule[("liability", "value")].astype(float)
    exact = (payments - interest).eq(liability)
    reconciled = schedule.index[exact]

    rates = _rates_against_yield(FOCUS_YEAR, reconciled)
    rates = rates.loc[rates.end_date.eq(schedule.loc[rates.index, ("liability", "end_date")])]
    carried = rates.index
    spread_bp = (rates.rate_pct - rates.yield_pct) * 100

    path = {year: _rates_against_yield(year, carried) for year in PATH_YEARS}
    majority = [year for year, frame in path.items() if (frame.rate_pct < frame.yield_pct).sum() * 2 > len(frame)]
    early, late = path[BASE_YEAR], path[FOCUS_YEAR]
    paired = early.index.intersection(late.index)
    rate_change = (late.loc[paired, "rate_pct"] - early.loc[paired, "rate_pct"]) * 100
    yield_change = (late.loc[paired, "yield_pct"] - early.loc[paired, "yield_pct"]) * 100

    asset = focus["asset"]
    asset = asset.loc[asset.index.isin(carried)]
    asset = asset.loc[asset.end_date.eq(rates.loc[asset.index, "end_date"])]
    ratio = asset.value.astype(float) / liability.loc[asset.index]

    return {
        "schedule_filer_count": len(schedule),
        "schedule_exact_count": int(exact.sum()),
        "negative_interest_count": int((interest < 0).sum()),
        "median_interest_share_pct": float((interest / payments * 100).median()),
        "rate_filer_count": len(rates),
        "median_discount_rate_pct": float(rates.rate_pct.median()),
        "below_treasury_count": int((rates.rate_pct < rates.yield_pct).sum()),
        "median_spread_to_treasury_bp": float(spread_bp.median()),
        "first_majority_below_year": int(min(majority)),
        "below_treasury_2021_count": int((early.rate_pct < early.yield_pct).sum()),
        "paired_filer_count": len(paired),
        "median_rate_change_bp": float(rate_change.median()),
        "median_yield_change_bp": float(yield_change.median()),
        "rate_kept_pace_count": int((rate_change >= yield_change).sum()),
        "asset_filer_count": len(asset),
        "asset_below_liability_count": int((ratio < 1).sum()),
        "median_asset_to_liability": float(ratio.median()),
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
