"""Compare an interest-expense-to-deposit proxy across size bands and endpoints.

The BankFind field used here is total interest expense, not interest expense on
deposits.  Dividing that annual flow by period-end total deposits produces a
descriptive funding-expense proxy that also reflects nondeposit interest expense
and a stock/flow denominator mismatch.  It is neither a deposit rate nor a
deposit beta and cannot identify funding mix, repricing behavior, or franchise
value.

The case forms four mutually exclusive asset bands at each date: [0, $1bn),
[$1bn, $10bn), [$10bn, $100bn), and [$100bn, infinity).  It requires positive
assets and deposits and nonmissing total interest expense and reported net
interest margin.  Medians are unweighted across institutions, so they do not
describe an asset-weighted industry aggregate.

The final turn divides each band's change in the proxy between 2021-12-31 and
2024-12-31 by the change in single-day EFFR readings.  This is named an endpoint
sensitivity coefficient, not pass-through: annual interest expense does not
share the reference rate's one-day grain, and bands and member banks are formed
separately at each endpoint.  Net interest margin changes are reported alongside
the proxy changes, but the table cannot reconcile the asset-yield, funding-cost,
balance-mix, or cohort-composition drivers of either measure.

The pinned-reading comparison measures the deposit denominator, the band statistic, the
band basis, the policy-rate series and the margin measure.  Dividing by domestic
rather than total deposits raises the largest band's late proxy from 3.083308% to
3.342850%; taking band means instead of medians moves the late median margin from
3.4261% to 3.5772%; banding on equity instead of assets leaves only 4 banks in
the largest band rather than 33; using SOFR instead of EFFR reads the endpoints as
0.05% and 4.49% rather than 0.07% and 4.33%; and computing the margin from the
income statement instead of the reported field gives 3.1270% rather than 3.4261%.
The query pins all five choices.

Numeric validation uses the shared 0.6 x 10^-N rounding-boundary tolerance for
N requested decimals; counts match exactly.
"""

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


EARLY_DATE = "2021-12-31"
LATE_DATE = "2024-12-31"
POLICY_RATE = "EFFR"
BAND_EDGES = (0.0, 1e6, 1e7, 1e8, float("inf"))
BAND_LABELS = ("under_one_billion", "one_to_ten_billion", "ten_to_hundred_billion", "over_hundred_billion")


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = [
    "late_bank_count", "late_median_margin_pct", "late_median_expense_proxy_pct",
    "late_smallest_band_count", "late_largest_band_count",
]
TURN_2_NAMES = [
    "late_smallest_band_margin_pct", "late_largest_band_margin_pct",
    "late_smallest_band_expense_proxy_pct", "late_largest_band_expense_proxy_pct",
]
TURN_3_NAMES = [
    "early_bank_count", "early_smallest_band_expense_proxy_pct", "early_largest_band_expense_proxy_pct",
    "early_policy_rate_pct", "late_policy_rate_pct",
]
TURN_4_NAMES = [
    "smallest_band_endpoint_sensitivity", "second_band_endpoint_sensitivity",
    "third_band_endpoint_sensitivity", "largest_band_endpoint_sensitivity",
    "largest_band_margin_change_pp", "smallest_band_margin_change_pp",
]

variables = [
    _v(TURN_1_NAMES[0], "Store how many banks qualify at the later date as an integer."),
    _v(TURN_1_NAMES[1], "Store the median reported net interest margin at that date in percent, rounded to 4 decimals."),
    _v(TURN_1_NAMES[2], "Store the median total-interest-expense-to-period-end-deposits proxy at that date in percent, rounded to 4 decimals."),
    _v(TURN_1_NAMES[3], "Store how many banks fall in the smallest band at that date as an integer."),
    _v(TURN_1_NAMES[4], "Store how many fall in the largest band at that date as an integer."),
    _v(TURN_2_NAMES[0], "Store the smallest band's median margin at the later date in percent, rounded to 4 decimals."),
    _v(TURN_2_NAMES[1], "Store the largest band's median margin at that date, rounded to 4 decimals."),
    _v(TURN_2_NAMES[2], "Store the smallest band's median expense proxy at that date, rounded to 4 decimals."),
    _v(TURN_2_NAMES[3], "Store the largest band's median expense proxy at that date, rounded to 4 decimals."),
    _v(TURN_3_NAMES[0], "Store how many banks qualify at the earlier date as an integer."),
    _v(TURN_3_NAMES[1], "Store the smallest band's median expense proxy at the earlier date, rounded to 4 decimals."),
    _v(TURN_3_NAMES[2], "Store the largest band's median expense proxy at the earlier date, rounded to 4 decimals."),
    _v(TURN_3_NAMES[3], "Store the policy rate on the earlier date in percent, rounded to 4 decimals."),
    _v(TURN_3_NAMES[4], "Store the policy rate on the later date in percent, rounded to 4 decimals."),
    _v(TURN_4_NAMES[0], "Store the smallest band's endpoint sensitivity coefficient, rounded to 4 decimals."),
    _v(TURN_4_NAMES[1], "Store the second band's endpoint sensitivity coefficient, rounded to 4 decimals."),
    _v(TURN_4_NAMES[2], "Store the third band's endpoint sensitivity coefficient, rounded to 4 decimals."),
    _v(TURN_4_NAMES[3], "Store the largest band's endpoint sensitivity coefficient, rounded to 4 decimals."),
    _v(TURN_4_NAMES[4],
       "Store the largest band's change in median margin in percentage points, rounded to 4 "
       "decimals. Compute from the two unrounded endpoint medians, not from the rounded figures "
       "reported for them."),
    _v(TURN_4_NAMES[5],
       "Store the smallest band's change in median margin in percentage points, rounded to 4 "
       "decimals, on the same unrounded basis."),
]

DECIMALS = [
    0, 4, 4, 0, 0,
    4, 4, 4, 4,
    0, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4,
]


@lru_cache(maxsize=2)
def _banded(report_date):
    frame = load_expansion_table("fdic_bankfind").copy()
    frame = frame.loc[frame.REPDTE.astype(str).eq(report_date)].copy()
    for column in ("NIMY", "DEP", "ASSET", "EINTEXP"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[
        frame.DEP.gt(0) & frame.ASSET.gt(0) & frame.EINTEXP.notna() & frame.NIMY.notna()
    ].copy()
    frame["expense_proxy"] = frame.EINTEXP / frame.DEP * 100
    frame["band"] = pd.cut(frame.ASSET, list(BAND_EDGES), labels=list(BAND_LABELS), right=False)
    return frame


def _band_median(frame, band, column):
    rows = frame.loc[frame.band.astype(str).eq(band), column]
    if rows.empty:
        raise ValueError(f"no banks in band {band}")
    return float(rows.median())


def _policy_rate(report_date):
    rates = load_expansion_table("nyfed_reference_rates").copy()
    cell = rates.loc[
        rates["Effective Date"].astype(str).eq(report_date)
        & rates["Rate Type"].astype(str).eq(POLICY_RATE),
        "Rate (%)",
    ]
    if cell.empty:
        raise ValueError(f"no policy rate published for {report_date}")
    return float(pd.to_numeric(cell, errors="coerce").iloc[0])


@lru_cache(maxsize=1)
def ground_truth():
    early, late = _banded(EARLY_DATE), _banded(LATE_DATE)
    early_rate, late_rate = _policy_rate(EARLY_DATE), _policy_rate(LATE_DATE)
    rate_change = late_rate - early_rate

    late_cost = {band: _band_median(late, band, "expense_proxy") for band in BAND_LABELS}
    early_cost = {band: _band_median(early, band, "expense_proxy") for band in BAND_LABELS}
    late_margin = {band: _band_median(late, band, "NIMY") for band in BAND_LABELS}
    early_margin = {band: _band_median(early, band, "NIMY") for band in BAND_LABELS}
    betas = {band: (late_cost[band] - early_cost[band]) / rate_change for band in BAND_LABELS}

    smallest, largest = BAND_LABELS[0], BAND_LABELS[-1]
    return (
        int(len(late)), float(late.NIMY.median()), float(late.expense_proxy.median()),
        int(late.band.astype(str).eq(smallest).sum()), int(late.band.astype(str).eq(largest).sum()),
        late_margin[smallest], late_margin[largest], late_cost[smallest], late_cost[largest],
        int(len(early)), early_cost[smallest], early_cost[largest],
        early_rate, late_rate,
        betas[BAND_LABELS[0]], betas[BAND_LABELS[1]], betas[BAND_LABELS[2]], betas[BAND_LABELS[3]],
        late_margin[largest] - early_margin[largest], late_margin[smallest] - early_margin[smallest],
    )


def _validate_subset(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip(by_name, ground_truth()))
    decimals = dict(zip(by_name, DECIMALS))
    return validate_ordered_outputs(
        outputs, [by_name[name] for name in names], [truth[name] for name in names],
        [decimals[name] for name in names],
    )


def validate_turn_1(outputs): return _validate_subset(outputs, TURN_1_NAMES)
def validate_turn_2(outputs): return _validate_subset(outputs, TURN_2_NAMES)
def validate_turn_3(outputs): return _validate_subset(outputs, TURN_3_NAMES)
def validate_turn_4(outputs): return _validate_subset(outputs, TURN_4_NAMES)
def validate(outputs): return _validate_subset(outputs, [variable.name for variable in variables])


validators = {
    "validate_cross_section": turn_validator(validate_turn_1),
    "validate_size_gradient": turn_validator(validate_turn_2),
    "validate_cycle_endpoints": turn_validator(validate_turn_3),
    "validate_pass_through": turn_validator(validate_turn_4),
}
