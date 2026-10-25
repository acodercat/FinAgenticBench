"""Measure share-class gross-to-net yield wedges within money market series.

A money market fund reports a seven-day gross yield at the series level and a
seven-day net yield for each share class. The runtime class-yield table pairs the
series-level gross figure with each class-level net figure. At the 2025-06-30
yield date the file carries
1,062 class rows across 308 series; 183 series report more than one class, and in
all 183 the gross yield is identical across every class the series reports. The
case defines the difference between the paired gross and net fields as a wedge;
it does not treat that constructed wedge as a stated expense ratio.

The wedge is not small and is not evenly borne. On the 1,008 class rows that
survive the stated anomaly screen, the median class gives up 33.0000 basis
points and the mean 39.5427. Weighting instead by the assets in each class, a
dollar gives up 21.4018 basis points. The gap between 39.5427 and 21.4018 is the
whole point: higher-asset classes carry lower wedges in this observed population,
so the average class and the average dollar are not answering the same question.
The source does not by itself establish an investor-type or causal explanation.
The widest class pays 234.0000 basis points,
and within Allspring Money Market Fund alone the cheapest and dearest classes sit
116.0000 basis points apart on an identical portfolio.

The screen matters because the field is not always populated. At the same date 29
rows report a net yield of exactly zero against a positive gross yield, and 15 of
those belong to a class holding no assets at all; across the file 274 rows report a
net yield above the gross yield. That ordering is inconsistent with treating the
paired fields as gross-before-expenses and net-after-expenses, so the case screens
the rows as anomalies. The screen does not establish whether the source filing,
field matching, rounding or another data issue caused any individual observation.

Coverage is the other thing that has to be checked before any change over time is
computed. Of the 11,931 yield rows dated before 2025, 11,909, or 99.8156 percent,
come from amended filings; the original filings begin only at the end of 2024. In
June 2024 just 79 share classes appear against 1,029 in June 2025, so the early
window is a sample of funds that later amended rather than a picture of the
industry, and an unqualified industry-wide difference against the later window is
not supported. A comparison restricted to an explicitly common observed sample
would answer a different question. Within the available 2025 table, the
asset-weighted gross yield falls from 4.4327 percent in
January to 4.0241 in November while the three-month Treasury bill averages fall
from 4.3429 to 3.9372: declines of 40.86 and 40.57 basis points, with fund yields
sitting a little above bills at both ends.

Measured rejected alternatives, from the `money_market_fee_wedge` comparison, include retaining anomalous rows, substituting the equal-class mean for the
asset-weighted result, using the net yield alone instead of the defined difference,
and treating the observed June 2024 and June 2025 populations as comparable
industry snapshots. The query explicitly requests the relevant screen, both
weighting views and the coverage diagnostic, so these are instruction-following
probes rather than a live hidden-method trap. This is a hard baseline case.

Knowledge the case assumes and does not supply: that Form N-MFP reports the gross
yield at the series level and the net yield at the class level, that an average
over classes and an average over dollars answer different questions, and that an
amended filing is not an independent fund observation.

Boundaries: a seven-day yield is an annualized snapshot rather than a realized
return, and the wedge inferred from it is a yield reduction rather than a stated
expense ratio, which may differ where a fee waiver applies. The 2025-06-30 date is
one date, and class composition changes across the year. The bill comparison uses
a simple mean of daily quotes over each calendar month.

Numeric validation uses the shared 0.6 x 10^-N rounding-boundary tolerance for
N requested decimals; counts and fund names match exactly.
"""

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


YIELD_DATE = "2025-06-30"
EARLY_MONTH, LATE_MONTH = "2024-06", "2025-06"
COMPARISON_MONTHS = ("2025-01", "2025-11")
COVERAGE_CUTOFF = "2025-01-01"
BILL_COLUMN = "3 Mo"


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = ["class_rows", "series_count", "multi_class_series", "identical_gross_series", "median_classes_per_series"]
TURN_2_NAMES = ["zero_net_rows", "zero_net_empty_class_rows", "net_above_gross_rows", "clean_rows", "clean_series"]
TURN_3_NAMES = ["median_wedge_bp", "mean_wedge_bp", "asset_weighted_wedge_bp", "widest_class_wedge_bp",
                "widest_series_name", "widest_series_spread_bp"]
TURN_4_NAMES = ["early_month_classes", "late_month_classes", "pre_2025_rows", "amended_pre_2025_share_pct",
                "january_gross_yield_pct", "november_gross_yield_pct", "january_bill_pct", "november_bill_pct"]

variables = [
    _v(TURN_1_NAMES[0], "Store how many class rows the yield date carries as an integer."),
    _v(TURN_1_NAMES[1], "Store how many distinct series they belong to as an integer."),
    _v(TURN_1_NAMES[2], "Store how many series report more than one class as an integer."),
    _v(TURN_1_NAMES[3], "Store how many of those report one identical gross yield across their classes as an integer."),
    _v(TURN_1_NAMES[4], "Store the median number of classes per series, rounded to 4 decimals."),
    _v(TURN_2_NAMES[0], "Store how many rows at the date report a zero net yield against a positive gross yield as an integer."),
    _v(TURN_2_NAMES[1], "Store how many of those belong to a class holding no assets as an integer."),
    _v(TURN_2_NAMES[2], "Store how many rows in the whole table report a net yield above the gross yield as an integer."),
    _v(TURN_2_NAMES[3], "Store how many class rows survive the screen as an integer."),
    _v(TURN_2_NAMES[4], "Store how many distinct series they cover as an integer."),
    _v(TURN_3_NAMES[0], "Store the median wedge in basis points, rounded to 4 decimals."),
    _v(TURN_3_NAMES[1], "Store the mean wedge across classes in basis points, rounded to 4 decimals."),
    _v(TURN_3_NAMES[2], "Store the asset-weighted wedge in basis points, rounded to 4 decimals."),
    _v(TURN_3_NAMES[3], "Store the largest wedge any single class bears in basis points, rounded to 4 decimals."),
    _v(TURN_3_NAMES[4], "Store the name of the series whose classes are furthest apart as text."),
    _v(TURN_3_NAMES[5], "Store that spread in basis points, rounded to 4 decimals."),
    _v(TURN_4_NAMES[0], "Store how many distinct share classes report in the early month as an integer."),
    _v(TURN_4_NAMES[1], "Store how many report in the late month as an integer."),
    _v(TURN_4_NAMES[2], "Store how many yield rows are dated before 2025 as an integer."),
    _v(TURN_4_NAMES[3], "Store the percent of those that come from an amended filing, rounded to 4 decimals."),
    _v(TURN_4_NAMES[4], "Store the asset-weighted gross yield for January 2025 in percent, rounded to 4 decimals."),
    _v(TURN_4_NAMES[5], "Store the same for November 2025, rounded to 4 decimals."),
    _v(TURN_4_NAMES[6], "Store the mean three-month bill yield for January 2025 in percent, rounded to 4 decimals."),
    _v(TURN_4_NAMES[7], "Store the same for November 2025, rounded to 4 decimals."),
]

DECIMALS = [0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 4, 4, 4, 4, None, 4, 0, 0, 0, 4, 4, 4, 4, 4]


def _normalized_name(value):
    return " ".join(str(value).split()).casefold() if isinstance(value, str) else value


@lru_cache(maxsize=1)
def _yields():
    frame = load_expansion_table("sec_nmfp_class_yields").copy()
    for column in ("seven_day_net_yield", "seven_day_gross_yield", "class_net_assets_usd"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["day"] = pd.to_datetime(frame.yield_date, errors="coerce")
    frame["month"] = frame.day.dt.to_period("M").astype(str)
    series = load_expansion_table("sec_nmfp")[["accession", "series_id", "series_name", "form"]]
    return frame.merge(series, on="accession", how="left")


def _screened(frame):
    return frame.loc[
        frame.class_net_assets_usd.gt(0)
        & frame.seven_day_gross_yield.gt(0)
        & frame.seven_day_net_yield.gt(0)
        & frame.seven_day_net_yield.le(frame.seven_day_gross_yield)
    ].copy()


def _weighted_gross(frame):
    return float((frame.seven_day_gross_yield * frame.class_net_assets_usd).sum() / frame.class_net_assets_usd.sum()) * 100


@lru_cache(maxsize=1)
def ground_truth():
    frame = _yields()
    dated = frame.loc[frame.yield_date.astype(str).eq(YIELD_DATE) & frame.series_id.notna()].copy()
    grouped = dated.groupby("series_id").agg(
        classes=("class_id", "nunique"),
        gross_spread=("seven_day_gross_yield", lambda values: values.max() - values.min()),
    )
    multi = grouped.loc[grouped.classes.gt(1)]

    defective = dated.seven_day_net_yield.eq(0) & dated.seven_day_gross_yield.gt(0)
    clean = _screened(dated)
    clean["wedge"] = (clean.seven_day_gross_yield - clean.seven_day_net_yield) * 10000
    weighted = float((clean.wedge * clean.class_net_assets_usd).sum() / clean.class_net_assets_usd.sum())
    by_series = clean.groupby(["series_id", "series_name"]).wedge.agg(["min", "max", "count"])
    by_series = by_series.loc[by_series["count"].gt(1)]
    by_series["spread"] = by_series["max"] - by_series["min"]
    widest = by_series.sort_values(["spread", "series_name"], ascending=[False, True]).reset_index().iloc[0]

    early = frame.loc[frame.month.eq(EARLY_MONTH)]
    late = frame.loc[frame.month.eq(LATE_MONTH)]
    pre = frame.loc[frame.day.lt(COVERAGE_CUTOFF)]
    amended = pre.form.astype(str).str.endswith("/A")

    monthly = _screened(frame)
    january, november = (monthly.loc[monthly.month.eq(month)] for month in COMPARISON_MONTHS)

    curve = load_expansion_table("treasury_yield_curve").copy()
    curve["month"] = pd.to_datetime(curve.Date, errors="coerce").dt.to_period("M").astype(str)
    curve["bill"] = pd.to_numeric(curve[BILL_COLUMN], errors="coerce")
    bills = [float(curve.loc[curve.month.eq(month), "bill"].mean()) for month in COMPARISON_MONTHS]

    return (
        int(len(dated)), int(dated.series_id.nunique()), int(len(multi)),
        int(multi.gross_spread.abs().lt(1e-9).sum()), float(grouped.classes.median()),
        int(defective.sum()), int((defective & dated.class_net_assets_usd.eq(0)).sum()),
        int(frame.seven_day_net_yield.gt(frame.seven_day_gross_yield).sum()),
        int(len(clean)), int(clean.series_id.nunique()),
        float(clean.wedge.median()), float(clean.wedge.mean()), weighted, float(clean.wedge.max()),
        str(widest.series_name), float(widest.spread),
        int(early.class_id.nunique()), int(late.class_id.nunique()), int(len(pre)),
        float(amended.mean()) * 100,
        _weighted_gross(january), _weighted_gross(november), bills[0], bills[1],
    )


NAME_OUTPUTS = ("widest_series_name",)


def _validate_subset(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip(by_name, ground_truth()))
    decimals = dict(zip(by_name, DECIMALS))
    normalized = dict(outputs)
    for name in NAME_OUTPUTS:
        if name in normalized:
            normalized[name] = _normalized_name(normalized[name])
            truth[name] = _normalized_name(truth[name])
    return validate_ordered_outputs(
        normalized, [by_name[name] for name in names], [truth[name] for name in names],
        [decimals[name] for name in names],
    )


def validate_turn_1(outputs): return _validate_subset(outputs, TURN_1_NAMES)
def validate_turn_2(outputs): return _validate_subset(outputs, TURN_2_NAMES)
def validate_turn_3(outputs): return _validate_subset(outputs, TURN_3_NAMES)
def validate_turn_4(outputs): return _validate_subset(outputs, TURN_4_NAMES)
def validate(outputs): return _validate_subset(outputs, [variable.name for variable in variables])


validators = {
    "validate_class_structure": turn_validator(validate_turn_1),
    "validate_reporting_screen": turn_validator(validate_turn_2),
    "validate_fee_wedge": turn_validator(validate_turn_3),
    "validate_coverage_and_passthrough": turn_validator(validate_turn_4),
}
