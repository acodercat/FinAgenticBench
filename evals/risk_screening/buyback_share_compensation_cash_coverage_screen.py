"""Compare repurchases, stock compensation and subsequent share-count outcomes.

Operating cash flow adds back share-based compensation because it is noncash.
That does not make the resulting cash unavailable for distribution; the cost to
existing owners can instead arrive through dilution. Operating cash flow less
reported PP&E purchases includes growth as well as maintenance investment and
does not by itself measure sustainable distributable cash.
Across the 1,251 annual filings in this window, share-based compensation is
176.353726 USD billions against 1585.787171 of operating cash flow, a median
8.1852% of a filer's own operating cash flow.

The case turns on an association that can be checked. Of the
1,039 filers reporting a repurchase, 352 repurchase less than they expensed in
stock compensation. Among the 989 with
a prior-year annual filing to compare against, 72.5076% of the ones repurchasing
below their compensation saw weighted-average basic shares rise, against
19.7568% of the ones repurchasing at or above it. The first group's median
share count rose 0.7811%; the second group's fell.

Three conventions the data forces:

1. ``PaymentsToAcquirePropertyPlantAndEquipment``, ``PaymentsForRepurchaseOf
   CommonStock`` and ``ShareBasedCompensation`` are outflow and expense
   concepts can carry unexpected negative values. A reversal, recovery,
   presentation choice or data fault may explain one; these data do not
   distinguish them. Eight of the 1,259 filers tagging all four are dropped
   by the stated sign screen rather than sign-flipped: 1 capital expenditure, 3 repurchases and
   5 compensation amounts are negative, nine flags across eight filers because
   one trips two. Reading them as positives would substitute unreported amounts.
2. The prior-year filing is the latest original annual report 300 to 430 days
   before the current one, found by that distance rather than by a calendar
   year: three 52/53-week filers closed their preceding year on 2024-01-01 or
   2024-01-02, 364 days earlier, and a search of 2023 period ends finds no
   prior year for them. The window keeps out the one filer whose nearest
   earlier report sits 731 days back, so a year-over-year share count stays a
   year.
3. ``PaymentsOfDividends`` carries no total on its own. Only 825 of the 5,524
   annual filings tag it, most filers using a variant concept this table does
   not carry, so the dividends are taken from the separate fact file instead:
   of the 122 eligible filers recording a selected annual dividend fact there,
   including zero-valued facts, the single
   concept would find 49. Two conventions hold that total together. An
   aggregate supersedes its own components, without which one filer's payment
   is counted twice and the total reads 291.380830 rather than 281.566830 USD
   billions; and a year is taken as first filed, one filer otherwise carrying
   782,000,000 restated a year later in place of the 782,048,000 it first
   reported. The aggregate and common/ordinary/preferred fallback do not
   have a uniform recipient scope; this is a selected-concept screen rather
   than a homogeneous parent-shareholder payout total.

Both medians are taken with linear interpolation, which the query states. It
binds for the compensation share, whose group has an even count with two
differing middle values: 8.1311 or 8.2392 against the reported 8.1852 under the
lower or the upper middle value. The below-compensation group has an odd count,
331, so its median share-count change is one filer's and no convention moves it.

Every selection above changes a reported number. The one that does not is the
guard requiring a positive prior share count, which no filer trips; it stays in
the code because it is a divisor and is left out of the query because a solver
would spend a step on a condition that never binds.

The repurchase total is what filers tagged, not an audited figure. 177 of the
repurchasers spend more than their own operating cash flow, which can reflect
financing, other cash sources or a tagging mismatch: NetApp reports
15,600,000,000 against operating cash flow of 1,685,000,000, and its series
across 2019 to 2025 runs 11.7, 13.1, 13.3, 13.9, 0.85, 15.6 and 16.8 billions,
a cumulative program total sitting in an annual field in every year but one.
The screen is reported for what it is and no threshold is applied to repair it.
"""

from decimal import Decimal
from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_facts, load_financial_statements
from core.validation import turn_validator, validate_ordered_outputs


WINDOW = ("2024-01-01", "2024-12-31")
PAIR_GAP_DAYS = (300, 430)
BILLION = Decimal(10) ** 9

OCF = "NetCashProvidedByUsedInOperatingActivities_ytd"
CAPEX = "PaymentsToAcquirePropertyPlantAndEquipment_ytd"
REPURCHASE = "PaymentsForRepurchaseOfCommonStock_ytd"
COMPENSATION = "ShareBasedCompensation_ytd"
BASIC_SHARES = "WeightedAverageNumberOfSharesOutstandingBasic_ytd"
NON_NEGATIVE = (CAPEX, REPURCHASE, COMPENSATION)
DIVIDEND_AGGREGATE = "PaymentsOfDividends"
DIVIDEND_COMPONENTS = (
    "PaymentsOfDividendsCommonStock",
    "PaymentsOfOrdinaryDividends",
    "PaymentsOfDividendsPreferredStockAndPreferenceStock",
)
ANNUAL_FACT_DAYS = (300, 430)

OUTPUTS = (
    ("eligible_filer_count", 0, "the count of filers kept after the negative-value exclusion, as an integer"),
    ("negative_value_excluded_filer_count", 0, "the count dropped for a negative outflow or expense, as an integer"),
    ("repurchasing_filer_count", 0, "the count reporting a repurchase above zero as an integer"),
    ("operating_cash_flow_bn", 6, "the eligible filers' operating cash flow summed, in USD billions"),
    ("free_cash_flow_bn", 6, "their operating cash flow after capital expenditure, summed, in USD billions"),
    ("negative_free_cash_flow_filers", 0, "how many of them have a negative such figure, as an integer"),
    ("repurchase_bn", 6, "their repurchases summed, in USD billions"),
    ("repurchase_above_fcf_filers", 0, "how many repurchasers spend more on it than they generate after capital expenditure, as an integer"),
    ("share_based_compensation_bn", 6, "their share-based compensation summed, in USD billions"),
    ("compensation_to_operating_cash_flow_median_pct", 4, "the median of that expense as a percentage of the filer's own operating cash flow"),
    ("repurchase_below_compensation_filers", 0, "how many repurchasers spend less than they expensed in stock, as an integer"),
    ("repurchase_above_operating_cash_flow_filers", 0, "how many repurchasers spend more on it than their operating cash flow, as an integer"),
    ("comparable_filer_count", 0, "the count of repurchasers with a comparable prior year as an integer"),
    ("below_compensation_share_rise_pct", 4, "the percentage of the below-compensation group whose share count rose"),
    ("above_compensation_share_rise_pct", 4, "the percentage of the remaining group whose share count rose"),
    ("below_compensation_median_share_change_pct", 4, "the below-compensation group's median share-count change in percent"),
    ("dividend_reporting_filer_count", 0, "the count of eligible filers with a selected dividend fact, including zero-valued facts, as an integer"),
    ("single_concept_reporting_filer_count", 0, "how many of those report the first table's dividend concept, as an integer"),
    ("aggregate_and_component_filer_count", 0, "how many report both an aggregate and a component, as an integer"),
    ("dividend_bn", 6, "the selected dividend facts summed, in USD billions"),
)
variables = [
    Variable(name, None, f"Store {text}" + (f", rounded to {p} decimals from unrounded source rows." if p is not None else "."))
    for name, p, text in OUTPUTS
]
PRECISION = {name: p for name, p, _ in OUTPUTS}
TURNS = (
    tuple(n for n, _, _ in OUTPUTS[:4]),
    tuple(n for n, _, _ in OUTPUTS[4:8]),
    tuple(n for n, _, _ in OUTPUTS[8:12]),
    tuple(n for n, _, _ in OUTPUTS[12:16]),
    tuple(n for n, _, _ in OUTPUTS[16:]),
)


def _annual(window):
    """One annual filing per filer: latest period end, then filed date, then accession."""
    statements = load_financial_statements()
    frame = statements.loc[
        statements.form.eq("10-K") & statements.fiscal_period.eq("FY")
    ].copy()
    frame["period_end"] = frame.report_date.astype(str)
    frame = frame.loc[frame.period_end.between(*window)]
    if frame.empty:
        raise ValueError(f"no annual filing has a period end in {window}")
    ordered = frame.sort_values(["cik", "period_end", "filed_date", "accession"])
    return ordered.drop_duplicates("cik", keep="last")


@lru_cache(maxsize=1)
def _annual_reports():
    """Every original full-year 10-K, one per filer and period end."""
    statements = load_financial_statements()
    frame = statements.loc[
        statements.form.eq("10-K") & statements.fiscal_period.eq("FY")
    ].copy()
    frame["period_end"] = frame.report_date.astype(str)
    ordered = frame.sort_values(["cik", "period_end", "filed_date", "accession"])
    return ordered.drop_duplicates(["cik", "period_end"], keep="last")


@lru_cache(maxsize=1)
def _tagged():
    return _annual(WINDOW).dropna(subset=[OCF, CAPEX, REPURCHASE, COMPENSATION])


@lru_cache(maxsize=1)
def _eligible():
    tagged = _tagged()
    keep = pd.Series(True, index=tagged.index)
    for column in NON_NEGATIVE:
        keep &= tagged[column] >= 0
    frame = tagged.loc[keep].copy()
    if frame.empty:
        raise ValueError("no filer survives the sign screen")
    frame["free_cash_flow"] = frame[OCF] - frame[CAPEX]
    return frame


@lru_cache(maxsize=1)
def _comparable():
    """Repurchasers whose prior annual filing supports a year-over-year count."""
    current = _eligible()
    current = current.loc[current[REPURCHASE] > 0]
    # The prior year is found by its distance, not by a calendar year: a 52/53-week
    # filer's preceding year can end in the first days of the current one.
    prior = _annual_reports()[["cik", "period_end", BASIC_SHARES]].rename(
        columns={"period_end": "prior_period_end", BASIC_SHARES: "prior_basic_shares"}
    )
    frame = current.merge(prior, on="cik")
    frame["gap_days"] = (
        pd.to_datetime(frame.period_end) - pd.to_datetime(frame.prior_period_end)
    ).dt.days
    frame = frame.loc[frame.gap_days.between(*PAIR_GAP_DAYS)]
    # The preceding fiscal year is the latest report in that window.
    frame = frame.sort_values(["cik", "prior_period_end"]).drop_duplicates("cik", keep="last")
    frame = frame.dropna(subset=[BASIC_SHARES, "prior_basic_shares"])
    frame = frame.loc[frame.prior_basic_shares > 0]
    if frame.empty:
        raise ValueError("no repurchaser has a comparable prior year")
    frame["share_change_pct"] = (frame[BASIC_SHARES] / frame.prior_basic_shares - 1) * 100
    return frame


@lru_cache(maxsize=1)
def _dividend_facts():
    """One dividend fact per filer and concept, as that year was first filed."""
    eligible = _eligible()[["cik", "period_end"]].copy()
    eligible["cik"] = eligible.cik.astype(str)

    facts = load_facts()
    wanted = (DIVIDEND_AGGREGATE,) + DIVIDEND_COMPONENTS
    facts = facts.loc[
        facts.concept.astype(str).isin(wanted) & facts.form.eq("10-K") & facts.unit.eq("USD")
    ].copy()
    facts["cik"] = facts.cik.astype(str)
    facts["concept"] = facts.concept.astype(str)
    facts["fact_end"] = facts.end_date.astype(str)

    matched = eligible.merge(
        facts[["cik", "fact_end", "concept", "value", "accession", "filed_date", "start_date"]],
        left_on=["cik", "period_end"], right_on=["cik", "fact_end"],
    )
    matched["fact_days"] = (
        pd.to_datetime(matched.fact_end) - pd.to_datetime(matched.start_date.astype(str))
    ).dt.days
    matched = matched.loc[matched.fact_days.between(*ANNUAL_FACT_DAYS)]
    if matched.empty:
        raise ValueError("no eligible filer carries an annual dividend fact")
    # A later annual report restates the prior year, so the same concept can
    # appear twice at one period end; the year as first filed is taken.
    ordered = matched.sort_values(["cik", "concept", "filed_date", "accession"])
    return ordered.drop_duplicates(["cik", "concept"], keep="first")


def _dividend_total(facts):
    """An aggregate supersedes its own components rather than adding to them."""
    total = Decimal(0)
    for _, rows in facts.groupby("cik"):
        aggregate = rows.loc[rows.concept.eq(DIVIDEND_AGGREGATE)]
        chosen = aggregate if len(aggregate) else rows
        total += sum((Decimal(str(x)) for x in chosen.value), Decimal(0))
    return total


def _sum_bn(series):
    return float(sum((Decimal(str(x)) for x in series), Decimal(0)) / BILLION)


@lru_cache(maxsize=1)
def analysis():
    eligible = _eligible()
    repurchasers = eligible.loc[eligible[REPURCHASE] > 0]
    generating = eligible.loc[eligible[OCF] > 0]
    if generating.empty:
        raise ValueError("no eligible filer reports positive operating cash flow")

    comparable = _comparable()
    dividends = _dividend_facts()
    below = comparable.loc[comparable[REPURCHASE] < comparable[COMPENSATION]]
    above = comparable.loc[comparable[REPURCHASE] >= comparable[COMPENSATION]]
    if below.empty or above.empty:
        raise ValueError("the compensation comparison does not split the repurchasers")

    return {
        "eligible_filer_count": len(eligible),
        "negative_value_excluded_filer_count": len(_tagged()) - len(eligible),
        "repurchasing_filer_count": len(repurchasers),
        "operating_cash_flow_bn": _sum_bn(eligible[OCF]),
        "free_cash_flow_bn": _sum_bn(eligible.free_cash_flow),
        "negative_free_cash_flow_filers": int((eligible.free_cash_flow < 0).sum()),
        "repurchase_bn": _sum_bn(eligible[REPURCHASE]),
        "repurchase_above_fcf_filers": int(
            (repurchasers[REPURCHASE] > repurchasers.free_cash_flow).sum()
        ),
        "share_based_compensation_bn": _sum_bn(eligible[COMPENSATION]),
        "compensation_to_operating_cash_flow_median_pct": float(
            (generating[COMPENSATION] / generating[OCF] * 100).median()
        ),
        "repurchase_below_compensation_filers": int(
            (repurchasers[REPURCHASE] < repurchasers[COMPENSATION]).sum()
        ),
        "repurchase_above_operating_cash_flow_filers": int(
            (repurchasers[REPURCHASE] > repurchasers[OCF]).sum()
        ),
        "comparable_filer_count": len(comparable),
        "below_compensation_share_rise_pct": float(
            (below.share_change_pct > 0).mean() * 100
        ),
        "above_compensation_share_rise_pct": float(
            (above.share_change_pct > 0).mean() * 100
        ),
        "below_compensation_median_share_change_pct": float(below.share_change_pct.median()),
        "dividend_reporting_filer_count": dividends.cik.nunique(),
        "single_concept_reporting_filer_count": int(
            dividends.loc[dividends.concept.eq(DIVIDEND_AGGREGATE), "cik"].nunique()
        ),
        "aggregate_and_component_filer_count": int(
            sum(
                1 for _, rows in dividends.groupby("cik")
                if rows.concept.eq(DIVIDEND_AGGREGATE).any() and len(rows) > 1
            )
        ),
        "dividend_bn": float(_dividend_total(dividends) / BILLION),
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


def validate_turn_5(outputs):
    return _check(outputs, TURNS[4])


validators = {
    f"validate_turn_{i}": turn_validator(globals()[f"validate_turn_{i}"])
    for i in range(1, 6)
}
