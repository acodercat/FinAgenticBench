"""Leverage is a definition before it is a number, and goodwill sits inside it.

A filer's balance sheet supports several defensible leverage ratios and the
gap between them is not small. Of the 5,524 annual filings in this window, 709
tag long-term debt, its noncurrent portion, the operating lease liability,
goodwill and stockholders' equity together, and 624 of those carry positive
equity. Across those 624 the median ratio to equity runs 0.6305 on the
noncurrent portion of long-term debt alone, 0.6805 on the whole of it, and
0.7852 once the operating lease liability joins the numerator. Struck in
aggregate against tangible equity instead, the lease-inclusive figure is
1.4399. The same balance sheets, read four defensible ways.

Each step is a real choice. The noncurrent portion excludes debt maturing
inside a year, so a filer refinancing next year looks less levered on it than
it is. The operating lease liability has sat on the balance sheet since the
2019 leases standard and is a contractual obligation the filer cannot walk
away from, but the conventional definition of debt leaves it out; here it adds
419.849886 USD billions against equity of 3394.290464. Goodwill is 1612.234198
of that equity, so removing it takes the denominator down by nearly half, and
138 of the 624 are left with negative tangible equity: their goodwill is
larger than their whole book.

Where that goodwill sits is a separate question, because impairment is tested
below the entity. 180 filers break goodwill out along a business-segment axis
inside the very filing that carries their entity figure, and for 162 the
segment figures sum to that total. The other 18 do not, and the query keeps
only the 162 rather than guessing which member to drop. Among the 162 the
largest segment carries a median 63.5991% of the goodwill and 125 of them have
a single segment holding more than half, so the entity-level ratio understates
how concentrated the impairment exposure is.

Four things the data forces:

1. ``LongTermDebt`` and ``LongTermDebtNoncurrent`` are different concepts and
   the file carries both. On 23 of the 709 the total is below the noncurrent
   portion, which the two tags cannot both be right about; they are reported
   rather than corrected, since nothing here says which one is wrong.
2. The ratios are struck only where equity is positive. A negative denominator
   turns a more levered filer into a smaller-looking ratio.
3. Every median is taken with linear interpolation. It matters: the 624 are an
   even count whose two middle values differ on all three ratios, so the lower
   and upper middle values give 0.6293 and 0.6317 against the reported 0.6305,
   0.6802 and 0.6809 against 0.6805, and 0.7837 and 0.7867 against 0.7852.
4. Segment rows are matched on the accession as well as the filer and period
   end. Matching on the period end alone would let a later annual report's
   comparative column stand against the earlier filing's entity total; inside
   one filing a member can still appear under more than one presentation
   index, and only index 0 is taken. Both rules bite: 133 of the 693 member
   groups in the window carry more than one row, 102 of them across filings
   and 31 inside one, and 17 carry more than one value. After the accession
   match 582 groups survive, 32 of them still holding more than one row but
   only one holding more than one value: Celanese reports EngineeredMaterials
   twice in the same filing, 5,025 USD millions at index 0 against 5,000 at
   index 1. Taking the highest index instead of index 0 drops the tie count
   to 161, the median share to 63.0318% and the majority count to 124, while
   leaving the matched filer count at 180. Every one of the 582 groups carries
   an index 0 row, so the rule selects rather than discards.

5. The segment median is also an even count whose middle values differ, at
   63.0318 and 64.1664 against the reported 63.5991, so the interpolation rule
   carries it too.

6. The aggregate after goodwill sums every filer in, including the 138 whose
   own equity after goodwill is below zero. Dropping them, which following the
   turn's own reading of a negative denominator would suggest, gives 0.9202
   rather than 1.4399. The aggregate stays well defined where the individual
   ratios do not, which is the reason for reporting it beside them.

Every stated selection binds except one: nine filers close twice inside the
window, seven of them at different period ends and two on different filing
dates, so the accession never has to separate them. The rule is stated anyway
and the case test asserts it does not fire.

One relation worth naming so it is not mistaken for an oversight: the
lease-inclusive aggregate on equity can be rebuilt from the debt aggregate
plus the lease total over the equity total, which is three reported figures
rather than two, and it lands on the same value either way. It is reported
because it is the comparison the turn is about.
"""

from decimal import Decimal
from functools import lru_cache

from cave_agent import Variable

from core.data import load_expansion_table, load_financial_statements
from core.validation import turn_validator, validate_ordered_outputs


PRIMARY_PRESENTATION = 0
WINDOW = ("2024-01-01", "2024-12-31")
SEGMENT_AXIS = "BusinessSegments"
GOODWILL_TAG = "Goodwill"
TIE_USD = Decimal(1)
DEBT = "LongTermDebt"
DEBT_NONCURRENT = "LongTermDebtNoncurrent"
LEASE = "OperatingLeaseLiability"
GOODWILL = "Goodwill"
EQUITY = "StockholdersEquity"
REQUIRED = (DEBT, DEBT_NONCURRENT, LEASE, GOODWILL, EQUITY)
BILLION = Decimal(10) ** 9

OUTPUTS = (
    ("annual_filer_count", 0, "the count of filers in the annual population as an integer"),
    ("complete_filer_count", 0, "how many tag every balance-sheet item the case needs, as an integer"),
    ("positive_equity_count", 0, "how many of those carry positive equity, as an integer"),
    ("total_below_noncurrent_count", 0, "how many report a long-term total below its own noncurrent portion, as an integer"),
    ("median_noncurrent_leverage", 4, "the median ratio of the noncurrent portion to equity"),
    ("median_total_debt_leverage", 4, "the median ratio of the whole long-term total to equity"),
    ("median_lease_inclusive_leverage", 4, "the median ratio once the lease liability joins the numerator"),
    ("operating_lease_total_bn", 6, "those filers' operating lease liability summed, in USD billions"),
    ("equity_total_bn", 6, "their equity summed, in USD billions"),
    ("goodwill_total_bn", 6, "their goodwill summed, in USD billions"),
    ("negative_tangible_equity_count", 0, "how many are left with negative equity once goodwill is removed, as an integer"),
    ("aggregate_lease_tangible_leverage", 4, "their summed lease-inclusive debt over their summed equity after goodwill"),
    ("segment_filer_count", 0, "the count of filers breaking goodwill out by segment as an integer"),
    ("segment_tie_count", 0, "how many of those have segment figures summing to the entity total, as an integer"),
    ("median_largest_segment_share_pct", 4, "the median share of goodwill in the largest segment, in percent"),
    ("single_segment_majority_count", 0, "how many have one segment holding more than half of it, as an integer"),
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
    tuple(n for n, _, _ in OUTPUTS[12:]),
)


@lru_cache(maxsize=1)
def _annual():
    statements = load_financial_statements()
    frame = statements.loc[
        statements.form.eq("10-K") & statements.fiscal_period.eq("FY")
    ].copy()
    frame["period_end"] = frame.report_date.astype(str)
    frame = frame.loc[frame.period_end.between(*WINDOW)]
    if frame.empty:
        raise ValueError("no annual filing has a period end in the window")
    ordered = frame.sort_values(["cik", "period_end", "filed_date", "accession"])
    kept = ordered.drop_duplicates("cik", keep="last").copy()
    kept["cik"] = kept.cik.astype(str)
    return kept


@lru_cache(maxsize=1)
def _levered():
    """Filers tagging every needed item and carrying positive equity."""
    complete = _annual().dropna(subset=list(REQUIRED))
    frame = complete.loc[complete[EQUITY] > 0].copy()
    if frame.empty:
        raise ValueError("no filer tags every item and carries positive equity")
    frame["noncurrent_ratio"] = frame[DEBT_NONCURRENT] / frame[EQUITY]
    frame["total_ratio"] = frame[DEBT] / frame[EQUITY]
    frame["lease_ratio"] = (frame[DEBT] + frame[LEASE]) / frame[EQUITY]
    return frame


@lru_cache(maxsize=1)
def _segments():
    """Segment goodwill taken from the very filing the entity figure comes from.

    Matching on the period end alone would let a later annual report's
    comparative column stand against the entity figure of the earlier one, so
    the accession is part of the key. Inside one filing a member can still
    appear twice under different presentation indices, and only presentation
    index 0 is taken.
    """
    dimensional = load_expansion_table("sec_notes_dimensional")
    rows = dimensional.loc[
        dimensional.tag.astype(str).eq(GOODWILL_TAG)
        & dimensional.axis.astype(str).eq(SEGMENT_AXIS)
        & dimensional.form.astype(str).eq("10-K")
        & dimensional.qtrs.eq(0)
    ].copy()
    rows["period_end"] = rows.ddate.astype(str)
    rows = rows.loc[rows.period_end.between(*WINDOW)]
    rows["cik"] = rows.cik.astype(str)
    rows["member"] = rows.member.astype(str)
    rows["accession"] = rows.adsh.astype(str)
    if rows.empty:
        raise ValueError("no segment goodwill row falls in the window")

    entity = _annual()
    entity = entity.loc[entity[GOODWILL].notna(), ["cik", "period_end", "accession", GOODWILL]].copy()
    entity["accession"] = entity.accession.astype(str)
    joined = rows.merge(entity, on=["cik", "period_end", "accession"])
    if joined.empty:
        raise ValueError("no segment row shares a filing with an entity goodwill figure")

    primary = joined.loc[joined.iprx.eq(PRIMARY_PRESENTATION)]
    if primary.groupby(["cik", "member"]).ngroups != joined.groupby(["cik", "member"]).ngroups:
        raise ValueError("a member group carries no primary-presentation row")
    return primary.groupby("cik").agg(
        period_end=("period_end", "first"),
        segment_total=("value", "sum"),
        largest=("value", "max"),
        **{GOODWILL: (GOODWILL, "first")},
    ).reset_index()


@lru_cache(maxsize=1)
def analysis():
    annual = _annual()
    complete = annual.dropna(subset=list(REQUIRED))
    levered = _levered()

    equity = sum((Decimal(str(x)) for x in levered[EQUITY]), Decimal(0))
    goodwill = sum((Decimal(str(x)) for x in levered[GOODWILL]), Decimal(0))
    lease = sum((Decimal(str(x)) for x in levered[LEASE]), Decimal(0))
    debt = sum((Decimal(str(x)) for x in levered[DEBT]), Decimal(0))
    tangible = equity - goodwill
    if not tangible:
        raise ValueError("summed equity after goodwill leaves nothing to divide by")

    segments = _segments()
    # A filer whose segment figures miss the entity total is dropped whole
    # rather than repaired: nothing here says which member is the odd one.
    tied = segments.loc[
        (segments.segment_total - segments[GOODWILL]).abs() < float(TIE_USD)
    ].copy()
    if tied.empty:
        raise ValueError("no filer's segment goodwill sums to its entity figure")
    tied["largest_share"] = tied.largest / tied.segment_total * 100

    return {
        "annual_filer_count": len(annual),
        "complete_filer_count": len(complete),
        "positive_equity_count": len(levered),
        "total_below_noncurrent_count": int((complete[DEBT] < complete[DEBT_NONCURRENT]).sum()),
        "median_noncurrent_leverage": float(levered.noncurrent_ratio.median()),
        "median_total_debt_leverage": float(levered.total_ratio.median()),
        "median_lease_inclusive_leverage": float(levered.lease_ratio.median()),
        "operating_lease_total_bn": float(lease / BILLION),
        "equity_total_bn": float(equity / BILLION),
        "goodwill_total_bn": float(goodwill / BILLION),
        "negative_tangible_equity_count": int(
            ((levered[EQUITY] - levered[GOODWILL]) < 0).sum()
        ),
        "aggregate_lease_tangible_leverage": float((debt + lease) / tangible),
        "segment_filer_count": len(segments),
        "segment_tie_count": len(tied),
        "median_largest_segment_share_pct": float(tied.largest_share.median()),
        "single_segment_majority_count": int((tied.largest_share > 50).sum()),
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
