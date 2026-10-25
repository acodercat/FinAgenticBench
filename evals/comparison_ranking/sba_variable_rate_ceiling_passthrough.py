"""A price ceiling you can see in a histogram, and the loans it does not bind.

The 7(a) programme caps what a lender may charge on a variable-rate loan at a
published base rate plus a statutory spread, and the spread tier depends on how
large the loan is. A binding ceiling does not shade a distribution; it stacks it.
Fixed-rate loans in the same file are written under a different schedule and are
the control that says the stacking is the ceiling and not the market.

The base rate the statute names is not in the frozen data, but the overnight
policy rate is, and the base rate moves with it one for one. Measuring each
loan's rate against the policy rate on its approval date therefore leaves the
statutory spread plus a constant, which is enough to see the tiers and is why the
second source is load bearing: without it there is no spread to look at.

Across the 281,056 approvals that fall inside the rate file and carry a rate, an
amount and a fixed-or-variable flag, 239,767 are variable and 41,289 fixed. The
variable median spread is 5.9100 points over the policy rate.

The stacking is the first result. One value, 5.92, holds 16.3488 percent of the
variable loans, and the five most common values hold 46.3004 percent. The fixed
loans, written without that ceiling, put only 15.0960 percent in their own five
most common values -- a third as concentrated on a group a fifth the size. Their
own most common spread leads its runner-up by 0.0993 points, far too little to
select on, which is why the case reports the five-value coverage for both groups
rather than the fixed mode.

The tiers are the second. Splitting the variable loans at 50,000, 250,000 and
350,000 dollars of approval, the smallest tier's 60,285 loans stack at 9.67
rather than 5.92, with 14.4895 percent of the tier on that one value and their
own runner-up 3.00 points of spread away, while the largest tier's 77,121 stack
at 5.92. The variable group also uses fewer distinct spreads in absolute terms
than the fixed group -- 835 against 936 -- on 5.8 times as many loans. A single ceiling would not produce two
stacks four and a half points apart inside one programme.

The pass-through is the third, and it is the sharpest. Over the five calendar
years the file covers, the most common variable rate runs 6.00, 6.00, 10.50,
11.25 and 10.25, a range of 5.25 points, while the policy rate itself ranges
5.2800. The most common spread is 5.92 in every one of those five years. A rate
that moves with policy while its spread does not is what pricing at a floating
ceiling looks like.

Fragility. Nothing ties, but two boundaries are close. The fixed group's fifth
and sixth most common spreads are 1,019 and 989 loans apart out of 41,289, so a
refresh that swapped them would move the five-value coverage by 0.0727 points
against a tolerance of 0.00006; and the 2023 most common rate leads by 378 loans
of 45,091. Every selection states a tie rule even though none currently binds,
and a case-local test asserts that none does.

Conventions the query pins because each moves an answer. The policy rate is the
one published for the approval date itself, so approvals on days the rate file
skips are outside the population. Spreads are rounded to two decimals before
they are counted, since the ceiling is quoted in quarter points and unrounded
arithmetic would split one stack across neighbouring values. The approval-size
tiers are closed at the upper edge.

Numeric validation uses the shared 0.6 x 10^-N rounding-boundary tolerance for N
requested decimals; counts match exactly.
"""

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


POLICY_RATE = "EFFR"
VARIABLE = "V"
FIXED = "F"
SPREAD_DECIMALS = 2
TOP_VALUES = 5
TIER_EDGES = (50_000.0, 250_000.0, 350_000.0)
MIN_YEAR_LOANS = 200


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = [
    "matched_loan_count",
    "variable_rate_loan_count",
    "fixed_rate_loan_count",
    "variable_median_spread_pct",
]
TURN_2_NAMES = [
    "variable_modal_spread_pct",
    "variable_modal_spread_share_pct",
    "variable_top_five_spread_share_pct",
    "fixed_top_five_spread_share_pct",
]
TURN_3_NAMES = [
    "smallest_tier_loan_count",
    "smallest_tier_modal_spread_pct",
    "smallest_tier_modal_share_pct",
    "largest_tier_modal_spread_pct",
]
TURN_4_NAMES = [
    "covered_year_count",
    "highest_yearly_modal_rate_pct",
    "modal_rate_range_pp",
    "policy_rate_range_pp",
]

variables = [
    _v(TURN_1_NAMES[0],
       "Store how many approvals survive the matching rules as an integer."),
    _v(TURN_1_NAMES[1], "Store how many of them carry a variable rate as an integer."),
    _v(TURN_1_NAMES[2], "Store how many carry a fixed rate as an integer."),
    _v(TURN_1_NAMES[3],
       "Store the median variable spread in percentage points, rounded to 4 decimals."),
    _v(TURN_2_NAMES[0],
       "Store the most common variable spread in percentage points, rounded to 2 decimals."),
    _v(TURN_2_NAMES[1],
       "Store the share of variable loans carrying it in percent, rounded to 4 decimals."),
    _v(TURN_2_NAMES[2],
       "Store the share of variable loans carrying any of the five most common variable "
       "spreads in percent, rounded to 4 decimals."),
    _v(TURN_2_NAMES[3],
       "Store the same share for the fixed group against its own five most common spreads, in "
       "percent, rounded to 4 decimals."),
    _v(TURN_3_NAMES[0],
       "Store how many variable loans fall in the smallest approval tier as an integer."),
    _v(TURN_3_NAMES[1],
       "Store that tier's most common spread in percentage points, rounded to 2 decimals."),
    _v(TURN_3_NAMES[2],
       "Store the share of the tier carrying it in percent, rounded to 4 decimals."),
    _v(TURN_3_NAMES[3],
       "Store the largest tier's most common spread in percentage points, rounded to 2 "
       "decimals."),
    _v(TURN_4_NAMES[0],
       "Store how many calendar years clear the stated minimum as an integer."),
    _v(TURN_4_NAMES[1],
       "Store the highest of those years' most common variable rates in percent, rounded to 2 "
       "decimals."),
    _v(TURN_4_NAMES[2],
       "Store the range of those most common rates in percentage points, rounded to 2 "
       "decimals."),
    _v(TURN_4_NAMES[3],
       "Store the range of the policy rate across the matched approvals in percentage points, "
       "rounded to 4 decimals."),
]

DECIMALS = [
    0, 0, 0, 4,
    2, 4, 4, 4,
    0, 2, 4, 2,
    0, 2, 2, 4,
]


@lru_cache(maxsize=1)
def _policy_rates():
    """The overnight policy rate published for each effective date."""
    rates = load_expansion_table("nyfed_reference_rates")
    frame = rates.loc[rates["Rate Type"].astype(str).eq(POLICY_RATE),
                      ["Effective Date", "Rate (%)"]].copy()
    del rates
    frame["observed_on"] = pd.to_datetime(frame["Effective Date"])
    frame["policy_rate"] = pd.to_numeric(frame["Rate (%)"], errors="coerce")
    series = frame.dropna(subset=["policy_rate"]).set_index("observed_on").policy_rate
    if series.index.has_duplicates:
        raise ValueError("the policy rate is published more than once for a date")
    return series.sort_index()


@lru_cache(maxsize=1)
def _matched():
    """Approvals carrying a rate, an amount, a rate kind and a same-day policy rate."""
    loans = load_expansion_table("sba_7a")
    frame = loans.loc[:, ["ApprovalDate", "InitialInterestRate",
                          "FixedorVariableInterestInd", "GrossApproval"]].copy()
    del loans
    frame["approved_on"] = pd.to_datetime(frame.ApprovalDate, errors="coerce")
    frame["rate"] = pd.to_numeric(frame.InitialInterestRate, errors="coerce")
    frame["approval"] = pd.to_numeric(frame.GrossApproval, errors="coerce")
    frame["kind"] = frame.FixedorVariableInterestInd.astype(str).str.upper().str[0]
    frame = frame.loc[frame.kind.isin((VARIABLE, FIXED))]
    frame = frame.dropna(subset=["approved_on", "rate", "approval"])
    # The policy rate must exist for the approval date itself; an approval on a day
    # the rate file skips is left out rather than carried from a neighbouring day.
    frame["policy_rate"] = frame.approved_on.map(_policy_rates())
    frame = frame.dropna(subset=["policy_rate"])
    if frame.empty:
        raise ValueError("no approval falls inside the policy rate window")
    frame["spread"] = (frame.rate - frame.policy_rate).round(SPREAD_DECIMALS)
    return frame


def _ranked(counts):
    """Spreads ordered by how many loans carry them, ties going to the lower spread."""
    return sorted(counts.index, key=lambda value: (-int(counts[value]), float(value)))


def _top_share(frame):
    counts = frame.spread.value_counts()
    chosen = _ranked(counts)[:TOP_VALUES]
    return 100.0 * float(counts[chosen].sum()) / float(len(frame))


def _modal(frame):
    counts = frame.spread.value_counts()
    value = _ranked(counts)[0]
    return float(value), 100.0 * float(counts[value]) / float(len(frame))


@lru_cache(maxsize=1)
def ground_truth():
    matched = _matched()
    variable = matched.loc[matched.kind.eq(VARIABLE)]
    fixed = matched.loc[matched.kind.eq(FIXED)]

    modal_spread, modal_share = _modal(variable)

    # The lowest cut is open below, so no approval falls outside the tiers; a zero
    # left edge would have silently dropped a nil or negative approval amount.
    edges = (float("-inf"),) + TIER_EDGES + (float("inf"),)
    tier = pd.cut(variable.approval, bins=edges, labels=False, right=True)
    smallest = variable.loc[tier.eq(0)]
    largest = variable.loc[tier.eq(len(TIER_EDGES))]
    smallest_spread, smallest_share = _modal(smallest)

    by_year = {
        int(year): group
        for year, group in variable.groupby(variable.approved_on.dt.year)
        if len(group) >= MIN_YEAR_LOANS
    }
    modal_rates = {}
    for year, group in by_year.items():
        counts = group.rate.round(SPREAD_DECIMALS).value_counts()
        modal_rates[year] = float(
            sorted(counts.index, key=lambda value: (-int(counts[value]), float(value)))[0]
        )

    return (
        int(len(matched)),
        int(len(variable)),
        int(len(fixed)),
        float(variable.spread.median()),
        modal_spread,
        modal_share,
        _top_share(variable),
        _top_share(fixed),
        int(len(smallest)),
        smallest_spread,
        smallest_share,
        _modal(largest)[0],
        int(len(by_year)),
        max(modal_rates.values()),
        max(modal_rates.values()) - min(modal_rates.values()),
        float(matched.policy_rate.max() - matched.policy_rate.min()),
    )


def _validate_subset(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip(by_name, ground_truth()))
    decimals = dict(zip(by_name, DECIMALS))
    return validate_ordered_outputs(
        outputs,
        [by_name[name] for name in names],
        [truth[name] for name in names],
        [decimals[name] for name in names],
    )


def validate_population(outputs):
    return _validate_subset(outputs, TURN_1_NAMES)


def validate_stacking(outputs):
    return _validate_subset(outputs, TURN_2_NAMES)


def validate_tiers(outputs):
    return _validate_subset(outputs, TURN_3_NAMES)


def validate_passthrough(outputs):
    return _validate_subset(outputs, TURN_4_NAMES)


def validate(outputs):
    return _validate_subset(outputs, [variable.name for variable in variables])


validators = {
    "validate_population": turn_validator(validate_population),
    "validate_stacking": turn_validator(validate_stacking),
    "validate_tiers": turn_validator(validate_tiers),
    "validate_passthrough": turn_validator(validate_passthrough),
}
