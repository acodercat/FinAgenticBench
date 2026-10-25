"""Rank metropolitan house price growth against the index's own standard error.

A repeat-sales house price index is an estimate, and the expanded-data
metropolitan series publishes the standard error of every level it reports.  Read
beside the estimate, that error changes what a ranking means.

Across the 410 metropolitan areas the panel covers, the median area gained
2.978911 percent over the year to 2025 Q4 while the median standard error on the
ending level is 2.380000 index points.  For 41 of them the whole annual change in
the index is smaller than the error published on the level it ended at; 87 are
inside twice the error and 135 inside three times it.

The same programme publishes a second metropolitan index built only from purchase
transactions, and the two disagree.  On the 100 areas carrying both, annual growth
differs by 1.749336 percentage points on average and by 5.957833 at the widest,
and on 23 of the 100 the two published indexes disagree about whether prices rose
or fell at all.

The mortgage register says why the errors differ.  Over the 410 matched areas it
records 5,395,611 originations, and the rank correlation between an area's
origination count and its published standard error is -0.879796.  The 41 areas
with the fewest originations carry a median standard error of 5.900000 index
points against 0.590000 for the 41 with the most -- a tenfold spread in precision.
The ranking therefore selects thin markets: the fastest riser is Elmira, New York
at 23.991991 percent on 1,352 originations and a standard error of 5.990000, and
the ten fastest have a median of 2,037 originations against 5,536 across the panel.

Rejected conventions.  The panel is the traditional expanded-data metropolitan
quarterly series, which is the only one carrying a standard error; the
purchase-only series covers 100 areas and publishes none, and the
all-transactions series publishes neither a standard error nor a seasonally
adjusted level.  Growth is measured on the not-seasonally-adjusted level between
the same quarter of two years, so no seasonal convention enters.  The register's
own metropolitan field leaves 796,214 originations outside the panel, 783,114 of
them recorded against no metropolitan area at all and the rest in Puerto Rico,
which the metropolitan panel does not cover.
"""

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


LEVEL = "MSA"
FREQUENCY = "quarterly"
HPI_TYPE = "traditional"
PUBLISHED_ERROR_FLAVOR = "expanded-data"
PURCHASE_FLAVOR = "purchase-only"
END_YEAR, END_QUARTER = 2025, 4
ORIGINATED = "1"
# The register writes this code when the tract sits in no metropolitan area.
NO_METROPOLITAN_AREA = "99999"
TAIL_SIZE = 41


variables = [
    Variable("msa_index_count", None,
             "Store how many metropolitan areas the standard-error panel covers as an integer."),
    Variable("median_annual_growth_pct", None,
             "Store the median metropolitan annual index growth in percent, rounded to 6 decimals."),
    Variable("median_ending_standard_error", None,
             "Store the median standard error published on the ending index level, in index points, rounded to 6 decimals."),
    Variable("change_within_standard_error_msa_count", None,
             "Store how many metropolitan areas moved less over the year than that published error as an integer."),

    Variable("dual_flavor_msa_count", None,
             "Store how many metropolitan areas carry both published index series as an integer."),
    Variable("mean_absolute_flavor_difference_pp", None,
             "Store the mean absolute difference between the two series' annual growth, in percentage points, rounded to 6 decimals."),
    Variable("largest_flavor_difference_pp", None,
             "Store the widest such difference in percentage points, rounded to 6 decimals."),
    Variable("opposite_sign_flavor_msa_count", None,
             "Store how many metropolitan areas the two series disagree with about the direction of the year as an integer."),

    Variable("matched_origination_count", None,
             "Store the mortgage register's originations inside the covered metropolitan areas as an integer."),
    Variable("unplaced_origination_count", None,
             "Store the register's originations recorded against no metropolitan area as an integer."),
    Variable("thin_market_median_standard_error", None,
             "Store the median published standard error among the metropolitan areas with the fewest originations, in index points, rounded to 6 decimals."),
    Variable("deep_market_median_standard_error", None,
             "Store the same median among those with the most originations, in index points, rounded to 6 decimals."),

    Variable("fastest_growth_msa_code", None,
             "Store the fastest-growing metropolitan area's five-digit code as a string."),
    Variable("fastest_growth_pct", None,
             "Store that area's annual index growth in percent, rounded to 6 decimals."),
    Variable("fastest_ending_standard_error", None,
             "Store the standard error published on that area's ending level, in index points, rounded to 6 decimals."),
    Variable("fastest_ten_median_origination_count", None,
             "Store the median origination count across the ten fastest-growing metropolitan areas as an integer."),
]

DECIMALS = [0, 6, 6, 0, 0, 6, 6, 0, 0, 0, 6, 6, None, 6, 6, 0]


def _metropolitan_panel(flavor):
    """One published metropolitan quarterly series, indexed by area and quarter."""
    prices = load_expansion_table("fhfa_hpi")
    frame = pd.DataFrame({
        "area": prices.place_id.astype(str),
        "name": prices.place_name.astype(str),
        "year": pd.to_numeric(prices.yr, errors="coerce"),
        "quarter": pd.to_numeric(prices.period, errors="coerce"),
        "index": pd.to_numeric(prices.index_nsa, errors="coerce"),
        "error": pd.to_numeric(prices.rstderr, errors="coerce"),
        "level": prices.level.astype(str),
        "frequency": prices.frequency.astype(str),
        "kind": prices.hpi_type.astype(str),
        "flavor": prices.hpi_flavor.astype(str),
    })
    del prices
    return frame.loc[
        frame.level.eq(LEVEL) & frame.frequency.eq(FREQUENCY)
        & frame.kind.eq(HPI_TYPE) & frame.flavor.eq(flavor)
    ]


def _annual_growth(flavor):
    panel = _metropolitan_panel(flavor)
    ending = panel.loc[panel.year.eq(END_YEAR) & panel.quarter.eq(END_QUARTER)].set_index("area")
    starting = panel.loc[panel.year.eq(END_YEAR - 1) & panel.quarter.eq(END_QUARTER)].set_index("area")
    shared = ending.index.intersection(starting.index)
    return pd.DataFrame({
        "name": ending.name[shared],
        "ending": ending["index"][shared],
        "starting": starting["index"][shared],
        "error": ending.error[shared],
        "growth": 100.0 * (ending["index"][shared] / starting["index"][shared] - 1.0),
    })


def _originations():
    register = load_expansion_table("hmda")
    originated = register.action_taken.astype(str).eq(ORIGINATED)
    area = register.derived_msa_md.astype(str)
    counts = area[originated].value_counts()
    del register
    return counts


@lru_cache(maxsize=1)
def ground_truth():
    published = _annual_growth(PUBLISHED_ERROR_FLAVOR)
    purchase = _annual_growth(PURCHASE_FLAVOR)

    shared = published.index.intersection(purchase.index)
    difference = published.growth[shared] - purchase.growth[shared]
    disagree = int(((published.growth[shared] > 0) & (purchase.growth[shared] < 0)).sum()
                   + ((published.growth[shared] < 0) & (purchase.growth[shared] > 0)).sum())

    counts = _originations()
    matched = published.join(counts.rename("loans"), how="inner")
    matched = matched.assign(code=matched.index)

    # Ties are broken by the lower area code at both tails, so the deep tail is
    # ranked downward rather than read off the end of the ascending frame.
    thinnest = matched.sort_values(["loans", "code"], ascending=[True, True])
    deepest = matched.sort_values(["loans", "code"], ascending=[False, True])
    fastest = matched.sort_values(["growth", "code"], ascending=[False, True])

    return (
        len(published),
        round(float(published.growth.median()), 6),
        round(float(published.error.median()), 6),
        int((published.ending - published.starting).abs().lt(published.error).sum()),

        len(shared),
        round(float(difference.abs().mean()), 6),
        round(float(difference.abs().max()), 6),
        disagree,

        int(matched.loans.sum()),
        int(counts[NO_METROPOLITAN_AREA]),
        round(float(thinnest.head(TAIL_SIZE).error.median()), 6),
        round(float(deepest.head(TAIL_SIZE).error.median()), 6),

        str(fastest.index[0]),
        round(float(fastest.growth.iloc[0]), 6),
        round(float(fastest.error.iloc[0]), 6),
        int(round(float(fastest.head(10).loans.median()))),
    )


TURN_1_NAMES = ["msa_index_count", "median_annual_growth_pct",
                "median_ending_standard_error", "change_within_standard_error_msa_count"]
TURN_2_NAMES = ["dual_flavor_msa_count", "mean_absolute_flavor_difference_pp",
                "largest_flavor_difference_pp", "opposite_sign_flavor_msa_count"]
TURN_3_NAMES = ["matched_origination_count", "unplaced_origination_count",
                "thin_market_median_standard_error", "deep_market_median_standard_error"]
TURN_4_NAMES = ["fastest_growth_msa_code", "fastest_growth_pct",
                "fastest_ending_standard_error", "fastest_ten_median_origination_count"]

CODE_OUTPUTS = ("fastest_growth_msa_code",)


def _same_code(value, expected):
    if value is None:
        return False
    try:
        return int(str(value).strip()) == int(expected)
    except (TypeError, ValueError):
        return False


def _validate_subset(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip(by_name, ground_truth()))
    decimals = dict(zip(by_name, DECIMALS))
    candidate = dict(outputs)
    for name in CODE_OUTPUTS:
        if _same_code(candidate.get(name), truth[name]):
            candidate[name] = truth[name]
    return validate_ordered_outputs(
        candidate,
        [by_name[name] for name in names],
        [truth[name] for name in names],
        [decimals[name] for name in names],
    )


def validate_turn_1(outputs): return _validate_subset(outputs, TURN_1_NAMES)
def validate_turn_2(outputs): return _validate_subset(outputs, TURN_2_NAMES)
def validate_turn_3(outputs): return _validate_subset(outputs, TURN_3_NAMES)
def validate_turn_4(outputs): return _validate_subset(outputs, TURN_4_NAMES)
def validate(outputs): return _validate_subset(outputs, [v.name for v in variables])


# The turn contract: `validators` maps the name a turn's `validator`
# field cites to the callable that judges it.
validators = {
    "validate_published_error": turn_validator(validate_turn_1),
    "validate_series_disagreement": turn_validator(validate_turn_2),
    "validate_volume_and_precision": turn_validator(validate_turn_3),
    "validate_thin_market_leader": turn_validator(validate_turn_4),
}
