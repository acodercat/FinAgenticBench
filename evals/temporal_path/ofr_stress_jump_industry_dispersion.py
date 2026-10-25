"""Link the largest 2020 OFR FSI jump to same-day industry dispersion.

Across 253 dates common to OFR and the Ken French 10-industry portfolios, the
largest single-step FSI increase between consecutive common dates is 3.454 points from 1.680 on 2020-03-06
to 5.134 on 2020-03-09.  Volatility is the largest category-contribution
increase at 2.178 points.  On the event date all ten industry portfolios are
negative: Enrgy is worst at -19.68%, Shops is best at -5.05%, their equal-
weighted mean is -8.531%, and the cross-industry spread is 14.63 points.

Selecting the highest FSI level rather than its largest increase, taking the
largest rise over a run of consecutive dates of any length rather than a single
step, shifting industry returns to the next common date, ranking category levels
rather than changes, or using the median industry return changes at least one
output.  The
query pins these conventions, so they are regression probes for a hard baseline.
OFR and Ken French happen to contain the same 253 calendar-2020 dates, so an
OFR-only event grid is a measured numerical equivalent in this frozen sample.
"""

from cave_agent import Variable

from core.data import load_expansion_table, load_ken_french_table
from core.validation import validate_ordered_outputs, turn_validator


INDUSTRIES = {
    "NoDur": "nodur_pct",
    "Durbl": "durbl_pct",
    "Manuf": "manuf_pct",
    "Enrgy": "enrgy_pct",
    "HiTec": "hitec_pct",
    "Telcm": "telcm_pct",
    "Shops": "shops_pct",
    "Hlth": "hlth_pct",
    "Utils": "utils_pct",
    "Other": "other_pct",
}
CATEGORIES = {
    "Credit": "credit",
    "Equity valuation": "equity_valuation",
    "Safe assets": "safe_assets",
    "Funding": "funding",
    "Volatility": "volatility",
}

variables = [
    Variable("common_2020_observation_date_count", None, "Store the common 2020 observation-date count as an integer."),
    Variable("stress_window_start_date", None, "Store the starting date as a YYYY-MM-DD string."),
    Variable("stress_window_end_date", None, "Store the ending date as a YYYY-MM-DD string."),
    Variable("start_ofr_fsi", None, "Store the starting OFR FSI index value, rounded to 3 decimals."),
    Variable("end_ofr_fsi", None, "Store the ending OFR FSI index value, rounded to 3 decimals."),
    Variable("largest_category_contribution_increase_name", None, "Store the leading OFR market-category name; casing and repeated whitespace are not significant."),
    Variable("largest_category_contribution_increase", None, "Store that category's contribution increase in OFR FSI index points, rounded to 3 decimals."),
    Variable("worst_event_date_industry_label", None, "Store the worst-return official Ken French 10-industry label; casing and repeated whitespace are not significant."),
    Variable("worst_event_date_industry_return_pct", None, "Store the worst event-date industry return in percent, rounded to 2 decimals."),
    Variable("best_event_date_industry_label", None, "Store the best-return official Ken French 10-industry label; casing and repeated whitespace are not significant."),
    Variable("best_event_date_industry_return_pct", None, "Store the best event-date industry return in percent, rounded to 2 decimals."),
    Variable("event_date_negative_industry_count", None, "Store the negative-return industry count as an integer."),
    Variable("event_date_equal_weighted_industry_mean_return_pct", None, "Store the equal-weighted mean industry return in percent, rounded to 3 decimals."),
]


def _normalized_label(value):
    if not isinstance(value, str):
        return value
    return " ".join(value.split()).casefold()


def _event_panel():
    stress = load_expansion_table("ofr_market_stress").copy()
    industries = load_ken_french_table("daily_industry_returns").copy()
    panel = stress.merge(industries, on="date", how="inner", validate="one_to_one")
    panel = panel.loc[panel.date.between("2020-01-01", "2020-12-31")].sort_values(
        "date"
    ).reset_index(drop=True)
    if len(panel) < 2 or panel.date.duplicated().any():
        raise ValueError("unexpected 2020 OFR/industry common-date panel")
    panel["ofr_fsi_change"] = panel.ofr_fsi.diff()
    return panel


def ground_truth():
    panel = _event_panel()
    maximum = panel.ofr_fsi_change.max()
    ending_index = int(panel.index[panel.ofr_fsi_change.eq(maximum)][0])
    if ending_index == 0:
        raise ValueError("selected stress change lacks a prior common date")
    start = panel.iloc[ending_index - 1]
    end = panel.iloc[ending_index]

    category_changes = {
        label: float(end[column] - start[column])
        for label, column in CATEGORIES.items()
    }
    category = sorted(category_changes, key=lambda label: (-category_changes[label], label))[0]
    returns = {label: float(end[column]) for label, column in INDUSTRIES.items()}
    worst = sorted(returns, key=lambda label: (returns[label], label))[0]
    best = sorted(returns, key=lambda label: (-returns[label], label))[0]
    values = list(returns.values())
    return (
        len(panel),
        str(start.date),
        str(end.date),
        float(start.ofr_fsi),
        float(end.ofr_fsi),
        category,
        category_changes[category],
        worst,
        returns[worst],
        best,
        returns[best],
        sum(value < 0 for value in values),
        sum(values) / len(values),
    )


DECIMALS = [0, None, None, 3, 3, None, 3, None, 2, None, 2, 0, 3]
LABEL_NAMES = (
    "largest_category_contribution_increase_name",
    "worst_event_date_industry_label",
    "best_event_date_industry_label",
)


def validate(outputs):
    expected = ground_truth()
    expected_by_name = dict(zip((variable.name for variable in variables), expected))
    candidate = dict(outputs)
    for name in LABEL_NAMES:
        if _normalized_label(candidate.get(name)) == _normalized_label(expected_by_name[name]):
            candidate[name] = expected_by_name[name]
    return validate_ordered_outputs(candidate, variables, expected, DECIMALS)



TURN_1_NAMES = [
    "common_2020_observation_date_count",
    "stress_window_start_date",
    "stress_window_end_date",
    "start_ofr_fsi",
    "end_ofr_fsi",
]
TURN_2_NAMES = [
    "largest_category_contribution_increase_name",
    "largest_category_contribution_increase",
]
TURN_3_NAMES = [
    "worst_event_date_industry_label",
    "worst_event_date_industry_return_pct",
    "best_event_date_industry_label",
    "best_event_date_industry_return_pct",
    "event_date_negative_industry_count",
    "event_date_equal_weighted_industry_mean_return_pct",
]


def _turn_subset(names):
    """Score only `names`, filling the other outputs with their own answers."""
    def run(outputs):
        merged = dict(zip((variable.name for variable in variables), ground_truth()))
        merged.update({name: outputs.get(name) for name in names})
        return validate(merged)

    return run


# The turn contract: `validators` maps the name a turn's `validator`
# field cites to the callable that judges it.
validators = {
    "validate_stress_window": turn_validator(_turn_subset(TURN_1_NAMES)),
    "validate_category_decomposition": turn_validator(_turn_subset(TURN_2_NAMES)),
    "validate_industry_dispersion": turn_validator(_turn_subset(TURN_3_NAMES)),
}
