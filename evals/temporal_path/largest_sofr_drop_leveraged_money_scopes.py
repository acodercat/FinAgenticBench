"""Select a SOFR event window before comparing CFTC position scopes.

Among consecutive 2025 CFTC SOFR-3M report dates with same-date New York Fed
overnight SOFR observations, the largest decline is from 2025-10-28 to
2025-11-04: 4.31% to 4.00%, or -0.31 percentage points.  Over those snapshots,
leveraged-money directional net (long minus short) changes from -684,788 to
-961,361 futures-only contracts and from -925,858 to -1,218,695 combined
contracts.  The combined-minus-futures scope difference moves from -241,070 to
-257,334, a -16,264-contract change.

Using the New York Fed 30-day average instead of overnight SOFR selects
2025-01-07 to 2025-01-14 and a -0.07559-point decline. Adding spreading to the
directional net at the pinned event instead gives changes of -264,331 and
-149,059 contracts and a +115,272 change in the scope difference. The query
pins both choices, so these are regression probes for a hard baseline rather
than live diagnostic traps.

Numeric validation uses exact matching for contract counts and 0.6 × 10^-2
rounding-boundary tolerance for rates and percentage-point changes. Dates match
exactly in YYYY-MM-DD form.
"""

from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import validate_ordered_outputs, turn_validator


CONTRACT_CODE = "134741"
SCOPES = {"futures_only", "futures_and_options_combined"}

variables = [
    Variable("window_start_date", None, "Store the starting report date as a YYYY-MM-DD string."),
    Variable("window_end_date", None, "Store the ending report date as a YYYY-MM-DD string."),
    Variable("start_overnight_sofr_pct", None, "Store starting overnight SOFR in percent, rounded to 2 decimals."),
    Variable("end_overnight_sofr_pct", None, "Store ending overnight SOFR in percent, rounded to 2 decimals."),
    Variable("futures_only_start_directional_net", None, "Store starting futures-only leveraged-money directional net as a signed integer."),
    Variable("futures_only_end_directional_net", None, "Store ending futures-only leveraged-money directional net as a signed integer."),
    Variable("futures_only_directional_net_change", None, "Store ending minus starting futures-only directional net as a signed integer."),
    Variable("combined_start_directional_net", None, "Store starting combined leveraged-money directional net as a signed integer."),
    Variable("combined_end_directional_net", None, "Store ending combined leveraged-money directional net as a signed integer."),
    Variable("combined_directional_net_change", None, "Store ending minus starting combined directional net as a signed integer."),
    Variable("start_combined_minus_futures_net", None, "Store starting combined minus futures-only directional net as a signed integer."),
    Variable("end_combined_minus_futures_net", None, "Store ending combined minus futures-only directional net as a signed integer."),
    Variable("combined_minus_futures_net_change", None, "Store ending minus starting change in the combined-minus-futures directional net as a signed integer."),
]


def _one(frame, description):
    if len(frame) != 1:
        raise ValueError(f"expected one {description} row, found {len(frame)}")
    return frame.iloc[0]


def ground_truth():
    cot = load_expansion_table("cftc_cot")
    cot = cot.loc[cot.CFTC_Contract_Market_Code.eq(CONTRACT_CODE)].copy()
    scope_counts = cot.groupby("Report_Date_as_YYYY-MM-DD").report_scope.agg(
        lambda values: set(values)
    )
    complete_dates = set(
        scope_counts.loc[
            scope_counts.map(lambda scopes: set(scopes) == SCOPES).astype(bool)
        ].index
    )

    rates = load_expansion_table("nyfed_reference_rates")
    rates = rates.loc[
        rates["Rate Type"].eq("SOFR")
        & rates["Effective Date"].str.startswith("2025-")
        & rates["Effective Date"].isin(complete_dates),
        ["Effective Date", "Rate (%)"],
    ].sort_values("Effective Date")
    if rates["Effective Date"].duplicated().any() or len(rates) < 2:
        raise ValueError("expected unique same-date SOFR observations")
    rates = rates.reset_index(drop=True)
    rates["change"] = rates["Rate (%)"].diff()
    minimum = rates["change"].min()
    ending_index = int(rates.index[rates["change"].eq(minimum)][0])
    if ending_index == 0:
        raise ValueError("selected SOFR change has no preceding report date")
    start_rate = rates.iloc[ending_index - 1]
    end_rate = rates.iloc[ending_index]

    def position(date, scope):
        row = _one(
            cot.loc[
                cot["Report_Date_as_YYYY-MM-DD"].eq(date)
                & cot.report_scope.eq(scope)
            ],
            f"{date} {scope} CFTC",
        )
        return int(
            row.Lev_Money_Positions_Long_All - row.Lev_Money_Positions_Short_All
        )

    start_date = str(start_rate["Effective Date"])
    end_date = str(end_rate["Effective Date"])
    futures_start = position(start_date, "futures_only")
    futures_end = position(end_date, "futures_only")
    combined_start = position(start_date, "futures_and_options_combined")
    combined_end = position(end_date, "futures_and_options_combined")
    start_scope_difference = combined_start - futures_start
    end_scope_difference = combined_end - futures_end
    return (
        start_date,
        end_date,
        float(start_rate["Rate (%)"]),
        float(end_rate["Rate (%)"]),
        futures_start,
        futures_end,
        futures_end - futures_start,
        combined_start,
        combined_end,
        combined_end - combined_start,
        start_scope_difference,
        end_scope_difference,
        end_scope_difference - start_scope_difference,
    )


def validate(outputs):
    return validate_ordered_outputs(
        outputs,
        variables,
        ground_truth(),
        [
    None, None, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0,
],
    )



TURN_1_NAMES = [
    "window_start_date",
    "window_end_date",
    "start_overnight_sofr_pct",
    "end_overnight_sofr_pct",
]
TURN_2_NAMES = [
    "futures_only_start_directional_net",
    "futures_only_end_directional_net",
    "futures_only_directional_net_change",
    "combined_start_directional_net",
    "combined_end_directional_net",
    "combined_directional_net_change",
]
TURN_3_NAMES = [
    "start_combined_minus_futures_net",
    "end_combined_minus_futures_net",
    "combined_minus_futures_net_change",
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
    "validate_rate_window": turn_validator(_turn_subset(TURN_1_NAMES)),
    "validate_scope_positions": turn_validator(_turn_subset(TURN_2_NAMES)),
    "validate_scope_difference": turn_validator(_turn_subset(TURN_3_NAMES)),
}
