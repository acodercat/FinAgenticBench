"""Compare two fixed WTI forecast origins with monthly and daily PET observations.

Across the eight common strictly future target months from May through December
2024, the 2024-01-04 STEO vintage has a $3.95125/bbl mean absolute error and the
2024-04-04 vintage has a $10.92875/bbl mean absolute error.  The later vintage
improves none of the eight months.  August has the largest deterioration:
$0.82/bbl versus $10.82/bbl absolute error, a $10.00/bbl increase.  The
PET.RWTC.D series has 22 reported observations in August, spanning $72.76 to
$81.45/bbl, an $8.69/bbl high-low range that the monthly average cannot supply.

Including April as a target month, reversing signed-error direction, substituting
the PET first-purchase-price series, or measuring percentage rather than dollar
errors changes at least one requested result.  The query pins these conventions,
so they are regression probes for a hard baseline rather than live traps.
Validation uses 0.6 x 10^-N rounding-boundary tolerance for N requested decimals;
the count and target month are exact.
"""

from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import validate_ordered_outputs, turn_validator


JANUARY_ORIGIN = "2024-01-04"
APRIL_ORIGIN = "2024-04-04"
TARGET_MONTHS = [f"2024-{month:02d}" for month in range(5, 13)]

variables = [
    Variable("common_future_target_month_count", None, "Store the common strictly future target-month count as an integer."),
    Variable("january_vintage_mean_absolute_error_usd_per_barrel", None, "Store the January-vintage mean absolute dollar error in USD per barrel, rounded to 3 decimals."),
    Variable("april_vintage_mean_absolute_error_usd_per_barrel", None, "Store the April-vintage mean absolute dollar error in USD per barrel, rounded to 3 decimals."),
    Variable("april_lower_absolute_error_month_count", None, "Store the number of target months in which the April vintage has lower absolute dollar error as an integer."),
    Variable("largest_absolute_error_deterioration_month", None, "Store the target month with the largest increase in absolute dollar error as a YYYY-MM string."),
    Variable("selected_month_january_forecast_usd_per_barrel", None, "Store the January-vintage forecast for that month in USD per barrel, rounded to 2 decimals."),
    Variable("selected_month_april_forecast_usd_per_barrel", None, "Store the April-vintage forecast for that month in USD per barrel, rounded to 2 decimals."),
    Variable("selected_month_actual_usd_per_barrel", None, "Store the PET monthly spot-price observation for that month in USD per barrel, rounded to 2 decimals."),
    Variable("selected_month_daily_observation_count", None, "Store the count of reported PET.RWTC.D observations in the selected calendar month as an integer."),
    Variable("selected_month_daily_low_usd_per_barrel", None, "Store the lowest reported PET.RWTC.D price in the selected month in USD per barrel, rounded to 2 decimals."),
    Variable("selected_month_daily_high_usd_per_barrel", None, "Store the highest reported PET.RWTC.D price in the selected month in USD per barrel, rounded to 2 decimals."),
]


def _comparison():
    vintages = load_expansion_table("eia_steo_vintages")
    selected = vintages.loc[
        vintages.publication_date.isin([JANUARY_ORIGIN, APRIL_ORIGIN])
        & vintages.target_month.isin(TARGET_MONTHS)
        & vintages.value_status_in_vintage.eq("forecast")
    ]
    forecasts = selected.pivot(
        index="target_month",
        columns="publication_date",
        values="wti_spot_average_usd_per_barrel",
    )
    actuals = load_expansion_table("eia_bulk_wti").loc[
        lambda frame: frame.series_id.eq("PET.RWTC.M")
        & frame.observation_month.isin(TARGET_MONTHS),
        ["observation_month", "wti_spot_price_usd_per_barrel"],
    ].set_index("observation_month")
    comparison = forecasts.join(actuals, how="inner").rename(
        columns={
            JANUARY_ORIGIN: "january_forecast",
            APRIL_ORIGIN: "april_forecast",
            "wti_spot_price_usd_per_barrel": "actual",
        }
    )
    if len(comparison) != 8 or comparison.isna().any().any():
        raise ValueError("unexpected WTI forecast/actual comparison panel")
    comparison["january_signed_error"] = (
        comparison.january_forecast - comparison.actual
    )
    comparison["april_signed_error"] = comparison.april_forecast - comparison.actual
    comparison["january_absolute_error"] = comparison.january_signed_error.abs()
    comparison["april_absolute_error"] = comparison.april_signed_error.abs()
    comparison["absolute_error_deterioration"] = (
        comparison.april_absolute_error - comparison.january_absolute_error
    )
    return comparison


def ground_truth():
    comparison = _comparison()
    january_mae = comparison.january_absolute_error.mean()
    april_mae = comparison.april_absolute_error.mean()
    leader = comparison.reset_index().sort_values(
        ["absolute_error_deterioration", "target_month"],
        ascending=[False, True],
    ).iloc[0]
    daily = load_expansion_table("eia_bulk_wti_daily").loc[
        lambda frame: frame.series_id.eq("PET.RWTC.D")
        & frame.observation_date.str.startswith(str(leader.target_month)),
        "wti_spot_price_usd_per_barrel",
    ]
    if daily.empty or daily.isna().any():
        raise ValueError("unexpected selected-month PET.RWTC.D population")
    daily_low = daily.min()
    daily_high = daily.max()
    return (
        len(comparison),
        january_mae,
        april_mae,
        int((comparison.april_absolute_error < comparison.january_absolute_error).sum()),
        leader.target_month,
        leader.january_forecast,
        leader.april_forecast,
        leader.actual,
        len(daily),
        daily_low,
        daily_high,
    )


def validate(outputs):
    return validate_ordered_outputs(
        outputs,
        variables,
        ground_truth(),
        [
    0, 3, 3, 0, None, 2, 2, 2, 0, 2, 2,
],
    )



TURN_1_NAMES = [
    "common_future_target_month_count",
    "january_vintage_mean_absolute_error_usd_per_barrel",
    "april_vintage_mean_absolute_error_usd_per_barrel",
    "april_lower_absolute_error_month_count",
]
TURN_2_NAMES = [
    "largest_absolute_error_deterioration_month",
    "selected_month_january_forecast_usd_per_barrel",
    "selected_month_april_forecast_usd_per_barrel",
    "selected_month_actual_usd_per_barrel",
]
TURN_3_NAMES = [
    "selected_month_daily_observation_count",
    "selected_month_daily_low_usd_per_barrel",
    "selected_month_daily_high_usd_per_barrel",
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
    "validate_vintage_accuracy": turn_validator(_turn_subset(TURN_1_NAMES)),
    "validate_deterioration_month": turn_validator(_turn_subset(TURN_2_NAMES)),
    "validate_daily_dispersion": turn_validator(_turn_subset(TURN_3_NAMES)),
}
