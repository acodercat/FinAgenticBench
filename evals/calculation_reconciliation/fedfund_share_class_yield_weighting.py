"""Compare FEDFUND class yields with same-date New York Fed SOFR.

On 2025-10-31 the 15 FEDFUND classes have a 3.8340% simple mean seven-day net
yield and a 3.9875% class-net-asset-weighted mean.  The Institutional class holds
86.8505% of class assets.  SELECT has the unique lowest net yield, 3.18%, and the
series gross-minus-SELECT-net gap is 1.00 percentage point.

The same-date New York Fed overnight SOFR is 4.22%.  The series gross yield minus
SOFR is -0.04 point and the asset-weighted class net yield minus SOFR is -0.2325
point.  Equal class weighting, leaving the fund yields in decimal form, or using
the 30-day average SOFR instead of the overnight rate produces different outputs.
Because the query pins the weighting, units and named reference rate, those paths
are regression probes rather than diagnostic traps.  Numeric validation uses
0.6 x 10^-N rounding-boundary tolerance; counts and the class label are exact.
"""

import numpy as np
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import validate_ordered_outputs, turn_validator


ACCESSION = "0002071691-25-004249"
YIELD_DATE = "2025-10-31"

variables = [
    Variable("reported_class_count", None, "Store the share-class count with a net yield as an integer."),
    Variable("class_net_assets_usd_billions", None, "Store total class net assets in USD billions, rounded to 6 decimals."),
    Variable("series_gross_seven_day_yield_pct", None, "Store the series gross seven-day yield in percent, rounded to 2 decimals."),
    Variable("simple_mean_class_net_yield_pct", None, "Store the equal-class mean net seven-day yield in percent, rounded to 4 decimals."),
    Variable("asset_weighted_class_net_yield_pct", None, "Store the class-net-asset-weighted mean net seven-day yield in percent, rounded to 4 decimals."),
    Variable("institutional_class_asset_share_pct", None, "Store the Institutional class share of total class net assets in percent, rounded to 4 decimals."),
    Variable("lowest_net_yield_class", None, "Store the unique class name with the lowest net seven-day yield."),
    Variable("lowest_class_net_yield_pct", None, "Store that class's net seven-day yield in percent, rounded to 2 decimals."),
    Variable("same_date_sofr_pct", None, "Store SOFR in percent, rounded to 2 decimals."),
]


def ground_truth():
    data = load_expansion_table("sec_nmfp_class_yields")
    data = data.loc[
        data.accession.eq(ACCESSION) & data.yield_date.eq(YIELD_DATE)
    ].copy()
    if len(data) != 15 or data.class_id.nunique() != len(data):
        raise ValueError("unexpected FEDFUND share-class population")
    data["net_pct"] = data.seven_day_net_yield * 100
    data["gross_pct"] = data.seven_day_gross_yield * 100
    gross = data.gross_pct.drop_duplicates()
    if len(gross) != 1:
        raise ValueError("expected one series gross yield")
    simple = data.net_pct.mean()
    weighted = np.average(data.net_pct, weights=data.class_net_assets_usd)
    institutional = data.loc[data.class_name.eq("INSTITUTIONAL")]
    if len(institutional) != 1:
        raise ValueError("expected one Institutional class")
    ranked = data.sort_values(["net_pct", "class_name"])
    if ranked.iloc[0].net_pct == ranked.iloc[1].net_pct:
        raise ValueError("lowest class net yield is tied")
    lowest = ranked.iloc[0]

    rates = load_expansion_table("nyfed_reference_rates")
    sofr_row = rates.loc[
        rates["Effective Date"].eq(YIELD_DATE) & rates["Rate Type"].eq("SOFR")
    ]
    if len(sofr_row) != 1 or sofr_row["Rate (%)"].isna().any():
        raise ValueError("expected one same-date overnight SOFR observation")
    sofr = float(sofr_row.iloc[0]["Rate (%)"])
    gross_value = float(gross.iloc[0])
    total_assets = data.class_net_assets_usd.sum()
    return (
        len(data),
        total_assets / 1e9,
        gross_value,
        simple,
        weighted,
        float(institutional.iloc[0].class_net_assets_usd / total_assets * 100),
        lowest.class_name,
        lowest.net_pct,
        sofr,
    )


def validate(outputs):
    return validate_ordered_outputs(
        outputs, variables, ground_truth(),
        [
    0, 6, 2, 4, 4, 4, None, 2, 2,
],
    )



TURN_1_NAMES = [
    "reported_class_count",
    "class_net_assets_usd_billions",
    "series_gross_seven_day_yield_pct",
]
TURN_2_NAMES = [
    "simple_mean_class_net_yield_pct",
    "asset_weighted_class_net_yield_pct",
    "institutional_class_asset_share_pct",
    "lowest_net_yield_class",
    "lowest_class_net_yield_pct",
]
TURN_3_NAMES = [
    "same_date_sofr_pct",
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
    "validate_class_panel": turn_validator(_turn_subset(TURN_1_NAMES)),
    "validate_weighting": turn_validator(_turn_subset(TURN_2_NAMES)),
    "validate_sofr_comparison": turn_validator(_turn_subset(TURN_3_NAMES)),
}
