"""Link the FY2025 USFR accounting bridge to federal debt-stock changes.

The non-restated statement starts at -$2,093.2 billion of net operating cost.
Its three signed top-level adjustment totals are +$476.0 billion, -$120.5
billion and -$37.7 billion, producing the reported -$1,775.4 billion budget
deficit with a zero residual.  Summing every detail and subtotal row double
counts the hierarchy and produces -$529.0 billion with a +$1,246.4 billion
residual.

Using the specified 2024-09-30 and 2025-09-30 fiscal-year-end balance endpoints,
debt held by the public increases by $1,970.4541 billion, intragovernmental
holdings by $202.4254 billion and total public debt by $2,172.8796 billion.  At
one-decimal output precision, the debt component identity has a zero residual;
the held-by-public and total increases exceed the reported deficit magnitude by
$195.1 billion and $397.5 billion.  A literal 2024-10-01 fiscal-start endpoint
instead gives $1,965.1374 billion, $3.4687 billion and $1,968.6061 billion for
the three debt changes; calendar-year endpoints give $2,009.7204 billion and
$2,295.4039 billion for the held-by-public and total changes.  The hierarchy and
window alternatives form a 2 x 3 grid; all five non-pinned combinations fall
outside the validator.  Numeric validation uses 0.6 x 10^-N
rounding-boundary tolerance.
"""

from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import validate_ordered_outputs, turn_validator


variables = [
    Variable("net_operating_cost_billion_usd", None, "Store signed net operating cost in USD billions, rounded to 1 decimal."),
    Variable("net_cost_excluded_components_billion_usd", None, "Store the signed total of net-cost components outside the budget deficit in USD billions, rounded to 1 decimal."),
    Variable("budget_only_components_billion_usd", None, "Store the signed total of budget-deficit components outside net operating cost in USD billions, rounded to 1 decimal."),
    Variable("other_reconciling_items_billion_usd", None, "Store the signed total of other reconciling items in USD billions, rounded to 1 decimal."),
    Variable("calculated_budget_deficit_billion_usd", None, "Store the calculated signed budget deficit in USD billions, rounded to 1 decimal."),
    Variable("reported_budget_deficit_billion_usd", None, "Store the reported signed budget deficit in USD billions, rounded to 1 decimal."),
    Variable("bridge_residual_billion_usd", None, "Store calculated minus reported budget deficit in USD billions, rounded to 1 decimal."),
    Variable("debt_held_by_public_change_billion_usd", None, "Store the change in debt held by the public in USD billions, rounded to 1 decimal."),
    Variable("intragovernmental_holdings_change_billion_usd", None, "Store the change in intragovernmental holdings in USD billions, rounded to 1 decimal."),
    Variable("total_public_debt_change_billion_usd", None, "Store the change in total public debt outstanding in USD billions, rounded to 1 decimal."),
    Variable("debt_component_residual_billion_usd", None, "Store total debt change minus the two component changes in USD billions, rounded to 1 decimal."),
    Variable("total_debt_change_minus_deficit_magnitude_billion_usd", None, "Store total public debt change minus the reported budget-deficit magnitude in USD billions, rounded to 1 decimal."),
]


def _statement():
    data = load_expansion_table("treasury_usfr_reconciliations")
    rows = data.loc[
        data.record_date.eq("2025-09-30")
        & data.statement_fiscal_year.eq(2025)
        & data.restatement_flag.eq("N")
    ]
    if len(rows) != 30 or rows.source_line_number.nunique() != len(rows):
        raise ValueError("unexpected FY2025 reconciliation hierarchy")
    return rows.set_index("source_line_number")


def _debt_changes():
    data = load_expansion_table("treasury_debt_to_penny").set_index("record_date")
    required_dates = {"2024-09-30", "2025-09-30"}
    if not required_dates.issubset(data.index):
        raise ValueError("missing specified fiscal-year-end balance endpoint")
    start, end = data.loc["2024-09-30"], data.loc["2025-09-30"]
    held = (end.debt_held_by_public_usd - start.debt_held_by_public_usd) / 1e9
    intragovernmental = (
        end.intragovernmental_holdings_usd - start.intragovernmental_holdings_usd
    ) / 1e9
    total = (
        end.total_public_debt_outstanding_usd
        - start.total_public_debt_outstanding_usd
    ) / 1e9
    return held, intragovernmental, total


def ground_truth():
    rows = _statement()
    start = rows.at[1, "position_billion_usd"]
    net_cost_only = rows.at[19, "position_billion_usd"]
    budget_only = rows.at[26, "position_billion_usd"]
    other = rows.at[29, "position_billion_usd"]
    calculated = start + net_cost_only + budget_only + other
    reported = rows.at[30, "position_billion_usd"]
    held, intragovernmental, total = _debt_changes()
    deficit_magnitude = abs(reported)
    return (
        start, net_cost_only, budget_only, other, calculated, reported,
        calculated - reported, held, intragovernmental, total,
        total - held - intragovernmental,
        total - deficit_magnitude,
    )


def validate(outputs):
    return validate_ordered_outputs(
        outputs, variables, ground_truth(), [1] * len(variables)
    )



TURN_1_NAMES = [
    "net_operating_cost_billion_usd",
    "net_cost_excluded_components_billion_usd",
    "budget_only_components_billion_usd",
    "other_reconciling_items_billion_usd",
    "calculated_budget_deficit_billion_usd",
    "reported_budget_deficit_billion_usd",
    "bridge_residual_billion_usd",
]
TURN_2_NAMES = [
    "debt_held_by_public_change_billion_usd",
    "intragovernmental_holdings_change_billion_usd",
    "total_public_debt_change_billion_usd",
    "debt_component_residual_billion_usd",
]
TURN_3_NAMES = [
    "total_debt_change_minus_deficit_magnitude_billion_usd",
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
    "validate_reconciliation_bridge": turn_validator(_turn_subset(TURN_1_NAMES)),
    "validate_debt_stocks": turn_validator(_turn_subset(TURN_2_NAMES)),
    "validate_deficit_debt_link": turn_validator(_turn_subset(TURN_3_NAMES)),
}
