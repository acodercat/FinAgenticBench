"""Bridge the largest FY2025 TGA drawdown through DTS debt flows and debt stocks.

This is a component-crosswalk trap that first becomes live in turn 4.  Total
DTS net issues of -90.084 billion USD reconcile to the -90.084377 billion USD
Debt to the Penny change within 0.377 million USD, but Marketable net issues
are -1.868 billion while debt held by the public changes by -8.791762 billion.
Treating the classifications as identical produces the plausible wrong pair
of 0.000 and 0.000 billion USD for the two component residuals.  The correct
pair is +6.923762 and -6.923385 billion USD.  At three-decimal storage precision
the validator tolerance is 0.0006 billion USD, so it excludes both wrong zeros.
The wrong path changes the two component-residual outputs in the final turn;
there is no later downstream turn.  The query asks for a neutral interpretation
without stating the result.

Numeric validation uses the shared 0.6 x 10^-N rounding-boundary tolerance for
N requested decimals; counts match exactly.
"""

from functools import lru_cache

from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


T1 = [
    "fy2025_tga_reported_date_count", "largest_tga_decline_start_date",
    "largest_tga_decline_end_date", "start_tga_closing_balance_usd_billions",
    "end_tga_closing_balance_usd_billions", "tga_closing_balance_change_usd_billions",
    "tga_decline_magnitude_usd_billions", "tga_decline_leader_margin_usd_billions",
]
T2 = [
    "selected_day_tga_opening_balance_usd_billions",
    "opening_minus_prior_close_usd_millions", "selected_day_tga_deposits_usd_billions",
    "selected_day_tga_withdrawals_usd_billions", "deposits_minus_withdrawals_usd_billions",
    "cash_flow_minus_closing_change_usd_millions",
]
T3 = [
    "marketable_debt_issues_usd_billions", "marketable_debt_redemptions_usd_billions",
    "marketable_debt_net_issues_usd_billions", "nonmarketable_debt_issues_usd_billions",
    "nonmarketable_debt_redemptions_usd_billions", "nonmarketable_debt_net_issues_usd_billions",
    "total_debt_issues_usd_billions", "total_debt_redemptions_usd_billions",
    "total_debt_net_issues_usd_billions",
]
T4 = [
    "debt_stock_start_date", "debt_stock_end_date",
    "held_by_public_start_usd_billions", "held_by_public_end_usd_billions",
    "held_by_public_change_usd_billions",
    "intragovernmental_start_usd_billions", "intragovernmental_end_usd_billions",
    "intragovernmental_change_usd_billions",
    "total_debt_start_usd_billions", "total_debt_end_usd_billions",
    "total_debt_change_usd_billions", "dts_net_issues_minus_total_debt_change_usd_millions",
    "marketable_net_minus_public_change_usd_billions",
    "nonmarketable_net_minus_intragovernmental_change_usd_billions",
    "tga_change_minus_total_debt_change_usd_billions",
]


def _v(name, description):
    return Variable(name, None, description)


variables = [
    _v(T1[0], "Store the FY2025 reported TGA date count as an integer."),
    _v(T1[1], "Store the starting reported date as an ISO YYYY-MM-DD string."),
    _v(T1[2], "Store the ending reported date as an ISO YYYY-MM-DD string."),
    *[_v(name, "Store the TGA balance or signed change in USD billions, rounded to 3 decimals.") for name in T1[3:7]],
    _v(T1[7], "Store the largest-over-runner-up decline-magnitude margin in USD billions, rounded to 3 decimals."),
    _v(T2[0], "Store the selected-day TGA opening balance in USD billions, rounded to 3 decimals."),
    _v(T2[1], "Store opening balance minus prior reported close in USD millions, rounded to 1 decimal."),
    *[_v(name, "Store the daily TGA flow or signed change in USD billions, rounded to 3 decimals.") for name in T2[2:5]],
    _v(T2[5], "Store daily cash-flow net minus closing-balance change in USD millions, rounded to 1 decimal."),
    *[_v(name, "Store the daily debt transaction flow in USD billions, rounded to 3 decimals.") for name in T3],
    _v(T4[0], "Store the starting Debt to the Penny date as an ISO YYYY-MM-DD string."),
    _v(T4[1], "Store the ending Debt to the Penny date as an ISO YYYY-MM-DD string."),
    *[_v(name, "Store the Debt to the Penny stock or signed stock change in USD billions, "
               "rounded to 3 decimals. Take each stock from the unrounded published series and "
               "each change from its own unrounded endpoints, not from the other stocks or "
               "changes reported here.") for name in T4[2:11]],
    _v(T4[11], "Store DTS total net issues minus Debt to the Penny total change in USD millions, rounded to 3 decimals."),
    *[_v(name, "Store the signed cross-classification residual in USD billions, rounded to 3 decimals.") for name in T4[12:]],
]


@lru_cache(maxsize=1)
def ground_truth():
    cash = load_expansion_table("treasury_dts_operating_cash").copy()
    closings = cash.loc[
        cash.record_fiscal_year.eq(2025)
        & cash.account_type.eq("Treasury General Account (TGA) Closing Balance")
    ].sort_values("record_date").copy()
    closings["change"] = closings.today_amount_million_usd.diff()
    ranked = closings.dropna(subset=["change"]).sort_values(
        ["change", "record_date"], ascending=[True, True]
    )
    end = ranked.iloc[0]
    end_position = closings.index.get_loc(end.name)
    start = closings.iloc[end_position - 1]
    runner = ranked.iloc[1]
    decline = float(end.change) / 1000

    day = cash.loc[cash.record_date.eq(end.record_date)].set_index("account_type")
    daily = lambda label: float(day.loc[label, "today_amount_million_usd"]) / 1000
    opening = daily("Treasury General Account (TGA) Opening Balance")
    deposits = daily("Total TGA Deposits (Table II)")
    withdrawals = daily("Total TGA Withdrawals (Table II) (-)")
    flow_net = deposits - withdrawals

    debt_flows = load_expansion_table("treasury_dts_public_debt").loc[
        lambda frame: frame.record_date.eq(end.record_date)
    ]
    grouped = debt_flows.groupby(
        ["transaction_type", "security_market"]
    ).transaction_today_amount_million_usd.sum()
    issue = lambda market: float(grouped.loc[("Issues", market)]) / 1000
    redemption = lambda market: float(grouped.loc[("Redemptions", market)]) / 1000
    market_issues, market_redemptions = issue("Marketable"), redemption("Marketable")
    nonmarket_issues, nonmarket_redemptions = issue("Nonmarketable"), redemption("Nonmarketable")
    total_issues = market_issues + nonmarket_issues
    total_redemptions = market_redemptions + nonmarket_redemptions
    market_net = market_issues - market_redemptions
    nonmarket_net = nonmarket_issues - nonmarket_redemptions
    total_net = total_issues - total_redemptions

    debt = load_expansion_table("treasury_debt_to_penny").copy()
    debt = debt.loc[debt.record_date.isin([start.record_date, end.record_date])].sort_values("record_date")
    if len(debt) != 2:
        raise ValueError("selected DTS dates do not both exist in Debt to the Penny")
    columns = [
        "debt_held_by_public_usd", "intragovernmental_holdings_usd",
        "total_public_debt_outstanding_usd",
    ]
    levels = []
    changes = []
    for column in columns:
        start_value, end_value = float(debt.iloc[0][column]) / 1e9, float(debt.iloc[1][column]) / 1e9
        levels.append((start_value, end_value))
        changes.append(end_value - start_value)
    public_change, intra_change, total_change = changes
    return (
        len(closings), start.record_date, end.record_date,
        float(start.today_amount_million_usd) / 1000,
        float(end.today_amount_million_usd) / 1000, decline, -decline,
        (-float(end.change) + float(runner.change)) / 1000,
        opening, (opening - float(start.today_amount_million_usd) / 1000) * 1000,
        deposits, withdrawals, flow_net, (flow_net - decline) * 1000,
        market_issues, market_redemptions, market_net,
        nonmarket_issues, nonmarket_redemptions, nonmarket_net,
        total_issues, total_redemptions, total_net,
        debt.iloc[0].record_date, debt.iloc[1].record_date,
        levels[0][0], levels[0][1], public_change,
        levels[1][0], levels[1][1], intra_change,
        levels[2][0], levels[2][1], total_change,
        (total_net - total_change) * 1000,
        market_net - public_change, nonmarket_net - intra_change,
        decline - total_change,
    )


DECIMALS = [0, None, None, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 1, *([3] * 9), None, None, *([3] * 9), 3, 3, 3, 3]


def _validate(outputs, names):
    expected = dict(zip((v.name for v in variables), ground_truth()))
    by_name = {v.name: v for v in variables}
    places = dict(zip((v.name for v in variables), DECIMALS))
    return validate_ordered_outputs(outputs, [by_name[n] for n in names], [expected[n] for n in names], [places[n] for n in names])


def validate_turn_1(outputs): return _validate(outputs, T1)
def validate_turn_2(outputs): return _validate(outputs, T2)
def validate_turn_3(outputs): return _validate(outputs, T3)
def validate_turn_4(outputs): return _validate(outputs, T4)
def validate(outputs): return _validate(outputs, [v.name for v in variables])


validators = {
    "validate_tga_screen": turn_validator(validate_turn_1),
    "validate_cash_identity": turn_validator(validate_turn_2),
    "validate_debt_flows": turn_validator(validate_turn_3),
    "validate_stock_flow_bridge": turn_validator(validate_turn_4),
}
