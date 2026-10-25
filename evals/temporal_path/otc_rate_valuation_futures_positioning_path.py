"""Trace an OTC interest-rate valuation event into Treasury-futures positioning.

The baseline first finds the largest absolute consecutive half-year change in
BIS interest-rate gross market value over 2019-S1 through 2025-S2.  That event
window then governs a fixed five-contract U.S. Treasury futures basket in the
CFTC futures-only report.  The case selects the largest first-to-last change in
leveraged-money net positioning, finds the selected contract's peak absolute
net/open-interest share and applies a ten-percent long-position reduction at
that report.

Gross market value is not notional amount, and the global OTC aggregate is not
the same market, participant population or unit as CFTC contract counts.  The
cross-market path is descriptive and cannot establish a causal hedge response.
The query pins measure, scope, basket, direction, endpoints and tie rules, so
notional substitution, futures-and-options scope, short-minus-long signs and
latest-row selection are regression probes rather than a live trap.  Numeric
validation uses the shared display-precision boundary and one canonical result.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


START_HALF_YEAR = "2019-S1"
END_HALF_YEAR = "2025-S2"
TREASURY_CODES = ("020601", "020604", "042601", "043602", "044601")

TURN_1_NAMES = ["otc_event_half_year", "otc_interest_rate_gmv_change_pct"]
TURN_2_NAMES = ["treasury_contract_code", "treasury_net_position_change_contracts"]
TURN_3_NAMES = ["peak_position_date", "peak_signed_net_open_interest_pct"]
TURN_4_NAMES = ["long_reduction_scenario_net_open_interest_pct"]

variables = [
    Variable(
        TURN_1_NAMES[0], None,
        "Store the selected BIS half-year as an exact YYYY-Sn period token string.",
    ),
    Variable(
        TURN_1_NAMES[1], None,
        "Store the signed consecutive change in interest-rate gross market value in percent, rounded to 4 decimals.",
    ),
    Variable(
        TURN_2_NAMES[0], None,
        "Store the selected six-digit CFTC contract market code as a string.",
    ),
    Variable(
        TURN_2_NAMES[1], None,
        "Store the signed last-minus-first leveraged-money net-position change in contracts as an integer.",
    ),
    Variable(
        TURN_3_NAMES[0], None,
        "Store the selected CFTC report date as an ISO YYYY-MM-DD string.",
    ),
    Variable(
        TURN_3_NAMES[1], None,
        "Store the signed leveraged-money net position as a percent of total open interest, rounded to 4 decimals.",
    ),
    Variable(
        TURN_4_NAMES[0], None,
        "Store the scenario signed leveraged-money net position as a percent of total open interest, rounded to 4 decimals.",
    ),
]

DECIMALS = [None, 4, None, 0, None, 4, 4]


def _half_year_bounds(period: str) -> tuple[str, str]:
    year = int(period[:4])
    if period.endswith("S1"):
        return f"{year}-01-01", f"{year}-06-30"
    if period.endswith("S2"):
        return f"{year}-07-01", f"{year}-12-31"
    raise ValueError(f"invalid half-year {period!r}")


@lru_cache(maxsize=1)
def _trajectory() -> dict[str, object]:
    otc = load_expansion_table("bis_otc_derivatives").copy()
    selected = otc.loc[
        otc["DER_TYPE:Measure"].eq("D: Outstanding - gross market values")
        & otc["DER_INSTR:Instrument"].eq("A: Total (all instruments)")
        & otc["DER_RISK:Risk category"].eq("D: Interest rate")
        & otc["DER_SECTOR_CPY:Counterparty sector"].eq("A: Total (all counterparties)")
        & otc["DER_BASIS:Basis"].eq("C: Net - net")
        & otc["UNIT_MEASURE:Unit of measure"].eq("USD: US dollar")
        & otc["UNIT_MULT:Unit Multiplier"].eq("6: Millions")
        & otc["TIME_PERIOD:Time period or range"].between(START_HALF_YEAR, END_HALF_YEAR)
    ].copy()
    selected = selected.sort_values("TIME_PERIOD:Time period or range")
    if len(selected) != 14:
        raise ValueError(f"expected 14 BIS half-years, found {len(selected)}")
    selected["change_pct"] = selected["OBS_VALUE:Observation Value"].pct_change() * 100
    ranked = selected.dropna(subset=["change_pct"]).assign(
        absolute_change=lambda frame: frame.change_pct.abs()
    ).sort_values(
        ["absolute_change", "TIME_PERIOD:Time period or range"],
        ascending=[False, True],
    )
    event = ranked.iloc[0]
    event_period = str(event["TIME_PERIOD:Time period or range"])

    start_date, end_date = _half_year_bounds(event_period)
    cot = load_expansion_table("cftc_cot").copy()
    window = cot.loc[
        cot.report_scope.eq("futures_only")
        & cot.CFTC_Contract_Market_Code.isin(TREASURY_CODES)
        & cot["Report_Date_as_YYYY-MM-DD"].between(start_date, end_date)
    ].copy()
    window["net"] = (
        window.Lev_Money_Positions_Long_All - window.Lev_Money_Positions_Short_All
    )
    changes = []
    for code, rows in window.groupby("CFTC_Contract_Market_Code", observed=True):
        rows = rows.sort_values("Report_Date_as_YYYY-MM-DD")
        changes.append({
            "code": str(code),
            "change": int(rows.iloc[-1].net - rows.iloc[0].net),
        })
    contract_changes = pd.DataFrame(changes)
    contract_changes["absolute_change"] = contract_changes.change.abs()
    contract_changes = contract_changes.sort_values(
        ["absolute_change", "code"], ascending=[False, True]
    )
    contract = contract_changes.iloc[0]
    contract_rows = window.loc[
        window.CFTC_Contract_Market_Code.eq(contract.code)
    ].copy()
    contract_rows["net_open_interest_pct"] = (
        contract_rows.net / contract_rows.Open_Interest_All * 100
    )
    contract_rows["absolute_share"] = contract_rows.net_open_interest_pct.abs()
    contract_rows = contract_rows.sort_values(
        ["absolute_share", "Report_Date_as_YYYY-MM-DD"],
        ascending=[False, True],
    )
    peak = contract_rows.iloc[0]
    scenario = (
        0.9 * peak.Lev_Money_Positions_Long_All
        - peak.Lev_Money_Positions_Short_All
    ) / peak.Open_Interest_All * 100
    return {
        "event_period": event_period,
        "event_change_pct": float(event.change_pct),
        "contract_code": str(contract.code),
        "contract_change": int(contract.change),
        "peak_date": str(peak["Report_Date_as_YYYY-MM-DD"]),
        "peak_share_pct": float(peak.net_open_interest_pct),
        "scenario_share_pct": float(scenario),
    }


@lru_cache(maxsize=1)
def ground_truth() -> tuple:
    result = _trajectory()
    return (
        result["event_period"], result["event_change_pct"],
        result["contract_code"], result["contract_change"],
        result["peak_date"], result["peak_share_pct"],
        result["scenario_share_pct"],
    )


def _validate_subset(outputs: dict, names: list[str]):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip((variable.name for variable in variables), ground_truth()))
    places = dict(zip((variable.name for variable in variables), DECIMALS))
    return validate_ordered_outputs(
        outputs,
        [by_name[name] for name in names],
        [truth[name] for name in names],
        [places[name] for name in names],
    )


def validate(outputs):
    return _validate_subset(outputs, [variable.name for variable in variables])


validators = {
    "validate_otc_event": turn_validator(lambda outputs: _validate_subset(outputs, TURN_1_NAMES)),
    "validate_contract_move": turn_validator(lambda outputs: _validate_subset(outputs, TURN_2_NAMES)),
    "validate_peak_position": turn_validator(lambda outputs: _validate_subset(outputs, TURN_3_NAMES)),
    "validate_long_sensitivity": turn_validator(lambda outputs: _validate_subset(outputs, TURN_4_NAMES)),
}
