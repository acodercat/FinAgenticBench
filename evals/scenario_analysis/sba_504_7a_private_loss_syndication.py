"""FY2025 SBA 504/7(a) approval-exposure proxy and hypothetical risk transfer.

504 third-party financing is not the SBA-guaranteed CDC debenture. Seven(a)
lender-retained approval is gross less the explicitly guaranteed amount. The
combined proxy joins two unlike programs at state/NAICS-sector grain, not at
borrower or bank level. Approvals are not outstanding balances. Source-native
rows have no guaranteed unique loan ID and are intentionally not deduplicated.

Measured wrong path: using 7(a) gross instead of retained principal makes the
CA/72 private exposure USD980.587173m instead of USD528.98918422m. Summing
504 SBA gross instead of third-party principal changes the same group's 504
private component from USD374.68097322m to USD239.492m. Neither is accepted.
The last turn uses whole-row sales; a fractional-dollar cure is not the stated
execution policy.
"""

from functools import lru_cache

import numpy as np
import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


STATES = set("AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC".split())
TURNS = [
    ["matched_state_sectors", "leader_state", "leader_naics2", "leader_private_approval_m"],
    ["leader_504_rows", "leader_7a_rows", "leader_504_private_share_pct", "leader_7a_guarantee_share_pct"],
    ["modeled_private_loss_m", "seven_a_loss_share_pct"],
    ["minimum_whole_504_sales", "sold_504_thirdparty_m"],
]
variables = [
    Variable("matched_state_sectors", None, "Store the count of matched state-sector pairs as a finite integer number."),
    Variable("leader_state", None, "Store the two-letter USPS state abbreviation as an exact uppercase string."),
    Variable("leader_naics2", None, "Store the two-digit NAICS sector code as an exact string, preserving leading zeros."),
    Variable("leader_private_approval_m", None, "Store combined private approval exposure in USD millions, rounded to 4 decimals from unrounded rows."),
    Variable("leader_504_rows", None, "Store the number of eligible 504 reported rows as a finite integer number."),
    Variable("leader_7a_rows", None, "Store the number of eligible 7(a) reported rows as a finite integer number."),
    Variable("leader_504_private_share_pct", None, "Store the 504 share of combined private approval exposure in percent, rounded to 4 decimals from unrounded rows."),
    Variable("leader_7a_guarantee_share_pct", None, "Store the 7(a) guaranteed share of 7(a) gross approvals in percent, rounded to 4 decimals from unrounded rows."),
    Variable("modeled_private_loss_m", None, "Store the combined hypothetical private principal loss in USD millions, rounded to 3 decimals from unrounded rows."),
    Variable("seven_a_loss_share_pct", None, "Store the 7(a) share of that hypothetical loss in percent, rounded to 3 decimals from unrounded rows."),
    Variable("minimum_whole_504_sales", None, "Store the count of whole 504 reported rows sold as a finite integer number."),
    Variable("sold_504_thirdparty_m", None, "Store their total third-party dollars in USD millions, rounded to 4 decimals from unrounded rows."),
]
DECIMALS = [0, None, None, 4, 0, 0, 4, 4, 3, 3, 0, 4]


def _prepare(name, data):
    rows = data.loc[data["ApprovalFY"].astype(str).eq("2025") & data["ProjectState"].isin(STATES)].copy()
    rows["sector"] = rows["NaicsCode"].astype(str).str[:2]
    rows = rows.loc[rows["sector"].str.fullmatch(r"\d{2}")].copy()
    for column in (["GrossApproval", "ThirdPartyDollars"] if name == "sba_504" else ["GrossApproval", "SBAGuaranteedApproval"]):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    if name == "sba_504":
        rows = rows.loc[np.isfinite(rows["GrossApproval"]) & rows["GrossApproval"].gt(0) & np.isfinite(rows["ThirdPartyDollars"]) & rows["ThirdPartyDollars"].gt(0)].copy()
        rows["private"] = rows["ThirdPartyDollars"]
    else:
        rows = rows.loc[np.isfinite(rows["GrossApproval"]) & rows["GrossApproval"].gt(0) & np.isfinite(rows["SBAGuaranteedApproval"]) & rows["SBAGuaranteedApproval"].ge(0) & rows["SBAGuaranteedApproval"].le(rows["GrossApproval"])].copy()
        rows["private"] = rows["GrossApproval"] - rows["SBAGuaranteedApproval"]
    return rows


@lru_cache(maxsize=1)
def analysis():
    five = _prepare("sba_504", load_expansion_table("sba_504"))
    seven = _prepare("sba_7a", load_expansion_table("sba_7a"))
    keys = ["ProjectState", "sector"]
    five_group = five.groupby(keys, sort=True).agg(five_rows=("private", "size"), five_private=("private", "sum"))
    seven_group = seven.groupby(keys, sort=True).agg(seven_rows=("private", "size"), seven_private=("private", "sum"))
    joined = five_group.join(seven_group, how="inner")
    joined["combined"] = joined["five_private"] + joined["seven_private"]
    joined = joined.sort_values("combined", ascending=False, kind="stable")
    state, sector = map(str, joined.index[0])
    a = five.loc[five["ProjectState"].eq(state) & five["sector"].eq(sector)]
    b = seven.loc[seven["ProjectState"].eq(state) & seven["sector"].eq(sector)]
    five_private = float(a["private"].sum())
    seven_private = float(b["private"].sum())
    combined = five_private + seven_private
    five_loss = 0.08 * 0.50 * five_private
    seven_loss = 0.12 * 0.50 * seven_private
    total_loss = five_loss + seven_loss
    ordered = a["private"].sort_values(ascending=False, kind="stable").to_numpy(dtype=float)
    remaining = total_loss - 0.04 * np.cumsum(ordered)
    qualifying = np.flatnonzero(remaining <= 20_000_000)
    if len(qualifying) == 0:
        raise ValueError("whole-row syndication cannot meet reserve")
    count = int(qualifying[0] + 1)
    if count > 1 and remaining[count - 2] <= 20_000_000:
        raise ValueError("sale boundary not minimal")
    return {
        "matched_state_sectors": len(joined),
        "leader_state": state,
        "leader_naics2": sector,
        "leader_private_approval_m": combined / 1_000_000,
        "leader_504_rows": len(a),
        "leader_7a_rows": len(b),
        "leader_504_private_share_pct": 100 * five_private / combined,
        "leader_7a_guarantee_share_pct": 100 * b["SBAGuaranteedApproval"].sum() / b["GrossApproval"].sum(),
        "modeled_private_loss_m": total_loss / 1_000_000,
        "seven_a_loss_share_pct": 100 * seven_loss / total_loss,
        "minimum_whole_504_sales": count,
        "sold_504_thirdparty_m": ordered[:count].sum() / 1_000_000,
    }


@lru_cache(maxsize=1)
def ground_truth():
    answers = analysis()
    return tuple(answers[v.name] for v in variables)


def _validate(outputs, names):
    by_name = {v.name: v for v in variables}
    truth = dict(zip(by_name, ground_truth()))
    decimals = {v.name: d for v, d in zip(variables, DECIMALS)}
    return validate_ordered_outputs(outputs, [by_name[n] for n in names], [truth[n] for n in names], [decimals[n] for n in names])


def validate(outputs):
    return _validate(outputs, [v.name for v in variables])


validators = {f"validate_turn_{i}": turn_validator(lambda outputs, names=names: _validate(outputs, names)) for i, names in enumerate(TURNS, 1)}
