"""Allowance establishment, covered charge-off, recovery, and CET1 restoration.

This is a transparent internal stress, not a CECL calibration or regulatory
forecast. Its four financial states are provision, charge-off, recovery, and
ending allowance reassessment. A covered charge-off consumes the allowance;
recognizing it again through earnings would double-count the same loss.
"""

from decimal import Decimal
from functools import lru_cache

from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


def D(value):
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("Non-finite source value")
    return result


variables = [
    Variable("opening_net_loans_bn", None, "Store reported net loans and leases in USD billions as a finite number, rounded to 6 decimals from the unrounded source."),
    Variable("opening_allowance_bn", None, "Store the reported allowance in USD billions as a finite number, rounded to 6 decimals from the unrounded source."),
    Variable("opening_noncurrent_bn", None, "Store reported noncurrent loans in USD billions as a finite number, rounded to 6 decimals from the unrounded source."),
    Variable("opening_cet1_pct", None, "Store the calculated opening standardized CET1 ratio in percent as a finite number, rounded to 4 decimals from unrounded inputs."),
    Variable("stress_provision_bn", None, "Store the first-stage signed provision in USD billions as a finite number, rounded to 4 decimals from unrounded inputs."),
    Variable("post_provision_cet1_pct", None, "Store the ratio after the first-stage provision in percent as a finite number, rounded to 4 decimals from unrounded inputs."),
    Variable("charged_off_bn", None, "Store the covered charge-off in USD billions as a finite number, rounded to 4 decimals from unrounded inputs."),
    Variable("post_chargeoff_allowance_bn", None, "Store the allowance after the covered charge-off in USD billions as a finite number, rounded to 4 decimals from unrounded inputs."),
    Variable("cash_recovery_bn", None, "Store the cash recovery in USD billions as a finite number, rounded to 4 decimals from unrounded inputs."),
    Variable("ending_topup_provision_bn", None, "Store the ending reassessment provision in USD billions as a finite number, rounded to 4 decimals from unrounded inputs."),
    Variable("ending_cet1_pct", None, "Store the final standardized CET1 ratio before new capital in percent as a finite number, rounded to 4 decimals from unrounded inputs."),
    Variable("new_cet1_required_bn", None, "Store the minimum new CET1 needed for the opening-ratio target in USD billions as a finite number, rounded to 4 decimals from unrounded inputs."),
]

TURN_1_NAMES = ["opening_net_loans_bn", "opening_allowance_bn", "opening_noncurrent_bn", "opening_cet1_pct"]
TURN_2_NAMES = ["stress_provision_bn", "post_provision_cet1_pct"]
TURN_3_NAMES = ["charged_off_bn", "post_chargeoff_allowance_bn"]
TURN_4_NAMES = ["cash_recovery_bn", "ending_topup_provision_bn", "ending_cet1_pct", "new_cet1_required_bn"]
DECIMALS = [6 if v.name in TURN_1_NAMES[:3] else 4 for v in variables]


def analysis():
    fdic = load_expansion_table("fdic_bankfind")
    row = fdic.loc[fdic.REPDTE.eq("2025-09-30") & fdic.CERT.astype("string").eq("628")]
    if len(row) != 1:
        raise ValueError("Expected one FDIC bank-date record")
    row = row.iloc[0]

    call = load_expansion_table("ffiec_call_reports_capital")
    capital = call.loc[
        call.report_date.eq("2025-09-30")
        & call.bank_id.astype("string").eq("852218")
        & call.capital_reporting_basis.eq("consolidated")
        & call.cblr_election.eq(False)
    ]
    if len(capital) != 1:
        raise ValueError("Expected one consolidated capital report")
    capital = capital.iloc[0]

    net_loans, allowance, noncurrent = [D(row[name]) for name in ("LNLSNET", "LNATRES", "NCLNLS")]
    cet1 = D(capital.common_equity_tier1_capital_thousand_usd)
    rwa = D(capital.standardized_risk_weighted_assets_thousand_usd)
    if min(net_loans, allowance, noncurrent, cet1, rwa) <= 0:
        raise ValueError("Invalid positive banking input")

    target = noncurrent * D("2.25")
    first_provision = target - allowance
    if first_provision <= 0:
        raise ValueError("First-stage target no longer calls for a provision")
    post_provision_capital = cet1 - first_provision * D("0.75")
    chargeoff = target * D("0.40")
    after_chargeoff = target - chargeoff
    recovery = chargeoff * D("0.10")
    after_recovery = after_chargeoff + recovery
    ending_target = net_loans * D("0.02")
    topup = ending_target - after_recovery
    if topup <= 0:
        raise ValueError("Ending target no longer calls for an additional provision")
    final_capital = post_provision_capital - topup * D("0.75")
    if final_capital <= 0:
        raise ValueError("Scenario depleted CET1")
    required = max(D(0), cet1 - final_capital)
    bn = lambda amount: float(amount / D(1000000))
    return {
        "opening_net_loans_bn": bn(net_loans),
        "opening_allowance_bn": bn(allowance),
        "opening_noncurrent_bn": bn(noncurrent),
        "opening_cet1_pct": float(cet1 / rwa * 100),
        "stress_provision_bn": bn(first_provision),
        "post_provision_cet1_pct": float(post_provision_capital / rwa * 100),
        "charged_off_bn": bn(chargeoff),
        "post_chargeoff_allowance_bn": bn(after_chargeoff),
        "cash_recovery_bn": bn(recovery),
        "ending_topup_provision_bn": bn(topup),
        "ending_cet1_pct": float(final_capital / rwa * 100),
        "new_cet1_required_bn": bn(required),
    }


@lru_cache(maxsize=1)
def ground_truth():
    answer = analysis()
    return tuple(answer[v.name] for v in variables)


def _validate_subset(outputs, names):
    by_name = {v.name: v for v in variables}
    truth = dict(zip(by_name, ground_truth()))
    precision = dict(zip(by_name, DECIMALS))
    return validate_ordered_outputs(
        outputs, [by_name[name] for name in names],
        [truth[name] for name in names], [precision[name] for name in names],
    )


def validate_turn_1(outputs): return _validate_subset(outputs, TURN_1_NAMES)
def validate_turn_2(outputs): return _validate_subset(outputs, TURN_2_NAMES)
def validate_turn_3(outputs): return _validate_subset(outputs, TURN_3_NAMES)
def validate_turn_4(outputs): return _validate_subset(outputs, TURN_4_NAMES)
def validate(outputs): return _validate_subset(outputs, [v.name for v in variables])


validators = {
    "validate_opening": turn_validator(validate_turn_1),
    "validate_provision": turn_validator(validate_turn_2),
    "validate_chargeoff": turn_validator(validate_turn_3),
    "validate_reassessment": turn_validator(validate_turn_4),
}
