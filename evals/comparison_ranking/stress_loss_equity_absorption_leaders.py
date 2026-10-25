"""Scale supervisory stress projections by same-date Y-9C total equity.

Twenty-two holding companies appear in both the 2025 supervisory severely
adverse results and the same-date Y-9C balance sheet.  Their $471.900 billion of
projected total loan losses is 25.4407% of their $1,854.898991 billion of
combined total equity, while the simple mean of the twenty-two entity ratios is
26.3632% -- a -0.9224 point difference, because the ratio of sums weights each
firm by its own equity.  Capital One leads loss-to-equity at 88.3465% (RSSD
2277860, $60.7834 billion of equity against $53.700 billion of losses) and
American Express leads provision-to-equity at 89.8758% (RSSD 1275216).  A stress
projection is a supervisory scenario output, not a forecast, and equity absorbing
a projected loss on paper is not a statement about either firm's solvency.

The pinned-reading comparison measures the Y-9C period, the equity denominator, cohort
aggregation, the loss measure and the CET1 endpoint.  Taking the latest available
Y-9C instead of the stress exercise's own as-of date raises combined equity to
$1,969.255617 billion and flips the aggregation difference to +0.3462 points;
using consolidated assets as the denominator reports 2.3476% rather than
25.4407%; aggregating as a simple mean of the entity ratios reports the cohort
ratio as 26.3632% and the difference as zero by construction; ranking the second
leader on projected loan losses instead of projected provisions moves it from
American Express to Capital One; and reading the stress end-of-horizon CET1
instead of the projected minimum gives 9.3% and 12.7% rather than 9.2% and 9.4%.
The query pins all five, so these are regression probes for a hard baseline
rather than live traps.

Numeric validation uses the shared 0.6 x 10^-N rounding-boundary tolerance for N
requested decimals; the entity count, names and RSSD identifiers match exactly.
"""

from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import validate_ordered_outputs, turn_validator


variables = [
    Variable("matched_stress_y9c_entity_count", None, "Store the matched legal-entity count as an integer."),
    Variable("cohort_y9c_total_equity_usd_billions", None, "Store summed Y-9C consolidated total equity in USD billions, rounded to 3 decimals."),
    Variable("cohort_projected_total_loan_losses_usd_billions", None, "Store summed projected total loan losses in USD billions, rounded to 1 decimal."),
    Variable("cohort_loan_loss_to_equity_ratio_pct", None, "Store the cohort loan-loss-to-equity ratio of sums in percent, rounded to 4 decimals."),
    Variable("simple_mean_entity_loan_loss_to_equity_pct", None, "Store the equal-entity mean loan-loss-to-equity ratio in percent, rounded to 4 decimals."),
    Variable("loan_loss_to_equity_leader_name", None, "Store the loan-loss-to-equity leader legal name; casing and repeated whitespace are not significant."),
    Variable("loan_loss_to_equity_leader_rssd_id", None, "Store the loan-loss-to-equity leader RSSD identifier as text."),
    Variable("loan_loss_leader_y9c_equity_usd_billions", None, "Store the loan-loss leader's Y-9C consolidated total equity in USD billions, rounded to 3 decimals."),
    Variable("loan_loss_leader_projected_total_loan_losses_usd_billions", None, "Store the loan-loss leader's projected total loan losses in USD billions, rounded to 1 decimal."),
    Variable("loan_loss_leader_loss_to_equity_ratio_pct", None, "Store the leading loan-loss-to-equity ratio in percent, rounded to 4 decimals."),
    Variable("loan_loss_leader_actual_cet1_ratio_pct", None, "Store the loan-loss leader's actual CET1 ratio in percent, rounded to 1 decimal."),
    Variable("loan_loss_leader_projected_minimum_cet1_ratio_pct", None, "Store the loan-loss leader's projected-minimum CET1 ratio in percent, rounded to 1 decimal."),
    Variable("provision_to_equity_leader_name", None, "Store the provision-to-equity leader legal name; casing and repeated whitespace are not significant."),
    Variable("provision_to_equity_leader_rssd_id", None, "Store the provision-to-equity leader RSSD identifier as text."),
    Variable("provision_leader_y9c_equity_usd_billions", None, "Store the provision leader's Y-9C consolidated total equity in USD billions, rounded to 3 decimals."),
    Variable("provision_leader_projected_provision_usd_billions", None, "Store the provision leader's projected provision in USD billions, rounded to 1 decimal."),
    Variable("provision_leader_provision_to_equity_ratio_pct", None, "Store the leading provision-to-equity ratio in percent, rounded to 4 decimals."),
    Variable("provision_leader_actual_cet1_ratio_pct", None, "Store the provision leader's actual CET1 ratio in percent, rounded to 1 decimal."),
    Variable("provision_leader_projected_minimum_cet1_ratio_pct", None, "Store the provision leader's projected-minimum CET1 ratio in percent, rounded to 1 decimal."),
]


def _normalized_label(value):
    return " ".join(value.split()).casefold() if isinstance(value, str) else value


def _matched_panel():
    stress = load_expansion_table("fed_stress_tests").loc[
        lambda frame: frame.exercise_name.eq("2025 Stress Test")
        & frame.scenario_name.eq("Supervisory Severely Adverse")
        & frame.id_rssd.notna()
    ].copy()
    y9c = load_expansion_table("ffiec_y9c").loc[
        lambda frame: frame.report_date.eq("2024-12-31")
        & frame.consolidated_total_equity_thousand_usd.notna()
    ].copy()
    if stress.id_rssd.duplicated().any() or y9c.reporter_rssd_id.duplicated().any():
        raise ValueError("duplicate RSSD identifier in the selected source population")
    panel = stress.merge(
        y9c[[
            "reporter_rssd_id",
            "reporter_legal_name",
            "consolidated_total_equity_thousand_usd",
        ]],
        left_on="id_rssd",
        right_on="reporter_rssd_id",
        how="inner",
        validate="one_to_one",
    )
    if len(panel) != 22:
        raise ValueError("expected all 22 legal-entity stress rows to match Y-9C")
    panel["y9c_equity_usd_billions"] = (
        panel.consolidated_total_equity_thousand_usd / 1e6
    )
    panel["loan_loss_to_equity_pct"] = (
        panel.loss_total_loan_amt / panel.y9c_equity_usd_billions * 100
    )
    panel["provision_to_equity_pct"] = (
        panel.provision_amt / panel.y9c_equity_usd_billions * 100
    )
    return panel


def ground_truth():
    panel = _matched_panel()
    loss_leader = panel.sort_values(
        ["loan_loss_to_equity_pct", "disclosure_legal_name"],
        ascending=[False, True],
    ).iloc[0]
    provision_leader = panel.sort_values(
        ["provision_to_equity_pct", "disclosure_legal_name"],
        ascending=[False, True],
    ).iloc[0]
    equity_total = panel.y9c_equity_usd_billions.sum()
    loss_total = panel.loss_total_loan_amt.sum()
    cohort_ratio = loss_total / equity_total * 100
    simple_mean = panel.loan_loss_to_equity_pct.mean()
    return (
        len(panel),
        equity_total,
        loss_total,
        cohort_ratio,
        simple_mean,
        loss_leader.disclosure_legal_name,
        loss_leader.id_rssd,
        loss_leader.y9c_equity_usd_billions,
        loss_leader.loss_total_loan_amt,
        loss_leader.loan_loss_to_equity_pct,
        loss_leader.common_equity_tier1_actual_rat,
        loss_leader.common_equity_tier1_min_rat,
        provision_leader.disclosure_legal_name,
        provision_leader.id_rssd,
        provision_leader.y9c_equity_usd_billions,
        provision_leader.provision_amt,
        provision_leader.provision_to_equity_pct,
        provision_leader.common_equity_tier1_actual_rat,
        provision_leader.common_equity_tier1_min_rat,
    )


DECIMALS = [0, 3, 1, 4, 4, None, None, 3, 1, 4, 1, 1, None, None, 3, 1, 4, 1, 1]
LEGAL_NAME_OUTPUTS = ("loan_loss_to_equity_leader_name", "provision_to_equity_leader_name")
RSSD_OUTPUTS = ("loan_loss_to_equity_leader_rssd_id", "provision_to_equity_leader_rssd_id")


def validate(outputs):
    expected = ground_truth()
    expected_by_name = dict(zip((variable.name for variable in variables), expected))
    candidate = dict(outputs)
    for name in LEGAL_NAME_OUTPUTS:
        if _normalized_label(candidate.get(name)) == _normalized_label(expected_by_name[name]):
            candidate[name] = expected_by_name[name]
    for name in RSSD_OUTPUTS:
        if str(candidate.get(name, "")).strip() == str(expected_by_name[name]):
            candidate[name] = expected_by_name[name]
    return validate_ordered_outputs(candidate, variables, expected, DECIMALS)



TURN_1_NAMES = [
    "matched_stress_y9c_entity_count",
    "cohort_y9c_total_equity_usd_billions",
    "cohort_projected_total_loan_losses_usd_billions",
    "cohort_loan_loss_to_equity_ratio_pct",
    "simple_mean_entity_loan_loss_to_equity_pct",
]
TURN_2_NAMES = [
    "loan_loss_to_equity_leader_name",
    "loan_loss_to_equity_leader_rssd_id",
    "loan_loss_leader_y9c_equity_usd_billions",
    "loan_loss_leader_projected_total_loan_losses_usd_billions",
    "loan_loss_leader_loss_to_equity_ratio_pct",
    "loan_loss_leader_actual_cet1_ratio_pct",
    "loan_loss_leader_projected_minimum_cet1_ratio_pct",
]
TURN_3_NAMES = [
    "provision_to_equity_leader_name",
    "provision_to_equity_leader_rssd_id",
    "provision_leader_y9c_equity_usd_billions",
    "provision_leader_projected_provision_usd_billions",
    "provision_leader_provision_to_equity_ratio_pct",
    "provision_leader_actual_cet1_ratio_pct",
    "provision_leader_projected_minimum_cet1_ratio_pct",
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
    "validate_cohort_scaling": turn_validator(_turn_subset(TURN_1_NAMES)),
    "validate_loss_leader": turn_validator(_turn_subset(TURN_2_NAMES)),
    "validate_provision_leader": turn_validator(_turn_subset(TURN_3_NAMES)),
}
