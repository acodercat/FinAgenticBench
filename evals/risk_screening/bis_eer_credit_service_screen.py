"""Screen a cross-market credit-service cohort and carry the selected economy forward.

The canonical 2025-Q4/common-vintage population contains 17 economies with actual
private credit, credit gap, all three DSR sectors, and 2024-12/2025-12 broad nominal
and real EER observations.  Three pass the pinned leverage/service/real-EER screen.
Japan has the most negative nominal EER percentage change, narrowly ahead of Korea.

The measured interactions cover quarter choice, reported-private versus an arithmetic
sector mean, percentage versus index-point EER change, nominal versus real ranking,
and actual versus trend credit level.  Replacing the published private-sector DSR
with the household/corporate mean admits the United States and changes the leader;
the query explicitly names the reported private-sector series, so this is a hard
baseline regression path rather than a live trap.  Numeric validation uses the
shared rounding-boundary tolerance; counts, ranks and labels are exact.

Fragility: the selected economy leads the runner-up by 0.0062 percentage points
on the nominal effective-exchange-rate change, so a BIS revision would change
the leader and every downstream output; the frozen artifact fixes the vintage.
"""

from functools import lru_cache

from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import validate_ordered_outputs, turn_validator


COUNTRY = "BORROWERS_CTY:Borrowers' country"
PERIOD = "TIME_PERIOD:Time period or range"
VALUE = "OBS_VALUE:Observation Value"
ACTUAL = "A: Credit-to-GDP ratios (actual data)"
GAP = "C: Credit-to-GDP gaps (actual-trend)"
HOUSEHOLD = "H: Households & NPISHs"
CORPORATE = "N: Non-financial corporations"
PRIVATE = "P: Private non-financial sector"
NOMINAL = "N: Nominal"
REAL = "R: Real"


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = [
    "joint_economy_count", "screened_economy_count", "selected_economy",
    "nominal_eer_leader_margin_pp",
]
TURN_2_NAMES = [
    "selected_actual_credit_to_gdp_pct", "selected_credit_gap_pp",
    "selected_actual_credit_rank", "selected_credit_gap_rank",
]
TURN_3_NAMES = [
    "selected_household_dsr_2024_q4_pct", "selected_household_dsr_2025_q4_pct",
    "selected_household_dsr_change_pp", "selected_corporate_dsr_2024_q4_pct",
    "selected_corporate_dsr_2025_q4_pct", "selected_corporate_dsr_change_pp",
    "selected_private_dsr_2024_q4_pct", "selected_private_dsr_2025_q4_pct",
    "selected_private_dsr_change_pp", "selected_sector_mean_dsr_2025_q4_pct",
]
TURN_4_NAMES = [
    "selected_nominal_eer_2024_12", "selected_nominal_eer_2025_12",
    "selected_nominal_eer_change_pct", "selected_real_eer_2024_12",
    "selected_real_eer_2025_12", "selected_real_eer_change_pct",
    "selected_nominal_eer_decline_rank", "selected_real_eer_decline_rank",
]

variables = [
    _v("joint_economy_count", "Store the common-economy count as an integer."),
    _v("screened_economy_count", "Store the count passing the joint screen as an integer."),
    _v("selected_economy", "Store the selected economy display name as a string; an optional two-letter source prefix is accepted."),
    _v("nominal_eer_leader_margin_pp", "Store the runner-up nominal-EER percentage change minus the selected economy's change in percentage points, rounded to 4 decimals."),
    _v("selected_actual_credit_to_gdp_pct", "Store the selected economy's actual credit-to-GDP ratio in percent of GDP, rounded to 4 decimals."),
    _v("selected_credit_gap_pp", "Store the selected economy's actual-minus-trend credit gap in percentage points, rounded to 4 decimals."),
    _v("selected_actual_credit_rank", "Store the selected economy's descending actual-credit rank in the common population as an integer."),
    _v("selected_credit_gap_rank", "Store the selected economy's descending credit-gap rank in the common population as an integer."),
    _v("selected_household_dsr_2024_q4_pct", "Store the selected economy's 2024-Q4 household DSR in percent, rounded to 1 decimal."),
    _v("selected_household_dsr_2025_q4_pct", "Store the selected economy's 2025-Q4 household DSR in percent, rounded to 1 decimal."),
    _v("selected_household_dsr_change_pp", "Store the household DSR change in percentage points, rounded to 1 decimal."),
    _v("selected_corporate_dsr_2024_q4_pct", "Store the selected economy's 2024-Q4 non-financial-corporate DSR in percent, rounded to 1 decimal."),
    _v("selected_corporate_dsr_2025_q4_pct", "Store the selected economy's 2025-Q4 non-financial-corporate DSR in percent, rounded to 1 decimal."),
    _v("selected_corporate_dsr_change_pp", "Store the non-financial-corporate DSR change in percentage points, rounded to 1 decimal."),
    _v("selected_private_dsr_2024_q4_pct", "Store the selected economy's 2024-Q4 reported private-sector DSR in percent, rounded to 1 decimal."),
    _v("selected_private_dsr_2025_q4_pct", "Store the selected economy's 2025-Q4 reported private-sector DSR in percent, rounded to 1 decimal."),
    _v("selected_private_dsr_change_pp", "Store the reported private-sector DSR change in percentage points, rounded to 1 decimal."),
    _v("selected_sector_mean_dsr_2025_q4_pct", "Store the arithmetic mean of the selected economy's 2025-Q4 household and corporate DSRs in percent, rounded to 2 decimals."),
    _v("selected_nominal_eer_2024_12", "Store the selected economy's 2024-12 broad nominal EER index, rounded to 2 decimals."),
    _v("selected_nominal_eer_2025_12", "Store the selected economy's 2025-12 broad nominal EER index, rounded to 2 decimals."),
    _v("selected_nominal_eer_change_pct", "Store the selected economy's 2024-12-to-2025-12 nominal EER percentage change, rounded to 4 decimals."),
    _v("selected_real_eer_2024_12", "Store the selected economy's 2024-12 broad real EER index, rounded to 2 decimals."),
    _v("selected_real_eer_2025_12", "Store the selected economy's 2025-12 broad real EER index, rounded to 2 decimals."),
    _v("selected_real_eer_change_pct", "Store the selected economy's 2024-12-to-2025-12 real EER percentage change, rounded to 4 decimals."),
    _v("selected_nominal_eer_decline_rank", "Store the selected economy's nominal-EER-change rank from most negative in the common population as an integer."),
    _v("selected_real_eer_decline_rank", "Store the selected economy's real-EER-change rank from most negative in the common population as an integer."),
]


def _display(value):
    text = " ".join(str(value).split())
    prefix, separator, remainder = text.partition(": ")
    return remainder if separator and len(prefix) == 2 else text


def _population():
    credit = load_expansion_table("bis_credit_gap")
    credit = credit.loc[
        credit["TC_BORROWERS:Borrowing sector"].eq(PRIVATE)
        & credit["TC_LENDERS:Lending sector"].str.startswith("A:")
        & credit[PERIOD].eq("2025-Q4")
    ].pivot_table(index=COUNTRY, columns="CG_DTYPE:Credit gap data type", values=VALUE, aggfunc="first")
    credit = credit.dropna(subset=[ACTUAL, GAP])[[ACTUAL, GAP]]
    credit.index = credit.index.map(_display)

    dsr = load_expansion_table("bis_debt_service")
    dsr_levels = dsr.loc[dsr[PERIOD].isin(["2024-Q4", "2025-Q4"])].pivot_table(
        index=[COUNTRY, "DSR_BORROWERS:Borrowers"], columns=PERIOD,
        values=VALUE, aggfunc="first",
    ).reset_index()
    dsr_levels[COUNTRY] = dsr_levels[COUNTRY].map(_display)
    dsr_2025 = dsr_levels.pivot(index=COUNTRY, columns="DSR_BORROWERS:Borrowers", values="2025-Q4")
    dsr_2025 = dsr_2025.dropna(subset=[HOUSEHOLD, CORPORATE, PRIVATE])

    eer = load_expansion_table("bis_effective_exchange_rates")
    eer_levels = eer.loc[eer["EER_TYPE:Type"].isin([NOMINAL, REAL])].pivot_table(
        index=["REF_AREA:Reference area", "EER_TYPE:Type"], columns=PERIOD,
        values=VALUE, aggfunc="first",
    ).dropna(subset=["2024-12", "2025-12"])
    eer_levels["change_pct"] = (eer_levels["2025-12"] / eer_levels["2024-12"] - 1) * 100
    eer_change = eer_levels["change_pct"].unstack("EER_TYPE:Type")
    eer_change.index = eer_change.index.map(_display)

    population = credit.join(dsr_2025).join(eer_change).dropna().reset_index()
    if len(population) != 17 or population[COUNTRY].duplicated().any():
        raise ValueError("unexpected BIS common population")
    return population, dsr_levels, eer_levels


def _rank(population, column, *, ascending):
    ordered = population.sort_values([column, COUNTRY], ascending=[ascending, True]).reset_index(drop=True)
    return {_display(row[COUNTRY]): rank for rank, (_, row) in enumerate(ordered.iterrows(), 1)}


@lru_cache(maxsize=1)
def ground_truth():
    population, dsr_levels, eer_levels = _population()
    screened = population.loc[
        population[ACTUAL].ge(100)
        & population[PRIVATE].ge(15)
        & population[REAL].lt(0)
    ].sort_values([NOMINAL, COUNTRY], ascending=[True, True]).reset_index(drop=True)
    if len(screened) != 3 or len(screened) < 2:
        raise ValueError("unexpected BIS screened population")
    leader, runner = screened.iloc[0], screened.iloc[1]
    selected = _display(leader[COUNTRY])

    actual_ranks = _rank(population, ACTUAL, ascending=False)
    gap_ranks = _rank(population, GAP, ascending=False)
    nominal_ranks = _rank(population, NOMINAL, ascending=True)
    real_ranks = _rank(population, REAL, ascending=True)

    selected_dsr = dsr_levels.loc[dsr_levels[COUNTRY].eq(selected)].set_index("DSR_BORROWERS:Borrowers")
    hh_24, hh_25 = selected_dsr.loc[HOUSEHOLD, ["2024-Q4", "2025-Q4"]]
    co_24, co_25 = selected_dsr.loc[CORPORATE, ["2024-Q4", "2025-Q4"]]
    pr_24, pr_25 = selected_dsr.loc[PRIVATE, ["2024-Q4", "2025-Q4"]]
    sector_mean = (hh_25 + co_25) / 2

    selected_eer = eer_levels.reset_index()
    selected_eer["display"] = selected_eer["REF_AREA:Reference area"].map(_display)
    selected_eer = selected_eer.loc[selected_eer.display.eq(selected)].set_index("EER_TYPE:Type")
    nom = selected_eer.loc[NOMINAL]
    real = selected_eer.loc[REAL]

    return (
        len(population),
        len(screened),
        selected,
        float(runner[NOMINAL] - leader[NOMINAL]),
        float(leader[ACTUAL]),
        float(leader[GAP]),
        actual_ranks[selected],
        gap_ranks[selected],
        float(hh_24),
        float(hh_25),
        float(hh_25 - hh_24),
        float(co_24),
        float(co_25),
        float(co_25 - co_24),
        float(pr_24),
        float(pr_25),
        float(pr_25 - pr_24),
        float(sector_mean),
        float(nom["2024-12"]),
        float(nom["2025-12"]),
        float(nom.change_pct),
        float(real["2024-12"]),
        float(real["2025-12"]),
        float(real.change_pct),
        nominal_ranks[selected],
        real_ranks[selected],
    )


DECIMALS = [
    0, 0, None, 4,
    4, 4, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
    2, 2, 4, 2, 2, 4, 0, 0,
]


def _validate_subset(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip(by_name, ground_truth()))
    decimals = dict(zip(by_name, DECIMALS))
    candidate = dict(outputs)
    if "selected_economy" in candidate and _display(candidate["selected_economy"]).casefold() == _display(truth["selected_economy"]).casefold():
        candidate["selected_economy"] = truth["selected_economy"]
    return validate_ordered_outputs(candidate, [by_name[n] for n in names], [truth[n] for n in names], [decimals[n] for n in names])


def validate_turn_1(outputs): return _validate_subset(outputs, TURN_1_NAMES)
def validate_turn_2(outputs): return _validate_subset(outputs, TURN_2_NAMES)
def validate_turn_3(outputs): return _validate_subset(outputs, TURN_3_NAMES)
def validate_turn_4(outputs): return _validate_subset(outputs, TURN_4_NAMES)
def validate(outputs): return _validate_subset(outputs, [v.name for v in variables])


validators = {
    "validate_joint_screen": turn_validator(validate_turn_1),
    "validate_credit_position": turn_validator(validate_turn_2),
    "validate_dsr_structure": turn_validator(validate_turn_3),
    "validate_eer_path": turn_validator(validate_turn_4),
}
