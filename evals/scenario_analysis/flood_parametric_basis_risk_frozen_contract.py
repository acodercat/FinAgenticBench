"""Frozen county flood-index benefits and out-of-sample NFIP reference basis risk.

NOAA damage is only a reported county-month index. NFIP signed net payments
are an insured-claim reference measured by loss month, not total flood damage
or cash paid during that month. The two sources are not event matched.
"""

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from cave_agent import Variable
from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


CASE_ID = "flood_parametric_basis_risk_frozen_contract"
STATE = "LA"
TRAIN_END = "2022-12"
FIPS = ("22019", "22033", "22051", "22103", "22089")
WEIGHT_UNDER = 2.0
BUDGET_SHARE = .15
PREMIUM_LOAD = 1.2


def _v(name, description):
    return Variable(name, None, description)


variables = [
    *[_v(f"county_{j}_fips", f"Store selected county rank {j} as its exact five-character FIPS token.") for j in range(1, 6)],
    _v("train_county_months", "Store the training county-month support count as an integer."),
    _v("train_no_trigger_reference_mn", "Store the positive NFIP reference payment in training county-months with no qualifying NOAA index trigger, summed in USD millions, rounded to 4 decimals."),
    *[_v(f"initial_county_{j}_benefit_mn", f"Store the optimized fixed benefit for selected county rank {j} in USD millions, rounded to 4 decimals.") for j in range(1, 6)],
    _v("initial_premium_mn_per_book_month", "Store the loaded training premium for the five-county book per calendar month in USD millions, rounded to 4 decimals."),
    _v("initial_train_weighted_mse", "Store the optimized training weighted mean squared county-month basis error in squared USD millions, rounded to 4 decimals."),
    _v("validation_negative_basis_mn", "Store validation aggregate positive reference shortfall after the frozen contract, USD millions, rounded to 4 decimals."),
    _v("validation_overpay_mn", "Store validation aggregate positive contract excess above reference, USD millions, rounded to 4 decimals."),
    _v("worst_gap_county_fips", "Store the exact five-character FIPS token for the largest validation county-month shortfall."),
    _v("worst_gap_month", "Store the largest-shortfall validation loss/index month as a YYYY-MM string."),
    _v("worst_gap_reference_mn", "Store NFIP reference payment in the dynamically selected worst validation county-month, USD millions, rounded to 4 decimals."),
    _v("worst_gap_index_mn", "Store NOAA reported property-damage index in the selected worst county-month, USD millions, rounded to 4 decimals."),
    _v("worst_gap_contract_mn", "Store the frozen contract payment in the selected worst county-month, USD millions, rounded to 4 decimals."),
    _v("deleted_train_month", "Store the highest-reference whole five-county training book month as a YYYY-MM string."),
    *[_v(f"revised_county_{j}_benefit_mn", f"Store the revised fixed benefit for selected county rank {j} in USD millions, rounded to 4 decimals.") for j in range(1, 6)],
    _v("revised_validation_negative_basis_mn", "Store validation aggregate positive reference shortfall under the reestimated contract, USD millions, rounded to 4 decimals."),
    _v("revised_validation_weighted_mse", "Store revised validation weighted mean squared county-month basis error, squared USD millions, rounded to 4 decimals."),
    _v("validation_mse_winner", "Store exactly ORIGINAL or REVISED for the lower validation weighted mean squared basis error."),
]
TURN_NAMES = [
    [*(f"county_{j}_fips" for j in range(1, 6)), "train_county_months", "train_no_trigger_reference_mn"],
    [*(f"initial_county_{j}_benefit_mn" for j in range(1, 6)), "initial_premium_mn_per_book_month", "initial_train_weighted_mse"],
    ["validation_negative_basis_mn", "validation_overpay_mn", "worst_gap_county_fips", "worst_gap_month"],
    ["worst_gap_reference_mn", "worst_gap_index_mn", "worst_gap_contract_mn"],
    ["deleted_train_month", *(f"revised_county_{j}_benefit_mn" for j in range(1, 6)), "revised_validation_negative_basis_mn", "revised_validation_weighted_mse", "validation_mse_winner"],
]
DECIMALS = [None]*5 + [0, 4] + [4]*7 + [4, 4, None, None] + [4]*3 + [None] + [4]*5 + [4, 4, None]


@lru_cache(maxsize=1)
def _panel():
    claims = load_expansion_table("fema_nfip")
    claims = claims.loc[
        claims.state_usps.eq(STATE)
        & claims.date_of_loss.between("2019-01-01", "2024-12-31")
        & claims.county_code.str.fullmatch(r"\d{5}", na=False)
    ].copy()
    claims["month"] = claims.date_of_loss.str[:7]
    claims["reference"] = claims[[
        "net_building_payment_usd", "net_contents_payment_usd", "net_icc_payment_usd"
    ]].fillna(0).sum(axis=1).astype(float) / 1e6
    if claims.claim_record_id.duplicated().any() or claims["reference"].isna().any():
        raise ValueError("NFIP claim identity or payment changed")

    storms = load_expansion_table("noaa_storm_events")
    storms = storms.loc[
        storms.state_usps.eq(STATE)
        & storms.county_zone_type.eq("C")
        & storms.event_type.isin(("Flood", "Flash Flood"))
        & storms.begin_datetime.str[:4].between("2019", "2024")
    ].copy()
    storms["county_code"] = storms.state_fips + storms.county_zone_fips.str.zfill(3)
    storms["month"] = storms.begin_datetime.str[:7]
    if storms.event_id.duplicated().any():
        raise ValueError("NOAA event identity changed")
    missing_keys = pd.MultiIndex.from_frame(storms.loc[storms.property_damage_usd.isna(), ["county_code", "month"]].drop_duplicates())
    available = storms.loc[storms.month.le(TRAIN_END)].groupby(["county_code", "month"])["property_damage_usd"].sum(min_count=1)
    available = available.loc[~available.index.isin(missing_keys)]
    positive_counts = available[available.gt(0)].groupby(level=0).size()
    eligible = positive_counts[positive_counts.ge(3)].index
    rank = claims.loc[claims.month.le(TRAIN_END) & claims.county_code.isin(eligible)].groupby("county_code")["reference"].sum()
    rank = rank.sort_index().sort_values(ascending=False, kind="stable")
    selected = tuple(rank.head(5).index)
    if selected != FIPS:
        raise ValueError("Training-only county cohort changed")

    months = pd.period_range("2019-01", "2024-12", freq="M").astype(str)
    grid = pd.MultiIndex.from_product([selected, months], names=["county_code", "month"])
    reference = claims.loc[claims.county_code.isin(selected)].groupby(["county_code", "month"])["reference"].sum().reindex(grid, fill_value=0)
    filtered = storms.loc[storms.county_code.isin(selected)].copy()
    missing_months = filtered.loc[filtered.property_damage_usd.isna(), ["county_code", "month"]].drop_duplicates()
    damage = filtered.groupby(["county_code", "month"])["property_damage_usd"].sum(min_count=1).reindex(grid, fill_value=0) / 1e6
    panel = pd.DataFrame({"reference": reference, "index": damage}).reset_index()
    if not missing_months.empty:
        unavailable = pd.MultiIndex.from_frame(missing_months)
        panel = panel.loc[~pd.MultiIndex.from_frame(panel[["county_code", "month"]]).isin(unavailable)].copy()
    if (len(panel) != 360 or panel[["reference", "index"]].isna().any().any()
            or len(claims.loc[claims.county_code.isin(selected)]) != 15297
            or len(filtered) != 238):
        raise ValueError("Frozen county-month source support changed")
    return panel


def _fit(train):
    threshold = np.array([train.loc[(train.county_code.eq(c)) & (train["index"].gt(0)), "index"].median() for c in FIPS])
    cap = np.array([train.loc[(train.county_code.eq(c)) & (train.reference.gt(0)), "reference"].quantile(.9) for c in FIPS])
    if not np.all(np.isfinite(threshold)) or not np.all(np.isfinite(cap)) or np.any(cap <= 0):
        raise ValueError("Nonidentifiable county threshold or benefit cap")
    trigger = np.column_stack([(train.county_code.eq(c) & train["index"].ge(t)).to_numpy(float) for c, t in zip(FIPS, threshold)])
    counts = trigger.sum(axis=0)
    if np.any(counts <= 0):
        raise ValueError("No positive trigger in a county")
    y = train.reference.to_numpy(float)
    premium_coeff = PREMIUM_LOAD * counts / train.month.nunique()
    budget = BUDGET_SHARE * y.sum() / train.month.nunique()

    def objective(p):
        residual = y - trigger @ p
        weight = np.where(residual >= 0, WEIGHT_UNDER, 1.0)
        return float(np.dot(weight, residual**2) / len(y))

    def gradient(p):
        residual = y - trigger @ p
        weight = np.where(residual >= 0, WEIGHT_UNDER, 1.0)
        return -2 * trigger.T @ (weight * residual) / len(y)

    result = minimize(
        objective, np.minimum(cap/2, 5), jac=gradient, method="SLSQP",
        bounds=list(zip(np.zeros(5), cap)),
        constraints=[{"type": "ineq", "fun": lambda p: budget - premium_coeff @ p,
                      "jac": lambda p: -premium_coeff}],
        options={"ftol": 1e-15, "maxiter": 3000},
    )
    if not result.success or premium_coeff @ result.x > budget + 1e-6 or np.any(result.x < -1e-7):
        raise ValueError(f"County-benefit solve failed: {result.message}")
    return threshold, cap, np.maximum(result.x, 0), float(premium_coeff @ result.x), objective(result.x)


def _evaluate(panel, threshold, benefit):
    payout = np.zeros(len(panel))
    for c, t, b in zip(FIPS, threshold, benefit):
        payout += (panel.county_code.eq(c) & panel["index"].ge(t)).to_numpy(float) * b
    reference = panel.reference.to_numpy(float)
    residual = reference - payout
    return payout, float(np.maximum(residual, 0).sum()), float(np.maximum(-residual, 0).sum()), float(np.mean(np.where(residual >= 0, 2., 1.) * residual**2))


@lru_cache(maxsize=1)
def analysis():
    panel = _panel()
    train = panel.loc[panel.month.le(TRAIN_END)].copy()
    validation = panel.loc[panel.month.gt(TRAIN_END)].copy()
    if len(train) != 240 or len(validation) != 120:
        raise ValueError("Training/validation panel changed")
    threshold, cap, benefit, premium, train_error = _fit(train)
    validation_pay, negative, overpay, original_error = _evaluate(validation, threshold, benefit)
    gap = np.maximum(validation.reference.to_numpy(float) - validation_pay, 0)
    worst_position = int(validation.assign(gap=gap, position=np.arange(len(validation)))
                         .sort_values(["gap", "county_code", "month"], ascending=[False, True, True])
                         .iloc[0].position)
    worst = validation.iloc[worst_position]
    monthly_ref = train.groupby("month").reference.sum().sort_index()
    removed = monthly_ref.idxmax()
    revised_train = train.loc[train.month.ne(removed)]
    revised_threshold, _, revised_benefit, _, _ = _fit(revised_train)
    _, revised_negative, _, revised_error = _evaluate(validation, revised_threshold, revised_benefit)
    county_trigger = np.zeros(len(train), dtype=bool)
    for c, t in zip(FIPS, threshold):
        county_trigger |= (train.county_code.eq(c) & train["index"].ge(t)).to_numpy()
    no_trigger_positive = train.loc[~county_trigger & train.reference.gt(0), "reference"].sum()
    result = {
        **{f"county_{j}_fips": c for j, c in enumerate(FIPS, 1)},
        "train_county_months": len(train),
        "train_no_trigger_reference_mn": float(no_trigger_positive),
        **{f"initial_county_{j}_benefit_mn": float(b) for j, b in enumerate(benefit, 1)},
        "initial_premium_mn_per_book_month": premium,
        "initial_train_weighted_mse": train_error,
        "validation_negative_basis_mn": negative,
        "validation_overpay_mn": overpay,
        "worst_gap_county_fips": str(worst.county_code),
        "worst_gap_month": str(worst.month),
        "worst_gap_reference_mn": float(worst.reference),
        "worst_gap_index_mn": float(worst["index"]),
        "worst_gap_contract_mn": float(validation_pay[worst_position]),
        "deleted_train_month": removed,
        **{f"revised_county_{j}_benefit_mn": float(b) for j, b in enumerate(revised_benefit, 1)},
        "revised_validation_negative_basis_mn": revised_negative,
        "revised_validation_weighted_mse": revised_error,
        "validation_mse_winner": "REVISED" if revised_error < original_error else "ORIGINAL",
    }
    return result


@lru_cache(maxsize=1)
def ground_truth():
    state = analysis()
    return tuple(state[v.name] for v in variables)


def _validate_subset(outputs, names):
    by_name = {v.name: v for v in variables}
    truth = analysis()
    precision = dict(zip((v.name for v in variables), DECIMALS))
    return validate_ordered_outputs(outputs, [by_name[n] for n in names], [truth[n] for n in names], [precision[n] for n in names])


def validate(outputs):
    return _validate_subset(outputs, [v.name for v in variables])


validators = {
    f"validate_turn_{i}": turn_validator(lambda outputs, names=names: _validate_subset(outputs, names))
    for i, names in enumerate(TURN_NAMES, start=1)
}
