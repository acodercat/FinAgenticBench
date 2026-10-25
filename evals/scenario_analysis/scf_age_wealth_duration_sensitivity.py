"""Carry an SCF age-cohort wealth concentration result into a Treasury-rate shock.

The five SCF implicates are multiple-imputation records, not five independent
households.  The query pins implicate-wise estimates and their arithmetic mean,
so the case is a hard baseline.  Its difficulty is the weighted distribution,
result-dependent cohort handoff, path-dependent yield draw-up and duration
sensitivity.  The final loss is a mechanical first-order scenario, not an
observed household portfolio loss.

The pinned-reading comparison measures implicate handling for the cohort summary and for
the distribution, survey versus equal weighting, the shocked tenor, the yield
draw-up path and the exposure statistic.  Dropping the survey weights is the
largest departure: the weighted household count falls from 109.603096 million to
0.019885 million records and the selected cohort's wealth share rises from
33.4026% to 45.4138%.  Shocking the five-year instead of the ten-year par yield
selects a 2025-05-14 endpoint and a 0.4500-point draw-up rather than 2025-05-21
and 0.5700; taking the largest unordered yield pair instead of the ordered
draw-up selects 2025-10-22 to 2025-01-13; and using the mean rather than the
median exposure reports a $19,792.43 duration loss instead of $4,572.24.  Pooling
the implicates instead of averaging their estimates moves the cohort's wealth
share from 33.4026% to 33.3990% and its selection margin from 2.6257 to 2.6206
points, and reports a $410,000 weighted median net worth instead of $411,358.
The query pins every one of these.
"""

from functools import lru_cache

import numpy as np
import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import validate_ordered_outputs, turn_validator


BANDS = [(25, 35, "25-34"), (35, 45, "35-44"), (45, 55, "45-54"),
         (55, 65, "55-64"), (65, 75, "65-74")]


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = [
    "eligible_scf_household_count", "survey_weighted_household_count_millions",
    "selected_age_band", "selected_household_share_pct", "selected_wealth_share_pct",
    "selection_margin_pp",
]
TURN_2_NAMES = [
    "selected_age_band_implicate_row_count", "selected_age_band_household_count",
    "selected_age_band_weighted_median_net_worth_usd",
    "selected_age_band_weighted_mean_net_worth_usd",
    "selected_age_band_negative_net_worth_share_pct",
]
TURN_3_NAMES = [
    "treasury_2025_observation_count", "ten_year_drawup_start_date",
    "ten_year_drawup_end_date", "ten_year_drawup_start_yield_pct",
    "ten_year_drawup_end_yield_pct", "ten_year_drawup_calendar_days",
    "median_household_bond_allocation_usd",
]
TURN_4_NAMES = [
    "median_household_duration_loss_usd", "median_household_duration_loss_to_net_worth_pct",
    "median_household_post_shock_net_worth_usd",
    "selected_cohort_aggregate_duration_loss_usd_billions",
]

variables = [
    _v("eligible_scf_household_count", "Store the number of distinct eligible SCF households as an integer."),
    _v("survey_weighted_household_count_millions", "Store the survey-weighted eligible household count in millions, rounded to 6 decimals."),
    _v("selected_age_band", "Store the selected age band as a string."),
    _v("selected_household_share_pct", "Store the selected band's survey-weighted household share in percent, rounded to 4 decimals."),
    _v("selected_wealth_share_pct", "Store the selected band's survey-weighted aggregate net-worth share in percent, rounded to 4 decimals."),
    _v("selection_margin_pp", "Store the selected gap minus the runner-up gap in percentage points, rounded to 4 decimals."),
    _v("selected_age_band_implicate_row_count", "Store the selected band's implicate-row count as an integer."),
    _v("selected_age_band_household_count", "Store the selected band's distinct household count as an integer."),
    _v("selected_age_band_weighted_median_net_worth_usd", "Store the selected band's weighted median net worth in USD, rounded to 2 decimals."),
    _v("selected_age_band_weighted_mean_net_worth_usd", "Store the selected band's weighted mean net worth in USD, rounded to 2 decimals."),
    _v("selected_age_band_negative_net_worth_share_pct", "Store the selected band's survey-weighted negative-net-worth share in percent, rounded to 4 decimals."),
    _v("treasury_2025_observation_count", "Store the number of reported 2025 Treasury curve observations as an integer."),
    _v("ten_year_drawup_start_date", "Store the selected starting date as an ISO YYYY-MM-DD string."),
    _v("ten_year_drawup_end_date", "Store the selected ending date as an ISO YYYY-MM-DD string."),
    _v("ten_year_drawup_start_yield_pct", "Store the starting 10-year par yield in percent, rounded to 3 decimals."),
    _v("ten_year_drawup_end_yield_pct", "Store the ending 10-year par yield in percent, rounded to 3 decimals."),
    _v("ten_year_drawup_calendar_days", "Store the elapsed calendar-day count as an integer."),
    _v("median_household_bond_allocation_usd", "Store the scenario bond allocation in USD, rounded to 2 decimals."),
    _v("median_household_duration_loss_usd", "Store the positive estimated scenario loss in USD, rounded to 2 decimals."),
    _v("median_household_duration_loss_to_net_worth_pct", "Store the estimated loss divided by median net worth in percent, rounded to 4 decimals."),
    _v("median_household_post_shock_net_worth_usd", "Store median net worth after the estimated loss in USD, rounded to 2 decimals."),
    _v("selected_cohort_aggregate_duration_loss_usd_billions", "Store the selected cohort's aggregate estimated scenario loss in USD billions, rounded to 6 decimals."),
]


def _weighted_median(values, weights):
    order = np.argsort(values, kind="mergesort")
    values = np.asarray(values)[order]
    weights = np.asarray(weights)[order]
    return float(values[np.searchsorted(np.cumsum(weights), weights.sum() / 2, side="left")])


@lru_cache(maxsize=1)
def _analysis():
    scf = load_expansion_table("fed_scf").copy()
    scf = scf.loc[scf.reference_person_age.between(25, 74)].copy()
    scf["age_band"] = pd.cut(
        scf.reference_person_age, [25, 35, 45, 55, 65, 75], right=False,
        labels=[band[2] for band in BANDS],
    ).astype("string")
    scf["analysis_weight"] = scf.survey_weight
    scf["weighted_wealth"] = scf.analysis_weight * scf.net_worth_usd
    if not scf.groupby("household_id").implicate.nunique().eq(5).all():
        raise ValueError("eligible SCF households must retain five implicates")
    shares = []
    for implicate, frame in scf.groupby("implicate"):
        for band, group in frame.groupby("age_band"):
            shares.append({
                "implicate": implicate, "age_band": band,
                "household_share": float(group.analysis_weight.sum() / frame.analysis_weight.sum()),
                "wealth_share": float(group.weighted_wealth.sum() / frame.weighted_wealth.sum()),
            })
    ranking = pd.DataFrame(shares).groupby("age_band", as_index=False)[["household_share", "wealth_share"]].mean()
    ranking["gap"] = ranking.wealth_share - ranking.household_share
    ranking = ranking.sort_values(["gap", "age_band"], ascending=[False, True]).reset_index(drop=True)
    selected = scf.loc[scf.age_band.eq(ranking.iloc[0].age_band)].copy()

    curve = load_expansion_table("treasury_yield_curve").copy()
    curve["Date"] = pd.to_datetime(curve.Date)
    curve = curve.loc[curve.Date.dt.year.eq(2025), ["Date", "10 Yr"]].dropna().sort_values("Date")
    candidates = []
    for i, start in curve.iterrows():
        later = curve.loc[curve.Date.gt(start.Date)].copy()
        if len(later):
            later["change"] = later["10 Yr"] - start["10 Yr"]
            end = later.sort_values(["change", "Date"], ascending=[False, True]).iloc[0]
            candidates.append((float(end.change), start.Date, end.Date, float(start["10 Yr"]), float(end["10 Yr"])))
    event = sorted(candidates, key=lambda x: (-x[0], x[1], x[2]))[0]
    return scf, ranking, selected, curve, event


@lru_cache(maxsize=1)
def ground_truth():
    scf, ranking, selected, curve, event = _analysis()
    leader, runner = ranking.iloc[0], ranking.iloc[1]
    medians = []
    means = []
    negatives = []
    for _, group in selected.groupby("implicate"):
        medians.append(_weighted_median(group.net_worth_usd.to_numpy(), group.analysis_weight.to_numpy()))
        means.append(float(np.average(group.net_worth_usd, weights=group.analysis_weight)))
        negatives.append(float(group.loc[group.net_worth_usd.lt(0), "analysis_weight"].sum() / group.analysis_weight.sum()))
    median = float(np.mean(medians))
    mean = float(np.mean(means))
    negative = float(np.mean(negatives))
    change, start_date, end_date, start_yield, end_yield = event
    allocation = median * 0.30
    loss = allocation * 6.5 * change / 100
    aggregate_allocation = float(selected.weighted_wealth.sum()) * 0.30
    aggregate_loss = aggregate_allocation * 6.5 * change / 100
    return (
        scf.household_id.nunique(),
        float(scf.analysis_weight.sum()) / 1e6,
        str(leader.age_band),
        float(leader.household_share) * 100,
        float(leader.wealth_share) * 100,
        float(leader.gap - runner.gap) * 100,
        len(selected),
        selected.household_id.nunique(),
        median,
        mean,
        negative * 100,
        len(curve),
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
        start_yield,
        end_yield,
        (end_date - start_date).days,
        allocation,
        loss,
        loss / median * 100,
        median - loss,
        aggregate_loss / 1e9,
    )


DECIMALS = [
    0, 6, None, 4, 4, 4,
    0, 0, 2, 2, 4,
    0, None, None, 3, 3, 0, 2,
    2, 4, 2, 6,
]


def _validate_subset(outputs, names):
    by_name = {v.name: v for v in variables}
    truth = dict(zip(by_name, ground_truth()))
    decimals = dict(zip(by_name, DECIMALS))
    return validate_ordered_outputs(outputs, [by_name[n] for n in names], [truth[n] for n in names], [decimals[n] for n in names])


def validate_turn_1(outputs): return _validate_subset(outputs, TURN_1_NAMES)
def validate_turn_2(outputs): return _validate_subset(outputs, TURN_2_NAMES)
def validate_turn_3(outputs): return _validate_subset(outputs, TURN_3_NAMES)
def validate_turn_4(outputs): return _validate_subset(outputs, TURN_4_NAMES)
def validate(outputs): return _validate_subset(outputs, [v.name for v in variables])


validators = {
    "validate_age_wealth_concentration": turn_validator(validate_turn_1),
    "validate_selected_age_distribution": turn_validator(validate_turn_2),
    "validate_ten_year_drawup": turn_validator(validate_turn_3),
    "validate_duration_sensitivity": turn_validator(validate_turn_4),
}
