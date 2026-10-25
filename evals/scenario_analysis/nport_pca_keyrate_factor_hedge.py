"""Principal-component key-rate hedge for a carried N-PORT bond fund.

A duration hedge that matches one number leaves a portfolio exposed to the shape of the
curve. Decomposing key-rate DV01s onto principal components separates the level, slope
and curvature moves that actually drive a bond fund's profit and loss, and a
three-instrument overlay can zero the first three of them exactly.

The solved overlay is 34,722.119 of one-year DV01, -300,321.124 of five-year and
-4,154,201.614 of thirty-year. The hedge-factor matrix has a 2-norm condition number of
3.8329, so the three-instrument system is well posted rather than nearly singular, and
the maximum absolute residual across the first three factor exposures is 0.0. Zeroing
three factors is not zeroing risk: the overlay leaves 4,884.877 of PC4 DV01 and
-668,238.560 of PC5, and it takes 4,489,244.857 of gross DV01 to put on.

The realized move over the following period was a twist rather than a level shift, with
the three-month yield down 65 basis points, the one-year down 55, the five-year down 23,
the ten-year down 5 and the thirty-year up 25. On that path the unhedged fund loses
94,307,323.86 dollars, 5.910295 percent of net assets, while the hedged fund gains
4,550,047.21, 0.285154 percent, a 98,857,371.07 dollar difference and a 95.1753 percent
reduction in absolute first-order profit and loss. A level-only duration hedge would not
merely have missed this move, it would have made things worse: offsetting the fund's
total DV01 at the ten-year point turns the loss into -116,337,028.23 dollars, because
the realized path fell hardest at the front end and rose at the long end. What the
three-factor overlay leaves behind is the unhedged fourth and fifth factors.

Every realized profit-and-loss figure is computed from the unrounded DV01s and overlay
notionals, not from the rounded figures reported for them, and the post-hedge residual is
pinned at 7 decimals so the shared rounding-boundary tolerance absorbs the noise any
convergent linear solve leaves behind.

The `nport_pca_hedge` pinned-reading comparison records sensitivity to the training-window start,
to decomposing the correlation rather than the covariance matrix, and to the choice of
hedge tenors. The query fixes all three, so these are robustness comparisons rather than
hidden answer paths.

Boundaries: this is a first-order DV01 approximation, so convexity and the actual
repricing of the underlying holdings are outside it, and the overlay is hypothetical with
no financing cost, bid-offer or margin. Principal components are estimated on this
training window and are not stable structural factors; a different window yields different
loadings, which is why the window is part of the question. The fund's key-rate DV01s are
derived from a period-end holdings snapshot and do not reflect intra-period trading.
"""

from functools import lru_cache

import numpy as np
import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import ValidatorResult, turn_validator, validate_ordered_outputs


TENORS = ("3month", "1year", "5year", "10year", "30year")
YIELD_COLUMNS = ("3 Mo", "1 Yr", "5 Yr", "10 Yr", "30 Yr")
DV01_COLUMNS = tuple(f"dv01_{tenor}" for tenor in TENORS)
HEDGE_INDICES = (1, 2, 4)


def _v(name, description):
    return Variable(name, None, description)

TURN_1_NAMES = (
    ["pca_yield_change_count"]
    + [f"pca_eigenvalue_{i}_bp2" for i in range(1, 6)]
    + ["pca_first_three_explained_pct", "pca_covariance_condition_number"]
    + [f"pc1_{tenor}_loading" for tenor in TENORS]
    + [f"pc2_{tenor}_loading" for tenor in TENORS]
)
TURN_2_NAMES = (
    [
        "nport_eligible_fund_count", "selected_fund_name", "selected_fund_series_id",
        "selected_fund_accession", "selected_fund_net_assets_usd",
    ]
    + [f"selected_{tenor}_dv01_usd" for tenor in TENORS]
    + [f"selected_pc{i}_duration_exposure" for i in range(1, 6)]
    + ["selected_pc1_absolute_margin"]
)
TURN_3_NAMES = [
    "hedge_1year_dv01_usd", "hedge_5year_dv01_usd", "hedge_30year_dv01_usd",
    "hedge_factor_matrix_condition_number", "hedge_maximum_first_three_factor_residual",
    "hedged_pc4_dv01_usd", "hedged_pc5_dv01_usd", "hedge_gross_dv01_usd",
]
TURN_4_NAMES = (
    [f"realized_{tenor}_yield_change_bp" for tenor in TENORS]
    + [
        "unhedged_first_order_pnl_usd", "hedged_first_order_pnl_usd",
        "unhedged_pnl_pct_net_assets", "hedged_pnl_pct_net_assets",
        "absolute_pnl_reduction_pct",
    ]
)

variables = [_v("pca_yield_change_count", "Store the complete daily Treasury yield-change count as an integer.")]
variables += [_v(f"pca_eigenvalue_{i}_bp2", f"Store PCA covariance eigenvalue {i} in squared basis points, rounded to 6 decimals.") for i in range(1, 6)]
variables += [
    _v("pca_first_three_explained_pct",
       "Store variance explained by the first three components in percent, rounded to 4 decimals."),
    _v("pca_covariance_condition_number",
       "Store the yield-change covariance 2-norm condition number, rounded to 4 decimals."),
]
variables += [_v(f"pc1_{tenor}_loading", f"Store the PC1 loading at {tenor}, rounded to 4 decimals.") for tenor in TENORS]
variables += [_v(f"pc2_{tenor}_loading", f"Store the PC2 loading at {tenor}, rounded to 4 decimals.") for tenor in TENORS]
variables += [
    _v("nport_eligible_fund_count",
       "Store the latest-per-series eligible N-PORT fund count as an integer."),
    _v("selected_fund_name",
       "Store the selected fund's registered series name verbatim from N-PORT, preserving every word, "
       "punctuation mark, and legal-style suffix."),
    _v("selected_fund_series_id", "Store the selected SEC series identifier."),
    _v("selected_fund_accession", "Store the selected N-PORT accession."),
    _v("selected_fund_net_assets_usd",
       "Store selected fund net assets in USD, rounded to 2 decimals."),
]
variables += [_v(f"selected_{tenor}_dv01_usd", f"Store selected fund {tenor} reported USD DV01, rounded to 3 decimals.") for tenor in TENORS]
variables += [_v(f"selected_pc{i}_duration_exposure", f"Store selected fund PC{i} exposure after DV01-to-net-assets duration normalization, rounded to 4 decimals.") for i in range(1, 6)]
variables += [_v("selected_pc1_absolute_margin", "Store the selected absolute PC1 duration exposure minus the runner-up's, rounded to 4 decimals.")]
variables += [
    _v("hedge_1year_dv01_usd",
       "Store the 1-year key-rate hedge overlay in USD DV01, rounded to 3 decimals."),
    _v("hedge_5year_dv01_usd",
       "Store the 5-year key-rate hedge overlay in USD DV01, rounded to 3 decimals."),
    _v("hedge_30year_dv01_usd",
       "Store the 30-year key-rate hedge overlay in USD DV01, rounded to 3 decimals."),
    _v("hedge_factor_matrix_condition_number",
       "Store the first-three-factor hedge matrix 2-norm condition number, rounded to 4 decimals."),
    _v("hedge_maximum_first_three_factor_residual",
       "Store the maximum absolute post-hedge first-three-factor DV01 residual in USD, rounded to 7 "
       "decimals; an exact overlay drives it to zero."),
    _v("hedged_pc4_dv01_usd",
       "Store post-hedge PC4 exposure in USD DV01, rounded to 3 decimals."),
    _v("hedged_pc5_dv01_usd",
       "Store post-hedge PC5 exposure in USD DV01, rounded to 3 decimals."),
    _v("hedge_gross_dv01_usd",
       "Store sum of absolute overlay DV01 in USD, rounded to 3 decimals."),
]
variables += [_v(f"realized_{tenor}_yield_change_bp", f"Store 2025-03-31 to 2025-12-31 {tenor} Treasury yield change in basis points, rounded to 2 decimals.") for tenor in TENORS]
variables += [
    _v("unhedged_first_order_pnl_usd",
       "Store unhedged static-DV01 first-order P&L in USD, rounded to 2 decimals. Compute from the "
       "unrounded DV01s and overlay notionals, not from the rounded figures reported for them."),
    _v("hedged_first_order_pnl_usd",
       "Store hedged static-DV01 first-order P&L in USD, rounded to 2 decimals. Compute from the "
       "unrounded DV01s and overlay notionals, not from the rounded figures reported for them."),
    _v("unhedged_pnl_pct_net_assets",
       "Store unhedged first-order P&L as percent of report-date net assets, rounded to 6 decimals. "
       "Compute from the unrounded DV01s and overlay notionals, not from the rounded figures reported "
       "for them."),
    _v("hedged_pnl_pct_net_assets",
       "Store hedged first-order P&L as percent of report-date net assets, rounded to 6 decimals. "
       "Compute from the unrounded DV01s and overlay notionals, not from the rounded figures "
       "reported for them."),
    _v("absolute_pnl_reduction_pct",
       "Store one minus absolute hedged P&L divided by absolute unhedged P&L, in percent rounded to 4 "
       "decimals. Compute from the unrounded DV01s and overlay notionals, not from the rounded figures "
       "reported for them."),
]

DECIMALS = [0] + [6] * 5 + [4, 4] + [4] * 10 + [0, None, None, None, 2] + [3] * 5 + [4] * 5 + [4] + [3, 3, 3, 4, 7, 3, 3, 3] + [2] * 5 + [2, 2, 6, 6, 4]


@lru_cache(maxsize=1)
def _state():
    yields = load_expansion_table("treasury_yield_curve").copy()
    yields["Date"] = pd.to_datetime(yields.Date)
    training = yields.loc[yields.Date.between("2020-01-02", "2024-12-31")].dropna(subset=list(YIELD_COLUMNS)).sort_values("Date")
    changes = training[list(YIELD_COLUMNS)].astype(float).diff().dropna().to_numpy() * 100
    covariance = np.cov(changes, rowvar=False, ddof=1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    for index in range(5):
        if eigenvectors[:, index].sum() < 0:
            eigenvectors[:, index] *= -1

    risk = load_expansion_table("sec_nport_interest_rate_risk")
    population = risk.loc[
        risk.report_date.astype(str).eq("2025-03-31") & risk.currency_code.astype(str).eq("USD")
        & risk.net_assets_usd.gt(0)
    ].dropna(subset=["series_id", *DV01_COLUMNS]).copy()
    population = population.sort_values(["series_id", "filing_date", "accession"]).drop_duplicates("series_id", keep="last")
    dv01 = population[list(DV01_COLUMNS)].to_numpy(float)
    duration_vectors = dv01 / population.net_assets_usd.to_numpy(float)[:, None] * 10000
    factor_duration = duration_vectors @ eigenvectors
    ranking = np.lexsort((population.series_id.astype(str).to_numpy(), -np.abs(factor_duration[:, 0])))
    leader_index, runner_index = ranking[:2]
    selected = population.iloc[leader_index]
    selected_dv01 = dv01[leader_index]
    selected_factors = selected_dv01 @ eigenvectors

    hedge_matrix = eigenvectors[list(HEDGE_INDICES), :3].T
    hedge_values = np.linalg.solve(hedge_matrix, -selected_factors[:3])
    overlay = np.zeros(5)
    overlay[list(HEDGE_INDICES)] = hedge_values
    hedged = selected_dv01 + overlay
    hedged_factors = hedged @ eigenvectors
    start = yields.loc[yields.Date.eq(pd.Timestamp("2025-03-31")), list(YIELD_COLUMNS)].iloc[0].to_numpy(float)
    end = yields.loc[yields.Date.eq(pd.Timestamp("2025-12-31")), list(YIELD_COLUMNS)].iloc[0].to_numpy(float)
    realized_changes = (end - start) * 100
    return (changes, covariance, eigenvalues, eigenvectors, population, factor_duration,
            leader_index, runner_index, selected, selected_dv01, hedge_matrix, overlay,
            hedged, hedged_factors, realized_changes)


@lru_cache(maxsize=1)
def ground_truth():
    (changes, covariance, eigenvalues, eigenvectors, population, factor_duration,
     leader_index, runner_index, selected, selected_dv01, hedge_matrix, overlay,
     hedged, hedged_factors, realized_changes) = _state()
    unhedged_pnl = float(-selected_dv01 @ realized_changes)
    hedged_pnl = float(-hedged @ realized_changes)
    nav = float(selected.net_assets_usd)
    result = [len(changes), *eigenvalues, eigenvalues[:3].sum() / eigenvalues.sum() * 100,
              np.linalg.cond(covariance), *eigenvectors[:, 0], *eigenvectors[:, 1], len(population),
              str(selected.series_name), str(selected.series_id), str(selected.accession), nav,
              *selected_dv01, *factor_duration[leader_index],
              abs(factor_duration[leader_index, 0]) - abs(factor_duration[runner_index, 0]),
              overlay[1], overlay[2], overlay[4], np.linalg.cond(hedge_matrix),
              np.max(np.abs(hedged_factors[:3])), hedged_factors[3], hedged_factors[4],
              np.abs(overlay).sum(), *realized_changes, unhedged_pnl, hedged_pnl,
              unhedged_pnl / nav * 100, hedged_pnl / nav * 100,
              (1 - abs(hedged_pnl) / abs(unhedged_pnl)) * 100]
    return tuple(result)


def _validate_subset(outputs, names):
    by_name = {v.name:v for v in variables}
    truth = dict(zip(by_name, ground_truth()))
    decimals = dict(zip(by_name, DECIMALS))
    return validate_ordered_outputs(outputs, [by_name[n] for n in names], [truth[n] for n in names], [decimals[n] for n in names])


def _finish(result):
    items = result.variable_results or {}
    missing = any(not item["is_set"] for item in items.values())
    errors = [item["message"] for item in items.values() if item["is_set"] and not item["correct"]]
    return ValidatorResult(not missing and not errors, "required output not set" if missing else "; ".join(errors) or "correct", missing, items)




def validate_turn_1(outputs): return _validate_subset(outputs, TURN_1_NAMES)
def validate_turn_2(outputs): return _validate_subset(outputs, TURN_2_NAMES)
def validate_turn_3(outputs): return _finish(_validate_subset(outputs, TURN_3_NAMES))
def validate_turn_4(outputs): return _finish(_validate_subset(outputs, TURN_4_NAMES))
def validate(outputs): return _validate_subset(outputs, [v.name for v in variables])
validators = {
    "validate_pca": turn_validator(validate_turn_1),
    "validate_selection": turn_validator(validate_turn_2),
    "validate_hedge": turn_validator(validate_turn_3),
    "validate_realized": turn_validator(validate_turn_4),
}
