"""Minimum damaging five-tenor curve shock and cash-first fund redemption.

This is an explicitly first-order, USD-rate-only scenario using a reported
N-PORT DV01 row and Treasury historical yield-change covariance, not a fund
reprice or actual liquidation plan. For the frozen selected fund, collapsing
the five signed tenor DV01s to one parallel net loses the large offsetting
shape exposures: the 50bp normalized cancellation is 4.0337994% of NAV even
though absolute signed-net/gross DV01 is only 15.26656%. The unboxed 2%-NAV
loss direction needs -70.8570bp at five years; a +/-50bp coordinate limit
raises minimum Mahalanobis distance from 11.4468114 to 14.2733466 and fixes
the five-year component at -50bp. These are different scenario contracts, not
numerical-tolerance alternatives. Cash-first 20%-NAV redemption leaves
84.401217% of the noncash DV01 while NAV falls to 80%; scaling all DV01 by
80% would miss the new 10.8499017 distance. Borrowing does not supply the
USD 16.9631686m of fresh NAV capital required at the original radius.
"""

from functools import lru_cache

import numpy as np
from scipy.optimize import minimize

from cave_agent import Variable
from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


CASE_ID = "nport_bounded_curve_reverse_stress"
TENORS = ("3m", "1y", "5y", "10y", "30y")
YIELD_COLUMNS = ("3 Mo", "1 Yr", "5 Yr", "10 Yr", "30 Yr")
DV01_COLUMNS = tuple(f"dv01_{x}" for x in ("3month", "1year", "5year", "10year", "30year"))
TARGET_LOSS_SHARE = .02
BOX_BP = 50.0
REDEMPTION_SHARE = .20


def _v(name, description):
    return Variable(name, None, description)


variables = [
    _v("selected_series_id", "Store the selected SEC series identifier as its exact string."),
    _v("selected_accession", "Store the selected N-PORT accession as its exact string."),
    _v("selected_nav_mn", "Store report-date net assets in USD millions, rounded to 4 decimals from the unrounded reported amount."),
    _v("selected_cash_mn", "Store report-date cash not elsewhere reported in USD millions, rounded to 4 decimals from the unrounded reported amount."),
    _v("selected_cancellation_pct_nav", "Store the selected 50-basis-point normalized signed-tenor cancellation in percent of net assets, rounded to 4 decimals."),
    _v("curve_change_count", "Store the complete consecutive Treasury yield-change count as an integer."),
    _v("unboxed_distance", "Store the minimum unboxed Mahalanobis distance as a nonnegative number, rounded to 4 decimals."),
    _v("unboxed_5y_shock_bp", "Store the signed five-year component of the unboxed shock in basis points, rounded to 4 decimals."),
    _v("unboxed_10y_shock_bp", "Store the signed ten-year component of the unboxed shock in basis points, rounded to 4 decimals."),
    _v("boxed_distance", "Store the minimum boxed Mahalanobis distance as a nonnegative number, rounded to 4 decimals."),
    _v("boxed_10y_shock_bp", "Store the signed ten-year component of the boxed shock in basis points, rounded to 4 decimals."),
    _v("boxed_first_binding", "Store exactly one token from NONE, 3M_LOW, 3M_HIGH, 1Y_LOW, 1Y_HIGH, 5Y_LOW, 5Y_HIGH, 10Y_LOW, 10Y_HIGH, 30Y_LOW, 30Y_HIGH."),
    _v("redemption_noncash_sale_mn", "Store noncash assets sold at reported book value for the redemption in USD millions, rounded to 4 decimals."),
    _v("redemption_dv01_retained_pct", "Store the post-redemption USD DV01 fraction retained in percent, rounded to 4 decimals."),
    _v("redemption_unboxed_distance", "Store the post-redemption minimum unboxed Mahalanobis distance as a nonnegative number, rounded to 4 decimals."),
    _v("pre_injection_worst_loss_pct_nav", "Store the original-distance worst first-order loss as percent of post-redemption, pre-injection net assets, rounded to 4 decimals."),
    _v("required_new_equity_mn", "Store the minimum fresh common-equity cash injection in USD millions, rounded to 4 decimals."),
]
TURN_NAMES = [
    ["selected_series_id", "selected_accession", "selected_nav_mn", "selected_cash_mn", "selected_cancellation_pct_nav", "curve_change_count"],
    ["unboxed_distance", "unboxed_5y_shock_bp", "unboxed_10y_shock_bp"],
    ["boxed_distance", "boxed_10y_shock_bp", "boxed_first_binding"],
    ["redemption_noncash_sale_mn", "redemption_dv01_retained_pct", "redemption_unboxed_distance"],
    ["pre_injection_worst_loss_pct_nav", "required_new_equity_mn"],
]
DECIMALS = [None, None, 4, 4, 4, 0, 4, 4, 4, 4, 4, None, 4, 4, 4, 4, 4]


@lru_cache(maxsize=1)
def _state():
    reports = load_expansion_table("sec_nport")
    filings = reports.loc[
        reports.report_date.eq("2025-03-31")
        & reports.form.eq("NPORT-P")
        & reports.filing_date.le("2025-06-30")
    ].dropna(subset=["series_id"])
    # The query takes each series' latest filing first and only then applies the
    # balance-sheet requirements; screening first would fall back to an earlier
    # filing whenever the latest one reports a null.
    filings = filings.sort_values(["series_id", "filing_date", "accession"]).drop_duplicates("series_id", keep="last")
    filings = filings.dropna(subset=["net_assets_usd", "cash_not_elsewhere_reported_usd", "total_assets_usd", "total_liabilities_usd"])
    risk = load_expansion_table("sec_nport_interest_rate_risk")
    risk = risk.loc[risk.report_date.eq("2025-03-31") & risk.currency_code.eq("USD")].dropna(subset=list(DV01_COLUMNS))
    if risk.accession.duplicated().any():
        raise ValueError("USD risk accession is not unique")
    cohort = filings.merge(risk[["accession", *DV01_COLUMNS]], on="accession", validate="one_to_one")
    cohort = cohort.loc[
        cohort.net_assets_usd.ge(100_000_000)
        & cohort.cash_not_elsewhere_reported_usd.gt(0)
        & cohort.cash_not_elsewhere_reported_usd.lt(cohort.net_assets_usd)
        & cohort.total_assets_usd.sub(cohort.total_liabilities_usd).sub(cohort.net_assets_usd).abs().le(1)
    ].copy()
    dv = cohort[list(DV01_COLUMNS)].to_numpy(float)
    cohort["cancellation"] = 50 * (np.abs(dv).sum(axis=1) - np.abs(dv.sum(axis=1))) / cohort.net_assets_usd.to_numpy(float) * 100
    cohort = cohort.sort_values(["cancellation", "series_id"], ascending=[False, True]).reset_index(drop=True)
    if len(cohort) != 964 or str(cohort.iloc[0].series_id) != "S000044524":
        raise ValueError("Frozen N-PORT selected population changed")
    selected = cohort.iloc[0]
    vector = selected[list(DV01_COLUMNS)].to_numpy(float)

    treasury = load_expansion_table("treasury_yield_curve")
    treasury = treasury.loc[treasury.Date.between("2020-01-02", "2024-12-31")].dropna(subset=list(YIELD_COLUMNS)).sort_values("Date")
    changes = np.diff(treasury[list(YIELD_COLUMNS)].to_numpy(float), axis=0) * 100
    covariance = np.cov(changes, rowvar=False, ddof=1)
    eigenvalues = np.linalg.eigvalsh(covariance)
    if len(changes) != 1250 or eigenvalues[0] <= 0:
        raise ValueError("Historical five-tenor covariance changed or is singular")
    return selected, vector, covariance, len(changes)


def _unboxed(vector, covariance, target):
    variance = float(vector @ covariance @ vector)
    if variance <= 0:
        raise ValueError("Zero rate-risk quadratic form")
    shock = target * (covariance @ vector) / variance
    distance = target / np.sqrt(variance)
    return shock, float(distance)


def _boxed(vector, covariance, target):
    inverse = np.linalg.inv(covariance)
    # A sign-aligned feasible seed is well inside the box for this frozen
    # target. The loss constraint is scaled by NAV to keep SLSQP conditioned.
    seed = np.sign(vector) * target / np.abs(vector).sum()
    scale = target / TARGET_LOSS_SHARE
    result = minimize(
        lambda x: float(x @ inverse @ x) / 2,
        seed,
        jac=lambda x: inverse @ x,
        method="SLSQP",
        bounds=[(-BOX_BP, BOX_BP)] * 5,
        constraints=[{
            "type": "eq",
            "fun": lambda x: float(vector @ x / scale - TARGET_LOSS_SHARE),
            "jac": lambda x: vector / scale,
        }],
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    if (not result.success or not np.all(np.isfinite(result.x))
            or abs(vector @ result.x - target) > 1e-4
            or np.max(np.abs(result.x)) > BOX_BP + 1e-7):
        raise ValueError(f"Bounded reverse stress did not converge: {result.message}")
    return result.x, float(np.sqrt(result.x @ inverse @ result.x))


@lru_cache(maxsize=1)
def analysis():
    selected, dv, covariance, change_count = _state()
    nav = float(selected.net_assets_usd)
    cash = float(selected.cash_not_elsewhere_reported_usd)
    assets = float(selected.total_assets_usd)
    liabilities = float(selected.total_liabilities_usd)
    target = TARGET_LOSS_SHARE * nav
    unboxed_shock, unboxed_distance = _unboxed(dv, covariance, target)
    boxed_shock, boxed_distance = _boxed(dv, covariance, target)
    if boxed_distance + 1e-9 < unboxed_distance:
        raise ValueError("Boxed optimum cannot improve the unconstrained distance")
    active = []
    for tenor, value in zip(TENORS, boxed_shock):
        if abs(value + BOX_BP) < 1e-5:
            active.append(tenor.upper() + "_LOW")
        elif abs(value - BOX_BP) < 1e-5:
            active.append(tenor.upper() + "_HIGH")
    binding = active[0] if active else "NONE"
    if abs(assets - liabilities - nav) > 1:
        raise ValueError("Report-date accounting identity fails")
    withdrawal = REDEMPTION_SHARE * nav
    noncash_sale = max(0., withdrawal - cash)
    noncash_assets = assets - cash
    retained = 1 - noncash_sale / noncash_assets
    if not 0 < retained <= 1:
        raise ValueError("Proportional sale exceeds noncash assets")
    after_nav = nav - withdrawal
    after_dv = retained * dv
    _, after_distance = _unboxed(after_dv, covariance, TARGET_LOSS_SHARE * after_nav)
    radius_loss = unboxed_distance * np.sqrt(float(after_dv @ covariance @ after_dv))
    new_equity = max(0., radius_loss / TARGET_LOSS_SHARE - after_nav)
    if abs(radius_loss - TARGET_LOSS_SHARE * (after_nav + new_equity)) > 1e-6:
        raise ValueError("New-equity risk boundary failed")
    return {
        "selected_series_id": str(selected.series_id),
        "selected_accession": str(selected.accession),
        "selected_nav_mn": nav / 1e6,
        "selected_cash_mn": cash / 1e6,
        "selected_cancellation_pct_nav": float(selected.cancellation),
        "curve_change_count": change_count,
        "unboxed_distance": unboxed_distance,
        "unboxed_5y_shock_bp": float(unboxed_shock[2]),
        "unboxed_10y_shock_bp": float(unboxed_shock[3]),
        "boxed_distance": boxed_distance,
        "boxed_10y_shock_bp": float(boxed_shock[3]),
        "boxed_first_binding": binding,
        "redemption_noncash_sale_mn": noncash_sale / 1e6,
        "redemption_dv01_retained_pct": retained * 100,
        "redemption_unboxed_distance": after_distance,
        "pre_injection_worst_loss_pct_nav": radius_loss / after_nav * 100,
        "required_new_equity_mn": new_equity / 1e6,
    }


@lru_cache(maxsize=1)
def ground_truth():
    state = analysis()
    return tuple(state[v.name] for v in variables)


def _validate_subset(outputs, names):
    by_name = {v.name: v for v in variables}
    truth = analysis()
    precision = dict(zip((v.name for v in variables), DECIMALS))
    return validate_ordered_outputs(
        outputs, [by_name[n] for n in names], [truth[n] for n in names],
        [precision[n] for n in names],
    )


def validate(outputs):
    return _validate_subset(outputs, [v.name for v in variables])


validators = {
    f"validate_turn_{i}": turn_validator(lambda outputs, names=names: _validate_subset(outputs, names))
    for i, names in enumerate(TURN_NAMES, start=1)
}
