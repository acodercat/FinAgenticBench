"""Estimate a generalized-FEVD connectedness network for EME credit and stress.

The three credit series are BIS total credit, loans plus debt securities from all
lenders, to non-bank borrowers in emerging-market and developing economies, one series
each for dollar-, euro- and yen-denominated lending, alongside the quarterly mean OFR
Financial Stress Index. A generalized forecast-error variance decomposition asks how
much of each series' forecast error is explained by shocks to the others without
imposing the ordering a Cholesky decomposition needs, which matters because there is no
defensible causal ordering between three funding currencies and a stress index.

On the VAR(2) estimated over 2001-Q1 through 2024-Q4 the decomposition keeps 78.8363
percent of the dollar forecast error at home and 91.1030 percent of the yen, and total
connectedness across the four nodes is 18.1924 percent. Dollar credit growth is the net
transmitter at 7.3174 percentage points; the euro is the largest net receiver at
-17.6226, taking 22.2167 percent of its variance from the others while sending 4.5941.
The largest single directed edge runs from the stress index into dollar credit growth at
14.3627 percent rather than between two currencies.

Each decomposition row is normalized to sum to 100 percent, and the reported row-sum
residual is zero at the precision the output is scored to. The innovation covariance condition number is 10.9757, so the
generalized weights are not read off a near-singular matrix, and the maximum OLS normal
moment is zero at the scored precision.

The `bis_var_connectedness` pinned-reading comparison records sensitivity to differencing the
published levels rather than using quarter-over-quarter log growth, to the VAR lag
order, and to the decomposition horizon. The query fixes all three, so these are
robustness comparisons rather than hidden answer paths.

Boundaries: a generalized decomposition is invariant to variable ordering but is not a
causal identification, and the shares describe this estimated system over this sample
rather than a structural transmission mechanism. The impulse responses are generalized
rather than orthogonalized, so they are not the effect of one shock holding the other
series fixed. Credit is measured at quarter-end in nominal terms, so valuation effects
from exchange-rate moves are inside the growth series rather than separated from it.
"""

from functools import lru_cache

import numpy as np
import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


PERIOD = "TIME_PERIOD:Time period or range"
VALUE = "OBS_VALUE:Observation Value"
CURRENCY = "CURR_DENOM:Currency of denomination"
NODES = ("usd", "eur", "jpy", "ofr")
DISPLAY = {"usd": "USD credit growth", "eur": "EUR credit growth", "jpy": "JPY credit growth", "ofr": "OFR FSI"}


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = [
    "connectedness_quarter_count",
    "var_estimation_observation_count",
    "var_spectral_radius",
    "var_design_condition_number",
    "innovation_covariance_condition_number",
    "maximum_ols_normal_moment",
]
TURN_2_NAMES = [f"gfevd_{source}_to_{target}_pct" for target in NODES for source in NODES] + ["gfevd_maximum_row_sum_residual", "total_connectedness_index_pct"]
TURN_3_NAMES = (
    [
        item
        for node in NODES
        for item in (f"{node}_from_others_pct", f"{node}_to_others_pct")
    ]
    + [
        "net_transmitter", "largest_directed_edge_source",
        "largest_directed_edge_target", "largest_directed_edge_pct",
    ]
)
TURN_4_NAMES = (
    [
        "girf_shock_node", "girf_shock_size", "girf_ofr_impact_response",
        "girf_ofr_peak_absolute_horizon", "girf_ofr_peak_absolute_response",
    ]
    + [f"girf_cumulative_{node}_response" for node in NODES]
    + ["girf_terminal_ofr_response"]
)

variables = [
    _v("connectedness_quarter_count", "Store the aligned VAR panel quarter count as an integer."),
    _v("var_estimation_observation_count", "Store the VAR(2) regression observation count as an integer."),
    _v("var_spectral_radius", "Store the companion-matrix spectral radius, rounded to 4 decimals."),
    _v("var_design_condition_number", "Store the intercept-and-lag design matrix 2-norm condition number, rounded to 4 decimals."),
    _v("innovation_covariance_condition_number", "Store the innovation covariance 2-norm condition number, rounded to 4 decimals."),
    _v("maximum_ols_normal_moment", "Store the maximum absolute intercept/lag normal-equation moment, rounded to 10 decimals."),
]
for target in NODES:
    for source in NODES:
        variables.append(_v(f"gfevd_{source}_to_{target}_pct", f"Store the row-normalized generalized FEVD share from {DISPLAY[source]} to {DISPLAY[target]} in percent, rounded to 4 decimals."))
variables += [
    _v("gfevd_maximum_row_sum_residual", "Store the maximum absolute row-sum-minus-one residual, rounded to 10 decimals."),
    _v("total_connectedness_index_pct", "Store the off-diagonal total connectedness index in percent, rounded to 4 decimals."),
]
for node in NODES:
    variables += [
        _v(f"{node}_from_others_pct", f"Store {DISPLAY[node]}'s directional connectedness received from all other nodes in percent, rounded to 4 decimals."),
        _v(f"{node}_to_others_pct", f"Store {DISPLAY[node]}'s directional connectedness transmitted to all other nodes in percent, rounded to 4 decimals."),
    ]
variables += [
    _v("net_transmitter", "Store exactly one lower-case node token: usd, eur, jpy, or ofr."),
    _v("largest_directed_edge_source", "Store the source as exactly one lower-case node token: usd, eur, jpy, or ofr."),
    _v("largest_directed_edge_target", "Store the target as exactly one lower-case node token: usd, eur, jpy, or ofr."),
    _v("largest_directed_edge_pct", "Store that row-normalized directed FEVD share in percent, rounded to 4 decimals."),
    _v("girf_shock_node", "Store the shocked node as exactly one lower-case node token: usd, eur, jpy, or ofr."),
    _v("girf_shock_size", "Store its one-innovation-standard-deviation shock in the node's native units, rounded to 4 decimals."),
    _v("girf_ofr_impact_response", "Store the generalized OFR-FSI response at horizon zero, rounded to 4 decimals."),
    _v("girf_ofr_peak_absolute_horizon", "Store the horizon from zero through eight with greatest absolute OFR-FSI response as an integer."),
    _v("girf_ofr_peak_absolute_response", "Store the signed OFR-FSI response at that horizon, rounded to 4 decimals."),
]
variables += [_v(f"girf_cumulative_{node}_response", f"Store the cumulative horizon-zero-through-eight response of {DISPLAY[node]}, rounded to 4 decimals.") for node in NODES]
variables += [_v("girf_terminal_ofr_response", "Store the OFR-FSI generalized impulse response at horizon eight, rounded to 4 decimals.")]

DECIMALS = [0, 0, 4, 4, 4, 10] + [4] * 16 + [10, 4] + [4] * 8 + [None, None, None, 4] + [None, 4, 4, 0, 4] + [4] * 4 + [4]


def _panel():
    liquidity = load_expansion_table("bis_global_liquidity")
    liquidity = liquidity.loc[
        liquidity["L_INSTR:Type of instruments"].str.startswith("B:")
        & liquidity["LENDERS_SECTOR:Lending sector"].str.startswith("A:")
    ].pivot(index=PERIOD, columns=CURRENCY, values=VALUE).sort_index()
    liquidity = np.log(liquidity).diff() * 100
    liquidity.columns = ["eur" if str(c).startswith("EUR") else "jpy" if str(c).startswith("JPY") else "usd" for c in liquidity.columns]

    ofr = load_expansion_table("ofr_market_stress").copy()
    ofr["date"] = pd.to_datetime(ofr.date)
    ofr[PERIOD] = ofr.date.dt.to_period("Q").astype(str).str.replace("Q", "-Q")
    quarterly_ofr = ofr.groupby(PERIOD).ofr_fsi.mean().rename("ofr")
    return liquidity.join(quarterly_ofr).dropna().loc["2001-Q1":"2024-Q4", list(NODES)]


@lru_cache(maxsize=1)
def _state():
    panel = _panel()
    values = panel.to_numpy(float)
    y = values[2:]
    design = np.column_stack([np.ones(len(y)), values[1:-1], values[:-2]])
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    residuals = y - design @ coefficients
    covariance = residuals.T @ residuals / len(residuals)
    a1, a2 = coefficients[1:5].T, coefficients[5:9].T
    companion = np.block([[a1, a2], [np.eye(4), np.zeros((4, 4))]])
    phis = [np.eye(4), a1]
    for _ in range(2, 9):
        phis.append(a1 @ phis[-1] + a2 @ phis[-2])
    return panel, y, design, coefficients, residuals, covariance, companion, phis


@lru_cache(maxsize=1)
def ground_truth():
    panel, y, design, coefficients, residuals, covariance, companion, phis = _state()
    theta = np.zeros((4, 4))
    for i in range(4):
        denominator = sum((phi @ covariance @ phi.T)[i, i] for phi in phis[:8])
        for j in range(4):
            theta[i, j] = sum((phi @ covariance)[i, j] ** 2 / covariance[j, j] for phi in phis[:8]) / denominator
    normalized = theta / theta.sum(axis=1, keepdims=True)
    received = (normalized.sum(axis=1) - np.diag(normalized)) * 100
    transmitted = (normalized.sum(axis=0) - np.diag(normalized)) * 100
    net = transmitted - received
    transmitter = int(np.argmax(net))
    edges = normalized.copy()
    np.fill_diagonal(edges, -np.inf)
    target, source = np.unravel_index(np.argmax(edges), edges.shape)
    girf = np.asarray([phi @ covariance[:, transmitter] / np.sqrt(covariance[transmitter, transmitter]) for phi in phis[:9]])
    ofr_peak = int(np.argmax(np.abs(girf[:, 3])))
    result = [
        len(panel), len(y), max(abs(np.linalg.eigvals(companion))), np.linalg.cond(design),
        np.linalg.cond(covariance), np.max(np.abs(design.T @ residuals / len(residuals))),
    ]
    result.extend(float(normalized[target_index, source_index] * 100) for target_index in range(4) for source_index in range(4))
    result.extend([np.max(np.abs(normalized.sum(axis=1) - 1)), (normalized.sum() - np.trace(normalized)) / 4 * 100])
    for i in range(4):
        result.extend([received[i], transmitted[i]])
    result.extend([NODES[transmitter], NODES[source], NODES[target], edges[target, source] * 100])
    result.extend([NODES[transmitter], np.sqrt(covariance[transmitter, transmitter]), girf[0, 3], ofr_peak, girf[ofr_peak, 3]])
    result.extend(girf.sum(axis=0))
    result.append(girf[8, 3])
    return tuple(result)


def _validate_subset(outputs, names):
    by_name = {v.name: v for v in variables}
    truth = dict(zip(by_name, ground_truth()))
    decimals = dict(zip(by_name, DECIMALS))
    candidate = dict(outputs)
    for name in ("net_transmitter", "largest_directed_edge_source", "largest_directed_edge_target", "girf_shock_node"):
        if name in candidate and str(candidate[name]).strip().casefold() == str(truth[name]).casefold():
            candidate[name] = truth[name]
    return validate_ordered_outputs(candidate, [by_name[n] for n in names], [truth[n] for n in names], [decimals[n] for n in names])


def validate_turn_1(outputs): return _validate_subset(outputs, TURN_1_NAMES)
def validate_turn_2(outputs): return _validate_subset(outputs, TURN_2_NAMES)
def validate_turn_3(outputs): return _validate_subset(outputs, TURN_3_NAMES)
def validate_turn_4(outputs): return _validate_subset(outputs, TURN_4_NAMES)
def validate(outputs): return _validate_subset(outputs, [v.name for v in variables])

validators = {
    "validate_var": turn_validator(validate_turn_1),
    "validate_gfevd": turn_validator(validate_turn_2),
    "validate_directional": turn_validator(validate_turn_3),
    "validate_girf": turn_validator(validate_turn_4),
}
