"""Two-state Gaussian HMM for aggregate money-fund flow/liquidity regimes.

Money market fund flows and portfolio liquidity are usually described in regime language:
funds are said to be in a calm state or a stressed one. A hidden Markov model makes that
language testable, because it produces both a fitted state sequence and a predictive
density that can be scored against a single-Gaussian alternative.

The aligned panel runs 372 days from 2024-06-03 to 2025-11-28 across 330 matched filings.
The heaviest single-day net outflow, 2024-11-14 at -0.8741 percent of assets, occurs
while weekly liquid assets stand at 81.3107 percent and SOFR at 4.58 percent, so the
worst flow day of the sample is not a day of impaired liquidity.

Expectation-maximization converges in 12 iterations over 300 training days to a log
likelihood of -653.606581, and both states are persistent: the calm-to-stress transition
probability is 0.01931034 and stress-to-calm 0.01419855, implying mean durations of tens
of days rather than day-to-day switching. What separates the states is liquidity rather
than direction of flow. The stress state carries mean weekly liquid assets of 80.4091
percent against 82.5289 in the calm state, while mean net flow is positive in both,
0.0460 against 0.0836 percent. Posterior mass and scaled forward likelihood residuals
are both 0.0.

The out-of-sample result is negative and is the point of the case. Over the 72 test days
the two-state model beats the single Gaussian on average predictive log score,
-1.62295459 against -2.24292589, an improvement of 0.6199713, and the filtered stress
probability reaches 0.9994187 on 2025-09-24. Yet the classification carries no next-day
flow information: the 54 stress-classified days are followed by mean net flow of 0.0827
percent and the calm-classified days by 0.0734, so stress-classified days are followed
by slightly stronger inflows, a difference of 0.0094 percentage points in the wrong
direction for a stress signal. A model that fits the density better is not thereby a
usable early warning.

The `nmfp_hmm_liquidity_regime` pinned-reading comparison records sensitivity to the flow scaling,
to the posterior cutoff used to classify a day, and to which fitted state is labelled the
stress state. The query fixes all three, so these are robustness comparisons rather than
hidden answer paths.

Boundaries: two states are imposed, not selected by an information criterion, and the
Gaussian emission is assumed rather than tested. The state labels are a modelling
convention attached to fitted moments and carry no regulatory or supervisory meaning.
Aggregate flows conceal offsetting movements between individual funds, and 72 test days
with 54 classified one way cannot establish predictive value in either direction.
"""

from functools import lru_cache

import numpy as np
import pandas as pd
from cave_agent import Variable
from scipy.special import logsumexp

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


TRAIN_COUNT = 300
MAX_ITER = 500
TOLERANCE = 1e-10


def _v(name, description):
    return Variable(name, None, description)
TURN_1_NAMES = [
    "aligned_day_count",
    "aligned_start_date",
    "aligned_end_date",
    "matched_accession_count",
    "minimum_net_flow_date",
    "minimum_net_flow_pct",
    "minimum_flow_weekly_liquidity_pct",
    "minimum_flow_sofr_pct",
]
TURN_2_NAMES = [
    "hmm_training_day_count",
    "hmm_em_iteration_count",
    "hmm_log_likelihood",
    "calm_to_stress_probability",
    "stress_to_calm_probability",
    "stress_state_mean_net_flow_pct",
    "stress_state_mean_weekly_liquidity_pct",
    "calm_state_mean_net_flow_pct",
    "calm_state_mean_weekly_liquidity_pct",
    "hmm_maximum_posterior_mass_residual",
    "hmm_scaled_forward_log_likelihood_residual",
]
TURN_3_NAMES = [
    "hmm_test_start_date",
    "hmm_test_end_date",
    "hmm_test_day_count",
    "maximum_filtered_stress_date",
    "maximum_filtered_stress_probability",
    "maximum_stress_date_net_flow_pct",
    "maximum_stress_date_weekly_liquidity_pct",
    "maximum_stress_date_sofr_pct",
]
TURN_4_NAMES = [
    "hmm_average_predictive_log_score",
    "single_gaussian_average_log_score",
    "stress_classified_test_day_count",
    "stress_classified_next_day_mean_net_flow_pct",
    "calm_classified_next_day_mean_net_flow_pct",
    "hmm_beats_single_gaussian",
]

_descriptions = {
    "aligned_day_count": "Store the aligned daily observation count as an integer.",
    "aligned_start_date": "Store the first aligned date as an ISO YYYY-MM-DD string.",
    "aligned_end_date": "Store the last aligned date as an ISO YYYY-MM-DD string.",
    "matched_accession_count": "Store the maximum same-day matched accession count as an integer.",
    "minimum_net_flow_date": "Store the minimum aggregate net-flow date as an ISO YYYY-MM-DD string.",
    "minimum_net_flow_pct": "Store the signed minimum aggregate net flow as a percent of matched net assets, rounded to 4 decimals.",
    "minimum_flow_weekly_liquidity_pct": "Store the net-asset-weighted weekly-liquid-assets percentage on that date, rounded to 4 decimals.",
    "minimum_flow_sofr_pct": "Store SOFR on that date in percent, rounded to 2 decimals.",
    "hmm_training_day_count": "Store the HMM training-day count as an integer.",
    "hmm_em_iteration_count": "Store the number of completed EM updates as an integer.",
    "hmm_log_likelihood": "Store the fitted training log likelihood, rounded to 6 decimals.",
    "calm_to_stress_probability": "Store the one-day calm-to-stress transition probability, rounded to 8 decimals.",
    "stress_to_calm_probability": "Store the one-day stress-to-calm transition probability, rounded to 8 decimals.",
    "stress_state_mean_net_flow_pct": "Store the stress-state mean signed net flow in percent, rounded to 4 decimals.",
    "stress_state_mean_weekly_liquidity_pct": "Store the stress-state mean weekly liquidity in percent, rounded to 4 decimals.",
    "calm_state_mean_net_flow_pct": "Store the calm-state mean signed net flow in percent, rounded to 4 decimals.",
    "calm_state_mean_weekly_liquidity_pct": "Store the calm-state mean weekly liquidity in percent, rounded to 4 decimals.",
    "hmm_maximum_posterior_mass_residual": "Store the maximum absolute posterior row-mass residual, rounded to 10 decimals.",
    "hmm_scaled_forward_log_likelihood_residual": "Store the absolute log-likelihood residual between the log-domain and scaled forward recursions, rounded to 10 decimals.",
    "hmm_test_start_date": "Store the first held-out test date as an ISO YYYY-MM-DD string.",
    "hmm_test_end_date": "Store the last held-out test date as an ISO YYYY-MM-DD string.",
    "hmm_test_day_count": "Store the held-out test-day count as an integer.",
    "maximum_filtered_stress_date": "Store the date of maximum filtered stress probability as an ISO YYYY-MM-DD string.",
    "maximum_filtered_stress_probability": "Store the maximum filtered stress-state probability, rounded to 8 decimals.",
    "maximum_stress_date_net_flow_pct": "Store that date's signed aggregate net flow in percent, rounded to 4 decimals.",
    "maximum_stress_date_weekly_liquidity_pct": "Store that date's weighted weekly liquidity in percent, rounded to 4 decimals.",
    "maximum_stress_date_sofr_pct": "Store that date's SOFR in percent, rounded to 2 decimals.",
    "hmm_average_predictive_log_score": "Store the HMM average one-step predictive log score, rounded to 8 decimals.",
    "single_gaussian_average_log_score": "Store the single-Gaussian average predictive log score, rounded to 8 decimals.",
    "stress_classified_test_day_count": "Store the number of test days classified as stress as an integer.",
    "stress_classified_next_day_mean_net_flow_pct": "Store mean next-day signed net flow after stress-classified days in percent, rounded to 4 decimals.",
    "calm_classified_next_day_mean_net_flow_pct": "Store mean next-day signed net flow after calm-classified days in percent, rounded to 4 decimals.",
    "hmm_beats_single_gaussian": "Store whether the HMM average predictive log score is higher as a boolean.",
}
variables = [
    _v(name, _descriptions[name])
    for name in TURN_1_NAMES + TURN_2_NAMES + TURN_3_NAMES + TURN_4_NAMES
]
DECIMALS = [0, None, None, 0, None, 4, 4, 2, 0, 0, 6, 8, 8, 4, 4, 4, 4, 10, 10, None, None, 0, None, 8, 4, 4, 2, 8, 8, 0, 4, 4, None]


@lru_cache(maxsize=1)
def _panel():
    filings = load_expansion_table("sec_nmfp")[["accession", "net_assets_usd"]].drop_duplicates("accession")
    flows = load_expansion_table("sec_nmfp_flows").copy()
    flows["net_flow"] = flows.gross_subscriptions_usd-flows.gross_redemptions_usd
    flow = flows.groupby(["accession", "flow_date"], as_index=False).net_flow.sum().rename(columns={"flow_date":"date"})
    liq = load_expansion_table("sec_nmfp_liquidity")[["accession", "liquidity_date", "weekly_liquid_assets_fraction"]].rename(columns={"liquidity_date":"date"})
    matched = flow.merge(liq, on=["accession", "date"], how="inner", validate="one_to_one").merge(filings, on="accession", validate="many_to_one")
    matched["weighted"] = matched.weekly_liquid_assets_fraction*matched.net_assets_usd
    aggregate = matched.groupby("date").agg(net_flow=("net_flow", "sum"), assets=("net_assets_usd", "sum"), accessions=("accession", "nunique"), weighted=("weighted", "sum")).reset_index()
    aggregate["net_flow_pct"] = aggregate.net_flow/aggregate.assets*100
    aggregate["weekly_liquidity_pct"] = aggregate.weighted/aggregate.assets*100
    rates = load_expansion_table("nyfed_reference_rates").copy()
    rates = rates.loc[rates["Rate Type"].eq("SOFR"), ["Effective Date", "Rate (%)"]].rename(columns={"Effective Date":"date", "Rate (%)":"sofr_pct"})
    for frame in (aggregate, rates):
        frame["date"] = pd.to_datetime(frame.date)
    panel = aggregate.merge(rates, on="date", validate="one_to_one").sort_values("date").reset_index(drop=True)
    return panel


def _log_density(x, means, variances):
    return -.5*(np.sum(np.log(2*np.pi*variances), axis=1)[None, :]+np.sum((x[:, None, :]-means[None, :, :])**2/variances[None, :, :], axis=2))


def _forward_backward(x, initial, transition, means, variances):
    emit = _log_density(x, means, variances)
    logt = np.log(transition)
    n = len(x)
    forward = np.empty((n, 2))
    forward[0] = np.log(initial)+emit[0]
    for t in range(1, n):
        forward[t] = emit[t]+logsumexp(forward[t-1][:, None]+logt, axis=0)
    likelihood = float(logsumexp(forward[-1]))
    backward = np.zeros((n, 2))
    for t in range(n-2, -1, -1):
        backward[t] = logsumexp(logt+emit[t+1][None, :]+backward[t+1][None, :], axis=1)
    gamma = np.exp(forward+backward-likelihood)
    xi = np.empty((n-1, 2, 2))
    for t in range(n-1):
        xi[t] = np.exp(forward[t][:, None]+logt+emit[t+1][None, :]+backward[t+1][None, :]-likelihood)
    return likelihood, gamma, xi, forward, emit


@lru_cache(maxsize=1)
def _fit():
    panel = _panel()
    raw = panel[["net_flow_pct", "weekly_liquidity_pct"]].to_numpy(float)
    train = raw[:TRAIN_COUNT]
    center = train.mean(0)
    scale = train.std(0, ddof=1)
    x = (train-center)/scale
    score = x[:, 0]+x[:, 1]
    groups = score<=np.median(score)
    means = np.vstack([x[~groups].mean(0), x[groups].mean(0)])
    variances = np.vstack([x[~groups].var(0)+.1, x[groups].var(0)+.1])
    transition = np.array([[.95, .05], [.08, .92]])
    initial = np.array([.5, .5])
    previous = None
    for iteration in range(1, MAX_ITER+1):
        _, gamma, xi, _, _ = _forward_backward(x, initial, transition, means, variances)
        initial = np.maximum(gamma[0], 1e-12)
        initial/=initial.sum()
        transition = xi.sum(0)/gamma[:-1].sum(0)[:, None]
        transition = np.maximum(transition, 1e-10)
        transition/=transition.sum(1)[:, None]
        weight = gamma.sum(0)
        means = (gamma.T@x)/weight[:, None]
        variances = np.maximum(
            np.vstack([
                np.average((x-means[s])**2, axis=0, weights=gamma[:, s])
                for s in range(2)
            ]),
            1e-6,
        )
        likelihood, gamma, xi, _, _ = _forward_backward(x, initial, transition, means, variances)
        if previous is not None and likelihood-previous<TOLERANCE:
            break
        previous = likelihood
    # Canonical label: stress has lower sum of original-unit standardized feature means.
    stress = int(np.argmin(means.sum(1)))
    calm = 1-stress
    order = [calm, stress]
    initial = initial[order]
    transition = transition[np.ix_(order, order)]
    means = means[order]
    variances = variances[order]
    likelihood, gamma, xi, forward, emit = _forward_backward(x, initial, transition, means, variances)
    return panel, center, scale, initial, transition, means, variances, likelihood, gamma, iteration


@lru_cache(maxsize=1)
def ground_truth():
    panel, center, scale, initial, transition, means, variances, ll, gamma, iterations = _fit()
    worst = panel.sort_values(["net_flow_pct", "date"], ascending=[True, True]).iloc[0]
    original_means = means*scale+center
    raw = panel[["net_flow_pct", "weekly_liquidity_pct"]].to_numpy(float)
    x = (raw-center)/scale
    test = x[TRAIN_COUNT:]
    prior = gamma[-1]@transition
    filtered = []
    hmm_scores = []
    single_mean = x[:TRAIN_COUNT].mean(0)
    single_var = x[:TRAIN_COUNT].var(0, ddof=0)
    single_scores = []
    for observation in test:
        logemit = _log_density(observation[None, :], means, variances)[0]
        hmm_scores.append(float(logsumexp(np.log(prior)+logemit)))
        posterior = np.exp(np.log(prior)+logemit-hmm_scores[-1])
        filtered.append(posterior)
        single_scores.append(float(_log_density(observation[None, :], single_mean[None, :], single_var[None, :])[0, 0]))
        prior = posterior@transition
    filtered = np.asarray(filtered)
    test_panel = panel.iloc[TRAIN_COUNT:].reset_index(drop=True)
    maxindex = int(np.argmax(filtered[:, 1]))
    classifications = filtered[:, 1]>=.5
    # Compare next-day flow conditional on today's filtered state; last test date has no next day.
    usable = classifications[:-1]
    next_flow = test_panel.net_flow_pct.to_numpy(float)[1:]
    stress_next = float(next_flow[usable].mean())
    calm_next = float(next_flow[~usable].mean())
    residual = float(np.max(np.abs(gamma.sum(1)-1)))
    training = x[:TRAIN_COUNT]
    density = np.exp(_log_density(training, means, variances))
    alpha = initial*density[0]
    scale0 = float(alpha.sum())
    alpha/=scale0
    scaled_ll = np.log(scale0)
    for index in range(1, len(training)):
        alpha = (alpha@transition)*density[index]
        factor = float(alpha.sum())
        alpha/=factor
        scaled_ll+=np.log(factor)
    likelihood_residual = abs(float(scaled_ll)-ll)
    return (
        len(panel),
        panel.date.iloc[0].strftime("%Y-%m-%d"),
        panel.date.iloc[-1].strftime("%Y-%m-%d"),
        int(panel.accessions.max()),
        worst.date.strftime("%Y-%m-%d"),
        float(worst.net_flow_pct),
        float(worst.weekly_liquidity_pct),
        float(worst.sofr_pct),
        TRAIN_COUNT,
        iterations,
        ll,
        float(transition[0, 1]),
        float(transition[1, 0]),
        float(original_means[1, 0]),
        float(original_means[1, 1]),
        float(original_means[0, 0]),
        float(original_means[0, 1]),
        residual,
        likelihood_residual,
        test_panel.date.iloc[0].strftime("%Y-%m-%d"),
        test_panel.date.iloc[-1].strftime("%Y-%m-%d"),
        len(test_panel),
        test_panel.date.iloc[maxindex].strftime("%Y-%m-%d"),
        float(filtered[maxindex, 1]),
        float(test_panel.net_flow_pct.iloc[maxindex]),
        float(test_panel.weekly_liquidity_pct.iloc[maxindex]),
        float(test_panel.sofr_pct.iloc[maxindex]),
        float(np.mean(hmm_scores)),
        float(np.mean(single_scores)),
        int(classifications.sum()),
        stress_next,
        calm_next,
        bool(np.mean(hmm_scores)>np.mean(single_scores)),
    )


def _validate_subset(outputs, names):
    by = {v.name:v for v in variables}
    truth = dict(zip(by, ground_truth()))
    dec = dict(zip(by, DECIMALS))
    return validate_ordered_outputs(outputs, [by[n] for n in names], [truth[n] for n in names], [dec[n] for n in names])
def validate_turn_1(o): return _validate_subset(o, TURN_1_NAMES)
def validate_turn_2(o): return _validate_subset(o, TURN_2_NAMES)
def validate_turn_3(o): return _validate_subset(o, TURN_3_NAMES)
def validate_turn_4(o): return _validate_subset(o, TURN_4_NAMES)
def validate(o): return _validate_subset(o, [v.name for v in variables])
validators = {
    "validate_panel":turn_validator(validate_turn_1),
    "validate_hmm":turn_validator(validate_turn_2),
    "validate_filter":turn_validator(validate_turn_3),
    "validate_scores":turn_validator(validate_turn_4),
}
