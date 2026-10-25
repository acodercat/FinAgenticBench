"""A parametric payroll-protection book, a health-care quote, and capacity allocated to the original.

Five Louisiana counties are insured for a quarter of their 2019 manufacturing payroll
against 2021 NOAA flood events, with a time deductible, a county cap and an annual book
limit. An independent health-care book is quoted on the same counties with a shorter
deductible and allocated its own annual capacity; the capacity allocation and the
reinsurance layer then return to the original manufacturing book.

Rejected alternative conventions, each measured against the pinned reading:

- Allocating capacity with the health-care exposure instead of the manufacturing book's
  moves the exhaustion episode from 162128 to 165384 and its payment from 0.6873 to
  0.3857 million.
- Carrying the comparison's 12-hour deductible back into the original book moves the
  exhaustion episode to 162213 with a payment of 0.5000 million.
- Summing overlapping trigger intervals instead of covering overlapped time once raises
  original indemnity before the annual limit from 10.8127 to 11.2798 million.
- Paying the largest occurrences first instead of in chronological order raises the
  comparison's paid share of its exhaustion occurrence from 63.1958 to 73.9528 percent.
"""

from functools import lru_cache
import numpy as np, pandas as pd
from core.data import load_expansion_table


@lru_cache(None)
def inputs():
    q = load_expansion_table("bls_qcew")
    n = load_expansion_table("noaa_storm_events")
    x = q[
        q.year.eq(2019)
        & q.state_fips.eq("22")
        & q.aggregation_level.eq("county_naics_sector")
        & q.ownership_scope.eq("private")
        & q.industry_code.isin(["31-33", "62"])
    ]
    assert not x.duplicated(["area_fips", "industry_code"]).any()
    p = x.pivot(
        index="area_fips", columns="industry_code", values="total_annual_wages_usd"
    ).dropna()
    p = (
        p[(p > 0).all(axis=1)]
        .reset_index()
        .sort_values(["31-33", "area_fips"], ascending=[False, True])
        .head(5)
        .set_index("area_fips")
    )
    assert len(p) == 5
    s = n[
        n.state_usps.eq("LA")
        & n.county_zone_type.eq("C")
        & n.event_type.isin(["Flood", "Flash Flood"])
    ].copy()
    s["fips"] = "22" + s.county_zone_fips.str.zfill(3)
    s = s[s.fips.isin(p.index) & s.begin_datetime.str.startswith("2021")].copy()
    s["begin"] = pd.to_datetime(s.begin_datetime)
    s["end"] = pd.to_datetime(s.end_datetime) + pd.Timedelta(hours=48)
    s["end"] = s["end"].clip(upper=pd.Timestamp("2022-01-01"))
    assert (
        s.event_id.is_unique
        and s[["begin", "end"]].notna().all().all()
        and (s.end >= s.begin).all()
    )
    return p, s


def claims(p, s, sector, wait, interval="union"):
    rows = []
    for (ep, c), z in s.groupby(["episode_id", "fips"]):
        intervals = sorted(zip(z.begin, z.end))
        segments = []
        for a, b in intervals:
            if interval == "union" and segments and a <= segments[-1][1]:
                segments[-1] = (segments[-1][0], max(segments[-1][1], b))
            else:
                segments.append((a, b))
        hours = sum((b - a).total_seconds() / 3600 for a, b in segments)
        loss = min(max(hours - wait, 0) / 24 * 0.25 * float(p.loc[c, sector]) / 365, 500000)
        rows.append(dict(episode=ep, fips=c, hours=hours, start=min(z.begin), loss=loss))
    d = pd.DataFrame(rows)
    e = (
        d.groupby("episode")
        .agg(start=("start", "min"), loss=("loss", "sum"))
        .reset_index()
        .sort_values(["start", "episode"])
    )
    return e, d


def settle(e, order="chronological"):
    e = (
        e.sort_values(["start", "episode"])
        if order == "chronological"
        else e.sort_values(["loss", "episode"], ascending=[False, True])
    )
    e = e.copy()
    prior = e.loss.cumsum().shift(fill_value=0)
    e["paid"] = np.minimum(e.loss, np.maximum(8e6 - prior, 0))
    ex = e[e.paid.cumsum() >= 8e6 - 1e-7].iloc[0]
    return e, ex


def calculate(
    exposure_state="original",
    waiting_state="original",
    interval="union",
    allocation_order="chronological",
):
    p, s = inputs()
    a, ad = claims(p, s, "31-33", 24, interval)
    b, bd = claims(p, s, "62", 12, interval)
    bs, bex = settle(b, allocation_order)
    late, _ = claims(
        p,
        s,
        "31-33" if exposure_state == "original" else "62",
        24 if waiting_state == "original" else 12,
        interval,
    )
    paid, ex = settle(late, allocation_order)
    ceded = np.minimum(np.maximum(paid.paid.to_numpy(float) - 250000, 0), 750000)
    retained = paid.paid.to_numpy(float) - ceded
    rank = sorted(range(len(paid)), key=lambda i: (-retained[i], str(paid.episode.iloc[i])))
    lead = a.sort_values(["loss", "episode"], ascending=[False, True]).iloc[0]
    out = dict(
        original_insured_payroll_million=float(p["31-33"].sum() * 0.25 / 1e6),
        largest_exposure_county=str(p.index[0]),
        original_preaggregate_claims_million=float(a.loss.sum() / 1e6),
        largest_original_occurrence_million=float(lead.loss / 1e6),
        comparison_insured_payroll_million=float(p["62"].sum() * 0.25 / 1e6),
        comparison_preaggregate_claims_million=float(b.loss.sum() / 1e6),
        comparison_exhaustion_episode=str(bex.episode),
        comparison_exhaustion_paid_pct=float(bex.paid / bex.loss * 100),
        original_exhaustion_episode=str(ex.episode),
        original_exhaustion_payment_million=float(ex.paid / 1e6),
        original_ceded_claims_million=float(ceded.sum() / 1e6),
        largest_retained_episode=str(paid.episode.iloc[rank[0]]),
    )
    return out, dict(
        counties=p.index.tolist(),
        source_rows=len(s),
        original_episodes=a.assign(start=a.start.astype(str)).to_dict("records"),
        comparison_episodes=b.assign(start=b.start.astype(str)).to_dict("records"),
        original_county_occurrences=ad.assign(start=ad.start.astype(str)).to_dict("records"),
        final_paid_episodes=paid.assign(start=paid.start.astype(str)).to_dict("records"),
        ceded=ceded.tolist(),
        retained=retained.tolist(),
        retained_leader_margin=float(retained[rank[0]] - retained[rank[1]]),
        aggregate_conservation=float(paid.paid.sum() - 8e6),
        reinsurance_conservation=float((ceded + retained - paid.paid).abs().max()),
    )


from cave_agent import Variable
from core.validation import validate_ordered_outputs, turn_validator

TURN_1_NAMES = ["original_insured_payroll_million", "largest_exposure_county"]
TURN_2_NAMES = ["original_preaggregate_claims_million", "largest_original_occurrence_million"]
TURN_3_NAMES = ["comparison_insured_payroll_million", "comparison_preaggregate_claims_million"]
TURN_4_NAMES = ["comparison_exhaustion_episode", "comparison_exhaustion_paid_pct"]
TURN_5_NAMES = ["original_exhaustion_episode", "original_exhaustion_payment_million"]
TURN_6_NAMES = ["original_ceded_claims_million", "largest_retained_episode"]

variables = [
    Variable(
        "original_insured_payroll_million",
        None,
        "Store original total insured annual payroll in USD millions, rounded to 4 decimals. Compute from your own unrounded source inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "largest_exposure_county",
        None,
        "Store largest insured-exposure county as an exact five-character FIPS string.",
    ),
    Variable(
        "original_preaggregate_claims_million",
        None,
        "Store original total indemnity before the annual limit in USD millions, rounded to 4 decimals. Compute from your own unrounded source inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "largest_original_occurrence_million",
        None,
        "Store largest original occurrence indemnity before the annual limit in USD millions, rounded to 4 decimals. Compute from your own unrounded source inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "comparison_insured_payroll_million",
        None,
        "Store comparison total insured annual payroll in USD millions, rounded to 4 decimals. Compute from your own unrounded source inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "comparison_preaggregate_claims_million",
        None,
        "Store comparison total indemnity before the annual limit in USD millions, rounded to 4 decimals. Compute from your own unrounded source inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "comparison_exhaustion_episode",
        None,
        "Store comparison exhaustion episode identifier exactly as a source string.",
    ),
    Variable(
        "comparison_exhaustion_paid_pct",
        None,
        "Store comparison exhaustion occurrence paid share in percent, rounded to 4 decimals. Compute from your own unrounded source inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "original_exhaustion_episode",
        None,
        "Store original exhaustion episode identifier exactly as a source string.",
    ),
    Variable(
        "original_exhaustion_payment_million",
        None,
        "Store original exhaustion occurrence payment in USD millions, rounded to 4 decimals. Compute from your own unrounded source inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "original_ceded_claims_million",
        None,
        "Store original total reinsurance recoveries in USD millions, rounded to 4 decimals. Compute from your own unrounded source inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "largest_retained_episode",
        None,
        "Store largest original retained-payment episode identifier exactly as a source string.",
    ),
]
DECIMALS = [4, None, 4, 4, 4, 4, None, 4, None, 4, 4, None]


@lru_cache(maxsize=1)
def ground_truth():
    out = calculate()[0]
    return tuple(out[v.name] for v in variables)


def _validate_subset(outputs, names):
    by = {v.name: v for v in variables}
    truth = dict(zip(by, ground_truth()))
    places = dict(zip(by, DECIMALS))
    return validate_ordered_outputs(
        outputs, [by[n] for n in names], [truth[n] for n in names], [places[n] for n in names]
    )


def validate(outputs):
    return _validate_subset(outputs, [v.name for v in variables])


def validate_turn_1(outputs):
    return _validate_subset(outputs, TURN_1_NAMES)


def validate_turn_2(outputs):
    return _validate_subset(outputs, TURN_2_NAMES)


def validate_turn_3(outputs):
    return _validate_subset(outputs, TURN_3_NAMES)


def validate_turn_4(outputs):
    return _validate_subset(outputs, TURN_4_NAMES)


def validate_turn_5(outputs):
    return _validate_subset(outputs, TURN_5_NAMES)


def validate_turn_6(outputs):
    return _validate_subset(outputs, TURN_6_NAMES)


validators = {
    f"validate_turn_{i}": turn_validator(globals()[f"validate_turn_{i}"]) for i in range(1, 7)
}
