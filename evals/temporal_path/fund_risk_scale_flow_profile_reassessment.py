"""Baseline: distinguish fund size, risk intensity and share flows.

N-PORT is one source family; Treasury yields supply the external curve scenario.
The scenario reuses endpoint exposures at a common September NAV and is not a
realized-return calculation. All decisions use exact decimal-input fractions;
validators never read author answer files. Non-derivative gains are intentionally
not used to label the NAV bridge remainder as total investment performance.

Rejected alternative conventions, each measured against the pinned reading:

- Ranking declines by percentage rather than dollar DV01 selects a different five-fund
  cohort, an aggregate decline of 266,136.2434 instead of 3,003,448.0912, and series
  S000032956 instead of S000020634.
- Applying the USD 1 billion floor to September as well as June leaves 261 eligible pairs
  instead of 263 and replaces S000021221 in the cohort with S000006579.
- Using the June profile without rescaling it to September net assets gives a June
  scenario value change of 27,753,560.54 instead of 22,456,653.60 USD and a largest
  tenor difference of -1,987,252.39 instead of 530,781.57 USD.
- Crediting positive DV01 for a yield rise instead of a decline reverses the sign of all
  three scenario outputs.
"""

from fractions import Fraction as F
from functools import lru_cache
import math

import pandas as pd
from cave_agent import Variable
from core.data import load_expansion_table
from core.validation import validate_ordered_outputs, turn_validator

BUCKETS = ["dv01_3month", "dv01_1year", "dv01_5year", "dv01_10year", "dv01_30year"]
TENORS = ["3 Mo", "1 Yr", "5 Yr", "10 Yr", "30 Yr"]
TOKENS = ["3M", "1Y", "5Y", "10Y", "30Y"]
DATES = ["2025-06-30", "2025-09-30"]
SPECS = [
    (
        "eligible_pairs",
        "Store the eligible paired-series count before selecting positive DV01 declines as an integer.",
        0,
    ),
    (
        "decliner_series",
        "Store a Python list of exactly five verbatim SEC series-ID strings, in descending DV01-decline order with the specified tie-break; no extra elements or aliases.",
        None,
    ),
    (
        "cohort_dv01_decline",
        "Store the cohort aggregate DV01 decline in USD per basis point, rounded to 4 decimals.",
        4,
    ),
    ("selected_series", "Store the selected SEC series ID verbatim as a string.", None),
    (
        "june_unit_dv01",
        "Store June USD DV01 in USD per basis point per USD 1 million of net assets, rounded to 4 decimals.",
        4,
    ),
    (
        "september_unit_dv01",
        "Store September USD DV01 in USD per basis point per USD 1 million of net assets, rounded to 4 decimals.",
        4,
    ),
    (
        "selected_dv01_change_pct",
        "Store the selected fund September-minus-June total USD DV01 change as a percentage of June total USD DV01, rounded to 4 decimals.",
        4,
    ),
    ("september_nav_usd", "Store September net assets in USD, rounded to 2 decimals.", 2),
    (
        "net_share_transactions_usd",
        "Store the quarter net reported share-transaction amount in USD, positive for net issuance and negative for net redemption, rounded to 2 decimals.",
        2,
    ),
    (
        "nav_bridge_remainder_usd",
        "Store the portion of net-asset change not covered by net reported share transactions in USD, rounded to 2 decimals.",
        2,
    ),
    (
        "june_profile_value_change_usd",
        "Store the June risk-profile scenario value change at September net assets in USD, rounded to 2 decimals.",
        2,
    ),
    (
        "september_profile_value_change_usd",
        "Store the September risk-profile scenario value change at September net assets in USD, rounded to 2 decimals.",
        2,
    ),
    (
        "largest_profile_difference_tenor",
        "Store exactly one tenor token: 3M, 1Y, 5Y, 10Y or 30Y.",
        None,
    ),
    (
        "largest_profile_difference_usd",
        "Store the selected tenor signed September-profile-minus-June-profile scenario contribution in USD, rounded to 2 decimals.",
        2,
    ),
]
variables = [Variable(n, None, d) for n, d, _ in SPECS]
DECIMALS = [places for _, _, places in SPECS]
TURN_NAMES = [[s[0] for s in SPECS[a:b]] for a, b in [(0, 3), (3, 7), (7, 10), (10, 14)]]


def fraction(value):
    return F(str(value))


def finite(value):
    return pd.notna(value) and not isinstance(value, bool) and math.isfinite(float(value))


def rank_profiles(records, decline="dollar"):
    decliners = [r for r in records if r["drop"] > 0]

    def size(r):
        return r["drop"] if decline == "dollar" else r["drop"] / r["dv0"]

    cohort = sorted(decliners, key=lambda r: (-size(r), r["series"]))[:5]
    if len(cohort) != 5:
        raise ValueError("fewer than five qualifying decliners")
    selected = min(
        cohort, key=lambda r: (-(r["dv1"] / r["nav1"] - r["dv0"] / r["nav0"]), r["series"])
    )
    return cohort, selected


def scenario(dv0, dv1, nav0, nav1, changes, scaling="scaled", sign="decline_gains"):
    if nav0 <= 0 or nav1 <= 0 or not len(dv0) == len(dv1) == len(changes) == 5:
        raise ValueError("invalid scenario domain")
    scale = nav1 / nav0 if scaling == "scaled" else 1
    direction = -1 if sign == "decline_gains" else 1
    old = [direction * x * d * scale for x, d in zip(changes, dv0)]
    new = [direction * x * d for x, d in zip(changes, dv1)]
    difference = [b - a for a, b in zip(old, new)]
    winner = min(range(5), key=lambda i: (-abs(difference[i]), i))
    return sum(old), sum(new), TOKENS[winner], difference[winner], old, new


@lru_cache(maxsize=None)
def analysis(decline="dollar", nav_floor="june", scaling="scaled", sign="decline_gains"):
    """Every output; the keyword defaults are the conventions the query pins."""
    funds = load_expansion_table("sec_nport")
    f = funds[
        funds.form.eq("NPORT-P")
        & funds.filing_date.le("2025-12-31")
        & funds.report_date.isin(DATES)
        & funds.series_id.notna()
    ].copy()
    if f.accession.duplicated().any():
        raise ValueError("duplicate fund accession")
    f = f.sort_values(["filing_date", "accession"]).drop_duplicates(
        ["series_id", "report_date"], keep="last"
    )
    r = load_expansion_table("sec_nport_interest_rate_risk")
    r = r[r.currency_code.eq("USD")].copy()
    counts = r.groupby("accession").size()
    r = r[r.accession.isin(counts[counts.eq(1)].index)]
    # Eligibility applies after deterministic filing selection; no fallback filing.
    joined = f.merge(
        r[["accession", "report_date", "series_id"] + BUCKETS],
        on="accession",
        suffixes=("", "_risk"),
        validate="one_to_one",
    )
    if not (
        joined.report_date.eq(joined.report_date_risk) & joined.series_id.eq(joined.series_id_risk)
    ).all():
        raise ValueError("inconsistent accession identity across fund and risk tables")
    snapshots = {}
    for row in joined.to_dict("records"):
        if not all(finite(row[c]) for c in ["net_assets_usd"] + BUCKETS):
            continue
        row["nav"] = fraction(row["net_assets_usd"])
        row["vector"] = [fraction(row[c]) for c in BUCKETS]
        row["dv"] = sum(row["vector"])
        snapshots.setdefault(row["series_id"], {})[row["report_date"]] = row
    records = []
    for series, dates in snapshots.items():
        if not all(d in dates for d in DATES):
            continue
        a, b = [dates[d] for d in DATES]
        floor_failed = a["nav"] < 10**9 or (nav_floor == "both" and b["nav"] < 10**9)
        if floor_failed or b["nav"] <= 0 or a["dv"] <= 0 or b["dv"] <= 0:
            continue
        records.append(
            dict(
                series=series,
                nav0=a["nav"],
                nav1=b["nav"],
                dv0=a["dv"],
                dv1=b["dv"],
                drop=a["dv"] - b["dv"],
            )
        )
    cohort, selected = rank_profiles(records, decline)
    a, b = [snapshots[selected["series"]][d] for d in DATES]
    fields = [
        f"{kind}_month_{m}_usd"
        for m in [1, 2, 3]
        for kind in ["sales", "reinvestment", "redemptions"]
    ]
    if not all(finite(b[k]) and b[k] >= 0 for k in fields):
        raise ValueError("selected share transactions require nonnegative finite disclosed amounts")
    net = sum(
        fraction(b[f"sales_month_{m}_usd"])
        + fraction(b[f"reinvestment_month_{m}_usd"])
        - fraction(b[f"redemptions_month_{m}_usd"])
        for m in [1, 2, 3]
    )
    change = b["nav"] - a["nav"]
    remainder = change - net
    curve = load_expansion_table("treasury_yield_curve")
    curve = curve[curve.Date.isin(DATES)].set_index("Date")
    if len(curve) != 2 or curve.index.duplicated().any():
        raise ValueError("missing or duplicate scenario curve")
    if not all(finite(curve.loc[d, t]) for d in DATES for t in TENORS):
        raise ValueError("incomplete curve")
    changes = [
        100 * (fraction(curve.loc[DATES[1], t]) - fraction(curve.loc[DATES[0], t])) for t in TENORS
    ]
    old, new, tenor, contribution, old_parts, new_parts = scenario(
        a["vector"], b["vector"], a["nav"], b["nav"], changes, scaling, sign
    )
    values = [
        len(records),
        [r["series"] for r in cohort],
        sum(r["drop"] for r in cohort),
        selected["series"],
        a["dv"] / a["nav"] * 10**6,
        b["dv"] / b["nav"] * 10**6,
        (b["dv"] / a["dv"] - 1) * 100,
        b["nav"],
        net,
        remainder,
        old,
        new,
        tenor,
        contribution,
    ]
    answer = {s[0]: float(v) if isinstance(v, F) else v for s, v in zip(SPECS, values)}
    evidence = dict(
        start_accession=a["accession"],
        end_accession=b["accession"],
        name=b["series_name"],
        start_nav=str(a["nav"]),
        end_nav=str(b["nav"]),
        start_dv01=str(a["dv"]),
        end_dv01=str(b["dv"]),
        start_vector=[str(x) for x in a["vector"]],
        end_vector=[str(x) for x in b["vector"]],
        nav_change_usd=float(change),
        net_asset_change_pct=float(change / a["nav"] * 100),
        curve_changes_bp=[float(x) for x in changes],
        june_contributions=[float(x) for x in old_parts],
        september_contributions=[float(x) for x in new_parts],
        share_fields={k: float(b[k]) for k in fields},
        cohort=[{k: str(v) if isinstance(v, F) else v for k, v in x.items()} for x in cohort],
        invariant_bridge_exact=(change == net + remainder),
        unscaled_june_scenario=float(sum(-s * d for s, d in zip(changes, a["vector"]))),
    )
    return answer, evidence


def ground_truth():
    answer, _ = analysis()
    return tuple(answer[v.name] for v in variables)


def _validate(outputs, names):
    answer, _ = analysis()
    specs = [s for s in SPECS if s[0] in names]
    clean = dict(outputs)
    if "decliner_series" in names and clean.get("decliner_series") is not None:
        value = clean["decliner_series"]
        if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
            clean["decliner_series"] = "<invalid ordered series list>"
    return validate_ordered_outputs(
        clean,
        [Variable(n, None, d) for n, d, _ in specs],
        [answer[n] for n, _, _ in specs],
        [p for _, _, p in specs],
    )


def validate(outputs):
    return _validate(outputs, [v.name for v in variables])


def validate_cohort(outputs):
    return _validate(outputs, TURN_NAMES[0])


def validate_intensity(outputs):
    return _validate(outputs, TURN_NAMES[1])


def validate_share_bridge(outputs):
    return _validate(outputs, TURN_NAMES[2])


def validate_profile_scenario(outputs):
    return _validate(outputs, TURN_NAMES[3])


validators = {
    name: turn_validator(fn)
    for name, fn in [
        ("validate_cohort", validate_cohort),
        ("validate_intensity", validate_intensity),
        ("validate_share_bridge", validate_share_bridge),
        ("validate_profile_scenario", validate_profile_scenario),
    ]
}
