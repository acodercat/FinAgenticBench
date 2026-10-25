"""Trace a large reported sale through disclosure lineage and market windows.

The frozen exact-Form-4 screen selects accession 0001883850-24-000012: its
eligible S/D sale block reports 16,848 shares at 900 USD, or 15,163,200 USD,
and the filing date follows the transaction date by 120 calendar days.  The
same issuer, reporting-owner set and specified transaction fingerprint occur in
0001883850-24-000007, filed 2024-01-04.  This establishes an earlier matching
record in the supplied original-Form-4 universe; field equality alone neither
proves unique economic-event identity nor establishes the first publication in
every possible source.

The next-observed-trading-day convention maps the matching and selected filings
to 2024-01-05 and 2024-05-02.  MIDAS begins 2024-01-02, leaving only three
observations before the first date, while both ten-day post-windows are complete.
Their mean on-exchange volumes are 238,754.5 and 216,937.6 shares.  Hidden
volume divided by the MIDAS volume eligible for hidden orders is 47.6949 and
51.7716 percent.  Dividing by all trade volume instead gives 39.5121 and
46.1972 percent; averaging daily hidden shares gives 47.5475 and 51.1214.
The query pins the ratio-of-sums over the applicable denominator.  Counting
the filing date itself as usable moves the matched clock to 2024-01-04 with two
pre-window observations and a 236,888.2-share post-window mean.

The case is a baseline because the public task explicitly requests the lineage
search, observation convention and MIDAS denominator.  Its diagnostic value is
whether an analyst preserves the information clock and refuses to manufacture a
complete pre-window.  The market outputs are descriptive venue-level activity,
not security returns, trading profit or a causal disclosure effect.

Turn 4 is not reconstructible from turns 1-3: adding 1,000 shares to one daily
post-window volume in a synthetic data perturbation leaves all earlier outputs
unchanged but raises the turn-4 daily mean by 100 shares.
"""

from collections import Counter
from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table, load_insider_table
from core.validation import boolean_equal, turn_validator, validate_ordered_outputs

FILING_START = "2024-01-01"
FILING_END = "2024-12-31"
MINIMUM_CALENDAR_LAG = 7
FORM_TYPE = "4"
WINDOW_DAYS = 10
VOLUME_COLUMNS = [
    "trade_volume_thousands",
    "hidden_volume_thousands",
    "trade_volume_for_hidden_thousands",
]
FINGERPRINT = [
    "trans_date",
    "trans_code",
    "trans_acquired_disp_cd",
    "security_title",
    "trans_shares",
    "trans_pricepershare",
    "shrs_ownd_folwng_trans",
    "direct_indirect_ownership",
]


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = ["selected_accession", "sale_amount_usd", "maximum_lag_days"]
TURN_2_NAMES = ["matched_accession", "matched_filing_date"]
TURN_3_NAMES = [
    "matched_usable_date",
    "matched_pre_days",
    "matched_post_days",
    "matched_complete_window",
    "selected_usable_date",
    "selected_pre_days",
    "selected_post_days",
    "selected_complete_window",
]
TURN_4_NAMES = [
    "matched_post_mean_shares",
    "matched_post_hidden_pct",
    "selected_post_mean_shares",
    "selected_post_hidden_pct",
]

variables = [
    _v(
        TURN_1_NAMES[0],
        "Store the selected filing accession as its exact hyphenated SEC accession string.",
    ),
    _v(
        TURN_1_NAMES[1],
        "Store the selected filing's aggregate eligible sale value in USD, rounded to 2 decimals.",
    ),
    _v(TURN_1_NAMES[2], "Store the selected filing's largest calendar-day interval as an integer."),
    _v(
        TURN_2_NAMES[0],
        "Store the matched filing accession as its exact hyphenated SEC accession string.",
    ),
    _v(TURN_2_NAMES[1], "Store the matched filing date as an ISO YYYY-MM-DD string."),
    _v(
        TURN_3_NAMES[0],
        "Store the matched filing's analysis availability date as an ISO YYYY-MM-DD string.",
    ),
    _v(
        TURN_3_NAMES[1],
        "Store the matched filing's available pre-window observation count as an integer from 0 through 10.",
    ),
    _v(
        TURN_3_NAMES[2],
        "Store the matched filing's available post-window observation count as an integer from 0 through 10.",
    ),
    _v(
        TURN_3_NAMES[3],
        "Store the matched filing's full-window availability as a Boolean true or false.",
    ),
    _v(
        TURN_3_NAMES[4],
        "Store the selected filing's analysis availability date as an ISO YYYY-MM-DD string.",
    ),
    _v(
        TURN_3_NAMES[5],
        "Store the selected filing's available pre-window observation count as an integer from 0 through 10.",
    ),
    _v(
        TURN_3_NAMES[6],
        "Store the selected filing's available post-window observation count as an integer from 0 through 10.",
    ),
    _v(
        TURN_3_NAMES[7],
        "Store the selected filing's full-window availability as a Boolean true or false.",
    ),
    _v(
        TURN_4_NAMES[0],
        "Store the matched filing's post-window mean daily volume in shares, rounded to 1 decimal.",
    ),
    _v(
        TURN_4_NAMES[1],
        "Store the matched filing's post-window hidden-volume percentage, rounded to 4 decimals.",
    ),
    _v(
        TURN_4_NAMES[2],
        "Store the selected filing's post-window mean daily volume in shares, rounded to 1 decimal.",
    ),
    _v(
        TURN_4_NAMES[3],
        "Store the selected filing's post-window hidden-volume percentage, rounded to 4 decimals.",
    ),
]

DECIMALS = [None, 2, 0, None, None, None, 0, 0, None, None, 0, 0, None, 1, 4, 1, 4]


def _original_form_four(submissions):
    return submissions.loc[
        submissions.document_type.astype(str).eq(FORM_TYPE)
        & submissions.filing_date.astype(str).between(FILING_START, FILING_END)
    ]


@lru_cache(maxsize=1)
def _event_and_lineage():
    submissions = load_insider_table("submission")
    transactions = load_insider_table("nonderiv_trans")
    owners = load_insider_table("reportingowner")
    if submissions.accession_number.duplicated().any():
        raise ValueError("submission accession must be unique")
    if transactions.nonderiv_trans_sk.duplicated().any():
        raise ValueError("non-derivative transaction key must be unique")

    filing_window = _original_form_four(submissions)
    sales = transactions.merge(
        filing_window[["accession_number", "filing_date", "issuercik", "issuertradingsymbol"]],
        on="accession_number",
        validate="many_to_one",
    )
    sales = sales.loc[
        sales.trans_code.astype(str).eq("S")
        & sales.trans_acquired_disp_cd.astype(str).eq("D")
        & sales.trans_date.astype(str).between(FILING_START, FILING_END)
    ].copy()
    for column in ("trans_shares", "trans_pricepershare"):
        sales[column] = pd.to_numeric(sales[column], errors="coerce")
    sales = sales.loc[sales.trans_shares.gt(0) & sales.trans_pricepershare.gt(0)].copy()
    sales["sale_value"] = sales.trans_shares * sales.trans_pricepershare
    sales["calendar_lag"] = (
        pd.to_datetime(sales.filing_date) - pd.to_datetime(sales.trans_date)
    ).dt.days
    ranked = (
        sales.groupby(["accession_number", "issuertradingsymbol"], dropna=False)
        .agg(sale_value=("sale_value", "sum"), maximum_lag=("calendar_lag", "max"))
        .reset_index()
    )
    ranked = ranked.loc[
        ranked.maximum_lag.ge(MINIMUM_CALENDAR_LAG) & ranked.issuertradingsymbol.notna()
    ].sort_values(["sale_value", "accession_number"], ascending=[False, True])
    if ranked.empty:
        raise ValueError("no eligible delayed-sale filing")
    selected = ranked.iloc[0]
    selected_rows = sales.loc[sales.accession_number.eq(selected.accession_number)]
    block = selected_rows.loc[selected_rows.calendar_lag.eq(selected.maximum_lag)]
    if block.empty or not block[FINGERPRINT].notna().all().all():
        raise ValueError("selected maximum-lag block has an incomplete fingerprint")

    issuer_cik = selected_rows.issuercik.iloc[0]
    history = submissions.loc[
        submissions.issuercik.eq(issuer_cik)
        & submissions.document_type.astype(str).eq(FORM_TYPE)
        & submissions.filing_date.astype(str).le(FILING_END)
    ]
    history_accessions = set(history.accession_number)
    owner_sets = {
        accession: frozenset(group.rptownercik.dropna().astype(str))
        for accession, group in owners.loc[
            owners.accession_number.isin(history_accessions)
        ].groupby("accession_number")
    }
    selected_owner_set = owner_sets.get(selected.accession_number, frozenset())
    if not selected_owner_set:
        raise ValueError("selected filing has no reporting-owner CIK")
    required = Counter(block[FINGERPRINT].itertuples(index=False, name=None))
    matches = []
    relevant_transactions = transactions.loc[transactions.accession_number.isin(history_accessions)]
    for accession, group in relevant_transactions.groupby("accession_number"):
        if owner_sets.get(accession) != selected_owner_set:
            continue
        observed = Counter(group[FINGERPRINT].itertuples(index=False, name=None))
        if all(observed[fingerprint] >= count for fingerprint, count in required.items()):
            matches.append(accession)
    lineage = history.loc[history.accession_number.isin(matches)].sort_values(
        ["filing_date", "accession_number"]
    )
    if lineage.empty:
        raise ValueError("selected block has no matching disclosure")
    first = lineage.iloc[0]
    return {
        "selected_accession": str(selected.accession_number),
        "selected_filing_date": str(selected_rows.filing_date.iloc[0]),
        "ticker": str(selected.issuertradingsymbol),
        "sale_value": float(selected.sale_value),
        "maximum_lag": int(selected.maximum_lag),
        "matched_accession": str(first.accession_number),
        "matched_filing_date": str(first.filing_date),
    }


def _market_window(
    market,
    ticker,
    filing_date,
    availability="after_filing",
    denominator="applicable",
    aggregation="ratio_of_sums",
):
    """One filing's event clock; the keyword defaults are the pinned conventions."""
    rows = (
        market.loc[
            market.security_type.astype(str).eq("Stock")
            & market.ticker.astype(str).eq(ticker)
            & market.date.astype(str).le(FILING_END)
        ]
        .copy()
        .sort_values("date")
    )
    if rows.empty or rows.date.duplicated().any():
        raise ValueError("selected stock must have one MIDAS row per observed date")
    for column in VOLUME_COLUMNS:
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    dates = rows.date.astype(str)
    opens = dates.gt(filing_date) if availability == "after_filing" else dates.ge(filing_date)
    usable = rows.loc[opens, "date"]
    if usable.empty:
        raise ValueError("no observed trading date after filing")
    usable_date = str(usable.iloc[0])
    pre = rows.loc[rows.date.astype(str).lt(usable_date)].tail(WINDOW_DAYS)
    post = rows.loc[rows.date.astype(str).ge(usable_date)].head(WINDOW_DAYS)
    if len(post) != WINDOW_DAYS or post[VOLUME_COLUMNS].isna().any().any():
        raise ValueError("post-window is incomplete or nonnumeric")
    if denominator == "applicable":
        base = post.trade_volume_for_hidden_thousands
    else:
        base = post.trade_volume_thousands
    if float(base.sum()) <= 0:
        raise ValueError("hidden-volume denominator must be positive")
    if aggregation == "ratio_of_sums":
        hidden_pct = float(post.hidden_volume_thousands.sum() / base.sum() * 100)
    else:
        hidden_pct = float((post.hidden_volume_thousands / base).mean() * 100)
    return (
        usable_date,
        int(len(pre)),
        int(len(post)),
        bool(len(pre) == len(post) == WINDOW_DAYS),
        float(post.trade_volume_thousands.mean() * 1000),
        hidden_pct,
    )


@lru_cache(maxsize=None)
def outputs_under(
    availability="after_filing", denominator="applicable", aggregation="ratio_of_sums"
):
    """Every output under one set of market-clock conventions; the defaults are pinned."""
    event = _event_and_lineage()
    market = load_expansion_table("sec_midas_security_exchange")
    clock = dict(availability=availability, denominator=denominator, aggregation=aggregation)
    matched = _market_window(market, event["ticker"], event["matched_filing_date"], **clock)
    selected = _market_window(market, event["ticker"], event["selected_filing_date"], **clock)
    return (
        event["selected_accession"],
        event["sale_value"],
        event["maximum_lag"],
        event["matched_accession"],
        event["matched_filing_date"],
        *matched[:4],
        *selected[:4],
        matched[4],
        matched[5],
        selected[4],
        selected[5],
    )


def ground_truth():
    return outputs_under()


def _validate_subset(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip(by_name, ground_truth()))
    decimals = dict(zip(by_name, DECIMALS))
    normalized = dict(outputs)
    for name in ("matched_complete_window", "selected_complete_window"):
        value = normalized.get(name)
        if value is not None:
            normalized[name] = (
                truth[name] if boolean_equal(value, truth[name]) else "<invalid Boolean>"
            )
    return validate_ordered_outputs(
        normalized,
        [by_name[name] for name in names],
        [truth[name] for name in names],
        [decimals[name] for name in names],
    )


def validate_turn_1(outputs):
    return _validate_subset(outputs, TURN_1_NAMES)


def validate_turn_2(outputs):
    return _validate_subset(outputs, TURN_2_NAMES)


def validate_turn_3(outputs):
    return _validate_subset(outputs, TURN_3_NAMES)


def validate_turn_4(outputs):
    return _validate_subset(outputs, TURN_4_NAMES)


def validate(outputs):
    return _validate_subset(outputs, [variable.name for variable in variables])


validators = {
    "validate_sale_filing_selection": turn_validator(validate_turn_1),
    "validate_disclosure_lineage": turn_validator(validate_turn_2),
    "validate_market_window_coverage": turn_validator(validate_turn_3),
    "validate_post_window_activity": turn_validator(validate_turn_4),
}
