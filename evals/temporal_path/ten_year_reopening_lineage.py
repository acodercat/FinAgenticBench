"""Hard cross-source baseline joining an auction lineage to its MSPD representation.

CUSIP 91282CMM0 has an original 10-year auction and two reopenings whose shorter
remaining-term labels must not split the lineage.  A literal 10-Year filter keeps
one auction, 42.0 billion USD offered and 53.5379607 billion accepted instead of
three auctions, 120.0 billion offered and 138.7086321 billion accepted.

At 2025-04-30 MSPD represents the same CUSIP with three issue-date rows.  Keeping
only the original-issue row gives 53.5386697 billion USD issued instead of the
138.7042713 billion cumulative issued amount.  CUSIP-level outstanding is reported
once, not once per issue row, and is also 138.7042713 billion.  Auction accepted
minus MSPD issued is +0.0043608 billion, so forcing a zero cross-source residual is
not valid.  Numeric validation uses 0.6 x 10^-N rounding-boundary tolerance; dates,
lists, counts and the CUSIP are exact.  Because the query explicitly requests the
CUSIP lineage, reopenings, cumulative issue rows and stock-versus-flow explanation,
the literal-term and original-row alternatives are regression probes rather than
unprompted diagnostic traps.
"""

from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import validate_ordered_outputs, turn_validator


CUSIP = "91282CMM0"
MSPD_DATE = "2025-04-30"

variables = [
    Variable("auction_count", None, "Store the number of 2025 auctions in the CUSIP lineage as an integer."),
    Variable("original_auction_date", None, "Store the original non-reopening auction date as a YYYY-MM-DD string."),
    Variable("reopening_auction_dates", None, "Store the reopening auction dates as a chronological list of YYYY-MM-DD strings."),
    Variable("reported_security_terms", None, "Store the distinct reported security-term labels in chronological first-appearance order."),
    Variable("original_issue_date", None, "Store the lineage's original issue date as a YYYY-MM-DD string."),
    Variable("maturity_date", None, "Store the common maturity date as a YYYY-MM-DD string."),
    Variable("coupon_rate_pct", None, "Store the common coupon rate as a percentage, rounded to 3 decimals."),
    Variable("auction_offering_total_usd_billions", None, "Store cumulative auction offering amount in USD billions, rounded to 3 decimals."),
    Variable("auction_accepted_total_usd_billions", None, "Store cumulative auction total accepted in USD billions, rounded to 6 decimals."),
    Variable("mspd_issue_row_count", None, "Store the number of issue-date rows for the CUSIP in the 2025-04-30 MSPD detail as an integer."),
    Variable("mspd_cumulative_issued_usd_billions", None, "Store cumulative issued amount across those MSPD issue-date rows in USD billions, rounded to 6 decimals."),
    Variable("mspd_reported_outstanding_usd_billions", None, "Store the CUSIP-level outstanding amount reported in the 2025-04-30 MSPD detail in USD billions, rounded to 6 decimals."),
    Variable("accepted_minus_mspd_issued_usd_billions", None, "Store cumulative auction accepted minus cumulative MSPD issued in USD billions, rounded to 6 decimals."),
]


def ground_truth():
    auctions = load_expansion_table("treasury_auctions")
    rows = auctions.loc[
        auctions.cusip.eq(CUSIP)
        & auctions.auction_date.between("2025-01-01", "2025-12-31")
    ].sort_values("auction_date")
    original = rows.loc[rows.reopening.eq("No")]
    reopenings = rows.loc[rows.reopening.eq("Yes")]
    if len(rows) != 3 or len(original) != 1 or len(reopenings) != 2:
        raise ValueError("unexpected auction lineage")
    original_issue = str(original.iloc[0].issue_date)
    if not reopenings.original_issue_date.eq(original_issue).all():
        raise ValueError("reopenings do not identify the original issue")
    maturity = rows.maturity_date.drop_duplicates()
    coupon = rows.int_rate.drop_duplicates()
    if len(maturity) != 1 or len(coupon) != 1:
        raise ValueError("maturity or coupon changed within the CUSIP lineage")

    mspd = load_expansion_table("treasury_marketable_securities")
    stock = mspd.loc[
        mspd.record_date.eq(MSPD_DATE)
        & mspd.security_identifier_or_total_label.eq(CUSIP)
    ].sort_values("issue_date")
    if len(stock) != 3 or set(stock.issue_date) != set(rows.issue_date):
        raise ValueError("MSPD issue rows do not match the auction lineage")
    outstanding = stock.outstanding_million_usd.dropna()
    if len(outstanding) != 1:
        raise ValueError("expected one CUSIP-level MSPD outstanding amount")

    accepted = float(rows.total_accepted.sum() / 1e9)
    issued = float(stock.issued_million_usd.sum() / 1e3)
    terms = list(dict.fromkeys(rows.security_term.astype(str)))
    return (
        len(rows), str(original.iloc[0].auction_date),
        reopenings.auction_date.astype(str).tolist(), terms, original_issue,
        str(maturity.iloc[0]), float(coupon.iloc[0]),
        float(rows.offering_amt.sum() / 1e9), accepted, len(stock), issued,
        float(outstanding.iloc[0] / 1e3), accepted - issued,
    )


def validate(outputs):
    return validate_ordered_outputs(
        outputs, variables, ground_truth(),
        [0, None, None, None, None, None, 3, 3, 6, 0, 6, 6, 6],
    )



TURN_1_NAMES = [
    "auction_count",
    "original_auction_date",
    "reopening_auction_dates",
    "reported_security_terms",
    "original_issue_date",
    "maturity_date",
    "coupon_rate_pct",
]
TURN_2_NAMES = [
    "mspd_issue_row_count",
    "mspd_cumulative_issued_usd_billions",
    "mspd_reported_outstanding_usd_billions",
]
TURN_3_NAMES = [
    "auction_offering_total_usd_billions",
    "auction_accepted_total_usd_billions",
    "accepted_minus_mspd_issued_usd_billions",
]


def _turn_subset(names):
    """Score only `names`, filling the other outputs with their own answers."""
    def run(outputs):
        merged = dict(zip((variable.name for variable in variables), ground_truth()))
        merged.update({name: outputs.get(name) for name in names})
        return validate(merged)

    return run


# The turn contract: `validators` maps the name a turn's `validator`
# field cites to the callable that judges it.
validators = {
    "validate_auction_lineage": turn_validator(_turn_subset(TURN_1_NAMES)),
    "validate_debt_statement": turn_validator(_turn_subset(TURN_2_NAMES)),
    "validate_reconciliation": turn_validator(_turn_subset(TURN_3_NAMES)),
}
