"""Screen 7(a) lenders on the exposure the guarantee leaves with them.

A 7(a) approval is not one credit.  The Small Business Administration guarantees
a statutory share of it -- three quarters on the standard product, one half on
the express product -- and the government, not the lender, carries that share.
What the lender is itself at risk on is the remainder.

The trap is the second turn, which screens lenders whose own risk from a fiscal
year of approvals exceeds their entire commercial and industrial loan book.  On
the pinned reading 6 lenders clear that bar; measuring exposure by the gross
approved amount instead reports 45.  Both are counts, compared for exact
equality, so the rounding-boundary tolerance excludes the naive reading by 39
lenders.  The margin either side of the bar is wide: the sixth lender sits at
1.102126 and the seventh at 0.866684, so the count does not turn on a hair.

The error reaches every turn.  Six of the sixteen outputs move under the gross
reading: the screened count 6 to 45, the largest single exposure 0.757135 to
2.852669, the intensity leader's ratio 3.546408 to 14.562687, the largest
lender's own intensity 0.164963 to 0.621534, the top-ten concentration 38.812902
to 38.068233 percent, and the exposure carried outside the banks 1.065450 to
4.304624.  The two certificates do not move, so a solver that falls into the
trap still names the right two lenders and reports every magnitude wrong.

The reordering is real but sits below the very top, which is what makes the trap
subtle rather than obvious.  The first five lenders rank the same either way; the
top ten share only seven members, with TD Bank, Bank of America and Wells Fargo
entering on retained risk in place of Celtic Bank, Byline Bank and GBank.  TD
Bank is thirteenth by gross approvals and seventh by what it keeps, because the
one-half guarantee on the express product leaves a lender twice as much per
approved dollar as the three-quarter guarantee on the standard one.

Rejected conventions.  The period is the federal fiscal year, 2024-10-01 through
2025-09-30 inclusive by approval date: the file runs 2019-10-01 to 2025-09-30, so
the last calendar year in it is truncated and only fiscal years are complete.
Lender identity is the certificate the loan file records, not the lender name --
all 1,130 certificates in the fiscal year match a bank reporting on 2025-09-30,
and 9,562 loans carry no certificate at all because their lenders are not
FDIC-insured banks.  Call-report amounts are thousands of dollars.  Leader
margins are comfortable: 13.17 percent on the largest exposure and 1.044771 on
the intensity leader.  The module keeps a positive-commercial-book guard on the
match, but it currently removes nothing: all 1,130 report one, so the query does
not claim the filter does any work.

Every one of the sixteen outputs was recomputed by a second implementation built
from plain dictionaries with no pandas aggregation, and all sixteen agreed.  Each
of the eight rounded figures lies at most 0.2687 of a unit from the value it is
rounded to, against the half-unit that would put it on a rounding boundary, and
all three totals are invariant to summation order to the cent.
"""

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


FISCAL_YEAR_START = pd.Timestamp("2024-10-01")
FISCAL_YEAR_END = pd.Timestamp("2025-09-30")
CALL_REPORT_DATE = pd.Timestamp("2025-09-30")
THOUSANDS = 1_000.0
LEADER_COUNT = 10


variables = [
    Variable("fiscal_year_loan_count", None,
             "Store the number of loans approved in the fiscal year as an integer."),
    Variable("gross_approval_usd_billions", None,
             "Store their total gross approved amount in USD billions, rounded to 6 decimals."),
    Variable("guaranteed_approval_usd_billions", None,
             "Store the total amount the agency guaranteed on them in USD billions, rounded to 6 decimals."),
    Variable("unidentified_lender_loan_count", None,
             "Store how many of those loans record no lender certificate as an integer."),

    Variable("matched_bank_lender_count", None,
             "Store how many certificates match a bank reporting at the call-report date as an integer."),
    Variable("own_exposure_exceeds_commercial_book_lender_count", None,
             "Store how many of those lenders are at risk on more than their whole commercial book as an integer."),
    Variable("largest_own_exposure_certificate", None,
             "Store that lender's FDIC certificate as an unpadded identifier string."),
    Variable("largest_own_exposure_usd_billions", None,
             "Store the largest amount a single lender is at risk on in USD billions, rounded to 6 decimals."),

    Variable("highest_exposure_intensity_certificate", None,
             "Store the most intense lender's FDIC certificate as an unpadded identifier string."),
    Variable("highest_exposure_intensity_ratio", None,
             "Store its amount at risk divided by its commercial book, rounded to 6 decimals."),
    Variable("highest_exposure_intensity_commercial_book_usd_billions", None,
             "Store that lender's commercial book in USD billions, rounded to 6 decimals."),
    Variable("largest_own_exposure_intensity_ratio", None,
             "Store the same ratio for the lender carrying the largest amount, rounded to 6 decimals."),

    Variable("three_quarter_guarantee_loan_count", None,
             "Store how many fiscal-year loans were guaranteed for three quarters of the approved amount as an integer."),
    Variable("one_half_guarantee_loan_count", None,
             "Store how many were guaranteed for one half of it as an integer."),
    Variable("top_ten_lender_own_exposure_share_pct", None,
             "Store the ten largest matched lenders' share of all matched exposure at risk in percent, rounded to 6 decimals."),
    Variable("unidentified_lender_own_exposure_usd_billions", None,
             "Store the amount at risk on loans recording no certificate in USD billions, rounded to 6 decimals."),
]

# The two certificates are text: their comparison is exact equality, not a
# rounding-boundary tolerance, so they carry no decimal count.
DECIMALS = [0, 6, 6, 0, 0, 0, None, 6, None, 6, 6, 6, 0, 0, 6, 6]


def _fiscal_year_loans():
    """7(a) approvals dated inside the fiscal year, with the lender's own risk."""
    loans = load_expansion_table("sba_7a")
    approved_on = pd.to_datetime(loans.ApprovalDate, errors="coerce")
    frame = pd.DataFrame({
        "gross": pd.to_numeric(loans.GrossApproval, errors="coerce"),
        "guaranteed": pd.to_numeric(loans.SBAGuaranteedApproval, errors="coerce"),
        "certificate": pd.to_numeric(loans.BankFDICNumber, errors="coerce"),
    }).loc[approved_on.between(FISCAL_YEAR_START, FISCAL_YEAR_END)]
    del loans
    frame["own"] = frame.gross - frame.guaranteed
    return frame


def _commercial_books():
    """Each bank's commercial and industrial loan stock at the call-report date, in dollars."""
    banks = load_expansion_table("fdic_bankfind")
    reported_on = pd.to_datetime(banks.REPDTE, errors="coerce")
    frame = pd.DataFrame({
        "certificate": pd.to_numeric(banks.CERT, errors="coerce"),
        "commercial": pd.to_numeric(banks.LNCI, errors="coerce") * THOUSANDS,
    }).loc[reported_on.eq(CALL_REPORT_DATE)]
    del banks
    frame = frame.loc[frame.certificate.notna()]
    return frame.set_index(frame.certificate.astype(int)).commercial


@lru_cache(maxsize=1)
def ground_truth():
    loans = _fiscal_year_loans()
    unidentified = loans.certificate.isna()

    identified = loans.loc[~unidentified]
    by_lender = identified.groupby(identified.certificate.astype(int))
    lenders = pd.DataFrame({"gross": by_lender.gross.sum(), "own": by_lender.own.sum()})
    lenders["commercial"] = _commercial_books()
    matched = lenders.commercial.notna() & lenders.commercial.gt(0)
    screened = lenders.loc[matched].copy()
    screened["intensity"] = screened.own / screened.commercial

    # The query breaks every tie by the lower certificate, so the selections are
    # ordered explicitly rather than left to an unstable sort.
    screened = screened.assign(cert_key=screened.index)
    by_own = screened.sort_values(["own", "cert_key"], ascending=[False, True])
    by_intensity = screened.sort_values(["intensity", "cert_key"], ascending=[False, True])
    largest = by_own.index[0]
    most_intense = by_intensity.index[0]
    ranked = by_own.own

    guaranteed_four = loans.guaranteed.mul(4)
    guaranteed_two = loans.guaranteed.mul(2)

    return (
        len(loans),
        round(float(loans.gross.sum()) / 1e9, 6),
        round(float(loans.guaranteed.sum()) / 1e9, 6),
        int(unidentified.sum()),

        int(matched.sum()),
        int(screened.intensity.gt(1.0).sum()),
        str(largest),
        round(float(screened.own[largest]) / 1e9, 6),

        str(most_intense),
        round(float(screened.intensity[most_intense]), 6),
        round(float(screened.commercial[most_intense]) / 1e9, 6),
        round(float(screened.intensity[largest]), 6),

        int(guaranteed_four.eq(loans.gross.mul(3)).sum()),
        int(guaranteed_two.eq(loans.gross).sum()),
        round(100.0 * float(ranked.head(LEADER_COUNT).sum()) / float(ranked.sum()), 6),
        round(float(loans.own[unidentified].sum()) / 1e9, 6),
    )


TURN_1_NAMES = [
    "fiscal_year_loan_count", "gross_approval_usd_billions",
    "guaranteed_approval_usd_billions", "unidentified_lender_loan_count",
]
TURN_2_NAMES = [
    "matched_bank_lender_count", "own_exposure_exceeds_commercial_book_lender_count",
    "largest_own_exposure_certificate", "largest_own_exposure_usd_billions",
]
TURN_3_NAMES = [
    "highest_exposure_intensity_certificate", "highest_exposure_intensity_ratio",
    "highest_exposure_intensity_commercial_book_usd_billions",
    "largest_own_exposure_intensity_ratio",
]
TURN_4_NAMES = [
    "three_quarter_guarantee_loan_count", "one_half_guarantee_loan_count",
    "top_ten_lender_own_exposure_share_pct", "unidentified_lender_own_exposure_usd_billions",
]

CERTIFICATE_OUTPUTS = ("largest_own_exposure_certificate", "highest_exposure_intensity_certificate")


def _same_certificate(value, expected):
    """A certificate is a number written as text; leading zeros do not change it."""
    if value is None:
        return False
    try:
        return int(str(value).strip()) == int(expected)
    except (TypeError, ValueError):
        return False


def _validate_subset(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip(by_name, ground_truth()))
    decimals = dict(zip(by_name, DECIMALS))
    candidate = dict(outputs)
    for name in CERTIFICATE_OUTPUTS:
        if _same_certificate(candidate.get(name), truth[name]):
            candidate[name] = truth[name]
    return validate_ordered_outputs(
        candidate,
        [by_name[name] for name in names],
        [truth[name] for name in names],
        [decimals[name] for name in names],
    )


def validate_turn_1(outputs): return _validate_subset(outputs, TURN_1_NAMES)
def validate_turn_2(outputs): return _validate_subset(outputs, TURN_2_NAMES)
def validate_turn_3(outputs): return _validate_subset(outputs, TURN_3_NAMES)
def validate_turn_4(outputs): return _validate_subset(outputs, TURN_4_NAMES)
def validate(outputs): return _validate_subset(outputs, [v.name for v in variables])


# The turn contract: `validators` maps the name a turn's `validator`
# field cites to the callable that judges it.
validators = {
    "validate_programme_year": turn_validator(validate_turn_1),
    "validate_exposure_screen": turn_validator(validate_turn_2),
    "validate_size_versus_intensity": turn_validator(validate_turn_3),
    "validate_guarantee_structure": turn_validator(validate_turn_4),
}
