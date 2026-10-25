"""A loss rate is a denominator question before it is a credit question.

The public 7(a) file records an outcome for every approval, except that the
agency withholds it on 218,901 of the 347,514 rows. What is left is not a
sample of the file: on the 293,909 loans that were actually disbursed, the
withheld share climbs from 37.2223% of fiscal 2020 approvals to 96.3493% of
fiscal 2025, and the loans whose outcome is withheld are the larger ones, a
median approval 1.3933 times the disclosed median. A rate computed on the
disclosed rows is close to a statement about two early vintages.

The denominator moves the answer by a factor of five. Charged-off loans are
6893. Against every row in the file that is 1.9835%; against the loans that
were disbursed, 2.3453%; against only the loans whose outcome is disclosed,
9.1897%. None of the three is wrong, and the query asks for all three because
naming the denominator is the whole of the exercise.

The disbursement date, not the status code, is what defines the population at
risk, and the two agree without being the same test. Every one of the 44,921
cancelled approvals and all 8,681 that are committed but not yet drawn carry no
first disbursement date, so no money was ever at risk on them; every withheld
and every charged-off row carries one. Three rows recorded as paid in full
carry no disbursement date, which the status and the dated events cannot both
be right about.

What can be said is an interval rather than a number. Charge-offs are at least
2.3453% of the disbursed file, the figure already computed, and at most
76.8245% if every withheld outcome were a charge-off. That is not a useful
bound, which is the finding: this file does not support a portfolio loss rate,
only a disclosed-subset rate whose selection is documented alongside it.

Three features of the extract the query has to pin:

1. Statuses are stored as source-native codes and one of them carries interior
   spaces, so a screen written against a tidied spelling silently matches
   nothing. The codes are read as the file stores them.
2. 513 rows repeat a row already in the file across every column. The extract
   has no unique loan identifier and its own documentation says exact-looking
   rows are preserved, so they are counted, not collapsed.
3. Six rows disagree with their own dated events: five are charged off with no
   charge-off date, and one carries a paid-in-full date under another status.
   They are reported rather than reassigned, since either field could be the
   wrong one and the file says nothing about which. The two dispositions the
   query names are exhaustive here: no paid-in-full row lacks its date and no
   other status carries a charge-off date.

The withholding tracks the lender as well. 31,585 of the disbursed loans name
no deposit-insurance certificate, their lender being outside the insured
banking system, and the withheld share there is 78.7304% against 73.9673%
where a certificate is named. Of the 1,823 certificates the loans do name, one
has no row in the bank file at 2025-12-31 and covers 5 loans: the certificate
identifies an insured institution as of the loan, not one still reporting at a
later snapshot.

Two readings the query leaves open because the data closes them. The two median
approval amounts come out at 209,000 and 150,000 whether they are struck over
the disbursed loans or over every row, so the ratio is 1.3933 either way; and
the earliest and latest approval fiscal years are 2020 and 2025 in both
populations. Neither is pinned further, because the wording would grow without
any reported number moving.
"""

from functools import lru_cache

from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


CANCELLED, COMMITTED, CHARGED_OFF, WITHHELD, PAID_IN_FULL = (
    "CANCLD", "COMMIT", "CHGOFF", "EXEMPT", "P I F",
)
FDIC_SNAPSHOT = "2025-12-31"

OUTPUTS = (
    ("loan_row_count", 0, "the count of rows in the loan file as an integer"),
    ("cancelled_count", 0, "the count of cancelled approvals as an integer"),
    ("committed_not_disbursed_count", 0, "the count recorded as committed as an integer"),
    ("charged_off_count", 0, "the count recorded as charged off as an integer"),
    ("disbursed_loan_count", 0, "the count of disbursed loans as an integer"),
    ("chargeoff_pct_all_rows", 4, "charge-offs as a percentage of every row in the file"),
    ("chargeoff_pct_disbursed", 4, "charge-offs as a percentage of the disbursed loans"),
    ("chargeoff_pct_disclosed_only", 4, "charge-offs as a percentage of the disbursed loans whose outcome is disclosed"),
    ("status_withheld_count", 0, "the count of disbursed loans whose outcome is withheld, as an integer"),
    ("withheld_pct_earliest_fy", 4, "the withheld percentage among the earliest approval fiscal year's disbursed loans"),
    ("withheld_pct_latest_fy", 4, "the same percentage for the latest approval fiscal year"),
    ("withheld_to_disclosed_median_ratio", 4, "the withheld group's median approval amount as a multiple of the disclosed group's"),
    ("chargeoff_upper_bound_pct", 4, "the highest charge-off percentage of the disbursed loans the file still permits"),
    ("status_date_inconsistency_count", 0, "the count of rows whose status contradicts their dated events, as an integer"),
    ("paid_in_full_without_disbursement_count", 0, "the count paid in full with no disbursement date, as an integer"),
    ("exact_duplicate_row_count", 0, "the count of rows repeating a row already present, as an integer"),
    ("lender_without_certificate_count", 0, "the count of disbursed loans carrying no lender certificate, as an integer"),
    ("withheld_pct_uninsured_lender", 4, "the withheld percentage among those loans"),
    ("withheld_pct_insured_lender", 4, "the withheld percentage among the disbursed loans that do carry one"),
    ("unmatched_certificate_loan_count", 0, "the count of disbursed loans whose lender certificate is absent from the bank file, as an integer"),
)
variables = [
    Variable(name, None, f"Store {text}" + (f", rounded to {p} decimals from unrounded source rows." if p is not None else "."))
    for name, p, text in OUTPUTS
]
PRECISION = {name: p for name, p, _ in OUTPUTS}
TURNS = (
    tuple(n for n, _, _ in OUTPUTS[:4]),
    tuple(n for n, _, _ in OUTPUTS[4:8]),
    tuple(n for n, _, _ in OUTPUTS[8:12]),
    tuple(n for n, _, _ in OUTPUTS[12:16]),
    tuple(n for n, _, _ in OUTPUTS[16:]),
)


@lru_cache(maxsize=1)
def _loans():
    frame = load_expansion_table("sba_7a")
    if frame.LoanStatus.isna().any():
        raise ValueError("a loan row carries no status at all")
    return frame


@lru_cache(maxsize=1)
def _disbursed():
    frame = _loans()
    disbursed = frame.loc[frame.FirstDisbursementDate.notna()]
    if disbursed.empty:
        raise ValueError("no loan records a first disbursement")
    return disbursed


@lru_cache(maxsize=1)
def _insured_certificates():
    """Certificates the bank file carries at the stated snapshot date."""
    banks = load_expansion_table("fdic_bankfind")
    snapshot = banks.loc[banks.REPDTE.astype(str).eq(FDIC_SNAPSHOT)]
    if snapshot.empty:
        raise ValueError(f"the bank file carries no rows at {FDIC_SNAPSHOT}")
    return frozenset(snapshot.CERT.astype(str))


@lru_cache(maxsize=1)
def analysis():
    loans = _loans()
    disbursed = _disbursed()
    status = disbursed.LoanStatus

    charged_off = int(status.eq(CHARGED_OFF).sum())
    withheld = int(status.eq(WITHHELD).sum())
    disclosed = disbursed.loc[status.ne(WITHHELD)]
    if not charged_off or not withheld or disclosed.empty:
        raise ValueError("the status field does not separate the three populations")

    by_year = disbursed.groupby("ApprovalFY").LoanStatus.apply(
        lambda s: float(s.eq(WITHHELD).mean() * 100)
    )
    if len(by_year) < 2:
        raise ValueError("the file spans a single approval fiscal year")

    withheld_median = float(disbursed.loc[status.eq(WITHHELD), "GrossApproval"].median())
    disclosed_median = float(disclosed.GrossApproval.median())
    if not disclosed_median:
        raise ValueError("the disclosed group has no median approval amount to divide by")

    # A status and its dated events can disagree in two directions; both are
    # reported rather than reassigned to whichever field looks more plausible.
    contradicted = (
        loans.LoanStatus.eq(CHARGED_OFF) & loans.ChargeOffDate.isna()
    ) | (
        loans.LoanStatus.ne(PAID_IN_FULL) & loans.PaidInFullDate.notna()
    )

    no_certificate = disbursed.loc[disbursed.fdic_certificate.isna()]
    with_certificate = disbursed.loc[disbursed.fdic_certificate.notna()]
    if no_certificate.empty or with_certificate.empty:
        raise ValueError("every disbursed loan reports the same kind of lender")

    return {
        "loan_row_count": len(loans),
        "cancelled_count": int(loans.LoanStatus.eq(CANCELLED).sum()),
        "committed_not_disbursed_count": int(loans.LoanStatus.eq(COMMITTED).sum()),
        "charged_off_count": charged_off,
        "disbursed_loan_count": len(disbursed),
        "chargeoff_pct_all_rows": charged_off / len(loans) * 100,
        "chargeoff_pct_disbursed": charged_off / len(disbursed) * 100,
        "chargeoff_pct_disclosed_only": charged_off / len(disclosed) * 100,
        "status_withheld_count": withheld,
        "withheld_pct_earliest_fy": by_year.loc[by_year.index.min()],
        "withheld_pct_latest_fy": by_year.loc[by_year.index.max()],
        "withheld_to_disclosed_median_ratio": withheld_median / disclosed_median,
        "chargeoff_upper_bound_pct": (charged_off + withheld) / len(disbursed) * 100,
        "status_date_inconsistency_count": int(contradicted.sum()),
        "paid_in_full_without_disbursement_count": int(
            (loans.LoanStatus.eq(PAID_IN_FULL) & loans.FirstDisbursementDate.isna()).sum()
        ),
        "exact_duplicate_row_count": int(loans.duplicated().sum()),
        "lender_without_certificate_count": len(no_certificate),
        "withheld_pct_uninsured_lender": float(no_certificate.LoanStatus.eq(WITHHELD).mean() * 100),
        "withheld_pct_insured_lender": float(with_certificate.LoanStatus.eq(WITHHELD).mean() * 100),
        "unmatched_certificate_loan_count": int(
            (~with_certificate.fdic_certificate.astype(str).isin(_insured_certificates())).sum()
        ),
    }


@lru_cache(maxsize=1)
def ground_truth():
    answers = analysis()
    return tuple(answers[variable.name] for variable in variables)


def _check(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip(by_name, ground_truth()))
    return validate_ordered_outputs(
        outputs,
        [by_name[n] for n in names],
        [truth[n] for n in names],
        [PRECISION[n] for n in names],
    )


def validate_turn_1(outputs):
    return _check(outputs, TURNS[0])


def validate_turn_2(outputs):
    return _check(outputs, TURNS[1])


def validate_turn_3(outputs):
    return _check(outputs, TURNS[2])


def validate_turn_4(outputs):
    return _check(outputs, TURNS[3])


def validate(outputs):
    return _check(outputs, tuple(v.name for v in variables))


def validate_turn_5(outputs):
    return _check(outputs, TURNS[4])


validators = {
    f"validate_turn_{i}": turn_validator(globals()[f"validate_turn_{i}"])
    for i in range(1, 6)
}
