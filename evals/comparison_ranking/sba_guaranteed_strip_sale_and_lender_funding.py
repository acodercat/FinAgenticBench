"""Who sells the guaranteed strip, and what that says about how they fund.

A 7(a) lender may sell the guaranteed portion of a loan into the secondary
market. 108,175 of the 293,909 disbursed loans here were sold, and which ones
they are is not random: the sale needs a fixed amortisation schedule and a
floating coupon the buyer can price. Five of the 75,649 revolving lines were
sold against 49.5602% of the rest, and 98.2796% of the sold loans carry a
variable rate against 74.5302% of those retained.

The seller's own funding model shows through more strongly than any loan
feature. A lender outside the insured banking system has no deposits to hold
the loan against, and sells 70.9007% of what it disburses against 32.7004% for
a lender the bank file matches to a deposit-insurance certificate. Inside that
group the split is sharper still: the 111 lenders above ten billion USD of
total assets sell 16.0144% while the 1,711 below sell 55.9638%.

The market is also concentrated. 2,116 lenders appear here and 1,465 of them
never sell a loan; of the 108,175 that are sold, the ten largest sellers
account for 45.3108% and BayFirst National Bank alone for 7.7957%.

Four things the data forces:

1. ``SoldSecMrktInd`` is a positive-only indicator. It reads ``Y`` on a sold
   loan and is absent on every other row; there is no ``N``. Absence is read as
   not sold, which is what the field means, but it is an absence and not a
   reported denial.
2. A lender is counted by its reported name. The file carries 2,116 distinct
   lender names against 1,823 distinct deposit-insurance certificates, so the
   two keys do not give the same lender count and the query names which one.
3. The size split uses total assets above ten billion USD at the bank file's
   2025-12-31 report date. No lender sits exactly on that figure, so the
   boundary is strict or inclusive to the same effect, and the case test says so.
4. Charge-off shares are struck against every loan in each cohort, sold or
   retained, not against the loans whose outcome is disclosed.

Three readings the query has to close because each one moves a reported
number, one of them by reversing the finding:

- The size split is a percentage of loans, not of lenders. Among lenders
  instead, the share that sell anything runs 57.6577% above ten billion USD
  against 27.8200% below, and the mean of the per-lender sold rates runs
  26.0825% against 15.4978% — both the opposite way round from the 16.0144%
  and 55.9638% the loan-weighted reading gives. A large bank sells a small
  part of a large book.
- The certificate comparison spans every loan whose lender carries one, while
  the size split can only span the 262,319 of those 262,324 whose certificate
  the bank file carries at the snapshot date; on the matched subset the
  certificate share reads 32.7010 rather than 32.7004.
- Six loans report no rate type. They sit in the denominator, without which
  the variable-rate shares read 98.2815 and 74.5318 rather than 98.2796 and
  74.5302.

A reported lender name is not a certificate: 76 names carry more than one
certificate, so counting lenders by name merges nothing but also splits
nothing the file keeps separate, and the count differs from the 1,823
certificates for that reason as well as for the lenders holding none.

The last pair of figures is where the case stops short of a conclusion. Sold
loans are charged off on 1.6464% of the cohort against 2.7523% for retained
ones, which reads as the sold book performing better. It is not a clean
comparison: the agency withholds the outcome on 78.8204% of the sold loans
against 71.9507% of the retained, so the cohort that looks safer is also the
one whose outcomes are less often published. Which way that cuts is not
settled by anything in this file.

A charge-off is a loan whose status reads charged off, as the query states.
Five such loans, two sold and three retained, carry no charge-off date and a
zero gross charge-off amount; counting a dated or positive charge-off instead
reads 1.6446% and 2.7507%.
"""

from functools import lru_cache

from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


SOLD = "Y"
CHARGED_OFF = "CHGOFF"
REVOLVER = "Y"
VARIABLE = "V"
BANK_SNAPSHOT = "2025-12-31"
LARGE_BANK_ASSETS_THOUSAND_USD = 10_000_000
TOP_SELLERS = 10

OUTPUTS = (
    ("sold_loan_count", 0, "the count of disbursed loans sold into the secondary market as an integer"),
    ("sold_share_pct", 4, "those loans as a percentage of the disbursed loans"),
    ("lender_count", 0, "the count of distinct lenders in the disbursed population as an integer"),
    ("never_selling_lender_count", 0, "how many of them sell none of their loans, as an integer"),
    ("revolver_sold_count", 0, "the count of revolving loans sold as an integer"),
    ("nonrevolver_sold_share_pct", 4, "the sold percentage among the loans that are not revolving"),
    ("sold_variable_rate_pct", 4, "the variable-rate percentage among the sold loans"),
    ("retained_variable_rate_pct", 4, "the variable-rate percentage among the rest"),
    ("noncertificate_sold_share_pct", 4, "the sold percentage among loans from lenders carrying no certificate"),
    ("certificate_sold_share_pct", 4, "the sold percentage among loans from lenders that carry one"),
    ("large_bank_sold_share_pct", 4, "the sold percentage among loans from the largest of those banks"),
    ("small_bank_sold_share_pct", 4, "the sold percentage among loans from the rest of them"),
    ("top_ten_seller_share_pct", 4, "the ten largest sellers' share of the sold loans, in percent"),
    ("largest_seller_share_pct", 4, "the single largest seller's share of them, in percent"),
    ("sold_chargeoff_pct", 4, "charge-offs as a percentage of the sold loans"),
    ("retained_chargeoff_pct", 4, "charge-offs as a percentage of the rest"),
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
    tuple(n for n, _, _ in OUTPUTS[12:]),
)


@lru_cache(maxsize=1)
def _disbursed():
    frame = load_expansion_table("sba_7a")
    disbursed = frame.loc[frame.FirstDisbursementDate.notna()].copy()
    if disbursed.empty:
        raise ValueError("no loan records a first disbursement")
    if disbursed.BankName.isna().any():
        raise ValueError("a disbursed loan names no lender")
    # The indicator is positive-only: Y on a sold loan, absent everywhere else.
    if not disbursed.SoldSecMrktInd.dropna().eq(SOLD).all():
        raise ValueError("the secondary-market indicator carries a value other than Y")
    disbursed["sold"] = disbursed.SoldSecMrktInd.astype(str).eq(SOLD)
    return disbursed


@lru_cache(maxsize=1)
def _bank_assets():
    """Total assets by certificate at the stated bank-file report date."""
    banks = load_expansion_table("fdic_bankfind")
    snapshot = banks.loc[banks.REPDTE.astype(str).eq(BANK_SNAPSHOT)].drop_duplicates("CERT")
    if snapshot.empty:
        raise ValueError(f"the bank file carries no rows at {BANK_SNAPSHOT}")
    return snapshot.set_index(snapshot.CERT.astype(str)).ASSET


def _pct(mask, within):
    total = int(within.sum())
    if not total:
        raise ValueError("an empty group has no percentage to report")
    return float((mask & within).sum() / total * 100)


@lru_cache(maxsize=1)
def analysis():
    loans = _disbursed()
    sold = loans.sold
    if not sold.any() or sold.all():
        raise ValueError("the indicator does not split the disbursed loans")

    by_lender = loans.groupby("BankName").sold.sum()
    revolving = loans.RevolverStatus.astype(str).eq(REVOLVER)
    variable = loans.FixedorVariableInterestInd.astype(str).eq(VARIABLE)
    has_certificate = loans.fdic_certificate.notna()

    matched = loans.loc[has_certificate].copy()
    matched["assets"] = matched.fdic_certificate.astype(str).map(_bank_assets())
    matched = matched.loc[matched.assets.notna()]
    if matched.empty:
        raise ValueError("no certificate matches the bank file")
    large = matched.assets > LARGE_BANK_ASSETS_THOUSAND_USD
    if not large.any() or large.all():
        raise ValueError("the asset threshold does not split the matched banks")

    # groupby orders the names, and a stable sort by count keeps that order
    # inside a tie, which is the alphabetical rule the query states.
    sellers = loans.loc[sold].groupby("BankName").size().sort_values(
        ascending=False, kind="stable"
    )
    if len(sellers) <= TOP_SELLERS:
        raise ValueError("fewer sellers than the reported cut")
    charged_off = loans.LoanStatus.astype(str).eq(CHARGED_OFF)

    return {
        "sold_loan_count": int(sold.sum()),
        "sold_share_pct": float(sold.mean() * 100),
        "lender_count": int(loans.BankName.nunique()),
        "never_selling_lender_count": int((by_lender == 0).sum()),
        "revolver_sold_count": int((sold & revolving).sum()),
        "nonrevolver_sold_share_pct": _pct(sold, ~revolving),
        "sold_variable_rate_pct": _pct(variable, sold),
        "retained_variable_rate_pct": _pct(variable, ~sold),
        "noncertificate_sold_share_pct": _pct(sold, ~has_certificate),
        "certificate_sold_share_pct": _pct(sold, has_certificate),
        "large_bank_sold_share_pct": float(matched.loc[large].sold.mean() * 100),
        "small_bank_sold_share_pct": float(matched.loc[~large].sold.mean() * 100),
        "top_ten_seller_share_pct": float(sellers.head(TOP_SELLERS).sum() / sellers.sum() * 100),
        "largest_seller_share_pct": float(sellers.iloc[0] / sellers.sum() * 100),
        "sold_chargeoff_pct": _pct(charged_off, sold),
        "retained_chargeoff_pct": _pct(charged_off, ~sold),
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


validators = {
    f"validate_turn_{i}": turn_validator(globals()[f"validate_turn_{i}"])
    for i in range(1, 5)
}
