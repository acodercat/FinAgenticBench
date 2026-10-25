"""A field that cannot be averaged, and a denial reason that names it.

The mortgage disclosure file reports debt-to-income three different ways in one
column. Between 36 and 49 percent it reports the exact integer; outside that
band it reports a range, ``<20%`` through ``>60%``; and where the ratio is not
collected it reports ``NA`` or ``Exempt``. Twenty-one values, of which nineteen
sit on a scale and two do not. 4,320,765 of the 12,043,028 records carry one of
the two that do not, so a third of the file cannot be placed on the axis at
all, and no mean of this column is defined: ``>60%`` has no number behind it
and ``30%-<36%`` is not 33.

``Exempt`` is not a missing value. It marks a filer using the partial reporting
exemption, which is open to smaller insured depositories and credit unions, and
the institution file shows it: of the 12,043,028 records, the 2,999,048 filed
by a lender the file maps to a deposit-insurance certificate carry ``Exempt``
on 5.2955% against 1.3361% for the rest. The code is about who filed, not
about the borrower.

The denial reason points at the same field. 668,880 records are denied with
debt-to-income named as the first reason, and the reported ratio agrees with
that most of the time: 545,342 of them sit at 50% or above, and the class at
the median of the denied records is ``>60%`` against ``41`` across every
orderable record in the file. It does not always agree. 25,287 of the records
denied for debt-to-income report a ratio below 36%, and 21,101 report no ratio
at all. Neither is evidence of an error: the reason codes an underwriting
judgement against a programme's own limit, while the field records a ratio
computed under the reporting rules, and one can fail where the other looks fine.

Three conventions the data forces:

1. The ordering runs ``<20%``, ``20%-<30%``, ``30%-<36%``, then the integers 36
   through 49, then ``50%-60%`` and ``>60%``. Sorting the column as text does
   not reproduce it: the digits sort ahead of ``<`` and ``>``, so the text
   order opens at ``20%-<30%`` and leaves ``<20%`` sitting second from the top,
   between ``50%-60%`` and ``>60%``. The whole-file median class reads 42 that
   way against the 41 the scale gives.
2. The median is the first class whose cumulative count reaches half the
   orderable records. Neither median lands on a class edge: across the file
   the running count is 3,756,195 before class 41 and 4,027,713 through it
   against a half of 3,861,131.5, and the denied records clear their own half
   inside ``>60%`` by a comparable margin. No tie rule is needed.
The denial reason is named by its code in the query. The table description
points at the public register schema rather than carrying the code list, and
the other cases drawing on this file name their codes the same way, so the
alternative would be a test of a codebook rather than of the column.

3. A lender is matched to a certificate through the institution file, which
   maps 2,126 identifiers to one and never to two, so no uniqueness rule is
   stated. 1,711 of them appear as filers here; the lenders with no
   certificate are the independent mortgage companies that file the majority
   of this data.

The exempt records are excluded from neither count. They are records of real
applications whose ratio the filer was not required to report, and dropping
them would shrink the denominator of every share in this case.
"""

from functools import lru_cache

from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


ACTIVITY_YEAR = "2024"
DTI_DENIAL_CODE = "1"
LOW_BINS = ("<20%", "20%-<30%", "30%-<36%")
HIGH_BINS = ("50%-60%", ">60%")
EXACT = tuple(str(value) for value in range(36, 50))
ORDERED = LOW_BINS + EXACT + HIGH_BINS
UNORDERABLE = ("NA", "Exempt")
EXEMPT = "Exempt"

OUTPUTS = (
    ("record_count", 0, "the count of records in the file as an integer"),
    ("distinct_dti_values", 0, "the count of distinct values the ratio column takes as an integer"),
    ("ordered_class_count", 0, "how many of those sit on the ratio scale, as an integer"),
    ("unorderable_record_count", 0, "the count of records carrying one of the rest, as an integer"),
    ("bank_linked_lei_count", 0, "the count of lender identifiers matched to a certificate as an integer"),
    ("bank_linked_record_count", 0, "the count of records those lenders filed as an integer"),
    ("bank_exempt_pct", 4, "the exempt percentage among those records"),
    ("nonbank_exempt_pct", 4, "the exempt percentage among the rest"),
    ("dti_denied_count", 0, "the count denied with the ratio named first as an integer"),
    ("dti_denied_above_50_count", 0, "how many of those report a ratio of 50 percent or more, as an integer"),
    ("dti_denied_below_36_count", 0, "how many report a range below 36 percent, as an integer"),
    ("dti_denied_unreported_count", 0, "how many report no ratio on the scale, as an integer"),
    ("exact_integer_record_count", 0, "the count of records reporting an exact integer ratio as an integer"),
    ("dti_denied_exact_integer_count", 0, "how many of the denied records do, as an integer"),
    ("all_record_median_class", None, "the median class across every orderable record, as a source-verbatim string"),
    ("dti_denied_median_class", None, "the median class across the denied records, as a source-verbatim string"),
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
def _applications():
    frame = load_expansion_table("hmda")
    if not frame.activity_year.astype(str).eq(ACTIVITY_YEAR).all():
        raise ValueError("the file spans more than the stated activity year")
    if frame.debt_to_income_ratio.isna().any():
        raise ValueError("a record carries no debt-to-income value at all")
    return frame


@lru_cache(maxsize=1)
def _certificate_lenders():
    """Lender identifiers the institution file matches to a certificate."""
    institutions = load_expansion_table("ffiec_nic_institutions")
    matched = institutions.loc[
        institutions.lei.notna() & institutions.fdic_certificate.notna(), "lei"
    ]
    if matched.empty:
        raise ValueError("the institution file matches no lender to a certificate")
    return frozenset(matched.astype(str))


def _median_class(values):
    """The first class whose cumulative count reaches half the orderable records."""
    counts = {name: 0 for name in ORDERED}
    for name, count in values.value_counts().items():
        if name in counts:
            counts[name] = int(count)
    total = sum(counts.values())
    if not total:
        raise ValueError("no record in this group sits on the ratio scale")
    running = 0
    for name in ORDERED:
        running += counts[name]
        if running * 2 >= total:
            return name
    raise ValueError("the cumulative count never reached half")


@lru_cache(maxsize=1)
def analysis():
    applications = _applications()
    ratio = applications.debt_to_income_ratio
    lenders = _certificate_lenders()

    on_scale = ratio.isin(ORDERED)
    if not ratio.isin(ORDERED + UNORDERABLE).all():
        raise ValueError("the ratio column carries a value outside the known encodings")

    bank = applications.lei.astype(str).isin(lenders)
    if not bank.any() or bank.all():
        raise ValueError("the certificate match does not split the filers")
    exempt = ratio.eq(EXEMPT)

    denied = applications.loc[applications.denial_reason_1.eq(DTI_DENIAL_CODE)]
    if denied.empty:
        raise ValueError("no record names the ratio as its first denial reason")
    denied_ratio = denied.debt_to_income_ratio

    return {
        "record_count": len(applications),
        "distinct_dti_values": int(ratio.nunique()),
        "ordered_class_count": len(frozenset(ratio.unique()) & frozenset(ORDERED)),
        "unorderable_record_count": int((~on_scale).sum()),
        "bank_linked_lei_count": len(lenders & frozenset(applications.lei.astype(str))),
        "bank_linked_record_count": int(bank.sum()),
        "bank_exempt_pct": float(exempt[bank].mean() * 100),
        "nonbank_exempt_pct": float(exempt[~bank].mean() * 100),
        "dti_denied_count": len(denied),
        "dti_denied_above_50_count": int(denied_ratio.isin(HIGH_BINS).sum()),
        "dti_denied_below_36_count": int(denied_ratio.isin(LOW_BINS).sum()),
        "dti_denied_unreported_count": int(denied_ratio.isin(UNORDERABLE).sum()),
        "exact_integer_record_count": int(ratio.isin(EXACT).sum()),
        "dti_denied_exact_integer_count": int(denied_ratio.isin(EXACT).sum()),
        "all_record_median_class": _median_class(ratio),
        "dti_denied_median_class": _median_class(denied_ratio),
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
