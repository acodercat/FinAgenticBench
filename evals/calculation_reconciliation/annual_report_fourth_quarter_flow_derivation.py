"""The fourth quarter is the one quarter no filing reports directly.

A company files three 10-Qs and then a 10-K. The 10-K covers the year, so the
fourth quarter has no filing of its own and an analyst who wants it has to take
the year less the nine months already reported. The tagged data says the same
thing: of the 5,524 annual filings in this window, 5,170 carry a year-to-date
net income and only 268 carry a quarterly one.

That the derivation is the right one is checkable rather than assumed. On the
256 filings that do carry both, the year less the paired nine months reproduces
the disclosed quarterly figure to the dollar 214 times, including at scale:
Accenture's 7,264,787,000 less 5,580,486,000 is the 1,684,301,000 it tags, and
AbbVie's -22,000,000 comes out of 4,278,000,000 less 4,300,000,000. No match is
degenerate; not one of the 214 has a nine-month figure of zero.

The remaining 42 are the point of the case. The derivation inherits every
restatement, reclassification and discontinued operation booked between the
third-quarter filing and the 10-K, and the two figures then disagree with no
way to say from these fields alone which one an analyst should carry. Five of
the 42 are a tagging fault visible without judgement, the annual figure sitting
in the quarterly slot: North Haven Private Income Fund tags 259,971,000 as its
quarter, which is exactly its year. The other 37 differ for reasons the fields
do not record, Ally Financial by 248,000,000 the widest among them.

Four conventions, each of which moves a reported number:

1. Fiscal years are selected by period end falling in calendar 2024, not by the
   ``fiscal_year`` label. The label puts two different fiscal years under 2024
   for the 52/53-week filers, so Children's Place appears under it for both the
   year ended 2024-02-03 and the year ended 2025-02-01. Selecting on the label
   gives 5,504 filers against 5,524. Seven filers still close twice inside
   calendar 2024 and the later period end is the fiscal 2024 one, while two
   more file the same period end twice; retaining the earlier row instead of
   the later gives 488.831902 rather than 488.833662 USD billions.
2. The paired nine-month filing is the Q3 10-Q whose period end is nearest
   before the annual period end and no more than 120 days before it. Without
   the cap a filer that skipped its third-quarter filing pairs against a much
   older one, which raises the derived total to 491.108491.
3. Amendments are outside the population on both sides. Admitting 10-K/A gives
   490.578436 and admitting 10-Q/A gives 488.946485.
4. Agreement is an absolute difference below one dollar. This one changes
   nothing measured here, the 42 disagreements being the same set under exact
   inequality; it is stated so the comparison survives float arithmetic rather
   than to admit a second reading.

Two selections the query leaves open because the data closes them. One paired
filer files its third-quarter period end twice and both rows carry the same
nine-month figure, so which row is taken does not reach a reported number. And
no filer has two third-quarter period ends inside the 120-day window, so the
nearest-filing rule never has to choose. The query still states the rule, since
a solver cannot know either of these before computing. The two tie-breaks that
do not fire are stated for the same reason and asserted not to fire in the
case-local test: no filer needs an accession to separate its annual rows, and
the widest disagreement leads the second by 27,000,000.

What the derivation rests on is dated and revisable. The submission index
matches 5,522 of these filings on filer and accession together, 85 of the
accessions being filed jointly by several filers so that the accession alone
duplicates rows. 574 of the filings are later superseded by an amendment for
the same period end, and the median gap from period end to public acceptance
is 65 days. Two filings the statements extract labels 10-K the index labels
10-K/A, neither field being checked against the other.

Revenue is not used. Only 1,945 of the annual filings tag a year-to-date
revenue and 91 a quarterly one, too thin a validation set to carry the same
argument.

The annual population is read from the financial statements, which the query
names, because the filing index lists filings that carry no statement row: its
unamended 10-Ks with a 2024 period end come from 5,711 filers against the 5,524
here, and turn five's form comparison only means something when the statements
are the side being checked.
"""

from decimal import Decimal
from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_filings, load_financial_statements
from core.validation import turn_validator, validate_ordered_outputs


WINDOW = ("2024-01-01", "2024-12-31")
MAX_PAIR_GAP_DAYS = 120
AMENDED_FORM = "10-K/A"
AGREEMENT_USD = Decimal(1)
BILLION = Decimal(10) ** 9
MILLION = Decimal(10) ** 6

OUTPUTS = (
    ("annual_filer_count", 0, "the count of filers in the annual population as an integer"),
    ("noncalendar_year_end_filers", 0, "the count of those closing their year other than 31 December, as an integer"),
    ("annual_with_year_net_income", 0, "the count reporting a year-to-date net income as an integer"),
    ("annual_with_quarter_net_income", 0, "the count reporting a quarterly net income as an integer"),
    ("paired_filer_count", 0, "the count of filers paired to a third-quarter filing as an integer"),
    ("derived_filer_count", 0, "the count whose fourth quarter can be derived as an integer"),
    ("year_net_income_bn", 6, "those filers' year-to-date net income summed, in USD billions"),
    ("derived_q4_net_income_bn", 6, "their derived fourth-quarter net income summed, in USD billions"),
    ("positive_year_filers", 0, "the count of those filers whose year is positive as an integer"),
    ("positive_year_negative_quarter_filers", 0, "how many of those have a negative derived quarter, as an integer"),
    ("validation_filer_count", 0, "the count of filers carrying both a derived and a disclosed quarter, as an integer"),
    ("validation_disagreement_count", 0, "how many of those disagree, as an integer"),
    ("annual_figure_in_quarter_slot_count", 0, "how many disagreements report the year in the quarterly field, as an integer"),
    ("largest_disagreement_cik", None, "the widest disagreement's CIK, as a ten-character zero-padded string"),
    ("largest_disagreement_derived_mn", 6, "that filer's derived fourth-quarter net income, in USD millions"),
    ("largest_disagreement_reported_mn", 6, "that filer's disclosed quarterly net income, in USD millions"),
    ("shared_accession_count", 0, "the count of annual filings whose accession is filed under more than one CIK, as an integer"),
    ("superseded_by_amendment_count", 0, "the count later superseded by an amendment for the same period, as an integer"),
    ("median_days_to_public", 1, "the median days from period end to public acceptance"),
    ("form_disagreement_count", 0, "the count the two files label with different forms, as an integer"),
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


def _latest_per_filer(frame, keys):
    """One row per filer: the latest period end, then filed date, then accession."""
    return frame.sort_values(list(keys)).drop_duplicates("cik", keep="last")


@lru_cache(maxsize=1)
def _annual():
    statements = load_financial_statements()
    frame = statements.loc[
        statements.form.eq("10-K") & statements.fiscal_period.eq("FY")
    ].copy()
    frame["period_end"] = frame.report_date.astype(str)
    frame = frame.loc[frame.period_end.between(*WINDOW)]
    if frame.empty:
        raise ValueError("no annual filing has a period end in the window")
    return _latest_per_filer(frame, ("cik", "period_end", "filed_date", "accession"))


@lru_cache(maxsize=1)
def _paired():
    statements = load_financial_statements()
    quarterly = statements.loc[
        statements.form.eq("10-Q") & statements.fiscal_period.eq("Q3")
    ].copy()
    quarterly["period_end"] = quarterly.report_date.astype(str)
    quarterly = quarterly.sort_values(["cik", "period_end", "filed_date", "accession"])
    quarterly = quarterly.drop_duplicates(["cik", "period_end"], keep="last")

    annual = _annual()[["cik", "company_name", "period_end", "NetIncomeLoss_ytd", "NetIncomeLoss_q"]]
    nine_month = quarterly[["cik", "period_end", "NetIncomeLoss_ytd"]].rename(
        columns={"period_end": "q3_end", "NetIncomeLoss_ytd": "nine_month_ytd"}
    )
    paired = annual.merge(nine_month, on="cik")
    paired["gap_days"] = (
        pd.to_datetime(paired.period_end) - pd.to_datetime(paired.q3_end)
    ).dt.days
    paired = paired.loc[paired.gap_days.between(1, MAX_PAIR_GAP_DAYS)]
    # The nearest qualifying third quarter, not the latest filed: a filer that
    # skipped one would otherwise pair against a year it did not report.
    return paired.sort_values(["cik", "gap_days"]).drop_duplicates("cik", keep="first")


@lru_cache(maxsize=1)
def _derived():
    paired = _paired()
    frame = paired.loc[
        paired.NetIncomeLoss_ytd.notna() & paired.nine_month_ytd.notna()
    ].copy()
    if frame.empty:
        raise ValueError("no paired filer reports both year-to-date figures")
    frame["year_usd"] = [Decimal(str(x)) for x in frame.NetIncomeLoss_ytd]
    frame["nine_month_usd"] = [Decimal(str(x)) for x in frame.nine_month_ytd]
    frame["derived_q4_usd"] = frame.year_usd - frame.nine_month_usd
    return frame


@lru_cache(maxsize=1)
def _submissions():
    """Annual filings joined to the submission index on filer and accession.

    An accession filed jointly by several filers appears once per filer, so the
    accession alone does not identify a row.
    """
    annual = _annual()[["cik", "accession", "period_end"]].copy()
    annual["cik"] = annual.cik.astype(str)
    annual["accession"] = annual.accession.astype(str)

    index = load_filings().copy()
    index["cik"] = index.cik.astype(str)
    index["accession"] = index.accession.astype(str)
    joined = annual.merge(
        index[["cik", "accession", "form", "filing_date", "acceptance_datetime"]],
        on=["cik", "accession"],
    )
    if joined.empty:
        raise ValueError("no annual filing appears in the submission index")
    joined["days_to_public"] = (
        pd.to_datetime(joined.acceptance_datetime.astype(str).str[:10])
        - pd.to_datetime(joined.period_end)
    ).dt.days
    return joined, index


@lru_cache(maxsize=1)
def analysis():
    annual = _annual()
    derived = _derived()

    positive = derived.loc[derived.year_usd > 0]
    validation = derived.loc[derived.NetIncomeLoss_q.notna()].copy()
    if validation.empty:
        raise ValueError("no filer discloses a quarterly figure to check against")
    validation["reported_q4_usd"] = [Decimal(str(x)) for x in validation.NetIncomeLoss_q]
    validation["gap_usd"] = validation.derived_q4_usd - validation.reported_q4_usd
    disagreeing = validation.loc[validation.gap_usd.abs() >= AGREEMENT_USD]
    if disagreeing.empty:
        raise ValueError("every disclosed quarter matches, so there is nothing to classify")

    widest = disagreeing.assign(magnitude=disagreeing.gap_usd.abs()).sort_values(
        ["magnitude", "cik"], ascending=[False, True], kind="stable"
    )
    if widest.magnitude.iloc[0] == widest.magnitude.iloc[1]:
        raise ValueError("the widest disagreement is not a strict leader")
    leader = widest.iloc[0]

    joined, index = _submissions()
    per_accession = index.groupby("accession").cik.nunique()
    shared = frozenset(per_accession.index[per_accession > 1])

    amendments = index.loc[index.form.eq(AMENDED_FORM)].copy()
    amendments["amended_period"] = amendments.report_date.astype(str)
    against = joined.merge(
        amendments[["cik", "amended_period", "filing_date"]], on="cik", suffixes=("", "_amend")
    )
    superseded = against.loc[
        against.amended_period.eq(against.period_end)
        & (pd.to_datetime(against.filing_date_amend) > pd.to_datetime(against.filing_date))
    ]
    if superseded.empty:
        raise ValueError("no annual filing in the window was later amended")

    return {
        "annual_filer_count": len(annual),
        "noncalendar_year_end_filers": int((~annual.period_end.str.endswith("-12-31")).sum()),
        "annual_with_year_net_income": int(annual.NetIncomeLoss_ytd.notna().sum()),
        "annual_with_quarter_net_income": int(annual.NetIncomeLoss_q.notna().sum()),
        "paired_filer_count": len(_paired()),
        "derived_filer_count": len(derived),
        "year_net_income_bn": float(sum(derived.year_usd, Decimal(0)) / BILLION),
        "derived_q4_net_income_bn": float(sum(derived.derived_q4_usd, Decimal(0)) / BILLION),
        "positive_year_filers": len(positive),
        "positive_year_negative_quarter_filers": int((positive.derived_q4_usd < 0).sum()),
        "validation_filer_count": len(validation),
        "validation_disagreement_count": len(disagreeing),
        "annual_figure_in_quarter_slot_count": int(
            (disagreeing.reported_q4_usd == disagreeing.year_usd).sum()
        ),
        "largest_disagreement_cik": str(leader.cik).zfill(10),
        "largest_disagreement_derived_mn": float(leader.derived_q4_usd / MILLION),
        "largest_disagreement_reported_mn": float(leader.reported_q4_usd / MILLION),
        "shared_accession_count": int(joined.accession.isin(shared).sum()),
        "superseded_by_amendment_count": int(superseded.accession.nunique()),
        "median_days_to_public": float(joined.days_to_public.median()),
        "form_disagreement_count": int(joined.form.ne("10-K").sum()),
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
