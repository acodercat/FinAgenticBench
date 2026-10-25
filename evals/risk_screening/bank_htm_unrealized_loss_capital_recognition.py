"""The loss a bank holds at amortised cost never reaches its capital ratio.

A security classified as held to maturity stays on the balance sheet at
amortised cost. Its fair value is disclosed beside it and the gap between the
two is real money, but it is not carried anywhere that regulatory capital can
see. At 2025-09-30 the 4,435 banks filing these schedules held 221.749484 USD
billions of unrealised loss inside held-to-maturity portfolios, against
115.374711 in available for sale, where a mark does pass through other
comprehensive income for a bank that did not elect out of it.

Recognising the held-to-maturity gap is the standard adjustment and it is
large. Across the 2,710 banks that report risk-weighted assets, common equity
tier 1 capital of 2192.620735 USD billions against 15260.143275 of risk-weighted
assets is a 14.3683% ratio; charging that group's own 220.207743 of
held-to-maturity loss against it leaves 12.9253%. The count below the 7% that a
4.5% minimum and a 2.5% conservation buffer together imply goes from 4 banks to
9, and for two banks the loss is larger than the whole of their common equity
tier 1: the largest, RSSD 480330, carries a held-to-maturity loss equal to
235.2045% of it.

Five things the schedules force:

1. A bank electing the community bank leverage ratio reports a leverage ratio
   and no risk-weighted assets. All 1,725 electors here are exactly the 1,725
   banks with no risk-weighted-asset figure, so the ratio work covers 2,710 of
   the 4,435 and the excluded group is not a random sample of the file.
   The three counts add up for that reason: 1,725 electors plus 2,710 with
   risk-weighted assets is the whole 4,435. The identity is the finding, so
   all three are reported rather than two of them and a subtraction.
2. The available-for-sale mark is already inside reported capital for a bank
   that did not elect the opt-out, and outside it for one that did. The
   schedules do not say which, so the adjustment made here is the
   held-to-maturity one, which is outside reported capital for every bank.
3. Unrealised amounts are fair value less amortised cost, so a loss is
   negative and is added to capital rather than subtracted from it. The two
   totals reported for the whole file are struck over all 4,435 banks, while
   the adjustment uses only the 2,710 in the ratio group, whose own
   held-to-maturity loss is 220.207743 rather than 221.749484.
4. Securities are reported on each bank's own basis, domestic for 4,355 of
   them and consolidated for 80, and are taken as reported. Within the ratio
   group the two schedules never disagree: 2,633 of those banks file both on
   the domestic basis and 77 file both consolidated. The 1,725 rows whose
   capital basis is blank are exactly the electors, and none of them reaches
   the ratio group, so no held-to-maturity figure is ever set against capital
   struck on a different basis.

5. A held-to-maturity portfolio counts when its amortised cost is above zero.
   One bank reports a zero cost against a fair value of 5 thousand USD, so the
   looser reading of holding a portfolio at all would give 1,640 rather than
   1,639. Its 5 thousand USD sits in the file total either way.

40.3287% of the securities carried here are pledged. A pledged security cannot
be sold to realise anything, so the gap between a held-to-maturity loss on
paper and a loss a bank could act on is wider than the adjustment alone
suggests.
"""

from decimal import Decimal
from functools import lru_cache

from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


REPORT_DATE = "2025-09-30"
BUFFERED_MINIMUM_PCT = 7.0
BILLION = Decimal(10) ** 6  # the schedules report thousands of USD

OUTPUTS = (
    ("bank_count", 0, "the count of banks filing both schedules as an integer"),
    ("htm_holder_count", 0, "how many hold a held-to-maturity portfolio, as an integer"),
    ("htm_loss_holder_count", 0, "how many of those carry an unrealised loss on it, as an integer"),
    ("htm_unrealized_bn", 6, "the held-to-maturity unrealised amount summed, in USD billions"),
    ("afs_unrealized_bn", 6, "the available-for-sale unrealised amount summed, in USD billions"),
    ("cblr_elector_count", 0, "the count electing the community bank leverage ratio as an integer"),
    ("ratio_bank_count", 0, "the count reporting risk-weighted assets as an integer"),
    ("cet1_capital_bn", 6, "those banks' common equity tier 1 capital summed, in USD billions"),
    ("aggregate_cet1_ratio_pct", 4, "their capital over their risk-weighted assets, in percent"),
    ("aggregate_adjusted_ratio_pct", 4, "the same ratio after their held-to-maturity amount is taken into capital"),
    ("below_seven_before_count", 0, "how many of them sit below the buffered minimum beforehand, as an integer"),
    ("below_seven_after_count", 0, "how many do afterwards, as an integer"),
    ("worst_bank_rssd", None, "the bank carrying the largest such loss against its capital, by its source-verbatim RSSD identifier"),
    ("worst_htm_loss_to_cet1_pct", 4, "that loss as a percentage of the bank's capital"),
    ("htm_loss_exceeds_cet1_count", 0, "how many banks carry a loss larger than their capital, as an integer"),
    ("pledged_share_pct", 4, "pledged securities as a percentage of the securities carried across the file"),
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
def _banks():
    securities = load_expansion_table("ffiec_call_reports_securities")
    capital = load_expansion_table("ffiec_call_reports_capital")
    capital = capital.loc[capital.report_date.astype(str).eq(REPORT_DATE)]
    if capital.empty:
        raise ValueError(f"the capital schedule carries no rows at {REPORT_DATE}")
    if not securities.report_date.astype(str).eq(REPORT_DATE).all():
        raise ValueError("the securities schedule spans more than the stated date")

    frame = securities.merge(
        capital, left_on="bank_rssd_id", right_on="bank_id", validate="one_to_one"
    )
    if len(frame) != len(securities):
        raise ValueError("a securities row has no capital row at the same date")
    frame = frame.copy()
    frame["htm_unrealized"] = (
        frame.htm_fair_value_thousand_usd - frame.htm_amortized_cost_thousand_usd
    )
    frame["afs_unrealized"] = (
        frame.afs_fair_value_thousand_usd - frame.afs_amortized_cost_thousand_usd
    )
    # A held-to-maturity portfolio is carried at cost and available for sale at
    # fair value, so the file's carrying amount mixes the two bases.
    frame["securities_carried"] = (
        frame.htm_amortized_cost_thousand_usd + frame.afs_fair_value_thousand_usd
    )
    return frame


@lru_cache(maxsize=1)
def _ratio_group():
    frame = _banks()
    weighted = frame.standardized_risk_weighted_assets_thousand_usd
    group = frame.loc[
        weighted.notna() & (weighted > 0) & frame.common_equity_tier1_capital_thousand_usd.notna()
    ]
    if group.empty:
        raise ValueError("no bank reports both risk-weighted assets and capital")
    return group


def _sum(series):
    return sum((Decimal(str(x)) for x in series), Decimal(0))


@lru_cache(maxsize=1)
def analysis():
    frame = _banks()
    group = _ratio_group()

    holds = frame.htm_amortized_cost_thousand_usd > 0
    capital = _sum(group.common_equity_tier1_capital_thousand_usd)
    weighted = _sum(group.standardized_risk_weighted_assets_thousand_usd)
    group_htm = _sum(group.htm_unrealized)
    if not weighted:
        raise ValueError("the ratio group reports no risk-weighted assets")

    before = group.common_equity_tier1_capital_thousand_usd / group.standardized_risk_weighted_assets_thousand_usd * 100
    after = (
        group.common_equity_tier1_capital_thousand_usd + group.htm_unrealized
    ) / group.standardized_risk_weighted_assets_thousand_usd * 100

    positive = group.loc[group.common_equity_tier1_capital_thousand_usd > 0].copy()
    positive["load"] = -positive.htm_unrealized / positive.common_equity_tier1_capital_thousand_usd * 100
    # The query breaks a tie on this ranking by the lower identifier; the case
    # test asserts that nothing currently ties.
    ranked = positive.assign(
        identifier=positive.bank_rssd_id.astype(int)
    ).sort_values(["load", "identifier"], ascending=[False, True])
    worst = ranked.iloc[0]

    carried = _sum(frame.securities_carried)
    if not carried:
        raise ValueError("the file carries no securities to take a share of")

    return {
        "bank_count": len(frame),
        "htm_holder_count": int(holds.sum()),
        "htm_loss_holder_count": int((holds & (frame.htm_unrealized < 0)).sum()),
        "htm_unrealized_bn": float(_sum(frame.htm_unrealized) / BILLION),
        "afs_unrealized_bn": float(_sum(frame.afs_unrealized) / BILLION),
        "cblr_elector_count": int(frame.cblr_election.sum()),
        "ratio_bank_count": len(group),
        "cet1_capital_bn": float(capital / BILLION),
        "aggregate_cet1_ratio_pct": float(capital / weighted * 100),
        "aggregate_adjusted_ratio_pct": float((capital + group_htm) / weighted * 100),
        "below_seven_before_count": int((before < BUFFERED_MINIMUM_PCT).sum()),
        "below_seven_after_count": int((after < BUFFERED_MINIMUM_PCT).sum()),
        "worst_bank_rssd": str(worst.bank_rssd_id),
        "worst_htm_loss_to_cet1_pct": float(worst.load),
        "htm_loss_exceeds_cet1_count": int((positive.load > 100).sum()),
        "pledged_share_pct": float(
            _sum(frame.pledged_securities_book_value_thousand_usd) / carried * 100
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


validators = {
    f"validate_turn_{i}": turn_validator(globals()[f"validate_turn_{i}"])
    for i in range(1, 5)
}
