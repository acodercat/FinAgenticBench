"""A crowdfunding offering ceiling, who states it, and what those issuers do next.

Regulation Crowdfunding caps what an issuer may raise in twelve months. Before
the amendments effective 2021-03-15 the cap was 1,070,000 dollars; from that
date it is 5,000,000. The original Form C filings show both as hard ceilings.
Of the 2,124 original offerings stating a maximum before the change, none
states more than 1,070,000 and 636 state exactly that; of the 6,729 from the
change through 2025, none states more than 5,000,000 and 391 state exactly
that. The first offering above the old ceiling was filed on 2021-05-11, almost
two months after the higher cap took effect.

Stating the ceiling does not by itself mean the cap held an issuer back. The
stated maximum is the most an issuer may accept if oversubscribed: the filings
at the old ceiling aim for a median target of 25,000 dollars, the same as the
others, with a maximum 42.8 times the target against 5.0 for the rest, so many
simply put the maximum at the legal limit. What the filings do show is that
issuers stating the ceiling behave differently afterwards. Among original
Form C filings from 2020-07-01 through 2024-12-31, so that every filing has a
full year of Form D record after it, 15.67 percent of those at the ceiling
before the change were followed within 365 days by an original Form D notice
from the same issuer, against 6.83 percent of those below it; after the change
the figures are 12.50 against 5.36. The pattern holds under both caps, and it
can reflect which issuers choose to state the ceiling as much as the ceiling
itself.

The same issuers show it across the change. Of the 618 distinct issuers that
filed at the old ceiling before the change, 81 later filed an offering above
it; of the 1,346 whose earlier offerings never stated the ceiling, 95 did,
13.11 against 7.06 percent. The uncapped group counts only issuers whose
earlier filings state a maximum; admitting issuers with no stated maximum at
all would make it 1,378, with the same 95 returning.

The runtime Form D table holds notices only from 2020-07-01: the six earlier
quarters use a different date format in the source files and were dropped when
the table was built, a data-layer defect recorded during review. Because the
follow-on window looks only forward from Form C filings dated 2020-07-01 or
later, no output here depends on the missing quarters.

Readings measured so they are not mistaken for oversights:

1. The follow-on window includes the Form C filing date itself. Seventeen
   filings have a Form D notice on the same day; reading the window as
   strictly after the filing drops them. One follow-on arrives 366 days
   later and none at 364 or 365, so the far end does not sit on a boundary.
2. Shares are per Form C filing. Counting issuers instead, an issuer counting
   as followed if any of its filings is, gives 15.88 and 6.91 before the
   change and 12.96 and 5.44 after.
3. Filings that state no maximum are left out: 20 of the pre-change filings
   in the follow-on window carry none.
4. Every filing counts as filed. In 31 cases an issuer filed more than one
   original statement on the same day, 24 of them with identical contents.
   Treating identical same-day filings as one reads 2,117 and 633 before the
   change and 6,707 after, and moves the pre-change capped follow-on share to
   15.77 and the uncapped one to 6.84; the post-change shares are unchanged.
   Only original statements, and progress updates and amendments that also
   carry a maximum, state one at all, so no other Form C variant can enter.
"""

from functools import lru_cache

import numpy as np
import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


CHANGE_DATE = "2021-03-15"
CAP_WINDOW = ("2019-01-01", "2025-12-31")
FOLLOW_WINDOW = ("2020-07-01", "2024-12-31")
FOLLOW_DAYS = 365

OUTPUTS = (
    ("pre_change_filing_count", 0, "the count of original offerings stating a maximum before the change as an integer"),
    ("pre_change_highest_maximum_usd", 0, "the highest maximum any of them states, in dollars, as an integer"),
    ("pre_change_at_highest_count", 0, "how many state exactly that amount, as an integer"),
    ("post_change_filing_count", 0, "the count of original offerings stating a maximum from the change on as an integer"),
    ("post_change_highest_maximum_usd", 0, "the highest maximum any of them states, in dollars, as an integer"),
    ("post_change_at_highest_count", 0, "how many state exactly that amount, as an integer"),
    ("first_filing_above_old_highest", None, "the filing date of the first offering above the earlier highest maximum, as an ISO YYYY-MM-DD string"),
    ("pre_change_capped_follow_on_pct", 2, "the pre-change share of filings at the ceiling followed by a Form D notice, in percent"),
    ("pre_change_uncapped_follow_on_pct", 2, "the pre-change share of filings below the ceiling followed by one, in percent"),
    ("post_change_capped_follow_on_pct", 2, "the post-change share of filings at the ceiling followed by one, in percent"),
    ("post_change_uncapped_follow_on_pct", 2, "the post-change share of filings below the ceiling followed by one, in percent"),
    ("capped_issuer_count", 0, "how many distinct issuers filed an earlier-period offering at the earlier ceiling, as an integer"),
    ("capped_issuers_returning_above_count", 0, "how many of them later filed an offering above that ceiling, as an integer"),
    ("uncapped_issuer_count", 0, "how many distinct issuers filed earlier-period offerings none of which stated that ceiling, as an integer"),
    ("uncapped_issuers_returning_above_count", 0, "how many of them later filed an offering above that ceiling, as an integer"),
)


def _description(precision, text):
    if precision in (None, 0):
        return f"Store {text}."
    return f"Store {text}, rounded to {precision} decimals from unrounded source rows."


variables = [Variable(name, None, _description(p, text)) for name, p, text in OUTPUTS]
PRECISION = {name: p for name, p, _ in OUTPUTS}
TURNS = (
    tuple(n for n, _, _ in OUTPUTS[:3]),
    tuple(n for n, _, _ in OUTPUTS[3:7]),
    tuple(n for n, _, _ in OUTPUTS[7:11]),
    tuple(n for n, _, _ in OUTPUTS[11:]),
)


def _cik(series):
    return pd.to_numeric(series.astype(str).str.strip(), errors="coerce").astype("Int64")


@lru_cache(maxsize=1)
def _offerings():
    """Original Form C filings that state a maximum offering amount."""
    table = load_expansion_table("sec_form_c")
    frame = table.loc[table.SUBMISSION_TYPE.astype(str).eq("C")].copy()
    frame["filed"] = frame.FILING_DATE.astype(str)
    frame["cik"] = _cik(frame.CIK)
    frame["maximum"] = pd.to_numeric(frame.MAXIMUMOFFERINGAMOUNT, errors="coerce")
    frame = frame.loc[frame.maximum.notna() & frame.filed.between(*CAP_WINDOW)]
    frame["after_change"] = frame.filed.ge(CHANGE_DATE)
    return frame


@lru_cache(maxsize=1)
def _form_d_dates():
    table = load_expansion_table("sec_form_d")
    notices = table.loc[table.SUBMISSIONTYPE.astype(str).eq("D")].copy()
    notices["cik"] = _cik(notices.CIK)
    notices["filed"] = pd.to_datetime(notices.FILING_DATE.astype(str))
    return notices.dropna(subset=["cik"]).groupby("cik").filed.apply(lambda s: np.sort(s.to_numpy()))


def _followed(frame):
    dates = _form_d_dates()
    followed = []
    for cik, filed in zip(frame.cik, frame.filed):
        later = dates.get(cik)
        start = np.datetime64(filed)
        if later is None:
            followed.append(False)
            continue
        first = np.searchsorted(later, start)
        followed.append(bool(first < len(later) and later[first] <= start + np.timedelta64(FOLLOW_DAYS, "D")))
    return pd.Series(followed, index=frame.index)


def _share(mask):
    return float(mask.mean() * 100)


@lru_cache(maxsize=1)
def analysis():
    offerings = _offerings()
    before = offerings.loc[~offerings.after_change]
    after = offerings.loc[offerings.after_change]
    old_highest, new_highest = before.maximum.max(), after.maximum.max()

    window = offerings.loc[offerings.filed.between(*FOLLOW_WINDOW)].copy()
    window["followed"] = _followed(window)
    shares = {}
    for label, part, ceiling in (("pre", window.loc[~window.after_change], old_highest),
                                 ("post", window.loc[window.after_change], new_highest)):
        capped = part.maximum.eq(ceiling)
        shares[f"{label}_change_capped_follow_on_pct"] = _share(part.loc[capped, "followed"])
        shares[f"{label}_change_uncapped_follow_on_pct"] = _share(part.loc[~capped, "followed"])

    capped_issuers = set(before.loc[before.maximum.eq(old_highest), "cik"])
    uncapped_issuers = set(before.cik) - capped_issuers
    returning = set(after.loc[after.maximum > old_highest, "cik"])

    return {
        "pre_change_filing_count": len(before),
        "pre_change_highest_maximum_usd": int(old_highest),
        "pre_change_at_highest_count": int(before.maximum.eq(old_highest).sum()),
        "post_change_filing_count": len(after),
        "post_change_highest_maximum_usd": int(new_highest),
        "post_change_at_highest_count": int(after.maximum.eq(new_highest).sum()),
        "first_filing_above_old_highest": str(after.loc[after.maximum > old_highest, "filed"].min()),
        **shares,
        "capped_issuer_count": len(capped_issuers),
        "capped_issuers_returning_above_count": len(capped_issuers & returning),
        "uncapped_issuer_count": len(uncapped_issuers),
        "uncapped_issuers_returning_above_count": len(uncapped_issuers & returning),
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


def validate(outputs):
    return _check(outputs, tuple(v.name for v in variables))


def validate_turn_3(outputs):
    return _check(outputs, TURNS[2])


def validate_turn_4(outputs):
    return _check(outputs, TURNS[3])


validators = {
    f"validate_turn_{i}": turn_validator(globals()[f"validate_turn_{i}"])
    for i in range(1, 5)
}
