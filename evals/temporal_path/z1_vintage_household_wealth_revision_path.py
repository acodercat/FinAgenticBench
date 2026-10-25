"""Read a Z.1 balance-sheet figure as it stood at a decision date, then as revised.

The Financial Accounts vintages are full restatements: each release republishes
every quarter back to 2015, so the value of a quarter is a function of which
release you read. On 2024-10-01 the newest release was 2024-09-12, whose last
quarter is 2024-06-30, and household and nonprofit net worth (FL152090005.Q)
for that quarter stood at 163,796,961 million USD. The 2025-09-11 release puts
the same quarter at 166,159,502 million USD, 2,362,541 million USD (1.4424
percent) higher, after five releases covered it. Across the b101 stock series
that all seven releases cover for 2015Q1-2023Q4, the largest single revision is
LM153064105.Q in 2021Q4: 31,656,063 restated to 33,033,743 million USD, 112,680
million USD ahead of the runner-up pair. The case closes on the Philadelphia Fed
real-time file, where 2024Q2 real GDP growth was first released at 2.8401 and
now reads 3.5895 percent, a 0.7494 point revision.

Rejected alternatives (each measured in the `z1_vintage_revision` comparison): reading the newest release for the decision date instead of the
newest release published on or before it (163,796,961 becomes 166,159,502);
including FA flow series alongside FL and LM stocks; and ranking by each
series' summed absolute revision rather than by the single largest revision,
which keeps the same series but moves the quarter. Three further pins are
documented equivalents rather than answer-changing choices: ranking on signed
rather than absolute revisions cannot move a leader whose revision is
+1,377,680 million USD when the largest negative revision in the population is
-552,436; and the population and revision-base choices are inert on their own,
because differencing against the earliest release already drops the pairs that
release lacks, while restricting to pairs all seven releases report already
makes the first covering release the earliest one. Only switching both at once
moves the answer, to FL893131573.Q in 2023Q4 over 1,836 pairs. The query pins
all six so a reader need not derive those equivalences.

The vintage table carries no series labels, only Financial Accounts codes, so
turn 3 can only be answered by code; the query names concepts for the series it
states outright and asks for a code for the one it does not.

One row is a vintage, statement table, series and quarter, not a vintage,
series and quarter: household net worth is printed in b1, b101 and b101e, 819
rows across the panel. Every copy agrees -- no series-quarter-vintage triple in
the quarterly panel carries two different values -- so reading a series without
naming a table is safe here, and the lookups assert it rather than assume it.
The frequency field holds the last segment of the series code, so the suffixed
`.Q.1` and `.Q.2` presentations sit outside the Q selection; table b101 has
none of them.

Fragility: the panel starts at the 2024-03-07 release, so the count of releases
available at the decision date, the earliest-vintage base of every revision and
the seven-release population rule all move if the artifact is ever extended
backwards; the frozen artifact fixes them. The leading pair's 8.18 percent
margin and the 2,362,541 million USD turn-2 revision are wide enough that a
single restated quarter would not reorder them.

Numeric validation uses the shared 0.6 x 10^-N rounding-boundary tolerance for
N requested decimals; counts, codes and dates match exactly.
"""

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


DECISION_DATE = "2024-10-01"
HOUSEHOLD_NET_WORTH = "FL152090005.Q"
REVISION_TABLE = "b101"
REVISION_WINDOW = ("2015-03-31", "2023-12-31")
STOCK_PREFIXES = ("FL", "LM")
RTDSM_SERIES = "real_gdp_annualized_pct"
RTDSM_PERIOD = "2024:Q2"


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = [
    "available_vintage_date", "vintage_count_at_decision_date", "latest_covered_quarter",
    "publication_lag_days", "as_published_net_worth_usd_millions",
]
TURN_2_NAMES = [
    "newest_vintage_date", "revised_net_worth_usd_millions", "net_worth_revision_pct",
    "covering_vintage_count",
]
TURN_3_NAMES = [
    "revision_series_count", "revision_pair_count", "largest_revision_series",
    "largest_revision_quarter", "largest_revision_first_value_usd_millions",
    "largest_revision_latest_value_usd_millions", "largest_revision_usd_millions",
    "largest_revision_margin_usd_millions", "household_net_worth_largest_revision_rank",
]
TURN_4_NAMES = [
    "rtdsm_first_release_pct", "rtdsm_third_release_pct", "rtdsm_most_recent_pct",
    "rtdsm_quarters_with_larger_absolute_revision",
]

variables = [
    _v(TURN_1_NAMES[0], "Store the release date of the newest Financial Accounts vintage available at the decision date as an ISO YYYY-MM-DD string."),
    _v(TURN_1_NAMES[1], "Store how many vintages in the table were published on or before the decision date as an integer."),
    _v(TURN_1_NAMES[2], "Store the latest quarter that vintage covers as an ISO YYYY-MM-DD quarter-end date string."),
    _v(TURN_1_NAMES[3], "Store the number of calendar days from that quarter end to that vintage's release date as an integer."),
    _v(TURN_1_NAMES[4], "Store household and nonprofit net worth for that quarter as that vintage published it, in millions of USD rounded to 3 decimals."),
    _v(TURN_2_NAMES[0], "Store the release date of the newest vintage in the table as an ISO YYYY-MM-DD string."),
    _v(TURN_2_NAMES[1], "Store the newest vintage's value for the same series and quarter in millions of USD rounded to 3 decimals."),
    _v(TURN_2_NAMES[2], "Store that revision as a percent of the as-published value, rounded to 4 decimals."),
    _v(TURN_2_NAMES[3], "Store how many vintages in the table report that series for that quarter as an integer."),
    _v(TURN_3_NAMES[0], "Store the number of distinct series in the revision population as an integer."),
    _v(TURN_3_NAMES[1], "Store the number of series-quarter pairs in the revision population as an integer."),
    _v(TURN_3_NAMES[2], "Store the Financial Accounts code of the series carrying the largest absolute revision as text."),
    _v(TURN_3_NAMES[3], "Store that revision's quarter as an ISO YYYY-MM-DD quarter-end date string."),
    _v(TURN_3_NAMES[4], "Store that series-quarter's earliest-vintage value in millions of USD rounded to 3 decimals."),
    _v(TURN_3_NAMES[5], "Store that series-quarter's newest-vintage value in millions of USD rounded to 3 decimals."),
    _v(TURN_3_NAMES[6], "Store the signed revision of that series-quarter in millions of USD rounded to 3 decimals."),
    _v(TURN_3_NAMES[7], "Store the leading absolute revision minus the runner-up absolute revision in millions of USD rounded to 3 decimals."),
    _v(TURN_3_NAMES[8], "Store the rank of household and nonprofit net worth's own largest absolute revision within the population, 1 being largest, as an integer."),
    _v(TURN_4_NAMES[0], "Store the first-release real GDP growth for the reference quarter in percent, rounded to 4 decimals."),
    _v(TURN_4_NAMES[1], "Store the third-release value for the same quarter in percent, rounded to 4 decimals."),
    _v(TURN_4_NAMES[2], "Store the most recent value for the same quarter in percent, rounded to 4 decimals."),
    _v(TURN_4_NAMES[3], "Store how many quarters of that series carry a larger absolute first-to-latest revision as an integer."),
]

DECIMALS = [
    None, 0, None, 0, 3,
    None, 3, 4, 0,
    0, 0, None, None, 3, 3, 3, 3, 0,
    4, 4, 4, 0,
]


def _vintages():
    frame = load_expansion_table("fed_z1_vintages").copy()
    frame["value"] = pd.to_numeric(frame.value, errors="coerce")
    return frame.loc[frame.frequency.astype(str).eq("Q")]


def _single_value(rows, label):
    """The value of one series-quarter, which repeats once per statement table."""
    values = rows.value.dropna().unique()
    if len(values) != 1:
        raise ValueError(f"{label}: expected one value across statement tables, found {list(values)}")
    return float(values[0])


def _revision_population(quarterly, vintages):
    table = quarterly.loc[
        quarterly.table.astype(str).eq(REVISION_TABLE)
        & quarterly.series.astype(str).str.startswith(STOCK_PREFIXES)
        & quarterly.date.astype(str).between(*REVISION_WINDOW)
    ]
    covered = table.groupby(["series", "date"]).vintage.nunique()
    complete = covered[covered.eq(len(vintages))].index
    panel = table.pivot_table(index=["series", "date"], columns="vintage", values="value", aggfunc="first")
    panel = panel.loc[panel.index.isin(complete)].copy()
    panel["revision"] = panel[vintages[-1]] - panel[vintages[0]]
    panel["absolute"] = panel.revision.abs()
    return panel.sort_values(["absolute", "series", "date"], ascending=[False, True, True])


@lru_cache(maxsize=1)
def ground_truth():
    quarterly = _vintages()
    vintages = sorted(quarterly.vintage.astype(str).unique())
    available = [vintage for vintage in vintages if vintage <= DECISION_DATE]
    as_of = available[-1]

    snapshot = quarterly.loc[quarterly.vintage.astype(str).eq(as_of)]
    covered_quarter = str(snapshot.date.astype(str).max())
    published = snapshot.loc[
        snapshot.series.astype(str).eq(HOUSEHOLD_NET_WORTH) & snapshot.date.astype(str).eq(covered_quarter)
    ]
    if published.empty:
        raise ValueError("the household net worth series is missing from the decision-date vintage")
    published_value = _single_value(published, f"{HOUSEHOLD_NET_WORTH} at {as_of}")
    lag_days = (pd.Timestamp(as_of) - pd.Timestamp(covered_quarter)).days

    newest = vintages[-1]
    revised = quarterly.loc[
        quarterly.vintage.astype(str).eq(newest)
        & quarterly.series.astype(str).eq(HOUSEHOLD_NET_WORTH)
        & quarterly.date.astype(str).eq(covered_quarter)
    ]
    revised_value = _single_value(revised, f"{HOUSEHOLD_NET_WORTH} at {newest}")
    covering = quarterly.loc[
        quarterly.series.astype(str).eq(HOUSEHOLD_NET_WORTH) & quarterly.date.astype(str).eq(covered_quarter), "vintage"
    ].nunique()

    panel = _revision_population(quarterly, vintages)
    leader = panel.iloc[0]
    runner_up = panel.iloc[1]
    leader_series, leader_quarter = panel.index[0]
    per_series = (
        panel.groupby(level=0).absolute.max().rename("absolute").reset_index()
        .sort_values(["absolute", "series"], ascending=[False, True])
        .reset_index(drop=True)
    )
    household_rank = int(per_series.index[per_series.series.eq(HOUSEHOLD_NET_WORTH)][0]) + 1

    rtdsm = load_expansion_table("philadelphia_fed_rtdsm").copy()
    series = rtdsm.loc[rtdsm.series.astype(str).eq(RTDSM_SERIES)].copy()
    for column in ("first_release", "second_release", "third_release", "most_recent"):
        series[column] = pd.to_numeric(series[column], errors="coerce")
    reference = series.loc[series.period.astype(str).eq(RTDSM_PERIOD)]
    if reference.empty:
        raise ValueError("the reference quarter is missing from the real-time data set")
    reference = reference.iloc[0]
    reference_revision = float(reference.most_recent) - float(reference.first_release)
    complete = series.dropna(subset=["first_release", "most_recent"]).copy()
    complete["revision"] = complete.most_recent - complete.first_release
    larger = int((complete.revision.abs() > abs(reference_revision)).sum())

    return (
        as_of,
        len(available),
        covered_quarter,
        int(lag_days),
        published_value,
        newest,
        revised_value,
        (revised_value - published_value) / published_value * 100,
        int(covering),
        int(panel.index.get_level_values(0).nunique()),
        int(len(panel)),
        str(leader_series),
        str(leader_quarter),
        float(leader[vintages[0]]),
        float(leader[vintages[-1]]),
        float(leader.revision),
        float(leader.absolute - runner_up.absolute),
        household_rank,
        float(reference.first_release),
        float(reference.third_release),
        float(reference.most_recent),
        larger,
    )


def _validate_subset(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip(by_name, ground_truth()))
    decimals = dict(zip(by_name, DECIMALS))
    return validate_ordered_outputs(
        outputs, [by_name[name] for name in names], [truth[name] for name in names],
        [decimals[name] for name in names],
    )


def validate_turn_1(outputs): return _validate_subset(outputs, TURN_1_NAMES)
def validate_turn_2(outputs): return _validate_subset(outputs, TURN_2_NAMES)
def validate_turn_3(outputs): return _validate_subset(outputs, TURN_3_NAMES)
def validate_turn_4(outputs): return _validate_subset(outputs, TURN_4_NAMES)
def validate(outputs): return _validate_subset(outputs, [variable.name for variable in variables])


validators = {
    "validate_decision_date_reading": turn_validator(validate_turn_1),
    "validate_observation_revision": turn_validator(validate_turn_2),
    "validate_revision_ranking": turn_validator(validate_turn_3),
    "validate_release_sequence_contrast": turn_validator(validate_turn_4),
}
