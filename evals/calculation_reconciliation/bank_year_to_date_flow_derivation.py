"""Turn cumulative bank income into a quarter, and see which ratios survive it.

Bank income statements in the regulatory files are reported year to date. A figure
dated 2024-09-30 covers nine months, not three, and the only way to a quarter is to
subtract the previous quarter's cumulative figure within the same year, with the
first quarter needing no subtraction at all.

Across the 163 banks holding more than 10 billion USD at the 2024-09-30 report date,
followed through the four 2024 dates, the cumulative net interest income divided by
the same panel's own quarterly income comes to 1.0000 at the first date, 2.0047 at
the second, 2.9517 at the third and 3.9125 at the fourth. The multiple is the
quarter number, a little under it because later quarters were larger. At the third
date the panel reports 435.8897 billion USD cumulative against 147.6760 billion
earned in the quarter, and its largest member reports 71.1180 billion against
24.1350, a multiple of 2.9467.

What the convention does to a ratio depends on what is underneath it. A ratio of a
flow to a stock inherits the period: net income over assets at the third date is a
nine-month return, and its median of 0.7334 percent is exactly 0.7500 of the 0.9779
percent that the same income annualized returns, the annualization factor and
nothing else. The denominator is a separate choice again. Bridging the panel to
reported risk-weighted assets through the branch file leaves 158 banks whose
risk-weighted assets are a median 0.7320 of total assets, so the same annualized
income measured against them returns a median 1.3918 percent, 0.4139 points above
the figure measured against assets.

A ratio of one flow to another does not inherit the period, because the factor
appears above and below and cancels. The efficiency ratio needs no annualization at
all. It is still not the same number: computed from cumulative flows it is the
year-to-date average across quarters whose mix differs, not the latest quarter. On
the 156 banks whose revenue base is positive on both bases the cumulative reading
has a median of 58.9811 percent against 57.8794 from the quarter alone, a median
absolute gap of 1.7819 points, and 25 of them differ by more than 5.

Measured rejected alternatives, from the `bank_ytd_flow_derivation` comparison, include annualizing the third quarter by four rather than by four thirds, setting
the panel at 1 billion USD of assets rather than 10, retaining banks whose
quarterly revenue base is not positive, substituting risk-weighted assets for
total assets, and counting a wide efficiency gap from 3 points rather than 5.
The query explicitly fixes each convention needed for the scored calculation, so
these probes test instruction following and reconciliation rather than a live
hidden-method trap. This is therefore a hard baseline case.

The first turn names the 2024-09-30 report date for the cumulative figure. The
earlier wording "at the report date", written straight after the four dates had
been listed, was read by 24 of 42 runs across 10 model configurations as the final
2024-12-31 date, which gives 585.5519 USD billions against 435.8897; the four
multiples were right in all 42, so the misreading was referential, not a
misunderstanding of the cumulative convention. The date is pinned in the query and
the variable description rather than inferred.

Knowledge made explicit in the query: regulatory income statements accumulate
within a year, a partial-year flow is annualized by four over the quarter number,
returns on total and risk-weighted assets use different denominators, and a ratio
of two flows from the same period requires no separate annualization factor.

Boundaries: this is one panel at one date, and the multiples depend on how income
was distributed across the quarters of that year rather than being fixed by the
convention. Differencing assumes the cumulative series is internally consistent; a
restatement inside the year would show up as a distorted quarter. The efficiency
ratio is unstable where a bank's quarterly revenue base is small, and the largest
gap in this panel reaches 139.5809 points on that account.

Numeric validation uses the shared 0.6 x 10^-N rounding-boundary tolerance for
N requested decimals; counts and bank names match exactly.
"""

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


REPORT_DATE = "2024-09-30"
YEAR = "2024"
QUARTER_DATES = ("2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31")
SIZE_THRESHOLD = 1e7
QUARTER_NUMBER = 3
FLOW_COLUMNS = ("INTINC", "EINTEXP", "NONII", "NONIX", "NETINC")
EFFICIENCY_GAP_THRESHOLD = 5.0


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = ["panel_banks", "first_quarter_multiple", "second_quarter_multiple",
                "third_quarter_multiple", "fourth_quarter_multiple", "cumulative_income_usd_billions"]
TURN_2_NAMES = ["quarter_income_usd_billions", "leader_name", "leader_cumulative_usd_billions",
                "leader_quarter_usd_billions", "leader_multiple"]
TURN_3_NAMES = ["median_return_from_cumulative", "median_return_annualized", "annualization_ratio",
                "bridged_banks", "median_risk_weight_share", "median_return_on_risk_weighted"]
TURN_4_NAMES = ["efficiency_panel", "median_efficiency_from_cumulative", "median_efficiency_from_quarter",
                "median_efficiency_gap", "banks_gap_above_five"]

variables = [
    _v(TURN_1_NAMES[0], "Store how many banks the panel holds as an integer."),
    _v(TURN_1_NAMES[1], "Store the cumulative-to-quarterly income multiple at the first date, rounded to 4 decimals."),
    _v(TURN_1_NAMES[2], "Store it at the second date, rounded to 4 decimals."),
    _v(TURN_1_NAMES[3], "Store it at the third date, rounded to 4 decimals."),
    _v(TURN_1_NAMES[4], "Store it at the fourth date, rounded to 4 decimals."),
    _v(TURN_1_NAMES[5], "Store the panel's cumulative net interest income at the 2024-09-30 report date in USD billions, rounded to 4 decimals."),
    _v(TURN_2_NAMES[0], "Store the panel's net interest income earned in the report quarter in USD billions, rounded to 4 decimals."),
    _v(TURN_2_NAMES[1], "Store the selected bank's name verbatim from FDIC financials as a string, preserving capitalization, punctuation and spacing."),
    _v(TURN_2_NAMES[2], "Store its cumulative net interest income in USD billions, rounded to 4 decimals."),
    _v(TURN_2_NAMES[3], "Store its quarterly net interest income in USD billions, rounded to 4 decimals."),
    _v(TURN_2_NAMES[4], "Store the ratio of the two, rounded to 4 decimals."),
    _v(TURN_3_NAMES[0], "Store the median of net income over assets using the cumulative figure, in percent, rounded to 4 decimals."),
    _v(TURN_3_NAMES[1], "Store the median using the annualized figure, in percent, rounded to 4 decimals."),
    _v(TURN_3_NAMES[2], "Store the first median divided by the second, rounded to 4 decimals."),
    _v(TURN_3_NAMES[3], "Store how many panel banks carry reported risk-weighted assets as an integer."),
    _v(TURN_3_NAMES[4], "Store the median ratio of risk-weighted assets to total assets, rounded to 4 decimals."),
    _v(TURN_3_NAMES[5], "Store the median annualized return on risk-weighted assets in percent, rounded to 4 decimals."),
    _v(TURN_4_NAMES[0], "Store how many banks carry a positive revenue base on both readings as an integer."),
    _v(TURN_4_NAMES[1], "Store their median efficiency ratio from cumulative flows in percent, rounded to 4 decimals."),
    _v(TURN_4_NAMES[2], "Store it from quarterly flows in percent, rounded to 4 decimals."),
    _v(TURN_4_NAMES[3], "Store the median absolute difference between the two in percentage points, rounded to 4 decimals."),
    _v(TURN_4_NAMES[4], "Store how many banks differ by more than 5 percentage points as an integer."),
]

DECIMALS = [0, 4, 4, 4, 4, 4,
            4, None, 4, 4, 4,
            4, 4, 4, 0, 4, 4,
            0, 4, 4, 4, 0]


@lru_cache(maxsize=1)
def _financials():
    frame = load_expansion_table("fdic_bankfind").copy()
    for column in (*FLOW_COLUMNS, "ASSET"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["day"] = frame.REPDTE.astype(str)
    frame["certificate"] = frame.CERT.astype(str).str.strip()
    frame["year"] = frame.day.str[:4]
    frame = frame.sort_values(["certificate", "day"])
    for column in FLOW_COLUMNS:
        previous = frame.groupby(["certificate", "year"])[column].shift(1).fillna(0)
        frame["quarter_" + column] = frame[column] - previous
    frame["cumulative_nii"] = frame.INTINC - frame.EINTEXP
    frame["quarter_nii"] = frame.quarter_INTINC - frame.quarter_EINTEXP
    return frame


@lru_cache(maxsize=1)
def _panel():
    frame = _financials()
    members = frame.loc[frame.day.eq(REPORT_DATE) & frame.ASSET.gt(SIZE_THRESHOLD), "certificate"]
    return frame.loc[frame.certificate.isin(set(members)) & frame.year.eq(YEAR)].copy()


@lru_cache(maxsize=1)
def _bridge():
    branches = load_expansion_table("fdic_sod").copy()
    branches["certificate"] = branches.fdic_certificate.astype(str).str.strip()
    branches["rssd"] = branches.bank_rssd_id.astype(str).str.strip()
    counted = branches.groupby(["certificate", "rssd"], as_index=False).size()
    counted = counted.sort_values(["certificate", "size", "rssd"], ascending=[True, False, True])
    return counted.drop_duplicates("certificate")[["certificate", "rssd"]]


@lru_cache(maxsize=1)
def ground_truth():
    panel = _panel()
    at_date = panel.loc[panel.day.eq(REPORT_DATE)].copy()
    multiples = [
        float(panel.loc[panel.day.eq(day), "cumulative_nii"].sum()
              / panel.loc[panel.day.eq(day), "quarter_nii"].sum())
        for day in QUARTER_DATES
    ]
    leader = at_date.sort_values(["ASSET", "certificate"], ascending=[False, True]).iloc[0]

    at_date["return_cumulative"] = at_date.NETINC / at_date.ASSET * 100
    at_date["return_annualized"] = at_date.NETINC * (4 / QUARTER_NUMBER) / at_date.ASSET * 100

    capital = load_expansion_table("ffiec_call_reports_capital").copy()
    capital["rssd"] = capital.bank_id.astype(str).str.strip()
    capital["risk_weighted"] = pd.to_numeric(
        capital.standardized_risk_weighted_assets_thousand_usd, errors="coerce")
    dated = capital.loc[capital.report_date.astype(str).eq(REPORT_DATE), ["rssd", "risk_weighted"]]
    bridged = at_date.merge(_bridge(), on="certificate", how="left").merge(dated, on="rssd", how="left")
    bridged = bridged.loc[bridged.risk_weighted.gt(0)].copy()
    bridged["risk_share"] = bridged.risk_weighted / bridged.ASSET
    bridged["return_on_risk_weighted"] = bridged.NETINC * (4 / QUARTER_NUMBER) / bridged.risk_weighted * 100

    at_date["revenue_cumulative"] = at_date.cumulative_nii + at_date.NONII
    at_date["revenue_quarter"] = at_date.quarter_nii + at_date.quarter_NONII
    priced = at_date.loc[at_date.revenue_cumulative.gt(0) & at_date.revenue_quarter.gt(0)].copy()
    priced["efficiency_cumulative"] = priced.NONIX / priced.revenue_cumulative * 100
    priced["efficiency_quarter"] = priced.quarter_NONIX / priced.revenue_quarter * 100
    gap = (priced.efficiency_cumulative - priced.efficiency_quarter).abs()

    return (
        int(len(at_date)), *[float(value) for value in multiples],
        float(at_date.cumulative_nii.sum()) / 1e6,
        float(at_date.quarter_nii.sum()) / 1e6, str(leader.NAME),
        float(leader.cumulative_nii) / 1e6, float(leader.quarter_nii) / 1e6,
        float(leader.cumulative_nii / leader.quarter_nii),
        float(at_date.return_cumulative.median()), float(at_date.return_annualized.median()),
        float(at_date.return_cumulative.median() / at_date.return_annualized.median()),
        int(len(bridged)), float(bridged.risk_share.median()),
        float(bridged.return_on_risk_weighted.median()),
        int(len(priced)), float(priced.efficiency_cumulative.median()),
        float(priced.efficiency_quarter.median()), float(gap.median()),
        int(gap.gt(EFFICIENCY_GAP_THRESHOLD).sum()),
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
    "validate_cumulative_structure": turn_validator(validate_turn_1),
    "validate_quarter_derivation": turn_validator(validate_turn_2),
    "validate_flow_to_stock": turn_validator(validate_turn_3),
    "validate_flow_to_flow": turn_validator(validate_turn_4),
}
