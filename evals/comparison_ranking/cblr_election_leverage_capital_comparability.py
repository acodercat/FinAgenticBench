"""Compare capital measures across CBLR electors and non-electors.

The case links the 2025-09-30 Call Report capital schedule to BankFind through
the current NIC RSSD-to-certificate bridge, then applies a BankFind total-asset
band. NIC's primary federal regulator supplies independent institution context
for the lowest-leverage elector and prevents the identifier table from serving
as a replaceable bridge only.

CBLR electors and non-electors both report the tier-1 leverage ratio, while the
selected electors do not report standardized risk-weighted assets or CET1 ratios.
An elector below 9 percent is not, from this snapshot alone, proof of a violation:
the framework includes grace-period mechanics above its lower boundary, and
eligibility history is not present in the declared tables.

The final turn contrasts BankFind book equity over ending total assets with tier-1
capital over the regulatory leverage denominator. A divergence can reflect
regulatory capital adjustments, measurement scope and different denominators;
these sources do not isolate AOCI treatment or any other cause for a bank's gap.

The pinned-reading comparison measures central tendency, the CET1 population, the
threshold rule and the leverage measure. Taking means instead of medians reports
12.5139% and 11.2255% rather than 11.3685% and 10.1835%, and widens the gap to
1.2884 points. Substituting BankFind book equity over ending assets for the
regulatory tier-1 leverage ratio reports 11.0315% and 10.0264%, a 1.0050-point
gap, and raises the count below the threshold from 1 to 21 -- the two measures
are not interchangeable even though both are leverage ratios. Widening the CET1
population to the whole asset band and reading the threshold inclusively are both
numerically equivalent on this frozen snapshot, because no elector sits exactly
on the boundary and the non-electors are the only band members reporting CET1.
The query pins all four choices.

This is a medium baseline case. Numeric validation uses the shared rounding
tolerance; counts and labels match exactly.
"""

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


REPORT_DATE = "2025-09-30"
BAND_LOW_THOUSAND = 1_000_000
BAND_HIGH_THOUSAND = 10_000_000
CBLR_THRESHOLD_PCT = 9.0


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = [
    "linked_bank_count", "band_bank_count", "cblr_elector_count",
    "non_elector_count", "cblr_elector_share_pct",
]
TURN_2_NAMES = [
    "cblr_median_tier1_leverage_pct", "non_elector_median_tier1_leverage_pct",
    "cblr_below_threshold_count", "cblr_below_threshold_certificate",
    "cblr_below_threshold_name", "cblr_below_threshold_primary_regulator",
    "cblr_below_threshold_leverage_pct", "non_elector_below_threshold_count",
]
TURN_3_NAMES = [
    "non_elector_cet1_reported_count", "cblr_cet1_reported_count",
    "non_elector_median_cet1_ratio_pct", "lowest_cet1_certificate", "lowest_cet1_name",
    "lowest_cet1_ratio_pct", "lowest_cet1_reporting_basis",
    "lowest_cet1_standardized_rwa_usd_billions", "lowest_cet1_capital_usd_millions",
    "lowest_cet1_recomputed_ratio_pct",
]
TURN_4_NAMES = [
    "cblr_median_book_equity_to_assets_pct", "non_elector_median_book_equity_to_assets_pct",
    "book_below_leverage_bank_count", "largest_negative_divergence_certificate",
    "largest_negative_divergence_name",
    "largest_negative_divergence_book_equity_to_assets_pct",
    "largest_negative_divergence_tier1_leverage_pct",
]

variables = [
    _v(TURN_1_NAMES[0], "Store the count of 2025-09-30 Call Report capital rows linked to a BankFind row as an integer."),
    _v(TURN_1_NAMES[1], "Store the count of linked banks in the $1B-to-under-$10B BankFind total-asset band as an integer."),
    _v(TURN_1_NAMES[2], "Store the count of band banks with a CBLR election as an integer."),
    _v(TURN_1_NAMES[3], "Store the count of band banks without a CBLR election as an integer."),
    _v(TURN_1_NAMES[4], "Store CBLR electors as a percent of the band, rounded to 4 decimals."),
    _v(TURN_2_NAMES[0], "Store the median Call Report tier-1 leverage ratio of CBLR electors in percent, rounded to 4 decimals."),
    _v(TURN_2_NAMES[1], "Store the median Call Report tier-1 leverage ratio of non-electors in percent, rounded to 4 decimals."),
    _v(TURN_2_NAMES[2], "Store the count of CBLR electors with a tier-1 leverage ratio strictly below 9 percent as an integer."),
    _v(TURN_2_NAMES[3], "Store the FDIC certificate of the lowest-leverage CBLR elector as an unpadded string."),
    _v(TURN_2_NAMES[4], "Store that bank's BankFind name verbatim as a string, preserving capitalization, punctuation and spacing."),
    _v(TURN_2_NAMES[5], "Store that bank's NIC primary federal regulator code verbatim as a string."),
    _v(TURN_2_NAMES[6], "Store that bank's tier-1 leverage ratio in percent, rounded to 4 decimals."),
    _v(TURN_2_NAMES[7], "Store the count of non-electors with a tier-1 leverage ratio strictly below 9 percent as an integer."),
    _v(TURN_3_NAMES[0], "Store the count of non-electors reporting a standardized CET1 ratio as an integer."),
    _v(TURN_3_NAMES[1], "Store the count of CBLR electors reporting a standardized CET1 ratio as an integer."),
    _v(TURN_3_NAMES[2], "Store the median standardized CET1 ratio of non-electors in percent, rounded to 4 decimals."),
    _v(TURN_3_NAMES[3], "Store the FDIC certificate of the non-elector with the lowest standardized CET1 ratio as an unpadded string."),
    _v(TURN_3_NAMES[4], "Store that bank's BankFind name verbatim as a string, preserving capitalization, punctuation and spacing."),
    _v(TURN_3_NAMES[5], "Store that bank's reported standardized CET1 ratio in percent, rounded to 4 decimals."),
    _v(TURN_3_NAMES[6], "Store that bank's reported capital basis verbatim as a string (domestic or consolidated)."),
    _v(TURN_3_NAMES[7], "Store that bank's standardized risk-weighted assets in USD billions, rounded to 6 decimals."),
    _v(TURN_3_NAMES[8], "Store that bank's CET1 capital in USD millions, rounded to 6 decimals."),
    _v(TURN_3_NAMES[9], "Store CET1 capital divided by standardized RWA in percent, rounded to 4 decimals."),
    _v(TURN_4_NAMES[0], "Store the median BankFind book equity divided by total assets for CBLR electors, in percent, rounded to 4 decimals."),
    _v(TURN_4_NAMES[1], "Store the median BankFind book equity divided by total assets for non-electors, in percent, rounded to 4 decimals."),
    _v(TURN_4_NAMES[2], "Store the count of band banks whose book equity ratio is below their tier-1 leverage ratio as an integer."),
    _v(TURN_4_NAMES[3], "Store the FDIC certificate of the bank with the most negative book-minus-leverage divergence (ties by certificate ascending) as an unpadded string."),
    _v(TURN_4_NAMES[4], "Store that bank's BankFind name verbatim as a string, preserving capitalization, punctuation and spacing."),
    _v(TURN_4_NAMES[5], "Store that bank's book equity divided by total assets in percent, rounded to 4 decimals."),
    _v(TURN_4_NAMES[6], "Store that bank's tier-1 leverage ratio in percent, rounded to 4 decimals."),
]

DECIMALS = [
    0, 0, 0, 0, 4,
    4, 4, 0, None, None, None, 4, 0,
    0, 0, 4, None, None, 4, None, 6, 6, 4,
    4, 4, 0, None, None, 4, 4,
]


@lru_cache(maxsize=1)
def _linked():
    capital = load_expansion_table("ffiec_call_reports_capital").copy()
    capital = capital.loc[capital.report_date.astype(str).eq(REPORT_DATE)].copy()
    capital["bank_id"] = capital.bank_id.astype(str)
    identifiers = load_expansion_table("ffiec_nic_institutions")[[
        "rssd_id", "fdic_certificate", "primary_federal_regulator",
    ]].dropna(subset=["rssd_id", "fdic_certificate"]).copy()
    identifiers[["rssd_id", "fdic_certificate"]] = identifiers[["rssd_id", "fdic_certificate"]].astype(str)
    bankfind = load_expansion_table("fdic_bankfind").copy()
    bankfind = bankfind.loc[bankfind.REPDTE.astype(str).eq(REPORT_DATE)].copy()
    bankfind["CERT"] = bankfind.CERT.astype(str)
    linked = capital.merge(identifiers, left_on="bank_id", right_on="rssd_id", how="inner", validate="one_to_one")
    linked = linked.merge(bankfind[["CERT", "NAME", "ASSET", "EQ", "RBC1AAJ"]], left_on="fdic_certificate", right_on="CERT", how="inner", validate="one_to_one")
    linked["cblr"] = linked.cblr_election.astype("boolean").fillna(False)
    linked["leverage"] = pd.to_numeric(linked.reported_tier1_leverage_ratio_pct, errors="coerce")
    linked["cet1_ratio"] = pd.to_numeric(linked.reported_standardized_cet1_ratio_pct, errors="coerce")
    linked["asset_thousand"] = pd.to_numeric(linked.ASSET, errors="coerce")
    linked["book_equity_ratio"] = pd.to_numeric(linked.EQ, errors="coerce") / linked.asset_thousand * 100
    linked["book_minus_leverage"] = linked.book_equity_ratio - linked.leverage
    return linked


@lru_cache(maxsize=1)
def ground_truth():
    linked = _linked()
    band = linked.loc[linked.asset_thousand.ge(BAND_LOW_THOUSAND) & linked.asset_thousand.lt(BAND_HIGH_THOUSAND)].copy()
    electors = band.loc[band.cblr]
    non = band.loc[~band.cblr]
    if band.empty or electors.empty or non.empty:
        raise ValueError("unexpected CBLR band population")
    elector_median = float(electors.leverage.median())
    non_median = float(non.leverage.median())
    low_electors = electors.loc[electors.leverage.lt(CBLR_THRESHOLD_PCT)]
    lowest_elector = electors.sort_values(["leverage", "fdic_certificate"], ascending=[True, True]).iloc[0]

    cet1_non = non.loc[non.cet1_ratio.notna()]
    lowest_cet1 = cet1_non.sort_values(["cet1_ratio", "fdic_certificate"], ascending=[True, True]).iloc[0]
    rwa = float(lowest_cet1.standardized_risk_weighted_assets_thousand_usd)
    cet1_capital = float(lowest_cet1.common_equity_tier1_capital_thousand_usd)

    most_negative = band.sort_values(["book_minus_leverage", "fdic_certificate"], ascending=[True, True]).iloc[0]

    return (
        len(linked),
        len(band),
        len(electors),
        len(non),
        len(electors) / len(band) * 100,
        elector_median,
        non_median,
        len(low_electors),
        str(lowest_elector.fdic_certificate),
        str(lowest_elector.NAME),
        str(lowest_elector.primary_federal_regulator),
        float(lowest_elector.leverage),
        int(non.leverage.lt(CBLR_THRESHOLD_PCT).sum()),
        len(cet1_non),
        int(electors.cet1_ratio.notna().sum()),
        float(cet1_non.cet1_ratio.median()),
        str(lowest_cet1.fdic_certificate),
        str(lowest_cet1.NAME),
        float(lowest_cet1.cet1_ratio),
        str(lowest_cet1.capital_reporting_basis),
        rwa * 1000 / 1e9,
        cet1_capital * 1000 / 1e6,
        cet1_capital / rwa * 100,
        float(electors.book_equity_ratio.median()),
        float(non.book_equity_ratio.median()),
        int(band.book_minus_leverage.lt(0).sum()),
        str(most_negative.fdic_certificate),
        str(most_negative.NAME),
        float(most_negative.book_equity_ratio),
        float(most_negative.leverage),
    )


def _validate_subset(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip(by_name, ground_truth()))
    decimals = dict(zip(by_name, DECIMALS))
    return validate_ordered_outputs(outputs, [by_name[n] for n in names], [truth[n] for n in names], [decimals[n] for n in names])


def validate_turn_1(outputs): return _validate_subset(outputs, TURN_1_NAMES)
def validate_turn_2(outputs): return _validate_subset(outputs, TURN_2_NAMES)
def validate_turn_3(outputs): return _validate_subset(outputs, TURN_3_NAMES)
def validate_turn_4(outputs): return _validate_subset(outputs, TURN_4_NAMES)
def validate(outputs): return _validate_subset(outputs, [v.name for v in variables])


validators = {
    "validate_band_population": turn_validator(validate_turn_1),
    "validate_leverage_comparison": turn_validator(validate_turn_2),
    "validate_cet1_scope": turn_validator(validate_turn_3),
    "validate_book_equity_divergence": turn_validator(validate_turn_4),
}
