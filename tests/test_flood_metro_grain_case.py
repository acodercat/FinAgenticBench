"""Case-local gate for county_flood_loss_metro_price_grain.

The case rests on three measured properties of the frozen data: the house price
panel has no county level, the mortgage register is the only county-to-metro
bridge, and the two extremes of the price path sit on the same coast. Each is
asserted here so a data refresh that dissolves one fails loudly rather than
quietly turning the case into an arithmetic exercise.
"""

import json

import pandas as pd

from config import PROJECT_ROOT
from core.data import load_expansion_table
from evals.risk_screening import county_flood_loss_metro_price_grain as case
from scripts.audit_case_design import has_closed_text_representation


SPEC = json.loads(
    (PROJECT_ROOT / "evals/risk_screening/county_flood_loss_metro_price_grain.json").read_text()
)


def _truth():
    return dict(zip((variable.name for variable in case.variables), case.ground_truth()))


def _turns():
    return SPEC["conversations"][0]["turns"]


def test_anchor_values_are_stable():
    truth = _truth()
    assert truth["county_claim_count"] == 102009
    assert truth["affected_county_count"] == 1193
    assert truth["metropolitan_county_count"] == 710
    assert truth["non_metropolitan_county_count"] == 472
    assert truth["unmatched_county_count"] == 11
    assert (
        truth["metropolitan_county_count"]
        + truth["non_metropolitan_county_count"]
        + truth["unmatched_county_count"]
        == truth["affected_county_count"]
    )
    assert truth["largest_loss_county_code"] == "12103"
    assert truth["largest_loss_metro_code"] == "41304"
    assert round(truth["total_net_paid_usd_millions"], 4) == 8798.2578
    assert round(truth["metropolitan_net_paid_share_pct"], 4) == 97.0167


def test_place_count_and_money_share_point_in_opposite_directions():
    """The financial distinction: a metro reading keeps the dollars and loses the places."""
    truth = _truth()
    outside_share = truth["non_metropolitan_county_count"] / truth["affected_county_count"]
    assert outside_share > 0.35, "non-metro counties are no longer a large minority of places"
    assert truth["metropolitan_net_paid_share_pct"] > 95.0


def test_price_extremes_diverge_on_the_same_coast():
    truth = _truth()
    assert truth["weakest_price_index_growth_pct"] < 0 < truth["strongest_price_index_growth_pct"]
    spread = truth["strongest_price_index_growth_pct"] - truth["weakest_price_index_growth_pct"]
    assert spread > 5.0, "the two extremes no longer disagree materially"
    panel = load_expansion_table("fhfa_hpi")
    names = panel.loc[panel.place_id.astype(str).isin(
        (truth["weakest_price_metro_code"], truth["strongest_price_metro_code"])
    ), "place_name"].astype(str).unique()
    assert all(name.rstrip().endswith("FL") or ", FL" in name for name in names), list(names)


def test_house_price_panel_has_no_county_level():
    """Without this the metropolitan bridge would be unnecessary and the case trivial."""
    panel = load_expansion_table("fhfa_hpi")
    assert "County" not in set(panel.level.astype(str))
    counties = load_expansion_table("census_counties")
    assert not [column for column in counties.columns if "cbsa" in column.lower()]


def test_mortgage_register_assigns_each_county_one_metropolitan_code():
    bridge = case._county_to_metro()
    assert bridge.index.is_unique
    register = load_expansion_table("hmda")
    frame = register.loc[:, ["county_code", "derived_msa_md"]].copy()
    del register
    frame["county_code"] = frame.county_code.astype(str)
    frame = frame.loc[frame.county_code.str.len().eq(5)]
    frame["derived_msa_md"] = frame.derived_msa_md.astype(str)
    distinct = frame.groupby("county_code").derived_msa_md.nunique()
    assert int(distinct.gt(1).sum()) == 0, "a county now reports more than one metropolitan code"


def test_both_readings_of_index_coverage_agree():
    """"Covered" could mean present in the panel or present at both pinned quarters.

    The query says only that the index covers an area; the two quarters arrive a
    turn later. Both readings currently give the same count and the same leading
    five, so the query needs no extra clause. If a refresh ever separates them,
    this fires and the query has to pin which one counts.
    """
    truth = _truth()
    levels = case._metro_index()
    claims = case._claims()
    by_county = claims.groupby("county_code").net_paid.sum()
    bridge = case._county_to_metro()
    metro_of = pd.Series(by_county.index.map(bridge), index=by_county.index, dtype="object")
    metropolitan = ~metro_of.isna() & ~metro_of.eq(case.NON_METRO_SENTINEL)
    by_metro = (
        by_county.loc[metropolitan]
        .groupby(metro_of.loc[metropolitan])
        .sum()
        .sort_values(ascending=False)
    )
    anywhere = [code for code in by_metro.index if code in levels.index.get_level_values(0)]
    endpoints = [
        code for code in by_metro.index
        if (code, case.BASE_YEAR, case.BASE_QUARTER) in levels.index
        and (code, case.FINAL_YEAR, case.FINAL_QUARTER) in levels.index
    ]
    assert len(anywhere) == len(endpoints) == truth["indexed_metro_area_count"]
    assert anywhere[:5] == endpoints[:5]


def test_no_state_rollup_code_hides_in_the_population():
    """The query asks only for a five-digit county, so the module must filter only that.

    A code ending in three zeros would be a state-level rollup rather than a
    county. The clause naming them was removed from the query after measuring
    that it changed no output, and the module was aligned with the query. If one
    ever appears, this fires instead of the module quietly applying an unstated
    rule.
    """
    claims = case._claims()
    rollups = claims.county_code.str.endswith("000")
    assert not rollups.any(), sorted(claims.loc[rollups].county_code.unique())


def test_every_text_output_declares_a_closed_representation():
    truth = _truth()
    text_outputs = [name for name, value in truth.items() if isinstance(value, str)]
    assert text_outputs
    by_name = {variable.name: variable for variable in case.variables}
    for name in text_outputs:
        assert has_closed_text_representation(by_name[name].description), name


def test_public_contract_pins_the_loss_basis_and_the_index_series():
    turns = _turns()
    assert len(turns) == 4
    assert SPEC["case_type"] == "baseline"
    first = turns[0]["query"]
    assert "date of loss falls in 2024" in first
    assert "keeping the reported sign" in first
    # Screening on the open year instead moves the total; the query must pin it.
    assert "opened in a calendar year later than their loss year" in first
    third = turns[2]["query"]
    assert "all-transactions" in third and "not seasonally adjusted" in third
    # The state-rollup clause was removed after measuring that it changed nothing.
    assert "three zeros" not in first


def test_open_year_screen_is_a_different_population():
    claims = load_expansion_table("fema_nfip")
    opened = claims.loc[claims.open_date.astype(str).str.slice(0, 4).eq("2024")]
    loss_year = claims.loc[claims.date_of_loss.astype(str).str.slice(0, 4).eq("2024")]
    del claims
    assert len(opened) != len(loss_year)
    truth = _truth()
    assert truth["loss_year_claims_opened_later_count"] > 0


def test_stores_partition_the_module_variables():
    declared = [name for turn in _turns() for name in turn["stores"]]
    assert declared == [variable.name for variable in case.variables]
    assert len(declared) == len(set(declared))


def test_every_turn_validator_rejects_a_wrong_value():
    truth = _truth()
    for turn in _turns():
        checker = getattr(case, turn["validator"].replace("validate_", "validate_", 1))
        outputs = {name: truth[name] for name in turn["stores"]}
        assert checker(dict(outputs)).success
        broken = dict(outputs)
        first = turn["stores"][0]
        broken[first] = (
            "__WRONG__" if isinstance(broken[first], str) else float(broken[first]) + 1_000_000.0
        )
        assert not checker(broken).success, turn["validator"]


def test_claim_contract_is_well_formed():
    for turn in _turns():
        for claim in turn["claims"]:
            if claim["mode"] == "conditional":
                assert claim["description"].startswith("If ")
            if claim["mode"] == "required":
                assert " does not " not in claim["description"].lower()


def test_judge_reference_quotes_recomputed_figures():
    truth = _truth()
    facts = " ".join(SPEC["judge_reference"]["facts"])
    assert f"{truth['affected_county_count']:,}" in facts
    assert f"{truth['total_net_paid_usd_millions']:,.4f}" in facts
    assert f"{truth['metropolitan_net_paid_share_pct']:.4f}" in facts
    assert truth["largest_loss_metro_code"] in facts


def test_index_growth_uses_the_pinned_quarters():
    levels = case._metro_index()
    code = _truth()["weakest_price_metro_code"]
    base = float(levels[(code, case.BASE_YEAR, case.BASE_QUARTER)])
    final = float(levels[(code, case.FINAL_YEAR, case.FINAL_QUARTER)])
    assert pd.notna(base) and pd.notna(final)
    assert round(100.0 * (final / base - 1.0), 4) == round(
        _truth()["weakest_price_index_growth_pct"], 4
    )
