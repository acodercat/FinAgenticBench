"""Case-local gate for nfip_coverage_exhaustion_and_icc_component_split.

Both trap axes have to keep separating.  The building limit must stay consumable
only by the building payment, so that folding in the other two coverages still
reports a different count and a different total; and the cause column must keep
storing one code under two numeric spellings alongside letter codes, so that both
the as-recorded reading and the numeric coercion still miss.  Every assertion
here is written to be capable of failing: where a property is currently vacuous
the test says so in its message rather than passing quietly.
"""

import json

import pandas as pd
import pytest

from config import PROJECT_ROOT
from core.data import load_expansion_table
from core.validation import numeric_equal
from evals.calculation_reconciliation import (
    nfip_coverage_exhaustion_and_icc_component_split as case,
)


SPEC = json.loads(
    (PROJECT_ROOT
     / "evals/calculation_reconciliation/nfip_coverage_exhaustion_and_icc_component_split.json"
     ).read_text()
)

GIVEAWAYS = (
    "building payment only", "only the building", "separate pot", "additional coverage",
    "does not count", "not count toward", "exclude the contents", "exclude the compliance",
    "numeric formatting", "decimal point", "same code", "spelling", "as a string",
    # A singular, definite "the payment" implies one payment per coverage, which is the
    # distinction the second turn is measuring.  An earlier draft leaked it this way.
    "the payment drawn", "drawn against it", "payment against the coverage",
    "one payment", "its own payment",
)


def _truth():
    return dict(zip((variable.name for variable in case.variables), case.ground_truth()))


def _turns():
    return SPEC["conversations"][0]["turns"]


@pytest.fixture(scope="module")
def claims():
    frame = load_expansion_table("fema_nfip")
    numeric = {
        column: pd.to_numeric(frame[column], errors="coerce")
        for column in ("gross_building_payment_usd", "gross_contents_payment_usd",
                       "gross_icc_payment_usd", "building_coverage_usd")
    }
    numeric["cause"] = frame.cause_of_damage
    return pd.DataFrame(numeric)


def test_anchor_values_are_stable():
    truth = _truth()
    assert truth["claim_record_count"] == 312322
    assert truth["cause_coded_claim_count"] == 308230
    assert truth["distinct_cause_code_count"] == 12
    assert truth["leading_cause_claim_count"] == 145913
    assert truth["positive_building_coverage_claim_count"] == 305292
    assert truth["building_coverage_exhausted_claim_count"] == 11934
    assert truth["exhausted_gross_building_payment_usd_millions"] == 2233.450134
    assert truth["exhausted_at_maximum_residential_coverage_claim_count"] == 4686
    assert truth["icc_paid_claim_count"] == 1402
    assert truth["icc_at_statutory_cap_claim_count"] == 637
    assert truth["icc_gross_payment_usd_millions"] == 33.158440
    assert truth["exhausted_claim_icc_recipient_count"] == 229
    assert truth["noaa_matched_county_month_count"] == 6855
    assert truth["noaa_matched_claim_count"] == 138621
    assert truth["unmatched_claim_gross_payment_usd_millions"] == 13818.473127
    assert truth["leading_noaa_flood_cause_county_month_count"] == 6264


def test_folding_the_other_coverages_still_misses_the_limit(claims):
    """Trap axis one, on both of the turn's measured outputs."""
    truth = _truth()
    coverage = claims.building_coverage_usd
    building = claims.gross_building_payment_usd
    contents = claims.gross_contents_payment_usd.fillna(0.0)
    compliance = claims.gross_icc_payment_usd.fillna(0.0)
    insured = coverage.gt(0)

    pinned = truth["building_coverage_exhausted_claim_count"]
    folded = {
        "compliance": int((insured & building.fillna(0.0).add(compliance).ge(coverage)).sum()),
        "contents": int((insured & building.fillna(0.0).add(contents).ge(coverage)).sum()),
        "both": int((insured & building.fillna(0.0).add(contents).add(compliance).ge(coverage)).sum()),
    }
    assert folded == {"compliance": 12097, "contents": 18137, "both": 18308}
    for label, count in folded.items():
        assert count != pinned, label
        # Counts are compared for exact equality, so any difference is excluded.
        assert not numeric_equal(count, pinned, decimals=0), label

    folded_payment = float(
        building.fillna(0.0).add(compliance)[
            insured & building.fillna(0.0).add(compliance).ge(coverage)
        ].sum()
    ) / 1e6
    assert round(folded_payment, 6) == 2262.423430
    assert not numeric_equal(
        round(folded_payment, 6),
        truth["exhausted_gross_building_payment_usd_millions"],
        decimals=6,
    )


def test_reaching_the_limit_is_not_the_same_as_passing_it(claims):
    """The query pins 'reaches'; 'passes' would report a different population."""
    truth = _truth()
    insured = claims.building_coverage_usd.gt(0)
    passing = int((insured & claims.gross_building_payment_usd.gt(claims.building_coverage_usd)).sum())
    settling = int((insured & claims.gross_building_payment_usd.eq(claims.building_coverage_usd)).sum())
    assert passing == 794
    assert settling == 11140
    assert passing + settling == truth["building_coverage_exhausted_claim_count"]
    assert settling > passing, "the paid-to-limit signature the second claim asserts is gone"
    assert "reaches that amount" in _turns()[1]["query"]


def test_both_naive_cause_readings_still_miss(claims):
    """Trap axis two: as-recorded over-counts, numeric coercion drops the letters."""
    truth = _truth()
    recorded = claims.cause.map(
        lambda value: None
        if value is None or value is pd.NA or (isinstance(value, float) and value != value)
        or not str(value).strip()
        else str(value).strip()
    )
    coerced = pd.to_numeric(claims.cause, errors="coerce")

    assert int(recorded.nunique()) == 19
    assert int(coerced.nunique()) == 8
    assert truth["distinct_cause_code_count"] == 12
    for naive in (int(recorded.nunique()), int(coerced.nunique())):
        assert not numeric_equal(naive, truth["distinct_cause_code_count"], decimals=0)

    assert int(recorded.value_counts().iloc[0]) == 133349
    assert not numeric_equal(133349, truth["leading_cause_claim_count"], decimals=0)

    # The two halves of the axis must both still be present in the data.
    canonical = claims.cause.map(case.canonical_cause_code)
    dual = {code for code in canonical.dropna().unique()
            if f"{code}.0" in set(recorded.dropna())}
    assert dual, "no code is stored under two numeric spellings any more; the axis is gone"
    letters = [code for code in canonical.dropna().unique() if not code.lstrip("-").isdigit()]
    assert sorted(letters) == ["A", "B", "C", "D"]
    assert int(canonical.isin(letters).sum()) == 702


def test_compliance_coverage_pays_past_an_exhausted_building_limit(claims):
    """The third turn's payoff: the required claim has to stay true of the data."""
    truth = _truth()
    insured = claims.building_coverage_usd.gt(0)
    exhausted = insured & claims.gross_building_payment_usd.ge(claims.building_coverage_usd)
    recipients = exhausted & claims.gross_icc_payment_usd.gt(0)
    assert int(recipients.sum()) == truth["exhausted_claim_icc_recipient_count"] > 0, (
        "no exhausted claim receives compliance coverage any more, so the turn's required "
        "claim is no longer supported by the file"
    )
    assert round(float(claims.gross_icc_payment_usd[recipients].sum()) / 1e6, 6) == 4.936133
    above_cap = int(claims.gross_icc_payment_usd.gt(case.ICC_STATUTORY_CAP_USD).sum())
    assert above_cap == 5, (
        "the conditional claim says a handful of rows exceed the cap; if this reaches zero "
        "that claim must be withdrawn"
    )


def test_the_county_granularity_filter_is_a_guard_and_currently_removes_nothing():
    """The query no longer claims this filter does anything; the code keeps it as a guard.

    If this ever stops being vacuous the query must say so again, because the
    pairing would then silently drop events the reader was not told about.
    """
    storms = load_expansion_table("noaa_storm_events")
    flood = storms.flood_cause.notna() & storms.flood_cause.astype(str).str.strip().ne("")
    assert int(flood.sum()) == 49617
    zoned = flood & storms.county_zone_type.astype(str).str.strip().str.upper().ne(
        case.COUNTY_GRANULARITY_TOKEN
    )
    assert int(zoned.sum()) == 0, (
        "flood-cause rows are no longer all placed on a county, so the fourth turn's query "
        "must state the county-granularity restriction it now depends on"
    )
    # The only rows the event frame loses are the ones the state restriction removes.
    state = storms.state_usps.astype(str).str.strip().str.upper().isin(case.STATE_USPS)
    assert len(case._county_flood_events()) == int((flood & ~zoned & state).sum())
    assert "placed on a county" not in _turns()[3]["query"]


def test_coastal_and_tropical_flooding_never_enters_the_pairing():
    """The required claim the previous version got backwards, pinned to the data."""
    storms = load_expansion_table("noaa_storm_events")
    flood = storms.flood_cause.notna() & storms.flood_cause.astype(str).str.strip().ne("")
    assert sorted(storms.event_type[flood].unique()) == ["Debris Flow", "Flash Flood", "Flood"]

    water = storms.event_type.astype(str).str.lower().str.contains(
        "flood|surge|hurricane|tropical|tsunami|seiche"
    )
    uncaused = water & ~flood
    assert int(uncaused.sum()) == 5485, (
        "the population the required claim describes has changed size"
    )
    zoned = storms.county_zone_type.astype(str).str.strip().str.upper().ne(
        case.COUNTY_GRANULARITY_TOKEN
    )
    assert int((uncaused & ~zoned).sum()) == 0, (
        "some coastal or tropical row now sits on a county, so the claim that they are "
        "filed against broad warning areas no longer holds"
    )
    labels = storms.event_type[uncaused].value_counts()
    assert labels.get("Tropical Storm") == 2143
    assert labels.get("Coastal Flood") == 1709


@pytest.mark.parametrize("removed", ["fema_nfip", "noaa_storm_events"])
def test_both_sources_are_load_bearing(monkeypatch, removed):
    original = case.load_expansion_table

    def without(name):
        if name == removed:
            raise FileNotFoundError("removed source")
        return original(name)

    case.ground_truth.cache_clear()
    try:
        with monkeypatch.context() as ctx:
            ctx.setattr(case, "load_expansion_table", without)
            with pytest.raises(FileNotFoundError):
                case.ground_truth()
    finally:
        case.ground_truth.cache_clear()


def test_the_query_withholds_both_distinctions():
    """P5: a trap whose query states the avoidance is a baseline."""
    assert SPEC["case_type"] == "trap"
    trapped = " ".join(turn["query"] for turn in _turns()[:2]).lower()
    for giveaway in GIVEAWAYS:
        assert giveaway not in trapped, giveaway


def test_no_variable_description_states_either_avoidance():
    """Variable descriptions are a prompt surface too."""
    trapped = set(_turns()[0]["stores"]) | set(_turns()[1]["stores"])
    for variable in case.variables:
        if variable.name not in trapped:
            continue
        lowered = variable.description.lower()
        for giveaway in GIVEAWAYS:
            assert giveaway not in lowered, (variable.name, giveaway)


def test_stores_partition_the_module_variables():
    declared = [name for turn in _turns() for name in turn["stores"]]
    assert declared == [variable.name for variable in case.variables]
    assert len(declared) == len(set(declared))


def test_every_turn_validator_rejects_a_wrong_value():
    truth = _truth()
    for turn in _turns():
        checker = case.validators[turn["validator"]]
        outputs = {name: truth[name] for name in turn["stores"]}
        assert case._validate_subset(outputs, turn["stores"]).success
        for name in turn["stores"]:
            broken = dict(outputs)
            broken[name] = float(broken[name]) + 1.0
            assert not case._validate_subset(broken, turn["stores"]).success, (turn["validator"], name)
        assert checker is not None


def test_claim_contract_is_well_formed():
    for turn in _turns():
        assert turn["claims"]
        for claim in turn["claims"]:
            if claim["mode"] == "conditional":
                assert claim["description"].startswith("If ")
            if claim["mode"] == "required":
                assert " does not " not in claim["description"].lower()
            if "overlap_waiver" in claim:
                assert claim["overlap_waiver"].strip()


def test_judge_reference_quotes_recomputed_figures():
    truth = _truth()
    facts = " ".join(SPEC["judge_reference"]["facts"])
    for name in ("claim_record_count", "building_coverage_exhausted_claim_count",
                 "leading_cause_claim_count", "noaa_matched_claim_count",
                 "exhausted_claim_icc_recipient_count"):
        assert f"{truth[name]:,}" in facts, name
    assert f"{truth['exhausted_gross_building_payment_usd_millions']:,.6f}" in facts
    assert "12,097" in facts and "18,308" in facts
    assert set(SPEC["judge_reference"]) == {"source", "facts", "known_overclaims"}
    assert SPEC["judge_reference"]["known_overclaims"]


def test_the_unrecorded_county_rule_is_stated_and_material(claims):
    """683 rows can never be paired; which side they fall on moves the total."""
    truth = _truth()
    frame = case._claims()
    unplaced = frame.county.isna()
    assert int(unplaced.sum()) == 683
    payments = (claims.gross_building_payment_usd.fillna(0.0)
                + claims.gross_contents_payment_usd.fillna(0.0)
                + claims.gross_icc_payment_usd.fillna(0.0))
    carried = round(float(payments[unplaced.to_numpy()].sum()) / 1e6, 6)
    assert carried == 21.153562
    dropped = round(truth["unmatched_claim_gross_payment_usd_millions"] - carried, 6)
    assert not numeric_equal(dropped, truth["unmatched_claim_gross_payment_usd_millions"], decimals=6)
    assert "county is blank falls outside the pairing" in _turns()[3]["query"], (
        "the rule these rows follow is no longer stated, so the turn has two readings"
    )


def test_the_cause_count_survives_every_reasonable_normalisation(claims):
    """P1: one answer, several valid routes to it; only the naive readings miss."""
    truth = _truth()
    recorded = claims.cause.map(
        lambda value: None
        if value is None or value is pd.NA or (isinstance(value, float) and value != value)
        or not str(value).strip()
        else str(value).strip()
    )

    def strip_suffix(text):
        return text[:-2] if isinstance(text, str) and text.endswith(".0") else text

    routes = {
        "integer cast": truth["distinct_cause_code_count"],
        "strip a trailing point zero": int(recorded.map(strip_suffix).nunique()),
        "casefold then strip": int(
            recorded.map(lambda t: t.lower() if isinstance(t, str) else t).map(strip_suffix).nunique()
        ),
    }
    assert set(routes.values()) == {12}, routes
    assert int(recorded.nunique()) == 19
    assert int(pd.to_numeric(claims.cause, errors="coerce").nunique()) == 8


def test_the_payment_basis_and_boundary_are_jointly_unique(claims):
    """The full sweep: exactly one of sixteen readings reproduces the pinned pair."""
    truth = _truth()
    coverage = claims.building_coverage_usd
    building = claims.gross_building_payment_usd.fillna(0.0)
    bases = {
        "building": building,
        "icc": building + claims.gross_icc_payment_usd.fillna(0.0),
        "contents": building + claims.gross_contents_payment_usd.fillna(0.0),
        "both": building + claims.gross_contents_payment_usd.fillna(0.0)
        + claims.gross_icc_payment_usd.fillna(0.0),
    }
    pinned = (truth["building_coverage_exhausted_claim_count"],
              truth["exhausted_gross_building_payment_usd_millions"])
    reproducing = 0
    for payment in bases.values():
        for reached in (payment.ge(coverage), payment.gt(coverage)):
            for scope in (coverage.gt(0), coverage.notna()):
                selected = scope & reached
                candidate = (int(selected.sum()),
                             round(float(claims.gross_building_payment_usd[selected].sum()) / 1e6, 6))
                reproducing += candidate == pinned
    assert reproducing == 1, reproducing


def test_the_state_restriction_settles_the_reading_without_moving_the_answer():
    """The storm file numbers the territories outside the standard range.

    Those rows can never meet a claim county as recorded, so restricting the storm
    side costs nothing -- but without the restriction a reader who repaired the
    codes would get a different answer for every one of the fourth turn's outputs,
    which is why the query pins it.
    """
    truth = _truth()
    storms = load_expansion_table("noaa_storm_events")
    flood = storms.flood_cause.notna() & storms.flood_cause.astype(str).str.strip().ne("")
    county = storms.county_zone_type.astype(str).str.strip().str.upper().eq(
        case.COUNTY_GRANULARITY_TOKEN
    )
    state = storms.state_usps.astype(str).str.strip().str.upper().isin(case.STATE_USPS)
    assert int((flood & county).sum()) == 49617
    assert int((flood & county & ~state).sum()) == 1766
    assert len(case._county_flood_events()) == 47851

    outside = storms.state_name[flood & county & ~state].value_counts()
    assert outside.get("PUERTO RICO") == 1655
    numbering = pd.to_numeric(storms.state_fips, errors="coerce")[flood & county & ~state]
    assert sorted(numbering.unique()) == [96, 97, 98, 99], (
        "the territories are no longer numbered outside the standard range, so the "
        "restriction may have stopped being what settles the reading"
    )

    # Repairing the numbering instead would move every fourth-turn output.
    repaired = {"99": "72", "97": "60", "98": "66", "96": "78"}
    events = pd.DataFrame({
        "cause": storms.flood_cause.map(case._flood_cause_label),
        "county": case._digits(storms.state_fips, 2) + case._digits(storms.county_zone_fips, 3),
        "month": pd.to_datetime(storms.begin_datetime, errors="coerce").dt.to_period("M").astype(str),
    }).loc[flood & county]
    events["county"] = events.county.map(lambda code: repaired.get(code[:2], code[:2]) + code[2:])

    claims = case._claims()
    claim_months = set(zip(claims.county, claims.month))
    shared = claim_months & set(zip(events.county, events.month))
    assert len(shared) == 6877
    assert len(shared) != truth["noaa_matched_county_month_count"], (
        "the two readings now agree, so the state restriction no longer settles anything"
    )
    matched = pd.Series([key in shared for key in zip(claims.county, claims.month)],
                        index=claims.index)
    assert int(matched.sum()) == 138693
    assert int(matched.sum()) != truth["noaa_matched_claim_count"]
    assert "fifty states or the District of Columbia" in _turns()[3]["query"]
