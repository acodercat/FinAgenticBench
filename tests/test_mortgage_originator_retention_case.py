"""Case-local gate for mortgage_originator_retention_and_purchase_limits.

Three things have to keep holding: the supervisory snapshot has to remain the
only way to split the lenders, the two eligibility cliffs have to stay in the
amount distribution with their ordering intact, and the population rules the
query spends words on have to keep changing the answer.
"""

import json

import pytest

from config import PROJECT_ROOT
from core.validation import numeric_equal
from evals.risk_screening import mortgage_originator_retention_and_purchase_limits as case
from scripts.audit_case_design import has_closed_text_representation


SPEC = json.loads(
    (PROJECT_ROOT / "evals/risk_screening/mortgage_originator_retention_and_purchase_limits.json").read_text()
)


def _truth():
    return dict(zip((variable.name for variable in case.variables), case.ground_truth()))


def _turns():
    return SPEC["conversations"][0]["turns"]


def test_anchor_values_are_stable():
    truth = _truth()
    assert truth["origination_count"] == 6191825
    assert truth["certificate_linked_lender_count"] == 1710
    assert truth["unlinked_lender_count"] == 3159
    assert truth["widest_buyer_gap_code"] == "2"
    assert truth["agency_purchased_count"] == 1320636
    assert truth["steepest_fall_band_floor"] == 1140000
    assert truth["steepest_fall_band_count"] == 1307
    assert truth["second_steepest_fall_band_floor"] == 760000
    assert truth["above_national_limit_count"] == 23117
    assert truth["largest_above_limit_state"] == "CA"
    assert round(truth["certificate_linked_sold_share_pct"], 4) == 34.6269
    assert round(truth["unlinked_sold_share_pct"], 4) == 68.4937
    assert round(truth["steepest_fall_pct"], 4) == 98.4698


@pytest.mark.parametrize("removed", ["hmda", "ffiec_nic_institutions"])
def test_removing_either_source_makes_the_case_impossible(removed, monkeypatch):
    """Strong source removal, enforced rather than asserted."""
    original = case.load_expansion_table

    def blocked(name):
        if name == removed:
            raise FileNotFoundError(f"removed source {name}")
        return original(name)

    case._originations.cache_clear()
    case._certificate_linked_identifiers.cache_clear()
    case.ground_truth.cache_clear()
    try:
        with monkeypatch.context() as context:
            context.setattr(case, "load_expansion_table", blocked)
            with pytest.raises(FileNotFoundError):
                case.ground_truth()
    finally:
        case._originations.cache_clear()
        case._certificate_linked_identifiers.cache_clear()
        case.ground_truth.cache_clear()


def test_the_snapshot_is_the_only_way_to_split_the_lenders():
    """Not decorative: it separates a quarter of the originations from the rest."""
    truth = _truth()
    originations = case._originations()
    linked = originations.certificate_linked
    assert 0.05 < float(linked.mean()) < 0.95, float(linked.mean())
    assert truth["certificate_linked_lender_count"] + truth["unlinked_lender_count"] == (
        originations.identifier.nunique()
    )
    # And the two groups behave differently, which is what the split buys.
    sold = originations.buyer.ne(case.NOT_SOLD)
    assert float(sold[~linked].mean()) > 1.5 * float(sold[linked].mean())


def test_the_two_population_rules_change_the_answer():
    """Originated-only, and exactly one certificate. Each clause must earn its words."""
    truth = _truth()
    records = case.load_expansion_table("hmda")
    everything = records.loc[:, ["lei", "purchaser_type", "action_taken"]].copy()
    del records
    originated = everything.action_taken.astype(str).eq(case.ORIGINATED)
    assert int((~originated).sum()) > 1_000_000
    # The buyer field is populated on records the filer bought, not only made.
    bought_rows = everything.loc[~originated, "purchaser_type"].astype(str)
    assert float(bought_rows.ne(case.NOT_SOLD).mean()) > 0.05, "the clause is now idle"

    institutions = case.load_expansion_table("ffiec_nic_institutions")
    frame = institutions.loc[institutions.lei.notna() & institutions.fdic_certificate.notna()]
    del institutions
    counts = frame.groupby(frame.lei.astype(str)).fdic_certificate.nunique()
    # The module discards identifiers carrying several certificates. On this
    # release nothing is discarded, which is why the query does not mention it.
    # If that ever changes the clause has to go back into the query.
    ambiguous = int(counts.gt(1).sum())
    assert ambiguous == 0, (
        f"{ambiguous} identifiers now carry several certificates; the discard rule has "
        "become load-bearing and the first turn must state it"
    )
    assert "carrying exactly one" not in _turns()[0]["query"]
    assert "ties to a deposit-insurance certificate" in _turns()[0]["query"]
    assert "action code 1" in _turns()[0]["query"]
    assert truth["origination_count"] == int(originated.sum())


def test_both_eligibility_cliffs_are_present_and_ordered():
    truth = _truth()
    originations = case._originations()
    agency = originations.loc[originations.buyer.isin(case.AGENCY_CODES)]
    amounts = agency.amount.dropna()
    counts = case._band_counts(amounts)
    ratios = (counts.shift(-1) / counts).dropna().loc[case.SEARCH_LOW:case.SEARCH_HIGH]
    ordered = ratios.nsmallest(3)
    assert int(ordered.index[0]) == truth["steepest_fall_band_floor"]
    assert int(ordered.index[1]) == truth["second_steepest_fall_band_floor"]
    # The ordering of the top two is the closest call the case makes; pin the gap.
    first, second = 100 * (1 - ordered.iloc[0]), 100 * (1 - ordered.iloc[1])
    assert round(float(first), 4) == 98.4698
    assert round(float(second), 4) == 97.8944
    assert 0.3 < abs(first - second) < 1.0, (first, second)
    # Both cliffs dwarf the third, so the pair is not an artefact of the window.
    assert 100 * (1 - ordered.iloc[2]) < 95.0
    # Neither band is a lone spike: its neighbours below are populated.
    for floor in (truth["steepest_fall_band_floor"], truth["second_steepest_fall_band_floor"]):
        assert counts.loc[floor] > 5 * counts.loc[floor + case.BAND]


def test_the_reported_amounts_are_not_already_banded():
    """The query says the amounts are off the grid; a grid would make the bands trivial."""
    originations = case._originations()
    amounts = originations.loc[
        originations.buyer.isin(case.AGENCY_CODES), "amount"].dropna()
    assert float((amounts % case.BAND == 0).mean()) < 0.01
    assert "not themselves on that grid" in _turns()[2]["query"]


def test_the_above_limit_set_is_geographically_concentrated():
    truth = _truth()
    originations = case._originations()
    agency = originations.loc[originations.buyer.isin(case.AGENCY_CODES)]
    above = agency.loc[agency.amount.gt(case.NATIONAL_LIMIT)]
    states = above.state_code.astype(str).value_counts()
    assert len(above) == truth["above_national_limit_count"]
    assert states.index[0] == truth["largest_above_limit_state"]
    assert float(states.iloc[0]) / float(states.iloc[1]) > 3.0, "the leading state no longer leads"
    assert float(states.head(6).sum()) / float(states.sum()) > 0.6
    # The residue above the high-cost ceiling is a real population, and the
    # overclaim that explains it has to keep naming multi-unit properties.
    ceiling = above.loc[above.amount.gt(1_149_825)]
    assert len(ceiling) > 1000, len(ceiling)
    joined = " ".join(SPEC["judge_reference"]["known_overclaims"]).lower()
    assert "multi-unit" in joined


def test_the_widest_buyer_gap_has_a_real_margin():
    truth = _truth()
    originations = case._originations()
    linked = originations.certificate_linked
    gaps = {}
    for code in originations.buyer.unique():
        if code == case.NOT_SOLD:
            continue
        taken = originations.buyer.eq(code)
        gaps[code] = 100.0 * (float(taken[~linked].mean()) - float(taken[linked].mean()))
    ranked = sorted(gaps.values(), reverse=True)
    assert round(ranked[0], 4) == round(truth["widest_buyer_gap_pp"], 4)
    assert ranked[0] - ranked[1] > 1.0, (ranked[0], ranked[1])
    assert "breaking a tie by the lower code" in _turns()[1]["query"]


def test_every_text_output_declares_a_closed_representation():
    truth = _truth()
    by_name = {variable.name: variable for variable in case.variables}
    text_outputs = [name for name, value in truth.items() if isinstance(value, str)]
    assert len(text_outputs) == 2
    for name in text_outputs:
        assert has_closed_text_representation(by_name[name].description), name


def test_stores_partition_the_module_variables():
    declared = [name for turn in _turns() for name in turn["stores"]]
    assert declared == [variable.name for variable in case.variables]
    assert len(declared) == len(set(declared))


def test_every_turn_validator_rejects_a_wrong_value():
    truth = _truth()
    for turn in _turns():
        checker = getattr(case, turn["validator"])
        outputs = {name: truth[name] for name in turn["stores"]}
        assert checker(dict(outputs)).success
        broken = dict(outputs)
        first = turn["stores"][0]
        broken[first] = (
            "__WRONG__" if isinstance(broken[first], str) else float(broken[first]) + 1_000.0
        )
        assert not checker(broken).success, turn["validator"]


def test_claim_contract_is_well_formed():
    for turn in _turns():
        for claim in turn["claims"]:
            if claim["mode"] == "conditional":
                assert claim["description"].startswith("If ")
            if claim["mode"] == "required":
                assert " does not " not in claim["description"].lower()
            if "overlap_waiver" in claim:
                assert claim["overlap_waiver"].strip()


def test_judge_reference_carries_the_full_contract():
    reference = SPEC["judge_reference"]
    assert set(reference) == {"source", "facts", "known_overclaims"}
    assert reference["source"].strip() and len(reference["known_overclaims"]) >= 5
    facts = " ".join(reference["facts"])
    truth = _truth()
    assert f"{truth['origination_count']:,}" in facts
    assert f"{truth['above_national_limit_count']:,}" in facts
    assert "97.8944" in facts and "98.4698" in facts
    # The caveat that names the second group honestly has to survive edits.
    joined = " ".join(reference["known_overclaims"]).lower()
    assert "nonbank" in joined and "credit union" in joined


def test_the_reported_fall_is_positive_like_every_text_that_names_it():
    """The query, the variable, the docstring and the judge reference all say "a fall of".

    An earlier draft returned the signed change instead, so a solver reporting the
    fall the way all four texts describe it was marked wrong and only the negated
    value passed. The sign now follows the prose.
    """
    truth = _truth()
    assert truth["steepest_fall_pct"] > 0
    assert numeric_equal(98.4698, truth["steepest_fall_pct"], decimals=4)
    assert not numeric_equal(-98.4698, truth["steepest_fall_pct"], decimals=4)

    by_name = {variable.name: variable for variable in case.variables}
    assert "fall" in by_name["steepest_fall_pct"].description.lower()
    facts = " ".join(SPEC["judge_reference"]["facts"])
    assert "a fall of 98.4698 percent" in facts


def test_the_buyer_code_tie_rule_orders_codes_as_numbers():
    """The codes are strings; "71" sorts before "8" lexicographically."""
    observed = ["0", "1", "2", "3", "4", "5", "6", "71", "72", "8", "9"]
    assert sorted(observed, key=int) == ["0", "1", "2", "3", "4", "5", "6", "8", "9", "71", "72"]
    assert sorted(["8", "71"]) == ["71", "8"], (
        "string codes no longer sort lexicographically, so this guard is moot"
    )
    assert "breaking a tie by the lower code" in _turns()[1]["query"]


def test_an_empty_band_would_not_be_skipped():
    """value_counts drops empty bands; the query compares adjacent grid bands."""
    originations = case._originations()
    agency = originations.loc[originations.buyer.isin(case.AGENCY_CODES)]
    counts = case._band_counts(agency.amount.dropna())
    floors = list(counts.index)
    assert floors == list(range(floors[0], floors[-1] + case.BAND, case.BAND))
    window = counts.loc[case.SEARCH_LOW:case.SEARCH_HIGH]
    assert int(window.eq(0).sum()) == 0, (
        "a band in the search window is now empty, so the query must say what "
        "dividing by it means"
    )


def test_the_state_tie_rule_is_stated_and_currently_idle():
    """The query breaks a tie alphabetically; no tie binds on this file."""
    originations = case._originations()
    agency = originations.loc[originations.buyer.isin(case.AGENCY_CODES)]
    above = agency.loc[agency.amount.gt(case.NATIONAL_LIMIT)]
    states = above.state_code.astype(str).value_counts()
    assert int(states.iloc[0]) != int(states.iloc[1]), (
        "the leading state is now tied, so the alphabetical rule has started to bind"
    )
    assert "breaking a tie by the alphabetically earlier code" in _turns()[3]["query"]


def test_every_turn_asks_for_an_interpretation():
    """The claims demand interpretation; the query has to ask for it."""
    for index, turn in enumerate(_turns()):
        assert turn["query"].rstrip().split(". ")[-1].startswith("Explain"), index
