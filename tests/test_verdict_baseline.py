from scripts.verdict_baseline import check_baseline


def test_every_validator_accepts_truth_and_rejects_single_output_mutations():
    report = check_baseline()
    assert report["case_count"] > 0
    assert report["probe_count"] > report["case_count"]
    assert report["failures"] == []
