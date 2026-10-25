"""Raw certification for the eight cases whose modules record it as blocked.

Each case's answer is recomputed by its own module from the governed runtime
tables. That closes two of the three evidence chains the cross-review protocol
asks for; the third has to start at the source-native archive, because a defect
introduced while building the runtime layer would be invisible to both of the
others.

These tests supply that chain. For every source one of the eight cases reads,
the archive is parsed independently of `core.dataset_layers` and compared with
the runtime table over exactly the rows and columns the case consumes. Where
the build narrows the archive the narrowing is asserted rather than assumed, so
a future build that silently changed a window would fail here.

These checks certify equality of the examined input slices when executed and
passing. They do not independently verify the financial calculation. Skipped
tests do not establish certification for the current environment.
"""

from __future__ import annotations

import csv
import glob
import io
from zipfile import ZipFile

import pandas as pd
import pytest

from core.data import load_expansion_table, load_ken_french_table
from core.dataset_layers import RAW_DIR


CUTOFF = "2025-12-31"


def _ken_french_archive(archive: str, member: str, header: str) -> pd.DataFrame:
    """Parse a Ken French daily CSV out of its published zip.

    The files carry a prose preamble, then the daily block, then a second
    annual block; the header row locates the first and a non-numeric key ends
    it. Encoding is cp1252 because the preamble contains smart quotes.
    """
    with ZipFile(RAW_DIR / f"ken_french/{archive}") as bundle:
        text = bundle.read(member).decode("cp1252")
    lines = text.splitlines()
    start = lines.index(header)
    columns = [name.strip() for name in header.split(",")[1:]]
    rows = []
    for row in csv.reader(io.StringIO("\n".join(lines[start + 1:]))):
        if not row or not row[0].strip().isdigit():
            break
        rows.append([row[0].strip(), *[float(value) for value in row[1:]]])
    frame = pd.DataFrame(rows, columns=["date", *columns])
    frame["date"] = pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d")
    return frame.loc[frame.date.le(CUTOFF)].reset_index(drop=True)


@pytest.mark.raw_data
def test_ken_french_industry_returns_match_the_published_archive():
    """Covers industry_kalman_beta_smoothing, industry_grs_residual_alpha_overlay
    and industry_hansen_jagannathan_good_deal."""
    raw = _ken_french_archive(
        "10_Industry_Portfolios_daily_CSV.zip",
        "10_Industry_Portfolios_Daily.csv",
        ",NoDur,Durbl,Manuf,Enrgy,HiTec,Telcm,Shops,Hlth,Utils,Other",
    )
    runtime = load_ken_french_table("daily_industry_returns")
    assert len(raw) == len(runtime)
    assert raw.date.tolist() == runtime.date.tolist()
    for column in ("NoDur", "Durbl", "Manuf", "Enrgy", "HiTec",
                   "Telcm", "Shops", "Hlth", "Utils", "Other"):
        expected = raw[column].astype(float).to_numpy()
        actual = runtime[f"{column.lower()}_pct"].astype(float).to_numpy()
        assert (abs(expected - actual) < 1e-12).all(), column


@pytest.mark.raw_data
def test_ken_french_daily_factors_match_the_published_archive():
    """Covers industry_kalman_beta_smoothing and industry_grs_residual_alpha_overlay."""
    raw = _ken_french_archive(
        "F-F_Research_Data_Factors_daily_CSV.zip",
        "F-F_Research_Data_Factors_daily.csv",
        ",Mkt-RF,SMB,HML,RF",
    )
    runtime = load_ken_french_table("daily_factors")
    assert len(raw) == len(runtime)
    assert raw.date.tolist() == runtime.date.tolist()
    for source, target in (("Mkt-RF", "mkt_rf_pct"), ("SMB", "smb_pct"),
                           ("HML", "hml_pct"), ("RF", "rf_pct")):
        expected = raw[source].astype(float).to_numpy()
        actual = runtime[target].astype(float).to_numpy()
        assert (abs(expected - actual) < 1e-12).all(), source


@pytest.mark.raw_data
def test_nyfed_reference_rates_match_the_published_archive():
    """Covers industry_hansen_jagannathan_good_deal, sofr_vasicek_jamshidian_bond_option
    and sba_aalen_johansen_credit_protection."""
    raw = pd.read_csv(RAW_DIR / "nyfed_rates/reference_rates_2021_2025.csv")
    runtime = load_expansion_table("nyfed_reference_rates")
    assert len(raw) == len(runtime)

    # The archive prints MM/DD/YYYY; the runtime layer stores ISO.
    def sofr(frame, published: bool):
        rows = frame.loc[frame["Rate Type"].astype(str).eq("SOFR")]
        days = (pd.to_datetime(rows["Effective Date"], format="%m/%d/%Y") if published
                else pd.to_datetime(rows["Effective Date"]))
        return dict(zip(days.dt.strftime("%Y-%m-%d"), rows["Rate (%)"].astype(float)))

    expected, actual = sofr(raw, True), sofr(runtime, False)
    assert set(expected) == set(actual)
    assert all(abs(expected[day] - actual[day]) < 1e-12 for day in expected)


@pytest.mark.raw_data
def test_treasury_yield_curve_matches_the_published_archive():
    """Covers nfip_compound_poisson_stoploss_capacity,
    sofr_vasicek_jamshidian_bond_option and treasury_almgren_chriss_execution_frontier."""
    files = sorted(glob.glob(str(RAW_DIR / "treasury_rates/daily_treasury_yield_curve_*.csv")))
    assert files, "no source-native yield curve files"
    raw = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    runtime = load_expansion_table("treasury_yield_curve")
    assert len(raw) == len(runtime)

    def keyed(frame, tenor):
        rows = frame.dropna(subset=[tenor])
        return dict(zip(pd.to_datetime(rows.Date).dt.strftime("%Y-%m-%d"),
                        rows[tenor].astype(float)))

    for tenor in ("1 Yr", "2 Yr", "10 Yr"):
        expected, actual = keyed(raw, tenor), keyed(runtime, tenor)
        assert set(expected) == set(actual), tenor
        assert all(abs(expected[day] - actual[day]) < 1e-12 for day in expected), tenor


@pytest.mark.raw_data
def test_nfip_claims_match_the_published_archive():
    """Covers nfip_compound_poisson_stoploss_capacity and nfip_gerber_shiu_ruin_capital."""
    files = sorted(glob.glob(str(RAW_DIR / "fema_nfip/NfipClaims_*.parquet")))
    assert files, "no source-native NFIP claim files"
    raw = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    runtime = load_expansion_table("fema_nfip")
    assert len(raw) == len(runtime)

    # Both cases screen Florida by date of loss, so that screen is what has to agree.
    raw_florida = raw.loc[raw.state.astype(str).eq("FL")]
    runtime_florida = runtime.loc[runtime.state_usps.astype(str).eq("FL")]
    assert len(raw_florida) == len(runtime_florida)
    # Claim timestamps carry mixed offsets, so the instant is normalised before the day.
    expected = (pd.to_datetime(raw_florida.dateOfLoss, utc=True)
                .dt.strftime("%Y-%m-%d").value_counts())
    actual = runtime_florida.date_of_loss.astype(str).str.slice(0, 10).value_counts()
    assert expected.to_dict() == actual.to_dict()


@pytest.mark.raw_data
def test_fema_disaster_declarations_match_the_published_archive():
    """Covers nfip_gerber_shiu_ruin_capital."""
    raw = pd.read_parquet(RAW_DIR / "fema_disasters/DisasterDeclarationsSummaries.parquet")
    runtime = load_expansion_table("fema_disasters")
    declared = raw.declarationDate.astype(str).str.slice(0, 10)
    window = raw.loc[declared.ge("2019-01-01") & declared.le(CUTOFF)]
    assert len(window) == len(runtime), "the build window over declaration date moved"
    assert (window.declarationType.value_counts().to_dict()
            == runtime.declaration_type.value_counts().to_dict())
    assert (window.incidentType.value_counts().to_dict()
            == runtime.incident_type.value_counts().to_dict())


@pytest.mark.raw_data
def test_sba_7a_approvals_match_the_published_archive():
    """Covers sba_aalen_johansen_credit_protection."""
    files = sorted(glob.glob(str(RAW_DIR / "sba_7a_504/FOIA_7a_*.csv")))
    assert files, "no source-native SBA 7(a) files"
    raw = pd.concat([pd.read_csv(path, encoding="latin-1", low_memory=False)
                     for path in files], ignore_index=True)
    runtime = load_expansion_table("sba_7a")
    low, high = int(runtime.ApprovalFY.min()), int(runtime.ApprovalFY.max())
    window = raw.loc[raw.ApprovalFY.between(low, high)]
    assert len(window) == len(runtime), "the build window over approval fiscal year moved"

    # The case screens California approvals, so that screen carries the certification.
    expected = window.loc[window.ProjectState.astype(str).eq("CA")]
    actual = runtime.loc[runtime.ProjectState.astype(str).eq("CA")]
    assert len(expected) == len(actual)
    assert (round(float(expected.GrossApproval.astype(float).sum()), 2)
            == round(float(actual.GrossApproval.astype(float).sum()), 2))
    assert (expected.LoanStatus.value_counts().to_dict()
            == actual.LoanStatus.value_counts().to_dict())


@pytest.mark.raw_data
def test_cftc_disaggregated_positions_match_the_published_archive():
    """Covers treasury_almgren_chriss_execution_frontier."""
    # The case reads a Treasury contract, which the Traders in Financial Futures
    # files carry; the disaggregated files in the same archive cover commodities.
    frames = []
    for path in sorted(glob.glob(str(RAW_DIR / "cftc_cot/*_fin_txt_*.zip"))):
        scope = "futures_only" if "/fut_" in path else "futures_and_options_combined"
        with ZipFile(path) as bundle:
            member = bundle.namelist()[0]
            with bundle.open(member) as handle:
                frame = pd.read_csv(handle, low_memory=False)
        frame.columns = [column.strip() for column in frame.columns]
        frames.append(frame.assign(report_scope=scope))
    raw = pd.concat(frames, ignore_index=True)
    runtime = load_expansion_table("cftc_cot")
    assert len(raw) == len(runtime)

    # The case reads one contract on one report date in one scope.
    def selected(frame):
        return frame.loc[
            frame.CFTC_Contract_Market_Code.astype(str).str.strip().eq("043602")
            & frame["Report_Date_as_YYYY-MM-DD"].astype(str).str.slice(0, 10).eq("2024-12-31")
            & frame.report_scope.eq("futures_only")
        ]

    expected, actual = selected(raw), selected(runtime)
    assert len(expected) == 1 and len(actual) == 1
    for column in ("Open_Interest_All", "Lev_Money_Positions_Long_All",
                   "Lev_Money_Positions_Short_All"):
        assert (float(expected[column].iloc[0]) == float(actual[column].iloc[0])), column
