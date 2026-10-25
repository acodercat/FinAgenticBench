"""Build governed inputs for the eleventh cross-source batch.

The projections make four source-native formats usable inside the runtime while
preserving their analytical grain.  NOAA damage suffixes are decoded without
turning missing amounts into zero; NFIP claim records retain their reported
``policyCount`` and signed net-payment fields; BEA nonnumeric source markers
remain explicit without assigning meanings absent from the frozen artifact; and
Census end-use rows are not pre-aggregated to country
answers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from config import PROJECT_ROOT


DATASETS = PROJECT_ROOT / "datasets_build"   # reviewed staging inputs
RAW_DIR = PROJECT_ROOT / "datasets_raw"      # source-native archives
GZIP = {"method": "gzip", "compresslevel": 6, "mtime": 0}
STATE_FIPS_TO_USPS = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA",
    "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL",
    "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN",
    "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME",
    "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS",
    "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
    "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
    "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI",
    "56": "WY",
}


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, na_rep="", compression=GZIP)


def _damage_usd(value: object) -> float | None:
    if pd.isna(value) or not str(value).strip():
        return None
    match = re.fullmatch(
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))([KMB]?)",
        str(value).strip().upper(),
    )
    if match is None:
        raise ValueError(f"unrecognized Storm Events damage token: {value!r}")
    return float(match.group(1)) * {
        "": 1.0,
        "K": 1_000.0,
        "M": 1_000_000.0,
        "B": 1_000_000_000.0,
    }[match.group(2)]


def build_noaa_storm_details() -> dict[str, object]:
    root = DATASETS / "noaa_storm_events"
    source_path = next(
        (RAW_DIR / "census_international_trade").glob("StormEvents_details-ftp_v1.0_d2024_*.csv.gz")
    )
    source = pd.read_csv(
        source_path,
        low_memory=False,
        dtype={"STATE_FIPS": "string", "CZ_FIPS": "string"},
    )
    fields = [
        "EPISODE_ID", "EVENT_ID", "STATE", "STATE_FIPS", "EVENT_TYPE",
        "CZ_TYPE", "CZ_FIPS", "CZ_NAME", "BEGIN_DATE_TIME", "END_DATE_TIME",
        "DAMAGE_PROPERTY", "DAMAGE_CROPS", "FLOOD_CAUSE", "DATA_SOURCE",
    ]
    result = source[fields].rename(
        columns={
            "EPISODE_ID": "episode_id",
            "EVENT_ID": "event_id",
            "STATE": "state_name",
            "STATE_FIPS": "state_fips",
            "EVENT_TYPE": "event_type",
            "CZ_TYPE": "county_zone_type",
            "CZ_FIPS": "county_zone_fips",
            "CZ_NAME": "county_zone_name",
            "BEGIN_DATE_TIME": "begin_datetime",
            "END_DATE_TIME": "end_datetime",
            "DAMAGE_PROPERTY": "property_damage_source_token",
            "DAMAGE_CROPS": "crop_damage_source_token",
            "FLOOD_CAUSE": "flood_cause",
            "DATA_SOURCE": "data_source",
        }
    )
    result["state_fips"] = result["state_fips"].str.zfill(2)
    result.insert(3, "state_usps", result["state_fips"].map(STATE_FIPS_TO_USPS))
    result["begin_datetime"] = pd.to_datetime(
        result["begin_datetime"], format="%d-%b-%y %H:%M:%S"
    ).dt.strftime("%Y-%m-%dT%H:%M:%S")
    result["end_datetime"] = pd.to_datetime(
        result["end_datetime"], format="%d-%b-%y %H:%M:%S"
    ).dt.strftime("%Y-%m-%dT%H:%M:%S")
    result.insert(
        result.columns.get_loc("property_damage_source_token") + 1,
        "property_damage_usd",
        result["property_damage_source_token"].map(_damage_usd),
    )
    result.insert(
        result.columns.get_loc("crop_damage_source_token") + 1,
        "crop_damage_usd",
        result["crop_damage_source_token"].map(_damage_usd),
    )
    result["source_file"] = source_path.name
    result = result.sort_values("event_id").reset_index(drop=True)
    if len(result) != 69_801 or result["event_id"].duplicated().any():
        raise ValueError("unexpected 2024 NOAA Storm Events detail population")
    if result.loc[result.state_usps.notna(), "state_usps"].nunique() != 51:
        raise ValueError("NOAA state/DC geography bridge is incomplete")
    output = root / "storm_event_details_2024.csv.gz"
    _write(result, output)
    return {
        "rows": len(result),
        "state_dc_rows": int(result.state_usps.notna().sum()),
        "reported_property_damage_rows": int(result.property_damage_usd.notna().sum()),
        "output": str(output.relative_to(DATASETS)),
        "source_file": source_path.name,
    }


def build_nfip_claims() -> dict[str, object]:
    root = DATASETS / "fema_nfip"
    source_path = RAW_DIR / "census_international_trade" / "NfipClaims_2024.parquet"
    source = pd.read_parquet(source_path)
    fields = [
        "id", "asOfDate", "dateOfLoss", "openDate", "state", "countyCode",
        "censusGeoid", "policyCount", "floodEvent", "causeOfDamage",
        "amountPaidOnBuildingClaim", "amountPaidOnContentsClaim",
        "amountPaidOnIncreasedCostOfComplianceClaim",
        "netBuildingPaymentAmount", "netContentsPaymentAmount",
        "netIccPaymentAmount", "totalBuildingInsuranceCoverage",
        "totalContentsInsuranceCoverage",
    ]
    result = source[fields].rename(
        columns={
            "id": "claim_record_id",
            "asOfDate": "snapshot_as_of_date",
            "dateOfLoss": "date_of_loss",
            "openDate": "open_date",
            "state": "state_usps",
            "countyCode": "county_code",
            "censusGeoid": "census_geoid",
            "policyCount": "policy_count",
            "floodEvent": "flood_event",
            "causeOfDamage": "cause_of_damage",
            "amountPaidOnBuildingClaim": "gross_building_payment_usd",
            "amountPaidOnContentsClaim": "gross_contents_payment_usd",
            "amountPaidOnIncreasedCostOfComplianceClaim": "gross_icc_payment_usd",
            "netBuildingPaymentAmount": "net_building_payment_usd",
            "netContentsPaymentAmount": "net_contents_payment_usd",
            "netIccPaymentAmount": "net_icc_payment_usd",
            "totalBuildingInsuranceCoverage": "building_coverage_usd",
            "totalContentsInsuranceCoverage": "contents_coverage_usd",
        }
    )
    result["snapshot_as_of_date"] = pd.to_datetime(
        result["snapshot_as_of_date"], utc=True
    ).dt.strftime("%Y-%m-%d")
    for column in ("date_of_loss", "open_date"):
        result[column] = pd.to_datetime(result[column], errors="coerce").dt.strftime(
            "%Y-%m-%d"
        )
    for column in ("county_code", "census_geoid"):
        result[column] = result[column].astype("string")
    result["source_file"] = source_path.name
    result = result.sort_values("claim_record_id").reset_index(drop=True)
    if len(result) != 102_045 or result["claim_record_id"].duplicated().any():
        raise ValueError("unexpected 2024 NFIP claims population")
    if result["policy_count"].min() < 1 or result["policy_count"].max() <= 1:
        raise ValueError("NFIP policyCount grain check failed")
    output = root / "claims_2024.csv.gz"
    _write(result, output)
    return {
        "rows": len(result),
        "represented_policy_count": int(result.policy_count.sum()),
        "max_policy_count_per_record": int(result.policy_count.max()),
        "signed_negative_net_payment_cells": int(
            result[[
                "net_building_payment_usd",
                "net_contents_payment_usd",
                "net_icc_payment_usd",
            ]].lt(0).sum().sum()
        ),
        "output": str(output.relative_to(DATASETS)),
        "source_file": source_path.name,
    }


def _bea_value(value: object) -> tuple[float | None, str]:
    if pd.isna(value) or not str(value).strip():
        return None, "missing"
    token = str(value).strip()
    if token == "(D)":
        return None, "source_marker_D"
    if token == "(*)":
        return None, "source_marker_star"
    return float(token), "reported"


def build_bea_affiliate_trade() -> dict[str, object]:
    root = DATASETS / "bea_mne"
    source_path = RAW_DIR / "census_international_trade" / "usdia2023p-CombinedP.zip"
    member = "Part-II-H1-H14.xls"
    with ZipFile(source_path) as archive:
        with archive.open(member) as handle:
            source = pd.read_excel(handle, sheet_name="Table II.H 1", header=None)
    source = source.iloc[7:95, :7].copy()
    source.columns = [
        "geography_label",
        "affiliate_exports_total",
        "affiliate_exports_by_us_parents",
        "affiliate_exports_by_unaffiliated_us_persons",
        "affiliate_imports_total",
        "affiliate_imports_to_us_parents",
        "affiliate_imports_to_unaffiliated_us_persons",
    ]
    source.insert(0, "source_row_number", source.index + 1)
    source = source.loc[source.geography_label.notna()].copy()
    value_columns = [column for column in source if column.startswith("affiliate_")]
    for column in value_columns:
        parsed = source[column].map(_bea_value)
        source[f"{column}_million_usd"] = parsed.map(lambda pair: pair[0])
        source[f"{column}_status"] = parsed.map(lambda pair: pair[1])
        source[f"{column}_source_token"] = source[column].astype("string")
        source = source.drop(columns=column)
    source["statistics_year"] = 2023
    source["release_status"] = "preliminary"
    source["source_archive"] = source_path.name
    source["source_member"] = member
    source["source_sheet"] = "Table II.H 1"
    source = source.sort_values("source_row_number").reset_index(drop=True)
    if len(source) != 76 or source.source_row_number.duplicated().any():
        raise ValueError("unexpected BEA Table II.H 1 row population")
    output = root / "affiliate_goods_trade_by_geography_2023.csv.gz"
    _write(source, output)
    return {
        "rows": len(source),
        "source_marker_d_cells": int(
            sum(source[f"{column}_status"].eq("source_marker_D").sum() for column in value_columns)
        ),
        "source_marker_star_cells": int(
            sum(
                source[f"{column}_status"].eq("source_marker_star").sum()
                for column in value_columns
            )
        ),
        "output": str(output.relative_to(DATASETS)),
        "source_member": member,
    }


def build_census_country_end_use_trade() -> dict[str, object]:
    root = DATASETS / "census_international_trade"
    parts: list[pd.DataFrame] = []
    source_files: list[str] = []
    for direction in ("exports", "imports"):
        source_path = RAW_DIR / "census_international_trade" / f"country_enduse_{direction}_2016_2025.xlsx"
        sheet = "EnduseExp" if direction == "exports" else "EnduseImp"
        source = pd.read_excel(
            source_path,
            sheet_name=sheet,
            dtype={"CTY_CODE": "string", "END_USE": "string"},
        ).rename(
            columns={
                "CTY_CODE": "country_code",
                "CTY_DESC": "country_name",
                "END_USE": "end_use_code",
                "COMM_DESC": "end_use_description",
            }
        )
        source.insert(0, "trade_direction", direction)
        source["country_code"] = source["country_code"].str.zfill(4)
        source["end_use_code"] = source["end_use_code"].str.zfill(5)
        source = source.rename(
            columns={f"value_{str(year)[-2:]}": f"value_{year}_usd" for year in range(2016, 2026)}
        )
        source["source_file"] = source_path.name
        source["source_sheet"] = sheet
        parts.append(source)
        source_files.append(source_path.name)
    result = pd.concat(parts, ignore_index=True).sort_values(
        ["trade_direction", "country_code", "end_use_code"]
    ).reset_index(drop=True)
    key = ["trade_direction", "country_code", "end_use_code"]
    if len(result) != 48_268 or result.duplicated(key).any():
        raise ValueError("unexpected Census country/end-use trade population")
    output = root / "country_end_use_trade_2016_2025.csv.gz"
    _write(result, output)
    return {
        "rows": len(result),
        "country_codes": int(result.country_code.nunique()),
        "directions": sorted(result.trade_direction.unique().tolist()),
        "output": str(output.relative_to(DATASETS)),
        "source_files": source_files,
    }


def main() -> None:
    report = {
        "noaa_storm_event_details": build_noaa_storm_details(),
        "fema_nfip_claims": build_nfip_claims(),
        "bea_affiliate_goods_trade": build_bea_affiliate_trade(),
        "census_country_end_use_trade": build_census_country_end_use_trade(),
        "rules": [
            "NOAA K/M/B damage suffixes are decoded to USD while source tokens and missing values remain separate; only the standard 50-state-plus-DC USPS bridge is added.",
            "NFIP physical claim records retain source IDs, policyCount, signed gross/net payment fields and coverage fields; no record is expanded, deduplicated or netted away.",
            "BEA Table II.H 1 row order, geography labels, preliminary status and nonnumeric (D)/(*) source markers are preserved without assigning an unfrozen legend; marker cells are missing, never zero.",
            "Census country/end-use rows remain unaggregated and retain both source country/end-use identifiers and all annual values from 2016 through 2025.",
            "All four projections support retrospective analysis from later-frozen source artifacts, not strict historical as-known snapshots.",
        ],
    }
    path = PROJECT_ROOT / "metadata/build_reports/phase11_cross_source_table_build.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
