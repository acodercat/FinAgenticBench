"""Track a fixed money-fund share-class cohort across two original N-MFP3 filings.

This is a baseline. September gross subscriptions alone would make Y Shares
2,164.7783 USD million rather than its signed 2,088.3710 million net inflow;
I Shares gross redemptions of 1,725.6992 million are not its signed
-908.6856 million net outflow. Aggregating only those two classes gives
1,179.6854 million, not the six-class 554.8497 million. In October the
previously selected I Shares turn positive (+17.5722 million), so repeating
September's outflow classification is not a valid fixed-ID comparison.
The original-only N-MFP3 scope rules out amended or different-month rows.
"""

from __future__ import annotations

from decimal import Decimal
from functools import lru_cache

from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


SERIES_ID = "S000005047"
MILLION = Decimal(1_000_000)
OUTPUTS = (
    ("class_flow_sep_accession", None, "the September filing's exact SEC accession string in 10-2-6 digit form"),
    ("class_flow_sep_assets_m", 4, "September series net assets in USD millions"),
    ("class_flow_sep_matched_classes", 0, "count of September classes with both same-day flow and yield rows as an integer"),
    ("class_flow_inflow_id", None, "largest September net-inflow class's exact SEC class ID as C followed by 9 digits"),
    ("class_flow_inflow_m", 4, "that class's signed September net flow in USD millions"),
    ("class_flow_outflow_id", None, "largest September net-outflow class's exact SEC class ID as C followed by 9 digits"),
    ("class_flow_outflow_m", 4, "that class's signed September net flow in USD millions"),
    ("class_flow_sep_total_m", 4, "September net flow across the matched classes in USD millions"),
    ("class_flow_inflow_net_yield_pct", 2, "selected September inflow class's seven-day net yield in percent"),
    ("class_flow_outflow_net_yield_pct", 2, "selected September outflow class's seven-day net yield in percent"),
    ("class_flow_sep_gross_yield_pct", 2, "the same-date series seven-day gross yield in percent"),
    ("class_flow_oct_accession", None, "the October filing's exact SEC accession string in 10-2-6 digit form"),
    ("class_flow_oct_inflow_id_net_m", 4, "October signed net flow for September's selected inflow class in USD millions"),
    ("class_flow_oct_outflow_id_net_m", 4, "October signed net flow for September's selected outflow class in USD millions"),
    ("class_flow_oct_inflow_id_yield_pct", 2, "October seven-day net yield for September's selected inflow class in percent"),
    ("class_flow_oct_outflow_id_yield_pct", 2, "October seven-day net yield for September's selected outflow class in percent"),
    ("class_flow_oct_total_m", 4, "October net flow across the carried classes in USD millions"),
)
variables = [Variable(name, None, "Store " + description +
                      (f", rounded to {places} decimals from unrounded source values."
                       if places is not None and places > 0 else "."))
             for name, places, description in OUTPUTS]
PRECISION = {name: places for name, places, _ in OUTPUTS}
TURNS = tuple(tuple(name for name, _, _ in OUTPUTS[a:b]) for a, b in
              ((0, 3), (3, 8), (8, 11), (11, 17)))


def _decimal(value) -> Decimal:
    number = Decimal(str(value))
    if not number.is_finite():
        raise ValueError("non-finite N-MFP value")
    return number


def _filing(series, date: str):
    rows = series.loc[
        series.series_id.eq(SERIES_ID) & series.report_date.eq(date)
        & series.form.eq("N-MFP3") & series.filing_date.le("2025-12-31")
    ]
    if len(rows) != 1:
        raise ValueError(f"expected one original filing for {date}, found {len(rows)}")
    return rows.iloc[0]


def _panel(flows, yields, accession: str, date: str):
    daily_flows = flows.loc[flows.accession.eq(accession) & flows.flow_date.eq(date)].copy()
    daily_yields = yields.loc[yields.accession.eq(accession) & yields.yield_date.eq(date)].copy()
    for name, frame in (("flow", daily_flows), ("yield", daily_yields)):
        if frame.empty or frame.class_id.isna().any() or frame.class_id.duplicated().any():
            raise ValueError(f"missing or duplicate {name} class-day key")
    joined = daily_flows.merge(
        daily_yields[["class_id", "class_name", "seven_day_net_yield", "seven_day_gross_yield"]],
        on="class_id", how="inner", validate="one_to_one",
    )
    if joined.empty:
        raise ValueError("no matched class-day rows")
    return {str(row.class_id): row for row in joined.itertuples(index=False)}


def _net(row) -> Decimal:
    return _decimal(row.gross_subscriptions_usd) - _decimal(row.gross_redemptions_usd)


def analysis() -> dict[str, str | int | float]:
    series = load_expansion_table("sec_nmfp")
    flows = load_expansion_table("sec_nmfp_flows")
    yields = load_expansion_table("sec_nmfp_class_yields")
    sep = _filing(series, "2025-09-30")
    oct_ = _filing(series, "2025-10-31")
    september = _panel(flows, yields, str(sep.accession), "2025-09-30")
    october = _panel(flows, yields, str(oct_.accession), "2025-10-31")
    if set(september) != set(october):
        raise ValueError("carried class IDs missing from October")
    if any(september[key].class_name != october[key].class_name for key in september):
        raise ValueError("class name changed for a carried ID")
    flow_by_class = {key: _net(row) for key, row in september.items()}
    inflow_id = min(flow_by_class, key=lambda key: (-flow_by_class[key], key))
    outflow_id = min(flow_by_class, key=lambda key: (flow_by_class[key], key))
    if flow_by_class[inflow_id] <= 0 or flow_by_class[outflow_id] >= 0:
        raise ValueError("selected flow signs do not match the question")
    gross = {_decimal(row.seven_day_gross_yield) for row in september.values()}
    if len(gross) != 1:
        raise ValueError("gross yield is not unique within the September series-day")
    return {
        "class_flow_sep_accession": str(sep.accession),
        "class_flow_sep_assets_m": float(_decimal(sep.net_assets_usd) / MILLION),
        "class_flow_sep_matched_classes": len(september),
        "class_flow_inflow_id": inflow_id,
        "class_flow_inflow_m": float(flow_by_class[inflow_id] / MILLION),
        "class_flow_outflow_id": outflow_id,
        "class_flow_outflow_m": float(flow_by_class[outflow_id] / MILLION),
        "class_flow_sep_total_m": float(sum(flow_by_class.values(), Decimal(0)) / MILLION),
        "class_flow_inflow_net_yield_pct": float(_decimal(september[inflow_id].seven_day_net_yield) * 100),
        "class_flow_outflow_net_yield_pct": float(_decimal(september[outflow_id].seven_day_net_yield) * 100),
        "class_flow_sep_gross_yield_pct": float(next(iter(gross)) * 100),
        "class_flow_oct_accession": str(oct_.accession),
        "class_flow_oct_inflow_id_net_m": float(_net(october[inflow_id]) / MILLION),
        "class_flow_oct_outflow_id_net_m": float(_net(october[outflow_id]) / MILLION),
        "class_flow_oct_inflow_id_yield_pct": float(_decimal(october[inflow_id].seven_day_net_yield) * 100),
        "class_flow_oct_outflow_id_yield_pct": float(_decimal(october[outflow_id].seven_day_net_yield) * 100),
        "class_flow_oct_total_m": float(sum((_net(october[key]) for key in september), Decimal(0)) / MILLION),
    }


@lru_cache(maxsize=1)
def ground_truth():
    result = analysis()
    return tuple(result[v.name] for v in variables)


def _check(outputs, names):
    by = {v.name: v for v in variables}
    truth = dict(zip(by, ground_truth()))
    return validate_ordered_outputs(outputs, [by[name] for name in names],
                                    [truth[name] for name in names],
                                    [PRECISION[name] for name in names])


def validate_turn_1(outputs): return _check(outputs, TURNS[0])
def validate_turn_2(outputs): return _check(outputs, TURNS[1])
def validate_turn_3(outputs): return _check(outputs, TURNS[2])
def validate_turn_4(outputs): return _check(outputs, TURNS[3])
def validate(outputs): return _check(outputs, tuple(v.name for v in variables))


validators = {f"validate_turn_{i}": turn_validator(globals()[f"validate_turn_{i}"])
              for i in range(1, 5)}
