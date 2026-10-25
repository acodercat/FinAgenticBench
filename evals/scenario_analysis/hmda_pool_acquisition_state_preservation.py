"""A stressed mortgage-pool acquisition whose FHA comparison must not replace the pool being bought.

The acquisition pool is built from 2024 Florida conventional first-lien purchase
originations, re-underwritten as hypothetical level-payment loans and stressed through a
joint payment-burden and collateral default test. An FHA pool is then evaluated as a
comparison and put through a severe alternative, after which the quote and the leveraged
purchase return to the conventional pool under its base underwriting case. The case
measures whether that original state survives two intervening analyses.

Rejected alternative conventions, each measured against the pinned reading:

- Quoting the FHA comparison instead of the acquisition pool moves the bid from 99.1857
  to 93.5013 per 100 and turns the leveraged equity return from 9.1488 to -6.9655
  percent.
- Carrying the severe FHA shock into the acquisition quote gives a bid of 93.8739 and a
  return of -5.8247 percent.
- Leaving property values unindexed instead of applying the 2023Q4-2024Q4 FHFA growth
  raises the base terminal credit loss from 974.1388 to 1,085.1055 million.
- A 240-month term instead of 360 raises the aggregate monthly payment from 237.6023 to
  279.1054 million and changes 11 of the 14 outputs.
"""

from functools import lru_cache
import numpy as np
import pandas as pd
from core.data import load_expansion_table
from core.dataset_layers import load_catalog, DATASETS_DIR


@lru_cache(None)
def inputs():
    meta = load_catalog()["tables"]["hmda.loan_applications"]
    cols = [
        "activity_year",
        "state_code",
        "action_taken",
        "lien_status",
        "loan_type",
        "loan_purpose",
        "occupancy_type",
        "total_units",
        "loan_amount",
        "income",
        "interest_rate",
        "property_value",
    ]
    parts = []
    for d in pd.read_csv(
        DATASETS_DIR / meta["layers"]["runtime"]["path"], usecols=cols, dtype=str, chunksize=300000
    ):
        keep = (
            d.activity_year.eq("2024")
            & d.state_code.eq("FL")
            & d.action_taken.eq("1")
            & d.lien_status.eq("1")
            & d.loan_type.isin(["1", "2"])
            & d.loan_purpose.eq("1")
            & d.occupancy_type.eq("1")
            & d.total_units.eq("1")
        )
        z = d[keep].copy()
        for c in ["loan_amount", "income", "interest_rate", "property_value"]:
            z[c] = pd.to_numeric(z[c], errors="coerce")
        z = z[
            z.loan_amount.between(100000, 750000)
            & z.income.between(40, 250)
            & z.interest_rate.between(3, 10)
            & z.property_value.between(100000, 1500000)
        ]
        parts.append(z)
    x = pd.concat(parts, ignore_index=True)
    h = load_expansion_table("fhfa_hpi")
    h = h[
        h.hpi_type.eq("traditional")
        & h.hpi_flavor.eq("purchase-only")
        & h.frequency.eq("quarterly")
        & h.level.eq("State")
        & h.place_id.eq("FL")
        & h.yr.isin([2023, 2024])
        & h.period.eq(4)
    ]
    assert len(h) == 2 and h.yr.is_unique
    ratio = float(h[h.yr.eq(2024)].index_nsa.iloc[0] / h[h.yr.eq(2023)].index_nsa.iloc[0])
    q = load_expansion_table("nyfed_reference_rates")
    q = q[q["Rate Type"].eq("SOFR") & q["Effective Date"].eq("2024-12-31")]
    assert len(q) == 1
    return x, ratio, float(q["Rate (%)"].iloc[0]) / 100


def pool(z, ratio, shock=0.2, term=360):
    p = z.loan_amount.to_numpy(float)
    r = z.interest_rate.to_numpy(float) / 1200
    a = p * r / -np.expm1(-term * np.log1p(r))
    b = p * (1 + r) ** 24 - a * np.expm1(24 * np.log1p(r)) / r
    home = z.property_value.to_numpy(float) * ratio * (1 - shock)
    ds = 12 * a / (1000 * z.income.to_numpy(float) * (1 - shock))
    default = (ds > 0.35) & (b > home)
    loss = np.where(default, np.maximum(b - 0.92 * home, 0), 0)
    cf = np.full(24, a.sum())
    cf[-1] += float((b - loss).sum())
    return dict(p=p, a=a, b=b, loss=loss, cf=cf, default=default, ds=ds, home=home)


def calculate(contamination="none", shock_path="base", hpi="indexed", term=360):
    x, ratio, rate = inputs()
    ratio = ratio if hpi == "indexed" else 1.0
    z = x[x.loan_type.eq("1")]
    f = x[x.loan_type.eq("2")]
    a = pool(z, ratio, term=term)
    b = pool(f, ratio, term=term)
    c = pool(f, ratio, 0.3, term)
    scale = a["p"].sum() / b["p"].sum()
    active = pool(
        f if contamination == "fha" else z, ratio, 0.3 if shock_path == "severe" else 0.2, term
    )
    w = a["p"].sum() / active["p"].sum()
    cf = active["cf"] * w
    principal = a["p"].sum()
    times = np.arange(1, 25) / 12
    disc = (1 + rate + 0.015) ** (-times)
    pv = cf * disc
    purchase = 0.98 * principal
    debt = 0.8 * purchase
    equity = 0.2 * purchase
    terminal = float(cf @ (1 + rate + 0.015) ** (2 - times) - debt * (1 + rate + 0.015) ** 2)
    out = dict(
        pool_count=len(z),
        pool_principal_million=principal / 1e6,
        pool_monthly_payment_million=a["a"].sum() / 1e6,
        month24_scheduled_balance_million=a["b"].sum() / 1e6,
        base_terminal_credit_loss_million=a["loss"].sum() / 1e6,
        fha_count=len(f),
        fha_equal_notional_payment_million=b["a"].sum() * scale / 1e6,
        fha_base_loss_million=b["loss"].sum() * scale / 1e6,
        fha_severe_loss_million=c["loss"].sum() * scale / 1e6,
        fha_severe_default_principal_pct=c["p"][c["default"]].sum() / c["p"].sum() * 100,
        acquisition_bid_per100=pv.sum() / principal * 100,
        acquisition_duration_years=pv @ times / pv.sum(),
        equity_terminal_million=terminal / 1e6,
        equity_annual_return_pct=((terminal / equity) ** 0.5 - 1) * 100,
    )
    assert terminal > 0 and np.isfinite(list(out.values())).all()
    return out, dict(
        base_default_count=int(a["default"].sum()),
        fha_default_count=int(b["default"].sum()),
        fha_severe_default_count=int(c["default"].sum()),
        scale=scale,
        hpi_ratio=ratio,
        sofr=rate,
        base_ds_min_boundary=float(np.min(abs(a["ds"] - 0.35))),
        base_equity_boundary=float(np.min(abs(a["b"] - a["home"]))),
        conservation=float(np.max(abs(a["p"] - (a["p"] - a["b"]) - a["b"]))),
    )


from cave_agent import Variable
from core.validation import validate_ordered_outputs, turn_validator

TURN_1_NAMES = ["pool_count", "pool_principal_million", "pool_monthly_payment_million"]
TURN_2_NAMES = ["month24_scheduled_balance_million", "base_terminal_credit_loss_million"]
TURN_3_NAMES = ["fha_count", "fha_equal_notional_payment_million", "fha_base_loss_million"]
TURN_4_NAMES = ["fha_severe_loss_million", "fha_severe_default_principal_pct"]
TURN_5_NAMES = ["acquisition_bid_per100", "acquisition_duration_years"]
TURN_6_NAMES = ["equity_terminal_million", "equity_annual_return_pct"]

variables = [
    Variable("pool_count", None, "Store the eligible conventional loan count as an integer."),
    Variable(
        "pool_principal_million",
        None,
        "Store initial conventional pool principal in USD millions, rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "pool_monthly_payment_million",
        None,
        "Store aggregate conventional monthly principal-and-interest payment in USD millions, rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "month24_scheduled_balance_million",
        None,
        "Store conventional scheduled balance after payment24 in USD millions, rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "base_terminal_credit_loss_million",
        None,
        "Store terminal acquisition underwriting credit loss in USD millions, rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "fha_count", None, "Store eligible FHA loan count before notional scaling as an integer."
    ),
    Variable(
        "fha_equal_notional_payment_million",
        None,
        "Store equal-notional FHA monthly payment in USD millions, rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "fha_base_loss_million",
        None,
        "Store equal-notional FHA baseline terminal loss in USD millions, rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "fha_severe_loss_million",
        None,
        "Store equal-notional FHA severe terminal loss in USD millions, rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "fha_severe_default_principal_pct",
        None,
        "Store severe FHA defaulting original principal share in percent, rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "acquisition_bid_per100",
        None,
        "Store maximum acquisition price per100 initial principal, rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "acquisition_duration_years",
        None,
        "Store Macaulay duration in years, rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "equity_terminal_million",
        None,
        "Store terminal equity wealth in USD millions, rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
    Variable(
        "equity_annual_return_pct",
        None,
        "Store two-year compound annual equity return in percent, rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.",
    ),
]
DECIMALS = [0, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 4, 4]


@lru_cache(maxsize=1)
def ground_truth():
    out = calculate()[0]
    return tuple(out[v.name] for v in variables)


def _validate_subset(outputs, names):
    by = {v.name: v for v in variables}
    truth = dict(zip(by, ground_truth()))
    places = dict(zip(by, DECIMALS))
    return validate_ordered_outputs(
        outputs, [by[n] for n in names], [truth[n] for n in names], [places[n] for n in names]
    )


def validate(outputs):
    return _validate_subset(outputs, [v.name for v in variables])


def validate_turn_1(outputs):
    return _validate_subset(outputs, TURN_1_NAMES)


def validate_turn_2(outputs):
    return _validate_subset(outputs, TURN_2_NAMES)


def validate_turn_3(outputs):
    return _validate_subset(outputs, TURN_3_NAMES)


def validate_turn_4(outputs):
    return _validate_subset(outputs, TURN_4_NAMES)


def validate_turn_5(outputs):
    return _validate_subset(outputs, TURN_5_NAMES)


def validate_turn_6(outputs):
    return _validate_subset(outputs, TURN_6_NAMES)


validators = {
    f"validate_turn_{i}": turn_validator(globals()[f"validate_turn_{i}"]) for i in range(1, 7)
}
