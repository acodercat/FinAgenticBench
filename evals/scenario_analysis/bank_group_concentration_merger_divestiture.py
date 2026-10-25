"""Baseline: organization boundary, overlap markets and branch disposal.

Current NIC organization membership reclassifies 2025 SOD geography. Counties
are analytical units, not asserted regulatory markets. The hypothetical transfer
preserves all deposits and creates a new independent local competitor. Runtime
ground truth uses exact rational comparisons; no author answer file is read.

Rejected alternative conventions, each measured against the pinned reading:

- Building groups from every direct NIC relationship rather than controlled ones leaves
  2,516 eligible counties instead of 2,531.
- Admitting counties with two positive groups rather than three gives 2,885 eligible
  counties.
- Choosing the transfer that minimizes post-transfer HHI rather than the deposits given
  up selects branch 2025_21591_0 of bank 214106, relinquishing 1,372,629 rather than
  1,184,514 thousand USD, with a final HHI of 4,317.8043 instead of 4,536.1711.
- Reporting HHI on a 0-1 scale rather than 0-10,000 misstates all four HHI outputs.
"""

from collections import defaultdict, deque
from fractions import Fraction
from functools import lru_cache
import math

import pandas as pd
from cave_agent import Variable
from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs

SPECS = [
    ("eligible_counties", "Store the number of eligible county analysis units as an integer.", 0),
    ("origin_county", "Store the selected county FIPS as an exact five-digit string.", None),
    (
        "origin_bank_hhi",
        "Store the selected county bank-level HHI on the 0–10,000 scale, rounded to 4 decimals.",
        4,
    ),
    (
        "origin_group_hhi",
        "Store the selected county group-level HHI on the 0–10,000 scale, rounded to 4 decimals.",
        4,
    ),
    (
        "acquiring_group",
        "Store the selected group root RSSD as a decimal-digit string without leading zeros.",
        None,
    ),
    (
        "group_bank_count",
        "Store the count of group banks with positive deposits in the specified broader branch population as an integer.",
        0,
    ),
    (
        "origin_group_deposit_share_pct",
        "Store the selected county share of group deposits in the specified broader branch population in percent, rounded to 4 decimals.",
        4,
    ),
    (
        "target_group",
        "Store the acquired group root RSSD as a decimal-digit string without leading zeros.",
        None,
    ),
    (
        "overlap_counties",
        "Store the number of eligible counties with positive deposits from both merging groups as an integer.",
        0,
    ),
    (
        "merger_county",
        "Store the county selected for its merger HHI increase as an exact five-digit FIPS string.",
        None,
    ),
    (
        "merger_after_hhi",
        "Store that county post-merger HHI on the 0–10,000 scale, rounded to 4 decimals.",
        4,
    ),
    (
        "qualifying_transfers",
        "Store the number of qualifying single-branch transfer alternatives as an integer.",
        0,
    ),
    (
        "transfer_bank",
        "Store the selected branch bank RSSD as a decimal-digit string without leading zeros, or exactly NONE when no transfer is made.",
        None,
    ),
    (
        "transfer_branch",
        "Store the selected source branch identifier verbatim, or exactly NONE when no transfer is made.",
        None,
    ),
    (
        "transferred_deposits_thousand_usd",
        "Store transferred deposits in thousand USD as an integer, including zero when no transfer is made.",
        0,
    ),
    (
        "final_hhi",
        "Store the county HHI after the chosen action on the 0–10,000 scale, rounded to 4 decimals.",
        4,
    ),
]
variables = [Variable(n, None, d) for n, d, _ in SPECS]
DECIMALS = [places for _, _, places in SPECS]
TURN_NAMES = [[s[0] for s in SPECS[a:b]] for a, b in [(0, 4), (4, 7), (7, 11), (11, 16)]]


def group_roots(edges, banks):
    """Unique roots; cyclic ancestry remains unprocessed by Kahn propagation."""
    children = defaultdict(set)
    parents = defaultdict(set)
    nodes = set(banks)
    for parent, child in edges:
        nodes.update([parent, child])
        children[parent].add(child)
        parents[child].add(parent)
    degree = {n: len(parents[n]) for n in nodes}
    queue = deque(sorted(n for n in nodes if degree[n] == 0))
    roots = {n: {n} for n in queue}
    while queue:
        node = queue.popleft()
        for child in children[node]:
            roots.setdefault(child, set()).update(roots[node])
            degree[child] -= 1
            if degree[child] == 0:
                queue.append(child)
    return {
        b: next(iter(roots[b])) for b in banks if degree[b] == 0 and len(roots.get(b, set())) == 1
    }


def hhi(amounts):
    amounts = [int(x) for x in amounts]
    total = sum(amounts)
    if total <= 0:
        raise ValueError("HHI requires positive total deposits")
    return Fraction(10000 * sum(x * x for x in amounts), total * total)


def transfer_options(groups, acquirer, target, branches, objective="least_deposits"):
    """branches are (bank RSSD, source branch ID, nonnegative integer deposits)."""
    before = hhi(groups.values())
    merged = int(groups[acquirer]) + int(groups[target])
    other = [int(v) for k, v in groups.items() if k not in {acquirer, target}]
    options = []
    for bank, branch, amount in branches:
        amount = int(amount)
        if amount < 0 or amount > merged:
            raise ValueError("invalid transfer amount")
        post = hhi(other + [merged - amount, amount])
        options.append((bank, branch, amount, post))
    feasible = [x for x in options if x[3] <= before]
    if objective == "least_deposits":
        feasible.sort(key=lambda x: (x[2], int(x[0]), x[1]))
    else:
        feasible.sort(key=lambda x: (x[3], x[2], int(x[0]), x[1]))
    chosen = feasible[0] if feasible else ("NONE", "NONE", 0, hhi(other + [merged]))
    return chosen, feasible, options


@lru_cache(maxsize=None)
def analysis(control="controlled", minimum_groups=3, objective="least_deposits"):
    """Every output; the keyword defaults are the conventions the query pins."""
    s = load_expansion_table("fdic_sod").copy()
    s = s[s.report_date.eq("2025-06-30")].copy()
    s = s[
        s.branch_county_fips.notna() & s.branch_county_fips.astype(str).str.fullmatch(r"\d{5}")
    ].copy()
    if s.duplicated(["bank_rssd_id", "branch_id"]).any():
        raise ValueError("duplicate branch keys")
    o = load_expansion_table("ffiec_nic_ownership")
    o = o[o.relationship_level.astype(str).eq("1")]
    if control == "controlled":
        o = o[o.control_indicator.astype(str).eq("1")]
    if o[["parent_rssd_id", "offspring_rssd_id"]].isna().any().any():
        raise ValueError("missing control endpoints")
    edges = set(zip(o.parent_rssd_id.astype(str), o.offspring_rssd_id.astype(str)))
    s["bank"] = s.bank_rssd_id.astype("string")
    bank_valid = s.bank.notna() & s.bank.str.fullmatch(r"[1-9]\d*", na=False)
    amount = pd.to_numeric(s.branch_deposits_thousand_usd, errors="coerce")
    amount_valid = (
        amount.notna()
        & amount.ge(0)
        & amount.map(lambda x: math.isfinite(float(x)) if pd.notna(x) else False)
    )
    if (amount[amount_valid] % 1).ne(0).any():
        raise ValueError("noninteger source thousand-dollar amounts")
    mapping = group_roots(edges, set(s.loc[bank_valid, "bank"]))
    s["group"] = s.bank.map(mapping)
    s["usable"] = bank_valid & amount_valid & s.group.notna()
    s["amount"] = amount
    complete = s.groupby("branch_county_fips").usable.all()
    broad = s[s.usable].copy()
    population = broad[broad.branch_county_fips.isin(complete[complete].index)]
    banks = defaultdict(dict)
    groups = defaultdict(lambda: defaultdict(int))
    for (county, bank), amount in (
        population.groupby(["branch_county_fips", "bank"]).amount.sum().items()
    ):
        banks[county][bank] = int(amount)
        groups[county][mapping[bank]] += int(amount)
    eligible = {c for c, g in groups.items() if sum(v > 0 for v in g.values()) >= minimum_groups}
    if not eligible:
        raise ValueError("no eligible county")
    bank_hhi = {c: hhi(banks[c].values()) for c in eligible}
    group_hhi = {c: hhi(groups[c].values()) for c in eligible}
    origin = min(eligible, key=lambda c: (-(group_hhi[c] - bank_hhi[c]), c))
    total = sum(groups[origin].values())
    contribution = {
        g: Fraction(
            10000 * (v * v - sum(d * d for b, d in banks[origin].items() if mapping[b] == g)),
            total * total,
        )
        for g, v in groups[origin].items()
    }
    acquirer = min(contribution, key=lambda g: (-contribution[g], int(g)))
    assert sum(contribution.values()) == group_hhi[origin] - bank_hhi[origin]
    footprint = broad[broad.group.eq(acquirer)]
    national = footprint.groupby("bank").amount.sum()
    share = Fraction(100 * int(groups[origin][acquirer]), int(national.sum()))
    target = min(
        (g for g, v in groups[origin].items() if g != acquirer and v > 0),
        key=lambda g: (-groups[origin][g], int(g)),
    )
    overlaps = sorted(
        c for c in eligible if groups[c].get(acquirer, 0) > 0 and groups[c].get(target, 0) > 0
    )
    increments = {
        c: Fraction(20000 * groups[c][acquirer] * groups[c][target], sum(groups[c].values()) ** 2)
        for c in overlaps
    }
    county = min(overlaps, key=lambda c: (-increments[c], c))
    selected = broad[broad.branch_county_fips.eq(county) & broad.group.isin([acquirer, target])]
    branches = [(r.bank, str(r.branch_id), int(r.amount)) for r in selected.itertuples()]
    choice, feasible, options = transfer_options(
        groups[county], acquirer, target, branches, objective
    )
    post = hhi(
        [v for g, v in groups[county].items() if g not in {acquirer, target}]
        + [groups[county][acquirer] + groups[county][target]]
    )
    assert post - group_hhi[county] == increments[county]
    values = [
        len(eligible),
        origin,
        bank_hhi[origin],
        group_hhi[origin],
        acquirer,
        int(national.gt(0).sum()),
        share,
        target,
        len(overlaps),
        county,
        post,
        len(feasible),
        choice[0],
        choice[1],
        choice[2],
        choice[3],
    ]
    answer = {s[0]: float(v) if isinstance(v, Fraction) else v for s, v in zip(SPECS, values)}
    evidence = {
        "contributions": {g: float(v) for g, v in contribution.items()},
        "group_national_deposits_thousand_usd": int(national.sum()),
        "group_member_banks": national.astype(int).to_dict(),
        "overlap_increases": {c: float(v) for c, v in increments.items()},
        "selected_county_groups": dict(groups[county]),
        "options": [(b, k, v, float(h)) for b, k, v, h in options],
        "excluded_incomplete_counties": int((~complete).sum()),
        "unmapped_bank_count": len(set(s.loc[bank_valid, "bank"]) - set(mapping)),
        "eligible_counties": sorted(eligible),
        "merger_before_hhi": float(group_hhi[county]),
    }
    return answer, evidence


def ground_truth():
    answer, _ = analysis()
    return tuple(answer[v.name] for v in variables)


def _validate(outputs, names):
    answer, _ = analysis()
    specs = [s for s in SPECS if s[0] in names]
    return validate_ordered_outputs(
        outputs,
        [Variable(n, None, d) for n, d, _ in specs],
        [answer[n] for n, _, _ in specs],
        [p for _, _, p in specs],
    )


def validate(outputs):
    return _validate(outputs, [v.name for v in variables])


def validate_selection(outputs):
    return _validate(outputs, TURN_NAMES[0])


def validate_group_footprint(outputs):
    return _validate(outputs, TURN_NAMES[1])


def validate_merger(outputs):
    return _validate(outputs, TURN_NAMES[2])


def validate_transfer(outputs):
    return _validate(outputs, TURN_NAMES[3])


validators = {
    name: turn_validator(fn)
    for name, fn in [
        ("validate_selection", validate_selection),
        ("validate_group_footprint", validate_group_footprint),
        ("validate_merger", validate_merger),
        ("validate_transfer", validate_transfer),
    ]
}
