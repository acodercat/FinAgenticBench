"""Shared grouping of the 14 study configurations.

Everything that compares models imports from here so that the thinking map, the
capability tiers and the family labels are defined once. Thinking mode is NOT
reliable in run.json (`reasoning` is false for most thinking configs), so it is
taken from the study id.

Capability tiers are defined by PV task accuracy over 213 cases x 3 runs on the
final results; the thresholds live here and the paper must call them capability
tiers, not parameter sizes (no parameter counts are known for the API models).
"""
from __future__ import annotations

# study id -> (display model, mode)
STUDIES = {
    "deepseek-v4-pro-0813-thinking-20261006": ("DeepSeek V4 Pro", "T"),
    "gemini-3.8-flash-thinking-20261007": ("Gemini 3.8 Flash", "T"),
    "gpt-6-luna-thinking-20261006": ("GPT-6 Luna", "T"),
    "claude-sonnet-5-thinking-20261007": ("Claude Sonnet 5.5", "T"),
    "claude-sonnet-5-nonthinking-20261007": ("Claude Sonnet 5.5", "N"),
    "glm-5.3-thinking-high-20261007": ("GLM-5.3", "T"),
    "qwen3.8-flash-thinking-20261006": ("Qwen3.8 Flash", "T"),
    "qwen3.8-27b-thinking-20261008": ("Qwen3.8-27B", "T"),
    "deepseek-v4-flash-thinking-20261006": ("DeepSeek V4 Flash", "T"),
    "deepseek-v4-flash-nonthinking-20261006": ("DeepSeek V4 Flash", "N"),
    "deepseek-v4-pro-0813-nonthinking-20261006": ("DeepSeek V4 Pro", "N"),
    "gpt-6-luna-nonthinking-20261006": ("GPT-6 Luna", "N"),
    "qwen3.8-flash-nonthinking-20261006": ("Qwen3.8 Flash", "N"),
    "qwen3.8-27b-nonthinking-20261008": ("Qwen3.8-27B", "N"),
}
MODEL = {s: m for s, (m, _) in STUDIES.items()}
MODE = {s: t for s, (_, t) in STUDIES.items()}
THINKING = {s for s, t in MODE.items() if t == "T"}

# Same model with and without extended thinking: (thinking study, non-thinking study)
PAIRS = {
    "DeepSeek V4 Pro": ("deepseek-v4-pro-0813-thinking-20261006", "deepseek-v4-pro-0813-nonthinking-20261006"),
    "DeepSeek V4 Flash": ("deepseek-v4-flash-thinking-20261006", "deepseek-v4-flash-nonthinking-20261006"),
    "GPT-6 Luna": ("gpt-6-luna-thinking-20261006", "gpt-6-luna-nonthinking-20261006"),
    "Claude Sonnet 5.5": ("claude-sonnet-5-thinking-20261007", "claude-sonnet-5-nonthinking-20261007"),
    "Qwen3.8 Flash": ("qwen3.8-flash-thinking-20261006", "qwen3.8-flash-nonthinking-20261006"),
    "Qwen3.8-27B": ("qwen3.8-27b-thinking-20261008", "qwen3.8-27b-nonthinking-20261008"),
}

# Model family: the lab's product line. Six vendors, eight families.
FAMILY = {
    "deepseek-v4-pro-0813-thinking-20261006": "deepseek-pro", "deepseek-v4-pro-0813-nonthinking-20261006": "deepseek-pro",
    "deepseek-v4-flash-thinking-20261006": "deepseek-flash", "deepseek-v4-flash-nonthinking-20261006": "deepseek-flash",
    "gemini-3.8-flash-thinking-20261007": "gemini",
    "gpt-6-luna-thinking-20261006": "gpt-6-luna", "gpt-6-luna-nonthinking-20261006": "gpt-6-luna",
    "claude-sonnet-5-thinking-20261007": "claude", "claude-sonnet-5-nonthinking-20261007": "claude",
    "glm-5.3-thinking-high-20261007": "glm",
    "qwen3.8-flash-thinking-20261006": "qwen-flash", "qwen3.8-flash-nonthinking-20261006": "qwen-flash",
    "qwen3.8-27b-thinking-20261008": "qwen-27b", "qwen3.8-27b-nonthinking-20261008": "qwen-27b",
}

# Vendor (training provenance as far as it is public): the unit for "shared across vendors" and for
# cross-vendor agreement. FAMILY above distinguishes product lines of one vendor (DeepSeek Pro vs Flash,
# Qwen Flash vs 27B) and must not be read as independent training sources.
VENDOR = {s: {"deepseek-pro": "deepseek", "deepseek-flash": "deepseek", "gemini": "google", "gpt-6-luna": "openai",
              "claude": "anthropic", "glm": "zhipu", "qwen-flash": "alibaba", "qwen-27b": "alibaba"}[f] for s, f in FAMILY.items()}

# Capability tiers by PV task accuracy (final results, 2026-10-12). Thresholds are the only
# definition; tier membership is derived from accuracy at run time via tiers().
TIER_THRESHOLDS = {"strong": 0.78, "mid": 0.60}   # strong >= 0.78; mid in [0.60, 0.78); weak < 0.60

# Two pairs where a size relation is actually known (same lab, small vs large product; open 27B vs API).
SIZE_PAIRS = {
    "DeepSeek Flash -> Pro (thinking)": ("deepseek-v4-flash-thinking-20261006", "deepseek-v4-pro-0813-thinking-20261006"),
    "DeepSeek Flash -> Pro (non-thinking)": ("deepseek-v4-flash-nonthinking-20261006", "deepseek-v4-pro-0813-nonthinking-20261006"),
    "Qwen3.8-27B -> Qwen3.8 Flash (thinking)": ("qwen3.8-27b-thinking-20261008", "qwen3.8-flash-thinking-20261006"),
    "Qwen3.8-27B -> Qwen3.8 Flash (non-thinking)": ("qwen3.8-27b-nonthinking-20261008", "qwen3.8-flash-nonthinking-20261006"),
}


def task_accuracy(turns):
    """PV task accuracy per study from final_data turns: a case passes a run only if every turn passes."""
    import collections
    runs = collections.defaultdict(list)
    for t in turns:
        runs[(t["study"], t["case"], t["run_id"])].append(t["ok"])
    per = collections.defaultdict(list)
    for (s, _, _), oks in runs.items():
        per[s].append(all(oks))
    return {s: sum(v) / len(v) for s, v in per.items()}


def tiers(acc):
    """Map study -> tier from accuracies; also return the ordered membership for printing."""
    out = {}
    for s, a in acc.items():
        out[s] = "strong" if a >= TIER_THRESHOLDS["strong"] else ("mid" if a >= TIER_THRESHOLDS["mid"] else "weak")
    return out


def label(study):
    m, t = STUDIES[study]
    return f"{m} {'thinking' if t == 'T' else 'non-thinking'}"


if __name__ == "__main__":
    import json, sys
    d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "analysis/scratch/final_data.json"))
    acc = task_accuracy(d["turns"]); tr = tiers(acc)
    for s in sorted(acc, key=lambda s: -acc[s]):
        print(f"{tr[s]:6} {MODE[s]} {acc[s]:.1%}  {label(s)}")
    import collections
    print({k: v for k, v in collections.Counter(tr.values()).items()})


# Model-level tier for the thinking x capability cross. Config-level tiers are degenerate for
# that purpose (every weak config is non-thinking, every strong config is thinking), so the
# model is placed by its thinking configuration's accuracy and both of its modes inherit the
# tier. Only paired models enter the 2x2.
def model_tiers(acc):
    out = {}
    for model, (T, N) in PAIRS.items():
        tier = "top" if acc[T] >= TIER_THRESHOLDS["strong"] else "lower"
        out[T] = tier; out[N] = tier
    return out
