"""One pass over run.json and transcripts: per-turn delivery-layer and self-check flags.

Writes analysis/scratch/turn_flags.json: {"study|case|run_id|turn": {...}} with
- n_wrong_numeric, n_computed_not_delivered : wrong numeric outputs whose reference value (>=4 significant digits, any of
  several rounding forms, delimited by non-digit characters) appears in the tool output of the SAME turn. Only tool
  returns are searched: never the query, the system prompt or the model's own text. n_cnd_loose keeps the earlier
  definition (substring anywhere in the run's tool AND user messages, all turns) for comparison; it is not reported.
- n_missing, n_near_name : unset contract variables, and those a near-name variant (ratio>=0.75) was assigned instead
  (name_contract.py)
- phantom : completed turn with unset variables whose prose claims completion and writes a missing var as name=value
  but never assigned it in executed code (phantom.py)
- has_check, check_failed : explicit assert/isclose/allclose in code; a visible failure in the same turn's tool output
- alt_mentions : code snippets mentioning alternatives/conventions/options (convention_paralysis.py)
Runs ~20 min; consumed by tier_axis.py.
"""
import collections
import difflib
import glob
import json
import re

from common import load

cases, turns, vars_ = load()
assign_re = re.compile(r"^\s*([A-Za-z_]\w*)\s*(?:\[[^\]]*\])?\s*=[^=]", re.M)
dict_re = re.compile(r"""[\[\(]\s*['"]([A-Za-z_]\w*)['"]\s*[\]\)]\s*=[^=]""")
claim = re.compile(r"variables? (are|have been|is) (now |all )?(assigned|set|stored|defined|populated)|assigned (all|the) (required |output )?variables|all (required |output )?variables (are )?(assigned|set)", re.I)
chk = re.compile(r"\b(assert|np\.isclose|allclose)\b")
alt_re = re.compile(r"\b(opt\d|option\s?\d|alt\d|alternative|variant|approach\s?\d|method\s?\d|interpretation\s?\d|convention)\b", re.I)
fail_re = re.compile(r"AssertionError|Mismatch|mismatch|False\b.*isclose|isclose.*False|allclose.*False|False$", re.M)


def parse(msg):
    m = re.match(r"^(\S+?)=(.*?),? expected (.*)$", msg)
    return m.group(3).strip() if m else None


def variants(s):
    try:
        e = float(s.replace(",", ""))
    except ValueError:
        return set()
    vs = {s.replace(",", "").strip(), f"{e:g}"}
    if e != int(e):
        for nd in (2, 3, 4, 6):
            vs.add(f"{e:.{nd}f}"); vs.add(f"{round(e, nd):g}")
    if abs(e) >= 1000:
        vs.add(f"{e:,.0f}")
        if e != int(e):
            vs.add(f"{e:,.2f}")
    return {v for v in vs if sig_digits(v) >= 4}


def sig_digits(v):
    """Significant digits of a printed number: digits after dropping sign, separators and leading zeros
    (0.0523 -> 3, 11.0000 -> 6 as printed but 11 -> 2, 4,640 -> 4). Variants with fewer than four are too
    short to be matched safely and are dropped."""
    return len(re.sub(r"[^0-9]", "", v).lstrip("0"))


def strict_re(vs):
    """One regex matching any variant as a whole number: no digit, sign or decimal point touching it on the left,
    no digit on the right (so 0.0523 is not found inside 10.0523, -0.0523 or 0.05234)."""
    return re.compile(r"(?<![\d.\-])(?:" + "|".join(re.escape(v) for v in sorted(vs, key=len, reverse=True)) + r")(?!\d)")


def _text(x):
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return "\n".join(_text(y) for y in x)
    if isinstance(x, dict):
        if "text" in x and isinstance(x["text"], str):
            return x["text"]
        return json.dumps(x, ensure_ascii=False)
    return "" if x is None else str(x)


# Messages the harness injects inside a turn (continuation after an output limit; repair after an unreadable
# tool call). They arrive as user messages but do not start a turn.
HARNESS_USER = ("Output limit hit. Resume directly", "Your last tool call could not be read")


def is_query(text):
    return isinstance(text, str) and not text.startswith(HARNESS_USER)


def turn_tool_texts(path):
    """Tool returns grouped by turn: [[text, ...], ...]. A new turn starts at each user query message
    (harness-injected user messages excepted, see HARNESS_USER).

    Four transcript shapes: Anthropic (user str = query, user list of tool_result = tool output), OpenAI chat
    (user str, role tool), OpenAI Responses (user str, type function_call_output) and Gemini (parts with role user
    text = query, parts with function_response = tool output)."""
    out = []
    for l in open(path, encoding="utf-8"):
        try:
            m = json.loads(l)
        except ValueError:
            continue
        if "parts" in m:
            parts = m.get("parts") or []
            if any("function_response" in p for p in parts):
                if out:
                    out[-1].extend(_text((p["function_response"].get("response") or {}).get("output", p["function_response"].get("response"))) for p in parts if "function_response" in p)
            elif m.get("role") == "user" and any("text" in p for p in parts):
                if is_query(next((p["text"] for p in parts if "text" in p), "")):
                    out.append([])
            continue
        if m.get("type") == "function_call_output":
            if out:
                out[-1].append(_text(m.get("output")))
            continue
        role, c = m.get("role"), m.get("content")
        if role == "user" and isinstance(c, str):
            if is_query(c):
                out.append([])
        elif role == "tool":
            if out:
                out[-1].append(_text(c))
        elif role == "user" and isinstance(c, list):
            if out:
                out[-1].extend(_text(b.get("content")) for b in c if isinstance(b, dict) and b.get("type") == "tool_result")
    return out


def tool_text(path):
    out = []
    for l in open(path, encoding="utf-8"):
        try:
            m = json.loads(l)
        except ValueError:
            continue
        s = json.dumps(m, ensure_ascii=False)
        if m.get("role") in ("tool", "user") or '"function_response"' in s or '"tool_result"' in s or "function_call_output" in s:
            out.append(s)
    return "\n".join(out)


wrong_by_run = collections.defaultdict(lambda: collections.defaultdict(list))
for v in vars_:
    if v["is_set"] and v["correct"] is False:
        exp = parse(v["message"])
        if exp:
            vs = variants(exp)
            if vs:
                wrong_by_run[(v["study"], v["case"], v["run_id"])][v["turn"]].append((v["var"], exp, vs))
samples = []            # (hit?, study, case, run, turn, var, expected, context) for the manual spot check
turn_count_mismatch = 0
T = {(t["study"], t["case"], t["run_id"], t["turn"]): t for t in turns}
out = {}
n = 0
for f in sorted(glob.glob("experiments/*/runs/*/*/*/run.json")):
    study = f.split("/")[1]
    r = json.load(open(f)); case, rid = r["case"], r["run"]["run_id"]
    key = (study, case, rid)
    need_tt = key in wrong_by_run
    tt = None; per_turn = None
    def transcript():
        tr = glob.glob(f"experiments/{study}/runs/{case}/*/{rid}/transcripts/main.jsonl")
        return tr[0] if tr else None
    for c in r["conversations"]:
        for t in c["turns"]:
            meta = T.get((study, case, rid, t["turn"]))
            if not meta:
                continue
            code = t.get("code_snippets") or []; resp = t.get("response") or ""
            stores = cases[case]["turn_stores"][t["turn"] - 1]; outs = t.get("outputs") or {}
            missing = [s for s in stores if outs.get(s) is None]
            fl = dict(n_missing=len(missing), n_near_name=0, phantom=False, has_check=False, check_failed=False,
                      alt_mentions=sum(1 for c_ in code if alt_re.search(c_)), n_wrong_numeric=0, n_computed_not_delivered=0, n_cnd_loose=0)
            if missing and meta["kind"] == "variables_not_set":
                assigned = set(assign_re.findall("\n".join(code))) | set(dict_re.findall("\n".join(code)))
                extra = assigned - set(stores)
                for m in missing:
                    best = max((difflib.SequenceMatcher(None, m.lower(), a.lower()).ratio() for a in extra), default=0)
                    fl["n_near_name"] += best >= 0.75
                if meta["stop"] == "completed":
                    names_in_prose = [s for s in missing if re.search(rf"`?{re.escape(s)}`?\s*(=|:)", resp)]
                    assigned_in_code = [s for s in missing if any(re.search(rf"^\s*{re.escape(s)}\s*=", c_, re.M) for c_ in code)]
                    fl["phantom"] = bool(claim.search(resp)) and bool(names_in_prose) and not assigned_in_code
            if any(chk.search(c_) for c_ in code) or (need_tt and t["turn"] in wrong_by_run[key]):
                if per_turn is None:
                    tr = transcript()
                    per_turn = turn_tool_texts(tr) if tr else []
                    tt = tool_text(tr) if tr else ""
                    if len(per_turn) != len(c["turns"]):
                        turn_count_mismatch += 1
                own = "\n".join(per_turn[t["turn"] - 1]) if t["turn"] - 1 < len(per_turn) else ""
            if any(chk.search(c_) for c_ in code):
                fl["has_check"] = True
                fl["check_failed"] = bool(fail_re.search(own))
            if need_tt and t["turn"] in wrong_by_run[key]:
                for var, exp, vs in wrong_by_run[key][t["turn"]]:
                    fl["n_wrong_numeric"] += 1
                    fl["n_cnd_loose"] += any(v in tt for v in vs)
                    m_ = strict_re(vs).search(own)
                    fl["n_computed_not_delivered"] += bool(m_)
                    ctx = own[max(0, m_.start() - 120):m_.end() + 60].replace("\n", " ") if m_ else ""
                    samples.append((bool(m_), study, case, rid, t["turn"], var, exp, ctx))
            out[f"{study}|{case}|{rid}|{t['turn']}"] = fl
    n += 1
    if n % 500 == 0:
        print(n, "runs", flush=True)
json.dump(out, open("analysis/scratch/turn_flags.json", "w"))
print("runs whose transcript turn count differs from run.json:", turn_count_mismatch)
# spot-check sample: 20 strict hits and 20 non-hits, fixed seed, for manual annotation in spot_cnd.md
import random
rng = random.Random(20261013)
hits = [s for s in samples if s[0]]; misses = [s for s in samples if not s[0]]
with open("analysis/final_20261012/spot_cnd_sample.md", "w") as f:
    f.write(f"# Computed-but-not-delivered detector: spot-check sample\n\nstrict hits {len(hits)} / wrong numeric outputs {len(samples)}\n\n")
    for title, pool in (("## 20 hits (is the matched number this variable's own candidate value?)", hits), ("## 20 non-hits (does the correct value appear in another form?)", misses)):
        f.write(title + "\n\n")
        for s in rng.sample(pool, min(20, len(pool))):
            f.write(f"- `{s[1]}` / `{s[2]}` / `{s[3]}` turn {s[4]} var `{s[5]}` expected `{s[6]}`\n" + (f"  - context: `{s[7]}`\n" if s[7] else ""))
agg = collections.Counter()
for fl in out.values():
    for k, v in fl.items():
        agg[k] += int(v)
print("turns", len(out), dict(agg))
