"""One pass over the final results: per-turn and per-variable rows for the defect analysis."""
import json, importlib, sys
from pathlib import Path
sys.path.insert(0, ".")
E = Path("experiments")
reg = json.loads(Path("benchmarks.json").read_text())["cases"]
cases = {}
for name, path in reg.items():
    s = json.loads(Path(path).read_text())
    turns = [t for c in s["conversations"] for t in c["turns"]]
    cases[name] = dict(family=s["task_family"], difficulty=s.get("difficulty"), case_type=s.get("case_type"),
                       sources=s["data_sources"], n_turns=len(turns),
                       turn_stores=[t["stores"] for t in turns], turn_claims=[len(t.get("claims", [])) for t in turns],
                       query_len=[len(t["query"]) for t in turns])
    try:
        m = importlib.import_module(s["module"])
        cases[name]["var_desc"] = {v.name: v.description for v in m.variables}
    except Exception as e:
        cases[name]["var_desc"] = {}
turn_rows, var_rows = [], []
for sd in sorted(E.iterdir()):
    study = sd.name
    for d in sorted((sd / "runs").glob("*/*/*")):
        case = d.parts[-3]; run_id = d.name
        r = json.loads((d / "run.json").read_text()); p = json.loads((d / "programmatic_verification.json").read_text())
        pvt = [t for c in p["conversations"] for t in c["turns"]]
        rt = [t for c in r["conversations"] for t in c["turns"]]
        prev_ok = None
        for i, (t, v) in enumerate(zip(rt, pvt)):
            pv = v.get("programmatic_verification") or {}
            vv = pv.get("variable_verification") or {}
            ok = v.get("status") == "scored" and bool(pv.get("success"))
            kind = "pass" if ok else (pv.get("failure_type") or v.get("status"))
            code = t.get("code_snippets") or []
            turn_rows.append(dict(study=study, case=case, run_id=run_id, turn=i + 1, n_turns=len(rt), ok=ok, kind=kind,
                status=v.get("status"), msg=(pv.get("validation_message") or "")[:200], steps=t.get("steps"),
                prompt_tokens=t.get("prompt_tokens"), completion_tokens=t.get("completion_tokens"), elapsed=t.get("elapsed"),
                stop=t.get("stop_reason"), executed_code=t.get("executed_code"), n_code=len(code),
                code_chars=sum(len(c) for c in code), n_unref=len(t.get("tables_unreferenced") or []),
                n_ref=len(t.get("tables_referenced") or []), unref=t.get("tables_unreferenced") or [],
                n_vars=vv.get("total"), n_set=vv.get("set"), n_correct=vv.get("correct"),
                interaction_failure=vv.get("interaction_failure"), prev_ok=prev_ok, resp_len=len(t.get("response") or ""),
                unserializable=t.get("outputs_unserializable") or []))
            # The per-turn validators of 21 cases score the turn by filling every *other* turn's
            # outputs with the reference answers before validating; the stored variable_results
            # therefore carry those projected outputs as "set, correct, value None". Only the
            # outputs the turn itself declares (its `stores`) are the turn's own verdicts.
            own = set(cases[case]["turn_stores"][i])
            items = vv.get("items") or {}
            turn_rows[-1]["n_out_of_turn"] = sum(1 for name in items if name not in own)
            for name, item in items.items():
                if name not in own:
                    continue
                var_rows.append(dict(study=study, case=case, run_id=run_id, turn=i + 1, var=name, is_set=item.get("is_set"),
                    correct=item.get("correct"), message=(item.get("message") or "")[:160],
                    value=json.dumps((t.get("outputs") or {}).get(name), default=str)[:120]))
            prev_ok = ok
json.dump(dict(cases=cases, turns=turn_rows, vars=var_rows), open("analysis/scratch/final_data.json", "w"), default=str)
out_of_turn = sum(t["n_out_of_turn"] for t in turn_rows)
print("turns", len(turn_rows), "vars", len(var_rows), "cases", len(cases), "| out-of-turn projected rows excluded", out_of_turn,
      "in", len({t["case"] for t in turn_rows if t["n_out_of_turn"]}), "cases")
