"""Model provenance for the supplement: per model and role, the requested and returned model
identifiers, routes and fallback shares, collection dates, and final-reply versus total call
counts. Reads the call logs only (records of the current study: the latest record per call id).

    PYTHONPATH=src python scripts/revision_provenance.py
"""
import collections
import json

import pandas as pd

from tlt.config import load_run

run = load_run("full")
LOGS = ["generations", "repair_generations", "repair_steps", "judgements", "repair_judgements",
        "translated_judgements", "utility_calls", "b5_student", "revision_e2_judgements", "revision_e3_fidelity",
        "revision_e3_ablation_generations", "revision_e3_ablation_judgements", "revision_e4_generations",
        "revision_e4_judgements", "sens_template_generations", "sens_template_judgements"]
agg = collections.defaultdict(lambda: {"calls": 0, "ok": 0, "fallback": 0, "routes": collections.Counter(),
                                       "requested": collections.Counter(), "returned": collections.Counter(),
                                       "dates": set(), "in": 0, "out": 0})
for name in LOGS:
    p = run.path("runs", f"{name}.jsonl")
    if not p.exists():
        continue
    last = {}
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        last[r["call_id"]] = r
    for r in last.values():
        a = agg[(name, r.get("judge") or r.get("model"), r.get("role"))]  # judge logs: model = tutor judged
        a["calls"] += 1
        a["ok"] += r.get("status") == "ok"
        a["fallback"] += bool(r.get("fallback_used"))
        a["routes"][str(r.get("route_used")).split(":")[0]] += 1
        a["requested"][str(r.get("model_requested"))] += 1
        a["returned"][str(r.get("model_returned"))] += 1
        if r.get("timestamp"):
            a["dates"].add(str(r["timestamp"])[:10])
        a["in"] += int(r.get("input_tokens") or 0)
        a["out"] += int(r.get("output_tokens") or 0)
rows = []
for (log, model, role), a in sorted(agg.items(), key=lambda x: (x[0][0], str(x[0][1]))):
    fmt = lambda c: "; ".join(f"{k} ({v / a['calls']:.1%})" for k, v in c.most_common(4))  # noqa: E731
    d = sorted(a["dates"])
    rows.append({"log": log, "model": model, "role": role, "calls": a["calls"], "ok": a["ok"],
                 "fallback_share": a["fallback"] / a["calls"], "routes": fmt(a["routes"]),
                 "model_requested": fmt(a["requested"]), "model_returned": fmt(a["returned"]),
                 "first_date": d[0] if d else "", "last_date": d[-1] if d else "",
                 "input_tokens": a["in"], "output_tokens": a["out"]})
t = pd.DataFrame(rows)
t.to_csv(run.path("results", "tables", "revision_provenance.csv"), index=False)
print(t[["log", "model", "role", "calls", "fallback_share", "routes", "first_date", "last_date"]]
      .to_string(index=False, max_colwidth=60))
