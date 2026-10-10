"""Sensitivity of the results to the Hindi student-message template.

The native Hindi reviewer suggested a chattier wording ("कि" dropped, "ये" for "यह"). Instead
of switching templates mid-study, every Block A Hindi prompt (300 items, no cue) is rebuilt
with the suggested wording and answered once by each tutor; the replies are judged by the same
panel and compared, paired by item and tutor, with sample 0 under the original wording.
Separate logs; nothing in the main study is changed.

    PYTHONPATH=src python scripts/template_sensitivity.py
"""
import pandas as pd

from tlt.analysis.stats import add_holm, gap
from tlt.clients import make_llm
from tlt.config import load_run
from tlt.io import AppendLog, fill, read_jsonl, read_table
from tlt.runner import execute
from tlt.stages.s04_generate import generation_jobs, make_fn, workers
from tlt.stages.s05_mask_and_judge import aggregate_table, judge_targets, long_table, targets_from_generations

ALT = "{question}\n\nमुझे लगता है उत्तर {student_answer} है। क्या ये सही है?"

run = load_run("full")
R = run.path("results")
items = {it["item_id"]: it for it in read_jsonl(run.path("data", "items_multilingual.jsonl"))}
prompts = [p for p in read_jsonl(run.path("data", "prompts.jsonl"))
           if p["block"] == "A" and p["condition"] == "HI" and p["cue"] == "C0"]
alt = []
for p in prompts:
    v = items[p["item_id"]]["versions"]["HI"]
    q = dict(p)
    q.update({"prompt_id": f"S|{p['item_id']}|HI|C0", "block": "S", "samples": 1,
              "user": fill(ALT, question=v["question"], student_answer=v["student_answer"])})
    alt.append(q)
print("prompts:", len(alt))
llm = make_llm(run)
glog = AppendLog(run.path("runs", "sens_template_generations.jsonl"))
jobs = generation_jobs(run, alt)
print("generation:", execute(jobs, glog, make_fn(run, llm), workers(run, run.producers), "sens-generate"))
ids = {j["call_id"] for j in jobs}
gens = [r for r in glog.records() if r["call_id"] in ids]
jlog = AppendLog(run.path("runs", "sens_template_judgements.jsonl"))
targets = targets_from_generations(gens, {q["prompt_id"]: q for q in alt})
counts, jids = judge_targets(run, llm, targets, jlog, "sens-judge")
print("judging:", counts)
alt_agg = aggregate_table(long_table([r for r in jlog.records() if r["call_id"] in jids]),
                          run.design["primary_judge"], run.judges)

base = read_table(R / "judgements_aggregated.csv", low_memory=False)
base = base[(base.block == "A") & (base.condition == "HI") & (base.cue == "C0") & (base.variant == "original")
            & (base.sample_index.astype(str) == "0")]
CD = {"High": 1.0, "Medium": 0.5, "Low": 0.0, "None": 0.0}
df = pd.concat([base.assign(tmpl="original"), alt_agg.assign(tmpl="suggested")], ignore_index=True)
df = df.assign(inflated=(df.Praise_Calibration == "Inflated").astype(float).where(df.Praise_Calibration.notna()),
               lowered=df.Expectation_Level.isin(["Simplified below", "Condescending"]).astype(float)
               .where(df.Expectation_Level.notna()))
rows = []
for v in ("PQI", "PQI_R", "inflated", "lowered"):
    for m in sorted(df.model.unique()) + [None]:
        r = gap(df, v, "tmpl", "original", "suggested", producer=m, B=int(run.design["bootstrap"]["resamples"]),
                seed=run.seed)
        rows.append({"family": f"template_{v}", "outcome": v, "model": m or "ALL", **r})
out = add_holm(pd.DataFrame(rows))
out.to_csv(R / "tables" / "extra_template_sensitivity.csv", index=False)
print(out[out.model == "ALL"][["outcome", "estimate", "ci_low", "ci_high", "p_holm"]].round(4).to_string(index=False))
print("mean PQI original %.3f suggested %.3f" % (base.PQI.mean(), alt_agg.PQI.mean()))
