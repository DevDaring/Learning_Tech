"""Stage 6 (B4): judge validation on MRBench_V2 gold labels (Maurya et al., NAACL 2025).

A seeded random sample of tutor responses is rated by each judge with the MRBench variant
of the judge prompt (conversation history + ground-truth solution in place of our item
fields; dimensions 1-8 only; same definitions). Reports accuracy, macro-F1 and Cohen's
kappa per dimension for each judge and for the aggregate. Linear-weighted kappa is
added for ordinal dimensions.
"""
from __future__ import annotations

import json
import logging
import random

import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from ..clients import make_llm
from ..config import Run, stable_hash
from ..io import AppendLog, fill, prompt
from ..jsonfix import parse_with_repair
from ..rubric import CATEGORICAL, DIMENSIONS, MRBENCH_DIMS, aggregate, norm_label, normalise_scores
from ..runner import call_record, execute
from .s05_mask_and_judge import definitions_block

log = logging.getLogger("tlt.s06")

GOLD_KEYS = {"Humanlikeness": "humanlikeness"}


def load_mrbench(run: Run) -> list[dict]:
    data = json.load(open(run.source(run.design["mrbench"]["file"]), encoding="utf-8"))
    rows = []
    for conv in data:
        for tutor, rr in conv["anno_llm_responses"].items():
            ann = rr.get("annotation") or {}
            gold = {}
            for d in MRBENCH_DIMS:
                gold[d] = norm_label(d, ann.get(GOLD_KEYS.get(d, d), ann.get(d)))
            rows.append({"mr_id": f"{conv['conversation_id']}|{tutor}", "tutor": tutor, "data": conv.get("Data"),
                         "conversation_history": conv["conversation_history"],
                         "ground_truth_solution": conv["Ground_Truth_Solution"], "response": rr["response"],
                         **{f"gold__{d}": gold[d] for d in MRBENCH_DIMS}})
    return rows


def run_stage(run: Run) -> dict:
    rows = load_mrbench(run)
    n = min(int(run.design["mrbench"]["n_responses"]), len(rows))
    sample = random.Random(run.seed + 6).sample(sorted(rows, key=lambda r: r["mr_id"]), n)
    llm = make_llm(run)
    applog = AppendLog(run.path("runs", "mrbench_judgements.jsonl"))
    def user_for(r):
        return fill(prompt("judge_mrbench"), definitions_block=definitions_block("mrbench"),
                    conversation_history=r["conversation_history"],
                    ground_truth_solution=r["ground_truth_solution"], tutor_response=r["response"])

    jobs = [{"call_id": f"mrbench|{r['mr_id']}|{j}|{stable_hash(user_for(r), n=10)}", "model": j, "row": r,
             "user": user_for(r)} for r in sample for j in run.judges]
    current = {j["call_id"] for j in jobs}

    def fn(job):
        r = job["row"]
        user = job["user"]
        res = llm.call(job["model"], user, None, seed_key=job["call_id"])
        obj, info = parse_with_repair(res.text, llm, job["call_id"]) if res.ok else (None, {})
        scores, bad = normalise_scores(obj or {}, MRBENCH_DIMS)
        rec = call_record({"call_id": job["call_id"], "model": job["model"], "judge": job["model"],
                           "role": "judge_validation", "mr_id": r["mr_id"], "scores": scores,
                           "invalid_dims": bad, "parse_ok": obj is not None, **info}, res)
        if res.ok and obj is None:
            rec["status"] = "unparseable"
        return rec

    counts = execute(jobs, applog, fn, {j: int(run.models["models"][j].get("workers", 4)) for j in run.judges},
                     "mrbench")
    recs = [r for r in applog.records() if r["call_id"] in current]
    pred = {(r["mr_id"], r["judge"]): (r.get("scores") or {}) for r in recs}
    out = []
    for who in run.judges + ["aggregate"]:
        for d in MRBENCH_DIMS:
            g, p = [], []
            for r in sample:
                if who == "aggregate":
                    lab = aggregate(d, {j: pred.get((r["mr_id"], j), {}).get(d) for j in run.judges},
                                    run.design["primary_judge"])
                else:
                    lab = pred.get((r["mr_id"], who), {}).get(d)
                if lab is None or r[f"gold__{d}"] is None:
                    continue
                g.append(r[f"gold__{d}"])
                p.append(lab)
            labels = DIMENSIONS[d]
            row = {"judge": who, "dimension": d, "n": len(g)}
            if g:
                row["accuracy"] = accuracy_score(g, p)
                row["macro_f1"] = f1_score(g, p, labels=labels, average="macro", zero_division=0)
                row["cohen_kappa"] = cohen_kappa_score(g, p, labels=labels) if len(set(g) | set(p)) > 1 else float("nan")
                if d not in CATEGORICAL and len(set(g) | set(p)) > 1:
                    row["kappa_linear"] = cohen_kappa_score(g, p, labels=labels, weights="linear")
                row["gold_distribution"] = json.dumps(pd.Series(g).value_counts().to_dict())
            out.append(row)
    df = pd.DataFrame(out)
    df.to_csv(run.path("results", "judge_validation.csv"), index=False)
    counts.update({"sampled": n, "unparseable": sum(r["status"] == "unparseable" for r in recs)})
    log.info("judge validation: %s", counts)
    return counts
