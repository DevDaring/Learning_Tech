"""Stage 5: mask names, blind-judge every generation with three judges, aggregate.

The judge never sees the student's message (Section 6.2): it gets the canonical English
question, answers and misconception, the language label, and the tutor reply with every
listed name replaced by [STUDENT]. Parsing: strict JSON -> utility syntax repair (logged)
-> one fresh judge call if still unparseable (logged). Invalid labels are left missing,
never guessed. Outputs:
  runs/judgements.jsonl                one line per (response, judge) call
  results/judgements_long.csv          normalised labels per (response, judge)
  results/judgements_aggregated.csv    aggregated labels + PQI per response
"""
from __future__ import annotations

import logging
import math

import pandas as pd

from ..clients import make_llm
from ..config import Run, stable_hash
from ..io import AppendLog, fill, prompt, read_jsonl
from ..jsonfix import parse_with_repair
from ..names import mask_names, residual_names
from ..rubric import DIMENSIONS, aggregate, component_scores, normalise_scores, pqi, pqi_reliable
from ..runner import call_record, execute
from .s04_generate import current_generations

log = logging.getLogger("tlt.s05")

TARGET_META = ["target_id", "block", "item_id", "source", "condition", "student_lang", "reply_lang", "cue",
               "model", "sample_index", "first_name", "surname", "arm", "variant", "subject"]


def definitions_block(dims: str = "all") -> str:
    text = prompt("judge_definitions")
    if dims == "mrbench":
        text = text.split("Dimensions 9-14")[0].rstrip()
    return text


def judge_prompt(t: dict) -> tuple[str, int]:
    masked, n = mask_names(t["response_text"])
    p = fill(prompt("judge"), definitions_block=definitions_block(), question_en=t["judge_question_en"],
             correct_answer=t["judge_correct_answer"], student_answer=t["judge_student_answer"],
             misconception_name=t["misconception_name"], language_label=t["language_label"],
             tutor_response_masked=masked)
    return p, n


def targets_from_generations(gens: list[dict], prompts: dict[str, dict]) -> list[dict]:
    out = []
    for g in gens:
        if g["status"] != "ok":
            continue
        p = prompts[g["prompt_id"]]
        t = {k: g.get(k) for k in TARGET_META if k in g}
        t.update({"target_id": g["call_id"], "response_text": g["raw_text"] or "",
                  "judge_question_en": p["judge_question_en"], "judge_correct_answer": p["judge_correct_answer"],
                  "judge_student_answer": p["judge_student_answer"],
                  "misconception_name": p["misconception_name"], "language_label": p["language_label"],
                  "arm": "R0", "variant": "original"})
        out.append(t)
    return out


def judge_targets(run: Run, llm, targets: list[dict], applog: AppendLog, label: str) -> dict:
    """Judge every target with every judge. Returns (counts, ids of the current judge calls).
    The call id includes a hash of the exact judge prompt, so editing the rubric, the
    definitions or a target's fields triggers re-judging instead of reusing old scores."""
    judges = run.judges
    jobs = []
    for t in targets:
        user, n_masked = judge_prompt(t)
        js = run.models["judge_settings"]
        h = stable_hash(user, js["temperature"], js["max_tokens"], n=10)
        for j in judges:
            jobs.append({"call_id": f"judge|{t['target_id']}|{t.get('variant', 'original')}|{j}|{h}", "model": j,
                         "target": t, "user": user, "n_masked": n_masked})

    def fn(job):
        t, j = job["target"], job["model"]
        user, n_masked = job["user"], job["n_masked"]
        res = llm.call(j, user, None, seed_key=job["call_id"])
        obj, info = parse_with_repair(res.text, llm, job["call_id"]) if res.ok else (None, {})
        retried = False
        if res.ok and obj is None:  # one fresh judge call, logged
            retried = True
            res2 = llm.call(j, user, None, seed_key=job["call_id"] + "|retry")
            if res2.ok:
                obj2, info2 = parse_with_repair(res2.text, llm, job["call_id"] + "|retry")
                res, obj, info = res2, obj2, {**info2, "first_attempt_unparseable": True}
        scores, bad = normalise_scores(obj or {}, list(DIMENSIONS))
        meta = {k: t.get(k) for k in TARGET_META}
        meta.update({"call_id": job["call_id"], "judge": j, "role": "judge", "names_masked": n_masked,
                     "residual_names": residual_names(mask_names(t["response_text"])[0]),
                     "parse_ok": obj is not None, "invalid_dims": bad, "judge_retried": retried,
                     "evidence": (obj or {}).get("evidence"), "scores": scores, **info})
        rec = call_record(meta, res)
        if res.ok and obj is None:
            rec["status"] = "unparseable"
        return rec

    w = {j: int(run.models["models"][j].get("workers", 4)) for j in judges}
    return execute(jobs, applog, fn, w, label), {j["call_id"] for j in jobs}


def long_table(recs: list[dict]) -> pd.DataFrame:
    rows = []
    for r in recs:
        row = {k: r.get(k) for k in TARGET_META}
        row.update({"judge": r["judge"], "status": r["status"], "parse_ok": r.get("parse_ok"),
                    "json_repair_fired": r.get("json_repair_fired"), "judge_route": r.get("route_used"),
                    "judge_fallback": r.get("fallback_used"), "judge_model_returned": r.get("model_returned")})
        for d in DIMENSIONS:
            row[d] = (r.get("scores") or {}).get(d)
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_table(long: pd.DataFrame, primary: str, judges: list[str]) -> pd.DataFrame:
    rows = []
    key = ["target_id", "variant"]
    for (tid, var), g in long.groupby(key, sort=False, dropna=False):
        base = g.iloc[0]
        row = {k: base.get(k) for k in TARGET_META}
        by_judge = {j: g[g.judge == j].iloc[0] if (g.judge == j).any() else None for j in judges}
        labels = {}
        for d in DIMENSIONS:
            lab = {j: (None if r is None or (isinstance(r[d], float) and math.isnan(r[d])) else r[d])
                   for j, r in by_judge.items()}
            labels[d] = aggregate(d, lab, primary)
            row[d] = labels[d]
            for j in judges:
                row[f"{d}__{j}"] = lab[j]
        row["n_judges_valid"] = int(sum(r is not None and bool(r["parse_ok"] == True) for r in by_judge.values()))
        row["PQI"] = pqi(labels)
        row["PQI_R"] = pqi_reliable(labels)
        for j in judges:
            r = by_judge[j]
            row[f"PQI__{j}"] = pqi({d: (r[d] if r is not None and isinstance(r[d], str) else None)
                                    for d in DIMENSIONS}) if r is not None else math.nan
        row.update(component_scores(labels))
        rows.append(row)
    return pd.DataFrame(rows)


def write_tables(run: Run, applog: AppendLog, stem: str, keep_ids: set[str]) -> pd.DataFrame:
    """Tables from the current judge calls only (``keep_ids``); superseded records stay in
    the append-only log but never reach an analysis table."""
    recs = [r for r in applog.records() if r["call_id"] in keep_ids]
    if not recs:
        return pd.DataFrame()
    long = long_table(recs)
    long.to_csv(run.path("results", f"{stem}_long.csv"), index=False)
    agg = aggregate_table(long, run.design["primary_judge"], run.judges)
    agg.to_csv(run.path("results", f"{stem}_aggregated.csv"), index=False)
    return agg


def run_stage(run: Run) -> dict:
    gens = current_generations(run)
    prompts = {p["prompt_id"]: p for p in read_jsonl(run.path("data", "prompts.jsonl"))}
    if not gens:
        raise SystemExit("runs/generations.jsonl empty: run generate first")
    llm = make_llm(run)
    applog = AppendLog(run.path("runs", "judgements.jsonl"))
    targets = targets_from_generations(gens, prompts)
    counts, ids = judge_targets(run, llm, targets, applog, "judge")
    agg = write_tables(run, applog, "judgements", ids)
    recs = [r for r in applog.records() if r["call_id"] in ids]
    counts.update({"targets": len(targets), "judge_records": len(recs),
                   "unparseable": sum(r["status"] == "unparseable" for r in recs),
                   "json_repair_fired": sum(bool(r.get("json_repair_fired")) for r in recs),
                   "residual_name_hits": sum(bool(r.get("residual_names")) for r in recs),
                   "pqi_missing": int(agg["PQI"].isna().sum()) if len(agg) else 0})
    log.info("judge: %s", counts)
    return counts
