"""Stage 4: tutor generations for Blocks A-D (append-only, resumable)."""
from __future__ import annotations

import logging

from ..clients import make_llm
from ..config import Run, stable_hash
from ..io import AppendLog, read_jsonl
from ..runner import call_record, execute

log = logging.getLogger("tlt.s04")

META_KEYS = ["prompt_id", "block", "item_id", "source", "condition", "student_lang", "reply_lang", "cue",
             "first_name", "surname", "subject"]


def generation_jobs(run: Run, prompts: list[dict], producers: list[str] | None = None) -> list[dict]:
    jobs = []
    for p in prompts:
        for m in producers or run.producers:
            for s in range(int(p["samples"])):
                st = run.models["producer_settings"]
                # the generation settings are part of the identity: changing the cap or the
                # temperature regenerates instead of reusing replies made under the old one
                h = stable_hash(p["system"], p["user"], st["temperature"], st["max_tokens"], n=10)
                jobs.append({"call_id": f"gen|{p['prompt_id']}|{m}|s{s}|{h}", "model": m, "sample": s, "prompt": p})
    return jobs


def current_generations(run: Run, ok_only: bool = False) -> list[dict]:
    """Generation records for the CURRENT prompts and producers only: a record made from
    an older version of a prompt (e.g. before a translation correction) is ignored."""
    prompts = read_jsonl(run.path("data", "prompts.jsonl"))
    ids = {j["call_id"] for j in generation_jobs(run, prompts)}
    recs = [r for r in AppendLog(run.path("runs", "generations.jsonl")).records() if r["call_id"] in ids]
    return [r for r in recs if r["status"] == "ok"] if ok_only else recs


def make_fn(run: Run, llm):
    def fn(job: dict) -> dict:
        p = job["prompt"]
        res = llm.call(job["model"], p["user"], p["system"], seed_key=job["call_id"])
        meta = {k: p.get(k) for k in META_KEYS}
        meta.update({"call_id": job["call_id"], "model": job["model"], "role": "producer",
                     "sample_index": job["sample"], "system": p["system"], "user": p["user"]})
        return call_record(meta, res)
    return fn


def workers(run: Run, models: list[str]) -> dict[str, int]:
    return {m: int(run.models["models"][m].get("workers", 4)) for m in models}


def run_stage(run: Run) -> dict:
    prompts = read_jsonl(run.path("data", "prompts.jsonl"))
    if not prompts:
        raise SystemExit("data/prompts.jsonl missing: run build_prompts first")
    llm = make_llm(run)
    applog = AppendLog(run.path("runs", "generations.jsonl"))
    jobs = generation_jobs(run, prompts)
    counts = execute(jobs, applog, make_fn(run, llm), workers(run, run.producers), "generate")
    recs = applog.records()
    counts.update({"records": len(recs), "ok_total": sum(r["status"] == "ok" for r in recs),
                   "failed_total": sum(r["status"] != "ok" for r in recs),
                   "empty_text": sum(r["status"] == "ok" and not (r["raw_text"] or "").strip() for r in recs),
                   "truncated": sum(bool(r.get("truncated")) for r in recs),
                   "fallback_used": sum(bool(r.get("fallback_used")) for r in recs)})
    log.info("generate: %s", counts)
    return counts
