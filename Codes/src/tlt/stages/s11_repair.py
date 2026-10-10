"""Stage 11 (RQ5): repair arms R1-R5 on the repair subset, then blind judging as in stage 5.

Arms (Section 8):
  R1 fairness instruction, R2 pedagogy instruction (appended to the default system prompt)
  R3 Bridge strategy-guided prompt (B6): only the REPLY section is judged; decisions logged
  R4 translate pipeline: message -> English, tutor in English, reply -> student language
  R5 plan-then-realise: canonicalise (drop cue, translate to English), plan (JSON), realise
R0 is reused from Blocks A/B. Translation routes: IndicTrans2 for HI/BN; the utility model
for code-mixed text (IndicTrans2 does not accept romanised input; D7).
"""
from __future__ import annotations

import json
import random
import logging
import re

from ..clients import make_llm
from ..config import Run, stable_hash
from ..io import AppendLog, fill, prompt, read_jsonl
from ..jsonfix import parse_with_repair
from ..names import mask_names
from ..runner import cached_call, call_record, execute, protected_utility
from ..translate import make_translator
from .s02_translate import CM_SOURCE, CODEMIX
from .s04_generate import META_KEYS
from .s05_mask_and_judge import judge_targets, write_tables

log = logging.getLogger("tlt.s11")


def parse_bridge(text: str) -> tuple[str, dict]:
    """Split a Bridge-format answer into (reply, decisions). If the REPLY marker is
    missing the whole text is judged and the failure is recorded."""
    dec = {}
    for k in ("ERROR TYPE", "STRATEGY", "INTENTION"):
        m = re.search(rf"^\s*\**{k}\**\s*:\s*(.+)$", text or "", re.M | re.I)
        dec[k.lower().replace(" ", "_")] = m.group(1).strip() if m else None
    m = re.search(r"^\s*\**REPLY\**\s*:\s*\n?(.*)\Z", text or "", re.S | re.M | re.I)
    dec["bridge_format_ok"] = bool(m) and all(dec[k] for k in ("error_type", "strategy", "intention"))
    return (m.group(1).strip() if m else (text or "").strip()), dec


ARM_TEMPLATES = {"R1": ["tutor_system", "repair_R1_instruction"], "R2": ["tutor_system", "repair_R2_instruction"],
                 "R3": ["repair_R3_bridge"], "R4": ["tutor_system", "utility_translate_to_en", "utility_codemix_reply"],
                 "R5": ["repair_plan", "repair_realise", "utility_translate_to_en"],
                 # revision baselines (docs/revision_plan_frozen.md, E1)
                 "R2L": ["tutor_system", "repair_R2_instruction", "repair_R2L_language"],
                 "R2T": ["tutor_system", "repair_R2_instruction", "utility_translate_to_en", "utility_codemix_reply"],
                 # contemporaneous re-runs of R2 / R5 to estimate drift (E1); separate arm names so
                 # they never mix with the original arms in any table
                 "R2-DRIFT": ["tutor_system", "repair_R2_instruction"],
                 "R5-DRIFT": ["repair_plan", "repair_realise", "utility_translate_to_en"]}
SAME_AS = {"R2-DRIFT": "R2", "R5-DRIFT": "R5"}


# Version of the R5 plan parser written into every R5 record. Version 1 (records without the
# field) could return a nested object in place of an invalid outer plan (D46).
PLAN_PARSER = 2


def stale_r5_ids(run: Run) -> set[str]:
    """R5 replies realised from a plan the version-1 parser mis-read: they fail the plan
    schema and carry no parser version. They are redone under a '|reparse1' id, reusing the
    cached plan text (same plan call, same seed), so only the parse and the realise step
    change."""
    return {r["call_id"] for r in AppendLog(run.path("runs", "repair_generations.jsonl")).records()
            if r.get("arm") == "R5" and "plan_parser" not in r and r.get("plan_schema_ok") is False}


def repair_jobs(run: Run, cells: list[dict], prompts: dict[str, dict], stale: set[str] | None = None) -> list[dict]:
    """Call ids carry a hash of the base prompt and the arm's templates, so editing either
    regenerates the reply instead of reusing an old one."""
    stale = stale_r5_ids(run) if stale is None else stale
    jobs = []
    for c in cells:
        st = run.models["producer_settings"]
        h = stable_hash(prompts[c["base_prompt_id"]]["user"], *(prompt(t) for t in ARM_TEMPLATES[c["arm"]]),
                        st["temperature"], st["max_tokens"], n=10)
        for m in run.producers:
            base = f"repair|{c['arm']}|{c['base_prompt_id']}|{m}|{h}"
            cid = f"{base}|reparse1" if base in stale else base
            jobs.append({"call_id": cid, "base_key": base, "model": m, "cell": c})
    return jobs


def current_repair_records(run: Run, ok_only: bool = False) -> list[dict]:
    items = read_jsonl(run.path("data", "items_multilingual.jsonl"))
    prompts = {p["prompt_id"]: p for p in read_jsonl(run.path("data", "prompts.jsonl"))}
    ids = {j["call_id"] for j in repair_jobs(run, repair_cells(run, items), prompts)}
    recs = [r for r in AppendLog(run.path("runs", "repair_generations.jsonl")).records() if r["call_id"] in ids]
    return [r for r in recs if r["status"] == "ok"] if ok_only else recs


PLAN_KEYS = ("diagnosis", "next_step", "question_to_ask", "do_not_reveal", "cognitive_level")


def plan_schema_ok(plan: dict | None) -> bool:
    return isinstance(plan, dict) and all(k in plan and str(plan[k]).strip() for k in PLAN_KEYS)


def repair_cells(run: Run, items: list[dict]) -> list[dict]:
    r = run.design["repair"]
    langs = list(r["languages"]) + (["EN"] if r.get("include_english_reference") else [])
    cells = []
    for it in (x for x in items if x.get("in_repair")):
        for lang in langs:
            for arm in r["arms"]:
                cells.append({"item_id": it["item_id"], "lang": lang, "cue": "C0", "arm": arm,
                              "base_prompt_id": f"A|{it['item_id']}|{lang}|C0"})
        for lang in r.get("revision_languages", []):
            for arm in r.get("revision_arms", []):
                cells.append({"item_id": it["item_id"], "lang": lang, "cue": "C0", "arm": arm,
                              "base_prompt_id": f"A|{it['item_id']}|{lang}|C0"})
        for cue in r["identity_cues"]:
            for arm in r["identity_arms"]:
                if arm == "R0":
                    continue  # reused from Block B
                cells.append({"item_id": it["item_id"], "lang": r["identity_language"], "cue": cue, "arm": arm,
                              "base_prompt_id": f"B|{it['item_id']}|{r['identity_language']}|{cue}"})
    dr = r.get("drift_check")
    if dr:
        pool = sorted(x["item_id"] for x in items if x.get("in_repair"))
        chosen = sorted(random.Random(run.seed + int(dr["seed_offset"])).sample(pool, int(dr["items"])))
        for iid in chosen:
            for lang in dr["languages"]:
                for arm in dr["arms"]:
                    cells.append({"item_id": iid, "lang": lang, "cue": "C0", "arm": f"{arm}-DRIFT",
                                  "base_prompt_id": f"A|{iid}|{lang}|C0"})
    return cells


class Repair:
    def __init__(self, run: Run, prompts: dict[str, dict]):
        self.run, self.prompts = run, prompts
        self.llm, self.tr = make_llm(run), make_translator(run)
        self.steps = AppendLog(run.path("runs", "repair_steps.jsonl"))
        self.sys0 = prompt("tutor_system")
        self.lang_names = run.design["languages"]["names"]

    def to_english(self, text: str, lang: str, key: str) -> tuple[str, str]:
        if lang == "EN":
            return text, "identity"
        if lang in ("HI", "BN"):
            return self.tr.indic_to_en([text], lang)[0], self.tr.model_id_indic_en
        out, rec, _ = protected_utility(self.llm, self.steps, f"to_en|{key}", "utility_translate_to_en", "text", text,
                                        {"role": "repair_canonicalise"}, source_description=CM_SOURCE[lang])
        return out, rec.get("route_used") or "FAILED"

    def from_english(self, text: str, lang: str, key: str) -> tuple[str, str]:
        if lang == "EN":
            return text, "identity"
        if lang in ("HI", "BN"):
            return self.tr.en_to_indic([text], lang)[0], self.tr.model_id_en_indic
        region, matrix = CODEMIX[lang]
        out, rec, _ = protected_utility(self.llm, self.steps, f"from_en|{key}", "utility_codemix_reply", "text", text,
                                        {"role": "repair_back_translation"}, region=region, matrix_language=matrix)
        return out, rec.get("route_used") or "FAILED"

    def run_cell(self, job: dict) -> dict:
        c, m = job["cell"], job["model"]
        base = self.prompts[c["base_prompt_id"]]
        meta = {k: base.get(k) for k in META_KEYS}
        meta.update({"call_id": job["call_id"], "model": m, "role": "producer", "arm": c["arm"],
                     "sample_index": 0, "block": "R", "base_prompt_id": c["base_prompt_id"],
                     "repair_lang": c["lang"]})
        # the plan cache and seeds use the base id, so a '|reparse1' job reuses the same plan call
        arm, user, key = c["arm"], base["user"], job.get("base_key", job["call_id"])
        arm = SAME_AS.get(arm, arm)  # drift arms run exactly like R2 / R5 (fresh calls: distinct ids)
        extra: dict = {}
        if arm in ("R1", "R2"):
            system = f"{self.sys0}\n{prompt(f'repair_{arm}_instruction')}"
            res = self.llm.call(m, user, system, seed_key=key)
            text = res.text
        elif arm == "R2L":
            system = (f"{self.sys0}\n{prompt('repair_R2_instruction')}\n"
                      + fill(prompt("repair_R2L_language"), target_language_and_script=self.lang_names[c["lang"]]))
            res = self.llm.call(m, user, system, seed_key=key)
            text = res.text
        elif arm == "R2T":
            # the same canonical English message as R4 (shared by all producers), R2's prompt in English
            system = f"{self.sys0}\n{prompt('repair_R2_instruction')}"
            en_msg, r1 = self.to_english(user, c["lang"], f"r4|{c['item_id']}|{c['lang']}|{c['cue']}")
            res = self.llm.call(m, en_msg, system, seed_key=key)
            text, r2 = self.from_english(res.text, c["lang"], key) if res.ok else ("", "")
            extra.update({"canonical_message_en": en_msg, "english_reply": res.text,
                          "to_en_route": r1, "from_en_route": r2})
        elif arm == "R3":
            system = prompt("repair_R3_bridge")
            res = self.llm.call(m, user, system, seed_key=key)
            text, dec = parse_bridge(res.text)
            extra.update(dec)
        elif arm == "R4":
            # One canonical English message per item x language (shared by all producers), so
            # every producer answers exactly the same input.
            en_msg, r1 = self.to_english(user, c["lang"], f"r4|{c['item_id']}|{c['lang']}|{c['cue']}")
            res = self.llm.call(m, en_msg, self.sys0, seed_key=key)
            text, r2 = self.from_english(res.text, c["lang"], key) if res.ok else ("", "")
            extra.update({"canonical_message_en": en_msg, "english_reply": res.text,
                          "to_en_route": r1, "from_en_route": r2})
        elif arm == "R5":
            # Stage 1: drop the cue sentence (use the cue-free Block A message) and mask names.
            canon_src = self.prompts[f"A|{c['item_id']}|{c['lang']}|C0"]["user"]
            canon_src, _ = mask_names(canon_src)
            en_msg, r1 = self.to_english(canon_src, c["lang"], f"{c['item_id']}|{c['lang']}")
            plan_user = fill(prompt("repair_plan"), canonical_message_en=en_msg)
            plan_rec = cached_call(self.llm, self.steps, f"plan|{key}", m, plan_user,
                                   {"role": "repair_plan", "item_id": c["item_id"]})
            plan, pinfo = parse_with_repair(plan_rec.get("raw_text") or "", self.llm, f"plan|{key}")
            plan_retried = False
            if plan_rec.get("status") == "ok" and not plan_schema_ok(plan):
                # One fresh plan call when the JSON lacks the five required keys (logged).
                plan_retried = True
                retry = cached_call(self.llm, self.steps, f"plan_retry|{key}", m, plan_user,
                                    {"role": "repair_plan_retry", "item_id": c["item_id"]})
                plan2, pinfo2 = parse_with_repair(retry.get("raw_text") or "", self.llm, f"plan_retry|{key}")
                if retry.get("status") == "ok" and (plan_schema_ok(plan2) or plan is None):
                    plan_rec, plan, pinfo = retry, plan2, pinfo2
            plan_json = json.dumps(plan, ensure_ascii=False) if plan is not None else (plan_rec.get("raw_text") or "")
            user2 = fill(prompt("repair_realise"), target_language_and_script=self.lang_names[c["lang"]],
                         plan_json=plan_json, original_message=user)
            if plan_rec.get("status") != "ok":
                res = None
            else:
                res = self.llm.call(m, user2, None, seed_key=key)
            text = res.text if res is not None else ""
            extra.update({"canonical_message_en": en_msg, "to_en_route": r1, "plan": plan,
                          "plan_raw": plan_rec.get("raw_text"), "plan_parse_ok": plan is not None,
                          "plan_schema_ok": plan_schema_ok(plan), "plan_retried": plan_retried,
                          "plan_parser": PLAN_PARSER,
                          "plan_json_repair_fired": pinfo.get("json_repair_fired"),
                          "plan_route": plan_rec.get("route_used"), "realise_user": user2})
            if res is None:
                from ..clients import CallResult
                res = CallResult(ok=False, error=f"plan call failed: {plan_rec.get('error')}")
        else:
            raise ValueError(arm)
        rec = call_record({**meta, "system": None if arm == "R5" else (system if arm != "R4" else self.sys0),
                           "user": user, **extra}, res)
        rec["response_text"] = text
        if res.ok and arm in ("R4", "R2T") and not text.strip() and res.text.strip():
            rec["status"] = "error"
            rec["error"] = "back-translation failed"
        return rec


def run_stage(run: Run) -> dict:
    items = read_jsonl(run.path("data", "items_multilingual.jsonl"))
    prompts = {p["prompt_id"]: p for p in read_jsonl(run.path("data", "prompts.jsonl"))}
    cells = repair_cells(run, items)
    rp = Repair(run, prompts)
    # Batch the IndicTrans2 canonicalisation up front (R4 and R5 then hit the cache).
    for lang in ("HI", "BN"):
        msgs = sorted({prompts[c["base_prompt_id"]]["user"] for c in cells if c["lang"] == lang and c["arm"] == "R4"}
                      | {mask_names(prompts[f"A|{c['item_id']}|{lang}|C0"]["user"])[0]
                         for c in cells if c["lang"] == lang and c["arm"] == "R5"})
        if msgs:
            rp.tr.indic_to_en(msgs, lang)
    applog = AppendLog(run.path("runs", "repair_generations.jsonl"))
    jobs = repair_jobs(run, cells, prompts)
    counts = execute(jobs, applog, rp.run_cell,
                     {m: int(run.models["models"][m].get("workers", 4)) for m in run.producers}, "repair")
    # Judge the repaired replies exactly as in stage 5.
    ids = {j["call_id"] for j in jobs}
    targets = []
    for g in applog.records():
        if g["call_id"] not in ids or g["status"] != "ok":
            continue
        p = prompts[g["base_prompt_id"]]
        targets.append({"target_id": g["call_id"], "block": "R", "item_id": g["item_id"], "source": g["source"],
                        "condition": g["condition"], "student_lang": g["student_lang"],
                        "reply_lang": g["student_lang"], "cue": g["cue"], "model": g["model"],
                        "sample_index": 0, "arm": g["arm"], "variant": "original",
                        "first_name": g.get("first_name"), "surname": g.get("surname"),
                        "subject": g.get("subject"), "response_text": g.get("response_text") or "",
                        "judge_question_en": p["judge_question_en"], "judge_correct_answer": p["judge_correct_answer"],
                        "judge_student_answer": p["judge_student_answer"],
                        "misconception_name": p["misconception_name"], "language_label": p["language_label"]})
    jlog = AppendLog(run.path("runs", "repair_judgements.jsonl"))
    jc, jids = judge_targets(run, rp.llm, targets, jlog, "repair-judge")
    write_tables(run, jlog, "repair_judgements", jids)
    recs = [r for r in applog.records() if r["call_id"] in ids]
    counts.update({"cells": len(cells), "generations": len(jobs), "judged": jc,
                   "r3_format_failures": sum(r.get("arm") == "R3" and not r.get("bridge_format_ok") for r in recs),
                   "r5_plan_parse_failures": sum(r.get("arm") == "R5" and not r.get("plan_parse_ok") for r in recs),
                   "r5_plan_schema_failures": sum(r.get("arm") == "R5" and not r.get("plan_schema_ok") for r in recs),
                   "r5_plan_retried": sum(bool(r.get("plan_retried")) for r in recs)})
    log.info("repair: %s", counts)
    return counts
