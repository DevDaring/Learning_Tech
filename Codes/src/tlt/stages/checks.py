"""Acceptance checks (Section 11). Writes results/acceptance.json; exits non-zero on failure."""
from __future__ import annotations

import json
import logging
import random
import re

import pandas as pd

from .. import secrets
from ..config import CODES_DIR, DOCS_DIR, Run
from ..io import AppendLog, read_jsonl
from .s04_generate import current_generations
from ..names import loose_name_candidates, mask_names, residual_names

log = logging.getLogger("tlt.checks")


def _strip_cue(p: dict) -> str:
    if p["cue"] == "C0":
        return p["user"]
    return p["user"].split("\n\n", 1)[1]


def paired_prompt_diff(prompts: list[dict], n: int, seed: int) -> dict:
    """Every pair in a planned comparison must differ only in the intended factor."""
    by = {p["prompt_id"]: p for p in prompts}
    pairs = []
    for p in prompts:
        if p["block"] == "B":
            pairs.append(("cue", by.get(f"A|{p['item_id']}|{p['condition']}|C0"), p))
            for q_cue in ("N-MD", "N-FD", "N-MM", "N-FM", "S-ENG", "S-VER"):
                q = by.get(f"B|{p['item_id']}|{p['condition']}|{q_cue}")
                if q and q_cue != p["cue"]:
                    pairs.append(("cue", p, q))
        if p["block"] == "C":
            pairs.append(("reply_instruction", by.get(f"A|{p['item_id']}|EN|C0"), p))
        if p["block"] == "A" and p["condition"] != "EN":
            pairs.append(("language", by.get(f"A|{p['item_id']}|EN|C0"), p))
    pairs = [x for x in pairs if x[1] is not None]
    sample = random.Random(seed).sample(pairs, min(n, len(pairs)))
    fails = []
    for kind, a, b in sample:
        if kind == "cue":
            ok = a["system"] == b["system"] and _strip_cue(a) == _strip_cue(b)
            na, nb = (a.get("first_name"), a.get("surname")), (b.get("first_name"), b.get("surname"))
            if ok and a["cue"].startswith("N-") and b["cue"].startswith("N-"):
                same_gender = a["cue"][2] == b["cue"][2]
                same_caste = a["cue"][3] == b["cue"][3]
                ok = (na[0] == nb[0]) == same_gender and (na[1] == nb[1]) == same_caste
        elif kind == "reply_instruction":
            ok = a["user"] == b["user"] and b["system"].startswith(a["system"] + "\n") \
                and b["system"].count("\n") == a["system"].count("\n") + 1
        else:  # language: system identical, and judge-facing canonical fields identical
            ok = a["system"] == b["system"] and all(a[k] == b[k] for k in (
                "judge_question_en", "judge_correct_answer", "judge_student_answer", "misconception_name"))
        if not ok:
            fails.append({"kind": kind, "a": a["prompt_id"], "b": b["prompt_id"]})
    return {"pairs_tested": len(sample), "failures": fails, "pass": not fails and len(sample) > 0}


def masking_scan(run: Run) -> dict:
    hits, loose = [], []
    n = 0
    for name in ("judgements", "repair_judgements", "translated_judgements"):
        for r in AppendLog(run.path("runs", f"{name}.jsonl")).records():
            n += 1
            if r.get("residual_names"):
                hits.append(r["call_id"])
    for g in current_generations(run):
        masked = mask_names(g.get("raw_text") or "")[0]
        if residual_names(masked):
            hits.append(g["call_id"])
        loose += loose_name_candidates(masked)
    return {"judge_records_scanned": n, "residual_hits": hits[:50], "n_residual": len(hits),
            "loose_candidates_for_review": sorted(set(loose))[:100], "pass": not hits}


def secret_scan(run: Run) -> dict:
    vals = secrets.all_secret_values()
    bad = []
    for d in ("runs", "results", "review", "data"):
        base = run.path(d, "x").parent
        for f in base.rglob("*"):
            if f.is_file() and f.suffix in (".jsonl", ".csv", ".json", ".md", ".txt", ".log"):
                text = f.read_text(encoding="utf-8", errors="ignore")
                if any(v in text for v in vals):
                    bad.append(str(f))
    return {"files_with_secret_values": bad, "pass": not bad}


def run_stage(run: Run) -> dict:
    res = {}
    vlog = DOCS_DIR / "verification_log.md"
    vtext = vlog.read_text(encoding="utf-8") if vlog.exists() else ""
    res["verification_log"] = {"exists": vlog.exists(),
                               "automated_section_present": "## Automated checks" in vtext,
                               "unresolved_mentions": len(re.findall(r"UNRESOLVED", vtext)),
                               "pass": vlog.exists() and "## Automated checks" in vtext}
    prompts = read_jsonl(run.path("data", "prompts.jsonl"))
    res["paired_prompt_diff"] = paired_prompt_diff(prompts, 200, run.seed) if prompts else {"pass": False}
    res["masking_scan"] = masking_scan(run)
    auto = run.path("results", "auto_metrics.csv")
    gens = current_generations(run)
    if auto.exists():
        a = pd.read_csv(auto)
        res["refusals_and_empty_counted"] = {"generations": len(gens), "auto_metric_rows": len(a),
                                             "empty": int(a.is_empty.sum()), "refusal": int(a.refusal.sum()),
                                             "failed_calls": int((a.status != "ok").sum()),
                                             "pass": len(a) == len(gens)}
    op = run.path("results", "tables", "operational_counts.csv")
    res["repair_and_fallback_reported"] = {"operational_table": op.exists(), "pass": op.exists()}
    pilot_tables = CODES_DIR / "pilot" / "results" / "tables"
    res["pilot_completed"] = {"pass": pilot_tables.exists() and any(pilot_tables.glob("table4_*.csv"))
                              or run.mode != "full"}
    holm_ok, missing = True, []
    for f in run.path("results", "tables", "x").parent.glob("*.csv"):
        df = pd.read_csv(f, nrows=5)
        if "p_value" in df.columns and "p_holm" not in df.columns:
            holm_ok, missing = False, missing + [f.name]
    res["significance_uses_holm"] = {"tables_without_holm": missing, "pass": holm_ok}
    res["secret_scan"] = secret_scan(run)
    res["all_pass"] = all(v.get("pass", True) for v in res.values() if isinstance(v, dict))
    run.path("results", "acceptance.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    log.info("acceptance: all_pass=%s", res["all_pass"])
    return {k: v.get("pass") if isinstance(v, dict) else v for k, v in res.items()}
