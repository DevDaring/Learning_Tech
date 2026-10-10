"""Stage 3: every tutor prompt for Blocks A-D (Section 4.4, prompts in Section 7).

Each record carries what the generator needs (system, user) and what the judge needs
(canonical English question, answers, misconception, language label), so later stages
never re-derive a condition. Full mode refuses to build prompts while native-speaker
template approval, translation review or name validation is outstanding.
"""
from __future__ import annotations

import json
import logging

import pandas as pd

from ..config import DOCS_DIR, Run
from ..io import fill, prompt, prompt_yaml, read_jsonl, write_jsonl
from ..names import cue_names, render
from .s02_translate import pending_reviews

log = logging.getLogger("tlt.s03")

TEMPLATES_NEEDING_APPROVAL = (
    ["student_HI", "student_BN", "student_HI-EN", "student_BN-EN", "cue_name_HI", "cue_name_BN",
     "cue_S-ENG_HI", "cue_S-ENG_BN", "cue_S-VER_HI", "cue_S-VER_BN"]
    + [f"b5_control_{l}" for l in ("HI", "BN", "HI-EN", "BN-EN")]
    + [f"name_{n}_{l}" for n in ("Rahul", "Amit", "Arjun", "Sourav", "Priya", "Pooja", "Ananya", "Riya",
                                 "Sharma", "Trivedi", "Mukherjee", "Chatterjee", "Paswan", "Valmiki",
                                 "Jatav", "Bauri") for l in ("HI", "BN")]
)


def template_text(name: str) -> str:
    from ..names import SCRIPT_FORMS
    if name.startswith("b5_control_"):
        return prompt_yaml("b5_control_replies")[name.split("_", 2)[2]]
    if name.startswith("name_"):
        _, n, lang = name.split("_")
        return SCRIPT_FORMS[n][lang]
    return prompt(name)


def check_template_approval(run: Run) -> int:
    """Writes review/template_approval.csv if absent; returns number not yet approved."""
    p = run.path("review", "template_approval.csv")
    rows = [{"template": t, "text": template_text(t)} for t in TEMPLATES_NEEDING_APPROVAL]
    if p.exists():
        old = pd.read_csv(p, dtype=str, keep_default_na=False).set_index("template")
        for r in rows:
            o = old.loc[r["template"]] if r["template"] in old.index else None
            same = o is not None and o["text"] == r["text"]
            r["status"] = o["status"] if same else "pending"
            r["approver"] = o["approver"] if same else ""
            r["notes"] = o["notes"] if same else ""
    else:
        for r in rows:
            r.update(status="pending", approver="", notes="")
    pd.DataFrame(rows).to_csv(p, index=False)
    return sum(r["status"] != "approved" for r in rows)


def names_confirmed() -> bool:
    p = DOCS_DIR / "name_validation.md"
    import re
    return p.exists() and re.search(r"^STATUS: CONFIRMED\s*$", p.read_text(encoding="utf-8"), re.M) is not None


def student_message(lang: str, question: str, answer: str) -> str:
    return fill(prompt(f"student_{lang}"), question=question, student_answer=answer)


def cue_sentence(cue: str, lang: str, rank: int) -> tuple[str, dict]:
    if cue == "C0":
        return "", {}
    if cue in ("S-ENG", "S-VER"):
        return prompt(f"cue_{cue}_{lang}"), {}
    first, surname = cue_names(cue, rank)
    return (fill(prompt(f"cue_name_{lang}"), first=render(first, lang), surname=render(surname, lang)),
            {"first_name": first, "surname": surname})


def with_cue(cue_text: str, message: str) -> str:
    return f"{cue_text}\n\n{message}" if cue_text else message


def make_prompts(run: Run, items: list[dict]) -> list[dict]:
    d = run.design
    labels = d["languages"]["labels"]
    sys0 = prompt("tutor_system")
    out = []

    def rec(block, it, cond_lang, student_lang, cue, system, user, label, extra=None):
        en = it["versions"]["EN-OFFICIAL" if it["source"] == "benqa" else "EN"]
        r = {"prompt_id": f"{block}|{it['item_id']}|{cond_lang}|{cue}", "block": block,
             "item_id": it["item_id"], "source": it["source"], "condition": cond_lang,
             "student_lang": student_lang, "cue": cue, "system": system, "user": user,
             "judge_question_en": en["question"], "judge_correct_answer": en["correct_answer"],
             "judge_student_answer": en["student_answer"], "misconception_name": it["misconception_name"],
             "language_label": label,
             "correct_answer_student_lang": it["versions"][cond_lang]["correct_answer"]
             if cond_lang in it["versions"] else en["correct_answer"],
             "subject": it.get("subject"), "first_name": None, "surname": None}
        r.update(extra or {})
        out.append(r)

    eedi = [it for it in items if it["source"] == "eedi"]
    for it in eedi:  # Block A
        for lang in d["languages"]["core"]:
            v = it["versions"][lang]
            rec("A", it, lang, lang, "C0", sys0, student_message(lang, v["question"], v["student_answer"]),
                labels[lang])
    for it in (x for x in eedi if x["in_block_b"]):  # Block B
        for lang in d["languages"]["identity"]:
            v = it["versions"][lang]
            msg = student_message(lang, v["question"], v["student_answer"])
            for cue in d["cues"]["block_b"]:
                ctext, nm = cue_sentence(cue, lang, it["name_rank"])
                rec("B", it, lang, lang, cue, sys0, with_cue(ctext, msg), labels[lang], nm)
    for it in (x for x in eedi if x["in_block_c"]):  # Block C
        v = it["versions"]["EN"]
        msg = student_message("EN", v["question"], v["student_answer"])
        for tgt in d["languages"]["decomposition_targets"]:
            rec("C", it, "EN", "EN", "C0", f"{sys0}\n{prompt(f'block_c_reply_{tgt}')}", msg,
                d["languages"]["block_c_labels"][tgt], {"prompt_id": f"C|{it['item_id']}|EN>{tgt}|C0",
                                                        "condition": f"EN>{tgt}", "reply_lang": tgt})
    for it in (x for x in items if x["source"] == "benqa"):  # Block D
        for cond in d["blocks"]["D"]["conditions"]:
            v = it["versions"][cond]
            tmpl = "EN" if cond == "EN-OFFICIAL" else "BN"
            rec("D", it, cond, tmpl, "C0", sys0, student_message(tmpl, v["question"], v["student_answer"]),
                labels[tmpl])
    for r in out:
        r.setdefault("reply_lang", r["student_lang"])
        r["samples"] = int(d["blocks"][r["block"]]["samples"])
    return out


def run_stage(run: Run) -> dict:
    items = read_jsonl(run.path("data", "items_multilingual.jsonl"))
    if not items:
        raise SystemExit("data/items_multilingual.jsonl missing: run translate first")
    gates = {"templates_not_approved": check_template_approval(run),
             "translation_reviews_pending": pending_reviews(run),
             "names_confirmed": names_confirmed()}
    outstanding = bool(gates["templates_not_approved"] or gates["translation_reviews_pending"]
                       or not gates["names_confirmed"])
    import os
    if run.mode == "full" and outstanding and os.environ.get("TLT_ALLOW_UNREVIEWED") == "1":
        log.warning("FULL RUN WITH HUMAN GATES OUTSTANDING (--allow-unreviewed): outputs are PRELIMINARY %s", gates)
        run.path("results", "PRELIMINARY_human_gates_outstanding.json").write_text(json.dumps(gates, indent=2))
    elif run.mode == "full" and outstanding:
        raise SystemExit(
            f"Full run blocked by human gates {gates}. Complete review/template_approval.csv, "
            "review/translation_review.csv and docs/name_validation.md (STATUS: CONFIRMED), "
            "or use --pilot.")
    marker = run.path("results", "PRELIMINARY_human_gates_outstanding.json")
    if run.mode == "full" and not outstanding and marker.exists():
        marker.unlink()  # every gate is closed: results are no longer preliminary
    if any(gates.values()) and run.mode != "full":
        log.warning("human gates outstanding (allowed in %s mode): %s", run.mode, gates)
    prompts = make_prompts(run, items)
    write_jsonl(run.path("data", "prompts.jsonl"), prompts)
    by_block = {}
    for p in prompts:
        by_block[p["block"]] = by_block.get(p["block"], 0) + 1
    n_prod = len(run.producers)
    gens = {b: sum(p["samples"] for p in prompts if p["block"] == b) * n_prod for b in by_block}
    stats = {"prompts": by_block, "generations": gens, "total_generations": sum(gens.values()),
             "judge_calls": sum(gens.values()) * len(run.judges), **gates}
    log.info("prompts: %s", stats)
    return stats
