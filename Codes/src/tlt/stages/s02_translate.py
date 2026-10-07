"""Stage 2: HI/BN by IndicTrans2, HI-EN/BN-EN by the utility model, quality gate, review file.

Quality gate (Section 4.2), applied to every non-English version:
  * back-translate to English and compute embedding cosine with the original (< 0.75 -> flag);
  * every LaTeX span must survive unchanged and the multiset of numerals must match
    (Section 4.1 step 5) -> flag otherwise;
  * an empty output -> flag.
The review file lists every version with a ``status``: ``flagged`` (must be reviewed),
``review_sample`` (random 20% of the rest, must be reviewed) or ``auto_ok``. The study owner
sets reviewed rows to ``approved`` or ``corrected`` (and fills ``corrected_text``).
Rerunning this stage keeps those decisions and applies the corrections.
"""
from __future__ import annotations

import logging
import random
import re

import numpy as np
import pandas as pd

from ..clients import make_llm
from ..config import Run
from ..io import AppendLog, read_jsonl, write_jsonl
from ..runner import protected_utility
from ..textutil import latex_spans, numbers_in, strip_latex_commands
from ..translate import make_embedder, make_translator

log = logging.getLogger("tlt.s02")

CODEMIX = {"HI-EN": ("North India", "Hindi"), "BN-EN": ("West Bengal", "Bengali")}
CM_SOURCE = {"HI-EN": "Hinglish (Hindi written in Roman script, with English maths terms)",
             "BN-EN": "Banglish (Bengali written in Roman script, with English maths terms)"}
REVIEWED = {"approved", "corrected"}


def needs_translation(answer: str) -> bool:
    """An answer is translated only if it contains words outside LaTeX (e.g. 'Only Tom')."""
    plain = re.sub(r"\\\(.+?\\\)|\\\[.+?\\\]|\$.+?\$", " ", answer or "")
    plain = strip_latex_commands(plain)
    return bool(re.search(r"[A-Za-z]{2,}", plain))


def integrity(src: str, out: str, missing_spans: list[str]) -> list[str]:
    reasons = []
    if not (out or "").strip():
        reasons.append("empty")
    if missing_spans:
        reasons.append(f"latex_lost:{len(missing_spans)}")
    src_spans, out_spans = latex_spans(src), latex_spans(out)
    if sorted(src_spans) != sorted(out_spans):
        reasons.append("latex_changed")
    if numbers_in(src) != numbers_in(out):
        reasons.append("numbers_changed")
    return reasons


def run_stage(run: Run) -> dict:
    items = read_jsonl(run.path("data", "items.jsonl"))
    if not items:
        raise SystemExit("data/items.jsonl missing: run build_items first")
    tr = make_translator(run)
    emb = make_embedder(run)
    llm = make_llm(run)
    ulog = AppendLog(run.path("runs", "utility_calls.jsonl"))
    thr = float(run.design["translation"]["similarity_threshold"])

    # ---- forward translation
    rows = []  # one per (item, lang, field)
    for lang in ("HI", "BN"):
        eedi = [it for it in items if it["source"] == "eedi"]
        qs = tr.en_to_indic_checked([it["question_en"] for it in eedi], lang)
        for it, (t, miss) in zip(eedi, qs):
            rows.append({"item_id": it["item_id"], "lang": lang, "field": "question",
                         "source_en": it["question_en"], "translation": t, "missing": miss,
                         "route": tr.model_id_en_indic})
        for field in ("student_answer", "correct_answer"):
            sel = [it for it in eedi if needs_translation(it[f"{field}_en"])]
            outs = tr.en_to_indic_checked([it[f"{field}_en"] for it in sel], lang) if sel else []
            done = {it["item_id"]: o for it, o in zip(sel, outs)}
            for it in eedi:
                t, miss = done.get(it["item_id"], (it[f"{field}_en"], []))
                rows.append({"item_id": it["item_id"], "lang": lang, "field": field,
                             "source_en": it[f"{field}_en"], "translation": t, "missing": miss,
                             "route": tr.model_id_en_indic if it["item_id"] in done else "unchanged(maths)"})
    # Block D machine-translated Bengali (D2)
    benqa = [it for it in items if it["source"] == "benqa"]
    for field in ("question", "student_answer", "correct_answer"):
        src = [it[f"{field}_en"] for it in benqa]
        outs = tr.en_to_indic_checked(src, "BN") if benqa else []
        for it, (t, miss) in zip(benqa, outs):
            rows.append({"item_id": it["item_id"], "lang": "BN-MT", "field": field,
                         "source_en": it[f"{field}_en"], "translation": t, "missing": miss,
                         "route": tr.model_id_en_indic})
    # Code-mixed (utility model, Section 7.5); answers stay in English (D7).
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(8)  # independent utility calls; results kept in item order
    for lang, (region, matrix) in CODEMIX.items():
        eedi = [x for x in items if x["source"] == "eedi"]
        futs = [pool.submit(protected_utility, llm, ulog, f"codemix|{lang}|{it['item_id']}", "codemix",
                            "question_en", it["question_en"], {"role": "codemix", "item_id": it["item_id"], "lang": lang},
                            region=region, matrix_language=matrix) for it in eedi]
        for it, fut in zip(eedi, futs):
            text, rec, miss = fut.result()
            rows.append({"item_id": it["item_id"], "lang": lang, "field": "question",
                         "source_en": it["question_en"], "translation": text,
                         "missing": miss, "route": rec.get("route_used") or "FAILED"})
            for field in ("student_answer", "correct_answer"):
                rows.append({"item_id": it["item_id"], "lang": lang, "field": field,
                             "source_en": it[f"{field}_en"], "translation": it[f"{field}_en"],
                             "missing": [], "route": "unchanged(english)"})

    # ---- back-translation and gate (only rows whose text was actually transformed)
    def is_transformed(r):
        return not r["route"].startswith("unchanged")

    for lang in ("HI", "BN", "BN-MT"):
        sel = [r for r in rows if r["lang"] == lang and is_transformed(r)]
        code = "BN" if lang == "BN-MT" else lang
        backs = tr.indic_to_en([r["translation"] for r in sel], code) if sel else []
        for r, b in zip(sel, backs):
            r["back_translation"], r["back_route"] = b, tr.model_id_indic_en
    for lang in CODEMIX:
        sel = [x for x in rows if x["lang"] == lang and is_transformed(x)]
        futs = [pool.submit(protected_utility, llm, ulog, f"backtr|{lang}|{r['item_id']}|{r['field']}",
                            "utility_translate_to_en", "text", r["translation"],
                            {"role": "back_translation", "item_id": r["item_id"], "lang": lang},
                            source_description=CM_SOURCE[lang]) for r in sel]
        for r, fut in zip(sel, futs):
            text, rec, _ = fut.result()
            r["back_translation"], r["back_route"] = text, rec.get("route_used")
    pool.shutdown()

    gated = [r for r in rows if is_transformed(r)]
    if gated:
        a = emb.encode([r["source_en"] for r in gated])
        b = emb.encode([r.get("back_translation") or "" for r in gated])
        sims = np.sum(a * b, axis=1)
        for r, s in zip(gated, sims):
            r["similarity"] = round(float(s), 4)
    rng = random.Random(run.seed + 2)
    share = float(run.design["translation"]["review_random_share"])
    for r in rows:
        reasons = []
        if is_transformed(r):
            reasons = integrity(r["source_en"], r["translation"], r["missing"])
            if r.get("similarity", 0.0) < thr:
                reasons.append(f"similarity<{thr}")
        r["flag_reasons"] = ";".join(reasons)
        if reasons:
            r["status"] = "flagged"
        elif is_transformed(r):
            r["status"] = "review_sample" if rng.random() < share else "auto_ok"
        else:
            r["status"] = "auto_ok"

    # ---- merge earlier human decisions
    review_path = run.path("review", "translation_review.csv")
    prior = {}
    if review_path.exists():
        old = pd.read_csv(review_path, dtype=str, keep_default_na=False)
        for _, o in old.iterrows():
            prior[(o["item_id"], o["lang"], o["field"])] = o
    for r in rows:
        o = prior.get((r["item_id"], r["lang"], r["field"]))
        r["corrected_text"], r["reviewer"], r["notes"] = "", "", ""
        if o is not None and o["status"] in REVIEWED and o["translation"] == r["translation"]:
            r["status"], r["corrected_text"] = o["status"], o.get("corrected_text", "")
            r["reviewer"], r["notes"] = o.get("reviewer", ""), o.get("notes", "")
    cols = ["item_id", "lang", "field", "status", "flag_reasons", "similarity", "source_en", "translation",
            "back_translation", "corrected_text", "reviewer", "notes", "route", "back_route"]
    pd.DataFrame(rows).reindex(columns=cols).to_csv(review_path, index=False)

    # ---- multilingual items
    final = {}
    for r in rows:
        text = r["corrected_text"] if r["status"] == "corrected" and r["corrected_text"] else r["translation"]
        final[(r["item_id"], r["lang"], r["field"])] = text
    out = []
    for it in items:
        v = {"EN": {"question": it["question_en"], "student_answer": it["student_answer_en"],
                    "correct_answer": it["correct_answer_en"]}}
        langs = ["HI", "BN", "HI-EN", "BN-EN"] if it["source"] == "eedi" else ["BN-MT"]
        for lang in langs:
            v[lang] = {f: final.get((it["item_id"], lang, f), "") for f in ("question", "student_answer", "correct_answer")}
        if it["source"] == "benqa":
            v["BN-NATIVE"] = {"question": it["question_bn_native"], "student_answer": it["student_answer_bn_native"],
                              "correct_answer": it["correct_answer_bn_native"]}
            v["EN-OFFICIAL"] = v.pop("EN")
        out.append({**it, "versions": v})
    write_jsonl(run.path("data", "items_multilingual.jsonl"), out)
    df = pd.DataFrame(rows)
    stats = {"rows": len(df), "flagged": int((df.status == "flagged").sum()),
             "review_sample": int((df.status == "review_sample").sum()),
             "reviewed": int(df.status.isin(REVIEWED).sum()),
             "translation_model": tr.model_id_en_indic, "back_translation_model": tr.model_id_indic_en,
             "embedding_model": getattr(emb, "model_id", "")}
    log.info("translate: %s", stats)
    return stats


def pending_reviews(run: Run) -> int:
    p = run.path("review", "translation_review.csv")
    if not p.exists():
        return -1
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    return int(df.status.isin(["flagged", "review_sample"]).sum())
