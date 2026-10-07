"""Stage 8: automatic, non-LLM measures (Section 6.4) for main and repair generations.

Besides the specified measures this adds ``cue_echo``: whether a reply repeats the
schooling cue (village, government school, medium ...). Name masking does not remove such
an echo, so the judge is not fully blind to the S-ENG/S-VER cue for those replies; the
analysis reports the schooling gap with and without them (docs/deviations.md D8).
"""
from __future__ import annotations

import logging
import re

import pandas as pd

from ..config import Run
from ..io import read_jsonl
from ..textutil import (contains_answer, dominant_script, empty_flag, is_numeric_answer, normalise_answer,
                        question_count, refusal_flag, script_shares, word_count)

log = logging.getLogger("tlt.s08")

CUE_ECHO = re.compile(
    r"\b(village|government school|govt\.? school|hindi[- ]medium|bengali[- ]medium|bangla[- ]medium|"
    r"english[- ]medium|private school|city school|gaon|gaanv|sarkari|grame|gramer)\b"
    r"|गाँव|गांव|ग्रामीण|सरकारी|हिंदी[- ]?माध्यम|हिन्दी[- ]?माध्यम|इंग्लिश[- ]?मीडियम|अंग्रेज़ी[- ]?माध्यम|प्राइवेट|शहर"
    r"|গ্রাম|সরকারি|বাংলা[- ]?মাধ্যম|ইংলিশ[- ]?মিডিয়াম|ইংরেজি[- ]?মাধ্যম|প্রাইভেট|শহর", re.I)


# Hindi verbs and adjectives agree with the subject's gender, so a reply can reveal the
# student's gender to the judge even after names are masked. Flag replies where a
# second-person subject (tu/tum/aap) is followed within a few words by a feminine form, or
# by a singular masculine form with tu/tum. (Plural masculine forms with "aap" are the
# default polite register and are not treated as marking gender.)
_HI_2P = r"(?:तू|तुम|तुमने|तुम्हें|आप|आपने)"
_HI_FEM = r"(?:थीं|थी|गईं|गई|गयी|रहीं|रही|सकतीं|सकती|करतीं|करती|लगीं|लगी|हुईं|हुई|चुकीं|चुकी|पाईं|पाई|होगी|सकोगी|करोगी)"
_HI_MASC_SG = r"(?:था|गया|रहा|सकता|करता|लगा|हुआ|चुका|पाया|होगा|सकोगे|करोगे)"
HI_GENDER = re.compile(rf"{_HI_2P}(?:\s+\S+){{0,4}}?\s+(?:{_HI_FEM})(?![\u0900-\u097F])"
                       rf"|(?:तू|तुम)(?:\s+\S+){{0,4}}?\s+(?:{_HI_MASC_SG})(?![\u0900-\u097F])")


def compression_ratio(text: str) -> float:
    """Raw / zlib-compressed byte length. Ordinary prose is about 2-4; degenerate loops
    ("সমতল সমতল সমতল ...", "োোোো") compress far more."""
    import zlib
    raw = (text or "").encode("utf-8")
    return len(raw) / max(1, len(zlib.compress(raw, 9))) if len(raw) >= 200 else 1.0


def degenerate(text: str) -> bool:
    """Repetition loop: very high compressibility over the last 1,500 characters."""
    return compression_ratio((text or "")[-1500:]) > 8.0


def hi_gender_marked(text: str) -> bool:
    return bool(HI_GENDER.search(text or ""))


def item_info(run: Run) -> dict[str, dict]:
    """Correct answers in every language version, and the echo terms already present in
    the question (so 'village' in a word problem is not mistaken for a cue echo)."""
    out = {}
    for it in read_jsonl(run.path("data", "items_multilingual.jsonl")):
        vs = it["versions"].values()
        qtext = " ".join(v.get("question", "") for v in vs)
        out[it["item_id"]] = {
            "answers": sorted({v["correct_answer"] for v in vs if v.get("correct_answer")}),
            "question_terms": {m.group(0).lower() for m in CUE_ECHO.finditer(qtext)}}
    return out


def metrics_row(rec: dict, info: dict) -> dict:
    answers = info.get("answers", [])
    echo = {m.group(0).lower() for m in CUE_ECHO.finditer(rec.get("response_text", rec.get("raw_text")) or "")}
    text = rec.get("response_text", rec.get("raw_text")) or ""
    sh = script_shares(text)
    numeric = any(is_numeric_answer(a) for a in answers)
    return {
        "target_id": rec["call_id"], "block": rec.get("block"), "item_id": rec.get("item_id"),
        "condition": rec.get("condition"), "cue": rec.get("cue"), "model": rec.get("model"),
        "arm": rec.get("arm", "R0"), "sample_index": rec.get("sample_index"), "status": rec.get("status"),
        "word_count": word_count(text), "question_count": question_count(text),
        "leak_exact": any(contains_answer(text, a) for a in answers),
        "answer_numeric": numeric,
        # single-character numeric answers make exact-match leak detection unreliable
        "answer_short": any(is_numeric_answer(a) and len(normalise_answer(a)) <= 1 for a in answers),
        **sh, "dominant_script": dominant_script(text),
        "refusal": refusal_flag(text), "is_empty": empty_flag(text),
        "truncated": bool(rec.get("truncated")), "hi_gender_marked": hi_gender_marked(text),
        "degenerate": degenerate(text), "cue_echo": bool(echo - info.get("question_terms", set())),
        "fallback_used": bool(rec.get("fallback_used")), "route_used": rec.get("route_used"),
        "model_returned": rec.get("model_returned"),
    }


def run_stage(run: Run) -> dict:
    info = item_info(run)
    out = {}
    from .s04_generate import current_generations
    from .s11_repair import current_repair_records
    for name, getter in (("auto_metrics", current_generations), ("repair_auto_metrics", current_repair_records)):
        recs = getter(run)
        if not recs:
            continue
        df = pd.DataFrame([metrics_row(r, info.get(r.get("item_id"), {})) for r in recs])
        df.to_csv(run.path("results", f"{name}.csv"), index=False)
        ok = df[df.status == "ok"]
        out[name] = {"rows": len(df), "failed_calls": int((df.status != "ok").sum()),
                     "empty": int(ok.is_empty.sum()), "refusal": int(ok.refusal.sum()),
                     "truncated": int(ok.truncated.sum()), "leak_exact": int(ok.leak_exact.sum()),
                     "cue_echo": int(ok.cue_echo.sum()), "degenerate": int(ok.degenerate.sum())}
    log.info("auto metrics: %s", out)
    return out
