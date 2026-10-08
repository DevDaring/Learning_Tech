"""Stage 7 (RQ4): judge English back-translations of a 20% sample of non-English replies.

Sample: stratified by block x condition x producer, seeded, from every judged reply whose
student or requested reply language is not English. Back-translation routing, per reply:
  * code-mixed conditions (HI-EN, BN-EN) whose reply is mostly Roman script -> utility
    model (IndicTrans2 does not accept romanised Hindi/Bengali; docs/deviations.md D7);
  * otherwise sentence by sentence: Devanagari -> IndicTrans2 hin->eng, Bengali script ->
    IndicTrans2 ben->eng, Latin sentences kept as they are.
Names are masked before translation and again after it. The translated reply is judged
with the same prompt and language label. Report: per judge x language x dimension, the
paired difference, signed so that POSITIVE means the judge rated the original-language
reply worse than its English translation (penalising the language, not the teaching),
bootstrapped over items. The language label is held fixed between the two judgements so
the reply's language is the only thing that differs; Language_Match is excluded. Failed or
empty back-translations are excluded and counted, never judged.
"""
from __future__ import annotations

import logging
import random
from collections import defaultdict

import numpy as np
import pandas as pd

from ..analysis.stats import add_holm, bootstrap_mean
from ..clients import make_llm
from ..config import Run
from ..io import AppendLog, read_jsonl, read_table
from ..names import mask_names
from ..rubric import CATEGORICAL, DIMENSIONS, ordinal_index, pqi

# Indicator used to compare categorical dimensions (the undesirable outcome).
CATEGORICAL_BAD = {"Revealing_of_the_Answer": {"Yes (correct)", "Yes (incorrect)"},
                   "Praise_Calibration": {"Inflated", "Discouraging"}}
from ..runner import protected_utility
from ..textutil import dominant_script
from ..translate import join_segments, make_translator, split_segments
from .s02_translate import CM_SOURCE
from .s04_generate import current_generations
from .s05_mask_and_judge import judge_targets, targets_from_generations, write_tables

log = logging.getLogger("tlt.s07")


def eligible(t: dict) -> bool:
    return t["student_lang"] != "EN" or t.get("reply_lang", "EN") != "EN"


def back_translate(run: Run, tr, llm, ulog, targets: list[dict]) -> None:
    seg_jobs = defaultdict(list)  # script -> [(target_idx, seg_idx, text)]
    plans = {}
    for ti, t in enumerate(targets):
        masked, _ = mask_names(t["response_text"])
        t["_masked"] = masked
        if t["condition"] in CM_SOURCE and dominant_script(masked) == "latin":
            cid = f"tj_backtr|{t['target_id']}"
            text, rec, _ = protected_utility(llm, ulog, cid, "utility_translate_to_en", "text", masked,
                                             {"role": "translated_judging_backtr", "target_id": t["target_id"]},
                                             source_description=CM_SOURCE[t["condition"]])
            t["translated_text"] = mask_names(text)[0]
            t["backtr_route"] = rec.get("route_used") or "FAILED"
            continue
        pairs = split_segments(masked)
        plans[ti] = pairs
        for si, (_, s) in enumerate(pairs):
            sc = dominant_script(s)
            if sc in ("deva", "beng"):
                seg_jobs[sc].append((ti, si, s))
    results = {}
    for sc, lang in (("deva", "HI"), ("beng", "BN")):
        jobs = seg_jobs.get(sc, [])
        if jobs:
            outs = tr.indic_to_en([s for _, _, s in jobs], lang)
            for (ti, si, _), o in zip(jobs, outs):
                results[(ti, si)] = o
    for ti, pairs in plans.items():
        new = [(ln, results.get((ti, si), s)) for si, (ln, s) in enumerate(pairs)]
        targets[ti]["translated_text"] = mask_names(join_segments(new))[0]
        targets[ti]["backtr_route"] = tr.model_id_indic_en


def run_stage(run: Run) -> dict:
    gens = current_generations(run)
    prompts = {p["prompt_id"]: p for p in read_jsonl(run.path("data", "prompts.jsonl"))}
    pool = [t for t in targets_from_generations(gens, prompts) if eligible(t)]
    share = float(run.design["translated_judging"]["share"])
    strata = defaultdict(list)
    for t in pool:
        strata[(t["block"], t["condition"], t["model"])].append(t)
    rng = random.Random(run.seed + 7)
    sample = []
    for k in sorted(strata):
        grp = sorted(strata[k], key=lambda t: t["target_id"])
        sample.extend(rng.sample(grp, max(1, round(share * len(grp)))))
    tr, llm = make_translator(run), make_llm(run)
    ulog = AppendLog(run.path("runs", "utility_calls.jsonl"))
    back_translate(run, tr, llm, ulog, sample)
    trans_targets, failed = [], []
    for t in sample:
        if not (t.get("translated_text") or "").strip() or t.get("backtr_route") == "FAILED":
            failed.append(t["target_id"])  # never judge an empty translation; counted below
            continue
        u = dict(t)
        u["response_text"], u["variant"] = t.get("translated_text", ""), "translated"
        trans_targets.append(u)
    pd.DataFrame([{k: t.get(k) for k in ("target_id", "condition", "model", "backtr_route", "translated_text")}
                  for t in sample]).to_csv(run.path("results", "translated_judging_texts.csv"), index=False)
    applog = AppendLog(run.path("runs", "translated_judgements.jsonl"))
    counts, jids = judge_targets(run, llm, trans_targets, applog, "translated-judge")
    write_tables(run, applog, "translated_judgements", jids)

    # Paired comparison with the original-language judgements of the same replies.
    orig = read_table(run.path("results", "judgements_long.csv"))
    trans = read_table(run.path("results", "translated_judgements_long.csv"))
    ids = {t["target_id"] for t in trans_targets}
    orig = orig[orig.target_id.isin(ids)]
    m = orig.merge(trans, on=["target_id", "judge"], suffixes=("_o", "_t"))
    B = int(run.design["bootstrap"]["resamples"])
    rows = []
    dims = [d for d in DIMENSIONS if d != "Language_Match"]
    for j in run.judges:
        for lang in sorted(m["condition_o"].unique()):
            g = m[(m.judge == j) & (m.condition_o == lang)]
            if g.empty:
                continue
            for d in dims + ["PQI"]:
                if d == "PQI":
                    vo = g.apply(lambda r: pqi({x: (r[f"{x}_o"] if isinstance(r[f"{x}_o"], str) else None) for x in DIMENSIONS}), axis=1)
                    vt = g.apply(lambda r: pqi({x: (r[f"{x}_t"] if isinstance(r[f"{x}_t"], str) else None) for x in DIMENSIONS}), axis=1)
                    diff = vt - vo
                    note = "PQI(translated) - PQI(original)"
                elif d in CATEGORICAL:
                    bad = CATEGORICAL_BAD[d]
                    vo = g[f"{d}_o"].map(lambda x: float(x in bad) if isinstance(x, str) else np.nan)
                    vt = g[f"{d}_t"].map(lambda x: float(x in bad) if isinstance(x, str) else np.nan)
                    diff = vo - vt
                    note = f"rate of {sorted(bad)} in original - in translated"
                else:
                    io = g[f"{d}_o"].map(lambda x: ordinal_index(d, x if isinstance(x, str) else None))
                    it = g[f"{d}_t"].map(lambda x: ordinal_index(d, x if isinstance(x, str) else None))
                    diff = io - it  # index 0 is best, so a larger index is a worse rating
                    note = "category index(original) - index(translated); index 0 = best"
                per_item = pd.DataFrame({"item": g["item_id_o"], "d": diff}).dropna().groupby("item")["d"].mean()
                res = bootstrap_mean(per_item, B, run.seed)
                agree = float(np.mean(g[f"{d}_o"].values == g[f"{d}_t"].values)) if d != "PQI" else np.nan
                rows.append({"family": "translated_judging", "judge": j, "language": lang, "dimension": d,
                             "exact_agreement": agree, "computed_as": note,
                             "sign": "positive = judge rated the original-language reply WORSE than its translation",
                             **res})
    out = add_holm(pd.DataFrame(rows)) if rows else pd.DataFrame()
    out.to_csv(run.path("results", "translated_judging.csv"), index=False)
    counts.update({"pool": len(pool), "sampled": len(sample), "translation_failed_excluded": len(failed),
                   "utility_routed": sum(str(t.get("backtr_route", "")).startswith(("linkapi", "openrouter", "mock"))
                                         for t in sample)})
    log.info("translated judging: %s", counts)
    return counts
