"""Stage 10: related-study baselines B1, B2, B3 and B5 (B4 is stage 6, B6 is repair arm R3).

All reuse the main generations and judgements; only B2's sentence labelling and B5's
student re-answers call a model, and both run on subsets (design.yaml) to limit cost.
Outputs go to results/baselines/.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..analysis.stats import add_holm, bootstrap_mean, gap
from ..baselines import b1_mab_mdb, b2_marked, b3_semantic_shift, b5_simstudent
from ..clients import make_llm
from ..config import Run, stable_hash
from ..io import AppendLog, fill, prompt, prompt_yaml, read_jsonl
from ..jsonfix import parse_with_repair
from ..names import mask_names
from ..runner import call_record, execute
from ..translate import make_embedder
from .s04_generate import current_generations
from .s11_repair import current_repair_records

log = logging.getLogger("tlt.s10")


def _gens_with_c0(run: Run, langs: list[str]) -> pd.DataFrame:
    """Block B replies plus Block A C0 replies (samples < Block B's) for the same items."""
    gens = current_generations(run, ok_only=True)
    items_b = {it["item_id"] for it in read_jsonl(run.path("data", "items.jsonl")) if it.get("in_block_b")}
    nb = int(run.design["blocks"]["B"]["samples"])
    rows = []
    for g in gens:
        if g["student_lang"] not in langs:
            continue
        if g["block"] == "B" or (g["block"] == "A" and g["cue"] == "C0" and g["item_id"] in items_b
                                 and g["sample_index"] < nb):
            rows.append({"target_id": g["call_id"], "item_id": g["item_id"], "model": g["model"],
                         "student_lang": g["student_lang"], "cue": g["cue"], "sample_index": g["sample_index"],
                         "text": g["raw_text"] or ""})
    return pd.DataFrame(rows)


def run_b1(run: Run, out_dir) -> dict:
    agg = pd.read_csv(run.path("results", "judgements_aggregated.csv"))
    df = b1_mab_mdb.compute(agg, int(run.design["bootstrap"]["resamples"]) // 10, run.seed)
    df.to_csv(out_dir / "b1_mab_mdb.csv", index=False)
    return {"rows": len(df)}


def run_b2(run: Run, llm, out_dir) -> dict:
    gens = _gens_with_c0(run, ["EN"])
    mw = b2_marked.marked_words(gens)
    mw.to_csv(out_dir / "b2_marked_words.csv", index=False)
    # Sentence labelling on the Block C item subset, sample 0.
    sub_items = {it["item_id"] for it in read_jsonl(run.path("data", "items.jsonl")) if it.get("in_block_c")}
    sub = gens[gens.item_id.isin(sub_items) & (gens.sample_index == 0)]
    applog = AppendLog(run.path("runs", "b2_praise_labels.jsonl"))
    jobs = [{"call_id": f"b2|{r.target_id}|{stable_hash(prompt('praise_critique'), r.text, n=10)}",
             "model": "utility", "row": r} for r in sub.itertuples()]
    current = {j["call_id"] for j in jobs}

    def fn(job):
        r = job["row"]
        sents = b2_marked.split_sentences(mask_names(r.text)[0])
        numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sents)) or "1. (empty reply)"
        res = llm.call("utility", fill(prompt("praise_critique"), numbered_sentences=numbered),
                       seed_key=job["call_id"])
        obj, info = parse_with_repair(res.text, llm, job["call_id"]) if res.ok else (None, {})
        labels = {str(k): str(v).upper().strip() for k, v in (obj or {}).items()}
        n = len(sents)
        rec = call_record({"call_id": job["call_id"], "model": "utility", "role": "b2_labeller",
                           "target_id": r.target_id, "n_sentences": n, "labels": labels, **info}, res)
        rec.update({"n_praise": sum(labels.get(str(i + 1)) == "PRAISE" for i in range(n)),
                    "n_corrective": sum(labels.get(str(i + 1)) == "CORRECTIVE" for i in range(n)),
                    "labels_complete": all(str(i + 1) in labels for i in range(n))})
        if res.ok and obj is None:
            rec["status"] = "unparseable"
        return rec

    counts = execute(jobs, applog, fn, {"utility": int(run.models["models"]["utility"].get("workers", 8))}, "b2")
    lab = pd.DataFrame([r for r in applog.records() if r["call_id"] in current and r["status"] == "ok"
                        and r.get("labels_complete")])
    if lab.empty:
        return {"marked_words": len(mw), **counts}
    m = sub.merge(lab[["target_id", "n_praise", "n_corrective", "n_sentences"]], on="target_id")
    B = int(run.design["bootstrap"]["resamples"])
    rows = []
    for pair, (adv, dis) in b2_marked.PAIRS.items():
        for model in sorted(m.model.unique()) + [None]:
            for metric, label in (("n_praise", "positive_feedback_bias"), ("n_corrective", "feedback_withholding")):
                res = gap(m, metric, "cue", dis, adv, producer=model, B=B, seed=run.seed)
                rows.append({"family": f"b2_{label}", "pair": pair, "model": model or "ALL", "metric": metric,
                             "estimand": f"mean({metric} | disadvantaged) - mean(... | advantaged)", **res})
    out = add_holm(pd.DataFrame(rows))
    out.to_csv(out_dir / "b2_praise_critique.csv", index=False)
    return {"marked_words": len(mw), "labelled": len(lab), **counts}


def run_b3(run: Run, out_dir) -> dict:
    gens = _gens_with_c0(run, run.design["languages"]["identity"])
    gens = gens[gens.cue.isin(["N-MD", "N-FD", "N-MM", "N-FM"])]
    if gens.empty:
        return {"rows": 0}
    emb = make_embedder(run)
    texts = [mask_names(t)[0] for t in gens.text]
    vecs = emb.encode(texts)
    vec = dict(zip(gens.target_id, vecs))
    shifts = b3_semantic_shift.item_shifts(vec, gens)
    shifts.to_csv(out_dir / "b3_item_shifts.csv", index=False)
    B = int(run.design["bootstrap"]["resamples"])
    rows = []
    for (swap, lang), g in shifts.groupby(["swap", "language"]):
        for model in sorted(g.model.unique()) + [None]:
            s = g if model is None else g[g.model == model]
            for col in ("shift", "cross", "noise"):
                per_item = s.groupby("item_id")[col].mean()
                res = bootstrap_mean(per_item, B, run.seed)
                if col == "shift":  # Du et al.'s (unstudentized) permutation test, reported beside p_value
                    res["p_permutation_du"] = b3_semantic_shift.sign_flip_test(per_item.values, 5000, run.seed)
                rows.append({"family": "b3_semantic_shift", "swap": swap, "language": lang,
                             "model": model or "ALL", "quantity": col, **res})
    out = pd.DataFrame(rows)
    shift_rows = out.quantity == "shift"
    out.loc[~shift_rows, "p_value"] = np.nan  # cross and noise are levels, not tested contrasts
    out = pd.concat([add_holm(out[shift_rows]), out[~shift_rows]])
    out["embedding_model"] = getattr(emb, "model_id", "")
    out.to_csv(out_dir / "b3_semantic_shift.csv", index=False)
    return {"rows": len(out)}


def b5_targets(run: Run) -> list[dict]:
    """(tutor reply, student language) pairs: Block A sample 0 for B5 items in all core
    languages, every repair-arm reply for those items, and one control per item x language."""
    items = {it["item_id"]: it for it in read_jsonl(run.path("data", "items_multilingual.jsonl"))
             if it.get("in_b5")}
    si = int(run.design["b5"]["sample_index"])
    out = []
    for g in current_generations(run, ok_only=True):
        if g["block"] == "A" and g["item_id"] in items and g["sample_index"] == si:
            out.append({"source_id": g["call_id"], "item_id": g["item_id"], "lang": g["student_lang"],
                        "model": g["model"], "arm": "R0", "cue": "C0", "reply": g["raw_text"] or ""})
    for g in current_repair_records(run, ok_only=True):
        if g["item_id"] in items and g["cue"] == "C0":
            out.append({"source_id": g["call_id"], "item_id": g["item_id"], "lang": g["repair_lang"],
                        "model": g["model"], "arm": g["arm"], "cue": "C0", "reply": g.get("response_text") or ""})
    if run.design["b5"].get("control"):
        ctrl = prompt_yaml("b5_control_replies")
        langs = sorted({t["lang"] for t in out})
        for iid in items:
            for lang in langs:
                out.append({"source_id": f"control|{iid}|{lang}", "item_id": iid, "lang": lang,
                            "model": "control", "arm": "control", "cue": "C0", "reply": ctrl[lang]})
    for t in out:
        v = items[t["item_id"]]["versions"][t["lang"]]
        t.update({"question": v["question"], "student_answer": v["student_answer"],
                  "references": sorted({x["correct_answer"] for x in items[t["item_id"]]["versions"].values()}),
                  "wrong": sorted({x["student_answer"] for x in items[t["item_id"]]["versions"].values()}),
                  "question_en": items[t["item_id"]]["versions"]["EN"]["question"],
                  "reference_en": items[t["item_id"]]["versions"]["EN"]["correct_answer"]})
    return out


def run_b5(run: Run, llm, out_dir) -> dict:
    targets = b5_targets(run)
    applog = AppendLog(run.path("runs", "b5_student.jsonl"))
    jobs = [{"call_id": f"b5|{t['source_id']}|{stable_hash(t['question'], t['student_answer'], t['reply'], n=10)}",
             "model": "student", "t": t} for t in targets]
    current = {j["call_id"] for j in jobs}

    def fn(job):
        t = job["t"]
        user = fill(prompt("simulated_student"), question_in_student_language=t["question"],
                    student_answer=t["student_answer"], tutor_response=t["reply"])
        res = llm.call("student", user, None, temperature=0.0, max_tokens=200, seed_key=job["call_id"])
        verdict, method, eq_info = None, None, {}
        if res.ok:
            verdict = b5_simstudent.deterministic_match(res.text, t["references"], t["wrong"])
            method = "deterministic"
            if verdict is None:
                eq = llm.call("utility", fill(prompt("answer_equivalence"), question=t["question_en"],
                                              reference=t["reference_en"], answer=res.text.strip()),
                              seed_key=job["call_id"] + "|eq")
                word = (eq.text or "").strip().upper()
                verdict = True if word.startswith("SAME") else False if word.startswith("DIFFERENT") else None
                method = f"utility:{eq.route}" if verdict is not None else "undecided"
                eq_info = {"aux_utility_route": eq.route, "aux_utility_returned": eq.model_returned,
                           "aux_utility_text": eq.text, "aux_utility_ok": eq.ok}
        meta = {k: t[k] for k in ("source_id", "item_id", "lang", "arm", "cue")}
        # model = the simulated student that answered; tutor_model = the producer whose reply it read
        return call_record({"call_id": job["call_id"], "model": "student", "tutor_model": t["model"],
                            "role": "student", **meta, "correct": verdict, "scoring_method": method,
                            **eq_info}, res)

    counts = execute(jobs, applog, fn, {"student": int(run.models["models"]["student"].get("workers", 8))}, "b5")
    recs = [r for r in applog.records() if r["call_id"] in current]
    df = pd.DataFrame([r for r in recs if r["status"] == "ok"])
    if df.empty:
        return counts
    df = df[df.correct.notna()].copy()
    df["model"] = df["tutor_model"]
    df["correct"] = df.correct.astype(float)
    # Leak flag of the tutor reply (auto metrics, Tonga-style stand-alone match).
    leak = {}
    for name in ("auto_metrics", "repair_auto_metrics"):
        p = run.path("results", f"{name}.csv")
        if p.exists():
            am = pd.read_csv(p)
            leak.update(dict(zip(am.target_id, am.leak_exact)))
    df["leak"] = df.source_id.map(leak).fillna(False).astype(bool)
    ctrl = df[df.arm == "control"].set_index(["item_id", "lang"])["correct"]
    B = int(run.design["bootstrap"]["resamples"])
    rows = []
    for (arm, lang), g in df[df.arm != "control"].groupby(["arm", "lang"]):
        for model in sorted(g.model.unique()) + [None]:
            s = g if model is None else g[g.model == model]
            per_item = s.groupby("item_id")["correct"].mean()
            c = ctrl.reindex([(i, lang) for i in per_item.index]).values
            gain = per_item.values - c
            res = bootstrap_mean(pd.Series(gain, index=per_item.index), B, run.seed)
            correct = s[s.correct == 1]
            rows.append({"family": "b5_gain", "arm": arm, "language": lang, "model": model or "ALL",
                         "acc_after_tutor": float(s.correct.mean()), "acc_control": float(np.nanmean(c)),
                         "gain_estimate": res["estimate"], "gain_ci_low": res["ci_low"], "gain_ci_high": res["ci_high"],
                         "p_value": res["p_value"], "n_items": res["n_items"],
                         "leak_share_among_correct": float(correct.leak.mean()) if len(correct) else np.nan,
                         "acc_after_tutor_no_leak": float(s[~s.leak].correct.mean()) if (~s.leak).any() else np.nan,
                         "n_responses": len(s)})
    out = add_holm(pd.DataFrame(rows))
    out.to_csv(out_dir / "b5_simulated_student.csv", index=False)
    counts.update({"scored": len(df), "undecided_dropped": int(sum(r.get("correct") is None for r in recs))})
    return counts


def run_stage(run: Run, only: list[str] | None = None) -> dict:
    out_dir = run.path("results", "baselines", "x").parent
    llm = make_llm(run)
    res = {}
    for name, fn in (("B1", lambda: run_b1(run, out_dir)), ("B2", lambda: run_b2(run, llm, out_dir)),
                     ("B3", lambda: run_b3(run, out_dir)), ("B5", lambda: run_b5(run, llm, out_dir))):
        if only and name not in only:
            continue
        res[name] = fn()
        log.info("%s: %s", name, res[name])
    return res
