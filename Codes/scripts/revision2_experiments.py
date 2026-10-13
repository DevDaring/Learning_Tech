"""Second-revision experiments (docs/revision_plan_frozen.md, second addendum, 2026-10-13).

    PYTHONPATH=src python scripts/revision2_experiments.py e8prep     # segments for NLLB
    PYTHONPATH=src python scripts/revision2_experiments.py e8remote   # GPU job (after vast_translator.py up)
    PYTHONPATH=src python scripts/revision2_experiments.py e8         # assemble, judge, analyse
    PYTHONPATH=src python scripts/revision2_experiments.py e9 [--dry] # school-cue factors
Separate logs (runs/revision_e8_*.jsonl, runs/revision_e9_*.jsonl); nothing in the main study changes.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tlt.analysis.stats import add_holm, gap
from tlt.clients import make_llm
from tlt.config import load_run, stable_hash
from tlt.io import AppendLog, prompt, read_jsonl, read_table
from tlt.runner import execute
from tlt.stages.s03_build_prompts import student_message, with_cue
from tlt.stages.s04_generate import generation_jobs, make_fn, workers
from tlt.stages.s05_mask_and_judge import aggregate_table, judge_targets, long_table, targets_from_generations
from tlt.textutil import has_words, protect_math, restore_math
from tlt.translate import join_segments, split_segments

run = load_run("full")
R = run.path("results")
T = R / "tables"
B = int(run.design["bootstrap"]["resamples"])
SEED = run.seed
DRY = "--dry" in sys.argv
MODELS = sorted(run.producers)
SEGF = run.path("runs", "revision_e8_segments.jsonl")
OUTF = run.path("runs", "revision_e8_nllb_out.jsonl")
NLLB = "facebook/nllb-200-distilled-1.3B"
_llm = None


def llm():
    global _llm
    if _llm is None:
        _llm = make_llm(run)
    return _llm


def judge_agg(targets, log, label):
    jlog = AppendLog(run.path("runs", log))
    counts, jids = judge_targets(run, llm(), targets, jlog, label)
    print(label, counts, flush=True)
    return aggregate_table(long_table([r for r in jlog.records() if r["call_id"] in jids]),
                           run.design["primary_judge"], run.judges)


def r2t_records():
    keep = {}
    for line in open(run.path("runs", "repair_generations.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        if r.get("arm") == "R2T" and r.get("status") == "ok":
            keep[r["call_id"]] = r
    return list(keep.values())


# =================================================================== E8: MT sensitivity
def e8prep():
    rows, n = [], 0
    for r in r2t_records():
        if r["repair_lang"] not in ("HI", "BN"):
            continue
        n += 1
        prot, _ = protect_math(r.get("english_reply") or "")
        for j, (_, s) in enumerate(split_segments(prot)):
            if has_words(s):
                rows.append({"k": f"{r['call_id']}#{j}", "lang": r["repair_lang"], "text": s})
    with open(SEGF, "w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"{n} replies, {len(rows)} segments -> {SEGF}")


def e8remote():
    sys.path.insert(0, str(Path(__file__).parent))
    import vast_translator as v  # noqa: E402
    st = v.load()
    if not st.get("ip"):
        raise SystemExit("no instance: run scripts/vast_translator.py up first")
    base = ["-i", v.KEYFILE, *v.SSH_OPTS]
    tgt = f"root@{st['ip']}"
    subprocess.run(["scp", "-P", str(st["port"]), *base, str(SEGF), str(Path(__file__).parent / "nllb_job.py"),
                    f"{tgt}:/root/"], check=True, timeout=600)
    r = v.ssh(st, "cd /root && python nllb_job.py revision_e8_segments.jsonl nllb_out.jsonl > nllb.log 2>&1; "
                  "tail -3 nllb.log", timeout=6000)
    print(r.stdout[-500:], r.stderr[-300:])
    subprocess.run(["scp", "-P", str(st["port"]), *base, f"{tgt}:/root/nllb_out.jsonl", str(OUTF)], check=True,
                   timeout=600)
    print("fetched", sum(1 for _ in open(OUTF)), "translations")


def e8():
    outs = {json.loads(x)["k"]: json.loads(x)["v"] for x in open(OUTF, encoding="utf-8")}
    prompts = {p["prompt_id"]: p for p in read_jsonl(run.path("data", "prompts.jsonl"))}
    glog = AppendLog(run.path("runs", "revision_e8_nllb_generations.jsonl"))
    lost = 0
    recs = []
    for r in r2t_records():
        if r["repair_lang"] not in ("HI", "BN"):
            continue
        prot, mapping = protect_math(r.get("english_reply") or "")
        pairs = split_segments(prot)
        tr = [(ln, outs.get(f"{r['call_id']}#{j}", "") if has_words(s) else s) for j, (ln, s) in enumerate(pairs)]
        text, missing = restore_math(join_segments(tr), mapping)
        lost += len(missing)
        cid = f"r2t-nllb|{r['call_id']}|{stable_hash(text, n=10)}"
        rec = {k: r.get(k) for k in ("block", "item_id", "source", "condition", "student_lang", "reply_lang", "cue",
                                     "first_name", "surname", "subject", "base_prompt_id", "repair_lang", "model")}
        rec.update({"call_id": cid, "arm": "R2T-NLLB", "role": "producer", "status": "ok", "source_call_id": r["call_id"],
                    "translator": NLLB, "missing_math_spans": missing, "response_text": text,
                    "english_reply": r.get("english_reply")})
        recs.append(rec)
    done = {x["call_id"] for x in glog.records()}
    for rec in recs:
        if rec["call_id"] not in done:
            glog.append(rec)
    print(f"E8: {len(recs)} replies assembled; maths spans lost {lost}")
    targets = []
    for g in recs:
        p = prompts[g["base_prompt_id"]]
        targets.append({"target_id": g["call_id"], "block": "R", "item_id": g["item_id"], "source": g["source"],
                        "condition": g["condition"], "student_lang": g["student_lang"], "reply_lang": g["student_lang"],
                        "cue": g["cue"], "model": g["model"], "sample_index": 0, "arm": "R2T-NLLB",
                        "variant": "original", "first_name": None, "surname": None, "subject": g.get("subject"),
                        "response_text": g["response_text"], "judge_question_en": p["judge_question_en"],
                        "judge_correct_answer": p["judge_correct_answer"],
                        "judge_student_answer": p["judge_student_answer"],
                        "misconception_name": p["misconception_name"], "language_label": p["language_label"]})
    new = judge_agg(targets, "revision_e8_judgements.jsonl", "e8-judge")
    new.to_csv(R / "revision_e8_judgements_aggregated.csv", index=False)
    rep = read_table(R / "repair_judgements_aggregated.csv", low_memory=False)
    rep = rep[(rep.variant == "original") & (rep.cue == "C0") & rep.condition.isin(["EN", "HI", "BN"])
              & rep.arm.isin(["R2T", "R5", "R2L", "R2"])]
    en = rep[(rep.arm == "R2T") & (rep.condition == "EN")].assign(arm="R2T-NLLB")  # EN replies are untranslated
    df = pd.concat([rep, new, en], ignore_index=True)
    df["withheld"] = (df.Revealing_of_the_Answer == "No").astype(float)
    df["lang_match"] = (df.Language_Match == "Yes").astype(float)
    df["math_ok"] = (df.Math_Correctness == "Yes").astype(float)
    df["arm_lang"] = df.arm + "|" + df.condition
    rows = []
    for v in ("PQI_R", "PQI", "withheld", "lang_match", "math_ok"):
        for fam, a, b in (("nllb_vs_it2", "R2T-NLLB", "R2T"), ("r5_vs_nllb", "R5", "R2T-NLLB")):
            for l in ("HI", "BN"):
                for m in MODELS + [None]:
                    r = gap(df, v, "arm_lang", f"{a}|{l}", f"{b}|{l}", producer=m, B=B, seed=SEED)
                    rows.append({"family": f"e8_{fam}_{v}" if m is None else "descriptive", "outcome": v,
                                 "contrast": f"{a} - {b}", "language": l, "model": m or "ALL",
                                 "mean_a": float(df[(df.arm_lang == f"{a}|{l}") & ((df.model == m) if m else True)][v].mean()),
                                 "mean_b": float(df[(df.arm_lang == f"{b}|{l}") & ((df.model == m) if m else True)][v].mean()),
                                 **r})
        for arm in ("R2T", "R2T-NLLB", "R5"):
            for l in ("HI", "BN"):
                r = gap(df, v, "arm_lang", f"{arm}|EN", f"{arm}|{l}", B=B, seed=SEED)
                rows.append({"family": f"e8_gap_{v}", "outcome": v, "contrast": f"{arm} gap EN-{l}", "language": l,
                             "model": "ALL", **r})
    t = add_holm(pd.DataFrame(rows))
    t.loc[t.family == "descriptive", "p_holm"] = np.nan
    t.to_csv(T / "revision_e8_mt_sensitivity.csv", index=False)
    print(t[t.model == "ALL"][["outcome", "contrast", "language", "estimate", "ci_low", "ci_high", "p_holm"]]
          .round(4).to_string(index=False))
    pd.DataFrame([{"replies": len(recs), "maths_spans_lost": lost, "translator": NLLB}]).to_csv(
        T / "revision_e8_integrity.csv", index=False)


# =================================================================== E9: school-cue factors
CUES = {"S-ENG": prompt("cue_S-ENG_EN"), "S-VER": prompt("cue_S-VER_EN"),
        "S-MED": "I am in Class 8 at a Hindi-medium private school in the city.",
        "S-TYPE": "I am in Class 8 at an English-medium government school in the city.",
        "S-LOC": "I am in Class 8 at an English-medium private school in a village."}
DEVA = re.compile(r"[ऀ-ॿ]")


def e9():
    items = [it for it in read_jsonl(run.path("data", "items_multilingual.jsonl")) if it.get("in_block_b")]
    prompts = {p["prompt_id"]: p for p in read_jsonl(run.path("data", "prompts.jsonl"))}
    new = []
    for it in items:
        base = prompts[f"A|{it['item_id']}|EN|C0"]
        v = it["versions"]["EN"]
        for cue, text in CUES.items():
            p = dict(base)
            p.update({"prompt_id": f"Y|{it['item_id']}|EN|{cue}", "block": "Y", "cue": cue, "samples": 1,
                      "user": with_cue(text, student_message("EN", v["question"], v["student_answer"]))})
            new.append(p)
    jobs = generation_jobs(run, new)
    print("E9: items", len(items), "prompts", len(new), "replies", len(jobs), "judge calls", len(jobs) * 3)
    print("example:", new[2]["user"][:160].replace("\n", " | "))
    if DRY:
        return
    glog = AppendLog(run.path("runs", "revision_e9_generations.jsonl"))
    print(execute(jobs, glog, make_fn(run, llm()), workers(run, run.producers), "e9-generate"), flush=True)
    ids = {j["call_id"] for j in jobs}
    gens = [r for r in glog.records() if r["call_id"] in ids]
    agg = judge_agg(targets_from_generations(gens, {p["prompt_id"]: p for p in new}),
                    "revision_e9_judgements.jsonl", "e9-judge")
    agg.to_csv(R / "revision_e9_judgements_aggregated.csv", index=False)
    texts = {g["call_id"]: g.get("raw_text") or "" for g in gens}
    agg["devanagari"] = [float(bool(DEVA.search(texts.get(t, "")))) for t in agg.target_id]
    agg["inflated"] = (agg.Praise_Calibration == "Inflated").astype(float).where(agg.Praise_Calibration.notna())
    rows = []
    for v in ("PQI", "PQI_R", "inflated", "devanagari"):
        for c in ("S-MED", "S-TYPE", "S-LOC", "S-VER"):
            for m in MODELS + [None]:
                r = gap(agg, v, "cue", "S-ENG", c, producer=m, B=B, seed=SEED)
                fam = "descriptive"
                if m is None:
                    fam = f"e9_{v}" if c != "S-VER" else f"e9_replication_{v}"
                rows.append({"family": fam, "outcome": v, "contrast": f"S-ENG - {c}", "model": m or "ALL",
                             "mean_S-ENG": float(agg[(agg.cue == "S-ENG") & ((agg.model == m) if m else True)][v].mean()),
                             "mean_cue": float(agg[(agg.cue == c) & ((agg.model == m) if m else True)][v].mean()), **r})
    t = add_holm(pd.DataFrame(rows))
    t.loc[t.family == "descriptive", "p_holm"] = np.nan
    t.to_csv(T / "revision_e9_cue_factors.csv", index=False)
    print(t[t.model.isin(["ALL", "sarvam_105b"])][["outcome", "contrast", "model", "estimate", "ci_low", "ci_high",
                                                   "p_holm"]].round(4).to_string(index=False))


if __name__ == "__main__":
    {"e8prep": e8prep, "e8remote": e8remote, "e8": e8, "e9": e9}[sys.argv[1]]()
