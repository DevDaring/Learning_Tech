"""Third follow-up experiments (docs/revision_plan_frozen.md, third
addendum, frozen 2026-11-02 before any call).

    PYTHONPATH=src python scripts/revision3_experiments.py e10 [--dry]     # reply-language policy
    PYTHONPATH=src python scripts/revision3_experiments.py e11 [--dry]     # R2-T on code-mixed messages
    PYTHONPATH=src python scripts/revision3_experiments.py e12api [--dry]  # tutor-call timing
    PYTHONPATH=src python scripts/revision3_experiments.py e12prep         # MT timing jobs for the GPU
    PYTHONPATH=src python scripts/revision3_experiments.py e12gpu          # run them (after vast_translator.py up)
    PYTHONPATH=src python scripts/revision3_experiments.py e12             # assemble the timing tables
Separate logs (runs/revision_e10_*, revision_e11_*, revision_e12_*); nothing in the main study
changes. Owner's instruction (2026-11-02): no Google (GCP) key, so the Google route of Gemini is
removed here and Gemini is served through LinkAPI (same model string), OpenRouter as fallback.
"""
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from gaptrace.analysis.stats import add_holm, bootstrap_mean, gap
from gaptrace.clients import make_llm
from gaptrace.config import load_run, stable_hash, stable_int
from gaptrace.io import AppendLog, fill, prompt, read_jsonl, read_table
from gaptrace.jsonfix import parse_with_repair
from gaptrace.names import mask_names
from gaptrace.runner import execute
from gaptrace.stages.s02_translate import CM_SOURCE
from gaptrace.stages.s03_build_prompts import student_message, with_cue
from gaptrace.stages.s04_generate import generation_jobs, make_fn, workers
from gaptrace.stages.s05_mask_and_judge import aggregate_table, judge_targets, long_table, targets_from_generations
from gaptrace.stages.s11_repair import Repair, repair_jobs
from gaptrace.textutil import has_words, protect_math
from gaptrace.translate import make_translator, split_segments

run = load_run("full")
R = run.path("results")
T = R / "tables"
B = int(run.design["bootstrap"]["resamples"])
SEED = run.seed
DRY = "--dry" in sys.argv
MODELS = sorted(run.producers)
NAMES = run.design["languages"]["names"]
prompts = {p["prompt_id"]: p for p in read_jsonl(run.path("data", "prompts.jsonl"))}
items = {it["item_id"]: it for it in read_jsonl(run.path("data", "items_multilingual.jsonl"))}

# no Google (GCP) key: drop Gemini's Google route for every call made by this script
_g = run.models["models"]["gemini_flash"]
_g["routes"] = [r for r in _g["routes"] if r.get("provider") != "gemini"]
assert _g["routes"] and _g["routes"][0]["provider"] == "linkapi_gemini", _g["routes"]

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


def save(t: pd.DataFrame, name: str, cols=None) -> None:
    t.to_csv(T / name, index=False)
    show = t if cols is None else t[[c for c in cols if c in t.columns]]
    print(show.round(4).to_string(index=False), flush=True)


def did(df: pd.DataFrame, v: str) -> pd.Series:
    """Per-item difference-in-differences [S-ENG - S-VER](P0) - [S-ENG - S-VER](P1), averaged
    over the tutors that have all four cells (only items complete for every tutor are kept)."""
    w = df.groupby(["item_id", "model", "cue", "arm"])[v].mean().unstack(["cue", "arm"])
    need = [("S-ENG", "P0"), ("S-VER", "P0"), ("S-ENG", "P1"), ("S-VER", "P1")]
    w = w.dropna(subset=need)
    d = ((w[("S-ENG", "P0")] - w[("S-VER", "P0")]) - (w[("S-ENG", "P1")] - w[("S-VER", "P1")])).reset_index()
    cnt = d.groupby("item_id").model.nunique()
    keep = cnt[cnt == df.model.nunique()].index
    return d[d.item_id.isin(keep)].groupby("item_id")[0].mean()


# =================================================================== E10: reply-language policy
POLICY = "Always reply in the same language and script that the student used in their message."
CUES = {"S-ENG": prompt("cue_S-ENG_EN"), "S-VER": prompt("cue_S-VER_EN")}
DEVA = re.compile(r"[ऀ-ॿ]")


def e10():
    bitems = sorted(i for i, it in items.items() if it.get("in_block_b"))
    new = []
    for iid in bitems:
        base = prompts[f"A|{iid}|EN|C0"]
        v = items[iid]["versions"]["EN"]
        msg = student_message("EN", v["question"], v["student_answer"])
        for cue, text in CUES.items():
            for arm, system in (("P0", base["system"]), ("P1", f"{base['system']}\n{POLICY}")):
                p = dict(base)
                p.update({"prompt_id": f"Z|{iid}|EN|{cue}|{arm}", "block": "Z", "cue": cue, "arm": arm,
                          "samples": 1, "system": system, "user": with_cue(text, msg)})
                new.append(p)
    assert new[0]["system"] == prompt("tutor_system"), "P0 must use the default system prompt"
    jobs = generation_jobs(run, new)
    print("E10: items", len(bitems), "prompts", len(new), "replies", len(jobs), "judge calls", len(jobs) * 3)
    print("P1 system:", new[1]["system"].replace("\n", " | "))
    print("example user:", new[2]["user"][:160].replace("\n", " | "))
    if DRY:
        return
    glog = AppendLog(run.path("runs", "revision_e10_generations.jsonl"))
    print(execute(jobs, glog, make_fn(run, llm()), workers(run, run.producers), "e10-generate"), flush=True)
    ids = {j["call_id"] for j in jobs}
    gens = [r for r in glog.records() if r["call_id"] in ids]
    pmap = {p["prompt_id"]: p for p in new}
    targets = targets_from_generations(gens, pmap)
    for t, g in zip(targets, [g for g in gens if g["status"] == "ok"]):
        t["arm"] = pmap[g["prompt_id"]]["arm"]
    agg = judge_agg(targets, "revision_e10_judgements.jsonl", "e10-judge")
    agg["arm"] = agg.target_id.str.split("|").str[5]
    texts = {g["call_id"]: g.get("raw_text") or "" for g in gens}
    agg["devanagari"] = [float(bool(DEVA.search(texts.get(t, "")))) for t in agg.target_id]
    agg["inflated"] = (agg.Praise_Calibration == "Inflated").astype(float).where(agg.Praise_Calibration.notna())
    agg.to_csv(R / "revision_e10_judgements_aggregated.csv", index=False)
    routes = pd.DataFrame([{"model": g["model"], "route": g.get("route_used")} for g in gens if g["status"] == "ok"])
    print(routes.value_counts().to_string(), flush=True)
    cells = agg.groupby(["arm", "cue"]).agg(n=("target_id", "size"), PQI=("PQI", "mean"), PQI_R=("PQI_R", "mean"),
                                            devanagari=("devanagari", "mean"), inflated=("inflated", "mean"),
                                            language_match=("Language_Match", lambda s: float((s == "Yes").mean())))
    save(cells.reset_index(), "revision_e10_cells.csv")
    agg["arm_cue"] = agg.arm + "|" + agg.cue
    rows = []
    # primary family
    r = gap(agg, "devanagari", "arm_cue", "P0|S-VER", "P1|S-VER", B=B, seed=SEED)
    rows.append({"family": "e10_primary", "outcome": "devanagari", "contrast": "S-VER: P0 - P1", "model": "ALL", **r})
    for v in ("PQI", "PQI_R"):
        rows.append({"family": "e10_primary", "outcome": v, "contrast": "schooling gap: P0 - P1 (DiD)", "model": "ALL",
                     **bootstrap_mean(did(agg, v), B, SEED)})
    # secondary (descriptive)
    for v in ("PQI", "PQI_R", "inflated", "devanagari"):
        for m in MODELS + [None]:
            for arm in ("P0", "P1"):
                r = gap(agg, v, "arm_cue", f"{arm}|S-ENG", f"{arm}|S-VER", producer=m, B=B, seed=SEED)
                rows.append({"family": "descriptive", "outcome": v, "contrast": f"{arm}: S-ENG - S-VER",
                             "model": m or "ALL", **r})
            cue_mean = agg.groupby(["item_id", "model", "arm"])[v].mean().reset_index()
            r = gap(cue_mean, v, "arm", "P1", "P0", producer=m, B=B, seed=SEED)
            rows.append({"family": "descriptive", "outcome": v, "contrast": "policy cost: P1 - P0 (both cues)",
                         "model": m or "ALL", **r})
        sub = agg[agg.model == "sarvam_105b"]
        rows.append({"family": "descriptive", "outcome": v, "contrast": "schooling gap: P0 - P1 (DiD)",
                     "model": "sarvam_105b", **bootstrap_mean(did(sub, v), B, SEED)})
    t = add_holm(pd.DataFrame(rows))
    t.loc[t.family == "descriptive", "p_holm"] = np.nan
    save(t[(t.model.isin(["ALL", "sarvam_105b"]))], "revision_e10_policy_view.csv",
         ["family", "outcome", "contrast", "model", "estimate", "ci_low", "ci_high", "p_value", "p_holm"])
    t.to_csv(T / "revision_e10_policy.csv", index=False)
    dev = agg.groupby(["model", "arm", "cue"]).devanagari.mean().unstack(["arm", "cue"])
    dev.to_csv(T / "revision_e10_devanagari_by_tutor.csv")
    print(dev.round(3).to_string(), flush=True)


# =================================================================== E11: R2-T on code-mixed messages
class SeededLog(AppendLog):
    """Writes only to its own file, but treats the finished records of an older log as done,
    so cached steps (here R4's canonical English messages) are reused, never re-called."""

    def __init__(self, path, seed_path):
        super().__init__(path)
        for r in read_jsonl(seed_path):
            k = r.get("call_id")
            if k and r.get("status") == "ok" and not self.done(k):
                self.latest[k] = r


class Repair3(Repair):
    def __init__(self, steps_name: str):
        super().__init__(run, prompts)
        self.llm = llm()
        self.steps = SeededLog(run.path("runs", steps_name), run.path("runs", "repair_steps.jsonl"))


CM = ("HI-EN", "BN-EN")


def e11():
    pool = sorted(i for i, it in items.items() if it.get("in_repair"))
    dr = run.design["repair"]["drift_check"]
    drift = sorted(random.Random(SEED + int(dr["seed_offset"])).sample(pool, int(dr["items"])))
    cells = [{"item_id": i, "lang": l, "cue": "C0", "arm": "R2T", "base_prompt_id": f"A|{i}|{l}|C0"}
             for i in pool for l in CM + ("EN",)]
    cells += [{"item_id": i, "lang": l, "cue": "C0", "arm": "R2-DRIFT", "base_prompt_id": f"A|{i}|{l}|C0"}
              for i in drift for l in CM]
    jobs = repair_jobs(run, cells, prompts, stale=set())
    for j in jobs:
        j["call_id"], j["base_key"] = "e11|" + j["call_id"], "e11|" + j["base_key"]
    rp = Repair3("revision_e11_steps.jsonl")
    # every code-mixed message must already have R4's canonical English (no new to-English call)
    miss = 0
    for c in cells:
        if c["lang"] in CM:
            user = prompts[c["base_prompt_id"]]["user"]
            prot, _ = protect_math(user)
            tmpl_user = fill(prompt("utility_translate_to_en"), text=prot, source_description=CM_SOURCE[c["lang"]])
            cid = f"to_en|r4|{c['item_id']}|{c['lang']}|C0|{stable_hash(None, tmpl_user, 'utility', n=10)}"
            miss += not rp.steps.done(cid)
    print("E11: cells", len(cells), "replies", len(jobs), "judge calls", len(jobs) * 3,
          "| drift items", len(drift), "| canonical English missing:", miss, flush=True)
    if DRY:
        return
    glog = AppendLog(run.path("runs", "revision_e11_generations.jsonl"))
    print(execute(jobs, glog, rp.run_cell, workers(run, run.producers), "e11-generate"), flush=True)
    ids = {j["call_id"] for j in jobs}
    gens = [g for g in glog.records() if g["call_id"] in ids]
    targets = []
    for g in gens:
        if g["status"] != "ok":
            continue
        p = prompts[g["base_prompt_id"]]
        targets.append({"target_id": g["call_id"], "block": "R", "item_id": g["item_id"], "source": g["source"],
                        "condition": g["condition"], "student_lang": g["student_lang"],
                        "reply_lang": g["student_lang"], "cue": g["cue"], "model": g["model"], "sample_index": 0,
                        "arm": g["arm"], "variant": "original", "first_name": g.get("first_name"),
                        "surname": g.get("surname"), "subject": g.get("subject"),
                        "response_text": g.get("response_text") or "",
                        "judge_question_en": p["judge_question_en"], "judge_correct_answer": p["judge_correct_answer"],
                        "judge_student_answer": p["judge_student_answer"],
                        "misconception_name": p["misconception_name"], "language_label": p["language_label"]})
    new = judge_agg(targets, "revision_e11_judgements.jsonl", "e11-judge")
    new["arm"] = new.target_id.str.split("|").str[2].replace({"R2T": "R2T-new", "R2-DRIFT": "R2-new"})
    texts = {g["call_id"]: g.get("response_text") or "" for g in gens}
    new["latin"] = [float(bool(re.search(r"[A-Za-z]", re.sub(r"\\\(.*?\\\)", " ", texts.get(t, ""))))
                          and not re.search(r"[ऀ-ॿঀ-৿]", texts.get(t, ""))) for t in new.target_id]
    new.to_csv(R / "revision_e11_judgements_aggregated.csv", index=False)
    routes = pd.DataFrame([{"model": g["model"], "route": g.get("route_used"), "back": g.get("from_en_route")}
                           for g in gens if g["status"] == "ok"])
    print(routes.value_counts().to_string(), flush=True)
    rep = read_table(R / "repair_judgements_aggregated.csv", low_memory=False)
    rep = rep[(rep.variant == "original") & (rep.cue == "C0") & rep.condition.isin(CM + ("EN",))
              & rep.arm.isin(["R0", "R2", "R4", "R5"])]
    df = pd.concat([rep, new], ignore_index=True)
    df["withheld"] = (df.Revealing_of_the_Answer == "No").astype(float)
    df["lang_match"] = (df.Language_Match == "Yes").astype(float)
    df["math_ok"] = (df.Math_Correctness == "Yes").astype(float)
    df["arm_lang"] = df.arm + "|" + df.condition
    cells_t = df.groupby(["arm", "condition"]).agg(n=("target_id", "size"), PQI_R=("PQI_R", "mean"), PQI=("PQI", "mean"),
                                                   withheld=("withheld", "mean"), lang_match=("lang_match", "mean"),
                                                   latin=("latin", "mean")).reset_index()
    save(cells_t, "revision_e11_cells.csv")
    rows = []
    for v in ("PQI_R", "PQI", "withheld", "lang_match", "math_ok"):
        for l in CM:
            fam = "e11_primary" if v == "PQI_R" else "descriptive"
            for m in MODELS + [None]:
                f = fam if m is None else "descriptive"
                r = gap(df, v, "arm_lang", f"R2T-new|{l}", f"R2|{l}", producer=m, B=B, seed=SEED)
                rows.append({"family": f, "outcome": v, "contrast": "R2-T - R2", "language": l, "model": m or "ALL", **r})
                sub = df[df.arm == "R2T-new"]
                r = gap(sub, v, "condition", "EN", l, producer=m, B=B, seed=SEED)
                rows.append({"family": f, "outcome": v, "contrast": "R2-T gap EN - l", "language": l,
                             "model": m or "ALL", **r})
            for other in ("R4", "R5"):
                r = gap(df, v, "arm_lang", f"R2T-new|{l}", f"{other}|{l}", B=B, seed=SEED)
                rows.append({"family": "descriptive", "outcome": v, "contrast": f"R2-T - {other}", "language": l,
                             "model": "ALL", **r})
            for arm in ("R2", "R4", "R5"):
                r = gap(df[df.arm == arm], v, "condition", "EN", l, B=B, seed=SEED)
                rows.append({"family": "descriptive", "outcome": v, "contrast": f"{arm} gap EN - l", "language": l,
                             "model": "ALL", **r})
            r = gap(df, v, "arm_lang", f"R2-new|{l}", f"R2|{l}", B=B, seed=SEED)
            rows.append({"family": "descriptive", "outcome": v, "contrast": "drift: R2 now - R2 original (20 items)",
                         "language": l, "model": "ALL", **r})
    t = add_holm(pd.DataFrame(rows))
    t.loc[t.family == "descriptive", "p_holm"] = np.nan
    t.to_csv(T / "revision_e11_codemix.csv", index=False)
    save(t[(t.model == "ALL") & t.outcome.isin(["PQI_R", "withheld", "lang_match"])], "revision_e11_codemix_view.csv",
         ["family", "outcome", "contrast", "language", "estimate", "ci_low", "ci_high", "p_value", "p_holm"])


# =================================================================== E12: response time
E12_LANGS = ("HI", "BN")
E12_ARMS = ("R2", "R2L", "R2T", "R5")
MTJOBS = run.path("runs", "revision_e12_mt_jobs.jsonl")
MTOUT = run.path("runs", "revision_e12_mt_times.jsonl")


def e12_items():
    pool = sorted(i for i, it in items.items() if it.get("in_repair"))
    return sorted(random.Random(SEED + 81).sample(pool, 20))


def e12api():
    chosen = e12_items()
    jobs = [{"call_id": f"e12|{i}|{l}|{m}", "model": m, "item_id": i, "lang": l}
            for i in chosen for l in E12_LANGS for m in MODELS]
    print("E12 timing: items", len(chosen), "messages", len(jobs), "tutor calls", len(jobs) * 5, flush=True)
    tr = make_translator(run)
    sys0 = prompt("tutor_system")
    r2 = prompt("repair_R2_instruction")
    # canonical English messages must come from the IndicTrans2 cache (no GPU here)
    canon = {}
    for i in chosen:
        for l in E12_LANGS:
            for kind, text in (("r2t", prompts[f"A|{i}|{l}|C0"]["user"]),
                               ("r5", mask_names(prompts[f"A|{i}|{l}|C0"]["user"])[0])):
                prot, _ = protect_math(text)
                segs = [s for _, s in split_segments(prot) if has_words(s)]
                keys = [stable_hash(tr.model_id_indic_en, tr.tags[l], tr.tags["EN"], s, n=24) for s in segs]
                assert all(k in tr.cache for k in keys), f"uncached canonical message {i} {l} {kind}"
                canon[(i, l, kind)] = tr.indic_to_en([text], l)[0]
    print("canonical English messages: all", len(canon), "from the IndicTrans2 cache", flush=True)
    if DRY:
        return

    def timed(m, user, system, key):
        t0 = time.perf_counter()
        res = llm().call(m, user, system, seed_key=key)
        return res, time.perf_counter() - t0

    def fn(job):
        m, i, l = job["model"], job["item_id"], job["lang"]
        user = prompts[f"A|{i}|{l}|C0"]["user"]
        order = list(E12_ARMS)
        random.Random(stable_int(SEED, job["call_id"])).shuffle(order)
        out = {"call_id": job["call_id"], "model": m, "item_id": i, "lang": l, "arm_order": order, "status": "ok",
               "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        for arm in order:
            k = f"{job['call_id']}|{arm}"
            if arm == "R2":
                res, s = timed(m, user, f"{sys0}\n{r2}", k)
                out["R2"] = {"wall_s": s, "ok": res.ok, "out_tokens": res.output_tokens, "route": res.route}
            elif arm == "R2L":
                system = f"{sys0}\n{r2}\n" + fill(prompt("repair_R2L_language"), target_language_and_script=NAMES[l])
                res, s = timed(m, user, system, k)
                out["R2L"] = {"wall_s": s, "ok": res.ok, "out_tokens": res.output_tokens, "route": res.route}
            elif arm == "R2T":
                res, s = timed(m, canon[(i, l, "r2t")], f"{sys0}\n{r2}", k)
                out["R2T"] = {"wall_s": s, "ok": res.ok, "out_tokens": res.output_tokens, "route": res.route,
                              "english_reply": res.text if res.ok else ""}
            else:
                pres, s1 = timed(m, fill(prompt("repair_plan"), canonical_message_en=canon[(i, l, "r5")]), None, k + "|plan")
                t0 = time.perf_counter()
                plan, info = parse_with_repair(pres.text, llm(), k + "|plan") if pres.ok else (None, {})
                s_parse = time.perf_counter() - t0
                pj = json.dumps(plan, ensure_ascii=False) if plan is not None else (pres.text or "")
                rres, s2 = timed(m, fill(prompt("repair_realise"), target_language_and_script=NAMES[l],
                                         plan_json=pj, original_message=user), None, k + "|realise")
                out["R5"] = {"plan_s": s1, "parse_s": s_parse, "realise_s": s2, "ok": pres.ok and rres.ok,
                             "json_repair_fired": bool(info.get("json_repair_fired")),
                             "out_tokens": pres.output_tokens + rres.output_tokens, "route": rres.route}
        if not all(out[a]["ok"] for a in E12_ARMS):
            out["status"] = "error"
        return out

    glog = AppendLog(run.path("runs", "revision_e12_timing.jsonl"))
    print(execute(jobs, glog, fn, {m: 1 for m in MODELS}, "e12-timing"), flush=True)


def e12prep():
    tr = make_translator(run)
    rows = []
    for r in AppendLog(run.path("runs", "revision_e12_timing.jsonl")).records():
        if r.get("status") != "ok":
            continue
        l = r["lang"]
        for direction, text, model, src, tgt in (
                ("in", prompts[f"A|{r['item_id']}|{l}|C0"]["user"], tr.model_id_indic_en, tr.tags[l], tr.tags["EN"]),
                ("out", r["R2T"]["english_reply"], tr.model_id_en_indic, tr.tags["EN"], tr.tags[l])):
            prot, _ = protect_math(text)
            segs = [s for _, s in split_segments(prot) if has_words(s)]
            rows.append({"k": f"{r['call_id']}|{direction}", "model": model,
                         "revision": (tr.cfg.get("revisions") or {}).get(model), "src": src, "tgt": tgt,
                         "segments": segs, "chars": sum(len(s) for s in segs)})
    with open(MTJOBS, "w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"{len(rows)} MT timing jobs -> {MTJOBS}")


def e12gpu():
    sys.path.insert(0, str(Path(__file__).parent))
    import vast_translator as v  # noqa: E402
    st = v.load()
    if not st.get("ip"):
        raise SystemExit("no instance: run scripts/vast_translator.py up first")
    base = ["-i", v.KEYFILE, *v.SSH_OPTS]
    tgt = f"root@{st['ip']}"
    subprocess.run(["scp", "-P", str(st["port"]), *base, str(MTJOBS), str(Path(__file__).parent / "mt_timing_job.py"),
                    f"{tgt}:/root/"], check=True, timeout=600)
    # stop the translation service so the timing has the GPU to itself
    v.ssh(st, "pkill -f indictrans_server.py; sleep 3; nvidia-smi --query-gpu=name,memory.used --format=csv", 120)
    r = v.ssh(st, f"cd /root && HF_HUB_OFFLINE=1 GAPTRACE_ATTN={st.get('attn', 'sdpa')} python mt_timing_job.py "
                  f"{MTJOBS.name} mt_times.jsonl > mt.log 2>&1; tail -5 mt.log", timeout=7200)
    print(r.stdout[-800:], r.stderr[-300:])
    subprocess.run(["scp", "-P", str(st["port"]), *base, f"{tgt}:/root/mt_times.jsonl", str(MTOUT)], check=True,
                   timeout=600)
    print("fetched", sum(1 for _ in open(MTOUT)), "timings")


def e12():
    mt = {json.loads(x)["k"]: json.loads(x) for x in open(MTOUT, encoding="utf-8")}
    rows = []
    for r in AppendLog(run.path("runs", "revision_e12_timing.jsonl")).records():
        if r.get("status") != "ok":
            continue
        mi, mo = mt.get(f"{r['call_id']}|in"), mt.get(f"{r['call_id']}|out")
        if not mi or not mo:
            continue
        base = {"model": r["model"], "item_id": r["item_id"], "lang": r["lang"]}
        rows.append({**base, "design": "R2", "total_s": r["R2"]["wall_s"], "tutor_s": r["R2"]["wall_s"], "mt_s": 0.0})
        rows.append({**base, "design": "R2-L", "total_s": r["R2L"]["wall_s"], "tutor_s": r["R2L"]["wall_s"], "mt_s": 0.0})
        rows.append({**base, "design": "R2-T", "tutor_s": r["R2T"]["wall_s"], "mt_s": mi["median_s"] + mo["median_s"],
                     "total_s": mi["median_s"] + r["R2T"]["wall_s"] + mo["median_s"]})
        t5 = r["R5"]["plan_s"] + r["R5"]["parse_s"] + r["R5"]["realise_s"]
        rows.append({**base, "design": "R5", "tutor_s": t5, "mt_s": mi["median_s"], "total_s": mi["median_s"] + t5})
    d = pd.DataFrame(rows)
    d.to_csv(R / "revision_e12_latency_long.csv", index=False)
    q = lambda p: (lambda s: float(np.quantile(s, p)))  # noqa: E731
    summ = d.groupby("design").agg(n=("total_s", "size"), median_total_s=("total_s", "median"),
                                   p90_total_s=("total_s", q(0.9)), median_tutor_s=("tutor_s", "median"),
                                   median_mt_s=("mt_s", "median")).reset_index()
    save(summ, "revision_e12_latency.csv")
    by = d.groupby(["model", "design"]).total_s.median().unstack("design")
    by.to_csv(T / "revision_e12_latency_by_tutor.csv")
    print(by.round(2).to_string())
    dev = sorted({x.get("device", "") for x in mt.values()})
    mtd = pd.DataFrame(mt.values())
    print("MT device:", dev, "| median s per message: in", round(mtd[mtd.k.str.endswith("|in")].median_s.median(), 3),
          "out", round(mtd[mtd.k.str.endswith("|out")].median_s.median(), 3))
    pd.DataFrame([{"device": "; ".join(dev), "jobs": len(mtd),
                   "median_in_s": float(mtd[mtd.k.str.endswith("|in")].median_s.median()),
                   "median_out_s": float(mtd[mtd.k.str.endswith("|out")].median_s.median())}]).to_csv(
        T / "revision_e12_mt_device.csv", index=False)


if __name__ == "__main__":
    {"e10": e10, "e11": e11, "e12api": e12api, "e12prep": e12prep, "e12gpu": e12gpu, "e12": e12}[sys.argv[1]]()
