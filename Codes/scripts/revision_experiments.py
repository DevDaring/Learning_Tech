"""Revision experiments E2 (evaluator challenge set), E3 (plan-fidelity check and plan-only
ablation) and E4 (completed input/output crossing), as frozen in docs/revision_plan_frozen.md.
Separate logs (runs/revision_*.jsonl); nothing in the main study is changed. Samples are drawn
with fixed seeds before any call; re-running reuses the logged calls.

    PYTHONPATH=src python scripts/revision_experiments.py e2|e3sem|e3abl|e4|all [--dry]
"""
import json
import math
import random
import re
import sys

import numpy as np
import pandas as pd

from tlt.analysis.stats import add_holm, bootstrap_mean, gap
from tlt.clients import make_llm
from tlt.config import load_run, stable_hash
from tlt.io import AppendLog, fill, prompt, read_jsonl, read_table
from tlt.jsonfix import parse_with_repair
from tlt.rubric import PQI_RELIABLE, PQI_SCORES
from tlt.runner import call_record, execute
from tlt.stages.s04_generate import generation_jobs, make_fn, workers
from tlt.stages.s05_mask_and_judge import aggregate_table, judge_targets, long_table, targets_from_generations

run = load_run("full")
R = run.path("results")
T = R / "tables"
B = int(run.design["bootstrap"]["resamples"])
SEED = run.seed
DRY = "--dry" in sys.argv
CORE = list(run.design["languages"]["core"])
LABELS = run.design["languages"]["labels"]
NAMES = run.design["languages"]["names"]
MODELS = sorted(run.producers)
prompts = {p["prompt_id"]: p for p in read_jsonl(run.path("data", "prompts.jsonl"))}
items = {it["item_id"]: it for it in read_jsonl(run.path("data", "items_multilingual.jsonl"))}
_llm = None


def llm():
    global _llm
    if _llm is None:
        _llm = make_llm(run)
    return _llm


def recs_by_id(log: str) -> dict:
    out = {}
    for line in open(run.path("runs", log), encoding="utf-8"):
        r = json.loads(line)
        out[r["call_id"]] = r  # last record wins (append-only log)
    return out


def stratified(df: pd.DataFrame, per: int, seed: int, by=("condition", "model")) -> pd.DataFrame:
    """`per` rows per language x tutor, drawn with a fixed seed from a sorted frame."""
    rng = random.Random(seed)
    picks = []
    for _, g in df.sort_values("target_id").groupby(list(by)):
        ids = sorted(g.target_id)
        picks += rng.sample(ids, min(per, len(ids)))
    return df[df.target_id.isin(picks)].sort_values(["condition", "model", "target_id"])


def judge_and_aggregate(targets: list[dict], log: str, label: str) -> pd.DataFrame:
    jlog = AppendLog(run.path("runs", log))
    counts, jids = judge_targets(run, llm(), targets, jlog, label)
    print(label, counts)
    return aggregate_table(long_table([r for r in jlog.records() if r["call_id"] in jids]),
                           run.design["primary_judge"], run.judges)


def add_pqi_r(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for j in run.judges:
        df[f"PQI_R__{j}"] = pd.concat([df[f"{d}__{j}"].map(PQI_SCORES[d]) for d in PQI_RELIABLE], axis=1).mean(
            axis=1, skipna=False)
    return df


def boot_rate(x: np.ndarray, seed: int = SEED) -> tuple:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return math.nan, math.nan, math.nan, 0
    b = x[np.random.default_rng(seed).integers(0, len(x), size=(B, len(x)))].mean(1)
    return float(x.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975)), len(x)


def target_from(meta: dict, text: str, target_id: str, variant: str, prompt_id: str, arm: str) -> dict:
    p = prompts[prompt_id]
    t = {k: meta.get(k) for k in ("block", "item_id", "source", "condition", "student_lang", "reply_lang", "cue",
                                  "model", "first_name", "surname", "subject")}
    t.update({"target_id": target_id, "variant": variant, "arm": arm, "sample_index": meta.get("sample_index", 0),
              "response_text": text, "judge_question_en": p["judge_question_en"],
              "judge_correct_answer": p["judge_correct_answer"], "judge_student_answer": p["judge_student_answer"],
              "misconception_name": p["misconception_name"], "language_label": p["language_label"]})
    return t


# =================================================================== E2: evaluator challenge set
ANSWER_SENTENCE = {"EN": "The answer is {a}.", "HI": "उत्तर {a} है।", "BN": "উত্তর হল {a}।",
                   "HI-EN": "Answer {a} hai.", "BN-EN": "Answer holo {a}."}
FALSE_LEAD = {"EN": "Also,", "HI": "साथ ही,", "BN": "এছাড়া,", "HI-EN": "Aur,", "BN-EN": "Aar,"}
FALSE_STMT = r"\( 7 \times 8 = 54 \)"


def normalise_ws(text: str) -> str:
    """Whitespace only: runs of spaces/tabs -> one space, trailing spaces dropped, blank lines
    removed. Words, maths and line order are unchanged."""
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n")]
    return "\n".join(ln for ln in lines if ln)


def e2():
    rep = read_table(R / "repair_judgements_aggregated.csv", low_memory=False)
    pool_a = rep[(rep.variant == "original") & rep.arm.isin(["R2", "R5"]) & (rep.cue == "C0")
                 & (rep.Revealing_of_the_Answer == "No")]
    sel_a = stratified(pool_a, 2, SEED + 61)
    anc = read_table(R / "human_anchor_judgements_aggregated.csv", low_memory=False)
    sel_b = stratified(anc[anc.Math_Correctness == "Yes"], 2, SEED + 62)
    print("E2 eligible: (a)", len(pool_a), "selected", len(sel_a), "| (b,c)", int((anc.Math_Correctness == "Yes").sum()),
          "selected", len(sel_b))
    rgen, ggen = recs_by_id("repair_generations.jsonl"), recs_by_id("generations.jsonl")
    targets, rows = [], []
    for _, r in sel_a.iterrows():
        g = rgen[r.target_id]
        lang = g["repair_lang"]
        ans = items[g["item_id"]]["versions"][lang]["correct_answer"]
        text = (g.get("response_text") or "").rstrip() + "\n\n" + fill(ANSWER_SENTENCE[lang], a=ans)
        targets.append(target_from(g, text, f"chal-a|{r.target_id}", "challenge_answer", g["base_prompt_id"], g["arm"]))
        rows.append({"variant": "answer", "orig": r.target_id, "lang": lang, "model": g["model"]})
    for _, r in sel_b.iterrows():
        g = ggen[r.target_id]
        lang = g["student_lang"]
        raw = g.get("raw_text") or ""
        t1 = raw.rstrip() + "\n\n" + FALSE_LEAD[lang] + " " + FALSE_STMT
        t2 = normalise_ws(raw)
        targets.append(target_from(g, t1, f"chal-b|{r.target_id}", "challenge_false_math", g["prompt_id"], "R0"))
        targets.append(target_from(g, t2, f"chal-c|{r.target_id}", "challenge_whitespace", g["prompt_id"], "R0"))
        rows += [{"variant": "false_math", "orig": r.target_id, "lang": lang, "model": g["model"]},
                 {"variant": "whitespace", "orig": r.target_id, "lang": lang, "model": g["model"],
                  "text_changed": t2 != raw}]
    man = pd.DataFrame(rows)
    man.to_csv(T / "revision_e2_manifest.csv", index=False)
    print("targets", len(targets), "judge calls", len(targets) * len(run.judges),
          "| whitespace variants that changed the text:", int(man.text_changed.fillna(False).sum()))
    if DRY:
        return
    va = add_pqi_r(judge_and_aggregate(targets, "revision_e2_judgements.jsonl", "e2-judge"))
    va.to_csv(R / "revision_e2_judgements_aggregated.csv", index=False)
    orig = add_pqi_r(pd.concat([rep, anc], ignore_index=True)).drop_duplicates("target_id").set_index("target_id")
    va["orig"] = va.target_id.str.split("|", n=1).str[1]
    va["kind"] = va.target_id.str.split("|", n=1).str[0]
    out = []
    for who in ["panel"] + list(run.judges):
        suf = "" if who == "panel" else f"__{who}"
        for kind, dim, hit in (("chal-a", "Revealing_of_the_Answer", lambda s: s.isin(["Yes (correct)", "Yes (incorrect)"])),
                               ("chal-b", "Math_Correctness", lambda s: s == "No")):
            v = va[va.kind == kind]
            o = orig.loc[v.orig, dim + suf].values
            # detection counted where this judge (or the panel) gave the expected original label
            base_ok = (o == ("No" if kind == "chal-a" else "Yes"))
            det = hit(v[dim + suf]).astype(float).where(v[dim + suf].notna()).values
            m, lo, hi, n = boot_rate(det[base_ok])
            m2, lo2, hi2, n2 = boot_rate(det)
            out.append({"who": who, "check": {"chal-a": "(a) appended correct answer -> Revealing = Yes",
                                              "chal-b": "(b) appended 7x8=54 -> Math_Correctness = No"}[kind],
                        "detection_rate": m, "ci_low": lo, "ci_high": hi, "n": n,
                        "detection_rate_all_selected": m2, "all_ci_low": lo2, "all_ci_high": hi2, "n_all": n2})
        v = va[va.kind == "chal-c"]
        for score in ("PQI", "PQI_R"):
            col = score + suf
            d = np.abs(v[col].values - orig.loc[v.orig, col].values)
            m, lo, hi, n = boot_rate(d)
            sd = v[col].values - orig.loc[v.orig, col].values
            out.append({"who": who, "check": f"(c) whitespace only -> |delta {score}|", "detection_rate": m,
                        "ci_low": lo, "ci_high": hi, "n": n, "mean_signed_delta": float(np.nanmean(sd)),
                        "share_unchanged": float(np.nanmean(d == 0))})
        lab = []
        for d in PQI_SCORES:
            col = d + suf
            lab.append(float(np.mean(v[col].values != orig.loc[v.orig, col].values)))
        out.append({"who": who, "check": "(c) whitespace only -> share of PQI component labels changed",
                    "detection_rate": float(np.mean(lab)), "n": len(v)})
    t = pd.DataFrame(out)
    t.to_csv(T / "revision_e2_challenge.csv", index=False)
    print(t.round(3).to_string(index=False))


# =================================================================== E3: plan fidelity
def r5_records(langs=None) -> pd.DataFrame:
    rep = read_table(R / "repair_judgements_aggregated.csv", low_memory=False)
    keep = rep[(rep.variant == "original") & (rep.arm == "R5") & (rep.cue == "C0")]
    if langs:
        keep = keep[keep.condition.isin(langs)]
    return keep


def plan_string(rec: dict) -> str:
    """The exact plan text the saved realiser saw (taken from its logged prompt)."""
    u = rec.get("realise_user") or ""
    m = re.search(r"\nPlan: (.*)\nStudent's original message: ", u, re.S)
    if not m:
        raise ValueError(f"no plan in realise prompt: {rec['call_id']}")
    return m.group(1)


def e3sem():
    sel = stratified(r5_records(), 2, SEED + 63)
    rgen = recs_by_id("repair_generations.jsonl")
    tmpl = prompt("revision_plan_fidelity")
    jobs = []
    for _, r in sel.iterrows():
        g = rgen[r.target_id]
        user = fill(tmpl, plan_json=plan_string(g), reply=g.get("response_text") or "")
        h = stable_hash(user, n=10)
        for j in run.judges:
            jobs.append({"call_id": f"fidelity|{r.target_id}|{j}|{h}", "model": j, "user": user, "target": r.target_id,
                         "lang": g["repair_lang"], "tutor": g["model"], "reply": g.get("response_text") or ""})
    print("E3 semantic: pairs", len(sel), "calls", len(jobs))
    if DRY:
        return

    def fn(job):
        res = llm().call(job["model"], job["user"], None, seed_key=job["call_id"])
        obj, info = parse_with_repair(res.text, llm(), job["call_id"]) if res.ok else (None, {})
        rec = call_record({"call_id": job["call_id"], "model": job["model"], "role": "plan_fidelity_judge",
                           "target_id": job["target"], "lang": job["lang"], "tutor": job["tutor"],
                           "parsed": obj, "parse_ok": obj is not None, **info}, res)
        if res.ok and obj is None:
            rec["status"] = "unparseable"
        return rec

    log = AppendLog(run.path("runs", "revision_e3_fidelity.jsonl"))
    print(execute(jobs, log, fn, workers(run, run.judges), "e3-fidelity"))
    ids = {j["call_id"]: j for j in jobs}
    rows = []
    for r in log.records():
        if r["call_id"] not in ids or r.get("status") != "ok":
            continue
        reply = ids[r["call_id"]]["reply"]
        for q in ("contradicts_plan", "omits_next_step", "reveals_forbidden"):
            o = (r.get("parsed") or {}).get(q) or {}
            ans = str(o.get("answer", "")).strip().lower()
            quote = str(o.get("quote", "") or "").strip()
            rows.append({"target_id": r["target_id"], "judge": r["model"], "lang": r["lang"], "tutor": r["tutor"],
                         "question": q, "flag": 1.0 if ans.startswith("y") else (0.0 if ans.startswith("n") else np.nan),
                         "quote_given": bool(quote), "quote_verbatim": bool(quote) and quote in reply})
    d = pd.DataFrame(rows)
    d.to_csv(R / "revision_e3_fidelity_long.csv", index=False)
    out = []
    for q, s in d.groupby("question"):
        for j, sj in s.groupby("judge"):
            m, lo, hi, n = boot_rate(sj.flag.values)
            fl = sj[sj.flag == 1]
            out.append({"question": q, "who": j, "flag_rate": m, "ci_low": lo, "ci_high": hi, "n": n,
                        "quote_verbatim_share_of_flags": float(fl.quote_verbatim.mean()) if len(fl) else math.nan})
        wide = s.pivot_table(index="target_id", columns="judge", values="flag")
        maj = (wide.sum(1) >= 2).astype(float).where(wide.notna().sum(1) == len(run.judges))
        m, lo, hi, n = boot_rate(maj.values)
        js = list(wide.columns)
        agree = np.mean([np.nanmean(wide[a] == wide[b]) for i, a in enumerate(js) for b in js[i + 1:]])
        out.append({"question": q, "who": "majority (2 of 3)", "flag_rate": m, "ci_low": lo, "ci_high": hi, "n": n,
                    "pairwise_judge_agreement": float(agree)})
    t = pd.DataFrame(out)
    t.to_csv(T / "revision_e3_fidelity.csv", index=False)
    print(t.round(3).to_string(index=False))


ABL_TUTORS = ["gemini_flash", "sarvam_105b", "nova_lite"]  # frozen: one per tier, largest default gap in tier
ABL_LANGS = ["EN", "HI", "BN"]


def e3abl():
    saved = r5_records(ABL_LANGS)
    pool = sorted(set(saved.item_id))
    chosen = sorted(random.Random(SEED + 64).sample(pool, 50))
    saved = saved[saved.item_id.isin(chosen) & saved.model.isin(ABL_TUTORS)]
    rgen = recs_by_id("repair_generations.jsonl")
    tmpl = prompt("repair_realise_planonly")
    jobs = []
    for _, r in saved.iterrows():
        g = rgen[r.target_id]
        user = fill(tmpl, target_language_and_script=NAMES[g["repair_lang"]], plan_json=plan_string(g))
        h = stable_hash(user, run.models["producer_settings"]["temperature"],
                        run.models["producer_settings"]["max_tokens"], n=10)
        jobs.append({"call_id": f"ablation|planonly|{r.target_id}|{h}", "model": g["model"], "user": user, "g": g})
    print("E3 ablation: items", len(chosen), "replies", len(jobs), "judge calls", len(jobs) * len(run.judges))
    if DRY:
        return

    def fn(job):
        g = job["g"]
        res = llm().call(job["model"], job["user"], None, seed_key=job["call_id"])
        meta = {k: g.get(k) for k in ("block", "item_id", "source", "condition", "student_lang", "reply_lang", "cue",
                                      "first_name", "surname", "subject", "base_prompt_id", "repair_lang")}
        meta.update({"call_id": job["call_id"], "model": job["model"], "role": "producer", "arm": "R5-PLANONLY",
                     "source_r5_call_id": g["call_id"], "system": None, "user": job["user"]})
        rec = call_record(meta, res)
        rec["response_text"] = res.text if res.ok else ""
        return rec

    glog = AppendLog(run.path("runs", "revision_e3_ablation_generations.jsonl"))
    print(execute(jobs, glog, fn, workers(run, ABL_TUTORS), "e3-ablation"))
    ids = {j["call_id"] for j in jobs}
    targets = [target_from(g, g.get("response_text") or "", g["call_id"], "original", g["base_prompt_id"], "R5-PLANONLY")
               for g in glog.records() if g["call_id"] in ids and g["status"] == "ok"]
    new = judge_and_aggregate(targets, "revision_e3_ablation_judgements.jsonl", "e3-ablation-judge")
    new.to_csv(R / "revision_e3_ablation_judgements_aggregated.csv", index=False)
    old = saved.copy()
    df = pd.concat([old.assign(arm="R5"), new.assign(arm="R5-PLANONLY")], ignore_index=True)
    df["withheld"] = (df.Revealing_of_the_Answer == "No").astype(float)
    df["lang_match"] = (df.Language_Match == "Yes").astype(float)
    df["arm_lang"] = df.arm + "|" + df.condition
    rows = []
    for v in ("PQI_R", "PQI", "withheld", "lang_match"):
        for l in ABL_LANGS + ["pooled"]:
            a = [f"R5-PLANONLY|{x}" for x in ABL_LANGS] if l == "pooled" else f"R5-PLANONLY|{l}"
            b = [f"R5|{x}" for x in ABL_LANGS] if l == "pooled" else f"R5|{l}"
            for m in ABL_TUTORS + [None]:
                r = gap(df, v, "arm_lang", a, b, producer=m, B=B, seed=SEED)
                rows.append({"family": f"e3_planonly_{v}" if (m is None and l != "pooled") else "descriptive",
                             "outcome": v, "contrast": "plan-only - saved R5", "language": l, "model": m or "ALL", **r})
        for arm in ("R5", "R5-PLANONLY"):
            for l in ("HI", "BN"):
                r = gap(df, v, "arm_lang", f"{arm}|EN", f"{arm}|{l}", B=B, seed=SEED)
                rows.append({"family": "descriptive", "outcome": v, "contrast": f"{arm} gap EN-{l}", "language": l,
                             "model": "ALL", **r})
    t = add_holm(pd.DataFrame(rows))
    t.loc[t.family == "descriptive", "p_holm"] = np.nan
    t.to_csv(T / "revision_e3_ablation.csv", index=False)
    print(t[t.model == "ALL"][["outcome", "contrast", "language", "estimate", "ci_low", "ci_high", "p_value",
                               "p_holm"]].round(4).to_string(index=False))


# =================================================================== E4: completed crossing
E4_CELLS = [("EN", "EN"), ("HI", "HI"), ("BN", "BN"), ("HI", "EN"), ("BN", "EN")]
LANG_WORD = {"EN": "English", "HI": "Hindi", "BN": "Bengali"}
SCRIPT = {"EN": re.compile(r"[A-Za-z]"), "HI": re.compile(r"[ऀ-ॿ]"), "BN": re.compile(r"[ঀ-৿]")}


def dominant_script(text: str) -> str:
    t = re.sub(r"\\\(.*?\\\)|\\\[.*?\\\]|\$[^$]*\$", " ", text or "", flags=re.S)  # ignore maths
    counts = {k: len(p.findall(t)) for k, p in SCRIPT.items()}
    return max(counts, key=counts.get) if any(counts.values()) else "none"


def e4():
    sys0 = prompt("tutor_system")
    citems = sorted(i for i, it in items.items() if it.get("in_block_c"))
    new = []
    for iid in citems:
        for inp, out in E4_CELLS:
            base = prompts[f"A|{iid}|{inp}|C0"]
            p = dict(base)
            p.update({"prompt_id": f"X|{iid}|{inp}>{out}|C0", "block": "X", "condition": f"{inp}>{out}",
                      "student_lang": inp, "reply_lang": out, "samples": 1,
                      "system": f"{sys0}\n{prompt(f'block_c_reply_{out}')}",
                      "language_label": f"{LABELS[inp].split(' (')[0]} (the tutor was instructed to reply in "
                                        f"{LANG_WORD[out]})"})
            new.append(p)
    jobs = generation_jobs(run, new)
    print("E4: items", len(citems), "prompts", len(new), "replies", len(jobs), "judge calls", len(jobs) * len(run.judges))
    print("example label:", new[3]["language_label"], "| instruction:", new[3]["system"].split("\n")[-1])
    if DRY:
        return
    glog = AppendLog(run.path("runs", "revision_e4_generations.jsonl"))
    print(execute(jobs, glog, make_fn(run, llm()), workers(run, run.producers), "e4-generate"))
    ids = {j["call_id"] for j in jobs}
    gens = [r for r in glog.records() if r["call_id"] in ids]
    targets = targets_from_generations(gens, {p["prompt_id"]: p for p in new})
    newagg = judge_and_aggregate(targets, "revision_e4_judgements.jsonl", "e4-judge")
    newagg.to_csv(R / "revision_e4_judgements_aggregated.csv", index=False)
    # EN>HI and EN>BN: Block C sample 0 (same instruction wording, collected 2026-10-07/08)
    agg = read_table(R / "judgements_aggregated.csv", low_memory=False)
    c0 = agg[(agg.block == "C") & (agg.variant == "original") & (agg.sample_index.astype(str) == "0")
             & agg.item_id.isin(citems)]
    df = pd.concat([newagg, c0], ignore_index=True)
    texts = {g["call_id"]: g.get("raw_text") or "" for g in gens}
    c0_ids = set(c0.target_id)
    texts.update({g["call_id"]: g.get("raw_text") or "" for g in
                  (json.loads(x) for x in open(run.path("runs", "generations.jsonl"), encoding="utf-8"))
                  if g["call_id"] in c0_ids})
    df["out_lang"] = df.condition.str.split(">").str[1]
    df["script_ok"] = [float(dominant_script(texts.get(t, "")) == o) for t, o in zip(df.target_id, df.out_lang)]
    comp = df.groupby("condition").agg(n=("target_id", "size"), script_compliance=("script_ok", "mean"),
                                       language_match_yes=("Language_Match", lambda s: float((s == "Yes").mean())),
                                       PQI=("PQI", "mean"), PQI_R=("PQI_R", "mean")).reset_index()
    comp.to_csv(T / "revision_e4_cells.csv", index=False)
    print(comp.round(3).to_string(index=False))
    rows = []
    for v in ("PQI_R", "PQI"):
        for l in ("HI", "BN"):
            cells = {k: f"{a}>{b}" for k, (a, b) in {"ee": ("EN", "EN"), "el": ("EN", l), "le": (l, "EN"),
                                                     "ll": (l, l)}.items()}
            w = (df[df.condition.isin(cells.values())].groupby(["item_id", "model", "condition"])[v].mean()
                 .unstack("condition"))
            w = w.dropna(subset=list(cells.values()))
            ee, el, le, ll = (w[cells[k]] for k in ("ee", "el", "le", "ll"))
            effects = {"output effect (English minus target output, averaged over input)": ((ee - el) + (le - ll)) / 2,
                       "input effect (English minus target input, averaged over output)": ((ee - le) + (el - ll)) / 2,
                       "interaction (output effect with English input minus with target input)": (ee - el) - (le - ll)}
            for name, s in effects.items():
                s = s.reset_index()
                cnt = s.groupby("item_id").model.nunique()
                per_item = s[s.item_id.isin(cnt[cnt == len(MODELS)].index)].groupby("item_id")[0].mean()
                r = bootstrap_mean(per_item, B, SEED)
                rows.append({"family": f"e4_{v}", "outcome": v, "language": l, "effect": name, **r})
    t = add_holm(pd.DataFrame(rows))
    t.to_csv(T / "revision_e4_crossing.csv", index=False)
    print(t[["outcome", "language", "effect", "estimate", "ci_low", "ci_high", "p_value", "p_holm"]].round(4)
          .to_string(index=False))


def e4diag():
    """Secondary diagnostics for E4 (labelled as such: compliance is post-assignment, so
    compliant-only contrasts are selected samples). Per-tutor output-script compliance, and the
    English-output contrast EN>EN minus l>EN among compliant replies."""
    new = read_table(R / "revision_e4_judgements_aggregated.csv", low_memory=False)
    texts = {g["call_id"]: g.get("raw_text") or "" for g in
             (json.loads(x) for x in open(run.path("runs", "revision_e4_generations.jsonl"), encoding="utf-8"))}
    new["out_lang"] = new.condition.str.split(">").str[1]
    new["comply"] = [float(dominant_script(texts.get(t, "")) == o) for t, o in zip(new.target_id, new.out_lang)]
    comp = new.groupby(["condition", "model"]).comply.mean().unstack(0)
    comp.loc["ALL"] = new.groupby("condition").comply.mean()
    comp.to_csv(T / "revision_e4_compliance_by_tutor.csv")
    print(comp.round(2).to_string())
    rows = []
    for l in ("HI", "BN"):
        sub = new[new.condition.isin(["EN>EN", f"{l}>EN"])]
        for label, s in (("all replies (as assigned)", sub),
                         ("compliant replies only (diagnostic)", sub[(sub.condition == "EN>EN") | (sub.comply == 1)])):
            for v in ("PQI_R", "PQI"):
                pdf = s.groupby(["item_id", "model", "condition"])[v].mean().unstack("condition").dropna()
                d = (pdf["EN>EN"] - pdf[f"{l}>EN"]).groupby("item_id").mean()
                r = bootstrap_mean(d, B, SEED)
                rows.append({"language": l, "sample": label, "outcome": v, "n_item_tutor_pairs": len(pdf),
                             "mean_compliant_share": float(new[new.condition == f"{l}>EN"].comply.mean()), **r})
    t = pd.DataFrame(rows)
    t["family"] = "e4diag_" + t["sample"] + "_" + t.outcome  # Holm across HI and BN
    t = add_holm(t)
    t.to_csv(T / "revision_e4_compliance_contrast.csv", index=False)
    print(t[["language", "sample", "outcome", "n_item_tutor_pairs", "estimate", "ci_low", "ci_high"]].round(4)
          .to_string(index=False))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, f in (("e2", e2), ("e3sem", e3sem), ("e3abl", e3abl), ("e4", e4), ("e4diag", e4diag)):
        if cmd in (name, "all"):
            print(f"\n===== {name} =====")
            f()
