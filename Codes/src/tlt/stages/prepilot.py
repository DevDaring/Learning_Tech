"""Pre-pilot (exploratory go/no-go study), as fixed in docs/prepilot_criteria.md.

    python -m tlt prepilot

40 BEnQA maths items NOT in the main Block D sample; native Bengali and official English;
cues C0, S-ENG, S-VER, N-MD, N-FD, N-MM, N-FM; all six producers; one sample; the final
three-judge panel. Uses the production generation, judging, auto-metric and statistics
code; writes everything under Codes/prepilot/. The report is
prepilot/results/prepilot_report.md (+ JSON and CSV tables).
"""
from __future__ import annotations

import copy
import json
import logging
import math
import random
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..analysis.stats import add_holm, diff_in_diff, gap, krippendorff_alpha
from ..config import CONFIG_DIR, Run, load_yaml
from ..io import AppendLog, prompt, read_table, write_jsonl
from ..rubric import ALPHA_LEVEL, DIMENSIONS, PQI_SCORES, ordinal_index
from . import s04_generate, s05_mask_and_judge, s08_auto_metrics
from .s01_build_items import benqa_pool, load_benqa, stratified_subset
from .s03_build_prompts import cue_sentence, student_message, with_cue

log = logging.getLogger("tlt.prepilot")

N_ITEMS = 40
LANGS = {"EN": "EN-OFFICIAL", "BN": "BN-NATIVE"}
CUES = ["C0", "S-ENG", "S-VER", "N-MD", "N-FD", "N-MM", "N-FM"]


def prepilot_run(base: Run) -> Run:
    design = copy.deepcopy(load_yaml(CONFIG_DIR / "design.yaml"))
    design["blocks"]["P"] = {"samples": 1}
    return Run(design=design, models=base.models, mode="prepilot")


def build(run: Run) -> list[dict]:
    """Items and prompts. Items exclude the full study's Block D sample."""
    full_items, _ = load_benqa(Run(design=load_yaml(CONFIG_DIR / "design.yaml"), models=run.models, mode="full"))
    taken = {it["item_id"] for it in full_items}
    pool = [it for it in benqa_pool(run)[0] if it["item_id"] not in taken]
    rng = random.Random(run.seed + 77)
    chosen_ids = {r["item_id"] for r in stratified_subset([{"item_id": it["item_id"], "grade": it["grade"]}
                                                           for it in pool], N_ITEMS, "grade", rng)}
    items = sorted([it for it in pool if it["item_id"] in chosen_ids], key=lambda x: x["item_id"])
    labels = run.design["languages"]["labels"]
    sys0 = prompt("tutor_system")
    prompts, multi = [], []
    for rank, it in enumerate(items):
        versions = {"EN": {"question": it["question_en"], "student_answer": it["student_answer_en"],
                           "correct_answer": it["correct_answer_en"]},
                    "BN": {"question": it["question_bn_native"], "student_answer": it["student_answer_bn_native"],
                           "correct_answer": it["correct_answer_bn_native"]}}
        multi.append({**it, "versions": versions, "name_rank": rank})
        for lang in LANGS:
            v = versions[lang]
            msg = student_message(lang, v["question"], v["student_answer"])
            for cue in CUES:
                ctext, nm = cue_sentence(cue, lang, rank)
                prompts.append({"prompt_id": f"P|{it['item_id']}|{lang}|{cue}", "block": "P", "item_id": it["item_id"],
                                "source": "benqa", "condition": lang, "student_lang": lang, "reply_lang": lang,
                                "cue": cue, "system": sys0, "user": with_cue(ctext, msg),
                                "judge_question_en": it["question_en"], "judge_correct_answer": it["correct_answer_en"],
                                "judge_student_answer": it["student_answer_en"],
                                "misconception_name": it["misconception_name"], "language_label": labels[lang],
                                "correct_answer_student_lang": v["correct_answer"], "subject": it["grade"],
                                "first_name": nm.get("first_name"), "surname": nm.get("surname"), "samples": 1})
    write_jsonl(run.path("data", "items_multilingual.jsonl"), multi)
    write_jsonl(run.path("data", "prompts.jsonl"), prompts)
    return prompts


# ------------------------------------------------------------------------- analysis

def analyse(run: Run) -> dict:
    agg = read_table(run.path("results", "judgements_aggregated.csv"))
    long = read_table(run.path("results", "judgements_long.csv"))
    auto = pd.read_csv(run.path("results", "auto_metrics.csv"))
    agg = agg.merge(auto[["target_id", "truncated", "is_empty", "refusal", "cue_echo", "word_count",
                          "dominant_script", "leak_exact", "degenerate"]], on="target_id", how="left")
    B = int(run.design["bootstrap"]["resamples"])
    models = sorted(agg.model.unique())
    out: dict = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    tdir = run.path("results", "tables", "x").parent

    # ---- A. feasibility
    gens = s04_generate.current_generations(run)
    jrecs = [r for r in AppendLog(run.path("runs", "judgements.jsonl")).records()]
    ok_g = [g for g in gens if g["status"] == "ok"]
    full_valid = [r for r in jrecs if r.get("status") == "ok" and not r.get("invalid_dims")]
    trunc = auto.groupby(["condition", "model"])["truncated"].mean().unstack("model")
    trunc.to_csv(tdir / "truncation_rate.csv")
    real_trunc = auto.assign(t=auto.truncated & ~auto.degenerate).groupby(["condition", "model"])["t"].mean()
    degen = auto.groupby(["condition", "model"])["degenerate"].mean().unstack("model")
    degen.to_csv(tdir / "degenerate_rate.csv")
    out["A_feasibility"] = {
        "generation_success": len(ok_g) / max(1, len(gens)), "generations": len(gens),
        "judge_calls": len(jrecs), "judge_fully_valid": len(full_valid) / max(1, len(jrecs)),
        "json_repair_rate": float(np.mean([bool(r.get("json_repair_fired")) for r in jrecs])) if jrecs else None,
        "max_truncation_cell": float(np.nanmax(trunc.values)) if trunc.size else None,
        "max_truncation_cell_excluding_degenerate_loops": float(real_trunc.max()) if len(real_trunc) else None,
        "degenerate_loops": {f"{m}|{c}": int(v) for (c, m), v in
                             auto.groupby(["condition", "model"])["degenerate"].sum().items() if v},
        "empty_replies": int(auto.is_empty.sum()), "refusals": int(auto.refusal.sum()),
        "fallback_rate_generation": float(np.mean([bool(g.get("fallback_used")) for g in ok_g])) if ok_g else None,
    }
    # measured cost
    prices = load_yaml(CONFIG_DIR / "prices.yaml")
    fx = float(prices["inr_per_usd"])

    def usd(model, tin, tout):
        p = prices["models"].get(model, {})
        pin = p.get("input", p.get("input_inr", 0) / fx if "input_inr" in p else 0)
        pout = p.get("output", p.get("output_inr", 0) / fx if "output_inr" in p else 0)
        return (tin * pin + tout * pout) / 1e6
    cost_rows = [{"model": g["model"], "role": "producer", "usd": usd(g["model"], g.get("input_tokens", 0), g.get("output_tokens", 0)),
                  "route": g.get("route_used")} for g in ok_g]
    cost_rows += [{"model": r["judge"], "role": "judge", "usd": usd(r["judge"], r.get("input_tokens", 0), r.get("output_tokens", 0)),
                   "route": r.get("route_used")} for r in jrecs if r.get("status") in ("ok", "unparseable")]
    cdf = pd.DataFrame(cost_rows)
    cdf.groupby(["role", "model", "route"])["usd"].agg(["count", "sum"]).to_csv(tdir / "measured_cost.csv")
    per_gen = cdf.usd.sum() / max(1, len(ok_g))
    out["A_feasibility"]["measured_usd_total"] = round(float(cdf.usd.sum()), 2)
    out["A_feasibility"]["measured_usd_per_generation_incl_judging"] = round(float(per_gen), 5)
    out["A_feasibility"]["projected_full_main_audit_usd"] = round(float(per_gen) * 67200 * 1.15, 0)  # +15% translated judging, MRBench
    tok = pd.DataFrame([{"model": g["model"], "lang": g["student_lang"], "in": g.get("input_tokens", 0),
                         "out": g.get("output_tokens", 0)} for g in ok_g])
    tok.groupby(["model", "lang"])[["in", "out"]].mean().round(0).to_csv(tdir / "producer_tokens.csv")
    jt = pd.DataFrame([{"judge": r["judge"], "in": r.get("input_tokens", 0), "out": r.get("output_tokens", 0)} for r in jrecs])
    jt.groupby("judge")[["in", "out"]].mean().round(0).to_csv(tdir / "judge_tokens.csv")

    # ---- B. measurement quality
    alpha_rows = []
    for lang, g in long.groupby("student_lang"):
        for d in DIMENSIONS:
            piv = g.pivot_table(index="target_id", columns="judge", values=d, aggfunc="first")
            mat = np.vstack([piv[j].map(lambda x, d=d: ordinal_index(d, x if isinstance(x, str) else None)).values
                             for j in piv.columns])
            alpha_rows.append({"language": lang, "dimension": d, "in_PQI": d in PQI_SCORES,
                               "alpha": krippendorff_alpha(mat, ALPHA_LEVEL[d]), "n": len(piv)})
    alpha = pd.DataFrame(alpha_rows)
    alpha.to_csv(tdir / "inter_judge_alpha.csv", index=False)
    pa = alpha[alpha.in_PQI]
    judges = run.judges
    per_judge = []
    for j in judges:
        c0 = agg[agg.cue == "C0"].copy()
        c0["pq"] = c0[f"PQI__{j}"]
        r = gap(c0.dropna(subset=["pq"]), "pq", "condition", "EN", "BN", B=B, seed=run.seed)
        per_judge.append({"judge": j, "EN_minus_BN": r["estimate"], "ci_low": r["ci_low"], "ci_high": r["ci_high"]})
    pj = pd.DataFrame(per_judge)
    pj.to_csv(tdir / "per_judge_language_gap.csv", index=False)
    out["B_measurement"] = {
        "alpha_PQI_components": pa.pivot_table(index="dimension", columns="language", values="alpha").round(3).to_dict(),
        "components_below_0.40": pa[pa.alpha < 0.40][["language", "dimension", "alpha"]].round(3).to_dict("records"),
        "per_judge_EN_minus_BN_PQI": pj.round(4).to_dict("records"),
        "max_judge_divergence": float(pj.EN_minus_BN.max() - pj.EN_minus_BN.min()),
    }

    # ---- C. signal
    rows = []
    c0 = agg[agg.cue == "C0"]
    for m in models + [None]:
        r = gap(c0, "PQI", "condition", "EN", "BN", producer=m, B=B, seed=run.seed)
        rows.append({"family": "language_gap", "comparison": "EN - BN (C0)", "model": m or "ALL", **r})
    pairs = run.design["cues"]["pairs"]
    for lang in LANGS:
        s = agg[agg.condition == lang]
        for pair, spec in list(pairs.items()) + [("schooling_vs_none", {"adv": ["C0"], "dis": ["S-VER"]})]:
            for m in models + [None]:
                r = gap(s, "PQI", "cue", spec["adv"], spec["dis"], producer=m, B=B, seed=run.seed)
                rows.append({"family": "identity_gap", "comparison": f"{pair} ({lang})", "model": m or "ALL", **r})
    for pair, spec in pairs.items():
        r = diff_in_diff(agg, "PQI", "cue", spec["adv"], spec["dis"], "condition", "BN", "EN", B=B, seed=run.seed)
        rows.append({"family": "interaction", "comparison": f"IG({pair}) BN - EN", "model": "ALL", **r})
    sig = add_holm(pd.DataFrame(rows))
    sig.to_csv(tdir / "gaps.csv", index=False)
    means = agg.groupby(["model", "condition", "cue"])["PQI"].mean().unstack(["condition", "cue"]).round(3)
    means.to_csv(tdir / "pqi_means.csv")
    # expectation indicators by language and cue (pooled)
    ind = agg.assign(inflated=(agg.Praise_Calibration == "Inflated").astype(float),
                     lowered=agg.Expectation_Level.isin(["Simplified below", "Condescending"]).astype(float),
                     revealed=(agg.Revealing_of_the_Answer != "No").astype(float),
                     lang_match=(agg.Language_Match == "Yes").astype(float),
                     math_ok=(agg.Math_Correctness == "Yes").astype(float))
    ind.groupby(["condition", "cue"])[["PQI", "inflated", "lowered", "revealed", "math_ok", "lang_match", "word_count"]] \
        .mean().round(3).to_csv(tdir / "indicators_by_language_cue.csv")
    ind.groupby(["model", "condition"])[["PQI", "revealed", "math_ok", "lang_match", "inflated", "lowered"]] \
        .mean().round(3).to_csv(tdir / "indicators_by_model_language.csv")
    # power: minimum detectable pooled language gap at the full design (300 items)
    c0d = c0.groupby(["item_id", "model", "condition"])["PQI"].mean().unstack("condition").dropna()
    item_sd = float((c0d["EN"] - c0d["BN"]).groupby(level="item_id").mean().std(ddof=1))
    k = 4 * 7  # Holm family for language gaps: 4 languages x (6 producers + pooled)
    z_a = float(abs(norm.ppf(0.05 / k / 2)))  # Bonferroni bound: Holm is never stricter
    z_b = 0.8416
    mde = (z_a + z_b) * item_sd / math.sqrt(300)
    mde_prod = (z_a + z_b) * float((c0d["EN"] - c0d["BN"]).std(ddof=1)) / math.sqrt(300)
    lg_all = sig[(sig.family == "language_gap") & (sig.model == "ALL")].iloc[0]
    any_gap = sig[(sig.model == "ALL") & (sig.family.isin(["language_gap", "identity_gap"]))
                  & (sig.estimate.abs() >= 0.03) & ((sig.ci_low > 0) | (sig.ci_high < 0))]
    out["C_signal"] = {
        "language_gap_pooled": {k2: round(float(lg_all[k2]), 4) for k2 in ("estimate", "ci_low", "ci_high", "p_value")},
        "item_level_sd_of_pooled_gap": round(item_sd, 4),
        "mde_pooled_full_design_worst_case_holm": round(mde, 4),
        "mde_per_producer_full_design_worst_case_holm": round(mde_prod, 4),
        "gaps_ge_0.03_with_ci_excluding_0": any_gap[["family", "comparison", "estimate", "ci_low", "ci_high",
                                                     "p_holm"]].round(4).to_dict("records"),
    }
    # ---- decision against the pre-registered criteria
    A = out["A_feasibility"]
    # A3 concerns the cap; degenerate repetition loops are a model failure, reported separately
    a_ok = (A["generation_success"] >= 0.98 and A["judge_fully_valid"] >= 0.97
            and (A["max_truncation_cell_excluding_degenerate_loops"] or 0) <= 0.05)
    b_ok = len(out["B_measurement"]["components_below_0.40"]) <= 2 and out["B_measurement"]["max_judge_divergence"] <= 0.10
    c7 = mde <= 0.03
    c8 = len(any_gap) > 0
    out["decision"] = {"A_feasibility_met": a_ok, "B_measurement_met": b_ok, "C7_power_met": c7,
                       "C8_signal_met": c8,
                       "verdict": "GO" if (a_ok and b_ok and c7) else
                       ("GO with changes" if c7 or c8 else "STOP / redesign")}
    run.path("results", "prepilot_report.json").write_text(json.dumps(out, indent=2, default=float))
    return out


def run_stage(base: Run, phase: str | None = None) -> dict:
    run = prepilot_run(base)
    if phase in (None, "build"):
        prompts = build(run)
        log.info("prepilot: %d prompts x %d producers", len(prompts), len(run.producers))
    if phase in (None, "generate"):
        s04_generate.run_stage(run)
    if phase in (None, "judge"):
        s05_mask_and_judge.run_stage(run)
        s08_auto_metrics.run_stage(run)
    if phase in (None, "analyse"):
        res = analyse(run)
        log.info("prepilot decision: %s", res["decision"])
        return res
    return {"phase": phase}
