"""Estimate API cost per provider and model from the design, prices and token profile.

    python -m tlt estimate_cost            # full design
    python -m tlt estimate_cost --pilot    # pilot design (and measured tokens if the pilot ran)

Call volumes are derived from config/design.yaml exactly as the stages generate them.
Prices come from config/prices.yaml (dated, with sources). Token counts per call come from
the token profile there, or, when run logs exist for this mode, from their measured means.
Three scenarios are reported: as configured; Claude judge with thinking disabled; and that
plus judge prompt caching of the static instruction prefix. Output:
results/cost_estimate.csv and a printed summary.
"""
from __future__ import annotations

import logging
from collections import defaultdict

import pandas as pd

from ..config import CONFIG_DIR, Run, load_yaml
from ..io import AppendLog

log = logging.getLogger("tlt.cost")


def volumes(run: Run) -> dict[str, dict[str, float]]:
    """Number of calls per (model, call type)."""
    d = run.design
    it, bl = d["items"], d["blocks"]
    nA, nB, nC, nD = it["n_core"], it["n_block_b"], it["n_block_c"], it["n_block_d"]
    sA, sB, sC, sD = (bl[k]["samples"] for k in "ABCD")
    L, Li, cues = len(d["languages"]["core"]), len(d["languages"]["identity"]), len(d["cues"]["block_b"])
    nDc, nCt = len(bl["D"]["conditions"]), len(d["languages"]["decomposition_targets"])
    P, J = len(run.producers), len(run.judges)
    rp = d["repair"]
    r_langs = len(rp["languages"]) + (1 if rp.get("include_english_reference") else 0)
    gen = nA * L * sA + nB * Li * cues * sB + nC * nCt * sC + nD * nDc * sD           # per producer
    rep_c0 = nC * r_langs * len(rp["arms"])                                             # per producer
    rep_id = nC * len(rp["identity_cues"]) * sum(a != "R0" for a in rp["identity_arms"])
    r5 = (nC * r_langs if "R5" in rp["arms"] else 0) + rep_id                           # realise calls
    non_en = nA * (L - 1) * sA + nB * (Li - 1) * cues * sB + nC * nCt * sC + nD * (nDc - 1) * sD
    codemix_A = nA * 2 * sA
    tj = d["translated_judging"]["share"]
    v: dict[str, dict[str, float]] = defaultdict(dict)
    for m in run.producers:
        v[m]["reply"] = gen + rep_c0 + rep_id - r5
        v[m]["realise"] = r5
        v[m]["plan"] = r5 * 1.05                       # + ~5% schema retries (D17)
    judged = (gen + rep_c0 + rep_id) * P + tj * non_en * P + d["mrbench"]["n_responses"]
    for j in run.judges:
        v[j]["judge"] = judged * 1.01                 # + ~1% unparseable retries (D19)
    student = nC * L * P + nC * r_langs * len(rp["arms"]) * P + nC * r_langs
    v["student"]["answer"] = student
    v["utility"]["calls"] = (2 * nA * 2                # code-mix + back-translation, HI-EN and BN-EN
                             + tj * codemix_A * P      # translated judging of code-mixed replies
                             + nC * 2 * 2              # R4 + R5 canonical English (shared per item)
                             + nC * 2 * P              # R4 code-mixed back-translation of replies
                             + 0.03 * judged * J       # JSON syntax repair (~3%)
                             + nC * (cues + 1) * P     # B2 sentence labels (English, sample 0)
                             + 0.3 * student)          # B5 answer-equivalence fallback (~30%)
    return v


def measured_tokens(run: Run) -> dict[tuple[str, str], tuple[float, float]]:
    """Mean (input, output) tokens per (model, role) from this mode's logs, if any."""
    out = {}
    rows = []
    for name in ("generations", "judgements", "repair_generations", "repair_judgements", "utility_calls",
                 "b5_student", "repair_steps"):
        for r in AppendLog(run.path("runs", f"{name}.jsonl")).records():
            if r.get("status") in ("ok", "unparseable") and r.get("input_tokens"):
                rows.append({"model": r.get("judge") or r.get("model"), "role": r.get("role"),
                             "in": r["input_tokens"], "out": r.get("output_tokens") or 0})
    if rows:
        df = pd.DataFrame(rows)
        for (m, role), g in df.groupby(["model", "role"]):
            out[(m, role)] = (float(g["in"].mean()), float(g["out"].mean()))
    return out


def run_stage(run: Run) -> dict:
    cfg = load_yaml(CONFIG_DIR / "prices.yaml")
    prices, tp, fx = cfg["models"], cfg["token_profile"], float(cfg["inr_per_usd"])
    meas = measured_tokens(run)

    def price(m, kind):
        p = prices[m]
        if f"{kind}_inr" in p:
            return float(p[f"{kind}_inr"]) / fx
        return float(p.get(kind, p["input"] if kind == "cache_read" else 0.0))

    def toks(m, call):
        if call == "judge":
            base = tp["judge"][m]
            key = (m, "judge")
        elif call in ("reply", "plan", "realise"):
            base = tp[f"producer_{call}"]
            key = (m, {"reply": "producer", "plan": "repair_plan", "realise": "producer"}[call])
        else:
            base = tp["utility" if m == "utility" else "student"]
            key = (m, "student" if m == "student" else None)
        i, o = base["input"], base["output"]
        if call == "reply":
            o += tp.get("producer_reasoning_extra", {}).get(m, 0)
        if key in meas:
            i, o = meas[key]
        return i, o, key in meas

    rows = []
    for scenario in ("as_configured", "claude_thinking_off", "thinking_off_plus_judge_caching"):
        for m, calls in volumes(run).items():
            for call, n in calls.items():
                i, o, measured = toks(m, call)
                if scenario != "as_configured" and m == "claude_sonnet5" and call == "judge" and not measured:
                    o = 220  # JSON verdict without thinking
                cost_in = n * i * price(m, "input") / 1e6
                if scenario == "thinking_off_plus_judge_caching" and call == "judge":
                    pre = min(tp["judge_cacheable_prefix"], i)
                    cr = price(m, "cache_read")
                    cost_in = n * ((i - pre) * price(m, "input") + pre * cr) / 1e6
                    if m == "claude_sonnet5":  # cache writes, assuming a hit rate of about 97% at our call rate
                        cost_in += 0.03 * n * pre * (float(prices[m].get("cache_write", 0)) - cr) / 1e6
                cost_out = n * o * price(m, "output") / 1e6
                rows.append({"scenario": scenario, "provider": prices[m]["provider"], "model": m, "call": call,
                             "calls": round(n), "input_tokens_per_call": round(i), "output_tokens_per_call": round(o),
                             "tokens_measured": measured, "input_usd": round(cost_in, 2),
                             "output_usd": round(cost_out, 2), "total_usd": round(cost_in + cost_out, 2)})
    df = pd.DataFrame(rows)
    df.to_csv(run.path("results", "cost_estimate.csv"), index=False)
    summary = df.groupby(["scenario", "provider"])["total_usd"].sum().unstack("scenario").round(2)
    summary.loc["TOTAL"] = summary.sum()
    log.info("cost estimate (%s, USD):\n%s", run.mode, summary.to_string())
    return {"by_provider": summary.to_dict(), "measured_tokens_used": bool(meas)}
