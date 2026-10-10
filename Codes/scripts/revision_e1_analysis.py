"""E1 analysis exactly as frozen in docs/revision_plan_frozen.md (written before the data).

Primary: paired PQI-R, R5 - R2-L and R5 - R2-T in each of EN, HI, BN (6 tests, one Holm
family). Equivalence only via TOST with margin +/-0.02 (alpha 0.05 per contrast, 90%
interval, Holm across the 6). PQI reported for continuity; pooled-over-languages estimates are
descriptive. Secondary: within-arm gap EN - l, language match, answer withheld, mathematical
correctness, PQI components, calls/tokens/latency per final reply. Wrong-language replies stay
in. Drift: contemporaneous R2/R5 re-runs on 20 seeded items, new minus saved.

    PYTHONPATH=src python scripts/revision_e1_analysis.py
"""
import json
import math

import numpy as np
import pandas as pd

from tlt.analysis.stats import add_holm, gap, paired_diff
from tlt.config import CONFIG_DIR, load_run, load_yaml
from tlt.io import read_table
from tlt.rubric import PQI_SCORES

run = load_run("full")
R = run.path("results")
T = R / "tables"
B = int(run.design["bootstrap"]["resamples"])
seed = run.seed
L3 = ["EN", "HI", "BN"]
MODELS = sorted(run.producers)
DELTA = 0.02

rep = read_table(R / "repair_judgements_aggregated.csv", low_memory=False)
rep = rep[(rep.variant == "original") & (rep.cue == "C0") & rep.condition.isin(L3)].copy()
rep["withheld"] = (rep.Revealing_of_the_Answer == "No").astype(float).where(rep.Revealing_of_the_Answer.notna())
rep["lang_match"] = (rep.Language_Match == "Yes").astype(float).where(rep.Language_Match.notna())
rep["math_ok"] = (rep.Math_Correctness == "Yes").astype(float).where(rep.Math_Correctness.notna())
rep["arm_lang"] = rep.arm + "|" + rep.condition
agg = read_table(R / "judgements_aggregated.csv", low_memory=False)
R0 = agg[(agg.variant == "original") & (agg.block == "A") & (agg.cue == "C0") & agg.condition.isin(L3)
         & agg.item_id.isin(set(rep.item_id))].copy()
R0["arm"] = "R0"
for c, v in (("withheld", "Revealing_of_the_Answer"), ("lang_match", "Language_Match"), ("math_ok", "Math_Correctness")):
    yes = "No" if c == "withheld" else "Yes"
    R0[c] = (R0[v] == yes).astype(float).where(R0[v].notna())
R0["arm_lang"] = R0.arm + "|" + R0.condition
df = pd.concat([R0, rep], ignore_index=True)
counts = rep.groupby(["arm", "condition"]).size().unstack()
print("judged replies per arm x language:\n", counts.to_string())
assert {"R2L", "R2T"} <= set(rep.arm), "E1 arms not yet judged"


def g(v, a, b, m=None, data=df):
    return gap(data, v, "arm_lang", a, b, producer=m, B=B, seed=seed)


def item_diffs(v, a, b, data=df):
    pdf = paired_diff(data, v, "arm_lang", a, b, ["item_id", "model"])
    cnt = pdf.groupby("item_id").model.nunique()
    return pdf[pdf.item_id.isin(cnt[cnt == data.model.nunique()].index)].groupby("item_id")["diff"].mean().values


def tost(x: np.ndarray, delta: float) -> dict:
    """Two one-sided tests of |mean| < delta; each one-sided p = max(bootstrap-t, sign-flip)."""
    x = x[~np.isnan(x)]
    n = len(x)
    xb = x[np.random.default_rng(seed).integers(0, n, size=(B, n))]
    est, se = x.mean(), x.std(ddof=1) / math.sqrt(n)
    tb = (xb.mean(1) - est) / (xb.std(1, ddof=1) / math.sqrt(n))
    out = {}
    for side, shift, sgn in (("lower", delta, 1), ("upper", -delta, -1)):
        y = x + shift
        t_obs = y.mean() / se
        p_bt = (np.sum(sgn * tb >= sgn * t_obs) + 1) / (B + 1)
        ys = np.random.default_rng(seed + 1).choice([-1.0, 1.0], size=(B, n)) * y
        ts = ys.mean(1) / (ys.std(1, ddof=1) / math.sqrt(n))
        out[f"p_{side}"] = float(max(p_bt, (np.sum(sgn * ts >= sgn * t_obs) + 1) / (B + 1)))
    q = np.quantile(tb, [0.05, 0.95])
    return {**out, "tost_estimate": float(est), "ci90_low": float(est - q[1] * se),
            "ci90_high": float(est - q[0] * se), "p_tost": max(out["p_lower"], out["p_upper"])}


# ---------------------------------------------------------------- primary + continuity + secondary contrasts
rows = []
for v in ("PQI_R", "PQI", "withheld", "lang_match", "math_ok"):
    for base in ("R2L", "R2T"):
        for l in L3 + ["pooled"]:
            a = [f"R5|{x}" for x in L3] if l == "pooled" else f"R5|{l}"
            b = [f"{base}|{x}" for x in L3] if l == "pooled" else f"{base}|{l}"
            for m in MODELS + [None]:
                r = g(v, a, b, m)
                fam = "descriptive"
                if m is None and l != "pooled":
                    fam = "E1_primary_PQI_R" if v == "PQI_R" else f"E1_secondary_{v}"
                row = {"family": fam, "outcome": v, "contrast": f"R5 - {base}", "language": l, "model": m or "ALL",
                       "mean_R5": float(df[df.arm_lang.isin([a] if isinstance(a, str) else a)
                                           & ((df.model == m) if m else True)][v].mean()),
                       "mean_base": float(df[df.arm_lang.isin([b] if isinstance(b, str) else b)
                                              & ((df.model == m) if m else True)][v].mean()), **r}
                if m is None and l != "pooled" and v in ("PQI_R", "PQI"):
                    row.update(tost(item_diffs(v, a, b), DELTA))
                rows.append(row)
t = add_holm(pd.DataFrame(rows))
t.loc[t.family == "descriptive", "p_holm"] = np.nan
for v in ("PQI_R", "PQI"):  # TOST Holm across the 6 contrasts of each outcome
    mask = (t.outcome == v) & (t.model == "ALL") & (t.language != "pooled")
    t.loc[mask, "p_tost_holm"] = add_holm(t[mask].assign(family="tost"), p_col="p_tost")["p_holm"].values
t.to_csv(T / "revision_e1_contrasts.csv", index=False)
cols = ["outcome", "contrast", "language", "mean_R5", "mean_base", "estimate", "ci_low", "ci_high", "p_value", "p_holm"]
print("\n[E1] contrasts (ALL tutors)")
print(t[t.model == "ALL"][cols].round(4).to_string(index=False))
print("\nTOST +/-0.02")
print(t[(t.model == "ALL") & t.p_tost.notna()][["outcome", "contrast", "language", "tost_estimate", "ci90_low",
                                                "ci90_high", "p_tost", "p_tost_holm"]].round(4).to_string(index=False))

# ---------------------------------------------------------------- arm summaries and within-arm gaps
rows = []
for arm in ("R0", "R2", "R2L", "R2T", "R4", "R5"):
    for v in ("PQI_R", "PQI", "withheld", "lang_match", "math_ok"):
        row = {"arm": arm, "outcome": v}
        for l in L3:
            row[f"mean_{l}"] = float(df[df.arm_lang == f"{arm}|{l}"][v].mean())
        for l in ("HI", "BN"):
            r = g(v, f"{arm}|EN", f"{arm}|{l}")
            row.update({f"gap_EN_{l}": r["estimate"], f"gap_EN_{l}_ci_low": r["ci_low"],
                        f"gap_EN_{l}_ci_high": r["ci_high"], f"gap_EN_{l}_p": r["p_value"]})
        rows.append(row)
ts = pd.DataFrame(rows)
ts.to_csv(T / "revision_e1_arm_summary.csv", index=False)
print("\n[E1] arm summary (EN/HI/BN, 100 repair items, six tutors)")
print(ts[["arm", "outcome", "mean_EN", "mean_HI", "mean_BN", "gap_EN_HI", "gap_EN_BN"]].round(3).to_string(index=False))

rows = []
for d in PQI_SCORES:
    v = f"pqi_{d}"
    for base in ("R2L", "R2T"):
        r = g(v, [f"R5|{x}" for x in L3], [f"{base}|{x}" for x in L3])
        rows.append({"component": d, "contrast": f"R5 - {base}", "language": "pooled", **r})
tcomp = pd.DataFrame(rows)
tcomp["family"] = "e1_component_" + tcomp.contrast  # Holm across the eight components of a contrast
add_holm(tcomp).to_csv(T / "revision_e1_components.csv", index=False)

# pooled HI+BN quality and within-arm gap for every arm (Fig. repair), PQI-R and PQI
from tlt.analysis.stats import bootstrap_mean  # noqa: E402

rows = []
HB = ["HI", "BN"]
for arm in ("R0", "R1", "R2", "R2L", "R2T", "R3", "R4", "R5"):
    for v in ("PQI_R", "PQI"):
        cell = df[df.arm_lang.isin([f"{arm}|{x}" for x in HB])]
        per = cell.groupby(["item_id", "model"])[v].mean()
        cnt = per.reset_index().groupby("item_id").model.nunique()
        per_item = per.reset_index()
        per_item = per_item[per_item.item_id.isin(cnt[cnt == len(MODELS)].index)].groupby("item_id")[v].mean()
        a_ = bootstrap_mean(per_item, B, seed)
        gp = g(v, f"{arm}|EN", [f"{arm}|{x}" for x in HB])
        rows.append({"arm": arm, "outcome": v, "quality_HI_BN": a_["estimate"], "q_ci_low": a_["ci_low"],
                     "q_ci_high": a_["ci_high"], "quality_EN": float(df[df.arm_lang == f"{arm}|EN"][v].mean()),
                     "gap": gp["estimate"], "gap_ci_low": gp["ci_low"], "gap_ci_high": gp["ci_high"],
                     "gap_p": gp["p_value"]})
tp = pd.DataFrame(rows)
tp.to_csv(T / "revision_e1_arms_pooled_HI_BN.csv", index=False)
print("\n[E1] all arms, pooled HI+BN")
print(tp.round(3).to_string(index=False))

# residual within-arm gaps by tutor (unadjusted 95% intervals), PQI-R
rows = []
for arm in ("R0", "R2", "R2L", "R2T", "R5"):
    for l in ("HI", "BN"):
        for m in MODELS + [None]:
            r = g("PQI_R", f"{arm}|EN", f"{arm}|{l}", m)
            rows.append({"arm": arm, "language": l, "model": m or "ALL", **r})
tr_ = pd.DataFrame(rows)
tr_["family"] = ("resid_" + tr_.arm + "_" + tr_.language + "_"
                 + tr_.model.where(tr_.model == "ALL", "tutors"))  # Holm across tutors
tr_ = add_holm(tr_)
tr_.to_csv(T / "revision_e1_residuals_by_tutor.csv", index=False)
print("\n[E1] within-arm gap EN - l by tutor (PQI-R)")
print(tr_.pivot_table(index="model", columns=["arm", "language"], values="estimate").round(3).to_string())

# exploratory (not in the frozen list): non-PQI rubric dimensions that bear on fluency,
# relevant because R2-T's HI/BN replies are machine translations
rows = []
FLU = {"Coherence": "Yes", "Humanlikeness": "Yes", "Tutor_Tone": "Encouraging"}
for d, good in FLU.items():
    df[f"flu_{d}"] = (df[d] == good).astype(float).where(df[d].notna())
    for arm in ("R0", "R2", "R2L", "R2T", "R4", "R5"):
        row = {"dimension": f"{d} = {good}", "arm": arm}
        for l in L3:
            row[l] = float(df[df.arm_lang == f"{arm}|{l}"][f"flu_{d}"].mean())
        rows.append(row)
    for base in ("R2L", "R2T"):
        r = g(f"flu_{d}", [f"R5|{x}" for x in ("HI", "BN")], [f"{base}|{x}" for x in ("HI", "BN")])
        rows.append({"dimension": f"{d} = {good}", "arm": f"R5 - {base} (HI+BN)", "estimate": r["estimate"],
                     "ci_low": r["ci_low"], "ci_high": r["ci_high"]})
tf = pd.DataFrame(rows)
tf.to_csv(T / "revision_e1_fluency_exploratory.csv", index=False)
print("\n[E1] exploratory fluency-related dimensions")
print(tf.round(3).to_string(index=False))

# ---------------------------------------------------------------- drift (contemporaneous R2 / R5 re-runs)
rows = []
for arm in ("R2", "R5"):
    dr = df[df.arm == f"{arm}-DRIFT"]
    items = set(dr.item_id)
    sub = df[df.item_id.isin(items)]
    for v in ("PQI_R", "PQI", "withheld", "lang_match"):
        for l in L3 + ["pooled"]:
            a = [f"{arm}-DRIFT|{x}" for x in L3] if l == "pooled" else f"{arm}-DRIFT|{l}"
            b = [f"{arm}|{x}" for x in L3] if l == "pooled" else f"{arm}|{l}"
            rows.append({"arm": arm, "outcome": v, "language": l, "n_items": len(items),
                         **gap(sub, v, "arm_lang", a, b, B=B, seed=seed)})
for v in ("PQI_R", "PQI"):  # does the R5 - R2 contrast itself move over time?
    dr_items = set(df[df.arm == "R5-DRIFT"].item_id)
    sub = df[df.item_id.isin(dr_items)].copy()
    new = gap(sub, v, "arm_lang", [f"R5-DRIFT|{x}" for x in L3], [f"R2-DRIFT|{x}" for x in L3], B=B, seed=seed)
    old = gap(sub, v, "arm_lang", [f"R5|{x}" for x in L3], [f"R2|{x}" for x in L3], B=B, seed=seed)
    rows.append({"arm": "R5 - R2 (re-run)", "outcome": v, "language": "pooled", **new})
    rows.append({"arm": "R5 - R2 (saved)", "outcome": v, "language": "pooled", **old})
tdr = pd.DataFrame(rows)
tdr["family"] = "drift_" + tdr.arm + "_" + tdr.outcome  # Holm within each arm x outcome
tdr = add_holm(tdr)
tdr.to_csv(T / "revision_e1_drift.csv", index=False)
print("\n[E1] drift: re-run minus saved, same 20 items")
print(tdr[["arm", "outcome", "language", "estimate", "ci_low", "ci_high", "p_value"]].round(4).to_string(index=False))

# ---------------------------------------------------------------- cost / latency per final reply
PR = load_yaml(CONFIG_DIR / "prices.yaml")


def usd(model, route, ti, to):
    p = PR["models"].get(model, {})
    if "input_inr" in p:
        return (ti * p["input_inr"] + to * p["output_inr"]) / float(PR["inr_per_usd"]) / 1e6
    if model == "gemini_flash" and "linkapi" not in str(route):
        return (ti * 0.75 + to * 3.75) / 1e6
    return (ti * p.get("input", 0) + to * p.get("output", 0)) / 1e6


ids = set(rep.target_id)
finals = {}
for line in open(run.path("runs", "repair_generations.jsonl"), encoding="utf-8"):
    r = json.loads(line)
    if r["call_id"] in ids and r.get("status") == "ok":
        finals[r["call_id"]] = r
steps = {}
for line in open(run.path("runs", "repair_steps.jsonl"), encoding="utf-8"):
    r = json.loads(line)
    if r.get("status") != "ok" or r["call_id"].startswith("to_en|"):
        continue
    rest = r["call_id"].split("|", 1)[1]
    parent = next((c for c in (rest, rest.rsplit("|", 1)[0]) if c in finals), None)
    if parent:
        steps.setdefault(parent, []).append(r)
rows = []
for cid, f in finals.items():
    calls = [f] + steps.get(cid, [])
    rows.append({"arm": f["arm"], "lang": f["repair_lang"], "llm_calls": len(calls),
                 "input_tokens": sum(int(c.get("input_tokens") or 0) for c in calls),
                 "output_tokens": sum(int(c.get("output_tokens") or 0) for c in calls),
                 "latency_s": sum(float(c.get("latency_s") or 0) for c in calls),
                 "usd": sum(usd(c["model"], c.get("route_used"), int(c.get("input_tokens") or 0),
                                int(c.get("output_tokens") or 0)) for c in calls),
                 "indictrans2_calls": ({"R4": 2, "R2T": 2, "R5": 1, "R5-DRIFT": 1}.get(f["arm"], 0)
                                       if f["repair_lang"] in ("HI", "BN") else 0),
                 "collected": str(f.get("timestamp", ""))[:10]})
c = pd.DataFrame(rows)
tc = c.groupby("arm").agg(final_replies=("llm_calls", "size"), llm_calls_per_reply=("llm_calls", "mean"),
                          input_tokens_per_reply=("input_tokens", "mean"),
                          output_tokens_per_reply=("output_tokens", "mean"),
                          sequential_latency_s_median=("latency_s", "median"),
                          usd_per_1000_replies=("usd", lambda x: 1000 * x.mean()),
                          indictrans2_calls_per_reply=("indictrans2_calls", "mean"),
                          collected=("collected", lambda x: ", ".join(sorted(set(x))))).reset_index()
tc.to_csv(T / "revision_e1_cost.csv", index=False)
print("\n[E1] cost per final reply (EN/HI/BN cells; judging excluded)")
print(tc.round(3).to_string(index=False))
print("\ndone")
