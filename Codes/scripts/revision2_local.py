"""Second-revision local reanalyses (no API calls; docs/revision_plan_frozen.md, second addendum).

    PYTHONPATH=src python scripts/revision2_local.py
Writes results/tables/revision2_*.csv.
"""
import json
import re

import numpy as np
import pandas as pd

from tlt.analysis.stats import add_holm, bootstrap_mean, gap
from tlt.config import CONFIG_DIR, load_run, load_yaml
from tlt.io import read_jsonl, read_table
from tlt.rubric import PQI_RELIABLE, PQI_SCORES

run = load_run("full")
R = run.path("results")
T = R / "tables"
B = int(run.design["bootstrap"]["resamples"])
SEED = run.seed
MODELS = sorted(run.producers)
items = {it["item_id"]: it for it in read_jsonl(run.path("data", "items_multilingual.jsonl"))}
agg = read_table(R / "judgements_aggregated.csv", low_memory=False)
rep = read_table(R / "repair_judgements_aggregated.csv", low_memory=False)
rep = rep[rep.variant == "original"]


def comps(df):
    df = df.copy()
    df["inflated"] = (df.Praise_Calibration == "Inflated").astype(float).where(df.Praise_Calibration.notna())
    df["PQI_R_noReveal"] = df[[f"pqi_{d}" for d in PQI_RELIABLE if d != "Revealing_of_the_Answer"]].mean(
        axis=1, skipna=False)
    return df


# ---------------------------------------------------------------- 1. Block X per dimension (Q2)
x = read_table(R / "revision_e4_judgements_aggregated.csv", low_memory=False)
citems = set(x.item_id)
c0 = agg[(agg.block == "C") & (agg.variant == "original") & (agg.sample_index.astype(str) == "0")
         & agg.item_id.isin(citems)]
X = comps(pd.concat([x, c0], ignore_index=True))
DIMS = ["PQI_R"] + [f"pqi_{d}" for d in PQI_SCORES] + ["inflated"]
rows = []
for l in ("HI", "BN"):
    cells = {"ee": "EN>EN", "el": f"EN>{l}", "le": f"{l}>EN", "ll": f"{l}>{l}"}
    for v in DIMS:
        w = X[X.condition.isin(cells.values())].groupby(["item_id", "model", "condition"])[v].mean().unstack("condition")
        w = w.dropna(subset=list(cells.values()))
        ee, el, le, ll = (w[cells[k]] for k in ("ee", "el", "le", "ll"))
        sgn = -1 if v == "inflated" else 1  # positive = worse for the student writing in l
        for name, s in (("output", ((ee - el) + (le - ll)) / 2), ("input", ((ee - le) + (el - ll)) / 2),
                        ("interaction", (ee - el) - (le - ll))):
            s = (sgn * s).reset_index()
            cnt = s.groupby("item_id").model.nunique()
            per = s[s.item_id.isin(cnt[cnt == len(MODELS)].index)].groupby("item_id")[0].mean()
            rows.append({"family": f"x_dims_{l}_{name}", "language": l, "effect": name, "dimension": v,
                         **bootstrap_mean(per, B, SEED)})
t1 = add_holm(pd.DataFrame(rows))
t1.to_csv(T / "revision2_crossing_by_dimension.csv", index=False)
print("[1] Block X effects by dimension (positive = worse in l)")
print(t1.pivot_table(index="dimension", columns=["language", "effect"], values="estimate").round(3).to_string())

# ---------------------------------------------------------------- 2. canonical-message loss over all R5 records (Q4)
cf = read_table(T / "revision_r5_canonical_fidelity.csv")  # per-variety summary from the first revision
MAN = {("eedi-741-D", "BN"): "answer lost", ("eedi-741-D", "HI"): "answer mistranslated",
       ("eedi-1487-B", "BN"): "garbled", ("eedi-1487-B", "HI"): "garbled",
       ("eedi-1015-A", "HI"): "meaning shifted", ("eedi-1303-A", "HI"): "meaning shifted"}
keep = set(rep.target_id)
r5 = []
for line in open(run.path("runs", "repair_generations.jsonl"), encoding="utf-8"):
    r = json.loads(line)
    if r["call_id"] in keep and r.get("arm") == "R5":
        r5.append(r)
rows = []
for r in r5:
    k = (r["item_id"], r["repair_lang"])
    # (a string test of whether the plan repeats the answer was dropped: one-word answers such as
    # "term" also match the question text, so it does not measure plan quality)
    rows.append({"target_id": r["call_id"], "item_id": k[0], "lang": k[1], "cue": r["cue"], "model": r["model"],
                 "canon_class": MAN.get(k, "kept")})
d = pd.DataFrame(rows)
s2 = d.groupby(["lang", "canon_class"]).agg(r5_outputs=("target_id", "size")).reset_index()
s2.to_csv(T / "revision2_canonical_loss_records.csv", index=False)
print("\n[2] R5 outputs by canonical-message class (all cues)")
print(s2.round(3).to_string(index=False))
# downstream: R5 minus R2-T on the same item x language x tutor, affected vs other HI/BN items (C0)
sc = rep[(rep.cue == "C0") & rep.condition.isin(["HI", "BN"]) & rep.arm.isin(["R5", "R2T", "R2L"])]
w = sc.pivot_table(index=["item_id", "condition", "model"], columns="arm", values="PQI_R").dropna().reset_index()
w["affected"] = [MAN.get((i, l), "kept") in ("answer lost", "answer mistranslated", "garbled")
                 for i, l in zip(w.item_id, w.condition)]
rows = []
for base in ("R2T", "R2L"):
    for aff, g in w.groupby("affected"):
        per = (g["R5"] - g[base]).groupby([g.item_id, g.condition]).mean()
        row = {"contrast": f"R5 - {base}", "canonical_answer": "lost or garbled" if aff else "kept",
               "item_language_pairs": len(per), "replies_each_arm": len(g), "mean_R5": float(g["R5"].mean()),
               "estimate": float(per.mean()), "min_pair": float(per.min()), "max_pair": float(per.max())}
        if len(per) >= 20:  # an interval only where there are enough item-language pairs
            bm = bootstrap_mean(per, B, SEED)
            row.update({"ci_low": bm["ci_low"], "ci_high": bm["ci_high"], "p_value": bm["p_value"]})
        rows.append(row)
t2 = pd.DataFrame(rows)
t2["family"] = "canonical_loss_kept"  # Holm across the two contrasts with enough pairs for a test
t2 = add_holm(t2) if "p_value" in t2 else t2
t2.to_csv(T / "revision2_canonical_loss_downstream.csv", index=False)
print(t2[["contrast", "canonical_answer", "item_language_pairs", "mean_R5", "estimate", "min_pair", "max_pair",
          "ci_low", "ci_high"]]
      .round(3).to_string(index=False))

# ---------------------------------------------------------------- 3. script-based language identification (Q5)
SCRIPT = {"Latin": re.compile(r"[A-Za-z]"), "Devanagari": re.compile(r"[ऀ-ॿ]"),
          "Bengali": re.compile(r"[ঀ-৿]")}
EXPECT = {"EN": "Latin", "HI": "Devanagari", "BN": "Bengali", "HI-EN": "Latin", "BN-EN": "Latin"}


def script_of(text):
    t = re.sub(r"\\\(.*?\\\)|\\\[.*?\\\]|\$[^$]*\$", " ", text or "", flags=re.S)
    c = {k: len(p.findall(t)) for k, p in SCRIPT.items()}
    tot = sum(c.values())
    if not tot:
        return "none", 0.0
    k = max(c, key=c.get)
    return k, c[k] / tot


texts = {}
for line in open(run.path("runs", "repair_generations.jsonl"), encoding="utf-8"):
    r = json.loads(line)
    if r["call_id"] in keep:
        texts[r["call_id"]] = r.get("response_text") or ""
e8p = run.path("runs", "revision_e8_nllb_generations.jsonl")
if e8p.exists():
    for line in open(e8p, encoding="utf-8"):
        r = json.loads(line)
        texts[r["call_id"]] = r.get("response_text") or ""
gl = rep[rep.cue == "C0"].copy()
try:
    e8 = read_table(R / "revision_e8_judgements_aggregated.csv", low_memory=False)
    gl = pd.concat([gl, e8], ignore_index=True)
except FileNotFoundError:
    pass
res = [script_of(texts.get(t, "")) for t in gl.target_id]
gl["script"] = [a for a, _ in res]
gl["script_share"] = [b for _, b in res]
gl["script_ok"] = [s == EXPECT[c] for s, c in zip(gl.script, gl.condition)]
gl["judge_match"] = gl.Language_Match == "Yes"
t3 = gl.groupby(["arm", "condition"]).agg(replies=("target_id", "size"), script_ok=("script_ok", "mean"),
                                          judge_match=("judge_match", "mean"),
                                          agree=("judge_match", lambda s: float((s == gl.loc[s.index, "script_ok"]).mean())),
                                          judge_no_but_script_ok=("judge_match", lambda s: int(((~s) & gl.loc[s.index, "script_ok"]).sum())),
                                          judge_yes_but_script_wrong=("judge_match", lambda s: int((s & ~gl.loc[s.index, "script_ok"]).sum()))
                                          ).reset_index()
t3.to_csv(T / "revision2_script_lid.csv", index=False)
print("\n[3] script-based language identification vs judge Language_Match (HI/BN rows)")
print(t3[t3.condition.isin(["HI", "BN"])].round(3).to_string(index=False))

# ---------------------------------------------------------------- 4. components without answer withholding (Q6)
A = agg[(agg.block == "A") & (agg.cue == "C0") & (agg.variant == "original") & agg.item_id.isin(set(rep.item_id))
        & agg.condition.isin(["EN", "HI", "BN"])].assign(arm="R0")
D = comps(pd.concat([A, rep[(rep.cue == "C0") & rep.condition.isin(["EN", "HI", "BN"])]], ignore_index=True))
D["arm_lang"] = D.arm + "|" + D.condition
CC = [f"pqi_{d}" for d in PQI_RELIABLE] + ["PQI_R_noReveal", "PQI_R", "inflated"]
rows = []
for arm in ("R0", "R1", "R2", "R2L", "R2T", "R3", "R4", "R5"):
    for v in CC:
        row = {"arm": arm, "component": v,
               "HI_BN": float(D[D.arm_lang.isin([f"{arm}|HI", f"{arm}|BN"])][v].mean()),
               "EN": float(D[D.arm_lang == f"{arm}|EN"][v].mean())}
        if arm != "R0":
            r = gap(D, v, "arm_lang", [f"{arm}|HI", f"{arm}|BN"], ["R0|HI", "R0|BN"], B=B, seed=SEED)
            row.update({"vs_R0": r["estimate"], "vs_R0_ci_low": r["ci_low"], "vs_R0_ci_high": r["ci_high"],
                        "p_value": r["p_value"], "family": f"comp_{v}"})
        rows.append(row)
t4 = pd.DataFrame(rows)
t4["family"] = t4["family"].fillna("reference")
t4 = add_holm(t4)
t4.loc[t4.family == "reference", "p_holm"] = np.nan
t4.to_csv(T / "revision2_components_by_arm.csv", index=False)
print("\n[4] components on HI+BN by arm")
print(t4.pivot_table(index="component", columns="arm", values="HI_BN").round(3).to_string())
rows = []
for arm in ("R2", "R2L", "R2T", "R5"):
    for m in MODELS:
        s = D[(D.model == m) & D.arm_lang.isin([f"{arm}|HI", f"{arm}|BN"])]
        z = D[(D.model == m) & D.arm_lang.isin(["R0|HI", "R0|BN"])]
        rows.append({"arm": arm, "model": m, "PQI_R_noReveal": float(s.PQI_R_noReveal.mean()),
                     "R0_PQI_R_noReveal": float(z.PQI_R_noReveal.mean()),
                     "mistake_location": float(s.pqi_Mistake_Location.mean()),
                     "R0_mistake_location": float(z.pqi_Mistake_Location.mean())})
pd.DataFrame(rows).to_csv(T / "revision2_components_by_tutor.csv", index=False)

# ---------------------------------------------------------------- 5. cost under price changes
cost = read_table(T / "revision_e1_cost.csv")
PR = load_yaml(CONFIG_DIR / "prices.yaml")
gpu_per_1000 = 0.27 * 0.2689 / 1200 * 1000  # E1: RTX 4080 0.27 h at $0.269/h for 1,200 back-translations
rows = []
for f_in in (0.5, 1, 2):
    for f_out in (0.5, 1, 2):
        for _, r in cost.iterrows():
            # tutor-call cost scales linearly with the price of each token type; the split uses the
            # arm's own token counts and the mean tutor price
            pin = np.mean([PR["models"][m].get("input", PR["models"][m].get("input_inr", 0) / PR["inr_per_usd"])
                           for m in MODELS])
            pout = np.mean([PR["models"][m].get("output", PR["models"][m].get("output_inr", 0) / PR["inr_per_usd"])
                            for m in MODELS])
            base_in = r.input_tok_per_reply if "input_tok_per_reply" in r else r.input_tokens_per_reply
            base_out = r.output_tokens_per_reply
            usd = (base_in * pin * f_in + base_out * pout * f_out) / 1e6 * 1000 * r.llm_calls_per_reply / max(
                r.llm_calls_per_reply, 1)
            gpu = gpu_per_1000 * r.indictrans2_calls_per_reply / 1.333 if r.indictrans2_calls_per_reply else 0.0
            rows.append({"input_price_x": f_in, "output_price_x": f_out, "arm": r.arm,
                         "usd_per_1000_tokens_only": usd, "gpu_usd_per_1000": gpu, "total": usd + gpu})
t5 = pd.DataFrame(rows)
t5.to_csv(T / "revision2_cost_sensitivity.csv", index=False)
rank = t5[t5.arm.isin(["R2", "R2L", "R2T", "R4", "R5"])].groupby(["input_price_x", "output_price_x"]).apply(
    lambda g: " < ".join(g.sort_values("total").arm)).reset_index(name="cheapest_to_dearest")
print("\n[5] cost ranking under price changes (mean tutor price; GPU translation included)")
print(rank.to_string(index=False))
print("GPU translation per 1,000 R2-T replies (EN/HI/BN mix): $%.3f" % (gpu_per_1000 * 1.0))
