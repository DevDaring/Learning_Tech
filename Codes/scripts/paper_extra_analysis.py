"""Additional analyses for the paper, on the same aggregated judgements and with the same
paired item bootstrap as stage 9 (tlt.analysis.stats.gap). Writes results/tables/extra_*.csv.

1. extra_decomposition_components.csv: the Block C decomposition (generation / input) for
   every PQI component and for the expectation indicators, so it is visible WHICH aspects of
   teaching the generation component consists of.
2. extra_schooling_switch.csv: in English Block B, how often a vernacular-school cue makes
   the tutor switch into Hindi, and the schooling gap with and without switched replies.
3. extra_component_profile.csv: mean of each PQI component by language (Block A).
"""
import pandas as pd

from tlt.analysis.stats import add_holm, gap
from tlt.config import load_run
from tlt.io import read_table

run = load_run("full")
B, seed = int(run.design["bootstrap"]["resamples"]), run.seed
R = run.path("results")
agg = read_table(R / "judgements_aggregated.csv", low_memory=False)
agg = agg[agg.variant == "original"]
auto = read_table(R / "auto_metrics.csv", usecols=["target_id", "deva_share", "beng_share", "latin_share"])
COMP = [c for c in agg.columns if c.startswith("pqi_")]
CD = {"High": 1.0, "Medium": 0.5, "Low": 0.0, "None": 0.0}
agg = agg.assign(inflated=(agg.Praise_Calibration == "Inflated").astype(float).where(agg.Praise_Calibration.notna()),
                 lowered=agg.Expectation_Level.isin(["Simplified below", "Condescending"]).astype(float)
                 .where(agg.Expectation_Level.notna()))
OUTS = ["PQI", "PQI_R"] + COMP + ["inflated", "lowered"]

# ---- 1. decomposition per component
C = agg[agg.block == "C"]
rows = []
for l in ("HI", "BN"):
    items = set(C[C.condition == f"EN>{l}"].item_id)
    A = agg[(agg.block == "A") & (agg.cue == "C0") & agg.item_id.isin(items)]
    d = pd.concat([A[A.condition == "EN"].assign(dc="EN_EN"), C[C.condition == f"EN>{l}"].assign(dc="EN_l"),
                   A[A.condition == l].assign(dc="l_l")])
    for v in OUTS:
        for comp, a, b in (("total", "EN_EN", "l_l"), ("generation", "EN_EN", "EN_l"), ("input", "EN_l", "l_l")):
            r = gap(d, v, "dc", a, b, B=B, seed=seed)
            rows.append({"family": f"decomposition_{comp}", "language": l, "outcome": v, "component": comp, **r})
dec = add_holm(pd.DataFrame(rows))
dec.to_csv(R / "tables" / "extra_decomposition_components.csv", index=False)

# ---- 2. language switch under the vernacular-school cue (English students)
E = agg[(agg.block == "B") & (agg.condition == "EN") & agg.cue.isin(["S-ENG", "S-VER"])].merge(auto, on="target_id")
E["switched"] = (E.deva_share > 0).astype(float)
rows = []
sw = E[E.cue == "S-VER"].groupby("model").switched.mean().to_dict()
sw["ALL"] = float(E[E.cue == "S-VER"].switched.mean())
sw_eng = float(E[E.cue == "S-ENG"].switched.mean())
for name, df in (("all replies", E), ("S-VER replies kept in English", E[(E.cue == "S-ENG") | (E.switched == 0)])):
    for v in ("PQI", "PQI_R", "lowered", "inflated"):
        r = gap(df, v, "cue", "S-ENG", "S-VER", B=B, seed=seed)
        rows.append({"family": "schooling_switch", "subset": name, "outcome": v, **r})
V = E[E.cue == "S-VER"]
desc = {"share_S-VER_switched": sw["ALL"], "share_S-ENG_switched": sw_eng,
        "PQI_S-VER_switched": float(V[V.switched == 1].PQI.mean()),
        "PQI_S-VER_english": float(V[V.switched == 0].PQI.mean()),
        **{f"share_S-VER_switched__{k}": v for k, v in sw.items() if k != "ALL"}}
out = add_holm(pd.DataFrame(rows))
for k, v in desc.items():
    out[k] = v
out.to_csv(R / "tables" / "extra_schooling_switch.csv", index=False)

# ---- 3. component profile by language (Block A, C0)
A = agg[(agg.block == "A") & (agg.cue == "C0")]
prof = A.groupby("condition")[["PQI", "PQI_R"] + COMP + ["inflated", "lowered"]].mean()
prof.to_csv(R / "tables" / "extra_component_profile.csv")
print(prof.round(3).T.to_string())
print({k: round(v, 3) for k, v in desc.items()})
print(out[["subset", "outcome", "estimate", "ci_low", "ci_high", "p_holm"]].round(3).to_string(index=False))
print(dec[dec.outcome.isin(["PQI", "inflated", "lowered"] + COMP)][["language", "outcome", "component", "estimate",
                                                                    "ci_low", "ci_high", "p_holm"]].round(3).to_string(index=False))

# ---- 4. repair arms pooled over the four non-English varieties (same data as table 9)
from tlt.analysis.stats import bootstrap_mean  # noqa: E402

rep = read_table(R / "repair_judgements_aggregated.csv", low_memory=False)
rep = rep[rep.variant == "original"]
items = set(rep.item_id)
R0 = agg[(agg.block == "A") & (agg.cue == "C0") & agg.item_id.isin(items)].copy()
R0["arm"], R0["lang"] = "R0", R0.condition
RR = rep[rep.cue == "C0"].copy()
RR["lang"] = RR.condition
allr = pd.concat([R0, RR], ignore_index=True)
allr["arm_lang"] = allr.arm + "|" + allr.lang
LANGS = ["HI", "BN", "HI-EN", "BN-EN"]
rows = []
for arm in ["R0", "R1", "R2", "R3", "R4", "R5"]:
    nonen = [f"{arm}|{l}" for l in LANGS]
    cell = allr[allr.arm_lang.isin(nonen)]
    per = cell.groupby(["item_id", "model", "arm_lang"]).PQI.mean().groupby(["item_id", "model"]).mean()
    n_prod = cell.model.nunique()
    cnt = per.reset_index().groupby("item_id").model.nunique()
    per_item = per.reset_index()
    per_item = per_item[per_item.item_id.isin(cnt[cnt == n_prod].index)].groupby("item_id").PQI.mean()
    absr = bootstrap_mean(per_item, B, seed)
    g = gap(allr, "PQI", "arm_lang", f"{arm}|EN", nonen, B=B, seed=seed)
    row = {"arm": arm, "abs_pqi_nonEN": absr["estimate"], "abs_ci_low": absr["ci_low"], "abs_ci_high": absr["ci_high"],
           "gap_within_arm": g["estimate"], "gap_ci_low": g["ci_low"], "gap_ci_high": g["ci_high"], "gap_p": g["p_value"],
           "pqi_EN": float(allr[allr.arm_lang == f"{arm}|EN"].PQI.mean()),
           "language_match_yes_nonEN": float((cell.Language_Match == "Yes").mean())}
    if arm != "R0":
        imp = gap(allr, "PQI", "arm_lang", nonen, [f"R0|{l}" for l in LANGS], B=B, seed=seed)
        row.update({"improvement": imp["estimate"], "imp_ci_low": imp["ci_low"], "imp_ci_high": imp["ci_high"],
                    "imp_p": imp["p_value"]})
    rows.append(row)
rp = pd.DataFrame(rows)
rp.to_csv(R / "tables" / "extra_repair_pooled.csv", index=False)
print(rp.round(3).to_string(index=False))

# ---- 5. judges vs teachers pooled over all annotated replies (alpha and exact agreement)
import glob  # noqa: E402

import numpy as np  # noqa: E402

from tlt.analysis.stats import krippendorff_alpha  # noqa: E402
from tlt.rubric import ALPHA_LEVEL, DIMENSIONS, norm_label, ordinal_index  # noqa: E402
from tlt.stages.s09_analysis import _consensus  # noqa: E402

files = sorted(glob.glob(str(run.path("review", "human_returned", "x").parent / "*.csv")))
if files:
    raw = [pd.read_csv(f, dtype=str, keep_default_na=False).set_index("row_id") for f in files]
    key = read_table(run.path("review", "human_annotation_key.csv")).set_index("row_id")
    anchor = read_table(R / "human_anchor_judgements_aggregated.csv").set_index("target_id")
    rows = []
    for d in DIMENSIONS:
        hum = {rid: _consensus(d, [norm_label(d, r[d].get(rid, "")) if str(r[d].get(rid, "")).strip() else None
                                   for r in raw]) for rid in key.index}
        for scope in ["ALL"] + sorted(key.language.unique()):
            rids = [rid for rid in key.index if scope == "ALL" or key.at[rid, "language"] == scope]
            pairs = [(hum[rid], anchor.at[key.at[rid, "target_id"], d]) for rid in rids
                     if hum[rid] is not None and key.at[rid, "target_id"] in anchor.index
                     and isinstance(anchor.at[key.at[rid, "target_id"], d], str)]
            if len(pairs) < 5:
                continue
            h, j = zip(*pairs)
            m = np.vstack([[ordinal_index(d, x) for x in h], [ordinal_index(d, x) for x in j]]).astype(float)
            rows.append({"dimension": d, "scope": scope, "n": len(pairs),
                         "exact_agreement": float(np.mean([a == b for a, b in pairs])),
                         "alpha_judges_vs_teachers": krippendorff_alpha(m, ALPHA_LEVEL[d])})
    hj = pd.DataFrame(rows)
    hj.to_csv(R / "tables" / "extra_judge_teacher_pooled.csv", index=False)
    print(hj[hj.scope == "ALL"].round(3).to_string(index=False))
