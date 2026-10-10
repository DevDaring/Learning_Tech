"""Revision reanalyses that need no new generation (docs/revision_plan_frozen.md, priority 1
and the local parts of E2, E3 and E5). Reads the saved results and logs only; every table is
written to results/tables/revision_*.csv. Original repair arms only: the E1 arms and drift
re-runs are analysed by scripts/revision_e1_analysis.py.

    PYTHONPATH=src python scripts/revision_local_analysis.py

Inference is the study's own: paired item bootstrap within item x tutor, pooled over tutors on
items observed under every tutor, p = max(bootstrap-t, sign-flip), Holm within each family.
Unadjusted intervals are labelled as such.
"""
import json
import math
import re

import numpy as np
import pandas as pd

from tlt.analysis.stats import add_holm, bootstrap_mean, gap, paired_diff
from tlt.config import load_run
from tlt.io import read_jsonl, read_table
from tlt.rubric import DIMENSIONS, PQI_RELIABLE, PQI_SCORES, norm_label
from tlt.stages.s09_analysis import _consensus

run = load_run("full")
R = run.path("results")
T = R / "tables"
B = int(run.design["bootstrap"]["resamples"])
seed = run.seed
JUDGES = list(run.judges)
LANGS = ["HI", "BN", "HI-EN", "BN-EN"]
ARMS = ["R0"] + list(run.design["repair"]["arms"])
MODELS = sorted(run.producers)


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Derived outcomes, plus per-judge PQI-R and leave-one-judge-out PQI / PQI-R (mean of
    the two remaining judges' numeric scores)."""
    df = df.copy()
    df["withheld"] = (df.Revealing_of_the_Answer == "No").astype(float).where(df.Revealing_of_the_Answer.notna())
    df["lang_match"] = (df.Language_Match == "Yes").astype(float).where(df.Language_Match.notna())
    df["math_ok"] = (df.Math_Correctness == "Yes").astype(float).where(df.Math_Correctness.notna())
    for j in JUDGES:
        comps = [df[f"{d}__{j}"].map(PQI_SCORES[d]) for d in PQI_RELIABLE]
        df[f"PQI_R__{j}"] = pd.concat(comps, axis=1).mean(axis=1, skipna=False)
    for j in JUDGES:
        rest = [k for k in JUDGES if k != j]
        df[f"PQI__no_{j}"] = df[[f"PQI__{k}" for k in rest]].mean(axis=1, skipna=False)
        df[f"PQI_R__no_{j}"] = df[[f"PQI_R__{k}" for k in rest]].mean(axis=1, skipna=False)
    # component-dropped composites (sensitivity to the answer-withholding component)
    pq = [f"pqi_{d}" for d in PQI_SCORES if d != "Revealing_of_the_Answer"]
    df["PQI_noReveal"] = df[pq].mean(axis=1, skipna=False)
    pr = [f"pqi_{d}" for d in PQI_RELIABLE if d != "Revealing_of_the_Answer"]
    df["PQI_R_noReveal"] = df[pr].mean(axis=1, skipna=False)
    return df


def g(df, v, a, b, m=None, col="arm_lang"):
    return gap(df, v, col, a, b, producer=m, B=B, seed=seed)


agg = read_table(R / "judgements_aggregated.csv", low_memory=False)
agg = add_scores(agg[agg.variant == "original"])
A = agg[(agg.block == "A") & (agg.cue == "C0")]
rep = read_table(R / "repair_judgements_aggregated.csv", low_memory=False)
rep = add_scores(rep[(rep.variant == "original") & rep.arm.isin(ARMS[1:])])
items_rep = set(rep.item_id)
R0 = A[A.item_id.isin(items_rep)].copy()
R0["arm"] = "R0"
RR = rep[rep.cue == "C0"].copy()
allr = pd.concat([R0, RR], ignore_index=True)
allr["lang"] = allr.condition
allr["arm_lang"] = allr.arm + "|" + allr.lang
NON = lambda arm: [f"{arm}|{l}" for l in LANGS]  # noqa: E731
print("repair rows", len(allr), "items", len(items_rep))

# ---------------------------------------------------------------- 1. R5 vs R2, paired
rows = []
for v in ("PQI", "PQI_R", "withheld", "lang_match", "math_ok"):
    for l in ["EN"] + LANGS + ["nonEN"]:
        a, b = (NON("R5"), NON("R2")) if l == "nonEN" else (f"R5|{l}", f"R2|{l}")
        for m in MODELS + [None]:
            r = g(allr, v, a, b, m)
            rows.append({"family": f"r5_minus_r2_{v}" if (m is None and l != "nonEN") else "descriptive",
                         "outcome": v, "language": l, "model": m or "ALL",
                         "mean_R5": float(allr[allr.arm_lang.isin([a] if isinstance(a, str) else a)
                                               & ((allr.model == m) if m else True)][v].mean()),
                         "mean_R2": float(allr[allr.arm_lang.isin([b] if isinstance(b, str) else b)
                                               & ((allr.model == m) if m else True)][v].mean()), **r})
t1 = add_holm(pd.DataFrame(rows))
t1.loc[t1.family == "descriptive", "p_holm"] = np.nan
t1.to_csv(T / "revision_r5_vs_r2.csv", index=False)
print("\n[1] R5 - R2 (ALL tutors)")
print(t1[t1.model == "ALL"][["outcome", "language", "mean_R5", "mean_R2", "estimate", "ci_low", "ci_high",
                             "p_value", "p_holm"]].round(4).to_string(index=False))

# ---------------------------------------------------------------- 2. residual gaps per language x tutor
rows = []
for v in ("PQI", "PQI_R"):
    for arm in ARMS:
        for l in LANGS + ["nonEN"]:
            b = NON(arm) if l == "nonEN" else f"{arm}|{l}"
            for m in MODELS + [None]:
                r = g(allr, v, f"{arm}|EN", b, m)
                rows.append({"outcome": v, "arm": arm, "language": l, "model": m or "ALL", **r})
t2 = pd.DataFrame(rows)
# Holm across tutors within outcome x arm x language (pooled rows form their own family)
t2["family"] = ("resid_" + t2.outcome + "_" + t2.arm + "_" + t2.language + "_"
                + t2.model.where(t2.model == "ALL", "tutors"))
t2 = add_holm(t2)
t2.to_csv(T / "revision_repair_residual_gaps.csv", index=False)
print("\n[2] within-arm gap EN - l, R5, unadjusted 95% intervals (PQI-R)")
w = t2[(t2.arm == "R5") & (t2.outcome == "PQI_R")]
print(w.pivot_table(index="model", columns="language", values="estimate").round(3).to_string())
big = t2[(t2.arm == "R5") & (t2.model != "ALL") & (t2.language != "nonEN") & (t2.ci_low > 0)]
print("R5 tutor x language residuals with unadjusted interval above zero:")
print(big[["outcome", "model", "language", "estimate", "ci_low", "ci_high"]].round(3).to_string(index=False))

# ---------------------------------------------------------------- 3. per-judge and leave-one-judge-out
VARIANTS = [("panel", "")] + [(f"only_{j}", f"__{j}") for j in JUDGES] + [(f"drop_{j}", f"__no_{j}") for j in JUDGES]
A2 = A.copy()
rows = []
for name, suf in VARIANTS:
    for base in ("PQI", "PQI_R"):
        v = base + suf
        for l in LANGS:
            r = gap(A2, v, "condition", "EN", l, B=B, seed=seed)
            rows.append({"variant": name, "outcome": base, "effect": f"default gap EN-{l}", **r})
        for label, a, b in (("R5 improvement vs R0 (non-EN)", NON("R5"), NON("R0")),
                            ("R2 improvement vs R0 (non-EN)", NON("R2"), NON("R0")),
                            ("R5 - R2 (non-EN)", NON("R5"), NON("R2")),
                            ("R5 within-arm gap (EN - non-EN)", "R5|EN", NON("R5")),
                            ("R2 within-arm gap (EN - non-EN)", "R2|EN", NON("R2")),
                            ("R0 gap on repair items (EN - non-EN)", "R0|EN", NON("R0"))):
            rows.append({"variant": name, "outcome": base, "effect": label, **g(allr, v, a, b)})
t3 = pd.DataFrame(rows)
t3["family"] = "judge_" + t3.variant + "_" + t3.outcome  # Holm within each judge variant and outcome
t3 = add_holm(t3)
t3.to_csv(T / "revision_judge_sensitivity.csv", index=False)
print("\n[3] per-judge / leave-one-out (estimates)")
print(t3.pivot_table(index=["outcome", "effect"], columns="variant", values="estimate").round(3).to_string())

# ---------------------------------------------------------------- 4. components
rows = []
comps = [f"pqi_{d}" for d in PQI_SCORES] + ["PQI_noReveal", "PQI_R_noReveal"]
for v in comps:
    for l in LANGS:
        rows.append({"outcome": v, "effect": f"default gap EN-{l}", **gap(A, v, "condition", "EN", l, B=B, seed=seed)})
    for arm in ("R2", "R4", "R5"):
        rows.append({"outcome": v, "effect": f"{arm} improvement vs R0 (non-EN)", **g(allr, v, NON(arm), NON("R0"))})
        rows.append({"outcome": v, "effect": f"{arm} within-arm gap (EN - non-EN)", **g(allr, v, f"{arm}|EN", NON(arm))})
    rows.append({"outcome": v, "effect": "R5 - R2 (non-EN)", **g(allr, v, NON("R5"), NON("R2"))})
t4 = pd.DataFrame(rows)
t4["family"] = "component_" + t4.effect
t4 = add_holm(t4)
t4.to_csv(T / "revision_component_sensitivity.csv", index=False)
print("\n[4] components (estimates)")
print(t4.pivot_table(index="outcome", columns="effect", values="estimate").round(3).to_string())

# ---------------------------------------------------------------- 5. human anchor detail (E2 local)
key = read_table(run.path("review", "human_annotation_key.csv")).set_index("row_id")
files = sorted(run.path("review", "human_returned", "x").parent.glob("*.csv"))
raw = {f"teacher_{i}": pd.read_csv(f, dtype=str, keep_default_na=False).set_index("row_id")
       for i, f in enumerate(files, start=1)}
anchor = read_table(R / "human_anchor_judgements_aggregated.csv", low_memory=False).set_index("target_id")
rng = np.random.default_rng(seed)
arows, crows = [], []
for d in DIMENSIONS:
    labs = {n: (raw[n][d] if d in raw[n] else pd.Series(dtype=str)).map(
        lambda x, d=d: norm_label(d, x) if str(x).strip() else None) for n in raw}
    cons = pd.Series({rid: _consensus(d, [labs[n].get(rid) for n in raw]) for rid in key.index})
    judge = pd.Series(anchor[d].reindex(key.target_id).values, index=key.index) if d in anchor else None
    for lang in sorted(key.language.unique()) + ["ALL"]:
        ids = key.index if lang == "ALL" else key.index[key.language == lang]
        rated = {n: int(labs[n].reindex(ids).notna().sum()) for n in raw}
        n_rated_any = int(sum(any(pd.notna(labs[n].get(r)) for n in raw) for r in ids))
        h = cons.reindex(ids)
        jl = judge.reindex(ids) if judge is not None else pd.Series(index=ids, dtype=object)
        ok = h.notna() & jl.notna()
        eq = (h[ok] == jl[ok]).astype(float)
        items = key.loc[ok[ok].index, "item_id"]
        # item-cluster bootstrap of exact agreement
        if len(eq):
            per = pd.DataFrame({"eq": eq.values, "item": items.values})
            grp = per.groupby("item").eq.agg(["sum", "count"])
            idx = rng.integers(0, len(grp), size=(2000, len(grp)))
            boots = grp["sum"].values[idx].sum(1) / grp["count"].values[idx].sum(1)
            lo, hi = np.quantile(boots, [0.025, 0.975])
        else:
            lo = hi = math.nan
        row = {"dimension": d, "language": lang, "n_replies": len(ids), "n_rated_by_any": n_rated_any,
               "n_consensus": int(h.notna().sum()), "missing_consensus": int(n_rated_any - h.notna().sum()),
               "n_compared": int(ok.sum()), "exact_agreement": float(eq.mean()) if len(eq) else math.nan,
               "exact_ci_low": lo, "exact_ci_high": hi,
               "consensus_majority_label": h.dropna().value_counts().idxmax() if h.notna().any() else None,
               "consensus_majority_share": float(h.dropna().value_counts(normalize=True).max()) if h.notna().any()
               else math.nan,
               "judge_majority_share_same_replies": float(jl[ok].value_counts(normalize=True).max()) if ok.any()
               else math.nan}
        for n in raw:
            hn = labs[n].reindex(ids)
            okn = hn.notna() & jl.notna()
            row[f"n_{n}"] = rated[n]
            row[f"exact_judge_vs_{n}"] = float((hn[okn] == jl[okn]).mean()) if okn.any() else math.nan
        names = list(raw)
        for i in range(len(names)):
            for k in range(i + 1, len(names)):
                a_, b_ = labs[names[i]].reindex(ids), labs[names[k]].reindex(ids)
                okp = a_.notna() & b_.notna()
                row[f"n_{names[i]}_and_{names[k]}"] = int(okp.sum())
                row[f"exact_{names[i]}_vs_{names[k]}"] = float((a_[okp] == b_[okp]).mean()) if okp.any() else math.nan
        arows.append(row)
        if lang == "ALL":
            for (hl, jlab), c in pd.crosstab(h[ok], jl[ok]).stack().items():
                crows.append({"dimension": d, "teacher_consensus": hl, "judge_panel": jlab, "count": int(c)})
ta = pd.DataFrame(arows)
ta.to_csv(T / "revision_human_agreement_detail.csv", index=False)
pd.DataFrame(crows).to_csv(T / "revision_human_confusion.csv", index=False)
print("\n[5] human anchor (ALL languages)")
print(ta[ta.language == "ALL"][["dimension", "n_rated_by_any", "n_consensus", "missing_consensus", "exact_agreement",
                                "exact_ci_low", "exact_ci_high", "consensus_majority_label",
                                "consensus_majority_share"]].round(3).to_string(index=False))

# ---------------------------------------------------------------- 6. translation coverage (E5 local)
tr = read_table(run.path("review", "translation_review.csv"))
tr["transformed"] = tr.similarity.notna()
tr["flagged"] = tr.flag_reasons.notna()
tr["reviewed"] = tr.status.isin(["approved", "corrected"])
tr["corrected"] = tr.status == "corrected"
rows = []
for lang, s in list(tr.groupby("lang")) + [("ALL", tr)]:
    x = s[s.transformed]
    unf = x[~x.flagged]
    rows.append({"variety": lang, "field_rows": len(s), "unchanged_or_not_gated": int((~s.transformed).sum()),
                 "transformed_gated": len(x), "flagged": int(x.flagged.sum()),
                 "flagged_reviewed": int((x.flagged & x.reviewed).sum()),
                 "unflagged": len(unf), "unflagged_reviewed": int(unf.reviewed.sum()),
                 "unflagged_review_share": float(unf.reviewed.mean()) if len(unf) else math.nan,
                 "reviewed_total": int(x.reviewed.sum()), "corrected_total": int(x.corrected.sum()),
                 "corrected_among_flagged": int((x.flagged & x.corrected).sum()),
                 "corrected_among_unflagged_reviewed": int((unf.corrected).sum()),
                 "reviewed_outside_gate": int((s.reviewed & ~s.transformed).sum())})
t6 = pd.DataFrame(rows)
t6.to_csv(T / "revision_translation_coverage.csv", index=False)
print("\n[6] translation coverage")
print(t6.to_string(index=False))

# reviewed-subset sensitivity: items whose every transformed field in a variety was reviewed
items_all = {it["item_id"]: it for it in read_jsonl(run.path("data", "items_multilingual.jsonl"))}
full = {}
for (iid, lang), s in tr[tr.transformed].groupby(["item_id", "lang"]):
    full[(iid, lang)] = bool(s.reviewed.all())
rows = []
for l in LANGS:
    keep = {iid for (iid, lang), okr in full.items() if lang == l and okr}
    sub = A[A.item_id.isin(keep)]
    for v in ("PQI", "PQI_R"):
        r_sub = gap(sub, v, "condition", "EN", l, B=B, seed=seed)
        r_all = gap(A, v, "condition", "EN", l, B=B, seed=seed)
        rows.append({"variety": l, "outcome": v, "items_fully_reviewed": len(keep),
                     "items_fully_reviewed_in_audit": int(sub.item_id.nunique()),
                     "gap_reviewed_subset": r_sub["estimate"], "ci_low": r_sub["ci_low"], "ci_high": r_sub["ci_high"],
                     "gap_all_items": r_all["estimate"], "all_ci_low": r_all["ci_low"], "all_ci_high": r_all["ci_high"]})
t6b = pd.DataFrame(rows)
t6b.to_csv(T / "revision_translation_reviewed_subset.csv", index=False)
print(t6b.round(4).to_string(index=False))

# integrity: numbers and maths spans preserved between English source and final translation
NUM = re.compile(r"\d+(?:[.,]\d+)?")  # unsigned: Bengali case suffixes attach with "-"
MATH = re.compile(r"\\\(.*?\\\)|\\\[.*?\\\]", re.S)
DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯०१२३४५६७८९", "01234567890123456789")


def nums(s):
    s = str(s or "").translate(DIGITS)
    return sorted(NUM.findall(s.replace(",", "")))


rows = []
final = tr.corrected_text.where(tr.corrected_text.notna(), tr.translation)
tr["final"] = final
for lang, s in list(tr[tr.transformed].groupby("lang")) + [("ALL", tr[tr.transformed])]:
    num_ok = [nums(a) == nums(b) for a, b in zip(s.source_en, s.final)]
    # digit multiset: insensitive to digit grouping and spacing ("3 000" vs "3000")
    dig_ok = [sorted(c for c in str(a).translate(DIGITS) if c.isdigit())
              == sorted(c for c in str(b or "").translate(DIGITS) if c.isdigit()) for a, b in zip(s.source_en, s.final)]
    math_ok = [len(MATH.findall(str(a))) == len(MATH.findall(str(b or ""))) for a, b in zip(s.source_en, s.final)]
    rows.append({"variety": lang, "transformed_fields": len(s), "number_token_mismatch": int(len(s) - sum(num_ok)),
                 "digit_multiset_mismatch": int(len(s) - sum(dig_ok)),
                 "math_span_count_mismatch": int(len(s) - sum(math_ok)),
                 "empty_final": int(s.final.isna().sum())})
t6c = pd.DataFrame(rows)
t6c.to_csv(T / "revision_translation_integrity.csv", index=False)
print(t6c.to_string(index=False))

# ---------------------------------------------------------------- 7. cost and latency per final reply
P = run.models  # noqa: F841  (prices below)
from tlt.config import CONFIG_DIR, load_yaml  # noqa: E402

PR = load_yaml(CONFIG_DIR / "prices.yaml")
inr = float(PR["inr_per_usd"])


def usd(model, route, ti, to):
    p = PR["models"].get(model, {})
    if "input_inr" in p:
        return (ti * p["input_inr"] + to * p["output_inr"]) / inr / 1e6
    if model == "gemini_flash" and "linkapi" not in str(route):
        return (ti * 0.75 + to * 3.75) / 1e6  # Google direct (config/prices.yaml note)
    return (ti * p.get("input", 0) + to * p.get("output", 0)) / 1e6


keep_ids = set(rep.target_id)
finals = {}
for line in open(run.path("runs", "repair_generations.jsonl"), encoding="utf-8"):
    r = json.loads(line)
    if r["call_id"] in keep_ids and r.get("status") == "ok":
        finals[r["call_id"]] = r
steps = {}
shared = []
for line in open(run.path("runs", "repair_steps.jsonl"), encoding="utf-8"):
    r = json.loads(line)
    if r.get("status") != "ok":
        continue
    kind, rest = r["call_id"].split("|", 1)
    if kind == "to_en":
        shared.append(r)
        continue
    parent = next((cid for cid in (rest, rest.rsplit("|", 1)[0]) if cid in finals), None)
    if parent:
        steps.setdefault(parent, []).append(r)
rows = []
for cid, f in finals.items():
    calls = [f] + steps.get(cid, [])
    ti = sum(int(c.get("input_tokens") or 0) for c in calls)
    to = sum(int(c.get("output_tokens") or 0) + 0 for c in calls)
    lang = f["repair_lang"]
    indictrans = {"R4": 2, "R5": 1}.get(f["arm"], 0) if lang in ("HI", "BN") else 0
    rows.append({"arm": f["arm"], "lang": lang, "model": f["model"], "llm_calls": len(calls),
                 "input_tokens": ti, "output_tokens": to,
                 "latency_s": sum(float(c.get("latency_s") or 0) for c in calls),
                 "usd": sum(usd(c["model"], c.get("route_used"), int(c.get("input_tokens") or 0),
                                int(c.get("output_tokens") or 0)) for c in calls),
                 "indictrans2_calls": indictrans, "plan_retried": bool(f.get("plan_retried"))})
cst = pd.DataFrame(rows)
n_cm = len(cst[cst.arm.isin(["R4", "R5"]) & cst.lang.isin(["HI-EN", "BN-EN"])])
sh_usd = sum(usd("utility", s.get("route_used"), int(s.get("input_tokens") or 0), int(s.get("output_tokens") or 0))
             for s in shared)
t7 = cst.groupby("arm").agg(final_replies=("llm_calls", "size"), llm_calls_per_reply=("llm_calls", "mean"),
                            input_tokens_per_reply=("input_tokens", "mean"),
                            output_tokens_per_reply=("output_tokens", "mean"),
                            sequential_latency_s_median=("latency_s", "median"),
                            sequential_latency_s_mean=("latency_s", "mean"),
                            usd_per_1000_replies=("usd", lambda x: 1000 * x.mean()),
                            indictrans2_calls_per_reply=("indictrans2_calls", "mean"),
                            plan_retry_rate=("plan_retried", "mean")).reset_index()
t7["shared_canonicalisation_calls"] = np.where(t7.arm.isin(["R4", "R5"]), len(shared), 0)
t7["note"] = (f"{len(shared)} shared code-mixed-to-English utility calls (${sh_usd:.2f}) serve R4 and R5 for "
              f"{n_cm} code-mixed replies; IndicTrans2 runs locally/GPU and is not billed per call; "
              "judging cost is identical across arms and excluded")
t7.to_csv(T / "revision_cost_latency.csv", index=False)
print("\n[7] cost / latency per final reply")
print(t7.drop(columns="note").round(3).to_string(index=False))

# ---------------------------------------------------------------- 8. TOST (post hoc margin for the original arms)
DELTA = 0.02


def tost(x: np.ndarray, delta: float) -> dict:
    """Two one-sided tests of |mean| < delta, each with p = max(bootstrap-t, sign-flip) on the
    shifted differences (the study's own one-sample procedure, one-sided)."""
    x = x[~np.isnan(x)]
    n = len(x)
    rs = np.random.default_rng(seed)
    xb = x[rs.integers(0, n, size=(B, n))]
    est, se = x.mean(), x.std(ddof=1) / math.sqrt(n)
    tb = (xb.mean(1) - est) / (xb.std(1, ddof=1) / math.sqrt(n))
    out = {}
    for side, shift in (("lower", delta), ("upper", -delta)):
        y = x + shift  # lower: H0 mean <= -delta -> test y > 0; upper: H0 mean >= delta -> test y < 0
        t_obs = y.mean() / se
        sgn = 1 if side == "lower" else -1
        p_bt = (np.sum(sgn * tb >= sgn * t_obs) + 1) / (B + 1)
        signs = np.random.default_rng(seed + 1).choice([-1.0, 1.0], size=(B, n))
        ys = signs * y
        ts = ys.mean(1) / (ys.std(1, ddof=1) / math.sqrt(n))
        p_sf = (np.sum(sgn * ts >= sgn * t_obs) + 1) / (B + 1)
        out[f"p_{side}"] = float(max(p_bt, p_sf))
    q = np.quantile(tb, [0.05, 0.95])
    out.update({"estimate": float(est), "ci90_low": float(est - q[1] * se), "ci90_high": float(est - q[0] * se),
                "p_tost": max(out["p_lower"], out["p_upper"]), "n_items": n})
    return out


def item_diffs(df, v, a, b):
    pdf = paired_diff(df, v, "arm_lang", a, b, ["item_id", "model"])
    cnt = pdf.groupby("item_id").model.nunique()
    return pdf[pdf.item_id.isin(cnt[cnt == df.model.nunique()].index)].groupby("item_id")["diff"].mean().values


rows = []
for v in ("PQI", "PQI_R"):
    for arm in ("R4", "R5"):
        for l in LANGS + ["nonEN"]:
            b = NON(arm) if l == "nonEN" else f"{arm}|{l}"
            r = tost(item_diffs(allr, v, f"{arm}|EN", b), DELTA)
            rows.append({"family": f"tost_{arm}_{v}" if l != "nonEN" else "descriptive", "outcome": v, "arm": arm,
                         "language": l, "margin": DELTA, "margin_status": "post hoc (original arms)", **r})
t8 = add_holm(pd.DataFrame(rows), p_col="p_tost")
t8.loc[t8.family == "descriptive", "p_holm"] = np.nan
t8["equivalent_holm_0.05"] = t8.p_holm < 0.05
t8.to_csv(T / "revision_tost_original_arms.csv", index=False)
print("\n[8] TOST, margin +/-0.02, within-arm gap EN - l")
print(t8[["outcome", "arm", "language", "estimate", "ci90_low", "ci90_high", "p_tost", "p_holm"]].round(4)
      .to_string(index=False))

# ---------------------------------------------------------------- 9. name balance (Block B)
prompts = [p for p in read_jsonl(run.path("data", "prompts.jsonl")) if p["block"] == "B"]
pb = pd.DataFrame(prompts)
rows = []
for (cond, cue), s in pb[pb.first_name.notna() | pb.surname.notna()].groupby(["condition", "cue"]):
    for col in ("first_name", "surname"):
        vc = s[col].value_counts()
        if len(vc):
            rows.append({"condition": cond, "cue": cue, "field": col, "n_prompts": len(s), "distinct": len(vc),
                         "min_count": int(vc.min()), "max_count": int(vc.max()),
                         "counts": "; ".join(f"{k}={v}" for k, v in vc.sort_index().items())})
t9 = pd.DataFrame(rows)
t9.to_csv(T / "revision_name_balance.csv", index=False)
print("\n[9] name balance")
print(t9.drop(columns="counts").to_string(index=False))

# ---------------------------------------------------------------- 10. construct-cluster sensitivity
clusters = {iid: (it.get("construct") or iid) for iid, it in items_all.items()}
mis = {iid: (it.get("misconception_id") or iid) for iid, it in items_all.items()}


def cluster_ci(per_item: pd.Series, cmap: dict) -> tuple:
    df = pd.DataFrame({"d": per_item.values, "c": [cmap[i] for i in per_item.index]})
    grp = df.groupby("c").d.agg(["sum", "count"])
    rs = np.random.default_rng(seed)
    idx = rs.integers(0, len(grp), size=(B, len(grp)))
    boots = grp["sum"].values[idx].sum(1) / grp["count"].values[idx].sum(1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return len(grp), float(lo), float(hi)


def per_item_series(df, v, col, a, b):
    pdf = paired_diff(df, v, col, a, b, ["item_id", "model"])
    cnt = pdf.groupby("item_id").model.nunique()
    return pdf[pdf.item_id.isin(cnt[cnt == df.model.nunique()].index)].groupby("item_id")["diff"].mean()


rows = []
specs = [(f"default gap EN-{l}", A, "condition", "EN", l) for l in LANGS] + [
    ("R5 improvement vs R0 (non-EN)", allr, "arm_lang", NON("R5"), NON("R0")),
    ("R2 improvement vs R0 (non-EN)", allr, "arm_lang", NON("R2"), NON("R0")),
    ("R5 within-arm gap (EN - non-EN)", allr, "arm_lang", "R5|EN", NON("R5")),
    ("R5 - R2 (non-EN)", allr, "arm_lang", NON("R5"), NON("R2"))]
for v in ("PQI", "PQI_R"):
    for label, df, col, a, b in specs:
        s = per_item_series(df, v, col, a, b)
        base = bootstrap_mean(s, B, seed)
        nc, clo, chi = cluster_ci(s, clusters)
        nm, mlo, mhi = cluster_ci(s, mis)
        rows.append({"outcome": v, "effect": label, "n_items": len(s), "estimate": base["estimate"],
                     "item_ci_low": base["ci_low"], "item_ci_high": base["ci_high"],
                     "n_constructs": nc, "construct_ci_low": clo, "construct_ci_high": chi,
                     "n_misconceptions": nm, "misconception_ci_low": mlo, "misconception_ci_high": mhi})
t10 = pd.DataFrame(rows)
t10.to_csv(T / "revision_cluster_sensitivity.csv", index=False)
print("\n[10] construct / misconception cluster bootstrap (percentile)")
print(t10.round(4).to_string(index=False))

# ---------------------------------------------------------------- 11. R5 structure (E3 local)
r5 = [f for f in finals.values() if f["arm"] == "R5"]
df5 = pd.DataFrame([{"item_id": f["item_id"], "model": f["model"], "lang": f["repair_lang"], "cue": f["cue"],
                     "canon": f.get("canonical_message_en"), "plan": str(f.get("plan_raw") or f.get("plan")),
                     "parse_ok": bool(f.get("plan_parse_ok")), "schema_ok": bool(f.get("plan_schema_ok")),
                     "retried": bool(f.get("plan_retried")), "json_repair": bool(f.get("plan_json_repair_fired"))}
                    for f in r5])
grp = df5.groupby(["item_id", "model"])
t11 = pd.DataFrame([{
    "r5_outputs": len(df5), "item_tutor_groups": grp.ngroups,
    "conditions_per_group_median": float(grp.size().median()),
    "groups_all_plans_distinct": int((grp.plan.nunique() == grp.size()).sum()),
    "distinct_canonical_per_group": "; ".join(f"{k}:{v}" for k, v in
                                              grp.canon.nunique().value_counts().sort_index().items()),
    "plan_parse_ok": int(df5.parse_ok.sum()), "plan_schema_ok": int(df5.schema_ok.sum()),
    "plan_retried": int(df5.retried.sum()), "plan_json_repair": int(df5.json_repair.sum())}])
t11.to_csv(T / "revision_r5_structure.csv", index=False)
print("\n[11] R5 plan structure")
print(t11.T.to_string())
print("\ndone")

# ---------------------------------------------------------------- 12. R5 canonical-message fidelity (E3 local)
# Does the canonical English message keep the student's answer? Automatic check on the answer
# sentence (last paragraph); every miss was then read and classified by hand (2026-10-10).
MANUAL = {  # (item, variety): class; all other misses are faithful paraphrases
    ("eedi-741-D", "BN"): "answer lost", ("eedi-741-D", "HI"): "answer mistranslated",
    ("eedi-1487-B", "BN"): "garbled (name left in Bengali script)",
    ("eedi-1487-B", "HI"): "garbled (name left in Devanagari)",
    ("eedi-1015-A", "HI"): "meaning shifted", ("eedi-1303-A", "HI"): "meaning shifted"}
INDIC = re.compile(r"[ऀ-ॿঀ-৿]")
seen, rows = set(), []
for f in r5:
    k = (f["item_id"], f["repair_lang"])
    if f["cue"] != "C0" or k in seen:
        continue
    seen.add(k)
    canon = (f.get("canonical_message_en") or "").strip()
    last = re.split(r"\n\s*\n", canon)[-1]
    sa = items_all[f["item_id"]]["student_answer_en"]
    hit = re.sub(r"\s+", "", sa).lower() in re.sub(r"\s+", "", last).lower()
    rows.append({"item_id": k[0], "variety": k[1], "answer_string_found": hit,
                 "class": "exact" if hit else MANUAL.get(k, "paraphrase (meaning kept)"),
                 "residual_indic_script": bool(INDIC.search(canon))})
cf = pd.DataFrame(rows)
t12 = (cf.groupby("variety").agg(canonical_messages=("item_id", "size"), exact=("answer_string_found", "sum"),
                                 residual_indic_script=("residual_indic_script", "sum")).reset_index())
for cls in ["paraphrase (meaning kept)", "meaning shifted", "answer lost", "answer mistranslated"]:
    t12[cls] = t12.variety.map(cf[cf["class"] == cls].groupby("variety").size()).fillna(0).astype(int)
t12["garbled"] = t12.variety.map(cf[cf["class"].str.startswith("garbled")].groupby("variety").size()).fillna(0).astype(int)
t12.to_csv(T / "revision_r5_canonical_fidelity.csv", index=False)
print("\n[12] R5 canonical message keeps the student's answer?")
print(t12.to_string(index=False))
