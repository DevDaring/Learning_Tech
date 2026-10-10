"""Stage 9: estimands, statistics, tables 1-9 and figures 1-5 (Sections 6.5, 6.6, 10).

Every table is written to results/tables/ as CSV with a ``status`` column that reads
PROVISIONAL until human annotations have come back (Section 6.7). Every comparison is
kept, significant or not; significance is only ever claimed from Holm-adjusted p-values
within the stated family.
"""
from __future__ import annotations

import hashlib
import json
import os
import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from ..analysis import figures as F
from .. import names as names_mod
from ..analysis.stats import add_holm, diff_in_diff, gap, krippendorff_alpha, paired_diff
from ..config import CODES_DIR, Run
from ..io import AppendLog, read_table
from ..rubric import ALPHA_LEVEL, CATEGORICAL, DIMENSIONS, ORDINAL, norm_label, ordinal_index

log = logging.getLogger("tlt.s09")

MICROMAMBA = Path.home() / ".local/micromamba/bin/micromamba"
R_ROOT = Path.home() / ".local/micromamba/root"


class Ctx:
    def __init__(self, run: Run):
        self.run = run
        self.B = int(run.design["bootstrap"]["resamples"])
        self.seed = run.seed
        self.status = "PROVISIONAL" if run.provisional else "FINAL"
        if run.path("results", "PRELIMINARY_human_gates_outstanding.json").exists():
            self.status = "PRELIMINARY (human review of translations/templates/names outstanding)"
        self.tdir = run.path("results", "tables", "x").parent
        self.fdir = run.path("results", "figures", "x").parent

    def write(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        df = df.drop(columns=["result_status"], errors="ignore").copy()
        df.insert(0, "result_status", self.status)
        df.to_csv(self.tdir / f"{name}.csv", index=False)
        return df


def _read(run: Run, *parts) -> pd.DataFrame:
    p = run.path(*parts)
    return read_table(p) if p.exists() else pd.DataFrame()


# ------------------------------------------------------------------ table 1: models

def table1_models(ctx: Ctx) -> pd.DataFrame:
    """Models, roles, routes actually used, returned version strings and run dates. Every
    log is read (including superseded records) because this table documents what ran.
    Utility calls made inside other records (JSON repair, B5 answer equivalence) are
    counted from the routes those records store."""
    run = ctx.run
    logs = ["generations", "judgements", "repair_generations", "repair_judgements", "utility_calls",
            "b5_student", "b2_praise_labels", "mrbench_judgements", "translated_judgements", "repair_steps"]
    recs = []
    for name in logs:
        for r in AppendLog(run.path("runs", f"{name}.jsonl")).records():
            recs.append({"log": name, "model": r.get("judge") or r.get("model"), "status": r.get("status"),
                         "route": r.get("route_used"), "returned": r.get("model_returned"),
                         "fallback": bool(r.get("fallback_used")), "ts": r.get("timestamp"),
                         "json_repair": bool(r.get("json_repair_fired"))})
            for rk, mk in (("json_repair_route", None), ("aux_utility_route", "aux_utility_returned")):
                if r.get(rk):
                    recs.append({"log": f"{name}:embedded", "model": "utility", "status": "ok",
                                 "route": r[rk], "returned": r.get(mk) if mk else None,
                                 "fallback": not str(r[rk]).startswith(("linkapi", "mock")), "ts": r.get("timestamp"),
                                 "json_repair": False})
    df = pd.DataFrame(recs)
    if df.empty:
        return ctx.write(df, "table1_models")
    rows = []
    for m, g in df.groupby("model"):
        spec = run.models["models"].get(m, {})
        ok = g[g.status == "ok"]
        rows.append({"model_key": m, "display": spec.get("display", m), "vendor": spec.get("vendor"),
                     "tier": spec.get("tier", ""), "role": spec.get("role", ""),
                     "configured_routes": " > ".join(f"{r['kind']}:{r['model']}" for r in spec.get("routes", [])),
                     "routes_used": json.dumps(ok.route.value_counts().to_dict()),
                     "returned_version_strings": json.dumps(sorted(set(ok.returned.dropna()))),
                     "calls": len(g), "failed_calls": int((g.status != "ok").sum()),
                     "fallback_rate": float(ok.fallback.mean()) if len(ok) else np.nan,
                     "json_repair_rate": float(g.json_repair.mean()),
                     "first_call_utc": g.ts.min(), "last_call_utc": g.ts.max()})
    return ctx.write(pd.DataFrame(rows), "table1_models")


def route_by_condition(ctx: Ctx, auto: pd.DataFrame) -> pd.DataFrame:
    """Route and returned version per block x condition x producer: shows whether any
    contrast is confounded with a fallback route or a model update."""
    if auto.empty:
        return pd.DataFrame()
    ok = auto[auto.status == "ok"]
    t = (ok.groupby(["block", "condition", "model", "route_used", "model_returned"]).size()
         .rename("n").reset_index())
    return ctx.write(t, "route_by_block_condition")


# ------------------------------------------------------------ table 3 / agreement

def judge_alpha(ctx: Ctx, long: pd.DataFrame) -> pd.DataFrame:
    """Krippendorff's alpha among the judges, per dimension, per block x condition, so
    each 'language' is the language the replies are actually written in (Block C replies
    are Hindi/Bengali although the student wrote English; Block D is reported apart)."""
    rows = []
    for (block, cond), g in long[long.variant == "original"].groupby(["block", "condition"]):
        for d in DIMENSIONS:
            piv = g.pivot_table(index="target_id", columns="judge", values=d, aggfunc="first")
            if piv.shape[1] < 2:
                continue
            mat = np.vstack([piv[j].map(lambda x, d=d: ordinal_index(d, x if isinstance(x, str) else None)).values
                             for j in piv.columns])
            rows.append({"block": block, "condition": cond, "dimension": d, "level": ALPHA_LEVEL[d],
                         "n_units": len(piv), "alpha_judges": krippendorff_alpha(mat, ALPHA_LEVEL[d])})
    return ctx.write(pd.DataFrame(rows), "inter_judge_alpha")


def _consensus(d: str, labels: list[str | None]) -> str | None:
    """Human consensus: median (ordinal) or majority (categorical) of valid labels. Two
    annotators who disagree give no consensus (missing), so no annotator is favoured."""
    valid = [x for x in labels if x is not None]
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]
    if d in CATEGORICAL:
        counts = {v: valid.count(v) for v in set(valid)}
        top = max(counts.values())
        win = [v for v, c in counts.items() if c == top]
        return win[0] if len(win) == 1 else None
    idx = sorted(DIMENSIONS[d].index(v) for v in valid)
    if len(idx) % 2 == 0:
        mid = idx[len(idx) // 2 - 1:len(idx) // 2 + 1]
        return DIMENSIONS[d][mid[0]] if mid[0] == mid[1] else None
    return DIMENSIONS[d][idx[len(idx) // 2]]


def table3_human(ctx: Ctx, agg: pd.DataFrame) -> pd.DataFrame:
    run = ctx.run
    rdir = run.path("review", "human_returned", "x").parent
    files = sorted(rdir.glob("*.csv"))
    key = _read(run, "review", "human_annotation_key.csv")
    if not files or key.empty:
        return ctx.write(pd.DataFrame([{"note": "human annotations not yet returned; table pending"}]),
                         "table3_judge_human_agreement")
    # Annotators are labelled teacher_1, teacher_2, ... in every output: no annotator identity
    # (not even the file name) leaves review/human_returned/.
    raw = {f"teacher_{i}": pd.read_csv(f, dtype=str, keep_default_na=False).set_index("row_id")
           for i, f in enumerate(files, start=1)}
    names = sorted(raw)
    key = key.set_index("row_id")
    # judge labels for the very replies the teachers rated (see s05.write_anchor_table)
    anchor = _read(run, "results", "human_anchor_judgements_aggregated.csv")
    aggi = (anchor if not anchor.empty else agg).set_index("target_id")
    invalid = {}
    rows = []

    def idx(d, s):
        return np.array([ordinal_index(d, x if isinstance(x, str) else None) for x in s])

    for d in DIMENSIONS:
        labs = {}
        for n in names:
            col = raw[n][d] if d in raw[n] else pd.Series(dtype=str)
            norm = col.map(lambda x, d=d: norm_label(d, x) if str(x).strip() else None)
            invalid[(n, d)] = int(((col.astype(str).str.strip() != "") & norm.isna()).sum())
            labs[n] = norm
        human = pd.Series({rid: _consensus(d, [labs[n].get(rid) for n in names]) for rid in key.index})
        for lang, kg in key.groupby("language"):
            ids, tids = kg.index, kg.target_id.values
            h = human.reindex(ids).values
            row = {"language": lang, "dimension": d, "n_consensus": int(pd.notna(h).sum())}
            if len(names) > 1:
                row["alpha_humans"] = krippendorff_alpha(np.vstack([idx(d, labs[n].reindex(ids)) for n in names]),
                                                         ALPHA_LEVEL[d])
            tj = aggi[d].reindex(tids).values if d in aggi else np.full(len(tids), None)
            row["alpha_aggjudge_vs_consensus"] = krippendorff_alpha(np.vstack([idx(d, h), idx(d, tj)]), ALPHA_LEVEL[d])
            for n in names:  # per annotator, so no single annotator drives the consensus result
                row[f"alpha_aggjudge_vs_{n}"] = krippendorff_alpha(
                    np.vstack([idx(d, labs[n].reindex(ids)), idx(d, tj)]), ALPHA_LEVEL[d])
            for j in run.judges:
                if f"{d}__{j}" in aggi:
                    row[f"alpha_{j}_vs_consensus"] = krippendorff_alpha(
                        np.vstack([idx(d, h), idx(d, aggi[f"{d}__{j}"].reindex(tids).values)]), ALPHA_LEVEL[d])
            rows.append(row)
    ctx.write(pd.DataFrame([{"annotator": n, "dimension": d, "invalid_labels": v} for (n, d), v in invalid.items()]),
              "table3b_human_invalid_labels")
    return ctx.write(pd.DataFrame(rows), "table3_judge_human_agreement")


# --------------------------------------------------------------- tables 4-7 (gaps)

def _gap_rows(ctx, df, family, label, a, b, models, cond_col="condition", extra=None):
    rows = []
    for m in list(models) + [None]:
        r = gap(df, "PQI", cond_col, a, b, producer=m, B=ctx.B, seed=ctx.seed)
        rows.append({"family": family, "comparison": label, "model": m or "ALL", **(extra or {}), **r})
    return rows


def table4_language(ctx: Ctx, agg: pd.DataFrame) -> pd.DataFrame:
    A = agg[(agg.block == "A") & (agg.variant == "original")]
    models = sorted(A.model.unique())
    means = A.groupby(["model", "condition"])["PQI"].mean().unstack("condition")
    ctx.write(means.reset_index(), "table4a_pqi_by_language_producer")
    langs = [x for x in ctx.run.design["languages"]["core"] if x != "EN"]
    rows = []
    for l in langs:
        rows += _gap_rows(ctx, A, "language_gap", f"LG({l}) = PQI(EN) - PQI({l})", "EN", l, models,
                          extra={"language": l})
    # Sensitivity: PQI-R (only the components judges agree on; D41)
    if "PQI_R" in A:
        rr = []
        for l in langs:
            for m in list(models) + [None]:
                r = gap(A, "PQI_R", "condition", "EN", l, producer=m, B=ctx.B, seed=ctx.seed)
                rr.append({"family": "sens_language_gap_PQI_R", "comparison": f"LG({l}) on PQI-R",
                           "language": l, "model": m or "ALL", **r})
        ctx.write(add_holm(pd.DataFrame(rr)), "table4d_language_gaps_PQI_R")
    # Sensitivity: truncated replies removed (Indic text uses more tokens per word, so the
    # 600-token cap can cut HI/BN replies more often than English ones).
    if "truncated" in A:
        s = A[A.truncated != True]  # noqa: E712
        srows = []
        for l in langs:
            srows += _gap_rows(ctx, s, "sens_language_gap_no_truncated", f"LG({l}) without truncated replies",
                               "EN", l, models, extra={"language": l})
        ctx.write(add_holm(pd.DataFrame(srows)), "table4b_language_gaps_no_truncation")
        tr = A.groupby(["condition", "model"])["truncated"].mean().unstack("model").reset_index()
        ctx.write(tr, "table4c_truncation_rate_by_language")
    return ctx.write(add_holm(pd.DataFrame(rows)), "table4_language_gaps")


def _block_b_frame(ctx: Ctx, agg: pd.DataFrame) -> pd.DataFrame:
    """Block B replies plus Block A C0 replies for Block B items (EN/HI/BN)."""
    d = ctx.run.design
    B_ = agg[agg.block.isin(["A", "B"]) & (agg.variant == "original")].copy()
    items_b = set(agg[agg.block == "B"].item_id)
    B_ = B_[B_.item_id.isin(items_b) & B_.condition.isin(d["languages"]["identity"])]
    return B_[(B_.block == "B") | (B_.cue == "C0")]


def table5_identity(ctx: Ctx, agg: pd.DataFrame, auto: pd.DataFrame) -> pd.DataFrame:
    d = ctx.run.design
    B_ = _block_b_frame(ctx, agg)
    models = sorted(B_.model.unique())
    rows = []
    for lang in d["languages"]["identity"]:
        s = B_[B_.condition == lang]
        for pair, spec in d["cues"]["pairs"].items():
            rows += _gap_rows(ctx, s, "identity_gap", f"IG {pair}", spec["adv"], spec["dis"], models, "cue",
                              {"language": lang, "pair": pair})
        rows += _gap_rows(ctx, s, "identity_vs_C0", "C0 - S-VER", ["C0"], ["S-VER"], models, "cue",
                          {"language": lang, "pair": "schooling_vs_none"})
    t = add_holm(pd.DataFrame(rows))
    irows = []
    for pair, spec in d["cues"]["pairs"].items():
        for lang in ("HI", "BN"):
            for m in models + [None]:
                r = diff_in_diff(B_, "PQI", "cue", spec["adv"], spec["dis"], "condition", lang, "EN",
                                 producer=m, B=ctx.B, seed=ctx.seed)
                irows.append({"family": "interaction", "comparison": f"IG({lang}) - IG(EN), {pair}", "pair": pair,
                              "language": lang, "model": m or "ALL", **r})
    inter = ctx.write(add_holm(pd.DataFrame(irows)), "table5b_identity_language_interaction")
    srows = []
    if "cue_echo" in B_:
        s = B_[B_.cue_echo != True]  # noqa: E712
        for lang in d["languages"]["identity"]:
            srows += _gap_rows(ctx, s[s.condition == lang], "sens_no_cue_echo", "IG schooling (no cue echo)",
                               ["S-ENG"], ["S-VER"], models, "cue", {"language": lang, "pair": "schooling"})
    if "hi_gender_marked" in B_:
        s = B_[(B_.condition == "HI") & (B_.hi_gender_marked != True)]  # noqa: E712
        srows += _gap_rows(ctx, s, "sens_no_hi_gender_marking", "IG gender (HI, no gendered agreement)",
                           d["cues"]["pairs"]["gender"]["adv"], d["cues"]["pairs"]["gender"]["dis"], models, "cue",
                           {"language": "HI", "pair": "gender"})
        rate = (B_[B_.condition == "HI"].groupby("cue")["hi_gender_marked"].mean().rename("rate").reset_index())
        ctx.write(rate, "table5d_hi_gender_marking_rate")
    for sur in names_mod.SURNAME_DOMINANT + names_mod.SURNAME_DALIT:
        drop = set(B_[B_.surname == sur].item_id)
        s = B_[~B_.item_id.isin(drop)]
        for lang in d["languages"]["identity"]:
            r = gap(s[s.condition == lang], "PQI", "cue", d["cues"]["pairs"]["caste"]["adv"],
                    d["cues"]["pairs"]["caste"]["dis"], B=ctx.B, seed=ctx.seed)
            srows.append({"family": "sens_leave_one_surname_out", "comparison": f"IG caste without {sur}",
                          "language": lang, "pair": "caste", "model": "ALL", "dropped_surname": sur,
                          "items_dropped": len(drop), **r})
    if srows:
        ctx.write(add_holm(pd.DataFrame(srows)), "table5c_identity_sensitivity")
    if "PQI_R" in B_:
        rr = []
        for lang in d["languages"]["identity"]:
            s = B_[B_.condition == lang]
            for pair, spec in d["cues"]["pairs"].items():
                for m in models + [None]:
                    r = gap(s, "PQI_R", "cue", spec["adv"], spec["dis"], producer=m, B=ctx.B, seed=ctx.seed)
                    rr.append({"family": "sens_identity_gap_PQI_R", "comparison": f"IG {pair} on PQI-R",
                               "language": lang, "pair": pair, "model": m or "ALL", **r})
        ctx.write(add_holm(pd.DataFrame(rr)), "table5e_identity_gaps_PQI_R")
    ctx.write(t, "table5_identity_gaps")
    return pd.concat([t, inter])


def table6_decomposition(ctx: Ctx, agg: pd.DataFrame) -> pd.DataFrame:
    """Total = generation + input, computed on the SAME item x producer cells (those with
    all three conditions observed), so the components add up exactly."""
    O = agg[agg.variant == "original"]
    items_c = set(O[O.block == "C"].item_id)
    rows = []
    for l in ctx.run.design["languages"]["decomposition_targets"]:
        a_en = O[(O.block == "A") & (O.condition == "EN") & O.item_id.isin(items_c)].assign(k="EN_EN")
        a_l = O[(O.block == "A") & (O.condition == l) & O.item_id.isin(items_c)].assign(k="l_l")
        c_l = O[(O.block == "C") & (O.condition == f"EN>{l}")].assign(k="EN_l")
        df = pd.concat([a_en, a_l, c_l])
        have = df.dropna(subset=["PQI"]).groupby(["item_id", "model"])["k"].nunique()
        full = have[have == 3].index
        df = df.set_index(["item_id", "model"]).loc[lambda x: x.index.isin(full)].reset_index()
        for m in sorted(df.model.unique()) + [None]:
            for comp, a, b in (("total = PQI(EN,EN) - PQI(l,l)", "EN_EN", "l_l"),
                               ("generation = PQI(EN in,EN out) - PQI(EN in,l out)", "EN_EN", "EN_l"),
                               ("input = PQI(EN in,l out) - PQI(l in,l out)", "EN_l", "l_l")):
                r = gap(df, "PQI", "k", a, b, producer=m, B=ctx.B, seed=ctx.seed)
                rows.append({"family": "decomposition", "language": l, "component": comp, "model": m or "ALL", **r})
    return ctx.write(add_holm(pd.DataFrame(rows)), "table6_decomposition")


def table7_native(ctx: Ctx, agg: pd.DataFrame) -> pd.DataFrame:
    D = agg[(agg.block == "D") & (agg.variant == "original")]
    if D.empty:
        return ctx.write(pd.DataFrame(), "table7_native_control")
    models = sorted(D.model.unique())
    rows = []
    for label, a, b in (("EN-OFFICIAL - BN-NATIVE", "EN-OFFICIAL", "BN-NATIVE"),
                        ("EN-OFFICIAL - BN-MT", "EN-OFFICIAL", "BN-MT"),
                        ("BN-MT - BN-NATIVE (translation artefact)", "BN-MT", "BN-NATIVE")):
        rows += _gap_rows(ctx, D, "native_control", label, a, b, models)
    means = D.groupby(["model", "condition"])["PQI"].mean().unstack("condition").reset_index()
    ctx.write(means, "table7a_block_d_means")
    return ctx.write(add_holm(pd.DataFrame(rows)), "table7_native_control")


def _indicators(O: pd.DataFrame) -> pd.DataFrame:
    cd = {"High": 1.0, "Medium": 0.5, "Low": 0.0, "None": 0.0}
    return O.assign(inflated=(O.Praise_Calibration == "Inflated").astype(float).where(O.Praise_Calibration.notna()),
                    lowered=O.Expectation_Level.isin(["Simplified below", "Condescending"]).astype(float)
                    .where(O.Expectation_Level.notna()),
                    condescending=(O.Expectation_Level == "Condescending").astype(float)
                    .where(O.Expectation_Level.notna()),
                    cog=O.Cognitive_Demand.map(cd))


def expectation_indicators(ctx: Ctx, agg: pd.DataFrame) -> pd.DataFrame:
    """Expectation indicators (Section 6.5) with paired tests for BOTH the language and
    the identity contrasts (signs: advantaged - disadvantaged, EN - l)."""
    d = ctx.run.design
    O = _indicators(agg[(agg.variant == "original") & agg.block.isin(["A", "B"])])
    t = O.groupby(["block", "condition", "cue", "model"])[["inflated", "lowered", "condescending", "cog"]].mean()
    t = t.reset_index().rename(columns={"inflated": "rate_inflated_praise", "lowered": "rate_simplified_or_condescending",
                                        "condescending": "rate_condescending", "cog": "mean_cognitive_demand"})
    A = O[O.block == "A"]
    BB = _indicators(_block_b_frame(ctx, agg))
    models = sorted(BB.model.unique())
    rows = []
    for ind in ("inflated", "lowered", "condescending", "cog"):
        for l in [x for x in d["languages"]["core"] if x != "EN"]:
            for m in sorted(A.model.unique()) + [None]:
                r = gap(A, ind, "condition", "EN", l, producer=m, B=ctx.B, seed=ctx.seed)
                rows.append({"family": f"expectation_{ind}_language", "indicator": ind, "comparison": f"EN - {l}",
                             "model": m or "ALL", **r})
        for lang in d["languages"]["identity"]:
            s = BB[BB.condition == lang]
            for pair, spec in d["cues"]["pairs"].items():
                for m in models + [None]:
                    r = gap(s, ind, "cue", spec["adv"], spec["dis"], producer=m, B=ctx.B, seed=ctx.seed)
                    rows.append({"family": f"expectation_{ind}_identity", "indicator": ind,
                                 "comparison": f"{pair} ({lang}): adv - dis", "language": lang, "pair": pair,
                                 "model": m or "ALL", **r})
    ctx.write(add_holm(pd.DataFrame(rows)), "expectation_indicator_tests")
    return ctx.write(t, "expectation_indicators")


# ------------------------------------------------------------------- table 9 repair

def _ratio_closed(df: pd.DataFrame, arm: str, l: str, B: int, seed: int, m: str | None) -> dict:
    """Share of the R0 within-arm gap closed by an arm, 1 - gap_arm / gap_R0, with a
    percentile interval from the SAME item resamples. Reported only when the R0 gap's own
    interval excludes 0 (otherwise the ratio is unstable and is left missing)."""
    sub = df if m is None else df[df.model == m]
    g0 = paired_diff(sub, "PQI", "arm_lang", "R0|EN", f"R0|{l}", ["item_id", "model"])
    g1 = paired_diff(sub, "PQI", "arm_lang", f"{arm}|EN", f"{arm}|{l}", ["item_id", "model"])
    mm = g0.merge(g1, on=["item_id", "model"], suffixes=("_0", "_1"))
    if m is None:
        n_prod = sub.model.nunique()
        cnt = mm.groupby("item_id").model.nunique()
        mm = mm[mm.item_id.isin(cnt[cnt == n_prod].index)]
    per = mm.groupby("item_id")[["diff_0", "diff_1"]].mean()
    if len(per) < 2:
        return {"share_gap_closed": np.nan}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per), size=(B, len(per)))
    b0, b1 = per.diff_0.values[idx].mean(1), per.diff_1.values[idx].mean(1)
    lo0, hi0 = np.quantile(b0, [0.025, 0.975])
    est = 1 - per.diff_1.mean() / per.diff_0.mean() if per.diff_0.mean() != 0 else np.nan
    if lo0 <= 0 <= hi0:
        return {"share_gap_closed": np.nan, "share_note": "R0 gap interval includes 0: ratio not reported"}
    with np.errstate(divide="ignore", invalid="ignore"):
        rb = 1 - b1 / b0
    lo, hi = np.nanquantile(rb, [0.025, 0.975])
    return {"share_gap_closed": est, "share_ci_low": lo, "share_ci_high": hi, "share_note": ""}


def table9_repair(ctx: Ctx, agg: pd.DataFrame, rep: pd.DataFrame) -> pd.DataFrame:
    """Per arm x language: absolute PQI, Language_Match, improvement over R0 in the same
    language (tested), within-arm gap and its reduction relative to R0 (tested as a
    difference in differences), share of gap closed (ratio with interval), and the
    simulated-student gain from B5."""
    if rep.empty:
        return pd.DataFrame()
    d = ctx.run.design["repair"]
    items = set(rep.item_id)
    R0 = agg[(agg.block == "A") & (agg.cue == "C0") & agg.item_id.isin(items) & (agg.variant == "original")].copy()
    R0["arm"], R0["lang"] = "R0", R0.condition
    R = rep[rep.cue == "C0"].copy()
    R["lang"] = R.condition
    allr = pd.concat([R0, R], ignore_index=True)
    allr["arm_lang"] = allr.arm + "|" + allr.lang
    has_en = (R.lang == "EN").any()
    rows = []
    for arm in ["R0"] + list(d["arms"]):
        for l in d["languages"]:
            for m in sorted(allr.model.unique()) + [None]:
                s = allr if m is None else allr[allr.model == m]
                cell = s[s.arm_lang == f"{arm}|{l}"]
                if cell.empty:
                    continue
                row = {"arm": arm, "language": l, "model": m or "ALL", "abs_pqi": float(cell.PQI.mean()),
                       "language_match_yes": float((cell.Language_Match == "Yes").mean())}
                g_in = gap(allr, "PQI", "arm_lang", f"{arm}|EN", f"{arm}|{l}", producer=m, B=ctx.B, seed=ctx.seed) \
                    if (has_en or arm == "R0") else {}
                row.update({"gap_within_arm": g_in.get("estimate", np.nan),
                            "gap_within_arm_ci_low": g_in.get("ci_low", np.nan),
                            "gap_within_arm_ci_high": g_in.get("ci_high", np.nan)})
                g_def = gap(allr, "PQI", "arm_lang", "R0|EN", f"{arm}|{l}", producer=m, B=ctx.B, seed=ctx.seed)
                row.update({"gap_vs_default_EN": g_def["estimate"], "gap_vs_default_EN_ci_low": g_def["ci_low"],
                            "gap_vs_default_EN_ci_high": g_def["ci_high"]})
                if arm != "R0":
                    imp = gap(allr, "PQI", "arm_lang", f"{arm}|{l}", f"R0|{l}", producer=m, B=ctx.B, seed=ctx.seed)
                    row.update({"improvement_vs_R0": imp["estimate"], "improvement_ci_low": imp["ci_low"],
                                "improvement_ci_high": imp["ci_high"], "p_improvement": imp["p_value"],
                                "d_z_improvement": imp["d_z"], "d_response_improvement": imp.get("d_response"),
                                "n_items": imp["n_items"]})
                    if has_en:
                        red = diff_in_diff(allr.assign(lang_c=allr.lang), "PQI", "lang_c", "EN", l, "arm", "R0", arm,
                                           producer=m, B=ctx.B, seed=ctx.seed)
                        row.update({"gap_reduction_vs_R0": red["estimate"], "gap_reduction_ci_low": red["ci_low"],
                                    "gap_reduction_ci_high": red["ci_high"], "p_gap_reduction": red["p_value"]})
                        row.update(_ratio_closed(allr, arm, l, ctx.B, ctx.seed, m))
                rows.append(row)
    t = pd.DataFrame(rows)
    # Two families, each Holm-corrected: absolute improvement, and gap reduction.
    for col, fam in (("p_improvement", "repair_improvement"), ("p_gap_reduction", "repair_gap_reduction")):
        if col in t:
            tmp = t[[col]].rename(columns={col: "p_value"}).assign(family=fam)
            t[f"{col}_holm"] = add_holm(tmp)["p_holm"].values
    t["p_value"] = t.get("p_improvement")
    t["p_holm"] = t.get("p_improvement_holm")
    b5 = _read(ctx.run, "results", "baselines", "b5_simulated_student.csv")
    if not b5.empty:
        b5 = b5.rename(columns={"gain_estimate": "simstudent_gain", "acc_after_tutor": "simstudent_acc"})[
            ["arm", "language", "model", "simstudent_gain", "simstudent_acc", "leak_share_among_correct"]]
        t = t.merge(b5, on=["arm", "language", "model"], how="left")
    # Identity cues: does R5 shrink the C0 - cue gap? Tested as IG(R0) - IG(R5).
    irows = []
    lang = d["identity_language"]
    B0 = agg[(agg.block.isin(["A", "B"])) & (agg.condition == lang) & agg.item_id.isin(items)
             & agg.cue.isin(["C0"] + list(d["identity_cues"])) & (agg.variant == "original")].copy()
    B0["arm"] = "R0"
    both = pd.concat([B0, rep[(rep.condition == lang) & (rep.arm == "R5")]], ignore_index=True)
    for cue in d["identity_cues"]:
        for m in sorted(both.model.unique()) + [None]:
            for arm in d["identity_arms"]:
                r = gap(both[both.arm == arm], "PQI", "cue", "C0", cue, producer=m, B=ctx.B, seed=ctx.seed)
                irows.append({"family": "repair_identity_gap", "arm": arm, "language": lang,
                              "comparison": f"C0 - {cue}", "model": m or "ALL", **r})
            r = diff_in_diff(both, "PQI", "cue", "C0", cue, "arm", "R0", "R5", producer=m, B=ctx.B, seed=ctx.seed)
            irows.append({"family": "repair_identity_reduction", "arm": "R0 - R5", "language": lang,
                          "comparison": f"IG(R0) - IG(R5), C0 - {cue}", "model": m or "ALL", **r})
    if irows:
        ctx.write(add_holm(pd.DataFrame(irows)), "table9b_repair_identity")
    return ctx.write(t, "table9_repair")


# ------------------------------------------------------------------- mixed models

def mixed_models(ctx: Ctx, agg: pd.DataFrame) -> dict:
    run = ctx.run
    out_dir = run.path("results", "mixed", "x").parent
    rscript = None
    if MICROMAMBA.exists():
        rscript = [str(MICROMAMBA), "run", "-r", str(R_ROOT), "-n", "tlt_r", "Rscript"]
    elif shutil.which("Rscript"):
        rscript = ["Rscript"]
    sets = {"A": agg[(agg.block == "A") & (agg.variant == "original")]}
    items_b = set(agg[agg.block == "B"].item_id)
    for lang in run.design["languages"]["identity"]:
        s = agg[agg.block.isin(["A", "B"]) & (agg.condition == lang) & agg.item_id.isin(items_b)
                & (agg.variant == "original")]
        sets[f"B_{lang}"] = s[(s.block == "B") | (s.cue == "C0")].assign(condition=lambda x: x.cue)
    status, procs = {}, {}
    for name, df in sets.items():
        if df.empty or df.model.nunique() < 1:
            continue
        cols = {"item_id": df.item_id, "model": df.model, "condition": df.condition, "PQI": df.PQI}
        for dname in ORDINAL:
            cols[f"ord__{dname}"] = df[dname].map(lambda x, dn=dname: ordinal_index(dn, x if isinstance(x, str) else None))
        cols["Revealing__bad"] = df.Revealing_of_the_Answer.map(
            lambda x: np.nan if not isinstance(x, str) else float(x != "No"))
        cols["Inflated__bad"] = df.Praise_Calibration.map(
            lambda x: np.nan if not isinstance(x, str) else float(x == "Inflated"))
        inp = out_dir / f"input_{name}.csv"
        pd.DataFrame(cols).to_csv(inp, index=False)
        if rscript is None:
            status[name] = "R not available: mixed models skipped"
            continue
        if df.model.nunique() < 2 or df.item_id.nunique() < 5:
            status[name] = "too little data for mixed models (pilot)"
            continue
        # Skip a set whose input and R script are byte-identical to a finished fit.
        key = hashlib.sha256(inp.read_bytes() + (CODES_DIR / "src/tlt/analysis/mixed_models.R").read_bytes()).hexdigest()
        done = out_dir / f"done_{name}.json"
        outs = [out_dir / f"mixed_coefficients_block{name}.csv", out_dir / f"mixed_tests_block{name}.csv"]
        if done.exists() and json.loads(done.read_text()).get("sha256") == key and all(o.exists() for o in outs):
            status[name] = "ok (cached fit, identical input)"
            continue
        env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "TLT_R_CORES": os.environ.get("TLT_R_CORES", "6")}
        # the sets run concurrently; each fits its models in parallel inside R
        procs[name] = (subprocess.Popen(rscript + [str(CODES_DIR / "src/tlt/analysis/mixed_models.R"), str(inp),
                                                   str(out_dir), name], stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True, env=env), done, key)
    for name, (proc, done, key) in procs.items():
        try:
            _, err = proc.communicate(timeout=8 * 3600)
        except subprocess.TimeoutExpired:
            proc.kill()
            status[name] = "R timed out"
            continue
        if proc.returncode == 0:
            done.write_text(json.dumps({"sha256": key}))
            status[name] = "ok"
        else:
            status[name] = f"R failed: {err[-800:]}"
    # Holm within each family of omnibus tests
    for f in out_dir.glob("mixed_tests_block*.csv"):
        t = pd.read_csv(f)
        if "model_type" not in t:
            t = t.rename(columns={"family": "model_type"})
        # one Holm family per (model type, test) within a block, e.g. all CLMM condition effects
        t["family"] = t["model_type"] + " | " + t["test"]
        add_holm(t).to_csv(f, index=False)
    (out_dir / "status.json").write_text(json.dumps(status, indent=2))
    return status


# ------------------------------------------------------------------- operational

def operational(ctx: Ctx, auto: pd.DataFrame, long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not auto.empty:
        for (block, model), g in auto.groupby(["block", "model"]):
            ok = g[g.status == "ok"]
            rows.append({"block": block, "model": model, "calls": len(g), "failed": int((g.status != "ok").sum()),
                         "empty": int(ok.is_empty.sum()), "refusal": int(ok.refusal.sum()),
                         "truncated": int(ok.truncated.sum()), "fallback_rate": float(ok.fallback_used.mean()),
                         "leak_exact_rate": float(ok.leak_exact.mean()), "cue_echo_rate": float(ok.cue_echo.mean()),
                         "degenerate_rate": float(ok.degenerate.mean()) if "degenerate" in ok else None,
                         "mean_words": float(ok.word_count.mean()), "mean_questions": float(ok.question_count.mean()),
                         "mean_latin_share": float(ok.latin_share.mean())})
    if not long.empty:
        for j, g in long.groupby("judge"):
            rows.append({"block": "judging", "model": j, "calls": len(g), "failed": int((g.status != "ok").sum()),
                         "json_repair_rate": float(g.json_repair_fired.fillna(False).astype(bool).mean()),
                         "unparseable": int((g.status == "unparseable").sum()),
                         "fallback_rate": float(g.judge_fallback.fillna(False).astype(bool).mean())})
    return ctx.write(pd.DataFrame(rows), "operational_counts")


# ------------------------------------------------------------------------- stage

def run_stage(run: Run) -> dict:
    ctx = Ctx(run)
    agg = _read(run, "results", "judgements_aggregated.csv")
    long = _read(run, "results", "judgements_long.csv")
    auto = _read(run, "results", "auto_metrics.csv")
    rep = _read(run, "results", "repair_judgements_aggregated.csv")
    if agg.empty:
        raise SystemExit("results/judgements_aggregated.csv missing: run mask_and_judge first")
    flags = ["truncated", "cue_echo", "hi_gender_marked", "is_empty", "refusal"]
    if not auto.empty:
        agg = agg.merge(auto[["target_id"] + flags], on="target_id", how="left")
    rauto = _read(run, "results", "repair_auto_metrics.csv")
    if not rep.empty and not rauto.empty:
        rep = rep.merge(rauto[["target_id"] + flags], on="target_id", how="left")
    route_by_condition(ctx, auto)
    out = {"status": ctx.status}
    table1_models(ctx)
    jv = _read(run, "results", "judge_validation.csv")
    if not jv.empty:
        ctx.write(jv, "table2_judge_validation_mrbench")
    judge_alpha(ctx, long)
    table3_human(ctx, agg)
    t4 = table4_language(ctx, agg)
    t5 = table5_identity(ctx, agg, auto)
    table6_decomposition(ctx, agg)
    t7 = table7_native(ctx, agg)
    expectation_indicators(ctx, agg)
    tj = _read(run, "results", "translated_judging.csv")
    if not tj.empty:
        ctx.write(tj, "translated_judging")
    # Table 8: baselines beside PQI gaps
    parts = []
    for name in ("b1_mab_mdb", "b2_praise_critique", "b3_semantic_shift"):
        b = _read(run, "results", "baselines", f"{name}.csv")
        if not b.empty:
            parts.append(b.assign(baseline=name))
    ig = t5[t5.family == "identity_gap"].assign(baseline="PQI identity gap") if not t5.empty else pd.DataFrame()
    ctx.write(pd.concat(parts + [ig], ignore_index=True, sort=False), "table8_baselines_vs_pqi")
    t9 = table9_repair(ctx, agg, rep)
    out["mixed_models"] = mixed_models(ctx, agg)
    operational(ctx, auto, long)

    # figures
    prov = ctx.status == "PROVISIONAL"
    tiers = {m: run.models["models"][m].get("tier", m) for m in run.producers}
    F.fig1_design(ctx.fdir / "fig1_design", run.design, prov)
    F.fig2_profiles(agg[(agg.block == "A") & (agg.variant == "original")], tiers, ctx.fdir / "fig2_profiles", prov)
    forest = []
    for df_, lab in ((t4, lambda r: f"LG {r.language}"), (t5[t5.family == "identity_gap"] if not t5.empty else t5,
                                                          lambda r: f"IG {r.pair} {r.language}"),
                     (t7, lambda r: r.comparison)):
        if df_ is not None and not df_.empty:
            s = df_[df_.model == "ALL"].copy()
            s["label"] = s.apply(lab, axis=1)
            forest.append(s)
    if forest:
        F.fig3_forest(pd.concat(forest).dropna(subset=["estimate"]), ctx.fdir / "fig3_forest", prov)
    if not tj.empty:
        F.fig4_translated(tj, ctx.fdir / "fig4_translated_judging", prov)
    if t9 is not None and not t9.empty:
        F.fig5_repair(t9[(t9.model == "ALL") & t9.gap_within_arm.notna()], ctx.fdir / "fig5_repair", prov)
    out["tables"] = sorted(p.name for p in ctx.tdir.glob("*.csv"))
    log.info("analysis: %s", out)
    return out
