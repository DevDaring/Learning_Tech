"""Build the supplementary material (Submission1/TLT_Supplement.tex) from the result tables, so
no number is copied by hand. Compile with pdflatex afterwards.

    PYTHONPATH=src python scripts/make_supplement.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "tables"
OUT = ROOT / "Submission1" / "TLT_Supplement.tex"


def rd(name, **kw):
    return pd.read_csv(T / name, keep_default_na=False, na_values=[""], **kw)


def esc(x):
    s = str(x)
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#"),
                 ("$", r"\$"), ("^", r"\^{}")):
        s = s.replace(a, b)
    return s


def table(df, caption, fmt=None, size=r"\scriptsize", colspec=None):
    fmt = fmt or {}
    cols = list(df.columns)
    spec = colspec or ("l" * 1 + "r" * (len(cols) - 1))
    head = " & ".join(esc(c) for c in cols) + r" \\"
    lines = [r"{" + size, rf"\begin{{longtable}}{{@{{}}{spec}@{{}}}}", rf"\caption{{{caption}}}\\", r"\toprule",
             head, r"\midrule", r"\endfirsthead", r"\toprule", head, r"\midrule", r"\endhead"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append("" if pd.isna(v) else (fmt.get(c, "{:.3f}").format(v)))
            else:
                cells.append(esc(v))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{longtable}", r"}", ""]
    return "\n".join(lines)


def ci(df, est="estimate", lo="ci_low", hi="ci_high", name="Estimate [95\\% CI]", nd=3):
    df = df.copy()
    df[name] = [f"{a:.{nd}f} [{b:.{nd}f}, {c:.{nd}f}]" for a, b, c in zip(df[est], df[lo], df[hi])]
    return df


TUT = {"gemini_flash": "Gemini 3.8 Flash", "sarvam_105b": "Sarvam 105B", "kimi_k25": "Kimi K2.5",
       "nova_lite": "Nova Lite", "ministral_8b": "Ministral 3 8B", "nemotron_nano": "Nemotron 3 Nano",
       "ALL": "Pooled", "gpt41_mini": "GPT-4.1 mini", "deepseek_v32": "DeepSeek-V3.2",
       "qwen3_next": "Qwen3-Next 80B", "panel": "Panel", "gpt_oss_120b": "gpt-oss-120b (former stand-in)",
       "utility": "GPT-5 mini (utility)", "student": "Llama 3.1 8B (sim. student)"}
parts = [r"""\documentclass[10pt]{article}
\usepackage[margin=2cm]{geometry}
\usepackage{booktabs,amsmath,amssymb,longtable,graphicx}
\usepackage[T1]{fontenc}
\usepackage{newtxtext,newtxmath}
\usepackage[hidelinks]{hyperref}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thesection}{S\arabic{section}}
\title{Supplementary Material\\[4pt]\large Same Question, Lesser Lesson: Measuring and Reducing
Pedagogical Disparity in Multilingual LLM Maths Tutors}
\author{Koushik Deb and Abhinaba Basu}
\date{}
\begin{document}
\maketitle
Every table below is generated from the released result files by
\texttt{scripts/make\_supplement.py}; per-call logs, prompts and the frozen revision plan are in
the repository.
"""]

# ------------------------------------------------------------------ S1 provenance
parts.append(r"\section{Model provenance}")
parts.append(r"""All calls of the main study and the revision experiments, from the per-call logs
(including calls later superseded, for example replies regenerated after a translation
correction, so counts exceed final replies). Routes: share of calls per provider route. Returned
identifier: the model string the provider returned. Dates are UTC. Settings: tutors temperature
0.7, maximum 4,000 output tokens; judges temperature 0 where accepted, 1,200 tokens; IndicTrans2
revisions \texttt{10e65a9} (en--indic) and \texttt{ac3daf0} (indic--en), beam 5.
""")
pv = rd("revision_provenance.csv")
keep = ["generations", "repair_generations", "repair_steps", "judgements", "repair_judgements",
        "revision_e4_generations", "revision_e4_judgements", "revision_e3_ablation_generations",
        "revision_e2_judgements", "revision_e3_fidelity", "utility_calls", "b5_student"]
pv = pv[pv.log.isin(keep)].copy()
pv["model"] = pv.model.map(lambda m: TUT.get(m, m))
pv["returned"] = pv.model_returned.str.slice(0, 48)
pv["dates"] = pv.first_date + " to " + pv.last_date
pv = pv[["log", "model", "role", "calls", "routes", "returned", "dates"]]
pv["calls"] = pv.calls.astype(int).map("{:,}".format)
parts.append(table(pv, "Calls by log, model and role", size=r"\tiny", colspec="p{2.4cm}p{2.2cm}p{1.6cm}rp{3.1cm}p{3.4cm}p{1.6cm}"))

# ------------------------------------------------------------------ S2 statistics
parts.append(r"""\section{Statistical procedure}
All contrasts are paired within item and tutor: replies are averaged within item $\times$ tutor
$\times$ condition, differences are taken within item $\times$ tutor, and pooled estimates use
items observed under every tutor so tutors have equal weight. Items are resampled with
replacement ($B=10{,}000$), so all conditions of an item move together. The $p$-value is the
larger of the studentized-bootstrap $p$ (zero-spread resamples kept as $t^*=\pm\infty$) and the
studentized sign-flip $p$; in simulations of normal, skewed, sparse, binary and PQI-like
differences at $n=10$--100 neither component alone held the nominal level everywhere, while the
maximum stayed at or below 0.06 (not a proof of exact 0.05 control). Intervals are bootstrap-$t$,
falling back to percentile intervals when bootstrap-$t$ is unbounded. Holm's correction is
applied within each family named in the result files; per-tutor and pooled-over-language rows
are descriptive. TOST: each one-sided test uses the same maximum-$p$ rule on the shifted
differences; equivalence at margin $\delta$ requires $\max(p_{\text{lower}},p_{\text{upper}})<0.05$
(a 90\% interval inside $\pm\delta$), Holm across the contrasts of a family.
""")

# ------------------------------------------------------------------ S3 frozen plan
parts.append(r"""\section{Revision experiments: frozen plan}
The plan (\texttt{docs/revision\_plan\_frozen.md}) was written on 10 October 2026 before any new
reply was generated, with an implementation addendum fixed before any E2--E4 call. E1: R2-L and
R2-T on the 100 repair items, six tutors, EN/HI/BN, one sample (3,600 replies), plus R2 and R5
regenerated on 20 seeded items (720 replies); primary contrasts R5$-$R2-L and R5$-$R2-T on
paired PQI-R in each language, one Holm family of six; TOST margin $\pm0.02$; decision rule: if a
simpler arm matches or exceeds R5, narrow the contribution to it (applied). E2: challenge set of
60 replies per edit (2 per language $\times$ tutor), 540 judge calls. E3: structural checks,
semantic check of 60 plan/reply pairs (180 calls), plan-only ablation (50 items, Gemini, Sarvam,
Nova Lite, EN/HI/BN; 450 replies). E4: full crossing (3,000 replies). Not run: identity repair in
English and a learner trial (a second addendum added E8 and E9, Section S10). Deviations: an OpenRouter out-of-credit response was misclassified
as a permanent error during E1 and E4 judging (affected calls were re-run with the corrected
client; none was dropped); canonical-message answer loss was found by reading the stated-rule
example and is reported, not fixed.
""")

# ------------------------------------------------------------------ S4 E1
parts.append(r"\section{Stronger repair baselines (E1)}")
e1 = rd("revision_e1_contrasts.csv")
x = e1[(e1.outcome.isin(["PQI_R", "PQI"])) & (e1.model == "ALL")].copy()
x = ci(x)
x = x[["outcome", "contrast", "language", "mean_R5", "mean_base", "Estimate [95\\% CI]", "p_value", "p_holm",
       "ci90_low", "ci90_high", "p_tost"]]
parts.append(table(x, "E1 contrasts, all tutors (TOST: 90\\% interval and $p$ at margin $\\pm0.02$; pooled rows "
                      "descriptive)", fmt={"p_value": "{:.4f}", "p_holm": "{:.4f}", "p_tost": "{:.3f}"}))
y = e1[(e1.outcome == "PQI_R") & (e1.model != "ALL")].copy()
y["model"] = y.model.map(TUT)
y = ci(y)[["model", "contrast", "language", "Estimate [95\\% CI]"]]
parts.append(table(y, "E1 contrasts by tutor (PQI-R; unadjusted)", colspec="lllr"))
dr = rd("revision_e1_drift.csv")
dr = ci(dr[dr.outcome.isin(["PQI_R", "PQI"])])[["arm", "outcome", "language", "n_items", "Estimate [95\\% CI]",
                                                 "p_value"]]
parts.append(table(dr, "Drift: contemporaneous re-run minus saved reply (20 items, six tutors)"))
cs = rd("revision_e1_cost.csv")
cs = cs[["arm", "final_replies", "llm_calls_per_reply", "input_tokens_per_reply", "output_tokens_per_reply",
         "sequential_latency_s_median", "usd_per_1000_replies", "indictrans2_calls_per_reply", "collected"]]
cs.columns = ["arm", "replies", "calls_per_reply", "input_tok", "output_tok", "latency_s_median", "usd_per_1000",
              "indictrans2_per_reply", "collected"]
parts.append(table(cs, "Tutor calls, tokens, latency and estimated cost per final reply (EN/HI/BN cells; judging "
                       "excluded; IndicTrans2 not priced)",
                   fmt={"calls_per_reply": "{:.2f}", "input_tok": "{:.0f}", "output_tok": "{:.0f}",
                        "latency_s_median": "{:.2f}", "usd_per_1000": "{:.2f}", "indictrans2_per_reply": "{:.2f}"},
                   colspec="lrrrrrrrl", size=r"\tiny"))
fl = rd("revision_e1_fluency_exploratory.csv")
fl = fl[fl.EN.notna()][["dimension", "arm", "EN", "HI", "BN"]]
parts.append(table(fl, "Exploratory: fluency-related rubric dimensions (share of replies)"))

# ------------------------------------------------------------------ S5 original arms
parts.append(r"\section{Original repair arms: paired, residual, component and equivalence analyses}")
r52 = rd("revision_r5_vs_r2.csv")
r52 = ci(r52[r52.model == "ALL"])[["outcome", "language", "mean_R5", "mean_R2", "Estimate [95\\% CI]", "p_holm"]]
parts.append(table(r52, "R5 $-$ R2 on the original repair items (four varieties and English)",
                   fmt={"p_holm": "{:.4f}"}))
co = rd("revision_component_sensitivity.csv")
co = co[co.effect.isin(["R2 improvement vs R0 (non-EN)", "R5 improvement vs R0 (non-EN)", "R5 - R2 (non-EN)"])]
co = co.pivot_table(index="outcome", columns="effect", values="estimate").reset_index()
parts.append(table(co, "Component sensitivity: improvement over R0 and R5 $-$ R2 by component (non-English pooled)"))
to = rd("revision_tost_original_arms.csv")
to = to[["outcome", "arm", "language", "estimate", "ci90_low", "ci90_high", "p_tost", "p_holm"]]
parts.append(table(to, "TOST for within-arm gaps of R4 and R5 (margin $\\pm0.02$, post hoc)",
                   fmt={"p_tost": "{:.4f}", "p_holm": "{:.4f}"}))
rs = rd("revision_e1_residuals_by_tutor.csv")
rs["model"] = rs.model.map(TUT)
rs = ci(rs)[["arm", "language", "model", "Estimate [95\\% CI]"]]
parts.append(table(rs, "Within-arm gap EN$-l$ by tutor (PQI-R, unadjusted)", colspec="lllr"))

# ------------------------------------------------------------------ S6 crossing
parts.append(r"\section{Full input/output crossing (E4)}")
cr = rd("revision_e4_crossing.csv")
cr = ci(cr)[["outcome", "language", "effect", "Estimate [95\\% CI]", "p_holm"]]
parts.append(table(cr, "Crossing effects as assigned", fmt={"p_holm": "{:.4f}"}, colspec="llp{7cm}rr"))
cp = rd("revision_e4_compliance_by_tutor.csv")
cp = cp.rename(columns={cp.columns[0]: "tutor"})
cp["tutor"] = cp.tutor.map(lambda m: TUT.get(m, m))
parts.append(table(cp, "Share of replies in the requested script, by tutor and cell", fmt={c: "{:.2f}" for c in cp.columns}))
cc = rd("revision_e4_compliance_contrast.csv")
cc = ci(cc)[["language", "sample", "outcome", "n_item_tutor_pairs", "Estimate [95\\% CI]"]]
parts.append(table(cc, "EN$\\to$EN minus $l\\to$EN, all replies and compliant replies only (diagnostic; selected sample)",
                   colspec="llllr"))

# ------------------------------------------------------------------ S7 evaluator
parts.append(r"\section{Evaluator diagnostics}")
ha = rd("revision_human_agreement_detail.csv")
hcols = ["dimension", "language", "n_rated_by_any", "n_consensus", "missing_consensus", "exact_agreement",
         "exact_ci_low", "exact_ci_high", "consensus_majority_label", "consensus_majority_share"]
h = ha[ha.dimension.isin(["Mistake_Identification", "Mistake_Location", "Revealing_of_the_Answer",
                          "Providing_Guidance", "Math_Correctness", "Praise_Calibration", "Actionability",
                          "Expectation_Level"])][hcols]
parts.append(table(h, "Panel versus teacher consensus by language (default replies; teacher 1 rated EN, HI and HI-EN "
                      "only; two disagreeing teachers give no consensus)", size=r"\tiny"))
pr = ha[ha.language == "ALL"][["dimension"] + [c for c in ha.columns if c.startswith("exact_judge_vs") or
                                              (c.startswith("exact_teacher"))]]
parts.append(table(pr, "Per-rater exact agreement, all languages (panel vs each teacher; teacher pairs)",
                   size=r"\tiny"))
# figure: teacher consensus by panel label, row-normalised, four dimensions
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

cfa = rd("revision_human_confusion.csv")
DIMS4 = [("Revealing_of_the_Answer", ["No", "Yes (correct)", "Yes (incorrect)"]),
         ("Mistake_Location", ["Yes", "To some extent", "No"]),
         ("Actionability", ["Yes", "To some extent", "No"]),
         ("Expectation_Level", ["At level", "Simplified below", "Condescending"])]
fig, axes = plt.subplots(1, 4, figsize=(11, 3.2))
for ax, (dim, labs) in zip(axes, DIMS4):
    sub = cfa[cfa.dimension == dim].pivot_table(index="teacher_consensus", columns="judge_panel", values="count",
                                                 aggfunc="sum").reindex(index=labs, columns=labs).fillna(0)
    shares = sub.div(sub.sum(axis=1).where(sub.sum(axis=1) > 0), axis=0).fillna(0)
    ax.imshow(shares.values, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labs)):
        for j in range(len(labs)):
            n = int(sub.values[i, j])
            ax.text(j, i, str(n), ha="center", va="center", fontsize=8,
                    color="white" if shares.values[i, j] > 0.6 else "#1F2933")
    ax.set_xticks(range(len(labs)), labs, fontsize=7, rotation=30, ha="right")
    ax.set_yticks(range(len(labs)), labs, fontsize=7)
    ax.set_title(dim.replace("_", " "), fontsize=9)
    ax.set_xlabel("judge panel", fontsize=8)
axes[0].set_ylabel("teacher consensus", fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / "Submission1" / "figures" / "figS_teacher_judge.pdf")
plt.close(fig)
parts.append(r"""\begin{figure}[!ht]\centering
\includegraphics[width=\textwidth]{figures/figS_teacher_judge}
\caption{Teacher consensus against the judge panel on the 300 default replies (counts; shading
by share of the teacher row). Revealing the answer is dominated by one label; the judges rated
as actionable and at the expected level many replies the teachers did not.}
\end{figure}
""")
cf = rd("revision_human_confusion.csv")
cf = cf[cf.dimension.isin(["Revealing_of_the_Answer", "Actionability", "Expectation_Level", "Mistake_Location"])]
parts.append(table(cf, "Confusion counts, teacher consensus by panel label (all languages)", colspec="lllr"))
ch = rd("revision_e2_challenge.csv")
ch["who"] = ch.who.map(lambda m: TUT.get(m, m))
ch = ch[["who", "check", "detection_rate", "ci_low", "ci_high", "n", "detection_rate_all_selected"]]
parts.append(table(ch, "Controlled edits (E2): detection rate or mean absolute change, with bootstrap intervals",
                   colspec="llrrrrr"))
js = rd("revision_judge_sensitivity.csv")
js = js.pivot_table(index=["outcome", "effect"], columns="variant", values="estimate").reset_index()
parts.append(table(js, "Per-judge and leave-one-judge-out estimates", size=r"\tiny"))

# ------------------------------------------------------------------ S8 R5 plans
parts.append(r"\section{Plan-then-realise: structure, canonical messages, fidelity and ablation}")
st = rd("revision_r5_structure.csv").T.reset_index()
st.columns = ["quantity", "value"]
parts.append(table(st, "Structure of the 4,200 saved R5 outputs", colspec="lr"))
cfid = rd("revision_r5_canonical_fidelity.csv")
parts.append(table(cfid, "Does the canonical English message keep the student's answer? (cue-free messages; misses "
                         "read by hand)", fmt={c: "{:.0f}" for c in cfid.columns}))
fi = rd("revision_e3_fidelity.csv")
fi["who"] = fi.who.map(lambda m: TUT.get(m, m))
parts.append(table(fi, "Semantic plan-fidelity check (60 plan/reply pairs)"))
ab = rd("revision_e3_ablation.csv")
ab = ci(ab[ab.model == "ALL"])[["outcome", "contrast", "language", "Estimate [95\\% CI]", "p_holm"]]
parts.append(table(ab, "Plan-only realisation minus saved R5 (same plans; Gemini, Sarvam, Nova Lite)",
                   fmt={"p_holm": "{:.3f}"}))

# ------------------------------------------------------------------ S9 translation and items
parts.append(r"\section{Translation coverage, integrity and item-level sensitivity}")
tc = rd("revision_translation_coverage.csv")
parts.append(table(tc.T.reset_index().rename(columns={"index": "quantity"}),
                   "Translation review coverage by variety (field level; BN-MT: BEnQA machine translation)",
                   fmt={}, size=r"\tiny"))
ti = rd("revision_translation_integrity.csv")
parts.append(table(ti, "Integrity checks on transformed fields (numerals as tokens and as digit multisets; LaTeX "
                       "span counts)"))
trs = rd("revision_translation_reviewed_subset.csv")
parts.append(table(trs, "Language gaps on items whose every translated field was reviewed", size=r"\tiny"))
cl = rd("revision_cluster_sensitivity.csv")
cl = cl[["outcome", "effect", "estimate", "item_ci_low", "item_ci_high", "construct_ci_low", "construct_ci_high",
         "misconception_ci_low", "misconception_ci_high"]]
parts.append(table(cl, "Resampling items, constructs or misconceptions (percentile intervals for clusters)",
                   size=r"\tiny"))
nb = rd("revision_name_balance.csv")[["condition", "cue", "field", "n_prompts", "distinct", "min_count", "max_count"]]
parts.append(table(nb, "Name frequencies per cue and language (Block B, 150 items)", fmt={}))

# ------------------------------------------------------------------ S10 second revision
parts.append(r"""\section{Second revision: translation system, cue factors and further diagnostics}
Frozen in the second addendum of \texttt{docs/revision\_plan\_frozen.md} (13 October 2026) before
any call. E8 re-translates the English replies of R2-T with NLLB-200 distilled 1.3B; E9 separates
the school-background cue into single-factor cues in English (five cues collected together).
""")
e8 = rd("revision_e8_mt_sensitivity.csv")
e8 = ci(e8[(e8.model == "ALL") & e8.outcome.isin(["PQI_R", "PQI", "withheld", "math_ok"])])
parts.append(table(e8[["outcome", "contrast", "language", "Estimate [95\\% CI]", "p_holm"]],
                   "E8: R2-T with NLLB-200 instead of IndicTrans2", fmt={"p_holm": "{:.4f}"}, colspec="lllrr"))
e9 = rd("revision_e9_cue_factors.csv")
e9 = e9[e9.model.isin(["ALL", "sarvam_105b"])].copy()
e9["model"] = e9.model.map(TUT)
e9 = ci(e9)[["outcome", "contrast", "model", "mean_S-ENG", "mean_cue", "Estimate [95\\% CI]", "p_holm"]]
parts.append(table(e9, "E9: S-ENG minus each school cue (English messages; S-MED Hindi-medium, S-TYPE government "
                       "school, S-LOC village, S-VER all three)", fmt={"p_holm": "{:.4f}"}, colspec="lllrrrr"))
xd = rd("revision2_crossing_by_dimension.csv")
xd = xd.pivot_table(index="dimension", columns=["language", "effect"], values="estimate")
xd.columns = [f"{a} {b}" for a, b in xd.columns]
parts.append(table(xd.reset_index(), "Block X effects by rubric dimension (positive = worse for the student "
                                     "writing in $l$; inflated praise sign reversed)"))
cl2 = rd("revision2_canonical_loss_downstream.csv")
cl2 = cl2[["contrast", "canonical_answer", "item_language_pairs", "mean_R5", "estimate", "min_pair", "max_pair",
           "ci_low", "ci_high"]]
parts.append(table(cl2, "R5 relative to R2-T and R2-L on HI/BN items whose canonical message kept or lost the "
                        "student's answer (C0; intervals only with enough pairs)"))
lid = rd("revision2_script_lid.csv")
lid = lid[lid.condition.isin(["HI", "BN"])]
parts.append(table(lid, "Unicode-script language identification against the panel's Language\\_Match (HI/BN)",
                   colspec="llrrrrrr"))
cba = rd("revision2_components_by_arm.csv")
cba = cba.pivot_table(index="component", columns="arm", values="HI_BN").reset_index()
parts.append(table(cba, "Rubric components on Hindi and Bengali by arm"))
cs2 = rd("revision2_cost_sensitivity.csv")
cs2 = cs2[cs2.arm.isin(["R2", "R2L", "R2T", "R4", "R5"])].pivot_table(
    index=["input_price_x", "output_price_x"], columns="arm", values="total").reset_index()
parts.append(table(cs2, "Estimated cost per 1,000 replies (USD; mean tutor price, GPU translation included) under "
                        "price multipliers", fmt={c: "{:.2f}" for c in cs2.columns}))

parts.append(r"\end{document}")
OUT.write_text("\n".join(parts), encoding="utf-8")
print("written", OUT)
