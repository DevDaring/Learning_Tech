"""Analyse round-2 blinded human ratings (R2T vs R5, R2T vs R2L).

Reads:
  review/human_returned/November_Annotated/Rater_{A,B}_pack.xlsx
  review/repair_rating_key_round2.csv

Writes:
  results/tables/round2_*.csv   (aggregated tables for supplement)
  results/round2_summary.txt    (headline numbers for the paper)

No rater names are stored or printed.
"""
import math, sys, pathlib
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
RET = ROOT / "review" / "human_returned" / "November_Annotated"
KEY_PATH = ROOT / "review" / "repair_rating_key_round2.csv"
OUT = ROOT / "results" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

RATING_DIMS = [
    "Mistake_Identification", "Mistake_Location",
    "Revealing_of_the_Answer", "Providing_Guidance",
    "Math_Correctness", "Language_Naturalness",
]

SHORT = {
    "Mistake_Identification": "Identif.",
    "Mistake_Location": "Location",
    "Revealing_of_the_Answer": "Reveal",
    "Providing_Guidance": "Guidance",
    "Math_Correctness": "Math",
    "Language_Naturalness": "Natural.",
}

ORDINAL = {
    "Mistake_Identification":  {"Yes": 2, "To some extent": 1, "No": 0},
    "Mistake_Location":        {"Yes": 2, "To some extent": 1, "No": 0},
    "Revealing_of_the_Answer": {"Yes (correct)": 1, "Yes (incorrect)": 1, "No": 0},
    "Providing_Guidance":      {"Yes": 2, "To some extent": 1, "No": 0},
    "Math_Correctness":        {"Yes": 1, "No": 0},
    "Language_Naturalness":    {"Natural": 2, "Understandable but unnatural": 1, "Hard to understand": 0},
}

# "Good" direction for each dimension: higher ordinal = better, except Revealing
GOOD_HIGH = {
    "Mistake_Identification": True,
    "Mistake_Location": True,
    "Revealing_of_the_Answer": False,  # not revealing is better pedagogically
    "Providing_Guidance": True,
    "Math_Correctness": True,
    "Language_Naturalness": True,
}

# ── load key ──
key = pd.read_csv(KEY_PATH)

# ── load ratings ──
dfs = []
for wb_name in ["Rater_A_pack", "Rater_B_pack"]:
    path = RET / f"{wb_name}.xlsx"
    r = pd.read_excel(path, sheet_name="Ratings")
    r["workbook"] = wb_name
    dfs.append(r)
ratings = pd.concat(dfs, ignore_index=True)

# ── load pair preferences ──
pdfs = []
for wb_name in ["Rater_A_pack", "Rater_B_pack"]:
    path = RET / f"{wb_name}.xlsx"
    p = pd.read_excel(path, sheet_name="Pairs")
    p["workbook"] = wb_name
    pdfs.append(p)
pairs_raw = pd.concat(pdfs, ignore_index=True)
pairs_raw.rename(columns={"Which reply would help the student learn more?": "pref_letter"}, inplace=True)

# ── unblind ratings ──
merged = ratings.merge(key, on=["workbook", "row_id", "pair"], how="left", suffixes=("", "_key"))
assert merged.arm.notna().all(), "unblinding failed: some rows have no arm"

# ── unblind pair preferences ──
pair_key = key.groupby(["workbook", "pair"]).apply(
    lambda g: pd.Series({
        "arm_A": g.loc[g.part_label == "A", "arm"].values[0] if "part_label" in g else
                 g.loc[g.row_id.str.endswith("A"), "arm"].values[0],
        "arm_B": g.loc[g.part_label == "B", "arm"].values[0] if "part_label" in g else
                 g.loc[g.row_id.str.endswith("B"), "arm"].values[0],
        "item_id": g.item_id.iloc[0],
        "model": g.model.iloc[0],
        "language": g.language_key.iloc[0] if "language_key" in g else g.language.iloc[0],
        "part": g.part.iloc[0],
    }), include_groups=False
).reset_index()

pairs_merged = pairs_raw.merge(pair_key, on=["workbook", "pair"], how="left")

def resolve_pref(row):
    if pd.isna(row.pref_letter) or str(row.pref_letter).strip() == "":
        return None
    pl = str(row.pref_letter).strip()
    if pl == "A":
        return row.arm_A
    elif pl == "B":
        return row.arm_B
    elif pl.lower().startswith("no"):
        return "tie"
    return pl

pairs_merged["preferred_arm"] = pairs_merged.apply(resolve_pref, axis=1)

# ══════════════════════════════════════════════════════════════
# 1. PER-DIMENSION LABEL DISTRIBUTIONS BY ARM
# ══════════════════════════════════════════════════════════════
print("=" * 70)
print("1. Label distributions by arm and dimension")
print("=" * 70)

rows_dist = []
for dim in RATING_DIMS:
    for arm in sorted(merged.arm.unique()):
        sub = merged[merged.arm == arm][dim].dropna()
        total = len(sub)
        for label in ORDINAL[dim]:
            cnt = (sub == label).sum()
            rows_dist.append({"dimension": dim, "arm": arm, "label": label,
                              "count": cnt, "share": cnt / total if total else math.nan})

dist = pd.DataFrame(rows_dist)
dist.to_csv(OUT / "round2_label_distributions.csv", index=False)

pivot = dist.pivot_table(index=["dimension", "label"], columns="arm", values="share")
print(pivot.round(3).to_string())
print()

# ══════════════════════════════════════════════════════════════
# 2. ARM-LEVEL ORDINAL MEANS AND MANN-WHITNEY U TESTS
# ══════════════════════════════════════════════════════════════
print("=" * 70)
print("2. Ordinal means and Mann-Whitney U (arm comparisons)")
print("=" * 70)

def ordinal_score(dim, series):
    return series.map(ORDINAL[dim])

rows_cmp = []
for part, (arm_a, arm_b) in [(1, ("R2T", "R5")), (2, ("R2T", "R2L"))]:
    sub = merged[merged.part == part] if "part" in merged else merged
    # for part 2 we may have two raters: pool their ratings
    sa = sub[sub.arm == arm_a]
    sb = sub[sub.arm == arm_b]
    for dim in RATING_DIMS:
        xa = ordinal_score(dim, sa[dim]).dropna()
        xb = ordinal_score(dim, sb[dim]).dropna()
        if len(xa) < 2 or len(xb) < 2:
            continue
        u_stat, p_val = sp_stats.mannwhitneyu(xa, xb, alternative="two-sided")
        rows_cmp.append({
            "comparison": f"{arm_a} vs {arm_b}",
            "part": part,
            "dimension": dim,
            "n_a": len(xa), "mean_a": xa.mean(),
            "n_b": len(xb), "mean_b": xb.mean(),
            "diff": xa.mean() - xb.mean(),
            "U": u_stat, "p": p_val,
        })

cmp = pd.DataFrame(rows_cmp)
cmp.to_csv(OUT / "round2_arm_comparisons.csv", index=False)
print(cmp[["comparison", "dimension", "n_a", "mean_a", "n_b", "mean_b", "diff", "p"]].round(4).to_string(index=False))
print()

# ══════════════════════════════════════════════════════════════
# 3. "YES" / "BEST" RATE COMPARISON (binary simplification)
# ══════════════════════════════════════════════════════════════
print("=" * 70)
print("3. Binary 'good' rates by arm")
print("=" * 70)

GOOD_LABELS = {
    "Mistake_Identification": ["Yes"],
    "Mistake_Location": ["Yes"],
    "Revealing_of_the_Answer": ["No"],
    "Providing_Guidance": ["Yes"],
    "Math_Correctness": ["Yes"],
    "Language_Naturalness": ["Natural"],
}

rows_bin = []
for part, (arm_a, arm_b) in [(1, ("R2T", "R5")), (2, ("R2T", "R2L"))]:
    for dim in RATING_DIMS:
        for arm in [arm_a, arm_b]:
            # row_ids repeat across workbooks, so select by part, not row_id
            sub = merged[(merged.arm == arm) & (merged.part == part)]
            vals = sub[dim].dropna()
            good = vals.isin(GOOD_LABELS[dim]).sum()
            rows_bin.append({"comparison": f"{arm_a} vs {arm_b}", "part": part,
                             "arm": arm, "dimension": dim,
                             "n": len(vals), "n_good": int(good),
                             "good_rate": good / len(vals) if len(vals) else math.nan})

bindf = pd.DataFrame(rows_bin)
bindf.to_csv(OUT / "round2_binary_good_rates.csv", index=False)
piv = bindf.pivot_table(index=["comparison", "dimension"], columns="arm", values="good_rate")
print(piv.round(3).to_string())
print()

# ══════════════════════════════════════════════════════════════
# 4. PAIR PREFERENCES
# ══════════════════════════════════════════════════════════════
print("=" * 70)
print("4. Pair preferences")
print("=" * 70)

for part, label in [(1, "R2T vs R5"), (2, "R2T vs R2L")]:
    pm = pairs_merged[pairs_merged.part == part]
    total = len(pm)
    counts = pm.preferred_arm.value_counts()
    print(f"\n{label} ({total} pairs):")
    for arm, cnt in sorted(counts.items()):
        print(f"  {arm}: {cnt} ({cnt/total*100:.1f}%)")

    # Sign test (excluding ties)
    arms_in_play = [a for a in counts.index if a != "tie"]
    if len(arms_in_play) == 2:
        a1, a2 = sorted(arms_in_play)
        n1 = counts.get(a1, 0)
        n2 = counts.get(a2, 0)
        n_ties = counts.get("tie", 0)
        n_total_notie = n1 + n2
        p_sign = sp_stats.binomtest(max(n1, n2), n_total_notie, 0.5).pvalue
        print(f"  Sign test (excl {n_ties} ties): p = {p_sign:.4f}")
print()

# ══════════════════════════════════════════════════════════════
# 5. INTER-RATER AGREEMENT on Part 2 (both raters rated same items)
# ══════════════════════════════════════════════════════════════
print("=" * 70)
print("5. Inter-rater agreement (Part 2, double-rated items)")
print("=" * 70)

ka = key[key.part == 2]
p2a_pairs = set(ka.loc[ka.workbook == "Rater_A_pack", "pair"])
p2b_pairs = set(ka.loc[ka.workbook == "Rater_B_pack", "pair"])
p2a = merged[(merged.workbook == "Rater_A_pack") & merged.pair.isin(p2a_pairs)]
p2b = merged[(merged.workbook == "Rater_B_pack") & merged.pair.isin(p2b_pairs)]

a_map = {}
for _, row in p2a.iterrows():
    hits = ka[(ka.workbook == "Rater_A_pack") & (ka.row_id == row.row_id)]
    if hits.empty:
        continue
    krow = hits.iloc[0]
    a_map[(krow.item_id, krow.model, krow.language, krow.arm)] = row

b_map = {}
for _, row in p2b.iterrows():
    hits = ka[(ka.workbook == "Rater_B_pack") & (ka.row_id == row.row_id)]
    if hits.empty:
        continue
    krow = hits.iloc[0]
    b_map[(krow.item_id, krow.model, krow.language, krow.arm)] = row

shared_keys = set(a_map) & set(b_map)
print(f"Shared reply slots: {len(shared_keys)}")

agree_rows = []
for dim in RATING_DIMS:
    a_vals, b_vals = [], []
    for k in sorted(shared_keys):
        av = a_map[k][dim]
        bv = b_map[k][dim]
        if pd.notna(av) and pd.notna(bv):
            a_vals.append(av)
            b_vals.append(bv)

    n = len(a_vals)
    exact = sum(1 for a, b in zip(a_vals, b_vals) if a == b)
    pct = exact / n if n else math.nan

    # Cohen's kappa
    labels = sorted(ORDINAL[dim].keys())
    a_ord = [ORDINAL[dim].get(v, -1) for v in a_vals]
    b_ord = [ORDINAL[dim].get(v, -1) for v in b_vals]
    if n >= 5:
        mat = np.zeros((len(labels), len(labels)), dtype=int)
        for ao, bo in zip(a_ord, b_ord):
            if ao >= 0 and bo >= 0:
                mat[ao, bo] += 1
        pe = 0
        for i in range(len(labels)):
            pe += mat[i, :].sum() * mat[:, i].sum()
        pe /= (n * n) if n else 1
        po = exact / n
        kappa = (po - pe) / (1 - pe) if pe < 1 else math.nan
    else:
        kappa = math.nan

    agree_rows.append({"dimension": dim, "n": n, "exact_agreement": pct, "kappa": kappa})
    print(f"  {dim}: {exact}/{n} = {pct:.2%}, kappa = {kappa:.3f}")

agree_df = pd.DataFrame(agree_rows)
agree_df.to_csv(OUT / "round2_interrater_agreement.csv", index=False)

# Pair-preference agreement
pa_prefs = pairs_merged[pairs_merged.part == 2].copy()
# For each (item_id, model, language) that appears in both workbooks
pref_a = pa_prefs[pa_prefs.workbook == "Rater_A_pack"].set_index(["item_id", "model"])
pref_b = pa_prefs[pa_prefs.workbook == "Rater_B_pack"].set_index(["item_id", "model"])
shared_pref = pref_a.index.intersection(pref_b.index)
n_sp = len(shared_pref)
agree_pref = sum(1 for k in shared_pref if pref_a.loc[k, "preferred_arm"] == pref_b.loc[k, "preferred_arm"])
print(f"\n  Pair preference agreement: {agree_pref}/{n_sp} = {agree_pref/n_sp:.2%}" if n_sp else "  No shared pairs for preference")

print()

# ══════════════════════════════════════════════════════════════
# 6. SUMMARY FOR PAPER
# ══════════════════════════════════════════════════════════════
print("=" * 70)
print("6. Summary for paper")
print("=" * 70)

# R2T vs R5 pair preference
p1_prefs = pairs_merged[pairs_merged.part == 1]
r2t_wins_p1 = (p1_prefs.preferred_arm == "R2T").sum()
r5_wins_p1 = (p1_prefs.preferred_arm == "R5").sum()
tie_p1 = (p1_prefs.preferred_arm == "tie").sum()
total_p1 = len(p1_prefs)
print(f"R2T vs R5: R2T preferred in {r2t_wins_p1}/{total_p1} pairs, R5 in {r5_wins_p1}, ties {tie_p1}")

# R2T vs R2L pair preference
p2_prefs = pairs_merged[pairs_merged.part == 2]
r2t_wins_p2 = (p2_prefs.preferred_arm == "R2T").sum()
r2l_wins_p2 = (p2_prefs.preferred_arm == "R2L").sum()
tie_p2 = (p2_prefs.preferred_arm == "tie").sum()
total_p2 = len(p2_prefs)
print(f"R2T vs R2L: R2T preferred in {r2t_wins_p2}/{total_p2} pairs, R2L in {r2l_wins_p2}, ties {tie_p2}")

# Sign tests
if r2t_wins_p1 + r5_wins_p1 > 0:
    p_sign_p1 = sp_stats.binomtest(r2t_wins_p1, r2t_wins_p1 + r5_wins_p1, 0.5).pvalue
    print(f"  R2T vs R5 sign test (excl ties): p = {p_sign_p1:.4f}")
if r2t_wins_p2 + r2l_wins_p2 > 0:
    p_sign_p2 = sp_stats.binomtest(max(r2t_wins_p2, r2l_wins_p2), r2t_wins_p2 + r2l_wins_p2, 0.5).pvalue
    print(f"  R2T vs R2L sign test (excl ties): p = {p_sign_p2:.4f}")

# Write summary file
with open(ROOT / "results" / "round2_summary.txt", "w") as f:
    f.write("Round 2 Human Rating Summary\n")
    f.write("=" * 40 + "\n\n")
    f.write(f"Rater A: 96 replies (48 pairs: 24 R2T-vs-R5, 24 R2T-vs-R2L)\n")
    f.write(f"Rater B: 48 replies (24 pairs: 24 R2T-vs-R2L)\n")
    f.write(f"Double-rated reply slots (Part 2): {len(shared_keys)}\n\n")
    f.write(f"R2T vs R5 preference: R2T {r2t_wins_p1}, R5 {r5_wins_p1}, tie {tie_p1} (n={total_p1})\n")
    f.write(f"R2T vs R2L preference: R2T {r2t_wins_p2}, R2L {r2l_wins_p2}, tie {tie_p2} (n={total_p2})\n\n")
    f.write("Inter-rater agreement (Part 2):\n")
    for _, row in agree_df.iterrows():
        f.write(f"  {row.dimension}: {row.exact_agreement:.2%} exact, kappa={row.kappa:.3f}\n")
    f.write(f"\n  Pair preference agreement: {agree_pref}/{n_sp} = {agree_pref/n_sp:.2%}\n" if n_sp else "")

print("\nDone. Tables written to results/tables/round2_*.csv")
