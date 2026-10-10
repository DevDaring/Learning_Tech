"""Paper figures (IEEE TLT, vector PDF) from the result tables.

    PYTHONPATH=src python scripts/paper_figures.py <out_dir>

fig_language_gaps.pdf  language gap per tutor and pooled, one panel per language variety
fig_decomposition.pdf  Block C: generation vs input component, per rubric component
fig_repair.pdf         all eight repair arms on HI+BN: PQI-R vs the remaining language gap
Filled markers: Holm-adjusted p < .05. Error bars: 95% bootstrap-t intervals.
"""
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "tables"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
S1, S2 = "#2a78d6", "#eb6834"          # categorical slots 1 and 2 (validated palette)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "mathtext.fontset": "stix", "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42, "lines.linewidth": 1.0,
})
PRODUCERS = [("gemini_flash", "Gemini 3.8 Flash"), ("sarvam_105b", "Sarvam 105B"), ("kimi_k25", "Kimi K2.5"),
             ("nova_lite", "Nova Lite"), ("ministral_8b", "Ministral 3 8B"), ("nemotron_nano", "Nemotron 3 Nano")]
LANG = {"HI": "Hindi", "BN": "Bengali", "HI-EN": "Hinglish", "BN-EN": "Banglish"}


def rd(name):
    t = pd.read_csv(T / name, keep_default_na=False, na_values=[""])
    if "model" in t:
        t["model"] = t["model"].fillna("ALL")
    return t


def errdot(ax, x, y, lo, hi, sig, color, marker="o", size=4.2):
    ax.plot([lo, hi], [y, y], color=color, lw=1.0, solid_capstyle="round", zorder=2)
    ax.plot([x], [y], marker=marker, ms=size, mec=color, mew=1.0,
            mfc=color if sig else "white", zorder=3, ls="none")


# ---------------------------------------------------------------- Fig. language gaps
t = rd("table4_language_gaps.csv")
fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.85), sharey=True, sharex=True, layout="constrained")
rows = PRODUCERS + [("ALL", "Pooled")]
for ax, (code, name) in zip(axes, LANG.items()):
    for i, (m, label) in enumerate(rows):
        r = t[(t.language == code) & (t.model == m)].iloc[0]
        y = len(rows) - 1 - i - (0.4 if m == "ALL" else 0)
        errdot(ax, r.estimate, y, r.ci_low, r.ci_high, bool(r.significant_holm_05), INK if m == "ALL" else S1,
               marker="D" if m == "ALL" else "o")
    ax.axvline(0, color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.set_title(name, color=INK)
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
axes[0].set_yticks([len(rows) - 1 - i - (0.4 if m == "ALL" else 0) for i, (m, _) in enumerate(rows)])
axes[0].set_yticklabels([lab for _, lab in rows])
axes[0].get_yticklabels()[-1].set_fontweight("bold")
fig.supxlabel("Language gap: PQI(English) $-$ PQI(language)", fontsize=8)
fig.savefig(OUT / "fig_language_gaps.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Fig. decomposition
d = rd("extra_decomposition_components.csv")
OUTC = [("PQI", "PQI"), ("pqi_Mistake_Identification", "Mistake identification"),
        ("pqi_Mistake_Location", "Mistake location"), ("pqi_Providing_Guidance", "Guidance"),
        ("pqi_Actionability", "Actionability"), ("pqi_Revealing_of_the_Answer", "Not revealing answer"),
        ("pqi_Cognitive_Demand", "Cognitive demand"), ("pqi_Conceptual_Depth", "Conceptual depth"),
        ("pqi_Math_Correctness", "Mathematical correctness"), ("inflated", "Inflated praise"),
        ("lowered", "Lowered expectations")]
BAD = {"inflated", "lowered"}           # sign flipped so that positive = worse for the student in l
fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.6), sharey=True, sharex=True, layout="constrained")
for ax, l in zip(axes, ("HI", "BN")):
    for i, (o, _) in enumerate(OUTC):
        y = len(OUTC) - 1 - i
        for comp, col, mk, off in (("generation", S1, "o", 0.17), ("input", S2, "s", -0.17)):
            r = d[(d.language == l) & (d.outcome == o) & (d.component == comp)].iloc[0]
            s = -1 if o in BAD else 1
            lo, hi = sorted((s * r.ci_low, s * r.ci_high))
            errdot(ax, s * r.estimate, y + off, lo, hi, bool(r.significant_holm_05), col, mk, 3.8)
    ax.axvline(0, color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.axhline(len(OUTC) - 1.5, color=GRID, lw=0.6)
    ax.axhline(1.5, color=GRID, lw=0.6)
    ax.set_title({"HI": "Hindi", "BN": "Bengali"}[l], color=INK)
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
axes[0].set_yticks(range(len(OUTC)))
axes[0].set_yticklabels([lab for _, lab in OUTC][::-1])
h1, = axes[1].plot([], [], "o", color=S1, ms=4, label="Generation (EN in: EN out vs $l$ out)")
h2, = axes[1].plot([], [], "s", color=S2, ms=4, label="Input ($l$ out: EN in vs $l$ in)")
fig.legend(handles=[h1, h2], loc="outside upper center", ncol=2, frameon=False)
fig.supxlabel("Component of the language gap (positive = worse for the student writing in $l$)", fontsize=8)
fig.savefig(OUT / "fig_decomposition.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Fig. repair
# every arm on Hindi + Bengali (the languages all eight arms share), PQI-R (E1's primary outcome)
rp = rd("revision_e1_arms_pooled_HI_BN.csv")
rp = rp[rp.outcome == "PQI_R"]
NAMES = {"R0": "R0", "R1": "R1", "R2": "R2", "R2L": "R2-L", "R2T": "R2-T", "R3": "R3", "R4": "R4", "R5": "R5"}
OFF = {"R0": (5, 5), "R1": (-5, -5), "R2": (5, 5), "R2L": (-5, 5), "R2T": (5, -5), "R3": (5, 5),
       "R4": (-5, -5), "R5": (-5, -5)}
fig, ax = plt.subplots(figsize=(3.5, 2.6))
for _, r in rp.iterrows():
    col = S2 if r.arm in ("R2L", "R2T", "R5") else S1
    ax.plot([r.q_ci_low, r.q_ci_high], [r.gap] * 2, color=col, lw=1.0, zorder=2)
    ax.plot([r.quality_HI_BN] * 2, [r.gap_ci_low, r.gap_ci_high], color=col, lw=1.0, zorder=2)
    ax.plot([r.quality_HI_BN], [r.gap], "D" if r.arm in ("R2L", "R2T", "R5") else "o", ms=4.2, color=col, zorder=3)
    dx, dy = OFF[r.arm]
    ax.annotate(NAMES[r.arm], (r.quality_HI_BN, r.gap), textcoords="offset points", xytext=(dx, dy),
                ha="left" if dx > 0 else ("right" if dx < 0 else "center"), va="center", fontsize=7, color=INK)
ax.axhline(0, color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
ax.set_xlabel("PQI-R in Hindi and Bengali (pooled)")
ax.set_ylabel("Language gap within arm\nPQI-R(EN) $-$ PQI-R($l$)")
ax.grid(color=GRID, lw=0.5)
ax.set_axisbelow(True)
ax.set_xlim(0.66, 0.985)
fig.tight_layout()
fig.savefig(OUT / "fig_repair.pdf")
plt.close(fig)
print("written to", OUT)
