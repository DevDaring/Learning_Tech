"""Paper figures (Section 10). Static, vector (PDF) plus PNG preview.

Colour: the validated categorical order of the dataviz reference palette, assigned to
entities in a fixed order (never cycled); more than three series in one panel also get
distinct marker shapes so identity is never carried by colour alone. One y-scale per
panel, recessive grid, legend for every multi-series panel.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LANG_ORDER = ["EN", "HI", "BN", "HI-EN", "BN-EN"]


def _style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def _save(fig, path_stem, provisional: bool):
    if provisional:
        fig.text(0.99, 0.005, "PROVISIONAL", ha="right", va="bottom", fontsize=7, color=INK2)
    fig.savefig(f"{path_stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{path_stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig1_design(path_stem, design: dict, provisional: bool):
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    w, h = 0.20, 0.15

    def box(x, y, title, sub):
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor=INK2, linewidth=0.8))
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=7, color=INK)
        ax.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center", fontsize=6, color=INK2)
        return (x, y)

    def arrow(p, q):
        ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="->", color=INK2, lw=0.7))

    n_prod, n_j = len(design["producers"]), len(design["judges"])
    [box(0.01, 0.62, "Eedi items", f"{design['items']['n_core']} misconception-tagged"),
           box(0.01, 0.12, "BEnQA maths", f"{design['items']['n_block_d']} native Bengali")]
    blocks = [box(0.28, 0.80, "A: language", "EN HI BN HI-EN BN-EN"),
              box(0.28, 0.57, "B: identity cue", "6 cues x EN/HI/BN"),
              box(0.28, 0.34, "C: decomposition", "EN in -> HI/BN out"),
              box(0.28, 0.11, "D: native control", "BN native / EN / BN-MT")]
    prod = box(0.55, 0.455, f"{n_prod} producer tutors", "temp 0.7, max 600 tokens")
    judge = box(0.79, 0.62, f"{n_j} blind judges", "14 dims -> PQI")
    repair = box(0.79, 0.29, "Repair arms R1-R5", "R5 plan-then-realise")
    for b_ in blocks[:3]:
        arrow((0.21, 0.62 + h / 2), (0.28, b_[1] + h / 2))
    arrow((0.21, 0.12 + h / 2), (0.28, blocks[3][1] + h / 2))
    for b_ in blocks:
        arrow((0.48, b_[1] + h / 2), (0.55, prod[1] + h / 2))
    arrow((0.75, prod[1] + h / 2), (0.79, judge[1] + h / 2))
    arrow((0.75, prod[1] + h / 2), (0.79, repair[1] + h / 2))
    arrow((0.89, repair[1] + h), (0.89, judge[1]))
    _save(fig, path_stem, provisional)


def fig2_profiles(agg_a: pd.DataFrame, tiers: dict, path_stem, provisional: bool):
    """Per-dimension score profile by language, one panel per producer tier."""
    from ..rubric import PQI_SCORES
    dims = list(PQI_SCORES)
    tier_names = list(dict.fromkeys(tiers.values()))
    fig, axes = plt.subplots(1, len(tier_names), figsize=(3.0 * len(tier_names), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(dims))
    for ax, tier in zip(axes, tier_names):
        sub = agg_a[agg_a.model.map(tiers) == tier]
        langs = [l for l in LANG_ORDER if l in sub.condition.unique()]
        for k, lang in enumerate(langs):
            s = sub[sub.condition == lang]
            means = [s[f"pqi_{d}"].mean() for d in dims]
            off = (k - (len(langs) - 1) / 2) * 0.13  # dimensions are categories: dots, no joining lines
            ax.plot(x + off, means, linestyle="none", color=SERIES[k], marker=MARKERS[k], markersize=4.5,
                    markeredgecolor="white", markeredgewidth=0.6, label=lang)
        ax.set_title(tier, fontsize=8, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels([d.replace("_", " ") for d in dims], rotation=60, ha="right", fontsize=6.5)
        _style(ax)
    axes[0].set_ylabel("Mean component score (0-1)", fontsize=8, color=INK2)
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=7, frameon=False, loc="upper center",
               ncol=5, bbox_to_anchor=(0.5, 1.04))
    _save(fig, path_stem, provisional)


def fig3_forest(gaps: pd.DataFrame, path_stem, provisional: bool):
    """All pooled gaps with 95% intervals; Holm-significant gaps drawn filled."""
    g = gaps.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(5.2, 0.22 * len(g) + 1.0))
    y = np.arange(len(g))[::-1]
    for yi, (_, r) in zip(y, g.iterrows()):
        ax.plot([r.ci_low, r.ci_high], [yi, yi], color=INK2, linewidth=1.0)
        filled = bool(r.get("significant_holm_05", False))
        ax.plot(r.estimate, yi, marker="o", markersize=4.5, color=SERIES[0],
                markerfacecolor=SERIES[0] if filled else "white")
    ax.axvline(0, color=INK2, linewidth=0.7, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(g.label, fontsize=7)
    ax.set_xlabel("PQI gap (advantaged - disadvantaged), 95% CI; filled = Holm p < .05", fontsize=8, color=INK2)
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    _save(fig, path_stem, provisional)


def fig4_translated(tj: pd.DataFrame, path_stem, provisional: bool):
    sub = tj[tj.dimension == "PQI"]
    if sub.empty:
        return
    judges = list(dict.fromkeys(sub.judge))
    langs = [l for l in list(dict.fromkeys(sub.language))]
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    w = 0.8 / max(1, len(judges))
    for k, j in enumerate(judges):
        s = sub[sub.judge == j].set_index("language").reindex(langs)
        xs = np.arange(len(langs)) + (k - (len(judges) - 1) / 2) * w
        ax.errorbar(xs, s.estimate, yerr=[s.estimate - s.ci_low, s.ci_high - s.estimate], fmt=MARKERS[k],
                    color=SERIES[k], markersize=4, linewidth=1.0, capsize=2, label=j)
    ax.axhline(0, color=INK2, linewidth=0.7, linestyle="--")
    ax.set_xticks(np.arange(len(langs)))
    ax.set_xticklabels(langs, fontsize=8)
    ax.set_ylabel("PQI(English translation) - PQI(original)\n> 0: judge harsher on the original language",
                  fontsize=7.5, color=INK2)
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    _save(fig, path_stem, provisional)


def fig5_repair(rep: pd.DataFrame, path_stem, provisional: bool):
    """Gap within arm vs absolute PQI, pooled over producers. Small multiples by language;
    every point is directly labelled with its arm (one series colour, so no colour key)."""
    if rep.empty:
        return
    langs = [l for l in LANG_ORDER if l in set(rep.language)]
    fig, axes = plt.subplots(1, len(langs), figsize=(2.4 * len(langs), 2.8), sharey=True, sharex=True)
    axes = np.atleast_1d(axes)
    for ax, lang in zip(axes, langs):
        s = rep[rep.language == lang]
        ax.scatter(s.abs_pqi, s.gap_within_arm, s=22, color=SERIES[0], edgecolors="white", linewidths=0.8, zorder=3)
        for _, r in s.iterrows():
            ax.annotate(r.arm, (r.abs_pqi, r.gap_within_arm), xytext=(3, 3), textcoords="offset points",
                        fontsize=6.5, color=INK)
        ax.axhline(0, color=INK2, linewidth=0.7, linestyle="--")
        ax.set_title(lang, fontsize=8, color=INK)
        _style(ax)
    axes[0].set_ylabel("Language gap within arm (EN - l)", fontsize=8, color=INK2)
    fig.supxlabel("Absolute PQI in the student's language", fontsize=8, color=INK2)
    _save(fig, path_stem, provisional)
