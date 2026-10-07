"""The 14-dimension rubric (Section 6.1), label normalisation, judge aggregation and PQI."""
from __future__ import annotations

import math
import re
from statistics import median

# Ordered from best/highest to worst/lowest for ordinal dimensions.
DIMENSIONS: dict[str, list[str]] = {
    "Mistake_Identification": ["Yes", "To some extent", "No"],
    "Mistake_Location": ["Yes", "To some extent", "No"],
    "Revealing_of_the_Answer": ["Yes (correct)", "Yes (incorrect)", "No"],
    "Providing_Guidance": ["Yes", "To some extent", "No"],
    "Actionability": ["Yes", "To some extent", "No"],
    "Coherence": ["Yes", "To some extent", "No"],
    "Tutor_Tone": ["Encouraging", "Neutral", "Offensive"],
    "Humanlikeness": ["Yes", "To some extent", "No"],
    "Cognitive_Demand": ["High", "Medium", "Low", "None"],
    "Conceptual_Depth": ["Conceptual", "Procedural only", "None"],
    "Praise_Calibration": ["Calibrated", "Inflated", "None", "Discouraging"],
    "Expectation_Level": ["At level", "Simplified below", "Condescending"],
    "Math_Correctness": ["Yes", "No"],
    "Language_Match": ["Yes", "Partly", "No"],
}
MRBENCH_DIMS = list(DIMENSIONS)[:8]
ADDED_DIMS = list(DIMENSIONS)[8:]

# Aggregation (Section 5): median for ordinal, majority for categorical.
CATEGORICAL = {"Revealing_of_the_Answer", "Praise_Calibration"}
ORDINAL = [d for d in DIMENSIONS if d not in CATEGORICAL]

# Krippendorff metric per dimension.
ALPHA_LEVEL = {d: ("nominal" if d in CATEGORICAL else "ordinal") for d in DIMENSIONS}

_ALIASES = {
    "yes (and the answer is correct)": "Yes (correct)",
    "yes (and the revealed answer is correct)": "Yes (correct)",
    "yes (but the answer is incorrect)": "Yes (incorrect)",
    "yes (but the revealed answer is incorrect)": "Yes (incorrect)",
    "yes, correct": "Yes (correct)", "yes - correct": "Yes (correct)",
    "yes, incorrect": "Yes (incorrect)", "yes - incorrect": "Yes (incorrect)",
    "simplified below what the question needs": "Simplified below",
    "procedural": "Procedural only",
    "inflated (praise for a wrong answer with no substance)": "Inflated",
    "somewhat": "To some extent", "to some extent.": "To some extent",
    "partial": "Partly", "partially": "Partly",
}


def norm_label(dim: str, value) -> str | None:
    """Map a judge's label onto the allowed set; None if it cannot be mapped."""
    if value is None:
        return None
    v = str(value).strip().strip(".").strip()
    allowed = DIMENSIONS[dim]
    for a in allowed:
        if v.lower() == a.lower():
            return a
    low = re.sub(r"\s+", " ", v.lower())
    if low in _ALIASES and _ALIASES[low] in allowed:
        return _ALIASES[low]
    # "Yes" alone for the revealing dimension is ambiguous: do not guess.
    for a in sorted(allowed, key=len, reverse=True):
        if dim != "Revealing_of_the_Answer" and low.startswith(a.lower()):
            return a
    return None


def normalise_scores(obj: dict, dims: list[str] | None = None) -> tuple[dict, list[str]]:
    """Return ({dim: label|None}, [dims that were missing or invalid])."""
    dims = dims or list(DIMENSIONS)
    keymap = {re.sub(r"[\s_]", "", k).lower(): k for k in obj}
    out, bad = {}, []
    for d in dims:
        k = keymap.get(re.sub(r"[\s_]", "", d).lower())
        lab = norm_label(d, obj.get(k)) if k else None
        out[d] = lab
        if lab is None:
            bad.append(d)
    return out, bad


def ordinal_index(dim: str, label: str | None) -> float:
    """0 = best/highest category. NaN for missing."""
    if label is None:
        return math.nan
    return float(DIMENSIONS[dim].index(label))


def aggregate(dim: str, labels_by_judge: dict[str, str | None], primary: str) -> str | None:
    """Median of three (ordinal) or majority (categorical); the primary judge breaks ties.

    With two valid labels that disagree, the primary judge's label is used if present;
    otherwise the result is missing (no arbitrary tie-break). With one valid label, that
    label is used only if it is the primary judge's.
    """
    valid = {j: l for j, l in labels_by_judge.items() if l is not None}
    if not valid:
        return None
    if len(valid) == 1:
        return valid.get(primary)
    vals = list(valid.values())
    if dim in CATEGORICAL:
        counts = {v: vals.count(v) for v in set(vals)}
        top = max(counts.values())
        winners = [v for v, c in counts.items() if c == top]
        if len(winners) == 1:
            return winners[0]
        return valid.get(primary) if valid.get(primary) in winners else None
    idx = sorted(DIMENSIONS[dim].index(v) for v in vals)
    if len(idx) % 2 == 1:
        return DIMENSIONS[dim][int(median(idx))]
    if idx[0] == idx[1]:
        return DIMENSIONS[dim][idx[0]]
    return valid.get(primary)


# ---------------------------------------------------------------- PQI (Section 6.5)

PQI_SCORES = {
    "Mistake_Identification": {"Yes": 1, "To some extent": 0.5, "No": 0},
    "Mistake_Location": {"Yes": 1, "To some extent": 0.5, "No": 0},
    "Providing_Guidance": {"Yes": 1, "To some extent": 0.5, "No": 0},
    "Actionability": {"Yes": 1, "To some extent": 0.5, "No": 0},
    "Revealing_of_the_Answer": {"No": 1, "Yes (correct)": 0, "Yes (incorrect)": 0},
    "Cognitive_Demand": {"High": 1, "Medium": 0.5, "Low": 0, "None": 0},
    "Conceptual_Depth": {"Conceptual": 1, "Procedural only": 0.5, "None": 0},
    "Math_Correctness": {"Yes": 1, "No": 0},
}


# PQI components with adequate inter-judge reliability in the pre-pilot (alpha 0.45-0.70 in
# both languages). PQI-R, their mean, is the robustness score reported beside the specified
# PQI (docs/deviations.md D41).
PQI_RELIABLE = ["Mistake_Identification", "Mistake_Location", "Revealing_of_the_Answer",
                "Providing_Guidance", "Math_Correctness"]


def pqi_reliable(labels: dict[str, str | None]) -> float:
    vals = []
    for d in PQI_RELIABLE:
        if labels.get(d) is None:
            return math.nan
        vals.append(PQI_SCORES[d][labels[d]])
    return sum(vals) / len(vals)


def pqi(labels: dict[str, str | None]) -> float:
    """Mean of the eight components; NaN if any component is missing (never imputed)."""
    vals = []
    for d, m in PQI_SCORES.items():
        lab = labels.get(d)
        if lab is None:
            return math.nan
        vals.append(m[lab])
    return sum(vals) / len(vals)


def component_scores(labels: dict[str, str | None]) -> dict[str, float]:
    return {f"pqi_{d}": (PQI_SCORES[d][labels[d]] if labels.get(d) is not None else math.nan)
            for d in PQI_SCORES}
