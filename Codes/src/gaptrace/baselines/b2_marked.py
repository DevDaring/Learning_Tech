"""B2: praise vs critique and Marked Words (Tan, Phalen & Demszky, LAK 2026), English Block B.

* Marked Words, as in their Sec. 3.2: log-odds ratio with an informative Dirichlet prior
  built from the pooled counts of the two corpora (Monroe et al. 2008), z-scores, keep
  |z| > 1.96 with at least 30 occurrences, computed per LLM and pooled. Names are masked
  first so the cue itself cannot surface as a marked word. Runs on every English Block B
  reply (no LLM cost).
* Praise / corrective counts: Tan & Phalen derive these qualitatively from the marked-word
  lists and do not count them. The counting operationalisation here is this study's own:
  each sentence is labelled PRAISE / CORRECTIVE / OTHER by the utility model (prompt
  prompts/praise_critique.txt). To limit cost it runs on the Block C item subset, sample 0.
  "Positive feedback bias" = more praise for the disadvantaged cue; "feedback withholding"
  = fewer corrective statements for the disadvantaged cue.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from ..names import mask_names

PAIRS = {"schooling": (["S-ENG"], ["S-VER"]), "caste": (["N-MD", "N-FD"], ["N-MM", "N-FM"]),
         "gender": (["N-MD", "N-MM"], ["N-FD", "N-FM"])}


def tokens(text: str) -> list[str]:
    masked, _ = mask_names(text)
    masked = masked.replace("[STUDENT]", " ")
    return re.findall(r"[a-z]+(?:'[a-z]+)?", masked.lower())


def log_odds_dirichlet(c1: Counter, c2: Counter) -> pd.DataFrame:
    """Monroe et al. (2008) log-odds with informative Dirichlet prior = pooled counts."""
    prior = c1 + c2
    a0 = sum(prior.values())
    n1, n2 = sum(c1.values()), sum(c2.values())
    rows = []
    for w, aw in prior.items():
        y1, y2 = c1.get(w, 0), c2.get(w, 0)
        l1 = np.log((y1 + aw) / (n1 + a0 - y1 - aw))
        l2 = np.log((y2 + aw) / (n2 + a0 - y2 - aw))
        var = 1.0 / (y1 + aw) + 1.0 / (y2 + aw)
        rows.append({"word": w, "count_1": y1, "count_2": y2, "total": y1 + y2,
                     "delta": l1 - l2, "z": (l1 - l2) / np.sqrt(var)})
    return pd.DataFrame(rows)


def marked_words(gens: pd.DataFrame, min_freq: int = 30, top: int = 20) -> pd.DataFrame:
    """gens: English Block B (and C0) replies with columns model, cue, text."""
    out = []
    for pair, (adv, dis) in PAIRS.items():
        for model in sorted(gens.model.unique()) + ["ALL"]:
            g = gens if model == "ALL" else gens[gens.model == model]
            c_dis = Counter(t for x in g[g.cue.isin(dis)].text for t in tokens(x))
            c_adv = Counter(t for x in g[g.cue.isin(adv)].text for t in tokens(x))
            if not c_dis or not c_adv:
                continue
            lo = log_odds_dirichlet(c_dis, c_adv)
            lo = lo[(lo.total >= min_freq) & (lo.z.abs() > 1.96) & (~lo.word.isin(ENGLISH_STOP_WORDS))]
            for direction, sub in (("marked_for_disadvantaged", lo[lo.z > 0].nlargest(top, "z")),
                                   ("marked_for_advantaged", lo[lo.z < 0].nsmallest(top, "z"))):
                for _, r in sub.iterrows():
                    out.append({"pair": pair, "model": model, "direction": direction, **r.to_dict()})
    return pd.DataFrame(out)


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.?!])\s+|\n+", (text or "").strip())
    return [p.strip() for p in parts if p.strip() and re.search(r"[A-Za-z]", p)]
