"""B3: semantic shift under a cue swap (Du, Borchers & Cukurova, arXiv 2511.08225 v2).

Their test: mean embedding distance between paired outputs (cosine distance 1 - cos)
against a noise baseline from re-submitting unchanged inputs, with a permutation test.
Here (open model instead of their text-embedding-3-large, as the specification requires):
  cross(i,m)  = mean cosine distance between replies to cue A and cue B (all sample pairs)
  noise(i,m)  = mean distance between the two samples of the same condition (A and B)
  shift(i,m)  = cross - noise
for each item i and producer m, with names masked before embedding. Pooled shift is
tested two ways: item bootstrap CI, and a sign-flip permutation test on item-level shifts
(5,000 permutations, as in Du et al.).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

SWAPS = {"caste": [("N-MD", "N-MM"), ("N-FD", "N-FM")], "gender": [("N-MD", "N-FD"), ("N-MM", "N-FM")]}


def cosdist(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def item_shifts(emb: dict[str, np.ndarray], gens: pd.DataFrame) -> pd.DataFrame:
    """gens: Block B rows with target_id, item_id, model, student_lang, cue, sample_index."""
    rows = []
    for (item, model, lang), g in gens.groupby(["item_id", "model", "student_lang"]):
        by_cue = {c: [emb[t] for t in sub.sort_values("sample_index").target_id if t in emb]
                  for c, sub in g.groupby("cue")}
        for swap, pairs in SWAPS.items():
            for a, b in pairs:
                A, Bv = by_cue.get(a, []), by_cue.get(b, [])
                if not A or not Bv:
                    continue
                cross = np.mean([cosdist(x, y) for x, y in itertools.product(A, Bv)])
                noise_parts = [cosdist(*p) for V in (A, Bv) for p in itertools.combinations(V, 2)]
                noise = float(np.mean(noise_parts)) if noise_parts else np.nan
                rows.append({"item_id": item, "model": model, "language": lang, "swap": swap,
                             "pair": f"{a}|{b}", "cross": cross, "noise": noise, "shift": cross - noise})
    return pd.DataFrame(rows)


def sign_flip_test(x: np.ndarray, n_perm: int, seed: int) -> float:
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan
    rng = np.random.default_rng(seed)
    obs = abs(x.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(x)))
    perm = np.abs((signs * x).mean(axis=1))
    return float((np.sum(perm >= obs) + 1) / (n_perm + 1))
