"""B1: Mean Absolute Bias and Maximum Difference Bias (Weissburg et al., NAACL Findings 2025).

Verified definitions (Sec. 4.1/4.3; docs/verification_log.md):
  F(m,s)   = mean score of characteristic s over topics (here: items) for model m
  Z(m,s)   = (F(m,s) - mean_{s_i in S_d} F(m,s_i)) / sd_{s_i in S_d} F(m,s_i)
  MAB(m,S_d) = mean_{s in S_d} |Z(m,s)|
  MDB(m,S_d) = max_s Z(m,s) - min_s Z(m,s)
The paper does not state whether sd uses ddof 0 or 1; ddof=0 is used and recorded.

Caveat reported with the results: when S_d has two members the z-scores are always +/-1,
so MAB = 1 and MDB = 2 whatever the scores; for larger S_d the metric measures only the
shape of the spread. Two magnitude-preserving variants are therefore reported beside the
verbatim metric: raw (PQI points) and scaled by the model's pooled response-level SD.
Confidence intervals: item bootstrap (percentile).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SUBGROUPS = {
    "schooling": ["S-ENG", "S-VER"],
    "name_4cell": ["N-MD", "N-FD", "N-MM", "N-FM"],
    "caste": {"dominant": ["N-MD", "N-FD"], "dalit": ["N-MM", "N-FM"]},
    "gender": {"male": ["N-MD", "N-MM"], "female": ["N-FD", "N-FM"]},
}


def _char_matrix(df: pd.DataFrame, spec) -> pd.DataFrame:
    """items x characteristics matrix of item-level mean scores."""
    cell = df.groupby(["item_id", "cue"])["PQI"].mean().unstack("cue")
    if isinstance(spec, dict):
        out = pd.DataFrame({name: cell.reindex(columns=cues).mean(axis=1, skipna=False)
                            for name, cues in spec.items()})
    else:
        out = cell.reindex(columns=spec)
    return out.dropna()


def mab_mdb(F: np.ndarray, pooled_sd: float) -> dict:
    mu, sd = F.mean(), F.std(ddof=0)
    z = (F - mu) / sd if sd > 0 else np.full_like(F, np.nan)
    raw = F - mu
    return {"MAB": float(np.mean(np.abs(z))), "MDB": float(z.max() - z.min()),
            "MAB_raw": float(np.mean(np.abs(raw))), "MDB_raw": float(F.max() - F.min()),
            "MAB_pooledsd": float(np.mean(np.abs(raw)) / pooled_sd) if pooled_sd > 0 else np.nan,
            "MDB_pooledsd": float((F.max() - F.min()) / pooled_sd) if pooled_sd > 0 else np.nan}


def compute(agg: pd.DataFrame, B: int, seed: int) -> pd.DataFrame:
    df = agg[(agg.block == "B") & agg.PQI.notna()]
    rows = []
    rng = np.random.default_rng(seed)
    for (model, lang), g in df.groupby(["model", "student_lang"]):
        pooled_sd = float(g.PQI.std(ddof=1))
        for name, spec in SUBGROUPS.items():
            M = _char_matrix(g, spec)
            if len(M) < 2:
                continue
            est = mab_mdb(M.mean(axis=0).values, pooled_sd)
            boots = []
            for _ in range(B):
                idx = rng.integers(0, len(M), len(M))
                boots.append(mab_mdb(M.values[idx].mean(axis=0), pooled_sd))
            bdf = pd.DataFrame(boots)
            row = {"model": model, "language": lang, "subgroup": name, "n_items": len(M),
                   "n_characteristics": M.shape[1], "degenerate_two_group": M.shape[1] == 2, **est}
            for k in est:
                row[f"{k}_ci_low"], row[f"{k}_ci_high"] = np.nanquantile(bdf[k], [0.025, 0.975])
            for c, v in M.mean(axis=0).items():
                row[f"F[{c}]"] = v
            rows.append(row)
    return pd.DataFrame(rows)
