"""Statistics used across the analysis (Section 6.6).

Paired estimands are computed at the item level: responses are first averaged within an
item x producer x condition cell (over samples), the paired difference is taken within
item x producer, and, for pooled estimates, averaged over producers within item. Items
are the resampling unit of the bootstrap, so dependence among responses to the same item
is respected.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def cell_means(df: pd.DataFrame, value: str, by: list[str]) -> pd.DataFrame:
    return df.groupby(by, dropna=False)[value].mean().reset_index()


def paired_diff(df: pd.DataFrame, value: str, cond_col: str, a: list[str] | str, b: list[str] | str,
                unit: list[str]) -> pd.DataFrame:
    """Per-unit difference mean(value | cond in a) - mean(value | cond in b).

    ``a``/``b`` may list several conditions (e.g. both dominant-caste name cues); each
    condition's cell mean is computed first, then the conditions are averaged, so every
    condition gets equal weight. Units missing either side are dropped (and counted by
    the caller via the returned frame size)."""
    a = [a] if isinstance(a, str) else list(a)
    b = [b] if isinstance(b, str) else list(b)
    sub = df[df[cond_col].isin(a + b)]
    cm = sub.groupby(unit + [cond_col], dropna=False)[value].mean().unstack(cond_col)
    for c in a + b:
        if c not in cm.columns:
            cm[c] = np.nan
    out = pd.DataFrame(index=cm.index)
    out["a"] = cm[a].mean(axis=1, skipna=False)
    out["b"] = cm[b].mean(axis=1, skipna=False)
    out["diff"] = out["a"] - out["b"]
    return out.dropna(subset=["diff"]).reset_index()


def sign_flip_p(x: np.ndarray, B: int, seed: int) -> float:
    """Studentized paired randomisation test of mean(x) = 0.

    Each item-level difference keeps or flips its sign, and the statistic is t = mean/se
    recomputed on every flipped sample. This is exact under the sharp null (conditions
    exchangeable within item) and, being studentized, asymptotically valid when only the
    means are equal (Janssen 1997; Chung & Romano 2013). p = (#{|t*| >= |t|} + 1) / (B + 1)."""
    n = len(x)
    rng = np.random.default_rng(seed + 1)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(B, n))
    y = signs * x
    m = y.mean(axis=1)
    se = y.std(axis=1, ddof=1) / math.sqrt(n)
    se0 = x.std(ddof=1) / math.sqrt(n)
    if se0 == 0:
        return 1.0 if x.mean() == 0 else 0.0
    t_obs = abs(x.mean()) / se0
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, np.abs(m) / se, np.where(np.abs(m) > 0, np.inf, 0.0))
    return float((np.sum(t >= t_obs - 1e-12) + 1) / (B + 1))


def bootstrap_mean(values_by_item: pd.Series, B: int, seed: int, ci: float = 0.95) -> dict:
    """Inference for the mean of item-level paired differences.

    * p-value: max(bootstrap-t p, studentized sign-flip p). Each alone failed somewhere
      in simulation (percentile bootstrap: 0.10 at n=10; bootstrap-t dropping zero-spread
      resamples: 0.16 for sparse differences; sign-flip: 0.084 for skewed n=10). The
      maximum stayed <= 0.06 for normal, skewed, sparse, binary and PQI-like differences at
      n = 10-100 with no loss of power (docs/deviations.md D9). Both components are reported.
    * interval: studentized bootstrap (bootstrap-t), keeping zero-spread resamples as
      t* = +/-inf. If that makes an endpoint infinite (very sparse data) the percentile
      interval is used instead, and ``ci_method`` says which was used.
    * d_z: mean / SD of the item-level differences."""
    x = np.asarray(values_by_item, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    nan = math.nan
    if n < 2:
        return {"n_items": n, "estimate": float(x.mean()) if n else nan, "ci_low": nan, "ci_high": nan,
                "p_value": nan, "d_z": nan, "pct_ci_low": nan, "pct_ci_high": nan, "ci_method": "none"}
    rng = np.random.default_rng(seed)
    xb = x[rng.integers(0, n, size=(B, n))]
    means = xb.mean(axis=1)
    ses = xb.std(axis=1, ddof=1) / math.sqrt(n)
    est = float(x.mean())
    sd = float(x.std(ddof=1))
    se = sd / math.sqrt(n)
    a = (1 - ci) / 2
    plo, phi = (float(v) for v in np.quantile(means, [a, 1 - a]))
    out = {"n_items": n, "estimate": est, "d_z": est / sd if sd > 0 else nan,
           "pct_ci_low": plo, "pct_ci_high": phi}
    if se == 0:
        p = 1.0 if est == 0 else 0.0
        return {**out, "p_value": p, "p_bootstrap_t": p, "p_signflip": p, "ci_low": est, "ci_high": est,
                "ci_method": "degenerate"}
    dev = means - est
    with np.errstate(divide="ignore", invalid="ignore"):
        # zero-spread resamples are kept as the most extreme t* (never dropped)
        t = np.where(ses > 0, dev / np.where(ses > 0, ses, 1.0), np.where(dev != 0, np.sign(dev) * np.inf, 0.0))
    p_bt = float((np.sum(np.abs(t) >= abs(est / se)) + 1) / (B + 1))
    p_sf = sign_flip_p(x, B, seed)
    out.update({"p_value": max(p_bt, p_sf), "p_bootstrap_t": p_bt, "p_signflip": p_sf})
    qlo, qhi = np.quantile(t, [a, 1 - a])
    if np.isfinite(qlo) and np.isfinite(qhi):
        return {**out, "ci_low": est - qhi * se, "ci_high": est - qlo * se, "ci_method": "bootstrap-t"}
    return {**out, "ci_low": plo, "ci_high": phi, "ci_method": "percentile (bootstrap-t unbounded)"}


def _response_sd(sub: pd.DataFrame, value: str, cond_col: str, a, b) -> float:
    """Pooled response-level SD of the two sides, so effect sizes are comparable between
    pooled and per-producer rows: sqrt((var_a + var_b) / 2)."""
    a = [a] if isinstance(a, str) else list(a)
    b = [b] if isinstance(b, str) else list(b)
    va = sub.loc[sub[cond_col].isin(a), value].var(ddof=1)
    vb = sub.loc[sub[cond_col].isin(b), value].var(ddof=1)
    return float(np.sqrt((va + vb) / 2)) if np.isfinite(va) and np.isfinite(vb) else math.nan


def gap(df: pd.DataFrame, value: str, cond_col: str, a, b, item_col: str = "item_id",
        producer_col: str = "model", producer: str | None = None, B: int = 10000, seed: int = 0) -> dict:
    """Paired gap a - b within item x producer; pooled over producers when producer is None."""
    sub = df if producer is None else df[df[producer_col] == producer]
    pdiff = paired_diff(sub, value, cond_col, a, b, [item_col, producer_col])
    if producer is None:
        # Pooled: only items observed under every producer keep producer weights equal.
        n_prod = sub[producer_col].nunique()
        counts = pdiff.groupby(item_col)[producer_col].nunique()
        complete = counts[counts == n_prod].index
        per_item = pdiff[pdiff[item_col].isin(complete)].groupby(item_col)["diff"].mean()
    else:
        per_item = pdiff.set_index(item_col)["diff"]
    res = bootstrap_mean(per_item, B, seed)
    res["n_pairs"] = len(pdiff)
    sd = _response_sd(sub, value, cond_col, a, b)
    res["d_response"] = res["estimate"] / sd if sd and sd > 0 else math.nan
    return res


def diff_in_diff(df: pd.DataFrame, value: str, cond_col: str, a, b, group_col: str, g1: str, g0: str,
                 item_col: str = "item_id", producer_col: str = "model", producer: str | None = None,
                 B: int = 10000, seed: int = 0) -> dict:
    """(gap in g1) - (gap in g0), paired by item (x producer). Used for the language x cue
    interaction (Section 6.5)."""
    sub = df if producer is None else df[df[producer_col] == producer]
    d1 = paired_diff(sub[sub[group_col] == g1], value, cond_col, a, b, [item_col, producer_col])
    d0 = paired_diff(sub[sub[group_col] == g0], value, cond_col, a, b, [item_col, producer_col])
    m = d1.merge(d0, on=[item_col, producer_col], suffixes=("_1", "_0"))
    m["dd"] = m["diff_1"] - m["diff_0"]
    if producer is None:
        n_prod = sub[producer_col].nunique()
        counts = m.groupby(item_col)[producer_col].nunique()
        m = m[m[item_col].isin(counts[counts == n_prod].index)]
    per_item = m.groupby(item_col)["dd"].mean()
    res = bootstrap_mean(per_item, B, seed)
    res["n_pairs"] = len(m)
    return res


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (NaN-safe: NaNs stay NaN and are not counted)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    m = int(ok.sum())
    if m == 0:
        return out.tolist()
    order = np.argsort(p[ok])
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[ok][i])
        adj[i] = min(1.0, running)
    out[ok] = adj
    return out.tolist()


def add_holm(df: pd.DataFrame, family_col: str = "family", p_col: str = "p_value") -> pd.DataFrame:
    df = df.copy()
    df["p_holm"] = np.nan
    for fam, idx in df.groupby(family_col).groups.items():
        df.loc[idx, "p_holm"] = holm(df.loc[idx, p_col].tolist())
    df["significant_holm_05"] = df["p_holm"] < 0.05
    return df


def krippendorff_alpha(matrix: np.ndarray, level: str) -> float:
    """Reliability data: raters x units, NaN for missing. Wraps the `krippendorff` package."""
    import krippendorff
    m = np.asarray(matrix, dtype=float)
    pairable = (~np.isnan(m)).sum(axis=0) >= 2
    if pairable.sum() < 2:
        return math.nan
    m = m[:, pairable]
    if len(np.unique(m[~np.isnan(m)])) < 2:
        return math.nan  # no variation: alpha undefined
    return float(krippendorff.alpha(reliability_data=m, level_of_measurement=level))
