"""Likelihood-ratio tests with the reference distribution of the hypothesis (v1.3.2 H6; STAT-17 / 23 / 41).

Python twin of the R helper `sx_lr_test` (analysis/covstruct.py); tests check that both give the same answer.
  * |LR| <= NEG_TOL below zero is numerical noise: LR is set to 0 and the raw value is kept.
  * A materially negative or non-finite LR, or df <= 0, never yields a p-value: the reason is returned.
  * reference "chisq": chi-square(df); "mixture": 50:50 chi-square(df - 1) : chi-square(df), chi-square(0) = point mass 0.
p-values are computed on the log scale so tiny probabilities are not rounded to zero.
"""
from __future__ import annotations

import math
from typing import Any

NEG_TOL = 1e-6


def _log_sf(lr: float, k: int) -> float:
    from scipy.stats import chi2

    if k == 0:
        return -math.inf if lr > 0 else 0.0
    return float(chi2.logsf(lr, k))


def lr_test(ll0: Any, ll1: Any, df: Any, reference: str = "chisq", tol: float = NEG_TOL) -> dict[str, Any]:
    out: dict[str, Any] = {"LR_raw": None, "LR": None, "df": df, "p": None, "log_p": None, "reference": reference,
                           "status": "tested", "reason": ""}
    try:
        raw = 2.0 * (float(ll1) - float(ll0))
    except (TypeError, ValueError):
        raw = math.nan
    out["LR_raw"] = raw
    if not math.isfinite(raw):
        return {**out, "status": "withheld", "reason": "non-finite likelihood-ratio statistic"}
    try:
        dfv = float(df)
    except (TypeError, ValueError):
        dfv = math.nan
    if not math.isfinite(dfv) or dfv <= 0:
        return {**out, "status": "withheld", "reason": "no extra parameters (df <= 0)"}
    if raw < -tol:
        return {**out, "status": "withheld",
                "reason": f"the larger model has a lower likelihood (LR = {raw:.6g}): at least one optimization is "
                          "unreliable; no p-value"}
    lr = max(raw, 0.0)
    out["LR"] = lr
    if raw < 0:
        out["reason"] = f"negligible negative LR {raw:.3g} set to 0 (tolerance {tol:g})"
    k = int(round(dfv))
    if reference == "mixture":
        a, b = math.log(0.5) + _log_sf(lr, k - 1), math.log(0.5) + _log_sf(lr, k)
        mx = max(a, b)
        lp = mx if not math.isfinite(mx) else mx + math.log(math.exp(a - mx) + math.exp(b - mx))
    else:
        lp = _log_sf(lr, k)
    out["log_p"] = lp
    out["p"] = math.exp(lp)
    return out
