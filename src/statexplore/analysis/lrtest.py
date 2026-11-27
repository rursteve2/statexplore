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


def log_upper_gamma_q(a: float, x: float) -> float:
    """log Q(a, x) (regularized upper incomplete gamma) without underflow (V134-002): series / continued fraction
    evaluated in log space (Lentz), as in Numerical Recipes gammq, so tails far below the double range keep their
    exact magnitude. Used for chi-square upper tails: P(chi2_k > x) = Q(k / 2, x / 2)."""
    if x <= 0:
        return 0.0
    lg = math.lgamma(a)
    pre = -x + a * math.log(x) - lg
    if x < a + 1.0:                       # lower series: Q = 1 - P (no underflow problem in this region)
        s = term = 1.0 / a
        n = a
        for _ in range(10000):
            n += 1.0
            term *= x / n
            s += term
            if abs(term) < abs(s) * 1e-16:
                break
        p = math.exp(pre + math.log(s))
        return math.log1p(-p) if p < 1 else -math.inf
    tiny = 1e-300                         # continued fraction for Q (modified Lentz)
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        dl = d * c
        h *= dl
        if abs(dl - 1.0) < 1e-16:
            break
    return pre + math.log(h)


def _log_sf(lr: float, k: int) -> float:
    from scipy.stats import chi2

    if k == 0:
        return -math.inf if lr > 0 else 0.0
    sf = float(chi2.sf(lr, k))
    if sf > 1e-280:
        return math.log(sf)
    return log_upper_gamma_q(k / 2.0, lr / 2.0)


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
