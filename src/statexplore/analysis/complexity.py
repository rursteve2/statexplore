"""Configurable complexity limits for manually built models (M §10).

Automated discovery keeps six distinct predictors per model. Manual models get their own, adjustable
limits — one number per analysis, used both by the form and by backend validation, so the GUI and the
R code can never disagree. A limit counts distinct variables; the preflight separately reports model
terms, fixed coefficients after dummy coding and interactions, complete rows, residual df and (for
factorial designs) observed cells. No observations-per-predictor rule decides validity on its own:
only an unidentifiable model (coefficients >= rows, empty factorial cells) is refused.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import prod
from typing import Any

from ..core.dataset import Dataset
from .framework import AnalysisInputError, Option


@dataclass(frozen=True)
class Limit:
    slot: str
    option: str
    label: str
    default: int
    hard_max: int


LIMITS: dict[str, list[Limit]] = {
    **{k: [Limit("predictors", "max_predictors", "Maximum predictors (complexity limit)", 6, 30)]
       for k in ("lm", "logistic", "ordinal_reg", "likert_sensitivity", "multinom_reg")},
    "anova_factorial": [Limit("factors", "max_factors", "Maximum factors (complexity limit)", 3, 5)],
    "ancova": [Limit("factors", "max_factors", "Maximum factors (complexity limit)", 2, 3),
               Limit("covariates", "max_covariates", "Maximum covariates (complexity limit)", 4, 8)],
}
HELP = ("Default {d}. Raise it (e.g. 10 or 15) only after checking the coefficient count in the preflight below; "
        "more variables need more data.")


def install(specs: list[Any]) -> None:
    """Add the limit options to the analysis specs and let their slots accept up to the hard maximum."""
    for spec in specs:
        for lim in LIMITS.get(spec.key, []):
            slot = next(s for s in spec.slots if s.key == lim.slot)
            slot.max = lim.hard_max
            if not any(o.key == lim.option for o in spec.options):
                spec.options.insert(0, Option(lim.option, lim.label, "int", lim.default, min=1,
                                              max=lim.hard_max, help=HELP.format(d=lim.default)))
        if spec.key in LIMITS:
            spec.check = _chain(spec.check, spec.key)


def _chain(previous, key: str):
    def check(ds: Dataset, vars_: dict[str, list[str]], opts: dict[str, Any]) -> None:
        enforce(key, ds, vars_, opts)
        if previous is not None:
            previous(ds, vars_, opts)
    return check


def enforce(key: str, ds: Dataset, vars_: dict[str, list[str]], opts: dict[str, Any]) -> None:
    for lim in LIMITS.get(key, []):
        n = len(vars_.get(lim.slot, []))
        cap = int(opts.get(lim.option, lim.default) or lim.default)
        if n > cap:
            raise AnalysisInputError(f"{n} variables chosen but the complexity limit is {cap}. Raise "
                                     f"'{lim.label}' under Options to fit a larger model.")
    pre = preflight(key, ds, vars_, opts)
    if pre and pre["n"] and pre["coefficients"] >= pre["n"]:
        raise AnalysisInputError(f"The model needs {pre['coefficients']} fixed coefficients but only {pre['n']} rows "
                                 "are complete: it cannot be estimated.")
    if pre and pre.get("empty_cells"):
        raise AnalysisInputError(f"{pre['empty_cells']} of {pre['cells']} factor combinations have no rows: the "
                                 "highest-order interaction is not estimable. Use fewer factors or collapse levels.")


def preflight(key: str, ds: Dataset, vars_: dict[str, list[str]], opts: dict[str, Any]) -> dict[str, Any] | None:
    """Distinct predictors, model terms, fixed coefficients, complete rows, residual df, cells."""
    if key not in LIMITS:
        return None
    dep = (vars_.get("dependent") or [None])[0]
    if key in ("anova_factorial", "ancova"):
        factors, covs = vars_.get("factors", []), vars_.get("covariates", [])
        terms = [t for k in range(1, len(factors) + 1) for t in combinations(factors, k)] + [(c,) for c in covs]
        predictors = factors + covs
    else:
        predictors = vars_.get("predictors", [])
        mode = opts.get("interactions", "none")
        orders = {"none": 1, "two_way": 2}.get(mode, len(predictors))
        terms = [t for k in range(1, min(orders, len(predictors)) + 1) for t in combinations(predictors, k)]
    if not predictors:
        return None
    cols = [c for c in [dep] + predictors if c]
    frame = ds.analysis_frame(cols).dropna()
    levels = {v: int(frame[v].nunique()) for v in predictors if ds.var(v).is_categorical}
    coef = 1 + sum(prod((levels[v] - 1) if v in levels else 1 for v in t) for t in terms)
    k_out = 1
    if key == "multinom_reg" and dep:
        k_out = max(1, int(frame[dep].nunique()) - 1)
    elif key in ("ordinal_reg",) and dep:
        coef += int(frame[dep].nunique()) - 2           # thresholds replace the intercept
    out = {"predictors": len(set(predictors)), "terms": len(terms), "coefficients": coef * k_out,
           "n": len(frame), "residual_df": len(frame) - coef * k_out}
    if key == "anova_factorial" or (key == "ancova" and len(vars_.get("factors", [])) > 1):
        f = vars_.get("factors", [])
        cells = prod(levels.get(v, 1) for v in f)
        observed = int(frame.groupby(f).ngroups) if f and len(frame) else 0
        out.update(cells=cells, empty_cells=cells - observed if len(f) > 1 else 0)
    return out


def describe(pre: dict[str, Any] | None) -> str:
    if not pre:
        return ""
    txt = (f"Complexity preflight: {pre['predictors']} distinct predictors · {pre['terms']} model terms · "
           f"{pre['coefficients']} fixed coefficients (after dummy coding and interactions) · "
           f"{pre['n']:,} complete rows · residual df {pre['residual_df']:,}")
    if "cells" in pre:
        txt += f" · {pre['cells']} factor cells ({pre['empty_cells']} empty)"
    return txt
