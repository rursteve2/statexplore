"""Derived variables and row filters, defined as specs (never free-text code).

Every operation has two implementations that must agree: `compute()` here
(pandas, for previews/overviews) and the R code in rbridge.codegen (executed
in the load script so View Code reproduces from the source file). Tests
assert the two match for every operation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

DERIVE_OPS: dict[str, str] = {
    "mean_items": "Mean of items (allows some missing items)",
    "sum_items": "Sum of items (missing if any item is missing)",
    "recode": "Recode / collapse values",
    "log": "Natural log (values <= 0 become missing)",
    "sqrt": "Square root (values < 0 become missing)",
    "zscore": "Standardize (z-score)",
    "center": "Mean-center",
    "square": "Square",
    "binarize": "Binarize at a cutoff (1/0)",
}
MULTI_SOURCE_OPS = {"mean_items", "sum_items"}

FILTER_OPS: dict[str, str] = {
    "==": "equals", "!=": "does not equal", "<": "less than", "<=": "at most", ">": "greater than",
    ">=": "at least", "in": "is one of", "not_in": "is not one of", "is_missing": "is missing",
    "not_missing": "is not missing",
}
VALUELESS_FILTER_OPS = {"is_missing", "not_missing"}


class DeriveError(ValueError):
    pass


@dataclass
class DerivedSpec:
    name: str
    op: str
    sources: list[str]
    params: dict[str, Any] = field(default_factory=dict)
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.op == "recode":
            d["params"] = dict(self.params, mapping=[list(m) for m in self.params.get("mapping", [])])
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DerivedSpec":
        return cls(**d)

    def describe(self) -> str:
        src = ", ".join(self.sources)
        if self.op == "mean_items":
            return f"mean of [{src}] (at least {self.params['min_valid']} valid)"
        if self.op == "recode":
            pairs = ", ".join(f"{a}→{b}" for a, b in self.params["mapping"])
            other = "kept" if self.params.get("keep_other", True) else "set missing"
            return f"recode {src}: {pairs}; other values {other}"
        if self.op == "binarize":
            return f"1 if {src} {'>' if self.params.get('strict') else '>='} {self.params['cutoff']} else 0"
        return f"{self.op}({src})"


@dataclass
class FilterSpec:
    variable: str
    op: str
    value: Any = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FilterSpec":
        return cls(**d)

    def describe(self) -> str:
        if self.op in VALUELESS_FILTER_OPS:
            return f"{self.variable} {FILTER_OPS[self.op]}"
        return f"{self.variable} {FILTER_OPS[self.op]} {self.value}"


# ---- validation ----------------------------------------------------------

def validate_derived(spec: DerivedSpec, n_sources_available: Callable[[str], bool]) -> None:
    if spec.op not in DERIVE_OPS:
        raise DeriveError(f"Unknown operation {spec.op!r}.")
    if not spec.name.strip():
        raise DeriveError("The new variable needs a name.")
    if not spec.sources:
        raise DeriveError("Choose at least one source variable.")
    for s in spec.sources:
        if not n_sources_available(s):
            raise DeriveError(f"Source {s!r} is not a numeric variable defined before this one.")
    if spec.op not in MULTI_SOURCE_OPS and len(spec.sources) != 1:
        raise DeriveError(f"{DERIVE_OPS[spec.op]} uses exactly one source variable.")
    if spec.op in MULTI_SOURCE_OPS and len(spec.sources) < 2:
        raise DeriveError("Choose at least two items.")
    p = spec.params
    if spec.op == "mean_items":
        k = p.get("min_valid")
        if not isinstance(k, int) or not 1 <= k <= len(spec.sources):
            raise DeriveError(f"Minimum valid items must be between 1 and {len(spec.sources)}.")
    if spec.op == "recode":
        mapping = p.get("mapping")
        if not mapping or not all(len(m) == 2 for m in mapping):
            raise DeriveError("Add at least one 'old → new' pair.")
        olds = [float(m[0]) for m in mapping]
        if len(set(olds)) != len(olds):
            raise DeriveError("Each old value may appear only once.")
        p["mapping"] = [[float(a), None if b is None else float(b)] for a, b in mapping]
        p.setdefault("keep_other", True)
    if spec.op == "binarize":
        if p.get("cutoff") is None:
            raise DeriveError("Binarize needs a cutoff.")
        p["cutoff"] = float(p["cutoff"])
        p["strict"] = bool(p.get("strict", False))


# ---- computation (Python twin of the R code in codegen) ------------------

def compute(spec: DerivedSpec, get: Callable[[str], pd.Series]) -> pd.Series:
    cols = [pd.to_numeric(get(s), errors="coerce").astype(float) for s in spec.sources]
    x = cols[0]
    p = spec.params
    if spec.op == "mean_items":
        m = pd.concat(cols, axis=1)
        n_valid = m.notna().sum(axis=1)
        out = m.mean(axis=1, skipna=True)
        return out.where(n_valid >= p["min_valid"])
    if spec.op == "sum_items":
        return pd.concat(cols, axis=1).sum(axis=1, skipna=False, min_count=len(cols))
    if spec.op == "log":
        return x.where(x > 0).pipe(np.log)
    if spec.op == "sqrt":
        return x.where(x >= 0).pipe(np.sqrt)
    if spec.op == "zscore":
        return (x - x.mean()) / x.std(ddof=1)
    if spec.op == "center":
        return x - x.mean()
    if spec.op == "square":
        return x ** 2
    if spec.op == "binarize":
        hit = x > p["cutoff"] if p["strict"] else x >= p["cutoff"]
        return hit.astype(float).where(x.notna())
    if spec.op == "recode":
        lookup = {a: (np.nan if b is None else b) for a, b in p["mapping"]}
        mapped = x.map(lambda v: lookup.get(v, np.nan) if v in lookup else (v if p["keep_other"] else np.nan))
        return mapped.astype(float)
    raise DeriveError(f"Unknown operation {spec.op!r}.")


def filter_mask(f: FilterSpec, col: pd.Series) -> pd.Series:
    """True = keep. Comparisons involving missing values drop the row (like dplyr::filter)."""
    if f.op == "is_missing":
        return col.isna()
    if f.op == "not_missing":
        return col.notna()
    if f.op in ("in", "not_in"):
        values = list(f.value)
        hit = col.isin(values) & col.notna()
        return hit if f.op == "in" else (~col.isin(values)) & col.notna()
    v = f.value
    ops = {"==": col.eq, "!=": col.ne, "<": col.lt, "<=": col.le, ">": col.gt, ">=": col.ge}
    if f.op not in ops:
        raise DeriveError(f"Unknown filter operator {f.op!r}.")
    return ops[f.op](v).fillna(False).astype(bool) & col.notna()
