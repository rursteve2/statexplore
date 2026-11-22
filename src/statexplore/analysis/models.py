"""A hierarchical model specification shared by the Teaching Studio and Model Comparison.

Terms are tuples of variable names: ("x",) is a main effect, ("x", "m") an interaction.
Specs are always hierarchically closed (every lower-order term of an interaction is present).
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Any

from ..core.dataset import Dataset
from ..rbridge.codegen import alias

Term = tuple[str, ...]


def close_hierarchy(terms: list[Term]) -> list[Term]:
    """Add every lower-order term; order: main effects, then 2-way, then 3-way (stable)."""
    out: list[Term] = []
    for t in terms:
        for k in range(1, len(t) + 1):
            for sub in combinations(t, k):
                if sub not in out:
                    out.append(sub)
    return sorted(out, key=lambda t: (len(t), out.index(t)))


def drop_term(terms: list[Term], t: Term) -> list[Term]:
    """Remove a term and every higher-order term containing it (keeps the model hierarchical)."""
    return [u for u in terms if not set(t) <= set(u)]


@dataclass
class ModelSpec:
    outcome: str
    terms: list[Term]
    family: str                         # "gaussian" | "binomial"
    label: str = ""
    source: str = "manual"
    sample_vars: list[str] = field(default_factory=list)   # extra variables for a shared complete-case sample
    focal: Term | None = None            # interaction of interest (for plots/interpretation)
    selected_from: int | None = None     # number of candidates searched if it came from discovery
    treatments: list[dict[str, Any]] = field(default_factory=list)   # analysis-specific encodings (v1.1)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])

    def __post_init__(self) -> None:
        self.terms = close_hierarchy([tuple(t) for t in self.terms])
        if self.focal is not None:
            self.focal = tuple(self.focal)
        if self.focal is None:
            inter = [t for t in self.terms if len(t) > 1]
            self.focal = max(inter, key=len) if inter else None

    @property
    def variables(self) -> list[str]:
        out: list[str] = []
        for t in self.terms:
            for v in t:
                if v not in out:
                    out.append(v)
        return out

    def rhs(self, ds: Dataset) -> str:
        return " + ".join(":".join(alias(ds.var(v).alias) for v in t) for t in self.terms) or "1"

    def display(self, ds: Dataset) -> str:
        dn = lambda v: ds.var(v).display_name  # noqa: E731
        top = [t for t in self.terms if not any(set(t) < set(u) for u in self.terms)]
        parts = [" * ".join(dn(v) for v in t) for t in top]
        return f"{dn(self.outcome)} ~ {' + '.join(parts) if parts else '1'}"

    def nested_in(self, other: "ModelSpec") -> bool:
        """True if this model's terms are a strict subset of `other`'s (same outcome and family)."""
        return (self.outcome == other.outcome and self.family == other.family
                and set(self.terms) < set(other.terms))

    def copy_edited(self, terms: list[Term], label: str = "") -> "ModelSpec":
        return ModelSpec(outcome=self.outcome, terms=terms, family=self.family,
                         label=label or f"{self.label or 'model'} (edited)", source=f"edited from {self.id}",
                         sample_vars=list(self.sample_vars), focal=None, selected_from=self.selected_from,
                         treatments=[dict(t) for t in self.treatments])

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["terms"] = [list(t) for t in self.terms]
        d["focal"] = list(self.focal) if self.focal else None
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelSpec":
        d = dict(d)
        d["terms"] = [tuple(t) for t in d["terms"]]
        return cls(**d)


def from_candidate(run: Any, row: dict[str, Any]) -> ModelSpec:
    """ModelSpec for a discovery candidate (same terms and same shared sample)."""
    s = run.settings
    focal = tuple(row["focal"])
    if row.get("kind") in ("main", "combo", "factorial"):
        inter = [tuple(t) for t in (row.get("inter") or [])]
        terms: list[Term] = [(v,) for v in row["covariates"]] + inter + [(v,) for v in s.required]
        top = [t for t in inter if not any(set(t) < set(u) for u in inter)]
        focal = top[0] if len(top) == 1 else None
    elif row.get("kind") == "all_two_way":
        terms: list[Term] = list(combinations(focal, 2)) + [(v,) for v in s.required]
        focal = None
    else:
        terms = [focal] + [(v,) for v in list(row["covariates"]) + list(s.required)]
    sample = [s.outcome] + s.candidates + s.required if s.sample == "shared" else []
    return ModelSpec(outcome=s.outcome, terms=terms, family=run.family, label=f"Candidate #{row['idx']}",
                     source=f"discovery {run.id} #{row['idx']}", sample_vars=sample, focal=focal,
                     selected_from=run.total, treatments=[dict(t) for t in getattr(s, "treatments", [])])


def model_view(ds: Dataset, m: "ModelSpec | None") -> Dataset:
    """The dataset as this model sees it (its analysis-specific treatments applied)."""
    from .measurement import TreatmentError, apply_treatments, as_treatments

    if m is None:
        return ds
    try:
        return apply_treatments(ds, as_treatments(m.treatments))
    except TreatmentError as e:
        raise ModelCheckError(str(e)) from None


# ---- pre-fit checks (§7.2: reject unidentifiable models with a reason) ---------------------

MAX_CATEGORIES = 50


class ModelCheckError(ValueError):
    """User-facing reason why a model cannot be fitted."""


def variable_problem(ds: Dataset, name: str, frame=None, allow_many_levels: bool = False) -> str:
    """Why a variable cannot be used as a predictor/outcome in the analysed rows ('' if fine)."""
    from ..core.variables import Role

    v = ds.var(name)
    col = (frame[name] if frame is not None else ds.analysis_frame([name])[name]).dropna()
    k = col.nunique()
    if k == 0:
        return f"{v.display_name} has no non-missing values in the rows analysed."
    if k == 1:
        return (f"{v.display_name} has the same value in every row analysed ({col.iloc[0]!r}), so it cannot "
                f"explain anything. Remove it from the model.")
    if v.role is Role.ID and not allow_many_levels:
        return (f"{v.display_name} is the participant ID. IDs identify people; they are not predictors or "
                f"outcomes. Use the Mixed & Longitudinal workspace (or Repeated measures in Discovery).")
    if v.is_categorical and k > MAX_CATEGORIES and not allow_many_levels:
        return (f"{v.display_name} has {k:,} categories: it looks like an identifier, not a predictor (each "
                f"category adds a parameter). Set its role to id/cluster, recode it into fewer groups, or treat "
                f"it as continuous if it is a number.")
    return ""


def check_model(ds: Dataset, m: ModelSpec) -> int:
    """Raise ModelCheckError if the model cannot be fitted; return the complete-case n."""
    ds = model_view(ds, m)
    names = list(dict.fromkeys([m.outcome] + m.variables + m.sample_vars))
    frame = ds.analysis_frame(names).dropna()
    if len(frame) < 5:
        raise ModelCheckError(f"Only {len(frame)} rows have no missing values on the model's variables.")
    for n in [m.outcome] + m.variables:
        why = variable_problem(ds, n, frame)
        if why:
            raise ModelCheckError(why)
    params = 1
    for t in m.terms:
        p = 1
        for v in t:
            p *= (frame[v].nunique() - 1) if ds.var(v).is_categorical else 1
        params += p
    if params >= len(frame) - 1:
        raise ModelCheckError(f"The model needs about {params} parameters but only {len(frame)} rows are "
                              "complete: it cannot be estimated. Use fewer variables or interactions.")
    return len(frame)
