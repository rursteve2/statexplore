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
                         sample_vars=list(self.sample_vars), focal=None, selected_from=self.selected_from)

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
    terms: list[Term] = [focal] + [(v,) for v in list(row["covariates"]) + list(s.required)]
    sample = [s.outcome] + s.candidates + s.required if s.sample == "shared" else []
    return ModelSpec(outcome=s.outcome, terms=terms, family=run.family, label=f"Candidate #{row['idx']}",
                     source=f"discovery {run.id} #{row['idx']}", sample_vars=sample, focal=focal,
                     selected_from=run.total)
