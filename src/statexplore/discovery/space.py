"""Discovery search space: settings, candidate grammar, deterministic enumeration, exact counts.

A candidate is a (focal interaction, covariate set) pair. Its full model contains the
focal variables' complete hierarchy (every lower-order term), the covariates and the
required covariates as main effects. The reduced model drops only the focal term, and
the focal interaction is tested by comparing the two (see docs/formula-grammar.md).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import combinations
from math import comb
from typing import Any, Iterator

from ..core.dataset import Dataset
from ..core.variables import Measure, Storage

MAX_VARIABLES = 40          # outcome + candidates + required (§7)
MAX_PREDICTORS_LIMIT = 6    # hard cap on distinct predictors per model (§7.1)
LARGE_SEARCH = 1_000_000

PROFILES: dict[str, dict[str, Any]] = {
    "fast": {"max_models": 2_000, "max_minutes": 2.0, "workers": 1},
    "balanced": {"max_models": 50_000, "max_minutes": 30.0, "workers": 4},
    "thorough": {"max_models": None, "max_minutes": None, "workers": 0},  # 0 = cores - 1
}


class SearchSpaceError(ValueError):
    pass


@dataclass
class SearchSettings:
    outcome: str
    candidates: list[str]
    required: list[str] = field(default_factory=list)
    max_predictors: int = 3
    max_order: int = 2                 # 2 = two-way; 3 = two- and three-way
    covariate_mode: str = "focal"      # "focal" | "subsets"
    alpha: float = 0.05
    correction: str = "BH"             # "BH" | "holm"
    mode: str = "C"                    # A first significant | B target count | C exhaustive | D budget
    target: int = 10
    profile: str = "balanced"
    max_models: int | None = 50_000
    max_minutes: float | None = 30.0
    workers: int = 4
    sample: str = "shared"             # "shared" | "per_model"
    max_params: int = 40
    conf: float = 0.95
    batch_size: int = 40
    participant: str | None = None     # set: linear mixed models with a random intercept per participant

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SearchSettings":
        return cls(**d)

    def apply_profile(self, name: str) -> None:
        self.profile = name
        for k, v in PROFILES[name].items():
            setattr(self, k, v)


@dataclass(frozen=True)
class Candidate:
    index: int
    focal: tuple[str, ...]          # variable names, in candidate order
    covariates: tuple[str, ...]     # other subset members (subset mode)
    required: tuple[str, ...]

    @property
    def order(self) -> int:
        return len(self.focal)

    @property
    def variables(self) -> tuple[str, ...]:
        return self.focal + self.covariates + self.required


# ---- validation -----------------------------------------------------------------

def search_family(ds: Dataset, s: "SearchSettings") -> str:
    family = outcome_family(ds, s.outcome)
    if s.participant:
        if family != "gaussian":
            raise SearchSpaceError("Mixed-model discovery needs a continuous outcome.")
        return "lmm"
    return family


def outcome_family(ds: Dataset, outcome: str) -> str:
    v = ds.var(outcome)
    n_values = ds.analysis_frame([outcome])[outcome].dropna().nunique()
    if n_values == 2:
        return "binomial"
    if v.measure is Measure.CONTINUOUS and v.storage is Storage.NUMERIC:
        return "gaussian"
    raise SearchSpaceError(f"{v.display_name} must be continuous (linear models) or have exactly two "
                           "values (logistic models).")


def validate_settings(ds: Dataset, s: SearchSettings) -> str:
    """Raise SearchSpaceError with a user-facing reason; return the model family."""
    if s.participant and s.participant in [s.outcome] + s.candidates + s.required:
        raise SearchSpaceError("The participant ID cannot also be the outcome or a predictor.")
    family = search_family(ds, s)
    if s.outcome in s.candidates or s.outcome in s.required:
        raise SearchSpaceError("The outcome cannot also be a predictor.")
    overlap = set(s.candidates) & set(s.required)
    if overlap:
        raise SearchSpaceError(f"Variables cannot be both candidates and required: {sorted(overlap)}")
    if len(set(s.candidates)) != len(s.candidates) or len(set(s.required)) != len(s.required):
        raise SearchSpaceError("Duplicate variables in the selection.")
    if 1 + len(s.candidates) + len(s.required) > MAX_VARIABLES:
        raise SearchSpaceError(f"At most {MAX_VARIABLES} variables (outcome included) may be selected.")
    if len(s.candidates) < 2:
        raise SearchSpaceError("Choose at least two candidate predictors.")
    if not 2 <= s.max_predictors <= MAX_PREDICTORS_LIMIT:
        raise SearchSpaceError(f"Maximum predictors per model must be between 2 and {MAX_PREDICTORS_LIMIT}.")
    if s.max_order not in (2, 3):
        raise SearchSpaceError("Interaction order must be 2 or 3.")
    if 2 + len(s.required) > s.max_predictors:
        raise SearchSpaceError("Required covariates leave no room for a two-way interaction "
                               "within the maximum number of predictors.")
    if s.covariate_mode not in ("focal", "subsets") or s.mode not in "ABCD" or s.sample not in ("shared", "per_model"):
        raise SearchSpaceError("Invalid search option.")
    if not 0 < s.alpha < 1:
        raise SearchSpaceError("The significance threshold must be between 0 and 1.")
    for n in s.candidates + s.required:
        v = ds.var(n)
        if v.storage is Storage.DATETIME or (not v.is_categorical and v.storage is not Storage.NUMERIC):
            raise SearchSpaceError(f"{v.display_name} cannot be used as a predictor.")
    return family


# ---- enumeration ------------------------------------------------------------------

def _orders(s: SearchSettings) -> list[int]:
    room = s.max_predictors - len(s.required)
    return [o for o in (2, 3) if o <= s.max_order and o <= room]


def count_candidates(s: SearchSettings) -> int:
    p, r = len(s.candidates), len(s.required)
    orders = _orders(s)
    if s.covariate_mode == "focal":
        return sum(comb(p, o) for o in orders)
    return sum(comb(p, size) * sum(comb(size, o) for o in orders)
               for size in range(2, s.max_predictors - r + 1))


def enumerate_candidates(s: SearchSettings, start: int = 0) -> Iterator[Candidate]:
    """Deterministic order. Focal mode: all pairs, then all triples (lexicographic).
    Subset mode: by subset size, then subset (lexicographic), then focal pairs, then triples."""
    names, req = s.candidates, tuple(s.required)
    orders = _orders(s)
    idx = 0

    def emit(focal_idx: tuple[int, ...], subset_idx: tuple[int, ...]) -> Candidate:
        return Candidate(idx, tuple(names[i] for i in focal_idx),
                         tuple(names[i] for i in subset_idx if i not in focal_idx), req)

    if s.covariate_mode == "focal":
        for o in orders:
            for f in combinations(range(len(names)), o):
                if idx >= start:
                    yield emit(f, f)
                idx += 1
        return
    for size in range(2, s.max_predictors - len(req) + 1):
        for subset in combinations(range(len(names)), size):
            for o in orders:
                for f in combinations(subset, o):
                    if idx >= start:
                        yield emit(f, subset)
                    idx += 1


# ---- formulas ---------------------------------------------------------------------

def hierarchy(focal: tuple[str, ...]) -> list[tuple[str, ...]]:
    """All non-empty subsets of the focal variables, lower orders first."""
    return [c for k in range(1, len(focal) + 1) for c in combinations(focal, k)]


def model_terms(c: Candidate) -> tuple[list[tuple[str, ...]], list[tuple[str, ...]]]:
    """(full terms, reduced terms) as tuples of variable names."""
    full = hierarchy(c.focal) + [(v,) for v in c.covariates + c.required]
    reduced = [t for t in full if t != c.focal]
    return full, reduced


def candidate_key(c: Candidate) -> str:
    full, _ = model_terms(c)
    terms = sorted(":".join(sorted(t)) for t in full)
    return f"{':'.join(sorted(c.focal))}|{'+'.join(terms)}"


def display_formula(ds: Dataset, outcome: str, c: Candidate) -> str:
    dn = lambda n: ds.var(n).display_name  # noqa: E731
    parts = [" * ".join(dn(v) for v in c.focal)] + [dn(v) for v in c.covariates + c.required]
    return f"{dn(outcome)} ~ {' + '.join(parts)}"


def focal_label(ds: Dataset, c: Candidate) -> str:
    return " × ".join(ds.var(v).display_name for v in c.focal)


def shared_sample_size(ds: Dataset, s: SearchSettings) -> int:
    frame = ds.analysis_frame([s.outcome] + s.candidates + s.required)
    return int(frame.notna().all(axis=1).sum())


def dependence_warning(ds: Dataset, s: SearchSettings) -> str:
    """Repeated rows per participant (ID role) in the analysed sample violate independence."""
    from ..core.variables import Role

    ids = [v.name for v in ds.variables.values() if v.role is Role.ID]
    if not ids or s.participant:
        return ""  # mixed-model searches model the repeated measurements
    used = [s.outcome] + s.candidates + s.required
    frame = ds.analysis_frame(list(dict.fromkeys(used + ids[:1]))).dropna(subset=used)
    n_rows, n_ids = len(frame), frame[ids[0]].nunique()
    if n_ids >= n_rows:
        return ""
    return (f"The {n_rows:,} rows come from only {n_ids:,} participants ({ds.var(ids[0]).display_name}). "
            "Rows are not independent, so every p-value in this search is too small and false discoveries "
            "will be more common. Choose the participant ID under 'Repeated measures' to fit mixed models, "
            "or filter to one row per participant (e.g. one wave).")


def describe_space(ds: Dataset, s: SearchSettings) -> dict[str, Any]:
    family = validate_settings(ds, s)
    total = count_candidates(s)
    warnings = []
    if total >= LARGE_SEARCH:
        warnings.append(f"{total:,} candidate models: an exhaustive search will take a long time. "
                        "Consider fewer candidates, a smaller maximum, or focal mode.")
    n_shared = shared_sample_size(ds, s)
    n_all = len(ds.analysis_frame([s.outcome]))
    if s.sample == "shared" and n_shared < 0.5 * n_all:
        warnings.append(f"The shared complete-case sample keeps only {n_shared:,} of {n_all:,} rows.")
    dep = dependence_warning(ds, s)
    if dep:
        warnings.append(dep)
    budget = s.max_models
    if s.mode in "CD" and budget is not None and budget < total:
        warnings.append(f"The model budget ({budget:,}) is smaller than the search space: the run "
                        "will be labelled partial.")
    return {"family": family, "total": total, "n_shared": n_shared, "n_rows": n_all, "warnings": warnings}
