"""Discovery search space: settings, candidate grammar, deterministic enumeration, exact counts.

Strategies (docs/formula-grammar.md):
  focal         a (focal interaction, covariate set) pair (v1.0/v1.1 grammar). Full model = the focal
                variables' complete hierarchy + covariates + required covariates; the reduced model
                drops only the focal term.
  combinations  every hierarchical set of interaction terms (up to the maximum order) on every subset
                of candidate predictors, main-effects-only models included (v1.2 default).
  factorial     the full hierarchical factorial model of every predictor subset.
  main          main-effects-only models on every predictor subset.
Each candidate carries a test plan: the typed hypotheses it tests (focal / individual interaction
term / joint interaction / main-effect term / whole model), each a nested comparison.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import combinations
from math import comb
from typing import Any, Iterator

import pandas as pd

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
    all_two_way: bool = False          # subset mode: also test each subset's model with ALL two-way terms (§10.3)
    # ---- v1.1 (§10): explicit, user-confirmed outcome model family and coding ----
    family: str = ""                   # gaussian | binomial | ordinal | multinomial ("" = v1.0 automatic choice)
    event: Any = None                  # binomial: event category code (None = second category, as in v1.0)
    reference: Any = None              # multinomial: baseline category code (None = first category)
    treatments: list[dict[str, Any]] = field(default_factory=list)   # analysis-specific encodings
    # ---- v1.2: model-structure strategies (X §3; M §7) ----
    strategy: str = "focal"            # focal | combinations | factorial | main  (v1.1 runs load as focal)
    include_main: bool = False         # add main-effects-only models (focal / factorial); keep them (combinations)
    min_predictors: int = 1            # smallest predictor subset in combinations / main / factorial
    min_obs: int = 0                   # minimum complete rows for a model to be fitted (0 = no minimum)
    # ---- v1.2: longitudinal discovery (M §8). Longitudinal <=> `participant` is set. ----
    study: str = "independent"         # independent | longitudinal (recorded for the user; `participant` decides)
    time: str | None = None            # time variable (random slopes, residual correlation, time strategy)
    time_type: str = "continuous"      # continuous | categorical  (how time enters the fixed effects)
    structure: str = "ri"              # random-effect / covariance structure locked for workflow A
    workflow: str = "fixed"            # fixed (A: fixed-effect discovery) | structure (B: structure comparison)
    structures: list[str] = field(default_factory=list)   # workflow B: structures to compare
    time_interactions: bool = False    # workflow B: locked fixed part includes time x each candidate

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SearchSettings":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def apply_profile(self, name: str) -> None:
        self.profile = name
        for k, v in PROFILES[name].items():
            setattr(self, k, v)


STRATEGIES = {
    "combinations": "All eligible interaction combinations (every hierarchical set of interactions)",
    "focal": "Single focal interaction (+ optional covariates)",
    "factorial": "Full factorial model per predictor subset",
    "main": "Main effects only",
    "time": "Time × each eligible predictor (longitudinal)",
}

# Longitudinal structures (M §8.3-8.4): key -> (label, outcome families, engine group)
STRUCTURES: dict[str, tuple[str, tuple[str, ...], str]] = {
    "ri": ("Random intercept (1 | ID) — lme4", ("gaussian",), "gaussian"),
    "rs": ("Random intercept + time slope, correlated (time | ID) — lme4", ("gaussian",), "gaussian"),
    "rs_uncorr": ("Random intercept + time slope, uncorrelated (time || ID) — lme4", ("gaussian",), "gaussian"),
    "nlme_ri": ("Random intercept, independent residuals — nlme", ("gaussian",), "gaussian"),
    "nlme_rs": ("Random intercept + time slope, independent residuals — nlme", ("gaussian",), "gaussian"),
    "ar1": ("Random intercept + AR(1) residuals over the observed-occasion index — nlme corAR1", ("gaussian",), "gaussian"),
    "car1": ("Random intercept + continuous-time AR(1) residuals (actual time differences) — nlme corCAR1", ("gaussian",), "gaussian"),
    "mmrm_us": ("Marginal model, unstructured covariance — mmrm (no random effects)", ("gaussian",), "gaussian"),
    "mmrm_cs": ("Marginal model, compound-symmetry covariance — mmrm (no random effects)", ("gaussian",), "gaussian"),
    "mmrm_ar1": ("Marginal model, AR(1) covariance — mmrm (no random effects)", ("gaussian",), "gaussian"),
    "glmer_ri": ("Binary logistic mixed model, random intercept — lme4::glmer", ("binomial",), "binomial"),
    "glmer_rs": ("Binary logistic mixed model, random intercept + time slope — lme4::glmer", ("binomial",), "binomial"),
    "clmm_ri": ("Ordinal cumulative-logit mixed model, random intercept — ordinal::clmm", ("ordinal",), "ordinal"),
    "clmm_rs": ("Ordinal cumulative-logit mixed model, random intercept + time slope — ordinal::clmm",
                ("ordinal",), "ordinal"),
    "gee_exch": ("GEE, exchangeable working correlation — geepack (population-average)", ("gaussian", "binomial"), "gee"),
    "gee_ar1": ("GEE, AR(1) working correlation — geepack (population-average)", ("gaussian", "binomial"), "gee"),
    "gee_ind": ("GEE, independence working correlation — geepack (population-average)", ("gaussian", "binomial"), "gee"),
}
SLOPE_STRUCTURES = {"rs", "rs_uncorr", "nlme_rs", "glmer_rs", "clmm_rs"}
ORDERED_TIME_STRUCTURES = {"ar1", "car1", "mmrm_us", "mmrm_cs", "mmrm_ar1", "gee_ar1"}
LONG_FAMILIES = {"lmm": "Gaussian longitudinal model", "glmm": "Binary logistic mixed model (lme4::glmer)",
                 "clmm": "Ordinal cumulative-logit mixed model (ordinal::clmm)",
                 "gee": "GEE (geepack::geeglm, population-average)"}


def structures_for(family: str) -> list[str]:
    return [k for k, (_l, fams, _g) in STRUCTURES.items() if family in fams]
MODEL_TYPES = {"main": "Main effects only", "one": "One interaction", "multi": "Multiple interactions",
               "factorial": "Full factorial", "focal": "Single focal interaction",
               "focal_cov": "Focal interaction + covariates", "all_two_way": "All two-way interactions"}


@dataclass(frozen=True)
class Candidate:
    index: int
    focal: tuple[str, ...]          # focal kind: variable names, in candidate order
    covariates: tuple[str, ...]     # focal kind: other subset members; main/combo/factorial: the predictor subset
    required: tuple[str, ...]
    kind: str = "focal"             # focal | all_two_way | main | combo | factorial
    inter: tuple[tuple[str, ...], ...] = ()   # combo / factorial / structure: the interaction terms
    structure: str = ""                         # structure kind (workflow B): the structure compared

    @property
    def order(self) -> int:
        return len(self.focal)

    @property
    def variables(self) -> tuple[str, ...]:
        return self.focal + self.covariates + self.required


# ---- validation -----------------------------------------------------------------

FAMILIES = {
    "gaussian": "Gaussian linear regression (stats::lm)",
    "binomial": "Binary logistic regression (stats::glm, binomial)",
    "ordinal": "Ordinal logistic regression (ordinal::clm, cumulative logit, proportional odds)",
    "multinomial": "Multinomial logistic regression (nnet::multinom)",
}
FOCAL_TESTS = {
    "gaussian": "nested-model F test",
    "lmm": "likelihood-ratio test (ML fits, same random-effect / covariance structure)",
    "glmm": "likelihood-ratio test (Laplace ML fits, same random-effect structure)",
    "clmm": "likelihood-ratio test of nested cumulative-logit mixed models (same random effects)",
    "gee": "robust (sandwich) Wald test of the dropped coefficients",
    "binomial": "likelihood-ratio (deviance) test",
    "ordinal": "likelihood-ratio test of nested cumulative-logit models",
    "multinomial": "likelihood-ratio test of nested multinomial models (joint over all outcome contrasts)",
}


def view(ds: Dataset, s: "SearchSettings") -> Dataset:
    """The dataset as this search sees it (analysis-specific treatments applied)."""
    from ..analysis.measurement import TreatmentError, apply_treatments, as_treatments

    try:
        return apply_treatments(ds, as_treatments(s.treatments))
    except TreatmentError as e:
        raise SearchSpaceError(str(e)) from None


def suggest_family(ds: Dataset, outcome: str) -> tuple[str, str]:
    """A suggestion only (§10.1): the user must confirm the family. `ds` should be the view
    with the outcome's treatment applied."""
    v = ds.var(outcome)
    k = ds.analysis_frame([outcome])[outcome].dropna().nunique()
    if v.measure is Measure.CONTINUOUS and v.storage is Storage.NUMERIC:
        if k == 2:
            return "binomial", ("The outcome is numeric but has only two observed values: binary logistic "
                                "regression is usual; Gaussian is possible (linear probability model).")
        return "gaussian", "The outcome is continuous (or treated as continuous for this search)."
    if k == 2:
        return "binomial", "The outcome has exactly two categories."
    if v.measure is Measure.ORDINAL:
        return "ordinal", (f"The outcome is ordinal with {k} categories: ordinal logistic regression keeps their "
                           "order. Multinomial ignores the order; Gaussian needs 'treat as continuous'.")
    if v.is_categorical:
        return "multinomial", f"The outcome is nominal with {k} unordered categories."
    return "", "This variable cannot be an outcome."


def search_family(ds: Dataset, s: "SearchSettings") -> str:
    family = s.family or outcome_family(ds, s.outcome)
    if family not in FAMILIES:
        raise SearchSpaceError(f"Unknown outcome model family {family!r}.")
    if s.family:
        _check_family(ds, s)
    if s.participant:
        return longitudinal_family(family, s)
    return family


def longitudinal_family(family: str, s: "SearchSettings") -> str:
    """Engine family of a longitudinal search; refuses unsupported combinations with a reason."""
    if family == "multinomial":
        raise SearchSpaceError("Repeated nominal (multinomial) outcomes are not available in automated discovery: "
                               "the only engine (mclogit::mblogit, PQL) is approximate and has no valid likelihood-"
                               "ratio test. Use the Mixed & Longitudinal workspace, or choose another outcome family.")
    structs = s.structures if s.workflow == "structure" else [s.structure or "ri"]
    if not structs:
        raise SearchSpaceError("Choose at least one random-effect / covariance structure to compare.")
    groups = set()
    for st in structs:
        if st not in STRUCTURES:
            raise SearchSpaceError(f"Unknown structure {st!r}.")
        label, fams, group = STRUCTURES[st]
        if family not in fams:
            raise SearchSpaceError(f"{label} is not available for {family} outcomes.")
        groups.add(group)
    if len(groups) > 1:
        raise SearchSpaceError("Structures from different model families (e.g. GEE and mixed models) estimate "
                               "different things and cannot be ranked together. Compare them in separate runs.")
    group = groups.pop()
    if group == "gee":
        return "gee"
    return {"gaussian": "lmm", "binomial": "glmm", "ordinal": "clmm"}[family]


def _check_family(ds: Dataset, s: "SearchSettings") -> None:
    v = ds.var(s.outcome)
    levels_obs = sorted(ds.analysis_frame([s.outcome])[s.outcome].dropna().unique().tolist(), key=str)
    k = len(levels_obs)
    dn = v.display_name
    if s.family == "gaussian" and not (v.measure is Measure.CONTINUOUS and v.storage is Storage.NUMERIC):
        hint = (f" Tick 'Treat {dn} as continuous for this analysis' to model its numeric codes."
                if v.is_categorical and v.storage is Storage.NUMERIC else "")
        raise SearchSpaceError(f"Gaussian linear regression needs a continuous outcome; {dn} is {v.measure.value}.{hint}")
    if s.family == "binomial":
        if k != 2:
            raise SearchSpaceError(f"Binary logistic regression needs exactly two outcome values; {dn} has {k}.")
        if s.event is not None and s.event not in levels_obs:
            raise SearchSpaceError(f"The event category {s.event!r} is not a value of {dn}.")
    if s.family in ("ordinal", "multinomial"):
        if not v.is_categorical:
            raise SearchSpaceError(f"{FAMILIES[s.family].split(' (')[0]} needs a categorical outcome; {dn} is continuous.")
        if k < 3:
            raise SearchSpaceError(f"{dn} has {k} observed categories; use binary logistic regression for two.")
        if k > 15:
            raise SearchSpaceError(f"{dn} has {k} categories (limit 15 for {s.family} outcomes).")
    if s.family == "ordinal" and v.measure is not Measure.ORDINAL:
        raise SearchSpaceError(f"{dn} is {v.measure.value}: ordinal regression needs ordered categories. Approve an "
                               "ordinal level in Dataset Explorer ▸ Measurement Review, or choose multinomial.")
    if s.family == "multinomial" and s.reference is not None and s.reference not in ds.levels(s.outcome):
        raise SearchSpaceError(f"The reference category {s.reference!r} is not a category of {dn}.")


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
    ds = view(ds, s)
    if s.participant and s.participant in [s.outcome] + s.candidates + s.required:
        raise SearchSpaceError("The participant ID cannot also be the outcome or a predictor.")
    if s.time and not s.participant:
        raise SearchSpaceError("A time variable needs a participant ID (Study structure: repeated measures).")
    if s.participant:
        _check_longitudinal(ds, s)
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
    if s.strategy not in STRATEGIES:
        raise SearchSpaceError(f"Unknown search strategy {s.strategy!r}.")
    if s.participant and s.workflow == "structure":
        n_fixed = len(set(s.candidates) | set(s.required) | ({s.time} if s.time_interactions and s.time else set()))
        if n_fixed > MAX_PREDICTORS_LIMIT:
            raise SearchSpaceError(f"The locked fixed part has {n_fixed} distinct predictors (limit "
                                   f"{MAX_PREDICTORS_LIMIT}).")
    structure_run = bool(s.participant) and s.workflow == "structure"     # strategy does not apply
    needs_pair = s.strategy in ("focal", "factorial") and not structure_run
    if len(s.candidates) < (2 if needs_pair else 1):
        raise SearchSpaceError("Choose at least two candidate predictors." if needs_pair else
                               "Choose at least one candidate predictor.")
    if not (1 if s.strategy in ("main", "combinations", "time") or structure_run else 2) <= s.max_predictors \
            <= MAX_PREDICTORS_LIMIT:
        raise SearchSpaceError(f"Maximum predictors per model must be between 2 and {MAX_PREDICTORS_LIMIT}.")
    if s.max_order not in (2, 3):
        raise SearchSpaceError("Interaction order must be 2 or 3.")
    if needs_pair and 2 + len(s.required) > s.max_predictors:
        raise SearchSpaceError("Required covariates leave no room for a two-way interaction "
                               "within the maximum number of predictors.")
    if len(s.required) >= s.max_predictors and s.strategy in ("main", "combinations", "time") and not structure_run:
        raise SearchSpaceError("Required covariates leave no room for a candidate predictor "
                               "within the maximum number of predictors.")
    if not 1 <= s.min_predictors <= s.max_predictors:
        raise SearchSpaceError("The minimum number of predictors must be between 1 and the maximum.")
    if s.covariate_mode not in ("focal", "subsets") or s.mode not in "ABCD" or s.sample not in ("shared", "per_model"):
        raise SearchSpaceError("Invalid search option.")
    if not 0 < s.alpha < 1:
        raise SearchSpaceError("The significance threshold must be between 0 and 1.")
    from ..analysis.models import variable_problem

    for n in [s.outcome] + s.candidates + s.required:
        v = ds.var(n)
        if n != s.outcome and (v.storage is Storage.DATETIME or (not v.is_categorical and v.storage is not Storage.NUMERIC)):
            raise SearchSpaceError(f"{v.display_name} cannot be used as a predictor.")
        why = variable_problem(ds, n)
        if why:
            raise SearchSpaceError(why)
    return family


def _check_longitudinal(ds: Dataset, s: "SearchSettings") -> None:
    """ID / time structure checks on the analysed rows (M §8.1)."""
    pid, t = s.participant, s.time
    if t and (t in (s.outcome, pid) or t in s.candidates):
        raise SearchSpaceError("The time variable cannot also be the outcome, the participant ID or a candidate "
                               "predictor (it enters the fixed effects through the time strategy or as a required "
                               "covariate).")
    structs = set(s.structures if s.workflow == "structure" else [s.structure or "ri"])
    if s.strategy == "time" and not t:
        raise SearchSpaceError("The 'Time × each predictor' strategy needs a time variable.")
    if s.workflow == "structure" and s.time_interactions and not t:
        raise SearchSpaceError("Time interactions need a time variable.")
    if structs & (SLOPE_STRUCTURES | ORDERED_TIME_STRUCTURES) and not t:
        raise SearchSpaceError("Random slopes and residual correlation structures need a time variable.")
    if s.workflow not in ("fixed", "structure") or s.time_type not in ("continuous", "categorical"):
        raise SearchSpaceError("Invalid longitudinal option.")
    cols = list(dict.fromkeys([pid] + ([t] if t else [])))
    f = ds.analysis_frame(cols)
    if f[pid].isna().all():
        raise SearchSpaceError(f"{ds.var(pid).display_name} has no values.")
    n_rows, n_ids = int(f[pid].notna().sum()), int(f[pid].nunique())
    if n_ids >= n_rows:
        raise SearchSpaceError(f"{ds.var(pid).display_name} is unique in every row: there are no repeated "
                               "measurements to model. Choose 'Independent observations' or another ID.")
    if t:
        if ds.var(t).storage is not Storage.NUMERIC:
            raise SearchSpaceError(f"{ds.var(t).display_name} is not numeric: time must have numeric codes.")
        if structs & ORDERED_TIME_STRUCTURES and f.dropna().duplicated([pid, t]).any():
            raise SearchSpaceError("Some participants have more than one row at the same time point; residual "
                                   "correlation structures and marginal models need one row per participant per "
                                   "occasion.")
        if "car1" in structs and (f[t].dropna() < 0).any():
            raise SearchSpaceError("Continuous-time AR(1) needs non-negative elapsed time values.")


def longitudinal_summary(ds: Dataset, s: "SearchSettings") -> dict[str, Any]:
    """Rows, participants, waves and observed outcomes by wave on the analysed rows (M §8.1, §8.6)."""
    from ..analysis.longitudinal_preflight import preflight

    v = view(ds, s)
    rep = preflight(v, s.outcome, s.participant, s.time, s.candidates + [r for r in s.required if r != s.time])
    used = [s.outcome] + s.candidates + s.required + [s.participant] + ([s.time] if s.time else [])
    f = v.analysis_frame(list(dict.fromkeys(used))).dropna()
    per = f.groupby(s.participant).size() if len(f) else pd.Series(dtype=int)
    return {"report": rep, "rows": rep.n_records, "participants": rep.n_participants,
            "waves": len(rep.time_points), "complete_rows": len(f), "complete_participants": int(per.size),
            "participants_2plus": int((per >= 2).sum()), "participants_3plus": int((per >= 3).sum())}


# ---- enumeration ------------------------------------------------------------------

def _orders(s: SearchSettings) -> list[int]:
    room = s.max_predictors - len(s.required)
    return [o for o in (2, 3) if o <= s.max_order and o <= room]


def _room(s: SearchSettings) -> int:
    return s.max_predictors - len(s.required)


def _sizes(s: SearchSettings, smallest: int = 1) -> range:
    return range(max(smallest, s.min_predictors), min(_room(s), len(s.candidates)) + 1)


def _block_counts(s: SearchSettings) -> list[tuple[str, int]]:
    from .modelspec import n_interaction_sets

    p = len(s.candidates)
    blocks: list[tuple[str, int]] = []
    if s.participant and s.workflow == "structure":
        return [("structures", len(s.structures))]
    if s.strategy in ("focal", "factorial", "time") and s.include_main or s.strategy == "main":
        blocks.append(("main", sum(comb(p, k) for k in _sizes(s))))
    if s.strategy == "combinations":
        drop = 0 if s.include_main else 1
        blocks.append(("combinations", sum(comb(p, k) * (n_interaction_sets(k, s.max_order) - drop) for k in _sizes(s))))
    elif s.strategy == "factorial":
        blocks.append(("factorial", sum(comb(p, k) for k in _sizes(s, 2))))
    elif s.strategy == "focal":
        blocks.append(("focal", _count_focal(s)))
    elif s.strategy == "time":
        blocks.append(("time", _count_time(s)))
    return blocks


def _count_time(s: SearchSettings) -> int:
    p, room = len(s.candidates), _room(s)
    orders = [o for o in (2, 3) if o <= s.max_order and o <= room]
    if s.covariate_mode == "focal":
        return sum(comb(p, o - 1) for o in orders)
    return sum(comb(p, k - 1) * sum(comb(k - 1, o - 1) for o in orders if o <= k) for k in range(2, room + 1))


def _enum_time(s: SearchSettings, idx: int, start: int) -> Iterator[Candidate]:
    """Time × predictor focal interactions (M §8.5): focal sets always contain the time variable."""
    names, req, t, room = s.candidates, tuple(s.required), s.time, _room(s)
    orders = [o for o in (2, 3) if o <= s.max_order and o <= room]
    if s.covariate_mode == "focal":
        for o in orders:
            for f in combinations(names, o - 1):
                if idx >= start:
                    yield Candidate(idx, (t,) + f, (), req)
                idx += 1
        return
    for k in range(2, room + 1):
        for rest in combinations(names, k - 1):
            for o in orders:
                if o > k:
                    continue
                for f in combinations(rest, o - 1):
                    if idx >= start:
                        yield Candidate(idx, (t,) + f, tuple(x for x in rest if x not in f), req)
                    idx += 1


def _enum_structures(s: SearchSettings, idx: int, start: int) -> Iterator[Candidate]:
    """Workflow B: one locked fixed part, one candidate per random-effect / covariance structure."""
    inter = tuple((s.time, x) for x in s.candidates) if s.time_interactions and s.time else ()
    for st in s.structures:
        if idx >= start:
            yield Candidate(idx, (), tuple(s.candidates), tuple(s.required), "structure", inter, st)
        idx += 1


def count_candidates(s: SearchSettings) -> int:
    return sum(n for _, n in _block_counts(s))


def _count_focal(s: SearchSettings) -> int:
    p, r = len(s.candidates), len(s.required)
    orders = _orders(s)
    if s.covariate_mode == "focal":
        return sum(comb(p, o) for o in orders)
    extra = lambda size: 1 if s.all_two_way and size >= 3 else 0  # noqa: E731
    return sum(comb(p, size) * (sum(comb(size, o) for o in orders) + extra(size))
               for size in range(2, s.max_predictors - r + 1))


def enumerate_candidates(s: SearchSettings, start: int = 0) -> Iterator[Candidate]:
    """Deterministic order: the main-effects block (when included), then the strategy's block.
    Resuming from `start` skips whole blocks and subsets by their exact counts."""
    offset = 0
    for name, n in _block_counts(s):
        if start < offset + n:
            gen = {"main": _enum_main, "combinations": _enum_combinations, "factorial": _enum_factorial,
                   "focal": _enum_focal, "time": _enum_time, "structures": _enum_structures}[name]
            yield from gen(s, offset, max(start, offset))
        offset += n


def _enum_main(s: SearchSettings, idx: int, start: int) -> Iterator[Candidate]:
    names, req = s.candidates, tuple(s.required)
    for size in _sizes(s):
        n = comb(len(names), size)
        if idx + n <= start:
            idx += n
            continue
        for sub in combinations(names, size):
            if idx >= start:
                yield Candidate(idx, (), sub, req, "main")
            idx += 1


def _enum_combinations(s: SearchSettings, idx: int, start: int) -> Iterator[Candidate]:
    from .modelspec import interaction_sets, n_interaction_sets

    names, req = s.candidates, tuple(s.required)
    for size in _sizes(s):
        per = n_interaction_sets(size, s.max_order) - (0 if s.include_main else 1)
        n = comb(len(names), size) * per
        if idx + n <= start:
            idx += n
            continue
        for sub in combinations(names, size):
            if idx + per <= start:
                idx += per
                continue
            for inter in interaction_sets(sub, s.max_order):
                if not inter and not s.include_main:
                    continue
                if idx >= start:
                    yield Candidate(idx, (), sub, req, "combo" if inter else "main", inter)
                idx += 1


def _factorial_terms(sub: tuple[str, ...], max_order: int) -> tuple[tuple[str, ...], ...]:
    return tuple(c for k in range(2, min(max_order, len(sub)) + 1) for c in combinations(sub, k))


def _enum_factorial(s: SearchSettings, idx: int, start: int) -> Iterator[Candidate]:
    names, req = s.candidates, tuple(s.required)
    for size in _sizes(s, 2):
        n = comb(len(names), size)
        if idx + n <= start:
            idx += n
            continue
        for sub in combinations(names, size):
            if idx >= start:
                yield Candidate(idx, (), sub, req, "factorial", _factorial_terms(sub, s.max_order))
            idx += 1


def _enum_focal(s: SearchSettings, base: int, start: int) -> Iterator[Candidate]:
    """v1.1 grammar. Focal mode: all pairs, then all triples (lexicographic). Subset mode: by subset
    size, then subset (lexicographic), then focal pairs, then triples."""
    names, req = s.candidates, tuple(s.required)
    orders = _orders(s)
    idx = base

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
            if s.all_two_way and size >= 3:
                if idx >= start:
                    yield Candidate(idx, tuple(names[i] for i in subset), (), req, "all_two_way")
                idx += 1


# ---- formulas ---------------------------------------------------------------------

def hierarchy(focal: tuple[str, ...]) -> list[tuple[str, ...]]:
    """All non-empty subsets of the focal variables, lower orders first."""
    return [c for k in range(1, len(focal) + 1) for c in combinations(focal, k)]


def _mains(c: Candidate) -> list[tuple[str, ...]]:
    return [(v,) for v in c.covariates]


def model_terms(c: Candidate) -> tuple[list[tuple[str, ...]], list[tuple[str, ...]]]:
    """(full terms, reduced terms of the declared focal test) as tuples of variable names."""
    if c.kind == "structure":
        return full_terms(c), []
    if c.kind in ("main", "combo", "factorial"):
        full = full_terms(c)
        return full, candidate_tests(c)[0]["reduced"]
    if c.kind == "all_two_way":
        mains = [(v,) for v in c.focal]
        req = [(v,) for v in c.required]
        return mains + list(combinations(c.focal, 2)) + req, mains + req
    full = hierarchy(c.focal) + [(v,) for v in c.covariates + c.required]
    reduced = [t for t in full if t != c.focal]
    return full, reduced


def full_terms(c: Candidate) -> list[tuple[str, ...]]:
    if c.kind == "structure":
        mains = _mains(c) + [(v,) for v in c.required]
        extra = [(t[0],) for t in c.inter[:1] if (t[0],) not in mains]     # time main effect for time x X
        return mains + extra + list(c.inter)
    if c.kind in ("main", "combo", "factorial"):
        return _mains(c) + list(c.inter) + [(v,) for v in c.required]
    return model_terms(c)[0]


def candidate_tests(c: Candidate) -> list[dict[str, Any]]:
    """The typed hypotheses a candidate tests, the declared focal test first. Each is a nested
    comparison of the full model with `reduced` (= full minus `drop`). Main effects of variables
    in an interaction are not tested (marginality); interaction terms contained in a higher-order
    term are not dropped alone (hierarchy)."""
    full = full_terms(c)
    req = [(v,) for v in c.required]
    if c.kind == "structure":
        return []          # structures are compared with each other after the run, never by a focal p-value
    if c.kind == "focal":
        return [{"id": 0, "type": "focal_interaction", "drop": [c.focal], "reduced": model_terms(c)[1]}]
    if c.kind == "all_two_way":
        return [{"id": 0, "type": "joint_interaction", "drop": list(combinations(c.focal, 2)),
                 "reduced": model_terms(c)[1]}]
    tests: list[dict[str, Any]] = []
    inter = list(c.inter)
    mains = _mains(c)
    if inter:
        top = [t for t in inter if not any(set(t) < set(u) for u in inter)]
        if len(inter) > 1:
            tests.append({"type": "joint_interaction", "drop": inter, "reduced": mains + req})
        for t in top:
            tests.append({"type": "interaction_term", "drop": [t], "reduced": [u for u in full if u != t]})
    tests.append({"type": "overall", "drop": mains + inter, "reduced": req})
    if not inter and len(mains) > 1:
        for m in mains:
            tests.append({"type": "main_term", "drop": [m], "reduced": [u for u in full if u != m]})
    for i, t in enumerate(tests):
        t["id"] = i
    return tests


def model_type(c: Candidate) -> str:
    if c.kind == "structure":
        return "Covariance structure: " + STRUCTURES[c.structure][0].split(" — ")[0]
    if c.kind == "focal":
        return MODEL_TYPES["focal_cov" if c.covariates else "focal"]
    if c.kind == "all_two_way":
        return MODEL_TYPES["all_two_way"]
    if c.kind == "factorial":
        return MODEL_TYPES["factorial"]
    n = len(c.inter)
    return MODEL_TYPES["main" if n == 0 else "one" if n == 1 else "multi"]


def candidate_key(c: Candidate) -> str:
    if c.kind == "structure":
        from .modelspec import canonical_key

        return f"structure:{c.structure}|{canonical_key(full_terms(c))}"
    if c.kind in ("main", "combo", "factorial"):
        from .modelspec import canonical_key

        return f"{c.kind}|{canonical_key(full_terms(c))}"
    full, _ = model_terms(c)
    terms = sorted(":".join(sorted(t)) for t in full)
    prefix = "ALL2|" if c.kind == "all_two_way" else ""
    return f"{prefix}{':'.join(sorted(c.focal))}|{'+'.join(terms)}"


def display_formula(ds: Dataset, outcome: str, c: Candidate) -> str:
    dn = lambda n: ds.var(n).display_name  # noqa: E731
    if c.kind == "structure":
        from .modelspec import display

        return display(full_terms(c), dn(outcome), dn, order=list(c.covariates) + list(c.required))
    if c.kind in ("main", "combo", "factorial"):
        from .modelspec import display

        body = display(_mains(c) + list(c.inter), dn(outcome), dn, order=list(c.covariates),
                       extra=[dn(v) for v in c.required])
        return body
    if c.kind == "all_two_way":
        parts = ["(" + " + ".join(dn(v) for v in c.focal) + ")^2"] + [dn(v) for v in c.required]
        return f"{dn(outcome)} ~ {' + '.join(parts)}"
    parts = [" * ".join(dn(v) for v in c.focal)] + [dn(v) for v in c.covariates + c.required]
    return f"{dn(outcome)} ~ {' + '.join(parts)}"


def term_label(ds: Dataset, t: tuple[str, ...]) -> str:
    return " × ".join(ds.var(v).display_name for v in t)


def test_label(ds: Dataset, c: Candidate, t: dict[str, Any]) -> str:
    """What one hypothesis tests, in words."""
    if t["type"] in ("focal_interaction", "interaction_term", "main_term"):
        return term_label(ds, t["drop"][0])
    if t["type"] == "joint_interaction":
        return "all interactions jointly: " + ", ".join(term_label(ds, d) for d in t["drop"])
    base = ", ".join(ds.var(v).display_name for v in c.required)
    return "whole model vs " + (f"required covariates only ({base})" if base else "intercept only")


def focal_label(ds: Dataset, c: Candidate) -> str:
    if c.kind == "structure":
        return STRUCTURES[c.structure][0]
    if c.kind == "all_two_way":
        return "all two-way: " + ", ".join(ds.var(v).display_name for v in c.focal)
    if c.kind in ("main", "combo", "factorial"):
        if not c.inter:
            return "main effects: " + ", ".join(ds.var(v).display_name for v in c.covariates)
        if c.kind == "factorial":
            return "full factorial: " + " × ".join(ds.var(v).display_name for v in c.covariates)
        return ", ".join(term_label(ds, t) for t in c.inter)
    return " × ".join(ds.var(v).display_name for v in c.focal)


def random_part(ds: Dataset, s: SearchSettings, structure: str | None = None) -> str:
    """The grouping / covariance part of a longitudinal model, in R notation with display names."""
    if not s.participant:
        return ""
    g = ds.var(s.participant).display_name
    t = ds.var(s.time).display_name if s.time else "time"
    st = structure or s.structure or "ri"
    return {"ri": f"(1 | {g})", "rs": f"(1 + {t} | {g})", "rs_uncorr": f"(1 + {t} || {g})",
            "nlme_ri": f"random = ~1 | {g}", "nlme_rs": f"random = ~1 + {t} | {g}",
            "ar1": f"random = ~1 | {g}, corAR1(~ occasion | {g})", "car1": f"random = ~1 | {g}, corCAR1(~ {t} | {g})",
            "mmrm_us": f"us({t} | {g})", "mmrm_cs": f"cs({t} | {g})", "mmrm_ar1": f"ar1({t} | {g})",
            "glmer_ri": f"(1 | {g})", "glmer_rs": f"(1 + {t} | {g})", "clmm_ri": f"(1 | {g})",
            "clmm_rs": f"(1 + {t} | {g})", "gee_exch": f"GEE id = {g}, exchangeable",
            "gee_ar1": f"GEE id = {g}, AR(1)", "gee_ind": f"GEE id = {g}, independence"}.get(st, "")


def shared_sample_size(ds: Dataset, s: SearchSettings) -> int:
    ds = view(ds, s)
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
    ds = view(ds, s)
    total = count_candidates(s)
    warnings = []
    if total >= LARGE_SEARCH:
        warnings.append(f"{total:,} candidate models: an exhaustive search will take a long time. "
                        "Consider fewer candidates, a smaller maximum, two-way interactions only, or the "
                        "single-focal-interaction strategy.")
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
    if family in ("ordinal", "multinomial"):
        k = len(ds.analysis_frame([s.outcome])[s.outcome].dropna().unique())
        warnings.append(f"{FAMILIES[family].split(' (')[0]}: each model estimates "
                        + ("one coefficient per term plus " + str(k - 1) + " thresholds" if family == "ordinal" else
                           f"{k - 1} coefficients per term (one per non-reference category)")
                        + "; the parameter limit applies to the total, so fewer predictors fit.")
    return {"family": family, "total": total, "n_shared": n_shared, "n_rows": n_all, "warnings": warnings,
            "coding": coding(ds, s, family), "blocks": _block_counts(s), "fits": estimate_fits(s, total)}


def estimate_fits(s: SearchSettings, total: int, sample: int = 2000) -> int:
    """Upper bound on R model fits: each candidate's full model plus one reduced model per planned
    test (reduced fits shared between candidates are cached on the shared sample, so the real number
    is usually lower). Averaged over the first `sample` candidates."""
    from itertools import islice

    first = list(islice(enumerate_candidates(s), sample))
    if not first:
        return 0
    per = sum(1 + len(candidate_tests(c)) for c in first) / len(first)
    return int(round(per * total))


def binomial_levels(ds: Dataset, s: SearchSettings) -> tuple[Any, Any]:
    """(event, reference) codes. Default event = the second category in R's level order
    (after a reference level is moved first), exactly as in v1.0."""
    v = ds.var(s.outcome)
    observed = set(ds.analysis_frame([s.outcome])[s.outcome].dropna().unique().tolist())
    if v.is_categorical:
        levels = [c for c in ds.levels(s.outcome) if c in observed]
        if v.reference_level in levels:
            levels = [v.reference_level] + [c for c in levels if c != v.reference_level]
    else:
        levels = sorted(observed)
    event = s.event if s.event is not None else levels[1]
    return event, next(c for c in levels if c != event)


def coding(ds: Dataset, s: SearchSettings, family: str) -> dict[str, Any]:
    """How the outcome and predictors are coded in this search (§10.4), recorded with the run.
    `ds` must be the search view."""
    from ..analysis.measurement import treatment_records

    v = ds.var(s.outcome)
    base = {"lmm": "gaussian", "glmm": "binomial", "clmm": "ordinal", "gee": s.family or "gaussian"}.get(family, family)
    fam_label = FAMILIES.get(base, base)
    if s.participant:
        fam_label = f"{LONG_FAMILIES.get(family, family)}; outcome: {FAMILIES.get(base, base).split(' (')[0]}"
    out: dict[str, Any] = {"family": fam_label,
                           "focal_test": FOCAL_TESTS.get(family, ""),
                           "outcome": v.display_name, "outcome_level": v.measure.value,
                           "predictor_coding": "categorical predictors: treatment (dummy) contrasts against the "
                                               "reference category (first category unless one is set)",
                           "treatments": [r["description"] for r in treatment_records(ds)]}
    if s.participant:
        out["structure"] = (", ".join(STRUCTURES[x][0] for x in s.structures) if s.workflow == "structure"
                            else STRUCTURES[s.structure or "ri"][0])
        out["participant"] = ds.var(s.participant).display_name
        if s.time:
            out["time"] = f"{ds.var(s.time).display_name} ({s.time_type})"
        out["workflow"] = ("B: random-effect / covariance-structure comparison (fixed part locked; REML for Gaussian)"
                           if s.workflow == "structure" else "A: fixed-effect discovery (structure locked; ML fits)")
        if family == "gee":
            out["estimand"] = "population-average (marginal) effects; robust sandwich standard errors"
        elif family in ("glmm", "clmm"):
            out["estimand"] = "subject-specific (conditional on the random effects) effects"
    if base == "binomial":
        event, ref = binomial_levels(ds, s)
        out["event"] = ds.level_label(s.outcome, event)
        out["reference"] = ds.level_label(s.outcome, ref)
        out["event_source"] = "chosen by the user" if s.event is not None else "second category (default)"
    elif base == "ordinal":
        out["ordering"] = [ds.level_label(s.outcome, c) for c in ds.levels(s.outcome)]
        if v.coding_direction:
            out["direction"] = v.coding_direction
    elif family == "multinomial":
        ref = s.reference if s.reference is not None else ds.levels(s.outcome)[0]
        out["reference"] = ds.level_label(s.outcome, ref)
    if any(t.get("variable") == s.outcome for t in s.treatments):
        out["outcome_treatment"] = next(r["description"] for r in treatment_records(ds) if r["variable"] == s.outcome)
    return out
