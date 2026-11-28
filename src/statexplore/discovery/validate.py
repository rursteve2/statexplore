"""AMD validation (v1.3.3, 02 §2; V133-VAL-01): holdout, grouped K-fold CV (optionally repeated) and cluster-aware
bootstrap stability for a discovery search, with an explicit prediction target.

Design (J7):
- Splits are made in Python, BEFORE anything is fitted, from a seeded numpy PCG64 generator over the units of the target
  (rows, participants, clusters, or participants within clusters) and are saved as a manifest keyed by stable source-row
  numbers (R rownames) and composite unit keys. The same manifest is used for every candidate. Splits never depend on
  fitted performance.
- Each training part runs either the fixed candidate (conditional / post-selection assessment of a model chosen on all
  data) or the WHOLE selection pipeline (every candidate of the search fitted on the training part only, then the
  prespecified rule picks one): only the latter evaluates the search procedure honestly.
- R fits with the search's own model helpers (same formulas, structures, estimator and grouping), predicts the held-out
  rows for the declared target without using their outcomes, and returns per-row predictions; Python computes metrics
  (observation- and group-weighted), keeps failed folds with reasons and never imputes a failure.
- Bootstrap: resampled units get NEW synthetic participant (and, for new-cluster targets, cluster) IDs, so a unit drawn
  twice is two sampled units; the original IDs are kept for traceability.
"""
from __future__ import annotations

import csv
import io
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..core.dataset import Dataset
from ..rbridge.codegen import r_str, r_vec

GRAMMAR = "validation-4"   # v1.3.6: execution ledger (plan / committed), exact-budget status, typed evidence schema 1
# validation-3 = v1.3.5: declared pool scopes, saved-specification fixed candidates, hard fit budget
# validation-2 = v1.3.4: a-priori pool, replayed stopping, per-sample caches, identity-keyed stability

TARGETS = {
    "rows": ("New independent observations",
             "rows are the sampling units (only for searches without repeated measures)"),
    "participant": ("New participants",
                    "all rows of a participant are in the same part; predictions use fixed effects only (the new "
                    "participant's random effects are unknown, i.e. zero)"),
    "cluster": ("New clusters (e.g. new schools)",
                "whole clusters with all their participants are held out; predictions use fixed effects only"),
    "participant_in_cluster": ("New participants in existing clusters",
                               "participants are split WITHIN each cluster, so every evaluated cluster also has training "
                               "participants; predictions use the cluster effect estimated from training participants "
                               "and a zero effect for the new participant (conditional on the cluster)"),
}
MODES = {"holdout": "Holdout", "cv": "Grouped K-fold cross-validation", "bootstrap": "Bootstrap stability"}
PIPELINES = {"pipeline": "Whole selection pipeline (search repeated inside each training part)",
             "fixed": "Fixed candidate (conditional on a model chosen on all data: post-selection, not independent)"}
RULES = {"p": "Smallest training p-value (natural-log tail) of the candidate's primary test (the search's discovery "
              "ranking); exact ties: lower candidate index",
         "aic": "Lowest training AIC (same likelihood basis for every candidate)"}
SUPPORTED_FAMILIES = {"gaussian", "binomial", "lmm", "gee"}


class ValidationError(ValueError):
    pass


@dataclass
class ValidationConfig:
    target: str = "participant"
    modes: list[str] = field(default_factory=lambda: ["holdout", "cv"])
    pipeline: str = "pipeline"
    candidate: int | None = None          # fixed mode: candidate index (row "idx")
    rule: str = "p"
    holdout_fraction: float = 0.2
    k: int = 5
    repeats: int = 1
    B: int = 200
    seed: int = 133
    max_fits: int = 20000                 # HARD budget of model-fitting invocations (AUD134-10; see `workload`)
    # AUD134-06 candidate-pool scope: "cumulative" = every declared execution budget of the search (default; = the whole
    # declared space unless the search is budget-limited, mode D), "original" = the first execution's budget only,
    # "prefix" = an explicit first `prefix_n` (restricted). "declared" is the v1.3.4 name of "original".
    pool: str = "cumulative"
    prefix_n: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ValidationConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---- units and feasibility ---------------------------------------------------------------------------------------

def default_target(run: Any) -> str:
    s = run.settings
    if not s.participant:
        return "rows"
    return "cluster" if getattr(s, "cluster", None) else "participant"


def allowed_targets(run: Any) -> dict[str, str]:
    """Targets valid for this search, else the reason (row splits are never offered for repeated data)."""
    s = run.settings
    out = {}
    for t in TARGETS:
        if t == "rows":
            out[t] = "" if not s.participant else "repeated measures: rows of a participant are never split at random"
        elif t == "participant":
            out[t] = "" if s.participant else "the search has no participant ID"
        else:
            out[t] = "" if s.participant and getattr(s, "cluster", None) else "the search has no higher-level cluster"
    return out


def unit_frame(ds: Dataset, run: Any) -> pd.DataFrame:
    """Every analysed row with a known unit: row_id (R rowname = source position + 1), participant key and cluster key
    as typed composite strings. Rows without an ID / cluster are not part of any model and have no unit."""
    from .space import view

    s = run.settings
    v = view(ds, s)
    cl = getattr(s, "cluster", None) if s.participant else None
    cols = [c for c in dict.fromkeys([s.outcome] + ([s.participant] if s.participant else []) + ([cl] if cl else []))]
    f = v.analysis_frame(cols)
    pos = pd.Series(np.arange(len(v.raw.index)), index=v.raw.index)
    out = pd.DataFrame({"row_id": (pos.loc[f.index].to_numpy() + 1).astype(int)})
    if cl:
        ok = f[cl].notna().to_numpy() & f[s.participant].notna().to_numpy()
        out["cluster"] = [repr(("c", x)) for x in f[cl]]
        out["participant"] = [repr(("c", c, "id", i)) for c, i in zip(f[cl], f[s.participant])]
        out = out[ok]
    elif s.participant:
        ok = f[s.participant].notna().to_numpy()
        out["participant"] = [repr(("id", i)) for i in f[s.participant]]
        out["cluster"] = ""
        out = out[ok]
    else:
        out["participant"] = out["row_id"].astype(str)
        out["cluster"] = ""
    return out.reset_index(drop=True)


def _unit_col(target: str) -> str:
    return "cluster" if target == "cluster" else "participant"


def feasibility(ds: Dataset, run: Any, cfg: ValidationConfig) -> dict[str, Any]:
    """Actual group counts and every reason the configuration cannot run (checked before any split is made)."""
    s = run.settings
    problems: list[str] = []
    warnings: list[str] = []
    why = allowed_targets(run).get(cfg.target, "unknown target")
    if why:
        problems.append(f"{TARGETS.get(cfg.target, (cfg.target,))[0]}: not available ({why}).")
    if run.family not in SUPPORTED_FAMILIES:
        problems.append(f"Validation predictions for {run.family} searches are not implemented: a marginal prediction "
                        "for a nonlinear mixed model needs integration over the random effects, and the inverse link "
                        "of the fixed part is not that prediction. Reported as unavailable instead of approximated.")
    if s.participant and (s.structure or "").startswith("mmrm") and s.workflow != "structure":
        problems.append("Marginal mmrm searches are not validated (no prediction for new participants is implemented).")
    if s.workflow == "structure" and cfg.rule == "p":
        problems.append("Workflow B compares covariance structures by information criteria: choose the AIC rule.")
    if cfg.rule == "aic" and run.family == "gee":
        problems.append("GEE has no likelihood, so there is no AIC: choose the p-value rule (robust Wald test). QIC-based "
                        "selection inside validation is not implemented (V134-005/006).")
    if cfg.rule == "aic" and getattr(s, "sample", "shared") != "shared" and cfg.pipeline == "pipeline":
        problems.append("The AIC rule needs every candidate fitted to the same training rows; this search uses per-model "
                        "complete cases (V134-006). Use the shared sample or the p-value rule.")
    scope = pool_scope(run, cfg) if cfg.pipeline == "pipeline" else {"problems": [], "notes": [], "adaptive": False}
    problems += scope["problems"]
    if cfg.pipeline == "fixed":
        try:
            fixed_candidate(run, cfg.candidate)
        except ValidationError as e:
            problems.append(str(e))
    elif getattr(run, "grammar_problem", None) and run.grammar_problem():
        problems.append("The declared search cannot be repeated inside the training parts: " + run.grammar_problem()
                        + ". Start a new search (it uses the current grammar), or validate a fixed candidate from its "
                          "saved specification.")
    if not set(cfg.modes) <= set(MODES) or not cfg.modes:
        problems.append("Choose at least one validation mode.")
    if not 0.05 <= cfg.holdout_fraction <= 0.5:
        problems.append("The holdout fraction must be between 5% and 50%.")
    if cfg.k < 2 or cfg.repeats < 1 or cfg.B < 1:
        problems.append("K must be at least 2; repeats and bootstrap draws at least 1.")
    info: dict[str, Any] = {"problems": problems, "warnings": warnings}
    if why:
        return info
    u = unit_frame(ds, run)
    info["rows"] = int(len(u))
    info["participants"] = int(u["participant"].nunique())
    info["clusters"] = int(u["cluster"].replace("", np.nan).nunique()) if s.participant and getattr(s, "cluster", None) else 0
    col = _unit_col(cfg.target)
    n_units = int(u[col].nunique()) if cfg.target != "rows" else int(len(u))
    info["units"] = n_units
    if cfg.target == "participant_in_cluster":
        per = u.groupby("cluster")["participant"].nunique()
        eligible = per[per >= 2]
        info["eligible_clusters"] = int(len(eligible))
        info["ineligible_clusters"] = sorted(map(str, per[per < 2].index))
        if len(eligible) == 0:
            problems.append("No cluster has 2 or more participants: new participants within existing clusters cannot "
                            "be evaluated (a cluster needs training participants). This target is not replaced by "
                            "another.")
        elif per[per >= 2].min() < cfg.k and "cv" in cfg.modes:
            warnings.append(f"Some clusters have fewer participants than K = {cfg.k}: in some folds they contribute "
                            "training rows only.")
    min_units = {"cluster": 2 * cfg.k if "cv" in cfg.modes else 5, "participant": 2 * cfg.k if "cv" in cfg.modes else 10,
                 "rows": 2 * cfg.k, "participant_in_cluster": 2}[cfg.target]
    if n_units < min_units:
        problems.append(f"Only {n_units} units for this target (need at least {min_units} for the chosen settings).")
    if "cv" in cfg.modes and cfg.k > n_units:
        problems.append(f"K = {cfg.k} exceeds the {n_units} units of this target.")
    if cfg.target == "cluster" and n_units < 20:
        warnings.append(f"Only {n_units} clusters: held-out performance for new clusters is imprecise and training "
                        "parts estimate the cluster variance from very few groups.")
    n_cand = 1 if cfg.pipeline == "fixed" else pool_size(run, cfg)
    info["pool"] = int(n_cand)
    info["pool_label"] = pool_label(run, cfg) if cfg.pipeline == "pipeline" else "fixed candidate"
    info["pool_scope"] = scope.get("scope", "fixed") if cfg.pipeline == "pipeline" else "fixed"
    info["adaptive"] = bool(scope.get("adaptive"))
    if scope.get("notes"):
        warnings += [n[0].upper() + n[1:] + "." for n in scope["notes"]]
    # AUD134-10: the conservative bound (never the optimistic estimate) is compared with the hard budget, computed
    # WITHOUT materializing a large candidate list
    wl = workload(run, cfg)
    info["workload"] = wl
    info["fits"] = wl["bound"]
    info["fits_estimate"] = wl["estimate"]
    info["candidate_fits"] = int(n_cand * wl["samples"])
    if cfg.max_fits < 1:
        problems.append("The fit budget must be at least 1 model-fitting invocation.")
    elif wl["bound"] > cfg.max_fits:
        problems.append(f"At most {wl['bound']:,} model-fitting invocations ({n_cand:,} candidates x {wl['samples']:,} "
                        f"training parts, with their reduced models and optimizer retries; conservative bound, "
                        f"{wl['method']}) exceed the budget of {cfg.max_fits:,}: reduce K, repeats or B, restrict the "
                        "pool explicitly to the first N candidates (labelled restricted), use fixed-candidate mode, or "
                        "raise the budget.")
    return info


def stop_target(run: Any) -> int | None:
    """The search's own outcome-dependent stopping rule, replayed inside every training part (V134-004)."""
    s = run.settings
    if s.participant and s.workflow == "structure":
        return None
    return 1 if s.mode == "A" else int(s.target) if s.mode == "B" else None


POOL_SCOPES = {"cumulative": "Complete declared procedure (every execution's model budget)",
               "original": "Original execution only (its declared model budget)",
               "prefix": "Explicit first N candidates (restricted)"}


def declared_budgets(run: Any) -> list[dict[str, Any]]:
    """The model budget each execution of the search DECLARED before it ran (never what it happened to evaluate)."""
    ex = list(getattr(run, "executions", None) or [])
    if not ex:
        ex = [{"start": 0, "max_models": run.settings.max_models, "max_minutes": run.settings.max_minutes}]
    return ex


def pool_scope(run: Any, cfg: ValidationConfig) -> dict[str, Any]:
    """AUD134-06: the a-priori candidate pool of a validation, from immutable declared quantities only: the grammar's
    exact count, the declared per-execution model budgets (mode D), or an explicit prefix. Never the rows evaluated, the
    statuses, observed p-values, the stop index or the number of significant findings. {n, label, problems, notes,
    adaptive}; `n` is None when the scope cannot be specified a priori (then an explicit prefix is required)."""
    total, s = int(run.total), run.settings
    scope = "original" if cfg.pool == "declared" else cfg.pool
    out: dict[str, Any] = {"scope": scope, "n": total, "problems": [], "notes": [], "adaptive": False}
    if scope == "prefix":
        if not (cfg.prefix_n or 0) >= 1:
            out["problems"].append("Restricted pool: give the number of leading candidates (at least 1).")
            out["n"] = None
            return out
        out["n"] = max(0, min(total, int(cfg.prefix_n)))
        out["label"] = (f"RESTRICTED pool: the first {out['n']:,} of {total:,} candidates in the declared order "
                        "(outcome-independent restriction; results are conditional on it)")
        return out
    if scope not in ("original", "cumulative"):
        out["problems"].append(f"Unknown candidate-pool scope {cfg.pool!r}.")
        out["n"] = None
        return out
    ex = declared_budgets(run)
    if s.mode != "D":
        out["label"] = f"the whole declared search space ({total:,} candidates)"
        if len(ex) > 1:
            out["notes"].append(f"the search ran in {len(ex)} executions; its declared procedure is the whole space "
                                f"(mode {s.mode}), which the validation uses whatever the budgets of the executions were")
        return out
    used = ex[:1] if scope == "original" else ex
    budgets = [e.get("max_models") for e in used]
    if any(not b for b in budgets):
        out["problems"].append("This budget-limited search (mode D) has an execution without a model budget (time limit "
                               "only): a time limit cannot be replayed on training data, so the declared pool is not "
                               "defined. Give an explicit number of leading candidates (restricted pool).")
        out["n"] = None
        return out
    n = min(total, int(sum(int(b) for b in budgets)))
    out["n"] = n
    out["adaptive"] = scope == "cumulative" and len(ex) > 1
    if scope == "original":
        out["label"] = (f"ORIGINAL declared model budget: the first {n:,} of {total:,} candidates (the first execution "
                        "only" + (f"; the search continued {len(ex) - 1} time(s) - use the cumulative scope for the "
                                  "complete procedure" if len(ex) > 1 else "") + ")")
    else:
        out["label"] = (f"CUMULATIVE declared model budgets: the first {n:,} of {total:,} candidates ("
                        + " + ".join(f"{int(b):,}" for b in budgets) + (f", capped at the {total:,} candidates of the "
                                                                          "space" if sum(budgets) > total else "") + ")")
    if out["adaptive"]:
        out["notes"].append("ADAPTIVE: continuation budgets were chosen after earlier results were seen, so validating "
                            "this frozen prefix is a conditional validation, not an unbiased validation of the adaptive "
                            "procedure")
    for i, e in enumerate(used, start=1):
        if "end" in e and e.get("max_models") and e["end"] - e.get("start", 0) < e["max_models"] and e["end"] < total:
            out["notes"].append(f"execution {i} stopped (time limit or cancellation) after {e['end'] - e.get('start', 0):,}"
                                f" of its {int(e['max_models']):,} declared models; time and cancellation cannot be "
                                "replayed, so the declared budget is used (deterministic frozen-prefix validation)")
    return out


def pool_size(run: Any, cfg: ValidationConfig) -> int:
    """Candidates of the a-priori pool (see `pool_scope`); 0 when the scope is not defined."""
    return int(pool_scope(run, cfg)["n"] or 0)


def pool_label(run: Any, cfg: ValidationConfig) -> str:
    sc, s = pool_scope(run, cfg), run.settings
    stop = stop_target(run)
    base = sc.get("label") or "candidate pool not defined"
    if sc["notes"]:
        base += " — " + "; ".join(sc["notes"])
    if stop:
        base += (f"; the search's stopping rule (stop at the {stop} candidate(s) with raw p < {s.alpha:g}) is replayed "
                 "on the training data of every part")
    return base


def fixed_candidate(run: Any, index: Any) -> Any:
    """AUD134-05 / AUD134-08: the fixed candidate from its SAVED specification (focal, covariates, kind, interaction
    terms, structure), verified against its saved canonical key - never a bare index into the current enumerator, and no
    enumeration at all (constant cost whatever the size of the space)."""
    from .space import Candidate, candidate_key

    if index is None:
        raise ValidationError("Choose the candidate to validate (fixed-candidate mode).")
    i = int(index)
    row = next((r for r in run.rows if int(r.get("idx", -1)) == i), None)
    if row is None:
        raise ValidationError(f"Candidate #{i} has no saved result in this search: a fixed-candidate validation needs the "
                              "candidate's saved specification.")
    key = row.get("key")
    if not key:
        raise ValidationError(f"Candidate #{i} has no saved candidate key, so its identity cannot be verified: refused "
                              "(start a new search to validate it).")
    s = run.settings
    kind = row.get("kind") or "focal"
    clustered = None
    if kind == "structure":
        clustered = False if "@nocluster" in str(key) else (True if getattr(s, "cluster_compare", False) else None)
    c = Candidate(i, tuple(row.get("focal") or ()), tuple(row.get("covariates") or ()), tuple(s.required), kind,
                  tuple(tuple(x) for x in (row.get("inter") or ())), str(row.get("structure") or "") if kind == "structure"
                  else "", clustered)
    if candidate_key(c) != key:
        raise ValidationError(f"Candidate #{i} cannot be rebuilt exactly from its saved specification (rebuilt key "
                              f"{candidate_key(c)!r}, saved key {key!r}): refused rather than validating another model.")
    return c


_EXACT_LIMIT = 20_000
_exact_cache: dict[tuple, int] = {}


def workload(run: Any, cfg: ValidationConfig, cands: list[Any] | None = None) -> dict[str, Any]:
    """AUD134-10: a CONSERVATIVE upper bound of the model-fitting invocations of a validation, the unit of `max_fits`:
    every training part / draw fits each pool candidate's full model and one reduced model per planned test with NO
    cache reuse, plus the selected model's refit; each fit may retry its optimizer (nlme: nlminb then optim = 2
    attempts), binary searches add the separation check and the null model, GEE adds the QIC independence refit. Exact
    per-candidate plans for a materialized or small pool (<= 20,000 candidates); otherwise the grammar's analytic maximum
    per candidate. The search's early stop and cache hits can only lower the actual count."""
    from itertools import islice

    from .space import candidate_fits, enumerate_candidates, estimate_fits, max_candidate_fits

    s = run.settings
    samples = ((1 if "holdout" in cfg.modes else 0) + (cfg.k * cfg.repeats if "cv" in cfg.modes else 0)
               + (cfg.B if "bootstrap" in cfg.modes else 0))
    retry = 1
    if run.family == "lmm":
        from ..analysis.covstruct import GAUSSIAN_STRUCTURES

        sts = s.structures if s.workflow == "structure" else [s.structure or "ri"]
        retry = max(2 if GAUSSIAN_STRUCTURES.get(st, ("", "", "lme4"))[2] == "nlme" else 1 for st in sts)
    aux = 1 if (run.family == "binomial" or run.family == "gee") else 0
    null = 1 if run.family in ("binomial", "ordinal", "multinomial") else 0
    if cfg.pipeline == "fixed":
        n_cand, plan, method = 1, 0, "fixed candidate: one refit per part, no candidate scoring"
    else:
        n_cand = pool_size(run, cfg)
        if cands is not None:
            plan, method = sum(candidate_fits(c) for c in cands), "exact plan of the materialized pool"
        elif n_cand <= _EXACT_LIMIT:
            key = (getattr(run, "id", ""), run.total, n_cand, repr(sorted(s.to_dict().items())))
            if key not in _exact_cache:
                _exact_cache[key] = sum(candidate_fits(c) for c in islice(enumerate_candidates(s), n_cand))
            plan, method = _exact_cache[key], "exact plan of the pool"
        else:
            plan, method = n_cand * max_candidate_fits(s), "analytic maximum per candidate (pool too large to enumerate)"
    tests = max(plan - n_cand, 0) if cfg.pipeline != "fixed" else 0
    per_sample = plan * retry + n_cand * aux + tests * null + retry          # + the selected model's refit
    bound = int(samples * per_sample)
    est_per = 1.0 if cfg.pipeline == "fixed" else max(1.0, estimate_fits(s, 1, sample=500))
    return {"bound": bound, "estimate": int(round(n_cand * est_per * samples + samples)), "samples": samples,
            "pool": n_cand, "plan_fits_per_sample": plan, "retry_factor": retry, "aux_per_candidate": aux,
            "null_per_test": null, "method": method,
            "unit": "model-fitting invocations (full, reduced, null and selected refits; each optimizer retry and each "
                    "auxiliary fit counts; cache hits are free and counted separately)"}


# ---- split manifests (Python, deterministic, before fitting) --------------------------------------------------------

def _rng(seed: int, stream: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64([int(seed), int(stream)]))


def _assign(units: list[str], n_test: int | None, k: int | None, rng: np.random.Generator) -> dict[str, int]:
    """Holdout (n_test units -> fold 1) or K folds (balanced, random order). Fold 0 = training only."""
    perm = list(rng.permutation(len(units)))
    if n_test is not None:
        return {units[i]: (1 if r < n_test else 0) for r, i in enumerate(perm)}
    return {units[i]: (r % k) + 1 for r, i in enumerate(perm)}


def make_splits(u: pd.DataFrame, cfg: ValidationConfig) -> list[dict[str, Any]]:
    """[{id, mode, repeat, folds: {row_id: fold}}]; fold 0 = never tested in this split."""
    out = []
    col = _unit_col(cfg.target)
    stream = 0

    def fold_rows(mapping: dict[str, int], key: str = col) -> dict[int, int]:
        return {int(r): int(mapping.get(k, 0)) for r, k in zip(u["row_id"], u[key])}

    def by_target(n_frac: float | None, k: int | None, rng: np.random.Generator) -> dict[int, int]:
        if cfg.target == "rows":
            units = [str(r) for r in u["row_id"]]
            n_test = max(1, math.ceil(n_frac * len(units))) if n_frac else None
            m = _assign(units, n_test, k, rng)
            return {int(r): m[str(r)] for r in u["row_id"]}
        if cfg.target == "participant_in_cluster":
            m: dict[str, int] = {}
            for c, grp in sorted(u.groupby("cluster")["participant"], key=lambda x: x[0]):
                ps = sorted(set(grp))
                if len(ps) < 2:
                    m.update({p: 0 for p in ps})            # ineligible: training only, never evaluated
                    continue
                n_test = min(len(ps) - 1, max(1, math.ceil(n_frac * len(ps)))) if n_frac else None
                m.update(_assign(ps, n_test, k, rng))
            return fold_rows(m, "participant")
        units = sorted(set(u[col]))
        n_test = max(1, math.ceil(n_frac * len(units))) if n_frac else None
        return fold_rows(_assign(units, n_test, k, rng))

    if "holdout" in cfg.modes:
        out.append({"id": "holdout", "mode": "holdout", "repeat": 1,
                    "folds": by_target(cfg.holdout_fraction, None, _rng(cfg.seed, stream))})
    stream += 1
    if "cv" in cfg.modes:
        for r in range(1, cfg.repeats + 1):
            out.append({"id": f"cv{r}", "mode": "cv", "repeat": r, "folds": by_target(None, cfg.k, _rng(cfg.seed, stream + r))})
    return out


def make_draws(u: pd.DataFrame, cfg: ValidationConfig) -> list[dict[str, Any]]:
    """Bootstrap draws: [{b, rows: [(row_id, draw_gid, draw_cluster, orig_participant, orig_cluster)]}]."""
    draws = []
    clustered = bool((u["cluster"] != "").any())
    by_cluster = {c: g for c, g in u.groupby("cluster")} if clustered else {}
    by_part = {p: g for p, g in u.groupby("participant")}
    for b in range(1, cfg.B + 1):
        rng = _rng(cfg.seed, 1000 + b)
        rows: list[tuple] = []
        j = 0
        if cfg.target == "rows":
            ids = u["row_id"].to_numpy()
            for r in ids[rng.integers(0, len(ids), len(ids))]:
                j += 1
                rows.append((int(r), f"d{j}", "", str(r), ""))
        elif cfg.target == "cluster":
            cl = sorted(by_cluster)
            for i, c in enumerate(rng.integers(0, len(cl), len(cl)), start=1):
                g = by_cluster[cl[c]]
                for p in sorted(set(g["participant"])):
                    j += 1
                    rows += [(int(r), f"d{j}", f"dc{i}", p, cl[c]) for r in g.loc[g["participant"] == p, "row_id"]]
        elif cfg.target == "participant_in_cluster":
            for c in sorted(by_cluster):
                g = by_cluster[c]
                ps = sorted(set(g["participant"]))
                for k in rng.integers(0, len(ps), len(ps)):
                    j += 1      # the cluster itself is kept (its existing ID): only participants are resampled
                    rows += [(int(r), f"d{j}", "", ps[k], c) for r in g.loc[g["participant"] == ps[k], "row_id"]]
        else:
            ps = sorted(by_part)
            for k in rng.integers(0, len(ps), len(ps)):
                j += 1
                g = by_part[ps[k]]
                rows += [(int(r), f"d{j}", "", ps[k], c) for r, c in zip(g["row_id"], g["cluster"])]
        draws.append({"b": b, "rows": rows, "units": j})
    return draws


def check_split(u: pd.DataFrame, split: dict[str, Any], target: str) -> list[str]:
    """Disjointness assertions on the manifest itself (also asserted again in R on the fitted rows)."""
    f = u.assign(fold=u["row_id"].map(split["folds"]))
    out = []
    for k in sorted(set(f["fold"]) - {0}):
        test, train = f[f["fold"] == k], f[f["fold"] != k]
        if set(test["row_id"]) & set(train["row_id"]):
            out.append(f"fold {k}: a row is in both parts")
        if target != "rows" and set(test["participant"]) & set(train["participant"]):
            out.append(f"fold {k}: a participant is in both parts")
        if target == "cluster" and set(test["cluster"]) & set(train["cluster"]):
            out.append(f"fold {k}: a cluster is in both parts")
        if target == "participant_in_cluster" and not set(test["cluster"]) <= set(train["cluster"]):
            out.append(f"fold {k}: a test cluster has no training participant")
    return out


# ---- R ---------------------------------------------------------------------------------------------------------------

R_VALIDATE = r"""
# ---- StatExplore AMD validation (v1.3.3): one split / bootstrap chunk per call ----
# test rows whose category levels never occur in the training part cannot be predicted: counted and excluded (reported)
sx_val_seen <- function(train, test, vars) {
  ok <- rep(TRUE, nrow(test))
  for (v in vars) if (is.factor(test[[v]])) ok <- ok & as.character(test[[v]]) %in% unique(as.character(train[[v]]))
  ok
}
sx_val_with <- function(a, expr_fun) {
  if (!is.null(a$struct)) { o <- sx_disc_struct; sx_disc_struct <<- a$struct; on.exit(sx_disc_struct <<- o) }
  if (!is.null(a$clustered)) { oc <- sx_disc_clustered; sx_disc_clustered <<- a$clustered
    on.exit(sx_disc_clustered <<- oc, add = TRUE) }
  expr_fun()
}
sx_val_need <- function(a) c("sx_y", a$vars, if (sx_long()) c("sx_gid", if (sx_disc_has_cluster) "sx_cluster",
                                                                if (sx_disc_has_time) c("sx_time", "sx_timef", "sx_occ")))
sx_val_fit <- function(a, d) sx_val_with(a, function() {
  d <- sx_complete(d, sx_val_need(a))
  if (sx_long() && sx_disc_has_time) d$sx_occ <- as.integer(d$sx_timef)
  sx_fit_model(stats::reformulate(if (length(a$full)) a$full else "1", response = "sx_y"), d)
})
sx_val_score <- function(a) {
  # V134-002 / AUD134-01: the p rule ranks the natural-log p of the primary test (sx_primary_logp, shared with the legacy
  # workspace); a budget stop is never a failed candidate: it aborts the training part
  r <- tryCatch(sx_val_with(a, function() do.call(sx_discover_one, a)),
                error = function(e) if (inherits(e, "sx_budget")) stop(e) else NULL)
  if (is.null(r) || !(r$status %in% c("ok", "warning"))) return(list(score = NA_real_, log_p = NA_real_, rows = "",
    status = if (is.null(r)) "failed" else r$status, reason = if (is.null(r)) "fit error" else r$reason,
    loglik = NA_real_, statistic = NA_real_, df = NA_real_, ref_loglik = NA_real_, flags = "", ref_flags = "", aic = NA_real_))
  t1 <- if (length(r$tests)) r$tests[[1]] else list()
  g <- function(x) if (is.null(x) || !length(x)) NA_real_ else as.numeric(x)[1]
  lp <- sx_primary_logp(r)
  # comparability key: the analysed row identities and the response values (SHA-256 fingerprints of the fit)
  rk <- if (is.null(r$identity)) r$rows_sig else paste(r$identity$rows, r$identity$response)
  list(score = if (sx_val_rule == "aic") r$aic else lp, log_p = lp, rows = rk, status = r$status, reason = "",
       loglik = g(r$loglik), statistic = g(t1$statistic), df = g(t1$df1), ref_loglik = g(t1$reference_loglik),
       flags = paste(r$flags, collapse = "; "), ref_flags = if (is.null(t1$reference_flags)) "" else t1$reference_flags,
       aic = g(r$aic))
}
sx_val_trace <- list()   # every candidate evaluated in every training part / draw (audit, V134-015)
# AUD134-05: the fitted object's fixed terms must be the candidate's terms (fail closed otherwise)
sx_val_check_terms <- function(m, a) {
  f <- tryCatch(if (inherits(m, "merMod")) stats::formula(m, fixed.only = TRUE) else stats::formula(m),
                error = function(e) NULL)
  if (is.null(f)) stop("the fitted model's terms could not be read back")
  got <- attr(stats::terms(f), "term.labels"); got <- got[!grepl("|", got, fixed = TRUE)]
  if (!setequal(sx_canon_names(got), sx_canon_names(a$full)))
    stop(sprintf("the fitted terms (%s) differ from the candidate's terms (%s)", paste(sort(got), collapse = " + "),
                 paste(sort(a$full), collapse = " + ")))
  invisible(TRUE)
}
# AUD135-05: `at_cap` = the cap was consumed (not an error); `denied` = fits the cap REFUSED (work truncated)
sx_val_budget <- function() list(used = sx_fit_budget$used, retries = sx_fit_budget$retries, aux = sx_fit_budget$aux,
  cap = if (is.finite(sx_fit_budget$cap)) sx_fit_budget$cap else NA_real_, at_cap = sx_fit_budget$used >= sx_fit_budget$cap,
  denied = sx_fit_budget$denied, cache_hits = sx_disc_cache_log$hits, cache_misses = sx_disc_cache_log$misses)
# a part during which the cap refused any fit is a budget stop, never a scored / fitted part (even when a wrapper
# swallowed the refusal as a failed candidate: its selection would not have considered every candidate)
sx_val_denied_since <- function(d0) sx_fit_budget$denied > d0
sx_val_budget_reason <- "fit budget: a model fit was refused by the hard cap (part not scored; the selection could not consider every candidate)"
sx_val_predict <- function(m, newd) {
  if (inherits(m, "merMod")) {
    rf <- switch(sx_val_target, participant_in_cluster = ~ (1 | sx_cluster), NA)
    return(as.numeric(stats::predict(m, newdata = newd, re.form = rf, allow.new.levels = TRUE)))
  }
  if (inherits(m, "lme")) return(as.numeric(stats::predict(m, newdata = newd,
                                 level = if (sx_val_target == "participant_in_cluster") 1 else 0)))
  as.numeric(stats::predict(m, newdata = newd, type = "response"))
}
sx_val_coef <- function(m, a) {
  # V134-011: EVERY coefficient of the candidate's primary tested term(s), one record each (never the first column only)
  none <- data.frame(term = character(), coefficient = character(), estimate = numeric(), stringsAsFactors = FALSE)
  b <- tryCatch(if (inherits(m, "merMod")) lme4::fixef(m) else if (inherits(m, "lme")) nlme::fixef(m) else stats::coef(m),
                error = function(e) NULL)
  if (is.null(b) || !length(a$tests)) return(none)
  drop <- sx_canon_names(unlist(a$tests[[1]]$drop))
  f <- stats::reformulate(if (length(a$full)) a$full else "1", response = "sx_y")
  lab <- sx_canon_names(attr(stats::terms(f), "term.labels"))
  X <- tryCatch(stats::model.matrix(stats::delete.response(stats::terms(f)), sx_val_lastd), error = function(e) NULL)
  if (is.null(X)) return(none)
  asg <- attr(X, "assign")
  keep <- asg %in% which(lab %in% drop) & colnames(X) %in% names(b)
  if (!any(keep)) return(none)
  data.frame(term = lab[asg[keep]], coefficient = colnames(X)[keep], estimate = unname(b[colnames(X)[keep]]),
             stringsAsFactors = FALSE)
}
# Training-only selection (V134-004): every outcome-dependent step - fitting, eligibility, ranking and the search's own
# stopping rule (replayed in candidate order) - uses the training rows only; the candidate pool is the a-priori space.
sx_val_select <- function(train, ctx) {
  sx_dd <<- train
  sx_cache_reset(ctx)           # V134-001: a fresh fit cache for every independent sample
  n <- length(sx_val_cands)
  if (n == 1) return(list(j = 1L, score = NA_real_, scores = NA_real_, visited = 1L, n_sig = NA_integer_, scored = 1L))
  sc <- rep(NA_real_, n); rows <- rep("", n); n_sig <- 0L; visited <- 0L
  for (j in seq_len(n)) {
    s <- sx_val_score(sx_val_cands[[j]]); visited <- j
    sc[j] <- s$score; rows[j] <- s$rows
    sx_val_trace[[length(sx_val_trace) + 1]] <<- data.frame(part = ctx, candidate_index = sx_val_index[j],
      status = s$status, reason = s$reason, loglik = s$loglik, statistic = s$statistic, df = s$df, log_p = s$log_p,
      reference_loglik = s$ref_loglik, aic = s$aic, score = s$score, flags = s$flags, reference_flags = s$ref_flags,
      rows_sig = s$rows, stringsAsFactors = FALSE)
    if (is.finite(s$log_p) && s$log_p < log(sx_val_alpha)) n_sig <- n_sig + 1L
    if (!is.na(sx_val_stop) && n_sig >= sx_val_stop) break    # the search's stopping rule, on training data
  }
  ok <- which(!is.na(sc))
  if (!length(ok)) return(list(j = NA_integer_, score = NA_real_, scores = sc, visited = visited, n_sig = n_sig,
                               scored = 0L))
  if (sx_val_rule == "aic" && length(unique(rows[ok])) > 1)
    stop("AIC is not comparable here: the candidates were fitted to different training rows (per-model complete cases)")
  j <- sx_select_min(sc)         # exact ties: the lower candidate index (documented secondary key); NA never wins
  list(j = j, score = sc[j], scores = sc, visited = visited, n_sig = n_sig, scored = length(ok))
}
sx_val_fold_budget <- function(split_id, f, n_train, reason) list(split = split_id, fold = f, status = "budget",
  reason = reason, selected = "", selected_index = NA_integer_, selected_key = "", train_score = NA_real_,
  visited = NA_integer_, scored = NA_integer_, n_sig_train = NA_integer_, n_train = n_train, n_test = 0L,
  train_mean = NA_real_, train_groups = NA_real_, test_groups = NA_real_, train_clusters = NA_real_,
  test_clusters = NA_real_, excluded_incomplete = NA_real_, excluded_unseen_levels = NA_real_,
  excluded_unpredicted = NA_real_, test_events = NA_real_, diagnostics = "")
sx_val_split <- function(split_id, fold_of) {
  sx_val_trace <<- list()
  folds <- sort(unique(fold_of[!is.na(fold_of) & fold_of > 0]))
  preds <- list(); info <- list()
  for (f in folds) {
    tr <- sx_val_full[is.na(fold_of) | fold_of != f, , drop = FALSE]
    te <- sx_val_full[!is.na(fold_of) & fold_of == f, , drop = FALSE]
    d0 <- sx_fit_budget$denied; pf <- NULL
    res <- if (d0 > 0) sx_val_fold_budget(split_id, f, nrow(tr), "fit budget used up before this part started (not run)") else tryCatch({
      if (length(intersect(rownames(tr), rownames(te)))) stop("internal: a row is in both parts")
      if (sx_val_target != "rows" && length(intersect(as.character(tr$sx_gid), as.character(te$sx_gid))))
        stop("internal: a participant is in both parts")
      if (sx_val_target == "cluster" && length(intersect(as.character(tr$sx_cluster), as.character(te$sx_cluster))))
        stop("internal: a cluster is in both parts")
      if (sx_val_target == "participant_in_cluster" && !all(as.character(te$sx_cluster) %in% as.character(tr$sx_cluster)))
        stop("internal: a test cluster has no training rows")
      sel <- sx_val_select(tr, paste0(split_id, ":fold", f))
      if (is.na(sel$j)) stop("no candidate could be fitted on the training part")
      a <- sx_val_cands[[sel$j]]
      m <- sx_val_fit(a, tr)
      sx_val_check_terms(m, a)
      trc <- sx_complete(tr, sx_val_need(a))
      tec <- sx_complete(te, sx_val_need(a)); n_incomplete <- nrow(te) - nrow(tec)
      seen <- sx_val_seen(trc, tec, setdiff(a$vars, c("sx_timef", "sx_gid", "sx_cluster")))
      n_unseen <- sum(!seen); tec <- tec[seen, , drop = FALSE]
      if (sx_long() && sx_disc_has_time) tec$sx_occ <- as.integer(tec$sx_timef)
      p <- sx_val_predict(m, tec)
      ok <- is.finite(p)
      pf <- data.frame(split = split_id, fold = f, row_id = as.integer(rownames(tec))[ok],
        observed = tec$sx_y[ok], predicted = p[ok],
        participant = if (sx_long()) as.character(tec$sx_gid)[ok] else as.character(rownames(tec))[ok],
        cluster = if (sx_long() && sx_disc_has_cluster) as.character(tec$sx_cluster)[ok] else "", stringsAsFactors = FALSE)
      list(split = split_id, fold = f, status = "scored", reason = "", selected = sx_val_labels[sel$j],
           selected_index = sx_val_index[sel$j], selected_key = sx_val_keys[sel$j], train_score = sel$score,
           visited = sel$visited,
           scored = sel$scored, n_sig_train = sel$n_sig, n_train = nrow(trc), n_test = sum(ok),
           train_mean = mean(trc$sx_y), train_groups = if (sx_long()) nlevels(droplevels(trc$sx_gid)) else nrow(trc),
           test_groups = if (sx_long()) length(unique(as.character(tec$sx_gid[ok]))) else sum(ok),
           train_clusters = if (sx_long() && sx_disc_has_cluster) nlevels(droplevels(trc$sx_cluster)) else 0L,
           test_clusters = if (sx_long() && sx_disc_has_cluster) length(unique(as.character(tec$sx_cluster[ok]))) else 0L,
           excluded_incomplete = n_incomplete, excluded_unseen_levels = n_unseen, excluded_unpredicted = sum(!ok),
           test_events = if (sx_disc_family == "binomial" || identical(sx_disc_gee_family, "binomial")) sum(tec$sx_y[ok] == 1) else NA_real_,
           diagnostics = tryCatch(paste(sx_diag(m, sx_last_run)$flags, collapse = "; "), error = function(e) ""))
    }, error = function(e) list(split = split_id, fold = f, status = if (inherits(e, "sx_budget")) "budget" else "failed",
                                 reason = conditionMessage(e), selected = "", selected_index = NA_integer_,
                                 selected_key = "", train_score = NA_real_, visited = NA_integer_,
                                 scored = NA_integer_, n_sig_train = NA_integer_, n_train = nrow(tr),
                                 n_test = 0L, train_mean = NA_real_, train_groups = NA_real_, test_groups = NA_real_,
                                 train_clusters = NA_real_, test_clusters = NA_real_, excluded_incomplete = NA_real_,
                                 excluded_unseen_levels = NA_real_, excluded_unpredicted = NA_real_, test_events = NA_real_,
                                 diagnostics = ""))
    if (d0 == 0 && sx_val_denied_since(d0) && res$status != "budget") {
      res <- sx_val_fold_budget(split_id, f, nrow(tr), sx_val_budget_reason); pf <- NULL
    }
    if (!is.null(pf) && identical(res$status, "scored")) preds[[length(preds) + 1]] <- pf
    info[[length(info) + 1]] <- as.data.frame(res, stringsAsFactors = FALSE)
  }
  sx_dd <<- sx_val_full
  sx_cache_reset("search")
  list(folds = do.call(rbind, info), predictions = if (length(preds)) do.call(rbind, preds) else NULL,
       trace = if (length(sx_val_trace)) do.call(rbind, sx_val_trace) else NULL)
}
sx_val_boot_budget <- function(b, reason) list(b = b, status = "budget", reason = reason, selected = "",
  selected_index = NA_integer_, selected_key = "", train_score = NA_real_, visited = NA_integer_, scored = NA_integer_,
  n_sig_train = NA_integer_, coefficient = "", estimate = NA_real_, n_coef = 0L, diagnostics = "budget", flags = "",
  n = NA_real_, groups = NA_real_)
sx_val_boot <- function(draws) {
  out <- list(); coefs <- list(); sx_val_trace <<- list()
  for (b in sort(unique(draws$b))) {
    dr <- draws[draws$b == b, , drop = FALSE]
    d0 <- sx_fit_budget$denied; cfb <- NULL
    res <- if (d0 > 0) sx_val_boot_budget(b, "fit budget used up before this draw started (not run)") else tryCatch({
      d <- sx_val_full[match(as.character(dr$row_id), rownames(sx_val_full)), , drop = FALSE]
      keep <- !is.na(match(as.character(dr$row_id), rownames(sx_val_full)))
      d <- d[keep, , drop = FALSE]; dr <- dr[keep, , drop = FALSE]
      if (sx_long()) {
        d$sx_gid <- factor(dr$draw_gid)            # a unit drawn twice = two sampled units (new IDs)
        if (sx_disc_has_cluster) d$sx_cluster <- factor(ifelse(nzchar(dr$draw_cluster), dr$draw_cluster, as.character(d$sx_cluster)))
      }
      # deterministic, draw-specific row identities: draw b, position k in the draw (source row kept in the manifest)
      rownames(d) <- paste0("b", b, ".", seq_len(nrow(d)))
      if (sx_long() && sx_disc_has_time) d <- d[order(d$sx_gid, d$sx_time), , drop = FALSE]
      sel <- sx_val_select(d, paste0("boot", b))
      if (is.na(sel$j)) stop("no candidate could be fitted on this resample")
      a <- sx_val_cands[[sel$j]]
      m <- sx_val_fit(a, d)
      sx_val_check_terms(m, a)
      sx_val_lastd <<- sx_complete(d, sx_val_need(a))
      cf <- sx_val_coef(m, a)
      if (nrow(cf)) cfb <- data.frame(b = b, selected_index = sx_val_index[sel$j], selected_key = sx_val_keys[sel$j],
                                      cf, stringsAsFactors = FALSE)
      dg <- tryCatch(sx_diag(m, sx_last_run), error = function(e) list(status = "unknown", flags = character()))
      list(b = b, status = "fitted", reason = "", selected = sx_val_labels[sel$j], selected_index = sx_val_index[sel$j],
           selected_key = sx_val_keys[sel$j], train_score = sel$score, visited = sel$visited, scored = sel$scored, n_sig_train = sel$n_sig,
           coefficient = if (nrow(cf)) cf$coefficient[nrow(cf)] else "", estimate = if (nrow(cf)) cf$estimate[nrow(cf)] else NA_real_,
           n_coef = nrow(cf), diagnostics = paste(dg$status, collapse = ""),
           flags = paste(dg$flags, collapse = "; "), n = nrow(sx_val_lastd),
           groups = if (sx_long()) nlevels(droplevels(sx_val_lastd$sx_gid)) else nrow(sx_val_lastd))
    }, error = function(e) list(b = b, status = if (inherits(e, "sx_budget")) "budget" else "failed",
                                 reason = conditionMessage(e), selected = "", selected_index = NA_integer_,
                                 selected_key = "", train_score = NA_real_, visited = NA_integer_,
                                 scored = NA_integer_, n_sig_train = NA_integer_, coefficient = "", estimate = NA_real_,
                                 n_coef = 0L, diagnostics = "failed", flags = "", n = NA_real_, groups = NA_real_))
    if (d0 == 0 && sx_val_denied_since(d0) && res$status != "budget") { res <- sx_val_boot_budget(b, sx_val_budget_reason); cfb <- NULL }
    if (!is.null(cfb) && identical(res$status, "fitted")) coefs[[length(coefs) + 1]] <- cfb
    out[[length(out) + 1]] <- as.data.frame(res, stringsAsFactors = FALSE)
  }
  sx_dd <<- sx_val_full
  sx_cache_reset("search")
  list(draws = do.call(rbind, out), coefs = if (length(coefs)) do.call(rbind, coefs) else NULL,
       trace = if (length(sx_val_trace)) do.call(rbind, sx_val_trace) else NULL)
}
"""


def _csv(rows: list[dict[str, Any]] | pd.DataFrame) -> str:
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False, lineterminator="\n")
    return buf.getvalue()


def _frame(x: Any) -> pd.DataFrame:
    """R data.frame from jsonlite (list of records or column dict) -> DataFrame."""
    if x is None:
        return pd.DataFrame()
    if isinstance(x, dict):
        return pd.DataFrame({k: (v if isinstance(v, list) else [v]) for k, v in x.items()})
    return pd.DataFrame(x)


class ValidationRunner:
    """Executes one immutable validation run in chunks (one split / a few bootstrap draws per R call), so a cancelled
    or failed run keeps every completed chunk and is reported as partial."""

    def __init__(self, session: Any, ds: Dataset, run: Any, cfg: ValidationConfig):
        from .history import context
        from .space import enumerate_candidates, focal_label, view

        self.session, self.ds, self.run, self.cfg = session, ds, run, cfg
        info = feasibility(ds, run, cfg)
        if info["problems"]:
            raise ValidationError(" ".join(info["problems"]))
        ctx = context(ds, run)
        if not ctx.current:
            raise ValidationError("The project changed since this search (" + "; ".join(ctx.differences) + "). "
                                  "Validation repeats the search on its own data view: run a new search first.")
        s = run.settings
        self.v = view(ds, s)
        leaky = [d.name for d in self.v.derived.values() if d.op in ("zscore", "center")]
        used = set([s.outcome] + s.candidates + s.required)
        if used & set(leaky):
            raise ValidationError("Standardized / centred derived variables (" + ", ".join(
                self.v.var(n).display_name for n in sorted(used & set(leaky))) + ") use whole-sample means: their "
                "scaling would leak information from the held-out data. Use the raw variables.")
        # V134-004: the a-priori pool (declared grammar, order and budget), enumerated lazily up to its exact size; the
        # full-data run's statuses, stop point and visited rows never define it
        from itertools import islice

        from .space import GRAMMAR_VERSION, candidate_key

        self.provenance: dict[str, Any] = {"search_grammar": int(getattr(run, "grammar", 2) or 2),
                                           "current_grammar": GRAMMAR_VERSION}
        if cfg.pipeline == "fixed":
            # AUD134-05 / 08: the saved specification, verified by its key; no enumeration of the space
            cands = [fixed_candidate(run, cfg.candidate)]
            row = next(r for r in run.rows if int(r.get("idx", -1)) == int(cfg.candidate))
            from .engine import FIT_RULES

            fr = (run.engine or {}).get("fit_rules")
            self.provenance.update({"candidate_key": row["key"], "model_id": row.get("model_id"),
                                    "resolved_from": "saved candidate specification (verified key)",
                                    "original_fit_rules": fr, "refit_rules": FIT_RULES,
                                    "refit_note": "" if fr == FIT_RULES else (
                                        f"the original candidate was fitted under fit rules {fr}; validation refits the "
                                        f"exact same specification under fit rules {FIT_RULES} (not a reproduction of "
                                        "the original fit)")})
        else:
            if run.grammar_problem():
                raise ValidationError("The declared search cannot be repeated inside the training parts: "
                                      + run.grammar_problem() + ". Start a new search, or validate a fixed candidate.")
            cands = list(islice(enumerate_candidates(s), pool_size(run, cfg)))
            # AUD134-05: the pool the training parts repeat must be the declared search: every saved key must match
            saved = {int(r["idx"]): r.get("key") for r in run.rows if r.get("key")}
            bad = [c.index for c in cands if c.index in saved and saved[c.index] != candidate_key(c)]
            if bad:
                raise ValidationError(f"{len(bad):,} saved candidate(s) (first #{bad[0]}) are not the candidates the "
                                      "current grammar produces at their positions: the declared search cannot be "
                                      "repeated. Start a new search.")
            self.provenance["verified_saved_keys"] = sum(1 for c in cands if c.index in saved)
        self.cands = cands
        # V134-003: display labels are for people and need not be unique; the candidate index (+ key) is the identity
        self.labels = [(focal_label(self.v, c) if c.kind != "structure" else c.structure + (
            "" if c.clustered is None else (" (with cluster)" if c.clustered else " (without cluster)"))) + f" [#{c.index}]"
            for c in cands]
        from .space import candidate_key
        self.keys = [candidate_key(c) for c in cands]
        self.info = info
        self.workload = workload(run, cfg, cands if cfg.pipeline == "pipeline" else None)
        # AUD135-05: `at_cap` = every allowed fit was used (not a truncation); `exhausted` = a fit was REFUSED or
        # planned work remained when no fit was left (the work was truncated); `counters_known` False after a chunk
        # whose response was lost (its fits are not counted: `used` is then a lower bound)
        self.budget = {"cap": int(cfg.max_fits), "used": 0, "retries": 0, "aux": 0, "cache_hits": 0, "cache_misses": 0,
                       "denied": 0, "at_cap": False, "exhausted": False, "counters_known": True}
        self.units = unit_frame(ds, run)
        self.splits = make_splits(self.units, cfg)
        for sp in self.splits:
            bad = check_split(self.units, sp, cfg.target)
            if bad:
                raise ValidationError("Internal split check failed: " + "; ".join(bad))
        self.draws = make_draws(self.units, cfg) if "bootstrap" in cfg.modes else []
        self.scripts: list[str] = []      # every DISPATCHED chunk (execution record, with its outcome)
        self.chunks: list[str] = []       # COMMITTED chunks only (validated responses): the reproduction replays these
        self.plan = execution_plan(self.splits, self.draws, int(getattr(self, "chunk", 10)))
        self.attempts: list[dict[str, Any]] = []
        self.committed: dict[int, dict[str, str]] = {}     # chunk -> {part: returned status}

    def _setup(self) -> str:
        from .rcode import candidate_args, setup_script

        s = self.run.settings
        setup = setup_script(self.v, s, self.run.family).replace("sx_result <- list(", "sx_setup_info <- list(")
        return "\n".join([
            setup, R_VALIDATE,
            "sx_val_cands <- list(" + ",\n  ".join(f"list({candidate_args(self.v, c, s)})" for c in self.cands) + ")",
            f"sx_val_labels <- {r_vec(self.labels)}",
            f"sx_val_keys <- {r_vec(self.keys)}",
            f"sx_val_index <- {r_vec([int(c.index) for c in self.cands])}",
            f"sx_val_rule <- {r_str(self.cfg.rule)}",
            f"sx_val_target <- {r_str(self.cfg.target)}",
            f"sx_val_alpha <- {float(s.alpha)!r}",
            f"sx_val_stop <- {stop_target(self.run) if self.cfg.pipeline == 'pipeline' and stop_target(self.run) else 'NA'}"
            "  # the search's stopping rule, replayed on training data (NA = none)",
            "if (!exists(\"sx_disc_gee_family\")) sx_disc_gee_family <- \"\"",
            "sx_val_full <- sx_dd; sx_val_lastd <- sx_dd",
        ])

    def _call(self, body: str, title: str, chunk: int, parts: list[str]) -> Any:
        """Dispatch one chunk. AUD135-04: the chunk is COMMITTED (replayable, counted) only after its response was
        received and validated - every dispatched part returned exactly once, counters well-formed; a lost or invalid
        response leaves the chunk uncommitted and the fit counters unknown (`used` becomes a lower bound)."""
        from ..discovery.engine import ENGINE_PACKAGES

        # AUD134-10: the remaining hard fit budget travels with every chunk; R stops fitting when it is used up
        remaining = max(0, self.budget["cap"] - self.budget["used"])
        guard = (f"sx_budget_reset({remaining})  # hard budget: model-fitting invocations left for this chunk\n"
                 "sx_disc_cache_log$hits <- 0L; sx_disc_cache_log$misses <- 0L\n")
        tail = "\nsx_result$budget <- sx_val_budget()"
        script = self._setup() + "\n" + guard + body + tail
        att = {"chunk": chunk, "title": title, "parts": list(parts), "remaining_before": remaining,
               "outcome": "dispatched", "error": ""}
        self.attempts.append(att)
        self.scripts.append(f"# ==== {title} (chunk {chunk}) ====\n" + guard + body)
        try:
            res, raw = self.session.run(self.v, kind="amd_validation", title=title, body=script,
                                        packages=ENGINE_PACKAGES, inputs={}, timeout=3600)
        except BaseException as e:
            att.update(outcome="uncertain", error=str(e))
            self.budget["counters_known"] = False
            raise
        if raw is None:
            att.update(outcome="uncertain", error="; ".join(res.messages) or "R failed")
            self.budget["counters_known"] = False
            raise ValidationError("; ".join(res.messages) or "R failed")
        b = raw.get("budget") or {}

        def g(k: str) -> float:
            x = b.get(k)
            x = x[0] if isinstance(x, list) and x else x
            if isinstance(x, bool):
                return float(x)
            if not isinstance(x, (int, float)) or x != x or x < 0:
                raise ValidationError(f"invalid fit counter {k!r} in the R response")
            return float(x)
        try:
            counts = {k: g(k) for k in ("used", "retries", "aux", "cache_hits", "cache_misses", "denied")}
            got = returned_parts(raw)
            if sorted(got) != sorted(parts):
                raise ValidationError(f"the R response returned parts {sorted(got)[:5]} for dispatched "
                                      f"{sorted(parts)[:5]}")
            if counts["used"] > remaining or any(c != int(c) for c in counts.values()):
                raise ValidationError("the R response reports more fits than the chunk was allowed")
        except ValidationError as e:
            att.update(outcome="uncertain", error=str(e))
            self.budget["counters_known"] = False
            raise
        for k in ("used", "retries", "aux", "cache_hits", "cache_misses", "denied"):
            self.budget[k] += int(counts[k])
        self.budget["at_cap"] = self.budget["used"] >= self.budget["cap"]
        if counts["denied"]:
            self.budget["exhausted"] = True
        att["outcome"] = "committed"
        att["used"] = int(counts["used"])
        self.committed[chunk] = got
        self.chunks.append(guard + body)
        return raw

    def execute(self, should_stop: Callable[[], bool] = lambda: False,
                progress: Callable[[str], None] = lambda m: None) -> dict[str, Any]:
        """Run the planned chunks in order. Stop causes (precedence: cancel > failure > budget): a cancel request
        before a chunk; a lost / failed chunk; no fit left while planned work remains, or a fit refused inside a
        chunk. AUD135-05: the final status follows the evidence - COMPLETE when every planned part was committed with
        a successful status and no fit was refused, even if the last allowed fit was used (used == cap)."""
        cfg = self.cfg
        if not self.attempts:      # the chunk size may be set after construction (draws per R call)
            self.plan = execution_plan(self.splits, self.draws, int(getattr(self, "chunk", 10)))
        folds, preds, boot, problems = [], [], [], []
        coefs: list[pd.DataFrame] = []
        trace: list[pd.DataFrame] = []
        stop = ""
        for ch in self.plan["chunks"]:
            if should_stop():
                stop = "cancelled"
                break
            if self.budget["exhausted"] or self.budget["used"] >= self.budget["cap"]:
                stop = "budget"            # planned work remains and no fit may run (or one was already refused)
                self.budget["exhausted"] = True
                break
            try:
                if ch["kind"] == "split":
                    sp = self.splits[ch["index"]]
                    progress(f"{MODES[sp['mode']]} {sp['id']}")
                    rows = sorted(sp["folds"].items())
                    body = "\n".join([
                        f"sx_vfold <- c({', '.join(str(f) for _, f in rows)})",
                        f"names(sx_vfold) <- c({', '.join(str(r) for r, _ in rows)})",
                        "sx_vf <- unname(sx_vfold[rownames(sx_val_full)])",
                        f"sx_vr <- sx_val_split({r_str(sp['id'])}, sx_vf)",
                        "sx_result <- list(folds = sx_vr$folds, predictions = sx_vr$predictions, trace = sx_vr$trace)",
                    ])
                    raw = self._call(body, f"AMD validation: {sp['id']}", ch["chunk"], ch["parts"])
                    fr = _frame(raw.get("folds"))
                    fr["mode"], fr["repeat"] = sp["mode"], sp["repeat"]
                    folds.append(fr)
                    preds.append(_frame(raw.get("predictions")))
                    trace.append(_frame(raw.get("trace")))
                else:
                    part = self.draws[ch["index"]:ch["index"] + ch["size"]]
                    progress(f"bootstrap draws {part[0]['b']}–{part[-1]['b']} of {cfg.B}")
                    recs = [{"b": d["b"], "row_id": r[0], "draw_gid": r[1], "draw_cluster": r[2]} for d in part
                            for r in d["rows"]]
                    df = pd.DataFrame(recs)
                    body = "\n".join([
                        f"sx_vdraws <- data.frame(b = c({', '.join(map(str, df['b']))}), row_id = c({', '.join(map(str, df['row_id']))}),",
                        f"  draw_gid = {r_vec(list(df['draw_gid']))}, draw_cluster = {r_vec(list(df['draw_cluster']))}, stringsAsFactors = FALSE)",
                        "sx_result <- list(boot = sx_val_boot(sx_vdraws))",
                    ])
                    raw = self._call(body, f"AMD validation: bootstrap {part[0]['b']}-{part[-1]['b']}", ch["chunk"],
                                     ch["parts"])
                    bt = raw.get("boot") or {}
                    boot.append(_frame(bt.get("draws") if isinstance(bt, dict) and "draws" in bt else bt))
                    coefs.append(_frame(bt.get("coefs") if isinstance(bt, dict) else None))
                    trace.append(_frame(bt.get("trace") if isinstance(bt, dict) else None))
            except Exception as e:  # cancellation kills the worker: committed chunks stay, the run is not complete
                problems.append(str(e))
                stop = "cancelled" if should_stop() else "failed"
                break
        late_cancel = not stop and should_stop()
        if self.budget["denied"]:
            self.budget["exhausted"] = True
            stop = stop or "budget"
        status = {"": "complete", "cancelled": "cancelled", "failed": "failed", "budget": "partial"}[stop]
        ledger = self.ledger()
        not_run = sum(1 for p in ledger if p["status"] == "not_run")
        uncertain = sum(1 for p in ledger if p["status"] == "uncommitted")
        if stop == "budget":
            denied = sum(1 for p in ledger if p["status"] == "budget")
            problems.append(f"fit budget reached: {self.budget['used']:,} of {self.budget['cap']:,} model-fitting "
                            f"invocations used; {denied:,} part(s) stopped by the budget and {not_run:,} planned "
                            "part(s) not run (finished parts are kept; a part stopped by the budget is never scored)")
        elif stop and (not_run or uncertain):
            problems.append(f"{not_run:,} planned part(s) not run" + (f"; {uncertain:,} part(s) dispatched without a "
                            "validated response (uncommitted: not counted, not reproducible)" if uncertain else ""))
        if late_cancel:
            problems.append("a cancel request arrived after the last planned part had finished: the validation is "
                            "complete")
        if not self.budget["counters_known"]:
            problems.append(f"fit counters are a lower bound: a chunk's response was lost ({self.budget['used']:,} "
                            "counted)")
        out = self.assemble(folds, preds, boot, status, problems, coefs)
        T = [x for x in trace if len(x)]
        out["trace"] = pd.concat(T, ignore_index=True).to_dict("records") if T else []
        try:      # V134-008 / AUD134-11: one complete, portable script (data load + setup + every chunk + checks)
            out["reproduce_script"] = reproduce_script(getattr(self.session, "load_script", "") or "", self._setup(),
                                                       self.chunks, expected_files(out),
                                                       {"status": out["status"], "target": cfg.target,
                                                        "binary": self.run.family == "binomial"
                                                        or getattr(self.run.settings, "family", "") == "binomial"},
                                                       replay=self.replay_spec(out))
        except Exception as e:  # noqa: BLE001 - never lose a finished validation over its export text
            out["reproduce_script"] = f"# reproduction script unavailable: {e}\n"
        return out

    def ledger(self) -> list[dict[str, Any]]:
        """AUD135-04: every PLANNED part with its execution state: the returned status of a committed part, 'uncommitted'
        (dispatched, no validated response), or 'not_run'."""
        dispatched = {p for a in self.attempts for p in a["parts"]}
        done = {p: st for got in self.committed.values() for p, st in got.items()}
        out = []
        for ch in self.plan["chunks"]:
            for p in ch["parts"]:
                meta = self.plan["parts"][p]
                st = done.get(p) or ("uncommitted" if p in dispatched else "not_run")
                out.append({"part": p, "mode": meta["mode"], "split": meta.get("split", ""), "fold": meta.get("fold"),
                            "b": meta.get("b"), "chunk": ch["chunk"], "status": st, "committed": p in done})
        return out

    def replay_spec(self, out: dict[str, Any]) -> dict[str, Any]:
        """What the reproduction replays (committed chunks, in commit order) and the planned scope it verifies against
        the embedded plan instead (splits / draws whose chunks were not committed)."""
        committed_splits = {self.plan["parts"][p]["split"] for got in self.committed.values() for p in got
                            if self.plan["parts"][p]["mode"] != "bootstrap"}
        committed_b = {self.plan["parts"][p]["b"] for got in self.committed.values() for p in got
                       if self.plan["parts"][p]["mode"] == "bootstrap"}
        pend_m = [r for r in out["manifest"] if r["split"] not in committed_splits]
        pend_d = [r for r in out["draws"] if r["b"] not in committed_b]
        return {"ledger": out["execution"]["parts"], "pending_manifest": pend_m, "pending_draws": pend_d,
                "splits": [{"split": sp["id"], "mode": sp["mode"], "repeat": sp["repeat"]} for sp in self.splits],
                "config": {"holdout_fraction": self.cfg.holdout_fraction, "k": self.cfg.k, "target": self.cfg.target},
                "alias_map": {v.alias: v.display_name for v in self.v.variables.values() if getattr(v, "alias", None)},
                "rules": {sp["id"]: _size_rule(sp["mode"], self.cfg) for sp in self.splits}}

    # ---- aggregation ----
    def assemble(self, folds: list[pd.DataFrame], preds: list[pd.DataFrame], boot: list[pd.DataFrame], status: str,
                 problems: list[str], coefs: list[pd.DataFrame] | None = None) -> dict[str, Any]:
        cfg = self.cfg
        F = pd.concat(folds, ignore_index=True) if folds else pd.DataFrame()
        P = pd.concat([p for p in preds if len(p)], ignore_index=True) if any(len(p) for p in preds) else pd.DataFrame()
        Bt = pd.concat(boot, ignore_index=True) if boot else pd.DataFrame()
        binary = self.run.family == "binomial" or getattr(self.run.settings, "family", "") == "binomial"
        n_failed = int((F["status"] != "scored").sum()) if len(F) else 0
        planned = (1 if "holdout" in cfg.modes else 0) * 1 + (cfg.k * cfg.repeats if "cv" in cfg.modes else 0)
        if status == "complete" and (n_failed or len(F) < planned or (cfg.B and "bootstrap" in cfg.modes
                                                                       and len(Bt) < cfg.B)):
            status = "partial"
        if status == "complete" and len(Bt) and (Bt["status"] != "fitted").any():
            status = "partial"
        if len(F) and (F["status"] == "budget").any() or len(Bt) and (Bt["status"] == "budget").any():
            status = "partial" if status in ("complete", "partial") else status
        summary = performance(F, P, binary, cfg.target) if len(P) else []
        C = pd.concat([c for c in (coefs or []) if len(c)], ignore_index=True) if any(len(c) for c in (coefs or [])) \
            else pd.DataFrame()
        if len(C):
            C["coefficient_label"] = [coef_display(self.v, x) for x in C["coefficient"]]
        if len(Bt) and "coefficient" in Bt:
            Bt["coefficient_label"] = [coef_display(self.v, x) if isinstance(x, str) and x else "" for x in Bt["coefficient"]]
        index_of = [int(c.index) for c in self.cands]
        stability = stability_table(Bt, self.labels, index_of, self.keys) if len(Bt) else []
        estimands = estimand_table(Bt, C, dict(zip(index_of, self.labels))) if len(Bt) else []
        manifest = [{"split": sp["id"], "mode": sp["mode"], "repeat": sp["repeat"], "row_id": int(r),
                     "participant": pk, "cluster": ck, "fold": int(sp["folds"][int(r)])}
                    for sp in self.splits for r, pk, ck in zip(self.units["row_id"], self.units["participant"],
                                                                 self.units["cluster"])]
        # AUD135-02: an explicit draw-position key (a source row drawn twice appears twice in a draw)
        draws = [{"b": d["b"], "position": k, "row_id": r[0], "draw_participant": r[1], "draw_cluster": r[2],
                  "original_participant": r[3], "original_cluster": r[4]} for d in self.draws
                 for k, r in enumerate(d["rows"], start=1)]
        import uuid

        from .space import GRAMMAR_VERSION
        realized = realized_sizes(self.units, self.splits, cfg) if self.splits else []
        from ..core.identity import IDENTITY_SCHEMA

        parts = self.ledger()
        execution = {"schema": 1, "planned_parts": len(parts),
                     "committed_parts": sum(1 for p in parts if p["committed"]),
                     "not_run_parts": sum(1 for p in parts if p["status"] == "not_run"),
                     "uncommitted_parts": sum(1 for p in parts if p["status"] == "uncommitted"),
                     "budget_parts": sum(1 for p in parts if p["status"] == "budget"),
                     "chunks_planned": len(self.plan["chunks"]), "chunks_committed": len(self.committed),
                     "parts": parts, "attempts": [dict(a) for a in self.attempts],
                     "scope": "COMPLETE: every planned part committed" if all(p["committed"] for p in parts) else
                              "PARTIAL: only committed parts are results; planned manifests describe the whole plan"}
        return {
            "grammar": GRAMMAR, "config": cfg.to_dict(), "status": status, "problems": problems, "binary": bool(binary),
            "validation_id": uuid.uuid4().hex[:12], "schema": 2,
            # V134-008: the real identities of the search this validation repeats (never blank / arbitrary)
            "search_id": self.run.id, "search_identity": self.run.fit_id or "",
            "search_grammar": GRAMMAR_VERSION, "search_signature": self.run.signature,
            "search_identity_schema": int(getattr(self.run, "identity_schema", 2) or 2),
            "search_content_id": (getattr(self.run, "data_snapshot", None) or {}).get("content_id", ""),
            "execution": execution, "identity_schema": IDENTITY_SCHEMA,
            "data_sha256": getattr(getattr(self.ds, "source", None), "sha256", "") or "",
            "pool": self.info.get("pool"), "pool_label": self.info.get("pool_label", ""),
            "stop_rule": stop_target(self.run) if cfg.pipeline == "pipeline" else None,
            "candidate_keys": [{"index": int(c.index), "key": k} for c, k in zip(self.cands, self.keys)],
            "realized": realized, "estimands": estimands,
            "target_label": TARGETS[cfg.target][0], "target_estimand": TARGETS[cfg.target][1],
            "pipeline_label": PIPELINES[cfg.pipeline], "rule_label": RULES[cfg.rule] if cfg.pipeline == "pipeline" else "",
            "candidates": [{"index": int(c.index), "label": lab} for c, lab in zip(self.cands, self.labels)],
            "pool_scope": self.info.get("pool_scope"), "adaptive": bool(self.info.get("adaptive")),
            "provenance": dict(self.provenance), "workload": dict(self.workload),
            "fit_budget": {**self.budget, "requested_bound": self.workload["bound"],
                           "estimate": self.workload["estimate"], "unit": self.workload["unit"]},
            "coefs": C.drop(columns=[c for c in ("coefficient_label",) if c in C]).to_dict("records") if len(C) else [],
            "feasibility": {k: v for k, v in self.info.items() if k not in ("problems",)},
            "rng": {"generator": "numpy PCG64([seed, stream])", "seed": cfg.seed, "numpy": np.__version__},
            "folds": F.to_dict("records"), "predictions": P.to_dict("records"), "bootstrap": Bt.to_dict("records"),
            "performance": summary, "stability": stability, "manifest": manifest, "draws": draws,
            "scripts": self.scripts,
        }


# ---- metrics ------------------------------------------------------------------------------------------------------

def _auc(y: np.ndarray, p: np.ndarray) -> float:
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(p).rank().to_numpy()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _metrics(y: np.ndarray, p: np.ndarray, base: np.ndarray, binary: bool) -> dict[str, float]:
    if binary:
        q = np.clip(p, 1e-12, 1 - 1e-12)
        return {"log_loss": float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))),
                "Brier": float(np.mean((y - p) ** 2)), "AUC": _auc(y, p)}
    e = y - p
    den = float(np.sum((y - base) ** 2))
    return {"RMSE": float(np.sqrt(np.mean(e ** 2))), "MAE": float(np.mean(np.abs(e))),
            "R2_out_of_sample": (1 - float(np.sum(e ** 2)) / den) if den > 0 else float("nan")}


def performance(F: pd.DataFrame, P: pd.DataFrame, binary: bool, target: str) -> list[dict[str, Any]]:
    """Per fold, then pooled per split mode: observation-weighted (all held-out rows) and group-weighted (each
    participant, or each cluster for the new-cluster target, counts once). R² is against the TRAINING mean of the fold."""
    out = []
    means = {(r["split"], r["fold"]): r["train_mean"] for r in F.to_dict("records") if r.get("status") == "scored"}
    P = P.copy()
    P["base"] = [means.get((s, f), np.nan) for s, f in zip(P["split"], P["fold"])]
    for (sp, fo), g in P.groupby(["split", "fold"], sort=True):
        m = _metrics(g["observed"].to_numpy(float), g["predicted"].to_numpy(float), g["base"].to_numpy(float), binary)
        out.append({"split": sp, "fold": int(fo), "weighting": "fold", "n": int(len(g)), **m})
    unit = "cluster" if target == "cluster" else "participant"
    for sp, g in P.groupby(P["split"].str.replace(r"\d+$", "", regex=True)):
        y, p, b = g["observed"].to_numpy(float), g["predicted"].to_numpy(float), g["base"].to_numpy(float)
        out.append({"split": sp, "fold": 0, "weighting": "observation-weighted (pooled rows)", "n": int(len(g)),
                    **_metrics(y, p, b, binary)})
        if not binary and target != "rows":
            per = g.assign(se=(g["observed"] - g["predicted"]) ** 2, ae=(g["observed"] - g["predicted"]).abs()) \
                .groupby(["split", unit]).agg(mse=("se", "mean"), mae=("ae", "mean"))
            out.append({"split": sp, "fold": 0, "weighting": f"{unit}-weighted (each {unit} once)", "n": int(len(per)),
                        "RMSE": float(np.sqrt(per["mse"].mean())), "MAE": float(per["mae"].mean()),
                        "R2_out_of_sample": float("nan")})
    return out


def coef_display(ds: Dataset, name: str) -> str:
    """Design-column name with internal aliases (v005, sx_time) replaced by display names (V134-011)."""
    import re

    if not isinstance(name, str):
        return ""
    amap = {v.alias: v.display_name for v in ds.variables.values() if getattr(v, "alias", None)}
    parts = []
    for p in name.split(":"):
        if p in ("sx_time", "sx_timef"):
            parts.append("Time")
            continue
        m = re.match(r"^(v\d{3,})(.*)$", p)
        if m and m.group(1) in amap:
            parts.append(amap[m.group(1)] + (f" [{m.group(2)}]" if m.group(2) else ""))
        else:
            parts.append(p)
    return " × ".join(parts)


def stability_table(Bt: pd.DataFrame, labels: list[str], index_of: list[int] | None = None,
                    keys: list[str] | None = None) -> list[dict[str, Any]]:
    """Selection frequency with BOTH denominators (all attempted draws, successful draws), sign stability conditional
    on selection, percentiles of the estimate among draws that selected the candidate. Failed draws are never zeros.
    V134-003: draws are counted by the candidate INDEX (identity), never by the display label; under the single-winner
    rule the selected counts sum to the successful draws."""
    att = int(len(Bt))
    ok = Bt[Bt["status"] == "fitted"]
    nok = int(len(ok))
    out = []
    index_of = index_of if index_of is not None else list(range(len(labels)))
    sel_idx = pd.to_numeric(ok["selected_index"], errors="coerce") if "selected_index" in ok else pd.Series(dtype=float)
    for j, lab in enumerate(labels):
        sel = ok[(sel_idx == index_of[j]).to_numpy()] if len(ok) else ok
        est = pd.to_numeric(sel["estimate"], errors="coerce").dropna()
        rec = {"candidate_index": int(index_of[j]), "candidate_key": keys[j] if keys else "",
               "candidate": lab, "selected": int(len(sel)), "attempted": att, "successful": nok,
               "freq_of_attempted": len(sel) / att if att else float("nan"),
               "freq_of_successful": len(sel) / nok if nok else float("nan"),
               "flagged_fits": int((sel["diagnostics"].astype(str) != "clean").sum()) if len(sel) else 0,
               "coefficient": (sel["coefficient_label"] if "coefficient_label" in sel else sel["coefficient"]).iloc[0]
               if len(sel) else "",
               "positive_sign": float((est > 0).mean()) if len(est) else float("nan"),
               "estimate_p2.5": float(est.quantile(0.025)) if len(est) else float("nan"),
               "estimate_median": float(est.median()) if len(est) else float("nan"),
               "estimate_p97.5": float(est.quantile(0.975)) if len(est) else float("nan")}
        out.append(rec)
    return out


def estimand_table(Bt: pd.DataFrame, C: pd.DataFrame, labels: dict[int, str]) -> list[dict[str, Any]]:
    """V134-011: one record per (selected candidate, coefficient) of the primary test: sign stability and percentiles
    conditional on that candidate being selected AND that coefficient existing; separate from selection frequency.
    Not a probability that the effect is real and not a post-selection confidence interval."""
    if C is None or not len(C):
        return []
    ok = Bt[Bt["status"] == "fitted"]
    out = []
    for (ix, coef), g in C.groupby(["selected_index", "coefficient"], sort=True):
        n_sel = int((pd.to_numeric(ok["selected_index"], errors="coerce") == ix).sum())
        est = pd.to_numeric(g["estimate"], errors="coerce").dropna()
        out.append({"candidate_index": int(ix), "candidate": labels.get(int(ix), str(ix)),
                    "term": str(g["term"].iloc[0]), "coefficient": str(coef),
                    "coefficient_label": str(g["coefficient_label"].iloc[0]) if "coefficient_label" in g else str(coef),
                    "draws_selected": n_sel, "draws_with_estimate": int(len(est)),
                    "positive_sign": float((est > 0).mean()) if len(est) else float("nan"),
                    "estimate_p2.5": float(est.quantile(0.025)) if len(est) else float("nan"),
                    "estimate_median": float(est.median()) if len(est) else float("nan"),
                    "estimate_p97.5": float(est.quantile(0.975)) if len(est) else float("nan")})
    return out


def realized_sizes(u: pd.DataFrame, splits: list[dict[str, Any]], cfg: ValidationConfig) -> list[dict[str, Any]]:
    """V134-008: requested vs realized test-part sizes per split and fold, in units of the target and in rows."""
    col = _unit_col(cfg.target)
    out = []
    for sp in splits:
        f = u.assign(fold=u["row_id"].map(sp["folds"]))
        units_all = f[col].nunique() if cfg.target != "rows" else len(f)
        for k in sorted(set(f["fold"]) - {0}):
            te = f[f["fold"] == k]
            n_units = te[col].nunique() if cfg.target != "rows" else len(te)
            out.append({"split": sp["id"], "mode": sp["mode"], "fold": int(k),
                        "requested_fraction": cfg.holdout_fraction if sp["mode"] == "holdout" else 1.0 / cfg.k,
                        "test_units": int(n_units), "units": int(units_all),
                        "realized_unit_fraction": n_units / units_all if units_all else float("nan"),
                        "test_rows": int(len(te)), "rows": int(len(f)),
                        "realized_row_fraction": len(te) / len(f) if len(f) else float("nan"),
                        "rule": _size_rule(sp["mode"], cfg)})
    return out


def _size_rule(mode: str, cfg: ValidationConfig) -> str:
    return (("ceil(fraction x units) test units" + (" per cluster (at least one training participant kept)"
                                                     if cfg.target == "participant_in_cluster" else ""))
            if mode == "holdout" else "units dealt round-robin over K folds in random order; rows per fold follow the "
                                      "units' sizes (no row balancing is promised)")


def execution_plan(splits: list[dict[str, Any]], draws: list[dict[str, Any]], chunk: int) -> dict[str, Any]:
    """AUD135-04: the immutable plan - every part (a fold of a split, or one bootstrap draw) with its chunk. One chunk
    per split (all its folds), and `chunk` draws per bootstrap chunk."""
    parts: dict[str, dict[str, Any]] = {}
    chunks: list[dict[str, Any]] = []
    k = 0
    for i, sp in enumerate(splits):
        k += 1
        ids = []
        for f in sorted(set(sp["folds"].values()) - {0}):
            pid = f"{sp['id']}:fold{int(f)}"
            parts[pid] = {"mode": sp["mode"], "split": sp["id"], "fold": int(f), "b": None}
            ids.append(pid)
        chunks.append({"chunk": k, "kind": "split", "index": i, "size": 1, "parts": ids})
    step = max(1, int(chunk))
    for i in range(0, len(draws), step):
        k += 1
        ids = []
        for d in draws[i:i + step]:
            pid = f"boot{int(d['b'])}"
            parts[pid] = {"mode": "bootstrap", "split": "", "fold": None, "b": int(d["b"])}
            ids.append(pid)
        chunks.append({"chunk": k, "kind": "boot", "index": i, "size": len(ids), "parts": ids})
    return {"parts": parts, "chunks": chunks}


def returned_parts(raw: dict[str, Any]) -> dict[str, str]:
    """{part id: status} of a chunk response (folds of a split, or bootstrap draws); duplicates are an error."""
    out: dict[str, str] = {}
    recs: list[dict[str, Any]] = []
    if raw.get("folds") is not None:
        recs = _frame(raw.get("folds")).to_dict("records")
        keys = [(f"{r['split']}:fold{int(r['fold'])}", str(r.get("status"))) for r in recs]
    else:
        bt = raw.get("boot") or {}
        recs = _frame(bt.get("draws") if isinstance(bt, dict) and "draws" in bt else bt).to_dict("records")
        keys = [(f"boot{int(r['b'])}", str(r.get("status"))) for r in recs]
    for pid, st in keys:
        if pid in out:
            raise ValidationError(f"the R response returned part {pid} twice")
        out[pid] = st
    return out


# ---- result tables / export -----------------------------------------------------------------------------------------

def summary_text(v: dict[str, Any]) -> str:
    c = v["config"]
    parts = [f"Validation status: {v['status'].upper()}" + (f" ({'; '.join(v['problems'])})" if v["problems"] else ""),
             f"Target: {v['target_label']} — {v['target_estimand']}.",
             f"Procedure: {v['pipeline_label']}" + (f"; selection rule: {v['rule_label']}" if v["rule_label"] else "") + ".",
             *([f"Candidate pool: {v['pool_label']}."] if v.get("pool_label") and c["pipeline"] == "pipeline" else []),
             *([f"Search: {v.get('search_id')} (fit identity {str(v.get('search_identity'))[:16]}…); validation run "
                f"{v.get('validation_id')}."] if v.get("validation_id") else []),
             f"Modes: {', '.join(MODES[m] for m in c['modes'])}; holdout {c['holdout_fraction']:.0%}, K = {c['k']}, "
             f"repeats = {c['repeats']}, B = {c['B']}, seed = {c['seed']}.",
             *([_coverage_text(v["execution"])] if v.get("execution") else []),
             "This is a predictive check of the stated target, separate from the discovery p-values; it is not a "
             "confirmatory test, and bootstrap selection frequencies are not probabilities that a hypothesis is true."]
    if c["pipeline"] == "fixed":
        parts.append("Fixed candidate: the model was chosen on ALL the data, so these scores are a conditional "
                     "(post-selection) assessment, not an independent validation of the search.")
    return "\n".join(parts)


def _coverage_text(e: dict[str, Any]) -> str:
    """AUD135-04: planned vs committed scope, in words."""
    t = (f"Execution: {e.get('committed_parts', 0):,} of {e.get('planned_parts', 0):,} planned parts committed "
         f"({e.get('chunks_committed', 0):,} of {e.get('chunks_planned', 0):,} chunks)")
    extra = [f"{e[k]:,} {lab}" for k, lab in (("budget_parts", "stopped by the fit budget"),
                                             ("not_run_parts", "not run"),
                                             ("uncommitted_parts", "dispatched without a validated response"))
             if e.get(k)]
    return t + ("; " + "; ".join(extra) if extra else "") + "."


def tables(v: dict[str, Any]) -> list[tuple[str, list[str], list[list[Any]], str]]:
    out = []

    def tab(name: str, recs: list[dict[str, Any]], notes: str, cols: list[str] | None = None) -> None:
        if not recs:
            return
        cols = cols or list(dict.fromkeys(k for r in recs for k in r))
        out.append((name, cols, [[r.get(k, "") for k in cols] for r in recs], notes))
    tab("Validation performance", v["performance"],
        "Held-out predictions only (never refitted on all data). R² out of sample = 1 − SSE / Σ(y − training mean)²; "
        "may be negative. 'fold' rows are per fold; pooled rows weight every held-out row (observation-weighted) or "
        "every participant / cluster once (group-weighted). No t-based CI across correlated folds is given.")
    tab("Folds", v["folds"], "Every fold is listed, including failed ones (with the reason); a failed fold is never scored "
        "as zero. Excluded test rows are counted (incomplete, unseen category levels, not predicted).")
    tab("Bootstrap stability", v["stability"],
        "Selection frequency over all attempted draws and over successful draws; sign stability and percentiles only "
        "among draws that selected the candidate (an unselected coefficient is not replaced by 0). Naive percentile "
        "intervals after selection are not valid confirmatory intervals.")
    tab("Bootstrap estimands", v.get("estimands") or [],
        "One row per coefficient of the selected candidate's primary test (a multi-df test has several): sign and "
        "percentiles among the draws that selected that candidate. Separate from selection frequency; not a "
        "probability that the effect is real and not a post-selection confidence interval.")
    tab("Requested vs realized test sizes", v.get("realized") or [],
        "Units of the prediction target and rows held out per split and fold. The rule is ceil(fraction x units); rows "
        "follow the units' sizes, so the row share can differ from the unit share.")
    if v["bootstrap"]:
        tab("Bootstrap draws", v["bootstrap"], "One row per draw (failed draws keep their reason).")
    return out


def bundle_data(v: dict[str, Any], ds: Dataset | None, run: Any = None) -> tuple[Dataset | None, str]:
    """AUD134-07: the data view a runnable reproduction bundle may carry, or (None, why not). The historical analysis is
    authoritative: the view must reproduce the validated search's data signature. With `run`, `ds` is the project
    dataset: the current view when the project is unchanged, else the search's original view rebuilt from its snapshot
    (only possible on byte-identical source data). Without `run`, `ds` is taken as the view (v1.3.4 callers) and must
    still match. Current data are never substituted for historical data."""
    from .engine import data_signature
    from .space import view

    if ds is None:
        return None, "no data were given"
    want = v.get("search_signature") or (run.signature if run is not None else "")
    if not want:
        return None, "the validation record has no data signature of its search, so the analysed data cannot be verified"
    try:
        if run is not None:
            from .history import context

            ctx = context(ds, run)
            if ctx.current:
                cand = view(ds, run.settings)
            elif ctx.original is not None:
                cand = view(ctx.original, run.settings)
            else:
                return None, "the project's data are not the data this validation analysed (" + "; ".join(
                    ctx.differences) + ")"
        else:
            cand = ds
        if data_signature(cand) != want:
            return None, ("the data view at hand does not reproduce the validated search's data signature (different "
                          "rows, values, filters or coding)")
    except Exception as e:  # noqa: BLE001 - a variable of the search no longer exists, ...
        return None, f"the analysed data view could not be rebuilt ({e})"
    return cand, ""


def expected_files(v: dict[str, Any]) -> list[dict[str, Any]]:
    """AUD134-11: the evidence files this validation must export, with their row counts (embedded in the reproduction
    script, so removing or truncating a file cannot hide a check). Mode-specific files are required only when the mode
    ran and produced rows."""
    modes = set((v.get("config") or {}).get("modes") or [])
    out = []
    for key, name in EVIDENCE_FILES:
        n = len(v.get(key) or [])
        # AUD135-01: applicability follows the declared modes only (never the exported rows): an applicable file is
        # always written (header-only when it has no rows) and always required
        applicable = (key in ("manifest", "folds", "predictions", "performance", "realized") and bool(modes & {"holdout", "cv"})
                      or key in ("draws", "bootstrap", "stability", "coefs", "estimands") and "bootstrap" in modes
                      or key == "trace")
        out.append({"file": name, "required": bool(applicable), "rows": n, "applicable": bool(applicable)})
    out.append({"file": "execution_ledger.csv", "required": True,
                "rows": len((v.get("execution") or {}).get("parts") or []), "applicable": True})
    return out


EVIDENCE_FILES = (("manifest", "split_manifest.csv"), ("draws", "bootstrap_draws.csv"),
                  ("predictions", "predictions.csv"), ("folds", "folds.csv"), ("performance", "performance.csv"),
                  ("stability", "stability.csv"), ("bootstrap", "bootstrap_results.csv"),
                  ("estimands", "bootstrap_estimands.csv"), ("coefs", "bootstrap_coefficients.csv"),
                  ("trace", "candidate_trace.csv"), ("realized", "realized_sizes.csv"))

# ---- typed evidence (AUD135-01) ----------------------------------------------------------------------------------------
# Every evidence CSV is written from this declared schema (fixed columns, header even when empty) and the generated R
# verifier embeds the same schema: it never infers types or applicability from the file under test.
# Types: int (exact integer), num (finite double; round-trip text), text (exact UTF-8; missing = ""), bool (TRUE/FALSE).
# Flags: na = may be missing ("" in the file); inf = signed infinity allowed (Inf / -Inf, the sign must match);
#        need=<status> = must be present when the REPRODUCED row has that status; tol=<class> numeric tolerance class
#        (tol: abs 1e-6 + rel 1e-6; fit: same, rel 1e-5 for optimizer-sensitive mixed models; tight: abs 1e-12 + rel
#        1e-12 for exactly computed fractions / percentiles of exported values).
EVIDENCE_SCHEMA_VERSION = 1
_I, _N, _T, _B = "int", "num", "text", "bool"
_PERF_G = [("RMSE", _N, "na"), ("MAE", _N, "na"), ("R2_out_of_sample", _N, "na")]
_PERF_B = [("log_loss", _N, "na"), ("Brier", _N, "na"), ("AUC", _N, "na")]
EVIDENCE: dict[str, dict[str, Any]] = {
    "manifest": {"keys": ["split", "row_id"], "cols": [
        ("split", _T, ""), ("mode", _T, ""), ("repeat", _I, ""), ("row_id", _I, ""), ("participant", _T, ""),
        ("cluster", _T, ""), ("fold", _I, "")]},
    "draws": {"keys": ["b", "position"], "cols": [
        ("b", _I, ""), ("position", _I, ""), ("row_id", _I, ""), ("draw_participant", _T, ""), ("draw_cluster", _T, ""),
        ("original_participant", _T, ""), ("original_cluster", _T, "")]},
    "predictions": {"keys": ["split", "fold", "row_id"], "cols": [
        ("split", _T, ""), ("fold", _I, ""), ("row_id", _I, ""), ("observed", _N, "tol=tol"),
        ("predicted", _N, "tol=tol"), ("participant", _T, ""), ("cluster", _T, "")]},
    "folds": {"keys": ["split", "fold"], "status_ok": "scored", "cols": [
        ("split", _T, ""), ("fold", _I, ""), ("status", _T, ""), ("reason", _T, ""), ("selected", _T, ""),
        ("selected_index", _I, "na need=scored"), ("selected_key", _T, ""),
        ("train_score", _N, "na inf tol=fit"), ("visited", _I, "na"), ("scored", _I, "na"),
        ("n_sig_train", _I, "na"), ("n_train", _I, ""), ("n_test", _I, ""), ("train_mean", _N, "na need=scored tol=tol"),
        ("train_groups", _I, "na"), ("test_groups", _I, "na"), ("train_clusters", _I, "na"),
        ("test_clusters", _I, "na"), ("excluded_incomplete", _I, "na"), ("excluded_unseen_levels", _I, "na"),
        ("excluded_unpredicted", _I, "na"), ("test_events", _I, "na"), ("diagnostics", _T, ""), ("mode", _T, ""),
        ("repeat", _I, "")]},
    "performance": {"keys": ["split", "fold", "weighting"], "cols": [
        ("split", _T, ""), ("fold", _I, ""), ("weighting", _T, ""), ("n", _I, "")]},     # + metric columns by family
    "bootstrap": {"keys": ["b"], "status_ok": "fitted", "cols": [
        ("b", _I, ""), ("status", _T, ""), ("reason", _T, ""), ("selected", _T, ""),
        ("selected_index", _I, "na need=fitted"), ("selected_key", _T, ""),
        ("train_score", _N, "na inf tol=fit"), ("visited", _I, "na"), ("scored", _I, "na"),
        ("n_sig_train", _I, "na"), ("coefficient", _T, ""), ("estimate", _N, "na tol=fit"), ("n_coef", _I, ""),
        ("diagnostics", _T, ""), ("flags", _T, ""), ("n", _I, "na need=fitted"), ("groups", _I, "na need=fitted"),
        ("coefficient_label", _T, "")]},
    "coefs": {"keys": ["b", "selected_index", "coefficient"], "cols": [
        ("b", _I, ""), ("selected_index", _I, ""), ("selected_key", _T, ""), ("term", _T, ""), ("coefficient", _T, ""),
        ("estimate", _N, "tol=fit")]},
    "stability": {"keys": ["candidate_index"], "cols": [
        ("candidate_index", _I, ""), ("candidate_key", _T, ""), ("candidate", _T, ""), ("selected", _I, ""),
        ("attempted", _I, ""), ("successful", _I, ""), ("freq_of_attempted", _N, "na tol=tight"),
        ("freq_of_successful", _N, "na tol=tight"), ("flagged_fits", _I, ""), ("coefficient", _T, ""),
        ("positive_sign", _N, "na tol=tight"), ("estimate_p2.5", _N, "na tol=fit"), ("estimate_median", _N, "na tol=fit"),
        ("estimate_p97.5", _N, "na tol=fit")]},
    "estimands": {"keys": ["candidate_index", "coefficient"], "cols": [
        ("candidate_index", _I, ""), ("candidate", _T, ""), ("term", _T, ""), ("coefficient", _T, ""),
        ("coefficient_label", _T, ""), ("draws_selected", _I, ""), ("draws_with_estimate", _I, ""),
        ("positive_sign", _N, "na tol=tight"), ("estimate_p2.5", _N, "na tol=fit"), ("estimate_median", _N, "na tol=fit"),
        ("estimate_p97.5", _N, "na tol=fit")]},
    "trace": {"keys": ["part", "candidate_index"], "cols": [
        ("part", _T, ""), ("candidate_index", _I, ""), ("status", _T, ""), ("reason", _T, ""),
        ("loglik", _N, "na tol=fit"), ("statistic", _N, "na inf tol=fit"), ("df", _N, "na tol=fit"),
        ("log_p", _N, "na inf tol=fit"), ("reference_loglik", _N, "na tol=fit"), ("aic", _N, "na tol=fit"),
        ("score", _N, "na inf tol=fit"), ("flags", _T, ""), ("reference_flags", _T, ""), ("rows_sig", _T, "")]},
    "realized": {"keys": ["split", "fold"], "cols": [
        ("split", _T, ""), ("mode", _T, ""), ("fold", _I, ""), ("requested_fraction", _N, "tol=tight"),
        ("test_units", _I, ""), ("units", _I, ""), ("realized_unit_fraction", _N, "na tol=tight"), ("test_rows", _I, ""),
        ("rows", _I, ""), ("realized_row_fraction", _N, "na tol=tight"), ("rule", _T, "")]},
    "ledger": {"keys": ["part"], "cols": [
        ("part", _T, ""), ("mode", _T, ""), ("split", _T, ""), ("fold", _I, "na"), ("b", _I, "na"), ("chunk", _I, ""),
        ("status", _T, ""), ("committed", _B, "")]},
}


def evidence_columns(key: str, recs: list[dict[str, Any]] | None = None, binary: bool | None = None) -> list[tuple]:
    cols = list(EVIDENCE[key]["cols"])
    if key == "performance":
        if binary is None:
            binary = any("log_loss" in r for r in (recs or []))
        cols += _PERF_B if binary else _PERF_G
    return cols


class EvidenceError(ValidationError):
    pass


def _cell(v: Any, typ: str, flags: str, where: str) -> str:
    missing = v is None or v is pd.NA or (isinstance(v, float) and math.isnan(v)) or \
        (isinstance(v, (np.floating,)) and np.isnan(v))
    if typ == _T:
        if missing:
            return ""
        if isinstance(v, str):
            return v
        raise EvidenceError(f"{where}: text expected, got {type(v).__name__}")
    if missing:
        if "na" in flags.split():
            return ""
        raise EvidenceError(f"{where}: a value is required")
    if typ == _B:
        if isinstance(v, (bool, np.bool_)):
            return "TRUE" if v else "FALSE"
        raise EvidenceError(f"{where}: TRUE / FALSE expected")
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
        raise EvidenceError(f"{where}: a number expected, got {type(v).__name__}")
    if typ == _I:
        f = float(v)
        if not math.isfinite(f) or f != int(f) or abs(f) > 2 ** 53:
            raise EvidenceError(f"{where}: an exact integer expected, got {v!r}")
        return str(int(f))
    f = float(v)
    if math.isinf(f):
        if "inf" in flags.split():
            return "Inf" if f > 0 else "-Inf"
        raise EvidenceError(f"{where}: infinite value not allowed")
    return repr(f)


def evidence_csv(key: str, recs: list[dict[str, Any]], binary: bool | None = None) -> str:
    """Typed, canonical CSV of an evidence table (header even when empty; unknown columns are refused)."""
    cols = evidence_columns(key, recs, binary)
    names = [c[0] for c in cols]
    extra = sorted({k for r in recs for k in r} - set(names))
    if extra:
        raise EvidenceError(f"{key}: undeclared evidence column(s) {extra}")
    buf = io.StringIO()
    wr = csv.writer(buf, lineterminator="\n")
    wr.writerow(names)
    for i, r in enumerate(recs):
        wr.writerow([_cell(r.get(n), t, f, f"{key} row {i + 1} column {n}") for n, t, f in cols])
    return buf.getvalue()


def _evidence_applicable(v: dict[str, Any], key: str) -> bool:
    return next((e["applicable"] for e in expected_files(v) if e["file"] == dict(EVIDENCE_FILES)[key]), False)


def export(v: dict[str, Any], folder: str, ds: Dataset | None = None, include_data: bool = False,
           run: Any = None) -> list[str]:
    """Write the validation evidence into a NEW folder inside `folder` (AUD135-03: never into a reused folder; the
    returned paths name the actual location): config/summary JSON, fold manifest, bootstrap draw map, execution ledger,
    predictions, folds, performance, stability, coefficient and trace CSVs (typed, header-only when empty), the exact
    R code of every committed chunk, and - only when the analysed data can be verified (`bundle_data`) - the runnable
    portable reproduction (the data file only with `include_data` consent; otherwise code-only). Without verified
    data the export is results and metadata only, labelled so (AUD134-07), and carries no reproduction script."""
    import json

    from ..export.bundle import export_transaction

    data_view, why_not = bundle_data(v, ds, run) if ds is not None else (None, "no data view was given to the export")
    if not v.get("execution") and v.get("reproduce_script"):      # a v1.3.5-or-earlier record: results only
        data_view, why_not = None, ("this validation was recorded by StatExplore 1.3.5 or earlier: its reproduction "
                                    "script lacks the typed, complete verification of v1.3.6 (AUD135-01/02), so it is "
                                    "not exported as a runnable reproduction") + (f"; also: {why_not}" if why_not else "")
    with export_transaction(folder, f"validation_{v.get('validation_id') or 'export'}", include_data) as tx:
        files: list[str] = []

        def w(name: str, text: str, kind: str = "results") -> None:
            tx.write_text(name, text, kind)
            files.append(name)
        meta = {k: v.get(k) for k in ("grammar", "schema", "validation_id", "config", "status", "problems", "search_id",
                                      "search_identity", "search_grammar", "search_signature", "search_identity_schema",
                                      "search_content_id", "data_sha256", "pool",
                                      "pool_label", "pool_scope", "adaptive", "stop_rule", "target_label",
                                      "target_estimand", "pipeline_label", "rule_label", "candidates", "candidate_keys",
                                      "feasibility", "rng", "realized", "provenance", "fit_budget", "execution")}
        meta["evidence_files"] = expected_files(v)
        meta["evidence_schema"] = EVIDENCE_SCHEMA_VERSION
        meta["reproduction"] = ({"runnable": True, "data": "verified against the search's data signature (identity "
                                 "schema 3)", "data_included": bool(include_data),
                                 "bundle_kind": "runnable" if include_data else "code-only"}
                                if data_view is not None and v.get("reproduce_script") else
                                {"runnable": False, "reason": why_not or "no reproduction script was recorded",
                                 "bundle_kind": "results-only"})
        w("validation_manifest.json", json.dumps(meta, indent=1, default=str), "manifest")
        typed = bool(v.get("execution"))      # v1.3.6 records; older records are exported as recorded (results only)
        for key, name in EVIDENCE_FILES:
            if typed and _evidence_applicable(v, key):
                w(name, evidence_csv(key, v.get(key) or [], binary=bool(v.get("binary"))))
            elif not typed and v.get(key):
                w(name, _csv(v[key]))
        if typed:
            w("execution_ledger.csv", evidence_csv("ledger", (v.get("execution") or {}).get("parts") or []))
        w("validation_expected.csv", _csv(meta["evidence_files"]), "metadata")
        w("validation_code.R", "# R code executed for each committed chunk (after the project's data-load script and "
          "the search setup).\n# Historical record; for a complete, runnable reproduction use validation_reproduce.R "
          "(written only when the analysed data are verified).\n" + "\n\n".join(v.get("scripts") or []), "metadata")
        if v.get("reproduce_script") and data_view is not None:
            # portable: verified data view (with consent), SHA-256 check, package check
            from ..export.exporters import portable_into

            files += portable_into(tx, "", v["reproduce_script"], data_view, name="validation_reproduce",
                                   packages=None, readme="README_bundle.txt",
                                   extra={"validation_id": v.get("validation_id"), "search_id": v.get("search_id"),
                                          "search_identity": v.get("search_identity"),
                                          "search_signature": v.get("search_signature")})
        else:
            # AUD134-07 / AUD135-03: never a runnable bundle over data that are not the analysed data
            w("REPRODUCTION_UNAVAILABLE.txt", "No runnable reproduction bundle was written: " + (
                why_not or "no reproduction script was recorded") + ".\nThe results, manifests and the exact R code "
              "(validation_code.R) are historical records of the analysed data; reopen the project with the original "
              "data (same imported content, filters and coding) to export a runnable reproduction.\n", "metadata")
            tx.bundle_kind = "results-only"
        w("README.txt", summary_text(v) + "\n\nFiles: validation_reproduce.R (complete reproduction with self-checks, "
          "only with verified data), execution_ledger.csv (planned / committed / not run parts), candidate_trace.csv "
          "(every candidate in every training part / draw), bootstrap_estimands.csv, realized_sizes.csv, "
          "split_manifest.csv (the PLANNED fold of every analysed row per split; fold 0 = training only), "
          "bootstrap_draws.csv (planned draws: draw IDs and original IDs), predictions.csv (held-out rows), folds.csv, "
          "performance.csv, stability.csv, validation_code.R, bundle_inventory.json.\n", "readme")
    return [str(p) for p in tx.published(files)]


R_REPRO_CHECKS = r"""
# ---- verification against the app's exported evidence (AUD134-11; typed and complete since v1.3.6, AUD135-01/02/04) ----
# Every check is listed in validation_check_results.csv (PASS / FAIL / NOT APPLICABLE); the script exits with status 1
# when any check fails. A check is NOT APPLICABLE only when the embedded declaration says so - an indeterminate or
# malformed comparison is a FAIL. Every evidence file is read as TEXT (no type guessing: "001" stays "001", the literal
# "NA" is text, "" is the declared missing value) and parsed by the declared field type embedded below (never inferred
# from the file under test). Numbers: finite / +Inf / -Inf / missing / malformed are distinct; signed infinities match
# only the same sign; malformed text never matches anything. Keys are length-prefixed UTF-8 (no delimiter collisions)
# and must be unique on both sides; missing / extra / duplicate rows fail.
# Values are recomputed from the reproduced fits - predictions, folds, draws, coefficients and trace by replaying the
# committed chunks; performance, stability (selection frequencies over the DECLARED candidate pool, sign stability,
# type-7 percentiles), estimands and realized sizes by independent R code from those reproduced values. The planned
# split / draw manifests and the execution ledger are metadata: verified against the plan embedded in this script.
# Tolerances: predictions / observed / metrics abs 1e-6 + rel 1e-6; log p, statistics, log-likelihoods, AIC and
# coefficient estimates abs 1e-6 + rel 1e-6 (optimizer-sensitive mixed-model fits: rel 1e-5); fractions abs 1e-12 +
# rel 1e-12; counts, statuses, indices, keys and text exact.
sx_chk <- data.frame(check = character(), status = character(), detail = character(), stringsAsFactors = FALSE)
sx_add_chk <- function(check, ok, detail = "", applicable = TRUE) {
  st <- if (identical(applicable, FALSE)) "NOT APPLICABLE" else if (length(ok) == 1L && isTRUE(ok)) "PASS" else "FAIL"
  sx_chk[nrow(sx_chk) + 1L, ] <<- list(check, st, as.character(detail)[1])
  invisible(st == "PASS")
}
sx_read <- function(f) if (file.exists(f)) utils::read.csv(f, colClasses = "character", na.strings = character(0),
  check.names = FALSE, encoding = "UTF-8", strip.white = FALSE, stringsAsFactors = FALSE) else NULL
sx_num_re <- "^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$"
# state of each value: "num" (finite), "inf", "-inf", "na" (declared missing) or "bad" (malformed / NaN)
sx_parse_num <- function(x) {
  n <- length(x)
  if (is.numeric(x) || is.logical(x)) {
    x <- as.numeric(x)
    st <- ifelse(is.nan(x), "bad", ifelse(is.na(x), "na", ifelse(is.infinite(x), ifelse(x > 0, "inf", "-inf"), "num")))
    return(list(value = x, state = st))
  }
  x <- as.character(x); st <- rep("bad", n); val <- rep(NA_real_, n)
  st[is.na(x) | x == ""] <- "na"
  pinf <- !is.na(x) & x %in% c("Inf", "+Inf"); ninf <- !is.na(x) & x == "-Inf"
  st[pinf] <- "inf"; val[pinf] <- Inf; st[ninf] <- "-inf"; val[ninf] <- -Inf
  fin <- which(!is.na(x) & grepl(sx_num_re, x))
  if (length(fin)) {
    v <- suppressWarnings(as.numeric(x[fin])); nz <- grepl("[1-9]", sub("[eE].*$", "", x[fin]))
    good <- is.finite(v) & !(v == 0 & nz)            # decimal overflow / silent underflow to 0 are malformed
    st[fin[good]] <- "num"; val[fin[good]] <- v[good]
  }
  list(value = val, state = st)
}
# typed numeric equality: total (never NA); missing equals missing only if `missing`, infinities only same-sign if `inf`
sx_num_ok <- function(a, b, abs_tol, rel_tol, missing = TRUE, inf = TRUE) {
  p <- sx_parse_num(a); q <- sx_parse_num(b)
  if (length(p$state) != length(q$state)) return(FALSE)
  ok <- rep(FALSE, length(p$state))
  ok[p$state == "na" & q$state == "na"] <- isTRUE(missing)
  i <- p$state %in% c("inf", "-inf") & q$state %in% c("inf", "-inf")
  ok[i] <- isTRUE(inf) & p$state[i] == q$state[i]
  f <- p$state == "num" & q$state == "num"
  x <- p$value[f]; y <- q$value[f]; sc <- pmax(abs(x), abs(y), 1)
  ok[f] <- abs(x / sc - y / sc) <= abs_tol / sc + rel_tol * pmax(abs(x), abs(y)) / sc
  ok
}
sx_parse_int <- function(x) {
  if (is.numeric(x) || is.logical(x)) {
    x <- as.numeric(x)
    st <- ifelse(is.na(x) & !is.nan(x), "na", ifelse(!is.na(x) & is.finite(x) & x == round(x) & abs(x) <= 2^53, "int", "bad"))
    v <- rep(NA_character_, length(x)); v[st == "int"] <- sprintf("%.0f", x[st == "int"] + 0)
    return(list(value = v, state = st))
  }
  x <- as.character(x)
  st <- ifelse(is.na(x) | x == "", "na", ifelse(grepl("^-?(0|[1-9][0-9]{0,15})$", x), "int", "bad"))
  v <- rep(NA_character_, length(x)); v[st == "int"] <- x[st == "int"]
  list(value = v, state = st)
}
sx_int_ok <- function(a, b, missing = FALSE) {
  p <- sx_parse_int(a); q <- sx_parse_int(b)
  if (length(p$state) != length(q$state)) return(FALSE)
  ok <- p$state == "int" & q$state == "int" & p$value == q$value
  ok[is.na(ok)] <- FALSE
  ok | (p$state == "na" & q$state == "na" & isTRUE(missing))
}
sx_text_ok <- function(a, b) {     # exact UTF-8 text; a missing text value is "" (declared convention)
  a <- as.character(a); b <- as.character(b); if (length(a) != length(b)) return(FALSE)
  a[is.na(a)] <- ""; b[is.na(b)] <- ""
  enc2utf8(a) == enc2utf8(b)
}
sx_bool_ok <- function(a, b) {
  tx <- function(x) if (is.logical(x)) ifelse(is.na(x), "bad", ifelse(x, "TRUE", "FALSE")) else
    ifelse(as.character(x) %in% c("TRUE", "FALSE"), as.character(x), "bad")
  x <- tx(a); y <- tx(b); if (length(x) != length(y)) return(FALSE)
  x != "bad" & x == y
}
# length-prefixed composite key (no delimiter collision); integer keys canonical; missing / malformed keys stop
sx_key <- function(d, cols, types) {
  if (!nrow(d)) return(character())
  parts <- lapply(seq_along(cols), function(i) {
    x <- d[[cols[i]]]
    if (is.null(x)) stop("key column ", cols[i], " missing")
    if (types[i] == "int") { p <- sx_parse_int(x); if (any(p$state != "int")) stop("invalid integer key ", cols[i]); x <- p$value }
    else { x <- as.character(x); if (anyNA(x)) stop("missing key ", cols[i]); x <- enc2utf8(x) }
    paste0(nchar(x, type = "bytes"), ":", x, ";")
  })
  do.call(paste0, parts)
}
sx_empty <- function(cols) as.data.frame(stats::setNames(lapply(cols, function(i) character()), cols),
                                         stringsAsFactors = FALSE, check.names = FALSE)
sx_tol <- c(1e-6, 1e-6)
sx_tol_fit <- c(1e-6, if (sx_long()) 1e-5 else 1e-6)
sx_tol_tight <- c(1e-12, 1e-12)
# compare one exported evidence file with its reproduced / recomputed table, field by declared field
sx_compare <- function(name, file, rep, spec, skip = character()) {
  app <- sx_read(file)
  if (is.null(app)) return(invisible(sx_add_chk(paste0(name, ": file"), FALSE, paste("required file missing:", file))))
  cols <- spec$cols$name; types <- spec$cols$type; flags <- spec$cols$flags
  okc <- identical(names(app), cols)
  sx_add_chk(paste0(name, ": columns"), okc, if (okc) "the declared columns, in order" else
             paste("expected", paste(cols, collapse = ","), "got", paste(names(app), collapse = ",")))
  if (!okc) return(invisible(FALSE))
  if (is.null(rep)) rep <- sx_empty(cols)
  lack <- setdiff(setdiff(cols, skip), names(rep))
  if (length(lack)) return(invisible(sx_add_chk(paste0(name, ": reproduced columns"), FALSE,
                                                paste("not reproduced:", paste(lack, collapse = ", ")))))
  kt <- types[match(spec$keys, cols)]
  ka <- tryCatch(sx_key(app, spec$keys, kt), error = function(e) e)
  kr <- tryCatch(sx_key(rep, spec$keys, kt), error = function(e) e)
  if (inherits(ka, "error") || inherits(kr, "error"))
    return(invisible(sx_add_chk(paste0(name, ": keys"), FALSE, conditionMessage(if (inherits(ka, "error")) ka else kr))))
  sx_add_chk(paste0(name, ": unique keys (app)"), !anyDuplicated(ka), paste(sum(duplicated(ka)), "duplicate key(s)"))
  sx_add_chk(paste0(name, ": unique keys (reproduced)"), !anyDuplicated(kr), paste(sum(duplicated(kr)), "duplicate key(s)"))
  sx_add_chk(paste0(name, ": same rows"), length(ka) == length(kr) && setequal(ka, kr),
             sprintf("app %d, reproduced %d, only in app %d, only reproduced %d", length(ka), length(kr),
                     length(setdiff(ka, kr)), length(setdiff(kr, ka))))
  i <- match(kr, ka); k <- !is.na(i)
  A <- app[i[k], , drop = FALSE]; R <- rep[k, , drop = FALSE]
  st <- if (!is.null(spec$status_ok) && "status" %in% names(R)) as.character(R$status) else NULL
  for (j in seq_along(cols)) {
    col <- cols[j]
    if (col %in% c(spec$keys, skip)) next
    fl <- strsplit(flags[j], " ", fixed = TRUE)[[1]]
    na_ok <- "na" %in% fl; inf_ok <- "inf" %in% fl
    need <- sub("^need=", "", fl[startsWith(fl, "need=")]); tc <- sub("^tol=", "", fl[startsWith(fl, "tol=")])
    a <- A[[col]]; r <- R[[col]]
    res <- tryCatch(switch(types[j],
      text = sx_text_ok(a, r), int = sx_int_ok(a, r, na_ok), bool = sx_bool_ok(a, r),
      num = { tl <- if (identical(tc, "tight")) sx_tol_tight else if (identical(tc, "fit")) sx_tol_fit else sx_tol
              sx_num_ok(a, r, tl[1], tl[2], na_ok, inf_ok) },
      stop("unknown declared type ", types[j])), error = function(e) FALSE)
    if (length(need) && !is.null(st) && length(res) == length(st)) {
      present <- !(sx_parse_num(a)$state %in% c("na", "bad")) & !(sx_parse_num(r)$state %in% c("na", "bad"))
      res[st %in% need & !present] <- FALSE
    }
    d <- if (types[j] == "num") { x <- sx_parse_num(a)$value; y <- sx_parse_num(r)$value; dd <- abs(x - y)
           if (any(is.finite(dd))) sprintf("; max |difference| %.3g", max(dd[is.finite(dd)])) else "" } else ""
    sx_add_chk(paste0(name, ": ", col), length(res) == nrow(A) && all(res),
               paste0(sum(!res), " of ", length(res), " row(s) differ or are invalid", d))
  }
  invisible(TRUE)
}
# same label <=> same R group (identity mapping of the app's unit keys onto the analysed data's grouping)
sx_partition_ok <- function(labels, groups) {
  a <- as.character(labels); g <- as.character(groups)
  if (anyNA(a) || anyNA(g)) return(FALSE)
  pa <- paste0(nchar(a, type = "bytes"), ":", a, ";", g)
  length(unique(pa)) == length(unique(a)) && length(unique(a)) == length(unique(g))
}
sx_pos <- function(row_id) match(as.character(row_id), rownames(sx_val_full))
sx_unit_part <- function(row_id) { p <- sx_pos(row_id)
  if (sx_long()) as.character(sx_val_full$sx_gid)[p] else ifelse(is.na(p), NA_character_, as.character(row_id)) }
sx_unit_clus <- function(row_id) { p <- sx_pos(row_id)
  if (sx_long() && sx_disc_has_cluster) as.character(sx_val_full$sx_cluster)[p] else ifelse(is.na(p), NA_character_, "") }
sx_coef_display <- function(x) vapply(as.character(x), function(nm) {
  if (is.na(nm) || !nzchar(nm)) return("")
  out <- vapply(strsplit(nm, ":", fixed = TRUE)[[1]], function(p) {
    if (p %in% c("sx_time", "sx_timef")) return("Time")
    m <- regmatches(p, regexec("^(v[0-9]{3,})(.*)$", p))[[1]]
    if (length(m) == 3 && m[2] %in% names(sx_alias)) return(paste0(sx_alias[[m[2]]], if (nzchar(m[3])) paste0(" [", m[3], "]") else ""))
    p }, "", USE.NAMES = FALSE)
  paste(out, collapse = " × ")
}, "", USE.NAMES = FALSE)
sx_auc2 <- function(y, p) { n1 <- sum(y == 1); n0 <- sum(y == 0); if (n1 == 0 || n0 == 0) return(NA_real_)
  r <- rank(p); (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0) }
sx_met <- function(y, p, b, binary) {
  if (binary) { q <- pmin(pmax(p, 1e-12), 1 - 1e-12)
    return(c(log_loss = -mean(y * log(q) + (1 - y) * log(1 - q)), Brier = mean((y - p)^2), AUC = sx_auc2(y, p))) }
  e <- y - p; den <- sum((y - b)^2)
  c(RMSE = sqrt(mean(e^2)), MAE = mean(abs(e)), R2_out_of_sample = if (den > 0) 1 - sum(e^2) / den else NA_real_)
}
# an independent R implementation of the app's aggregation (per fold; pooled observation-weighted; group-weighted)
sx_perf <- function(P, F, binary, target) {
  if (is.null(P) || !nrow(P)) return(NULL)
  F <- F[F$status == "scored", , drop = FALSE]
  P$base <- F$train_mean[match(paste(P$split, P$fold), paste(F$split, F$fold))]
  out <- list()
  for (k in unique(paste(P$split, P$fold))) {
    g <- P[paste(P$split, P$fold) == k, , drop = FALSE]
    out[[length(out) + 1]] <- data.frame(split = g$split[1], fold = g$fold[1], weighting = "fold", n = nrow(g),
                                         t(sx_met(g$observed, g$predicted, g$base, binary)), check.names = FALSE)
  }
  mode <- sub("[0-9]+$", "", P$split); unit <- if (target == "cluster") "cluster" else "participant"
  for (m in unique(mode)) {
    g <- P[mode == m, , drop = FALSE]
    out[[length(out) + 1]] <- data.frame(split = m, fold = 0, weighting = "observation-weighted (pooled rows)",
                                         n = nrow(g), t(sx_met(g$observed, g$predicted, g$base, binary)), check.names = FALSE)
    if (!binary && target != "rows") {
      u <- paste(g$split, g[[unit]]); se <- tapply((g$observed - g$predicted)^2, u, mean)
      ae <- tapply(abs(g$observed - g$predicted), u, mean)
      out[[length(out) + 1]] <- data.frame(split = m, fold = 0, weighting = paste0(unit, "-weighted (each ", unit, " once)"),
                                           n = length(se), RMSE = sqrt(mean(se)), MAE = mean(ae), R2_out_of_sample = NA_real_,
                                           check.names = FALSE)
    }
  }
  do.call(rbind, out)
}
# AUD135-02: selection frequencies over the DECLARED pool (index / key / label from the search setup, never from
# stability.csv), denominators = committed draws / successful draws, sign stability and type-7 percentiles of the
# estimates of the draws that selected the candidate; failed draws are never zeros
sx_pool <- data.frame(index = as.numeric(sx_val_index), key = as.character(sx_val_keys),
                      label = as.character(sx_val_labels), stringsAsFactors = FALSE)
sx_q <- function(e, p) if (length(e)) stats::quantile(e, p, type = 7, names = FALSE) else NA_real_
sx_stability <- function(B) {
  if (is.null(B) || !nrow(B)) return(NULL)
  ok <- B[B$status == "fitted", , drop = FALSE]; att <- nrow(B); nok <- nrow(ok)
  do.call(rbind, lapply(seq_len(nrow(sx_pool)), function(j) {
    sel <- ok[!is.na(ok$selected_index) & as.numeric(ok$selected_index) == sx_pool$index[j], , drop = FALSE]
    est <- suppressWarnings(as.numeric(sel$estimate)); est <- est[!is.na(est)]
    data.frame(candidate_index = sx_pool$index[j], candidate_key = sx_pool$key[j], candidate = sx_pool$label[j],
               selected = nrow(sel), attempted = att, successful = nok,
               freq_of_attempted = if (att) nrow(sel) / att else NA_real_,
               freq_of_successful = if (nok) nrow(sel) / nok else NA_real_,
               flagged_fits = sum(as.character(sel$diagnostics) != "clean"),
               coefficient = if (nrow(sel)) sx_coef_display(sel$coefficient[1]) else "",
               positive_sign = if (length(est)) mean(est > 0) else NA_real_, `estimate_p2.5` = sx_q(est, 0.025),
               estimate_median = if (length(est)) stats::median(est) else NA_real_, `estimate_p97.5` = sx_q(est, 0.975),
               check.names = FALSE, stringsAsFactors = FALSE)
  }))
}
sx_estimands <- function(B, C) {
  if (is.null(C) || !nrow(C)) return(NULL)
  ok <- B[B$status == "fitted", , drop = FALSE]
  g <- unique(C[, c("selected_index", "coefficient")])
  do.call(rbind, lapply(seq_len(nrow(g)), function(j) {
    ix <- as.numeric(g$selected_index[j]); cf <- g$coefficient[j]
    r <- C[as.numeric(C$selected_index) == ix & C$coefficient == cf, , drop = FALSE]
    est <- suppressWarnings(as.numeric(r$estimate)); est <- est[!is.na(est)]
    data.frame(candidate_index = ix, candidate = sx_pool$label[match(ix, sx_pool$index)], term = r$term[1],
               coefficient = cf, coefficient_label = sx_coef_display(cf),
               draws_selected = sum(!is.na(ok$selected_index) & as.numeric(ok$selected_index) == ix),
               draws_with_estimate = length(est), positive_sign = if (length(est)) mean(est > 0) else NA_real_,
               `estimate_p2.5` = sx_q(est, 0.025), estimate_median = if (length(est)) stats::median(est) else NA_real_,
               `estimate_p97.5` = sx_q(est, 0.975), check.names = FALSE, stringsAsFactors = FALSE)
  }))
}
# coefficient identity: every coefficient row belongs to a fitted draw of the same selected candidate, with the
# declared key, one term per (candidate, coefficient), and n_coef rows per draw
sx_coef_identity <- function(B, C) {
  if (is.null(C) || !nrow(C)) return(is.null(B) || !nrow(B) || all(as.numeric(B$n_coef[B$status == "fitted"]) == 0))
  ok <- B[B$status == "fitted", , drop = FALSE]; i <- match(C$b, ok$b)
  if (anyNA(i) || any(as.numeric(C$selected_index) != as.numeric(ok$selected_index[i]))) return(FALSE)
  if (any(C$selected_key != sx_pool$key[match(as.numeric(C$selected_index), sx_pool$index)])) return(FALSE)
  terms <- tapply(C$term, paste(C$selected_index, C$coefficient), function(t) length(unique(t)))
  n <- table(factor(C$b, levels = ok$b))
  all(terms == 1) && all(as.numeric(n) == as.numeric(ok$n_coef))
}
sx_realized <- function(M) {
  if (is.null(M) || !nrow(M)) return(NULL)
  unit <- if (sx_target == "rows") as.character(M$row_id) else if (sx_target == "cluster") sx_unit_clus(M$row_id) else
    sx_unit_part(M$row_id)
  out <- list()
  for (s in sx_plan_splits$split) {
    w <- M$split == s; f <- as.numeric(M$fold[w]); u <- unit[w]; md <- sx_plan_splits$mode[sx_plan_splits$split == s]
    n_all <- if (sx_target == "rows") length(u) else length(unique(u))
    for (k in sort(unique(f[f > 0]))) {
      te <- f == k; nu <- if (sx_target == "rows") sum(te) else length(unique(u[te]))
      out[[length(out) + 1]] <- data.frame(split = s, mode = md, fold = k,
        requested_fraction = if (md == "holdout") sx_val_cfg$holdout_fraction else 1 / sx_val_cfg$k,
        test_units = nu, units = n_all, realized_unit_fraction = if (n_all) nu / n_all else NA_real_,
        test_rows = sum(te), rows = length(f), realized_row_fraction = if (length(f)) sum(te) / length(f) else NA_real_,
        rule = sx_plan_splits$rule[sx_plan_splits$split == s], stringsAsFactors = FALSE)
    }
  }
  do.call(rbind, out)
}
# ---- required files (declared by the app from the validation's modes, embedded above) ----
for (i in seq_len(nrow(sx_expect))) {
  f <- sx_expect$file[i]
  if (!sx_expect$required[i]) { sx_add_chk(paste0("file ", f), TRUE, "not applicable to this validation", applicable = FALSE); next }
  ok <- file.exists(f)
  sx_add_chk(paste0("file ", f), ok, if (ok) "present" else "REQUIRED reference file missing")
  if (ok) { n <- nrow(sx_read(f)); sx_add_chk(paste0("file ", f, ": rows"), n == sx_expect$rows[i],
                                               sprintf("%d rows, expected %d", n, sx_expect$rows[i])) }
}
sx_has <- function(f) any(sx_expect$file == f & sx_expect$required)
# ---- execution scope (AUD135-04): the ledger, and replayed = committed ----
sx_compare("execution ledger", "execution_ledger.csv", sx_ledger, sx_schema$ledger)
sx_rep_parts <- c(if (!is.null(sx_F)) paste0(sx_F$split, ":fold", sx_F$fold), if (!is.null(sx_B)) paste0("boot", sx_B$b))
sx_rep_status <- c(if (!is.null(sx_F)) as.character(sx_F$status), if (!is.null(sx_B)) as.character(sx_B$status))
sx_com <- sx_ledger[sx_ledger$committed, , drop = FALSE]
sx_add_chk("replay scope: the replayed parts are exactly the committed parts", setequal(sx_rep_parts, sx_com$part) &&
           length(sx_rep_parts) == nrow(sx_com), sprintf("replayed %d, committed %d", length(sx_rep_parts), nrow(sx_com)))
sx_add_chk("replay scope: replayed statuses equal the committed statuses",
           identical(unname(sx_rep_status[match(sx_com$part, sx_rep_parts)]), as.character(sx_com$status)),
           "status of every committed part")
# ---- planned manifests: committed parts from the replayed chunk code, pending parts from the embedded plan ----
if (sx_has("split_manifest.csv")) {
  M <- rbind(if (!is.null(sx_M)) sx_M[, c("split", "row_id", "fold")], sx_pend_M)
  if (!is.null(M) && nrow(M)) {
    M$mode <- sx_plan_splits$mode[match(M$split, sx_plan_splits$split)]
    M$`repeat` <- sx_plan_splits$`repeat`[match(M$split, sx_plan_splits$split)]
  }
  sx_compare("split manifest", "split_manifest.csv", M, sx_schema$manifest, skip = c("participant", "cluster"))
  app <- sx_read("split_manifest.csv")
  if (!is.null(app) && nrow(app)) {
    sx_add_chk("split manifest: every row is an analysed row", !anyNA(sx_pos(app$row_id)), "row_id in the analysed data")
    sx_add_chk("split manifest: participant keys identify the analysed participants",
               sx_partition_ok(app$participant, sx_unit_part(app$row_id)), "same key <=> same participant in R")
    sx_add_chk("split manifest: cluster keys identify the analysed clusters",
               sx_partition_ok(app$cluster, sx_unit_clus(app$row_id)), "same key <=> same cluster in R")
    sx_M_all <- M
  } else sx_M_all <- NULL
} else sx_M_all <- NULL
if (sx_has("bootstrap_draws.csv")) {
  D <- rbind(if (!is.null(sx_D)) sx_D, sx_pend_D)
  sx_compare("bootstrap draw manifest", "bootstrap_draws.csv", D, sx_schema$draws,
             skip = c("original_participant", "original_cluster"))
  app <- sx_read("bootstrap_draws.csv")
  if (!is.null(app) && nrow(app)) {
    sx_add_chk("bootstrap draw manifest: every row is an analysed row", !anyNA(sx_pos(app$row_id)), "row_id")
    sx_add_chk("bootstrap draw manifest: original participant keys identify the analysed participants",
               sx_partition_ok(app$original_participant, sx_unit_part(app$row_id)), "same key <=> same participant")
    sx_add_chk("bootstrap draw manifest: original cluster keys identify the analysed clusters",
               sx_partition_ok(app$original_cluster, sx_unit_clus(app$row_id)), "same key <=> same cluster")
    man <- sx_read("split_manifest.csv")
    if (!is.null(man) && nrow(man)) {
      j <- match(app$row_id, man$row_id)
      sx_add_chk("bootstrap draw manifest: original keys equal the split manifest's keys of the same rows",
                 !anyNA(j) && identical(app$original_participant, man$participant[j]) &&
                   identical(app$original_cluster, man$cluster[j]), "cross-file identity")
    }
  }
}
# ---- reproduced results ----
if (sx_has("predictions.csv")) sx_compare("predictions", "predictions.csv", sx_P, sx_schema$predictions)
if (sx_has("folds.csv")) {
  Fr <- sx_F
  if (!is.null(Fr) && nrow(Fr)) { Fr$mode <- sx_plan_splits$mode[match(Fr$split, sx_plan_splits$split)]
    Fr$`repeat` <- sx_plan_splits$`repeat`[match(Fr$split, sx_plan_splits$split)] }
  sx_compare("folds", "folds.csv", Fr, sx_schema$folds)
}
if (sx_has("performance.csv"))
  sx_compare("performance (recomputed in R from the reproduced predictions)", "performance.csv",
             sx_perf(sx_P, sx_F, sx_binary, sx_target), sx_schema$performance)
if (sx_has("realized_sizes.csv"))
  sx_compare("realized sizes (recomputed from the verified plan and the analysed units)", "realized_sizes.csv",
             sx_realized(sx_M_all), sx_schema$realized)
if (sx_has("bootstrap_results.csv")) {
  Br <- sx_B
  if (!is.null(Br) && nrow(Br)) Br$coefficient_label <- sx_coef_display(Br$coefficient)
  sx_compare("bootstrap draws", "bootstrap_results.csv", Br, sx_schema$bootstrap)
}
if (sx_has("bootstrap_coefficients.csv")) {
  sx_compare("bootstrap coefficients", "bootstrap_coefficients.csv", sx_C, sx_schema$coefs)
  sx_add_chk("bootstrap coefficients: identity (draw, selected candidate, declared key, term, n_coef)",
             isTRUE(tryCatch(sx_coef_identity(sx_B, sx_C), error = function(e) FALSE)), "reproduced coefficients")
}
if (sx_has("stability.csv"))
  sx_compare("stability (recomputed from the reproduced draws and the declared pool)", "stability.csv",
             sx_stability(sx_B), sx_schema$stability)
if (sx_has("bootstrap_estimands.csv"))
  sx_compare("bootstrap estimands (recomputed from the reproduced coefficients)", "bootstrap_estimands.csv",
             sx_estimands(sx_B, sx_C), sx_schema$estimands)
if (sx_has("candidate_trace.csv")) sx_compare("candidate trace", "candidate_trace.csv", sx_T, sx_schema$trace)
utils::write.csv(sx_chk, "validation_check_results.csv", row.names = FALSE)
sx_fail <- sx_chk$status == "FAIL"
print(sx_chk[sx_chk$status != "PASS", , drop = FALSE], row.names = FALSE)
sx_n_plan <- nrow(sx_ledger); sx_n_com <- sum(sx_ledger$committed)
cat(sprintf("Execution coverage: %d of %d planned parts committed and replayed; %d not run; %d dispatched without a validated response (not reproducible).\n",
            sx_n_com, sx_n_plan, sum(sx_ledger$status == "not_run"), sum(sx_ledger$status == "uncommitted")))
if (any(sx_fail)) {
  msg <- paste0("VERIFICATION FAILED: ", sum(sx_fail), " of ", nrow(sx_chk), " checks (see validation_check_results.csv): ",
                paste(utils::head(sx_chk$check[sx_fail], 5), collapse = "; "))
  if (interactive()) stop(msg, call. = FALSE) else { message(msg); quit(save = "no", status = 1) }
}
if (!is.null(sx_P)) cat("Predictions reproduced:", nrow(sx_P), "held-out rows.\n")
if (!is.null(sx_B)) cat("Bootstrap selections reproduced:", nrow(sx_B), "draws.\n")
cat(sprintf("All %d checks passed (%d not applicable).\n", sum(sx_chk$status == "PASS"), sum(sx_chk$status == "NOT APPLICABLE")))
cat(if (identical(sx_app_status, "complete") && sx_n_com == sx_n_plan) "The complete validation was reproduced.\n" else
    paste0("The export is ", toupper(sx_app_status), ": only its ", sx_n_com, " committed part(s) of ", sx_n_plan,
           " planned were reproduced; this is not a reproduction of a complete validation.\n"))
"""


def _r_frame(cols: list[tuple[str, str]], recs: list[dict[str, Any]]) -> str:
    """An R data.frame literal of `recs` (typed columns; character for text, numeric otherwise; NA for missing)."""
    if not recs:
        return "NULL"
    parts = []
    for name, typ in cols:
        vals = [r.get(name) for r in recs]
        if typ == "text":
            parts.append(f"`{name}` = {r_vec(['' if v is None else str(v) for v in vals])}")
        elif typ == "bool":
            parts.append(f"`{name}` = c({', '.join('TRUE' if v else 'FALSE' for v in vals)})")
        else:
            parts.append(f"`{name}` = c({', '.join('NA' if v is None or (isinstance(v, float) and v != v) else repr(float(v)) if typ == 'num' else str(int(v)) for v in vals)})")
    return "data.frame(" + ", ".join(parts) + ", stringsAsFactors = FALSE, check.names = FALSE)"


def _r_schema() -> str:
    out = []
    for key, spec in EVIDENCE.items():
        cols = spec["cols"]
        out.append(f"  {key} = list(keys = {r_vec(spec['keys'])}, status_ok = "
                   + (r_str(spec["status_ok"]) if spec.get("status_ok") else "NULL")
                   + f", cols = data.frame(name = {r_vec([c[0] for c in cols])}, type = {r_vec([c[1] for c in cols])}, "
                     f"flags = {r_vec([c[2] for c in cols])}, stringsAsFactors = FALSE))")
    return "sx_schema <- list(\n" + ",\n".join(out) + ")"


def reproduce_script(load: str, setup: str, chunks: list[str], expected: list[dict[str, Any]] | None = None,
                     meta: dict[str, Any] | None = None, replay: dict[str, Any] | None = None) -> str:
    """V134-008 / AUD134-11 / AUD135-01/02/04: a complete script that repeats the validation outside StatExplore - the
    data load, the search setup and helpers, the exact split / draw manifests and fit budget of every COMMITTED chunk -
    collects the predictions, folds, bootstrap draws and coefficients, candidate trace and the executed manifests,
    writes them, and verifies every exported evidence file field by declared field (see R_REPRO_CHECKS). The planned
    scope that was not committed (pending splits / draws), the execution ledger, the declared evidence schema and the
    expected files are embedded, so a removed, truncated or altered file - or a partial run presented as complete -
    fails a named check."""
    meta = meta or {}
    exp = expected or []
    rp = replay or {"ledger": [], "pending_manifest": [], "pending_draws": [], "splits": [],
                    "config": {"holdout_fraction": 0.2, "k": 5}, "alias_map": {}, "rules": {}}
    L = ["# ==== StatExplore AMD validation: complete reproduction ====",
         "# Run from the exported folder (Rscript validation_reproduce.R). It needs the data file named in the data-load",
         "# section and verifies its results against the CSV files StatExplore exported (same folder).",
         load, setup,
         "sx_rep_pred <- list(); sx_rep_folds <- list(); sx_rep_boot <- list(); sx_rep_coef <- list()",
         "sx_rep_trace <- list(); sx_rep_man <- list(); sx_rep_draws <- list()"]
    for k, body in enumerate(chunks, start=1):
        L += [f"# ---- committed chunk {k} ----", "sx_vfold <- NULL; sx_vdraws <- NULL; sx_vr <- NULL", body,
              "if (!is.null(sx_vfold)) sx_rep_man[[length(sx_rep_man) + 1]] <- data.frame(split = sx_vr$folds$split[1],",
              "  row_id = as.integer(names(sx_vfold)), fold = unname(sx_vfold), stringsAsFactors = FALSE)",
              "if (!is.null(sx_vdraws)) sx_rep_draws[[length(sx_rep_draws) + 1]] <- data.frame(b = sx_vdraws$b,",
              "  position = stats::ave(seq_along(sx_vdraws$b), sx_vdraws$b, FUN = seq_along), row_id = sx_vdraws$row_id,",
              "  draw_participant = sx_vdraws$draw_gid, draw_cluster = sx_vdraws$draw_cluster, stringsAsFactors = FALSE)",
              "if (!is.null(sx_result$predictions)) sx_rep_pred[[length(sx_rep_pred) + 1]] <- sx_result$predictions",
              "if (!is.null(sx_result$folds)) sx_rep_folds[[length(sx_rep_folds) + 1]] <- sx_result$folds",
              "if (!is.null(sx_result$trace)) sx_rep_trace[[length(sx_rep_trace) + 1]] <- sx_result$trace",
              "if (!is.null(sx_result$boot)) { sx_rep_boot[[length(sx_rep_boot) + 1]] <- sx_result$boot$draws",
              "  if (!is.null(sx_result$boot$trace)) sx_rep_trace[[length(sx_rep_trace) + 1]] <- sx_result$boot$trace",
              "  if (!is.null(sx_result$boot$coefs)) sx_rep_coef[[length(sx_rep_coef) + 1]] <- sx_result$boot$coefs }"]
    sp = rp.get("splits") or []
    L += ["sx_bind <- function(x) if (length(x)) do.call(rbind, x) else NULL",
          "sx_P <- sx_bind(sx_rep_pred); sx_F <- sx_bind(sx_rep_folds); sx_B <- sx_bind(sx_rep_boot)",
          "sx_C <- sx_bind(sx_rep_coef); sx_T <- sx_bind(sx_rep_trace); sx_M <- sx_bind(sx_rep_man); sx_D <- sx_bind(sx_rep_draws)",
          'if (!is.null(sx_P)) utils::write.csv(sx_P, "reproduced_predictions.csv", row.names = FALSE)',
          'if (!is.null(sx_F)) utils::write.csv(sx_F, "reproduced_folds.csv", row.names = FALSE)',
          'if (!is.null(sx_B)) utils::write.csv(sx_B, "reproduced_bootstrap.csv", row.names = FALSE)',
          'if (!is.null(sx_C)) utils::write.csv(sx_C, "reproduced_coefficients.csv", row.names = FALSE)',
          'if (!is.null(sx_T)) utils::write.csv(sx_T, "reproduced_trace.csv", row.names = FALSE)',
          "# ---- declarations embedded by StatExplore (independent of the exported files under test) ----",
          _r_schema(),
          "sx_expect <- data.frame(file = " + r_vec([e["file"] for e in exp]) + ", required = c("
          + ", ".join("TRUE" if e["required"] else "FALSE" for e in exp) + "), rows = c("
          + ", ".join(str(int(e["rows"])) for e in exp) + "), applicable = c("
          + ", ".join("TRUE" if e["applicable"] else "FALSE" for e in exp) + "), stringsAsFactors = FALSE)"
          if exp else "sx_expect <- data.frame(file = character(), required = logical(), rows = integer(), "
                      "applicable = logical())",
          "sx_ledger <- " + (_r_frame([("part", "text"), ("mode", "text"), ("split", "text"), ("fold", "int"),
                                        ("b", "int"), ("chunk", "int"), ("status", "text"), ("committed", "bool")],
                                       rp.get("ledger") or []) if rp.get("ledger") else
                             "data.frame(part = character(), mode = character(), split = character(), fold = numeric(), "
                             "b = numeric(), chunk = numeric(), status = character(), committed = logical())"),
          "sx_pend_M <- " + _r_frame([("split", "text"), ("row_id", "int"), ("fold", "int")],
                                     rp.get("pending_manifest") or []) + "   # planned, not committed (not replayed)",
          "sx_pend_D <- " + _r_frame([("b", "int"), ("position", "int"), ("row_id", "int"), ("draw_participant", "text"),
                                      ("draw_cluster", "text")], rp.get("pending_draws") or [])
          + "   # planned, not committed",
          "sx_plan_splits <- " + (_r_frame([("split", "text"), ("mode", "text"), ("repeat", "int"), ("rule", "text")],
                                           [dict(x, rule=(rp.get("rules") or {}).get(x["split"], "")) for x in sp])
                                  if sp else "data.frame(split = character(), mode = character(), `repeat` = numeric(), "
                                             "rule = character(), check.names = FALSE)"),
          f"sx_val_cfg <- list(holdout_fraction = {float((rp.get('config') or {}).get('holdout_fraction', 0.2))!r}, "
          f"k = {int((rp.get('config') or {}).get('k', 5))})",
          "sx_alias <- " + ("c(" + ", ".join(f"{a} = {r_str(n)}" for a, n in sorted((rp.get("alias_map") or {}).items()))
                            + ")" if rp.get("alias_map") else "character()"),
          f"sx_app_status <- {r_str(str(meta.get('status', 'complete')))}",
          f"sx_binary <- {'TRUE' if meta.get('binary') else 'FALSE'}",
          f"sx_target <- {r_str(str(meta.get('target', 'participant')))}",
          "if (sx_binary) sx_schema$performance$cols <- rbind(sx_schema$performance$cols, data.frame(name = c(\"log_loss\", "
          "\"Brier\", \"AUC\"), type = \"num\", flags = \"na tol=tol\", stringsAsFactors = FALSE)) else "
          "sx_schema$performance$cols <- rbind(sx_schema$performance$cols, data.frame(name = c(\"RMSE\", \"MAE\", "
          "\"R2_out_of_sample\"), type = \"num\", flags = \"na tol=tol\", stringsAsFactors = FALSE))",
          R_REPRO_CHECKS]
    return "\n".join(L) + "\n"


def csv_text(recs: list[dict[str, Any]]) -> str:
    if not recs:
        return ""
    buf = io.StringIO()
    wr = csv.DictWriter(buf, fieldnames=list(dict.fromkeys(k for r in recs for k in r)), lineterminator="\n")
    wr.writeheader()
    wr.writerows(recs)
    return buf.getvalue()
