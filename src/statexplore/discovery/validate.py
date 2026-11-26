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

GRAMMAR = "validation-1"

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
RULES = {"p": "Smallest training p-value of the candidate's primary test (the search's discovery ranking)",
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
    max_fits: int = 20000                 # computational budget (candidate fits), checked before running

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
    if cfg.pipeline == "fixed" and cfg.candidate is None:
        problems.append("Choose the candidate to validate (fixed-candidate mode).")
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
    n_cand = 1 if cfg.pipeline == "fixed" else max(1, run.total)
    fits = n_cand * ((1 if "holdout" in cfg.modes else 0) + (cfg.k * cfg.repeats if "cv" in cfg.modes else 0)
                     + (cfg.B if "bootstrap" in cfg.modes else 0))
    info["fits"] = int(fits)
    if fits > cfg.max_fits:
        problems.append(f"About {fits:,} model fits exceed the budget of {cfg.max_fits:,}: reduce K, repeats or B, or use "
                        "fixed-candidate mode.")
    return info


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
sx_val_score <- function(a) { r <- tryCatch(sx_val_with(a, function() do.call(sx_discover_one, a)), error = function(e) NULL)
  if (is.null(r) || !(r$status %in% c("ok", "warning"))) return(NA_real_)
  if (sx_val_rule == "aic") r$aic else r$p }
sx_val_predict <- function(m, newd) {
  if (inherits(m, "merMod")) {
    rf <- switch(sx_val_target, participant_in_cluster = ~ (1 | sx_cluster), NA)
    return(as.numeric(stats::predict(m, newdata = newd, re.form = rf, allow.new.levels = TRUE)))
  }
  if (inherits(m, "lme")) return(as.numeric(stats::predict(m, newdata = newd,
                                 level = if (sx_val_target == "participant_in_cluster") 1 else 0)))
  as.numeric(stats::predict(m, newdata = newd, type = "response"))
}
sx_val_coef <- function(m, a) {   # coefficient(s) of the candidate's primary tested term(s)
  b <- tryCatch(if (inherits(m, "merMod")) lme4::fixef(m) else if (inherits(m, "lme")) nlme::fixef(m) else stats::coef(m),
                error = function(e) NULL)
  if (is.null(b) || !length(a$tests)) return(list(name = "", value = NA_real_, k = 0L))
  drop <- sx_canon_names(unlist(a$tests[[1]]$drop))
  f <- stats::reformulate(if (length(a$full)) a$full else "1", response = "sx_y")
  lab <- sx_canon_names(attr(stats::terms(f), "term.labels"))
  X <- tryCatch(stats::model.matrix(stats::delete.response(stats::terms(f)), sx_val_lastd), error = function(e) NULL)
  if (is.null(X)) return(list(name = "", value = NA_real_, k = 0L))
  cols <- colnames(X)[attr(X, "assign") %in% which(lab %in% drop)]
  cols <- intersect(cols, names(b))
  if (!length(cols)) return(list(name = "", value = NA_real_, k = 0L))
  list(name = cols[1], value = unname(b[[cols[1]]]), k = length(cols))
}
sx_val_select <- function(train) {
  sx_dd <<- train
  if (length(sx_val_cands) == 1) return(list(j = 1L, score = NA_real_, scores = NA_real_))
  sc <- vapply(sx_val_cands, sx_val_score, 0)
  if (all(is.na(sc))) return(list(j = NA_integer_, score = NA_real_, scores = sc))
  list(j = which.min(sc), score = min(sc, na.rm = TRUE), scores = sc)
}
sx_val_split <- function(split_id, fold_of) {
  folds <- sort(unique(fold_of[!is.na(fold_of) & fold_of > 0]))
  preds <- list(); info <- list()
  for (f in folds) {
    tr <- sx_val_full[is.na(fold_of) | fold_of != f, , drop = FALSE]
    te <- sx_val_full[!is.na(fold_of) & fold_of == f, , drop = FALSE]
    res <- tryCatch({
      if (length(intersect(rownames(tr), rownames(te)))) stop("internal: a row is in both parts")
      if (sx_val_target != "rows" && length(intersect(as.character(tr$sx_gid), as.character(te$sx_gid))))
        stop("internal: a participant is in both parts")
      if (sx_val_target == "cluster" && length(intersect(as.character(tr$sx_cluster), as.character(te$sx_cluster))))
        stop("internal: a cluster is in both parts")
      if (sx_val_target == "participant_in_cluster" && !all(as.character(te$sx_cluster) %in% as.character(tr$sx_cluster)))
        stop("internal: a test cluster has no training rows")
      sel <- sx_val_select(tr)
      if (is.na(sel$j)) stop("no candidate could be fitted on the training part")
      a <- sx_val_cands[[sel$j]]
      m <- sx_val_fit(a, tr)
      trc <- sx_complete(tr, sx_val_need(a))
      tec <- sx_complete(te, sx_val_need(a)); n_incomplete <- nrow(te) - nrow(tec)
      seen <- sx_val_seen(trc, tec, setdiff(a$vars, c("sx_timef", "sx_gid", "sx_cluster")))
      n_unseen <- sum(!seen); tec <- tec[seen, , drop = FALSE]
      if (sx_long() && sx_disc_has_time) tec$sx_occ <- as.integer(tec$sx_timef)
      p <- sx_val_predict(m, tec)
      ok <- is.finite(p)
      preds[[length(preds) + 1]] <- data.frame(split = split_id, fold = f, row_id = as.integer(rownames(tec))[ok],
        observed = tec$sx_y[ok], predicted = p[ok],
        participant = if (sx_long()) as.character(tec$sx_gid)[ok] else as.character(rownames(tec))[ok],
        cluster = if (sx_long() && sx_disc_has_cluster) as.character(tec$sx_cluster)[ok] else "", stringsAsFactors = FALSE)
      list(split = split_id, fold = f, status = "scored", reason = "", selected = sx_val_labels[sel$j],
           selected_index = sx_val_index[sel$j], train_score = sel$score, n_train = nrow(trc), n_test = sum(ok),
           train_mean = mean(trc$sx_y), train_groups = if (sx_long()) nlevels(droplevels(trc$sx_gid)) else nrow(trc),
           test_groups = if (sx_long()) length(unique(as.character(tec$sx_gid[ok]))) else sum(ok),
           train_clusters = if (sx_long() && sx_disc_has_cluster) nlevels(droplevels(trc$sx_cluster)) else 0L,
           test_clusters = if (sx_long() && sx_disc_has_cluster) length(unique(as.character(tec$sx_cluster[ok]))) else 0L,
           excluded_incomplete = n_incomplete, excluded_unseen_levels = n_unseen, excluded_unpredicted = sum(!ok),
           test_events = if (sx_disc_family == "binomial" || identical(sx_disc_gee_family, "binomial")) sum(tec$sx_y[ok] == 1) else NA_real_,
           diagnostics = tryCatch(paste(sx_diag(m, sx_last_run)$flags, collapse = "; "), error = function(e) ""))
    }, error = function(e) list(split = split_id, fold = f, status = "failed", reason = conditionMessage(e),
                                 selected = "", selected_index = NA_integer_, train_score = NA_real_, n_train = nrow(tr),
                                 n_test = 0L, train_mean = NA_real_, train_groups = NA_real_, test_groups = NA_real_,
                                 train_clusters = NA_real_, test_clusters = NA_real_, excluded_incomplete = NA_real_,
                                 excluded_unseen_levels = NA_real_, excluded_unpredicted = NA_real_, test_events = NA_real_,
                                 diagnostics = ""))
    info[[length(info) + 1]] <- as.data.frame(res, stringsAsFactors = FALSE)
  }
  sx_dd <<- sx_val_full
  list(folds = do.call(rbind, info), predictions = if (length(preds)) do.call(rbind, preds) else NULL)
}
sx_val_boot <- function(draws) {
  out <- list()
  for (b in sort(unique(draws$b))) {
    dr <- draws[draws$b == b, , drop = FALSE]
    res <- tryCatch({
      d <- sx_val_full[match(as.character(dr$row_id), rownames(sx_val_full)), , drop = FALSE]
      keep <- !is.na(match(as.character(dr$row_id), rownames(sx_val_full)))
      d <- d[keep, , drop = FALSE]; dr <- dr[keep, , drop = FALSE]
      if (sx_long()) {
        d$sx_gid <- factor(dr$draw_gid)            # a unit drawn twice = two sampled units (new IDs)
        if (sx_disc_has_cluster) d$sx_cluster <- factor(ifelse(nzchar(dr$draw_cluster), dr$draw_cluster, as.character(d$sx_cluster)))
        if (sx_disc_has_time) d <- d[order(d$sx_gid, d$sx_time), , drop = FALSE]
      }
      rownames(d) <- paste0("b", seq_len(nrow(d)))
      sel <- sx_val_select(d)
      if (is.na(sel$j)) stop("no candidate could be fitted on this resample")
      a <- sx_val_cands[[sel$j]]
      m <- sx_val_fit(a, d)
      sx_val_lastd <<- sx_complete(d, sx_val_need(a))
      cf <- sx_val_coef(m, a)
      dg <- tryCatch(sx_diag(m, sx_last_run), error = function(e) list(status = "unknown", flags = character()))
      list(b = b, status = "fitted", reason = "", selected = sx_val_labels[sel$j], selected_index = sx_val_index[sel$j],
           coefficient = cf$name, estimate = cf$value, n_coef = cf$k, diagnostics = paste(dg$status, collapse = ""),
           flags = paste(dg$flags, collapse = "; "), n = nrow(sx_val_lastd),
           groups = if (sx_long()) nlevels(droplevels(sx_val_lastd$sx_gid)) else nrow(sx_val_lastd))
    }, error = function(e) list(b = b, status = "failed", reason = conditionMessage(e), selected = "",
                                 selected_index = NA_integer_, coefficient = "", estimate = NA_real_, n_coef = 0L,
                                 diagnostics = "failed", flags = "", n = NA_real_, groups = NA_real_))
    out[[length(out) + 1]] <- as.data.frame(res, stringsAsFactors = FALSE)
  }
  sx_dd <<- sx_val_full
  do.call(rbind, out)
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
        cands = list(enumerate_candidates(s))
        valid = {int(r["idx"]) for r in run.rows if r.get("status") not in ("invalid",)} if run.rows else None
        if cfg.pipeline == "fixed":
            cands = [c for c in cands if c.index == cfg.candidate]
            if not cands:
                raise ValidationError(f"Candidate #{cfg.candidate} is not part of this search.")
        elif valid is not None:
            cands = [c for c in cands if c.index in valid] or cands
        self.cands = cands
        self.labels = [focal_label(self.v, c) if c.kind != "structure" else c.structure + (
            "" if c.clustered is None else (" (with cluster)" if c.clustered else " (without cluster)")) for c in cands]
        self.info = info
        self.units = unit_frame(ds, run)
        self.splits = make_splits(self.units, cfg)
        for sp in self.splits:
            bad = check_split(self.units, sp, cfg.target)
            if bad:
                raise ValidationError("Internal split check failed: " + "; ".join(bad))
        self.draws = make_draws(self.units, cfg) if "bootstrap" in cfg.modes else []
        self.scripts: list[str] = []

    def _setup(self) -> str:
        from .rcode import candidate_args, setup_script

        s = self.run.settings
        setup = setup_script(self.v, s, self.run.family).replace("sx_result <- list(", "sx_setup_info <- list(")
        return "\n".join([
            setup, R_VALIDATE,
            "sx_val_cands <- list(" + ",\n  ".join(f"list({candidate_args(self.v, c, s)})" for c in self.cands) + ")",
            f"sx_val_labels <- {r_vec(self.labels)}",
            f"sx_val_index <- {r_vec([int(c.index) for c in self.cands])}",
            f"sx_val_rule <- {r_str(self.cfg.rule)}",
            f"sx_val_target <- {r_str(self.cfg.target)}",
            "if (!exists(\"sx_disc_gee_family\")) sx_disc_gee_family <- \"\"",
            "sx_val_full <- sx_dd; sx_val_lastd <- sx_dd",
        ])

    def _call(self, body: str, title: str) -> Any:
        from ..discovery.engine import ENGINE_PACKAGES

        script = self._setup() + "\n" + body
        self.scripts.append(f"# ==== {title} ====\n" + body)
        res, raw = self.session.run(self.v, kind="amd_validation", title=title, body=script,
                                    packages=ENGINE_PACKAGES, inputs={}, timeout=3600)
        if raw is None:
            raise ValidationError("; ".join(res.messages) or "R failed")
        return raw

    def execute(self, should_stop: Callable[[], bool] = lambda: False,
                progress: Callable[[str], None] = lambda m: None) -> dict[str, Any]:
        cfg = self.cfg
        folds, preds, boot, problems = [], [], [], []
        status = "complete"
        try:
            for sp in self.splits:
                if should_stop():
                    status = "cancelled"
                    break
                progress(f"{MODES[sp['mode']]} {sp['id']}")
                rows = sorted(sp["folds"].items())
                body = "\n".join([
                    f"sx_vfold <- c({', '.join(str(f) for _, f in rows)})",
                    f"names(sx_vfold) <- c({', '.join(str(r) for r, _ in rows)})",
                    "sx_vf <- unname(sx_vfold[rownames(sx_val_full)])",
                    f"sx_vr <- sx_val_split({r_str(sp['id'])}, sx_vf)",
                    "sx_result <- list(folds = sx_vr$folds, predictions = sx_vr$predictions)",
                ])
                raw = self._call(body, f"AMD validation: {sp['id']}")
                fr = _frame(raw.get("folds"))
                fr["mode"], fr["repeat"] = sp["mode"], sp["repeat"]
                folds.append(fr)
                preds.append(_frame(raw.get("predictions")))
            chunk = 10
            for i in range(0, len(self.draws), chunk):
                if status != "complete" or should_stop():
                    status = "cancelled"
                    break
                part = self.draws[i:i + chunk]
                progress(f"bootstrap draws {part[0]['b']}–{part[-1]['b']} of {cfg.B}")
                recs = [{"b": d["b"], "row_id": r[0], "draw_gid": r[1], "draw_cluster": r[2]} for d in part for r in d["rows"]]
                df = pd.DataFrame(recs)
                body = "\n".join([
                    f"sx_vdraws <- data.frame(b = c({', '.join(map(str, df['b']))}), row_id = c({', '.join(map(str, df['row_id']))}),",
                    f"  draw_gid = {r_vec(list(df['draw_gid']))}, draw_cluster = {r_vec(list(df['draw_cluster']))}, stringsAsFactors = FALSE)",
                    "sx_result <- list(boot = sx_val_boot(sx_vdraws))",
                ])
                raw = self._call(body, f"AMD validation: bootstrap {part[0]['b']}-{part[-1]['b']}")
                boot.append(_frame(raw.get("boot")))
        except Exception as e:  # cancellation kills the worker: completed chunks stay, the run is partial
            problems.append(str(e))
            status = "cancelled" if should_stop() else "failed"
        return self.assemble(folds, preds, boot, status, problems)

    # ---- aggregation ----
    def assemble(self, folds: list[pd.DataFrame], preds: list[pd.DataFrame], boot: list[pd.DataFrame], status: str,
                 problems: list[str]) -> dict[str, Any]:
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
        summary = performance(F, P, binary, cfg.target) if len(P) else []
        stability = stability_table(Bt, self.labels) if len(Bt) else []
        manifest = [{"split": sp["id"], "mode": sp["mode"], "repeat": sp["repeat"], "row_id": int(r),
                     "participant": pk, "cluster": ck, "fold": int(sp["folds"][int(r)])}
                    for sp in self.splits for r, pk, ck in zip(self.units["row_id"], self.units["participant"],
                                                                 self.units["cluster"])]
        draws = [{"b": d["b"], "row_id": r[0], "draw_participant": r[1], "draw_cluster": r[2],
                  "original_participant": r[3], "original_cluster": r[4]} for d in self.draws for r in d["rows"]]
        return {
            "grammar": GRAMMAR, "config": cfg.to_dict(), "status": status, "problems": problems,
            "search_id": self.run.id, "search_identity": getattr(self.run, "identity", "") or "",
            "target_label": TARGETS[cfg.target][0], "target_estimand": TARGETS[cfg.target][1],
            "pipeline_label": PIPELINES[cfg.pipeline], "rule_label": RULES[cfg.rule] if cfg.pipeline == "pipeline" else "",
            "candidates": [{"index": int(c.index), "label": lab} for c, lab in zip(self.cands, self.labels)],
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


def stability_table(Bt: pd.DataFrame, labels: list[str]) -> list[dict[str, Any]]:
    """Selection frequency with BOTH denominators (all attempted draws, successful draws), sign stability conditional
    on selection, percentiles of the estimate among draws that selected the candidate. Failed draws are never zeros."""
    att = int(len(Bt))
    ok = Bt[Bt["status"] == "fitted"]
    nok = int(len(ok))
    out = []
    for lab in labels:
        sel = ok[ok["selected"] == lab]
        est = pd.to_numeric(sel["estimate"], errors="coerce").dropna()
        rec = {"candidate": lab, "selected": int(len(sel)), "attempted": att, "successful": nok,
               "freq_of_attempted": len(sel) / att if att else float("nan"),
               "freq_of_successful": len(sel) / nok if nok else float("nan"),
               "flagged_fits": int((sel["diagnostics"].astype(str) != "clean").sum()) if len(sel) else 0,
               "coefficient": sel["coefficient"].iloc[0] if len(sel) else "",
               "positive_sign": float((est > 0).mean()) if len(est) else float("nan"),
               "estimate_p2.5": float(est.quantile(0.025)) if len(est) else float("nan"),
               "estimate_median": float(est.median()) if len(est) else float("nan"),
               "estimate_p97.5": float(est.quantile(0.975)) if len(est) else float("nan")}
        out.append(rec)
    return out


# ---- result tables / export -----------------------------------------------------------------------------------------

def summary_text(v: dict[str, Any]) -> str:
    c = v["config"]
    parts = [f"Validation status: {v['status'].upper()}" + (f" ({'; '.join(v['problems'])})" if v["problems"] else ""),
             f"Target: {v['target_label']} — {v['target_estimand']}.",
             f"Procedure: {v['pipeline_label']}" + (f"; selection rule: {v['rule_label']}" if v["rule_label"] else "") + ".",
             f"Modes: {', '.join(MODES[m] for m in c['modes'])}; holdout {c['holdout_fraction']:.0%}, K = {c['k']}, "
             f"repeats = {c['repeats']}, B = {c['B']}, seed = {c['seed']}.",
             "This is a predictive check of the stated target, separate from the discovery p-values; it is not a "
             "confirmatory test, and bootstrap selection frequencies are not probabilities that a hypothesis is true."]
    if c["pipeline"] == "fixed":
        parts.append("Fixed candidate: the model was chosen on ALL the data, so these scores are a conditional "
                     "(post-selection) assessment, not an independent validation of the search.")
    return "\n".join(parts)


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
    if v["bootstrap"]:
        tab("Bootstrap draws", v["bootstrap"], "One row per draw (failed draws keep their reason).")
    return out


def export(v: dict[str, Any], folder: str) -> list[str]:
    """Write the validation evidence: config/summary JSON, fold manifest, bootstrap draw map, predictions, folds,
    performance, stability CSVs, and the exact R code of every chunk."""
    import json
    from pathlib import Path

    d = Path(folder)
    d.mkdir(parents=True, exist_ok=True)
    files = []

    def w(name: str, text: str) -> None:
        (d / name).write_text(text, encoding="utf-8", newline="\n")
        files.append(str(d / name))
    meta = {k: v[k] for k in ("grammar", "config", "status", "problems", "search_id", "search_identity",
                              "target_label", "target_estimand", "pipeline_label", "rule_label", "candidates",
                              "feasibility", "rng")}
    w("validation_manifest.json", json.dumps(meta, indent=1, default=str))
    for key, name in (("manifest", "split_manifest.csv"), ("draws", "bootstrap_draws.csv"),
                      ("predictions", "predictions.csv"), ("folds", "folds.csv"), ("performance", "performance.csv"),
                      ("stability", "stability.csv"), ("bootstrap", "bootstrap_results.csv")):
        if v.get(key):
            w(name, _csv(v[key]))
    w("validation_code.R", "# R code executed for each chunk (after the project's data-load script and the search setup).\n"
      + "\n\n".join(v.get("scripts") or []))
    w("README.txt", summary_text(v) + "\n\nFiles: split_manifest.csv (fold of every analysed row per split; fold 0 = "
      "training only), bootstrap_draws.csv (draw IDs and original IDs), predictions.csv (held-out rows), folds.csv, "
      "performance.csv, stability.csv, validation_code.R.\n")
    return files


def csv_text(recs: list[dict[str, Any]]) -> str:
    if not recs:
        return ""
    buf = io.StringIO()
    wr = csv.DictWriter(buf, fieldnames=list(dict.fromkeys(k for r in recs for k in r)), lineterminator="\n")
    wr.writeheader()
    wr.writerows(recs)
    return buf.getvalue()
