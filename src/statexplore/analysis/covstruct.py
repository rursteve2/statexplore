"""Gaussian longitudinal covariance structures, shared by the Mixed workspace, its comparisons, Discovery
workflows A and B and every exported script (v1.3.2, STAT-21 / H1).

A structure is (random effects, residual correlation, optional higher-level cluster). The R text that builds the
random part and the correlation object comes from ONE set of R helpers (`MIXED_R_HELPERS`), so a fit, its
comparison and its exported code cannot disagree about the hierarchy:

  random     lme4 (independent residuals)                 nlme (independent / AR(1) / CAR(1))
  ri         (1 | id)                                     ~ 1 | id
  diag       (1 + t || id)                                list(id = pdDiag(~ 1 + t))
  rs         (1 + t | id)                                 ~ 1 + t | id
  + cluster  (1 | cluster) + <participant block>          ~ 1 | cluster/id;  list(cluster = pdDiag(~ 1), id = pd*(~ 1 + t))
  residual   —                                            corAR1(~ occasion | [cluster/]id), corCAR1(~ time | [cluster/]id)

With a cluster, participants are identified by the composite (cluster, ID) key (`sx_id`), so locally reused IDs never
merge. Never `~ t | cluster/id`, which would add slopes at both levels.
"""
from __future__ import annotations

from dataclasses import dataclass

RANDOM_TYPES = {"ri": "Random intercept", "diag": "Random intercept + time slope, uncorrelated (DIAG)",
                "rs": "Random intercept + time slope, correlated (RS)"}
RESIDUAL_TYPES = {"independent": "Independent", "ar1": "AR(1) over the observed-occasion index",
                  "car1": "Continuous-time AR(1) over actual time"}

# Discovery structure key -> (random, residual, engine). Older keys keep their meaning (saved projects, candidate IDs).
GAUSSIAN_STRUCTURES: dict[str, tuple[str, str, str]] = {
    "ri": ("ri", "independent", "lme4"), "rs_uncorr": ("diag", "independent", "lme4"), "rs": ("rs", "independent", "lme4"),
    "nlme_ri": ("ri", "independent", "nlme"), "nlme_diag": ("diag", "independent", "nlme"),
    "nlme_rs": ("rs", "independent", "nlme"),
    "ar1": ("ri", "ar1", "nlme"), "diag_ar1": ("diag", "ar1", "nlme"), "rs_ar1": ("rs", "ar1", "nlme"),
    "car1": ("ri", "car1", "nlme"), "diag_car1": ("diag", "car1", "nlme"), "rs_car1": ("rs", "car1", "nlme"),
}
# Mixed workspace random setting <-> structure random type
MIXED_RANDOM = {"intercept": "ri", "slope_uncorrelated": "diag", "slope": "rs"}
RANDOM_MIXED = {v: k for k, v in MIXED_RANDOM.items()}
# covariance parameters at the participant level (variances + covariances) and of the residual correlation
RANDOM_PARAMS = {"ri": 1, "diag": 2, "rs": 3}
RESIDUAL_PARAMS = {"independent": 0, "ar1": 1, "car1": 1}


def structure_key(random: str, residual: str, engine: str) -> str:
    for k, v in GAUSSIAN_STRUCTURES.items():
        if v == (random, residual, engine):
            return k
    raise KeyError((random, residual, engine))


@dataclass(frozen=True)
class Structure:
    random: str
    residual: str
    clustered: bool = False
    engine: str = "lme4"

    @property
    def label(self) -> str:
        return (f"{'Three-level' if self.clustered else 'Two-level'} {self.random.upper()} + "
                f"{RESIDUAL_TYPES[self.residual].split(' over')[0]}")

    @property
    def cov_params(self) -> int:
        """Variance-covariance parameters apart from the residual variance."""
        return RANDOM_PARAMS[self.random] + RESIDUAL_PARAMS[self.residual] + (1 if self.clustered else 0)


def from_key(key: str, clustered: bool = False) -> Structure | None:
    v = GAUSSIAN_STRUCTURES.get(key)
    return Structure(v[0], v[1], clustered, v[2]) if v else None


# ---- nested structure contrasts (STAT-17 / STAT-23; 04 Workflow B) ------------------------------------------------

@dataclass(frozen=True)
class Contrast:
    nested: bool
    df: int = 0
    reference: str = ""            # "chisq" | "mixture" (50:50 chi-square(df - 1) : chi-square(df))
    null: str = ""                 # tested null hypothesis in words
    what: str = ""                 # what the larger model adds
    assumptions: str = ""
    needs_interior: bool = False   # DIAG -> RS: valid only when both variances are interior
    reason: str = ""               # why no likelihood-ratio test (not nested / several changes)

    @property
    def reference_text(self) -> str:
        if not self.nested:
            return "no likelihood-ratio test"
        if self.reference == "mixture":
            lo = "a point mass at 0" if self.df == 1 else f"chi-square({self.df - 1})"
            return f"50:50 mixture of {lo} and chi-square({self.df}) (boundary null)"
        return f"chi-square({self.df})"


REGULAR = ("the same rows and fixed part, REML likelihoods, both fits converged without boundary or optimizer "
           "flags; the regular-case asymptotic reference distribution")


def contrast(small: Structure, large: Structure) -> Contrast:
    """The test that compares two structures fitted to the same rows with the same fixed part, or the reason there
    is none. The reference distribution follows the hypothesis, never the df alone."""
    changes = [k for k in ("random", "residual", "clustered") if getattr(small, k) != getattr(large, k)]
    if not changes:
        return Contrast(False, reason="the same structure: there is nothing to test")
    if len(changes) > 1:
        return Contrast(False, reason="the structures differ in more than one component (" + ", ".join(changes)
                        + "); no validated reference distribution covers simultaneous changes, so they are compared "
                          "by AIC/BIC only")
    ch = changes[0]
    if ch == "random":
        pair = (small.random, large.random)
        if pair == ("ri", "diag"):
            return Contrast(True, 1, "mixture", "participant time-slope variance = 0", "random time-slope variance",
                            REGULAR + "; the variance lies on the boundary of its range under the null")
        if pair == ("ri", "rs"):
            return Contrast(True, 2, "mixture", "participant time-slope variance = 0 and its covariance with the "
                            "intercept = 0", "random time slope (variance + covariance)",
                            REGULAR + "; one variance on the boundary under the null (Self & Liang)")
        if pair == ("diag", "rs"):
            return Contrast(True, 1, "chisq", "intercept-slope covariance = 0", "intercept-slope covariance",
                            REGULAR + "; valid only when both variances are clearly positive (interior)",
                            needs_interior=True)
        return Contrast(False, reason="the smaller random-effects structure is not nested in the larger one")
    if ch == "residual":
        pair = (small.residual, large.residual)
        if pair == ("independent", "ar1"):
            return Contrast(True, 1, "chisq", "AR(1) correlation phi = 0", "AR(1) residual correlation (signed phi)",
                            REGULAR + "; phi = 0 is interior to (-1, 1), so the ordinary chi-square(1) applies")
        if pair == ("independent", "car1"):
            return Contrast(True, 1, "mixture", "continuous-time AR(1) correlation phi = 0",
                            "continuous-time AR(1) residual correlation",
                            REGULAR + "; phi lies in [0, 1), so phi = 0 is a boundary (one-sided) null")
        return Contrast(False, reason="discrete AR(1) over occasions and continuous-time AR(1) are not nested in each "
                                      "other: compare them by AIC/BIC")
    if not small.clustered and large.clustered:
        return Contrast(True, 1, "mixture", "higher-level cluster variance = 0", "higher-level cluster variance",
                        REGULAR + "; same composite participants and rows; the variance is on the boundary under the "
                                  "null. Exploratory: the study design may require the cluster regardless of this test")
    return Contrast(False, reason="removing the cluster is not a nested test in this direction")


# ---- shared R helpers ------------------------------------------------------------------------------------------------

MIXED_R_HELPERS = r"""# ---- StatExplore: shared longitudinal model helpers (v1.3.2) ----
# One builder for every Gaussian longitudinal fit (Mixed workspace, comparisons, Discovery, exports).
# random: "ri" | "diag" | "rs"; residual: "independent" | "ar1" | "car1"; clustered: participants (composite
# cluster+ID key in `id`) nested in `cluster`. Correlation objects are created fresh for every fit.
sx_re_lme4 <- function(random, clustered = FALSE, slope = "sx_time", id = "sx_id", cluster = "sx_cluster") {
  r <- switch(random, ri = paste0("(1 | ", id, ")"), diag = paste0("(1 + ", slope, " || ", id, ")"),
              rs = paste0("(1 + ", slope, " | ", id, ")"), stop("Unsupported random-effects structure: ", random))
  if (clustered) paste0("(1 | ", cluster, ") + ", r) else r
}
sx_re_nlme <- function(random, clustered = FALSE, slope = "sx_time", id = "sx_id", cluster = "sx_cluster") {
  fs <- stats::as.formula(paste("~ 1 +", slope))
  if (!clustered) return(switch(random,
    ri = stats::as.formula(paste("~ 1 |", id)),
    diag = stats::setNames(list(nlme::pdDiag(fs)), id),
    rs = stats::as.formula(paste("~ 1 +", slope, "|", id)),
    stop("Unsupported random-effects structure: ", random)))
  switch(random,
    ri = stats::as.formula(paste("~ 1 |", cluster, "/", id)),
    diag = stats::setNames(list(nlme::pdDiag(~ 1), nlme::pdDiag(fs)), c(cluster, id)),
    rs = stats::setNames(list(nlme::pdDiag(~ 1), nlme::pdSymm(fs)), c(cluster, id)),
    stop("Unsupported random-effects structure: ", random))
}
sx_cor_nlme <- function(residual, clustered = FALSE, occ = "sx_tidx", tnum = "sx_tnum", id = "sx_id",
                        cluster = "sx_cluster") {
  g <- if (clustered) paste0(cluster, "/", id) else id
  switch(residual, independent = NULL,
    ar1 = nlme::corAR1(form = stats::as.formula(paste("~", occ, "|", g))),
    car1 = nlme::corCAR1(form = stats::as.formula(paste("~", tnum, "|", g))),
    stop("Unsupported residual covariance structure: ", residual))
}
# Composite participant key for nested designs: length-prefixed (cluster, ID) labels, ordered by cluster then ID, so
# locally reused IDs never merge and no concatenation is ambiguous.
sx_nested_id <- function(cluster, id) {
  cl <- as.character(cluster); pid <- as.character(id)
  key <- paste0(nchar(cl, type = "chars"), ":", cl, "|", pid)
  o <- order(as.integer(factor(cluster)), as.integer(factor(id)))
  factor(key, levels = unique(key[o]))
}
# Optimizer errors that justify retrying the SAME model with another optimizer; any other error is structural.
sx_optimizer_error <- function(msg) grepl("convergence|iteration limit|evaluation limit|nlminb problem|false convergence|singular convergence|optim problem", msg, ignore.case = TRUE)
# Run fit(opt) for each optimizer in turn until one succeeds; record every attempt, its warnings and errors. Only an
# optimizer failure moves on to the next optimizer; the requested model is never changed.
sx_fit_attempts <- function(fit, optimizers) {
  attempts <- list(); warns <- character(); m <- NULL; err <- NULL
  for (opt in optimizers) {
    w <- character()
    # AUD134-10: every optimizer attempt is a fitting invocation (outside the tryCatch: a budget stop is not a fit error)
    if (exists("sx_budget_take", mode = "function")) sx_budget_take(if (length(attempts)) "retry" else "fit")
    m <- withCallingHandlers(tryCatch(fit(opt), error = function(e) e),
      warning = function(x) { w <<- c(w, conditionMessage(x)); invokeRestart("muffleWarning") },
      message = function(x) invokeRestart("muffleMessage"))
    bad <- inherits(m, "error")
    attempts[[length(attempts) + 1]] <- data.frame(optimizer = opt, status = if (bad) "failed" else "fitted",
      error = if (bad) conditionMessage(m) else "", warnings = paste(unique(w), collapse = "; "), stringsAsFactors = FALSE)
    if (!bad) { warns <- unique(w); break }
    err <- conditionMessage(m)
    if (!sx_optimizer_error(err)) break
  }
  list(fit = if (inherits(m, "error")) NULL else m, attempts = do.call(rbind, attempts), warnings = warns, error = err)
}
# Diagnostics of a fitted Gaussian mixed model (STAT-18 / STAT-32). Flags may coexist; `clean` only when none.
#   lme4: optimizer code, lme4 check code and messages, warnings, isSingular (lme4 tolerance 1e-4 on theta)
#   nlme: every random-effects block G / sigma^2: non positive definite -> singular; smallest eigenvalue < 1e-6, eigenvalue
#         ratio < 1e-6 or |correlation| > 0.999 -> near-boundary (positive definite but at the edge); apVar (approximate
#         covariance of the variance parameters) reported separately; optimizer fallback reported as a warning.
sx_diag <- function(m, run = NULL) {
  out <- list(engine = if (inherits(m, "merMod")) "lme4" else if (inherits(m, "lme")) "nlme" else class(m)[1],
              optimizer = NA_character_, opt_code = NA_real_, check_code = NA_real_, singular = NA,
              near_boundary = NA, boundary = character(), apvar = "not applicable", messages = character(),
              flags = character())
  att <- if (is.null(run)) NULL else run$attempts
  w <- if (is.null(run)) character() else run$warnings
  if (!is.null(att) && nrow(att)) {
    out$optimizer <- utils::tail(att$optimizer, 1)
    if (nrow(att) > 1) out$flags <- c(out$flags, paste0("optimizer fallback: ", paste(sprintf("%s failed (%s)",
      att$optimizer[att$status == "failed"], att$error[att$status == "failed"]), collapse = "; "), "; estimated with ",
      out$optimizer))
  }
  if (inherits(m, "merMod")) {
    oi <- m@optinfo
    out$optimizer <- oi$optimizer
    out$opt_code <- if (is.null(oi$conv$opt)) NA_real_ else as.numeric(oi$conv$opt)
    cc <- oi$conv$lme4$code; out$check_code <- if (is.null(cc) || !length(cc)) 0 else as.numeric(cc[1])
    msgs <- unique(c(unlist(oi$conv$lme4$messages), unlist(oi$warnings)))
    out$singular <- lme4::isSingular(m)
    if (isTRUE(out$singular)) out$flags <- c(out$flags, "Singular fit: a random-effect variance is ~0 or a correlation is +/-1 (lme4 tolerance 1e-4); simplify the random-effects structure")
    msgs <- msgs[!grepl("singular", msgs, ignore.case = TRUE) | !isTRUE(out$singular)]
    out$messages <- msgs
    if (isTRUE(out$opt_code != 0)) out$flags <- c(out$flags, sprintf("optimizer returned code %g", out$opt_code))
    if (isTRUE(out$check_code != 0) && !length(msgs)) out$flags <- c(out$flags, sprintf("lme4 convergence check code %g", out$check_code))
    if (length(msgs)) out$flags <- c(out$flags, paste("convergence:", msgs))
  } else if (inherits(m, "lme")) {
    s2 <- m$sigma^2; nb <- FALSE; sing <- FALSE
    for (g in names(m$modelStruct$reStruct)) {
      G <- tryCatch(nlme::pdMatrix(m$modelStruct$reStruct[[g]]), error = function(e) NULL)
      if (is.null(G)) { out$boundary <- c(out$boundary, paste0(g, ": covariance unavailable")); next }
      ev <- eigen(G, symmetric = TRUE, only.values = TRUE)$values
      r <- if (nrow(G) > 1) stats::cov2cor(G) else matrix(1)
      rmax <- if (nrow(G) > 1) max(abs(r[upper.tri(r)])) else 0
      if (!all(is.finite(ev)) || min(ev) <= 0) { sing <- TRUE
        out$boundary <- c(out$boundary, sprintf("%s: covariance not positive definite (smallest eigenvalue %.3g)", g, min(ev)))
      } else if (min(ev) < 1e-6 || min(ev) / max(ev) < 1e-6 || rmax > 0.999) { nb <- TRUE
        out$boundary <- c(out$boundary, sprintf("%s: near-boundary (smallest variance/sigma^2 eigenvalue %.3g, ratio %.3g, max |r| %.4f)",
                                                g, min(ev), min(ev) / max(ev), rmax)) }
    }
    out$singular <- sing; out$near_boundary <- nb
    if (sing) out$flags <- c(out$flags, paste("singular random-effects covariance:", paste(out$boundary, collapse = "; ")))
    if (nb) out$flags <- c(out$flags, paste("near-boundary random-effects covariance (positive definite; scaled eigenvalue / eigenvalue-ratio tolerance 1e-6, |r| > 0.999):", paste(out$boundary, collapse = "; ")))
    out$apvar <- if (is.matrix(m$apVar)) "positive definite" else as.character(m$apVar)[1]
    if (!is.matrix(m$apVar)) out$flags <- c(out$flags, paste0("approximate covariance of the variance parameters (apVar) unavailable: ", out$apvar,
                                                             " (variance-component standard errors and intervals are not available; this is not by itself singularity)"))
  } else if (inherits(m, "clmm")) {
    out$optimizer <- "ucminf (ordinal default)"
    code <- m$optRes$convergence; out$opt_code <- if (is.null(code)) NA_real_ else as.numeric(code)
    if (!isTRUE(out$opt_code == 0)) out$flags <- c(out$flags, sprintf("optimizer convergence code %s", format(out$opt_code)))
    ch <- tryCatch(summary(m)$condHess, error = function(e) NA_real_)
    if (isTRUE(ch > 1e4)) out$flags <- c(out$flags, sprintf("ill-conditioned Hessian (condition number %.3g)", ch))
    vc <- tryCatch(getExportedValue("ordinal", "VarCorr")(m), error = function(e) NULL); sing <- FALSE
    for (g in names(vc)) { ev <- eigen(as.matrix(vc[[g]]), symmetric = TRUE, only.values = TRUE)$values
      if (min(ev) < 1e-6) { sing <- TRUE; out$boundary <- c(out$boundary, sprintf("%s: smallest variance eigenvalue %.3g", g, min(ev))) } }
    out$singular <- sing
    if (sing) out$flags <- c(out$flags, paste("random-effect variance at or near zero:", paste(out$boundary, collapse = "; ")))
  } else if (inherits(m, "geeglm")) {
    out$optimizer <- "geese (Fisher scoring)"
    if (!identical(as.integer(m$geese$error), 0L)) out$flags <- c(out$flags, "GEE estimating equations did not converge")
  } else if (inherits(m, "mmrm")) {
    out$optimizer <- "mmrm default"
    if (!isTRUE(attr(m, "converged"))) out$flags <- c(out$flags, "mmrm did not report convergence")
  } else if (inherits(m, "mblogit")) {
    out$optimizer <- "PQL"
    if (!isTRUE(m$converged)) out$flags <- c(out$flags, "the PQL algorithm did not converge")
  } else {
    out$flags <- c(out$flags, "diagnostics unknown for this engine")
  }
  w <- unique(w[!vapply(w, function(x) any(grepl(x, out$messages, fixed = TRUE)), logical(1))])
  if (length(w)) out$flags <- c(out$flags, paste("warning:", w))
  out$flags <- unique(out$flags)
  out$clean <- !length(out$flags)
  out$status <- if (isTRUE(out$singular)) "singular" else if (isTRUE(out$near_boundary)) "near_boundary" else
    if (out$clean) "clean" else "warned"
  out
}
# Marginal / conditional R2 by direct variance decomposition (STAT-45): var(X beta) / (var(X beta) + sum over every
# grouping level of mean diag(Z G Z') + sigma^2). Never the package fallback that silently drops random variance.
sx_r2_direct <- function(m, d) {
  na <- function(why) list(m = NA_real_, c = NA_real_, fixed = NA_real_, random = NA_real_, residual = NA_real_, reason = why)
  tryCatch({
    if (inherits(m, "merMod")) {
      s2 <- stats::sigma(m)^2
      fv <- stats::var(as.vector(lme4::getME(m, "X") %*% lme4::fixef(m)))
      rv <- mean(Matrix::rowSums((lme4::getME(m, "Z") %*% lme4::getME(m, "Lambda"))^2)) * s2
    } else if (inherits(m, "lme")) {
      s2 <- m$sigma^2
      fv <- stats::var(as.numeric(stats::predict(m, level = 0)))
      rv <- 0
      for (g in names(m$modelStruct$reStruct)) {
        G <- nlme::pdMatrix(m$modelStruct$reStruct[[g]]) * s2
        Z <- stats::model.matrix(stats::formula(m$modelStruct$reStruct[[g]]), d)
        rv <- rv + mean(rowSums((Z %*% G) * Z))
      }
    } else return(na("not a random-effects model"))
    tot <- fv + rv + s2
    if (!is.finite(tot) || tot <= 0) return(na("variance components unavailable"))
    list(m = fv / tot, c = (fv + rv) / tot, fixed = fv, random = rv, residual = s2, reason = "")
  }, error = function(e) na(paste("variance decomposition failed:", conditionMessage(e))))
}
# Random-effects covariance of an nlme fit, every grouping level (STAT-43): variance, SD, covariance and correlation
# from the scaled pdMatrix; DIAG covariances are constrained to 0, not estimated.
sx_vc_nlme <- function(m) {
  s2 <- m$sigma^2; rows <- list()
  for (g in rev(names(m$modelStruct$reStruct))) {
    pd <- m$modelStruct$reStruct[[g]]; G <- nlme::pdMatrix(pd) * s2; nm <- colnames(G)
    diag_only <- inherits(pd, "pdDiag")
    for (i in seq_len(nrow(G))) rows[[length(rows) + 1]] <- data.frame(group = g, term = nm[i], with = "",
      variance_or_cov = G[i, i], sd_or_corr = sqrt(G[i, i]), estimated = "estimated", stringsAsFactors = FALSE)
    if (nrow(G) > 1) for (i in 1:(nrow(G) - 1)) for (j in (i + 1):nrow(G)) rows[[length(rows) + 1]] <- data.frame(
      group = g, term = nm[i], with = nm[j], variance_or_cov = G[i, j],
      sd_or_corr = if (diag_only) 0 else G[i, j] / sqrt(G[i, i] * G[j, j]),
      estimated = if (diag_only) "constrained to 0 (uncorrelated structure)" else "estimated", stringsAsFactors = FALSE)
  }
  rows[[length(rows) + 1]] <- data.frame(group = "Residual", term = "(residual)", with = "", variance_or_cov = s2,
    sd_or_corr = sqrt(s2), estimated = "estimated", stringsAsFactors = FALSE)
  do.call(rbind, rows)
}
# Grouping counts at every level of the fitted object (UI-57 / STAT-46): groups, smallest and largest group size.
sx_group_counts <- function(m) {
  fl <- if (inherits(m, "merMod")) lme4::getME(m, "flist") else if (inherits(m, "lme")) as.list(m$groups) else list()
  if (!length(fl)) return(NULL)
  do.call(rbind, lapply(names(fl), function(g) { k <- table(droplevels(as.factor(fl[[g]])))
    data.frame(level = g, groups = length(k), min_size = min(k), max_size = max(k)) }))
}
# What was ACTUALLY fitted (v1.3.3, J4), read back from the fitted object, never from the request: class, fixed-effect
# term labels, grouping factors, random-effect blocks (columns, correlated or diagonal), residual correlation class,
# estimation method, GEE working correlation, number of observations.
sx_effective <- function(m) {
  # optional helpers looked up only when needed (a Gaussian script does not depend on reformulas / ordinal)
  nobars <- function(x) getExportedValue(if (requireNamespace("reformulas", quietly = TRUE)) "reformulas" else "lme4",
                                         "nobars")(x)
  f <- tryCatch(if (inherits(m, "merMod")) stats::formula(m, fixed.only = TRUE)
                else if (inherits(m, "clmm")) nobars(m$formula) else stats::formula(m), error = function(e) NULL)
  blocks <- list(); groups <- character()
  if (inherits(m, "merMod") || inherits(m, "clmm")) {
    vc <- tryCatch(if (inherits(m, "clmm")) getExportedValue("ordinal", "VarCorr")(m) else lme4::VarCorr(m),
                   error = function(e) list())
    groups <- names(vc)
    blocks <- lapply(names(vc), function(g) { v <- as.matrix(vc[[g]])
      list(group = g, columns = I(colnames(v)), structure = if (ncol(v) > 1) "correlated" else "single") })
  } else if (inherits(m, "lme")) {
    groups <- names(m$groups)
    blocks <- lapply(names(m$modelStruct$reStruct), function(g) { p <- m$modelStruct$reStruct[[g]]
      list(group = g, columns = I(colnames(as.matrix(p))),
           structure = if (inherits(p, "pdDiag")) "diagonal" else if (ncol(as.matrix(p)) > 1) "correlated" else "single") })
  }
  list(class = class(m)[1], fixed = I(if (is.null(f)) character() else attr(stats::terms(f), "term.labels")),
       groups = I(groups), blocks = blocks,
       correlation = if (inherits(m, "lme") && !is.null(m$modelStruct$corStruct)) class(m$modelStruct$corStruct)[1] else "none",
       # nlme turns corAR1 into the equivalent corARMA(p = 1, q = 0) when the occasion index has gaps: keep the order
       cor_pq = I(if (inherits(m, "lme") && inherits(m$modelStruct$corStruct, "corARMA"))
                    c(attr(m$modelStruct$corStruct, "p"), attr(m$modelStruct$corStruct, "q")) else integer()),
       method = if (inherits(m, "merMod") && !inherits(m, "glmerMod")) (if (lme4::isREML(m)) "REML" else "ML")
                else if (inherits(m, "lme")) m$method else "",
       corstr = if (inherits(m, "geeglm")) m$corstr else "",
       nobs = tryCatch(stats::nobs(m), error = function(e) NA_real_))
}
# Likelihood-ratio test with the reference distribution of the hypothesis (STAT-17 / 23 / 41): |LR| <= tol below 0 is
# numerical noise (LR = 0, raw value kept); a materially negative, non-finite LR or df <= 0 gives no p-value.
# reference "chisq" = chi-square(df); "mixture" = 50:50 chi-square(df - 1) : chi-square(df) (chi-square(0) = point mass 0).
sx_lr_test <- function(ll0, ll1, df, reference = "chisq", tol = 1e-6) {
  raw <- 2 * (ll1 - ll0)
  out <- list(LR_raw = raw, LR = NA_real_, df = df, p = NA_real_, log_p = NA_real_, reference = reference,
              status = "tested", reason = "")
  if (!is.finite(raw)) { out$status <- "withheld"; out$reason <- "non-finite likelihood-ratio statistic"; return(out) }
  if (!is.finite(df) || df <= 0) { out$status <- "withheld"; out$reason <- "no extra parameters (df <= 0)"; return(out) }
  if (raw < -tol) { out$status <- "withheld"; out$reason <- sprintf(paste0("the larger model has a lower likelihood ",
    "(LR = %.6g): at least one optimization is unreliable; no p-value"), raw); return(out) }
  lr <- max(raw, 0); out$LR <- lr
  if (raw < 0) out$reason <- sprintf("negligible negative LR %.3g set to 0 (tolerance %g)", raw, tol)
  lp <- function(k) if (k == 0) (if (lr > 0) -Inf else 0) else stats::pchisq(lr, k, lower.tail = FALSE, log.p = TRUE)
  out$log_p <- if (reference == "mixture") { a <- log(0.5) + lp(df - 1); b <- log(0.5) + lp(df)
    mx <- max(a, b); if (!is.finite(mx)) mx else mx + log(exp(a - mx) + exp(b - mx)) } else lp(df)
  out$p <- exp(out$log_p)
  out
}
# Upper-tail F probability from full-precision statistics (STAT-42); log10 kept when the probability underflows.
sx_pf <- function(F, df1, df2) {
  p <- stats::pf(F, df1, df2, lower.tail = FALSE)
  lp <- stats::pf(F, df1, df2, lower.tail = FALSE, log.p = TRUE) / log(10)
  data.frame(p = p, log10_p = lp)
}
"""

# ---- model identity v2 (STAT-22 / H4) -----------------------------------------------------------------------------

IDENTITY_SCHEMA = 2
# Identity fingerprints (schema 2): rbridge.codegen.IDENTITY_R_HELPERS, part of every load script.

