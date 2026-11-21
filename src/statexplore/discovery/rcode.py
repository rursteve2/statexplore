"""R code for the discovery engine. `DISCOVERY_HELPERS` + `setup_script()` run once per
worker; each batch calls `sx_discover_one()` per candidate. `candidate_script()` gives
the exact code for a single candidate (View Code)."""
from __future__ import annotations

from ..core.dataset import Dataset
from ..rbridge.codegen import alias, r_num, r_str, r_vec
from .space import Candidate, SearchSettings, model_terms

DISCOVERY_HELPERS = r"""# ---- StatExplore: discovery helpers ----
# Fit one candidate: full model (focal hierarchy + covariates) vs reduced model (focal term
# removed). Returns the focal test plus screening results.
sx_discover_one <- function(full, reduced, focal, vars, focal_vars) {
  res <- list(status = "ok", reason = "", flags = character(), n = NA_real_, n_params = NA_real_,
              statistic = NA_real_, df1 = NA_real_, df2 = NA_real_, p = NA_real_, effect = NA_real_,
              partial_eta2 = NA_real_, estimate = NA_real_, se = NA_real_, ci_low = NA_real_,
              ci_high = NA_real_, r2 = NA_real_, aic = NA_real_, bic = NA_real_)
  invalid <- function(why) { res$status <- "invalid"; res$reason <- why; res$flags <- ""; res }
  d <- sx_complete(sx_dd, c("sx_y", vars, if (sx_disc_family == "lmm") "sx_gid"))
  n <- nrow(d); res$n <- n
  for (v in vars) if (is.factor(d[[v]]) && nlevels(d[[v]]) < 2)
    return(invalid(paste0(sx_names[[v]], " has fewer than 2 levels in the analysed rows")))
  if (sx_disc_family == "binomial" && length(unique(d$sx_y)) < 2) return(invalid("outcome has one value"))
  f1 <- stats::reformulate(full, response = "sx_y")
  f0 <- stats::reformulate(reduced, response = "sx_y")
  X <- stats::model.matrix(f1, d)
  k <- ncol(X); res$n_params <- k
  if (k > sx_disc_max_params) return(invalid(sprintf("%d parameters exceed the limit of %d", k, sx_disc_max_params)))
  if (n - k < 1) return(invalid("no residual degrees of freedom"))
  fv <- focal_vars[vapply(d[focal_vars], is.factor, logical(1))]
  if (length(fv)) {
    cells <- table(d[fv])
    if (length(fv) >= 2 && any(cells == 0)) return(invalid("empty cell among the focal factors"))
    if (min(cells) < 5) res$flags <- c(res$flags, sprintf("sparse focal cells (min n = %d)", min(cells)))
  }
  # (checked after empty cells, which are the more specific explanation)
  if (qr(X)$rank < k) return(invalid("design matrix is rank-deficient (not identifiable)"))
  cv <- vars[vapply(d[vars], is.numeric, logical(1))]
  if (length(cv) >= 2) {
    r <- abs(stats::cor(d[cv])); diag(r) <- 0
    if (max(r) > 0.9) res$flags <- c(res$flags, sprintf("|r| = %.2f between continuous predictors", max(r)))
  }
  if (sx_disc_family == "lmm") {
    # random intercept per participant; ML fits so the likelihood-ratio test of the focal term is valid
    g1 <- stats::update(f1, . ~ . + (1 | sx_gid)); g0 <- stats::update(f0, . ~ . + (1 | sx_gid))
    m1 <- lme4::lmer(g1, data = d, REML = FALSE); m0 <- lme4::lmer(g0, data = d, REML = FALSE)
    if (lme4::isSingular(m1)) res$flags <- c(res$flags, "singular fit (participant variance ~ 0)")
    conv <- unlist(m1@optinfo$conv$lme4$messages)
    if (length(conv)) res$flags <- c(res$flags, paste("convergence:", conv[1]))
    lr <- 2 * (as.numeric(stats::logLik(m1)) - as.numeric(stats::logLik(m0)))
    res$df1 <- attr(stats::logLik(m1), "df") - attr(stats::logLik(m0), "df")
    res$statistic <- lr; res$p <- stats::pchisq(lr, res$df1, lower.tail = FALSE)
    r2m <- function(m) { v <- stats::var(as.vector(lme4::getME(m, "X") %*% lme4::fixef(m)))
      vc <- as.data.frame(lme4::VarCorr(m)); v / (v + sum(vc$vcov)) }
    res$effect <- r2m(m1) - r2m(m0)                              # delta marginal R-squared
    res$r2 <- r2m(m1)
    res$df2 <- nlevels(droplevels(d$sx_gid))                      # participants
    crit <- stats::qnorm(1 - (1 - sx_disc_conf) / 2)
    co_tab <- function(m) { b <- lme4::fixef(m); se <- sqrt(diag(as.matrix(stats::vcov(m))))
      cbind(Estimate = b, SE = se) }
  } else if (sx_disc_family == "gaussian") {
    m1 <- stats::lm(f1, d); m0 <- stats::lm(f0, d)
    a <- stats::anova(m0, m1)
    res$statistic <- a$F[2]; res$df1 <- a$Df[2]; res$df2 <- a$Res.Df[2]; res$p <- a[["Pr(>F)"]][2]
    ss <- a[["Sum of Sq"]][2]
    res$effect <- ss / sum((d$sx_y - mean(d$sx_y))^2)          # delta R-squared
    res$partial_eta2 <- ss / (ss + a$RSS[2])
    res$r2 <- summary(m1)$r.squared
    crit <- stats::qt(1 - (1 - sx_disc_conf) / 2, m1$df.residual)
  } else {
    m1 <- stats::glm(f1, data = d, family = stats::binomial())
    m0 <- stats::glm(f0, data = d, family = stats::binomial())
    if (!m1$converged || !m0$converged) res$flags <- c(res$flags, "did not converge")
    sep <- stats::glm(f1, data = d, family = stats::binomial(), method = detectseparation::detect_separation)
    if (isTRUE(sep$outcome)) res$flags <- c(res$flags, "separation (infinite estimates)")
    lr <- m0$deviance - m1$deviance
    res$statistic <- lr; res$df1 <- m0$df.residual - m1$df.residual
    res$p <- stats::pchisq(lr, res$df1, lower.tail = FALSE)
    res$effect <- lr / m1$null.deviance                         # delta McFadden R-squared
    ph <- stats::fitted(m1); res$r2 <- mean(ph[d$sx_y == 1]) - mean(ph[d$sx_y == 0])  # Tjur
    epp <- min(sum(d$sx_y), n - sum(d$sx_y)) / max(1, k - 1)
    if (epp < 10) res$flags <- c(res$flags, sprintf("%.1f events per parameter", epp))
    crit <- stats::qnorm(1 - (1 - sx_disc_conf) / 2)
  }
  res$aic <- stats::AIC(m1); res$bic <- stats::BIC(m1)
  if (isTRUE(res$df1 == 1)) {
    col <- colnames(X)[attr(X, "assign") == match(focal, attr(stats::terms(f1), "term.labels"))]
    ctab <- if (sx_disc_family == "lmm") co_tab(m1) else stats::coef(summary(m1))
    if (length(col) == 1 && col %in% rownames(ctab)) {
      co <- ctab[col, ]
      res$estimate <- co[[1]]; res$se <- co[[2]]
      res$ci_low <- co[[1]] - crit * co[[2]]; res$ci_high <- co[[1]] + crit * co[[2]]
    }
  }
  if (length(res$flags)) res$status <- "warning"
  res$flags <- paste(unique(res$flags), collapse = "; ")
  res
}
# Run a batch; per-candidate errors and warnings are recorded, never fatal.
sx_discover_batch <- function(batch) lapply(batch, function(s) {
  warns <- character()
  r <- tryCatch(withCallingHandlers(
      do.call(sx_discover_one, s[c("full", "reduced", "focal", "vars", "focal_vars")]),
      warning = function(w) { warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") }),
    error = function(e) list(status = "failed", reason = conditionMessage(e), flags = ""))
  if (length(warns) && !identical(r$status, "failed") && !identical(r$status, "invalid")) {
    r$flags <- paste(c(if (nzchar(r$flags)) r$flags, unique(warns)), collapse = "; ")
    r$status <- "warning"
  }
  r$i <- s$i
  r
})
"""


def setup_script(ds: Dataset, s: SearchSettings, family: str) -> str:
    a = lambda n: alias(ds.var(n).alias)  # noqa: E731
    vars_ = [a(n) for n in [s.outcome] + s.candidates + s.required + ([s.participant] if s.participant else [])]
    y = a(s.outcome)
    lines = [
        DISCOVERY_HELPERS,
        "# ---- StatExplore: discovery setup ----",
        f"sx_disc_vars <- {r_vec(vars_)}",
        f"sx_disc_family <- {r_str(family)}",
        f"sx_disc_max_params <- {int(s.max_params)}",
        f"sx_disc_conf <- {r_num(s.conf)}",
        ("sx_dd <- sx_complete(dat, sx_disc_vars)  # shared complete-case sample"
         if s.sample == "shared" else
         "sx_dd <- droplevels(dat[, sx_disc_vars, drop = FALSE])  # complete cases are taken per model"),
    ]
    if s.participant:
        lines.append(f"sx_dd$sx_gid <- factor(sx_dd${a(s.participant)})  # participant (random intercept)")
    if family in ("gaussian", "lmm"):
        lines.append(f"sx_dd$sx_y <- sx_dd${y}")
    else:
        lines += [
            f"sx_lv <- if (is.factor(sx_dd${y})) levels(sx_dd${y}) else sort(unique(stats::na.omit(sx_dd${y})))",
            f"sx_dd$sx_y <- as.integer(sx_dd${y} == sx_lv[2])  # event = second (non-reference) level",
        ]
    lines.append("sx_result <- list(n = nrow(sx_dd), n_complete = sum(stats::complete.cases(sx_dd)))")
    return "\n".join(lines) + "\n"


def _term(ds: Dataset, t: tuple[str, ...]) -> str:
    return ":".join(alias(ds.var(v).alias) for v in t)


def candidate_args(ds: Dataset, c: Candidate) -> str:
    full, reduced = model_terms(c)
    return (f"full = {r_vec([_term(ds, t) for t in full])}, "
            f"reduced = {r_vec([_term(ds, t) for t in reduced])}, "
            f"focal = {r_str(_term(ds, c.focal) if c.kind == 'focal' else '')}, "
            f"vars = {r_vec([alias(ds.var(v).alias) for v in c.variables])}, "
            f"focal_vars = {r_vec([alias(ds.var(v).alias) for v in c.focal])}")


def batch_script(ds: Dataset, batch: list[Candidate]) -> str:
    items = ",\n  ".join(f"list(i = {c.index}L, {candidate_args(ds, c)})" for c in batch)
    return f"sx_result <- sx_discover_batch(list(\n  {items}))\n"


def candidate_script(load_script: str, ds: Dataset, s: SearchSettings, family: str, c: Candidate) -> str:
    setup = setup_script(ds, s, family).replace("sx_result <- list(n = nrow(sx_dd), n_complete = "
                                               "sum(stats::complete.cases(sx_dd)))\n", "")
    return (load_script + "\n" + setup + "\n# ---- candidate " + str(c.index) + " ----\n"
            + f"sx_result <- sx_discover_one({candidate_args(ds, c)})\nstr(sx_result)\n")
