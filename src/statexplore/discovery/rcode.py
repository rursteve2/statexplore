"""R code for the discovery engine. `DISCOVERY_HELPERS` + `setup_script()` run once per
worker; each batch calls `sx_discover_one()` per candidate. `candidate_script()` gives
the exact code for a single candidate (View Code)."""
from __future__ import annotations

from ..core.dataset import Dataset
from ..rbridge.codegen import alias, r_num, r_str, r_vec
from .space import Candidate, SearchSettings

DISCOVERY_HELPERS = r"""# ---- StatExplore: discovery helpers ----
# Fit one candidate's full model once, then evaluate each planned hypothesis as a nested
# comparison with a reduced model (tests[[1]] is the candidate's declared focal test).
sx_disc_cache <- new.env()            # reduced-model statistics, reused on the shared sample
sx_disc_cache_rows <- NULL            # the exact rows the cached statistics were fitted to
sx_cache_for <- function(d) {
  # callers (stability, procedure CV) may point sx_dd at other rows: never reuse fits across samples
  if (!identical(rownames(d), sx_disc_cache_rows)) {
    rm(list = ls(sx_disc_cache), envir = sx_disc_cache)
    sx_disc_cache_rows <<- rownames(d)
  }
}
sx_add <- function(f, extra) stats::update(f, stats::as.formula(paste(". ~ . +", extra)))
sx_fit_lmm <- function(f, d, st, reml) {
  if (st %in% c("ri", "rs", "rs_uncorr")) {
    re <- switch(st, ri = "(1 | sx_gid)", rs = "(1 + sx_time | sx_gid)", rs_uncorr = "(1 + sx_time || sx_gid)")
    return(lme4::lmer(sx_add(f, re), data = d, REML = reml))
  }
  if (startsWith(st, "mmrm")) {
    cs <- switch(st, mmrm_us = "us", mmrm_cs = "cs", mmrm_ar1 = "ar1")
    return(mmrm::mmrm(sx_add(f, paste0(cs, "(sx_timef | sx_gid)")), data = d, reml = reml))
  }
  random <- if (st == "nlme_rs") ~ 1 + sx_time | sx_gid else ~ 1 | sx_gid
  cor <- switch(st, ar1 = nlme::corAR1(form = ~ sx_occ | sx_gid), car1 = nlme::corCAR1(form = ~ sx_time | sx_gid),
                NULL)
  fit <- function(opt) nlme::lme(f, random = random, correlation = cor, data = d,
                                 method = if (reml) "REML" else "ML", control = nlme::lmeControl(opt = opt))
  m <- tryCatch(fit("nlminb"), error = function(e) NULL)
  if (is.null(m)) { m <- fit("optim"); attr(m, "sx_note") <- "nlminb failed; refitted with optim" }
  m
}
sx_fit_model <- function(f, d) {
  st <- sx_disc_struct
  switch(sx_disc_family,
    lmm = sx_fit_lmm(f, d, st, sx_disc_reml),
    glmm = lme4::glmer(sx_add(f, if (st == "glmer_rs") "(1 + sx_time | sx_gid)" else "(1 | sx_gid)"), data = d,
                       family = stats::binomial()),
    clmm = ordinal::clmm(sx_add(f, if (st == "clmm_rs") "(1 + sx_time | sx_gid)" else "(1 | sx_gid)"), data = d,
                         link = "logit"),
    gee = geepack::geeglm(f, data = d, id = sx_gid, waves = sx_occ, corstr = sx_disc_corstr,
                          family = if (sx_disc_gee_family == "binomial") stats::binomial() else stats::gaussian()),
    gaussian = stats::lm(f, d),
    ordinal = ordinal::clm(f, data = d, link = "logit"),
    multinomial = nnet::multinom(f, data = d, trace = FALSE, maxit = 1000, model = FALSE),
    stats::glm(f, data = d, family = stats::binomial()))
}
sx_fit_stats <- function(m) {
  if (sx_disc_family == "gee") return(list(ll = NA_real_, k = length(stats::coef(m)), conv = ""))
  ll <- stats::logLik(m)
  out <- list(ll = as.numeric(ll), k = attr(ll, "df"), conv = "")
  if (sx_disc_family == "gaussian") { out$rss <- stats::deviance(m); out$dfres <- m$df.residual }
  if (sx_disc_family == "binomial") { out$dev <- m$deviance; out$dfres <- m$df.residual
    if (!m$converged) out$conv <- "did not converge" }
  if (sx_disc_family == "multinomial") { out$dev <- m$deviance; out$k <- m$edf
    if (m$convergence != 0) out$conv <- "did not converge (iteration limit)" }
  if (sx_disc_family == "ordinal" && !identical(as.integer(m$convergence$code), 0L))
    out$conv <- paste("did not converge:", paste(m$convergence$messages, collapse = " "))
  if (sx_disc_family %in% c("lmm", "glmm") && inherits(m, "merMod")) {
    conv <- unlist(m@optinfo$conv$lme4$messages)
    if (length(conv)) out$conv <- paste("convergence:", conv[1])
  }
  out$r2m <- NA_real_
  if (sx_disc_family == "lmm" && sx_disc_struct == "ri") {
    v <- stats::var(as.vector(lme4::getME(m, "X") %*% lme4::fixef(m)))
    vc <- as.data.frame(lme4::VarCorr(m)); out$r2m <- v / (v + sum(vc$vcov))
  }
  out
}
sx_reduced_stats <- function(reduced, d) {
  key <- paste0("m:", paste(sort(reduced), collapse = "+"))      # "m:" = intercept-only model
  if (sx_disc_shared && !is.null(sx_disc_cache[[key]])) return(sx_disc_cache[[key]])
  f0 <- stats::reformulate(if (length(reduced)) reduced else "1", response = "sx_y")
  st <- sx_fit_stats(sx_fit_model(f0, d))
  if (sx_disc_shared) assign(key, st, envir = sx_disc_cache)
  st
}
sx_null_scale <- function(d) {
  # denominators of the pseudo-R2 effects: -2 log L (ordinal) or deviance (logistic, multinomial) of y ~ 1
  if (sx_disc_shared && !is.null(sx_disc_cache[["<null>"]])) return(sx_disc_cache[["<null>"]])
  st <- sx_reduced_stats(character(), d)
  v <- if (sx_disc_family == "ordinal") -2 * st$ll else st$dev
  if (sx_disc_shared) assign("<null>", v, envir = sx_disc_cache)
  v
}
# columns of the design matrix X that belong to model term `term` ("a:b" matches "b:a")
sx_term_cols <- function(X, f, term) {
  labs <- attr(stats::terms(f), "term.labels")
  norm <- function(x) vapply(strsplit(x, ":", fixed = TRUE), function(p) paste(sort(p), collapse = ":"), "")
  j <- match(norm(term), norm(labs))
  if (is.na(j)) character() else colnames(X)[attr(X, "assign") == j]
}
sx_wald <- function(m, X, f, drop) {
  # robust (sandwich) Wald test of all coefficients of the dropped terms (GEE)
  cols <- unlist(lapply(drop, function(t) sx_term_cols(X, f, t)))
  b <- stats::coef(m)[cols]; V <- as.matrix(stats::vcov(m))[cols, cols, drop = FALSE]
  w <- as.numeric(t(b) %*% solve(V, b))
  list(statistic = w, df1 = length(cols), df_den = NA_real_, p = stats::pchisq(w, length(cols), lower.tail = FALSE),
       effect = NA_real_, partial_eta2 = NA_real_)
}
sx_compare <- function(s1, s0, d) {
  r <- list(statistic = NA_real_, df1 = NA_real_, df_den = NA_real_, p = NA_real_, effect = NA_real_,
            partial_eta2 = NA_real_)
  if (sx_disc_family == "gaussian") {
    ss <- s0$rss - s1$rss; r$df1 <- s0$dfres - s1$dfres; r$df_den <- s1$dfres
    r$statistic <- (ss / r$df1) / (s1$rss / s1$dfres)
    r$p <- stats::pf(r$statistic, r$df1, r$df_den, lower.tail = FALSE)
    r$effect <- ss / sum((d$sx_y - mean(d$sx_y))^2)                # delta R-squared
    r$partial_eta2 <- ss / (ss + s1$rss)
    return(r)
  }
  if (sx_disc_family %in% c("binomial", "multinomial")) {
    lr <- s0$dev - s1$dev
    r$df1 <- if (sx_disc_family == "binomial") s0$dfres - s1$dfres else s1$k - s0$k
  } else {
    lr <- 2 * (s1$ll - s0$ll); r$df1 <- s1$k - s0$k
  }
  r$statistic <- lr
  r$p <- stats::pchisq(max(lr, 0), r$df1, lower.tail = FALSE)
  r$effect <- switch(sx_disc_family, lmm = s1$r2m - s0$r2m,             # delta marginal R-squared (lme4 RI)
                     glmm = NA_real_, clmm = NA_real_,
                     lr / sx_null_scale(d))                             # delta McFadden pseudo-R-squared
  r
}
sx_long <- function() sx_disc_family %in% c("lmm", "glmm", "clmm", "gee")
sx_discover_one <- function(full, tests, vars, focal_vars, struct = NULL) {
  if (!is.null(struct)) {      # workflow B: this candidate's structure
    old_struct <- sx_disc_struct; sx_disc_struct <<- struct; on.exit(sx_disc_struct <<- old_struct)
  }
  res <- list(status = "ok", reason = "", flags = character(), n = NA_real_, n_params = NA_real_,
              n_groups = NA_real_, statistic = NA_real_, df1 = NA_real_, df2 = NA_real_, p = NA_real_,
              effect = NA_real_, partial_eta2 = NA_real_, estimate = NA_real_, se = NA_real_,
              ci_low = NA_real_, ci_high = NA_real_, r2 = NA_real_, adj_r2 = NA_real_, aic = NA_real_,
              bic = NA_real_, loglik = NA_real_, tests = list())
  invalid <- function(why) { res$status <- "invalid"; res$reason <- why; res$flags <- ""; res$tests <- NULL; res }
  d <- sx_complete(sx_dd, c("sx_y", vars, if (sx_long()) c("sx_gid", if (sx_disc_has_time) c("sx_time", "sx_timef", "sx_occ"))))
  n <- nrow(d); res$n <- n
  if (n < sx_disc_min_obs) return(invalid(sprintf("%d complete rows; the minimum is %d", n, sx_disc_min_obs)))
  for (v in vars) if (is.factor(d[[v]]) && nlevels(d[[v]]) < 2)
    return(invalid(paste0(sx_names[[v]], " has fewer than 2 levels in the analysed rows")))
  if (sx_disc_family == "binomial" && length(unique(d$sx_y)) < 2) return(invalid("outcome has one value"))
  K <- if (is.factor(d$sx_y)) nlevels(d$sx_y) else NA_integer_
  if (sx_disc_family %in% c("glmm") || (sx_disc_family == "gee" && sx_disc_gee_family == "binomial"))
    if (length(unique(d$sx_y)) < 2) return(invalid("outcome has one value"))
  if (sx_disc_family %in% c("ordinal", "multinomial", "clmm") && K < 3)
    return(invalid(sprintf("only %d outcome categories observed in the analysed rows", K)))
  if (sx_disc_family == "multinomial" && !(sx_disc_ref %in% levels(d$sx_y)))
    return(invalid("the reference outcome category is not observed in the analysed rows"))
  f1 <- stats::reformulate(if (length(full)) full else "1", response = "sx_y")
  X <- stats::model.matrix(f1, d)
  # parameters: lm/glm = columns of X; ordinal = slopes + (K - 1) thresholds; multinomial = (K - 1) x columns
  k <- switch(sx_disc_family, ordinal = , clmm = ncol(X) - 1 + (K - 1), multinomial = (K - 1) * ncol(X), ncol(X))
  res$n_params <- k
  if (k > sx_disc_max_params) return(invalid(sprintf("%d parameters exceed the limit of %d", k, sx_disc_max_params)))
  if (n - k < 1) return(invalid("no residual degrees of freedom"))
  # focal_vars: one set of variables (focal interaction) or a list of sets (each interaction term)
  for (fs in if (is.list(focal_vars)) focal_vars else list(focal_vars)) {
    fv <- fs[vapply(d[fs], is.factor, logical(1))]
    if (length(fv)) {
      cells <- table(d[fv])
      if (length(fv) >= 2 && any(cells == 0)) return(invalid("empty cell among the focal factors"))
      if (min(cells) < 5) res$flags <- c(res$flags, sprintf("sparse focal cells (min n = %d)", min(cells)))
    }
  }
  # (checked after empty cells, which are the more specific explanation)
  if (qr(X)$rank < ncol(X)) return(invalid("design matrix is rank-deficient (not identifiable)"))
  cv <- vars[vapply(d[vars], is.numeric, logical(1))]
  if (length(cv) >= 2) {
    r <- abs(stats::cor(d[cv])); diag(r) <- 0
    if (max(r) > 0.9) res$flags <- c(res$flags, sprintf("|r| = %.2f between continuous predictors", max(r)))
  }
  if (sx_long() && sx_disc_struct %in% c("rs", "rs_uncorr", "nlme_rs", "glmer_rs", "clmm_rs")) {
    k2 <- sum(tapply(d$sx_time, droplevels(d$sx_gid), function(x) length(unique(x))) >= 2)
    if (k2 < 10) return(invalid(sprintf("only %d participants have 2+ time points: a random slope is not estimable", k2)))
  }
  if (sx_disc_shared) sx_cache_for(d)
  m1 <- sx_fit_model(f1, d)
  s1 <- sx_fit_stats(m1)
  if (nzchar(s1$conv)) res$flags <- c(res$flags, s1$conv)
  if (sx_long()) {
    res$n_groups <- nlevels(droplevels(d$sx_gid))
    if (inherits(m1, "merMod") && lme4::isSingular(m1))
      res$flags <- c(res$flags, "singular fit (a random-effect variance is ~ 0 or a correlation is +/-1)")
    if (!is.null(attr(m1, "sx_note"))) res$flags <- c(res$flags, attr(m1, "sx_note"))
    if (sx_disc_family == "gee" && !identical(as.integer(m1$geese$error), 0L))
      res$flags <- c(res$flags, "GEE did not converge")
    if (sx_disc_family == "clmm" && !is.null(m1$condHess) && is.finite(m1$condHess) && m1$condHess > 1e4)
      res$flags <- c(res$flags, sprintf("ill-conditioned Hessian (condition number %.0f)", m1$condHess))
    if (sx_disc_family %in% c("glmm") || (sx_disc_family == "gee" && sx_disc_gee_family == "binomial")) {
      epp <- min(sum(d$sx_y), n - sum(d$sx_y)) / max(1, k - 1)
      if (epp < 10) res$flags <- c(res$flags, sprintf("%.1f events per parameter", epp))
    }
    res$r2 <- s1$r2m
    ctab <- switch(sx_disc_family,
      lmm = if (inherits(m1, "merMod")) cbind(lme4::fixef(m1), sqrt(diag(as.matrix(stats::vcov(m1))))) else
        if (inherits(m1, "lme")) summary(m1)$tTable[, 1:2, drop = FALSE] else
        cbind(stats::coef(m1), sqrt(diag(as.matrix(stats::vcov(m1))))),
      glmm = stats::coef(summary(m1))[, 1:2, drop = FALSE],
      clmm = stats::coef(summary(m1))[, 1:2, drop = FALSE],
      gee = stats::coef(summary(m1))[, 1:2, drop = FALSE])
    crit <- stats::qnorm(1 - (1 - sx_disc_conf) / 2)
  } else if (sx_disc_family == "gaussian") {
    sm <- summary(m1); res$r2 <- sm$r.squared; res$adj_r2 <- sm$adj.r.squared
    ctab <- stats::coef(sm)
    crit <- stats::qt(1 - (1 - sx_disc_conf) / 2, m1$df.residual)
  } else if (sx_disc_family == "binomial") {
    sep <- stats::glm(f1, data = d, family = stats::binomial(), method = detectseparation::detect_separation)
    if (isTRUE(sep$outcome)) res$flags <- c(res$flags, "separation (infinite estimates)")
    ph <- stats::fitted(m1); res$r2 <- mean(ph[d$sx_y == 1]) - mean(ph[d$sx_y == 0])  # Tjur
    epp <- min(sum(d$sx_y), n - sum(d$sx_y)) / max(1, k - 1)
    if (epp < 10) res$flags <- c(res$flags, sprintf("%.1f events per parameter", epp))
    ctab <- stats::coef(summary(m1))
    crit <- stats::qnorm(1 - (1 - sx_disc_conf) / 2)
  } else {
    res$r2 <- if (sx_disc_family == "ordinal") 1 + s1$ll / (sx_null_scale(d) / 2) else 1 - s1$dev / sx_null_scale(d)
    small <- min(table(d$sx_y))
    if (small < 10) res$flags <- c(res$flags, sprintf("smallest outcome category has %d rows", small))
    ctab <- if (sx_disc_family == "ordinal") stats::coef(summary(m1)) else NULL
    crit <- stats::qnorm(1 - (1 - sx_disc_conf) / 2)
  }
  if (sx_disc_family != "gee") { res$aic <- stats::AIC(m1); res$bic <- stats::BIC(m1); res$loglik <- s1$ll }
  for (t in tests) {
    if (sx_disc_family == "gee") {
      tr <- sx_wald(m1, X, f1, t$drop)
    } else {
      s0 <- sx_reduced_stats(t$reduced, d)
      if (nzchar(s0$conv) && sx_disc_family %in% c("ordinal", "multinomial", "binomial", "glmm", "clmm"))
        res$flags <- c(res$flags, paste("reduced model", s0$conv))
      tr <- sx_compare(s1, s0, d)
    }
    if (sx_disc_family == "multinomial" && tr$statistic < -1e-6)
      res$flags <- c(res$flags, "optimizer: the reduced model fitted better than the full model")
    tr$id <- t$id; tr$estimate <- NA_real_; tr$se <- NA_real_; tr$ci_low <- NA_real_; tr$ci_high <- NA_real_
    if (length(t$drop) == 1 && isTRUE(tr$df1 == 1) && !is.null(ctab)) {
      col <- sx_term_cols(X, f1, t$drop)
      if (length(col) == 1 && col %in% rownames(ctab)) {
        co <- ctab[col, ]
        tr$estimate <- co[[1]]; tr$se <- co[[2]]
        tr$ci_low <- co[[1]] - crit * co[[2]]; tr$ci_high <- co[[1]] + crit * co[[2]]
      }
    }
    res$tests[[length(res$tests) + 1]] <- tr
  }
  if (sx_disc_family == "gee") {
    res$qic <- tryCatch(as.numeric(geepack::QIC(m1)[["QIC"]]), error = function(e) NA_real_)
    res$aic <- NA_real_; res$bic <- NA_real_; res$loglik <- NA_real_
  }
  # the declared focal test, also at the top level (v1.1 row layout)
  if (length(res$tests)) {
    t1 <- res$tests[[1]]
    for (nm in c("statistic", "df1", "p", "effect", "partial_eta2", "estimate", "se", "ci_low", "ci_high"))
      res[[nm]] <- t1[[nm]]
    if (sx_disc_family == "gaussian") res$df2 <- t1$df_den
  }
  if (length(res$flags)) res$status <- "warning"
  res$flags <- paste(unique(res$flags), collapse = "; ")
  res
}
# Run a batch; per-candidate errors and warnings are recorded, never fatal.
sx_discover_batch <- function(batch) lapply(batch, function(s) {
  warns <- character()
  r <- tryCatch(withCallingHandlers(
      do.call(sx_discover_one, s[intersect(c("full", "tests", "vars", "focal_vars", "struct"), names(s))]),
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
    from ..analysis.eda import category_codes
    from .space import binomial_levels, view

    ds = view(ds, s)
    a = lambda n: alias(ds.var(n).alias)  # noqa: E731
    long = ([s.participant] + ([s.time] if s.time else [])) if s.participant else []
    vars_ = [a(n) for n in dict.fromkeys([s.outcome] + s.candidates + s.required + long)]
    y = a(s.outcome)
    lines = [
        DISCOVERY_HELPERS,
        "# ---- StatExplore: discovery setup ----",
        f"sx_disc_vars <- {r_vec(vars_)}",
        f"sx_disc_family <- {r_str(family)}",
        f"sx_disc_max_params <- {int(s.max_params)}",
        f"sx_disc_conf <- {r_num(s.conf)}",
        f"sx_disc_min_obs <- {int(getattr(s, 'min_obs', 0) or 0)}",
        f"sx_disc_shared <- {'TRUE' if s.sample == 'shared' else 'FALSE'}",
        "rm(list = ls(sx_disc_cache), envir = sx_disc_cache)",
        ("sx_dd <- sx_complete(dat, sx_disc_vars)  # shared complete-case sample"
         if s.sample == "shared" else
         "sx_dd <- droplevels(dat[, sx_disc_vars, drop = FALSE])  # complete cases are taken per model"),
    ]
    struct = (s.structures[0] if s.workflow == "structure" and s.structures else s.structure) or "ri"
    lines += [f"sx_disc_struct <- {r_str(struct)}",
              f"sx_disc_reml <- {'TRUE' if s.participant and s.workflow == 'structure' else 'FALSE'}"
              "  # REML only to compare covariance structures with a locked fixed part",
              f"sx_disc_corstr <- {r_str({'gee_exch': 'exchangeable', 'gee_ar1': 'ar1', 'gee_ind': 'independence'}.get(struct, 'exchangeable'))}",
              f"sx_disc_gee_family <- {r_str(s.family or 'gaussian')}",
              f"sx_disc_has_time <- {'TRUE' if s.participant and s.time else 'FALSE'}"]
    if s.participant:
        lines.append(f"sx_dd$sx_gid <- factor(sx_dd${a(s.participant)})  # participant (grouping, not a predictor)")
        if s.time:
            tv = ds.var(s.time)
            if tv.is_categorical:
                codes, labels = category_codes(ds, s.time)
                lines.append(f"sx_dd$sx_time <- {r_vec([float(c) for c in codes])}[match(as.character("
                             f"sx_dd${a(s.time)}), {r_vec(labels)})]  # numeric time codes (labels mapped back)")
            else:
                lines.append(f"sx_dd$sx_time <- as.numeric(sx_dd${a(s.time)})  # numeric time")
            lines += ["sx_dd$sx_timef <- factor(sx_dd$sx_time)  # time as categories (categorical time, mmrm)",
                      "sx_dd$sx_occ <- as.integer(sx_dd$sx_timef)  # occasion index (AR(1), GEE waves)",
                      "sx_dd <- sx_dd[order(sx_dd$sx_gid, sx_dd$sx_time), , drop = FALSE]  # within-participant order"]
    family = {"lmm": "gaussian", "glmm": "binomial", "clmm": "ordinal", "gee": s.family or "gaussian"}.get(family, family)
    v = ds.var(s.outcome)
    if family in ("gaussian", "lmm"):
        lines.append(f"sx_dd$sx_y <- sx_dd${y}")
    elif family == "binomial":
        event, ref = binomial_levels(ds, s)
        if v.is_categorical:
            ev, rf = ds.level_label(s.outcome, event), ds.level_label(s.outcome, ref)
            lines.append(f"sx_dd$sx_y <- as.integer(as.character(sx_dd${y}) == {r_str(ev)})  "
                         f"# event = {r_str(ev)}; reference = {r_str(rf)}")
        else:
            lines.append(f"sx_dd$sx_y <- as.integer(sx_dd${y} == {r_num(event)})  # event = {r_num(event)}; reference = {r_num(ref)}")
    else:
        _codes, labels = category_codes(ds, s.outcome)
        if family == "ordinal":
            lines.append(f"sx_dd$sx_y <- factor(as.character(sx_dd${y}), levels = {r_vec(labels)}, ordered = TRUE)"
                         "  # ordered categories, lowest first")
        else:
            ref = s.reference if s.reference is not None else ds.levels(s.outcome)[0]
            lab = ds.level_label(s.outcome, ref)
            lines += [f"sx_disc_ref <- {r_str(lab)}",
                      f"sx_dd$sx_y <- stats::relevel(factor(as.character(sx_dd${y}), levels = {r_vec(labels)}), "
                      "ref = sx_disc_ref)  # unordered; baseline = reference category"]
    lines.append("sx_result <- list(n = nrow(sx_dd), n_complete = sum(stats::complete.cases(sx_dd)))")
    return "\n".join(lines) + "\n"


def _rname(ds: Dataset, v: str, s: SearchSettings | None) -> str:
    """R column for a model variable: its alias, or the prepared time column in longitudinal searches."""
    if s is not None and s.participant and s.time and v == s.time:
        return "sx_timef" if s.time_type == "categorical" else "sx_time"
    return alias(ds.var(v).alias)


def _term(ds: Dataset, t: tuple[str, ...], s: SearchSettings | None = None) -> str:
    return ":".join(_rname(ds, v, s) for v in t)


def candidate_args(ds: Dataset, c: Candidate, s: SearchSettings | None = None) -> str:
    from .space import candidate_tests, full_terms

    terms = lambda ts: r_vec([_term(ds, t, s) for t in ts])  # noqa: E731
    tests = ", ".join(f"list(id = {t['id']}L, type = {r_str(t['type'])}, reduced = {terms(t['reduced'])}, "
                      f"drop = {terms(t['drop'])})" for t in candidate_tests(c))
    av = lambda vs: r_vec([_rname(ds, v, s) for v in vs])  # noqa: E731
    cells = av(c.focal) if c.kind in ("focal", "all_two_way") else \
        "list(" + ", ".join(av(t) for t in c.inter) + ")"
    struct = f", struct = {r_str(c.structure)}" if c.kind == "structure" else ""
    variables = list(dict.fromkeys(v for t in full_terms(c) for v in t)) if c.kind == "structure" else c.variables
    return (f"full = {terms(full_terms(c))}, tests = list({tests}), "
            f"vars = {av(variables)}, focal_vars = {cells}{struct}")


def batch_script(ds: Dataset, batch: list[Candidate], s: SearchSettings | None = None) -> str:
    items = ",\n  ".join(f"list(i = {c.index}L, {candidate_args(ds, c, s)})" for c in batch)
    return f"sx_result <- sx_discover_batch(list(\n  {items}))\n"


def candidate_script(load_script: str, ds: Dataset, s: SearchSettings, family: str, c: Candidate) -> str:
    setup = setup_script(ds, s, family).replace("sx_result <- list(n = nrow(sx_dd), n_complete = "
                                               "sum(stats::complete.cases(sx_dd)))\n", "")
    return (load_script + "\n" + setup + "\n# ---- candidate " + str(c.index) + " ----\n"
            + f"sx_result <- sx_discover_one({candidate_args(ds, c, s)})\nstr(sx_result)\n")
