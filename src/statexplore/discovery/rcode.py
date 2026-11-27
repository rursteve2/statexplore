"""R code for the discovery engine. `DISCOVERY_HELPERS` + `setup_script()` run once per
worker; each batch calls `sx_discover_one()` per candidate. `candidate_script()` gives
the exact code for a single candidate (View Code)."""
from __future__ import annotations

import math

from ..core.dataset import Dataset
from ..rbridge.codegen import factor_label, alias, r_num, r_str, r_vec
from .space import Candidate, SearchSettings

FIT_RULES_R = 4   # = engine.FIT_RULES (kept here to avoid an import cycle; asserted equal in the tests)

DISCOVERY_HELPERS = r"""# ---- StatExplore: discovery helpers ----
# Fit one candidate's full model once, then evaluate each planned hypothesis as a nested
# comparison with a reduced model (tests[[1]] is the candidate's declared focal test).
# v1.3.4 (V134-001 / V134-012): fit statistics are cached only under a FIT KEY that proves what was fitted - the
# content of every column the model uses (response, design variables, grouping, time, occasion) in the exact row order
# with the row identities, plus the estimation context (family, structure, cluster, REML/ML, GEE family and working
# correlation, optimizer rules) and the sample context set by the caller. Row names, counts or formulas alone never
# justify reuse; validation resets the cache for every training part and bootstrap draw (sx_cache_reset).
sx_disc_cache <- new.env()            # fit key -> fit statistics (with the producing fit's own diagnostics)
sx_disc_ctx <- "search"               # sample context: "search", or a validation split / bootstrap draw id
sx_disc_cache_log <- new.env()        # hits / misses (audit)
sx_disc_cache_log$hits <- 0L; sx_disc_cache_log$misses <- 0L
sx_disc_current <- NA_integer_        # candidate being evaluated (cache provenance)
sx_cache_reset <- function(ctx = "search") {
  rm(list = ls(sx_disc_cache, all.names = TRUE), envir = sx_disc_cache)
  rm(list = ls(sx_col_hashes, all.names = TRUE), envir = sx_col_hashes)
  sx_disc_ctx <<- ctx
}
sx_col_hashes <- new.env()            # column name -> (row names, column, MD5): each column hashed once per sample
sx_col_hash <- function(d, v) {
  rn <- rownames(d); col <- d[[v]]; e <- sx_col_hashes[[v]]
  if (!is.null(e) && identical(e$rn, rn) && identical(e$col, col)) return(e$h)
  h <- unname(tools::md5sum(bytes = serialize(list(v, rn, if (is.factor(col)) levels(col), unclass(col)), NULL, xdr = TRUE)))
  assign(v, list(rn = rn, col = col, h = h), envir = sx_col_hashes)
  h
}
sx_fit_key <- function(f, d) {
  cols <- intersect(c("sx_y", all.vars(f), "sx_gid", "sx_cluster", "sx_time", "sx_timef", "sx_occ"), names(d))
  terms <- sort(sx_canon_names(attr(stats::terms(f), "term.labels")))
  paste(c(sx_disc_ctx, sx_disc_family, sx_disc_struct, sx_disc_clustered, sx_disc_reml,
          if (sx_disc_family == "gee") c(sx_disc_gee_family, sx_gee_corstr(sx_disc_struct)), sx_disc_fit_rules,
          paste(terms, collapse = "+"), nrow(d), vapply(cols, function(v) paste0(v, "=", sx_col_hash(d, v)), "")),
        collapse = "|")
}
sx_gee_corstr <- function(st) switch(st, gee_exch = "exchangeable", gee_ar1 = "ar1", gee_ind = "independence",
                                      stop("not a GEE structure: ", st))
# Analysis-row signature: MD5 of the row identities in fitting order plus the levels of every factor model column
# (participant codes excluded: only their grouping matters). Equal signatures = same rows, order and coding.
sx_rows_sig <- function(d) {
  fac <- setdiff(names(d)[vapply(d, is.factor, logical(1))], "sx_gid")
  txt <- c(rownames(d), vapply(fac, function(v) paste0(v, ":", paste(levels(d[[v]]), collapse = "|")), ""))
  tf <- tempfile(); on.exit(unlink(tf)); writeLines(txt, tf, useBytes = TRUE)
  unname(tools::md5sum(tf))
}
sx_add <- function(f, extra) stats::update(f, stats::as.formula(paste(". ~ . +", extra)))
sx_last_run <- NULL                   # optimizer attempts of the most recent Gaussian longitudinal fit
sx_fit_lmm <- function(f, d, st, reml, clustered = sx_disc_clustered) {
  if (startsWith(st, "mmrm")) {
    cs <- switch(st, mmrm_us = "us", mmrm_cs = "cs", mmrm_ar1 = "ar1")
    sx_last_run <<- NULL
    return(mmrm::mmrm(sx_add(f, paste0(cs, "(sx_timef | sx_gid)")), data = d, reml = reml))
  }
  g <- sx_disc_gstruct[[st]]
  if (is.null(g)) stop("Unknown covariance structure: ", st)
  # one structure builder for every module (random part, residual correlation, nested cluster)
  run <- if (g[3] == "lme4")
    sx_fit_attempts(function(opt) lme4::lmer(sx_add(f, sx_re_lme4(g[1], clustered, "sx_time", "sx_gid")), data = d,
                                             REML = reml), "default") else
    sx_fit_attempts(function(opt) nlme::lme(f, random = sx_re_nlme(g[1], clustered, "sx_time", "sx_gid"),
      correlation = sx_cor_nlme(g[2], clustered, "sx_occ", "sx_time", "sx_gid"), data = d,
      method = if (reml) "REML" else "ML", control = nlme::lmeControl(opt = opt, msMaxIter = 200)),
      c("nlminb", "optim"))
  sx_last_run <<- run
  if (is.null(run$fit)) stop(run$error)
  run$fit
}
sx_fit_model <- function(f, d) {
  st <- sx_disc_struct
  switch(sx_disc_family,
    lmm = sx_fit_lmm(f, d, st, sx_disc_reml, sx_disc_clustered),
    glmm = lme4::glmer(sx_add(f, if (st == "glmer_rs") "(1 + sx_time | sx_gid)" else "(1 | sx_gid)"), data = d,
                       family = stats::binomial()),
    clmm = ordinal::clmm(sx_add(f, if (st == "clmm_rs") "(1 + sx_time | sx_gid)" else "(1 | sx_gid)"), data = d,
                         link = "logit"),
    gee = sx_fit_gee(f, d, st),
    gaussian = stats::lm(f, d),
    ordinal = ordinal::clm(f, data = d, link = "logit"),
    multinomial = nnet::multinom(f, data = d, trace = FALSE, maxit = 1000, model = FALSE),
    stats::glm(f, data = d, family = stats::binomial()))
}
# GEE (V134-005): the working correlation of THIS candidate's structure, passed explicitly and read back from the
# fitted object; a mismatch fails closed.
sx_fit_gee <- function(f, d, st) {
  cs <- sx_gee_corstr(st)
  m <- geepack::geeglm(f, data = d, id = sx_gid, waves = sx_occ, corstr = cs,
                       family = if (sx_disc_gee_family == "binomial") stats::binomial() else stats::gaussian())
  if (!identical(as.character(m$corstr), cs))
    stop(sprintf("the fitted working correlation (%s) differs from the requested one (%s)", m$corstr, cs))
  m
}
sx_gee_readback <- function(m) {
  # what geepack actually estimated (never the request): working correlation, its parameter(s), scale, clusters
  list(corstr = as.character(m$corstr), alpha = I(as.numeric(m$geese$alpha)), scale = as.numeric(m$geese$gamma),
       clusters = length(unique(m$id)), nobs = length(m$y), family = m$family$family, link = m$family$link,
       robust_se = I(as.numeric(sqrt(diag(as.matrix(m$geese$vbeta))))))
}
# Fit a model and its statistics, capturing the warnings of THIS fit (V134-010): they belong to the fit that produced
# them and travel with it into the cache, so a reused reduced model keeps its diagnostics and an unrelated fit's warning
# never flags another test.
sx_fit_capture <- function(f, d) {
  w <- character()
  st <- withCallingHandlers({ m <- sx_fit_model(f, d); s <- sx_fit_stats(m, d); list(m = m, s = s) },
    warning = function(x) { w <<- c(w, conditionMessage(x)); invokeRestart("muffleWarning") })
  w <- unique(w)
  if (length(w)) {
    known <- if (is.null(st$s$diag)) character() else st$s$diag$flags
    extra <- w[!vapply(w, function(x) any(grepl(x, known, fixed = TRUE)), logical(1))]
    if (length(extra)) st$s$conv <- paste(c(if (nzchar(st$s$conv)) st$s$conv, paste("warning:", extra)), collapse = "; ")
  }
  st$s$warnings <- w
  st
}
sx_fit_stats <- function(m, d) {
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
  if (sx_disc_family == "glmm" && inherits(m, "merMod")) {
    conv <- unlist(m@optinfo$conv$lme4$messages)
    if (length(conv)) out$conv <- paste("convergence:", conv[1])
  }
  out$r2m <- NA_real_
  if (sx_disc_family == "lmm") {
    # every diagnostic of the fit (convergence, singular / near-boundary, apVar, optimizer fallback): STAT-18 / 32
    out$diag <- sx_diag(m, if (inherits(m, c("merMod", "lme"))) sx_last_run else NULL)
    out$conv <- paste(out$diag$flags, collapse = "; ")
    if (inherits(m, c("merMod", "lme"))) out$r2m <- sx_r2_direct(m, d)$m     # direct decomposition (STAT-45)
  }
  out
}
sx_reduced_stats <- function(reduced, d) {
  f0 <- stats::reformulate(if (length(reduced)) reduced else "1", response = "sx_y")
  key <- sx_fit_key(f0, d)
  hit <- sx_disc_cache[[key]]
  if (!is.null(hit)) {
    sx_disc_cache_log$hits <- sx_disc_cache_log$hits + 1L
    hit$source <- "cache"; return(hit)
  }
  sx_disc_cache_log$misses <- sx_disc_cache_log$misses + 1L
  st <- sx_fit_capture(f0, d)$s
  st$m <- NULL
  st$source <- "fitted"; st$fitted_for <- sx_disc_current; st$key_md5 <- unname(tools::md5sum(bytes = charToRaw(key)))
  assign(key, st, envir = sx_disc_cache)
  st
}
sx_null_scale <- function(d) {
  # denominators of the pseudo-R2 effects: -2 log L (ordinal) or deviance (logistic, multinomial) of y ~ 1, fitted to
  # exactly these rows (same fit key as any other model)
  st <- sx_reduced_stats(character(), d)
  if (sx_disc_family == "ordinal") -2 * st$ll else st$dev
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
       log_p = stats::pchisq(w, length(cols), lower.tail = FALSE, log.p = TRUE), effect = NA_real_, partial_eta2 = NA_real_)
}
sx_compare <- function(s1, s0, d) {
  # V134-002: the natural log of the upper tail comes from the distribution itself (log.p = TRUE), never log(p)
  r <- list(statistic = NA_real_, df1 = NA_real_, df_den = NA_real_, p = NA_real_, log_p = NA_real_, effect = NA_real_,
            partial_eta2 = NA_real_)
  if (sx_disc_family == "gaussian") {
    ss <- s0$rss - s1$rss; r$df1 <- s0$dfres - s1$dfres; r$df_den <- s1$dfres
    r$statistic <- (ss / r$df1) / (s1$rss / s1$dfres)
    if (!is.finite(r$statistic) || r$df1 <= 0) {
      r$statistic <- NA_real_; r$withheld <- "no extra parameters or non-finite F statistic"
    } else if (r$statistic < 0) {
      # residual SS of the larger model above the smaller one's: only rounding noise can do this for nested OLS
      if (ss < -1e-8 * max(1, s0$rss)) r$withheld <- sprintf("the larger model has a larger residual SS (%.6g): no p-value", ss)
      else { r$statistic <- 0; r$p <- 1; r$log_p <- 0 }
    } else {
      r$p <- stats::pf(r$statistic, r$df1, r$df_den, lower.tail = FALSE)
      r$log_p <- stats::pf(r$statistic, r$df1, r$df_den, lower.tail = FALSE, log.p = TRUE)
    }
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
  # STAT-41: a materially negative LR (larger model fits worse) means an unreliable optimization: no p-value
  t <- sx_lr_test(0, lr / 2, r$df1, "chisq")
  r$p <- t$p; r$log_p <- t$log_p; r$LR_raw <- t$LR_raw
  if (!identical(t$status, "tested")) r$withheld <- t$reason
  r$effect <- switch(sx_disc_family, lmm = s1$r2m - s0$r2m,             # delta marginal R-squared (lme4 RI)
                     glmm = NA_real_, clmm = NA_real_,
                     lr / sx_null_scale(d))                             # delta McFadden pseudo-R-squared
  r
}
sx_long <- function() sx_disc_family %in% c("lmm", "glmm", "clmm", "gee")
sx_discover_one <- function(full, tests, vars, focal_vars, struct = NULL, clustered = NULL, i = NA_integer_) {
  sx_disc_current <<- i
  if (!is.null(struct)) {      # workflow B: this candidate's structure
    old_struct <- sx_disc_struct; sx_disc_struct <<- struct; on.exit(sx_disc_struct <<- old_struct)
  }
  if (!is.null(clustered)) {   # workflow B exploratory: this candidate with or without the cluster variance
    old_cl <- sx_disc_clustered; sx_disc_clustered <<- clustered; on.exit(sx_disc_clustered <<- old_cl, add = TRUE)
  }
  res <- list(status = "ok", reason = "", flags = character(), n = NA_real_, n_params = NA_real_,
              n_groups = NA_real_, statistic = NA_real_, df1 = NA_real_, df2 = NA_real_, p = NA_real_,
              effect = NA_real_, partial_eta2 = NA_real_, estimate = NA_real_, se = NA_real_,
              ci_low = NA_real_, ci_high = NA_real_, r2 = NA_real_, adj_r2 = NA_real_, aic = NA_real_,
              bic = NA_real_, loglik = NA_real_, tests = list())
  invalid <- function(why) { res$status <- "invalid"; res$reason <- why; res$flags <- ""; res$tests <- NULL; res }
  d <- sx_complete(sx_dd, c("sx_y", vars, if (sx_long()) c("sx_gid", if (sx_disc_has_cluster) "sx_cluster",
                                                            if (sx_disc_has_time) c("sx_time", "sx_timef", "sx_occ"))))
  # occasion index over the model's own rows (sx_timef levels were dropped above): a time value no analysed row has
  # is compressed exactly as in Full Analysis; a participant's skipped occasion still leaves an index gap
  if (sx_long() && sx_disc_has_time) d$sx_occ <- as.integer(d$sx_timef)
  n <- nrow(d); res$n <- n; res$rows_sig <- sx_rows_sig(d)
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
  if (sx_long() && (sx_disc_struct %in% c("glmer_rs", "clmm_rs") ||
                    (!is.null(sx_disc_gstruct[[sx_disc_struct]]) && sx_disc_gstruct[[sx_disc_struct]][1] != "ri"))) {
    k2 <- sum(tapply(d$sx_time, droplevels(d$sx_gid), function(x) length(unique(x))) >= 2)
    if (k2 < 10) return(invalid(sprintf("only %d participants have 2+ time points: a random slope is not estimable", k2)))
  }
  fc <- sx_fit_capture(f1, d); m1 <- fc$m; s1 <- fc$s
  if (sx_disc_family == "gee") res$gee <- sx_gee_readback(m1)
  if (!is.null(s1$diag)) {
    res$flags <- c(res$flags, if (nzchar(s1$conv)) strsplit(s1$conv, "; ", fixed = TRUE)[[1]])
    res$diagnostics <- s1$diag[c("status", "optimizer", "apvar")]
    res$diagnostics$attempts <- if (is.null(sx_last_run)) "" else paste(sprintf("%s: %s%s", sx_last_run$attempts$optimizer,
      sx_last_run$attempts$status, ifelse(nzchar(sx_last_run$attempts$error), paste0(" (", sx_last_run$attempts$error, ")"), "")),
      collapse = "; ")
  } else if (nzchar(s1$conv)) res$flags <- c(res$flags, s1$conv)
  # identity fingerprints of exactly what was fitted (rows, outcome, design, grouping): STAT-22
  res$identity <- sx_identity(d, f1, "sx_y", if (sx_long()) c("sx_gid", if (sx_disc_clustered) "sx_cluster") else character())
  if (sx_long()) {
    res$n_groups <- nlevels(droplevels(d$sx_gid))
    res$clustered <- sx_disc_clustered
    if (sx_disc_has_cluster) res$observed_clusters <- nlevels(droplevels(d$sx_cluster))
    res$modeled_clusters <- if (sx_disc_clustered) nlevels(droplevels(d$sx_cluster)) else 0
    if (sx_disc_family != "lmm" && inherits(m1, "merMod") && lme4::isSingular(m1))
      res$flags <- c(res$flags, "singular fit (a random-effect variance is ~ 0 or a correlation is +/-1)")
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
  if (sx_disc_family != "gee") { ic <- sx_ic(m1); res$aic <- ic$aic; res$bic <- ic$bic; res$loglik <- s1$ll
    res$bic_n <- ic$n_bic; res$ic_basis <- ic$basis
    res$n_lik_params <- ic$k }   # every parameter of the likelihood (fixed + thresholds + covariance)
  for (t in tests) {
    if (sx_disc_family == "gee") {
      tr <- sx_wald(m1, X, f1, t$drop)
    } else {
      s0 <- sx_reduced_stats(t$reduced, d)
      tr <- sx_compare(s1, s0, d)
      # V134-010: the reference (reduced) fit's own diagnostics belong to THIS test only, with their provenance: a
      # flagged reference makes the test provisional; it does not flag the model or its other tests
      tr$reference_flags <- s0$conv
      tr$reference_source <- s0$source
      tr$reference_fitted_for <- if (is.null(s0$fitted_for)) NA_integer_ else s0$fitted_for
      tr$reference_key <- if (is.null(s0$key_md5)) "" else s0$key_md5
      tr$reference_loglik <- if (is.null(s0$ll)) NA_real_ else s0$ll
      tr$reference_k <- if (is.null(s0$k)) NA_real_ else s0$k
    }
    if (sx_disc_family == "multinomial" && isTRUE(tr$statistic < -1e-6))
      tr$reference_flags <- paste(c(if (nzchar(tr$reference_flags)) tr$reference_flags,
                                    "optimizer: the reduced model fitted better than the full model"), collapse = "; ")
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
    # QIC refits the independence model from the stored call (formula `f`, data `d`): give it both (P1-07)
    # V134-005: QIC, QICu and CIC of the ACTUAL fit (native geepack definitions), kept in their own fields
    qic_env <- list2env(list(f = f1, d = d), parent = environment())
    q <- tryCatch(geepack::QIC(m1, env = qic_env),
                  error = function(e) { res$flags <<- c(res$flags, paste("QIC unavailable:", conditionMessage(e))); NULL })
    pick <- function(nm) if (!is.null(q) && nm %in% names(q)) as.numeric(q[[nm]]) else NA_real_
    res$qic <- pick("QIC"); res$qicu <- pick("QICu"); res$cic <- pick("CIC"); res$quasi_lik <- pick("Quasi Lik")
    res$aic <- NA_real_; res$bic <- NA_real_; res$loglik <- NA_real_
  }
  # the declared focal test, also at the top level (v1.1 row layout)
  if (length(res$tests)) {
    t1 <- res$tests[[1]]
    for (nm in c("statistic", "df1", "p", "log_p", "effect", "partial_eta2", "estimate", "se", "ci_low", "ci_high"))
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
      do.call(sx_discover_one, s[intersect(c("full", "tests", "vars", "focal_vars", "struct", "clustered", "i"), names(s))]),
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

    from ..analysis.covstruct import GAUSSIAN_STRUCTURES, MIXED_R_HELPERS

    ds = view(ds, s)
    a = lambda n: alias(ds.var(n).alias)  # noqa: E731
    cluster = getattr(s, "cluster", None) if s.participant else None
    long = ([s.participant] + ([s.time] if s.time else []) + ([cluster] if cluster else [])) if s.participant else []
    vars_ = [a(n) for n in dict.fromkeys([s.outcome] + s.candidates + s.required + long)]
    y = a(s.outcome)
    gstruct = "list(" + ", ".join(f"{k} = {r_vec(list(v))}" for k, v in GAUSSIAN_STRUCTURES.items()) + ")"
    lines = [
        MIXED_R_HELPERS,
        DISCOVERY_HELPERS,
        "# ---- StatExplore: discovery setup ----",
        f"sx_disc_gstruct <- {gstruct}  # structure key -> (random, residual, engine)",
        f"sx_disc_clustered <- {'TRUE' if cluster else 'FALSE'}  # participants nested in a higher-level cluster",
        f"sx_disc_has_cluster <- {'TRUE' if cluster else 'FALSE'}",
        f"sx_disc_vars <- {r_vec(vars_)}",
        f"sx_disc_family <- {r_str(family)}",
        f"sx_disc_max_params <- {int(s.max_params)}",
        f"sx_disc_conf <- {r_num(s.conf)}",
        f"sx_disc_min_obs <- {int(getattr(s, 'min_obs', 0) or 0)}",
        f"sx_disc_shared <- {'TRUE' if s.sample == 'shared' else 'FALSE'}",
        f"sx_disc_fit_rules <- {r_str('fit-rules-' + str(FIT_RULES_R))}  # part of every fit key",
        "sx_cache_reset(\"search\")",
        ("sx_dd <- sx_complete(dat, sx_disc_vars)  # shared complete-case sample"
         if s.sample == "shared" else
         "sx_dd <- droplevels(dat[, sx_disc_vars, drop = FALSE])  # complete cases are taken per model"),
    ]
    struct = (s.structures[0] if s.workflow == "structure" and s.structures else s.structure) or "ri"
    lines += [f"sx_disc_struct <- {r_str(struct)}",
              f"sx_disc_reml <- {'TRUE' if s.participant and s.workflow == 'structure' else 'FALSE'}"
              "  # REML only to compare covariance structures with a locked fixed part",
              f"sx_disc_gee_family <- {r_str(s.family or 'gaussian')}",
              f"sx_disc_has_time <- {'TRUE' if s.participant and s.time else 'FALSE'}"]
    if s.participant and cluster:
        cl, pid = a(cluster), a(s.participant)
        lines.append(f"sx_dd$sx_cluster <- factor(sx_dd${cl})  # higher-level cluster (missing cluster = excluded row)")
        if getattr(s, "id_policy", "global") == "global":
            lines += [f"sx_moves <- tapply(as.character(sx_dd${cl}), as.character(sx_dd${pid}), function(z) length(unique(stats::na.omit(z))))",
                      "if (any(sx_moves > 1, na.rm = TRUE)) stop(\"Participant IDs occur in more than one cluster: declare IDs as "
                      "numbered within each cluster, or the design is crossed (not supported).\")"]
        lines.append(f"sx_dd$sx_gid <- sx_nested_id(sx_dd$sx_cluster, sx_dd${pid})  # participant = (cluster, ID) composite key"
                     + (" (IDs numbered within clusters)" if getattr(s, "id_policy", "global") == "local" else ""))
    elif s.participant:
        lines.append(f"sx_dd$sx_gid <- factor(sx_dd${a(s.participant)})  # participant (grouping, not a predictor)")
    if s.participant:
        if s.time:
            tv = ds.var(s.time)
            if tv.is_categorical:
                codes, labels = category_codes(ds, s.time)
                lines.append(f"sx_dd$sx_time <- {r_vec([float(c) for c in codes])}[match(as.character("
                             f"sx_dd${a(s.time)}), {r_vec(labels)})]  # numeric time codes (labels mapped back)")
            else:
                lines.append(f"sx_dd$sx_time <- as.numeric(sx_dd${a(s.time)})  # numeric time")
            lines += ["sx_dd$sx_timef <- factor(sx_dd$sx_time)  # time as categories (categorical time, mmrm)",
                      "sx_dd$sx_occ <- as.integer(sx_dd$sx_timef)  # global observed-occasion index (AR(1), GEE waves)",
                      "sx_dd <- sx_dd[order(sx_dd$sx_gid, sx_dd$sx_time), , drop = FALSE]  # within-participant order"]
    family = {"lmm": "gaussian", "glmm": "binomial", "clmm": "ordinal", "gee": s.family or "gaussian"}.get(family, family)
    v = ds.var(s.outcome)
    if family in ("gaussian", "lmm"):
        lines.append(f"sx_dd$sx_y <- sx_dd${y}")
    elif family == "binomial":
        event, ref = binomial_levels(ds, s)
        if v.is_categorical:
            ev, rf = factor_label(ds, s.outcome, event), factor_label(ds, s.outcome, ref)
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
            lab = factor_label(ds, s.outcome, ref)
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
    if c.kind == "structure" and c.clustered is not None:
        struct += f", clustered = {'TRUE' if c.clustered else 'FALSE'}"
    variables = list(dict.fromkeys(v for t in full_terms(c) for v in t)) if c.kind == "structure" else c.variables
    return (f"full = {terms(full_terms(c))}, tests = list({tests}), "
            f"vars = {av(variables)}, focal_vars = {cells}{struct}")


def batch_script(ds: Dataset, batch: list[Candidate], s: SearchSettings | None = None) -> str:
    items = ",\n  ".join(f"list(i = {c.index}L, {candidate_args(ds, c, s)})" for c in batch)
    return f"sx_result <- sx_discover_batch(list(\n  {items}))\n"


CANDIDATE_SUMMARY = (
    'cat(sprintf("status = %s; n = %s; groups = %s; AIC = %.3f; BIC = %.3f; logLik = %.3f; likelihood parameters = %s; '
    'flags = %s\\n", sx_result$status, sx_result$n, sx_result$n_groups, sx_result$aic, sx_result$bic, '
    'sx_result$loglik, if (is.null(sx_result$n_lik_params)) NA else sx_result$n_lik_params, '
    'if (nzchar(sx_result$flags)) sx_result$flags else "none"))\n')


def _mixture_r(weights: dict[int, float] | None, df: float) -> str:
    """R expression for the p-value of statistic sx_lr: chi-square(df), or a boundary chi-square mixture."""
    if not weights:
        return f"stats::pchisq(max(sx_lr, 0), {int(df)}, lower.tail = FALSE)"
    parts = [(f"{w:g} * (sx_lr <= 0)" if int(k) == 0 else f"{w:g} * stats::pchisq(max(sx_lr, 0), {int(k)}, lower.tail = FALSE)")
             for k, w in sorted(weights.items(), key=lambda kv: int(kv[0]))]
    return " + ".join(parts)


def comparison_script(ds: Dataset, s: SearchSettings, c: Candidate, comparisons: list[dict]) -> str:
    """Workflow B: refit each reference structure on the same rows and fixed part and recompute the reported
    covariance-structure likelihood-ratio tests with the same reference distribution (sx_lr_test, the R twin of the
    app's test service), so exports reproduce the claimed inference too (P0-02, STAT-17/23)."""
    import dataclasses

    from ..rbridge.codegen import r_num

    if not comparisons:
        return ""
    L = ["# ---- covariance-structure comparisons shown in the discovery results (same rows, same fixed part, "
         + ("REML" if s.workflow == "structure" else "ML") + ") ----", "sx_cmp <- list()"]
    for k, t in enumerate(comparisons, start=1):
        ref = dataclasses.replace(c, structure=t["reference_structure"], clustered=t.get("reference_clustered")
                                  if t.get("reference_clustered") is not None or c.clustered is not None else None)
        kind = t.get("reference_kind") or ("mixture" if t.get("boundary") else "chisq")
        L += [f"sx_ref <- sx_discover_one({candidate_args(ds, ref, s)})",
              "# nested comparison only on identical rows: same row identities, order and factor coding",
              "stopifnot(\"the reference model was fitted to different rows\" = identical(sx_ref$rows_sig, sx_result$rows_sig))",
              f"sx_t <- sx_lr_test(sx_ref$loglik, sx_result$loglik, {r_num(float(t['df1']))}, {r_str(kind)})",
              f"sx_cmp[[{k}]] <- data.frame(comparison = {r_str(t['label'])}, reference = {r_str(t['reference_structure'])},",
              f"  null = {r_str(t.get('null') or '')}, distribution = {r_str(t.get('reference_distribution') or kind)},",
              "  LR_raw = sx_t$LR_raw, LR = sx_t$LR, df = sx_t$df, p = sx_t$p, status = sx_t$status,",
              f"  app_LR = {r_num(t.get('statistic'))}, app_p = {r_num(t.get('p'))})"]
        if t.get("statistic") is not None:
            L.append(f"stopifnot({r_str('comparison ' + str(k) + ': LR differs from the app')} = "
                     f"sx_rel(sx_cmp[[{k}]]$LR, sx_cmp[[{k}]]$app_LR))")
        if t.get("p") is not None:
            L.append(f"stopifnot({r_str('comparison ' + str(k) + ': p differs from the app')} = "
                     f"sx_close_p(sx_cmp[[{k}]]$p, sx_cmp[[{k}]]$app_p))")
        else:
            L.append(f"# comparison {k}: no p-value in the app ({t.get('test_reason') or t.get('test_status') or 'withheld'})")
    L += ["sx_comparisons <- do.call(rbind, sx_cmp)", "print(sx_comparisons, digits = 6)",
          'cat("Covariance-structure comparisons reproduced within tolerance.\\n")']
    return "\n".join(L) + "\n"


VERIFY_HELPERS = """# ---- verification against the values StatExplore recorded ----
# Tolerances, fixed in advance: n and analysis-row signature exact; logLik / AIC / BIC 1e-3 absolute;
# test statistics 1e-4 relative; p-values 1e-4 relative on the log10 scale (both below 1e-300 count as equal).
sx_abs <- function(new, app, tol = 1e-3) isTRUE(abs(new - app) <= tol)
sx_rel <- function(new, app, tol = 1e-4) isTRUE(abs(new - app) <= tol * max(1, abs(app)))
sx_close_p <- function(new, app) isTRUE((new < 1e-300 && app < 1e-300) ||
  abs(log10(max(new, 1e-300)) - log10(max(app, 1e-300))) <= 1e-4 * max(1, abs(log10(max(app, 1e-300)))))
"""


def _finite(x: object) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def verification_script(row: dict, adjusted: list[dict] | None = None) -> str:
    """stopifnot() checks that the re-run reproduces what the app recorded for this candidate (P1-EXPORT): n,
    analysis-row signature, logLik, AIC, BIC and every test statistic and p. Adjusted p-values depend on the whole
    hypothesis family, so they are listed (with their family) rather than recomputed."""
    checks = []
    if _finite(row.get("n")):
        checks.append(f'"n differs from the app" = identical(as.numeric(sx_result$n), {r_num(float(row["n"]))})')
    if row.get("rows_sig"):
        checks.append(f'"analysis rows differ from the app (row signature)" = identical(sx_result$rows_sig, '
                      f'{r_str(row["rows_sig"])})')
    ident = row.get("identity") or {}
    if int(ident.get("schema") or 0) >= 2:      # SHA-256 fingerprints of what the engine used (STAT-22)
        for k in ("rows", "response", "design", "groups"):
            if ident.get(k):
                checks.append(f'"{k} differ from the app (identity fingerprint)" = identical(sx_result$identity${k}, '
                              f'{r_str(ident[k])})')
    for key, label in (("loglik", "logLik"), ("aic", "AIC"), ("bic", "BIC")):
        if _finite(row.get(key)):
            checks.append(f'"{label} differs from the app" = sx_abs(sx_result${key}, {r_num(float(row[key]))})')
    L = [VERIFY_HELPERS.rstrip("\n"),
         "sx_test <- function(id) Filter(function(t) identical(as.integer(t$id), as.integer(id)), sx_result$tests)[[1]]"]
    for t in row.get("tests") or []:
        if not (_finite(t.get("statistic")) and _finite(t.get("p"))):
            continue
        tid = int(t.get("test_id", 0))
        lab = str(t.get("label") or t.get("type") or tid)
        checks.append(f'{r_str("test " + lab + ": statistic differs from the app")} = '
                      f'sx_rel(sx_test({tid})$statistic, {r_num(float(t["statistic"]))})')
        checks.append(f'{r_str("test " + lab + ": p differs from the app")} = '
                      f'sx_close_p(sx_test({tid})$p, {r_num(float(t["p"]))})')
    if checks:
        L.append("stopifnot(\n  " + ",\n  ".join(checks) + ")")
        L.append(f'cat("Reproduced within tolerance: {len(checks)} checks against the app\'s recorded values.\\n")')
    if adjusted:
        L.append("# Multiplicity adjustment as reported (it depends on every test in the family, so it is listed, not "
                 "recomputed here):")
        for a in adjusted:
            L.append(f"#   {a.get('label', '')} [{a.get('test_label', '')}]: p = {a.get('p')}, adjusted p = "
                     f"{a.get('p_adj')} ({a.get('adj_method', '')}, family of {a.get('family_size')} tests; "
                     f"{a.get('search_scope', '')})")
    return "\n".join(L) + "\n"


def candidate_script(load_script: str, ds: Dataset, s: SearchSettings, family: str, c: Candidate,
                     comparisons: list[dict] | None = None, expected: dict | None = None,
                     adjusted: list[dict] | None = None, provenance: dict[str, str] | None = None) -> str:
    setup = setup_script(ds, s, family).replace("sx_result <- list(n = nrow(sx_dd), n_complete = "
                                               "sum(stats::complete.cases(sx_dd)))\n", "")
    head = ("# ---- model provenance (as recorded by the search) ----\n"
            + "".join(f"#   {k}: {v}\n" for k, v in provenance.items()) if provenance else "")
    return (head + load_script + "\n" + setup + "\n# ---- candidate " + str(c.index) + " ----\n"
            + f"sx_result <- sx_discover_one({candidate_args(ds, c, s)})\n" + CANDIDATE_SUMMARY
            + (verification_script(expected, adjusted) if expected else VERIFY_HELPERS)
            + comparison_script(ds, s, c, comparisons or []))
