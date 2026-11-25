# ---- StatExplore: shared longitudinal model helpers (v1.3.2) ----
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
    vc <- tryCatch(ordinal::VarCorr(m), error = function(e) NULL); sing <- FALSE
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

.libPaths("C:/Users/Stephen/Desktop/My stuff/statexplore/r/library")
set.seed(132)                               # predeclared seed and scenarios (written before running)
R <- 200; ns <- 20; np <- 8; nt <- 6
gen <- function(school_sd, group_eff) {
  ni <- ns * np
  d <- data.frame(sx_cluster = factor(rep(1:ns, each = np * nt)), ID = rep(1:ni, each = nt), sx_time = rep(0:(nt - 1), ni))
  d$GROUP <- factor(as.integer(d$ID %% 2 == 0))
  d$sx_id <- sx_nested_id(d$sx_cluster, d$ID)
  d$sx_tidx <- as.integer(factor(d$sx_time)); d$sx_tnum <- d$sx_time
  e <- unlist(lapply(1:ni, function(i) as.numeric(arima.sim(list(ar = .35), n = nt))))
  d$Y <- 20 + .5 * d$sx_time + group_eff * (d$GROUP == "1") + rnorm(ns, 0, school_sd)[d$sx_cluster] +
    rnorm(ni, 0, 1.5)[d$ID] + e
  d
}
fit <- function(d, f, clustered, method) sx_fit_attempts(function(opt) nlme::lme(f, random = sx_re_nlme("ri", clustered, "sx_time"),
  correlation = sx_cor_nlme("ar1", clustered), data = d, method = method, control = nlme::lmeControl(opt = opt, msMaxIter = 200)),
  c("nlminb", "optim"))
res <- list()
for (r in seq_len(R)) {
  # scenario 1: GROUP effect 0, school SD 2.5 -> Workflow A LR test of GROUP (ML), 3L RI + AR(1)
  d <- gen(2.5, 0)
  a <- fit(d, Y ~ sx_time + GROUP, TRUE, "ML"); b <- fit(d, Y ~ sx_time, TRUE, "ML")
  t1 <- if (is.null(a$fit) || is.null(b$fit)) NULL else sx_lr_test(as.numeric(logLik(b$fit)), as.numeric(logLik(a$fit)), 1, "chisq")
  ci <- if (is.null(a$fit)) c(NA, NA) else { tt <- summary(a$fit)$tTable["sx_time", ]; tt[1] + c(-1, 1) * qt(.975, tt[3]) * tt[2] }
  # scenario 2: no school variance -> cluster variance boundary test (REML, same rows and composite participants)
  d2 <- gen(0, .8)
  c1 <- fit(d2, Y ~ sx_time + GROUP, TRUE, "REML"); c0 <- fit(d2, Y ~ sx_time + GROUP, FALSE, "REML")
  t2 <- if (is.null(c1$fit) || is.null(c0$fit)) NULL else sx_lr_test(as.numeric(logLik(c0$fit)), as.numeric(logLik(c1$fit)), 1, "mixture")
  res[[r]] <- data.frame(rep = r, ok1 = !is.null(t1), p_group = if (is.null(t1)) NA else t1$p, st1 = if (is.null(t1)) "failed" else t1$status,
    cover_time = !is.na(ci[1]) && ci[1] <= .5 && .5 <= ci[2], ok2 = !is.null(t2), p_cluster = if (is.null(t2)) NA else t2$p,
    st2 = if (is.null(t2)) "failed" else t2$status, fallback = nrow(a$attempts) > 1 || (!is.null(c1$attempts) && nrow(c1$attempts) > 1))
}
x <- do.call(rbind, res)
write.csv(x, "E:/sx_v132_work/sim/sim_replicates.csv", row.names = FALSE)
mc <- function(p) sqrt(p * (1 - p) / R)
s <- data.frame(
  metric = c("A GROUP LR: rejection at .05 (null true)", "A: fits converged", "TIME 95% CI coverage",
             "cluster variance boundary test: rejection at .05 (null true)", "cluster test: p = 1 share (LR = 0)", "optimizer fallback used"),
  value = c(mean(x$p_group < .05, na.rm = TRUE), mean(x$ok1), mean(x$cover_time), mean(x$p_cluster < .05, na.rm = TRUE),
            mean(x$p_cluster >= 1 - 1e-12, na.rm = TRUE), mean(x$fallback)))
s$mc_se <- mc(s$value)
print(s, digits = 4); write.csv(s, "E:/sx_v132_work/sim/sim_summary.csv", row.names = FALSE)
writeLines(c(R.version.string, paste("nlme", packageVersion("nlme"))), "E:/sx_v132_work/sim/sim_session.txt")
