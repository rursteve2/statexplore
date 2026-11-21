"""Mixed-effects and longitudinal models (§9.6, §9.7, §15).

One default engine per covariance structure, recorded with every result:
  independent residuals  -> lme4/lmerTest (Satterthwaite df)
  AR(1) / CAR(1)         -> nlme::lme (random effects + residual correlation)
  unstructured           -> mmrm (marginal model, no random effects)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..core.dataset import Dataset
from ..core.schema import AnalysisResult, Status
from ..core.variables import Measure, Storage
from ..rbridge.codegen import alias, level_labels, r_num, r_str, r_vec
from .session import RSession

TIME_TYPES = {"continuous": "Continuous (linear)", "quadratic": "Continuous (quadratic)",
              "categorical": "Categorical (one mean per occasion)"}
RANDOM = {"intercept": "Random intercept", "slope": "Random intercept + time slope (correlated)",
          "slope_uncorrelated": "Random intercept + time slope (uncorrelated)", "none": "None (marginal model)"}
RESIDUAL = {"independent": "Independent", "ar1": "AR(1) — equally spaced occasions",
            "car1": "Continuous-time AR(1) — actual time values", "unstructured": "Unstructured (marginal, mmrm)"}
INTERACTIONS = {"none": "None", "time": "Time × each predictor", "all": "All two-way"}


class MixedInputError(ValueError):
    pass


@dataclass
class MixedSpec:
    outcome: str
    id: str
    time: str | None = None
    time_type: str = "continuous"
    predictors: list[str] = field(default_factory=list)
    interactions: str = "time"
    random: str = "intercept"
    residual: str = "independent"
    cluster: str | None = None          # higher-level grouping; participants nested within clusters
    method: str = "REML"
    conf: float = 0.95
    n_trajectories: int = 40
    seed: int = 20261121
    force_nlme: bool = False            # comparisons of residual structures fit every variant in nlme

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def engine(self) -> str:
        if self.residual == "unstructured":
            return "mmrm"
        if self.residual in ("ar1", "car1") or self.force_nlme:
            return "nlme"
        return "lme4"


def validate(ds: Dataset, s: MixedSpec) -> None:
    v = ds.var(s.outcome)
    if not (v.measure is Measure.CONTINUOUS and v.storage is Storage.NUMERIC):
        raise MixedInputError("The outcome must be continuous (binary mixed models are not available yet).")
    names = [s.outcome, s.id] + ([s.time] if s.time else []) + ([s.cluster] if s.cluster else []) + s.predictors
    if len(set(names)) != len(names):
        raise MixedInputError("Each variable can have only one role.")
    from .models import variable_problem

    frame = ds.analysis_frame(list(dict.fromkeys(names))).dropna()
    if len(frame) < 10:
        raise MixedInputError(f"Only {len(frame)} rows have no missing values on these variables.")
    for n in [s.outcome] + s.predictors:
        why = variable_problem(ds, n, frame)
        if why:
            raise MixedInputError(why)
    for n in s.predictors:
        pv = ds.var(n)
        if not (pv.is_categorical or (pv.measure is Measure.CONTINUOUS and pv.storage is Storage.NUMERIC)):
            raise MixedInputError(f"{pv.display_name} cannot be used as a predictor.")
    if s.time:
        tv = ds.var(s.time)
        if s.time_type in ("continuous", "quadratic") and tv.storage is not Storage.NUMERIC:
            raise MixedInputError("Continuous time needs a numeric time variable.")
    needs_time = s.random in ("slope", "slope_uncorrelated") or s.residual != "independent"
    if needs_time and not s.time:
        raise MixedInputError("This random-effects or residual structure needs a time variable.")
    if s.residual == "unstructured":
        if s.random != "none":
            raise MixedInputError("Unstructured covariance is a marginal model: set random effects to 'None'.")
        if s.time_type != "categorical":
            raise MixedInputError("Unstructured covariance needs categorical time (one parameter per occasion).")
        if s.cluster:
            raise MixedInputError("Clusters are not supported with the unstructured marginal model.")
    elif s.random == "none":
        raise MixedInputError("Random-effects models need at least a random intercept "
                              "(choose unstructured covariance for a marginal model).")
    if s.method not in ("REML", "ML") or s.time_type not in TIME_TYPES or s.random not in RANDOM \
            or s.residual not in RESIDUAL or s.interactions not in INTERACTIONS:
        raise MixedInputError("Invalid option.")


def _fixed_rhs(ds: Dataset, s: MixedSpec) -> str:
    a = lambda n: alias(ds.var(n).alias)  # noqa: E731
    preds = [a(p) for p in s.predictors]
    terms: list[str] = []
    if s.time:
        t = "sx_time"
        terms.append(t if s.time_type != "quadratic" else "poly(sx_time, 2, raw = TRUE)")
    tt = terms[0] if terms else None
    if s.interactions == "all" and preds:
        body = f"({' + '.join(([tt] if tt else []) + preds)})^2"
        return body
    terms += preds
    if s.interactions == "time" and tt:
        terms += [f"{tt}:{p}" for p in preds]
    return " + ".join(terms) if terms else "1"


def build(ds: Dataset, s: MixedSpec, plot_paths: list[str]) -> str:
    validate(ds, s)
    a = lambda n: alias(ds.var(n).alias)  # noqa: E731
    y, idv = a(s.outcome), a(s.id)
    used = [y, idv] + ([a(s.time)] if s.time else []) + ([a(s.cluster)] if s.cluster else []) + [a(p) for p in s.predictors]
    conf = r_num(s.conf)
    reml = "TRUE" if s.method == "REML" else "FALSE"
    L = [
        f"sx_used <- {r_vec(used)}",
        "sx_d <- sx_complete(dat, sx_used)",
        f"sx_d$sx_id <- factor(sx_d${idv})",
    ]
    if s.cluster:
        L.append(f"sx_d$sx_cluster <- factor(sx_d${a(s.cluster)})")
    if s.time:
        t = a(s.time)
        tv = ds.var(s.time)
        if tv.is_categorical and tv.storage is Storage.NUMERIC:
            # labelled factor in R: recover the numeric codes (e.g. "Wave 3" -> 3) for continuous time
            levels = ds.levels(s.time)
            L.append(f"sx_codes <- stats::setNames({r_vec(levels)}, {r_vec(level_labels(ds, s.time, levels))})")
            num = f"unname(sx_codes[as.character(sx_d${t})])"
        elif tv.is_categorical:
            num = f"as.numeric(sx_d${t})  # string categories: occasion order"
        else:
            num = f"sx_d${t}"
        L.append(f"sx_d$sx_tnum <- {num}")
        if s.time_type == "categorical":
            L.append(f"sx_d$sx_time <- if (is.factor(sx_d${t})) droplevels(sx_d${t}) else factor(sx_d${t})")
        else:
            L.append("sx_d$sx_time <- sx_d$sx_tnum")
        L += [
            "sx_d$sx_tidx <- as.integer(factor(sx_d$sx_tnum))  # occasion order (AR(1) assumes equal spacing)",
            "sx_d <- sx_d[order(sx_d$sx_id, sx_d$sx_tnum), ]",
        ]
        if s.residual != "independent":
            L += ["if (anyDuplicated(sx_d[, c(\"sx_id\", \"sx_tnum\")])) stop(\"Some participants have more than one row "
                  "at the same time point; residual correlation structures need one row per participant per occasion.\")"]
    L += [
        "sx_counts <- table(sx_d$sx_id)",
        "if (length(sx_counts) < 3) stop(\"Fewer than 3 participants with complete data.\")",
        f"sx_fixed <- {y} ~ {_fixed_rhs(ds, s)}",
    ]
    tslope = "sx_time" if s.time_type == "continuous" else "sx_tnum"
    grp = "sx_cluster/sx_id" if s.cluster else "sx_id"
    if s.engine == "lme4":
        re = {"intercept": f"(1 | {grp})", "slope": f"(1 + {tslope} | {grp})",
              "slope_uncorrelated": f"(1 + {tslope} || {grp})"}[s.random]
        if s.cluster and s.random != "intercept":
            re = "(1 | sx_cluster) + " + re.replace(grp, "sx_id")
        L += [
            f"sx_fit <- lmerTest::lmer(stats::update(sx_fixed, . ~ . + {re}), data = sx_d, REML = {reml})",
            "sx_co <- as.data.frame(coef(summary(sx_fit)))",
            "sx_fx <- data.frame(term = rownames(sx_co), estimate = sx_co$Estimate, se = sx_co[[\"Std. Error\"]],",
            "  df = sx_co$df, t = sx_co[[\"t value\"]], p = sx_co[[\"Pr(>|t|)\"]])",
            "sx_at <- as.data.frame(stats::anova(sx_fit, type = 3))",
            "sx_terms <- data.frame(term = rownames(sx_at), F = sx_at[[\"F value\"]], df1 = sx_at$NumDF, df2 = sx_at$DenDF,",
            "  p = sx_at[[\"Pr(>F)\"]])",
            "sx_vc <- as.data.frame(lme4::VarCorr(sx_fit))",
            "sx_vc_tab <- data.frame(group = sx_vc$grp, term = ifelse(is.na(sx_vc$var1), \"(residual)\", sx_vc$var1),",
            "  with = ifelse(is.na(sx_vc$var2), \"\", sx_vc$var2), variance_or_cov = sx_vc$vcov, sd_or_corr = sx_vc$sdcor)",
            "sx_singular <- lme4::isSingular(sx_fit)",
            "sx_conv <- unlist(sx_fit@optinfo$conv$lme4$messages)",
            "sx_conv <- sx_conv[!grepl(\"singular\", sx_conv)]  # reported by the singular-fit check",
            "sx_beta <- lme4::fixef(sx_fit); sx_V <- as.matrix(stats::vcov(sx_fit))",
            "sx_df_note <- \"Satterthwaite degrees of freedom (lmerTest); Type III F-tests.\"",
            "sx_fitted <- stats::fitted(sx_fit); sx_resid <- stats::residuals(sx_fit)",
        ]
    elif s.engine == "nlme":
        rnd = {"intercept": f"~ 1 | {grp}",
               "slope": f"~ 1 + {tslope} | sx_id",
               "slope_uncorrelated": f"list(sx_id = nlme::pdDiag(~ 1 + {tslope}))"}[s.random]
        cor = {"ar1": "nlme::corAR1(form = ~ sx_tidx | sx_id)", "car1": "nlme::corCAR1(form = ~ sx_tnum | sx_id)",
               "independent": "NULL"}[s.residual]
        if s.cluster:
            cor = cor.replace("| sx_id)", "| sx_cluster/sx_id)")
        L += [
            "# nlminb (nlme default) first; optim as a fallback if it fails to converge",
            f"sx_lme <- function(opt) nlme::lme(sx_fixed, random = {rnd}, correlation = {cor}, data = sx_d,",
            f"  method = {r_str(s.method)}, control = nlme::lmeControl(opt = opt, msMaxIter = 200))",
            "sx_opt <- \"nlminb\"",
            "sx_fit <- tryCatch(sx_lme(\"nlminb\"), error = function(e) { sx_opt <<- \"optim\"; sx_lme(\"optim\") })",
            "sx_tt <- as.data.frame(summary(sx_fit)$tTable)",
            "sx_fx <- data.frame(term = rownames(sx_tt), estimate = sx_tt$Value, se = sx_tt$Std.Error, df = sx_tt$DF,",
            "  t = sx_tt[[\"t-value\"]], p = sx_tt[[\"p-value\"]])",
            "sx_at <- as.data.frame(stats::anova(sx_fit, type = \"marginal\"))",
            "sx_terms <- data.frame(term = rownames(sx_at), F = sx_at[[\"F-value\"]], df1 = sx_at$numDF, df2 = sx_at$denDF,",
            "  p = sx_at[[\"p-value\"]])",
            "sx_vcm <- nlme::VarCorr(sx_fit)",
            "sx_vc_tab <- data.frame(term = rownames(sx_vcm), variance = suppressWarnings(as.numeric(sx_vcm[, 1])),",
            "  sd = suppressWarnings(as.numeric(sx_vcm[, 2])))",
            "sx_phi <- if (is.null(sx_fit$modelStruct$corStruct)) NA_real_ else",
            "  unname(stats::coef(sx_fit$modelStruct$corStruct, unconstrained = FALSE))",
            "sx_singular <- FALSE",
            "sx_conv <- if (sx_opt != \"nlminb\") \"nlminb failed; estimated with optim (check the results)\" else character()",
            "sx_beta <- nlme::fixef(sx_fit); sx_V <- as.matrix(stats::vcov(sx_fit))",
            "sx_df_note <- \"nlme containment degrees of freedom; marginal (Type III) F-tests.\"",
            "sx_fitted <- stats::fitted(sx_fit); sx_resid <- stats::residuals(sx_fit, type = \"normalized\")",
        ]
    else:
        L += [
            f"sx_fit <- mmrm::mmrm(stats::update(sx_fixed, . ~ . + us(sx_time | sx_id)), data = sx_d, reml = {reml})",
            "sx_co <- as.data.frame(coef(summary(sx_fit)))",
            "sx_fx <- data.frame(term = rownames(sx_co), estimate = sx_co$Estimate, se = sx_co[[\"Std. Error\"]],",
            "  df = sx_co$df, t = sx_co[[\"t value\"]], p = sx_co[[\"Pr(>|t|)\"]])",
            "sx_terms <- tryCatch({ a <- as.data.frame(car::Anova(sx_fit, type = \"III\"))",
            "  data.frame(term = rownames(a), F = a[[\"F Statistic\"]], df1 = a[[\"Num.Df\"]], df2 = a[[\"Denom.Df\"]], p = a[[\"Pr(>=F)\"]]) },",
            "  error = function(e) NULL)",
            "sx_cov <- mmrm::VarCorr(sx_fit)",
            "sx_vc_tab <- data.frame(occasion = rownames(sx_cov), as.data.frame(unclass(sx_cov), check.names = FALSE), check.names = FALSE)",
            "sx_singular <- FALSE; sx_conv <- character()",
            "sx_beta <- stats::coef(sx_fit); sx_V <- as.matrix(stats::vcov(sx_fit))",
            "sx_df_note <- \"Satterthwaite degrees of freedom (mmrm); Type III F-tests.\"",
            "sx_fitted <- stats::fitted(sx_fit); sx_resid <- stats::residuals(sx_fit, type = \"normalized\")",
        ]
    L += [
        "sx_crit <- stats::qt(1 - (1 - " + conf + ") / 2, pmax(sx_fx$df, 1))",
        "sx_fx$ci_low <- sx_fx$estimate - sx_crit * sx_fx$se; sx_fx$ci_high <- sx_fx$estimate + sx_crit * sx_fx$se",
        "sx_tabs <- list(sx_table(\"Fixed effects\", sx_fx, notes = paste0(sx_df_note, \" Treatment coding; time is \",",
        f"  {r_str(TIME_TYPES[s.time_type] if s.time else 'not modelled')}, \".\")))",
        "if (!is.null(sx_terms)) sx_tabs[[length(sx_tabs) + 1]] <- sx_table(\"Fixed-effect term tests\", sx_terms)",
    ]
    if s.engine == "mmrm":
        L.append("sx_tabs[[length(sx_tabs) + 1]] <- sx_table(\"Residual covariance (unstructured)\", sx_vc_tab, "
                 "notes = \"Marginal model: no random effects. Each occasion has its own variance and every pair of "
                 "occasions its own covariance.\")")
    else:
        L.append("sx_tabs[[length(sx_tabs) + 1]] <- sx_table(\"Random effects (variance components)\", sx_vc_tab, "
                 "notes = \"Variance of participant-specific deviations; correlations between random effects where estimated.\")")
    if s.engine == "nlme" and s.residual != "independent":
        L.append("sx_tabs[[length(sx_tabs) + 1]] <- sx_table(\"Residual autocorrelation\", data.frame(structure = "
                 + r_str(RESIDUAL[s.residual]) + ", phi = sx_phi), notes = \"phi = correlation between residuals one "
                 "occasion (AR1) or one time unit (CAR1) apart, after random effects. Random effects and residual "
                 "autocorrelation are different things.\")")
    # model fit + R2
    L += [
        "sx_r2 <- if (" + ("FALSE" if s.engine == "mmrm" else "TRUE") + ") tryCatch(suppressWarnings(performance::r2_nakagawa(sx_fit)),",
        "  error = function(e) NULL) else NULL",
        "sx_fit_tab <- data.frame(observations = nrow(sx_d), participants = nlevels(sx_d$sx_id),",
        "  obs_per_participant_min = min(sx_counts[sx_counts > 0]), obs_per_participant_max = max(sx_counts),",
        f"  method = {r_str(s.method)}, logLik = as.numeric(stats::logLik(sx_fit)), AIC = stats::AIC(sx_fit), BIC = stats::BIC(sx_fit),",
        "  R2_marginal = if (is.null(sx_r2)) NA_real_ else as.numeric(sx_r2$R2_marginal),",
        "  R2_conditional = if (is.null(sx_r2)) NA_real_ else as.numeric(sx_r2$R2_conditional),",
        "  singular = sx_singular)",
        "sx_tabs[[length(sx_tabs) + 1]] <- sx_table(\"Model fit\", sx_fit_tab, notes = paste0(\"Marginal R2: variance explained \",",
        "  \"by fixed effects; conditional R2: by fixed + random effects (Nakagawa). AIC/BIC from \", "
        + r_str(s.method) + ", \" fits; compare fixed effects only with ML fits.\"))",
        "sx_checks <- character()",
        "if (isTRUE(sx_singular)) sx_checks <- c(sx_checks, paste0(\"Singular fit: a random-effect variance is (near) \",",
        "  \"zero or a correlation is +/-1. Simplify the random-effects structure.\"))",
        "if (length(sx_conv)) sx_checks <- c(sx_checks, paste(\"Convergence:\", sx_conv))",
        "# residual autocorrelation within participants (lags 1-3)",
        "sx_rd <- data.frame(id = sx_d$sx_id, r = as.numeric(sx_resid))",
        "sx_lagcor <- vapply(1:3, function(k) { x <- unlist(lapply(split(sx_rd$r, sx_rd$id), function(v) if (length(v) > k) head(v, -k) else NULL))",
        "  y <- unlist(lapply(split(sx_rd$r, sx_rd$id), function(v) if (length(v) > k) tail(v, -k) else NULL))",
        "  if (length(x) > 3) stats::cor(x, y) else NA_real_ }, 0)",
        "sx_tabs[[length(sx_tabs) + 1]] <- sx_table(\"Residual autocorrelation by lag\", data.frame(lag = 1:3, correlation = sx_lagcor),",
        "  notes = paste0(\"Correlation of within-participant residuals k occasions apart (normalized residuals for \",",
        "  \"nlme/mmrm). Large values suggest an AR(1) residual structure.\"))",
    ]
    L += _trajectory(ds, s, plot_paths[:3] if s.time else [])
    L += _extra_plots(s, plot_paths[3:])
    L.append("sx_result <- list(tables = sx_tabs, checks = sx_checks, n_used = nrow(sx_d), n_excluded = nrow(dat) - nrow(sx_d),")
    L.append("  notes = c(\"Mixed and marginal models use every available observation: participants with missing \",")
    L.append("    \"occasions still contribute. Estimates are unbiased if data are missing at random given the model (MAR).\"))")
    return "\n".join(L) + "\n"


def _extra_plots(s: MixedSpec, paths: list[str]) -> list[str]:
    """Random-effect (caterpillar) plot and residual autocorrelation plot (§15.5, §16.1)."""
    L: list[str] = []
    if len(paths) >= 1 and s.engine != "mmrm":
        L += [
            "# random intercepts with approximate 95% intervals (lme4: conditional SDs)",
            "sx_re <- if (inherits(sx_fit, \"merMod\")) {",
            "  r <- lme4::ranef(sx_fit, condVar = TRUE)$sx_id; sdv <- sqrt(attr(r, \"postVar\")[1, 1, ])",
            "  data.frame(id = rownames(r), value = r[[1]], lo = r[[1]] - 1.96 * sdv, hi = r[[1]] + 1.96 * sdv)",
            "} else { r <- nlme::ranef(sx_fit); r <- if (is.list(r) && !is.data.frame(r)) r[[length(r)]] else r",
            "  data.frame(id = rownames(r), value = r[[1]], lo = NA_real_, hi = NA_real_) }",
            "sx_re <- sx_re[order(sx_re$value), ]; sx_re$rank <- seq_len(nrow(sx_re))",
            "sx_p4 <- ggplot2::ggplot(sx_re, ggplot2::aes(x = rank, y = value)) +",
            "  ggplot2::geom_hline(yintercept = 0, linetype = 2) +",
            "  (if (all(is.na(sx_re$lo))) NULL else ggplot2::geom_errorbar(ggplot2::aes(ymin = lo, ymax = hi), colour = \"grey70\", width = 0)) +",
            "  ggplot2::geom_point(size = 0.8, colour = \"#2166ac\") +",
            "  ggplot2::labs(title = \"Random intercepts (participant deviations from the average)\",",
            "    subtitle = \"Sorted; bars = approximate 95% intervals\", x = \"Participants (sorted)\", y = \"Deviation\") +",
            "  ggplot2::theme_minimal()",
            f"ggplot2::ggsave({r_str(paths[0])}, sx_p4, width = 7, height = 4.5, dpi = 120)",
        ]
    if len(paths) >= 2:
        L += [
            "sx_acf <- data.frame(lag = 1:3, correlation = sx_lagcor); sx_acf <- sx_acf[!is.na(sx_acf$correlation), ]",
            "sx_p5 <- ggplot2::ggplot(sx_acf, ggplot2::aes(x = factor(lag), y = correlation)) +",
            "  ggplot2::geom_col(fill = \"#b2182b\", width = 0.5) + ggplot2::geom_hline(yintercept = 0) +",
            "  ggplot2::coord_cartesian(ylim = c(-1, 1)) +",
            "  ggplot2::labs(title = \"Residual autocorrelation within participants\", x = \"Lag (occasions apart)\",",
            "    y = \"Correlation\") + ggplot2::theme_minimal()",
            f"ggplot2::ggsave({r_str(paths[1])}, sx_p5, width = 6, height = 4, dpi = 120)",
        ]
    return L


def _trajectory(ds: Dataset, s: MixedSpec, paths: list[str]) -> list[str]:
    """Population-level trajectory (fixed effects) with CI, observed individual trajectories,
    and subject-specific (conditional) predictions for a few participants."""
    if not s.time or not paths:
        return []
    a = lambda n: alias(ds.var(n).alias)  # noqa: E731
    y = a(s.outcome)
    group = next((p for p in s.predictors if ds.var(p).is_categorical), None)
    g = a(group) if group else None
    others = [a(p) for p in s.predictors if p != group]
    cont = s.time_type != "categorical"
    L = [
        "# ---- trajectories ----",
        "sx_tv <- if (is.factor(sx_d$sx_time)) factor(levels(sx_d$sx_time), levels = levels(sx_d$sx_time)) else",
        "  seq(min(sx_d$sx_time), max(sx_d$sx_time), length.out = 50)",
        "sx_grid <- list(sx_time = sx_tv)",
    ]
    if g:
        L.append(f"sx_grid${g} <- factor(levels(sx_d${g}), levels = levels(sx_d${g}))")
    L += [
        f"sx_held <- lapply(sx_d[, {r_vec(others) if others else 'character(0)'}, drop = FALSE],",
        "  function(v) if (is.factor(v)) factor(levels(v)[1], levels = levels(v)) else mean(v))",
        "sx_nd <- do.call(expand.grid, c(sx_grid, sx_held, list(KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)))",
        "sx_nd$sx_tnum <- as.numeric(sx_nd$sx_time)",
        "sx_tt <- stats::delete.response(stats::terms(sx_fixed))",
        "sx_xlev <- lapply(Filter(is.factor, sx_d[, intersect(names(sx_d), all.vars(sx_tt)), drop = FALSE]), levels)",
        "sx_X <- stats::model.matrix(sx_tt, stats::model.frame(sx_tt, sx_nd, xlev = sx_xlev))",
        "sx_X <- sx_X[, names(sx_beta), drop = FALSE]",
        "sx_nd$predicted <- as.vector(sx_X %*% sx_beta)",
        "sx_se <- sqrt(rowSums((sx_X %*% sx_V) * sx_X))",
        f"sx_z <- stats::qnorm(1 - (1 - {r_num(s.conf)}) / 2)",
        "sx_nd$ci_low <- sx_nd$predicted - sx_z * sx_se; sx_nd$ci_high <- sx_nd$predicted + sx_z * sx_se",
        "sx_held_txt <- if (length(sx_held)) paste(sprintf(\"%s = %s\", sx_names[names(sx_held)],",
        "  vapply(sx_held, function(v) if (is.factor(v)) as.character(v) else format(signif(v, 4)), \"\")), collapse = \"; \") else \"\"",
        "sx_tabs[[length(sx_tabs) + 1]] <- sx_table(\"Population-level trajectory\", sx_nd[, setdiff(names(sx_nd), \"sx_tnum\"), drop = FALSE],",
        "  notes = paste0(\"Fixed-effects (marginal, population-average) predictions with normal-approximation CI\",",
        "    if (nzchar(sx_held_txt)) paste0(\"; other predictors held at: \", sx_held_txt) else \"\", \".\"))",
        f"set.seed({int(s.seed)})",
        f"sx_ids <- sample(levels(droplevels(sx_d$sx_id)), min({int(s.n_trajectories)}, nlevels(droplevels(sx_d$sx_id))))",
        "sx_obs <- sx_d[sx_d$sx_id %in% sx_ids, ]",
        f"sx_p <- ggplot2::ggplot() + ggplot2::geom_line(data = sx_obs, ggplot2::aes(x = sx_time, y = {y}, group = sx_id),",
        "  colour = \"grey60\", alpha = 0.45)",
    ]
    fill_g = f", fill = {g}, group = {g}" if g else ", group = 1"
    col_g = f", colour = {g}, group = {g}" if g else ", group = 1"
    if cont:
        L += [f"sx_p <- sx_p + ggplot2::geom_ribbon(data = sx_nd, ggplot2::aes(x = sx_time, ymin = ci_low, ymax = ci_high{fill_g}), alpha = 0.2, colour = NA) +",
              f"  ggplot2::geom_line(data = sx_nd, ggplot2::aes(x = sx_time, y = predicted{col_g}), linewidth = 1.3)"]
    else:
        L += [f"sx_p <- sx_p + ggplot2::geom_errorbar(data = sx_nd, ggplot2::aes(x = sx_time, ymin = ci_low, ymax = ci_high{col_g}), width = 0.15) +",
              f"  ggplot2::geom_line(data = sx_nd, ggplot2::aes(x = sx_time, y = predicted{col_g}), linewidth = 1.2) +",
              f"  ggplot2::geom_point(data = sx_nd, ggplot2::aes(x = sx_time, y = predicted{col_g}), size = 2.5)"]
    L += [
        "sx_p <- sx_p + ggplot2::labs(title = \"Population-level trajectory (fixed effects)\",",
        f"  subtitle = paste0(\"Grey: observed trajectories of \", length(sx_ids), \" randomly sampled participants (seed {int(s.seed)})\"),",
        f"  x = sx_names[[\"{a(s.time)}\"]], y = sx_names[[\"{y}\"]]"
        + (f", colour = sx_names[[\"{g}\"]], fill = sx_names[[\"{g}\"]]" if g else "") + ") +",
        "  ggplot2::theme_minimal(base_size = 12) + ggplot2::theme(legend.position = \"bottom\")",
        f"ggplot2::ggsave({r_str(paths[0])}, sx_p, width = 7.5, height = 5.2, dpi = 120)",
    ]
    if len(paths) > 1:
        L += [
            "# conditional (participant-specific) predictions for 6 participants",
            "sx_six <- head(sx_ids, 6)",
            "sx_cd <- data.frame(sx_id = sx_d$sx_id, sx_time = sx_d$sx_time, observed = sx_d[[sx_used[1]]], conditional = as.numeric(sx_fitted))",
            "sx_cd <- sx_cd[sx_cd$sx_id %in% sx_six, ]",
            "sx_p2 <- ggplot2::ggplot(sx_cd, ggplot2::aes(x = sx_time, group = sx_id)) +",
            "  ggplot2::geom_point(ggplot2::aes(y = observed), colour = \"grey30\") +",
            "  ggplot2::geom_line(ggplot2::aes(y = conditional), colour = \"#b2182b\", linewidth = 1) +",
            "  ggplot2::facet_wrap(~ sx_id) +",
            "  ggplot2::labs(title = \"Conditional (participant-specific) predictions\",",
            "    subtitle = " + r_str("Red: fitted values including each participant's random effects" if s.engine != "mmrm"
                                     else "Red: marginal fitted values (mmrm has no random effects)") + ",",
            f"    x = sx_names[[\"{a(s.time)}\"]], y = sx_names[[\"{y}\"]]) + ggplot2::theme_minimal(base_size = 11)",
            f"ggplot2::ggsave({r_str(paths[1])}, sx_p2, width = 7.5, height = 5.2, dpi = 120)",
        ]
    if len(paths) > 2:
        L += [
            "sx_p3 <- ggplot2::ggplot(data.frame(fitted = as.numeric(sx_fitted), residual = as.numeric(sx_resid)),",
            "  ggplot2::aes(fitted, residual)) + ggplot2::geom_point(alpha = 0.3) + ggplot2::geom_hline(yintercept = 0, linetype = 2) +",
            "  (if (length(unique(round(as.numeric(sx_fitted), 8))) > 20) ggplot2::geom_smooth(method = \"loess\", formula = y ~ x,",
            "    se = FALSE, colour = \"#b2182b\") else NULL) +",
            "  ggplot2::labs(title = \"Residuals vs fitted\") + ggplot2::theme_minimal()",
            f"ggplot2::ggsave({r_str(paths[2])}, sx_p3, width = 6.5, height = 4.5, dpi = 120)",
        ]
    return L


def _internal_names(ds: Dataset, s: MixedSpec) -> dict[str, str]:
    out = {"sx_id": ds.var(s.id).display_name}
    if s.time:
        out["sx_time"] = ds.var(s.time).display_name
        out["sx_tnum"] = ds.var(s.time).display_name
    if s.cluster:
        out["sx_cluster"] = ds.var(s.cluster).display_name
    return out


def run(session: RSession, ds: Dataset, s: MixedSpec) -> tuple[AnalysisResult, dict[str, bytes]]:
    from .framework import _alias_map, _as_list, _parse_table

    paths = [session.new_plot_path() for _ in range(5)]  # 3 trajectory/diagnostic + random effects + ACF
    body = build(ds, s, [p.as_posix() for p in paths])
    pk = {"lme4": ["lme4", "lmerTest", "performance"], "nlme": ["nlme", "performance"], "mmrm": ["mmrm", "car"]}[s.engine]
    title = {"lme4": "Linear mixed model", "nlme": "Linear mixed model with residual autocorrelation",
             "mmrm": "Marginal repeated-measures model (unstructured)"}[s.engine]
    result, raw = session.run(ds, kind="mixed_model", title=title, body=body, packages=pk + ["ggplot2"],
                              inputs={"spec": s.to_dict(), "filters": [f.describe() for f in ds.filters]})
    result.missing_data = "All available observations (complete on the model variables within each row); MAR assumed."
    result.seed = s.seed
    artifacts: dict[str, bytes] = {}
    if raw is None:
        return result, artifacts
    names = _alias_map(ds)
    extra = _internal_names(ds, s)
    for t in _as_list(raw.get("tables")):
        result.tables.append(_parse_table(t, names, extra))
    result.n_used = raw.get("n_used")
    result.values["n_excluded"] = raw.get("n_excluded")
    result.values["notes"] = ["".join(str(x) for x in _as_list(raw.get("notes")))]
    result.values["engine_choice"] = s.engine
    checks = [str(c) for c in _as_list(raw.get("checks"))]
    # an R warning that a check already reports (e.g. lme4 convergence) is shown once
    result.messages = [m for m in result.messages if not any(m.removeprefix("R warning: ") in c for c in checks)]
    for c in checks:
        result.messages.append(f"Check: {c}")
        if result.status is Status.OK:
            result.status = Status.WARNING
    for p in paths:
        if p.is_file():
            key = f"plots/{p.name}"
            artifacts[key] = p.read_bytes()
            result.plots.append(key)
    return result, artifacts


# ---- specification comparisons (§15.3, §14) -------------------------------------------------

COMPARISONS = {
    "time": "Time specifications (linear / quadratic / categorical) — ML fits",
    "random": "Random effects (intercept vs intercept + slope) — REML fits",
    "residual": "Residual structure (independent / AR(1) / CAR(1)) — REML fits in nlme",
}


def comparison_variants(base: MixedSpec, which: str) -> list[tuple[str, MixedSpec]]:
    from dataclasses import replace

    if which == "time":
        if base.residual == "unstructured":
            raise MixedInputError("Unstructured models need categorical time; time specifications cannot be compared.")
        return [(TIME_TYPES[t], replace(base, time_type=t, method="ML"))
                for t in ("continuous", "quadratic", "categorical")]
    if which == "random":
        if base.residual == "unstructured":
            raise MixedInputError("The unstructured marginal model has no random effects.")
        return [(RANDOM[r], replace(base, random=r, method="REML")) for r in ("intercept", "slope")]
    if which == "residual":
        if base.random == "none":
            raise MixedInputError("Choose a random-effects model to compare residual structures.")
        return [(RESIDUAL[r], replace(base, residual=r, method="REML", force_nlme=True))
                for r in ("independent", "ar1", "car1")]
    raise MixedInputError(which)


# (smaller, larger, boundary) nested pairs by position in comparison_variants
NESTED = {"time": [(0, 1, False), (0, 2, False), (1, 2, False)], "random": [(0, 1, True)],
          "residual": [(0, 1, False), (0, 2, False)]}


def compare_specs(session: RSession, ds: Dataset, base: MixedSpec, which: str) -> tuple[AnalysisResult, dict]:
    from .framework import _alias_map, _as_list, _parse_table

    variants = comparison_variants(base, which)
    for _, v in variants:
        validate(ds, v)
    L = ["sx_cmp <- list()"]
    for k, (label, v) in enumerate(variants, start=1):
        L.append(f"# ---- variant {k}: {label} ----")
        L.append("sx_cmp[[" + str(k) + "]] <- tryCatch({")
        L.append(build(ds, v, []).replace("sx_result <- list(", "sx_unused <- list("))
        L.append(f"  list(label = {r_str(label)}, engine = {r_str(v.engine)}, method = {r_str(v.method)},")
        L.append("    logLik = as.numeric(stats::logLik(sx_fit)), df = attr(stats::logLik(sx_fit), \"df\"),")
        L.append("    AIC = stats::AIC(sx_fit), BIC = stats::BIC(sx_fit), n = nrow(sx_d), singular = isTRUE(sx_singular))")
        L.append(f"}}, error = function(e) list(label = {r_str(label)}, error = conditionMessage(e)))")
    pairs = [(a, b, bd) for a, b, bd in NESTED[which] if a < len(variants) and b < len(variants)]
    L += [
        "sx_ok <- vapply(sx_cmp, function(x) is.null(x$error), TRUE)",
        "sx_tab <- do.call(rbind, lapply(sx_cmp[sx_ok], function(x) data.frame(model = x$label, engine = x$engine,",
        "  estimation = x$method, parameters = x$df, logLik = x$logLik, AIC = x$AIC, BIC = x$BIC, n = x$n, singular = x$singular)))",
        "sx_tab$delta_AIC <- sx_tab$AIC - min(sx_tab$AIC)",
        "sx_tabs <- list(sx_table(\"Specification comparison\", sx_tab, notes = " + r_str(COMPARISONS[which]) + "))",
        "sx_lrt <- list()",
    ]
    for a, b, boundary in pairs:
        L += [
            f"if (sx_ok[{a + 1}] && sx_ok[{b + 1}] && sx_cmp[[{a + 1}]]$n == sx_cmp[[{b + 1}]]$n && sx_cmp[[{b + 1}]]$df > sx_cmp[[{a + 1}]]$df) {{",
            f"  sx_chi <- 2 * (sx_cmp[[{b + 1}]]$logLik - sx_cmp[[{a + 1}]]$logLik); sx_df <- sx_cmp[[{b + 1}]]$df - sx_cmp[[{a + 1}]]$df",
            "  sx_p <- stats::pchisq(sx_chi, sx_df, lower.tail = FALSE)",
            "  sx_lrt[[length(sx_lrt) + 1]] <- data.frame(smaller = sx_cmp[[" + str(a + 1) + "]]$label, larger = sx_cmp[[" + str(b + 1) + "]]$label,",
            "    chi2 = sx_chi, df = sx_df, p = sx_p, p_boundary_corrected = "
            + ("0.5 * sx_p + 0.5 * stats::pchisq(sx_chi, sx_df - 1, lower.tail = FALSE)" if boundary else "NA_real_") + ")",
            "}",
        ]
    note = ("Likelihood-ratio tests for nested pairs on the same rows. " +
            ("Variance tests sit on the boundary of the parameter space: the naive p is conservative; "
             "p_boundary_corrected uses the 50:50 chi-square mixture." if which == "random" else
             "Fixed-effect comparisons use ML fits (REML likelihoods are not comparable across fixed effects)."
             if which == "time" else "AR(1) and CAR(1) are not nested in each other: compare them by AIC."))
    L += [
        "if (length(sx_lrt)) sx_tabs[[2]] <- sx_table(\"Likelihood-ratio tests\", do.call(rbind, sx_lrt), notes = " + r_str(note) + ")",
        "sx_errors <- vapply(sx_cmp[!sx_ok], function(x) paste0(x$label, \": \", x$error), \"\")",
        "sx_result <- list(tables = sx_tabs, checks = sx_errors, n_used = max(sx_tab$n), n_excluded = nrow(dat) - max(sx_tab$n))",
    ]
    body = "\n".join(L) + "\n"
    result, raw = session.run(ds, kind="mixed_comparison", title=f"Compare {which} specifications", body=body,
                              packages=["lme4", "lmerTest", "nlme", "mmrm"], inputs={"base": base.to_dict(), "which": which})
    if raw is None:
        return result, {}
    names = _alias_map(ds)
    for t in _as_list(raw.get("tables")):
        result.tables.append(_parse_table(t, names, _internal_names(ds, base)))
    result.n_used = raw.get("n_used")
    for c in _as_list(raw.get("checks")):
        result.messages.append(f"Check: could not fit {c}")
        result.status = Status.WARNING
    return result, {}
