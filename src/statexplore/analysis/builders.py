"""Phase 2 analyses: R code builders and their specs.

Conventions for every body:
  - `sx_d` is the complete-case analysis data (see sx_complete in codegen.R_HELPERS)
  - the body assigns `sx_result <- list(tables = list(sx_table(...)), n_used, n_excluded,
    notes, checks)`; `checks` holds data problems that warrant a warning status.
  - effect-size CIs are two-sided at the chosen confidence level.
"""
from __future__ import annotations

from ..rbridge.codegen import r_str, r_vec
from .framework import (
    ADJUST, ALTERNATIVE, CONF, INTERACTIONS, AnalysisInputError, AnalysisSpec, Ctx, Option, Slot,
)

COMMON_TAIL = "n_used = nrow(sx_d), n_excluded = nrow(dat) - nrow(sx_d)"


def _prologue(ctx: Ctx, *slots: str) -> list[str]:
    used = ctx.used(*slots)
    extra = [a for a in ctx.opts.get("complete_on", []) if a not in used]
    if extra:  # same rows as a discovery search's shared complete-case sample
        sample = [f"sx_d <- droplevels(sx_complete(dat, c(sx_used, {r_vec(extra)}))[, sx_used, drop = FALSE])"]
    else:
        sample = ["sx_d <- sx_complete(dat, sx_used)"]
    return [
        f"sx_used <- {r_vec(used)}",
        *sample,
        "if (nrow(sx_d) < 3) stop(\"Fewer than 3 complete cases for these variables.\")",
    ]


def _require_levels(var: str, n: int | None = None, at_least: int = 2) -> str:
    if n is not None:
        return (f"if (nlevels(sx_d${var}) != {n}) stop(sx_names[[\"{var}\"]], \" must have exactly {n} "
                f"groups in the analysed data; it has \", nlevels(sx_d${var}), \".\")")
    return (f"if (nlevels(sx_d${var}) < {at_least}) stop(sx_names[[\"{var}\"]], \" needs at least {at_least} "
            f"groups with data.\")")


# ============================================================================ t-tests

def _one_sample(ctx: Ctx) -> str:
    x = ctx.one("dependent")
    L = _prologue(ctx, "dependent") + [
        f"sx_x <- sx_d${x}",
        f"sx_tt <- stats::t.test(sx_x, mu = {ctx.num('mu')}, alternative = {ctx.str_('alternative')}, "
        f"conf.level = {ctx.num('conf')})",
        f"sx_es <- effectsize::cohens_d(sx_x, mu = {ctx.num('mu')}, ci = {ctx.num('conf')})",
        f"sx_g <- effectsize::hedges_g(sx_x, mu = {ctx.num('mu')}, ci = {ctx.num('conf')})",
        "sx_sw <- sx_shapiro(sx_x)",
        "sx_result <- list(tables = list(",
        "  sx_table(\"One-sample t-test\", data.frame(n = length(sx_x), mean = mean(sx_x), sd = stats::sd(sx_x),",
        f"    test_value = {ctx.num('mu')}, t = unname(sx_tt$statistic), df = unname(sx_tt$parameter),",
        "    p = sx_tt$p.value, ci_low = sx_tt$conf.int[1], ci_high = sx_tt$conf.int[2]),",
        "    notes = paste0(\"CI is for the mean (\", sx_tt$alternative, \").\")),",
        "  sx_table(\"Effect size\", data.frame(measure = c(\"Cohen's d\", \"Hedges' g\"),",
        "    estimate = c(sx_es$Cohens_d, sx_g$Hedges_g), ci_low = c(sx_es$CI_low, sx_g$CI_low),",
        "    ci_high = c(sx_es$CI_high, sx_g$CI_high)), notes = \"d = (mean - test value) / SD; two-sided CI.\"),",
        "  sx_table(\"Assumption check\", data.frame(check = \"Shapiro-Wilk normality\", W = sx_sw[[\"W\"]],",
        "    p = sx_sw[[\"p\"]]), notes = \"Reported for 3 <= n <= 5000. With large n, small deviations are 'significant'.\")),",
        f"  {COMMON_TAIL})",
    ]
    return "\n".join(L) + "\n"


def _independent(ctx: Ctx, welch: bool) -> str:
    y, g = ctx.one("dependent"), ctx.one("group")
    eq = "FALSE" if welch else "TRUE"
    L = _prologue(ctx, "dependent", "group") + [
        _require_levels(g, 2),
        f"sx_tt <- stats::t.test({y} ~ {g}, data = sx_d, var.equal = {eq}, alternative = {ctx.str_('alternative')}, "
        f"conf.level = {ctx.num('conf')})",
        f"sx_es <- effectsize::cohens_d({y} ~ {g}, data = sx_d, pooled_sd = {eq}, ci = {ctx.num('conf')})",
        f"sx_g <- effectsize::hedges_g({y} ~ {g}, data = sx_d, pooled_sd = {eq}, ci = {ctx.num('conf')})",
        f"sx_lev <- car::leveneTest({y} ~ {g}, data = sx_d, center = stats::median)",
        f"sx_lv <- levels(sx_d${g})",
        f"sx_desc <- do.call(rbind, lapply(sx_lv, function(l) {{ x <- sx_d${y}[sx_d${g} == l]; sw <- sx_shapiro(x)",
        "  data.frame(group = l, n = length(x), mean = mean(x), sd = stats::sd(x), shapiro_W = sw[[\"W\"]], shapiro_p = sw[[\"p\"]]) }))",
        "sx_result <- list(tables = list(",
        f"  sx_table({r_str('Welch t-test' if welch else 'Independent-samples t-test (Student)')}, data.frame(",
        "    difference = unname(sx_tt$estimate[1] - sx_tt$estimate[2]), t = unname(sx_tt$statistic),",
        "    df = unname(sx_tt$parameter), p = sx_tt$p.value, ci_low = sx_tt$conf.int[1], ci_high = sx_tt$conf.int[2]),",
        "    notes = paste0(\"Difference = \", sx_lv[1], \" minus \", sx_lv[2], \"; \", "
        + ("\"Welch: variances not assumed equal.\"" if welch else "\"Student: equal variances assumed.\"") + ")),",
        "  sx_table(\"Group descriptives\", sx_desc),",
        "  sx_table(\"Effect size\", data.frame(measure = c(\"Cohen's d\", \"Hedges' g\"),",
        "    estimate = c(sx_es$Cohens_d, sx_g$Hedges_g), ci_low = c(sx_es$CI_low, sx_g$CI_low),",
        "    ci_high = c(sx_es$CI_high, sx_g$CI_high)), notes = "
        + ("\"Standardized by the unpooled (average) SD.\"" if welch else "\"Standardized by the pooled SD.\"") + "),",
        "  sx_table(\"Levene's test (median-centred)\", data.frame(F = sx_lev[1, \"F value\"], df1 = sx_lev[1, \"Df\"],",
        "    df2 = sx_lev[2, \"Df\"], p = sx_lev[1, \"Pr(>F)\"]))),",
        f"  {COMMON_TAIL})",
    ]
    return "\n".join(L) + "\n"


def _paired(ctx: Ctx) -> str:
    a, b = ctx.one("first"), ctx.one("second")
    L = _prologue(ctx, "first", "second") + [
        f"sx_tt <- stats::t.test(sx_d${a}, sx_d${b}, paired = TRUE, alternative = {ctx.str_('alternative')}, "
        f"conf.level = {ctx.num('conf')})",
        f"sx_dz <- effectsize::repeated_measures_d(sx_d${a}, sx_d${b}, method = \"z\", adjust = FALSE, ci = {ctx.num('conf')})",
        f"sx_diff <- sx_d${a} - sx_d${b}",
        "sx_sw <- sx_shapiro(sx_diff)",
        "sx_result <- list(tables = list(",
        "  sx_table(\"Paired-samples t-test\", data.frame(n_pairs = length(sx_diff), mean_difference = mean(sx_diff),",
        "    sd_difference = stats::sd(sx_diff), t = unname(sx_tt$statistic), df = unname(sx_tt$parameter),",
        "    p = sx_tt$p.value, ci_low = sx_tt$conf.int[1], ci_high = sx_tt$conf.int[2]),",
        f"    notes = paste0(\"Difference = \", sx_names[[\"{a}\"]], \" minus \", sx_names[[\"{b}\"]], \".\")),",
        "  sx_table(\"Effect size\", data.frame(measure = \"Cohen's d_z\", estimate = sx_dz$d_z, ci_low = sx_dz$CI_low,",
        "    ci_high = sx_dz$CI_high), notes = \"d_z = mean difference / SD of differences (no small-sample correction).\"),",
        "  sx_table(\"Assumption check\", data.frame(check = \"Shapiro-Wilk on differences\", W = sx_sw[[\"W\"]], p = sx_sw[[\"p\"]]))),",
        f"  {COMMON_TAIL})",
    ]
    return "\n".join(L) + "\n"


# ======================================================================= correlations

METHOD = Option("method", "Method", "choice", "pearson", [("pearson", "Pearson"), ("spearman", "Spearman")])


def _cor_pair(ctx: Ctx) -> str:
    x, y = ctx.one("x"), ctx.one("y")
    L = _prologue(ctx, "x", "y") + [
        f"sx_exact <- if ({ctx.str_('method')} == \"spearman\") !(anyDuplicated(sx_d${x}) || anyDuplicated(sx_d${y})) else NULL",
        f"sx_ct <- stats::cor.test(sx_d${x}, sx_d${y}, method = {ctx.str_('method')}, conf.level = {ctx.num('conf')}, exact = sx_exact)",
        "sx_ci <- if (is.null(sx_ct$conf.int)) c(NA_real_, NA_real_) else sx_ct$conf.int",
        "sx_result <- list(tables = list(",
        f"  sx_table(paste(if ({ctx.str_('method')} == \"pearson\") \"Pearson\" else \"Spearman\", \"correlation\"), data.frame(",
        "    n = nrow(sx_d), r = unname(sx_ct$estimate), statistic = unname(sx_ct$statistic),",
        "    df = if (is.null(sx_ct$parameter)) NA_real_ else unname(sx_ct$parameter), p = sx_ct$p.value,",
        "    ci_low = sx_ci[1], ci_high = sx_ci[2]),",
        "    notes = \"Pearson CI uses Fisher's z; no CI is reported for Spearman's rho.\")),",
        f"  {COMMON_TAIL})",
    ]
    return "\n".join(L) + "\n"


def _cor_partial(ctx: Ctx) -> str:
    x, y = ctx.one("x"), ctx.one("y")
    z = ctx.aliases("controls")
    L = _prologue(ctx, "x", "y", "controls") + [
        f"sx_pc <- ppcor::pcor.test(sx_d${x}, sx_d${y}, sx_d[, {r_vec(z)}, drop = FALSE], method = {ctx.str_('method')})",
        f"sx_k <- {len(z)}",
        "sx_se <- 1 / sqrt(nrow(sx_d) - 3 - sx_k)",
        f"sx_zc <- stats::qnorm(1 - (1 - {ctx.num('conf')}) / 2)",
        f"sx_ci <- if ({ctx.str_('method')} == \"pearson\") tanh(atanh(sx_pc$estimate) + c(-1, 1) * sx_zc * sx_se) else c(NA_real_, NA_real_)",
        "sx_result <- list(tables = list(",
        "  sx_table(\"Partial correlation\", data.frame(n = nrow(sx_d), controls = sx_k, r_partial = sx_pc$estimate,",
        "    statistic = sx_pc$statistic, p = sx_pc$p.value, ci_low = sx_ci[1], ci_high = sx_ci[2]),",
        f"    notes = paste0(\"Controlling for: \", paste(sx_names[{r_vec(z)}], collapse = \", \"),",
        "      \". Pearson CI uses Fisher's z with n - 3 - k.\"))),",
        f"  {COMMON_TAIL})",
    ]
    return "\n".join(L) + "\n"


def _cor_matrix(ctx: Ctx) -> str:
    vs = ctx.aliases("variables")
    L = [
        f"sx_used <- {r_vec(vs)}",
        "sx_d <- dat[, sx_used, drop = FALSE]  # pairwise deletion",
        f"sx_ct <- psych::corr.test(sx_d, use = \"pairwise\", method = {ctx.str_('method')}, adjust = {ctx.str_('adjust')})",
        "sx_pairs <- utils::combn(seq_along(sx_used), 2)",
        "sx_n <- function(i, j) if (is.matrix(sx_ct$n)) sx_ct$n[i, j] else sx_ct$n",
        "sx_tab <- do.call(rbind, lapply(seq_len(ncol(sx_pairs)), function(k) { i <- sx_pairs[1, k]; j <- sx_pairs[2, k]",
        "  r <- sx_ct$r[i, j]; n <- sx_n(i, j)",
        f"  ci <- if ({ctx.str_('method')} == \"pearson\" && n > 3) tanh(atanh(r) + c(-1, 1) * stats::qnorm(0.975) / sqrt(n - 3)) else c(NA, NA)",
        "  data.frame(variable_1 = sx_used[i], variable_2 = sx_used[j], r = r, n = n,",
        "    p_raw = sx_ct$p[j, i], p_adjusted = sx_ct$p[i, j], ci_low = ci[1], ci_high = ci[2]) }))",
        "sx_result <- list(tables = list(",
        "  sx_table(\"Correlation matrix (pairs)\", sx_tab, notes = paste0(\"psych::corr.test, pairwise deletion; p_adjusted uses \",",
        f"    {ctx.str_('adjust')}, \" across all \", nrow(sx_tab), \" pairs. 95% CI (Fisher z) for Pearson only.\"))),",
        "  n_used = sum(stats::complete.cases(sx_d)), n_excluded = 0,",
        "  notes = \"n_used = rows complete on every variable; each r uses all rows complete on that pair.\")",
    ]
    return "\n".join(L) + "\n"


# ============================================================================ ANOVA

def _cells_check(factors: list[str]) -> list[str]:
    return [
        f"sx_cells <- table(sx_d[, {r_vec(factors)}, drop = FALSE])",
        "if (any(sx_cells == 0)) stop(\"Some combinations of factor levels have no observations, so the \",",
        "  \"factorial model is not estimable. Collapse levels or remove a factor.\")",
    ]


def _emm_tables(fit: str, factors: list[str], adjust: str, conf: str, label_suffix: str = "") -> list[str]:
    L = []
    for f in factors:
        L += [
            f"sx_em <- emmeans::emmeans({fit}, ~ {f})",
            f"sx_tabs[[length(sx_tabs) + 1]] <- sx_table(paste0(\"Estimated marginal means: \", sx_names[[\"{f}\"]]{label_suffix}),",
            f"  as.data.frame(summary(sx_em, level = {conf})))",
            f"sx_pw <- summary(pairs(sx_em, adjust = {adjust}), infer = c(TRUE, TRUE), level = {conf})",
            f"sx_tabs[[length(sx_tabs) + 1]] <- sx_table(paste0(\"Pairwise comparisons: \", sx_names[[\"{f}\"]]),",
            f"  as.data.frame(sx_pw), notes = paste0(\"p-values and CIs adjusted: \", {adjust}, \".\"))",
        ]
    if len(factors) > 1:
        L += [
            f"sx_em <- emmeans::emmeans({fit}, ~ {' * '.join(factors)})",
            "sx_tabs[[length(sx_tabs) + 1]] <- sx_table(\"Estimated marginal means: all cells\",",
            f"  as.data.frame(summary(sx_em, level = {conf})))",
        ]
    return L


def _anova_table(fit: str, note: str) -> list[str]:
    return [
        f"sx_at <- as.data.frame({fit}$anova_table)",
        "sx_tabs <- list(sx_table(\"ANOVA table\", data.frame(effect = rownames(sx_at), df1 = sx_at[[\"num Df\"]],",
        "  df2 = sx_at[[\"den Df\"]], MSE = sx_at$MSE, F = sx_at$F, p = sx_at[[\"Pr(>F)\"]], partial_eta2 = sx_at$pes,",
        f"  generalized_eta2 = sx_at$ges), notes = {r_str(note)}))",
    ]


def _anova_between(ctx: Ctx) -> str:
    y = ctx.one("dependent")
    fs = ctx.aliases("factors")
    adj, conf = ctx.str_("adjust"), ctx.num("conf")
    L = _prologue(ctx, "dependent", "factors") + [_require_levels(f) for f in fs] + _cells_check(fs) + [
        "sx_d$sx_id <- factor(seq_len(nrow(sx_d)))",
        f"sx_fit <- afex::aov_car({y} ~ {' * '.join(fs)} + Error(sx_id), data = sx_d, type = 3,",
        "  anova_table = list(es = c(\"pes\", \"ges\")))",
    ] + _anova_table("sx_fit", "Type III sums of squares with sum-to-zero contrasts (afex).") + _emm_tables(
        "sx_fit", fs, adj, conf) + [
        f"sx_lev <- car::leveneTest({y} ~ interaction({', '.join(fs)}), data = sx_d, center = stats::median)",
        "sx_tabs[[length(sx_tabs) + 1]] <- sx_table(\"Levene's test (median-centred, across cells)\",",
        "  data.frame(F = sx_lev[1, \"F value\"], df1 = sx_lev[1, \"Df\"], df2 = sx_lev[2, \"Df\"], p = sx_lev[1, \"Pr(>F)\"]))",
        f"sx_result <- list(tables = sx_tabs, {COMMON_TAIL})",
    ]
    return "\n".join(L) + "\n"


def _anova_welch(ctx: Ctx) -> str:
    y, g = ctx.one("dependent"), ctx.one("factor")
    L = _prologue(ctx, "dependent", "factor") + [
        _require_levels(g),
        f"sx_ow <- stats::oneway.test({y} ~ {g}, data = sx_d, var.equal = FALSE)",
        f"sx_es <- suppressMessages(effectsize::effectsize(sx_ow, ci = {ctx.num('conf')}))",
        f"sx_desc <- do.call(rbind, lapply(levels(sx_d${g}), function(l) {{ x <- sx_d${y}[sx_d${g} == l]",
        "  data.frame(group = l, n = length(x), mean = mean(x), sd = stats::sd(x)) }))",
        "sx_result <- list(tables = list(",
        "  sx_table(\"Welch one-way ANOVA\", data.frame(F = unname(sx_ow$statistic), df1 = unname(sx_ow$parameter[1]),",
        "    df2 = unname(sx_ow$parameter[2]), p = sx_ow$p.value), notes = \"Does not assume equal variances.\"),",
        "  sx_table(\"Effect size\", data.frame(measure = \"eta squared (approximation)\", estimate = sx_es$Eta2,",
        "    ci_low = sx_es$CI_low, ci_high = sx_es$CI_high), notes = \"effectsize: eta^2 from Welch F is an approximation.\"),",
        "  sx_table(\"Group descriptives\", sx_desc)),",
        "  notes = \"Post hoc comparisons for Welch ANOVA (Games-Howell) are not available yet.\",",
        f"  {COMMON_TAIL})",
    ]
    return "\n".join(L) + "\n"


def _anova_rm(ctx: Ctx, mixed: bool) -> str:
    y, idv = ctx.one("dependent"), ctx.one("id")
    within = ctx.aliases("within")
    between = ctx.aliases("between") if mixed else []
    adj, conf = ctx.str_("adjust"), ctx.num("conf")
    L = _prologue(ctx, "dependent", "id", "within", "between") + [_require_levels(w) for w in within + between] + [
        f"sx_fit <- afex::aov_ez(id = \"{idv}\", dv = \"{y}\", data = sx_d, within = {r_vec(within)},"
        + (f" between = {r_vec(between)}," if between else ""),
        "  anova_table = list(es = c(\"pes\", \"ges\"), correction = \"GG\"))",
        "sx_ids_all <- length(unique(sx_d[[sx_used[2]]]))",
        "sx_ids_used <- nrow(sx_fit$data$wide)",
    ] + _anova_table("sx_fit", "Type III; within-subject effects use the Greenhouse-Geisser correction (afex default).") + [
        "sx_sum <- summary(sx_fit)",
        "if (length(sx_sum$sphericity.tests)) {",
        "  sx_sph <- as.data.frame(unclass(sx_sum$sphericity.tests)); sx_adj <- as.data.frame(unclass(sx_sum$pval.adjustments))",
        "  sx_tabs[[length(sx_tabs) + 1]] <- sx_table(\"Sphericity (Mauchly) and corrections\", data.frame(effect = rownames(sx_sph),",
        "    mauchly_W = sx_sph[[1]], mauchly_p = sx_sph[[2]], GG_epsilon = sx_adj[[1]], HF_epsilon = sx_adj[[3]]),",
        "    notes = \"Only defined for within factors with more than two levels.\")",
        "}",
    ] + _emm_tables("sx_fit", within + between, adj, conf) + [
        "sx_result <- list(tables = sx_tabs, n_used = sx_ids_used, n_excluded = sx_ids_all - sx_ids_used,",
        "  notes = c(paste0(\"n counts participants. \", sx_ids_all - sx_ids_used, \" participant(s) with a missing \",",
        "    \"within-subject cell were excluded (afex requires complete data per participant).\"),",
        "    \"Data must be in long format: one row per participant per within-level.\"))",
    ]
    return "\n".join(L) + "\n"


def _ancova(ctx: Ctx) -> str:
    y = ctx.one("dependent")
    fs, cs = ctx.aliases("factors"), ctx.aliases("covariates")
    adj, conf = ctx.str_("adjust"), ctx.num("conf")
    L = _prologue(ctx, "dependent", "factors", "covariates") + [_require_levels(f) for f in fs] + _cells_check(fs) + [
        f"for (sx_c in {r_vec(cs)}) sx_d[[sx_c]] <- sx_d[[sx_c]] - mean(sx_d[[sx_c]])  # mean-centre covariates",
        "sx_d$sx_id <- factor(seq_len(nrow(sx_d)))",
        f"sx_fit <- afex::aov_car({y} ~ {' * '.join(fs)} + {' + '.join(cs)} + Error(sx_id), data = sx_d, type = 3,",
        "  factorize = FALSE, anova_table = list(es = c(\"pes\", \"ges\")))",
    ] + _anova_table("sx_fit", "Type III, sum-to-zero contrasts; covariates mean-centred.") + _emm_tables(
        "sx_fit", fs, adj, conf, label_suffix=", \" (adjusted for covariates)\"") + [
        "# Homogeneity of regression slopes: does any factor interact with a covariate?",
        f"sx_h0 <- stats::lm({y} ~ {' * '.join(fs)} + {' + '.join(cs)}, data = sx_d)",
        f"sx_h1 <- stats::lm({y} ~ ({' * '.join(fs)}) * ({' + '.join(cs)}), data = sx_d)",
        "sx_hs <- stats::anova(sx_h0, sx_h1)",
        "sx_tabs[[length(sx_tabs) + 1]] <- sx_table(\"Homogeneity of regression slopes\", data.frame(",
        "  F = sx_hs$F[2], df1 = sx_hs$Df[2], df2 = sx_hs$Res.Df[2], p = sx_hs[[\"Pr(>F)\"]][2]),",
        "  notes = \"Tests all factor-by-covariate interactions together. A small p suggests slopes differ across groups.\")",
        f"sx_result <- list(tables = sx_tabs, {COMMON_TAIL})",
    ]
    return "\n".join(L) + "\n"


# ======================================================================= regression

def _rhs(ctx: Ctx, slot: str = "predictors") -> str:
    if ctx.opts.get("rhs"):
        return ctx.opts["rhs"]
    preds = ctx.aliases(slot)
    mode = ctx.opts.get("interactions", "none")
    if mode == "none" or len(preds) == 1:
        rhs = " + ".join(preds)
    elif mode == "two_way":
        rhs = f"({' + '.join(preds)})^2"
    else:
        if len(preds) > 4:
            raise AnalysisInputError("Full factorial is limited to 4 predictors.")
        rhs = " * ".join(preds)
    nl = ctx.aliases("nonlinear") if "nonlinear" in ctx.vars else []
    if nl:
        k = int(ctx.opts.get("nl_degree", 2))
        term = (f"poly({nl[0]}, {k}, raw = TRUE)" if ctx.opts.get("nl_type", "poly") == "poly"
                else f"splines::ns({nl[0]}, df = {k})")
        rhs = f"{rhs} + {term}" if rhs else term
    return rhs


def _main_effects_rhs(ctx: Ctx) -> str:
    return " + ".join(ctx.used("predictors", "nonlinear"))


def _slopes(ctx: Ctx, fit: str) -> list[str]:
    """Conditional (simple) slopes for each two-way interaction that has a continuous variable.
    With an explicit focal pair (discovery), only that pair."""
    focal = ctx.opts.get("focal")
    if focal:
        pairs = [tuple(focal)] if len(focal) == 2 else []
    else:
        preds = ctx.aliases("predictors")
        if ctx.opts.get("interactions", "none") == "none":
            return []
        pairs = [(a, b) for i, a in enumerate(preds) for b in preds[i + 1:]]
    L = []
    for a, b in pairs:
        x, m = (a, b) if not ctx.is_cat(a) else (b, a)
        if ctx.is_cat(x):
            continue
        if ctx.is_cat(m):
            at = ""
            note = "levels of the moderator"
        else:
            at = (f", at = list({m} = mean(sx_d${m}) + c(-1, 0, 1) * stats::sd(sx_d${m}))")
            note = "moderator at mean - 1 SD, mean, mean + 1 SD"
        L += [
            f"sx_et <- emmeans::emtrends({fit}, ~ {m}, var = \"{x}\"{at})",
            "sx_tabs[[length(sx_tabs) + 1]] <- sx_table(",
            f"  paste0(\"Conditional slopes of \", sx_names[[\"{x}\"]], \" by \", sx_names[[\"{m}\"]]),",
            f"  as.data.frame(summary(sx_et, infer = c(TRUE, TRUE), level = {ctx.num('conf')})),",
            f"  notes = \"Slope of {x} at {note}; other predictors at their means / averaged over factor levels. Unadjusted p-values.\")",
        ]
    return L


def _linear(ctx: Ctx) -> str:
    y = ctx.one("dependent")
    rhs = _rhs(ctx)
    conf = ctx.num("conf")
    n_main = len(ctx.used("predictors", "nonlinear"))
    L = _prologue(ctx, "dependent", "predictors", "nonlinear") + [
        f"sx_fit <- stats::lm({y} ~ {rhs}, data = sx_d)",
        "if (any(is.na(stats::coef(sx_fit)))) stop(\"The model is not identifiable (aliased coefficients): \",",
        "  paste(names(stats::coef(sx_fit))[is.na(stats::coef(sx_fit))], collapse = \", \"))",
        "sx_s <- summary(sx_fit)",
        f"sx_ci <- stats::confint(sx_fit, level = {conf})",
        f"sx_std <- tryCatch(as.data.frame(effectsize::standardize_parameters(sx_fit, method = \"refit\", ci = {conf})),",
        "  error = function(e) NULL)",
        "sx_co <- as.data.frame(sx_s$coefficients)",
        "sx_beta <- if (is.null(sx_std)) rep(NA_real_, nrow(sx_co)) else sx_std$Std_Coefficient[match(rownames(sx_co), sx_std$Parameter)]",
        "sx_tabs <- list(sx_table(\"Coefficients\", data.frame(term = rownames(sx_co), estimate = sx_co[[1]], se = sx_co[[2]],",
        "  t = sx_co[[3]], p = sx_co[[4]], ci_low = sx_ci[, 1], ci_high = sx_ci[, 2], std_beta = sx_beta),",
        "  notes = paste0(\"Treatment (dummy) coding: each factor coefficient compares a level with the reference level. \",",
        "  \"std_beta: all variables standardized and the model refitted (effectsize).\")))",
        "sx_f <- sx_s$fstatistic",
        "sx_tabs[[2]] <- sx_table(\"Model fit\", data.frame(n = nrow(sx_d), R2 = sx_s$r.squared, adj_R2 = sx_s$adj.r.squared,",
        "  F = unname(sx_f[1]), df1 = unname(sx_f[2]), df2 = unname(sx_f[3]),",
        "  p = stats::pf(sx_f[1], sx_f[2], sx_f[3], lower.tail = FALSE), sigma = sx_s$sigma,",
        "  AIC = stats::AIC(sx_fit), BIC = stats::BIC(sx_fit)))",
        f"sx_tss <- sum((sx_d${y} - mean(sx_d${y}))^2)",
        "sx_dr <- stats::drop1(sx_fit, test = \"F\")[-1, , drop = FALSE]",
        "sx_tabs[[3]] <- sx_table(\"Term tests (nested model comparison)\", data.frame(term = rownames(sx_dr), df = sx_dr$Df,",
        "  F = sx_dr[[\"F value\"]], p = sx_dr[[\"Pr(>F)\"]], delta_R2 = sx_dr[[\"Sum of Sq\"]] / sx_tss),",
        "  notes = paste0(\"Each row compares the model with and without that term; terms contained in a higher-order \",",
        "  \"interaction are not tested on their own (marginality).\"))",
    ] + _slopes(ctx, "sx_fit") + _diagnostics(ctx, n_main) + [
        f"sx_result <- list(tables = sx_tabs, checks = sx_checks, {COMMON_TAIL})",
    ]
    return "\n".join(L) + "\n"


def _diagnostics(ctx: Ctx, n_main: int) -> list[str]:
    L = [
        "sx_res <- stats::residuals(sx_fit); sx_cook <- stats::cooks.distance(sx_fit)",
        "sx_bp <- lmtest::bptest(sx_fit)",
        "sx_sw <- sx_shapiro(sx_res)",
        "sx_tabs[[length(sx_tabs) + 1]] <- sx_table(\"Diagnostics\", data.frame(",
        "  check = c(\"Breusch-Pagan (constant variance)\", \"Shapiro-Wilk (residual normality)\",",
        "    \"Cook's distance > 4/n (count)\", \"Max Cook's distance\"),",
        "  statistic = c(unname(sx_bp$statistic), sx_sw[[\"W\"]], sum(sx_cook > 4 / nrow(sx_d)), max(sx_cook)),",
        "  p = c(sx_bp$p.value, sx_sw[[\"p\"]], NA, NA)),",
        "  notes = \"With large samples these tests flag trivial departures; read them with the residual plots.\")",
        "sx_checks <- character()",
        "if (max(sx_cook) > 1) sx_checks <- c(sx_checks, \"At least one observation has Cook's distance > 1 (highly influential).\")",
    ]
    if n_main >= 2:
        L += [
            f"sx_main <- stats::lm(stats::reformulate({r_vec(ctx.used('predictors', 'nonlinear'))}, response = sx_used[1]), data = sx_d)",
            "sx_v <- car::vif(sx_main)",
            "sx_vt <- if (is.matrix(sx_v)) data.frame(term = rownames(sx_v), GVIF = sx_v[, 1], df = sx_v[, 2],",
            "  GVIF_adj = sx_v[, 3]) else data.frame(term = names(sx_v), VIF = unname(sx_v))",
            "sx_tabs[[length(sx_tabs) + 1]] <- sx_table(\"Multicollinearity\", sx_vt,",
            "  notes = \"Computed on the main-effects model. GVIF_adj = GVIF^(1/(2 df)) is comparable to sqrt(VIF).\")",
            "if (any((if (is.matrix(sx_v)) sx_v[, 3]^2 else sx_v) > 10)) sx_checks <- c(sx_checks, \"VIF above 10: strong multicollinearity.\")",
        ]
    if ctx.plot_paths:
        L += [
            "sx_pd <- data.frame(fitted = stats::fitted(sx_fit), residual = sx_res, std_res = stats::rstandard(sx_fit))",
            "sx_p1 <- ggplot2::ggplot(sx_pd, ggplot2::aes(fitted, residual)) + ggplot2::geom_point(alpha = 0.35) +",
            "  ggplot2::geom_hline(yintercept = 0, linetype = 2) +",
            "  ggplot2::geom_smooth(method = \"loess\", formula = y ~ x, se = FALSE, colour = \"#b2182b\") +",
            "  ggplot2::labs(title = \"Residuals vs fitted\", x = \"Fitted values\", y = \"Residuals\") + ggplot2::theme_minimal()",
            f"ggplot2::ggsave({r_str(ctx.plot_paths[0])}, sx_p1, width = 6.5, height = 4.5, dpi = 120)",
            "sx_p2 <- ggplot2::ggplot(sx_pd, ggplot2::aes(sample = std_res)) + ggplot2::stat_qq(alpha = 0.35) +",
            "  ggplot2::stat_qq_line(colour = \"#2166ac\") +",
            "  ggplot2::labs(title = \"Normal Q-Q plot of standardized residuals\", x = \"Theoretical quantiles\",",
            "    y = \"Standardized residuals\") + ggplot2::theme_minimal()",
            f"ggplot2::ggsave({r_str(ctx.plot_paths[1])}, sx_p2, width = 6.5, height = 4.5, dpi = 120)",
        ]
    return L


def _hierarchical(ctx: Ctx) -> str:
    y = ctx.one("dependent")
    blocks = [ctx.aliases(b) for b in ("block1", "block2", "block3") if ctx.aliases(b)]
    if len(blocks) < 2:
        raise AnalysisInputError("Hierarchical regression needs at least two blocks.")
    conf = ctx.num("conf")
    cumulative = []
    acc: list[str] = []
    for b in blocks:
        acc = acc + b
        cumulative.append(list(acc))
    L = _prologue(ctx, "dependent", "block1", "block2", "block3") + [
        "sx_fits <- list(" + ", ".join(f"stats::lm({y} ~ {' + '.join(c)}, data = sx_d)" for c in cumulative) + ")",
        "sx_r2 <- vapply(sx_fits, function(m) summary(m)$r.squared, 0)",
        "# Each step vs the previous one (F-change uses that step's residual variance, as in SPSS)",
        "sx_cmp <- do.call(rbind, lapply(seq_along(sx_fits)[-1], function(k) {",
        "  a <- stats::anova(sx_fits[[k - 1]], sx_fits[[k]])",
        "  data.frame(F = a$F[2], Df = a$Df[2], Res.Df = a$Res.Df[2], p = a[[\"Pr(>F)\"]][2]) }))",
        "sx_f1 <- summary(sx_fits[[1]])$fstatistic",
        "sx_first <- c(F = unname(sx_f1[1]), df1 = unname(sx_f1[2]), df2 = unname(sx_f1[3]),",
        "  p = unname(stats::pf(sx_f1[1], sx_f1[2], sx_f1[3], lower.tail = FALSE)))",
        "sx_added <- c(" + ", ".join(_names_joined(b) for b in blocks) + ")",
        "sx_step <- data.frame(model = seq_along(sx_fits), added = sx_added, R2 = sx_r2,",
        "  adj_R2 = vapply(sx_fits, function(m) summary(m)$adj.r.squared, 0),",
        "  delta_R2 = c(sx_r2[1], diff(sx_r2)),",
        "  F_change = c(sx_first[[\"F\"]], sx_cmp$F), df1 = c(sx_first[[\"df1\"]], sx_cmp$Df),",
        "  df2 = c(sx_first[[\"df2\"]], sx_cmp$Res.Df), p_change = c(sx_first[[\"p\"]], sx_cmp$p))",
        "sx_step$f2 <- sx_step$delta_R2 / (1 - sx_step$R2)",
        "sx_fit <- sx_fits[[length(sx_fits)]]",
        "sx_co <- as.data.frame(summary(sx_fit)$coefficients)",
        f"sx_ci <- stats::confint(sx_fit, level = {conf})",
        "sx_tabs <- list(",
        "  sx_table(\"Hierarchical steps\", sx_step, notes = paste0(\"Each step's F-change compares it with the previous model \",",
        "    \"(step 1 with the intercept-only model). All models use the same complete cases. f2 = delta R2 / (1 - R2).\")),",
        "  sx_table(\"Coefficients (final model)\", data.frame(term = rownames(sx_co), estimate = sx_co[[1]], se = sx_co[[2]],",
        "    t = sx_co[[3]], p = sx_co[[4]], ci_low = sx_ci[, 1], ci_high = sx_ci[, 2])))",
        f"sx_result <- list(tables = sx_tabs, {COMMON_TAIL})",
    ]
    return "\n".join(L) + "\n"


def _names_joined(aliases: list[str]) -> str:
    return "paste(sx_names[" + r_vec(aliases) + "], collapse = \", \")"


def _logistic(ctx: Ctx) -> str:
    y = ctx.one("dependent")
    rhs = _rhs(ctx)
    conf = ctx.num("conf")
    preds = ctx.aliases("predictors")
    focal = preds[0]
    others = preds[1:]
    L = _prologue(ctx, "dependent", "predictors") + [
        f"sx_y <- sx_d${y}",
        "sx_lv <- if (is.factor(sx_y)) levels(sx_y) else sort(unique(sx_y))",
        "if (length(sx_lv) != 2) stop(\"The outcome must have exactly two values in the analysed data.\")",
        "sx_d$sx_event <- as.integer(sx_y == sx_lv[2])",
        "sx_event_label <- as.character(sx_lv[2])",
        f"sx_form <- sx_event ~ {rhs}",
        "sx_fit <- stats::glm(sx_form, data = sx_d, family = stats::binomial())",
        "if (any(is.na(stats::coef(sx_fit)))) stop(\"The model is not identifiable (aliased coefficients): \",",
        "  paste(names(stats::coef(sx_fit))[is.na(stats::coef(sx_fit))], collapse = \", \"))",
        "sx_sep <- stats::glm(sx_form, data = sx_d, family = stats::binomial(), method = detectseparation::detect_separation)",
        "sx_checks <- character()",
        "if (isTRUE(sx_sep$outcome)) sx_checks <- c(sx_checks, paste0(\"Separation detected (infinite estimates for: \",",
        "  paste(names(sx_sep$coefficients)[is.infinite(sx_sep$coefficients)], collapse = \", \"),",
        "  \"). Coefficients, odds ratios and CIs for these terms are not trustworthy.\"))",
        "if (!sx_fit$converged) sx_checks <- c(sx_checks, \"The model did not converge.\")",
        "sx_ci_method <- \"profile likelihood\"",
        f"sx_ci <- tryCatch(suppressMessages(stats::confint(sx_fit, level = {conf})), error = function(e) {{",
        f"  sx_ci_method <<- \"Wald (profile likelihood failed)\"; stats::confint.default(sx_fit, level = {conf}) }})",
        "sx_ci <- matrix(sx_ci, ncol = 2)",
        "sx_co <- as.data.frame(summary(sx_fit)$coefficients)",
        "sx_events <- sum(sx_d$sx_event); sx_nonevents <- nrow(sx_d) - sx_events",
        "sx_epp <- min(sx_events, sx_nonevents) / max(1, length(stats::coef(sx_fit)) - 1)",
        "if (sx_epp < 10) sx_checks <- c(sx_checks, sprintf(\"Only %.1f events per estimated parameter; estimates may be unstable.\", sx_epp))",
        "sx_ph <- stats::fitted(sx_fit)",
        "sx_tabs <- list(",
        "  sx_table(\"Coefficients\", data.frame(term = rownames(sx_co), B = sx_co[[1]], se = sx_co[[2]], z = sx_co[[3]],",
        "    p = sx_co[[4]], odds_ratio = exp(sx_co[[1]]), or_ci_low = exp(sx_ci[, 1]), or_ci_high = exp(sx_ci[, 2])),",
        "    notes = paste0(\"Modelled event: \", sx_names[[sx_used[1]]], \" = \", sx_event_label, \". OR CIs: \", sx_ci_method, \".\")),",
        "  sx_table(\"Model fit\", data.frame(n = nrow(sx_d), events = sx_events, non_events = sx_nonevents,",
        "    null_deviance = sx_fit$null.deviance, residual_deviance = sx_fit$deviance,",
        "    LR_chi2 = sx_fit$null.deviance - sx_fit$deviance, df = sx_fit$df.null - sx_fit$df.residual,",
        "    p = stats::pchisq(sx_fit$null.deviance - sx_fit$deviance, sx_fit$df.null - sx_fit$df.residual, lower.tail = FALSE),",
        "    AIC = stats::AIC(sx_fit), BIC = stats::BIC(sx_fit),",
        "    Tjur_R2 = mean(sx_ph[sx_d$sx_event == 1]) - mean(sx_ph[sx_d$sx_event == 0]),",
        "    McFadden_R2 = 1 - sx_fit$deviance / sx_fit$null.deviance,",
        "    events_per_parameter = sx_epp, separation = isTRUE(sx_sep$outcome), converged = sx_fit$converged)))",
        "sx_dr <- stats::drop1(sx_fit, test = \"LRT\")[-1, , drop = FALSE]",
        "sx_tabs[[3]] <- sx_table(\"Term tests (likelihood-ratio)\", data.frame(term = rownames(sx_dr), df = sx_dr$Df,",
        "  LR_chi2 = sx_dr$LRT, p = sx_dr[[\"Pr(>Chi)\"]]), notes = \"Nested model comparisons; marginality respected.\")",
        "# Predicted probabilities across the first predictor; others held at mean / reference level",
        f"sx_grid_base <- lapply(sx_d[, {r_vec(others) if others else 'character(0)'}, drop = FALSE],",
        "  function(v) if (is.factor(v)) factor(levels(v)[1], levels = levels(v)) else mean(v))",
        f"sx_fv <- sx_d${focal}",
        "sx_fx <- if (is.factor(sx_fv)) factor(levels(sx_fv), levels = levels(sx_fv)) else",
        "  seq(min(sx_fv), max(sx_fv), length.out = 100)",
        f"sx_nd <- do.call(data.frame, c(list({focal} = sx_fx), sx_grid_base))",
        "sx_pr <- stats::predict(sx_fit, newdata = sx_nd, type = \"link\", se.fit = TRUE)",
        f"sx_zc <- stats::qnorm(1 - (1 - {conf}) / 2)",
        "sx_nd$probability <- stats::plogis(sx_pr$fit)",
        "sx_nd$ci_low <- stats::plogis(sx_pr$fit - sx_zc * sx_pr$se.fit)",
        "sx_nd$ci_high <- stats::plogis(sx_pr$fit + sx_zc * sx_pr$se.fit)",
        "sx_show <- if (is.factor(sx_fv)) sx_nd else sx_nd[round(seq(1, 100, length.out = 6)), ]",
        "sx_held <- if (length(sx_grid_base)) paste(sprintf(\"%s = %s\", sx_names[names(sx_grid_base)],",
        "  vapply(sx_grid_base, function(v) if (is.factor(v)) as.character(v) else format(signif(v, 4)), \"\")), collapse = \"; \") else \"none\"",
        "sx_tabs[[4]] <- sx_table(\"Predicted probabilities\", sx_show,",
        f"  notes = paste0(\"Varying \", sx_names[[\"{focal}\"]], \"; other predictors held at: \", sx_held, \". CI on the logit scale, back-transformed.\"))",
    ]
    if ctx.plot_paths:
        L += [
            f"sx_p <- ggplot2::ggplot(sx_nd, ggplot2::aes(x = {focal}, y = probability)) +",
            "  (if (is.factor(sx_fv)) list(ggplot2::geom_point(size = 3), ggplot2::geom_errorbar(ggplot2::aes(ymin = ci_low, ymax = ci_high), width = 0.15))",
            "   else list(ggplot2::geom_ribbon(ggplot2::aes(ymin = ci_low, ymax = ci_high), alpha = 0.2), ggplot2::geom_line(linewidth = 1))) +",
            "  ggplot2::coord_cartesian(ylim = c(0, 1)) +",
            "  ggplot2::labs(title = \"Predicted probability\", subtitle = paste0(\"Others held at: \", sx_held),",
            f"    x = sx_names[[\"{focal}\"]], y = paste0(\"P(\", sx_names[[sx_used[1]]], \" = \", sx_event_label, \")\")) +",
            "  ggplot2::theme_minimal()",
            f"ggplot2::ggsave({r_str(ctx.plot_paths[0])}, sx_p, width = 6.5, height = 4.5, dpi = 120)",
        ]
    L.append(f"sx_result <- list(tables = sx_tabs, checks = sx_checks, {COMMON_TAIL})")
    return "\n".join(L) + "\n"


# ========================================================================= mediation

def _mediation(ctx: Ctx) -> str:
    x, m, y = ctx.one("x"), ctx.one("mediator"), ctx.one("dependent")
    covs = ctx.aliases("covariates")
    cv = "".join(f" + {c}" for c in covs)
    B, seed, conf = int(ctx.opts["boot"]), int(ctx.opts["seed"]), ctx.num("conf")
    L = _prologue(ctx, "dependent", "x", "mediator", "covariates") + [
        f"sx_fa <- function(d) stats::lm({m} ~ {x}{cv}, data = d)          # path a: X -> M",
        f"sx_fb <- function(d) stats::lm({y} ~ {x} + {m}{cv}, data = d)     # paths b and c': M -> Y, X -> Y given M",
        f"sx_fc <- function(d) stats::lm({y} ~ {x}{cv}, data = d)          # total effect c",
        "sx_row <- function(fit, term, label) { co <- stats::coef(summary(fit))",
        f"  ci <- stats::confint(fit, term, level = {conf})",
        "  data.frame(path = label, estimate = co[term, 1], se = co[term, 2], p = co[term, 4], ci_low = ci[1], ci_high = ci[2]) }",
        "sx_a <- sx_fa(sx_d); sx_b <- sx_fb(sx_d); sx_c <- sx_fc(sx_d)",
        f"sx_ab <- unname(stats::coef(sx_a)[\"{x}\"] * stats::coef(sx_b)[\"{m}\"])",
        f"set.seed({seed})",
        f"sx_boot <- replicate({B}, {{ i <- sample.int(nrow(sx_d), replace = TRUE); d <- sx_d[i, , drop = FALSE]",
        f"  unname(stats::coef(sx_fa(d))[\"{x}\"] * stats::coef(sx_fb(d))[\"{m}\"]) }})",
        f"sx_q <- stats::quantile(sx_boot, c((1 - {conf}) / 2, 1 - (1 - {conf}) / 2), names = FALSE, na.rm = TRUE)",
        "sx_paths <- rbind(",
        f"  sx_row(sx_a, \"{x}\", \"a: X -> M\"), sx_row(sx_b, \"{m}\", \"b: M -> Y (given X)\"),",
        f"  sx_row(sx_b, \"{x}\", \"c': direct effect X -> Y\"), sx_row(sx_c, \"{x}\", \"c: total effect X -> Y\"),",
        "  data.frame(path = \"a*b: indirect effect\", estimate = sx_ab, se = stats::sd(sx_boot), p = NA_real_,",
        "    ci_low = sx_q[1], ci_high = sx_q[2]))",
        f"sx_cx <- unname(stats::coef(sx_c)[\"{x}\"])",
        "sx_result <- list(tables = list(",
        "  sx_table(\"Mediation paths\", sx_paths, notes = paste0(\"Indirect effect CI: percentile bootstrap, \",",
        f"    {B}, \" resamples (seed {seed}); its 'se' is the bootstrap SD. Other CIs are ordinary t-based.\")),",
        f"  sx_table(\"Summary\", data.frame(indirect = sx_ab, direct = unname(stats::coef(sx_b)[\"{x}\"]),",
        "    total = sx_cx, proportion_mediated = sx_ab / sx_cx, indirect_ci_excludes_zero = sx_q[1] > 0 || sx_q[2] < 0),",
        "    notes = \"Proportion mediated is unstable when the total effect is near zero.\")),",
        "  notes = paste0(\"Mediation is a statistical decomposition, not proof of a causal pathway: it assumes the \",",
        "    \"causal order X -> M -> Y and no unmeasured confounding of any path, which observational data cannot show.\"),",
        f"  {COMMON_TAIL})",
    ]
    return "\n".join(L) + "\n"


def _alpha(ctx: Ctx) -> str:
    L = _prologue(ctx, "items") + [
        "sx_al <- psych::alpha(sx_d, check.keys = FALSE, warnings = FALSE)",
        "sx_tot <- sx_al$total",
        "sx_drop <- sx_al$alpha.drop",
        "sx_result <- list(tables = list(",
        "  sx_table(\"Reliability\", data.frame(items = ncol(sx_d), n = nrow(sx_d), raw_alpha = sx_tot$raw_alpha,",
        "    std_alpha = sx_tot$std.alpha, average_r = sx_tot$average_r, mean = sx_tot$mean, sd = sx_tot$sd),",
        "    notes = \"Cronbach's alpha (psych::alpha). Items are not reverse-keyed automatically: reverse negatively worded items first.\"),",
        "  sx_table(\"If an item is dropped\", data.frame(item = unname(sx_names[rownames(sx_drop)]), raw_alpha = sx_drop$raw_alpha,",
        "    std_alpha = sx_drop$std.alpha), notes = \"Alpha of the scale without each item.\")),",
        f"  {COMMON_TAIL})",
    ]
    return "\n".join(L) + "\n"


# ============================================================================ specs

DV = Slot("dependent", "Dependent variable", "continuous")

SPECS: list[AnalysisSpec] = [
    AnalysisSpec("t_one_sample", "t-tests", "One-sample t-test", "Compare a mean with a fixed test value.",
                 [DV], [Option("mu", "Test value", "float", 0.0), ALTERNATIVE, CONF],
                 ["effectsize"], _one_sample),
    AnalysisSpec("t_independent", "t-tests", "Independent-samples t-test (Student)",
                 "Compare two group means assuming equal variances.",
                 [DV, Slot("group", "Grouping variable (2 groups)", "categorical")], [ALTERNATIVE, CONF],
                 ["effectsize", "car"], lambda c: _independent(c, welch=False)),
    AnalysisSpec("t_welch", "t-tests", "Welch t-test", "Compare two group means without assuming equal variances.",
                 [DV, Slot("group", "Grouping variable (2 groups)", "categorical")], [ALTERNATIVE, CONF],
                 ["effectsize", "car"], lambda c: _independent(c, welch=True)),
    AnalysisSpec("t_paired", "t-tests", "Paired-samples t-test",
                 "Compare two measurements on the same cases (two columns, wide format).",
                 [Slot("first", "First measurement", "continuous"), Slot("second", "Second measurement", "continuous")],
                 [ALTERNATIVE, CONF], ["effectsize"], _paired),
    AnalysisSpec("cor_pair", "Correlation", "Correlation (two variables)", "Pearson or Spearman correlation with test.",
                 [Slot("x", "Variable X", "continuous"), Slot("y", "Variable Y", "continuous")], [METHOD, CONF],
                 [], _cor_pair),
    AnalysisSpec("cor_partial", "Correlation", "Partial correlation",
                 "Correlation between X and Y controlling for other variables.",
                 [Slot("x", "Variable X", "continuous"), Slot("y", "Variable Y", "continuous"),
                  Slot("controls", "Control variables", "continuous", 1, 5)], [METHOD, CONF], ["ppcor"], _cor_partial),
    AnalysisSpec("cor_matrix", "Correlation", "Correlation matrix (psych::corr.test)",
                 "All pairwise correlations with raw and multiplicity-adjusted p-values.",
                 [Slot("variables", "Variables", "continuous", 2, 15)],
                 [METHOD, Option("adjust", "p-value adjustment", "choice", "holm",
                                 [("holm", "Holm"), ("BH", "Benjamini-Hochberg (FDR)"), ("bonferroni", "Bonferroni"),
                                  ("none", "None")])],
                 ["psych"], _cor_matrix, missing_data="Pairwise deletion (each correlation uses all rows complete on that pair)."),
    AnalysisSpec("anova_oneway", "ANOVA", "One-way ANOVA", "Compare means across groups of one factor.",
                 [DV, Slot("factors", "Factor", "categorical")], [ADJUST, CONF], ["afex", "emmeans", "car"], _anova_between),
    AnalysisSpec("anova_welch", "ANOVA", "Welch one-way ANOVA", "One-way ANOVA without assuming equal variances.",
                 [DV, Slot("factor", "Factor", "categorical")], [CONF], ["effectsize"], _anova_welch),
    AnalysisSpec("anova_factorial", "ANOVA", "Factorial ANOVA (2-3 factors)",
                 "Main effects and interactions of two or three between-subject factors.",
                 [DV, Slot("factors", "Factors", "categorical", 2, 3)], [ADJUST, CONF], ["afex", "emmeans", "car"],
                 _anova_between),
    AnalysisSpec("anova_rm", "ANOVA", "Repeated-measures ANOVA",
                 "Within-subject factor(s), long-format data (one row per participant per level).",
                 [DV, Slot("id", "Participant ID", "any"), Slot("within", "Within-subject factor(s)", "categorical", 1, 2)],
                 [ADJUST, CONF], ["afex", "emmeans"], lambda c: _anova_rm(c, mixed=False),
                 missing_data="Participants missing any within-subject cell are excluded (afex)."),
    AnalysisSpec("anova_mixed", "ANOVA", "Mixed-design ANOVA",
                 "Within- and between-subject factors, long-format data.",
                 [DV, Slot("id", "Participant ID", "any"), Slot("within", "Within-subject factor(s)", "categorical", 1, 2),
                  Slot("between", "Between-subject factor(s)", "categorical", 1, 2)],
                 [ADJUST, CONF], ["afex", "emmeans"], lambda c: _anova_rm(c, mixed=True),
                 missing_data="Participants missing any within-subject cell are excluded (afex)."),
    AnalysisSpec("ancova", "ANOVA", "ANCOVA", "Group comparison adjusted for continuous covariates.",
                 [DV, Slot("factors", "Factor(s)", "categorical", 1, 2), Slot("covariates", "Covariate(s)", "continuous", 1, 3)],
                 [ADJUST, CONF], ["afex", "emmeans"], _ancova),
    AnalysisSpec("lm", "Regression", "Linear regression",
                 "Simple, multiple, interaction, polynomial or spline regression (OLS).",
                 [DV, Slot("predictors", "Predictors", "any", 1, 10),
                  Slot("nonlinear", "Nonlinear term (optional)", "continuous", 0, 1,
                       "Adds a polynomial or natural-spline term for this variable.")],
                 [INTERACTIONS, Option("nl_type", "Nonlinear type", "choice", "poly",
                                       [("poly", "Polynomial"), ("spline", "Natural spline")]),
                  Option("nl_degree", "Polynomial degree / spline df", "int", 2, min=2, max=5), CONF],
                 ["effectsize", "emmeans", "lmtest", "car", "ggplot2"], _linear, n_plots=2),
    AnalysisSpec("lm_hierarchical", "Regression", "Hierarchical (blockwise) regression",
                 "Add predictor blocks in steps and test the R-squared change.",
                 [DV, Slot("block1", "Block 1", "any", 1, 10), Slot("block2", "Block 2", "any", 1, 10),
                  Slot("block3", "Block 3 (optional)", "any", 0, 10)], [CONF], [], _hierarchical),
    AnalysisSpec("logistic", "Regression", "Binary logistic regression",
                 "Odds ratios, predicted probabilities, separation checks.",
                 [Slot("dependent", "Outcome (two values)", "binary"), Slot("predictors", "Predictors", "any", 1, 10)],
                 [INTERACTIONS, CONF], ["detectseparation", "ggplot2"], _logistic, n_plots=1),
]
SPECS.append(AnalysisSpec(
    "mediation", "Mediation (optional)", "Simple mediation (X -> M -> Y)",
    "Indirect effect of X on Y through one mediator M, with a bootstrap confidence interval. Separate from "
    "discovery; does not establish causation.",
    [Slot("x", "Predictor X", "continuous"), Slot("mediator", "Mediator M", "continuous"), DV,
     Slot("covariates", "Covariates (optional)", "any", 0, 5)],
    [Option("boot", "Bootstrap resamples", "int", 2000, min=200, max=20000),
     Option("seed", "Random seed", "int", 1, min=1, max=10**9), CONF],
    ["stats"], _mediation, missing_data="Complete cases on X, M, Y and covariates (bootstrap resamples these rows)."))
SPECS.append(AnalysisSpec(
    "alpha", "Psychometrics (optional)", "Scale reliability (Cronbach's alpha)",
    "Internal consistency of a set of items (psych::alpha). Runs only when you choose it.",
    [Slot("items", "Items", "any", 2, 40)], [], ["psych"], _alpha,
    missing_data="Complete cases on all items."))
BY_KEY = {s.key: s for s in SPECS}
