"""Teaching Studio: interaction plots computed from the fitted model, conditional slopes /
simple effects, Johnson-Neyman intervals, and short template interpretations (§16).

Plotted lines are model predictions on an explicit grid: the x variable varies, the group and
panel variables take displayed values, and every other predictor is held at a displayed value
(continuous: mean; categorical: reference level).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.dataset import Dataset
from ..core.schema import AnalysisResult, Status
from ..rbridge.codegen import alias, r_num, r_str, r_vec
from .models import ModelCheckError, ModelSpec, check_model, model_view
from .session import RSession

BACKENDS = {"ggplot2": "ggplot2 (default)", "base": "Base R graphics", "flexplot": "flexplot (raw data + its own fit)"}
FORMATS = ("png", "svg", "pdf")


class TeachingInputError(ValueError):
    pass


@dataclass
class PlotSpec:
    x: str
    group: str | None = None
    panel: str | None = None
    ci: bool = True
    show_data: bool = False
    backend: str = "ggplot2"
    fmt: str = "png"
    title: str = ""
    xlab: str = ""
    ylab: str = ""
    moderator_values: str = "sd"        # continuous group/panel: "sd" (mean ± 1 SD) or "quantiles" (10/50/90%)
    conf: float = 0.95
    held: dict[str, Any] | None = None  # user-chosen values for other predictors (default: mean / reference)


def validate(ds: Dataset, m: ModelSpec, p: PlotSpec) -> None:
    ds = model_view(ds, m)
    try:
        check_model(ds, m)
    except ModelCheckError as e:
        raise TeachingInputError(str(e)) from None
    vs = m.variables
    for v in (p.x, p.group, p.panel):
        if v is not None and v not in vs:
            raise TeachingInputError(f"{ds.var(v).display_name} is not in the model.")
    chosen = [v for v in (p.x, p.group, p.panel) if v]
    if len(set(chosen)) != len(chosen):
        raise TeachingInputError("x, group and panel must be different variables.")
    if p.backend not in BACKENDS or p.fmt not in FORMATS:
        raise TeachingInputError("Unknown backend or format.")
    for name, value in (p.held or {}).items():
        if name not in vs or name in chosen:
            raise TeachingInputError(f"{ds.var(name).display_name} is not a held predictor of this model.")
        if ds.var(name).is_categorical:
            labels = [ds.level_label(name, c) for c in ds.levels(name)]
            if str(value) not in labels:
                raise TeachingInputError(f"{value!r} is not a category of {ds.var(name).display_name}.")
        else:
            try:
                float(value)
            except (TypeError, ValueError):
                raise TeachingInputError(f"{ds.var(name).display_name}: enter a number.") from None


def _continuous_values_r(v: str, how: str) -> str:
    if how == "quantiles":
        return f"stats::quantile(sx_d${v}, c(0.1, 0.5, 0.9), names = FALSE)"
    return f"mean(sx_d${v}) + c(-1, 0, 1) * stats::sd(sx_d${v})"


def _values_r(v: str, how: str) -> str:
    """R expression for the displayed values of a group/panel variable."""
    return f"if (is.factor(sx_d${v})) levels(sx_d${v}) else {_continuous_values_r(v, how)}"


def _label_r(v: str, how: str) -> str:
    tag = 'c("10th pct", "median", "90th pct")' if how == "quantiles" else 'c("-1 SD", "mean", "+1 SD")'
    return (f"if (is.factor(sx_d${v})) levels(sx_d${v}) else "
            f"paste0({tag}, \" (\", format(signif({_continuous_values_r(v, how)}, 3)), \")\")")


def build(ds: Dataset, m: ModelSpec, p: PlotSpec, plot_path: str) -> str:
    ds = model_view(ds, m)
    a = lambda n: alias(ds.var(n).alias)  # noqa: E731
    used = [a(v) for v in [m.outcome] + m.variables]
    extra = [a(v) for v in m.sample_vars if a(v) not in used]
    x = a(p.x)
    g = a(p.group) if p.group else None
    pn = a(p.panel) if p.panel else None
    others = [a(v) for v in m.variables if v not in (p.x, p.group, p.panel)]
    gauss = m.family == "gaussian"
    conf = r_num(p.conf)
    focal = ":".join(a(v) for v in m.focal) if m.focal else None
    L = [
        f"sx_used <- {r_vec(used)}",
        (f"sx_d <- droplevels(sx_complete(dat, c(sx_used, {r_vec(extra)}))[, sx_used, drop = FALSE])" if extra
         else "sx_d <- sx_complete(dat, sx_used)"),
        "if (nrow(sx_d) < 5) stop(\"Too few complete cases.\")",
    ]
    y = a(m.outcome)
    if gauss:
        L.append(f"sx_fit <- stats::lm({y} ~ {m.rhs(ds)}, data = sx_d)")
        resp = y
    else:
        L += [
            f"sx_lv <- if (is.factor(sx_d${y})) levels(sx_d${y}) else sort(unique(sx_d${y}))",
            f"sx_d$sx_event <- as.integer(sx_d${y} == sx_lv[2])",
            f"sx_fit <- stats::glm(sx_event ~ {m.rhs(ds)}, data = sx_d, family = stats::binomial())",
        ]
        resp = "sx_event"
    L += [
        "if (any(is.na(stats::coef(sx_fit)))) stop(\"The model is not identifiable (aliased coefficients).\")",
        "sx_tabs <- list()",
    ]
    if focal:
        test = "\"F\"" if gauss else "\"Chisq\""
        L += [
            f"sx_red <- stats::update(sx_fit, . ~ . - {focal})",
            f"sx_cmp <- stats::anova(sx_red, sx_fit, test = {test})",
            "sx_focal <- list(df1 = sx_cmp$Df[2], df2 = " + ("sx_cmp$Res.Df[2]" if gauss else "NA_real_")
            + ", statistic = " + ("sx_cmp$F[2]" if gauss else "sx_cmp$Deviance[2]")
            + ", p = sx_cmp[[ncol(sx_cmp)]][2])",
            "sx_co <- stats::coef(summary(sx_fit))",
            f"sx_fc <- grep(paste0(\"^\", gsub(\":\", \"[^:]*:\", {r_str(focal)}), \"[^:]*$\"), rownames(sx_co), value = TRUE)",
            # t-based CI for linear models; Wald CI for logistic (profile CIs are in the Analysis Builder)
            (f"sx_ci <- stats::confint(sx_fit, parm = sx_fc, level = {conf})" if gauss
             else f"sx_ci <- stats::confint.default(sx_fit, parm = sx_fc, level = {conf})"),
            f"sx_tabs[[1]] <- sx_table(\"Interaction test\", data.frame(term = paste(sx_names[{r_vec(focal.split(':'))}], collapse = \" x \"),",
            "  df1 = sx_focal$df1, df2 = sx_focal$df2, statistic = sx_focal$statistic, p = sx_focal$p),",
            "  notes = " + ("\"F-test comparing the model with and without the interaction.\"" if gauss
                            else "\"Likelihood-ratio chi-square comparing the model with and without the interaction.\"") + ")",
            "sx_tabs[[2]] <- sx_table(\"Interaction coefficient(s)\", data.frame(term = sx_fc, estimate = sx_co[sx_fc, 1],",
            "  se = sx_co[sx_fc, 2], p = sx_co[sx_fc, 4], ci_low = sx_ci[, 1], ci_high = sx_ci[, 2]),",
            "  notes = " + ("\"Change in the slope of one predictor per unit (or level) of the other.\"" if gauss
                            else "\"On the log-odds scale; exp(estimate) is a ratio of odds ratios.\"") + ")",
        ]
    else:
        L.append("sx_focal <- NULL")
    # ---- prediction grid ----
    L += [
        f"sx_xv <- if (is.factor(sx_d${x})) factor(levels(sx_d${x}), levels = levels(sx_d${x})) else "
        f"seq(min(sx_d${x}), max(sx_d${x}), length.out = 80)",
        f"sx_grid <- list({x} = sx_xv)",
    ]
    for v, role in ((g, "group"), (pn, "panel")):
        if v:
            L += [
                f"sx_{role}_values <- {_values_r(v, p.moderator_values)}",
                f"sx_{role}_labels <- {_label_r(v, p.moderator_values)}",
                f"sx_grid${v} <- if (is.factor(sx_d${v})) factor(sx_{role}_values, levels = levels(sx_d${v})) else sx_{role}_values",
            ]
    L += [
        f"sx_held <- lapply(sx_d[, {r_vec(others) if others else 'character(0)'}, drop = FALSE],",
        "  function(v) if (is.factor(v)) factor(levels(v)[1], levels = levels(v)) else mean(v))",
    ] + [
        (f"sx_held${a(n)} <- factor({r_str(v)}, levels = levels(sx_d${a(n)}))  # chosen by the user"
         if ds.var(n).is_categorical else f"sx_held${a(n)} <- {r_num(float(v))}  # chosen by the user")
        for n, v in (p.held or {}).items()
    ] + [
        "sx_nd <- do.call(expand.grid, c(sx_grid, sx_held, list(KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)))",
        "sx_pr <- stats::predict(sx_fit, newdata = sx_nd, se.fit = TRUE)",
    ]
    if gauss:
        L += [
            f"sx_crit <- stats::qt(1 - (1 - {conf}) / 2, sx_fit$df.residual)",
            "sx_nd$predicted <- sx_pr$fit",
            "sx_nd$ci_low <- sx_pr$fit - sx_crit * sx_pr$se.fit",
            "sx_nd$ci_high <- sx_pr$fit + sx_crit * sx_pr$se.fit",
        ]
    else:
        L += [
            f"sx_crit <- stats::qnorm(1 - (1 - {conf}) / 2)",
            "sx_nd$predicted <- stats::plogis(sx_pr$fit)",
            "sx_nd$ci_low <- stats::plogis(sx_pr$fit - sx_crit * sx_pr$se.fit)",
            "sx_nd$ci_high <- stats::plogis(sx_pr$fit + sx_crit * sx_pr$se.fit)",
        ]
    for v, role in ((g, "group"), (pn, "panel")):
        if v:
            L.append(f"sx_nd$sx_{role} <- factor(sx_{role}_labels[match(as.character(sx_nd${v}), "
                     f"as.character(sx_{role}_values))], levels = sx_{role}_labels)")
    L += [
        "sx_held_txt <- if (length(sx_held)) paste(sprintf(\"%s = %s\", sx_names[names(sx_held)],",
        "  vapply(sx_held, function(v) if (is.factor(v)) as.character(v) else format(signif(v, 4)), \"\")), collapse = \"; \") else \"none\"",
        "sx_tabs[[length(sx_tabs) + 1]] <- sx_table(\"Plotted predictions\", sx_nd[, setdiff(names(sx_nd), c(\"sx_group\", \"sx_panel\")), drop = FALSE],",
        f"  notes = paste0(\"Model predictions with {round(p.conf * 100)}% CI\", "
        + ("\"\"" if gauss else "\" (probability scale; CI computed on the logit scale)\"")
        + ", \". Other predictors held at: \", sx_held_txt, \".\"))",
    ]
    # ---- conditional slopes / simple effects ----
    mod = g or pn
    if mod and m.focal and len(m.focal) == 2 and set(m.focal) == {p.x, p.group or p.panel}:
        at_others = ", ".join(f"{o} = sx_held${o}" for o in others)
        if ds.var(p.x).is_categorical:
            L += [
                f"sx_em <- emmeans::emmeans(sx_fit, ~ {x} | {mod}, at = list({mod} = sx_grid${mod}"
                + (f", {at_others}" if others else "") + "))",
                f"sx_se <- as.data.frame(summary(pairs(sx_em, reverse = TRUE, adjust = \"none\"), infer = c(TRUE, TRUE), level = {conf}"
                + ("" if gauss else ", type = \"response\"") + "))",
                "sx_tabs[[length(sx_tabs) + 1]] <- sx_table(\"Simple effects\", sx_se, notes = paste0(\"Differences between \",",
                f"  sx_names[[\"{x}\"]], \" levels at each value of \", sx_names[[\"{mod}\"]], "
                + ("\". Unadjusted p-values.\"" if gauss else "\" (odds ratios). Unadjusted p-values.\"") + "))",
                "sx_slopes <- sx_se",
            ]
        else:
            L += [
                f"sx_et <- emmeans::emtrends(sx_fit, ~ {mod}, var = \"{x}\", at = list({mod} = sx_grid${mod}"
                + (f", {at_others}" if others else "") + "))",
                f"sx_slopes <- as.data.frame(summary(sx_et, infer = c(TRUE, TRUE), level = {conf}))",
                f"sx_slopes$label <- sx_{'group' if g else 'panel'}_labels",
                "sx_tabs[[length(sx_tabs) + 1]] <- sx_table(\"Conditional slopes\", sx_slopes, notes = paste0(\"Slope of \",",
                f"  sx_names[[\"{x}\"]], \" at each value of \", sx_names[[\"{mod}\"]], "
                + ("\". Unadjusted p-values.\"" if gauss else "\" (log-odds per unit). Unadjusted p-values.\"") + "))",
            ]
            if not ds.var(p.group or p.panel).is_categorical and gauss:
                L += _johnson_neyman(x, mod, focal, conf)
    else:
        L.append("sx_slopes <- NULL")
    # ---- plot ----
    L += _plot_code(ds, m, p, plot_path, x, g, pn, gauss, resp)
    L.append("sx_result <- list(tables = sx_tabs, n_used = nrow(sx_d), n_excluded = nrow(dat) - nrow(sx_d),")
    L.append("  values = list(focal = sx_focal, held = sx_held_txt, jn = if (exists(\"sx_jn\")) sx_jn else NULL))")
    return "\n".join(L) + "\n"


def _johnson_neyman(x: str, mod: str, focal: str, conf: str) -> list[str]:
    """Exact Johnson-Neyman bounds for slope(x) = b_x + b_int * m in a linear model."""
    return [
        "# Johnson-Neyman: moderator values where the slope's CI excludes zero (solved exactly)",
        f"sx_b <- stats::coef(sx_fit); sx_V <- stats::vcov(sx_fit); sx_t2 <- stats::qt(1 - (1 - {conf}) / 2, sx_fit$df.residual)^2",
        f"sx_bx <- sx_b[[\"{x}\"]]; sx_bi <- sx_b[[{r_str(focal)}]]",
        f"sx_A <- sx_bi^2 - sx_t2 * sx_V[{r_str(focal)}, {r_str(focal)}]",
        f"sx_B <- 2 * (sx_bx * sx_bi - sx_t2 * sx_V[\"{x}\", {r_str(focal)}])",
        f"sx_C <- sx_bx^2 - sx_t2 * sx_V[\"{x}\", \"{x}\"]",
        "sx_disc <- sx_B^2 - 4 * sx_A * sx_C",
        "sx_jn <- if (sx_disc < 0) c(NA_real_, NA_real_) else sort((-sx_B + c(-1, 1) * sqrt(sx_disc)) / (2 * sx_A))",
        f"sx_rng <- range(sx_d${mod})",
        "sx_jn_txt <- if (anyNA(sx_jn)) \"The slope's confidence interval never excludes zero for any moderator value.\" else",
        "  if (sx_A > 0) sprintf(\"The slope is significant when the moderator is below %.4g or above %.4g (observed range %.4g to %.4g).\", sx_jn[1], sx_jn[2], sx_rng[1], sx_rng[2]) else",
        "  sprintf(\"The slope is significant when the moderator is between %.4g and %.4g (observed range %.4g to %.4g).\", sx_jn[1], sx_jn[2], sx_rng[1], sx_rng[2])",
        "sx_tabs[[length(sx_tabs) + 1]] <- sx_table(\"Johnson-Neyman interval\", data.frame(lower = sx_jn[1], upper = sx_jn[2],",
        "  significant_outside = sx_A > 0, observed_min = sx_rng[1], observed_max = sx_rng[2]), notes = sx_jn_txt)",
        "sx_jn <- list(bounds = sx_jn, outside = sx_A > 0, text = sx_jn_txt)",
    ]


def _base_ci_line(ci: bool, xcat: bool) -> str:
    if not ci:
        return "    NULL"
    if xcat:
        return ("    graphics::arrows(sx_xx, sx_q$ci_low, sx_xx, sx_q$ci_high, angle = 90, code = 3, "
                "length = 0.05, col = sx_cols[sx_j])")
    return ("    graphics::polygon(c(sx_xx, rev(sx_xx)), c(sx_q$ci_low, rev(sx_q$ci_high)), "
            "col = grDevices::adjustcolor(sx_cols[sx_j], 0.2), border = NA)")


def _plot_code(ds: Dataset, m: ModelSpec, p: PlotSpec, path: str, x: str, g: str | None, pn: str | None,
               gauss: bool, resp: str) -> list[str]:
    xcat = ds.var(p.x).is_categorical
    xl = r_str(p.xlab) if p.xlab else f"sx_names[[\"{x}\"]]"
    default_y = f"sx_names[[\"{alias(ds.var(m.outcome).alias)}\"]]"
    yl = r_str(p.ylab) if p.ylab else (default_y if gauss else f"paste0(\"P(\", {default_y}, \" = \", sx_lv[2], \")\")")
    title = r_str(p.title) if p.title else "\"Model-predicted values\""
    sub = "if (length(sx_held)) paste0(\"Other predictors held at: \", sx_held_txt) else NULL"
    dev = {"png": "", "svg": ", device = grDevices::svg", "pdf": ", device = grDevices::cairo_pdf"}[p.fmt]
    export = path.rsplit(".", 1)[0] + "." + p.fmt
    png = path.rsplit(".", 1)[0] + ".png"
    if p.backend == "ggplot2":
        aes_g = ", colour = sx_group, fill = sx_group, group = sx_group" if g else ""
        L = [f"sx_p <- ggplot2::ggplot(sx_nd, ggplot2::aes(x = {x}, y = predicted{aes_g}))"]
        if p.show_data:
            pts = f"ggplot2::aes(x = {x}, y = {resp})"
            jitter = "ggplot2::position_jitter(width = 0.1, height = 0)" if xcat else \
                "ggplot2::position_jitter(width = 0, height = " + ("0" if gauss else "0.03") + ")"
            L.append(f"sx_p <- sx_p + ggplot2::geom_point(data = sx_d, mapping = {pts}, inherit.aes = FALSE, "
                     f"alpha = 0.15, colour = \"grey40\", position = {jitter})")
        if xcat:
            dodge = "ggplot2::position_dodge(width = 0.3)"
            if p.ci:
                L.append(f"sx_p <- sx_p + ggplot2::geom_errorbar(ggplot2::aes(ymin = ci_low, ymax = ci_high), width = 0.15, position = {dodge})")
            L.append(f"sx_p <- sx_p + ggplot2::geom_point(size = 3, position = {dodge})"
                     + (f" + ggplot2::geom_line(position = {dodge})" if g else ""))
        else:
            if p.ci:
                L.append("sx_p <- sx_p + ggplot2::geom_ribbon(ggplot2::aes(ymin = ci_low, ymax = ci_high), alpha = 0.18, colour = NA)")
            L.append("sx_p <- sx_p + ggplot2::geom_line(linewidth = 1.1)")
        if pn:
            L.append("sx_p <- sx_p + ggplot2::facet_wrap(~ sx_panel, labeller = ggplot2::label_both)")
        lab_g = f", colour = sx_names[[\"{g}\"]], fill = sx_names[[\"{g}\"]]" if g else ""
        L += [
            f"sx_p <- sx_p + ggplot2::labs(title = {title}, subtitle = {sub}, x = {xl}, y = {yl}{lab_g}) +",
            "  ggplot2::theme_minimal(base_size = 12) + ggplot2::theme(legend.position = \"bottom\")",
            f"ggplot2::ggsave({r_str(png)}, sx_p, width = 7.5, height = 5.2, dpi = 120)",
        ] + ([f"ggplot2::ggsave({r_str(export)}, sx_p, width = 7.5, height = 5.2{dev})"] if p.fmt != "png" else [])
        return L
    if p.backend == "base":
        openers = [f"grDevices::png({r_str(png)}, width = 7.5, height = 5.2, units = \"in\", res = 120)"]
        if p.fmt != "png":
            openers.append({"svg": f"grDevices::svg({r_str(export)}, width = 7.5, height = 5.2)",
                            "pdf": f"grDevices::cairo_pdf({r_str(export)}, width = 7.5, height = 5.2)"}[p.fmt])
        return [
            "sx_draw <- function() {",
            "sx_panels <- if (is.null(sx_nd$sx_panel)) list(sx_nd) else split(sx_nd, sx_nd$sx_panel)",
            "graphics::par(mfrow = c(1, length(sx_panels)), mar = c(4.5, 4.5, 3, 1))",
            "sx_cols <- grDevices::hcl.colors(max(1, nlevels(sx_nd$sx_group)), \"Dark 3\")",
            "for (sx_k in seq_along(sx_panels)) {",
            "  sx_pp <- sx_panels[[sx_k]]",
            "  sx_groups <- if (is.null(sx_pp$sx_group)) list(sx_pp) else split(sx_pp, sx_pp$sx_group)",
            f"  sx_xnum <- as.numeric(sx_pp${x})",
            f"  graphics::plot(range(sx_xnum), range(c(sx_pp$ci_low, sx_pp$ci_high)), type = \"n\", xlab = {xl}, ylab = {yl},",
            f"    main = if (length(sx_panels) > 1) names(sx_panels)[sx_k] else {title}, xaxt = if (is.factor(sx_pp${x})) \"n\" else \"s\")",
            f"  if (is.factor(sx_pp${x})) graphics::axis(1, at = seq_along(levels(sx_pp${x})), labels = levels(sx_pp${x}))",
            "  for (sx_j in seq_along(sx_groups)) { sx_q <- sx_groups[[sx_j]]; sx_xx <- as.numeric(sx_q[[1]])",
            _base_ci_line(p.ci, xcat),
            "    graphics::lines(sx_xx, sx_q$predicted, col = sx_cols[sx_j], lwd = 2, type = " + ("\"b\"" if xcat else "\"l\"") + ") }",
            "  if (!is.null(sx_pp$sx_group)) graphics::legend(\"topleft\", legend = names(sx_groups), col = sx_cols, lwd = 2, bty = \"n\")",
            "}",
            "if (length(sx_held)) graphics::mtext(paste0(\"Other predictors held at: \", sx_held_txt), side = 1, line = -1, outer = TRUE, cex = 0.7)",
            "}",
        ] + [f"{o}; sx_draw(); invisible(grDevices::dev.off())" for o in openers]
    # flexplot: raw data with flexplot's own fitted lines (not the full model with covariates)
    rhs = x + (f" + {g}" if g else "") + (f" | {pn}" if pn else "")
    method = "\"lm\"" if gauss else "\"logistic\""
    return [
        "if (!requireNamespace(\"flexplot\", quietly = TRUE)) stop(\"The flexplot package is not installed.\")",
        f"sx_p <- flexplot::flexplot({resp} ~ {rhs}, data = sx_d, method = {method})",
        f"ggplot2::ggsave({r_str(png)}, sx_p, width = 7.5, height = 5.2, dpi = 120)",
    ] + ([f"ggplot2::ggsave({r_str(export)}, sx_p, width = 7.5, height = 5.2{dev})"] if p.fmt != "png" else [])


# ---- run + interpretation ---------------------------------------------------------------

def run(session: RSession, ds: Dataset, m: ModelSpec, p: PlotSpec) -> tuple[AnalysisResult, dict[str, bytes]]:
    ds = model_view(ds, m)
    from ..analysis.framework import _alias_map, _as_list, _parse_table

    validate(ds, m, p)
    path = session.new_plot_path()                    # PNG preview (always)
    export = path.with_suffix("." + p.fmt)
    body = build(ds, m, p, path.as_posix())
    pk = ["ggplot2", "emmeans"] + (["flexplot"] if p.backend == "flexplot" else [])
    inputs = {"model": m.to_dict(), "plot": vars(p), "filters": [f.describe() for f in ds.filters]}
    result, raw = session.run(ds, kind="teaching_plot", title=f"Interaction plot: {m.display(ds)}", body=body,
                              packages=pk, inputs=inputs)
    result.missing_data = ("Shared complete-case sample of the source search" if m.sample_vars
                           else "Complete cases on the model variables")
    from .measurement import record

    record(result, ds)
    artifacts: dict[str, bytes] = {}
    if raw is None:
        return result, artifacts
    names = _alias_map(ds)
    for t in _as_list(raw.get("tables")):
        result.tables.append(_parse_table(t, names))
    result.n_used = raw.get("n_used")
    result.values["n_excluded"] = raw.get("n_excluded")
    vals = raw.get("values") or {}
    result.values["held"] = vals.get("held")
    if path.is_file():
        key = f"plots/{path.name}"
        artifacts[key] = path.read_bytes()
        result.plots.append(key)
        if p.fmt != "png" and export.is_file():
            result.values["export"] = f"plots/{export.name}"
            artifacts[result.values["export"]] = export.read_bytes()
    else:
        result.status = Status.FAILED
        result.messages.append("No plot file was produced.")
    result.values["interpretation"] = interpret(ds, m, p, result, vals)
    if p.backend == "flexplot":
        result.messages.append("flexplot draws the raw data with its own fitted lines; it does not include "
                               "covariates held constant, so it can differ from the model above.")
    return result, artifacts


def _fmt_p(p: float) -> str:
    return "p < .001" if p < 0.001 else f"p = {p:.3f}".replace("0.", ".")


def _n(x: float, d: int = 2) -> str:
    return f"{x:.{d}f}" if abs(x) >= 0.01 or x == 0 else f"{x:.2e}"


def interpret(ds: Dataset, m: ModelSpec, p: PlotSpec, result: AnalysisResult, vals: dict[str, Any]) -> str:
    """Three to six sentences from validated templates (§16.4, §23)."""
    ds = model_view(ds, m)
    dn = lambda v: ds.var(v).display_name  # noqa: E731
    gauss = m.family == "gaussian"
    y = dn(m.outcome)
    s: list[str] = []
    focal = m.focal
    others = [dn(v) for v in m.variables if not focal or v not in focal]
    adj = f", adjusting for {', '.join(others)}" if others else ""
    if focal and len(focal) == 2:
        x, mod = (p.x, next(v for v in focal if v != p.x)) if p.x in focal else focal
        s.append(f"This model examines whether the relationship between {dn(x)} and "
                 f"{'' if gauss else 'the probability of '}{y} depends on {dn(mod)}{adj}.")
    elif focal:
        s.append(f"This model examines whether the {dn(focal[0])} × {dn(focal[1])} interaction on {y} itself "
                 f"differs across {dn(focal[2])} (a three-way interaction){adj}.")
    else:
        s.append(f"This model relates {y} to {', '.join(dn(v) for v in m.variables)} without an interaction term.")
    ft = vals.get("focal")
    if focal and ft:
        stat = (f"F({int(ft['df1'])}, {int(ft['df2'])}) = {ft['statistic']:.2f}" if gauss
                else f"χ²({int(ft['df1'])}) = {ft['statistic']:.2f}")
        name = " × ".join(dn(v) for v in focal)
        if ft["p"] < 1 - p.conf:
            s.append(f"The {name} interaction was statistically significant ({stat}, {_fmt_p(ft['p'])}).")
        else:
            s.append(f"The {name} interaction was not statistically significant ({stat}, {_fmt_p(ft['p'])}); the "
                     "data do not show clear evidence that the relationship differs, which is not proof that it is the same.")
    slopes = next((t for t in result.tables if t.name in ("Conditional slopes", "Simple effects")), None)
    if slopes and focal and len(focal) == 2 and slopes.rows:
        cols = slopes.columns
        est_col = next((c for c in cols if c.endswith(".trend")), None) or \
            ("estimate" if "estimate" in cols else "odds.ratio" if "odds.ratio" in cols else None)
        lo = "lower.CL" if "lower.CL" in cols else "asymp.LCL" if "asymp.LCL" in cols else None
        hi = "upper.CL" if "upper.CL" in cols else "asymp.UCL" if "asymp.UCL" in cols else None
        if est_col and lo and hi:
            first, last = slopes.rows[0], slopes.rows[-1]
            lab_col = "label" if "label" in cols else cols[0]
            ix, il, ih, ilab = cols.index(est_col), cols.index(lo), cols.index(hi), cols.index(lab_col)
            unit = "" if gauss else (" (odds ratio)" if est_col == "odds.ratio" else " (log-odds)")
            what = (f"the slope of {y} on {dn(p.x)}" if est_col.endswith(".trend")
                    else f"the {slopes.rows[0][0]} difference")
            s.append(f"At {dn(p.group or p.panel)} = {first[ilab]}, {what} was {_n(first[ix])}{unit} "
                     f"(95% CI {_n(first[il])} to {_n(first[ih])}); at {last[ilab]} it was {_n(last[ix])} "
                     f"(95% CI {_n(last[il])} to {_n(last[ih])}).")
    jn = vals.get("jn")
    if jn and jn.get("text"):
        s.append(str(jn["text"]))
    if len(s) < 4:
        held = vals.get("held")
        s.append(f"Predictions use {result.n_used:,} complete cases"
                 + (f"; other predictors are held at {held}." if held and held != "none" else "."))
    lim = "These are associations in observational data, not causal effects."
    if m.selected_from:
        lim = (f"This model was selected from {m.selected_from:,} candidates, so its p-value and effect size are "
               "optimistic; treat it as a hypothesis to test in new data. " + lim)
    s.append(lim)
    return " ".join(s[:6])

