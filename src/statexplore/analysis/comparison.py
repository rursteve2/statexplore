"""Model Comparison (§14): side-by-side fit statistics; nested tests only where valid."""
from __future__ import annotations

import json

from ..core.dataset import Dataset
from ..core.schema import AnalysisResult
from ..rbridge.codegen import alias, r_str, r_vec
from .models import ModelCheckError, ModelSpec, check_model, model_view
from .session import RSession


class ComparisonError(ValueError):
    pass


def nested_pairs(models: list[ModelSpec]) -> list[tuple[int, int]]:
    """(i, j) where model i is nested in model j."""
    return [(i, j) for i, a in enumerate(models) for j, b in enumerate(models) if i != j and a.nested_in(b)]


def build(ds: Dataset, models: list[ModelSpec], common_sample: bool, plot_path: str | None = None) -> str:
    if len(models) < 2:
        raise ComparisonError("Select at least two models.")
    for m in models:
        try:
            check_model(ds, m)
        except ModelCheckError as e:
            raise ComparisonError(f"{m.label or m.display(ds)}: {e}") from None
    outcomes = {(m.outcome, m.family) for m in models}
    if len(outcomes) != 1:
        raise ComparisonError("All models must have the same outcome (and model family) to be compared.")
    a = lambda n: alias(ds.var(n).alias)  # noqa: E731
    outcome, family = next(iter(outcomes))
    gauss = family == "gaussian"
    y = a(outcome)
    all_vars: list[str] = []
    for m in models:
        for v in [m.outcome] + m.variables + m.sample_vars:
            if a(v) not in all_vars:
                all_vars.append(a(v))
    L = [f"sx_all <- {r_vec(all_vars)}"]
    if common_sample:
        L.append("sx_common <- sx_complete(dat, sx_all)")
    resp = y if gauss else "sx_event"
    L.append("sx_fits <- list(); sx_n <- integer()")
    for k, m in enumerate(models, start=1):
        need = [a(v) for v in dict.fromkeys([m.outcome] + m.variables + m.sample_vars)]
        data = "sx_common" if common_sample else f"sx_complete(dat, {r_vec(need)})"
        L.append(f"sx_dk <- {data}")
        if not gauss:
            L += [f"sx_lv <- if (is.factor(sx_dk${y})) levels(sx_dk${y}) else sort(unique(sx_dk${y}))",
                  f"sx_dk$sx_event <- as.integer(sx_dk${y} == sx_lv[2])"]
        fit = (f"stats::lm({resp} ~ {m.rhs(ds)}, data = sx_dk)" if gauss
               else f"stats::glm({resp} ~ {m.rhs(ds)}, data = sx_dk, family = stats::binomial())")
        L += [f"sx_fits[[{k}]] <- {fit}", f"sx_n[{k}] <- nrow(sx_dk)"]
    labels = [m.label or f"Model {k}" for k, m in enumerate(models, start=1)]
    L += [
        f"sx_labels <- {r_vec(labels)}",
        f"sx_formulas <- {r_vec([m.display(ds) for m in models])}",
        "sx_fit_tab <- data.frame(model = sx_labels, formula = sx_formulas, n = sx_n,",
        "  parameters = vapply(sx_fits, function(f) length(stats::coef(f)), 0),",
        "  logLik = vapply(sx_fits, function(f) as.numeric(stats::logLik(f)), 0),",
        "  AIC = vapply(sx_fits, stats::AIC, 0), BIC = vapply(sx_fits, stats::BIC, 0))",
    ]
    if gauss:
        L += [
            "sx_fit_tab$R2 <- vapply(sx_fits, function(f) summary(f)$r.squared, 0)",
            "sx_fit_tab$adj_R2 <- vapply(sx_fits, function(f) summary(f)$adj.r.squared, 0)",
            "sx_fit_tab$sigma <- vapply(sx_fits, function(f) summary(f)$sigma, 0)",
        ]
    else:
        L += [
            "sx_fit_tab$Tjur_R2 <- vapply(sx_fits, function(f) { p <- stats::fitted(f); y <- f$y; mean(p[y == 1]) - mean(p[y == 0]) }, 0)",
            "sx_fit_tab$McFadden_R2 <- vapply(sx_fits, function(f) 1 - f$deviance / f$null.deviance, 0)",
            "sx_fit_tab$converged <- vapply(sx_fits, function(f) f$converged, TRUE)",
        ]
    L += [
        "sx_fit_tab$delta_AIC <- sx_fit_tab$AIC - min(sx_fit_tab$AIC)",
        "sx_fit_tab$delta_BIC <- sx_fit_tab$BIC - min(sx_fit_tab$BIC)",
        "sx_tabs <- list(sx_table(\"Model fit\", sx_fit_tab, notes = paste0(\"AIC/BIC comparisons are only meaningful \",",
        "  \"for models fitted to the same observations. Lower is better; differences under about 2 are small. \",",
        "  \"Linear-model AIC/BIC count the residual variance as a parameter (R convention).\")))",
        "sx_checks <- character()",
        "if (length(unique(sx_n)) > 1) sx_checks <- c(sx_checks, paste0(\"The models use different observations (n = \",",
        "  paste(sx_n, collapse = \", \"), \"): AIC, BIC and R-squared are not comparable and nested tests are skipped.\"))",
    ]
    pairs = nested_pairs(models)
    if pairs:
        test = "\"F\"" if gauss else "\"Chisq\""
        L.append("sx_nest <- list()")
        for i, j in pairs:
            L += [
                f"if (sx_n[{i + 1}] == sx_n[{j + 1}]) {{",
                f"  sx_a <- stats::anova(sx_fits[[{i + 1}]], sx_fits[[{j + 1}]], test = {test})",
                f"  sx_nest[[length(sx_nest) + 1]] <- data.frame(smaller = sx_labels[{i + 1}], larger = sx_labels[{j + 1}],",
                "    df = sx_a$Df[2], statistic = " + ("sx_a$F[2]" if gauss else "sx_a$Deviance[2]")
                + ", p = sx_a[[ncol(sx_a)]][2])",
                "}",
            ]
        L += [
            "if (length(sx_nest)) sx_tabs[[2]] <- sx_table(\"Nested model tests\", do.call(rbind, sx_nest),",
            "  notes = " + r_str("F-test of the added terms (same observations)." if gauss
                                 else "Likelihood-ratio test of the added terms (same observations).") + ")",
        ]
    if plot_path:
        L += [
            "sx_pl <- data.frame(model = factor(sx_labels, levels = rev(sx_labels)), delta_AIC = sx_fit_tab$delta_AIC,",
            "  delta_BIC = sx_fit_tab$delta_BIC)",
            "sx_pl <- rbind(data.frame(model = sx_pl$model, criterion = \"AIC\", delta = sx_pl$delta_AIC),",
            "  data.frame(model = sx_pl$model, criterion = \"BIC\", delta = sx_pl$delta_BIC))",
            "sx_p <- ggplot2::ggplot(sx_pl, ggplot2::aes(x = delta, y = model, colour = criterion)) +",
            "  ggplot2::geom_point(size = 3) + ggplot2::geom_vline(xintercept = 2, linetype = 2, colour = \"grey60\") +",
            "  ggplot2::labs(title = \"Model comparison\", subtitle = \"Difference from the best model (0 = best; dashed line = 2)\",",
            "    x = \"Difference in information criterion\", y = NULL) + ggplot2::theme_minimal()",
            f"ggplot2::ggsave({r_str(plot_path)}, sx_p, width = 7, height = 1.5 + 0.5 * length(sx_labels), dpi = 120)",
        ]
    non_nested = len(models) * (len(models) - 1) // 2 - len(pairs)
    L += [
        "sx_result <- list(tables = sx_tabs, checks = sx_checks, n_used = max(sx_n), n_excluded = nrow(dat) - max(sx_n),",
        f"  notes = c({r_str(f'{len(pairs)} nested pair(s) tested; {non_nested} non-nested pair(s) compared by AIC/BIC only (no significance test).')}))",
    ]
    return "\n".join(L) + "\n"


def compare(session: RSession, ds: Dataset, models: list[ModelSpec],
            common_sample: bool = True) -> tuple[AnalysisResult, dict[str, bytes]]:
    from .framework import _alias_map, _as_list, _parse_table
    from ..core.schema import Status

    keys = {json.dumps(m.treatments, sort_keys=True, default=str) for m in models}
    if len(keys) > 1:
        raise ComparisonError("These models encode some variables differently (analysis-specific treatments). "
                              "Models compared by AIC or nested tests must use the same data coding; use the "
                              "Likert sensitivity comparison to contrast treatments.")
    ds = model_view(ds, models[0] if models else None)
    plot = session.new_plot_path()
    body = build(ds, models, common_sample, plot.as_posix())
    result, raw = session.run(ds, kind="model_comparison", title="Model comparison", body=body,
                              packages=["stats", "ggplot2"], inputs={"models": [m.to_dict() for m in models],
                                                          "common_sample": common_sample})
    from .measurement import record

    record(result, ds)
    result.missing_data = ("Rows complete on every model's variables (common sample)" if common_sample
                           else "Each model uses its own complete cases")
    if raw is None:
        return result, {}
    names = _alias_map(ds)
    for t in _as_list(raw.get("tables")):
        result.tables.append(_parse_table(t, names))
    result.n_used = raw.get("n_used")
    result.values["notes"] = _as_list(raw.get("notes"))
    for c in _as_list(raw.get("checks")):
        result.messages.append(f"Check: {c}")
        result.status = Status.WARNING
    artifacts: dict[str, bytes] = {}
    if plot.is_file():
        key = f"plots/{plot.name}"
        artifacts[key] = plot.read_bytes()
        result.plots.append(key)
    return result, artifacts
