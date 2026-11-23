"""Model Comparison (§14): side-by-side fit statistics; nested tests only where valid."""
from __future__ import annotations

import json

from ..core.dataset import Dataset
from ..core.schema import AnalysisResult
from ..rbridge.codegen import alias, r_num, r_str, r_vec
from .models import ModelCheckError, ModelSpec, check_model, model_view
from .session import RSession


class ComparisonError(ValueError):
    pass


FAMILY_LABELS = {"gaussian": "linear (stats::lm)", "binomial": "binary logistic (stats::glm)",
                 "ordinal": "ordinal cumulative logit (ordinal::clm)",
                 "multinomial": "multinomial baseline-category logit (nnet::multinom)"}
FIT_NOTES = {
    "gaussian": "Linear-model AIC/BIC count the residual variance as a parameter (R convention).",
    "binomial": "Binary logistic ML fits; parameters = coefficients.",
    "ordinal": "Ordinal ML fits; parameters = coefficients + thresholds (cut-points).",
    "multinomial": "Multinomial ML fits; parameters = (categories - 1) x coefficients.",
}


def _outcome_coding(ds: Dataset, m: ModelSpec, y: str) -> list[str]:
    """sx_code_y(d): the outcome coded exactly as the model's family needs it (same event / order / reference
    for every model compared)."""
    from .eda import category_codes

    v = ds.var(m.outcome)
    if m.family == "gaussian":
        return ["sx_code_y <- function(d) d"]
    if m.family == "binomial":
        if m.event is None:
            ev = ("sx_lv <- if (is.factor(d$Y)) levels(droplevels(d$Y)) else sort(unique(d$Y))\n"
                  "  d$sx_event <- as.integer(d$Y == sx_lv[2])  # event = second category (default)")
        elif v.is_categorical:
            ev = (f"d$sx_event <- as.integer(as.character(d$Y) == {r_str(ds.level_label(m.outcome, m.event))})"
                  "  # event category of the source model")
        else:
            ev = f"d$sx_event <- as.integer(d$Y == {r_num(m.event)})  # event category of the source model"
        return ["sx_code_y <- function(d) {", "  " + ev.replace("$Y", f"${y}"), "  d", "}"]
    _codes, labels = category_codes(ds, m.outcome)
    if m.family == "ordinal":
        return ["sx_code_y <- function(d) {",
                f"  d${y} <- droplevels(factor(as.character(d${y}), levels = {r_vec(labels)}, ordered = TRUE))",
                "  d", "}"]
    ref = ds.level_label(m.outcome, m.reference) if m.reference is not None else labels[0]
    return ["sx_code_y <- function(d) {",
            f"  d${y} <- stats::relevel(droplevels(factor(as.character(d${y}), levels = {r_vec(labels)})), "
            f"ref = {r_str(ref)})", "  d", "}"]


def _fit_call(family: str, rhs: str) -> str:
    y = "sx_event" if family == "binomial" else None
    if family == "gaussian":
        return f"stats::lm(sx_yname ~ {rhs}, data = sx_dk)"
    if family == "binomial":
        return f"stats::glm({y} ~ {rhs}, data = sx_dk, family = stats::binomial())"
    if family == "ordinal":
        return f"ordinal::clm(sx_yname ~ {rhs}, data = sx_dk, link = \"logit\")"
    return f"nnet::multinom(sx_yname ~ {rhs}, data = sx_dk, trace = FALSE, maxit = 1000)"


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
        raise ComparisonError("All models must have the same outcome and model family to be compared. "
                              + "; ".join(f"{m.label or m.display(ds)}: {FAMILY_LABELS.get(m.family, m.family)}"
                                          for m in models))
    outcome, family = next(iter(outcomes))
    if family not in FAMILY_LABELS:
        raise ComparisonError(f"Model Comparison cannot fit {family!r} models; the model was not replaced by another "
                              "family.")
    codings = {(repr(m.event), repr(m.reference)) for m in models}
    if len(codings) != 1:
        raise ComparisonError("These models code the outcome differently (event or reference category): their "
                              "likelihoods are not comparable.")
    a = lambda n: alias(ds.var(n).alias)  # noqa: E731
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
    L += _outcome_coding(ds, models[0], y)
    L.append("sx_fits <- list(); sx_n <- integer()")
    for k, m in enumerate(models, start=1):
        need = [a(v) for v in dict.fromkeys([m.outcome] + m.variables + m.sample_vars)]
        data = "sx_common" if common_sample else f"sx_complete(dat, {r_vec(need)})"
        L += [f"sx_dk <- sx_code_y({data})", f"sx_fits[[{k}]] <- {_fit_call(family, m.rhs(ds)).replace('sx_yname', y)}",
              f"sx_n[{k}] <- nrow(sx_dk)"]
    labels = [m.label or f"Model {k}" for k, m in enumerate(models, start=1)]
    L += [
        f"sx_labels <- {r_vec(labels)}",
        f"sx_formulas <- {r_vec([m.display(ds) for m in models])}",
        "sx_fit_tab <- data.frame(model = sx_labels, formula = sx_formulas, n = sx_n,",
        "  parameters = vapply(sx_fits, function(f) attr(stats::logLik(f), \"df\"), 0),",
        "  logLik = vapply(sx_fits, function(f) as.numeric(stats::logLik(f)), 0),",
        "  AIC = vapply(sx_fits, stats::AIC, 0), BIC = vapply(sx_fits, stats::BIC, 0))",
    ]
    if gauss:
        L += [
            "sx_fit_tab$R2 <- vapply(sx_fits, function(f) summary(f)$r.squared, 0)",
            "sx_fit_tab$adj_R2 <- vapply(sx_fits, function(f) summary(f)$adj.r.squared, 0)",
            "sx_fit_tab$sigma <- vapply(sx_fits, function(f) summary(f)$sigma, 0)",
        ]
    elif family == "binomial":
        L += [
            "sx_fit_tab$Tjur_R2 <- vapply(sx_fits, function(f) { p <- stats::fitted(f); y <- f$y; mean(p[y == 1]) - mean(p[y == 0]) }, 0)",
            "sx_fit_tab$McFadden_R2 <- vapply(sx_fits, function(f) 1 - f$deviance / f$null.deviance, 0)",
            "sx_fit_tab$converged <- vapply(sx_fits, function(f) f$converged, TRUE)",
        ]
    elif family == "ordinal":
        L += [
            "sx_fit_tab$thresholds <- vapply(sx_fits, function(f) length(f$alpha), 0)",
            "sx_fit_tab$converged <- vapply(sx_fits, function(f) identical(as.integer(f$convergence$code), 0L), TRUE)",
        ]
    else:
        L += ["sx_fit_tab$converged <- vapply(sx_fits, function(f) f$convergence == 0, TRUE)"]
    L += [
        "sx_fit_tab$delta_AIC <- sx_fit_tab$AIC - min(sx_fit_tab$AIC)",
        "sx_fit_tab$delta_BIC <- sx_fit_tab$BIC - min(sx_fit_tab$BIC)",
        "sx_tabs <- list(sx_table(\"Model fit\", sx_fit_tab, notes = paste0(\"AIC/BIC comparisons are only meaningful \",",
        "  \"for models fitted to the same observations. Lower is better; differences under about 2 are small. \",",
        "  " + r_str(FIT_NOTES[family]) + ")))",
        "sx_checks <- character()",
        "if (length(unique(sx_n)) > 1) sx_checks <- c(sx_checks, paste0(\"The models use different observations (n = \",",
        "  paste(sx_n, collapse = \", \"), \"): AIC, BIC and R-squared are not comparable and nested tests are skipped.\"))",
    ]
    pairs = nested_pairs(models)
    if pairs:
        L.append("sx_nest <- list()")
        for i, j in pairs:
            L += [f"if (sx_n[{i + 1}] == sx_n[{j + 1}]) {{"]
            if gauss:
                L += [f"  sx_a <- stats::anova(sx_fits[[{i + 1}]], sx_fits[[{j + 1}]], test = \"F\")",
                      "  sx_df <- sx_a$Df[2]; sx_stat <- sx_a$F[2]; sx_p <- sx_a[[ncol(sx_a)]][2]"]
            else:   # likelihood-ratio statistic from the two ML log-likelihoods (glm, clm, multinom alike)
                L += [f"  sx_l0 <- stats::logLik(sx_fits[[{i + 1}]]); sx_l1 <- stats::logLik(sx_fits[[{j + 1}]])",
                      "  sx_stat <- 2 * (as.numeric(sx_l1) - as.numeric(sx_l0)); sx_df <- attr(sx_l1, \"df\") - attr(sx_l0, \"df\")",
                      "  sx_p <- stats::pchisq(max(sx_stat, 0), sx_df, lower.tail = FALSE)"]
            L += [f"  sx_nest[[length(sx_nest) + 1]] <- data.frame(smaller = sx_labels[{i + 1}], larger = sx_labels[{j + 1}],",
                  "    df = sx_df, statistic = sx_stat, p = sx_p)",
                  "}"]
        L += [
            "if (length(sx_nest)) sx_tabs[[2]] <- sx_table(\"Nested model tests\", do.call(rbind, sx_nest),",
            "  notes = " + r_str("F-test of the added terms (same observations)." if gauss
                                 else f"Likelihood-ratio test of the added terms (same observations; {FAMILY_LABELS[family]} "
                                      "ML fits; df = difference in estimated parameters).") + ")",
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


def _check_identities(result: AnalysisResult, models: list[ModelSpec]) -> None:
    """Each unedited discovery candidate is checked against the fit the search reported (P0-01)."""
    from ..core.schema import Status
    from .descriptor import verify

    fit = next((t for t in result.tables if t.name == "Model fit"), None)
    if fit is None:
        return
    col = {c: i for i, c in enumerate(fit.columns)}
    for k, m in enumerate(models):
        if not m.descriptor or k >= len(fit.rows):
            continue
        row = fit.rows[k]
        ok, text = verify(m.descriptor, row[col["n"]] if "n" in col else None,
                          row[col["logLik"]] if "logLik" in col else None, row[col["AIC"]] if "AIC" in col else None)
        if text:
            result.messages.append(f"{m.label}: {text}" if ok else f"Check: {m.label}: {text}")
        if ok is False:
            result.status = Status.WARNING


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
                              packages=["stats", "ggplot2"] + {"ordinal": ["ordinal"], "multinomial": ["nnet"]}.get(
                                  models[0].family if models else "", []), inputs={"models": [m.to_dict() for m in models],
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
    _check_identities(result, models)
    artifacts: dict[str, bytes] = {}
    if plot.is_file():
        key = f"plots/{plot.name}"
        artifacts[key] = plot.read_bytes()
        result.plots.append(key)
    return result, artifacts
