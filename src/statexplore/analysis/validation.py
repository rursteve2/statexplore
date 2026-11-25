"""Validation and post-selection uncertainty (§12.5, §13): cross-validation (standard,
grouped, temporal), holdout evaluation with replication test, discovery stability by
resampling, full-procedure CV (selection inside training folds), prediction intervals."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.dataset import Dataset
from ..core.schema import AnalysisResult, Status
from ..core.variables import Role
from ..rbridge.codegen import alias, r_num, r_str, r_vec
from .models import ModelCheckError, ModelSpec, check_model, model_view
from .session import RSession

R_VALIDATION_HELPERS = r"""# ---- StatExplore: validation helpers ----
sx_auc <- function(y, p) { r <- rank(p); n1 <- sum(y == 1); n0 <- sum(y == 0)
  if (n1 == 0 || n0 == 0) return(NA_real_); (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0) }
sx_metrics <- function(y, pred, family) {
  if (family == "gaussian") return(c(RMSE = sqrt(mean((y - pred)^2)), MAE = mean(abs(y - pred)),
    R2 = 1 - sum((y - pred)^2) / sum((y - mean(y))^2)))
  p <- pmin(pmax(pred, 1e-12), 1 - 1e-12)
  c(log_loss = -mean(y * log(p) + (1 - y) * log(1 - p)), Brier = mean((y - p)^2), AUC = sx_auc(y, p))
}
# Drop test rows whose factor levels were not seen in training (cannot be predicted).
sx_seen <- function(train, test, vars) {
  ok <- rep(TRUE, nrow(test))
  for (v in vars) if (is.factor(test[[v]])) ok <- ok & as.character(test[[v]]) %in% unique(as.character(train[[v]]))
  ok
}
"""

SCHEMES = {"kfold": "k-fold (random rows)", "grouped": "Grouped k-fold (whole participants)",
           "temporal": "Temporal (train on earlier occasions, test on the next)"}


class ValidationInputError(ValueError):
    pass


def _id_var(ds: Dataset) -> str | None:
    return next((v.name for v in ds.variables.values() if v.role is Role.ID), None)


def _time_var(ds: Dataset) -> str | None:
    return next((v.name for v in ds.variables.values() if v.role is Role.TIME), None)


def _model_lines(ds: Dataset, m: ModelSpec, data_expr: str, extra: list[str], reuse_levels: bool = False) -> list[str]:
    if data_expr == "dat":
        try:
            check_model(ds, m)
        except ModelCheckError as e:
            raise ValidationInputError(str(e)) from None
    """Complete-case data `sx_d` (model vars + extra columns) and the response column `sx_y`."""
    a = lambda n: alias(ds.var(n).alias)  # noqa: E731
    used = list(dict.fromkeys([a(m.outcome)] + [a(v) for v in m.variables] + [a(v) for v in m.sample_vars]))
    y = a(m.outcome)
    L = [f"sx_used <- {r_vec(used)}",
         f"sx_keep <- {r_vec(list(dict.fromkeys(used + [e for e in extra if e not in used])))}",
         f"sx_d <- sx_complete({data_expr}, sx_keep)  # model variables + ID/time needed for splitting"]
    if m.family == "gaussian":
        L.append(f"sx_d$sx_y <- sx_d${y}")
    else:
        if not reuse_levels:  # the event level is fixed by the training data
            L.append(f"sx_lv <- if (is.factor(sx_d${y})) levels(sx_d${y}) else sort(unique(sx_d${y}))")
        L.append(f"sx_d$sx_y <- as.integer(as.character(sx_d${y}) == as.character(sx_lv[2]))")
    fam = "stats::gaussian()" if m.family == "gaussian" else "stats::binomial()"
    L += [f"sx_form <- sx_y ~ {m.rhs(ds)}",
          f"sx_fitfun <- function(d) stats::glm(sx_form, data = d, family = {fam})",
          f"sx_family <- {r_str(m.family)}",
          f"sx_mvars <- {r_vec([a(v) for v in m.variables])}"]
    return L


def _finish(session: RSession, ds: Dataset, kind: str, title: str, body: str, inputs: dict[str, Any],
            packages: list[str]) -> tuple[AnalysisResult, dict[str, bytes], Any]:
    from .framework import _alias_map, _as_list, _parse_table

    from .measurement import record

    result, raw = session.run(ds, kind=kind, title=title, body=R_VALIDATION_HELPERS + body, packages=packages,
                              inputs=inputs)
    record(result, ds)
    artifacts: dict[str, bytes] = {}
    if raw is None:
        return result, artifacts, None
    names = _alias_map(ds)
    for t in _as_list(raw.get("tables")):
        result.tables.append(_parse_table(t, names))
    result.n_used = raw.get("n_used")
    result.values["notes"] = [str(n) for n in _as_list(raw.get("notes"))]
    for c in _as_list(raw.get("checks")):
        result.messages.append(f"Check: {c}")
        if result.status is Status.OK:
            result.status = Status.WARNING
    return result, artifacts, raw


# ---- cross-validation ------------------------------------------------------------------

def cross_validate(session: RSession, ds: Dataset, m: ModelSpec, scheme: str = "kfold", k: int = 5,
                   seed: int = 1) -> tuple[AnalysisResult, dict[str, bytes]]:
    ds = model_view(ds, m)
    if scheme not in SCHEMES:
        raise ValidationInputError("Unknown cross-validation scheme.")
    idv, tv = _id_var(ds), _time_var(ds)
    if scheme == "grouped" and not idv:
        raise ValidationInputError("Grouped cross-validation needs a variable with the participant-ID role.")
    if scheme == "temporal" and not tv:
        raise ValidationInputError("Temporal validation needs a variable with the time role.")
    a = lambda n: alias(ds.var(n).alias)  # noqa: E731
    extra = ([a(idv)] if idv else []) + ([a(tv)] if tv and scheme == "temporal" else [])
    L = _model_lines(ds, m, "dat", extra)
    L.append(f"set.seed({int(seed)})")
    if scheme == "kfold":
        L += [f"sx_fold <- sample(rep(seq_len({k}), length.out = nrow(sx_d)))", f"sx_k <- {k}"]
    elif scheme == "grouped":
        L += [f"sx_g <- as.character(sx_d${a(idv)})", "sx_ug <- unique(sx_g)",
              f"sx_gf <- sample(rep(seq_len({k}), length.out = length(sx_ug)))",
              "sx_fold <- sx_gf[match(sx_g, sx_ug)]", f"sx_k <- {k}"]
    else:
        L += [f"sx_t <- sx_d${a(tv)}", "sx_tl <- if (is.factor(sx_t)) levels(droplevels(sx_t)) else sort(unique(sx_t))",
              "if (length(sx_tl) < 3) stop(\"Temporal validation needs at least 3 occasions.\")",
              "sx_k <- length(sx_tl) - 1"]
    L += [
        "sx_rows <- list(); sx_overlap <- integer(); sx_dropped <- 0",
        "for (sx_f in seq_len(sx_k)) {",
    ]
    if scheme == "temporal":
        L += ["  sx_pos <- match(as.character(sx_t), as.character(sx_tl))",
              "  sx_train <- sx_d[sx_pos <= sx_f, , drop = FALSE]; sx_test <- sx_d[sx_pos == sx_f + 1, , drop = FALSE]",
              "  sx_label <- paste0(\"train <= \", sx_tl[sx_f], \", test = \", sx_tl[sx_f + 1])"]
    else:
        L += ["  sx_train <- sx_d[sx_fold != sx_f, , drop = FALSE]; sx_test <- sx_d[sx_fold == sx_f, , drop = FALSE]",
              "  sx_label <- paste(\"fold\", sx_f)"]
    L += [
        "  sx_ok <- sx_seen(sx_train, sx_test, sx_mvars); sx_dropped <- sx_dropped + sum(!sx_ok); sx_test <- sx_test[sx_ok, , drop = FALSE]",
        "  sx_m <- sx_fitfun(sx_train)",
        "  sx_pred <- stats::predict(sx_m, newdata = sx_test, type = \"response\")",
        "  sx_rows[[sx_f]] <- data.frame(split = sx_label, n_train = nrow(sx_train), n_test = nrow(sx_test),",
        "    t(sx_metrics(sx_test$sx_y, sx_pred, sx_family)))",
    ]
    if idv:
        L.append(f"  sx_overlap[sx_f] <- length(intersect(unique(as.character(sx_train${a(idv)})), unique(as.character(sx_test${a(idv)}))))")
    L += [
        "}",
        "sx_cv <- do.call(rbind, sx_rows)",
        "sx_mcols <- setdiff(names(sx_cv), c(\"split\", \"n_train\", \"n_test\"))",
        "sx_full <- sx_fitfun(sx_d)",
        "sx_in <- sx_metrics(sx_d$sx_y, stats::fitted(sx_full), sx_family)",
        "sx_sum <- data.frame(metric = sx_mcols, cv_mean = colMeans(sx_cv[sx_mcols]), cv_sd = apply(sx_cv[sx_mcols], 2, stats::sd),",
        "  in_sample = unname(sx_in[sx_mcols]))",
        f"sx_tabs <- list(sx_table(\"Cross-validation summary\", sx_sum, notes = paste0({r_str(SCHEMES[scheme])}, \". In-sample \",",
        "  \"values use the same rows the model was fitted to; the gap to cross-validated values is the optimism.\")),",
        "  sx_table(\"Per-fold results\", sx_cv))",
    ]
    if idv:
        L += ["sx_tabs[[3]] <- sx_table(\"Participant overlap between training and test\", data.frame(split = sx_cv$split,",
              "  participants_in_both = sx_overlap), notes = paste0(\"Non-zero values mean the same participants appear in \",",
              "  \"training and test folds: errors are optimistic for new participants (data leakage). Use grouped CV.\"))",
              "sx_checks <- if (any(sx_overlap > 0)) \"Participants appear in both training and test folds (leakage for "
              "generalization to new participants).\" else character()"]
    else:
        L.append("sx_checks <- character()")
    L += ["if (sx_dropped > 0) sx_checks <- c(sx_checks, paste(sx_dropped, \"test rows had factor levels unseen in training and were not predicted.\"))",
          "sx_result <- list(tables = sx_tabs, checks = sx_checks, n_used = nrow(sx_d))"]
    res, art, _ = _finish(session, ds, "cross_validation", f"Cross-validation: {m.display(ds)}", "\n".join(L) + "\n",
                          {"model": m.to_dict(), "scheme": scheme, "k": k, "seed": seed}, ["stats"])
    res.seed = seed
    return res, art


# ---- holdout ----------------------------------------------------------------------------

def holdout_evaluate(session: RSession, ds: Dataset, m: ModelSpec) -> tuple[AnalysisResult, dict[str, bytes]]:
    ds = model_view(ds, m)
    if ds.holdout is None:
        raise ValidationInputError("No holdout has been reserved.")
    focal = ":".join(alias(ds.var(v).alias) for v in m.focal) if m.focal else None
    L = _model_lines(ds, m, "dat", [])
    L += [
        "sx_train <- sx_d; sx_m <- sx_fitfun(sx_train)",
    ] + _model_lines(ds, m, "sx_holdout_dat", [], reuse_levels=True) + [
        "sx_test <- sx_d",
        "sx_ok <- sx_seen(sx_train, sx_test, sx_mvars); sx_test <- sx_test[sx_ok, , drop = FALSE]",
        "sx_pred <- stats::predict(sx_m, newdata = sx_test, type = \"response\")",
        "sx_out <- sx_metrics(sx_test$sx_y, sx_pred, sx_family)",
        "sx_in <- sx_metrics(sx_train$sx_y, stats::fitted(sx_m), sx_family)",
        "sx_tabs <- list(sx_table(\"Holdout performance\", data.frame(metric = names(sx_out), training = unname(sx_in),",
        "  holdout = unname(sx_out)), notes = \"Model fitted on the training rows only, then applied to the reserved rows.\"))",
    ]
    if focal:
        test = "\"F\"" if m.family == "gaussian" else "\"Chisq\""
        L += [
            "# Replication: does the interaction hold up when the same model is fitted to the holdout alone?",
            "sx_rep <- function(d) { full <- sx_fitfun(d); red <- stats::update(full, . ~ . - " + focal + ")",
            f"  a <- stats::anova(red, full, test = {test}); c(df = a$Df[2], p = a[[ncol(a)]][2]) }}",
            "sx_r_train <- sx_rep(sx_train); sx_r_hold <- sx_rep(sx_test)",
            "sx_tabs[[2]] <- sx_table(\"Interaction replication\", data.frame(sample = c(\"training\", \"holdout\"),",
            "  n = c(nrow(sx_train), nrow(sx_test)), df = c(sx_r_train[[\"df\"]], sx_r_hold[[\"df\"]]),",
            "  p = c(sx_r_train[[\"p\"]], sx_r_hold[[\"p\"]])), notes = paste0(\"The holdout p-value is a fresh test: the \",",
            "  \"holdout played no part in choosing this model. It is smaller-sample, so a non-significant result is not \",",
            "  \"proof the effect is absent.\"))",
        ]
    L.append("sx_result <- list(tables = sx_tabs, n_used = nrow(sx_test), checks = if (sum(!sx_ok)) paste(sum(!sx_ok), "
             "\"holdout rows had unseen factor levels and were not predicted.\") else character())")
    ds.holdout.evaluations += 1
    res, art, _ = _finish(session, ds, "holdout_evaluation", f"Holdout evaluation: {m.display(ds)}",
                          "\n".join(L) + "\n", {"model": m.to_dict(), "evaluation_number": ds.holdout.evaluations},
                          ["stats"])
    if ds.holdout.evaluations > 1:
        res.messages.append(f"Check: the holdout has now been used {ds.holdout.evaluations} times. Each extra "
                            "look makes it less of an independent test (it starts to act as a tuning set).")
        if res.status is Status.OK:
            res.status = Status.WARNING
    return res, art


# ---- discovery stability & full-procedure CV -------------------------------------------------

@dataclass
class ResampleSpec:
    resamples: int = 20
    fraction: float = 0.8
    seed: int = 1
    folds: int = 5
    max_candidates: int = 2000
    target: str = "auto"      # longitudinal procedure CV: auto | cluster | participant | temporal (CD-11)


TARGETS = {
    "cluster": "New organizations: whole clusters held out; population-level predictions (fixed effects only)",
    "participant": "New participants in known organizations: whole participants held out; predictions use the "
                   "estimated cluster effects of the training data (participant effects unknown)",
    "temporal": "Future visits of known participants: each fold trains on earlier occasions only and predicts the next "
                "occasion with the participants' training-data random effects",
}


def _discovery_setup(ds: Dataset, run: Any) -> tuple[str, list[Any], str | None]:
    from ..discovery.rcode import candidate_args, setup_script
    from ..discovery.space import enumerate_candidates

    s = run.settings
    if run.family in ("lmm", "glmm", "clmm", "gee"):
        raise ValidationInputError("Resampling stability is available for single-level (non-mixed) searches.")
    if run.family in ("ordinal", "multinomial"):
        raise ValidationInputError("Resampling stability and procedure cross-validation are available for Gaussian "
                                   "and binary searches; not yet for ordinal or multinomial outcomes.")
    if run.total > 2000:
        raise ValidationInputError("Resampling repeats the whole search many times: limit it to searches with at "
                                   "most 2,000 candidates.")
    cands = list(enumerate_candidates(s))
    idv = _id_var(ds)
    setup = setup_script(ds, s, run.family).replace("sx_result <- list(", "sx_setup_info <- list(")
    lines = [setup]
    if idv:
        lines.append(f"sx_dd$sx_group <- as.character(dat[rownames(sx_dd), {r_str(alias(ds.var(idv).alias))}])")
    else:
        lines.append("sx_dd$sx_group <- rownames(sx_dd)")
    lines.append("sx_cands <- list(" + ",\n  ".join(f"list({candidate_args(ds, c)})" for c in cands) + ")")
    lines.append("sx_pvals <- function() vapply(sx_cands, function(a) tryCatch(do.call(sx_discover_one, a)$p, "
                 "error = function(e) NA_real_), 0)")
    return "\n".join(lines) + "\n", cands, idv


def stability(session: RSession, ds: Dataset, run: Any, rs: ResampleSpec) -> tuple[AnalysisResult, dict[str, bytes]]:
    from ..discovery.space import focal_label, view

    ds = view(ds, run.settings)

    body, cands, idv = _discovery_setup(ds, run)
    labels = [focal_label(ds, c) + (f" | adj. {', '.join(ds.var(v).display_name for v in c.covariates)}" if c.covariates else "")
              for c in cands]
    body += "\n".join([
        f"sx_labels <- {r_vec(labels)}",
        f"set.seed({int(rs.seed)})",
        "sx_full <- sx_dd; sx_units <- unique(sx_full$sx_group)",
        f"sx_P <- matrix(NA_real_, {int(rs.resamples)}, length(sx_cands))",
        f"for (sx_b in seq_len({int(rs.resamples)})) {{",
        f"  sx_take <- sample(sx_units, floor({r_num(rs.fraction)} * length(sx_units)))",
        "  sx_dd <- sx_full[sx_full$sx_group %in% sx_take, , drop = FALSE]",
        "  sx_P[sx_b, ] <- sx_pvals()",
        "}",
        "sx_dd <- sx_full",
        "sx_p_full <- sx_pvals()",
        "sx_top <- apply(sx_P, 1, function(p) if (all(is.na(p))) NA_integer_ else which.min(p))",
        f"sx_idx <- {r_vec([float(c.index) for c in cands])}",
        "sx_freq <- data.frame(candidate = sx_idx, model = sx_labels, p_full_data = sx_p_full,",
        f"  selected_significant = colMeans(sx_P < {r_num(run.settings.alpha)}, na.rm = TRUE),",
        "  ranked_first = vapply(seq_along(sx_cands), function(j) mean(sx_top == j, na.rm = TRUE), 0))",
        "sx_freq <- sx_freq[order(-sx_freq$selected_significant, sx_freq$p_full_data), ]",
        "sx_result <- list(tables = list(sx_table(\"Discovery stability\", sx_freq, notes = paste0(\"Share of \",",
        f"  {int(rs.resamples)}, \" subsamples (\", {int(round(rs.fraction * 100))}, \"% of \", "
        + ("\"participants\"" if idv else "\"rows\"") + ", \") in which each model's interaction had p < \", "
        + r_num(run.settings.alpha) + ", \", and how often it ranked first. A frequency is not a p-value: \",",
        "  \"it shows how much the discovery depends on which observations happened to be sampled.\"))),",
        "  n_used = nrow(sx_full))",
    ]) + "\n"
    res, art, raw = _finish(session, ds, "discovery_stability", "Discovery stability (resampling)", body,
                            {"run": run.id, "resample": vars(rs)}, ["detectseparation"])
    res.seed = rs.seed
    if raw is not None and res.tables:
        t = res.tables[0]
        ci, si, fi = t.columns.index("candidate"), t.columns.index("selected_significant"), t.columns.index("ranked_first")
        for row in t.rows:
            run.validation[int(row[ci])] = (f"stability: p < {run.settings.alpha:g} in {row[si]:.0%} of "
                                            f"{rs.resamples} resamples, ranked first in {row[fi]:.0%}")
    return res, art


def procedure_cv(session: RSession, ds: Dataset, run: Any, rs: ResampleSpec) -> tuple[AnalysisResult, dict[str, bytes]]:
    """Repeat the whole search inside each training fold; evaluate the selected model on the test fold."""
    from ..discovery.space import focal_label, model_terms, view

    if run.family == "lmm":
        return procedure_cv_longitudinal(session, ds, run, rs)
    ds = view(ds, run.settings)

    body, cands, idv = _discovery_setup(ds, run)
    labels = [focal_label(ds, c) for c in cands]

    def rhs(c) -> str:
        full, _ = model_terms(c)
        return " + ".join(":".join(alias(ds.var(v).alias) for v in t) for t in full)
    fam = "stats::gaussian()" if run.family == "gaussian" else "stats::binomial()"
    body += "\n".join([
        f"sx_labels <- {r_vec(labels)}",
        f"sx_rhs <- {r_vec([rhs(c) for c in cands])}",
        f"sx_family <- {r_str(run.family)}",
        f"set.seed({int(rs.seed)})",
        "sx_full <- sx_dd; sx_units <- unique(sx_full$sx_group)",
        f"sx_gf <- sample(rep(seq_len({int(rs.folds)}), length.out = length(sx_units)))",
        "sx_fold <- sx_gf[match(sx_full$sx_group, sx_units)]",
        "sx_rows <- list()",
        f"for (sx_f in seq_len({int(rs.folds)})) {{",
        "  sx_dd <- sx_full[sx_fold != sx_f, , drop = FALSE]      # selection uses the training fold only",
        "  sx_p <- sx_pvals(); sx_j <- which.min(sx_p)",
        "  sx_train <- sx_dd; sx_test <- sx_full[sx_fold == sx_f, , drop = FALSE]",
        f"  sx_m <- stats::glm(stats::as.formula(paste(\"sx_y ~\", sx_rhs[sx_j])), data = sx_train, family = {fam})",
        "  sx_vars <- all.vars(stats::as.formula(paste(\"~\", sx_rhs[sx_j])))",
        "  sx_test <- sx_complete(sx_test, c(\"sx_y\", sx_vars)); sx_test <- sx_test[sx_seen(sx_train, sx_test, sx_vars), , drop = FALSE]",
        "  sx_pred <- stats::predict(sx_m, newdata = sx_test, type = \"response\")",
        "  sx_dd <- sx_test; sx_p_test <- tryCatch(do.call(sx_discover_one, sx_cands[[sx_j]])$p, error = function(e) NA_real_)",
        "  sx_rows[[sx_f]] <- data.frame(fold = sx_f, selected = sx_labels[sx_j], p_training = sx_p[sx_j],",
        "    p_test_fold = sx_p_test, n_test = nrow(sx_test), t(sx_metrics(sx_test$sx_y, sx_pred, sx_family)))",
        "}",
        "sx_dd <- sx_full",
        "sx_cvp <- do.call(rbind, sx_rows)",
        "sx_sel <- as.data.frame(table(selected = sx_cvp$selected), stringsAsFactors = FALSE)",
        "sx_result <- list(tables = list(",
        "  sx_table(\"Full-procedure cross-validation\", sx_cvp, notes = paste0(\"In each fold the entire search ran on \",",
        "    \"the training part; the top-ranked model was then evaluated on the held-out fold. p_test_fold is a fresh \",",
        "    \"test of the selected interaction in data that played no part in selecting it.\")),",
        "  sx_table(\"Models selected across folds\", sx_sel[order(-sx_sel$Freq), , drop = FALSE],",
        "    notes = \"If different folds select different interactions, the discovery is unstable.\")),",
        "  n_used = nrow(sx_full))",
    ]) + "\n"
    res, art, _ = _finish(session, ds, "procedure_cv", "Full-procedure cross-validation", body,
                          {"run": run.id, "resample": vars(rs)}, ["detectseparation"])
    res.seed = rs.seed
    return res, art


# ---- prediction uncertainty -------------------------------------------------------------------

def prediction_uncertainty(session: RSession, ds: Dataset, m: ModelSpec, x: str | None = None,
                           conf: float = 0.95) -> tuple[AnalysisResult, dict[str, bytes]]:
    ds = model_view(ds, m)
    x = x or (m.focal[0] if m.focal else m.variables[0])
    if x not in m.variables:
        raise ValidationInputError("The x variable must be in the model.")
    a = lambda n: alias(ds.var(n).alias)  # noqa: E731
    xa = a(x)
    others = [a(v) for v in m.variables if v != x]
    p1, p2 = session.new_plot_path(), session.new_plot_path()
    L = _model_lines(ds, m, "dat", []) + [
        "sx_m <- sx_fitfun(sx_d)",
        f"sx_xv <- if (is.factor(sx_d${xa})) factor(levels(sx_d${xa}), levels = levels(sx_d${xa})) else "
        f"seq(min(sx_d${xa}), max(sx_d${xa}), length.out = 60)",
        f"sx_held <- lapply(sx_d[, {r_vec(others) if others else 'character(0)'}, drop = FALSE],",
        "  function(v) if (is.factor(v)) factor(levels(v)[1], levels = levels(v)) else mean(v))",
        f"sx_nd <- do.call(data.frame, c(list({xa} = sx_xv), sx_held))",
    ]
    if m.family == "gaussian":
        L += [
            "sx_lm <- stats::lm(sx_form, data = sx_d)",
            f"sx_ci <- stats::predict(sx_lm, newdata = sx_nd, interval = \"confidence\", level = {r_num(conf)})",
            f"sx_pi <- stats::predict(sx_lm, newdata = sx_nd, interval = \"prediction\", level = {r_num(conf)})",
            "sx_nd$predicted <- sx_ci[, 1]; sx_nd$ci_low <- sx_ci[, 2]; sx_nd$ci_high <- sx_ci[, 3]",
            "sx_nd$pi_low <- sx_pi[, 2]; sx_nd$pi_high <- sx_pi[, 3]",
            f"sx_p <- ggplot2::ggplot(sx_nd, ggplot2::aes(x = {xa}, y = predicted, group = 1)) +",
            "  ggplot2::geom_ribbon(ggplot2::aes(ymin = pi_low, ymax = pi_high), fill = \"#fdae61\", alpha = 0.25) +",
            "  ggplot2::geom_ribbon(ggplot2::aes(ymin = ci_low, ymax = ci_high), fill = \"#2166ac\", alpha = 0.35) +",
            "  ggplot2::geom_line(linewidth = 1.1) +",
            f"  ggplot2::labs(title = \"Confidence vs prediction intervals\", subtitle = \"Blue: {round(conf * 100)}% CI "
            f"for the mean outcome. Orange: {round(conf * 100)}% prediction interval for one new person.\",",
            f"    x = sx_names[[\"{xa}\"]], y = sx_names[[sx_used[1]]]) + ggplot2::theme_minimal()",
        ]
        note = ("The confidence interval describes uncertainty about the AVERAGE outcome at each x; the prediction "
                "interval describes where ONE new person's outcome is likely to fall, so it is much wider.")
    else:
        L += [
            "sx_pr <- stats::predict(sx_m, newdata = sx_nd, type = \"link\", se.fit = TRUE)",
            f"sx_z <- stats::qnorm(1 - (1 - {r_num(conf)}) / 2)",
            "sx_nd$predicted <- stats::plogis(sx_pr$fit); sx_nd$ci_low <- stats::plogis(sx_pr$fit - sx_z * sx_pr$se.fit)",
            "sx_nd$ci_high <- stats::plogis(sx_pr$fit + sx_z * sx_pr$se.fit)",
            f"sx_p <- ggplot2::ggplot(sx_nd, ggplot2::aes(x = {xa}, y = predicted, group = 1)) +",
            "  ggplot2::geom_ribbon(ggplot2::aes(ymin = ci_low, ymax = ci_high), fill = \"#2166ac\", alpha = 0.3) +",
            f"  ggplot2::geom_line(linewidth = 1.1) + ggplot2::labs(title = \"Predicted probability with CI\", x = sx_names[[\"{xa}\"]],",
            "    y = \"Probability\") + ggplot2::theme_minimal()",
        ]
        note = ("For a binary outcome an individual prediction interval is not meaningful (the outcome is 0 or 1); "
                "the calibration plot shows whether predicted probabilities match observed proportions.")
    L += [f"ggplot2::ggsave({r_str(p1.as_posix())}, sx_p, width = 7, height = 4.8, dpi = 120)"]
    if m.family == "gaussian":
        L += ["sx_po <- data.frame(predicted = stats::fitted(sx_m), observed = sx_d$sx_y)",
              "sx_p2 <- ggplot2::ggplot(sx_po, ggplot2::aes(predicted, observed)) + ggplot2::geom_point(alpha = 0.3) +",
              "  ggplot2::geom_abline(linetype = 2) + ggplot2::labs(title = \"Predicted vs observed (in-sample)\") + ggplot2::theme_minimal()"]
    else:
        L += ["sx_po <- data.frame(p = stats::fitted(sx_m), y = sx_d$sx_y)",
              "sx_po$bin <- cut(sx_po$p, unique(stats::quantile(sx_po$p, seq(0, 1, 0.1))), include.lowest = TRUE)",
              "sx_cal <- stats::aggregate(cbind(p, y) ~ bin, data = sx_po, FUN = mean)",
              "sx_p2 <- ggplot2::ggplot(sx_cal, ggplot2::aes(p, y)) + ggplot2::geom_point(size = 2.5) + ggplot2::geom_line() +",
              "  ggplot2::geom_abline(linetype = 2) + ggplot2::coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +",
              "  ggplot2::labs(title = \"Calibration (deciles of predicted probability)\", x = \"Mean predicted\", y = \"Observed proportion\") +",
              "  ggplot2::theme_minimal()"]
    L += [f"ggplot2::ggsave({r_str(p2.as_posix())}, sx_p2, width = 6, height = 4.8, dpi = 120)",
          f"sx_result <- list(tables = list(sx_table(\"Predictions\", sx_nd, notes = {r_str(note)})), n_used = nrow(sx_d))"]
    res, art, _ = _finish(session, ds, "prediction_uncertainty", f"Prediction uncertainty: {m.display(ds)}",
                          "\n".join(L) + "\n", {"model": m.to_dict(), "x": x}, ["ggplot2"])
    for p in (p1, p2):
        if p.is_file():
            key = f"plots/{p.name}"
            art[key] = p.read_bytes()
            res.plots.append(key)
    return res, art


# ---- clustered / longitudinal full-procedure validation (v1.3.2, CD-11) ----------------------------------------

def _validation_target(run: Any, n_clusters: int | None, target: str) -> str:
    s = run.settings
    if target != "auto":
        return target
    return "cluster" if getattr(s, "cluster", None) and (n_clusters or 0) >= 10 else "participant"


def procedure_cv_longitudinal(session: RSession, ds: Dataset, run: Any,
                              rs: ResampleSpec) -> tuple[AnalysisResult, dict[str, bytes]]:
    """Gaussian longitudinal searches (workflow A): the entire search runs inside each training fold and the selected
    model is evaluated on data that played no part in selecting it. The split matches the declared prediction target;
    individual repeated rows are never split at random, and too few groups means 'unavailable', not a row split."""
    from ..discovery.history import context
    from ..discovery.rcode import candidate_args, setup_script
    from ..discovery.space import STRUCTURES, enumerate_candidates, focal_label, view

    s = run.settings
    if s.workflow == "structure":
        raise ValidationInputError("Workflow B compares covariance structures for one fixed part; there is no "
                                   "fixed-effect selection to cross-validate.")
    if (s.structure or "ri").startswith("mmrm"):
        raise ValidationInputError(f"Procedure validation is not available for {STRUCTURES[s.structure][0]}.")
    if run.total > rs.max_candidates:
        raise ValidationInputError(f"The whole search is repeated in every fold: limit it to searches with at most "
                                   f"{rs.max_candidates:,} candidates.")
    ctx = context(ds, run)
    if not ctx.current:
        raise ValidationInputError("The project changed since this search (" + "; ".join(ctx.differences) + "). "
                                   "Validation repeats the search on its own data view: run a new search first.")
    v = view(ds, s)
    leaky = [d.name for d in v.derived.values() if d.op in ("zscore", "center")]
    used = set([s.outcome] + s.candidates + s.required)
    if used & set(leaky):
        raise ValidationInputError("Standardized / centred derived variables (" + ", ".join(
            v.var(n).display_name for n in sorted(used & set(leaky))) + ") use whole-sample means: their scaling would "
            "leak information from the test folds. Use the raw variables for procedure validation.")
    n_cl = int(v.analysis_frame([s.cluster])[s.cluster].nunique()) if getattr(s, "cluster", None) else None
    target = _validation_target(run, n_cl, rs.target)
    if target not in TARGETS:
        raise ValidationInputError(f"Unknown validation target {target!r}.")
    if target == "cluster" and not getattr(s, "cluster", None):
        raise ValidationInputError("Holding out organizations needs a higher-level cluster in the search.")
    if target == "temporal" and not s.time:
        raise ValidationInputError("Future-visit validation needs a time variable.")
    cands = list(enumerate_candidates(s))
    setup = setup_script(v, s, run.family).replace("sx_result <- list(", "sx_setup_info <- list(")
    labels = [focal_label(v, c) for c in cands]
    k = int(rs.folds)
    min_units = {"cluster": max(2 * k, 10), "participant": max(2 * k, 20), "temporal": 0}[target]
    unit = {"cluster": "sx_cluster", "participant": "sx_gid", "temporal": "sx_gid"}[target]
    body = "\n".join([
        setup,
        "sx_cands <- list(" + ",\n  ".join(f"list({candidate_args(v, c, s)})" for c in cands) + ")",
        f"sx_labels <- {r_vec(labels)}",
        "sx_pvals <- function() vapply(sx_cands, function(a) tryCatch(do.call(sx_discover_one, a)$p, "
        "error = function(e) NA_real_), 0)",
        "sx_fit_selected <- function(a, d) {",
        "  f <- stats::reformulate(if (length(a$full)) a$full else \"1\", response = \"sx_y\")",
        "  d <- sx_complete(d, c(\"sx_y\", a$vars, \"sx_gid\", if (sx_disc_has_cluster) \"sx_cluster\",",
        "                       if (sx_disc_has_time) c(\"sx_time\", \"sx_timef\", \"sx_occ\")))",
        "  if (sx_disc_has_time) d$sx_occ <- as.integer(d$sx_timef)",
        "  sx_fit_model(f, d)",
        "}",
        "sx_predict_target <- function(m, newd, target) {",
        "  if (inherits(m, \"merMod\")) {",
        "    rf <- switch(target, cluster = NA, participant = if (sx_disc_clustered) ~ (1 | sx_cluster) else NA, temporal = NULL)",
        "    return(as.numeric(stats::predict(m, newdata = newd, re.form = rf, allow.new.levels = TRUE)))",
        "  }",
        "  lev <- switch(target, cluster = 0, participant = if (sx_disc_clustered) 1 else 0,",
        "                temporal = if (sx_disc_clustered) 2 else 1)",
        "  as.numeric(stats::predict(m, newdata = newd, level = lev))",
        "}",
        f"sx_target <- {r_str(target)}",
        f"set.seed({int(rs.seed)})",
        "sx_full <- sx_dd",
        "sx_rows <- list(); sx_skip <- character()",
        "if (sx_target == \"temporal\") {",
        "  sx_times <- sort(unique(sx_full$sx_time))",
        f"  sx_cut <- utils::tail(sx_times, min({k}, max(length(sx_times) - 3, 0)))   # rolling origin: >= 3 earlier occasions",
        "  sx_folds <- lapply(sx_cut, function(t) list(train = sx_full$sx_time < t, test = sx_full$sx_time == t, label = t))",
        "} else {",
        f"  sx_units <- sort(unique(as.character(sx_full${unit})))",
        f"  sx_folds <- if (length(sx_units) < {min_units}) list() else {{",
        f"    sx_gf <- sample(rep(seq_len({k}), length.out = length(sx_units)))",
        f"    sx_uf <- sx_gf[match(as.character(sx_full${unit}), sx_units)]",
        f"    lapply(seq_len({k}), function(f) list(train = sx_uf != f, test = sx_uf == f, label = f)) }}",
        "}",
        "if (!length(sx_folds)) stop(paste0(\"Validation unavailable: too few \", switch(sx_target, cluster = \"clusters\",",
        f"  participant = \"participants\", temporal = \"occasions\"), \" for this target (need at least \", {min_units if min_units else 4},",
        "  \"). Rows of the same participant are never split at random instead.\"))",
        "for (sx_f in seq_along(sx_folds)) {",
        "  sx_dd <- sx_full[sx_folds[[sx_f]]$train, , drop = FALSE]      # the whole search sees the training part only",
        "  sx_p <- sx_pvals()",
        "  if (all(is.na(sx_p))) { sx_skip <- c(sx_skip, paste0(\"fold \", sx_f, \": no candidate could be fitted\")); next }",
        "  sx_j <- which.min(sx_p)",
        "  sx_m <- tryCatch(sx_fit_selected(sx_cands[[sx_j]], sx_dd), error = function(e) e)",
        "  if (inherits(sx_m, \"error\")) { sx_skip <- c(sx_skip, paste0(\"fold \", sx_f, \": \", conditionMessage(sx_m))); next }",
        "  sx_test <- sx_full[sx_folds[[sx_f]]$test, , drop = FALSE]",
        "  sx_vars <- sx_cands[[sx_j]]$vars",
        "  sx_test <- sx_complete(sx_test, c(\"sx_y\", sx_vars, \"sx_gid\", if (sx_disc_has_cluster) \"sx_cluster\",",
        "    if (sx_disc_has_time) c(\"sx_time\", \"sx_timef\", \"sx_occ\")))",
        "  sx_test <- sx_test[sx_seen(sx_dd, sx_test, setdiff(sx_vars, c(\"sx_timef\"))), , drop = FALSE]",
        "  if (sx_target == \"temporal\") sx_test <- sx_test[as.character(sx_test$sx_gid) %in% as.character(sx_dd$sx_gid), , drop = FALSE]",
        "  sx_pred <- tryCatch(sx_predict_target(sx_m, sx_test, sx_target), error = function(e) rep(NA_real_, nrow(sx_test)))",
        "  sx_ok <- is.finite(sx_pred)",
        "  if (!any(sx_ok)) { sx_skip <- c(sx_skip, paste0(\"fold \", sx_f, \": no test row could be predicted\")); next }",
        "  sx_rows[[length(sx_rows) + 1]] <- data.frame(fold = sx_folds[[sx_f]]$label, selected = sx_labels[sx_j],",
        "    p_training = sx_p[sx_j], n_train = nrow(sx_dd), n_test = sum(sx_ok), test_units =",
        f"    length(unique(as.character(sx_test${unit}[sx_ok]))), t(sx_metrics(sx_test$sx_y[sx_ok], sx_pred[sx_ok], \"gaussian\")))",
        "}",
        "sx_dd <- sx_full",
        "if (!length(sx_rows)) stop(paste(\"Validation unavailable:\", paste(sx_skip, collapse = \"; \")))",
        "sx_cvp <- do.call(rbind, sx_rows)",
        "sx_sel <- as.data.frame(table(selected = sx_cvp$selected), stringsAsFactors = FALSE)",
        "sx_result <- list(tables = list(",
        f"  sx_table(\"Full-procedure validation\", sx_cvp, notes = paste0({r_str(TARGETS[target])}, \". In each fold \",",
        "    \"the entire search ran on the training part only (selection, estimation, occasion index); the selected \",",
        "    \"model was then evaluated on the held-out part. Exploratory: post-selection limits still apply.\")),",
        "  sx_table(\"Models selected across folds\", sx_sel[order(-sx_sel$Freq), , drop = FALSE],",
        "    notes = \"If different folds select different models, the discovery is unstable.\")),",
        "  checks = sx_skip, n_used = nrow(sx_full))",
    ]) + "\n"
    res, art, _ = _finish(session, v, "procedure_cv", "Full-procedure validation (longitudinal)", body,
                          {"run": run.id, "resample": vars(rs), "target": target}, ["lme4", "nlme"])
    res.seed = rs.seed
    res.values["validation_target"] = TARGETS[target]
    return res, art
