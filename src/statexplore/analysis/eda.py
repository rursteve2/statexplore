"""Descriptive statistics (psych) and basic EDA plots (ggplot2 / psych) run in R."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.dataset import Dataset
from ..core.schema import AnalysisResult, Status, Table
from ..core.variables import Measure, Storage
from ..rbridge.codegen import alias, r_str, r_vec
from .session import RSession

DESCRIBE_COLUMNS = ["n", "mean", "sd", "median", "trimmed", "mad", "min", "max",
                    "range", "skew", "kurtosis", "se"]
DESCRIBE_NOTE = ("psych::describe(): sd uses n-1; skew/kurtosis type 3 (psych default); "
                 "trimmed = 10% trimmed mean; mad = scaled median absolute deviation. "
                 "Each variable uses its own non-missing values (available-case).")

PLOT_KINDS = {
    "histogram": "Histogram",
    "density": "Density plot",
    "boxplot": "Boxplot",
    "violin": "Violin plot",
    "scatter": "Scatterplot",
    "bar": "Bar chart (counts)",
    "bar_percent": "Bar chart (percentages)",
    "bar_horizontal": "Horizontal bar chart",
    "bar_clustered": "Clustered bar chart (by group)",
    "bar_stacked": "Stacked bar chart (by group)",
    "bar_stacked100": "100% stacked bar chart (by group)",
    "mosaic": "Mosaic plot (by group)",
    "ordinal_dist": "Ordinal response distribution",
    "pairs": "Scatterplot matrix (psych::pairs.panels)",
    "correlation": "Correlation heatmap",
    "missing": "Missing data (% per variable)",
}
MAX_MATRIX_VARS = 10
CATEGORICAL_PLOTS = ("bar", "bar_percent", "bar_horizontal", "bar_clustered", "bar_stacked", "bar_stacked100",
                     "mosaic", "ordinal_dist")
GROUPED_PLOTS = ("bar_clustered", "bar_stacked", "bar_stacked100", "mosaic")
MAX_CATEGORIES = 60


class AnalysisInputError(ValueError):
    """User-facing message explaining why an analysis cannot run."""


def _check_continuous(ds: Dataset, names: list[str]) -> None:
    bad = [ds.var(n).display_name for n in names
           if ds.var(n).measure is not Measure.CONTINUOUS or ds.var(n).storage is not Storage.NUMERIC]
    if bad:
        raise AnalysisInputError(f"These variables are not continuous numeric: {', '.join(bad)}")


def _check_categorical(ds: Dataset, name: str) -> None:
    v = ds.var(name)
    if not v.is_categorical:
        raise AnalysisInputError(f"{v.display_name} is not categorical (nominal/ordinal).")


def _record(result: AnalysisResult, ds: Dataset) -> None:
    from .measurement import record

    record(result, ds)


def _inputs(ds: Dataset, **named: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, val in named.items():
        if isinstance(val, list):
            out[key] = [{"name": n, "alias": ds.var(n).alias} for n in val]
        elif isinstance(val, str) and val in ds.variables:
            out[key] = {"name": val, "alias": ds.var(val).alias}
        else:
            out[key] = val
    return out


# ---- descriptives --------------------------------------------------------

def describe(session: RSession, ds: Dataset, names: list[str], group: str | None = None) -> AnalysisResult:
    if not names:
        raise AnalysisInputError("Select at least one continuous variable.")
    _check_continuous(ds, names)
    aliases = [alias(ds.var(n).alias) for n in names]
    lines = [f"sx_vars <- {r_vec(aliases)}"]
    if group is None:
        lines += [
            "sx_desc <- as.data.frame(psych::describe(dat[, sx_vars, drop = FALSE]))",
            "sx_result <- list(table = as.list(sx_desc))",
        ]
        title = "Descriptive statistics"
    else:
        _check_categorical(ds, group)
        g = alias(ds.var(group).alias)
        lines += [
            f"sx_desc <- as.data.frame(psych::describeBy(dat[, sx_vars, drop = FALSE], group = dat${g},",
            "                                            mat = TRUE, digits = 15))",
            "sx_result <- list(table = as.list(sx_desc))",
        ]
        title = f"Descriptive statistics by {ds.var(group).display_name}"
    body = "\n".join(lines) + "\n"
    result, raw = session.run(ds, kind="describe", title=title, body=body,
                              packages=["psych"], inputs=_inputs(ds, variables=names, group=group))
    _record(result, ds)
    if raw is None:
        return result
    tbl = raw["table"]
    n_rows = len(tbl["n"]) if isinstance(tbl["n"], list) else 1
    col = lambda k: tbl[k] if isinstance(tbl[k], list) else [tbl[k]]  # noqa: E731
    by_alias = {ds.var(n).alias: ds.var(n).display_name for n in names}
    if group is None:
        row_names = [ds.var(n).display_name for n in names]
    else:
        # describeBy(mat=TRUE) rows are item x group; "vars" indexes sx_vars (1-based).
        row_names = [f"{by_alias[aliases[int(i) - 1]]} | {g}" for i, g in zip(col("vars"), col("group1"))]
    rows = [[col(k)[i] for k in DESCRIBE_COLUMNS] for i in range(n_rows)]
    result.tables.append(Table(name="describe", columns=DESCRIBE_COLUMNS, rows=rows, row_names=row_names,
                               notes=DESCRIBE_NOTE))
    result.missing_data = "Available-case per variable (psych::describe default)."
    return result


# ---- categorical descriptives (v1.1 §8.2, §8.4) ---------------------------------------------

def category_codes(ds: Dataset, name: str) -> tuple[list[Any], list[str]]:
    """Codes and labels in the variable's category order (explicit order, else sorted codes)."""
    from ..rbridge.codegen import level_labels

    levels = ds.levels(name)
    return levels, level_labels(ds, name, levels)


def _ordered_factor_line(ds: Dataset, var: str, target: str) -> str:
    """Re-level a categorical column to its category order (a reference level may have moved
    a level first for modelling; tables and plots use the category order)."""
    _codes, labels = category_codes(ds, var)
    return f"{target} <- factor(as.character({target}), levels = {r_vec(labels)})"


FREQ_R = r"""sx_freq <- function(x, codes, ordinal) {
  n_total <- length(x); n_miss <- sum(is.na(x)); n_valid <- n_total - n_miss
  tab <- table(x, useNA = "no")
  df <- data.frame(code = codes, category = names(tab), n = as.numeric(tab), stringsAsFactors = FALSE)
  df$percent <- 100 * df$n / n_total
  df$valid_percent <- if (n_valid > 0) 100 * df$n / n_valid else NA_real_
  if (ordinal) df$cumulative_valid_percent <- cumsum(df$valid_percent)
  miss <- df[1, , drop = FALSE]; miss[1, ] <- NA
  miss$category <- "Missing"; miss$n <- n_miss; miss$percent <- 100 * n_miss / n_total
  tot <- miss; tot$category <- "Total"; tot$n <- n_total; tot$percent <- 100
  df <- rbind(df, miss, tot)
  modes <- if (n_valid > 0) names(tab)[tab == max(tab)] else character()
  med <- NA_character_
  if (ordinal && n_valid > 0) med <- names(tab)[which(cumsum(as.numeric(tab)) >= n_valid / 2)[1]]
  list(table = df, summary = data.frame(n_total = n_total, n_valid = n_valid, n_missing = n_miss,
       percent_missing = 100 * n_miss / n_total, categories = length(tab), observed_categories = sum(tab > 0),
       mode = paste(modes, collapse = " / "), median_category = med, stringsAsFactors = FALSE))
}
"""


def _check_categorical_any(ds: Dataset, names: list[str]) -> None:
    for n in names:
        v = ds.var(n)
        if not v.is_categorical:
            raise AnalysisInputError(f"{v.display_name} is not categorical. Frequencies are for nominal, ordinal and "
                                     "binary variables; use Describe for continuous variables.")
        k = len(ds.levels(n))
        if k > MAX_CATEGORIES:
            raise AnalysisInputError(f"{v.display_name} has {k} categories (limit {MAX_CATEGORIES}).")


def frequencies(session: RSession, ds: Dataset, names: list[str]) -> AnalysisResult:
    """Counts, total %, valid %, missing, mode, number of categories (cumulative % and median
    category for ordinal variables). No means are computed for categorical variables."""
    if not names:
        raise AnalysisInputError("Select at least one categorical variable.")
    _check_categorical_any(ds, names)
    L = [FREQ_R, "sx_tabs <- list(); sx_sum <- list()"]
    for n in names:
        a = alias(ds.var(n).alias)
        codes, _labels = category_codes(ds, n)
        ordinal = ds.var(n).measure is Measure.ORDINAL
        L += [
            f"sx_x <- dat${a}",
            _ordered_factor_line(ds, n, "sx_x"),
            f"sx_f <- sx_freq(sx_x, codes = {r_vec([_code_text(c) for c in codes])}, ordinal = {'TRUE' if ordinal else 'FALSE'})",
            f"sx_tabs[[length(sx_tabs) + 1]] <- sx_table(paste0(\"Frequencies: \", sx_names[[{r_str(a)}]]), sx_f$table,",
            f"  notes = {r_str(_freq_note(ds, n))})",
            f"sx_sum[[length(sx_sum) + 1]] <- cbind(variable = sx_names[[{r_str(a)}]], level = {r_str(ds.var(n).measure.value)}, sx_f$summary)",
        ]
    L += [
        "sx_tabs <- c(list(sx_table(\"Categorical summary\", do.call(rbind, sx_sum),",
        "  notes = \"Valid % excludes missing values. Mode = most frequent category (ties listed). Median category for ordinal variables only; no means are computed for categorical codes.\")), sx_tabs)",
        "sx_result <- list(tables = sx_tabs, n_used = nrow(dat))",
    ]
    body = "\n".join(L) + "\n"
    from .framework import _alias_map, _parse_table

    result, raw = session.run(ds, kind="frequencies", title="Frequencies: " + ", ".join(ds.var(n).display_name for n in names),
                              body=body, packages=[], inputs=_inputs(ds, variables=names))
    result.missing_data = "Each variable: all rows; missing values counted separately (valid % excludes them)."
    if raw is not None:
        names_map = _alias_map(ds)
        for t in raw.get("tables") or []:
            result.tables.append(_parse_table(t, names_map))
        result.n_used = raw.get("n_used")
    return result


def _code_text(c: Any) -> str:
    return str(int(c)) if isinstance(c, float) and c.is_integer() else str(c)


def _freq_note(ds: Dataset, name: str) -> str:
    v = ds.var(name)
    order = "approved category order" if v.category_order else "code order"
    note = f"Categories in {order}; percent = of all rows, valid percent = of non-missing rows."
    if v.measure is Measure.ORDINAL:
        note += " Cumulative % follows the category order."
        if v.coding_direction:
            note += f" Direction: {v.coding_direction}."
    return note


CROSSTAB_R = r"""sx_xtab <- function(r, c) {
  tab <- table(r, c)
  rs <- rowSums(tab); cs <- colSums(tab); N <- sum(tab)
  keep_r <- rs > 0; keep_c <- cs > 0
  t2 <- tab[keep_r, keep_c, drop = FALSE]
  exp <- outer(rowSums(t2), colSums(t2)) / sum(t2)
  m <- function(x) { d <- as.data.frame.matrix(x); d$Total <- rowSums(x); d }
  obs <- m(tab); obs <- rbind(obs, Total = c(cs, N))
  pct <- function(x) as.data.frame.matrix(x)
  rowp <- pct(100 * prop.table(tab, 1)); rowp$Total <- ifelse(rs > 0, 100, NA)
  colp <- pct(100 * prop.table(tab, 2)); colp <- rbind(colp, Total = ifelse(cs > 0, 100, NA))
  totp <- m(100 * tab / N); totp <- rbind(totp, Total = c(100 * cs / N, 100))
  list(tab = tab, t2 = t2, obs = obs, rowp = rowp, colp = colp, totp = totp, exp = as.data.frame.matrix(exp), N = N)
}
"""


def crosstab(session: RSession, ds: Dataset, row: str, col: str) -> AnalysisResult:
    """Contingency table with row/column/total percentages, expected counts, the Pearson
    chi-square test (no continuity correction) and an exact-test suggestion."""
    if row == col:
        raise AnalysisInputError("Choose two different variables.")
    _check_categorical_any(ds, [row, col])
    ra, ca = alias(ds.var(row).alias), alias(ds.var(col).alias)
    L = [
        CROSSTAB_R,
        f"sx_d <- dat[stats::complete.cases(dat[, {r_vec([ra, ca])}]), {r_vec([ra, ca])}, drop = FALSE]",
        _ordered_factor_line(ds, row, f"sx_d${ra}"),
        _ordered_factor_line(ds, col, f"sx_d${ca}"),
        f"sx_x <- sx_xtab(sx_d${ra}, sx_d${ca})",
        "sx_chi <- if (all(dim(sx_x$t2) >= 2)) suppressWarnings(stats::chisq.test(sx_x$t2, correct = FALSE)) else NULL",
        "sx_minexp <- if (length(sx_x$exp)) min(as.matrix(sx_x$exp)) else NA_real_",
        "sx_small <- if (length(sx_x$exp)) mean(as.matrix(sx_x$exp) < 5) else NA_real_",
        "sx_k <- min(dim(sx_x$t2))",
        "sx_v <- if (!is.null(sx_chi) && sx_k > 1) sqrt(unname(sx_chi$statistic) / (sx_x$N * (sx_k - 1))) else NA_real_",
        "sx_test <- data.frame(test = \"Pearson chi-square (no continuity correction)\",",
        "  statistic = if (is.null(sx_chi)) NA_real_ else unname(sx_chi$statistic),",
        "  df = if (is.null(sx_chi)) NA_real_ else unname(sx_chi$parameter),",
        "  p = if (is.null(sx_chi)) NA_real_ else sx_chi$p.value, cramers_v = sx_v, n = sx_x$N,",
        "  min_expected = sx_minexp, percent_cells_expected_below_5 = 100 * sx_small)",
        "sx_advice <- if (is.null(sx_chi)) \"A test needs at least two observed categories in each variable.\" else",
        "  if (sx_minexp < 1 || sx_small > 0.2) \"Some expected counts are small (below 5 in more than 20% of cells, or below 1): the chi-square approximation may be poor. Use Fisher's exact test (Analysis Builder > Categorical Association).\" else",
        "  \"Expected counts are adequate for the chi-square approximation (all >= 1 and at most 20% below 5).\"",
        "sx_result <- list(tables = list(",
        f"  sx_table(\"Observed counts\", sx_x$obs, row_names = rownames(sx_x$obs), notes = paste0(\"Rows: \", sx_names[[{r_str(ra)}]], \"; columns: \", sx_names[[{r_str(ca)}]], \".\")),",
        "  sx_table(\"Row percentages\", sx_x$rowp, row_names = rownames(sx_x$rowp)),",
        "  sx_table(\"Column percentages\", sx_x$colp, row_names = rownames(sx_x$colp)),",
        "  sx_table(\"Total percentages\", sx_x$totp, row_names = rownames(sx_x$totp)),",
        "  sx_table(\"Expected counts (under independence)\", sx_x$exp, row_names = rownames(sx_x$exp),",
        "    notes = \"Categories with no observations are left out of the expected counts and the test.\"),",
        "  sx_table(\"Chi-square test of independence\", sx_test, notes = sx_advice)),",
        "  n_used = sx_x$N, n_excluded = nrow(dat) - sx_x$N, notes = sx_advice)",
    ]
    body = "\n".join(L) + "\n"
    from .framework import _alias_map, _parse_table

    title = f"Cross-tabulation: {ds.var(row).display_name} × {ds.var(col).display_name}"
    result, raw = session.run(ds, kind="crosstab", title=title, body=body, packages=[],
                              inputs=_inputs(ds, row=row, col=col))
    result.missing_data = "Complete cases on both variables."
    if raw is not None:
        names_map = _alias_map(ds)
        for t in raw.get("tables") or []:
            result.tables.append(_parse_table(t, names_map))
        result.n_used = raw.get("n_used")
        result.values["n_excluded"] = raw.get("n_excluded")
        result.values["notes"] = [str(raw.get("notes") or "")]
    return result


# ---- plots ---------------------------------------------------------------

@dataclass
class PlotRequest:
    kind: str
    variables: list[str]           # primary variables (x, or x and y, or matrix set)
    group: str | None = None       # optional categorical grouping variable
    title: str = ""


def _validate_plot(ds: Dataset, req: PlotRequest) -> None:
    k, vs = req.kind, req.variables
    if k not in PLOT_KINDS:
        raise AnalysisInputError(f"Unknown plot type: {k}")
    if k in ("histogram", "density"):
        if len(vs) != 1:
            raise AnalysisInputError("Choose exactly one continuous variable.")
        _check_continuous(ds, vs)
    elif k in ("boxplot", "violin"):
        if len(vs) != 1:
            raise AnalysisInputError("Choose one continuous variable (and optionally a grouping variable).")
        _check_continuous(ds, vs)
        if k == "violin" and req.group is None:
            raise AnalysisInputError("A violin plot needs a categorical grouping variable.")
    elif k == "scatter":
        if len(vs) != 2:
            raise AnalysisInputError("Choose two continuous variables (x then y).")
        _check_continuous(ds, vs)
    elif k in CATEGORICAL_PLOTS:
        if len(vs) != 1:
            raise AnalysisInputError("Choose one categorical variable (and a grouping variable where needed).")
        _check_categorical_any(ds, vs)
        if k in GROUPED_PLOTS and req.group is None:
            raise AnalysisInputError(f"A {PLOT_KINDS[k].lower()} needs a categorical grouping variable (Group by).")
        if k == "ordinal_dist" and ds.var(vs[0]).measure is not Measure.ORDINAL:
            raise AnalysisInputError(f"{ds.var(vs[0]).display_name} is not ordinal. Approve an ordinal level in "
                                     "Dataset Explorer ▸ Measurement Review, or use a bar chart.")
        if req.group is not None:
            _check_categorical_any(ds, [req.group])
    elif k == "missing":
        if not 1 <= len(vs) <= 60:
            raise AnalysisInputError("Choose 1 to 60 variables.")
        return
    elif k in ("pairs", "correlation"):
        if not 2 <= len(vs) <= MAX_MATRIX_VARS:
            raise AnalysisInputError(f"Choose 2 to {MAX_MATRIX_VARS} continuous variables.")
        _check_continuous(ds, vs)
    if req.group is not None:
        _check_categorical(ds, req.group)


def plot_body(ds: Dataset, req: PlotRequest, png_path: str) -> tuple[str, list[str]]:
    a = [alias(ds.var(n).alias) for n in req.variables]
    g = alias(ds.var(req.group).alias) if req.group else None
    used = a + ([g] if g else [])
    title = req.title or _default_title(ds, req)
    L = [
        f"sx_plot_file <- {r_str(png_path)}",
        f"sx_used <- {r_vec(used)}",
    ]
    if req.kind == "missing":
        L += [
            "sx_df <- dat[, sx_used, drop = FALSE]",
            "sx_n <- nrow(sx_df)",
            "sx_mis <- data.frame(variable = unname(sx_names[sx_used]), percent = 100 * colMeans(is.na(sx_df)))",
            "sx_mis$variable <- factor(sx_mis$variable, levels = sx_mis$variable[order(sx_mis$percent)])",
            "sx_plot <- ggplot2::ggplot(sx_mis, ggplot2::aes(x = variable, y = percent)) +",
            "  ggplot2::geom_col(fill = \"#b2182b\") + ggplot2::coord_flip() +",
            "  ggplot2::geom_text(ggplot2::aes(label = sprintf(\"%.1f%%\", percent)), hjust = -0.1, size = 3) +",
            "  ggplot2::scale_y_continuous(limits = c(0, 105)) +",
            "  ggplot2::labs(x = NULL, y = \"% missing\")",
        ]
    elif req.kind in ("pairs", "correlation"):
        L.append("sx_df <- dat[, sx_used, drop = FALSE]  # pairwise deletion")
        L.append("sx_n <- sum(stats::complete.cases(sx_df))")
    else:
        L.append("sx_df <- dat[stats::complete.cases(dat[, sx_used, drop = FALSE]), sx_used, drop = FALSE]")
        L.append("sx_n <- nrow(sx_df)")
    if req.kind in CATEGORICAL_PLOTS:
        return _categorical_plot(ds, req, L, a[0], g, title)
    gg = "ggplot2::"
    lab_g = f", {('fill' if req.kind in ('histogram', 'density', 'bar', 'boxplot', 'violin') else 'colour')} = sx_names[[{r_str(g)}]]" if g else ""
    if req.kind == "histogram":
        fill = f", fill = {g}" if g else ""
        L.append(f"sx_plot <- {gg}ggplot(sx_df, {gg}aes(x = {a[0]}{fill})) +")
        L.append(f"  {gg}geom_histogram(bins = 30, colour = \"white\", alpha = {0.6 if g else 0.9}, position = \"identity\") +")
        L.append(f"  {gg}labs(x = sx_names[[{r_str(a[0])}]], y = \"Count\"{lab_g})")
    elif req.kind == "density":
        fill = f", fill = {g}, colour = {g}" if g else ""
        L.append(f"sx_plot <- {gg}ggplot(sx_df, {gg}aes(x = {a[0]}{fill})) +")
        L.append(f"  {gg}geom_density(alpha = 0.35) +")
        L.append(f"  {gg}labs(x = sx_names[[{r_str(a[0])}]], y = \"Density\"{lab_g}{', colour = sx_names[[' + r_str(g) + ']]' if g else ''})")
    elif req.kind in ("boxplot", "violin"):
        xg = g if g else "\"\""
        L.append(f"sx_plot <- {gg}ggplot(sx_df, {gg}aes(x = {xg}, y = {a[0]}{', fill = ' + g if g else ''})) +")
        if req.kind == "boxplot":
            L.append(f"  {gg}geom_boxplot(outlier.alpha = 0.5, show.legend = FALSE) +")
        else:
            L.append(f"  {gg}geom_violin(trim = FALSE, show.legend = FALSE) +")
            L.append(f"  {gg}geom_boxplot(width = 0.12, fill = \"white\", outlier.shape = NA) +")
        L.append(f"  {gg}labs(x = {'sx_names[[' + r_str(g) + ']]' if g else 'NULL'}, y = sx_names[[{r_str(a[0])}]])")
    elif req.kind == "scatter":
        col = f", colour = {g}" if g else ""
        L.append(f"sx_plot <- {gg}ggplot(sx_df, {gg}aes(x = {a[0]}, y = {a[1]}{col})) +")
        L.append(f"  {gg}geom_point(alpha = 0.4) +")
        L.append(f"  {gg}labs(x = sx_names[[{r_str(a[0])}]], y = sx_names[[{r_str(a[1])}]]{lab_g})")
    elif req.kind == "bar":
        fill = f", fill = {g}" if g else ""
        L.append(f"sx_plot <- {gg}ggplot(sx_df, {gg}aes(x = {a[0]}{fill})) +")
        L.append(f"  {gg}geom_bar(position = {gg}position_dodge(preserve = \"single\")) +")
        L.append(f"  {gg}labs(x = sx_names[[{r_str(a[0])}]], y = \"Count\"{lab_g})")
    elif req.kind == "correlation":
        L += [
            "sx_cor <- stats::cor(sx_df, use = \"pairwise.complete.obs\", method = \"pearson\")",
            "sx_long <- as.data.frame(as.table(sx_cor), stringsAsFactors = FALSE)",
            "names(sx_long) <- c(\"row\", \"col\", \"r\")",
            "sx_long$row <- factor(sx_names[sx_long$row], levels = rev(sx_names[sx_used]))",
            "sx_long$col <- factor(sx_names[sx_long$col], levels = sx_names[sx_used])",
            f"sx_plot <- {gg}ggplot(sx_long, {gg}aes(x = col, y = row, fill = r)) +",
            f"  {gg}geom_tile(colour = \"white\") +",
            f"  {gg}geom_text({gg}aes(label = sprintf(\"%.2f\", r)), size = 3.2) +",
            f"  {gg}scale_fill_gradient2(limits = c(-1, 1), low = \"#2166ac\", mid = \"white\", high = \"#b2182b\") +",
            f"  {gg}labs(x = NULL, y = NULL, fill = \"Pearson r\") +",
            f"  {gg}theme_minimal() + {gg}theme(axis.text.x = {gg}element_text(angle = 35, hjust = 1))",
        ]
    if req.kind == "pairs":
        L += [
            "grDevices::png(sx_plot_file, width = 8, height = 8, units = \"in\", res = 110)",
            f"psych::pairs.panels(sx_df, labels = unname(sx_names[sx_used]), main = {r_str(title)},",
            "                    method = \"pearson\", hist.col = \"#9ecae1\", density = TRUE, ellipses = FALSE)",
            "invisible(grDevices::dev.off())",
        ]
        packages = ["psych"]
    else:
        if req.kind == "correlation":
            subtitle = ('paste0("Each r uses all rows complete on that pair; ", sx_n, '
                        '" rows are complete on every variable")')
        elif req.kind == "missing":
            subtitle = 'paste0(sx_n, " rows (after filters); missing = system-missing or declared user-missing")'
        else:
            subtitle = 'paste0("n = ", sx_n, " complete cases")'
        L.append(f"sx_plot <- sx_plot + {gg}labs(title = {r_str(title)}, subtitle = {subtitle})")
        if req.kind != "correlation":
            L.append(f"sx_plot <- sx_plot + {gg}theme_minimal()")
        L.append(f"{gg}ggsave(sx_plot_file, sx_plot, width = 7, height = 4.8, dpi = 120)")
        packages = ["ggplot2"]
    L.append("sx_result <- list(plot = sx_plot_file, n = sx_n)")
    return "\n".join(L) + "\n", packages


def _categorical_plot(ds: Dataset, req: PlotRequest, L: list[str], x: str, g: str | None,
                      title: str) -> tuple[str, list[str]]:
    """Categorical plots draw a count table computed in R (returned as "Plotted counts", so the
    plot can be checked against the frequency table). Categories keep their category order."""
    gg = "ggplot2::"
    k = req.kind
    L.append(_ordered_factor_line(ds, req.variables[0], f"sx_df${x}"))
    if g:
        L.append(_ordered_factor_line(ds, req.group, f"sx_df${g}"))
        L += [
            f"sx_cnt <- as.data.frame(table(x = sx_df${x}, g = sx_df${g}), responseName = \"n\")",
            "sx_tot <- stats::ave(sx_cnt$n, sx_cnt$g, FUN = sum)",
            "sx_cnt$percent <- ifelse(sx_tot > 0, 100 * sx_cnt$n / sx_tot, 0)  # % within each group",
        ]
    else:
        L += [
            f"sx_cnt <- as.data.frame(table(x = sx_df${x}), responseName = \"n\")",
            "sx_cnt$percent <- if (sum(sx_cnt$n) > 0) 100 * sx_cnt$n / sum(sx_cnt$n) else 0",
        ]
    xl = f"sx_names[[{r_str(x)}]]"
    gl = f"sx_names[[{r_str(g)}]]" if g else "NULL"
    pct_lab = "\"% of valid responses\"" + (" # within each group" if g else "")
    if k == "mosaic":
        L += [
            f"sx_tab <- table(sx_df${x}, sx_df${g}, dnn = c({xl}, {gl}))  # categories down the side, groups across",
            "grDevices::png(sx_plot_file, width = 8.5, height = 6, units = \"in\", res = 110)",
            f"vcd::mosaic(sx_tab, shade = TRUE, legend = TRUE, main = {r_str(title)},",
            "             labeling = vcd::labeling_border(rot_labels = c(0, 0, 0, 0), just_labels = c(\"center\", \"center\", \"center\", \"right\"),",
            "                                             offset_varnames = c(0, 0, 0, 6), offset_labels = c(0, 0, 0, 0.5)),",
            "             margins = c(3, 3, 3, 9))",
            "invisible(grDevices::dev.off())",
        ]
        packages = ["vcd"]
    else:
        fill_g = ", fill = g" if g else ""
        if k == "bar":
            L.append(f"sx_plot <- {gg}ggplot(sx_cnt, {gg}aes(x = x, y = n{fill_g})) + {gg}geom_col(position = {gg}position_dodge()) +")
            L.append(f"  {gg}labs(x = {xl}, y = \"Count\", fill = {gl})")
        elif k == "bar_percent":
            L.append(f"sx_plot <- {gg}ggplot(sx_cnt, {gg}aes(x = x, y = percent{fill_g})) + {gg}geom_col(position = {gg}position_dodge()) +")
            L.append(f"  {gg}labs(x = {xl}, y = {pct_lab.split(' #')[0]}, fill = {gl})")
        elif k == "bar_horizontal":
            L.append(f"sx_plot <- {gg}ggplot(sx_cnt, {gg}aes(x = x, y = n{fill_g})) + {gg}geom_col(position = {gg}position_dodge()) +")
            L.append(f"  {gg}scale_x_discrete(limits = rev(levels(sx_cnt$x))) + {gg}coord_flip() +")
            L.append(f"  {gg}labs(x = {xl}, y = \"Count\", fill = {gl})")
        elif k == "bar_clustered":
            L.append(f"sx_plot <- {gg}ggplot(sx_cnt, {gg}aes(x = x, y = n, fill = g)) + {gg}geom_col(position = {gg}position_dodge()) +")
            L.append(f"  {gg}labs(x = {xl}, y = \"Count\", fill = {gl})")
        elif k == "bar_stacked":
            L.append(f"sx_plot <- {gg}ggplot(sx_cnt, {gg}aes(x = g, y = n, fill = x)) + {gg}geom_col(position = {gg}position_stack(reverse = TRUE)) +")
            L.append(f"  {gg}labs(x = {gl}, y = \"Count\", fill = {xl})")
        elif k == "bar_stacked100":
            L.append(f"sx_plot <- {gg}ggplot(sx_cnt, {gg}aes(x = g, y = percent, fill = x)) + {gg}geom_col(position = {gg}position_stack(reverse = TRUE)) +")
            L.append(f"  {gg}labs(x = {gl}, y = \"% within group\", fill = {xl})")
        elif k == "ordinal_dist":
            if not g:
                L.append("sx_cnt$g <- factor(\"All\")")
            L += [
                f"sx_plot <- {gg}ggplot(sx_cnt, {gg}aes(x = g, y = percent, fill = x)) +",
                f"  {gg}geom_col(position = {gg}position_stack(reverse = TRUE), colour = \"white\") +",
                f"  {gg}geom_text({gg}aes(label = ifelse(percent >= 5, sprintf(\"%.0f%%\", percent), \"\")),",
                f"    position = {gg}position_stack(vjust = 0.5, reverse = TRUE), size = 3) +",
                f"  {gg}scale_fill_brewer(palette = \"RdYlBu\", direction = 1) + {gg}coord_flip() +",
                f"  {gg}labs(x = {gl if g else 'NULL'}, y = \"% of valid responses\", fill = {xl}) +",
                f"  {gg}guides(fill = {gg}guide_legend(reverse = FALSE))",
            ]
        L.append(f"sx_plot <- sx_plot + {gg}labs(title = {r_str(title)}, subtitle = paste0(\"n = \", sx_n, \" complete cases\")) + {gg}theme_minimal()")
        L.append(f"{gg}ggsave(sx_plot_file, sx_plot, width = 7.5, height = 4.8, dpi = 120)")
        packages = ["ggplot2"]
    L.append("sx_result <- list(plot = sx_plot_file, n = sx_n, tables = list(sx_table(\"Plotted counts\", sx_cnt,")
    L.append("  notes = \"Counts and percentages drawn in the plot (percent within group when grouped).\")))")
    return "\n".join(L) + "\n", packages


def _default_title(ds: Dataset, req: PlotRequest) -> str:
    names = [ds.var(n).display_name for n in req.variables]
    base = f"{PLOT_KINDS[req.kind]}: {', '.join(names)}"
    if req.group:
        base += f" by {ds.var(req.group).display_name}"
    return base


def plot(session: RSession, ds: Dataset, req: PlotRequest) -> tuple[AnalysisResult, dict[str, bytes]]:
    _validate_plot(ds, req)
    png = session.new_plot_path()
    body, packages = plot_body(ds, req, png.as_posix())
    result, raw = session.run(
        ds, kind="eda_plot", title=req.title or _default_title(ds, req), body=body, packages=packages,
        inputs=_inputs(ds, plot=req.kind, variables=req.variables, group=req.group),
    )
    _record(result, ds)
    artifacts: dict[str, bytes] = {}
    if raw is not None:
        if not png.is_file():
            result.status = Status.FAILED
            result.messages.append("R reported success but no image file was written.")
        else:
            key = f"plots/{png.name}"
            artifacts[key] = png.read_bytes()
            result.plots.append(key)
            result.n_used = int(raw["n"])
            if raw.get("tables"):
                from .framework import _alias_map, _parse_table

                result.tables = [_parse_table(t, _alias_map(ds)) for t in raw["tables"]]
            result.missing_data = ("All rows" if req.kind == "missing" else
                                   "Pairwise deletion; n used = rows complete on every variable"
                                   if req.kind in ("pairs", "correlation")
                                   else "Complete cases on the plotted variables")
    return result, artifacts
