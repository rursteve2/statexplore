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
    "bar": "Bar chart",
    "pairs": "Scatterplot matrix (psych::pairs.panels)",
    "correlation": "Correlation heatmap",
}
MAX_MATRIX_VARS = 10


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
    elif k == "bar":
        if len(vs) != 1:
            raise AnalysisInputError("Choose one categorical variable.")
        _check_categorical(ds, vs[0])
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
    if req.kind in ("pairs", "correlation"):
        L.append("sx_df <- dat[, sx_used, drop = FALSE]  # pairwise deletion")
        L.append("sx_n <- sum(stats::complete.cases(sx_df))")
    else:
        L.append("sx_df <- dat[stats::complete.cases(dat[, sx_used, drop = FALSE]), sx_used, drop = FALSE]")
        L.append("sx_n <- nrow(sx_df)")
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
        L.append(f"sx_plot <- sx_plot + {gg}labs(title = {r_str(title)}, subtitle = paste0(\"n = \", sx_n"
                 + (", \" (pairwise deletion for each r)\"" if req.kind == "correlation" else ", \" complete cases\"")
                 + "))")
        if req.kind != "correlation":
            L.append(f"sx_plot <- sx_plot + {gg}theme_minimal()")
        L.append(f"{gg}ggsave(sx_plot_file, sx_plot, width = 7, height = 4.8, dpi = 120)")
        packages = ["ggplot2"]
    L.append("sx_result <- list(plot = sx_plot_file, n = sx_n)")
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
            result.missing_data = ("Pairwise deletion" if req.kind in ("pairs", "correlation")
                                   else "Complete cases on the plotted variables")
    return result, artifacts
