"""R code generation. The only place where R source text is built.

Rules (§23): user-derived text (names, labels, levels) only ever appears inside
string literals produced by `r_str`, which escapes everything and emits pure
ASCII. Identifiers in code are always internal aliases (v001, ...).
"""
from __future__ import annotations

import math
import re
from typing import Any, Iterable

from ..core.dataset import Dataset
from ..core.variables import Storage

_ALIAS_RE = re.compile(r"^v\d{3,}$")


def r_str(s: Any) -> str:
    """Safe R double-quoted string literal (ASCII-only output)."""
    out = ['"']
    for ch in str(s):
        o = ord(ch)
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif 32 <= o < 127:
            out.append(ch)
        elif o <= 0xFFFF:
            out.append(f"\\u{o:04x}")
        else:
            out.append(f"\\U{{{o:08x}}}")
    out.append('"')
    return "".join(out)


def r_num(x: Any) -> str:
    if x is None:
        return "NA_real_"
    f = float(x)
    if math.isnan(f):
        return "NA_real_"
    if math.isinf(f):
        return "Inf" if f > 0 else "-Inf"
    return repr(f)


def r_value(x: Any) -> str:
    return r_num(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else r_str(x)


def r_vec(items: Iterable[Any], names: Iterable[str] | None = None) -> str:
    items = list(items)
    vals = [r_value(x) for x in items]
    if names is not None:
        vals = [f"{alias(n)} = {v}" for n, v in zip(names, vals)]
    return "c(" + ", ".join(vals) + ")"


def alias(a: str) -> str:
    """Validate an internal alias before it is used as an R identifier."""
    if not _ALIAS_RE.match(a):
        raise ValueError(f"Not a valid internal alias: {a!r}")
    return a


def unique_labels(labels: list[str]) -> list[str]:
    """factor() merges duplicated labels; keep every level distinct."""
    seen: dict[str, int] = {}
    out = []
    for lab in labels:
        if lab in seen:
            seen[lab] += 1
            lab = f"{lab} [{seen[lab]}]"
        seen.setdefault(lab, 0)
        out.append(lab)
    return out


def level_labels(ds: Dataset, name: str, levels: list[Any]) -> list[str]:
    return unique_labels([ds.level_label(name, code) for code in levels])


_COLCLASS = {Storage.NUMERIC: "numeric", Storage.STRING: "character", Storage.DATETIME: "character"}

# Small helper functions used by StatExplore scripts. They are part of the
# load script, so every "View Code" listing is self-contained.
R_HELPERS = r"""# ---- StatExplore helper functions ----
sx_item_mean <- function(d, min_valid) {
  n <- rowSums(!is.na(d)); out <- rowMeans(d, na.rm = TRUE); out[n < min_valid] <- NA_real_; out
}
sx_item_sum <- function(d) rowSums(d)  # NA when any item is missing
sx_log <- function(x) { out <- rep(NA_real_, length(x)); ok <- !is.na(x) & x > 0; out[ok] <- log(x[ok]); out }
sx_sqrt <- function(x) { out <- rep(NA_real_, length(x)); ok <- !is.na(x) & x >= 0; out[ok] <- sqrt(x[ok]); out }
sx_z <- function(x) (x - mean(x, na.rm = TRUE)) / stats::sd(x, na.rm = TRUE)
sx_center <- function(x) x - mean(x, na.rm = TRUE)
sx_binarize <- function(x, cutoff, strict) as.numeric(if (strict) x > cutoff else x >= cutoff)
sx_recode <- function(x, from, to, keep_other) {
  m <- match(x, from)
  out <- if (keep_other) as.numeric(x) else rep(NA_real_, length(x))
  out[!is.na(m)] <- to[m[!is.na(m)]]
  out
}
# Complete cases on the analysis variables, unused factor levels dropped.
sx_complete <- function(data, vars) {
  droplevels(data[stats::complete.cases(data[, vars, drop = FALSE]), vars, drop = FALSE])
}
# Shapiro-Wilk is defined for 3 <= n <= 5000; otherwise not reported.
sx_shapiro <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) < 3 || length(x) > 5000 || stats::var(x) == 0) return(c(W = NA_real_, p = NA_real_))
  t <- stats::shapiro.test(x); c(W = unname(t$statistic), p = t$p.value)
}
# Package a data frame as a result table.
sx_table <- function(name, df, notes = "", row_names = NULL) {
  df <- as.data.frame(df, stringsAsFactors = FALSE, check.names = FALSE)
  list(name = name, columns = I(names(df)),
       data = lapply(df, function(col) if (is.factor(col)) I(as.character(col)) else I(col)),
       row_names = I(as.character(row_names)), notes = notes)
}
"""


def _derived_r(ds: Dataset, spec) -> str:
    a = alias(ds.var(spec.name).alias)
    src = [alias(ds.var(s).alias) for s in spec.sources]
    p = spec.params
    if spec.op == "mean_items":
        rhs = f"sx_item_mean(dat[, {r_vec(src)}], min_valid = {int(p['min_valid'])})"
    elif spec.op == "sum_items":
        rhs = f"sx_item_sum(dat[, {r_vec(src)}])"
    elif spec.op == "recode":
        frm = r_vec([m[0] for m in p["mapping"]])
        to = "c(" + ", ".join(r_num(m[1]) for m in p["mapping"]) + ")"
        rhs = f"sx_recode(dat${src[0]}, from = {frm}, to = {to}, keep_other = {'TRUE' if p['keep_other'] else 'FALSE'})"
    elif spec.op == "binarize":
        rhs = f"sx_binarize(dat${src[0]}, cutoff = {r_num(p['cutoff'])}, strict = {'TRUE' if p['strict'] else 'FALSE'})"
    elif spec.op == "square":
        rhs = f"dat${src[0]}^2"
    else:
        fn = {"log": "sx_log", "sqrt": "sx_sqrt", "zscore": "sx_z", "center": "sx_center"}[spec.op]
        rhs = f"{fn}(dat${src[0]})"
    return f"dat${a} <- {rhs}  # {r_str(spec.describe())}"


def _filter_r(ds: Dataset, f) -> str:
    a = f"dat${alias(ds.var(f.variable).alias)}"
    if f.op == "is_missing":
        cond = f"is.na({a})"
    elif f.op == "not_missing":
        cond = f"!is.na({a})"
    elif f.op == "in":
        cond = f"{a} %in% {r_vec(f.value)}"
    elif f.op == "not_in":
        cond = f"!is.na({a}) & !({a} %in% {r_vec(f.value)})"
    else:
        if f.op not in ("==", "!=", "<", "<=", ">", ">="):
            raise ValueError(f"Unknown filter operator {f.op!r}")
        cond = f"{a} {f.op} {r_value(f.value)}"
    # which() drops rows where the condition is NA, matching the Python filter.
    return f"dat <- dat[which({cond}), , drop = FALSE]"


def data_load_script(ds: Dataset, csv_path: str) -> str:
    """Script that reads the exported CSV, computes derived variables, applies
    filters, and applies variable coding. Executed verbatim by the worker and
    included verbatim in "View Code"."""
    variables = list(ds.variables.values())
    base = [v for v in variables if not ds.is_derived(v.name)]
    lines = [
        R_HELPERS,
        "# ---- StatExplore: data preparation ----",
        f"# Source file: {r_str(ds.source.path)}",
        "# Values declared as user-missing in the source were exported as NA.",
        f"sx_data_file <- {r_str(csv_path)}",
        "# Internal names (v001, ...) -> variable names",
        "sx_names <- " + r_vec([v.display_name for v in variables], [v.alias for v in variables]),
        "dat <- utils::read.csv(",
        "  sx_data_file, na.strings = \"\", fileEncoding = \"UTF-8\", check.names = FALSE,",
        "  colClasses = " + r_vec([_COLCLASS[v.storage] for v in base], [v.alias for v in base]),
        ")",
    ]
    if ds.derived:
        lines.append("# Derived variables (computed on numeric codes, before filters)")
        lines.extend(_derived_r(ds, spec) for spec in ds.derived.values())
    if ds.filters:
        lines.append("# Row filters")
        lines.extend(_filter_r(ds, f) for f in ds.filters)
    cat_lines = []
    for v in variables:
        if not v.is_categorical or v.storage is Storage.DATETIME:
            continue
        levels = ds.levels(v.name)
        if not levels:
            continue
        labels = level_labels(ds, v.name, levels)
        a = alias(v.alias)
        cat_lines.append(
            f"dat${a} <- factor(dat${a}, levels = {r_vec(levels)}, labels = {r_vec(labels)})"
        )
        if v.reference_level is not None and v.reference_level in levels:
            ref = labels[levels.index(v.reference_level)]
            cat_lines.append(f"dat${a} <- stats::relevel(dat${a}, ref = {r_str(ref)})")
    if cat_lines:
        lines.append("# Categorical variables: numeric codes -> labelled factor levels")
        lines.extend(cat_lines)
    lines.append("sx_result <- list(n_rows = nrow(dat), n_cols = ncol(dat))")
    return "\n".join(lines) + "\n"
