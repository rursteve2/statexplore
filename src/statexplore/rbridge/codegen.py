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


def data_load_script(ds: Dataset, csv_path: str) -> str:
    """Script that reads the exported CSV and applies variable coding.

    Executed verbatim by the worker and included verbatim in "View Code".
    """
    variables = list(ds.variables.values())
    lines = [
        "# ---- StatExplore: data preparation ----",
        f"# Source file: {r_str(ds.source.path)}",
        "# Values declared as user-missing in the source were exported as NA.",
        f"sx_data_file <- {r_str(csv_path)}",
        "# Internal names (v001, ...) -> original variable names",
        "sx_names <- " + r_vec([v.display_name for v in variables], [v.alias for v in variables]),
        "dat <- utils::read.csv(",
        "  sx_data_file, na.strings = \"\", fileEncoding = \"UTF-8\", check.names = FALSE,",
        "  colClasses = " + r_vec([_COLCLASS[v.storage] for v in variables], [v.alias for v in variables]),
        ")",
    ]
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

