"""Missing-data summaries (§17): counts, percentages, patterns, usable n per model."""
from __future__ import annotations

from typing import Any

import pandas as pd

from .dataset import Dataset


def by_variable(ds: Dataset, names: list[str]) -> pd.DataFrame:
    frame = ds.analysis_frame(names)
    n = len(frame)
    miss = frame.isna().sum()
    out = pd.DataFrame({
        "variable": [ds.var(c).display_name for c in frame.columns],
        "missing": miss.to_numpy(),
        "percent": (100 * miss / n).to_numpy() if n else 0.0,
        "user_missing_codes": [int(ds.user_missing_mask(c)[frame.index].sum()) for c in frame.columns],
    })
    return out.sort_values("missing", ascending=False, kind="stable").reset_index(drop=True)


def patterns(ds: Dataset, names: list[str], top: int = 15) -> pd.DataFrame:
    """Most common missingness patterns: one column per variable (X = missing), with counts."""
    frame = ds.analysis_frame(names).isna()
    if frame.empty:
        return pd.DataFrame()
    key = frame.apply(lambda r: "".join("1" if v else "0" for v in r), axis=1)
    counts = key.value_counts().head(top)
    rows = []
    for k, cnt in counts.items():
        row: dict[str, Any] = {ds.var(c).display_name: ("X" if ch == "1" else "") for c, ch in zip(frame.columns, k)}
        row["rows"] = int(cnt)
        row["percent"] = 100 * cnt / len(frame)
        row["variables_missing"] = k.count("1")
        rows.append(row)
    return pd.DataFrame(rows)


def usable_n(ds: Dataset, models: list[tuple[str, list[str]]]) -> pd.DataFrame:
    """Complete-case n for each (label, variables) model, and how it differs from the largest."""
    rows = []
    for label, vars_ in models:
        frame = ds.analysis_frame(list(dict.fromkeys(vars_)))
        rows.append({"model": label, "variables": len(set(vars_)), "usable_n": int(frame.notna().all(axis=1).sum())})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["difference_from_largest"] = out["usable_n"] - out["usable_n"].max()
    return out
