"""Dataset overview numbers shown immediately after import (computed in Python).

These are simple counts and summaries for the Dataset Explorer. Formal
descriptive statistics (psych::describe) are produced by the R engine.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .dataset import Dataset
from .variables import Storage


@dataclass
class VariableSummary:
    name: str
    display_name: str
    label: str
    storage: str
    measure: str
    role: str
    n_valid: int
    n_missing: int
    pct_missing: float
    n_unique: int
    n_user_missing_codes: int
    mean: float | None
    sd: float | None
    min: float | None
    max: float | None
    n_outliers: int | None  # Tukey fences: outside [Q1 - 1.5 IQR, Q3 + 1.5 IQR]


@dataclass
class DatasetSummary:
    n_rows: int                 # rows after filters
    n_rows_total: int
    n_cols: int
    n_missing_cells: int
    pct_missing_cells: float
    n_complete_rows: int
    variables: list[VariableSummary]


def tukey_outliers(values: np.ndarray) -> int:
    if len(values) < 4:
        return 0
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    return int(np.sum((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)))


def summarize(ds: Dataset) -> DatasetSummary:
    frame = ds.analysis_frame(ds.names(include_excluded=True))
    rows: list[VariableSummary] = []
    for name in frame.columns:
        v = ds.var(name)
        col = frame[name]
        valid = col.dropna()
        n_valid = int(len(valid))
        mean = sd = lo = hi = None
        n_out = None
        if v.storage is Storage.NUMERIC and n_valid:
            arr = valid.to_numpy(dtype=float)
            mean, lo, hi = float(arr.mean()), float(arr.min()), float(arr.max())
            sd = float(arr.std(ddof=1)) if n_valid > 1 else None
            if not v.is_categorical:
                n_out = tukey_outliers(arr)
        n = len(frame)
        rows.append(VariableSummary(
            name=name, display_name=v.display_name, label=v.label,
            storage=v.storage.value, measure=v.measure.value, role=v.role.value,
            n_valid=n_valid, n_missing=n - n_valid,
            pct_missing=100.0 * (n - n_valid) / n if n else 0.0,
            n_unique=int(valid.nunique()),
            n_user_missing_codes=int(ds.user_missing_mask(name)[frame.index].sum()),
            mean=mean, sd=sd, min=lo, max=hi, n_outliers=n_out,
        ))
    n_missing = int(frame.isna().sum().sum())
    cells = frame.shape[0] * frame.shape[1]
    return DatasetSummary(
        n_rows=len(frame), n_rows_total=ds.n_rows, n_cols=ds.n_cols, n_missing_cells=n_missing,
        pct_missing_cells=100.0 * n_missing / cells if cells else 0.0,
        n_complete_rows=int(frame.notna().all(axis=1).sum()),
        variables=rows,
    )


def suggest_plots(measures: list[str]) -> list[str]:
    """Suggest EDA plot kinds from the measurement levels of the chosen variables (§8)."""
    cont = sum(m == "continuous" for m in measures)
    cat = len(measures) - cont
    if len(measures) == 1:
        return ["histogram", "density", "boxplot"] if cont else ["bar"]
    if len(measures) == 2:
        if cont == 2:
            return ["scatter"]
        if cont == 1 and cat == 1:
            return ["boxplot", "violin"]
        return ["bar"]
    if cont >= 3 and cat == 0:
        return ["pairs", "correlation"]
    return []


def as_frame(summary: DatasetSummary) -> pd.DataFrame:
    return pd.DataFrame([vars(v) for v in summary.variables])
