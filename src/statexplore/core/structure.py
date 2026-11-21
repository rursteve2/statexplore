"""Assisted data-structure detection (§15.1) and wide-to-long restructuring.

Suggestions only: the user confirms them by setting variable roles.
"""
from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .dataset import Dataset, SourceInfo
from .variables import Measure, Role, Storage, VariableMeta, make_alias

ID_HINT = re.compile(r"(^|_)(id|pid|subj|subject|participant|person|respondent|case)(_|$)|hhidpn|^id", re.I)
TIME_HINT = re.compile(r"wave|time|visit|occasion|session|week|month|day|year|period|phase|^t$", re.I)


@dataclass
class Suggestion:
    variable: str
    role: str          # "id" | "time" | "cluster"
    score: float
    reason: str


def suggest_structure(ds: Dataset, max_rows: int = 200_000) -> list[Suggestion]:
    frame = ds.analysis_frame(ds.names(), filtered=False)
    if len(frame) > max_rows:
        frame = frame.sample(max_rows, random_state=0)
    n = len(frame)
    out: list[Suggestion] = []
    ids: list[tuple[str, int]] = []
    for name in frame.columns:
        v = ds.var(name)
        col = frame[name].dropna()
        if col.empty or v.storage is Storage.DATETIME:
            continue
        k = col.nunique()
        if 2 <= k <= n / 1.5:
            counts = col.value_counts()
            share_repeated = (counts >= 2).mean()
            integer_like = v.storage is Storage.STRING or np.all(np.mod(col.to_numpy(float), 1) == 0)
            if integer_like and share_repeated > 0.8 and k >= 10:
                score = share_repeated + (0.5 if ID_HINT.search(name) else 0) - (0.3 if v.is_categorical and k < 50 else 0)
                ids.append((name, k))
                out.append(Suggestion(name, "id", score, f"{k:,} distinct values, {share_repeated:.0%} repeat "
                                      f"(median {int(counts.median())} rows each)"))
    best_ids = sorted((s for s in out if s.role == "id"), key=lambda s: -s.score)
    if best_ids:
        id_name = best_ids[0].variable
        id_col = frame[id_name]
        for name in frame.columns:
            if name == id_name:
                continue
            v = ds.var(name)
            col = frame[name]
            k = col.nunique()
            if v.storage is Storage.NUMERIC and 2 <= k <= 60 and not v.label.lower().startswith("dummy"):
                pairs = frame[[id_name, name]].dropna()
                unique_pairs = 1 - pairs.duplicated().mean() if len(pairs) else 0
                varies = pairs.groupby(id_name)[name].nunique().mean() if len(pairs) else 0
                if unique_pairs > 0.95 and varies > 1.5:
                    score = unique_pairs + (0.5 if TIME_HINT.search(name) else 0)
                    out.append(Suggestion(name, "time", score, f"{k} distinct values; varies within participant "
                                          f"(mean {varies:.1f} per participant); participant × value pairs are unique"))
            # cluster: each participant belongs to exactly one value, and there are fewer clusters than participants
            if (v.is_categorical or v.storage is Storage.STRING or ID_HINT.search(name)) and 2 <= k < id_col.nunique():
                per_id = frame[[id_name, name]].dropna().groupby(id_name)[name].nunique()
                if len(per_id) and (per_id <= 1).all() and k >= 5:
                    out.append(Suggestion(name, "cluster", 0.5 + (k / id_col.nunique()),
                                          f"{k:,} groups; each participant belongs to one (participants nested within it)"))
    order = {"id": 0, "time": 1, "cluster": 2}
    return sorted(out, key=lambda s: (order[s.role], -s.score))


# ---- wide -> long -----------------------------------------------------------------

@dataclass
class ReshapeSpec:
    id_var: str | None                           # existing participant ID, or None to number rows
    measures: dict[str, list[str]]               # new variable -> wide columns in time order
    time_values: list[Any]                       # one value per occasion
    time_name: str = "time"
    keep: list[str] | None = None                # time-constant variables to carry (default: all others)


def wide_to_long(ds: Dataset, spec: ReshapeSpec) -> Dataset:
    """Return a NEW dataset in long format; the source dataset is untouched."""
    k = len(spec.time_values)
    if k < 2:
        raise ValueError("At least two occasions are needed.")
    for new, cols in spec.measures.items():
        if len(cols) != k:
            raise ValueError(f"{new}: {len(cols)} columns for {k} occasions.")
        for c in cols:
            ds.var(c)
    used = {c for cols in spec.measures.values() for c in cols}
    keep = spec.keep if spec.keep is not None else [n for n in ds.variables if n not in used and n != spec.id_var]
    raw = ds.analysis_frame(list(ds.variables), filtered=False)
    id_name = spec.id_var or "row_id"
    ids = raw[spec.id_var] if spec.id_var else pd.Series(np.arange(1, len(raw) + 1), index=raw.index, dtype=float)
    blocks = []
    for j, tv in enumerate(spec.time_values):
        b = pd.DataFrame({id_name: ids.to_numpy()})
        b[spec.time_name] = tv
        for new, cols in spec.measures.items():
            b[new] = raw[cols[j]].to_numpy()
        for kname in keep:
            b[kname] = raw[kname].to_numpy()
        b["_order"] = np.arange(len(raw))
        blocks.append(b)
    long = pd.concat(blocks, ignore_index=True).sort_values(["_order", spec.time_name], kind="stable")
    long = long.drop(columns="_order").reset_index(drop=True)
    variables: list[VariableMeta] = []
    for i, name in enumerate(long.columns):
        src = ds.variables.get(name) or (ds.var(spec.measures[name][0]) if name in spec.measures else None)
        if name == spec.time_name:
            storage = Storage.NUMERIC if all(isinstance(t, (int, float)) for t in spec.time_values) else Storage.STRING
            long[name] = long[name].astype(float) if storage is Storage.NUMERIC else long[name].astype("string")
            meta = VariableMeta(name=name, alias=make_alias(i), storage=storage,
                                measure=Measure.ORDINAL if storage is Storage.NUMERIC else Measure.NOMINAL,
                                label="Occasion (created by wide-to-long restructuring)", role=Role.TIME)
        elif src is not None:
            meta = VariableMeta(name=name, alias=make_alias(i), storage=src.storage, measure=src.measure,
                                label=src.label, value_labels=dict(src.value_labels),
                                role=Role.ID if name == id_name else Role.NONE)
        else:
            meta = VariableMeta(name=name, alias=make_alias(i), storage=Storage.NUMERIC, measure=Measure.NOMINAL,
                                label="Row number of the wide dataset", role=Role.ID)
        variables.append(meta)
    source = SourceInfo(
        path=ds.source.path, format=f"{ds.source.format}+reshaped",
        options={"reshape": {"id_var": spec.id_var, "measures": spec.measures, "time_values": spec.time_values,
                             "time_name": spec.time_name, "keep": keep}, "original_options": ds.source.options},
        sha256=ds.source.sha256, size_bytes=ds.source.size_bytes,
        imported_at=_dt.datetime.now().isoformat(timespec="seconds"),
    )
    return Dataset(long, variables, source)
