"""Import the R Console's `my_data` back into StatExplore (v1.1 §14.2), only on explicit request.

`diff_import` compares the exported data with the current dataset (rows, columns, added /
removed variables, changed types, metadata that cannot be carried over). `apply_import` first
saves a recoverable snapshot of the current project, then returns a new project whose dataset
is the imported data; analysis history, model sets and searches are carried over.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .dataset import Dataset, SourceInfo
from .project import Project
from .variables import Measure, Storage, VariableMeta, make_alias


@dataclass
class ImportDiff:
    orig_rows: int
    orig_cols: int
    new_rows: int
    new_cols: int
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    changed_types: list[tuple[str, str, str]] = field(default_factory=list)
    metadata_loss: list[str] = field(default_factory=list)

    def lines(self) -> list[str]:
        out = [f"Rows: {self.orig_rows:,} → {self.new_rows:,}", f"Columns: {self.orig_cols} → {self.new_cols}"]
        if self.added:
            out.append("Added variables: " + ", ".join(self.added))
        if self.removed:
            out.append("Removed variables: " + ", ".join(self.removed))
        for n, a, b in self.changed_types:
            out.append(f"Changed type: {n} ({a} → {b})")
        out += [f"Metadata lost: {m}" for m in self.metadata_loss]
        return out


def _kind(v: VariableMeta) -> str:
    return f"{v.storage.value}/{v.measure.value}"


def diff_import(ds: Dataset, csv: Path, meta: dict[str, Any]) -> tuple[ImportDiff, Dataset]:
    from .importers import _infer_meta, file_sha256

    info = {c["name"]: c for c in meta["columns"]}
    raw = pd.read_csv(csv, keep_default_na=False, na_values=[""], dtype=str, encoding="utf-8")
    by_display = {v.display_name: v for v in ds.variables.values()}
    diff = ImportDiff(ds.n_rows, ds.n_cols, len(raw), raw.shape[1])
    if ds.filters or ds.holdout:
        diff.metadata_loss.append("my_data held the filtered (training) rows only; rows outside the filters or in "
                                  "the holdout are not in the import.")
    cols: dict[str, pd.Series] = {}
    metas: list[VariableMeta] = []
    next_alias = max((int(v.alias[1:]) for v in ds.variables.values()), default=0)
    for name in raw.columns:
        cls = info.get(name, {}).get("class", "character")
        s = raw[name]
        numeric = cls in ("numeric", "integer", "logical", "double")
        old = by_display.get(name)
        if old is None:
            col = pd.to_numeric(s, errors="coerce") if numeric else s.astype("string")
            m = _infer_meta(next_alias, name, col)
            next_alias += 1
            m.alias = make_alias(next_alias - 1)
            if cls in ("factor", "ordered"):
                m.measure = Measure.ORDINAL if info[name].get("ordered") else Measure.NOMINAL
            m.source_measure = "r-console"
            diff.added.append(name)
            cols[name] = col
            metas.append(m)
            continue
        m = VariableMeta.from_dict(old.to_dict())
        if ds.is_derived(old.name):
            diff.metadata_loss.append(f"{name}: derived variable imported as stored values (its definition is not kept)")
            m.source_measure = "derived (imported)"
        if old.has_user_missing():
            diff.metadata_loss.append(f"{name}: declared user-missing codes were exported as NA")
            m.missing_values, m.missing_ranges = [], []
        if cls in ("factor", "ordered") and old.storage is Storage.NUMERIC and old.is_categorical:
            lab2code = {ds.level_label(old.name, c): c for c in ds.levels(old.name)}
            mapped = s.map(lambda x: lab2code.get(x) if pd.notna(x) else None)
            if s.notna().sum() == pd.Series(mapped).notna().sum():
                col = pd.to_numeric(mapped, errors="coerce")       # labels mapped back to the original codes
            else:
                col = s.astype("string")
                m.storage, m.measure, m.value_labels, m.category_order = Storage.STRING, Measure.NOMINAL, {}, None
                diff.changed_types.append((name, _kind(old), _kind(m)))
                diff.metadata_loss.append(f"{name}: new category labels; numeric codes and value labels dropped")
        elif numeric:
            col = pd.to_numeric(s, errors="coerce")
            if old.storage is not Storage.NUMERIC:
                m.storage, m.measure = Storage.NUMERIC, Measure.CONTINUOUS
                diff.changed_types.append((name, _kind(old), _kind(m)))
            elif old.is_categorical and not set(col.dropna().unique()) <= set(ds.levels(old.name)):
                m.measure, m.value_labels, m.category_order, m.reference_level = Measure.CONTINUOUS, {}, None, None
                diff.changed_types.append((name, _kind(old), _kind(m)))
                diff.metadata_loss.append(f"{name}: values are no longer the labelled codes; value labels dropped")
        else:
            col = s.astype("string")
            if old.storage is Storage.NUMERIC:
                m.storage, m.measure, m.value_labels, m.category_order = Storage.STRING, Measure.NOMINAL, {}, None
                diff.changed_types.append((name, _kind(old), _kind(m)))
        m.name = old.name
        cols[old.name] = col
        metas.append(m)
    kept = {m.name for m in metas}
    diff.removed = [v.display_name for v in ds.variables.values() if v.name not in kept]
    frame = pd.DataFrame(cols)
    frame.columns = [m.name for m in metas]
    for c in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[c]):
            frame[c] = frame[c].astype("float64")
    src = SourceInfo(path=f"R Console import (from {ds.source.path})", format="r-console",
                     options={"origin_sha256": ds.source.sha256}, sha256=file_sha256(csv),
                     size_bytes=csv.stat().st_size, imported_at=_dt.datetime.now().isoformat(timespec="seconds"))
    return diff, Dataset(frame.reset_index(drop=True), metas, src)


def snapshot_dir(project: Project) -> Path:
    if project.path is not None:
        return project.path.parent / "snapshots"
    import os

    return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "StatExplore" / "snapshots"


def apply_import(project: Project, new_ds: Dataset, directory: Path | None = None) -> tuple[Project, Path]:
    """Save a recoverable snapshot of `project`, then return the new project (history carried over)."""
    directory = directory or snapshot_dir(project)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = project.path.stem if project.path else "untitled"
    snap = directory / f"{stem}-before-R-import-{stamp}.sxp"
    path_before, version_before = project.path, project.loaded_version
    project.save(snap)
    project.path, project.loaded_version = path_before, version_before   # the snapshot is not the working file
    new = Project(new_ds)
    new.results, new.artifacts, new.models, new.searches = (list(project.results), dict(project.artifacts),
                                                            list(project.models), list(project.searches))
    new.path = None
    return new, snap


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]
