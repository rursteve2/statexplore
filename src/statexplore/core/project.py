"""Project files (.sxp): a single SQLite database holding everything needed to
reopen and reproduce a session — a project-local copy of the data, variable
metadata, the change log, and analysis results with their artifacts.

The original source file is referenced (path + SHA-256) but never modified.
"""
from __future__ import annotations

import io
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from .dataset import Change, Dataset, HoldoutSpec, SourceInfo
from .derive import DerivedSpec, FilterSpec
from .schema import AnalysisResult
from .variables import VariableMeta

PROJECT_FORMAT = "statexplore-project"
PROJECT_VERSION = 2     # 2 = v1.1: measurement review fields, analysis treatments, outcome families (additive)

_DDL = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE variables (position INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, meta TEXT NOT NULL);
CREATE TABLE changes (seq INTEGER PRIMARY KEY, change TEXT NOT NULL);
CREATE TABLE data (id INTEGER PRIMARY KEY CHECK (id = 1), parquet BLOB NOT NULL);
CREATE TABLE results (seq INTEGER PRIMARY KEY, id TEXT NOT NULL UNIQUE, result TEXT NOT NULL);
CREATE TABLE artifacts (key TEXT PRIMARY KEY, content BLOB NOT NULL);
CREATE TABLE searches (seq INTEGER PRIMARY KEY, id TEXT NOT NULL UNIQUE, meta TEXT NOT NULL, rows BLOB NOT NULL);
"""


class ProjectError(Exception):
    pass


class Project:
    """A dataset plus its analysis history. Artifacts are stored as bytes keyed by name."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.results: list[AnalysisResult] = []
        self.artifacts: dict[str, bytes] = {}
        self.searches: list = []  # discovery.engine.DiscoveryRun
        self.models: list = []    # analysis.models.ModelSpec (comparison set)
        self.settings: dict[str, Any] = {}   # per-project presentation preferences (e.g. discovery highlighting)
        self.path: Path | None = None
        self.loaded_version = PROJECT_VERSION   # format version of the file this project was opened from
        self.backup_path: Path | None = None

    def add_result(self, result: AnalysisResult, artifacts: dict[str, bytes] | None = None) -> None:
        self.results.append(result)
        if artifacts:
            self.artifacts.update(artifacts)

    # ---- save -------------------------------------------------------
    def save(self, path: str | os.PathLike) -> None:
        """Atomically write the project: build a temp file, then replace. Overwriting a file written by
        an older StatExplore first keeps a copy of it (reversible migration)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.loaded_version < PROJECT_VERSION and path.is_file() and self.path is not None \
                and path.resolve() == self.path.resolve():
            import shutil

            backup = path.with_name(f"{path.stem}.v{self.loaded_version}-backup{path.suffix}")
            if not backup.exists():
                shutil.copy2(path, backup)
            self.backup_path = backup
        fd, tmp = tempfile.mkstemp(suffix=".sxp.tmp", dir=path.parent)
        os.close(fd)
        try:
            con = sqlite3.connect(tmp)
            try:
                con.executescript(_DDL)
                self._write(con)
                con.commit()
            finally:
                con.close()
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        self.loaded_version = PROJECT_VERSION
        self.path = path

    def _write(self, con: sqlite3.Connection) -> None:
        ds = self.dataset
        meta = {
            "format": PROJECT_FORMAT,
            "version": PROJECT_VERSION,
            "source": ds.source.to_dict(),
            "derived": [d.to_dict() for d in ds.derived.values()],
            "filters": [f.to_dict() for f in ds.filters],
            "models": [m.to_dict() for m in self.models],
            "holdout": ds.holdout.to_dict() if ds.holdout else None,
            "settings": self.settings,
        }
        con.executemany("INSERT INTO meta VALUES (?, ?)", [(k, json.dumps(v)) for k, v in meta.items()])
        con.executemany(
            "INSERT INTO variables VALUES (?, ?, ?)",
            [(i, v.name, json.dumps(v.to_dict())) for i, v in enumerate(ds.variables.values())],
        )
        con.executemany(
            "INSERT INTO changes VALUES (?, ?)",
            [(i, json.dumps(c.to_dict())) for i, c in enumerate(ds.changes)],
        )
        buf = io.BytesIO()
        ds.raw.to_parquet(buf, engine="pyarrow", index=False)
        con.execute("INSERT INTO data VALUES (1, ?)", (buf.getvalue(),))
        con.executemany(
            "INSERT INTO results VALUES (?, ?, ?)",
            [(i, r.id, json.dumps(r.to_dict())) for i, r in enumerate(self.results)],
        )
        con.executemany("INSERT INTO artifacts VALUES (?, ?)", list(self.artifacts.items()))
        for i, run in enumerate(self.searches):
            buf = io.BytesIO()
            pd.DataFrame([_flat_row(r) for r in run.rows]).to_parquet(buf, engine="pyarrow", index=False)
            con.execute("INSERT INTO searches VALUES (?, ?, ?, ?)",
                        (i, run.id, json.dumps(run.meta_dict()), buf.getvalue()))

    # ---- load -------------------------------------------------------
    @classmethod
    def open(cls, path: str | os.PathLike) -> "Project":
        path = Path(path)
        if not path.is_file():
            raise ProjectError(f"Project file not found: {path}")
        # Read-only URI so opening can never alter the project file.
        con = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
        try:
            meta = {k: json.loads(v) for k, v in con.execute("SELECT key, value FROM meta")}
            if meta.get("format") != PROJECT_FORMAT:
                raise ProjectError("Not a StatExplore project file.")
            if meta.get("version", 0) > PROJECT_VERSION:
                raise ProjectError("Project was saved by a newer StatExplore version.")
            variables = [VariableMeta.from_dict(json.loads(m)) for (m,) in
                         con.execute("SELECT meta FROM variables ORDER BY position")]
            (blob,) = con.execute("SELECT parquet FROM data WHERE id = 1").fetchone()
            raw = pd.read_parquet(io.BytesIO(blob), engine="pyarrow")
            derived = [DerivedSpec.from_dict(d) for d in meta.get("derived", [])]
            derived_names = {d.name for d in derived}
            base = [v for v in variables if v.name not in derived_names]
            ds = Dataset(raw, base, SourceInfo(**meta["source"]))
            ds.restore_derived(derived, [v for v in variables if v.name in derived_names])
            # restore the saved variable order (derived variables may be interleaved after undo)
            ds.variables = {v.name: ds.variables[v.name] for v in variables}
            ds.filters = [FilterSpec.from_dict(f) for f in meta.get("filters", [])]
            if meta.get("holdout"):
                ds.restore_holdout(HoldoutSpec(**meta["holdout"]))
            ds.changes = [Change(**json.loads(c)) for (c,) in
                          con.execute("SELECT change FROM changes ORDER BY seq")]
            project = cls(ds)
            project.results = [AnalysisResult.from_dict(json.loads(r)) for (r,) in
                               con.execute("SELECT result FROM results ORDER BY seq")]
            project.artifacts = dict(con.execute("SELECT key, content FROM artifacts"))
            project.searches = _load_searches(con)
            from ..analysis.models import ModelSpec

            project.models = [ModelSpec.from_dict(m) for m in meta.get("models", [])]
            project.settings = dict(meta.get("settings") or {})
        except sqlite3.DatabaseError as e:
            raise ProjectError(f"Project file is damaged or unreadable: {e}") from e
        finally:
            con.close()
        project.path = path
        project.loaded_version = int(meta.get("version", 1))
        return project


NESTED_ROW_FIELDS = ("tests", "inter")   # v1.2 discovery rows: kept as JSON text inside the parquet


def _flat_row(r: dict[str, Any]) -> dict[str, Any]:
    out = dict(r)
    for k in NESTED_ROW_FIELDS:
        if k in out and not isinstance(out[k], str):
            out[k] = json.dumps(out[k], default=float)
    return out


def _load_searches(con: sqlite3.Connection) -> list:
    from ..discovery.engine import DiscoveryRun

    has = con.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'searches'").fetchone()
    if not has:
        return []
    runs = []
    for meta, blob in con.execute("SELECT meta, rows FROM searches ORDER BY seq"):
        frame = pd.read_parquet(io.BytesIO(blob), engine="pyarrow")
        rows = []
        for rec in frame.to_dict("records"):
            rec["focal"] = list(rec.get("focal", []))
            rec["covariates"] = list(rec.get("covariates", []))
            for k in NESTED_ROW_FIELDS:
                if isinstance(rec.get(k), str):
                    rec[k] = json.loads(rec[k])
            rows.append({k: (None if isinstance(v, float) and v != v else v) for k, v in rec.items()})
        runs.append(DiscoveryRun.from_meta(json.loads(meta), rows))
    return runs


def source_status(ds: Dataset) -> dict[str, Any]:
    """Whether the original source file still exists and is unchanged."""
    from .importers import file_sha256

    p = Path(ds.source.path)
    if not p.is_file():
        return {"exists": False, "unchanged": None}
    return {"exists": True, "unchanged": file_sha256(p) == ds.source.sha256}
