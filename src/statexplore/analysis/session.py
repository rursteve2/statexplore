"""Binds a Dataset to an R worker: exports data once, runs analysis scripts,
and packages responses as AnalysisResult objects."""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from ..core.dataset import Dataset
from ..core.schema import AnalysisResult, EngineInfo, Status
from ..core.variables import Storage
from ..rbridge.codegen import data_load_parts
from ..rbridge.worker import RScriptError, RWorker

ANALYSIS_HEADER = "\n# ---- StatExplore: analysis ----\n"


class RSession:
    def __init__(self, worker: RWorker, workdir: str | Path):
        self.worker = worker
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        (self.workdir / "plots").mkdir(exist_ok=True)
        self._signature: str | None = None
        self._data_signature: str | None = None
        self._generation = -1
        self._load_script = ""

    # ---- data -------------------------------------------------------
    @staticmethod
    def _metadata_signature(ds: Dataset) -> str:
        meta = [v.to_dict() for v in ds.variables.values()]
        extra = [[d.to_dict() for d in ds.derived.values()], [f.to_dict() for f in ds.filters],
                 {k: v for k, v in (ds.holdout.to_dict() if ds.holdout else {}).items() if k != "evaluations"}]
        payload = json.dumps([ds.source.sha256, id(ds.raw), meta, extra], sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()

    @staticmethod
    def _data_signature_of(ds: Dataset) -> str:
        """What the exported CSV depends on: the data, which columns are exported with which
        storage and user-missing codes, and the holdout split. Measures and treatments are
        not part of it (they only change the R-side coding)."""
        base = [[v.alias, v.storage.value, v.missing_values, v.missing_ranges]
                for v in ds.variables.values() if not ds.is_derived(v.name)]
        hold = {k: v for k, v in (ds.holdout.to_dict() if ds.holdout else {}).items() if k != "evaluations"}
        payload = json.dumps([ds.source.sha256, id(ds.raw), base, hold], sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()

    def export_csv(self, ds: Dataset, path: Path) -> None:
        export_data(ds, path)

    def ensure_data(self, ds: Dataset) -> None:
        """Make the worker's `dat` match `ds` (including any analysis-specific treatment).
        Two cached steps: the CSV read (data signature) and the coding (metadata signature),
        so a treatment change re-codes the data without re-reading it."""
        sig = self._metadata_signature(ds)
        fresh = self.worker.alive and self.worker.generation == self._generation
        if sig == self._signature and fresh:
            return
        csv_path = self.workdir / "data.csv"
        dsig = self._data_signature_of(ds)
        read, prepare = data_load_parts(ds, csv_path.as_posix())
        if dsig != self._data_signature or not fresh:
            self.export_csv(ds, csv_path)
            self.worker.run(read, timeout=300)
            self._data_signature = dsig
            self._generation = self.worker.generation
        self._signature = None
        self.worker.run(prepare, timeout=300)
        self._signature, self._load_script = sig, read + prepare

    @property
    def load_script(self) -> str:
        """The data-load script currently executed in the worker (call ensure_data first)."""
        return self._load_script

    def invalidate(self) -> None:
        self._signature = None
        self._data_signature = None

    def new_plot_path(self) -> Path:
        return self.workdir / "plots" / f"{uuid.uuid4().hex[:12]}.png"

    # ---- analysis ---------------------------------------------------
    def run(self, ds: Dataset, *, kind: str, title: str, body: str, packages: list[str],
            inputs: dict[str, Any], timeout: float = 300.0) -> tuple[AnalysisResult, Any]:
        """Run `body` after the data-load script. Returns (result shell, raw R result).

        The recorded script is exactly what the worker evaluated: the data
        load script (run when the data/metadata last changed) followed by `body`.
        """
        self.ensure_data(ds)
        script = self._load_script + ANALYSIS_HEADER + body
        t0 = time.perf_counter()
        try:
            resp = self.worker.run(body, packages=packages, timeout=timeout)
        except RScriptError as e:
            versions = self.worker.request("versions", {"packages": packages}).result
            return AnalysisResult(
                kind=kind, title=title, engine=_engine(versions), script=script, inputs=inputs,
                status=Status.FAILED, messages=[f"R error: {e}"] + explain_r_error(str(e)),
                elapsed_s=time.perf_counter() - t0,
            ), None
        result = AnalysisResult(
            kind=kind, title=title, engine=_engine(resp.versions), script=script, inputs=inputs,
            status=Status.WARNING if resp.warnings else Status.OK,
            messages=[f"R warning: {w}" for w in resp.warnings],
            elapsed_s=resp.elapsed,
        )
        return result, resp.result


def export_data(ds: Dataset, path: Path) -> None:
    """Source variables only, unfiltered: derived variables and filters are applied by
    the R load script itself so the recorded code reproduces them."""
    base = [n for n in ds.variables if not ds.is_derived(n)]
    frame = ds.analysis_frame(base, filtered=False)
    frame.columns = [ds.var(n).alias for n in frame.columns]
    for n in ds.variables.values():
        if n.storage is Storage.DATETIME and n.alias in frame.columns:
            col = frame[n.alias]
            frame[n.alias] = col.map(lambda x: "" if pd.isna(x) else pd.Timestamp(x).isoformat())
    # Default float formatting is repr(), so values round-trip exactly.
    labels = ds.holdout_labels()
    if labels is not None:
        frame["sx_split"] = labels  # train/holdout assignment computed in Python (seeded)
    frame.to_csv(path, index=False, na_rep="", encoding="utf-8", lineterminator="\n")


R_ERROR_HINTS = [
    ("contrasts can be applied only to factors with 2 or more levels",
     "A categorical variable has only one category in the rows analysed (after removing missing values). "
     "Remove it, or check filters and missing data."),
    ("0 (non-NA) cases", "No rows are complete on all the variables used. Check missing data and filters."),
    ("not identifiable", "Some coefficients cannot be estimated: variables are redundant (perfectly related) "
     "or a combination of categories has no data."),
    ("cannot allocate vector", "Not enough memory: the model is too large (often a categorical variable with "
     "very many categories)."),
    ("new levels", "A category appears in the evaluation data but not in the data used to fit the model."),
]


def explain_r_error(message: str) -> list[str]:
    return [f"What this means: {hint}" for pattern, hint in R_ERROR_HINTS if pattern in message]


def _engine(versions: dict[str, Any]) -> EngineInfo:
    pk = {k: (v or "not installed") for k, v in (versions.get("packages") or {}).items()}
    return EngineInfo(engine="R", version=versions.get("r_version", "unknown"), packages=pk)
