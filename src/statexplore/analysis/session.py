"""Binds a Dataset to an R worker: exports data once, runs analysis scripts,
and packages responses as AnalysisResult objects."""
from __future__ import annotations

import hashlib
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..core.dataset import Dataset
from ..core.schema import AnalysisResult, EngineInfo, Status
from ..core.variables import Storage
from ..rbridge.codegen import data_load_parts
from ..rbridge.worker import RScriptError, RWorker

ANALYSIS_HEADER = "\n# ---- StatExplore: analysis ----\n"
PROVENANCE_SCHEMA = 1     # v1.3.6 (AUD135-07): result.values["analysis_provenance"]


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _text_sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


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
        self._input_sha256 = ""
        self._input_rows = 0

    # ---- data -------------------------------------------------------
    @staticmethod
    def _metadata_signature(ds: Dataset) -> str:
        """Everything the worker's prepared `dat` depends on: the CSV inputs (below) plus every variable's metadata
        (coding, treatments), derived variables and filters. AUD135-06: keyed by the CONTENT identity of the raw
        values, never by the process-local address of the frame (which can be reused by another frame)."""
        from ..core.identity import canonical, content_id, view_metadata

        return hashlib.sha256(canonical(["prepare", content_id(ds), view_metadata(ds)]).encode()).hexdigest()

    @staticmethod
    def _data_signature_of(ds: Dataset) -> str:
        """What the exported CSV depends on: the data (content identity), which columns are exported with which
        storage and user-missing codes, and the holdout split. Measures and treatments are not part of it (they only
        change the R-side coding)."""
        from ..core.identity import canonical, content_id

        base = [[v.alias, v.storage.value, v.missing_values, v.missing_ranges]
                for v in ds.variables.values() if not ds.is_derived(v.name)]
        hold = {k: v for k, v in (ds.holdout.to_dict() if ds.holdout else {}).items() if k != "evaluations"}
        labels = ds.holdout_labels()
        realized = hashlib.sha256(np.packbits((labels == "holdout").to_numpy(dtype=bool)).tobytes()).hexdigest() \
            if labels is not None else None
        return hashlib.sha256(canonical(["csv", content_id(ds), base, hold or None, realized]).encode()).hexdigest()

    def export_csv(self, ds: Dataset, path: Path) -> None:
        export_data(ds, path)

    def ensure_data(self, ds: Dataset) -> None:
        """Make the worker's `dat` match `ds` (including any analysis-specific treatment).
        Two cached steps: the CSV read (data signature) and the coding (metadata signature),
        so a treatment change re-codes the data without re-reading it. The SHA-256 of the CSV bytes the worker read
        is kept (`input_sha256`): it is the fit-time proof of the analysed input (AUD135-07)."""
        sig = self._metadata_signature(ds)
        fresh = self.worker.alive and self.worker.generation == self._generation
        if sig == self._signature and fresh:
            return
        csv_path = self.workdir / "data.csv"
        dsig = self._data_signature_of(ds)
        read, prepare = data_load_parts(ds, csv_path.as_posix())
        if dsig != self._data_signature or not fresh:
            self._data_signature = None
            self._input_sha256 = ""
            self.export_csv(ds, csv_path)
            sha = _file_sha256(csv_path)
            self.worker.run(read, timeout=300)
            self._data_signature, self._input_sha256 = dsig, sha
            self._input_rows = int(len(ds.raw))
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
        self._input_sha256 = ""

    def provenance(self, ds: Dataset, script: str) -> dict[str, Any]:
        """AUD135-07: the immutable analysis-view provenance of a result, captured at fit time right after
        `ensure_data` (the worker, its CSV and the view are synchronized): content and view identities, the SHA-256 of
        the exact CSV bytes R read (before any projection or pseudonymization), the script, and the snapshot needed to
        rebuild the view later. Historical exports reproduce a result only when this proof can be re-established."""
        from ..core.identity import IDENTITY_SCHEMA, content_id, view_id
        from ..discovery.history import snapshot

        return {"schema": PROVENANCE_SCHEMA, "identity_schema": IDENTITY_SCHEMA, "content_id": content_id(ds),
                "view_id": view_id(ds), "input_sha256": self._input_sha256, "input_rows": self._input_rows,
                "input_file": "data.csv", "load_script_sha256": _text_sha256(self._load_script),
                "script_sha256": _text_sha256(script), "snapshot": snapshot(ds),
                "treatments": [dict(t.to_dict(), description=d) for t, d in getattr(ds, "treatments", [])]}

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
        prov = self.provenance(ds, script)
        t0 = time.perf_counter()
        try:
            resp = self.worker.run(body, packages=packages, timeout=timeout)
        except RScriptError as e:
            versions = self.worker.request("versions", {"packages": packages}).result
            failed = AnalysisResult(
                kind=kind, title=title, engine=_engine(versions), script=script, inputs=inputs,
                status=Status.FAILED, messages=[f"R error: {e}"] + explain_r_error(str(e)),
                elapsed_s=time.perf_counter() - t0,
            )
            failed.values["analysis_provenance"] = prov
            return failed, None
        result = AnalysisResult(
            kind=kind, title=title, engine=_engine(resp.versions), script=script, inputs=inputs,
            status=Status.WARNING if resp.warnings else Status.OK,
            messages=[f"R warning: {w}" for w in resp.warnings],
            elapsed_s=resp.elapsed,
        )
        result.values["analysis_provenance"] = prov
        return result, resp.result


def export_data(ds: Dataset, path: Path, aliases: set[str] | None = None) -> None:
    """Source variables only, unfiltered: derived variables and filters are applied by
    the R load script itself so the recorded code reproduces them. `aliases` limits the columns."""
    base = [n for n in ds.variables if not ds.is_derived(n)
            and (aliases is None or ds.var(n).alias in aliases)]
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
    ("Hessian is not positive definite", "Standard errors could not be computed: usually a random-effect variance is "
     "(near) zero, or the data are too sparse for this model. Simplify the random effects, or fit the model without "
     "random effects (e.g. Ordinal logistic regression in the Analysis Builder)."),
    ("Hessian is numerically singular", "Standard errors could not be computed: some parameters are not identified "
     "(sparse categories, separation, or redundant predictors)."),
]


def explain_r_error(message: str) -> list[str]:
    return [f"What this means: {hint}" for pattern, hint in R_ERROR_HINTS if pattern in message]


def _engine(versions: dict[str, Any]) -> EngineInfo:
    pk = {k: (v or "not installed") for k, v in (versions.get("packages") or {}).items()}
    return EngineInfo(engine="R", version=versions.get("r_version", "unknown"), packages=pk)
