"""R Console back end (v1.1 §14): run the user's own R code in a separate process.

- Clean mode (default): every run starts a new R process, loads a fresh copy of the dataset as
  `my_data`, runs the script and exits. Nothing survives between runs.
- Persistent mode: one console process keeps its workspace between runs until it is restarted,
  cleared, or the project/dataset changes.

The analysis workers are never used for user code, so custom scripts cannot alter their state.
User code is sent as JSON over stdin (never through a shell). This is not a sandbox: the code
runs with the user's Windows permissions.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.dataset import Dataset
from .codegen import data_load_script
from .locate import find_rscript, r_library
from .worker import REngineError, RScriptError, RWorker

CONSOLE_SCRIPT = Path(__file__).resolve().parents[1] / "r" / "console_worker.R"
SECURITY_WARNING = (
    "Custom R code runs with your Windows user permissions: it can read, change or delete files and start "
    "programs. The separate R session keeps StatExplore's own analyses safe from your script, but it is not a "
    "security sandbox. Run only code you trust."
)


@dataclass
class ConsoleEntry:
    kind: str          # input | output | message | warning | error
    text: str


@dataclass
class ConsoleResult:
    entries: list[ConsoleEntry]
    plots: list[bytes]
    error: bool
    elapsed: float
    mode: str
    data_rows: int | None = None
    data_script: str = ""

    def text(self, kinds: tuple[str, ...] = ("input", "output", "message", "warning", "error")) -> str:
        out = []
        for e in self.entries:
            if e.kind in kinds:
                out.append("> " + e.text.replace("\n", "\n+ ") if e.kind == "input" else e.text)
        return "\n".join(out)


def user_library() -> Path:
    from .package_manager import user_library as _ul

    return _ul()


class RConsole:
    """One console (clean or persistent). Thread-safe for one run at a time; cancel() may be
    called from another thread and kills the running R process."""

    def __init__(self, workdir: str | Path, rscript: str | None = None, persistent: bool = False):
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.rscript = rscript
        self.persistent = persistent
        self._worker: RWorker | None = None
        self._ds: Dataset | None = None
        self._loaded_sig: str | None = None
        self._lock = threading.Lock()
        self._running: RWorker | None = None
        self.cancelled = False

    # ---- dataset --------------------------------------------------------------------------
    def set_dataset(self, ds: Dataset | None) -> None:
        """A different dataset (or project) ends a persistent session: no stale objects survive."""
        if ds is not self._ds:
            self._ds = ds
            self.restart(start=False)

    @staticmethod
    def _signature(ds: Dataset) -> str:
        from ..analysis.session import RSession

        return RSession._metadata_signature(ds)

    def _load_data(self, w: RWorker) -> tuple[int | None, str]:
        if self._ds is None:
            return None, ""
        from ..analysis.session import export_data

        csv = self.workdir / "console_data.csv"
        export_data(self._ds, csv)
        script = data_load_script(self._ds, csv.as_posix())
        out = w.request("load", {"script": script}, timeout=300).result
        self._loaded_sig = self._signature(self._ds)
        return int(out["rows"]), script

    def _new_worker(self) -> RWorker:
        lib = user_library()
        lib.mkdir(parents=True, exist_ok=True)
        w = RWorker(find_rscript(self.rscript), script=CONSOLE_SCRIPT,
                    extra_env={"STATEXPLORE_R_LIB": str(r_library()), "STATEXPLORE_USER_LIB": str(lib)})
        w.start()
        return w

    # ---- running ---------------------------------------------------------------------------
    def run(self, code: str, timeout: float = 600.0) -> ConsoleResult:
        """Run `code` after an explicit user action. Blocks; call from a background thread."""
        with self._lock:
            self.cancelled = False
            plot_dir = self.workdir / f"plots_{uuid.uuid4().hex[:8]}"
            rows, script = None, ""
            if self.persistent:
                if self._worker is None or not self._worker.alive:
                    self._worker = self._new_worker()
                    self._loaded_sig = None
                w = self._worker
                if self._ds is not None and self._loaded_sig != self._signature(self._ds):
                    rows, script = self._load_data(w)
            else:
                w = self._new_worker()
                rows, script = self._load_data(w)
            self._running = w
            try:
                resp = w.request("run_user", {"code": code, "plot_dir": plot_dir.as_posix(), "width": 7, "height": 5},
                                 timeout=timeout)
            except REngineError as e:
                if self.cancelled:
                    raise REngineError("Execution stopped by the user. "
                                       + ("The persistent session was restarted; its objects are gone."
                                          if self.persistent else "")) from None
                raise REngineError(f"{e} (the console session was restarted)") from None
            finally:
                self._running = None
                if not self.persistent:
                    w.stop()
                elif not w.alive:
                    self._worker = None
            r = resp.result
            entries = [ConsoleEntry(e["type"], e["text"]) for e in (r.get("entries") or [])]
            paths = r.get("plots") or []
            paths = [paths] if isinstance(paths, str) else paths
            plots = [Path(p).read_bytes() for p in paths if Path(p).is_file()]
            shutil.rmtree(plot_dir, ignore_errors=True)
            return ConsoleResult(entries, plots, bool(r.get("error")), resp.elapsed,
                                 "persistent" if self.persistent else "clean", rows, script)

    def cancel(self) -> None:
        self.cancelled = True
        w = self._running
        if w is not None:
            w.kill()

    # ---- persistent-session tools ----------------------------------------------------------------
    def _require_session(self) -> RWorker:
        if not self.persistent:
            raise RScriptError("This needs the persistent console session (clean runs keep no workspace).")
        if self._worker is None or not self._worker.alive:
            self._worker = self._new_worker()
            self._load_data(self._worker)
        return self._worker

    def objects(self) -> list[dict[str, Any]]:
        return self._require_session().request("objects", timeout=60).result or []

    def packages(self) -> list[dict[str, Any]]:
        return self._require_session().request("packages", timeout=60).result or []

    def clear_workspace(self) -> None:
        self._require_session().request("clear", timeout=60)

    def restart(self, start: bool = False) -> None:
        if self._worker is not None:
            self._worker.stop()
        self._worker = None
        self._loaded_sig = None
        if start and self.persistent:
            self._require_session()

    def export_my_data(self) -> tuple[Path, dict[str, Any]]:
        """Write the session's `my_data` for an explicit import (persistent mode only)."""
        w = self._require_session()
        csv, meta = self.workdir / "my_data_export.csv", self.workdir / "my_data_export.json"
        w.request("export", {"csv": csv.as_posix(), "meta": meta.as_posix()}, timeout=300)
        return csv, json.loads(meta.read_text(encoding="utf-8"))

    def close(self) -> None:
        self.restart(start=False)


def script_digest(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]
