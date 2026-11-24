"""Application state shared by all pages, plus background task execution."""
from __future__ import annotations

import logging
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, QSettings, QThreadPool, Signal

from ..analysis.session import RSession
from ..core.project import Project
from ..core.schema import AnalysisResult
from ..rbridge.locate import find_rscript
from ..rbridge.worker import RWorker

log = logging.getLogger(__name__)


class _TaskSignals(QObject):
    done = Signal(object)
    failed = Signal(str)


class _Task(QRunnable):
    def __init__(self, fn: Callable[[], Any]):
        super().__init__()
        self.fn = fn
        self.signals = _TaskSignals()

    def run(self) -> None:
        from ..rbridge.worker import RTimeoutError

        try:
            out = self.fn()
        except RTimeoutError as e:  # a timeout is never a completed analysis (P1-MODEL)
            self.signals.failed.emit(f"Timed out: the analysis did not complete and no result was recorded. {e}")
        except Exception as e:  # surfaced to the user via `failed`
            log.debug("Background task failed:\n%s", traceback.format_exc())
            self.signals.failed.emit(str(e) or e.__class__.__name__)
        else:
            self.signals.done.emit(out)


class AppState(QObject):
    project_changed = Signal()
    metadata_changed = Signal()
    result_added = Signal(object)        # AnalysisResult
    busy_changed = Signal(bool)
    r_busy_changed = Signal(bool)       # an R computation (not file I/O) is running: the Stop R button
    r_status_changed = Signal(str)
    models_changed = Signal()
    show_in_teaching = Signal(object)    # ModelSpec
    show_in_mixed = Signal(object)       # MixedSpec (a longitudinal discovery candidate, refitted)

    def __init__(self) -> None:
        super().__init__()
        self.settings = QSettings("StatExplore", "StatExplore")
        self.project: Project | None = None
        self.dirty = False
        self._worker: RWorker | None = None
        self._session: RSession | None = None
        self._workdir = Path(tempfile.mkdtemp(prefix="statexplore_"))
        # R is single-threaded per worker: serialise all R work on one thread.
        self._r_pool = QThreadPool()
        self._r_pool.setMaxThreadCount(1)
        self._io_pool = QThreadPool()
        self._io_pool.setMaxThreadCount(2)
        self._tasks: set[_Task] = set()
        self._busy = 0
        self._r_busy = 0
        self._cancel_requested = False

    # ---- project ----------------------------------------------------
    @property
    def dataset(self):
        return self.project.dataset if self.project else None

    def set_project(self, project: Project | None) -> None:
        self.project = project
        self.dirty = False
        if self._session:
            self._session.invalidate()
        self.project_changed.emit()

    def notify_metadata_changed(self) -> None:
        self.dirty = True
        self.metadata_changed.emit()

    def add_model(self, model) -> None:
        if self.project is None:
            return
        self.project.models.append(model)
        self.dirty = True
        self.models_changed.emit()

    def remove_model(self, model_id: str) -> None:
        if self.project is None:
            return
        self.project.models = [m for m in self.project.models if m.id != model_id]
        self.dirty = True
        self.models_changed.emit()

    def add_result(self, result: AnalysisResult, artifacts: dict[str, bytes]) -> None:
        if self.project is None:
            return
        self.project.add_result(result, artifacts)
        self.dirty = True
        self.result_added.emit(result)

    # ---- R ----------------------------------------------------------
    def rscript_setting(self) -> str:
        return str(self.settings.value("r/rscript", "") or "")

    def set_rscript_setting(self, path: str) -> None:
        self.settings.setValue("r/rscript", path)
        self.shutdown_r()

    def r_session(self) -> RSession:
        """Create the worker lazily (called on the R thread)."""
        if self._session is None:
            worker = RWorker(find_rscript(self.rscript_setting() or None))
            worker.start()
            self._worker = worker
            self._session = RSession(worker, self._workdir)
            self.r_status_changed.emit("R ready: " + worker.ready_info.get("r_version", ""))
        return self._session

    def cleanup(self) -> None:
        """Stop R and delete this session's temporary work folder (exported data, plot files)."""
        import shutil

        self.shutdown_r()
        shutil.rmtree(self._workdir, ignore_errors=True)

    def cancel_r(self) -> None:
        """Stop the running R computation (Stop button). The worker is killed; the waiting task fails and reports a
        cancellation, and the session reloads its data in the restarted worker (generation check)."""
        if self._r_busy and self._worker is not None:
            self._cancel_requested = True
            self._worker.kill()
            self.r_status_changed.emit("R stopped by the user (restarts automatically when needed)")

    def shutdown_r(self) -> None:
        if self._worker:
            self._worker.stop()
        self._worker = self._session = None
        self.r_status_changed.emit("R stopped (restarts automatically when needed)")

    # ---- background work --------------------------------------------
    def _submit(self, pool: QThreadPool, fn: Callable[[], Any],
                on_done: Callable[[Any], None], on_error: Callable[[str], None], r: bool = False) -> None:
        task = _Task(fn)
        self._tasks.add(task)
        self._set_busy(+1)
        if r:
            self._set_r_busy(+1)

        def finish() -> None:
            self._tasks.discard(task)
            self._set_busy(-1)
            if r:
                self._set_r_busy(-1)

        def failed(msg: str) -> None:
            cancelled = r and self._cancel_requested
            finish()        # the last R task to finish clears the cancel request
            if cancelled:
                msg = ("Cancelled by the user: the R computation was stopped and no result was recorded. "
                       "R restarts on next use.")
            on_error(msg)

        task.signals.done.connect(lambda out: (finish(), on_done(out)))
        task.signals.failed.connect(failed)
        task.setAutoDelete(False)
        pool.start(task)

    def run_r(self, fn: Callable[[RSession], Any], on_done: Callable[[Any], None],
              on_error: Callable[[str], None]) -> None:
        self._submit(self._r_pool, lambda: fn(self.r_session()), on_done, on_error, r=True)

    def run_io(self, fn: Callable[[], Any], on_done: Callable[[Any], None],
               on_error: Callable[[str], None]) -> None:
        self._submit(self._io_pool, fn, on_done, on_error)

    def _set_r_busy(self, delta: int) -> None:
        was = self._r_busy > 0
        self._r_busy += delta
        if (self._r_busy > 0) != was:
            self.r_busy_changed.emit(self._r_busy > 0)
            if not self._r_busy:
                self._cancel_requested = False

    def _set_busy(self, delta: int) -> None:
        was = self._busy > 0
        self._busy += delta
        if (self._busy > 0) != was:
            self.busy_changed.emit(self._busy > 0)

    def wait_idle(self, msecs: int = 30000) -> None:
        self._r_pool.waitForDone(msecs)
        self._io_pool.waitForDone(msecs)
