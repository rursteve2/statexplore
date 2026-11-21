"""Persistent R worker process with a JSON-lines protocol (see r/worker.R).

No shell is ever involved: Rscript is started with an argument list, and all
user-derived content travels inside JSON on stdin.
"""
from __future__ import annotations

import itertools
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .locate import WORKER_SCRIPT, find_rscript, r_library

log = logging.getLogger(__name__)
PREFIX = "@@SX@@ "


class REngineError(Exception):
    """R is unavailable, crashed, or timed out."""


class RScriptError(Exception):
    """The R script raised an error (worker stays alive)."""

    def __init__(self, message: str, script: str = ""):
        super().__init__(message)
        self.script = script


@dataclass
class RResponse:
    result: Any
    warnings: list[str] = field(default_factory=list)
    versions: dict[str, Any] = field(default_factory=dict)
    elapsed: float = 0.0


def _kill_tree(proc: subprocess.Popen[str]) -> None:
    """Kill the process and any children (e.g. if a user-configured Rscript is a launcher)."""
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
    if proc.poll() is None:
        proc.kill()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        log.error("R worker process %s did not exit after kill", proc.pid)


class RWorker:
    """One long-lived Rscript process. Requests are serialised with a lock."""

    def __init__(self, rscript: str | Path | None = None, startup_timeout: float = 60.0):
        path = Path(rscript) if rscript else find_rscript()
        if path is None or not Path(path).is_file():
            raise REngineError(
                "R was not found. Install R for Windows (https://cloud.r-project.org) "
                "or set the Rscript path in Settings."
            )
        self.rscript = Path(path)
        self.startup_timeout = startup_timeout
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None
        self._responses: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self.stderr_tail: list[str] = []
        self.ready_info: dict[str, Any] = {}
        self.generation = 0  # increments on every (re)start; session state lives per generation

    # ---- lifecycle --------------------------------------------------
    def start(self) -> None:
        env = os.environ.copy()
        env["STATEXPLORE_R_LIB"] = str(r_library())
        env["R_DEFAULT_PACKAGES"] = "utils,stats,graphics,grDevices,methods"
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self._responses = queue.Queue()
        self._proc = subprocess.Popen(
            [str(self.rscript), "--vanilla", str(WORKER_SCRIPT)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            env=env, creationflags=flags,
        )
        # Each process gets its own queue so a dying process's EOF can't reach its successor.
        threading.Thread(target=self._read_stdout, args=(self._proc, self._responses), daemon=True).start()
        threading.Thread(target=self._read_stderr, args=(self._proc,), daemon=True).start()
        msg = self._await("ready", self.startup_timeout)
        if not msg.get("ok"):
            self.stop()
            raise REngineError(f"R worker failed to start: {msg.get('error')}")
        self.ready_info = msg.get("result") or {}
        self.generation += 1

    def stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.poll() is None and proc.stdin:
                proc.stdin.write(json.dumps({"id": "bye", "cmd": "shutdown", "args": {}}) + "\n")
                proc.stdin.flush()
                proc.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired, ValueError):
            pass
        if proc.poll() is None:
            _kill_tree(proc)

    def kill(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            _kill_tree(proc)

    def restart(self) -> None:
        self.stop()
        self.start()

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # ---- io ---------------------------------------------------------
    @staticmethod
    def _read_stdout(proc: subprocess.Popen[str], responses: queue.Queue[dict[str, Any] | None]) -> None:
        assert proc.stdout
        for line in proc.stdout:
            line = line.rstrip("\r\n")
            if line.startswith(PREFIX):
                try:
                    responses.put(json.loads(line[len(PREFIX):]))
                except json.JSONDecodeError:
                    log.error("Unparseable R response: %.200s", line)
            elif line:
                log.debug("R stdout: %s", line)
        responses.put(None)  # EOF sentinel

    def _read_stderr(self, proc: subprocess.Popen[str]) -> None:
        assert proc.stderr
        for line in proc.stderr:
            line = line.rstrip()
            if line:
                self.stderr_tail = (self.stderr_tail + [line])[-50:]
                log.debug("R stderr: %s", line)

    def _await(self, req_id: str, timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.kill()
                raise REngineError(f"R did not respond within {timeout:.0f}s; the R worker was stopped "
                                   "and will restart on next use.")
            try:
                msg = self._responses.get(timeout=remaining)
            except queue.Empty:
                continue
            if msg is None:
                tail = "\n".join(self.stderr_tail[-10:])
                self._proc = None
                raise REngineError(f"R worker exited unexpectedly.\n{tail}".strip())
            if msg.get("id") == req_id:
                return msg
            log.warning("Discarding stale R response %s", msg.get("id"))

    def request(self, cmd: str, args: dict[str, Any] | None = None, timeout: float = 120.0) -> RResponse:
        with self._lock:
            if not self.alive:
                self.start()
            req_id = str(next(self._ids))
            payload = json.dumps({"id": req_id, "cmd": cmd, "args": args or {}}, ensure_ascii=True)
            assert self._proc and self._proc.stdin
            try:
                self._proc.stdin.write(payload + "\n")
                self._proc.stdin.flush()
            except OSError as e:
                self._proc = None
                raise REngineError(f"Could not send request to R: {e}") from e
            msg = self._await(req_id, timeout)
        if not msg.get("ok"):
            raise RScriptError(msg.get("error") or "Unknown R error", (args or {}).get("script", ""))
        return RResponse(
            result=msg.get("result"), warnings=list(msg.get("warnings") or []),
            versions=msg.get("versions") or {}, elapsed=float(msg.get("elapsed") or 0.0),
        )

    def run(self, script: str, packages: list[str] | None = None, timeout: float = 120.0) -> RResponse:
        return self.request("run", {"script": script, "packages": packages or []}, timeout)

    def ping(self) -> bool:
        return self.request("ping", timeout=10).result == "pong"

    def __enter__(self) -> "RWorker":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
