"""Discovery engine: runs a search over a pool of R workers, records every attempt,
computes multiplicity adjustments and the two rankings."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os
import tempfile
import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..analysis.session import RSession
from ..core.dataset import Dataset
from ..rbridge.worker import REngineError, RScriptError, RWorker
from .rcode import batch_script, setup_script
from .space import (
    Candidate, SearchSettings, candidate_key, count_candidates, dependence_warning, display_formula,
    enumerate_candidates, focal_label, validate_settings,
)

log = logging.getLogger(__name__)

STATUS_LABELS = {
    "complete": "Complete — every candidate in the defined search space was evaluated or classified invalid",
    "partial": "Partial — stopped by the model or time budget",
    "stopped_early": "Stopped early — the significance target was reached (search incomplete)",
    "cancelled": "Cancelled by the user (search incomplete)",
    "failed": "Failed",
    "not_started": "Not started",
    "running": "Running",
}
TIERS = {
    "A": "clean diagnostics, significant after adjustment",
    "B": "clean diagnostics, significant before adjustment only",
    "C": "significant before adjustment, but flagged by screening",
    "D": "not significant",
}


def data_signature(ds: Dataset) -> str:
    meta = [v.to_dict() for v in ds.variables.values()]
    extra = [[d.to_dict() for d in ds.derived.values()], [f.to_dict() for f in ds.filters],
             {k: v for k, v in (ds.holdout.to_dict() if ds.holdout else {}).items() if k != "evaluations"}]
    payload = json.dumps([ds.source.sha256, meta, extra], sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class DiscoveryRun:
    settings: SearchSettings
    family: str
    total: int
    signature: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: str = field(default_factory=lambda: _dt.datetime.now().isoformat(timespec="seconds"))
    status: str = "not_started"
    stop_reason: str = ""
    n_done: int = 0                     # contiguous prefix of candidate indices processed
    stop_index: int | None = None       # modes A/B: index of the qualifying candidate that ended the run
    rows: list[dict[str, Any]] = field(default_factory=list)
    elapsed_s: float = 0.0
    n_sample: int | None = None
    engine: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return self.status == "complete"

    @property
    def resumable(self) -> bool:
        return self.status in ("partial", "cancelled") and self.n_done < self.total

    def meta_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in ("id", "family", "total", "signature", "created_at", "status",
                                               "stop_reason", "n_done", "stop_index", "elapsed_s",
                                               "n_sample", "engine", "warnings")} | {"settings": self.settings.to_dict()}

    @classmethod
    def from_meta(cls, meta: dict[str, Any], rows: list[dict[str, Any]]) -> "DiscoveryRun":
        meta = dict(meta)
        settings = SearchSettings.from_dict(meta.pop("settings"))
        run = cls(settings=settings, **{k: meta[k] for k in ("family", "total", "signature")})
        for k in ("id", "created_at", "status", "stop_reason", "n_done", "stop_index", "elapsed_s",
                  "n_sample", "engine"):
            setattr(run, k, meta[k])
        run.warnings = list(meta.get("warnings", []))
        run.rows = rows
        return run


# ---- execution ---------------------------------------------------------------------

class _PooledSession:
    def __init__(self, session: RSession, setup: str):
        self.session = session
        self.setup = setup
        self.setup_gen = -1
        self.n_sample: int | None = None

    def ensure(self, ds: Dataset) -> None:
        self.session.ensure_data(ds)
        if self.setup_gen != self.session.worker.generation:
            out = self.session.worker.run(self.setup, timeout=600).result
            self.n_sample = int(out["n"])
            self.setup_gen = self.session.worker.generation

    def run_batch(self, ds: Dataset, script: str, n: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        self.ensure(ds)
        resp = self.session.worker.run(script, packages=["detectseparation", "lme4", "stats"], timeout=120 + 20 * n)
        out = resp.result if isinstance(resp.result, list) else [resp.result]
        return out, resp.versions


def default_workers(s: SearchSettings) -> int:
    cores = os.cpu_count() or 2
    n = s.workers if s.workers > 0 else max(1, cores - 1)
    return max(1, min(n, cores))


def create_run(ds: Dataset, s: SearchSettings) -> DiscoveryRun:
    family = validate_settings(ds, s)
    run = DiscoveryRun(settings=s, family=family, total=count_candidates(s), signature=data_signature(ds))
    dep = dependence_warning(ds, s)
    if dep:
        run.warnings.append(dep)
    return run


def execute(run: DiscoveryRun, ds: Dataset, *, rscript: str | None = None,
            progress: Callable[[DiscoveryRun], None] | None = None,
            cancel: threading.Event | None = None, workdir: str | Path | None = None) -> DiscoveryRun:
    """Run (or resume) a search. Blocks; call from a background thread."""
    s = run.settings
    if run.signature != data_signature(ds):
        raise ValueError("The data or variable settings changed since this search was created; "
                         "start a new search.")
    if run.status not in ("not_started", "partial", "cancelled"):
        raise ValueError(f"A {run.status} search cannot be resumed.")
    cancel = cancel or threading.Event()
    workdir = Path(workdir or tempfile.mkdtemp(prefix="sx_discovery_"))
    setup = setup_script(ds, s, run.family)
    n_workers = default_workers(s)
    pooled: list[_PooledSession] = []
    run.status, run.stop_reason = "running", ""
    t_start = time.perf_counter()
    elapsed_before = run.elapsed_s
    try:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            # start workers in parallel
            starts = []
            for i in range(n_workers):
                w = RWorker(rscript)
                ps = _PooledSession(RSession(w, workdir / f"w{i}"), setup)
                pooled.append(ps)
                starts.append(ex.submit(lambda p=ps: (p.session.worker.start(), p.ensure(ds))))
            for f in starts:
                f.result()
            run.n_sample = pooled[0].n_sample if s.sample == "shared" else None
            _dispatch(run, ds, pooled, ex, cancel, progress, t_start, elapsed_before)
    except Exception as e:
        run.status, run.stop_reason = "failed", str(e)
        raise
    finally:
        run.elapsed_s = elapsed_before + (time.perf_counter() - t_start)
        for p in pooled:
            p.session.worker.stop()
    return run


def _dispatch(run: DiscoveryRun, ds: Dataset, pooled: list[_PooledSession], ex: ThreadPoolExecutor,
              cancel: threading.Event, progress, t_start: float, elapsed_before: float) -> None:
    s = run.settings
    it = enumerate_candidates(s, start=run.n_done)
    # Budgets apply per execution, so resuming a budget-limited search continues it.
    budget = s.max_models
    deadline = None if s.max_minutes is None else t_start + s.max_minutes * 60
    target = 1 if s.mode == "A" else s.target if s.mode == "B" else None
    idle = list(pooled)
    in_flight: dict[Future, tuple[_PooledSession, list[Candidate]]] = {}
    dispatched_to = run.n_done
    exhausted = stop = False
    reason = ""
    n_sig = sum(1 for r in run.rows if _qualifies(r, s.alpha))
    while True:
        while not stop and idle:
            size = s.batch_size if budget is None else min(s.batch_size, budget)
            batch = list(islice(it, size)) if size > 0 else []
            if not batch:
                stop = True
                if budget is not None and budget <= 0 and dispatched_to < run.total:
                    reason = "model budget reached"
                else:
                    exhausted = True
                break
            if budget is not None:
                budget -= len(batch)
            dispatched_to = batch[-1].index + 1
            ps = idle.pop()
            fut = ex.submit(ps.run_batch, ds, batch_script(ds, batch), len(batch))
            in_flight[fut] = (ps, batch)
        if not in_flight:
            break
        done, _ = wait(list(in_flight), timeout=0.5, return_when=FIRST_COMPLETED)
        for fut in done:
            ps, batch = in_flight.pop(fut)
            idle.append(ps)
            try:
                results, versions = fut.result()
                if versions and not run.engine:
                    run.engine = versions
            except (RScriptError, REngineError) as e:
                log.error("Batch failed: %s", e)
                results = [{"i": c.index, "status": "failed", "reason": f"R error: {e}", "flags": ""} for c in batch]
            by_idx = {int(r["i"]): r for r in results}
            for c in batch:
                row = _row(ds, run, c, by_idx.get(c.index, {"status": "failed", "reason": "no result", "flags": ""}))
                run.rows.append(row)
                n_sig += _qualifies(row, s.alpha)
        if not stop:
            if cancel.is_set():
                stop, reason = True, "cancelled"
            elif deadline is not None and time.perf_counter() > deadline:
                stop, reason = True, "time budget reached"
            elif target is not None and n_sig >= target:
                stop, reason = True, "target reached"
        if progress:
            run.n_done = dispatched_to
            run.elapsed_s = elapsed_before + (time.perf_counter() - t_start)
            progress(run)
    run.n_done = dispatched_to
    run.rows.sort(key=lambda r: r["idx"])
    if target is not None:
        qualifying = [r["idx"] for r in run.rows if _qualifies(r, s.alpha)]
        if len(qualifying) >= target:
            run.stop_index = qualifying[target - 1]
            for r in run.rows:
                r["speculative"] = r["idx"] > run.stop_index
            run.status = "stopped_early" if run.stop_index < run.total - 1 or len(run.rows) < run.total else "complete"
            run.stop_reason = f"mode {s.mode}: significance target of {target} reached at candidate {run.stop_index}"
            if run.status == "complete":
                run.stop_reason = ""
            return
    if exhausted and run.n_done >= run.total:
        run.status, run.stop_reason = "complete", ""
    elif reason == "cancelled":
        run.status, run.stop_reason = "cancelled", reason
    else:
        run.status, run.stop_reason = "partial", reason or "stopped"


def _qualifies(row: dict[str, Any], alpha: float) -> bool:
    return row["status"] in ("ok", "warning") and row.get("p") is not None and row["p"] < alpha \
        and not row.get("speculative")


def _num(x: Any) -> float | None:
    if x is None or isinstance(x, str):
        return None
    if isinstance(x, list):
        x = x[0] if x else None
        return _num(x)
    f = float(x)
    return None if np.isnan(f) else f


def _row(ds: Dataset, run: DiscoveryRun, c: Candidate, r: dict[str, Any]) -> dict[str, Any]:
    keys = ("n", "n_params", "statistic", "df1", "df2", "p", "effect", "partial_eta2", "estimate", "se",
            "ci_low", "ci_high", "r2", "aic", "bic")
    flags = r.get("flags") or ""
    if isinstance(flags, list):
        flags = "; ".join(flags)
    return {
        "idx": c.index,
        "key": candidate_key(c),
        "focal": list(c.focal),
        "covariates": list(c.covariates),
        "order": c.order,
        "interaction": focal_label(ds, c),
        "formula": display_formula(ds, run.settings.outcome, c),
        "status": str(r.get("status", "failed")),
        "reason": str(r.get("reason") or ""),
        "flags": str(flags),
        "speculative": False,
        **{k: _num(r.get(k)) for k in keys},
    }


# ---- adjustments & rankings -------------------------------------------------------------

def bh(p: np.ndarray) -> np.ndarray:
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p, kind="stable")
    ranked = p[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out


def holm(p: np.ndarray) -> np.ndarray:
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p, kind="stable")
    ranked = p[order] * (m - np.arange(m))
    adj = np.maximum.accumulate(ranked)
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out


def results_frame(run: DiscoveryRun) -> pd.DataFrame:
    """All attempts with adjusted p-values, significance and support tiers."""
    s = run.settings
    df = pd.DataFrame(run.rows)
    if df.empty:
        return df
    in_family = df["status"].isin(["ok", "warning"]) & df["p"].notna() & ~df["speculative"]
    df["in_family"] = in_family
    p = df.loc[in_family, "p"].to_numpy(float)
    df["p_bh"] = np.nan
    df["p_holm"] = np.nan
    df.loc[in_family, "p_bh"] = bh(p)
    df.loc[in_family, "p_holm"] = holm(p)
    df["p_adj"] = df["p_bh"] if s.correction == "BH" else df["p_holm"]
    df["sig_raw"] = in_family & (df["p"] < s.alpha)
    df["sig_adj"] = in_family & (df["p_adj"] < s.alpha)
    clean = df["status"] == "ok"
    df["tier"] = np.select(
        [~in_family, df["sig_adj"] & clean, df["sig_raw"] & clean, df["sig_raw"]],
        ["", "A", "B", "C"], default="D")
    return df


def discovery_ranking(df: pd.DataFrame, by: str = "p") -> pd.DataFrame:
    if df.empty:
        return df
    return df[df["in_family"]].sort_values([by, "idx"], kind="stable")


def support_ranking(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    sub = df[df["in_family"]].copy()
    sub["_tier"] = sub["tier"].map({"A": 0, "B": 1, "C": 2, "D": 3})
    sub["_effect"] = -sub["effect"].fillna(0).abs()
    return sub.sort_values(["_tier", "_effect", "n_params", "idx"], kind="stable").drop(columns=["_tier", "_effect"])


def summary(run: DiscoveryRun, df: pd.DataFrame) -> dict[str, Any]:
    n_attempted = len(df)
    counts = df["status"].value_counts().to_dict() if n_attempted else {}
    fam = int(df["in_family"].sum()) if n_attempted else 0
    return {
        "status": run.status,
        "status_label": STATUS_LABELS.get(run.status, run.status),
        "stop_reason": run.stop_reason,
        "total": run.total,
        "attempted": n_attempted,
        "family_size": fam,
        "ok": counts.get("ok", 0), "warning": counts.get("warning", 0),
        "invalid": counts.get("invalid", 0), "failed": counts.get("failed", 0),
        "speculative": int(df["speculative"].sum()) if n_attempted else 0,
        "sig_raw": int(df["sig_raw"].sum()) if n_attempted else 0,
        "sig_adj": int(df["sig_adj"].sum()) if n_attempted else 0,
        "expected_false_raw": fam * run.settings.alpha,
        "provisional": run.status != "complete",
    }
