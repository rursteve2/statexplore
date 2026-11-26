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
from ..rbridge.worker import REngineError, RScriptError, RTimeoutError, RWorker
from .rcode import batch_script, setup_script
from .space import (
    FAMILIES, Candidate, SearchSettings, candidate_key, candidate_tests, coding, count_candidates,
    dependence_warning, display_formula, enumerate_candidates, focal_label, full_terms, model_type, test_label,
    validate_settings, view,
)

log = logging.getLogger(__name__)

STATUS_LABELS = {
    "complete": "Complete — every candidate in the defined search space was evaluated or classified invalid",
    "partial": "Partial — stopped by the model or time budget",
    "incomplete": "Incomplete — every candidate was attempted, but some timed out (not evaluated)",
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
    validation: dict[int, str] = field(default_factory=dict)   # candidate index -> stability summary
    coding: dict[str, Any] = field(default_factory=dict)       # v1.1: family, outcome treatment, event/reference
    fit_id: str = ""                                           # v1.2: fit identity hash (F6)
    n_times: int | None = None                                 # v1.2 longitudinal: distinct time points
    executions: list[dict[str, Any]] = field(default_factory=list)  # v1.3: budget + range of each execution
    data_snapshot: dict[str, Any] = field(default_factory=dict)     # v1.3.2: the data view used (STAT-39)
    validation_runs: list[dict[str, Any]] = field(default_factory=list)  # v1.3.3: immutable AMD validation runs (J7)

    @property
    def is_complete(self) -> bool:
        return self.status == "complete"

    @property
    def resumable(self) -> bool:
        return self.status in ("partial", "cancelled") and self.n_done < self.total

    def budget_text(self) -> str:
        """Budgets of every execution (each continuation has its own model / time budget)."""
        s = self.settings
        if not self.executions:
            return f"{s.max_models or 'no'} models, {s.max_minutes or 'no'} minutes (per execution)"
        parts = []
        for i, e in enumerate(self.executions, start=1):
            rng = f"candidates {e.get('start', 0):,}–{max(e.get('end', e.get('start', 0)) - 1, e.get('start', 0)):,}" \
                if "end" in e else "running"
            parts.append(f"run {i}: {e.get('max_models') or 'no'} models / {e.get('max_minutes') or 'no'} min "
                         f"({rng})")
        return "; ".join(parts) + " — each continuation evaluates up to its own budget of further models"

    def meta_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in ("id", "family", "total", "signature", "created_at", "status",
                                               "stop_reason", "n_done", "stop_index", "elapsed_s",
                                               "n_sample", "engine", "warnings", "coding", "fit_id", "n_times",
                                               "executions", "data_snapshot", "validation_runs")} | {
            "settings": self.settings.to_dict(), "validation": {str(k): v for k, v in self.validation.items()}}

    @classmethod
    def from_meta(cls, meta: dict[str, Any], rows: list[dict[str, Any]]) -> "DiscoveryRun":
        meta = dict(meta)
        settings = SearchSettings.from_dict(meta.pop("settings"))
        run = cls(settings=settings, **{k: meta[k] for k in ("family", "total", "signature")})
        for k in ("id", "created_at", "status", "stop_reason", "n_done", "stop_index", "elapsed_s",
                  "n_sample", "engine"):
            setattr(run, k, meta[k])
        run.warnings = list(meta.get("warnings", []))
        run.coding = dict(meta.get("coding") or {})
        run.fit_id = str(meta.get("fit_id") or "")
        run.n_times = meta.get("n_times")
        run.executions = list(meta.get("executions") or [])
        run.data_snapshot = dict(meta.get("data_snapshot") or {})
        run.validation_runs = list(meta.get("validation_runs") or [])
        run.validation = {int(k): v for k, v in (meta.get("validation") or {}).items()}
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
        resp = self.session.worker.run(script, packages=ENGINE_PACKAGES,
                                       timeout=120 + 20 * n)
        out = resp.result if isinstance(resp.result, list) else [resp.result]
        return out, resp.versions


def default_workers(s: SearchSettings) -> int:
    cores = os.cpu_count() or 2
    n = s.workers if s.workers > 0 else max(1, cores - 1)
    return max(1, min(n, cores))


def create_run(ds: Dataset, s: SearchSettings) -> DiscoveryRun:
    family = validate_settings(ds, s)
    fit_id = fit_identity(ds, s, family)
    base = ds
    ds = view(ds, s)
    run = DiscoveryRun(settings=s, family=family, total=count_candidates(s), signature=data_signature(ds))
    run.coding = coding(ds, s, family)
    run.fit_id = fit_id
    from .history import snapshot

    run.data_snapshot = snapshot(base)
    if s.participant and s.time:
        run.n_times = int(ds.analysis_frame([s.time])[s.time].nunique())
    dep = dependence_warning(ds, s)
    if dep:
        run.warnings.append(dep)
    return run


def fit_identity(ds: Dataset, s: SearchSettings, family: str) -> str:
    """Everything that changes a fitted model (F6): the analysed data and metadata as this search
    sees them (treatments, filters, derived variables, holdout, missing codes, category order,
    reference levels), the outcome family and its resolved coding (event / reference category,
    ordered levels), the grouping and time structure, the sample strategy and the variable pool
    that defines the shared sample, and fitting settings. Presentation settings (alpha, adjustment,
    highlighting, sorting) are deliberately excluded: changing them never refits."""
    from .space import binomial_levels

    v = view(ds, s)
    coding_id: dict[str, Any] = {"levels": [str(x) for x in v.levels(s.outcome)] if v.var(s.outcome).is_categorical else None}
    base = {"glmm": "binomial", "clmm": "ordinal", "lmm": "gaussian", "gee": s.family or "gaussian"}.get(family, family)
    if base == "binomial":                      # event coding of every logistic engine (glm, glmer, GEE)
        ev, ref = binomial_levels(v, s)
        coding_id["event"], coding_id["reference"] = str(ev), str(ref)
    if family == "multinomial":
        coding_id["reference"] = str(s.reference if s.reference is not None else v.levels(s.outcome)[0])
    payload = {
        "data": data_signature(v), "family": family, "outcome": s.outcome, "coding": coding_id,
        "participant": s.participant, "time": getattr(s, "time", None), "time_type": getattr(s, "time_type", None),
        "structure": getattr(s, "structure", None), "gee_corstr": getattr(s, "gee_corstr", None),
        # v1.3.2: nesting changes the model (composite participants, cluster variance); identity schema 2
        "cluster": getattr(s, "cluster", None), "id_policy": getattr(s, "id_policy", "global") if getattr(s, "cluster", None) else None,
        "identity_schema": 2, "optimizer": "lme4 default; nlme nlminb then optim on optimizer errors, msMaxIter 200",
        "sample": s.sample, "pool": sorted(s.candidates + s.required) if s.sample == "shared" else None,
        "required": sorted(s.required), "max_params": s.max_params, "conf": s.conf,
        "min_obs": getattr(s, "min_obs", 0),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


ENGINE_PACKAGES = ["detectseparation", "lme4", "ordinal", "nnet", "stats", "nlme", "mmrm", "geepack"]


# Version of the rules that turn a fit into a row (bump when a stored statistic changes definition).
# 2 = v1.3.1: AIC/BIC from the fitted object (STAT-16). Rows from older rules are refitted, never reused.
FIT_RULES = 3   # 3 = v1.3.2: identity fingerprints, diagnostics, direct R2, shared structure builder, msMaxIter 200


def _engine_id(engine: dict[str, Any]) -> tuple:
    pk = engine.get("packages") or {}
    return (engine.get("r_version"), tuple((k, pk.get(k)) for k in ENGINE_PACKAGES), engine.get("fit_rules"))


def cache_from(run: DiscoveryRun, previous: list[DiscoveryRun], ds: Dataset | None = None) -> dict[str, dict[str, Any]]:
    """Rows already fitted by earlier searches with the same fit identity (§21 caching, F6).
    Without `ds`, runs are matched on their stored identity (runs created by v1.2)."""
    ident = run.fit_id or (fit_identity(ds, run.settings, run.family) if ds is not None else "")
    out: dict[str, dict[str, Any]] = {}
    if not ident:
        return out
    for prev in previous:
        if prev is run or prev.signature != run.signature or prev.family != run.family:
            continue
        pid = prev.fit_id
        if not pid and ds is not None:
            try:
                pid = fit_identity(ds, prev.settings, prev.family)
            except Exception:  # an old run whose variables no longer exist
                continue
        if pid != ident:
            continue
        for r in prev.rows:
            if r["status"] != "failed" and not r.get("speculative"):
                out.setdefault(r["key"], dict(r, _engine=_engine_id(prev.engine), _from=prev.id))
    return out


def execute(run: DiscoveryRun, ds: Dataset, *, rscript: str | None = None,
            progress: Callable[[DiscoveryRun], None] | None = None,
            cancel: threading.Event | None = None, workdir: str | Path | None = None,
            cache: dict[str, dict[str, Any]] | None = None) -> DiscoveryRun:
    """Run (or resume) a search. Blocks; call from a background thread."""
    s = run.settings
    ds = view(ds, s)
    if run.signature != data_signature(ds):
        raise ValueError("The data or variable settings changed since this search was created; "
                         "start a new search.")
    if run.status not in ("not_started", "partial", "cancelled"):
        raise ValueError(f"A {run.status} search cannot be resumed.")
    cancel = cancel or threading.Event()
    own_workdir = workdir is None
    workdir = Path(workdir or tempfile.mkdtemp(prefix="sx_discovery_"))
    setup = setup_script(ds, s, run.family)
    n_workers = default_workers(s)
    pooled: list[_PooledSession] = []
    run.status, run.stop_reason = "running", ""
    run.executions.append({"start": run.n_done, "max_models": s.max_models, "max_minutes": s.max_minutes})
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
            current = pooled[0].session.worker.request("versions", {"packages": ENGINE_PACKAGES}).result
            current = {**current, "fit_rules": FIT_RULES}
            if not run.engine:
                run.engine = current
            # cached fits are reused only when R and the fitting packages are the same versions (F6)
            usable = {k: r for k, r in (cache or {}).items() if r.get("_engine") == _engine_id(current)}
            _dispatch(run, ds, pooled, ex, cancel, progress, t_start, elapsed_before, usable)
    except Exception as e:
        run.status, run.stop_reason = "failed", str(e)
        raise
    finally:
        run.elapsed_s = elapsed_before + (time.perf_counter() - t_start)
        for p in pooled:
            p.session.worker.stop()
        if own_workdir:
            import shutil

            shutil.rmtree(workdir, ignore_errors=True)
    return run


def _dispatch(run: DiscoveryRun, ds: Dataset, pooled: list[_PooledSession], ex: ThreadPoolExecutor,
              cancel: threading.Event, progress, t_start: float, elapsed_before: float,
              cache: dict[str, dict[str, Any]]) -> None:
    s = run.settings
    have = {int(r["idx"]) for r in run.rows}
    # resume: candidates evaluated after an interrupted batch already have rows and are not fitted twice
    it = (c for c in enumerate_candidates(s, start=run.n_done) if c.index not in have)
    # Budgets apply per execution, so resuming a budget-limited search continues it.
    budget = s.max_models
    deadline = None if s.max_minutes is None else t_start + s.max_minutes * 60
    target = 1 if s.mode == "A" else s.target if s.mode == "B" else None
    idle = list(pooled)
    in_flight: dict[Future, tuple[_PooledSession, list[Candidate]]] = {}
    dispatched_to = run.n_done
    exhausted = stop = False
    reason = ""
    aborted: list[int] = []           # first candidate index of each batch interrupted by a cancellation
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
            cached = [c for c in batch if candidate_key(c) in cache]
            for c in cached:  # identical model already fitted on identical data: reuse its result
                src = cache[candidate_key(c)]
                row = {k: v for k, v in src.items() if not k.startswith("_")}
                row.update(idx=c.index, cached=True, speculative=False, run_id=run.id,
                           candidate_id=f"{run.id}:{c.index}",
                           cache_reason=f"reused from search {src.get('_from')}: identical fit identity "
                                        "(data, coding, structure, sample) and R/package versions")
                run.rows.append(row)
                n_sig += _qualifies(row, s.alpha)
            batch = [c for c in batch if candidate_key(c) not in cache]
            if not batch:
                continue
            ps = idle.pop()
            fut = ex.submit(ps.run_batch, ds, batch_script(ds, batch, s), len(batch))
            in_flight[fut] = (ps, batch)
        if not in_flight:
            break
        if cancel.is_set() and not aborted:
            # V133-QA-02 / STAT-31: cancel promptly - stop the R workers of the running batches instead of waiting
            # for them; their candidates are not evaluated (never recorded or cached) and stay pending for a resume
            stop, reason = True, "cancelled"
            for fut, (ps, batch) in in_flight.items():
                aborted.append(batch[0].index)
                ps.session.worker.kill()
            for fut in list(in_flight):
                try:
                    fut.result(timeout=30)
                except Exception:   # noqa: BLE001 - the killed worker's error is expected
                    pass
            in_flight.clear()
            break
        done, _ = wait(list(in_flight), timeout=0.5, return_when=FIRST_COMPLETED)
        for fut in done:
            ps, batch = in_flight.pop(fut)
            idle.append(ps)
            try:
                results, versions = fut.result()
                if versions and not run.engine:
                    run.engine = {**versions, "fit_rules": FIT_RULES}
            except RTimeoutError as e:     # P1-06: a timeout is not a completed evaluation
                log.error("Batch timed out: %s", e)
                results = [{"i": c.index, "status": "failed", "reason": f"timed out (worker execution limit): {e}",
                            "flags": "", "timed_out": True} for c in batch]
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
    if aborted:
        dispatched_to = min(aborted)        # contiguous prefix of evaluated candidates (later rows are kept)
    run.n_done = dispatched_to
    if run.executions:
        run.executions[-1]["end"] = dispatched_to
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
    n_timeout = sum(1 for r in run.rows if r.get("timed_out"))
    if exhausted and run.n_done >= run.total and n_timeout:
        run.status = "incomplete"
        run.stop_reason = (f"{n_timeout:,} candidate(s) timed out and were not evaluated; adjusted p-values are "
                           "provisional")
    elif exhausted and run.n_done >= run.total:
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


CI_SCALES = {"gaussian": "coefficient", "lmm": "coefficient", "binomial": "log-odds", "ordinal": "log-odds",
             "glmm": "log-odds (subject-specific)", "clmm": "log-odds (subject-specific)"}


def _ci_scale(run: DiscoveryRun) -> str:
    if run.family == "gee":
        return "log-odds (population-average)" if run.settings.family == "binomial" else "coefficient (population-average)"
    return CI_SCALES.get(run.family, "")


def _logit_scale(run: DiscoveryRun) -> bool:
    return run.family in ("binomial", "ordinal", "glmm", "clmm") or (run.family == "gee" and run.settings.family == "binomial")


def _test_record(ds: Dataset, run: DiscoveryRun, c: Candidate, plan: dict[str, Any], r: dict[str, Any]) -> dict[str, Any]:
    fam = run.family
    from .modelspec import display

    dn = lambda n: ds.var(n).display_name  # noqa: E731
    out = {"test_id": int(plan["id"]), "type": plan["type"], "label": test_label(ds, c, plan),
           "drop": [":".join(t) for t in plan["drop"]],
           "reference": display(plan["reduced"], dn(run.settings.outcome), dn),
           **{k: _num(r.get(k)) for k in ("statistic", "df1", "df_den", "p", "effect", "partial_eta2",
                                          "estimate", "se", "ci_low", "ci_high")}}
    single = len(plan["drop"]) == 1 and out["df1"] == 1 and fam != "multinomial"
    has_ci = out["ci_low"] is not None and out["ci_high"] is not None
    out["ci_applicable"] = "same_estimand" if single and has_ci else (
        "joint" if (out["df1"] or 0) > 1 or len(plan["drop"]) > 1 or fam == "multinomial" else "unavailable")
    if r.get("withheld"):          # STAT-41: materially negative LR or invalid df -> no p-value, reason kept
        out["test_status"], out["test_reason"], out["p"] = "withheld", str(_scalar_text(r.get("withheld"))), None
    out["ci_scale"] = _ci_scale(run)
    out["ci_null"] = 0.0
    out["ci_level"] = run.settings.conf
    out["ci_effect_id"] = out["p_test_id"] = "+".join(out["drop"])
    if out["ci_applicable"] == "same_estimand":
        out["ci_method"] = ("Wald (t, residual df)" if fam == "gaussian" else
                            "Wald (robust sandwich SE)" if fam == "gee" else "Wald (normal approximation)")
        out["ci_note"] = ("The 1-df F test and the t-based CI are the same test." if fam == "gaussian" else
                          "Robust Wald test and robust Wald CI: the same test." if fam == "gee" else
                          "p-value from the likelihood-ratio test, CI from the Wald approximation: they can "
                          "disagree near the significance boundary.")
        if _logit_scale(run):
            out["odds_ratio"] = float(np.exp(out["estimate"]))
            out["or_low"], out["or_high"] = float(np.exp(out["ci_low"])), float(np.exp(out["ci_high"]))
    else:
        out["ci_method"] = ""
        out["ci_note"] = ("Joint test of several coefficients: there is no single-coefficient CI. Use "
                          "conditional slopes or contrasts (Teaching Studio) for interval estimates."
                          if out["ci_applicable"] == "joint" else "")
    return out


def _scalar_text(x: Any) -> str:
    if isinstance(x, list):
        return "; ".join(str(v) for v in x)
    return "" if x is None else str(x)


def _row(ds: Dataset, run: DiscoveryRun, c: Candidate, r: dict[str, Any]) -> dict[str, Any]:
    from .modelspec import counts

    keys = ("n", "n_params", "n_groups", "statistic", "df1", "df2", "p", "effect", "partial_eta2", "estimate",
            "se", "ci_low", "ci_high", "r2", "adj_r2", "aic", "bic", "loglik")
    flags = r.get("flags") or ""
    if isinstance(flags, list):
        flags = "; ".join(flags)
    status = str(r.get("status", "failed"))
    tests: list[dict[str, Any]] = []
    if status in ("ok", "warning"):
        got = r.get("tests") or []
        by_id = {int(t["id"]): t for t in (got if isinstance(got, list) else [got])}
        tests = [_test_record(ds, run, c, plan, by_id[plan["id"]]) for plan in candidate_tests(c)
                 if plan["id"] in by_id]
    terms = full_terms(c)
    cnt = counts(terms)
    row = {
        "idx": c.index,
        "key": candidate_key(c),
        "focal": list(c.focal),
        "covariates": list(c.covariates),
        "kind": c.kind,
        "inter": [list(t) for t in c.inter],
        "order": max((len(t) for t in terms), default=1) if c.kind != "focal" else c.order,
        "interaction": focal_label(ds, c),
        "formula": display_formula(ds, run.settings.outcome, c) + _random_suffix(ds, run, c),
        "model_type": model_type(c),
        "structure": (c.structure or run.settings.structure) if run.settings.participant else "",
        **cnt,
        "n_re_params": _re_params(run, c),
        "cached": False,
        "timed_out": bool(r.get("timed_out")),
        "status": status,
        "reason": str(r.get("reason") or ""),
        "flags": str(flags),
        "speculative": False,
        **{k: _num(r.get(k)) for k in keys},
        "tests": tests,
        "run_id": run.id,
        "candidate_id": f"{run.id}:{c.index}",
        "model_id": hashlib.sha256(f"{run.fit_id}|{candidate_key(c)}".encode()).hexdigest()[:16],
        "fitted_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "cache_reason": "",
    }
    if run.family in ("lmm", "glmm", "clmm", "gee"):
        row["df2"] = None       # v1.1 stored the participant count here; it is n_groups (M §9)
    # AIC/BIC come from the fitted object (sx_ic): native for every engine but mmrm (STAT-16: recomputing BIC with
    # log(n) here disagreed with nlme's REML log(n - p) by ~0.004)
    row["bic_n"] = _num(r.get("bic_n"))
    row["n_lik_params"] = _num(r.get("n_lik_params"))       # every likelihood parameter, from the fitted object
    row["ic_basis"] = str(r.get("ic_basis") or "")
    row["rows_sig"] = str(r.get("rows_sig") or "")
    if r.get("qic") is not None:
        row["qic"] = _num(r.get("qic"))
    from ..analysis.descriptor import parse_identity

    row["identity"] = parse_identity(r.get("identity"))
    dg = r.get("diagnostics")
    if isinstance(dg, dict):
        row["diagnostics"] = {k: (v[0] if isinstance(v, list) and len(v) == 1 else v) for k, v in dg.items()}
    if run.settings.participant:
        cl = r.get("clustered")
        row["clustered"] = bool(cl[0] if isinstance(cl, list) else cl) if cl is not None else bool(
            getattr(run.settings, "cluster", None))
        row["observed_clusters"] = _num(r.get("observed_clusters"))
        row["modeled_clusters"] = _num(r.get("modeled_clusters"))
        row["participant_key_policy"] = (("local (cluster, ID)" if run.settings.id_policy == "local" else
                                          "global ID (one cluster per ID)") if getattr(run.settings, "cluster", None)
                                         else "participant ID")
    return row


def _random_suffix(ds: Dataset, run: DiscoveryRun, c: Candidate) -> str:
    from .space import random_part

    rp = random_part(ds, run.settings, c.structure or None, c.clustered)
    return f" + {rp}" if rp.startswith("(") else (f"  [{rp}]" if rp else "")


def _re_params(run: DiscoveryRun, c: Candidate) -> int | None:
    """Random-effect variance/covariance + residual covariance parameters (counted apart from fixed effects)."""
    from .modelspec import random_parameters

    s = run.settings
    if not s.participant:
        return None
    st = c.structure or s.structure or "ri"
    if st.startswith("gee"):
        return {"gee_exch": 1, "gee_ar1": 1, "gee_ind": 0}[st]         # working-correlation parameters
    clustered = bool(getattr(s, "cluster", None)) and c.clustered is not False
    rp = random_parameters(st, n_times=getattr(run, "n_times", None), clustered=clustered)
    return sum(rp.values())


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


TEST_TYPES = {
    "structure": "Random-effect / covariance-structure comparison",
    "focal_interaction": "Focal interaction (nested-model test)",
    "interaction_term": "Individual interaction term",
    "joint_interaction": "Joint test of all interaction terms",
    "main_term": "Main-effect term",
    "overall": "Whole-model test",
}
CI_APPLICABILITY = {
    "same_estimand": "CI of the tested coefficient (same estimand as the p-value)",
    "joint": "Joint test; no single focal-coefficient CI",
    "unavailable": "CI unavailable",
}
LRT_FAMILIES = ("lmm", "glmm", "clmm", "binomial", "ordinal", "multinomial")


def _legacy_test(row: dict[str, Any], family: str, conf: float) -> dict[str, Any]:
    """v1.0/v1.1 rows hold one focal test in the row itself. For mixed searches, v1.1 stored the
    number of participants in `df2`; it is not a denominator df (M §9)."""
    kind = row.get("kind") or "focal"
    one_coef = row.get("ci_low") is not None and row.get("ci_high") is not None
    df1 = row.get("df1")
    return {
        "test_id": 0, "type": "joint_interaction" if kind == "all_two_way" else "focal_interaction",
        "label": row.get("interaction", ""), "statistic": row.get("statistic"),
        "df1": df1, "df_den": row.get("df2") if family == "gaussian" else None,
        "p": row.get("p"), "effect": row.get("effect"), "partial_eta2": row.get("partial_eta2"),
        "estimate": row.get("estimate"), "se": row.get("se"), "ci_low": row.get("ci_low"),
        "ci_high": row.get("ci_high"),
        "ci_applicable": "same_estimand" if one_coef else (
            "joint" if kind == "all_two_way" or (df1 is not None and df1 > 1) or family == "multinomial"
            else "unavailable"),
        "ci_scale": "log-odds" if family in ("binomial", "ordinal") else "coefficient",
        "ci_null": 0.0, "ci_level": conf,
        "ci_method": ("Wald (t, residual df)" if family == "gaussian" else "Wald (normal)") if one_coef else "",
    }


# Nested covariance structures (smaller, larger): (extra parameters, chi-square mixture weights by df).
# A variance tested at zero lies on the boundary: the LR statistic follows a 50:50 mixture (Self & Liang).
# Discrete AR(1) (occasion index) and continuous-time AR(1) (actual time) are NOT nested: never LR-tested against each
# other; they are ranked by AIC/BIC in the model-comparison tab.
NESTED_STRUCTURES = {
    ("ri", "rs"): ("random time slope (variance + covariance)", {1: 0.5, 2: 0.5}),
    ("ri", "rs_uncorr"): ("random time slope variance", {0: 0.5, 1: 0.5}),
    ("rs_uncorr", "rs"): ("intercept-slope covariance", {1: 1.0}),
    ("nlme_ri", "nlme_rs"): ("random time slope (variance + covariance)", {1: 0.5, 2: 0.5}),
    ("nlme_ri", "ar1"): ("AR(1) residual correlation", {1: 1.0}),
    ("nlme_ri", "car1"): ("continuous-time AR(1) residual correlation", {0: 0.5, 1: 0.5}),
    ("mmrm_cs", "mmrm_us"): ("unstructured vs compound-symmetry covariance", None),
    ("mmrm_ar1", "mmrm_us"): ("unstructured vs AR(1) covariance", None),
    ("glmer_ri", "glmer_rs"): ("random time slope (variance + covariance)", {1: 0.5, 2: 0.5}),
    ("clmm_ri", "clmm_rs"): ("random time slope (variance + covariance)", {1: 0.5, 2: 0.5}),
}


def _mixture_p(lr: float, weights: dict[int, float]) -> float:
    from scipy.stats import chi2

    lr = max(lr, 0.0)
    return float(sum(w * (1.0 if k == 0 and lr <= 0 else 0.0 if k == 0 else chi2.sf(lr, k))
                     for k, w in weights.items()))


def _same_rows(a: dict[str, Any], b: dict[str, Any]) -> bool:
    ia, ib = a.get("identity") or {}, b.get("identity") or {}
    if ia.get("rows") and ib.get("rows"):
        return ia["rows"] == ib["rows"]
    if a.get("rows_sig") and b.get("rows_sig"):
        return a["rows_sig"] == b["rows_sig"]
    return a.get("n") == b.get("n")


def _boundary(row: dict[str, Any]) -> bool:
    dg = row.get("diagnostics") or {}
    if dg.get("status"):
        return dg["status"] in ("singular", "near_boundary")
    flags = str(row.get("flags") or "").lower()  # rows without the diagnostics record (older searches)
    return "singular fit" in flags or "near-boundary" in flags


def structure_tests(run: DiscoveryRun) -> dict[int, list[dict[str, Any]]]:
    """Workflow B: tests between nested structures fitted in this run (same fixed part, same rows, REML for Gaussian
    models). A pair is tested only when it differs in exactly one component (random part, residual correlation or
    the cluster variance); the reference distribution follows the tested null (H6). Attached to the larger
    structure's row; never a focal test."""
    from ..analysis.covstruct import contrast, from_key
    from ..analysis.lrtest import lr_test
    from .space import STRUCTURES

    fitted = [r for r in run.rows if r.get("kind") == "structure" and r["status"] in ("ok", "warning")]
    out: dict[int, list[dict[str, Any]]] = {}
    has_cluster = bool(getattr(getattr(run, "settings", None), "cluster", None))

    def struct(r: dict[str, Any]):
        cl = has_cluster and r.get("clustered") is not False
        return from_key(str(r.get("structure") or ""), cl)

    def label(r: dict[str, Any]) -> str:
        base = STRUCTURES.get(r.get("structure"), (str(r.get("structure")),))[0].split(" — ")[0]
        return base + (" without the cluster" if has_cluster and r.get("clustered") is False else "")

    chosen: dict[tuple, tuple[dict, dict, Any]] = {}
    for a in fitted:
        for b in fitted:
            if a is b:
                continue
            sa, sb = struct(a), struct(b)
            if sa is None or sb is None:
                continue
            c = contrast(sa, sb)
            if not c.nested:
                continue
            key = (sa.random, sa.residual, sa.clustered, sb.random, sb.residual, sb.clustered, b["idx"])
            same_engine = sa.engine == sb.engine
            prev = chosen.get(key)
            if prev is None or (same_engine and prev[2][1] is False):
                chosen[key] = (a, b, (c, same_engine))
    for a, b, (c, _) in sorted(chosen.values(), key=lambda v: (v[1]["idx"], v[0]["idx"])):
        t = lr_test(a.get("loglik"), b.get("loglik"), c.df, c.reference)
        status, reason = t["status"], t["reason"]
        if not _same_rows(a, b):
            status, reason = "withheld", "the two structures were fitted to different rows"
        elif status == "tested" and c.needs_interior and (_boundary(a) or _boundary(b)):
            status, reason = "withheld", ("a variance is at or near zero, so the covariance test has no regular "
                                          "reference distribution; compare by AIC/BIC")
        if status == "tested" and (a["status"] != "ok" or b["status"] != "ok"):
            status = "provisional"
            reason = "; ".join(x for x in (reason, "at least one of the two fits is flagged (see its diagnostics): the "
                                           "p-value is provisional, not clean evidence") if x)
        weights = ({c.df - 1: 0.5, c.df: 0.5} if c.reference == "mixture" else {c.df: 1.0})
        rec = {
            "test_id": len(out.get(int(b["idx"]), [])), "type": "structure",
            "label": f"{label(b)} vs {label(a)}: {c.what}",
            "statistic": t["LR"] if t["LR"] is not None else None, "LR_raw": t["LR_raw"], "df1": float(c.df),
            "p": t["p"] if status != "withheld" else None, "log_p": t["log_p"] if status != "withheld" else None,
            "ci_applicable": "unavailable", "ci_method": "",
            "structure_method": f"likelihood-ratio test; reference {c.reference_text}",
            "reference_distribution": c.reference_text, "null": c.null, "assumptions": c.assumptions,
            "reference_structure": a.get("structure"), "reference_clustered": a.get("clustered"),
            "reference_model_id": a.get("model_id"), "reference_idx": int(a["idx"]),
            "mixture": weights if c.reference == "mixture" else None, "reference_kind": c.reference,
            "boundary": c.reference == "mixture", "test_status": status, "test_reason": reason}
        out.setdefault(int(b["idx"]), []).append(rec)
    # non-Gaussian / marginal structures keep their own nested pairs
    by = {r.get("structure"): r for r in fitted if from_key(str(r.get("structure") or "")) is None}
    for (small, large), (what, weights) in NESTED_STRUCTURES.items():
        a, b = by.get(small), by.get(large)
        if not a or not b or a.get("loglik") is None or b.get("loglik") is None or not _same_rows(a, b):
            continue
        if weights is None:
            df = (b.get("n_re_params") or 0) - (a.get("n_re_params") or 0)
            ref, dfv = "chisq", df
        else:
            dfv = max(weights)
            ref = "mixture" if len(weights) > 1 else "chisq"
        t = lr_test(a["loglik"], b["loglik"], dfv, ref)
        status, reason = t["status"], t["reason"]
        if status == "tested" and (a["status"] != "ok" or b["status"] != "ok"):
            status, reason = "provisional", "at least one of the two fits is flagged: the p-value is provisional"
        out.setdefault(int(b["idx"]), []).append({
            "test_id": len(out.get(int(b["idx"]), [])), "type": "structure",
            "label": f"{STRUCTURES[large][0].split(' — ')[0]} vs {STRUCTURES[small][0].split(' — ')[0]}: {what}",
            "statistic": t["LR"], "LR_raw": t["LR_raw"], "df1": float(dfv or 0),
            "p": t["p"] if status != "withheld" else None, "ci_applicable": "unavailable", "ci_method": "",
            "structure_method": "likelihood-ratio test, " + ("boundary-corrected 50:50 chi-square mixture"
                                                             if ref == "mixture" else f"chi-square({dfv})"),
            "reference_distribution": "50:50 chi-square mixture" if ref == "mixture" else f"chi-square({dfv})",
            "reference_structure": small, "reference_clustered": None, "reference_model_id": a.get("model_id"),
            "reference_idx": int(a["idx"]), "mixture": weights if ref == "mixture" else None, "reference_kind": ref,
            "boundary": ref == "mixture", "test_status": status, "test_reason": reason})
    return out


def row_tests(row: dict[str, Any], family: str, conf: float = 0.95) -> list[dict[str, Any]]:
    """The hypotheses tested in one model attempt; the first is the model's declared focal test."""
    tests = row.get("tests")
    if isinstance(tests, str):
        tests = json.loads(tests)
    if tests:
        return list(tests)
    if row.get("status") in ("invalid", "failed"):
        return []
    return [_legacy_test(row, family, conf)]


def test_method(test_type: str, family: str) -> str:
    if test_type == "structure":
        return "likelihood-ratio test between nested structures (REML for Gaussian; same fixed part)"
    if family == "gaussian":
        return "nested-model F test"
    if family == "gee":
        return "robust (sandwich) Wald test"
    return "likelihood-ratio test (ML fits)" if family in ("lmm", "glmm", "clmm") else "likelihood-ratio test"


def _null_flags(df: pd.DataFrame) -> pd.Series:
    return df["status"].isin(["ok", "warning"]) & df["p"].notna() & ~df["speculative"].astype(bool)


def tests_frame(run: DiscoveryRun, interpret: bool = True) -> pd.DataFrame:
    """One row per tested hypothesis, with its model's fields. Multiplicity families are the test
    types within this run (F2): BH and Holm are computed separately for each type."""
    s = run.settings
    recs = []
    extra = structure_tests(run) if s.participant and s.workflow == "structure" else {}
    for r in run.rows:
        tests = extra.get(int(r["idx"]), []) if r.get("kind") == "structure" else row_tests(r, run.family, s.conf)
        for j, t in enumerate(tests):
            rec = {k: r.get(k) for k in MODEL_FIELDS if k in r}
            rec.update({k: v for k, v in t.items() if k != "test_id"})
            rec["test_id"] = f"{r['idx']}.{t.get('test_id', j)}"
            rec["idx"] = r["idx"]
            rec["primary"] = j == 0
            rec["test_type"] = t.get("type", "focal_interaction")
            recs.append(rec)
    df = pd.DataFrame(recs)
    if df.empty:
        return df
    for c in ("statistic", "df1", "df_den", "p", "effect", "partial_eta2", "estimate", "se", "ci_low", "ci_high",
              "ci_null", "ci_level"):
        if c not in df:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c, default in (("speculative", False), ("flags", ""), ("ci_applicable", "unavailable"), ("ci_scale", ""),
                       ("ci_method", ""), ("label", "")):
        if c not in df:
            df[c] = default
        df[c] = df[c].fillna(default)
    df["speculative"] = df["speculative"].astype(bool)
    df["n_groups"] = pd.to_numeric(df.get("n_groups", np.nan), errors="coerce")
    if run.family == "lmm" and df["n_groups"].isna().all() and "df2" in df and not run.fit_id:
        df["n_groups"] = pd.to_numeric(df["df2"], errors="coerce")       # legacy v1.1 mixed rows
    # V133-UI-08: log10 of the upper tail at full precision (R's log tail when recorded, so a p below the double range
    # is still ordered); missing when there is no p-value (withheld / not applicable), never a bogus 0
    lp = pd.to_numeric(df["log_p"], errors="coerce") / np.log(10) if "log_p" in df else pd.Series(np.nan, index=df.index)
    with np.errstate(divide="ignore"):
        direct = np.where(df["p"] > 0, np.log10(df["p"].where(df["p"] > 0)), np.nan)
    df["log10_p"] = lp.where(lp.notna(), direct)
    df["lrt_df"] = np.where(run.family in LRT_FAMILIES, df["df1"], np.nan)
    df["test_label"] = df["test_type"].map(TEST_TYPES).fillna(df["test_type"])
    df["test_method"] = [test_method(t, run.family) for t in df["test_type"]]
    df["in_family"] = _null_flags(df)
    df["p_bh"] = np.nan
    df["p_holm"] = np.nan
    df["family_size"] = 0
    for ttype, idx in df[df["in_family"]].groupby("test_type").groups.items():
        p = df.loc[idx, "p"].to_numpy(float)
        df.loc[idx, "p_bh"] = bh(p)
        df.loc[idx, "p_holm"] = holm(p)
        df.loc[idx, "family_size"] = len(idx)
    df["p_adj"] = df["p_bh"] if s.correction == "BH" else df["p_holm"]
    df["sig_raw"] = df["in_family"] & (df["p"] < s.alpha)
    df["sig_adj"] = df["in_family"] & (df["p_adj"] < s.alpha)
    if "test_status" not in df:
        df["test_status"] = ""
    df["test_status"] = df["test_status"].fillna("")
    # clean only when the model AND (for structure comparisons) both compared fits are unflagged
    clean = (df["status"] == "ok") & ~df["test_status"].isin(["provisional", "withheld"])
    df["tier"] = np.select(
        [~df["in_family"], df["sig_adj"] & clean, df["sig_raw"] & clean, df["sig_raw"]],
        ["", "A", "B", "C"], default="D")
    df["ci_status"] = df["ci_applicable"].map(CI_APPLICABILITY).fillna("CI unavailable")
    df["search_scope"] = "complete search" if run.status == "complete" else "partial search (provisional)"
    df["adj_method"] = ("Benjamini-Hochberg" if s.correction == "BH" else "Holm") + " within " + df["test_label"]
    df["validation"] = [run.validation.get(int(i), "not validated" if f else "")
                        for i, f in zip(df["idx"], df["in_family"])]
    # interpretations are text built per row: large tables compute them lazily for the visible rows only
    df["interpretation"] = interpretations(df, run) if interpret else ""
    return df


INTERPRET_FIELDS = ("status", "reason", "speculative", "formula", "test_type", "kind", "label", "interaction", "p",
                    "p_adj", "effect", "family_size", "sig_adj", "sig_raw", "estimate", "ci_applicable", "ci_level",
                    "ci_low", "ci_high", "flags", "test_method", "reference")


def interpretations(df: pd.DataFrame, run: DiscoveryRun) -> list[str]:
    cols = [c for c in INTERPRET_FIELDS if c in df]
    return [interpret_row(dict(zip(cols, vals)), run.settings, run.family)
            for vals in zip(*(df[c].tolist() for c in cols))]


MODEL_FIELDS = ("idx", "key", "focal", "covariates", "kind", "order", "interaction", "formula", "cached", "status",
                "reason", "flags", "speculative", "n", "n_params", "n_groups", "r2", "adj_r2", "aic", "bic",
                "loglik", "model_type", "predictors", "n_predictors", "n_main", "n_two_way", "n_three_way",
                "n_terms", "n_re_params", "candidate_id", "model_id", "cache_reason", "df2", "inter", "run_id",
                "fitted_at", "structure", "qic")


def results_frame(run: DiscoveryRun, tf: pd.DataFrame | None = None) -> pd.DataFrame:
    """All attempts (one row per model) with the model's declared focal test, its adjusted
    p-value (within its test-type family), significance and support tier. Pass the run's
    `tests_frame` as `tf` to avoid computing it twice."""
    df = pd.DataFrame(run.rows)
    if df.empty:
        return df
    if "tests" in df:
        df = df.drop(columns=["tests"])
    if tf is None:
        tf = tests_frame(run)
    prim = tf[tf["primary"]].set_index("idx") if len(tf) else pd.DataFrame()
    cols = ("test_type", "test_label", "label", "statistic", "df1", "df_den", "lrt_df", "p", "effect",
            "partial_eta2", "estimate", "se", "ci_low", "ci_high", "ci_applicable", "ci_scale", "ci_null",
            "ci_status", "in_family", "p_bh", "p_holm", "p_adj", "sig_raw", "sig_adj", "tier", "family_size",
            "validation", "interpretation", "test_method")
    for c in cols:
        if len(prim) and c in prim:
            df[c] = df["idx"].map(prim[c])
        elif c not in df:
            df[c] = np.nan
    if "n_groups" in prim:      # the model's own count wins; only fill gaps (STAT-46: RI rows had none mapped)
        own = pd.to_numeric(df["n_groups"], errors="coerce") if "n_groups" in df else pd.Series(np.nan, index=df.index)
        df["n_groups"] = own.fillna(df["idx"].map(prim["n_groups"]))
    for c in ("in_family", "sig_raw", "sig_adj"):
        df[c] = df[c].fillna(False).astype(bool)
    df["tier"] = df["tier"].fillna("")
    df["validation"] = df["validation"].fillna("")
    df["interpretation"] = [r if isinstance(r, str) and r else (f"Not tested: {reason}" if reason else "")
                            for r, reason in zip(df["interpretation"], df.get("reason", pd.Series("", index=df.index)))]
    df["n_tests"] = df["idx"].map(tf.groupby("idx").size()).fillna(0).astype(int) if len(tf) else 0
    return df


def legacy_model_type(row: dict[str, Any]) -> str:
    kind = row.get("kind") or "focal"
    if kind == "all_two_way":
        return "All two-way interactions"
    return "Focal interaction + covariates" if list(row.get("covariates") or []) else "Single focal interaction"


def add_display_columns(df: pd.DataFrame, run: DiscoveryRun) -> pd.DataFrame:
    """Columns for presentation: model type, 'Predictors k / max', term counts for legacy rows,
    diagnostics status text. Vectorised; legacy (v1.1) rows are filled individually. Does not change
    any statistic."""
    if df.empty:
        return df
    df = df.copy()
    s = run.settings
    req = len(s.required)
    for col in ("model_type", "n_predictors", "n_terms"):
        if col not in df:
            df[col] = np.nan
    df["model_type"] = df["model_type"].astype(object)
    legacy = df["model_type"].isna() | pd.to_numeric(df["n_predictors"], errors="coerce").isna() \
        | pd.to_numeric(df["n_terms"], errors="coerce").isna()
    if legacy.any():
        from .space import Candidate, model_terms

        sub = df.loc[legacy]
        recs = sub[[c for c in ("idx", "kind", "focal", "covariates") if c in sub]].to_dict("records")
        mt, npred, nterm = [], [], []
        for r in recs:
            focal, cov = list(r.get("focal") or []), list(r.get("covariates") or [])
            mt.append(legacy_model_type(r))
            npred.append(len(set(focal + cov)) + req)
            c = Candidate(int(r["idx"]), tuple(focal), tuple(cov), tuple(s.required), r.get("kind") or "focal")
            nterm.append(len(model_terms(c)[0]))
        df.loc[legacy, "model_type"] = np.where(sub["model_type"].isna(), mt, sub["model_type"])
        df.loc[legacy, "n_predictors"] = np.where(pd.to_numeric(sub["n_predictors"], errors="coerce").isna(), npred,
                                                  sub["n_predictors"])
        df.loc[legacy, "n_terms"] = np.where(pd.to_numeric(sub["n_terms"], errors="coerce").isna(), nterm,
                                             sub["n_terms"])
    # P1-04: separate parameter concepts and the estimation basis of every likelihood row
    npar = pd.to_numeric(df.get("n_params", np.nan), errors="coerce")
    nre = pd.to_numeric(df.get("n_re_params", np.nan), errors="coerce") if "n_re_params" in df else 0
    fam = run.family
    if fam == "gee":
        df["n_lik"] = np.nan
    elif fam == "gaussian":
        df["n_lik"] = npar + 1                       # + residual variance
    else:
        df["n_lik"] = npar + (nre.fillna(0) if isinstance(nre, pd.Series) else 0)
    df["estimation"] = estimation_label(run)
    if s.participant and "structure" in df:
        from .space import STRUCTURES

        df["structure_label"] = df["structure"].map(lambda x: STRUCTURES.get(x, (x or "",))[0]).fillna("")
    k = pd.to_numeric(df["n_predictors"], errors="coerce")
    df["predictors_txt"] = np.where(k.notna(), k.fillna(0).astype(int).astype(str) + f" / {s.max_predictors}", "")
    df["status_text"] = ""          # filled per row by the table from the current highlight mode
    flags = df["flags"].fillna("").astype(str)
    df["diag_text"] = np.where(df["status"].isin(["invalid", "failed"]), "not fitted",
                               np.where(flags != "", "⚠ " + flags, "clean"))
    return df


def estimation_label(run: DiscoveryRun) -> str:
    s = run.settings
    return {"lmm": "REML" if s.participant and s.workflow == "structure" else "ML",
            "glmm": "ML (Laplace)", "clmm": "ML (Laplace)", "gee": "GEE (robust SE)", "gaussian": "OLS",
            }.get(run.family, "ML")


def model_comparison(df: pd.DataFrame, run: DiscoveryRun) -> tuple[pd.DataFrame, str]:
    """Fitted models ordered by AIC with ΔAIC / ΔBIC — only when every model was fitted to the same
    rows (shared sample) and family; otherwise information criteria are not comparable (X §9)."""
    crit = "qic" if run.family == "gee" and "qic" in df else "aic"
    fitted = df[df["status"].isin(["ok", "warning"]) & df[crit].notna()].copy() if crit in df else df.iloc[0:0]
    if fitted.empty:      # STAT-34: an explicit status, never an empty table with Inf deltas
        n_att = int(df["status"].isin(["ok", "warning", "invalid", "failed"]).sum()) if "status" in df else 0
        return fitted, ((f"No model could be compared: none of the {n_att:,} attempted models produced "
                         + ("a QIC (GEE has no likelihood, so AIC/BIC are not defined; see each row's reason)."
                            if run.family == "gee" else "an information criterion (see each row's status and reason)."))
                        if n_att else "")
    if len(fitted) == 1:
        fitted["d_aic"] = 0.0 if crit == "aic" else np.nan
        fitted["d_bic"] = np.nan
        if crit == "qic":
            fitted["aic"] = fitted["qic"]
        return fitted, ("Only one model was fitted, so there is no comparison and no 'best' model: the ranking is "
                        "incomplete.")
    if crit == "qic":
        fitted = fitted.sort_values(["qic", "idx"], kind="stable")
        fitted["d_aic"] = fitted["qic"] - fitted["qic"].min()
        fitted["aic"] = fitted["qic"]
        fitted["d_bic"] = np.nan
        return fitted, ("GEE (population-average): the AIC column holds QIC (quasi-likelihood under the "
                        "independence model criterion) and ΔAIC holds ΔQIC. QIC is only comparable between models "
                        "with the same rows and outcome family; it is not an AIC.")
    same_rows = fitted["n"].nunique() <= 1
    if same_rows and "identity" in fitted:
        rows = {(i or {}).get("rows") for i in fitted["identity"] if isinstance(i, dict)}
        same_rows = len(rows - {None}) <= 1
    shared = run.settings.sample == "shared" and same_rows
    fitted = fitted.sort_values(["aic", "idx"], kind="stable")
    if shared:
        fitted["d_aic"] = fitted["aic"] - fitted["aic"].min()
        reml = run.settings.participant and run.settings.workflow == "structure" and run.family == "lmm"
        bic_n = pd.to_numeric(fitted.get("bic_n"), errors="coerce") if "bic_n" in fitted else pd.Series(dtype=float)
        mixed_basis = bic_n.notna().any() and bic_n.nunique() > 1
        if mixed_basis:
            # STAT-24: native REML BIC uses n - p (nlme) or n (lme4): never rank them on their native values. A
            # harmonized BIC on one stated basis (n - p, nlme's REML convention) is shown separately.
            fitted["d_bic"] = np.nan
            npar = pd.to_numeric(fitted.get("n_params"), errors="coerce")
            k = pd.to_numeric(fitted.get("n_lik_params"), errors="coerce")
            fitted["bic_harmonized"] = -2 * fitted["loglik"] + np.log(fitted["n"] - npar) * k
            fitted["d_bic_harmonized"] = fitted["bic_harmonized"] - fitted["bic_harmonized"].min()
        else:
            fitted["d_bic"] = fitted["bic"] - fitted["bic"].min()
        note = (f"All {len(fitted):,} fitted models use the same {int(fitted['n'].iloc[0]):,} rows and the same "
                "outcome family, so AIC/BIC are comparable"
                + (" (REML fits with an identical fixed part: valid for comparing covariance structures)." if reml
                   else ".") + " Smaller is better; ΔAIC < 2 is usually not a meaningful "
                "difference. Information criteria are not p-values, and a model chosen from many is optimistic."
                + (" Native BIC uses different sample sizes here (nlme REML: n − p; lme4 REML: n), so ΔBIC is not "
                   "computed on native values; 'BIC (harmonized, n − p)' recomputes every model's BIC on the n − p "
                   "basis for a like-for-like ranking." if mixed_basis else ""))
    else:
        fitted["d_aic"] = np.nan
        fitted["d_bic"] = np.nan
        note = ("Models were fitted to different rows (complete cases per model): AIC/BIC are not comparable "
                "across them, so no ΔAIC is shown.")
    return fitted, note


HIGHLIGHT_MODES = {
    "adjusted": "Adjusted p < α",
    "raw": "Raw p < α",
    "strict": "Strict: adjusted p < α AND the same-estimand CI excludes its null value",
    "off": "Off",
}


def ci_excludes_null(df: pd.DataFrame) -> pd.Series:
    """True only for a finite CI of the tested coefficient itself (same estimand) that excludes the
    null value of its scale; boundary values do not exclude. Joint tests, missing, non-finite or
    unlabelled intervals never qualify."""
    lo = pd.to_numeric(df["ci_low"], errors="coerce")
    hi = pd.to_numeric(df["ci_high"], errors="coerce")
    null = pd.to_numeric(df["ci_null"], errors="coerce")
    finite = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(null) & (lo <= hi)
    same = df["ci_applicable"] == "same_estimand"
    return (same & finite & ((lo > null) | (hi < null))).astype(bool)


def highlight_mask(df: pd.DataFrame, mode: str = "adjusted", alpha: float = 0.05) -> pd.Series:
    """Presentation only: which hypothesis rows to highlight. Never refits or re-tests."""
    if df.empty or mode == "off":
        return pd.Series(False, index=df.index)
    ok = df["in_family"].astype(bool)
    p = pd.to_numeric(df["p"], errors="coerce")
    pa = pd.to_numeric(df["p_adj"], errors="coerce")
    if mode == "raw":
        return (ok & (p < alpha)).fillna(False).astype(bool)
    sig = (ok & (pa < alpha)).fillna(False).astype(bool)
    if mode == "strict":
        return sig & ci_excludes_null(df)
    return sig


def warning_mask(df: pd.DataFrame) -> pd.Series:
    """Serious screening flags (non-convergence, singularity, separation, sparse cells, …) — shown
    independently of significance."""
    if df.empty:
        return pd.Series(False, index=df.index)
    return (df["status"] == "warning") | (df["flags"].fillna("").astype(str) != "")


def _p(p: float) -> str:
    return "< .001" if p < 0.001 else f"{p:.3f}".lstrip("0")


def interpret_row(r: dict[str, Any], s: SearchSettings, family: str = "gaussian") -> str:
    """Short teaching interpretation of one tested hypothesis (§10.8, M §11): what was tested,
    the result with its test and family, direction when one coefficient is tested, and caveats."""
    if r["status"] in ("invalid", "failed"):
        return f"Not tested: {r['reason']}"
    if r.get("speculative"):
        return "Evaluated after the stop point; not part of this search's results."
    outcome = r["formula"].split(" ~ ")[0]
    ttype = r.get("test_type") or ("joint_interaction" if r.get("kind") == "all_two_way" else "focal_interaction")
    label = r.get("label") or r["interaction"]
    parts = label.split(" × ")
    if ttype == "structure":
        better, rest = label.split(" vs ", 1)
        what = f"{better} fits better than {rest.split(':')[0]} ({rest.split(': ', 1)[-1]})"
        p = r["p"]
        verdict = "Evidence that" if r.get("sig_adj") else "No clear evidence that"
        return (f"{verdict} {what} (likelihood-ratio p {_p(p)}). Structure comparisons describe dependence "
                "between a participant's observations; they are not discoveries about predictors.")
    if ttype == "joint_interaction":
        what = f"the interaction terms ({label.split(': ', 1)[-1]}) jointly improve the model of {outcome}"
    elif ttype == "main_term":
        what = f"{label} is associated with {outcome} when the model's other predictors are held constant"
    elif ttype == "overall":
        what = f"the model's predictors together explain {outcome} better than the {label.split(' vs ', 1)[-1]}"
    elif len(parts) == 2:
        what = f"the {parts[0]}–{outcome} relationship differs across {parts[1]}"
    elif len(parts) == 3:
        what = f"the {parts[0]} × {parts[1]} interaction on {outcome} differs across {parts[2]}"
    else:
        what = f"{label} improves the model of {outcome}"
    p, pa = r["p"], r.get("p_adj")
    if r.get("effect") is None or r.get("effect") != r.get("effect"):
        eff = ""
    elif family == "gaussian":
        eff = f"; it uniquely explains {100 * r['effect']:.1f}% of the outcome's variation"
    elif family == "lmm":
        eff = f"; marginal R² rises by {100 * r['effect']:.1f} points"
    elif family in ("glmm", "clmm", "gee"):
        eff = ""
    else:
        eff = f"; McFadden pseudo-R² rises by {r['effect']:.3f}"
    fam = r.get("family_size")
    within = f" among {int(fam):,} {TEST_TYPES.get(ttype, ttype).lower()} tests" if fam else ""
    if r.get("sig_adj"):
        text = (f"Evidence that {what} (p {_p(p)}, adjusted p {_p(pa)}{within}{eff}). Still exploratory: "
                "check stability and new data.")
    elif r.get("sig_raw"):
        text = (f"Possible sign that {what} (p {_p(p)}), but not after adjusting for the number of tests "
                f"(adjusted p {_p(pa)}{within}): plausibly a chance finding.")
    else:
        text = f"No clear evidence that {what} (p {_p(p)}); this is not proof that there is no difference."
    est = r.get("estimate")
    if r.get("ci_applicable") == "same_estimand" and est is not None and est == est and est != 0:
        scale = "log-odds" if family in ("binomial", "ordinal", "glmm", "clmm") or \
            (family == "gee" and s.family == "binomial") else "coefficient"
        if family == "gee":
            scale += " (population-average)"
        elif family in ("glmm", "clmm"):
            scale += " (subject-specific)"
        text += (f" The {scale} is {'positive' if est > 0 else 'negative'} ({est:.3g}, "
                 f"{100 * float(r.get('ci_level') or s.conf):.0f}% CI {r['ci_low']:.3g} to {r['ci_high']:.3g}).")
    method = r.get("test_method") or test_method(ttype, family)
    ref = r.get("reference")
    text += f" Test: {method}" + (f", against {ref}." if isinstance(ref, str) and ref else ".")
    if r.get("flags"):
        text += f" Screening flags: {r['flags']}."
    text += (" This is an association in the analysed data, not evidence of cause and effect"
             + (", and it was selected from many models." if (r.get("family_size") or 0) > 1 else "."))
    return text


def discovery_ranking(df: pd.DataFrame, by: str = "p") -> pd.DataFrame:
    if df.empty:
        return df
    return df[df["in_family"]].sort_values([by, "idx"], kind="stable", na_position="last")


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
        "timed_out": int(df["timed_out"].fillna(False).astype(bool).sum()) if n_attempted and "timed_out" in df else 0,
        "reused": sum(1 for r in run.rows if r.get("cached")),     # taken from an earlier search's identical fit
        "not_visited": max(run.total - n_attempted, 0),
        "sig_raw": int(df["sig_raw"].sum()) if n_attempted else 0,
        "sig_adj": int(df["sig_adj"].sum()) if n_attempted else 0,
        "expected_false_raw": fam * run.settings.alpha,
        "provisional": run.status != "complete",
        "families": _family_counts(run),
    }


INTERACTION_TYPES = ("focal_interaction", "interaction_term", "joint_interaction")
FAMILY_ORDER = ("interaction_term", "focal_interaction", "joint_interaction", "main_term", "overall", "structure")
FAMILY_NOUNS = {   # (singular, plural)
    "interaction_term": ("individual interaction-term test", "individual interaction-term tests"),
    "focal_interaction": ("focal interaction test", "focal interaction tests"),
    "joint_interaction": ("joint test of interaction terms", "joint tests of interaction terms"),
    "main_term": ("main-effect term test", "main-effect term tests"),
    "overall": ("whole-model test", "whole-model tests"),
    "structure": ("covariance-structure comparison", "covariance-structure comparisons"),
}


def _n(k: int, noun: tuple[str, str]) -> str:
    return f"{k:,} {noun[0] if k == 1 else noun[1]}"


def headline(run: DiscoveryRun, families: list[dict[str, Any]] | None = None) -> list[str]:
    """Type-specific summary sentences (P0-03). Interaction findings are counted only from interaction
    hypotheses; whole-model, main-effect and covariance-structure results are reported as what they are."""
    s = run.settings
    fams = {f["type"]: f for f in (families if families is not None else _family_counts(run))}
    corr = "Benjamini-Hochberg" if s.correction == "BH" else "Holm"
    adj = lambda f: f["sig_bh"] if s.correction == "BH" else f["sig_holm"]  # noqa: E731
    out: list[str] = []
    if "structure" in fams or (s.participant and s.workflow == "structure"):
        f = fams.get("structure")
        if f is None:
            out.append("No nested covariance-structure comparison could be computed in this search; compare the "
                       "structures by AIC/BIC in the Model comparison tab.")
        else:
            out.append(f"{adj(f):,} of {_n(f['tests'], FAMILY_NOUNS['structure'])} significant after {corr} "
                       f"adjustment (α = {s.alpha:g}){_clean_split(f)}. These compare random-effect / residual-"
                       "correlation structures with the fixed part locked; they are not findings about predictors or "
                       "interactions.")
    inter = [fams[t] for t in INTERACTION_TYPES if t in fams]
    if not (s.participant and s.workflow == "structure"):
        if not inter:
            out.append("This search tested no interaction hypotheses, so it reports no interaction findings "
                       "(0 significant interactions).")
        else:
            n_sig = sum(adj(f) for f in inter)
            what = " and ".join(_n(f["tests"], FAMILY_NOUNS[f["type"]]) for f in inter)
            if n_sig == 0:
                out.append(f"0 significant interactions: none of the {what} reached {corr}-adjusted p < "
                           f"{s.alpha:g}.")
            else:
                out.append("Interaction findings: " + ", ".join(
                    f"{adj(f):,} of {_n(f['tests'], FAMILY_NOUNS[f['type']])}{_clean_split(f)}" for f in inter)
                    + f" significant after {corr} adjustment (α = {s.alpha:g}). Still exploratory.")
    other = [fams[t] for t in ("main_term", "overall") if t in fams]
    for f in other:
        noun = FAMILY_NOUNS[f["type"]]
        out.append(f"{adj(f):,} of {_n(f['tests'], noun)} significant after {corr} adjustment{_clean_split(f)}"
                   + (" — a significant whole-model test says the predictors together explain the outcome; it is "
                      "not evidence of an interaction." if f["type"] == "overall" else
                      " — main effects of variables that are not in an interaction; not evidence of an interaction."))
    return out


def _clean_split(f: dict[str, Any]) -> str:
    """'(k with clean diagnostics, m provisional)' — a significant test of a flagged fit is not clean evidence."""
    if "sig_adj_clean" not in f or not f.get("sig_adj"):
        return ""
    return (f": {f['sig_adj_clean']:,} with clean diagnostics, {f['sig_adj_flagged']:,} provisional (a compared fit is "
            "flagged)")


def _family_counts(run: DiscoveryRun) -> list[dict[str, Any]]:
    """Per test type: tests in the family, raw / adjusted significant, joint tests without a CI."""
    tf = tests_frame(run)
    if tf.empty:
        return []
    out = []
    groups = dict(list(tf[tf["in_family"]].groupby("test_type", sort=False)))
    for ttype in sorted(groups, key=lambda t: FAMILY_ORDER.index(t) if t in FAMILY_ORDER else len(FAMILY_ORDER)):
        g = groups[ttype]
        out.append({"type": ttype, "label": TEST_TYPES.get(ttype, ttype), "tests": len(g),
                    "sig_raw": int(g["sig_raw"].sum()), "sig_adj": int(g["sig_adj"].sum()),
                    "sig_bh": int((g["p_bh"] < run.settings.alpha).sum()),
                    "sig_holm": int((g["p_holm"] < run.settings.alpha).sum()),
                    "sig_adj_clean": int((g["sig_adj"] & (g["tier"] == "A")).sum()),
                    "sig_adj_flagged": int((g["sig_adj"] & (g["tier"] == "C")).sum()),
                    "withheld": int((g.get("test_status", pd.Series("", index=g.index)) == "withheld").sum()),
                    "no_ci": int((g["ci_applicable"] != "same_estimand").sum()),
                    "expected_false_raw": len(g) * run.settings.alpha})
    return out
