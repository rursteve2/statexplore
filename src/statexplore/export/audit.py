"""Audit trail of a discovery search and its validation runs (v1.3.4, V134-015).

`audit_record(run, ds)` builds a versioned, JSON-safe record: run identities and versions, environment, data
fingerprints and policies, the search grammar and space, every candidate with its fit identity, diagnostics and cache
provenance, every hypothesis with raw / log / adjusted p and its family, and every validation run with its manifest
identity. `write_audit` writes it atomically as JSON (canonical, NaN/Infinity -> null) plus a Markdown report.

Records are immutable per execution: they describe stored results and never recompute statistics. Absolute paths are
reduced to the file name (no user paths in a shared audit); row-level data are never included.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import os
import platform
import tempfile
from pathlib import Path
from typing import Any

SCHEMA = "statexplore-audit"
SCHEMA_VERSION = 2   # 2 = v1.3.5 (AUD134-07): historical data provenance separated from the export-time context
REQUIRED = ("schema", "schema_version", "generated_utc", "application", "search", "data", "environment", "grammar",
            "candidates", "hypotheses", "validation", "limitations")


class AuditError(ValueError):
    pass


def _clean(x: Any) -> Any:
    """JSON-safe: NaN / +-Inf -> None (explicit missing), numpy scalars -> Python, tuples -> lists."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_clean(v) for v in x]
    if np is not None and isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if x is None or isinstance(x, (bool, int, str)):
        return x
    return str(x)


def _app_version() -> str:
    try:
        from .. import __version__
        return str(__version__)
    except Exception:  # noqa: BLE001
        return ""


def historical_data(run: Any) -> dict[str, Any]:
    """AUD134-07: what the search analysed, from its OWN immutable record (data snapshot, signature, fit identity).
    Missing historical metadata is 'unknown', never filled from the current project."""
    snap = getattr(run, "data_snapshot", None) or {}
    unknown = "unknown (not recorded by the version that ran this search)"
    return {"source_file": snap.get("source_file") or unknown, "source_format": snap.get("source_format") or unknown,
            "source_sha256": snap.get("source_sha256") or unknown, "source_rows": snap.get("source_rows"),
            "source_size_bytes": snap.get("source_size_bytes"),
            "analysis_signature": run.signature, "fit_identity": run.fit_id or unknown, "n_sample": run.n_sample,
            "filters": snap.get("filters") if snap else unknown, "derived": snap.get("derived") if snap else unknown,
            "holdout": snap.get("holdout") if snap else unknown, "treatments": list(run.settings.treatments or []),
            "snapshot_schema": snap.get("schema"), "snapshot": snap or None,
            "fingerprints": {
                "source_sha256": "SHA-256 of the source FILE's bytes as imported (not of the analysed rows)",
                "analysis_signature": "SHA-256 of (source SHA-256, variable metadata, derived variables, filters, holdout) "
                                      "of the search's data view: the canonical analysed-content identity",
                "fit_identity": "SHA-256 of the analysed data view, outcome family / coding, grouping, sample policy and "
                                "fitting settings",
                "per_fit": "each candidate's identity: SHA-256 of its realized rows, response, design and grouping "
                           "(identity schema 2)"}}


def current_context(run: Any, ds: Any) -> dict[str, Any]:
    """The project at EXPORT time, explicitly labelled: it is not the analysed data unless `matches_analysis`."""
    from pathlib import PureWindowsPath

    src = getattr(ds, "source", None)
    out = {"label": "export-time project context (NOT the analysed data unless matches_analysis is true)",
           "source_file": PureWindowsPath(getattr(src, "path", "") or "").name, "source_sha256": getattr(src, "sha256", ""),
           "matches_analysis": False, "state": "unknown", "differences": []}
    try:
        from ..discovery.history import context

        ctx = context(ds, run)
        out.update(matches_analysis=ctx.current, state=ctx.state, differences=list(ctx.differences))
    except Exception as e:  # noqa: BLE001 - the current data may not even have the search's variables
        out.update(state="unrecoverable", differences=[f"the search's data view cannot be built on the current data ({e})"])
    return out


def audit_record(run: Any, ds: Any, include_current: bool = True) -> dict[str, Any]:
    from ..discovery import engine
    from ..discovery.space import GRAMMAR_VERSION, count_candidates

    s = run.settings
    tf = engine.tests_frame(run, interpret=False)
    cands = []
    for r in run.rows:
        cands.append({
            "index": r.get("idx"), "key": r.get("key"), "candidate_id": r.get("candidate_id"), "model_id": r.get("model_id"),
            "formula": r.get("formula"), "model_type": r.get("model_type"), "structure": r.get("structure"),
            "status": r.get("status"), "reason": r.get("reason"), "flags": r.get("flags"),
            "speculative": bool(r.get("speculative")), "timed_out": bool(r.get("timed_out")),
            "n": r.get("n"), "n_groups": r.get("n_groups"), "n_params": r.get("n_params"),
            "n_lik_params": r.get("n_lik_params"), "loglik": r.get("loglik"), "aic": r.get("aic"), "bic": r.get("bic"),
            "bic_n": r.get("bic_n"), "ic_basis": r.get("ic_basis"), "qic": r.get("qic"), "qicu": r.get("qicu"),
            "cic": r.get("cic"), "gee_fitted": r.get("gee"), "identity": r.get("identity"), "rows_sig": r.get("rows_sig"),
            "diagnostics": r.get("diagnostics"), "fitted_at": r.get("fitted_at"),
            "cache": {"reused": bool(r.get("cached")), "reason": r.get("cache_reason") or "",
                      "source_run": (r.get("cache_reason") or "").split("search ")[-1].split(":")[0] if r.get("cached")
                      else ""}})
    hyps = []
    if len(tf):
        keep = ("test_id", "idx", "test_type", "label", "reference", "test_method", "statistic", "LR_raw", "df1",
                "df_den", "p", "log_p", "log10_p", "log_p_source", "p_bh", "p_holm", "log_p_bh", "log_p_holm", "p_adj",
                "log_p_adj", "family_size", "in_family", "sig_raw", "sig_adj", "tier", "test_status", "test_reason",
                "reference_flags", "reference_source", "reference_fitted_for", "estimate", "se", "ci_low", "ci_high",
                "ci_level", "ci_applicable", "ci_scale", "effect")
        for rec in tf[[c for c in keep if c in tf.columns]].to_dict("records"):
            lp = rec.get("log_p")
            # an underflowed p is 0.0 in double precision: say so, and keep the exact log tail (AUD134-07)
            rec["p_underflow"] = bool(rec.get("p") == 0 and lp is not None and lp == lp)
            rec["log_p_unknown_magnitude"] = bool(lp is not None and lp == -math.inf)
            hyps.append(rec)
    vals = []
    for v in run.validation_runs or []:
        vals.append({k: v.get(k) for k in ("validation_id", "schema", "grammar", "status", "problems", "config",
                                           "search_id", "search_identity", "search_grammar", "search_signature",
                                           "data_sha256", "pool", "pool_label", "pool_scope", "adaptive", "stop_rule",
                                           "target_label", "pipeline_label", "rule_label", "rng", "realized",
                                           "performance", "stability", "estimands", "candidate_keys", "provenance",
                                           "fit_budget")})
    rec = {
        "schema": SCHEMA, "schema_version": SCHEMA_VERSION,
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "application": {"name": "StatExplore", "version": _app_version(), "fit_rules": engine.FIT_RULES,
                        "grammar_current": GRAMMAR_VERSION},
        "search": {"id": run.id, "fit_identity": run.fit_id, "data_signature": run.signature,
                   "created_at": run.created_at, "status": run.status,
                   "status_label": engine.STATUS_LABELS.get(run.status, run.status), "stop_reason": run.stop_reason,
                   "stop_index": run.stop_index, "n_done": run.n_done, "total": run.total, "elapsed_s": run.elapsed_s,
                   "executions": run.executions, "family": run.family, "coding": run.coding, "warnings": run.warnings,
                   "resume": run.resume_reason(ds) or "resumable" if run.status in ("partial", "cancelled") else
                   run.resume_reason(ds), "provisional_adjustment": run.status != "complete",
                   "fit_rules_of_rows": (run.engine or {}).get("fit_rules"),
                   "historical_old_rules": ((run.engine or {}).get("fit_rules") or 0) < engine.FIT_RULES,
                   "parent_id": getattr(run, "parent_id", "") or None, "lineage": getattr(run, "lineage", None) or None},
        "settings": s.to_dict(),
        "environment": {"engine": run.engine, "engine_label": "R / package versions that fitted this search's rows",
                        "export_generated_with": {"python": platform.python_version(),
                                                  "os": platform.platform(terse=True), "statexplore": _app_version()}},
        # AUD134-07: the historical analysis is authoritative; the current project is a separate, labelled block
        "data": {"historical": historical_data(run), "row_level_data": "not included",
                 **({"current_context": current_context(run, ds)} if include_current and ds is not None else {})},
        "grammar": {"version": run.grammar, "strategy": s.strategy, "stored_total": run.total,
                    "current_grammar": GRAMMAR_VERSION,
                    "space_size_under_current_grammar": count_candidates(s),
                    "current_grammar_note": "" if count_candidates(s) == run.total else (
                        "recomputed under the CURRENT grammar; the search itself declared stored_total candidates"), "candidate_order": list(s.candidates), "required": list(s.required),
                    "max_predictors": s.max_predictors, "min_candidates": s.min_predictors, "max_order": s.max_order,
                    "include_main": s.include_main, "time_eligible": bool(getattr(s, "time_eligible", False)),
                    "mode": s.mode, "budgets": run.budget_text(), "alpha": s.alpha, "correction": s.correction,
                    "attempted": len(run.rows), "speculative": sum(1 for r in run.rows if r.get("speculative")),
                    "reused": sum(1 for r in run.rows if r.get("cached"))},
        "candidates": cands, "hypotheses": hyps, "validation": vals,
        "limitations": ["Exploratory search: adjusted p-values are within each test-type family of THIS search; "
                        "provisional when the search is incomplete.",
                        "Rows fitted under older fit rules are historical evidence and are not reused.",
                        "Selection frequencies are not probabilities that a hypothesis is true."],
    }
    return _clean(rec)


def canonical_json(rec: dict[str, Any]) -> str:
    return json.dumps(rec, indent=1, sort_keys=True, ensure_ascii=False, allow_nan=False)


def validate_record(rec: Any) -> None:
    """Refuse a corrupted or incompatible audit record (missing fields, wrong schema, NaN)."""
    if not isinstance(rec, dict) or rec.get("schema") != SCHEMA:
        raise AuditError("not a StatExplore audit record")
    if int(rec.get("schema_version") or 0) > SCHEMA_VERSION:
        raise AuditError(f"audit schema {rec.get('schema_version')} is newer than this version ({SCHEMA_VERSION})")
    missing = [k for k in REQUIRED if k not in rec]
    if missing:
        raise AuditError("audit record is missing: " + ", ".join(missing))
    if not rec["search"].get("id"):
        raise AuditError("audit record has no search id")
    json.dumps(rec, allow_nan=False)


def report_md(rec: dict[str, Any]) -> str:
    s, g = rec["search"], rec["grammar"]
    L = [f"# StatExplore audit report — search {s['id']}", "",
         f"Generated {rec['generated_utc']} by StatExplore {rec['application']['version']} (fit rules "
         f"{rec['application']['fit_rules']}, audit schema {rec['schema_version']}).", "",
         "## Search", f"- Status: {s['status_label']}" + (f" ({s['stop_reason']})" if s.get("stop_reason") else ""),
         f"- Fit identity: `{s['fit_identity']}`; data signature `{s['data_signature']}`",
         f"- Resume: {s.get('resume') or 'n/a'}",
         f"- Candidates: {g['attempted']:,} attempted of {g.get('stored_total', g.get('exact_space_size')):,} declared (grammar "
         f"{g['version']}; the current grammar {g.get('current_grammar', '?')} defines "
         f"{g.get('space_size_under_current_grammar', g.get('exact_space_size')):,} for these settings; strategy "
         f"{g['strategy']}, mode {g['mode']}); {g['speculative']:,} after the stop point; {g['reused']:,} reused fits",
         f"- Candidate order: {', '.join(g['candidate_order'])}; required: {', '.join(g['required']) or 'none'}",
         f"- Budgets: {g['budgets']}",
         f"- Multiplicity: {g['correction']} within each test-type family, alpha = {g['alpha']}"
         + ("; PROVISIONAL (incomplete search)" if s.get("provisional_adjustment") else ""),
         "", "## Data analysed by this search (historical record)"] + _data_lines(rec) + ["", "## Environment"]
    eng = rec["environment"].get("engine") or {}
    L.append(f"- R {eng.get('r_version', '?')}; packages: "
             + ", ".join(f"{k} {v}" for k, v in (eng.get("packages") or {}).items()))
    L += ["", "## Hypotheses (first 50 by log p)", "", "| test | label | statistic | df | log10 p | adj. p | family | status |",
          "|---|---|---|---|---|---|---|---|"]
    hy = sorted([h for h in rec["hypotheses"] if h.get("log_p") is not None], key=lambda h: h["log_p"])[:50]
    for h in hy:
        L.append(f"| {h.get('test_id')} | {h.get('label')} | {h.get('statistic')} | {h.get('df1')} | "
                 f"{(h.get('log10_p') or 0):.3f} | {h.get('p_adj')} | {h.get('family_size')} | "
                 f"{h.get('test_status') or 'tested'} |")
    if rec["validation"]:
        L += ["", "## Validation runs"]
        for v in rec["validation"]:
            L.append(f"- {v.get('validation_id')}: {v.get('status')} — {v.get('pipeline_label')}; pool: "
                     f"{v.get('pool_label')}; search identity `{v.get('search_identity')}`")
    L += ["", "## Limitations"] + [f"- {x}" for x in rec["limitations"]]
    return "\n".join(L) + "\n"


def _data_lines(rec: dict[str, Any]) -> list[str]:
    d = rec["data"]
    if "historical" not in d:          # audit schema 1 (v1.3.4): one block that mixed export-time and historical data
        return [f"- File: {d.get('file')} (SHA-256 `{d.get('sha256')}`) - schema-1 record: this file description may be "
                "the export-time project, not the analysed data", f"- Shared sample: {d.get('n_sample')}"]
    h = d["historical"]
    out = [f"- Source file: {h['source_file']} (source-file SHA-256 `{h['source_sha256']}`)",
           f"- Analysed-content signature `{h['analysis_signature']}`; fit identity `{h['fit_identity']}`",
           f"- Shared sample: {h['n_sample']}"]
    c = d.get("current_context")
    if c:
        out += ["", "## Export-time project context (not the analysed data unless stated)",
                f"- {c['source_file']} (SHA-256 `{c['source_sha256']}`): "
                + ("matches the analysed data view" if c["matches_analysis"] else
                   "DIFFERS from the analysed data: " + "; ".join(c.get("differences") or []))]
    return out


def _atomic_write(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def write_audit(run: Any, ds: Any, folder: Path) -> list[Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rec = audit_record(run, ds)
    validate_record(rec)
    j, m = folder / f"search_{run.id}_audit.json", folder / f"search_{run.id}_audit_report.md"
    _atomic_write(j, canonical_json(rec))
    _atomic_write(m, report_md(rec))
    return [j, m]


def read_audit(path: Path) -> dict[str, Any]:
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_record(rec)
    return rec
