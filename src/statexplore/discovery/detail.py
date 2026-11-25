"""Open a discovery candidate as a full Phase 2 analysis (coefficients, CIs, diagnostics,
conditional slopes, plots) on exactly the rows the search used."""
from __future__ import annotations

from typing import Any

from ..analysis.builders import BY_KEY
from ..analysis.framework import run_analysis
from ..analysis.session import RSession
from ..core.dataset import Dataset
from ..core.schema import AnalysisResult
from ..rbridge.codegen import alias
from .engine import DiscoveryRun
from .space import Candidate


def candidate_from_row(run: DiscoveryRun, row: dict[str, Any]) -> Candidate:
    inter = row.get("inter") or []
    if isinstance(inter, str):
        import json

        inter = json.loads(inter)
    kind = str(row.get("kind") or "focal")
    # workflow-B candidates differ only in their structure: dropping it silently substituted another model (P0-01 B/C)
    structure = str(row.get("structure") or "") if kind == "structure" else ""
    # exploratory cluster on/off candidates: the row's own cluster mode (dropping it would refit another model)
    clustered = (bool(row.get("clustered")) if kind == "structure" and getattr(run.settings, "cluster_compare", False)
                 and row.get("clustered") is not None else None)
    return Candidate(int(row["idx"]), tuple(row["focal"]), tuple(row["covariates"]), tuple(run.settings.required),
                     kind, tuple(tuple(t) for t in inter), structure, clustered)


def full_analysis(session: RSession, ds: Dataset, run: DiscoveryRun,
                  c: Candidate, new_analysis: list[str] | None = None) -> tuple[AnalysisResult, dict[str, bytes]]:
    """`ds` is the search's own data view (current or rebuilt, history.context); `new_analysis` lists why the
    project differs when the user explicitly chose a NEW analysis under the current settings instead."""
    s = run.settings
    a = lambda n: alias(ds.var(n).alias)  # noqa: E731
    if c.kind in ("main", "combo", "factorial"):
        from .space import full_terms

        rhs = " + ".join(":".join(a(v) for v in t) for t in full_terms(c))
        internal: dict[str, Any] = {"rhs": rhs}
        top = [t for t in c.inter if not any(set(t) < set(u) for u in c.inter)]
        if len(top) == 1:
            internal["focal"] = [a(v) for v in top[0]]
    elif c.kind == "all_two_way":
        rhs = "(" + " + ".join(a(v) for v in c.focal) + ")^2" + "".join(f" + {a(v)}" for v in c.required)
        internal = {"rhs": rhs}
    else:
        rhs = " * ".join(a(v) for v in c.focal) + "".join(f" + {a(v)}" for v in c.covariates + c.required)
        internal = {"rhs": rhs, "focal": [a(v) for v in c.focal]}
    if s.sample == "shared":
        internal["complete_on"] = [a(n) for n in [s.outcome] + s.candidates + s.required]
    key = {"gaussian": "lm", "binomial": "logistic", "ordinal": "ordinal_reg", "multinomial": "multinom_reg"}[run.family]
    spec = BY_KEY[key]
    vars_ = {"dependent": [s.outcome], "predictors": list(c.variables)}
    opts: dict[str, Any] = {"conf": s.conf}
    if run.family == "binomial" and getattr(s, "event", None) is not None:
        opts["event"] = s.event
    if run.family == "multinomial" and getattr(s, "reference", None) is not None:
        opts["reference"] = s.reference
    from .space import focal_label

    title = f"Discovery candidate #{c.index}: {focal_label(ds, c)}"
    result, artifacts = run_analysis(session, ds, spec, vars_, opts, internal, title,
                                     treatments=getattr(s, "treatments", None))
    result.values["discovery"] = {"run": run.id, "candidate": c.index}
    row = next((r for r in run.rows if int(r["idx"]) == c.index), None)
    if row is not None:
        from ..analysis.descriptor import attach_check, from_discovery

        d = from_discovery(run, row).to_dict()
        if new_analysis:
            d.update(new_analysis=True, new_analysis_reasons=list(new_analysis))
        attach_check(result, d)
    return result, artifacts
