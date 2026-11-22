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
    return Candidate(int(row["idx"]), tuple(row["focal"]), tuple(row["covariates"]), tuple(run.settings.required),
                     str(row.get("kind") or "focal"))


def full_analysis(session: RSession, ds: Dataset, run: DiscoveryRun,
                  c: Candidate) -> tuple[AnalysisResult, dict[str, bytes]]:
    s = run.settings
    a = lambda n: alias(ds.var(n).alias)  # noqa: E731
    if c.kind == "all_two_way":
        rhs = "(" + " + ".join(a(v) for v in c.focal) + ")^2" + "".join(f" + {a(v)}" for v in c.required)
        internal: dict[str, Any] = {"rhs": rhs}
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
    title = f"Discovery candidate #{c.index}: " + " × ".join(ds.var(v).display_name for v in c.focal)
    result, artifacts = run_analysis(session, ds, spec, vars_, opts, internal, title,
                                     treatments=getattr(s, "treatments", None))
    result.values["discovery"] = {"run": run.id, "candidate": c.index}
    return result, artifacts
