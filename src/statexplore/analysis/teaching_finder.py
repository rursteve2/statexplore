"""Teaching Example Finder (v1.1 §17.5): point to completed analyses that illustrate a teaching
point. Uses only results and searches already in the project — nothing is fitted or invented here."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.project import Project

UNSTABLE = ("singular", "separation", "did not converge", "convergence", "condition number", "large standard errors")
ASSUMPTION = ("shapiro", "levene", "breusch", "proportional-odds", "proportional odds", "sphericity",
              "heteroscedastic", "non-normal", "nominal_test")


@dataclass
class Example:
    category: str
    title: str
    detail: str
    source: str          # "result <id>" or "search <id> #<idx>"


def find(project: Project, alpha: float = 0.05, limit: int = 200) -> list[Example]:
    out: list[Example] = []
    out += _from_searches(project, alpha)
    out += _from_results(project)
    return out[:limit]


def _pt(p: float, lp: float | None) -> str:
    """'3.2e-05', or the log10 magnitude when p is below the double range (never a literal 0)."""
    import math

    if lp is not None and lp == lp and lp / math.log(10) < -300:
        return f"< 1e-300 (log10 p = {lp / math.log(10):.1f})" if math.isfinite(lp) else "< 1e-300"
    return f"{p:.3g}"


def _from_searches(project: Project, alpha: float) -> list[Example]:
    from ..discovery.engine import results_frame

    out: list[Example] = []
    for run in project.searches:
        df = results_frame(run)
        if df.empty:
            continue
        fam = run.settings
        key = ["log_p", "idx"] if "log_p" in df else ["p", "idx"]      # AUD134-02: log tails order underflowed p
        for _, r in df[df["sig_adj"]].sort_values(key, kind="stable").head(5).iterrows():
            out.append(Example("Significant interaction (after adjustment)", r["formula"],
                               f"p = {_pt(r['p'], r.get('log_p'))}, adjusted p = {_pt(r['p_adj'], r.get('log_p_adj'))}; "
                               f"{r['interaction']}",
                               f"search {run.id} #{int(r['idx'])}"))
        lost = df[df["sig_raw"] & ~df["sig_adj"]].sort_values(key, kind="stable").head(5)
        for _, r in lost.iterrows():
            out.append(Example("Significant before but not after multiple-testing adjustment", r["formula"],
                               f"p = {r['p']:.3g} but adjusted p = {r['p_adj']:.3g} ({fam.correction}); "
                               f"{len(df[df['in_family']])} tests in the family",
                               f"search {run.id} #{int(r['idx'])}"))
        flagged = df[(df["status"] == "warning") & df["flags"].str.contains("|".join(UNSTABLE), case=False, na=False)]
        for _, r in flagged.head(3).iterrows():
            out.append(Example("Unstable estimates (screening flags)", r["formula"], r["flags"],
                               f"search {run.id} #{int(r['idx'])}"))
    return out


def _from_results(project: Project) -> list[Example]:
    out: list[Example] = []
    glmm: dict[str, Any] = {}
    gee: dict[str, Any] = {}
    for r in project.results:
        text = " ".join(r.messages).lower()
        if any(k in text for k in UNSTABLE):
            out.append(Example("Unstable estimates", r.title, next(m for m in r.messages
                                                                    if any(k in m.lower() for k in UNSTABLE)),
                               f"result {r.id}"))
        if any(k in text for k in ASSUMPTION):
            out.append(Example("Assumption violation", r.title, next(m for m in r.messages
                                                                      if any(k in m.lower() for k in ASSUMPTION)),
                               f"result {r.id}"))
        if r.kind == "likert_sensitivity" and any("DISAGREE" in str(n) for n in r.values.get("notes") or []):
            out.append(Example("Gaussian and ordinal approaches disagree", r.title,
                               "Direction or significance of a term depends on treating the outcome as numeric vs "
                               "ordinal.", f"result {r.id}"))
        if r.kind == "mixed_model":
            spec = r.inputs.get("spec") or {}
            key = (spec.get("outcome"), tuple(spec.get("predictors") or []))
            if spec.get("estimator") == "gee" and spec.get("family") == "binomial":
                gee[key] = r
            elif spec.get("family") == "binomial":
                glmm[key] = r
    for key in glmm.keys() & gee.keys():
        a, b = _first_or(glmm[key]), _first_or(gee[key])
        if a and b and a[0] == b[0]:
            out.append(Example("GLMM vs GEE interpretation", f"{glmm[key].title} vs {gee[key].title}",
                               f"{a[0]}: conditional OR {a[1]:.2f} (subject-specific) vs population-average OR "
                               f"{b[1]:.2f}. Different estimands: the GLMM odds ratio is usually further from 1.",
                               f"results {glmm[key].id}, {gee[key].id}"))
    return out


def _first_or(result) -> tuple[str, float] | None:
    for t in result.tables:
        if "odds_ratio" in t.columns and "term" in t.columns:
            ti, oi = t.columns.index("term"), t.columns.index("odds_ratio")
            for row in t.rows:
                if row[ti] != "(Intercept)" and row[oi] is not None:
                    return str(row[ti]), float(row[oi])
    return None
