"""Brief statistical interpretations (v1.1 §17.2): two or three sentences built from the fitted
results — what was tested, whether it was significant, what the estimate means, and the main
limitation. Never an essay, never invented numbers."""
from __future__ import annotations

from typing import Any

from ..core.schema import AnalysisResult, Table

COEF_TABLES = ("Coefficients", "Fixed effects", "Survey-weighted coefficients", "Coefficients (population-average)")
EST_COLS = ("estimate", "B", "log_odds", "gaussian_B")


def _p(p: float) -> str:
    return "p < .001" if p < 0.001 else f"p = {p:.3f}".replace("0.", ".", 1)


def _num(x: Any) -> float | None:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _table(result: AnalysisResult, prefixes: tuple[str, ...]) -> Table | None:
    return next((t for t in result.tables if t.name.startswith(prefixes)), None)


def _limitation(result: AnalysisResult) -> str:
    checks = [m.removeprefix("Check: ") for m in result.messages if m.startswith("Check:")]
    if checks:
        return "Caution: " + checks[0].rstrip(".") + "."
    if result.inputs.get("treatments"):
        return "The outcome or a predictor was treated as continuous for this analysis; results depend on that choice."
    return "This is an association in observed data, not evidence of cause and effect."


def brief(result: AnalysisResult, alpha: float = 0.05) -> str:
    if result.values.get("interpretation") or not result.tables:
        return result.values.get("interpretation", "")
    t = _table(result, ("Test of independence", "Chi-square test of independence"))
    if t is not None:
        row = dict(zip(t.columns, t.rows[0]))
        p = _num(row.get("p"))
        meas = _table(result, ("Association measures",))
        v = _num(meas.rows[0][1]) if meas and meas.rows else _num(row.get("cramers_v"))
        if p is None:
            return ""
        size = "" if v is None else (" Cramér's V = %.2f (%s association)." % (v, "negligible" if v < 0.1 else "small" if v < 0.3
                                                                              else "moderate" if v < 0.5 else "strong"))
        return (f"Tested whether the two categorical variables are associated: {'statistically significant' if p < alpha else 'not statistically significant'} "
                f"({_p(p)}).{size} {_limitation(result)}")
    t = _table(result, COEF_TABLES)
    if t is None:
        return ""
    cols = t.columns
    ti = cols.index("term") if "term" in cols else None
    ei = next((cols.index(c) for c in EST_COLS if c in cols), None)
    pi = next((cols.index(c) for c in ("p", "gaussian_p") if c in cols), None)
    if ti is None or ei is None or pi is None:
        return ""
    rows = [r for r in t.rows if str(r[ti]) != "(Intercept)" and _num(r[pi]) is not None]
    if not rows:
        return ""
    best = min(rows, key=lambda r: _num(r[pi]))
    term, est, p = best[ti], _num(best[ei]), _num(best[pi])
    sig = "statistically significant" if p < alpha else "not statistically significant"
    if "odds_ratio" in cols:
        orv = _num(best[cols.index("odds_ratio")])
        effect = f"odds ratio {orv:.2f}" + (" (conditional on the participant)" if "subject-specific" in t.notes else "")
    elif "RRR" in cols:
        cmp = best[cols.index("comparison_category")] if "comparison_category" in cols else None
        ref = best[cols.index("reference_category")] if "reference_category" in cols else None
        effect = (f"relative risk ratio {_num(best[cols.index('RRR')]):.2f} for outcome category "
                  + (f"'{cmp}' vs the reference category '{ref}'" if cmp is not None and ref is not None
                     else "one category vs the reference"))
    else:
        effect = f"estimate {est:.3g}"
    n_terms = len(rows)
    # P1-05: the smallest p-value is not "the strongest predictor"; with interactions, lower-order terms are
    # conditional effects at the reference level (or zero) of the interacting variables
    lead = "Term with the smallest p-value" if n_terms > 1 else "Tested term"
    cond = ""
    if any(":" in str(r[ti]) for r in rows):
        cond = (" The model contains interactions, so main-effect coefficients are conditional effects at the "
                "reference level (or zero) of the interacting variables, not overall effects.")
    return f"{lead}: {term} — {sig} ({_p(p)}; {effect}).{cond} {_limitation(result)}"


def tables_tsv(result: AnalysisResult) -> str:
    """All result tables as tab-separated text (pastes into Word or Excel as tables)."""
    out = []
    for t in result.tables:
        out.append(t.name)
        head = ([""] if t.row_names else []) + list(t.columns)
        out.append("\t".join(head))
        for i, r in enumerate(t.rows):
            cells = ([t.row_names[i]] if t.row_names else []) + ["" if v is None else str(v) for v in r]
            out.append("\t".join(cells))
        if t.notes:
            out.append(t.notes)
        out.append("")
    return "\n".join(out)
