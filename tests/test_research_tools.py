"""v1.1 §17: Likert sensitivity comparison, brief interpretations, copyable tables, Teaching Example Finder."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel

from statexplore.analysis import measurement as M
from statexplore.analysis import mixed as MX
from statexplore.analysis.builders import BY_KEY
from statexplore.analysis.framework import AnalysisInputError, run_analysis
from statexplore.analysis.interpret import tables_tsv
from statexplore.analysis.teaching_finder import find
from statexplore.core.derive import FilterSpec
from statexplore.core.importers import import_file, import_sav
from statexplore.core.project import Project

pytestmark = pytest.mark.r


def test_likert_sensitivity_matches_references(hrs_path, r_session):
    ds = import_sav(hrs_path)
    ds.add_filter(FilterSpec("WAVE", "==", 10.0))
    M.approve(ds, "R_SHLT", "ordinal")
    res, _ = run_analysis(r_session, ds, BY_KEY["likert_sensitivity"],
                          {"dependent": ["R_SHLT"], "predictors": ["R_BMI", "R_LBNEUR"]}, {})
    assert res.status.value in ("ok", "warning"), res.messages
    t = next(t for t in res.tables if t.name.startswith("Gaussian vs ordinal"))
    rows = {r[0]: dict(zip(t.columns, r)) for r in t.rows}
    f = ds.analysis_frame(["R_SHLT", "R_BMI", "R_LBNEUR"]).dropna()
    ols = sm.OLS(f.R_SHLT, sm.add_constant(f[["R_BMI", "R_LBNEUR"]])).fit()
    om = OrderedModel(f.R_SHLT.astype(int), f[["R_BMI", "R_LBNEUR"]], distr="logit").fit(method="bfgs", disp=False,
                                                                                        maxiter=3000)
    for term in ("R_BMI", "R_LBNEUR"):
        assert rows[term]["gaussian_B"] == pytest.approx(ols.params[term], rel=1e-8)
        assert rows[term]["ordinal_log_odds"] == pytest.approx(om.params[term], rel=2e-3)
    assert "not interchangeable" in BY_KEY["likert_sensitivity"].description
    assert "NOT comparable" in next(t for t in res.tables if t.name == "Model assumptions").notes
    pred = next(t for t in res.tables if t.name == "Predictions")
    assert all(1 <= r[2] <= 5 for r in pred.rows)                         # expected code within the scale
    with pytest.raises(AnalysisInputError, match="ordinal"):
        run_analysis(r_session, import_sav(hrs_path), BY_KEY["likert_sensitivity"],
                     {"dependent": ["R_SHLT"], "predictors": ["R_BMI"]}, {})   # still nominal in the project


def test_brief_interpretation_and_tsv(tmp_path, r_session):
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.normal(size=200)})
    df["y"] = 0.5 * df.x + rng.normal(size=200)
    p = tmp_path / "d.csv"
    df.to_csv(p, index=False)
    res, _ = run_analysis(r_session, import_file(p), BY_KEY["lm"], {"dependent": ["y"], "predictors": ["x"]}, {})
    text = res.values["interpretation"]
    assert text.startswith("Tested term: x — statistically significant (p < .001; estimate 0.") and "not evidence of cause" in text
    assert len(text) < 300
    tsv = tables_tsv(res)
    assert "Coefficients\nterm\testimate" in tsv and "\n(Intercept)\t" in tsv


def test_teaching_finder_uses_real_results(tmp_path, r_session):
    from statexplore.discovery.engine import create_run, execute
    from statexplore.discovery.space import SearchSettings

    rng = np.random.default_rng(5)
    n = 300
    df = pd.DataFrame(rng.normal(size=(n, 4)), columns=["a", "b", "c", "d"])
    df["y"] = 1.2 * df.a * df.b + rng.normal(size=n)
    p = tmp_path / "d.csv"
    df.to_csv(p, index=False)
    ds = import_file(p)
    proj = Project(ds)
    assert find(proj) == []                                               # nothing run, nothing found
    run = create_run(ds, SearchSettings(outcome="y", candidates=["a", "b", "c", "d"], family="gaussian", workers=1))
    execute(run, ds, workdir=tmp_path / "w")
    proj.searches.append(run)
    # binary repeated outcome: GLMM and GEE on the same model
    k = 6
    m = 120
    long = pd.DataFrame({"id": np.repeat(np.arange(m), k) + 1.0, "x": rng.normal(size=m * k)})
    b = rng.normal(0, 1.5, m)[(long.id - 1).astype(int)]
    long["y"] = (rng.random(m * k) < 1 / (1 + np.exp(-(0.9 * long.x + b)))).astype(float)
    p2 = tmp_path / "l.csv"
    long.to_csv(p2, index=False)
    ds2 = import_file(p2)
    ds2.set_variable_field("id", "role", "id")
    ds2.set_variable_field("y", "measure", "nominal")
    for spec in (MX.MixedSpec("y", "id", predictors=["x"], family="binomial"),
                 MX.MixedSpec("y", "id", predictors=["x"], family="binomial", estimator="gee", random="none")):
        r, _ = MX.run(r_session, ds2, spec)
        proj.add_result(r)
    ex = find(proj)
    cats = {e.category for e in ex}
    assert "Significant interaction (after adjustment)" in cats
    sig = next(e for e in ex if e.category.startswith("Significant interaction"))
    assert "a × b" in sig.detail and sig.source.startswith(f"search {run.id}")
    gl = next(e for e in ex if e.category == "GLMM vs GEE interpretation")
    a, b = [float(x.split("OR ")[1].split(" ")[0].rstrip(".")) for x in gl.detail.split(" vs ")]
    assert a > b > 1                                                       # conditional OR further from 1


def test_exports_include_v11_result_kinds(tmp_path, r_session, qtbot):
    """Reports export (Excel / Word / PDF / R) handles the new analyses and keeps treatments visible."""
    from statexplore.analysis import survey as S
    from statexplore.export import exporters

    rng = np.random.default_rng(2)
    df = pd.DataFrame({"y": rng.integers(1, 6, 200).astype(float), "x": rng.normal(size=200),
                       "g": rng.choice(["a", "b"], 200), "b": rng.integers(0, 2, 200).astype(float),
                       "w": rng.uniform(1, 2, 200)})
    p = tmp_path / "d.csv"
    df.to_csv(p, index=False)
    ds = import_file(p)
    ds.set_variable_field("y", "measure", "ordinal")
    ds.set_variable_field("b", "measure", "nominal")
    proj = Project(ds)
    tr = [M.VariableTreatment("y", "continuous", acknowledged=True)]
    for res, art in [
        run_analysis(r_session, ds, BY_KEY["lm"], {"dependent": ["y"], "predictors": ["x"]}, {}, treatments=tr),
        run_analysis(r_session, ds, BY_KEY["assoc_2x2"], {"outcome": ["b"], "group": ["g"]},
                     {"event": 1.0, "outcome_ref": 0.0, "exposed": "b", "comparison": "a"}),
        run_analysis(r_session, ds, BY_KEY["ordinal_reg"], {"dependent": ["y"], "predictors": ["x"]}, {}),
        S.run(r_session, ds, S.SurveyDesign(weight="w"), "mean", ["x"]),
    ]:
        proj.add_result(res, art)
    written = exporters.export(proj, [r.id for r in proj.results], tmp_path / "out", {"xlsx", "docx", "pdf", "R"})
    assert {p.suffix for p in written} >= {".xlsx", ".docx", ".pdf", ".R"}
    import docx

    doc = docx.Document(next(p for p in written if p.suffix == ".docx"))
    text = "\n".join(par.text for par in doc.paragraphs)
    assert "Continuous (numeric codes) for this analysis" in text or "treat" in text.lower()
