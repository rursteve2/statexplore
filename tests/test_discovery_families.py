"""v1.1 Phase 4 (§10, §20 H): explicit outcome families in discovery (Gaussian, binary, ordinal,
multinomial), continuous overrides, family-specific likelihood-ratio focal tests, and the
matching Analysis Builder regressions; references: statsmodels OrderedModel / MNLogit."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.stats as st
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import MNLogit
from statsmodels.miscmodels.ordinal_model import OrderedModel

from statexplore.analysis import measurement as M
from statexplore.analysis.builders import BY_KEY
from statexplore.analysis.framework import run_analysis
from statexplore.core.importers import import_file, import_sav
from statexplore.core.project import Project
from statexplore.discovery.detail import candidate_from_row, full_analysis
from statexplore.discovery.engine import create_run, execute, results_frame
from statexplore.discovery.space import SearchSettings, SearchSpaceError, suggest_family, validate_settings

pytestmark = pytest.mark.r


def ordinal_data(tmp_path, n=600, seed=11):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    g = rng.integers(0, 2, n)
    z = rng.normal(size=n)
    latent = 0.5 * x + 0.4 * g + 1.2 * x * g + rng.logistic(size=n)
    y = np.digitize(latent, [-1.0, 0.3, 1.5]) + 1           # 1..4
    df = pd.DataFrame({"y": y.astype(float), "x": x, "g": g.astype(float), "z": z})
    p = tmp_path / "ord.csv"
    df.to_csv(p, index=False)
    ds = import_file(p)
    ds.set_variable_field("y", "measure", "ordinal")
    ds.set_variable_field("g", "measure", "nominal")
    return ds, df


def multinomial_data(tmp_path, n=900, seed=5):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    g = rng.integers(0, 2, n)
    z = rng.normal(size=n)
    eta1 = 0.3 + 0.5 * x + 1.4 * x * g
    eta2 = -0.2 - 0.4 * x + 0.6 * g
    P = np.column_stack([np.ones(n), np.exp(eta1), np.exp(eta2)])
    P /= P.sum(1, keepdims=True)
    y = np.array([rng.choice(3, p=row) for row in P]) + 1
    df = pd.DataFrame({"y": y.astype(float), "x": x, "g": g.astype(float), "z": z})
    p = tmp_path / "mn.csv"
    df.to_csv(p, index=False)
    ds = import_file(p)
    ds.set_variable_field("y", "measure", "nominal")
    ds.set_variable_field("g", "measure", "nominal")
    return ds, df


def search(ds, tmp_path, **kw):
    s = SearchSettings(outcome="y", candidates=["x", "g", "z"], workers=1, max_models=None, max_minutes=None, **kw)
    run = create_run(ds, s)
    execute(run, ds, workdir=tmp_path / "w")
    return run, results_frame(run)


def row(df, a, b):
    return df[df["focal"].apply(lambda f: set(f) == {a, b})].iloc[0]


def test_H_ordinal_family_lr_matches_statsmodels(tmp_path):
    ds, df = ordinal_data(tmp_path)
    run, res = search(ds, tmp_path, family="ordinal")
    assert run.family == "ordinal" and run.status == "complete"
    assert run.coding["ordering"] == ["1", "2", "3", "4"] and "clm" in run.coding["family"]
    r = row(res, "x", "g")
    X1 = pd.DataFrame({"x": df.x, "g": df.g, "xg": df.x * df.g})
    full = OrderedModel(df.y, X1, distr="logit").fit(method="bfgs", disp=False, maxiter=2000)
    red = OrderedModel(df.y, X1[["x", "g"]], distr="logit").fit(method="bfgs", disp=False, maxiter=2000)
    lr = 2 * (full.llf - red.llf)
    assert r["statistic"] == pytest.approx(lr, rel=1e-5) and r["df1"] == 1
    assert r["p"] < 1e-6 and r["sig_adj"]
    assert r["estimate"] == pytest.approx(full.params["xg"], rel=1e-3)   # same sign convention (cut - x'b)
    assert r["n_params"] == 3 + 3                                        # x, g, x:g + 3 thresholds
    others = res[~res["focal"].apply(lambda f: set(f) == {"x", "g"})]
    assert (others["p"] > 0.001).all()
    assert "McFadden" in r["interpretation"]


def test_H_multinomial_family_lr_matches_statsmodels(tmp_path):
    ds, df = multinomial_data(tmp_path)
    run, res = search(ds, tmp_path, family="multinomial")
    assert run.coding["reference"] == "1"
    r = row(res, "x", "g")
    X1 = sm.add_constant(pd.DataFrame({"x": df.x, "g": df.g, "xg": df.x * df.g}))
    full = MNLogit(df.y, X1).fit(disp=False, method="newton", maxiter=200)
    red = MNLogit(df.y, X1[["const", "x", "g"]]).fit(disp=False, method="newton", maxiter=200)
    assert r["df1"] == 2 and r["n_params"] == 2 * 4                      # joint over both contrasts
    assert r["statistic"] == pytest.approx(2 * (full.llf - red.llf), rel=1e-4)
    assert r["p"] < 1e-6
    assert r["estimate"] is None or np.isnan(r["estimate"])               # no single coefficient for 2 df
    # reference category changes the parametrisation, not the likelihood-ratio test
    run2, res2 = search(ds, tmp_path / "b", family="multinomial", reference=3.0)
    assert run2.coding["reference"] == "3"
    assert row(res2, "x", "g")["statistic"] == pytest.approx(r["statistic"], rel=1e-4)


def test_H_binomial_event_choice(tmp_path):
    rng = np.random.default_rng(2)
    n = 500
    df = pd.DataFrame({"x": rng.normal(size=n), "g": rng.integers(0, 2, n).astype(float), "z": rng.normal(size=n)})
    eta = 0.3 * df.x + 0.2 * df.g + 1.0 * df.x * df.g
    df["y"] = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(float)
    p = tmp_path / "b.csv"
    df.to_csv(p, index=False)
    ds = import_file(p)
    _, a = search(ds, tmp_path / "a", family="binomial")
    run_b, b = search(ds, tmp_path / "b", family="binomial", event=0.0)
    assert run_b.coding["event"] == "0" and run_b.coding["event_source"] == "chosen by the user"
    ra, rb = row(a, "x", "g"), row(b, "x", "g")
    assert ra["p"] == pytest.approx(rb["p"], rel=1e-8)                    # LR test is invariant
    assert rb["estimate"] == pytest.approx(-ra["estimate"], rel=1e-6)    # coefficients flip sign
    fit = sm.Logit(df.y, sm.add_constant(pd.DataFrame({"x": df.x, "g": df.g, "xg": df.x * df.g}))).fit(disp=False)
    assert ra["estimate"] == pytest.approx(fit.params["xg"], rel=1e-5)


def test_H_family_validation_and_suggestions(hrs_path):
    ds = import_sav(hrs_path)
    v = lambda **kw: validate_settings(ds, SearchSettings(outcome="R_SHLT", candidates=["R_BMI", "RAGENDER"], **kw))  # noqa: E731
    with pytest.raises(SearchSpaceError, match="Treat R_SHLT as continuous"):
        v(family="gaussian")
    with pytest.raises(SearchSpaceError, match="ordered categories"):
        v(family="ordinal")                                             # project level is still nominal
    assert v(family="multinomial") == "multinomial"
    with pytest.raises(SearchSpaceError, match="exactly two"):
        v(family="binomial")
    tr = [M.VariableTreatment("R_SHLT", "continuous", acknowledged=True, numeric_confirmed=True).to_dict()]
    assert v(family="gaussian", treatments=tr) == "gaussian"
    assert suggest_family(ds, "R_SHLT")[0] == "multinomial"
    M.approve(ds, "R_SHLT", "ordinal")
    assert v(family="ordinal") == "ordinal"
    assert suggest_family(ds, "R_SHLT")[0] == "ordinal"
    assert suggest_family(ds, "RAGENDER")[0] == "binomial" and suggest_family(ds, "R_BMI")[0] == "gaussian"
    # v1.2: repeated ordinal outcomes use ordinal::clmm; a Gaussian structure is refused with a reason
    with pytest.raises(SearchSpaceError, match="not available for ordinal outcomes"):
        v(family="ordinal", participant="HHIDPN")
    assert v(family="ordinal", participant="HHIDPN", structure="clmm_ri") == "clmm"


def test_H_continuous_override_in_hrs_search(hrs_path, tmp_path):
    ds = import_sav(hrs_path)
    ds.add_filter(__import__("statexplore.core.derive", fromlist=["FilterSpec"]).FilterSpec("WAVE", "==", 10.0))
    tr = [M.VariableTreatment("R_SHLT", "continuous", acknowledged=True, numeric_confirmed=True).to_dict()]
    s = SearchSettings(outcome="R_SHLT", candidates=["R_BMI", "RAGENDER", "R_LBNEUR"], family="gaussian",
                       treatments=tr, workers=1)
    run = create_run(ds, s)
    execute(run, ds, workdir=tmp_path / "w")
    res = results_frame(run)
    assert run.status == "complete" and run.coding["outcome_treatment"].startswith("R_SHLT: Continuous")
    assert ds.var("R_SHLT").measure.value == "nominal"                   # project metadata unchanged
    # independent check of one candidate: nested F test on the numeric codes 1..5
    f = ds.analysis_frame(["R_SHLT", "R_BMI", "RAGENDER", "R_LBNEUR"]).dropna()
    import statsmodels.formula.api as smf

    full = smf.ols("R_SHLT ~ R_BMI * R_LBNEUR + 0 + 1", f).fit()
    red = smf.ols("R_SHLT ~ R_BMI + R_LBNEUR", f).fit()
    F = ((red.ssr - full.ssr) / 1) / (full.ssr / full.df_resid)
    assert row(res, "R_BMI", "R_LBNEUR")["statistic"] == pytest.approx(F, rel=1e-8)
    # project round-trip keeps family, treatment and coding
    p = tmp_path / "p.sxp"
    pr = Project(ds)
    pr.searches.append(run)
    pr.save(p)
    back = Project.open(p).searches[0]
    assert back.settings.family == "gaussian" and back.settings.treatments == tr and back.coding == run.coding


def test_H_drill_down_ordinal_matches_search(tmp_path, r_session):
    ds, _ = ordinal_data(tmp_path)
    run, res = search(ds, tmp_path, family="ordinal")
    r = row(res, "x", "g")
    result, _ = full_analysis(r_session, ds, run, candidate_from_row(run, r.to_dict()))
    assert result.status.value in ("ok", "warning"), result.messages
    terms = next(t for t in result.tables if t.name.startswith("Term tests"))
    lr = dict(zip([x[0] for x in terms.rows], [x[3] for x in terms.rows]))
    assert lr["x:g"] == pytest.approx(r["p"], rel=1e-6)
    assert any(t.name.startswith("Proportional-odds") for t in result.tables)


def test_ordinal_and_multinomial_regressions_vs_statsmodels(tmp_path, r_session):
    ds, df = ordinal_data(tmp_path)
    res, _ = run_analysis(r_session, ds, BY_KEY["ordinal_reg"], {"dependent": ["y"], "predictors": ["x", "z"]}, {})
    co = next(t for t in res.tables if t.name.startswith("Coefficients"))
    b = {r[0]: r[1] for r in co.rows}
    ref = OrderedModel(df.y, df[["x", "z"]], distr="logit").fit(method="bfgs", disp=False, maxiter=2000)
    assert b["x"] == pytest.approx(ref.params["x"], rel=1e-4) and b["z"] == pytest.approx(ref.params["z"], abs=2e-4)
    fit = next(t for t in res.tables if t.name == "Model fit")
    assert fit.rows[0][fit.columns.index("logLik")] == pytest.approx(ref.llf, rel=1e-8)

    ds2, df2 = multinomial_data(tmp_path)
    res2, _ = run_analysis(r_session, ds2, BY_KEY["multinom_reg"], {"dependent": ["y"], "predictors": ["x", "z"]}, {})
    co2 = next(t for t in res2.tables if t.name.startswith("Coefficients"))
    got = {(r[0], r[1]): (r[2], r[3]) for r in co2.rows}
    ref2 = MNLogit(df2.y, sm.add_constant(df2[["x", "z"]])).fit(disp=False, method="newton")
    for j, cat in enumerate(["2", "3"]):
        for term, col in (("(Intercept)", "const"), ("x", "x"), ("z", "z")):
            est, se = got[(cat, term)]
            assert est == pytest.approx(ref2.params.iloc[:, j][col], abs=2e-4)
            assert se == pytest.approx(ref2.bse.iloc[:, j][col], rel=2e-3)
    an = next(t for t in res2.tables if t.name.startswith("Term tests"))
    red = MNLogit(df2.y, sm.add_constant(df2[["z"]])).fit(disp=False, method="newton")
    assert dict((r[0], r[1]) for r in an.rows)["x"] == pytest.approx(2 * (ref2.llf - red.llf), rel=1e-4)
    # reference category option changes the baseline
    res3, _ = run_analysis(r_session, ds2, BY_KEY["multinom_reg"], {"dependent": ["y"], "predictors": ["x"]},
                           {"reference": 2.0})
    cats = {r[0] for r in next(t for t in res3.tables if t.name.startswith("Coefficients")).rows}
    assert cats == {"1", "3"}


def test_logistic_event_option(tmp_path, r_session):
    rng = np.random.default_rng(8)
    df = pd.DataFrame({"x": rng.normal(size=300)})
    df["y"] = np.where(rng.random(300) < 1 / (1 + np.exp(-df.x)), "yes", "no")
    p = tmp_path / "l.csv"
    df.to_csv(p, index=False)
    ds = import_file(p)
    yes = "yes"
    base, _ = run_analysis(r_session, ds, BY_KEY["logistic"], {"dependent": ["y"], "predictors": ["x"]}, {})
    flip, _ = run_analysis(r_session, ds, BY_KEY["logistic"], {"dependent": ["y"], "predictors": ["x"]}, {"event": "no"})
    b = lambda r: {x[0]: x[1] for x in next(t for t in r.tables if t.name == "Coefficients").rows}["x"]  # noqa: E731
    assert b(flip) == pytest.approx(-b(base), rel=1e-8)
    assert "= no" in next(t for t in flip.tables if t.name == "Coefficients").notes
    assert "= " + yes in next(t for t in base.tables if t.name == "Coefficients").notes
