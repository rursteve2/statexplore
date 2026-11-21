"""Phase 4 acceptance: graphs reflect the fitted model, confidence bands are correct, reference
groups are handled, code reproduces the results; model comparison uses valid tests only."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import scipy.stats as st

from statexplore.analysis import comparison, teaching
from statexplore.analysis.models import ModelSpec, close_hierarchy, drop_term
from statexplore.core.importers import import_file
from statexplore.core.schema import Status


def load(tmp_path, df, name="t.csv"):
    p = tmp_path / name
    df.to_csv(p, index=False)
    return import_file(p)


def tab(res, name):
    t = next(t for t in res.tables if t.name.startswith(name))
    return pd.DataFrame(t.rows, columns=t.columns)


# ---- model specs (no R) -------------------------------------------------------------

def test_hierarchy_and_nesting():
    assert close_hierarchy([("a", "b", "c")]) == [("a",), ("b",), ("c",), ("a", "b"), ("a", "c"), ("b", "c"),
                                                  ("a", "b", "c")]
    assert drop_term(close_hierarchy([("a", "b"), ("c",)]), ("a",)) == [("b",), ("c",)]
    small = ModelSpec("y", [("a",), ("b",)], "gaussian")
    big = ModelSpec("y", [("a", "b")], "gaussian")
    assert small.nested_in(big) and not big.nested_in(small)
    assert big.focal == ("a", "b") and small.focal is None
    assert not small.nested_in(ModelSpec("z", [("a", "b")], "gaussian"))
    again = ModelSpec.from_dict(big.to_dict())
    assert again.terms == big.terms and again.focal == big.focal
    assert comparison.nested_pairs([small, big, ModelSpec("y", [("c",)], "gaussian")]) == [(0, 1)]


# ---- Teaching Studio (R) ------------------------------------------------------------

@pytest.fixture
def lin(tmp_path):
    rng = np.random.default_rng(21)
    n = 250
    df = pd.DataFrame({"x": rng.normal(size=n), "m": rng.normal(1, 2, n), "c": rng.normal(size=n),
                       "g": rng.choice(["lo", "hi"], n), "h": rng.choice(["p", "q", "r"], n)})
    df["y"] = 1 + 0.4 * df["x"] + 0.3 * df["m"] + 0.35 * df["x"] * df["m"] + 0.5 * df["c"] \
        + np.where(df["g"] == "hi", 1.0, 0) * np.where(df["h"] == "q", 1.2, 0) + rng.normal(size=n)
    return df, load(tmp_path, df)


@pytest.mark.r
def test_continuous_interaction_plot_matches_statsmodels(lin, r_session):
    import statsmodels.formula.api as smf

    df, ds = lin
    m = ModelSpec("y", [("x", "m"), ("c",)], "gaussian")
    res, art = teaching.run(r_session, ds, m, teaching.PlotSpec(x="x", group="m"))
    assert res.status is Status.OK, res.messages
    ref = smf.ols("y ~ x * m + c", df).fit()
    pred = tab(res, "Plotted predictions")
    assert len(pred) == 80 * 3
    assert np.allclose(pred["c"], df["c"].mean())  # held at the mean (displayed)
    sp = ref.get_prediction(pred[["x", "m", "c"]]).summary_frame(alpha=0.05)
    np.testing.assert_allclose(pred["predicted"], sp["mean"], rtol=1e-9)
    np.testing.assert_allclose(pred["ci_low"], sp["mean_ci_lower"], rtol=1e-9)
    np.testing.assert_allclose(pred["ci_high"], sp["mean_ci_upper"], rtol=1e-9)
    # moderator values: mean -1 SD, mean, +1 SD
    np.testing.assert_allclose(sorted(pred["m"].unique()), df["m"].mean() + np.array([-1, 0, 1]) * df["m"].std(), rtol=1e-12)
    test = tab(res, "Interaction test").iloc[0]
    assert test["p"] == pytest.approx(ref.pvalues["x:m"], rel=1e-8)
    slopes = tab(res, "Conditional slopes")
    np.testing.assert_allclose(slopes["x.trend"], ref.params["x"] + ref.params["x:m"] * slopes["m"], rtol=1e-9)
    # Johnson-Neyman: at the bounds the slope's t equals the critical t
    jn = tab(res, "Johnson-Neyman").iloc[0]
    V = ref.cov_params()
    tcrit = st.t.ppf(0.975, ref.df_resid)
    for b in (jn["lower"], jn["upper"]):
        if b is None or (isinstance(b, float) and math.isnan(b)):
            continue
        slope = ref.params["x"] + ref.params["x:m"] * b
        se = math.sqrt(V.loc["x", "x"] + 2 * b * V.loc["x", "x:m"] + b * b * V.loc["x:m", "x:m"])
        assert abs(slope / se) == pytest.approx(tcrit, rel=1e-8)
    text = res.values["interpretation"]
    assert "statistically significant" in text and 3 <= text.count(". ") + 1 <= 7
    assert art[res.plots[0]][:4] == b"\x89PNG"


@pytest.mark.r
def test_categorical_interaction_reference_levels(lin, r_session):
    import statsmodels.formula.api as smf

    df, ds = lin
    ds.set_variable_field("h", "reference_level", "r")  # held factor will be "r"
    m = ModelSpec("y", [("g", "h"), ("x",)], "gaussian")
    res, _ = teaching.run(r_session, ds, m, teaching.PlotSpec(x="h", group="g"))
    pred = tab(res, "Plotted predictions")
    assert set(pred["x"].round(12)) == {round(df["x"].mean(), 12)}
    ref = smf.ols("y ~ C(g) * C(h, Treatment('r')) + x", df).fit()
    np.testing.assert_allclose(pred["predicted"], ref.predict(pred[["g", "h", "x"]]), rtol=1e-9)
    se = tab(res, "Simple effects")
    assert len(se) == 3 * 2  # 3 pairwise contrasts among h levels, at each of the 2 g levels
    # a held factor is shown at its reference level
    m2 = ModelSpec("y", [("x", "m"), ("h",)], "gaussian")
    res2, _ = teaching.run(r_session, ds, m2, teaching.PlotSpec(x="x", group="m"))
    assert set(tab(res2, "Plotted predictions")["h"]) == {"r"}
    assert "h = r" in res2.values["held"]


@pytest.mark.r
def test_logistic_probability_curves(tmp_path, r_session):
    import statsmodels.formula.api as smf

    rng = np.random.default_rng(8)
    n = 500
    df = pd.DataFrame({"x": rng.normal(size=n), "g": rng.choice(["a", "b"], n)})
    eta = -0.2 + 0.7 * df["x"] + 0.9 * df["x"] * (df["g"] == "b")
    df["y"] = rng.binomial(1, 1 / (1 + np.exp(-eta)))
    ds = load(tmp_path, df)
    m = ModelSpec("y", [("x", "g")], "binomial")
    res, _ = teaching.run(r_session, ds, m, teaching.PlotSpec(x="x", group="g", show_data=True))
    pred = tab(res, "Plotted predictions")
    ref = smf.logit("y ~ x * C(g)", df).fit(disp=0)
    np.testing.assert_allclose(pred["predicted"], ref.predict(pred[["x", "g"]]), rtol=1e-6)
    assert ((pred["ci_low"] < pred["predicted"]) & (pred["predicted"] < pred["ci_high"])).all()
    assert "probability" in res.values["interpretation"]


@pytest.mark.r
@pytest.mark.parametrize("backend,fmt,magic", [("ggplot2", "svg", b"<?xml"), ("ggplot2", "pdf", b"%PDF"),
                                               ("base", "png", b"\x89PNG"), ("base", "pdf", b"%PDF")])
def test_backends_and_formats(lin, r_session, backend, fmt, magic):
    _, ds = lin
    m = ModelSpec("y", [("x", "m", "g")], "gaussian")  # three-way, with a panel
    res, art = teaching.run(r_session, ds, m, teaching.PlotSpec(x="x", group="m", panel="g", backend=backend, fmt=fmt))
    assert res.status in (Status.OK, Status.WARNING), res.messages
    assert art[res.plots[0]][:4] == b"\x89PNG"  # preview is always PNG
    assert art[res.values["export"] if fmt != "png" else res.plots[0]][:len(magic)] == magic
    assert "three-way" in res.values["interpretation"]


@pytest.mark.r
def test_flexplot_backend_degrades_gracefully(lin, r_session):
    _, ds = lin
    m = ModelSpec("y", [("x", "g")], "gaussian")
    have = r_session.worker.run("sx_result <- requireNamespace('flexplot', quietly = TRUE)").result
    res, art = teaching.run(r_session, ds, m, teaching.PlotSpec(x="x", group="g", backend="flexplot"))
    if have:
        assert res.plots
    else:
        assert res.status is Status.FAILED and "not installed" in res.messages[0]


@pytest.mark.r
def test_teaching_code_reproduces(lin, r_session):
    _, ds = lin
    m = ModelSpec("y", [("x", "m")], "gaussian")
    res, _ = teaching.run(r_session, ds, m, teaching.PlotSpec(x="x", group="m"))
    fresh = r_session.worker.run("local({\n" + res.script + "\nsx_result\n}) -> sx_result").result
    got = [t for t in fresh["tables"] if t["name"] == "Plotted predictions"][0]["data"]["predicted"]
    np.testing.assert_allclose(got, tab(res, "Plotted predictions")["predicted"], rtol=1e-12)


# ---- Model Comparison (R) ---------------------------------------------------------------

@pytest.mark.r
def test_comparison_nested_tests_and_warnings(lin, r_session, tmp_path):
    import statsmodels.formula.api as smf

    df, ds = lin
    m1 = ModelSpec("y", [("x",), ("m",)], "gaussian", label="main")
    m2 = ModelSpec("y", [("x", "m")], "gaussian", label="interaction")
    m3 = ModelSpec("y", [("c",)], "gaussian", label="other")
    res, _ = comparison.compare(r_session, ds, [m1, m2, m3])
    fit = tab(res, "Model fit").set_index("model")
    r1, r2 = smf.ols("y ~ x + m", df).fit(), smf.ols("y ~ x * m", df).fit()
    # R's AIC counts the residual variance as a parameter (statsmodels does not)
    assert fit.loc["main", "AIC"] == pytest.approx(-2 * r1.llf + 2 * (r1.df_model + 2), rel=1e-9)
    assert fit.loc["interaction", "R2"] == pytest.approx(r2.rsquared, rel=1e-9)
    nest = tab(res, "Nested model tests")
    assert list(zip(nest["smaller"], nest["larger"])) == [("main", "interaction")]  # non-nested pairs not tested
    f, p, _ = r2.compare_f_test(r1)
    assert nest.loc[0, "statistic"] == pytest.approx(f, rel=1e-9) and nest.loc[0, "p"] == pytest.approx(p, rel=1e-8)
    assert "2 non-nested pair(s)" in res.values["notes"][0]
    assert len(res.plots) == 1  # model comparison plot

    # different observations: warned and nested tests skipped
    df2 = df.copy()
    df2.loc[:9, "c"] = np.nan
    ds2 = load(tmp_path, df2, "miss.csv")
    a = ModelSpec("y", [("x",)], "gaussian", label="a")
    b = ModelSpec("y", [("x",), ("c",)], "gaussian", label="b")
    res, _ = comparison.compare(r_session, ds2, [a, b], common_sample=False)
    assert res.status is Status.WARNING and "different observations" in res.messages[0]
    assert not any(t.name.startswith("Nested") for t in res.tables)
    res, _ = comparison.compare(r_session, ds2, [a, b], common_sample=True)
    assert tab(res, "Model fit")["n"].tolist() == [240, 240] and any(t.name.startswith("Nested") for t in res.tables)


def test_comparison_rejects_mixed_outcomes(lin):
    _, ds = lin
    with pytest.raises(comparison.ComparisonError):
        comparison.build(ds, [ModelSpec("y", [("x",)], "gaussian"), ModelSpec("m", [("x",)], "gaussian")], True)


def test_user_reported_bad_model_is_rejected_with_reason(hrs_path):
    """Reported: HHIDPN ~ HHID * PN + HHID * ADVTR_MATCHED * WAVE ... gave a raw R contrasts error."""
    from statexplore.analysis.models import ModelCheckError, check_model

    ds = import_file(hrs_path)
    bad = ModelSpec("HHIDPN", [("HHID", "PN"), ("HHID", "ADVTR_MATCHED", "WAVE")], "gaussian")
    with pytest.raises(ModelCheckError) as e:
        check_model(ds, bad)
    assert "ADVTR_MATCHED" in str(e.value) or "HHID" in str(e.value)
    with pytest.raises(ModelCheckError, match="same value in every row"):
        check_model(ds, ModelSpec("R_BMI", [("ADVTR_MATCHED",)], "gaussian"))
    with pytest.raises(ModelCheckError, match="categories: it looks like an identifier"):
        check_model(ds, ModelSpec("R_BMI", [("HHID",)], "gaussian"))
    ds.set_variable_field("HHIDPN", "role", "id")
    with pytest.raises(ModelCheckError, match="participant ID"):
        check_model(ds, ModelSpec("R_BMI", [("HHIDPN",)], "gaussian"))
    with pytest.raises(teaching.TeachingInputError):
        teaching.validate(ds, bad, teaching.PlotSpec(x="HHID"))
    assert check_model(ds, ModelSpec("R_BMI", [("RAGENDER", "R_LBNEUR")], "gaussian")) > 1000


def test_r_errors_are_explained():
    from statexplore.analysis.session import explain_r_error

    assert explain_r_error("contrasts can be applied only to factors with 2 or more levels")[0].startswith("What this means")
    assert explain_r_error("something else") == []


@pytest.mark.r
def test_user_held_values(lin, r_session):
    import statsmodels.formula.api as smf

    df, ds = lin
    m = ModelSpec("y", [("x", "m"), ("c",), ("h",)], "gaussian")
    res, _ = teaching.run(r_session, ds, m, teaching.PlotSpec(x="x", group="m", held={"c": 1.5, "h": "q"}))
    pred = tab(res, "Plotted predictions")
    assert set(pred["c"]) == {1.5} and set(pred["h"]) == {"q"}
    ref = smf.ols("y ~ x * m + c + C(h)", df).fit()
    np.testing.assert_allclose(pred["predicted"], ref.predict(pred[["x", "m", "c", "h"]]), rtol=1e-9)
    with pytest.raises(teaching.TeachingInputError):
        teaching.validate(ds, m, teaching.PlotSpec(x="x", held={"h": "zzz"}))
