"""Phase 6 acceptance: leakage prevention (holdout and grouped CV — Test N), correct
predictive metrics, selection kept inside training folds, unstable findings identified."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statexplore.analysis import validation
from statexplore.analysis.builders import BY_KEY
from statexplore.analysis.framework import run_analysis
from statexplore.analysis.models import ModelSpec
from statexplore.core import missing
from statexplore.core.dataset import HoldoutSpec
from statexplore.core.importers import import_file
from statexplore.core.project import Project
from statexplore.core.schema import Status
from statexplore.discovery.engine import create_run, execute
from statexplore.discovery.space import SearchSettings


def load(tmp_path, df, name="v.csv"):
    p = tmp_path / name
    df.to_csv(p, index=False)
    return import_file(p)


def tab(res, name):
    t = next(t for t in res.tables if t.name.startswith(name))
    return pd.DataFrame(t.rows, columns=t.columns)


@pytest.fixture
def repeated(tmp_path):
    rng = np.random.default_rng(12)
    ids = np.repeat(np.arange(1, 81), 4)
    person = rng.normal(0, 1.5, 80)[ids - 1]
    x = rng.normal(size=320) + person          # predictor partly person-level
    df = pd.DataFrame({"pid": ids, "wave": np.tile([1, 2, 3, 4], 80), "x": x,
                       "g": np.repeat(rng.choice(["a", "b"], 80), 4)})
    df["y"] = 0.5 * df["x"] + person + 0.3 * df["wave"] + rng.normal(size=320)
    ds = load(tmp_path, df)
    ds.set_variable_field("pid", "role", "id")
    ds.set_variable_field("wave", "role", "time")
    return df, ds


def test_holdout_split_deterministic_grouped_and_undoable(repeated):
    df, ds = repeated
    ds.set_holdout(HoldoutSpec(fraction=0.25, seed=7, group="pid"))
    hold = ds.analysis_frame(["pid"], part="holdout")["pid"]
    train = ds.analysis_frame(["pid"])["pid"]
    assert hold.nunique() == 20 and set(hold) & set(train) == set()   # whole participants
    assert len(hold) + len(train) == 320
    again = ds._compute_holdout()
    pd.testing.assert_series_equal(again, ds._holdout_mask)          # same seed, same split
    ds.undo()
    assert ds.holdout is None and len(ds.analysis_frame(["pid"])) == 320


def test_holdout_survives_project_roundtrip(repeated, tmp_path):
    _, ds = repeated
    ds.set_holdout(HoldoutSpec(fraction=0.2, seed=3))
    out = tmp_path / "h.sxp"
    Project(ds).save(out)
    again = Project.open(out).dataset
    pd.testing.assert_series_equal(again._holdout_mask, ds._holdout_mask)


@pytest.mark.r
def test_holdout_rows_never_used_by_analyses_or_discovery(repeated, r_session):
    df, ds = repeated
    ds.set_holdout(HoldoutSpec(fraction=0.25, seed=7, group="pid"))
    n_train = len(ds.analysis_frame(["y"]))
    res, _ = run_analysis(r_session, ds, BY_KEY["lm"], {"dependent": ["y"], "predictors": ["x"]}, {})
    assert res.n_used == n_train == 240, (res.status, res.n_used, n_train, res.messages)
    run = execute(create_run(ds, SearchSettings(outcome="y", candidates=["x", "g", "wave"], workers=1,
                                                max_models=None, max_minutes=None)), ds)
    assert run.n_sample == 240, (run.status, run.stop_reason, run.n_sample)


@pytest.mark.r
def test_holdout_evaluation_metrics_and_reuse_warning(repeated, r_session):
    df, ds = repeated
    ds.set_holdout(HoldoutSpec(fraction=0.25, seed=7, group="pid"))
    m = ModelSpec("y", [("x", "g")], "gaussian")
    res, _ = validation.holdout_evaluate(r_session, ds, m)
    assert res.status is Status.OK, res.messages
    # independent check: fit on training rows with numpy, score on holdout rows
    train = ds.analysis_frame(["y", "x", "g"]).dropna()
    hold = ds.analysis_frame(["y", "x", "g"], part="holdout").dropna()
    def X(d):  # noqa: E306
        gb = (d["g"] == "b").astype(float)
        return np.c_[np.ones(len(d)), d["x"], gb, d["x"] * gb]
    beta = np.linalg.lstsq(X(train), train["y"], rcond=None)[0]
    pred = X(hold) @ beta
    rmse = np.sqrt(np.mean((hold["y"] - pred) ** 2))
    perf = tab(res, "Holdout performance").set_index("metric")
    assert perf.loc["RMSE", "holdout"] == pytest.approx(rmse, rel=1e-9)
    assert perf.loc["R2", "holdout"] == pytest.approx(1 - np.sum((hold["y"] - pred) ** 2) / np.sum((hold["y"] - hold["y"].mean()) ** 2), rel=1e-9)
    assert len(tab(res, "Interaction replication")) == 2
    res2, _ = validation.holdout_evaluate(r_session, ds, m)
    assert res2.status is Status.WARNING and "used 2 times" in res2.messages[-1]


@pytest.mark.r
def test_N_grouped_cv_prevents_participant_leakage(repeated, r_session):
    _, ds = repeated
    m = ModelSpec("y", [("x",), ("wave",)], "gaussian")
    naive, _ = validation.cross_validate(r_session, ds, m, "kfold", k=5, seed=2)
    grouped, _ = validation.cross_validate(r_session, ds, m, "grouped", k=5, seed=2)
    assert (tab(naive, "Participant overlap")["participants_in_both"] > 0).all()
    assert naive.status is Status.WARNING and any("leakage" in msg for msg in naive.messages)
    assert (tab(grouped, "Participant overlap")["participants_in_both"] == 0).all()
    # per-fold metrics are internally consistent
    folds = tab(grouped, "Per-fold results")
    assert folds["n_test"].sum() == 320
    summ = tab(grouped, "Cross-validation summary").set_index("metric")
    assert summ.loc["RMSE", "cv_mean"] == pytest.approx(folds["RMSE"].mean(), rel=1e-12)
    temporal, _ = validation.cross_validate(r_session, ds, m, "temporal")
    assert list(tab(temporal, "Per-fold results")["split"]) == ["train <= 1, test = 2", "train <= 2, test = 3",
                                                                 "train <= 3, test = 4"]


@pytest.mark.r
def test_logistic_cv_metrics(tmp_path, r_session):

    rng = np.random.default_rng(4)
    df = pd.DataFrame({"x": rng.normal(size=300)})
    df["y"] = rng.binomial(1, 1 / (1 + np.exp(-1.2 * df["x"])))
    ds = load(tmp_path, df)
    res, _ = validation.cross_validate(r_session, ds, ModelSpec("y", [("x",)], "binomial"), "kfold", k=4, seed=1)
    summ = tab(res, "Cross-validation summary").set_index("metric")
    assert set(summ.index) == {"log_loss", "Brier", "AUC"} and 0.6 < summ.loc["AUC", "cv_mean"] < 0.95


@pytest.mark.r
def test_stability_and_procedure_cv(tmp_path, r_session):
    rng = np.random.default_rng(5)
    n = 300
    df = pd.DataFrame(rng.normal(size=(n, 4)), columns=["a", "b", "c", "d"])
    df["y"] = df["a"] * df["b"] * 0.5 + rng.normal(size=n)
    ds = load(tmp_path, df)
    run = execute(create_run(ds, SearchSettings(outcome="y", candidates=["a", "b", "c", "d"], workers=1,
                                                max_models=None, max_minutes=None)), ds)
    rs = validation.ResampleSpec(resamples=10, fraction=0.8, seed=3, folds=4)
    res, _ = validation.stability(r_session, ds, run, rs)
    st = tab(res, "Discovery stability")
    top = st.iloc[0]
    assert top["model"] == "a × b" and top["selected_significant"] == 1.0 and top["ranked_first"] == 1.0
    assert st["selected_significant"].between(0, 1).all()
    from statexplore.discovery.engine import results_frame as rf

    frame = rf(run).set_index("interaction")
    assert frame.loc["a × b", "validation"].startswith("stability: p < 0.05 in 100%")
    assert frame.loc["a × b", "interpretation"].startswith("Evidence that the a–y relationship differs across b")
    res, _ = validation.procedure_cv(r_session, ds, run, rs)
    cvp = tab(res, "Full-procedure cross-validation")
    assert len(cvp) == 4 and cvp["n_test"].sum() == n
    assert (cvp["selected"] == "a × b").all() and (cvp["p_test_fold"] < 0.05).all()


@pytest.mark.r
def test_prediction_intervals(tmp_path, r_session):
    import statsmodels.formula.api as smf

    rng = np.random.default_rng(8)
    df = pd.DataFrame({"x": rng.normal(size=100)})
    df["y"] = 2 + df["x"] + rng.normal(size=100)
    ds = load(tmp_path, df)
    res, art = validation.prediction_uncertainty(r_session, ds, ModelSpec("y", [("x",)], "gaussian"))
    pr = tab(res, "Predictions")
    sf = smf.ols("y ~ x", df).fit().get_prediction(pr[["x"]]).summary_frame(alpha=0.05)
    np.testing.assert_allclose(pr["pi_low"], sf["obs_ci_lower"], rtol=1e-9)
    np.testing.assert_allclose(pr["ci_high"], sf["mean_ci_upper"], rtol=1e-9)
    assert (pr["pi_high"] - pr["pi_low"] > pr["ci_high"] - pr["ci_low"]).all() and len(res.plots) == 2


def test_missing_summaries(sav_labelled):
    ds = import_file(sav_labelled)
    bv = missing.by_variable(ds, ["score", "mood", "income"]).set_index("variable")
    assert bv.loc["score", "missing"] == 2 and bv.loc["mood", "user_missing_codes"] == bv.loc["mood", "missing"]
    pat = missing.patterns(ds, ["score", "mood"])
    assert pat["rows"].sum() == 60 and set(pat.columns) >= {"score", "mood", "rows", "percent"}
    un = missing.usable_n(ds, [("a", ["score"]), ("b", ["score", "mood"])])
    assert un.loc[0, "usable_n"] == 58 and un.loc[1, "difference_from_largest"] < 0
