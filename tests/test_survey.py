"""v1.1 Phase 6 (§13, §20 M): survey-weighted analysis vs closed-form design-based estimators
(Taylor linearization, with-replacement PSUs, stratified)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statexplore.analysis import survey as S
from statexplore.core.importers import import_file, import_sav

pytestmark = pytest.mark.r


@pytest.fixture
def design_data(tmp_path):
    rng = np.random.default_rng(17)
    rows = []
    for h in range(1, 6):                       # 5 strata
        for c in range(1, 5):                   # 4 PSUs per stratum
            u = rng.normal(0, 0.7)
            for _ in range(rng.integers(8, 15)):
                x = rng.normal()
                g = rng.integers(0, 2)
                rows.append((h, h * 10 + c, rng.uniform(0.5, 4.0), x, 1 + 0.5 * x + 0.3 * h + u + rng.normal(),
                             g, rng.choice(["a", "b", "c"], p=[0.5, 0.3, 0.2])))
    df = pd.DataFrame(rows, columns=["stratum", "psu", "w", "x", "y", "g", "cat"])
    p = tmp_path / "svy.csv"
    df.to_csv(p, index=False)
    ds = import_file(p)
    ds.set_variable_field("g", "measure", "nominal")
    return ds, df


def lin_var(df, z):
    """Stratified with-replacement Taylor variance of a total of z (z already = linearized scores)."""
    v = 0.0
    for h, part in df.assign(z=z).groupby("stratum"):
        tot = part.groupby("psu")["z"].sum()
        n = len(tot)
        v += n / (n - 1) * ((tot - tot.mean()) ** 2).sum()
    return v


def tab(res, prefix):
    t = next(t for t in res.tables if t.name.startswith(prefix))
    return pd.DataFrame(t.rows, columns=t.columns, index=t.row_names or None)


D = S.SurveyDesign(weight="w", psu="psu", strata="stratum")


def test_M_weighted_mean_and_se(design_data, r_session):
    ds, df = design_data
    res, _ = S.run(r_session, ds, D, "mean", ["y", "x"])
    m = tab(res, "Weighted means").set_index("variable")
    W = df.w.sum()
    ybar = (df.w * df.y).sum() / W
    se = np.sqrt(lin_var(df, df.w * (df.y - ybar) / W))
    assert m.loc["y", "mean"] == pytest.approx(ybar, rel=1e-12)
    assert m.loc["y", "se"] == pytest.approx(se, rel=1e-10)
    info = tab(res, "Survey design").iloc[0]
    assert info["rows_analysed"] == len(df) and info["PSUs"] == 20 and info["strata"] == 5
    assert info["design_df"] == 20 - 5 and info["sum_of_weights"] == pytest.approx(W, rel=1e-12)


def test_M_weighted_proportions(design_data, r_session):
    ds, df = design_data
    res, _ = S.run(r_session, ds, D, "proportion", ["cat"])
    p = tab(res, "Weighted proportions").set_index("category")
    W = df.w.sum()
    for c in ("a", "b", "c"):
        ind = (df.cat == c).astype(float)
        est = (df.w * ind).sum() / W
        assert p.loc[c, "proportion"] == pytest.approx(est, rel=1e-12)
        assert p.loc[c, "se"] == pytest.approx(np.sqrt(lin_var(df, df.w * (ind - est) / W)), rel=1e-10)
        assert p.loc[c, "raw_n"] == (df.cat == c).sum()
        assert p.loc[c, "weighted_count"] == pytest.approx((df.w * ind).sum(), rel=1e-12)


def test_M_survey_regression_matches_closed_form(design_data, r_session):
    ds, df = design_data
    res, _ = S.run(r_session, ds, D, "linear", ["x"], outcome="y")
    co = tab(res, "Survey-weighted coefficients").set_index("term")
    X = np.column_stack([np.ones(len(df)), df.x])
    Wd = df.w.to_numpy()
    XtWX = X.T @ (X * Wd[:, None])
    beta = np.linalg.solve(XtWX, X.T @ (Wd * df.y))
    e = df.y.to_numpy() - X @ beta
    U = X * (Wd * e)[:, None]
    meat = np.zeros((2, 2))
    for h, idx in df.groupby("stratum").groups.items():
        part = df.loc[idx]
        tot = pd.DataFrame(U[idx], index=idx).groupby(part.psu).sum().to_numpy()
        n = len(tot)
        dev = tot - tot.mean(0)
        meat += n / (n - 1) * dev.T @ dev
    bread = np.linalg.inv(XtWX)
    se = np.sqrt(np.diag(bread @ meat @ bread))
    assert co.loc["(Intercept)", "estimate"] == pytest.approx(beta[0], rel=1e-10)
    assert co.loc["x", "estimate"] == pytest.approx(beta[1], rel=1e-10)
    assert co.loc["x", "se"] == pytest.approx(se[1], rel=1e-8)
    assert "not a substitute for modelling repeated measurements" in next(
        t for t in res.tables if t.name.startswith("Survey-weighted")).notes


def test_M_crosstab_and_logistic_run(design_data, r_session):
    ds, df = design_data
    res, _ = S.run(r_session, ds, D, "crosstab", ["cat", "g"])
    raw = tab(res, "Raw (unweighted) counts")
    assert raw.to_numpy().sum() == len(df)
    wc = tab(res, "Weighted counts")
    assert wc.to_numpy().sum() == pytest.approx(df.w.sum(), rel=1e-12)
    assert 0 <= tab(res, "Design-adjusted")["p"][0] <= 1
    res2, _ = S.run(r_session, ds, D, "logistic", ["x"], outcome="g")
    assert "odds_ratio" in tab(res2, "Survey-weighted coefficients").columns


def test_M_invalid_weights_and_design_checks(tmp_path, design_data):
    ds, df = design_data
    df2 = df.copy()
    df2.loc[0, "w"] = -1.0
    p = tmp_path / "neg.csv"
    df2.to_csv(p, index=False)
    with pytest.raises(S.SurveyInputError, match="negative"):
        S.check_design(import_file(p), D)
    df2.loc[0, "w"] = np.nan
    df2.loc[1, "w"] = 0.0
    df2.to_csv(p, index=False)
    warn = S.check_design(import_file(p), D)
    assert any("missing weight" in w for w in warn) and any("weight 0" in w for w in warn)
    with pytest.raises(S.SurveyInputError, match="at least"):
        S.check_design(ds, S.SurveyDesign())
    # a stratum with one PSU needs an explicit policy
    df3 = df[~((df.stratum == 1) & (df.psu != 11))]
    df3.to_csv(p, index=False)
    with pytest.raises(S.SurveyInputError, match="single PSU"):
        S.check_design(import_file(p), D)
    assert S.check_design(import_file(p), S.SurveyDesign("w", "psu", "stratum", lonely_psu="adjust"))


def test_M_hrs_has_no_design_variables(hrs_path):
    ds = import_sav(hrs_path)
    assert S.suggest_design(ds) == {"weight": [], "psu": [], "strata": []}   # R_WEIGHT is body weight
    text = " ".join(S.explain_design(ds, S.SurveyDesign()))
    assert "body weight is not a survey weight" in text
    ds.set_variable_field("HHIDPN", "role", "id")
    with pytest.raises(S.SurveyInputError, match="cross-sectional weight"):
        S.check_design(ds, S.SurveyDesign(weight="R_WEIGHT"))
