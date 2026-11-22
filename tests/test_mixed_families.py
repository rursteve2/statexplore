"""v1.1 Phase 5 (§11, §20 I-K): lmerTest inference options, binary / ordinal / multinomial mixed models
and GEE. GLMM references are independent maximum-likelihood fits written here with numpy/scipy
(Gauss-Hermite quadrature of the random intercept), compared with R's adaptive quadrature (nAGQ = 25)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.optimize as so
import scipy.stats as st
from scipy.special import expit

from statexplore.analysis import measurement as M
from statexplore.analysis import mixed as MX
from statexplore.core.importers import import_file, import_sav

pytestmark = pytest.mark.r
GH_X, GH_W = np.polynomial.hermite.hermgauss(60)


def load(tmp_path, df, name="m.csv", nominal=(), ordinal=()):
    p = tmp_path / name
    df.to_csv(p, index=False)
    ds = import_file(p)
    ds.set_variable_field("id", "role", "id")
    for n in nominal:
        ds.set_variable_field(n, "measure", "nominal")
    for n in ordinal:
        ds.set_variable_field(n, "measure", "ordinal")
    return ds


def tab(res, prefix):
    t = next(t for t in res.tables if t.name.startswith(prefix))
    return pd.DataFrame(t.rows, columns=t.columns)


def run(r_session, ds, **kw):
    res, art = MX.run(r_session, ds, MX.MixedSpec(**kw))
    assert res.status.value in ("ok", "warning"), res.messages
    return res, art


# ---- binary GLMM ---------------------------------------------------------------------------------

def binary_data(seed=3, n=200, k=6, sigma=1.1):
    rng = np.random.default_rng(seed)
    idx = np.repeat(np.arange(n), k)
    x = rng.normal(size=n * k)
    g = np.repeat(rng.integers(0, 2, n), k)
    b = rng.normal(0, sigma, n)[idx]
    y = (rng.random(n * k) < expit(-0.4 + 0.8 * x + 0.6 * g + b)).astype(float)
    return pd.DataFrame({"id": idx + 1.0, "x": x, "g": g.astype(float), "y": y})


def ghq_logistic(df):
    X = np.column_stack([np.ones(len(df)), df.x, df.g])
    y = df.y.to_numpy()
    groups = [np.flatnonzero(df.id.to_numpy() == i) for i in np.unique(df.id)]

    def nll(theta):
        beta, sigma = theta[:3], np.exp(theta[3])
        eta = X @ beta
        total = 0.0
        for rows in groups:
            e = eta[rows][:, None] + np.sqrt(2) * sigma * GH_X[None, :]
            p = expit(e)
            lik = np.prod(np.where(y[rows][:, None] == 1, p, 1 - p), axis=0)
            total += np.log(max(np.sum(GH_W * lik) / np.sqrt(np.pi), 1e-300))
        return -total
    fit = so.minimize(nll, np.array([0, 0, 0, 0.0]), method="BFGS", options={"gtol": 1e-8})
    return fit.x[:3], np.exp(fit.x[3])


def test_J_glmer_matches_independent_ghq(tmp_path, r_session):
    df = binary_data()
    ds = load(tmp_path, df, nominal=("y",))
    res, art = run(r_session, ds, outcome="y", id="id", predictors=["x", "g"], interactions="none",
                   family="binomial", nagq=25, event=1.0)
    beta, sigma = ghq_logistic(df)
    fx = tab(res, "Fixed effects").set_index("term")
    assert fx.loc["(Intercept)", "log_odds"] == pytest.approx(beta[0], abs=2e-4)
    assert fx.loc["x", "log_odds"] == pytest.approx(beta[1], abs=2e-4)
    assert fx.loc["g=1", "log_odds"] == pytest.approx(beta[2], abs=2e-4)       # 0/1 imports as nominal
    assert fx.loc["x", "odds_ratio"] == pytest.approx(np.exp(fx.loc["x", "log_odds"]), rel=1e-12)
    z = st.norm.ppf(0.975)
    assert fx.loc["x", "or_ci_low"] == pytest.approx(np.exp(fx.loc["x", "log_odds"] - z * fx.loc["x", "se"]), rel=1e-10)
    vc = tab(res, "Random effects")
    assert vc["variance_or_cov"][0] == pytest.approx(sigma ** 2, rel=2e-3)
    assert "lme4::glmer" in res.script and "lmerTest::" not in res.script
    assert "subject-specific" in next(t for t in res.tables if t.name.startswith("Fixed")).notes
    assert "Wald z" in res.values["inference"] and "lmerTest does not apply" in res.values["inference"]
    assert res.plots, "predicted-probability plot"


def test_J_event_coding_and_predicted_probabilities(tmp_path, r_session):
    df = binary_data(seed=9)
    ds = load(tmp_path, df, nominal=("y",))
    a, _ = run(r_session, ds, outcome="y", id="id", predictors=["x"], family="binomial", event=1.0)
    b, _ = run(r_session, ds, outcome="y", id="id", predictors=["x"], family="binomial", event=0.0)
    fa, fb = tab(a, "Fixed effects").set_index("term"), tab(b, "Fixed effects").set_index("term")
    assert fb.loc["x", "log_odds"] == pytest.approx(-fa.loc["x", "log_odds"], rel=1e-4)
    assert "Modelled event: 1" in next(t for t in a.tables if t.name.startswith("Fixed")).notes
    pr = tab(a, "Predicted probabilities")
    sigma2 = tab(a, "Random effects")["variance_or_cov"][0]
    for _, r in pr.iterrows():
        eta = fa.loc["(Intercept)", "log_odds"] + fa.loc["x", "log_odds"] * r["x"]
        assert r["conditional"] == pytest.approx(expit(eta), rel=1e-8)
        pa = np.sum(GH_W * expit(eta + np.sqrt(2 * sigma2) * GH_X)) / np.sqrt(np.pi)
        assert r["population_average"] == pytest.approx(pa, abs=0.01)       # 4,000 simulated participants
    # population-average effects are attenuated towards 0.5 relative to conditional ones
    assert abs(pr["population_average"] - 0.5).max() < abs(pr["conditional"] - 0.5).max()


def test_J_singular_fit_flagged(tmp_path, r_session):
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"id": np.repeat(np.arange(80), 5) + 1.0, "x": rng.normal(size=400)})
    df["y"] = (rng.random(400) < expit(0.5 * df.x)).astype(float)        # no participant variance at all
    ds = load(tmp_path, df, nominal=("y",))
    res, _ = run(r_session, ds, outcome="y", id="id", predictors=["x"], family="binomial")
    assert any("Singular" in m for m in res.messages) or tab(res, "Random effects")["variance_or_cov"][0] < 0.05


# ---- ordinal GLMM ---------------------------------------------------------------------------------

def ordinal_mixed_data(seed=4, n=150, k=5):
    rng = np.random.default_rng(seed)
    idx = np.repeat(np.arange(n), k)
    x = rng.normal(size=n * k)
    latent = 0.9 * x + rng.normal(0, 0.9, n)[idx] + rng.logistic(size=n * k)
    y = np.digitize(latent, [-1.0, 0.5, 1.8]) + 1.0
    return pd.DataFrame({"id": idx + 1.0, "x": x, "y": y})


def ghq_cumlogit(df):
    x, y = df.x.to_numpy(), df.y.to_numpy().astype(int)
    groups = [np.flatnonzero(df.id.to_numpy() == i) for i in np.unique(df.id)]

    def nll(t):
        th = np.cumsum(np.r_[t[0], np.exp(t[1:3])])                # ordered thresholds
        beta, sigma = t[3], np.exp(t[4])
        cuts = np.r_[-np.inf, th, np.inf]
        total = 0.0
        for rows in groups:
            e = beta * x[rows][:, None] + np.sqrt(2) * sigma * GH_X[None, :]
            up, lo = cuts[y[rows]][:, None], cuts[y[rows] - 1][:, None]
            p = expit(up - e) - expit(lo - e)
            total += np.log(max(np.sum(GH_W * np.prod(p, axis=0)) / np.sqrt(np.pi), 1e-300))
        return -total
    fit = so.minimize(nll, np.array([-1, 0, 0, 0, 0.0]), method="BFGS", options={"gtol": 1e-8})
    t = fit.x
    return np.cumsum(np.r_[t[0], np.exp(t[1:3])]), t[3], np.exp(t[4])


def test_K_clmm_matches_independent_ghq(tmp_path, r_session):
    df = ordinal_mixed_data()
    ds = load(tmp_path, df, ordinal=("y",))
    res, _ = run(r_session, ds, outcome="y", id="id", predictors=["x"], family="ordinal", nagq=25)
    th, beta, sigma = ghq_cumlogit(df)
    fx = tab(res, "Fixed effects").set_index("term")
    assert fx.loc["x", "log_odds"] == pytest.approx(beta, abs=2e-4)
    assert tab(res, "Thresholds")["estimate"].tolist() == pytest.approx(list(th), abs=5e-4)
    assert tab(res, "Random effects")["variance"][0] == pytest.approx(sigma ** 2, rel=3e-3)
    note = next(t for t in res.tables if t.name.startswith("Fixed")).notes
    assert "1 < 2 < 3 < 4" in note and "higher categories" in note and "proportional" in note.lower()
    assert "Gaussian" not in res.title and "ordinal::clmm" in res.script


def test_K_ordinal_order_reversal_flips_sign(tmp_path, r_session):
    df = ordinal_mixed_data(seed=6, n=80)
    ds = load(tmp_path, df, ordinal=("y",))
    a, _ = run(r_session, ds, outcome="y", id="id", predictors=["x"], family="ordinal")
    ds.set_variable_field("y", "category_order", [4.0, 3.0, 2.0, 1.0])
    b, _ = run(r_session, ds, outcome="y", id="id", predictors=["x"], family="ordinal")
    assert tab(b, "Fixed effects")["log_odds"][0] == pytest.approx(-tab(a, "Fixed effects")["log_odds"][0], rel=1e-4)


def test_K_multinomial_mixed_recovers_parameters(tmp_path, r_session):
    rng = np.random.default_rng(12)
    n, k = 250, 12
    idx = np.repeat(np.arange(n), k)
    x = rng.normal(size=n * k)
    b = rng.normal(0, 0.5, n)[idx]
    eta = np.column_stack([np.zeros(n * k), 0.4 + 0.8 * x + b, -0.3 - 0.6 * x + b])
    P = np.exp(eta) / np.exp(eta).sum(1, keepdims=True)
    y = np.array([rng.choice(3, p=p) for p in P]) + 1.0
    ds = load(tmp_path, pd.DataFrame({"id": idx + 1.0, "x": x, "y": y}), nominal=("y",))
    res, _ = run(r_session, ds, outcome="y", id="id", predictors=["x"], family="multinomial")
    fx = tab(res, "Fixed effects")
    est = {(r.category, r.term): r.B for r in fx.itertuples()}
    assert est[("2", "x")] == pytest.approx(0.8, abs=0.12) and est[("3", "x")] == pytest.approx(-0.6, abs=0.12)
    assert "PQL" in res.values["inference"] and "underestimate" in res.values["inference"]
    assert tab(res, "Term tests (joint Wald)")["df"][0] == 2
    assert "mclogit::mblogit" in res.script


def test_K_unsupported_specifications_rejected(tmp_path):
    ds = load(tmp_path, ordinal_mixed_data(n=20), ordinal=("y",))
    bad = [
        dict(family="ordinal", residual="ar1", time="x"),
        dict(family="multinomial", random="slope", time="x"),
        dict(family="ordinal", random="slope", time="x", nagq=10),
        dict(family="ordinal", inference="kenward_roger"),
        dict(family="ordinal", estimator="gee"),
        dict(family="gaussian"),                                        # ordinal outcome, no treatment
    ]
    for kw in bad:
        with pytest.raises(MX.MixedInputError):
            MX.validate(ds, MX.MixedSpec(outcome="y", id="id", predictors=[], **kw))


# ---- GEE ---------------------------------------------------------------------------------------------

def test_gee_matches_statsmodels(tmp_path, r_session):
    import statsmodels.api as sm

    df = binary_data(seed=21, n=150, k=5)
    ds = load(tmp_path, df, nominal=("y",))
    res, _ = run(r_session, ds, outcome="y", id="id", predictors=["x", "g"], interactions="none", family="binomial",
                 estimator="gee", gee_corstr="exchangeable", random="none", event=1.0)
    ref = sm.GEE(df.y, sm.add_constant(df[["x", "g"]]), groups=df.id, family=sm.families.Binomial(),
                 cov_struct=sm.cov_struct.Exchangeable()).fit()
    fx = tab(res, "Coefficients").set_index("term")
    for r_term, p_term in (("(Intercept)", "const"), ("x", "x"), ("g=1", "g")):
        assert fx.loc[r_term, "estimate"] == pytest.approx(ref.params[p_term], abs=2e-3)
        assert fx.loc[r_term, "robust_se"] == pytest.approx(ref.bse[p_term], rel=2e-2)
    assert "population" in next(t for t in res.tables if t.name.startswith("Coefficients")).notes
    assert "not interchangeable" in next(t for t in res.tables if t.name.startswith("Coefficients")).notes


# ---- lmerTest inference (§11.3, Test I) ----------------------------------------------------------------

def balanced_gaussian(seed=2, n=30, k=4):
    rng = np.random.default_rng(seed)
    idx = np.repeat(np.arange(n), k)
    t = np.tile(np.arange(k, dtype=float), n)
    g = np.repeat(np.arange(n) % 2, k).astype(float)
    y = 1 + 0.5 * t + 0.8 * g + 0.3 * t * g + rng.normal(0, 1, n)[idx] + rng.normal(size=n * k)
    return pd.DataFrame({"id": idx + 1.0, "t": t, "g": g, "y": y})


def test_I_satterthwaite_and_kr_balanced_exact_df(tmp_path, r_session):
    df = balanced_gaussian()
    ds = load(tmp_path, df, nominal=("g",))
    out = {}
    for inf in ("satterthwaite", "kenward_roger"):
        res, _ = run(r_session, ds, outcome="y", id="id", time="t", predictors=["g"], interactions="none", inference=inf)
        terms = tab(res, "Fixed-effect term tests").set_index("term")
        # balanced random-intercept design: between-participant effect df = n - 2, within = n(k-1) - 1
        assert terms.loc["g", "df2"] == pytest.approx(28, abs=1e-4)
        assert terms.loc["t", "df2"] == pytest.approx(30 * 3 - 1, abs=1e-4)
        out[inf] = terms
        assert ("Kenward-Roger" if inf == "kenward_roger" else "Satterthwaite") in res.values["inference"]
        assert "lmerTest::lmer" in res.script
    assert out["kenward_roger"].loc["g", "F"] == pytest.approx(out["satterthwaite"].loc["g", "F"], rel=1e-6)


def test_I_kr_matches_pbkrtest_directly(tmp_path, r_session, r_worker):
    df = balanced_gaussian(seed=5)
    df = df.drop(index=[3, 10, 17, 50]).reset_index(drop=True)         # unbalanced: KR != Satterthwaite
    ds = load(tmp_path, df, nominal=("g",))
    res, _ = run(r_session, ds, outcome="y", id="id", time="t", predictors=["g"], interactions="none",
                 inference="kenward_roger")
    g = tab(res, "Fixed-effect term tests").set_index("term").loc["g"]
    p = (tmp_path / "kr.csv").as_posix()
    df.to_csv(tmp_path / "kr.csv", index=False)
    ref = r_worker.run(
        f"d <- read.csv('{p}'); d$id <- factor(d$id); d$g <- factor(d$g)\n"
        "m1 <- lme4::lmer(y ~ t + g + (1 | id), d, REML = TRUE); m0 <- lme4::lmer(y ~ t + (1 | id), d, REML = TRUE)\n"
        "k <- pbkrtest::KRmodcomp(m1, m0)$test\n"
        "sx_result <- list(F = k['Ftest', 'stat'], ddf = k['Ftest', 'ddf'])", packages=["pbkrtest"]).result
    assert g["F"] == pytest.approx(ref["F"], rel=1e-6) and g["df2"] == pytest.approx(ref["ddf"], rel=1e-6)


def test_I_lrt_option_matches_ml_refits(tmp_path, r_session):
    import statsmodels.formula.api as smf

    df = balanced_gaussian(seed=8)
    ds = load(tmp_path, df, nominal=("g",))
    res, _ = run(r_session, ds, outcome="y", id="id", time="t", predictors=["g"], interactions="time", inference="lrt")
    t = tab(res, "Fixed-effect term tests").set_index("term")
    full = smf.mixedlm("y ~ t * C(g)", df, groups=df.id).fit(reml=False)
    red = smf.mixedlm("y ~ t + C(g)", df, groups=df.id).fit(reml=False)
    assert t.iloc[0]["LR_chi2"] == pytest.approx(2 * (full.llf - red.llf), rel=1e-4)
    assert list(t.index) == ["t:g"]                                     # marginality: only the interaction
    assert "not Type III" in res.values["inference"]


# ---- HRS: R_LTACTX becomes usable in mixed models (Test B) -------------------------------------------------

def test_B_ltactx_mixed_models_selectable(hrs_path):
    ds = import_sav(hrs_path)
    ds.set_variable_field("HHIDPN", "role", "id")
    base = dict(outcome="R_LTACTX", id="HHIDPN", time="WAVE", predictors=["RAGENDER"])
    with pytest.raises(MX.MixedInputError, match="Treat R_LTACTX as continuous"):
        MX.validate(ds, MX.MixedSpec(**base))
    tr = [M.VariableTreatment("R_LTACTX", "continuous", acknowledged=True, numeric_confirmed=True).to_dict()]
    MX.validate(ds, MX.MixedSpec(**base, treatments=tr))
    MX.validate(ds, MX.MixedSpec(**base, family="multinomial"))
    M.approve(ds, "R_LTACTX", "ordinal")
    MX.validate(ds, MX.MixedSpec(**base, family="ordinal"))
    assert ds.var("R_LTACTX").source_measure == "nominal"
