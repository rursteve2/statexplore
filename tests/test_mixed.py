"""Phase 5 acceptance: tests F (random intercept), G (random slope), H (time x group),
I (AR(1)), unstructured covariance, singular fits, structure detection, reshaping."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import multivariate_normal

from statexplore.analysis import mixed
from statexplore.core.importers import import_file
from statexplore.core.schema import Status
from statexplore.core.structure import ReshapeSpec, suggest_structure, wide_to_long


def load(tmp_path, df, name="m.csv"):
    p = tmp_path / name
    df.to_csv(p, index=False)
    ds = import_file(p)
    return ds


def tab(res, name):
    t = next(t for t in res.tables if t.name.startswith(name))
    return pd.DataFrame(t.rows, columns=t.columns)


def panel(rng, n=60, times=(0, 1, 2, 3, 4), slope_sd=0.3, int_sd=1.0, phi=0.0, resid_sd=0.7):
    rows = []
    for i in range(n):
        g = "b" if i % 2 else "a"
        u0, u1 = rng.normal(0, int_sd), rng.normal(0, slope_sd)
        e = np.zeros(len(times))
        e[0] = rng.normal(0, resid_sd)
        for k in range(1, len(times)):
            e[k] = phi * e[k - 1] + rng.normal(0, resid_sd * np.sqrt(1 - phi ** 2))
        for k, t in enumerate(times):
            y = 1 + u0 + (0.5 + u1) * t + 0.4 * t * (g == "b") + e[k]
            rows.append({"id": i + 1, "time": t, "g": g, "y": y})
    return pd.DataFrame(rows)


@pytest.fixture
def rng():
    return np.random.default_rng(31)


@pytest.mark.r
def test_F_random_intercept_matches_statsmodels(tmp_path, rng, r_session):
    import statsmodels.formula.api as smf

    df = panel(rng, slope_sd=0.0)
    ds = load(tmp_path, df)
    s = mixed.MixedSpec(outcome="y", id="id", time="time", predictors=["g"], interactions="time")
    res, art = mixed.run(r_session, ds, s)
    assert res.status is Status.OK, res.messages
    ref = smf.mixedlm("y ~ time * g", df, groups=df["id"]).fit(reml=True)
    fx = tab(res, "Fixed effects").set_index("term")
    for r_t, p_t in [("(Intercept)", "Intercept"), ("time", "time"), ("g=b", "g[T.b]"), ("time:g=b", "time:g[T.b]")]:
        assert fx.loc[r_t, "estimate"] == pytest.approx(ref.fe_params[p_t], rel=1e-4)
    vc = tab(res, "Random effects")
    assert vc.loc[vc["term"] == "(Intercept)", "variance_or_cov"].iloc[0] == pytest.approx(ref.cov_re.iloc[0, 0], rel=1e-3)
    assert vc.loc[vc["term"] == "(residual)", "variance_or_cov"].iloc[0] == pytest.approx(ref.scale, rel=1e-3)
    fit = tab(res, "Model fit").iloc[0]
    assert fit["participants"] == 60 and fit["observations"] == 300
    assert len(res.plots) == 5 and all(art[k][:4] == b"\x89PNG" for k in res.plots)  # incl. random effects + ACF


@pytest.mark.r
def test_G_random_slope_matches_statsmodels(tmp_path, rng, r_session):
    import statsmodels.formula.api as smf

    df = panel(rng, n=120, slope_sd=0.5)
    ds = load(tmp_path, df)
    s = mixed.MixedSpec(outcome="y", id="id", time="time", predictors=["g"], random="slope")
    res, _ = mixed.run(r_session, ds, s)
    ref = smf.mixedlm("y ~ time * g", df, groups=df["id"], re_formula="~time").fit(reml=True, method="lbfgs")
    vc = tab(res, "Random effects")
    get = lambda term, w="": vc[(vc["term"] == term) & (vc["with"] == w)]["variance_or_cov"].iloc[0]  # noqa: E731
    assert get("(Intercept)") == pytest.approx(ref.cov_re.iloc[0, 0], rel=5e-3)
    assert get("time") == pytest.approx(ref.cov_re.iloc[1, 1], rel=5e-3)
    assert get("(Intercept)", "time") == pytest.approx(ref.cov_re.iloc[0, 1], rel=2e-2, abs=2e-3)
    fit = tab(res, "Model fit").iloc[0]
    assert 0 < fit["R2_marginal"] < fit["R2_conditional"] < 1


@pytest.mark.r
def test_H_time_by_group_trajectory(tmp_path, rng, r_session):
    df = panel(rng)
    df.loc[df.sample(30, random_state=1).index, "y"] = np.nan  # missing occasions: participants are kept
    ds = load(tmp_path, df)
    s = mixed.MixedSpec(outcome="y", id="id", time="time", time_type="categorical", predictors=["g"])
    res, _ = mixed.run(r_session, ds, s)
    assert res.n_used == 270 and tab(res, "Model fit").iloc[0]["participants"] == 60
    fx = tab(res, "Fixed effects").set_index("term")
    traj = tab(res, "Population-level trajectory")
    # trajectory = fixed-effect predictions: check one cell by hand (time 3, group b)
    row = traj[(traj["time"] == "3") & (traj["g"] == "b")].iloc[0]
    expected = fx.loc["(Intercept)", "estimate"] + fx.loc["time3", "estimate"] + fx.loc["g=b", "estimate"] \
        + fx.loc["time3:g=b", "estimate"]
    assert row["predicted"] == pytest.approx(expected, rel=1e-10)
    terms = tab(res, "Fixed-effect term tests").set_index("term")
    assert terms.loc["time:g", "p"] < 0.01  # planted time x group interaction


def _ar1_loglik(df, beta_fn, tau2, sigma2, phi):
    ll = 0.0
    for _, sub in df.groupby("id"):
        idx = sub["tidx"].to_numpy()
        R = phi ** np.abs(idx[:, None] - idx[None, :])
        V = tau2 * np.ones((len(idx), len(idx))) + sigma2 * R
        ll += multivariate_normal(mean=beta_fn(sub), cov=V).logpdf(sub["y"].to_numpy())
    return ll


@pytest.mark.r
def test_I_ar1_likelihood_matches_independent_computation(tmp_path, rng, r_session):
    df = panel(rng, n=80, slope_sd=0.0, phi=0.6)
    df = df.drop(df.sample(25, random_state=3).index)  # unbalanced: AR(1) must use occasion order
    ds = load(tmp_path, df)
    s = mixed.MixedSpec(outcome="y", id="id", time="time", predictors=["g"], residual="ar1", method="ML")
    res, _ = mixed.run(r_session, ds, s)
    assert res.status is Status.OK, res.messages
    phi = tab(res, "Residual autocorrelation").iloc[0]["phi"]
    assert 0.4 < phi < 0.8
    fx = tab(res, "Fixed effects").set_index("term")["estimate"]
    vc = tab(res, "Random effects")
    # v1.3.2 (STAT-43): one row per variance / covariance of every grouping level, residual as group "Residual"
    tau2 = float(vc.loc[vc["term"] == "(Intercept)", "variance_or_cov"].iloc[0])
    sigma2 = float(vc.loc[vc["group"] == "Residual", "variance_or_cov"].iloc[0])
    df = df.assign(tidx=df["time"].rank(method="dense").astype(int))
    beta = lambda sub: (fx["(Intercept)"] + fx["time"] * sub["time"] + fx["g=b"] * (sub["g"] == "b")  # noqa: E731
                        + fx["time:g=b"] * sub["time"] * (sub["g"] == "b")).to_numpy()
    ll = _ar1_loglik(df, beta, tau2, sigma2, phi)
    assert tab(res, "Model fit").iloc[0]["logLik"] == pytest.approx(ll, rel=1e-6)


@pytest.mark.r
def test_unstructured_marginal_likelihood(tmp_path, rng, r_session):
    df = panel(rng, n=70, times=(0, 1, 2, 3), slope_sd=0.4)
    ds = load(tmp_path, df)
    s = mixed.MixedSpec(outcome="y", id="id", time="time", time_type="categorical", predictors=["g"],
                        random="none", residual="unstructured", method="ML")
    res, _ = mixed.run(r_session, ds, s)
    assert res.status is Status.OK, res.messages and res.values["engine_choice"] == "mmrm"
    cov = tab(res, "Residual covariance").set_index("occasion").to_numpy(float)
    fx = tab(res, "Fixed effects").set_index("term")["estimate"]
    ll = 0.0
    for _, sub in df.groupby("id"):
        mu = np.array([fx["(Intercept)"] + (fx.get(f"time{t}", 0.0) if t else 0.0)
                       + (fx["g=b"] + (fx.get(f"time{t}:g=b", 0.0) if t else 0.0)) * (g == "b")
                       for t, g in zip(sub["time"], sub["g"])])
        ll += multivariate_normal(mean=mu, cov=cov).logpdf(sub["y"].to_numpy())
    assert tab(res, "Model fit").iloc[0]["logLik"] == pytest.approx(ll, rel=1e-6)


@pytest.mark.r
def test_singular_fit_flagged(tmp_path, rng, r_session):
    # every participant has the identical response profile -> between-participant variance is exactly 0
    profile = 1 + 0.5 * np.arange(5) + rng.normal(0, 0.7, 5)
    df = pd.DataFrame({"id": np.repeat(np.arange(1, 41), 5), "time": np.tile(np.arange(5), 40),
                       "y": np.tile(profile, 40)})
    ds = load(tmp_path, df)
    res, _ = mixed.run(r_session, ds, mixed.MixedSpec(outcome="y", id="id", time="time", random="slope"))
    assert res.status is Status.WARNING and any("Singular" in m for m in res.messages)


def test_spec_validation(tmp_path, rng):
    ds = load(tmp_path, panel(rng, n=6))
    bad = [
        mixed.MixedSpec(outcome="y", id="id", time="time", residual="unstructured"),           # random effects
        mixed.MixedSpec(outcome="y", id="id", time="time", residual="unstructured", random="none"),  # continuous time
        mixed.MixedSpec(outcome="y", id="id", random="slope"),                                 # slope without time
        mixed.MixedSpec(outcome="g", id="id"),                                                 # categorical outcome
        mixed.MixedSpec(outcome="y", id="id", time="time", random="none"),
        mixed.MixedSpec(outcome="y", id="id", predictors=["id"]),
    ]
    for s in bad:
        with pytest.raises(mixed.MixedInputError):
            mixed.validate(ds, s)


@pytest.mark.r
def test_ar1_duplicate_occasions_rejected(tmp_path, rng, r_session):
    df = panel(rng, n=10)
    df = pd.concat([df, df.head(3)])
    ds = load(tmp_path, df)
    res, _ = mixed.run(r_session, ds, mixed.MixedSpec(outcome="y", id="id", time="time", residual="ar1"))
    assert res.status is Status.FAILED and "one row per participant per occasion" in res.messages[0]


def test_structure_detection_hrs(hrs_path):
    ds = import_file(hrs_path)
    sugg = suggest_structure(ds)
    ids = [s.variable for s in sugg if s.role == "id"]
    times = [s.variable for s in sugg if s.role == "time"]
    clusters = [s.variable for s in sugg if s.role == "cluster"]
    assert ids[0] == "HHIDPN"
    assert "WAVE" in times[:2]
    assert "HHID" in clusters


def test_wide_to_long(tmp_path):
    wide = pd.DataFrame({"pid": [1, 2, 3], "sex": ["m", "f", "f"], "dep1": [1.0, 2, None], "dep2": [2.0, 3, 4],
                         "dep3": [3.0, None, 5]})
    ds = load(tmp_path, wide, "w.csv")
    long = wide_to_long(ds, ReshapeSpec(id_var="pid", measures={"dep": ["dep1", "dep2", "dep3"]},
                                        time_values=[1, 2, 3], time_name="wave"))
    assert long.n_rows == 9 and list(long.raw.columns) == ["pid", "wave", "dep", "sex"]
    assert long.raw["dep"].tolist() == [1, 2, 3, 2, 3, None, None, 4, 5] or \
        np.allclose(long.raw["dep"].to_numpy(float), [1, 2, 3, 2, 3, np.nan, np.nan, 4, 5], equal_nan=True)
    assert long.var("pid").role.value == "id" and long.var("wave").role.value == "time"
    assert long.source.options["reshape"]["measures"] == {"dep": ["dep1", "dep2", "dep3"]}
    assert ds.n_rows == 3  # source dataset untouched


@pytest.mark.r
def test_specification_comparisons(tmp_path, rng, r_session):
    import statsmodels.formula.api as smf
    from scipy.stats import chi2

    df = panel(rng, n=80, slope_sd=0.4)
    ds = load(tmp_path, df)
    base = mixed.MixedSpec(outcome="y", id="id", time="time", predictors=["g"], interactions="none")
    res, _ = mixed.compare_specs(r_session, ds, base, "time")
    assert res.status is Status.OK, res.messages
    comp = tab(res, "Specification comparison")
    assert set(comp["estimation"]) == {"ML"}  # fixed-effect comparisons are refit with ML
    lrt = tab(res, "Likelihood-ratio tests")
    lin = smf.mixedlm("y ~ time + g", df, groups=df["id"]).fit(reml=False)
    quad = smf.mixedlm("y ~ time + I(time**2) + g", df, groups=df["id"]).fit(reml=False)
    row = lrt[(lrt["smaller"].str.startswith("Continuous (linear")) & (lrt["larger"].str.startswith("Continuous (quad"))].iloc[0]
    assert row["LR"] == pytest.approx(2 * (quad.llf - lin.llf), rel=1e-3, abs=1e-4)

    res, _ = mixed.compare_specs(r_session, ds, base, "random")
    # v1.3.2 (UI-54, STAT-23): RI, uncorrelated and correlated slopes; the reference follows each hypothesis
    lrts = tab(res, "Likelihood-ratio tests")
    lrt = lrts[lrts["null_hypothesis"].str.startswith("participant time-slope variance = 0 and")].iloc[0]
    assert set(tab(res, "Specification comparison")["estimation"]) == {"REML"}
    assert lrt["df"] == 2
    assert lrt["p"] == pytest.approx(0.5 * chi2.sf(lrt["LR"], 2) + 0.5 * chi2.sf(lrt["LR"], 1), rel=1e-9)

    res, _ = mixed.compare_specs(r_session, ds, base, "residual")
    comp = tab(res, "Specification comparison")
    lrts = tab(res, "Likelihood-ratio tests")      # 2 nested tests + AR(1) vs CAR(1) reported as not nested
    assert list(comp["engine"]) == ["nlme"] * 3 and (lrts["status"] == "tested").sum() == 2

    with pytest.raises(mixed.MixedInputError):
        mixed.comparison_variants(mixed.MixedSpec(outcome="y", id="id", time="time", time_type="categorical",
                                                  random="none", residual="unstructured"), "random")


@pytest.mark.r
def test_labelled_time_uses_numeric_codes(tmp_path, rng, r_session):
    """A time variable stored as labelled codes (e.g. SPSS 'Wave 3' = 3) must behave like the numbers."""
    df = panel(rng, n=40, times=(2, 4, 6, 8))
    plain = load(tmp_path, df, "plain.csv")
    labelled = load(tmp_path, df, "lab.csv")
    labelled.set_variable_field("time", "measure", "ordinal")
    labelled.var("time").value_labels = {2.0: "Wave 2", 4.0: "Wave 4", 6.0: "Wave 6", 8.0: "Wave 8"}
    s = mixed.MixedSpec(outcome="y", id="id", time="time", predictors=["g"], random="slope", residual="car1")
    a, _ = mixed.run(r_session, plain, s)
    b, _ = mixed.run(r_session, labelled, s)
    assert a.status is Status.OK and b.status is Status.OK, (a.messages, b.messages)
    np.testing.assert_allclose(tab(a, "Fixed effects")["estimate"], tab(b, "Fixed effects")["estimate"], rtol=1e-8)
