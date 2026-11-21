"""Phase 2 acceptance: every analysis checked against an independent Python reference
(scipy / statsmodels / closed-form). Tolerances: 1e-6 relative unless noted."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import scipy.stats as st

from statexplore.analysis.builders import BY_KEY
from statexplore.analysis.framework import AnalysisInputError, run_analysis
from statexplore.core.importers import import_file
from statexplore.core.schema import Status

pytestmark = pytest.mark.r
REL = 1e-6


def load(tmp_path, df: pd.DataFrame, name="d.csv"):
    p = tmp_path / name
    df.to_csv(p, index=False)
    return import_file(p)


def run(r_session, ds, key, opts=None, **vars_):
    vs = {k: ([v] if isinstance(v, str) else list(v)) for k, v in vars_.items()}
    res, art = run_analysis(r_session, ds, BY_KEY[key], vs, opts or {})
    assert res.status in (Status.OK, Status.WARNING), res.messages
    return res, art


def tab(res, name_start: str) -> pd.DataFrame:
    for t in res.tables:
        if t.name.startswith(name_start):
            return pd.DataFrame(t.rows, columns=t.columns)
    raise AssertionError(f"no table {name_start!r} in {[t.name for t in res.tables]}")


def hedges_j(df: float) -> float:
    return math.exp(math.lgamma(df / 2) - math.log(math.sqrt(df / 2)) - math.lgamma((df - 1) / 2))


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# ---- Test A: t-tests ---------------------------------------------------------

def test_A_independent_and_welch(r_session, tmp_path, rng):
    a, b = rng.normal(10, 2, 40), rng.normal(11.5, 3, 55)
    ds = load(tmp_path, pd.DataFrame({"y": np.r_[a, b], "g": ["a"] * 40 + ["b"] * 55}))
    res, _ = run(r_session, ds, "t_independent", dependent="y", group="g")
    t = tab(res, "Independent").iloc[0]
    ref = st.ttest_ind(a, b, equal_var=True)
    assert t["t"] == pytest.approx(ref.statistic, rel=REL) and t["p"] == pytest.approx(ref.pvalue, rel=REL)
    assert t["difference"] == pytest.approx(a.mean() - b.mean(), rel=REL)
    sp = math.sqrt(((40 - 1) * a.var(ddof=1) + (55 - 1) * b.var(ddof=1)) / (95 - 2))
    es = tab(res, "Effect size")
    assert es.loc[0, "estimate"] == pytest.approx((a.mean() - b.mean()) / sp, rel=REL)
    assert es.loc[1, "estimate"] == pytest.approx((a.mean() - b.mean()) / sp * hedges_j(93), rel=1e-4)
    lev = tab(res, "Levene").iloc[0]
    ref_lev = st.levene(a, b, center="median")
    assert lev["F"] == pytest.approx(ref_lev.statistic, rel=REL) and lev["p"] == pytest.approx(ref_lev.pvalue, rel=REL)
    assert res.n_used == 95

    res, _ = run(r_session, ds, "t_welch", dependent="y", group="g")
    t = tab(res, "Welch").iloc[0]
    ref = st.ttest_ind(a, b, equal_var=False)
    assert t["t"] == pytest.approx(ref.statistic, rel=REL) and t["p"] == pytest.approx(ref.pvalue, rel=REL)
    assert t["df"] == pytest.approx(ref.df, rel=REL)


def test_one_sample_and_paired(r_session, tmp_path, rng):
    x1 = rng.normal(5, 1, 30)
    x2 = x1 + rng.normal(0.4, 0.8, 30)
    ds = load(tmp_path, pd.DataFrame({"x1": x1, "x2": x2}))
    res, _ = run(r_session, ds, "t_one_sample", {"mu": 4.5}, dependent="x1")
    t = tab(res, "One-sample").iloc[0]
    ref = st.ttest_1samp(x1, 4.5)
    assert t["t"] == pytest.approx(ref.statistic, rel=REL) and t["p"] == pytest.approx(ref.pvalue, rel=REL)
    assert tab(res, "Effect size").loc[0, "estimate"] == pytest.approx((x1.mean() - 4.5) / x1.std(ddof=1), rel=REL)

    res, _ = run(r_session, ds, "t_paired", first="x1", second="x2")
    t = tab(res, "Paired").iloc[0]
    ref = st.ttest_rel(x1, x2)
    assert t["t"] == pytest.approx(ref.statistic, rel=REL) and t["p"] == pytest.approx(ref.pvalue, rel=REL)
    d = x1 - x2
    assert tab(res, "Effect size").loc[0, "estimate"] == pytest.approx(d.mean() / d.std(ddof=1), rel=REL)


def test_t_test_rejects_three_groups(r_session, tmp_path, rng):
    ds = load(tmp_path, pd.DataFrame({"y": rng.normal(size=30), "g": ["a", "b", "c"] * 10}))
    res, _ = run_analysis(r_session, ds, BY_KEY["t_independent"], {"dependent": ["y"], "group": ["g"]}, {})
    assert res.status is Status.FAILED and "exactly 2" in res.messages[0]


# ---- correlations --------------------------------------------------------------

def test_correlations(r_session, tmp_path, rng):
    n = 80
    z = rng.normal(size=n)
    x = z + rng.normal(size=n)
    y = 0.5 * x + z + rng.normal(size=n)
    w = np.round(rng.normal(size=n), 0)  # ties
    ds = load(tmp_path, pd.DataFrame({"x": x, "y": y, "z": z, "w": w}))
    res, _ = run(r_session, ds, "cor_pair", x="x", y="y")
    t = tab(res, "Pearson").iloc[0]
    ref = st.pearsonr(x, y)
    ci = ref.confidence_interval(0.95)
    assert t["r"] == pytest.approx(ref.statistic, rel=REL) and t["p"] == pytest.approx(ref.pvalue, rel=1e-5)
    assert (t["ci_low"], t["ci_high"]) == (pytest.approx(ci.low, rel=1e-6), pytest.approx(ci.high, rel=1e-6))

    res, _ = run(r_session, ds, "cor_pair", {"method": "spearman"}, x="x", y="w")
    t = tab(res, "Spearman").iloc[0]
    ref = st.spearmanr(x, w)
    assert t["r"] == pytest.approx(ref.statistic, rel=REL) and t["p"] == pytest.approx(ref.pvalue, rel=1e-4)

    res, _ = run(r_session, ds, "cor_partial", x="x", y="y", controls=["z"])
    t = tab(res, "Partial").iloc[0]
    zz = np.c_[np.ones(n), z]
    rx = x - zz @ np.linalg.lstsq(zz, x, rcond=None)[0]
    ry = y - zz @ np.linalg.lstsq(zz, y, rcond=None)[0]
    r = np.corrcoef(rx, ry)[0, 1]
    tval = r * math.sqrt((n - 3) / (1 - r * r))
    assert t["r_partial"] == pytest.approx(r, rel=REL)
    assert t["p"] == pytest.approx(2 * st.t.sf(abs(tval), n - 3), rel=1e-5)


def test_correlation_matrix_raw_and_adjusted(r_session, tmp_path, rng):
    from statsmodels.stats.multitest import multipletests

    df = pd.DataFrame(rng.normal(size=(60, 4)), columns=list("abcd"))
    df.loc[:4, "a"] = np.nan  # pairwise deletion
    df["b"] += df["c"]
    ds = load(tmp_path, df)
    res, _ = run(r_session, ds, "cor_matrix", variables=list("abcd"))
    t = tab(res, "Correlation matrix")
    raw = []
    for _, row in t.iterrows():
        sub = df[[row["variable_1"], row["variable_2"]]].dropna()
        ref = st.pearsonr(sub.iloc[:, 0], sub.iloc[:, 1])
        assert row["r"] == pytest.approx(ref.statistic, rel=REL)
        assert row["n"] == len(sub)
        assert row["p_raw"] == pytest.approx(ref.pvalue, rel=1e-5)
        raw.append(ref.pvalue)
    holm = multipletests(raw, method="holm")[1]
    np.testing.assert_allclose(t["p_adjusted"], holm, rtol=1e-5)


# ---- ANOVA (incl. Test B) -------------------------------------------------------

def test_oneway_and_welch_anova(r_session, tmp_path, rng):
    from statsmodels.stats.oneway import anova_oneway

    groups = [rng.normal(m, s, k) for m, s, k in [(0, 1, 20), (0.8, 1.5, 25), (1.2, 2, 30)]]
    ds = load(tmp_path, pd.DataFrame({"y": np.concatenate(groups), "g": np.repeat(["a", "b", "c"], [20, 25, 30])}))
    res, _ = run(r_session, ds, "anova_oneway", dependent="y", factors=["g"])
    row = tab(res, "ANOVA table").iloc[0]
    ref = st.f_oneway(*groups)
    assert row["F"] == pytest.approx(ref.statistic, rel=REL) and row["p"] == pytest.approx(ref.pvalue, rel=REL)
    allv = np.concatenate(groups)
    ssb = sum(len(g) * (g.mean() - allv.mean()) ** 2 for g in groups)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    assert row["partial_eta2"] == pytest.approx(ssb / (ssb + ssw), rel=REL)
    assert any(t.name.startswith("Pairwise comparisons") for t in res.tables)

    res, _ = run(r_session, ds, "anova_welch", dependent="y", factor="g")
    row = tab(res, "Welch one-way").iloc[0]
    ref = anova_oneway(groups, use_var="unequal")
    assert row["F"] == pytest.approx(ref.statistic, rel=REL) and row["p"] == pytest.approx(ref.pvalue, rel=1e-5)


def test_B_factorial_interaction_type3(r_session, tmp_path, rng):
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm

    n = 30
    a = np.repeat(["a1", "a2"], 2 * n)
    b = np.tile(np.repeat(["b1", "b2"], n), 2)
    effect = np.where((a == "a2") & (b == "b2"), 1.5, 0.0)  # planted interaction
    y = rng.normal(size=4 * n) + effect
    keep = np.ones(4 * n, bool)
    keep[:7] = False  # unbalanced
    df = pd.DataFrame({"y": y, "A": a, "B": b})[keep]
    ds = load(tmp_path, df)
    res, _ = run(r_session, ds, "anova_factorial", dependent="y", factors=["A", "B"])
    t = tab(res, "ANOVA table").set_index("effect")
    ref = anova_lm(smf.ols("y ~ C(A, Sum) * C(B, Sum)", df).fit(), typ=3)
    assert t.loc["A:B", "F"] == pytest.approx(ref.loc["C(A, Sum):C(B, Sum)", "F"], rel=REL)
    assert t.loc["A", "F"] == pytest.approx(ref.loc["C(A, Sum)", "F"], rel=REL)
    assert t.loc["A:B", "p"] < 0.001  # planted effect is found


def test_empty_cell_rejected(r_session, tmp_path, rng):
    df = pd.DataFrame({"y": rng.normal(size=30), "A": ["a"] * 10 + ["b"] * 20, "B": ["x"] * 20 + ["y"] * 10})
    ds = load(tmp_path, df)
    res, _ = run_analysis(r_session, ds, BY_KEY["anova_factorial"], {"dependent": ["y"], "factors": ["A", "B"]}, {})
    assert res.status is Status.FAILED and "no observations" in res.messages[0]


def test_rm_and_mixed_anova(r_session, tmp_path, rng):
    from statsmodels.stats.anova import AnovaRM

    ids = np.repeat(np.arange(1, 25), 3)
    time = np.tile(["t1", "t2", "t3"], 24)
    grp = np.where(ids <= 12, "A", "B")
    subj = rng.normal(size=24)[ids - 1]
    y = subj + np.tile([0, 0.5, 1.2], 24) + rng.normal(scale=0.7, size=72)
    df = pd.DataFrame({"id": ids, "time": time, "grp": grp, "y": y})
    df.loc[5, "y"] = np.nan  # participant 2 loses a cell -> excluded
    ds = load(tmp_path, df)
    ds.set_variable_field("id", "measure", "nominal")
    res, _ = run(r_session, ds, "anova_rm", dependent="y", id="id", within=["time"])
    assert res.n_used == 23 and res.values["n_excluded"] == 1
    complete = df[df["id"] != 2]
    ref = AnovaRM(complete, "y", "id", within=["time"]).fit().anova_table
    row = tab(res, "ANOVA table").iloc[0]
    assert row["F"] == pytest.approx(ref["F Value"].iloc[0], rel=REL)
    assert row["p"] >= ref["Pr > F"].iloc[0] - 1e-12  # GG-corrected p is never smaller
    assert any(t.name.startswith("Sphericity") for t in res.tables)

    res, _ = run(r_session, ds, "anova_mixed", dependent="y", id="id", within=["time"], between=["grp"])
    t = tab(res, "ANOVA table").set_index("effect")
    means = complete.groupby(["id", "grp"])["y"].mean().reset_index()
    ref_b = st.f_oneway(*[g["y"].to_numpy() for _, g in means.groupby("grp")])
    assert t.loc["grp", "F"] == pytest.approx(ref_b.statistic, rel=REL)  # between test = ANOVA on subject means


def test_ancova(r_session, tmp_path, rng):
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm

    n = 90
    g = np.repeat(["a", "b", "c"], 30)
    c = rng.normal(50, 10, n)
    y = 0.3 * c + np.select([g == "b", g == "c"], [2.0, 3.0], 0.0) + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "g": g, "c": c})
    ds = load(tmp_path, df)
    res, _ = run(r_session, ds, "ancova", dependent="y", factors=["g"], covariates=["c"])
    t = tab(res, "ANOVA table").set_index("effect")
    df["cc"] = df["c"] - df["c"].mean()
    ref = anova_lm(smf.ols("y ~ C(g, Sum) + cc", df).fit(), typ=3)
    assert t.loc["g", "F"] == pytest.approx(ref.loc["C(g, Sum)", "F"], rel=REL)
    hs = tab(res, "Homogeneity").iloc[0]
    full = smf.ols("y ~ C(g) * cc", df).fit()
    red = smf.ols("y ~ C(g) + cc", df).fit()
    assert hs["F"] == pytest.approx(full.compare_f_test(red)[0], rel=REL)


# ---- regression (Tests D, E, J) ------------------------------------------------

def test_D_continuous_interaction_and_slopes(r_session, tmp_path, rng):
    import statsmodels.formula.api as smf

    n = 200
    x, m = rng.normal(size=n), rng.normal(2, 1.5, n)
    y = 1 + 0.5 * x + 0.2 * m + 0.6 * x * m + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "x": x, "m": m})
    ds = load(tmp_path, df)
    res, art = run(r_session, ds, "lm", {"interactions": "two_way"}, dependent="y", predictors=["x", "m"])
    co = tab(res, "Coefficients").set_index("term")
    ref = smf.ols("y ~ x * m", df).fit()
    for r_term, py_term in [("(Intercept)", "Intercept"), ("x", "x"), ("m", "m"), ("x:m", "x:m")]:
        assert co.loc[r_term, "estimate"] == pytest.approx(ref.params[py_term], rel=REL)
        assert co.loc[r_term, "se"] == pytest.approx(ref.bse[py_term], rel=REL)
        assert co.loc[r_term, "p"] == pytest.approx(ref.pvalues[py_term], rel=1e-5, abs=1e-300)
    fit = tab(res, "Model fit").iloc[0]
    assert fit["R2"] == pytest.approx(ref.rsquared, rel=REL)
    slopes = tab(res, "Conditional slopes")
    at = m.mean() + np.array([-1, 0, 1]) * m.std(ddof=1)
    expected = ref.params["x"] + ref.params["x:m"] * at
    np.testing.assert_allclose(slopes["x.trend"], expected, rtol=1e-6)
    term = tab(res, "Term tests").set_index("term")
    assert term.loc["x:m", "p"] == pytest.approx(ref.pvalues["x:m"], rel=1e-5, abs=1e-300)
    assert len(res.plots) == 2 and all(a[:4] == b"\x89PNG" for a in art.values())


def test_lm_factor_coding_and_polynomial(r_session, tmp_path, rng):
    import statsmodels.formula.api as smf

    n = 120
    g = rng.choice(["ctl", "trt"], n)
    x = rng.uniform(-2, 2, n)
    y = 1 + 2 * (g == "trt") + x - 0.8 * x ** 2 + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "g": g, "x": x, "x2": rng.normal(size=n)})
    ds = load(tmp_path, df)
    ds.set_variable_field("g", "reference_level", "trt")  # reference = trt
    res, _ = run(r_session, ds, "lm", {"nl_type": "poly", "nl_degree": 2}, dependent="y", predictors=["g", "x2"],
                 nonlinear=["x"])
    co = tab(res, "Coefficients").set_index("term")
    ref = smf.ols("y ~ C(g, Treatment('trt')) + x2 + x + I(x**2)", df).fit()
    assert co.loc["g=ctl", "estimate"] == pytest.approx(ref.params["C(g, Treatment('trt'))[T.ctl]"], rel=REL)
    quad = [t for t in co.index if t.startswith("poly(x") and t.endswith("2")][0]
    assert co.loc[quad, "estimate"] == pytest.approx(ref.params["I(x ** 2)"], rel=REL)
    assert tab(res, "Multicollinearity") is not None


def test_hierarchical(r_session, tmp_path, rng):
    import statsmodels.formula.api as smf

    n = 150
    df = pd.DataFrame(rng.normal(size=(n, 4)), columns=["a", "b", "c", "d"])
    df["y"] = df["a"] + 0.5 * df["c"] + rng.normal(size=n)
    ds = load(tmp_path, df)
    res, _ = run(r_session, ds, "lm_hierarchical", dependent="y", block1=["a", "b"], block2=["c"], block3=["d"])
    steps = tab(res, "Hierarchical")
    m1, m2, m3 = (smf.ols(f, df).fit() for f in ["y ~ a + b", "y ~ a + b + c", "y ~ a + b + c + d"])
    np.testing.assert_allclose(steps["R2"], [m1.rsquared, m2.rsquared, m3.rsquared], rtol=REL)
    assert steps.loc[1, "delta_R2"] == pytest.approx(m2.rsquared - m1.rsquared, rel=REL)
    for k, (big, small) in enumerate([(m2, m1), (m3, m2)], start=1):
        f, p, df1 = big.compare_f_test(small)
        assert steps.loc[k, "F_change"] == pytest.approx(f, rel=REL)
        assert steps.loc[k, "p_change"] == pytest.approx(p, rel=1e-5, abs=1e-300)
        assert steps.loc[k, "df2"] == big.df_resid
    assert steps.loc[0, "F_change"] == pytest.approx(m1.fvalue, rel=REL)


def test_E_logistic_interaction(r_session, tmp_path, rng):
    import statsmodels.formula.api as smf

    n = 400
    x = rng.normal(size=n)
    g = rng.choice(["no", "yes"], n)
    eta = -0.5 + 0.8 * x + 0.6 * (g == "yes") + 0.7 * x * (g == "yes")
    yv = rng.binomial(1, 1 / (1 + np.exp(-eta)))
    df = pd.DataFrame({"out": np.where(yv == 1, "event", "none"), "x": x, "g": g})
    ds = load(tmp_path, df)
    ds.set_variable_field("out", "reference_level", "none")  # event = "event"
    res, art = run(r_session, ds, "logistic", {"interactions": "two_way"}, dependent="out", predictors=["x", "g"])
    co = tab(res, "Coefficients").set_index("term")
    df["yb"] = yv
    ref = smf.logit("yb ~ x * C(g)", df).fit(disp=0)
    pairs = [("(Intercept)", "Intercept"), ("x", "x"), ("g=yes", "C(g)[T.yes]"), ("x:g=yes", "x:C(g)[T.yes]")]
    for r_term, py_term in pairs:
        assert co.loc[r_term, "B"] == pytest.approx(ref.params[py_term], rel=1e-5)
        assert co.loc[r_term, "odds_ratio"] == pytest.approx(math.exp(ref.params[py_term]), rel=1e-5)
        assert co.loc[r_term, "p"] == pytest.approx(ref.pvalues[py_term], rel=1e-4)
    pp = tab(res, "Predicted probabilities")
    row = pp.iloc[0]
    xb = ref.params["Intercept"] + ref.params["x"] * row["x"]  # g at reference "no"
    assert row["probability"] == pytest.approx(1 / (1 + math.exp(-xb)), rel=1e-5)
    fit = tab(res, "Model fit").iloc[0]
    assert fit["events"] == yv.sum() and not fit["separation"]
    assert len(art) == 1


def test_logistic_separation_flagged(r_session, tmp_path):
    df = pd.DataFrame({"y": [0] * 10 + [1] * 10, "x": list(range(20))})
    ds = load(tmp_path, df)
    res, _ = run(r_session, ds, "logistic", dependent="y", predictors=["x"])
    assert res.status is Status.WARNING
    assert any("Separation detected" in m for m in res.messages)


def test_J_missing_values_n(r_session, tmp_path, rng):
    df = pd.DataFrame(rng.normal(size=(50, 3)), columns=["y", "a", "b"])
    df.loc[[1, 2, 3], "a"] = np.nan
    df.loc[[3, 4], "b"] = np.nan
    df.loc[[10], "y"] = np.nan
    ds = load(tmp_path, df)
    res, _ = run(r_session, ds, "lm", dependent="y", predictors=["a", "b"])
    expected_excluded = len({1, 2, 3} | {3, 4} | {10})  # overlapping missing rows count once
    assert res.n_used == 50 - expected_excluded and res.values["n_excluded"] == expected_excluded
    assert tab(res, "Model fit").iloc[0]["n"] == 50 - expected_excluded


def test_input_validation(tmp_path, rng):
    ds = load(tmp_path, pd.DataFrame({"y": rng.normal(size=10), "g": ["a", "b"] * 5}))
    with pytest.raises(AnalysisInputError):
        run_analysis(None, ds, BY_KEY["t_independent"], {"dependent": ["g"], "group": ["y"]}, {})
    with pytest.raises(AnalysisInputError):
        run_analysis(None, ds, BY_KEY["cor_pair"], {"x": ["y"], "y": ["y"]}, {})
    with pytest.raises(AnalysisInputError):
        run_analysis(None, ds, BY_KEY["t_one_sample"], {"dependent": ["y"]}, {"conf": 1.5})


# ---- derived variables & filters: R == Python -------------------------------------

def test_derived_and_filters_match_python(r_session, sav_labelled, tmp_path):
    from statexplore.core.derive import DerivedSpec, FilterSpec

    ds = import_file(sav_labelled)
    specs = [
        DerivedSpec("m_items", "mean_items", ["score", "income", "id"], {"min_valid": 2}),
        DerivedSpec("s_items", "sum_items", ["score", "id"]),
        DerivedSpec("lg", "log", ["score"]), DerivedSpec("sq", "sqrt", ["income"]),
        DerivedSpec("zs", "zscore", ["score"]), DerivedSpec("ce", "center", ["score"]),
        DerivedSpec("sqr", "square", ["score"]),
        DerivedSpec("bin", "binarize", ["score"], {"cutoff": 50, "strict": True}),
        DerivedSpec("rec", "recode", ["mood"], {"mapping": [[1, 0], [2, 0], [3, 1]], "keep_other": False}),
        DerivedSpec("rec2", "recode", ["income"], {"mapping": [[10, 1]], "keep_other": True}),
        DerivedSpec("zz", "zscore", ["m_items"]),  # derived from derived
    ]
    for s in specs:
        ds.add_derived(s)
    ds.add_filter(FilterSpec("sex", "==", 2.0))
    ds.add_filter(FilterSpec("score", ">", 40))
    r_session.ensure_data(ds)
    aliases = [ds.var(s.name).alias for s in specs]
    out = r_session.worker.run(
        "sx_result <- list(n = nrow(dat), cols = lapply(dat[, " + repr(aliases).replace("'", '"').replace("[", "c(").replace("]", ")")
        + "], I))").result
    py = ds.analysis_frame([s.name for s in specs])
    assert out["n"] == len(py)
    for s, a in zip(specs, aliases):
        r_vals = np.array([np.nan if v is None else v for v in out["cols"][a]], dtype=float)
        np.testing.assert_allclose(r_vals, py[s.name].to_numpy(float), rtol=1e-12, equal_nan=True, err_msg=s.name)


def test_repeated_ids_flagged(r_session, tmp_path, rng):
    df = pd.DataFrame({"pid": np.repeat(np.arange(20), 3), "y": rng.normal(size=60), "x": rng.normal(size=60)})
    ds = load(tmp_path, df)
    res, _ = run(r_session, ds, "lm", dependent="y", predictors=["x"])
    assert res.status is Status.OK  # no ID role set: nothing to check
    ds.set_variable_field("pid", "role", "id")
    res, _ = run(r_session, ds, "lm", dependent="y", predictors=["x"])
    assert res.status is Status.WARNING
    assert any("60 analysed rows come from only 20 participants" in m for m in res.messages)
    from statexplore.core.derive import FilterSpec

    ds.add_filter(FilterSpec("pid", "<", 100))  # still repeated
    ds.undo()
    one_per = load(tmp_path, df.drop_duplicates("pid"), "one.csv")
    one_per.set_variable_field("pid", "role", "id")
    res, _ = run(r_session, one_per, "lm", dependent="y", predictors=["x"])
    assert res.status is Status.OK
