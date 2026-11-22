"""v1.2 discovery: canonical model specs and counting (X Tests A, B, F, G, H, K), exhaustive interaction
combinations (X Tests C, D, E; M Test D), typed hypotheses and per-type multiplicity families
(X Tests I, J), search integrity (X Test L), fit identity / cache invalidation (M §5, Test B)."""
from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
import pytest

from statexplore.discovery import modelspec as M
from statexplore.discovery.engine import (
    bh, cache_from, create_run, execute, fit_identity, holm, results_frame, summary,
)
from statexplore.discovery.engine import tests_frame as hypotheses  # (a "test*" name would be collected)
from statexplore.discovery.space import (
    SearchSettings, candidate_key, candidate_tests, count_candidates, display_formula, enumerate_candidates,
    full_terms,
)

from test_discovery import make_ds


def S(**kw):
    base = dict(outcome="y", candidates=["X1", "X2", "X3"], strategy="combinations", include_main=True,
                max_predictors=6, max_models=None, max_minutes=None, workers=1)
    base.update(kw)
    return SearchSettings(**base)


# ---- X Tests A, B, F, H, K: canonical specification and counting ----------------------------

def test_A_counting_two_overlapping_interactions():
    c = M.counts(M.parse("Y ~ X1 + X2 + X3 + X1:X2 + X2:X3"), levels={})
    assert c["n_predictors"] == 3 and c["n_main"] == 3 and c["n_two_way"] == 2
    assert c["n_terms"] == 5 and c["n_fixed"] == 6 and c["n_predictors"] <= 6


def test_B_three_way_counts():
    c = M.counts(M.parse("Y ~ X1 * X2 * X3"), levels={})
    assert (c["n_predictors"], c["n_terms"], c["n_fixed"], c["n_three_way"]) == (3, 7, 8, 1)


def test_F_categorical_counts_once_but_adds_coefficients():
    c = M.counts(M.parse("Y ~ G * X1"), levels={"G": 4})
    assert c["n_predictors"] == 2 and c["n_terms"] == 3
    assert c["n_fixed"] == 1 + 3 + 1 + 3                        # intercept, G (3), X1, G:X1 (3)


def test_H_equivalent_formulas_share_one_canonical_spec():
    a = M.parse("Y ~ X1*X2 + X1 + X3")
    b = M.parse("Y ~ X3 + X2:X1 + X1 + X2")
    c = M.parse("(X1 + X2)^2 + X3")
    assert a == b == c
    assert M.canonical_key(a) == M.canonical_key(b) == "X1 + X2 + X3 + X1:X2"
    assert M.display(a, "Y") == "Y ~ X1 * X2 + X3"
    assert M.display(M.parse("X1 + X2 + X3 + X1:X2 + X2:X3"), "Y") == "Y ~ X1 + X2 + X3 + X1:X2 + X2:X3"
    with pytest.raises(ValueError):
        M.parse("X1 + (X2")


def test_K_grouping_variables_are_not_fixed_predictors():
    terms = M.parse("time + activity + gender + time:activity + activity:gender")
    c = M.counts(terms)
    assert c["n_predictors"] == 3 and c["n_two_way"] == 2
    rp = M.random_parameters("rs")
    assert rp == {"re_variances": 2, "re_covariances": 1, "residual": 1}
    assert M.random_parameters("mmrm_us", n_times=4)["residual"] == 10


# ---- X Tests C, D, E, G; M Test D: enumeration ------------------------------------------------

EIGHT = {"Y ~ X1 + X2 + X3", "Y ~ X1 * X2 + X3", "Y ~ X1 * X3 + X2", "Y ~ X2 * X3 + X1",
         "Y ~ X1 + X2 + X3 + X1:X2 + X1:X3", "Y ~ X1 + X2 + X3 + X1:X2 + X2:X3",
         "Y ~ X1 + X2 + X3 + X1:X3 + X2:X3", "Y ~ X1 + X2 + X3 + X1:X2 + X1:X3 + X2:X3"}


class _DS:   # display names = variable names
    class _V:
        def __init__(self, n):
            self.display_name = n

    def var(self, n):
        return self._V(n)


def formulas(s):
    return [display_formula(_DS(), "Y", c) for c in enumerate_candidates(s)]


def test_C_exactly_eight_two_way_models_on_three_predictors():
    s = S(min_predictors=3, max_predictors=3)
    got = formulas(s)
    assert len(got) == count_candidates(s) == 8 and set(got) == EIGHT
    for c in enumerate_candidates(s):
        assert M.counts(full_terms(c))["n_predictors"] == 3


def test_D_three_way_adds_the_full_hierarchical_model():
    s = S(min_predictors=3, max_predictors=3, max_order=3)
    got = formulas(s)
    assert len(got) == count_candidates(s) == 9 and set(got) == EIGHT | {"Y ~ X1 * X2 * X3"}


def test_C_smaller_subsets_are_included_by_default():
    s = S()
    got = formulas(s)
    assert "Y ~ X1" in got and "Y ~ X1 * X2" in got and "Y ~ X1 + X3" in got
    assert len(got) == len(set(got)) == count_candidates(s) == 3 * 1 + 3 * 2 + 8


def test_E_six_predictors_32768_two_way_subsets():
    vs = tuple(f"X{i}" for i in range(1, 7))
    keys = {M.canonical_key([(v,) for v in vs] + list(inter)) for inter in M.interaction_sets(vs, 2)}
    assert len(keys) == 2 ** 15 == 32768 == M.n_interaction_sets(6, 2)
    s = S(candidates=list(vs), min_predictors=6, max_predictors=6)
    assert count_candidates(s) == 32768
    first = list(enumerate_candidates(s, start=32760))                  # resume near the end is fast
    assert [c.index for c in first] == list(range(32760, 32768))


def test_three_way_set_counts_match_enumeration():
    for size in (3, 4):
        vs = tuple(f"v{i}" for i in range(size))
        sets = list(M.interaction_sets(vs, 3))
        assert len(sets) == len({frozenset(x) for x in sets}) == M.n_interaction_sets(size, 3)


def test_G_hierarchy_holds_for_every_candidate():
    s = S(candidates=["a", "b", "c", "d"], max_order=3, max_predictors=4)
    for c in enumerate_candidates(s):
        terms = full_terms(c)
        assert M.is_hierarchical(terms)
        for t in candidate_tests(c):
            assert M.is_hierarchical(t["reduced"])              # reduced models stay hierarchical too


@pytest.mark.parametrize("kw", [
    dict(strategy="combinations", candidates=list("abcde"), max_predictors=4),
    dict(strategy="combinations", candidates=list("abcde"), max_predictors=3, max_order=3, include_main=False),
    dict(strategy="combinations", candidates=list("abcd"), required=["r"], max_predictors=4, min_predictors=2),
    dict(strategy="factorial", candidates=list("abcde"), max_predictors=4, max_order=3),
    dict(strategy="factorial", candidates=list("abcd"), max_predictors=3, include_main=False),
    dict(strategy="main", candidates=list("abcdef"), max_predictors=4),
    dict(strategy="focal", candidates=list("abcde"), max_predictors=4, covariate_mode="subsets", include_main=True),
    dict(strategy="focal", candidates=list("abcde"), max_predictors=3),
])
def test_counts_keys_and_resume(kw):
    s = S(**kw)
    cands = list(enumerate_candidates(s))
    assert len(cands) == count_candidates(s)
    assert [c.index for c in cands] == list(range(len(cands)))
    keys = [candidate_key(c) for c in cands]
    assert len(set(keys)) == len(keys)
    for c in cands:
        assert len(set(c.variables)) <= s.max_predictors
    for mid in (1, len(cands) // 3, len(cands) - 1):
        assert list(enumerate_candidates(s, start=mid)) == cands[mid:]


def test_M_D_main_focal_and_covariate_models_for_four_predictors():
    s = S(strategy="focal", candidates=["X1", "X2", "X3", "X4"], max_predictors=4, covariate_mode="subsets",
          include_main=True, max_order=3)
    got = set(formulas(s))
    for f in ["Y ~ X1 + X2 + X3", "Y ~ X1 * X2", "Y ~ X1 * X2 + X3", "Y ~ X1 * X2 + X3 + X4", "Y ~ X1 * X2 * X3"]:
        assert f in got


def test_I_test_plans_distinguish_hypothesis_types():
    s = S(min_predictors=3, max_predictors=3)
    plans = {display_formula(_DS(), "Y", c): candidate_tests(c) for c in enumerate_candidates(s)}
    main = plans["Y ~ X1 + X2 + X3"]
    assert [t["type"] for t in main] == ["overall", "main_term", "main_term", "main_term"]
    one = plans["Y ~ X1 * X2 + X3"]
    assert [t["type"] for t in one] == ["interaction_term", "overall"]
    multi = plans["Y ~ X1 + X2 + X3 + X1:X2 + X1:X3 + X2:X3"]
    assert [t["type"] for t in multi] == ["joint_interaction"] + ["interaction_term"] * 3 + ["overall"]
    assert multi[0]["reduced"] == [("X1",), ("X2",), ("X3",)]
    three = next(c for c in enumerate_candidates(S(min_predictors=3, max_predictors=3, max_order=3))
                 if len(c.inter) == 4)
    tops = [t["drop"] for t in candidate_tests(three) if t["type"] == "interaction_term"]
    assert tops == [[("X1", "X2", "X3")]]               # two-way terms inside the three-way are not dropped alone


# ---- R: typed tests against statsmodels ------------------------------------------------------

@pytest.fixture
def planted3(tmp_path):
    rng = np.random.default_rng(21)
    n = 300
    df = pd.DataFrame(rng.normal(size=(n, 3)), columns=["x1", "x2", "x3"])
    df["g"] = rng.choice(["a", "b", "c", "d"], n)
    df["y"] = 0.5 * df["x1"] + 0.3 * df["x2"] + 0.6 * df["x1"] * df["x2"] + rng.normal(size=n)
    return df, make_ds(tmp_path, df)


@pytest.mark.r
def test_I_J_combination_search_statistics_and_families(planted3):
    import statsmodels.formula.api as smf
    from statsmodels.stats.multitest import multipletests

    df, ds = planted3
    s = S(outcome="y", candidates=["x1", "x2", "x3"], min_predictors=3, max_predictors=3)
    run = execute(create_run(ds, s), ds)
    assert run.status == "complete" and len(run.rows) == 8
    tf = hypotheses(run)
    by = {(r["formula"], r["test_type"], r["label"]): r for r in tf.to_dict("records")}
    full = smf.ols("y ~ x1 + x2 + x3 + x1:x2 + x1:x3 + x2:x3", df).fit()
    mains = smf.ols("y ~ x1 + x2 + x3", df).fit()
    # D: joint interaction test
    f, p, d1 = full.compare_f_test(mains)
    joint = by[("y ~ x1 + x2 + x3 + x1:x2 + x1:x3 + x2:x3", "joint_interaction",
                "all interactions jointly: x1 × x2, x1 × x3, x2 × x3")]
    assert joint["statistic"] == pytest.approx(f, rel=1e-9) and joint["df1"] == 3 and joint["ci_applicable"] == "joint"
    # C: individual interaction term (1 df): F == t^2, CI is the coefficient's t interval
    term = by[("y ~ x1 + x2 + x3 + x1:x2 + x1:x3 + x2:x3", "interaction_term", "x1 × x2")]
    assert term["statistic"] == pytest.approx(full.tvalues["x1:x2"] ** 2, rel=1e-9)
    assert term["estimate"] == pytest.approx(full.params["x1:x2"], rel=1e-9)
    lo, hi = full.conf_int().loc["x1:x2"]
    assert (term["ci_low"], term["ci_high"]) == (pytest.approx(lo, rel=1e-8), pytest.approx(hi, rel=1e-8))
    assert term["ci_applicable"] == "same_estimand" and term["ci_scale"] == "coefficient" and term["ci_null"] == 0
    # A: overall model test vs intercept only == the regression F statistic
    ov = by[("y ~ x1 + x2 + x3", "overall", "whole model vs intercept only")]
    assert ov["statistic"] == pytest.approx(mains.fvalue, rel=1e-9) and ov["df1"] == 3
    # B: main-effect term in a main-effects model
    mt = by[("y ~ x1 + x2 + x3", "main_term", "x2")]
    assert mt["p"] == pytest.approx(mains.pvalues["x2"], rel=1e-7)
    # J: adjusted p-values are computed within each hypothesis type
    for ttype, g in tf.groupby("test_type"):
        np.testing.assert_allclose(g["p_bh"], multipletests(g["p"], method="fdr_bh")[1], rtol=1e-12)
        assert (g["family_size"] == len(g)).all()
    assert set(tf["test_type"]) == {"overall", "main_term", "interaction_term", "joint_interaction"}
    sm = summary(run, results_frame(run))
    assert {f["type"] for f in sm["families"]} == set(tf["test_type"])
    top = tf[tf["test_type"] == "interaction_term"].sort_values("p").iloc[0]
    assert top["label"] == "x1 × x2" and top["sig_adj"]


@pytest.mark.r
def test_F_categorical_parameter_count_from_design_matrix(planted3):
    df, ds = planted3
    s = S(outcome="y", candidates=["g", "x1"], min_predictors=2, max_predictors=2)
    run = execute(create_run(ds, s), ds)
    row = next(r for r in run.rows if r["kind"] == "combo")
    assert row["n_predictors"] == 2 and row["n_terms"] == 3 and row["n_params"] == 8
    t = row["tests"][0]
    assert t["type"] == "interaction_term" and t["df1"] == 3 and t["ci_applicable"] == "joint"


@pytest.mark.r
def test_L_interrupted_search_resumes_without_duplicates(planted3):
    df, ds = planted3
    s = S(outcome="y", candidates=["x1", "x2", "x3", "g"], max_predictors=3, max_models=7)
    run = execute(create_run(ds, s), ds)
    assert run.status == "partial" and summary(run, results_frame(run))["provisional"]
    assert hypotheses(run)["search_scope"].str.startswith("partial").all()
    run.settings.max_models = None
    execute(run, ds)
    assert run.status == "complete"
    idx = [r["idx"] for r in run.rows]
    assert idx == list(range(run.total)) and len(set(r["key"] for r in run.rows)) == run.total
    whole = execute(create_run(ds, S(outcome="y", candidates=["x1", "x2", "x3", "g"], max_predictors=3)), ds)
    np.testing.assert_allclose(hypotheses(run)["p"], hypotheses(whole)["p"], rtol=1e-10)


# ---- M §5 / Test B: fit identity ---------------------------------------------------------------

@pytest.fixture
def binary_ds(tmp_path):
    rng = np.random.default_rng(5)
    n = 240
    df = pd.DataFrame(rng.normal(size=(n, 2)), columns=["a", "b"])
    df["y"] = rng.choice(["no", "yes"], n)
    df["m"] = rng.choice(["p", "q", "r"], n)
    df["k"] = rng.integers(1, 6, n)
    df["pid"] = np.repeat(np.arange(n // 4), 4)
    return make_ds(tmp_path, df)


def test_B_fit_identity_changes_with_every_fit_defining_setting(binary_ds):
    ds = binary_ds
    base = dict(outcome="y", candidates=["a", "b"], family="binomial", strategy="focal")
    ref = fit_identity(ds, SearchSettings(**base), "binomial")
    assert ref == fit_identity(ds, SearchSettings(**base, event="yes"), "binomial")   # default event == "yes"
    for change in (dict(event="no"), dict(sample="per_model"), dict(candidates=["a"], required=["b"]),
                   dict(conf=0.9), dict(max_params=10)):
        assert fit_identity(ds, SearchSettings(**{**base, **change}), "binomial") != ref, change
    # presentation settings never change the identity
    for change in (dict(alpha=0.01), dict(correction="holm"), dict(mode="D"), dict(max_models=5)):
        assert fit_identity(ds, SearchSettings(**{**base, **change}), "binomial") == ref, change
    mult = dict(outcome="m", candidates=["a", "b"], family="multinomial", strategy="focal")
    assert fit_identity(ds, SearchSettings(**mult, reference="p"), "multinomial") != \
        fit_identity(ds, SearchSettings(**mult, reference="q"), "multinomial")
    gauss = dict(outcome="a", candidates=["b", "k"], family="gaussian", strategy="focal")
    assert fit_identity(ds, SearchSettings(**gauss), "gaussian") != \
        fit_identity(ds, SearchSettings(**gauss, participant="pid"), "lmm")
    # an analysis treatment changes the data view (k as continuous vs categorical)
    ds.set_variable_field("k", "measure", "ordinal")
    t = [{"variable": "k", "treatment": "continuous", "acknowledged": True}]
    assert fit_identity(ds, SearchSettings(**gauss), "gaussian") != \
        fit_identity(ds, SearchSettings(**gauss, treatments=t), "gaussian")


@pytest.mark.r
def test_B_event_change_never_reuses_fits_but_alpha_change_does(binary_ds):
    ds = binary_ds
    base = dict(outcome="y", candidates=["a", "b"], family="binomial", strategy="combinations",
                include_main=True, max_models=None, max_minutes=None, workers=1)
    run = execute(create_run(ds, SearchSettings(**base, event="yes")), ds)
    flipped = create_run(ds, SearchSettings(**base, event="no"))
    assert cache_from(flipped, [run], ds) == {}
    execute(flipped, ds)
    a = hypotheses(run).set_index("test_id")
    b = hypotheses(flipped).set_index("test_id")
    one = a["ci_applicable"] == "same_estimand"
    np.testing.assert_allclose(b.loc[one, "estimate"], -a.loc[one, "estimate"], rtol=1e-6)   # coding flips sign
    stricter = create_run(ds, SearchSettings(**base, event="yes", alpha=0.01, correction="holm"))
    cache = cache_from(stricter, [run], ds)
    assert len(cache) == run.total
    execute(stricter, ds, cache=cache)
    c = hypotheses(stricter)
    assert all(r["cached"] and "identical fit identity" in r["cache_reason"] for r in stricter.rows)
    np.testing.assert_allclose(c["p"], hypotheses(run)["p"])
    np.testing.assert_allclose(c["p_adj"], c["p_holm"])          # adjusted p recomputed for the new setting


@pytest.mark.r
def test_project_round_trip_keeps_typed_tests(planted3, tmp_path):
    from statexplore.core.project import Project

    df, ds = planted3
    run = execute(create_run(ds, S(outcome="y", candidates=["x1", "x2", "g"], max_predictors=2)), ds)
    p = Project(ds)
    p.searches.append(run)
    p.settings["discovery_highlight"] = {"mode": "strict", "alpha": 0.01, "only": True}
    p.save(tmp_path / "p.sxp")
    q = Project.open(tmp_path / "p.sxp")
    assert q.settings["discovery_highlight"]["mode"] == "strict"
    r2 = q.searches[0]
    assert r2.fit_id == run.fit_id and r2.rows[3]["tests"] == run.rows[3]["tests"]
    pd.testing.assert_frame_equal(hypotheses(r2)[["test_id", "p", "p_adj", "ci_applicable"]],
                                  hypotheses(run)[["test_id", "p", "p_adj", "ci_applicable"]])


@pytest.mark.r
def test_combination_candidate_drill_down_and_model_spec(planted3, r_session):
    from statexplore.analysis.models import from_candidate
    from statexplore.discovery.detail import candidate_from_row, full_analysis

    df, ds = planted3
    run = execute(create_run(ds, S(outcome="y", candidates=["x1", "x2", "x3"], min_predictors=3,
                                   max_predictors=3)), ds)
    tf = hypotheses(run)
    hit = tf[(tf["formula"] == "y ~ x1 + x2 + x3 + x1:x2 + x2:x3") & (tf["label"] == "x1 × x2")].iloc[0]
    row = next(r for r in run.rows if r["idx"] == hit["idx"])
    cand = candidate_from_row(run, row)
    assert cand.kind == "combo" and cand.inter == (("x1", "x2"), ("x2", "x3"))
    res, _ = full_analysis(r_session, ds, run, cand)
    term = [t for t in res.tables if t.name.startswith("Term tests")][0]
    p_detail = dict(zip([r[0] for r in term.rows], [r[3] for r in term.rows]))
    assert p_detail["x1:x2"] == pytest.approx(hit["p"], rel=1e-9)
    spec = from_candidate(run, row)
    assert set(spec.terms) == {("x1",), ("x2",), ("x3",), ("x1", "x2"), ("x2", "x3")}
