"""Phase 3 acceptance: enumeration (Test M), planted interactions, null data (Test K),
invalid models, multiplicity, partial / early-stop labelling, determinism, resume."""
from __future__ import annotations

import threading
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
import pytest

from statexplore.core.importers import import_file
from statexplore.discovery.engine import (
    bh, create_run, discovery_ranking, execute, holm, results_frame, summary, support_ranking,
)
from statexplore.discovery.space import (
    SearchSettings, SearchSpaceError, candidate_key, count_candidates, describe_space, enumerate_candidates,
    model_terms, validate_settings,
)


def names(p):
    return [f"x{i}" for i in range(1, p + 1)]


# ---- Test M: enumeration ---------------------------------------------------------------

@pytest.mark.parametrize("p,k,order,mode,req", [
    (5, 2, 2, "focal", 0), (6, 3, 3, "focal", 0), (6, 3, 2, "subsets", 0), (6, 4, 3, "subsets", 0),
    (7, 4, 3, "subsets", 1), (8, 6, 3, "subsets", 2), (4, 3, 3, "focal", 1), (10, 6, 2, "subsets", 0),
])
def test_M_enumeration_matches_count(p, k, order, mode, req):
    s = SearchSettings(outcome="y", candidates=names(p), required=[f"r{i}" for i in range(req)],
                       max_predictors=k, max_order=order, covariate_mode=mode)
    cands = list(enumerate_candidates(s))
    assert len(cands) == count_candidates(s)
    assert [c.index for c in cands] == list(range(len(cands)))
    keys = [candidate_key(c) for c in cands]
    assert len(set(keys)) == len(keys)  # no duplicate specifications
    for c in cands:
        assert len(c.variables) <= k
        full, reduced = model_terms(c)
        assert c.focal in full and c.focal not in reduced
        # hierarchy: every lower-order term of the focal interaction is present in both models
        for size in range(1, len(c.focal)):
            for sub in combinations(c.focal, size):
                assert sub in full and sub in reduced
    # resuming from an index yields exactly the tail
    mid = len(cands) // 2
    assert list(enumerate_candidates(s, start=mid)) == cands[mid:]


def test_spec_example_subset():
    """§10.3: for {X1, X2, X3} the candidates include X1*X2 + X3, X1*X3 + X2, X2*X3 + X1, X1*X2*X3."""
    s = SearchSettings(outcome="y", candidates=["X1", "X2", "X3"], max_predictors=3, max_order=3,
                       covariate_mode="subsets")
    got = {(c.focal, c.covariates) for c in enumerate_candidates(s)}
    for expected in [(("X1", "X2"), ("X3",)), (("X1", "X3"), ("X2",)), (("X2", "X3"), ("X1",)),
                     (("X1", "X2", "X3"), ())]:
        assert expected in got


def test_large_space_count():
    s = SearchSettings(outcome="y", candidates=names(39), max_predictors=6, covariate_mode="subsets")
    assert comb(39, 6) > 3_000_000  # §10.4 example
    assert count_candidates(s) == sum(comb(39, k) * comb(k, 2) for k in range(2, 7))


def test_bh_holm_match_statsmodels():
    from statsmodels.stats.multitest import multipletests

    p = np.random.default_rng(3).uniform(size=200) ** 3
    np.testing.assert_allclose(bh(p), multipletests(p, method="fdr_bh")[1], rtol=1e-12)
    np.testing.assert_allclose(holm(p), multipletests(p, method="holm")[1], rtol=1e-12)


# ---- engine (R) ------------------------------------------------------------------------

def make_ds(tmp_path, df, name="d.csv"):
    p = tmp_path / name
    df.to_csv(p, index=False)
    return import_file(p)


@pytest.fixture
def planted(tmp_path):
    rng = np.random.default_rng(7)
    n = 400
    df = pd.DataFrame(rng.normal(size=(n, 6)), columns=names(6))
    df["g"] = rng.choice(["a", "b"], n)
    df["y"] = df["x1"] + 0.5 * df["x2"] + 0.6 * df["x1"] * df["x2"] + rng.normal(size=n)
    return make_ds(tmp_path, df)


def settings(**kw):
    base = dict(outcome="y", candidates=names(6) + ["g"], max_models=None, max_minutes=None, workers=1)
    base.update(kw)
    return SearchSettings(**base)




@pytest.mark.r
def test_planted_interaction_found_and_p_matches_analysis(planted, r_session):
    from statexplore.analysis.builders import BY_KEY
    from statexplore.analysis.framework import run_analysis

    run = execute(create_run(planted, settings()), planted)
    df = results_frame(run)
    assert run.status == "complete" and len(df) == comb(7, 2)
    top = discovery_ranking(df).iloc[0]
    assert top["interaction"] == "x1 × x2" and top["sig_adj"]
    # noise interactions: at most a few raw hits, adjusted p-values well controlled
    assert df.loc[df["interaction"] != "x1 × x2", "sig_adj"].sum() <= 1
    assert support_ranking(df).iloc[0]["interaction"] == "x1 × x2"
    # the focal p equals the Phase 2 regression term test on the same model
    res, _ = run_analysis(r_session, planted, BY_KEY["lm"], {"dependent": ["y"], "predictors": ["x1", "x2"]},
                          {"interactions": "two_way"})
    term = [t for t in res.tables if t.name.startswith("Term tests")][0]
    p_lm = dict(zip([r[0] for r in term.rows], [r[3] for r in term.rows]))["x1:x2"]
    assert top["p"] == pytest.approx(p_lm, rel=1e-9)
    assert top["df1"] == 1 and top["estimate"] == pytest.approx(0.6, abs=0.15)
    assert top["validation"] == "not validated"
    for _, r in df.iterrows():  # interpretation wording follows the significance status
        if r["sig_adj"]:
            assert r["interpretation"].startswith("Evidence")
        elif r["sig_raw"]:
            assert "plausibly a chance finding" in r["interpretation"]
        else:
            assert r["interpretation"].startswith("No clear evidence")


@pytest.mark.r
def test_K_null_data_raw_vs_adjusted(tmp_path):
    rng = np.random.default_rng(11)
    df = pd.DataFrame(rng.normal(size=(150, 11)), columns=names(10) + ["y"])
    ds = make_ds(tmp_path, df)
    run = execute(create_run(ds, settings(candidates=names(10))), ds)
    df_r = results_frame(run)
    sm = summary(run, df_r)
    assert sm["family_size"] == 45 and sm["sig_raw"] >= 1  # raw false positives occur by chance...
    assert sm["sig_adj"] == 0                               # ...and disappear after adjustment
    assert sm["expected_false_raw"] == pytest.approx(45 * 0.05)
    from statsmodels.stats.multitest import multipletests

    fam = df_r[df_r["in_family"]]
    np.testing.assert_allclose(fam["p_bh"], multipletests(fam["p"], method="fdr_bh")[1], rtol=1e-12)


@pytest.mark.r
def test_invalid_models_recorded(tmp_path):
    rng = np.random.default_rng(5)
    n = 80
    df = pd.DataFrame({"y": rng.normal(size=n), "x1": rng.normal(size=n),
                       "a": ["p"] * 40 + ["q"] * 40, "b": ["u"] * 40 + ["v"] * 40,  # a, b confounded
                       "c": rng.choice(["k", "l", "m"], n)})
    df["x2"] = df["x1"] * 2  # perfectly collinear
    ds = make_ds(tmp_path, df)
    run = execute(create_run(ds, settings(candidates=["x1", "x2", "a", "b", "c"])), ds)
    df_r = results_frame(run)
    assert len(df_r) == comb(5, 2)  # every attempt recorded, including invalid ones
    inv = df_r[df_r["status"] == "invalid"].set_index("interaction")
    assert "empty cell" in inv.loc["a × b", "reason"]
    assert "rank-deficient" in inv.loc["x1 × x2", "reason"]
    assert not df_r.loc[df_r["status"] == "invalid", "in_family"].any()


@pytest.mark.r
def test_partial_budget_and_resume_equals_full(planted):
    full = results_frame(execute(create_run(planted, settings()), planted))
    run = create_run(planted, settings(max_models=8, batch_size=3))
    execute(run, planted)
    assert run.status == "partial" and run.n_done == 8 and "budget" in run.stop_reason
    assert summary(run, results_frame(run))["provisional"]
    run.settings.max_models = None
    execute(run, planted)  # resume
    assert run.status == "complete" and run.n_done == run.total
    resumed = results_frame(run)
    pd.testing.assert_frame_equal(resumed[["idx", "p", "p_bh"]], full[["idx", "p", "p_bh"]])


@pytest.mark.r
def test_mode_A_stops_deterministically_with_two_workers(planted):
    a = create_run(planted, settings(mode="A", batch_size=2, workers=2))
    execute(a, planted)
    b = create_run(planted, settings(mode="A", batch_size=2, workers=1))
    execute(b, planted)
    assert a.stop_index == b.stop_index == 0  # x1 × x2 is the first candidate and is significant
    assert a.status == "stopped_early"
    df = results_frame(a)
    assert df.loc[df["idx"] > a.stop_index, "speculative"].all()
    assert int(df["in_family"].sum()) == 1


@pytest.mark.r
def test_two_workers_match_one(planted):
    one = results_frame(execute(create_run(planted, settings(batch_size=4)), planted))
    two = results_frame(execute(create_run(planted, settings(batch_size=4, workers=2)), planted))
    pd.testing.assert_frame_equal(one[["idx", "p", "effect", "status"]], two[["idx", "p", "effect", "status"]])


@pytest.mark.r
def test_cancel_marks_incomplete(planted):
    ev = threading.Event()
    ev.set()
    run = execute(create_run(planted, settings(batch_size=2)), planted, cancel=ev)
    assert run.status == "cancelled" and run.n_done < run.total


@pytest.mark.r
def test_logistic_discovery_and_subset_mode(tmp_path):
    rng = np.random.default_rng(9)
    n = 600
    df = pd.DataFrame(rng.normal(size=(n, 4)), columns=names(4))
    eta = -0.3 + 0.5 * df["x1"] + 0.9 * df["x1"] * df["x3"]
    df["y"] = rng.binomial(1, 1 / (1 + np.exp(-eta)))
    ds = make_ds(tmp_path, df)
    s = settings(candidates=names(4), max_predictors=3, covariate_mode="subsets")
    run = execute(create_run(ds, s), ds)
    assert run.family == "binomial" and run.status == "complete"
    df_r = results_frame(run)
    assert len(df_r) == count_candidates(s) == comb(4, 2) + comb(4, 3) * 3
    top = discovery_ranking(df_r).iloc[0]
    assert top["interaction"] == "x1 × x3" and top["sig_adj"]
    # the same focal interaction under different adjustments is kept as distinct analyses
    assert (df_r["interaction"] == "x1 × x3").sum() == 3


def test_settings_validation(planted):
    with pytest.raises(SearchSpaceError):
        validate_settings(planted, settings(candidates=["x1"]))
    with pytest.raises(SearchSpaceError):
        validate_settings(planted, settings(required=["x1"]))  # overlaps candidates
    with pytest.raises(SearchSpaceError):
        validate_settings(planted, settings(max_predictors=7))
    with pytest.raises(SearchSpaceError):
        validate_settings(planted, settings(candidates=names(5), required=["x6", "g"], max_predictors=3))
    info = describe_space(planted, settings(max_models=5))
    assert info["total"] == 21 and info["n_shared"] == 400 and any("partial" in w for w in info["warnings"])


@pytest.mark.r
def test_detail_analysis_matches_search_and_project_roundtrip(planted, r_session, tmp_path):
    from statexplore.core.project import Project
    from statexplore.discovery.detail import candidate_from_row, full_analysis

    run = execute(create_run(planted, settings(sample="shared")), planted)
    df = results_frame(run)
    row = discovery_ranking(df).iloc[0].to_dict()
    res, art = full_analysis(r_session, planted, run, candidate_from_row(run, row))
    assert res.n_used == row["n"]
    term = [t for t in res.tables if t.name.startswith("Term tests")][0]
    p_detail = dict(zip([r[0] for r in term.rows], [r[3] for r in term.rows]))["x1:x2"]
    assert p_detail == pytest.approx(row["p"], rel=1e-9)
    assert any(t.name.startswith("Conditional slopes") for t in res.tables)

    proj = Project(planted)
    proj.searches.append(run)
    out = tmp_path / "s.sxp"
    proj.save(out)
    again = Project.open(out).searches[0]
    assert again.meta_dict() == run.meta_dict()
    pd.testing.assert_frame_equal(results_frame(again)[["idx", "p", "p_bh", "tier", "flags"]],
                                  df[["idx", "p", "p_bh", "tier", "flags"]])
    assert candidate_from_row(again, again.rows[0]) == candidate_from_row(run, run.rows[0])


@pytest.mark.r
def test_shared_sample_detail_uses_same_rows(tmp_path, r_session):
    from statexplore.discovery.detail import candidate_from_row, full_analysis

    rng = np.random.default_rng(2)
    n = 120
    df = pd.DataFrame(rng.normal(size=(n, 4)), columns=names(4))
    df["y"] = df["x1"] * df["x2"] + rng.normal(size=n)
    df.loc[:19, "x4"] = np.nan  # x4 is missing for 20 rows -> shared sample = 100
    ds = make_ds(tmp_path, df)
    run = execute(create_run(ds, settings(candidates=names(4))), ds)
    row = results_frame(run).iloc[0].to_dict()
    assert row["interaction"] == "x1 × x2" and row["n"] == 100 and run.n_sample == 100
    res, _ = full_analysis(r_session, ds, run, candidate_from_row(run, row))
    assert res.n_used == 100


def test_dependence_warning(tmp_path):
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"pid": np.repeat(np.arange(30), 2), "y": rng.normal(size=60),
                       "a": rng.normal(size=60), "b": rng.normal(size=60)})
    ds = make_ds(tmp_path, df)
    s = settings(candidates=["a", "b"])
    assert create_run(ds, s).warnings == []
    ds.set_variable_field("pid", "role", "id")
    run = create_run(ds, s)
    assert "60 rows come from only 30 participants" in run.warnings[0]
    assert any("not independent" in w for w in describe_space(ds, s)["warnings"])


@pytest.mark.r
def test_mixed_model_discovery(tmp_path):
    """Repeated measures: a random-intercept search recovers the planted interaction, and its focal
    likelihood-ratio statistic equals an independent statsmodels MixedLM (ML) comparison."""
    import statsmodels.formula.api as smf

    rng = np.random.default_rng(13)
    n_id, k = 60, 4
    df = pd.DataFrame({"pid": np.repeat(np.arange(1, n_id + 1), k)})
    df["a"] = rng.normal(size=len(df))
    df["b"] = rng.normal(size=len(df))
    df["c"] = rng.normal(size=len(df))
    df["y"] = rng.normal(0, 1.2, n_id).repeat(k) + 0.4 * df["a"] + 0.5 * df["a"] * df["b"] + rng.normal(size=len(df))
    ds = make_ds(tmp_path, df)
    ds.set_variable_field("pid", "role", "id")
    s = settings(candidates=["a", "b", "c"], participant="pid")
    assert create_run(ds, s).warnings == []  # repeated measures are modelled: no dependence warning
    run = execute(create_run(ds, s), ds)
    assert run.family == "lmm" and run.status == "complete"
    df_r = results_frame(run)
    top = discovery_ranking(df_r).iloc[0]
    assert top["interaction"] == "a × b" and top["sig_adj"] and top["df2"] == n_id
    full = smf.mixedlm("y ~ a * b", df, groups=df["pid"]).fit(reml=False)
    red = smf.mixedlm("y ~ a + b", df, groups=df["pid"]).fit(reml=False)
    assert top["statistic"] == pytest.approx(2 * (full.llf - red.llf), rel=1e-3)
    with pytest.raises(SearchSpaceError):
        validate_settings(ds, settings(candidates=["a", "pid"], participant="pid"))


def test_all_two_way_enumeration():
    s = SearchSettings(outcome="y", candidates=names(5), max_predictors=4, covariate_mode="subsets", all_two_way=True)
    cands = list(enumerate_candidates(s))
    assert len(cands) == count_candidates(s)
    all2 = [c for c in cands if c.kind == "all_two_way"]
    assert len(all2) == comb(5, 3) + comb(5, 4)
    full, reduced = model_terms(all2[0])
    assert reduced == [(v,) for v in all2[0].focal] and len(full) == 3 + 3
    assert len({candidate_key(c) for c in cands}) == len(cands)


@pytest.mark.r
def test_all_two_way_joint_test_and_cache(tmp_path):
    import statsmodels.formula.api as smf

    from statexplore.discovery.engine import cache_from

    rng = np.random.default_rng(17)
    df = pd.DataFrame(rng.normal(size=(250, 3)), columns=["a", "b", "c"])
    df["y"] = df["a"] * df["b"] * 0.5 + df["b"] * df["c"] * 0.4 + rng.normal(size=250)
    ds = make_ds(tmp_path, df)
    s = settings(candidates=["a", "b", "c"], max_predictors=3, covariate_mode="subsets", all_two_way=True)
    run = execute(create_run(ds, s), ds)
    df_r = results_frame(run)
    row = df_r[df_r["kind"] == "all_two_way"].iloc[0]
    f, p, df1 = smf.ols("y ~ (a + b + c)**2", df).fit().compare_f_test(smf.ols("y ~ a + b + c", df).fit())
    assert row["df1"] == 3 and row["statistic"] == pytest.approx(f, rel=1e-9) and row["p"] == pytest.approx(p, rel=1e-6)
    # caching: same data/sample/settings except alpha -> nothing refitted, identical results
    s2 = settings(candidates=["a", "b", "c"], max_predictors=3, covariate_mode="subsets", all_two_way=True, alpha=0.01)
    run2 = create_run(ds, s2)
    cache = cache_from(run2, [run])
    assert len(cache) == run.total
    execute(run2, ds, cache=cache)
    df2 = results_frame(run2)
    assert df2["cached"].all() and run2.status == "complete"
    np.testing.assert_allclose(df2["p"], df_r["p"])
    # a different candidate pool changes the shared sample: no reuse
    assert cache_from(create_run(ds, settings(candidates=["a", "b"])), [run]) == {}


@pytest.mark.r
def test_C_known_three_way_interaction(tmp_path):
    """§25 Test C: a planted three-way interaction is detected and interpreted."""
    import statsmodels.formula.api as smf

    rng = np.random.default_rng(23)
    n = 600
    df = pd.DataFrame(rng.normal(size=(n, 4)), columns=["a", "b", "c", "d"])
    df["y"] = 0.3 * df["a"] + 0.6 * df["a"] * df["b"] * df["c"] + rng.normal(size=n)
    ds = make_ds(tmp_path, df)
    run = execute(create_run(ds, settings(candidates=["a", "b", "c", "d"], max_order=3)), ds)
    df_r = results_frame(run)
    three = df_r[df_r["order"] == 3]
    top = three.sort_values("p").iloc[0]
    assert top["interaction"] == "a × b × c" and top["sig_adj"]
    full = smf.ols("y ~ a * b * c", df).fit()
    assert top["p"] == pytest.approx(full.pvalues["a:b:c"], rel=1e-6)
    assert top["estimate"] == pytest.approx(full.params["a:b:c"], rel=1e-9)
    assert "the a × b interaction on y differs across c" in top["interpretation"]
    others = three[three["interaction"] != "a × b × c"]
    assert not others["sig_adj"].any()
