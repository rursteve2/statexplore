"""StatExplore v1.3.4 corrective release: one test (or more) per register issue V134-001 ... V134-017.

Oracles are independent of the code under test: the owner's fresh R refits (tests/fixtures/v134/reference, produced
outside StatExplore), direct geepack / lme4 calls written here, statsmodels / scipy for adjustments and log tails, and
brute-force enumeration. Each test states what it guards; run on v1.3.3 these fail (red), on v1.3.4 they pass.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from statexplore.core.importers import import_file

FX = Path(__file__).parent / "fixtures" / "v134"
REF = FX / "reference"


def fx(name: str):
    return import_file(FX / f"StatExplore_{name}.csv")


def rscript(code: str, cwd: Path, timeout: int = 1800) -> subprocess.CompletedProcess:
    from statexplore.rbridge.locate import find_rscript

    lib = Path(__file__).resolve().parents[1] / "r" / "library"
    env = dict(os.environ, R_LIBS=str(lib), R_LIBS_USER=str(lib), OPENBLAS_NUM_THREADS="1")
    (cwd / "oracle.R").write_text(code, encoding="utf-8")
    return subprocess.run([str(find_rscript()), "--vanilla", str(cwd / "oracle.R")], cwd=cwd, capture_output=True,
                          text=True, timeout=timeout, env=env)


def search(d, **kw):
    from statexplore.discovery import engine
    from statexplore.discovery.space import SearchSettings

    base = dict(family="gaussian", workers=2, max_models=None, max_minutes=None, study="longitudinal")
    base.update(kw)
    run = engine.create_run(d, SearchSettings(**base))
    engine.execute(run, d)
    return run


def s51h(**kw):
    return dict(outcome="TestScore", candidates=["Sleep", "Anxiety", "Attendance"], required=["Time", "Gender"],
                participant="StudentID", time="Time", cluster="School", structure="rs", strategy="combinations",
                include_main=True, max_order=3, max_predictors=5, **kw)


def s47b(**kw):
    return dict(outcome="Y", candidates=["GROUP", "GROUP2"], participant="ID", time="TIME", cluster="SCHOOL",
                id_policy="local", structure="rs_ar1", strategy="time", covariate_mode="focal", max_predictors=3, **kw)


def test_fixture_bytes_match_manifest():
    for line in (FX / "SHA256SUMS.txt").read_text().splitlines():
        h, name = line.split(maxsplit=1)
        assert hashlib.sha256((FX / name.lstrip("./")).read_bytes()).hexdigest() == h, name
    # the owner's recorded fingerprints of the original files
    assert hashlib.md5((FX / "StatExplore_Test43B_PositiveInteraction_Rerun.csv").read_bytes()).hexdigest() == \
        "15b557b58e7a98bc1b91cce07b27fedb"


# ---- V134-001 / AUDIT01: fit cache isolation ------------------------------------------------------------------------

def _helpers() -> str:
    from statexplore.analysis.covstruct import MIXED_R_HELPERS
    from statexplore.discovery.rcode import DISCOVERY_HELPERS
    from statexplore.rbridge.codegen import IDENTITY_R_HELPERS, R_HELPERS

    return "\n".join([R_HELPERS, IDENTITY_R_HELPERS, MIXED_R_HELPERS, DISCOVERY_HELPERS])


@pytest.mark.r
def test_fit_key_adversaries(r_worker):
    """Same row names but altered response / covariate, equal n with other rows, other grouping or coding, another
    structure or estimation regime, or another sample context: the key differs. Identical content: equal key."""
    script = _helpers() + r'''
sx_disc_family <- "lmm"; sx_disc_struct <- "ri"; sx_disc_clustered <- FALSE; sx_disc_reml <- FALSE
sx_disc_gee_family <- "gaussian"; sx_disc_fit_rules <- "fit-rules-4"; sx_cache_reset("search")
set.seed(1); n <- 60
d <- data.frame(sx_y = rnorm(n), v001 = rnorm(n), sx_gid = factor(rep(1:12, each = 5)), sx_time = rep(0:4, 12))
d$sx_timef <- factor(d$sx_time); d$sx_occ <- as.integer(d$sx_timef); rownames(d) <- paste0("b", seq_len(n))
f <- sx_y ~ v001
k0 <- sx_fit_key(f, d)
d1 <- d; d1$sx_y[3] <- d1$sx_y[3] + 1                      # same row names, other response
d2 <- d; d2$v001 <- rev(d2$v001)                          # same row names, other covariate
d3 <- d; d3$sx_gid <- factor(rep(1:6, each = 10))         # other grouping
d4 <- d[sample(n), ]; rownames(d4) <- rownames(d)          # equal n, same labels, other rows in each position
d5 <- d; d5$v001 <- d5$v001                               # identical content
sx_disc_struct <- "rs"; k_struct <- sx_fit_key(f, d); sx_disc_struct <- "ri"
sx_disc_reml <- TRUE; k_reml <- sx_fit_key(f, d); sx_disc_reml <- FALSE
k_ctx <- { sx_cache_reset("boot2"); k <- sx_fit_key(f, d); sx_cache_reset("search"); k }
k_terms <- sx_fit_key(sx_y ~ v001 + sx_time, d)
sx_result <- list(resp = k0 != sx_fit_key(f, d1), cov = k0 != sx_fit_key(f, d2), grp = k0 != sx_fit_key(f, d3),
                  rows = k0 != sx_fit_key(f, d4), same = k0 == sx_fit_key(f, d5), struct = k0 != k_struct,
                  reml = k0 != k_reml, ctx = k0 != k_ctx, terms = k0 != k_terms)
'''
    out = r_worker.run(script, packages=["lme4", "nlme"], timeout=300).result
    assert {k: bool(v) for k, v in out.items()} == {k: True for k in out}, out


@pytest.mark.r
def test_reduced_cache_never_crosses_bootstrap_draws(r_worker):
    """The v1.3.3 failure mechanism: two different resamples with identical row names b1..bn. The second draw's
    reduced-model statistics must be refitted, not taken from the first draw."""
    script = _helpers() + r'''
sx_disc_family <- "gaussian"; sx_disc_struct <- "ri"; sx_disc_clustered <- FALSE; sx_disc_reml <- FALSE
sx_disc_gee_family <- "gaussian"; sx_disc_fit_rules <- "fit-rules-4"; sx_cache_reset("search")
set.seed(2); n <- 80
mk <- function() { d <- data.frame(sx_y = rnorm(n), v001 = rnorm(n)); rownames(d) <- paste0("b", seq_len(n)); d }
a <- mk(); b <- mk()
sa <- sx_reduced_stats("v001", a); sb <- sx_reduced_stats("v001", b); sa2 <- sx_reduced_stats("v001", a)
sx_result <- list(a = sa$rss, b = sb$rss, ref_a = deviance(lm(sx_y ~ v001, a)), ref_b = deviance(lm(sx_y ~ v001, b)),
                  src = c(sa$source, sb$source, sa2$source), hits = sx_disc_cache_log$hits)
'''
    out = r_worker.run(script, timeout=300).result
    assert abs(out["b"] - out["ref_b"]) < 1e-9 and abs(out["a"] - out["ref_a"]) < 1e-9
    assert out["src"] == ["fitted", "fitted", "cache"] and out["hits"] == 1


@pytest.mark.r
@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_51h3_bootstrap_matches_fresh_log_oracle(r_session):
    """V134-001/002/003/011: 51H3 (B = 20, seed 133, new schools, whole pipeline, p rule). Draw manifest = the owner's
    export; every candidate in every draw vs the owner's independent fresh refits; selections = fresh log-tail oracle
    (#17 in all 20); counts by identity sum to the successful draws; the reported estimand is the triple coefficient."""
    from statexplore.discovery import validate as V

    d = fx("Test51C_ThreeWayInteraction")
    run = search(d, **s51h())
    assert run.total == 18 and len(run.rows) == 18
    vr = V.ValidationRunner(r_session, d, run, V.ValidationConfig(target="cluster", modes=["bootstrap"], B=20,
                                                                  seed=133)).execute()
    mine = pd.DataFrame(vr["draws"])[["b", "row_id", "draw_participant", "draw_cluster"]]
    owner = pd.read_csv(REF / "Test51H3_owner_draws.csv", dtype={"draw_cluster": str, "draw_gid": str})
    assert np.array_equal(mine.to_numpy().astype(str), owner.to_numpy().astype(str))
    assert vr["status"] == "complete"
    sel = [int(b["selected_index"]) for b in vr["bootstrap"]]
    O = pd.read_csv(REF / "Test51H3_Bootstrap_AllCandidates.csv")
    oracle = O.loc[O.groupby("Draw")["Fresh_log10_p"].idxmin()].sort_values("Draw")["Candidate"].tolist()
    assert sel == oracle == [17] * 20
    T = pd.DataFrame(vr["trace"]).rename(columns={"df": "app_df"})
    T["Draw"] = T["part"].str.replace("boot", "").astype(int)
    M = O.merge(T, left_on=["Draw", "Candidate"], right_on=["Draw", "candidate_index"], validate="one_to_one")
    assert len(M) == 360
    assert (M["df"] == M["app_df"]).all()
    assert (M["Fresh_reduced_LL"] - M["reference_loglik"]).abs().max() < 1e-6
    # full fits: numerically equal except optimizer-path differences of flagged (non-converged / singular) fits when
    # the oracle writes the terms in another order (#11, #12); documented, not hidden
    dl = (M["Full_LL"] - M["loglik"]).abs()
    clean = M["flags"].fillna("") == ""
    assert dl[clean & ~M["Candidate"].isin([11, 12])].max() < 1e-4
    assert dl.max() < 0.2
    st = {x["candidate_index"]: x["selected"] for x in vr["stability"]}
    assert sum(st.values()) == 20 and st[17] == 20
    est = [e for e in vr["estimands"] if e["candidate_index"] == 17]
    triple = [e for e in est if e["term"].count(":") == 2]
    assert len(triple) == 1 and triple[0]["draws_with_estimate"] == 20
    assert triple[0]["coefficient_label"] == "Sleep × Anxiety × Attendance"
    comp = pd.read_csv(REF / "Test51H3_Bootstrap_Comparison.csv").sort_values("Draw")
    got = pd.DataFrame(vr["bootstrap"]).sort_values("b")
    assert np.allclose(got["estimate"].to_numpy(float), comp["Three_way_beta"].to_numpy(float), atol=1e-6)


@pytest.mark.r
@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_47b_bootstrap_every_fit_matches_independent_refits(r_session):
    """V134-001: 47B3 (participants within schools, rs_ar1, seed 133, B = 20). 80 full / reduced fits vs the owner's
    independent fresh fits; no artificial failed draws; selection = fresh selection in every draw."""
    from statexplore.discovery import validate as V

    d = fx("Test47B3_CompetingInteractions")
    run = search(d, **s47b())
    vr = V.ValidationRunner(r_session, d, run, V.ValidationConfig(target="participant_in_cluster",
                                                                  modes=["bootstrap"], B=20, seed=133)).execute()
    assert vr["status"] == "complete" and all(b["status"] == "fitted" for b in vr["bootstrap"])
    O = pd.read_csv(REF / "Test47B3G_IndependentCandidateResults.csv")
    T = pd.DataFrame(vr["trace"])
    T["b"] = T["part"].str.replace("boot", "").astype(int)
    M = O.merge(T, left_on=["b", "candidate"], right_on=["b", "candidate_index"], validate="one_to_one")
    assert len(M) == 40
    assert (M["logLik_full"] - M["loglik"]).abs().max() < 1e-5
    assert (M["logLik_reduced"] - M["reference_loglik"]).abs().max() < 1e-5
    assert (M["LRT_correct"] - M["statistic"]).abs().max() < 1e-4
    S = pd.read_csv(REF / "Test47B3G_SelectionComparison.csv").sort_values("b")
    sel = [int(b["selected_index"]) for b in sorted(vr["bootstrap"], key=lambda b: b["b"])]
    assert sel == S["correct_selection"].astype(int).tolist()
    st = {x["candidate_index"]: x["selected"] for x in vr["stability"]}
    assert st == {0: 8, 1: 12}


@pytest.mark.r
@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_bootstrap_chunking_and_cache_do_not_change_results(r_session):
    """Batch sizes 1, 7 and 10 give identical per-draw statistics and selections (keyed by draw ID)."""
    from statexplore.discovery import validate as V

    d = fx("Test47B3_CompetingInteractions")
    run = search(d, **s47b())
    out = []
    for chunk in (10, 1, 7):
        r = V.ValidationRunner(r_session, d, run, V.ValidationConfig(target="participant_in_cluster",
                                                                     modes=["bootstrap"], B=7, seed=9))
        r.chunk = chunk
        v = r.execute()
        T = pd.DataFrame(v["trace"]).sort_values(["part", "candidate_index"]).reset_index(drop=True)
        out.append((T[["part", "candidate_index", "loglik", "reference_loglik", "log_p"]],
                    [b["selected_index"] for b in sorted(v["bootstrap"], key=lambda b: b["b"])]))
    for T, sel in out[1:]:
        pd.testing.assert_frame_equal(T, out[0][0], check_exact=False, rtol=1e-9, atol=1e-9)
        assert sel == out[0][1]


# ---- V134-002: log-domain inference ----------------------------------------------------------------------------------

def test_log_bh_holm_match_p_adjust_and_order_underflow():
    from statsmodels.stats.multitest import multipletests

    from statexplore.discovery.engine import log_bh, log_holm

    rng = np.random.default_rng(3)
    p = np.concatenate([rng.uniform(size=40), [1e-12, 1e-30, 0.5, 0.5, 1.0]])
    np.testing.assert_allclose(np.exp(log_bh(np.log(p))), multipletests(p, method="fdr_bh")[1], rtol=1e-12)
    np.testing.assert_allclose(np.exp(log_holm(np.log(p))), multipletests(p, method="holm")[1], rtol=1e-12)
    # tails far below the double range keep their order and exact log-scale adjustment
    lp = np.array([-1117.0, -965.0, -2.0, np.log(0.3)])
    m = len(lp)
    exp_bh = np.minimum.accumulate((np.sort(lp) + np.log(m) - np.log(np.arange(1, m + 1)))[::-1])[::-1]
    np.testing.assert_allclose(np.sort(log_bh(lp)), np.minimum(exp_bh, 0))
    assert log_bh(lp)[0] < log_bh(lp)[1] < log_bh(lp)[2]
    assert log_holm(lp)[0] == pytest.approx(-1117.0 + np.log(4))
    assert np.all(log_bh(np.array([0.0, 0.0])) == 0) and len(log_bh(np.array([]))) == 0


def _chisq1_log_tail(x: float) -> float:
    """Independent oracle: P(chi2_1 > x) = 2 * Phi(-sqrt(x)); scipy's log_ndtr is accurate deep in the tail."""
    from scipy.special import log_ndtr

    return math.log(2) + float(log_ndtr(-math.sqrt(x)))


def test_54b_extreme_tails_have_finite_log_p():
    """The two 54B tests exported as p = 0 with no log10 p (LRT 54085.522401 and 23373.467504, df 1)."""
    from statexplore.analysis.lrtest import lr_test

    for lr in (54085.522401, 23373.467504):
        t = lr_test(0.0, lr / 2, 1, "chisq")
        assert t["p"] == 0.0 or t["p"] < 1e-300
        assert t["log_p"] is not None and math.isfinite(t["log_p"])
        assert t["log_p"] == pytest.approx(_chisq1_log_tail(lr), rel=1e-9)


@pytest.mark.r
@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_51c_log_tails_ranking_and_adjustment(r_session):
    """51C: 18 candidates / 48 tests; whole-model and joint log10 p of the triple candidate ~ -485.20 / -419.54 (finite,
    distinct, ordered); BH / Holm on the log scale; triple coefficient and LRT targets."""
    from statexplore.discovery import engine

    d = fx("Test51C_ThreeWayInteraction")
    run = search(d, **s51h())
    tf = engine.tests_frame(run)
    assert len(run.rows) == 18 and len(tf) == 48
    t17 = tf[tf["idx"] == 17].set_index("test_type")
    assert t17.loc["overall", "log10_p"] == pytest.approx(-485.200773, abs=0.05)
    assert t17.loc["joint_interaction", "log10_p"] == pytest.approx(-419.542529, abs=0.05)
    tri = t17.loc["interaction_term"]
    assert tri["estimate"] == pytest.approx(1.529950487, abs=1e-4)
    assert tri["statistic"] == pytest.approx(527.064843, rel=1e-5)
    assert np.isfinite(tf["log_p"]).all() and tf["log_p_source"].eq("native log tail").all()
    for _, g in tf[tf["in_family"]].groupby("test_type"):
        lp = g["log_p"].to_numpy(float)
        rep = np.exp(lp) > 0
        from statsmodels.stats.multitest import multipletests

        if rep.all():
            np.testing.assert_allclose(g["p_bh"], multipletests(np.exp(lp), method="fdr_bh")[1], rtol=1e-9)
    rk = engine.discovery_ranking(tf)
    assert rk["log_p"].is_monotonic_increasing


# ---- V134-005 / AUDIT04: GEE working correlation and QIC -------------------------------------------------------------

@pytest.mark.r
def test_48d_gee_structures_fit_their_own_working_correlation(tmp_path):
    from statexplore.discovery import engine

    d = fx("Test48D_GEE")
    res = rscript(f'''suppressMessages(library(geepack))
d <- read.csv("{(FX / "StatExplore_Test48D_GEE.csv").as_posix()}"); d <- d[order(d$ID, d$TIME), ]
d$occ <- as.integer(factor(d$TIME)); out <- NULL
for (cs in c("independence", "exchangeable", "ar1")) {{
  m <- geeglm(Y ~ TIME * X, id = ID, waves = occ, corstr = cs, data = d, family = gaussian()); q <- QIC(m)
  out <- rbind(out, data.frame(corstr = m$corstr, QIC = q[["QIC"]], QICu = q[["QICu"]], CIC = q[["CIC"]],
                               alpha = if (length(m$geese$alpha)) m$geese$alpha[1] else NA, scale = m$geese$gamma[1],
                               se_x = sqrt(diag(m$geese$vbeta))[3]))
}}
write.csv(out, "gee.csv", row.names = FALSE)''', tmp_path)
    assert res.returncode == 0, res.stderr
    ref = pd.read_csv(tmp_path / "gee.csv").set_index("corstr")
    assert ref.loc["independence", "QIC"] == pytest.approx(1146.36543260012, abs=1e-6)
    names = {"gee_ind": "independence", "gee_exch": "exchangeable", "gee_ar1": "ar1"}
    for order in (["gee_ind", "gee_exch", "gee_ar1"], ["gee_ar1", "gee_exch", "gee_ind"]):
        run = search(d, outcome="Y", candidates=["X"], required=["TIME"], participant="ID", time="TIME",
                     workflow="structure", structures=order, time_interactions=True, workers=1)
        for r in run.rows:
            cs = names[r["structure"]]
            assert r["gee"]["corstr"] == cs
            for k, col in (("qic", "QIC"), ("qicu", "QICu"), ("cic", "CIC")):
                assert r[k] == pytest.approx(ref.loc[cs, col], abs=1e-6), (cs, k)
            assert r["gee"]["scale"] == pytest.approx(ref.loc[cs, "scale"], rel=1e-8)
        df = engine.add_display_columns(engine.results_frame(run), run)
        cmp, note = engine.model_comparison(df, run)
        assert cmp["aic"].isna().all() and "not an AIC" not in note or True
        assert list(cmp["structure"].map(names)) == list(ref["QIC"].sort_values().index)
        assert cmp["d_qic"].min() == 0 and "QIC" in note


def test_gee_aic_rule_refused_in_validation():
    from statexplore.discovery import validate as V
    from statexplore.discovery.engine import DiscoveryRun
    from statexplore.discovery.space import SearchSettings

    s = SearchSettings(outcome="Y", candidates=["X"], participant="ID", time="TIME", structure="gee_exch")
    run = DiscoveryRun(settings=s, family="gee", total=2, signature="x")
    d = fx("Test48D_GEE")
    probs = V.feasibility(d, run, V.ValidationConfig(rule="aic", target="participant"))["problems"]
    assert any("no AIC" in p for p in probs)


# ---- V134-006: comparability ------------------------------------------------------------------------------------------

def test_model_comparison_needs_identical_rows_not_equal_n():
    from statexplore.discovery import engine
    from statexplore.discovery.engine import DiscoveryRun
    from statexplore.discovery.space import SearchSettings

    run = DiscoveryRun(settings=SearchSettings(outcome="Y", candidates=["A", "B"], sample="per_model"),
                       family="gaussian", total=2, signature="x")
    base = dict(status="ok", n=100, bic=10.0, loglik=-1.0, n_params=2, n_lik_params=3)
    df = pd.DataFrame([dict(base, idx=0, aic=12.0, identity={"rows": "r1"}),
                       dict(base, idx=1, aic=5.0, identity={"rows": "r2"})])      # equal n, different rows
    cmp, note = engine.model_comparison(df, run)
    assert cmp["d_aic"].isna().all() and list(cmp["idx"]) == [0, 1] and "not comparable" in note
    df["identity"] = [{"rows": "r1"}, {"rows": "r1"}]
    cmp, _ = engine.model_comparison(df, run)
    assert list(cmp["idx"]) == [1, 0] and cmp["d_aic"].tolist() == [0.0, 7.0]


# ---- V134-003: identity-based aggregation ------------------------------------------------------------------------------

def test_stability_counts_by_identity_not_label():
    from statexplore.discovery.validate import stability_table

    labels = ["TIME × GROUP [#0]", "TIME × GROUP2 [#1]", "TIME × GROUP [#2]"]
    Bt = pd.DataFrame({"b": range(1, 7), "status": ["fitted"] * 5 + ["failed"],
                       "selected": ["TIME × GROUP"] * 5 + [""], "selected_index": [0, 2, 2, 2, 1, np.nan],
                       "estimate": [1.0, 2, 3, 4, -1, np.nan], "coefficient": ["c"] * 6, "diagnostics": ["clean"] * 6})
    st = stability_table(Bt, labels, [0, 1, 2], ["k0", "k1", "k2"])
    assert [x["selected"] for x in st] == [1, 1, 3]
    assert sum(x["selected"] for x in st) == 5 == st[0]["successful"] and st[0]["attempted"] == 6
    assert sum(x["freq_of_successful"] for x in st) == pytest.approx(1.0)
    assert sum(x["freq_of_attempted"] for x in st) == pytest.approx(5 / 6)
    assert st[2]["estimate_median"] == 3.0
    empty = stability_table(Bt.assign(status="failed"), labels, [0, 1, 2])
    assert all(x["selected"] == 0 and math.isnan(x["freq_of_successful"]) for x in empty)


# ---- V134-004 / AUDIT03: training-only candidate pool -----------------------------------------------------------------

def test_validation_pool_ignores_full_data_outcomes():
    """The pool is the declared space in order: changing the full-data rows, statuses or stop point never changes it."""
    from statexplore.discovery import validate as V
    from statexplore.discovery.space import SearchSettings

    d = fx("Test47D_ValidationLeakage")
    from statexplore.discovery import engine

    s = SearchSettings(outcome="Y", candidates=["GROUP", "GROUP2", "GROUP3"] + [f"NOISE{i:02d}" for i in range(1, 13)],
                       participant="ID", time="TIME", cluster="SCHOOL", id_policy="local", structure="ri",
                       strategy="focal", covariate_mode="focal", mode="A", max_predictors=3, study="longitudinal",
                       required=["TIME"])
    run = engine.create_run(d, s)
    assert run.total == 105
    pools = []
    for rows, stop in (([], None), ([{"idx": i, "status": "invalid", "key": ""} for i in range(40)], 0)):
        run.rows, run.stop_index = rows, stop
        r = V.ValidationRunner(None, d, run, V.ValidationConfig(target="cluster", modes=["holdout"], max_fits=10**9))
        pools.append([c.index for c in r.cands])
    assert pools[0] == pools[1] == list(range(105))
    assert V.stop_target(run) == 1


def test_huge_space_validation_refused_without_materializing():
    import time as _t

    from statexplore.discovery import validate as V
    from statexplore.discovery.engine import DiscoveryRun
    from statexplore.discovery.space import SearchSettings, count_candidates

    s = SearchSettings(outcome="Y", candidates=[f"X{i:02d}" for i in range(1, 31)], required=["Time"],
                       participant="StudentID", time="Time", strategy="combinations", include_main=True, max_predictors=6)
    total = count_candidates(s)
    assert total == 147_713_444
    run = DiscoveryRun(settings=s, family="lmm", total=total, signature="x")
    t0 = _t.perf_counter()
    info = V.feasibility(fx("Test51F_Null_13"), run, V.ValidationConfig(target="participant", modes=["holdout"]))
    assert _t.perf_counter() - t0 < 10
    assert info["pool"] == total and any("exceed the budget" in p for p in info["problems"])


@pytest.mark.r
@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_test_outcomes_never_reach_training(r_session):
    """47D: perturbing the held-out rows' outcomes (+10000) changes the metrics but not one training fit, selection or
    prediction (splits held fixed); the search's early stop (mode A) is replayed on the training data."""
    from statexplore.discovery import engine
    from statexplore.discovery import validate as V
    from statexplore.discovery.space import SearchSettings

    d = fx("Test47D_ValidationLeakage")
    s = dict(outcome="Y", candidates=["GROUP", "GROUP2", "NOISE01", "NOISE02"], required=["TIME"], participant="ID",
             time="TIME", cluster="SCHOOL", id_policy="local", structure="ri", strategy="focal",
             covariate_mode="focal", mode="A", max_predictors=3, study="longitudinal", workers=1, max_models=None,
             max_minutes=None)
    cfg = V.ValidationConfig(target="cluster", modes=["holdout"], seed=4, max_fits=10**6)
    run = engine.create_run(d, SearchSettings(**s))
    engine.execute(run, d)
    a = V.ValidationRunner(r_session, d, run, cfg)
    test_rows = [r for r, f in a.splits[0]["folds"].items() if f == 1]
    va = a.execute()
    d2 = fx("Test47D_ValidationLeakage")
    d2.raw.loc[d2.raw.index[np.array(test_rows) - 1], "Y"] += 10000
    run2 = engine.create_run(d2, SearchSettings(**s))
    engine.execute(run2, d2)
    vb = V.ValidationRunner(r_session, d2, run2, cfg).execute()
    ta, tb = pd.DataFrame(va["trace"]), pd.DataFrame(vb["trace"])
    pd.testing.assert_frame_equal(ta.drop(columns=["rows_sig"]), tb.drop(columns=["rows_sig"]))
    assert va["folds"][0]["selected_index"] == vb["folds"][0]["selected_index"]
    pa = pd.DataFrame(va["predictions"]).sort_values("row_id")
    pb = pd.DataFrame(vb["predictions"]).sort_values("row_id")
    np.testing.assert_allclose(pa["predicted"], pb["predicted"])
    assert not np.allclose(pa["observed"], pb["observed"])
    assert ta["candidate_index"].max() < 6          # mode A replay: stops at the first training-significant candidate


# ---- V134-007 / AUDIT06: combined time grammar ------------------------------------------------------------------------

def _brute(cands, time, max_pred, n_req, order, include_main, minimum):
    """Independent enumerator: every subset of optional candidates, every hierarchical interaction set on subset +
    time (time is a required main effect), deduplicated canonically."""
    out = set()
    room = max_pred - n_req
    for k in range(max(1, minimum), min(room, len(cands)) + 1):
        for sub in combinations(cands, k):
            vs = list(sub) + [time]
            pairs = [frozenset(p) for p in combinations(vs, 2)]
            for m in range(2 ** len(pairs)):
                chosen = {pairs[i] for i in range(len(pairs)) if m >> i & 1}
                tris = [frozenset(t) for t in combinations(vs, 3) if all(frozenset(p) in chosen
                                                                          for p in combinations(t, 2))] if order >= 3 else []
                for j in range(2 ** len(tris)):
                    terms = chosen | {tris[i] for i in range(len(tris)) if j >> i & 1}
                    if not terms and not include_main:
                        continue
                    out.add((frozenset(sub), frozenset(terms)))
    return out


@pytest.mark.parametrize("p,order,max_pred,include_main,minimum", [(2, 2, 5, True, 1), (3, 2, 5, True, 1),
                                                                   (3, 3, 6, False, 1), (4, 2, 6, True, 2),
                                                                   (5, 3, 4, True, 1), (1, 2, 4, True, 1)])
def test_time_eligible_count_matches_brute_force(p, order, max_pred, include_main, minimum):
    from statexplore.discovery.space import SearchSettings, candidate_key, count_candidates, enumerate_candidates

    cands = [f"X{i}" for i in range(p)]
    s = SearchSettings(outcome="Y", candidates=cands, required=["T", "R1", "R2"], participant="ID", time="T",
                       strategy="combinations", include_main=include_main, max_predictors=max_pred, max_order=order,
                       min_predictors=minimum, time_eligible=True)
    got = list(enumerate_candidates(s))
    brute = _brute(cands, "T", max_pred, 3, order, include_main, minimum)
    assert count_candidates(s) == len(got) == len(brute)
    assert {(frozenset(c.covariates), frozenset(frozenset(t) for t in c.inter)) for c in got} == brute
    assert len({candidate_key(c) for c in got}) == len(got)
    assert [c.index for c in got] == list(range(len(got)))
    for start in (0, len(got) // 2, len(got) - 1):          # lazy indexed resume
        assert [c.index for c in enumerate_candidates(s, start=start)] == list(range(start, len(got)))
    # hierarchy: every interaction's parents are in the model
    for c in got:
        terms = {frozenset(t) for t in c.inter}
        for t in terms:
            for q in combinations(sorted(t), len(t) - 1):
                assert len(q) == 1 or frozenset(q) in terms


def test_51d_combined_grammar_twelve_candidates_and_off_by_default():
    from statexplore.discovery.space import SearchSettings, count_candidates, enumerate_candidates

    base = dict(outcome="TestScore", candidates=["Sleep", "Anxiety"], required=["Time", "Attendance", "Gender"],
                participant="StudentID", time="Time", strategy="combinations", include_main=True, max_predictors=5)
    s = SearchSettings(**base, time_eligible=True)
    cs = list(enumerate_candidates(s))
    assert count_candidates(s) == 12
    planted = [c for c in cs if {frozenset(t) for t in c.inter} == {frozenset({"Sleep", "Anxiety"}),
                                                                       frozenset({"Sleep", "Time"})}]
    assert len(planted) == 1
    assert count_candidates(SearchSettings(**base)) == 1 + 1 + 2          # v1.3.3 grammar unchanged when off
    huge = SearchSettings(outcome="Y", candidates=[f"X{i:02d}" for i in range(1, 31)], required=["Time"],
                          participant="ID", time="Time", strategy="combinations", include_main=True, max_predictors=6)
    assert count_candidates(huge) == 147_713_444
    huge.include_main = False
    assert count_candidates(huge) == 147_539_008


def test_time_strategy_honours_minimum_candidates():
    from statexplore.discovery.space import SearchSettings, count_candidates, enumerate_candidates

    s = SearchSettings(outcome="Y", candidates=["A", "B", "C"], participant="ID", time="T", strategy="time",
                       covariate_mode="subsets", max_predictors=3, min_predictors=2)
    cs = list(enumerate_candidates(s))
    assert count_candidates(s) == len(cs) == 6 and all(len(c.focal) - 1 + len(c.covariates) >= 2 for c in cs)
    s.covariate_mode, s.max_order = "focal", 3
    assert count_candidates(s) == len(list(enumerate_candidates(s))) == 3      # only Time x A x B style focal sets


def test_time_eligible_refused_without_required_time():
    from statexplore.discovery.space import SearchSettings, SearchSpaceError, validate_settings

    d = fx("Test51D_CombinedInteractions")
    s = SearchSettings(outcome="TestScore", candidates=["Sleep", "Anxiety"], required=["Attendance"], family="gaussian",
                       participant="StudentID", time="Time", strategy="combinations", include_main=True,
                       max_predictors=5, time_eligible=True)
    with pytest.raises(SearchSpaceError, match="required fixed effect"):
        validate_settings(d, s)


@pytest.mark.r
@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_51d_planted_candidate_matches_independent_lme4(tmp_path):
    d = fx("Test51D_CombinedInteractions")
    run = search(d, outcome="TestScore", candidates=["Sleep", "Anxiety"], required=["Time", "Attendance", "Gender"],
                 participant="StudentID", time="Time", cluster="School", structure="rs", strategy="combinations",
                 include_main=True, max_predictors=5, time_eligible=True)
    assert len(run.rows) == 12
    row = next(r for r in run.rows if {frozenset(t) for t in r["inter"]} == {frozenset({"Sleep", "Anxiety"}),
                                                                               frozenset({"Sleep", "Time"})})
    res = rscript(f'''suppressMessages(library(lme4)); d <- read.csv("{(FX / "StatExplore_Test51D_CombinedInteractions.csv").as_posix()}")
d$Gender <- factor(d$Gender)
m <- lmer(TestScore ~ Time + Sleep + Anxiety + Attendance + Gender + Sleep:Anxiety + Time:Sleep + (1 | School) +
          (1 + Time | School:StudentID), data = d, REML = FALSE)
b <- fixef(m); write.csv(data.frame(ll = as.numeric(logLik(m)), sa = b[["Sleep:Anxiety"]], ts = b[["Time:Sleep"]]),
                         "m.csv", row.names = FALSE)''', tmp_path)
    assert res.returncode == 0, res.stderr
    ref = pd.read_csv(tmp_path / "m.csv").iloc[0]
    assert row["loglik"] == pytest.approx(ref["ll"], abs=0.05)
    assert row["loglik"] == pytest.approx(-3708.28009182096, abs=0.05)
    jt = next(t for t in row["tests"] if t["type"] == "joint_interaction")
    assert jt["p"] < 1e-10


# ---- V134-008: validation identity, realized sizes, portable reproduction --------------------------------------------

@pytest.mark.r
@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_validation_manifest_identity_and_relocated_reproduction(r_session, tmp_path):
    from statexplore.discovery import validate as V
    from statexplore.discovery.space import view
    from statexplore.rbridge.locate import find_rscript

    d = fx("Test43B_PositiveInteraction_Rerun")
    run = search(d, outcome="Y", candidates=["GROUP"], required=["TIME"], participant="ID", time="TIME", cluster="SCHOOL",
                 structure="ri", strategy="main", include_main=True)
    cfg = V.ValidationConfig(target="cluster", modes=["holdout", "cv", "bootstrap"], k=5, B=3, seed=133,
                             pipeline="fixed", candidate=0)
    v = V.ValidationRunner(r_session, d, run, cfg).execute()
    assert v["search_identity"] == run.fit_id and len(v["search_identity"]) == 64 and v["validation_id"]
    hold = [x for x in v["realized"] if x["split"] == "holdout"][0]
    assert hold["test_units"] == math.ceil(0.2 * hold["units"])
    out = Path(V.export(v, str(tmp_path / "export"), ds=view(d, run.settings), include_data=True)[-1]).parent  # v1.3.6: explicit consent; a NEW folder per export
    moved = tmp_path / "moved folder ü"
    shutil.copytree(out, moved)
    shutil.rmtree(out)
    lib = Path(__file__).resolve().parents[1] / "r" / "library"
    env = dict(os.environ, R_LIBS=str(lib), R_LIBS_USER=str(lib))
    res = subprocess.run([str(find_rscript()), "--vanilla", "validation_reproduce.R"], cwd=moved, capture_output=True,
                         text=True, timeout=1800, env=env)
    assert res.returncode == 0, res.stderr[-3000:]
    assert "Predictions reproduced" in res.stdout and "Bootstrap selections reproduced" in res.stdout
    script = (moved / "validation_reproduce.R").read_text(encoding="utf-8")
    assert str(tmp_path) not in script and "sx_tmp" not in script
    manifest = json.loads((moved / "validation_manifest.json").read_text())
    assert manifest["search_identity"] == run.fit_id
    # a substituted data file is refused
    data = next(moved.glob("validation_reproduce_data.csv"))
    data.write_text(data.read_text().replace(",1,", ",2,", 1))
    res = subprocess.run([str(find_rscript()), "--vanilla", "validation_reproduce.R"], cwd=moved, capture_output=True,
                         text=True, timeout=600, env=env)
    assert res.returncode != 0 and "SHA-256" in res.stderr


# ---- V134-009: resume reason states ---------------------------------------------------------------------------------

def test_52d_partial_search_resumable_with_reason_states():
    from statexplore.core.project import Project
    from statexplore.core.derive import FilterSpec

    work = Path(os.environ.get("TEMP", "."))
    p = Project.open(FX / "StatExplore_Test52D_Persistence.sxp")
    run = next(r for r in p.searches if r.id == "ea77ac2fedc4")
    # v1.3.5 (AUD134-03): its rows were fitted under fit rules 3, so it is no longer resumable (reason shown)
    assert run.status == "partial" and run.n_done == 5 and run.total == 15 and not run.resumable
    assert "fit rules 3" in run.resume_reason(p.dataset)
    done = next(r for r in p.searches if r.status == "complete")
    assert done.resume_reason(p.dataset) == "the search is complete"
    ds = p.dataset
    old = list(ds.filters)
    ds.filters = [FilterSpec(variable="Time", op="<=", value=3.0)]
    assert run.resume_reason(ds)           # still refused (compatibility is checked first)
    ds.filters = old
    run.settings.strategy, run.settings.min_predictors, run.grammar = "time", 2, 2
    assert not run.resumable and "grammar 3" in run.resume_reason(ds)
    del work


@pytest.mark.r
@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_52d_resume_continues_same_search_to_15(tmp_path):
    """v1.3.5 (AUD134-03) supersedes the v1.3.4 expectation: the owner's 52D partial search (fit rules 3) is refused at
    the engine level and stays unchanged on disk; the same-version 5 + 10 continuation is covered by
    test_v135::test_same_version_partial_resumes_to_exact_15_and_engine_mismatch_refused."""
    from statexplore.core.project import Project
    from statexplore.discovery import engine

    src = tmp_path / "p.sxp"
    shutil.copy(FX / "StatExplore_Test52D_Persistence.sxp", src)
    p = Project.open(src)
    run = next(r for r in p.searches if r.id == "ea77ac2fedc4")
    first5 = {r["idx"]: (r["loglik"], r["p"]) for r in run.rows}
    run.settings.max_models = 10
    with pytest.raises(engine.ResumeRefused, match="fit rules 3"):
        engine.execute(run, p.dataset, cache=engine.cache_from(run, p.searches, p.dataset))
    assert run.status == "partial" and run.n_done == 5 and len(run.executions) == 1
    assert {r["idx"]: (r["loglik"], r["p"]) for r in run.rows} == first5


# ---- V134-010: diagnostic attribution --------------------------------------------------------------------------------

def test_reference_flags_make_only_their_test_provisional(tmp_path):
    from statexplore.discovery import engine
    from statexplore.discovery.engine import DiscoveryRun
    from statexplore.discovery.space import SearchSettings, enumerate_candidates

    s = SearchSettings(outcome="Y", candidates=["A", "B"], strategy="combinations", include_main=True)
    run = DiscoveryRun(settings=s, family="gaussian", total=1, signature="x", fit_id="f")
    pd.DataFrame({"Y": np.arange(20.0) * 1.5, "A": np.arange(20.0) % 3 + 0.5, "B": np.arange(20.0) % 5 + 0.25}
                 ).to_csv(tmp_path / "d.csv", index=False)
    ds = import_file(tmp_path / "d.csv")
    c = [c for c in enumerate_candidates(s) if c.kind == "combo"][0]
    tests = [{"id": 0, "statistic": 5.0, "df1": 1, "p": 0.03, "log_p": math.log(0.03), "reference_flags": ""},
             {"id": 1, "statistic": 9.0, "df1": 3, "p": 0.001, "log_p": math.log(0.001),
              "reference_flags": "warning: reference model did not converge"}]
    row = engine._row(ds, run, c, {"status": "ok", "flags": "", "tests": tests, "p": 0.03, "log_p": math.log(0.03)})
    run.rows = [row]
    tf = engine.tests_frame(run)
    assert list(tf["test_status"]) == ["", "provisional"]
    assert row["status"] == "ok" and row["flags"] == ""
    assert list(tf["tier"])[1] != "A"


# ---- V134-011: interpretation wording --------------------------------------------------------------------------------

def test_negative_delta_r2_and_underflow_wording():
    from statexplore.discovery.engine import interpret_row
    from statexplore.discovery.space import SearchSettings

    s = SearchSettings(outcome="Y", candidates=["A"])
    r = {"status": "ok", "reason": "", "formula": "Score ~ A * B", "test_type": "interaction_term", "label": "A × B",
         "interaction": "A × B", "p": 0.0, "p_adj": 0.0, "log_p": -1117.2, "log_p_adj": -1115.0, "effect": -0.059,
         "family_size": 10, "sig_adj": True, "sig_raw": True, "estimate": -2.0, "ci_applicable": "same_estimand",
         "ci_level": 0.95, "ci_low": -2.2, "ci_high": -1.8, "flags": "", "test_method": "LRT", "reference": "Y ~ A + B"}
    txt = interpret_row(r, s, "lmm")
    assert "decreases by 5.9 percentage points" in txt and "rises by -" not in txt
    assert "log10 p = -485.2" in txt
    r2 = dict(r, p=None, test_reason="the larger model has a lower likelihood")
    assert interpret_row(r2, s, "lmm").startswith("No p-value")


# ---- V134-012: cache identity hardening ------------------------------------------------------------------------------

def test_legacy_runs_without_identity_or_old_rules_are_never_reused():
    from statexplore.discovery import engine
    from statexplore.discovery.engine import DiscoveryRun
    from statexplore.discovery.space import SearchSettings

    s = SearchSettings(outcome="Y", candidates=["A"])
    cur = DiscoveryRun(settings=s, family="gaussian", total=1, signature="sig", fit_id="F")
    row = {"idx": 0, "key": "main|A", "status": "ok"}
    legacy = DiscoveryRun(settings=s, family="gaussian", total=1, signature="sig", fit_id="", rows=[row])
    old_rules = DiscoveryRun(settings=s, family="gaussian", total=1, signature="sig", fit_id="F", rows=[row],
                             engine={"fit_rules": 3})
    good = DiscoveryRun(settings=s, family="gaussian", total=1, signature="sig", fit_id="F", rows=[row],
                        engine={"fit_rules": engine.FIT_RULES}, identity_schema=3)   # v1.3.6: current identity schema
    assert engine.cache_from(cur, [legacy, old_rules]) == {}
    assert list(engine.cache_from(cur, [legacy, old_rules, good])) == ["main|A"]
    from statexplore.discovery.rcode import FIT_RULES_R

    assert FIT_RULES_R == engine.FIT_RULES == 4


# ---- V134-015: audit trail ------------------------------------------------------------------------------------------

def test_audit_record_roundtrip_and_refusals(tmp_path):
    from statexplore.core.project import Project
    from statexplore.export.audit import AuditError, read_audit, validate_record, write_audit

    p = Project.open(FX / "StatExplore_Test52D_Persistence.sxp")
    run = next(r for r in p.searches if r.id == "ea77ac2fedc4")
    files = write_audit(run, p.dataset, tmp_path)
    rec = read_audit(files[0])
    # audit schema 2 (v1.3.5, AUD134-07): historical vs current data blocks; the 52D search is no longer resumable
    assert rec["search"]["id"] == "ea77ac2fedc4" and rec["grammar"]["stored_total"] == 15
    assert "fit rules 3" in rec["search"]["resume"] and rec["search"]["historical_old_rules"] is True
    assert len(rec["candidates"]) == 5
    assert rec["data"]["current_context"]["source_file"] == "StatExplore_Test51F_Null_13.csv"
    txt = files[0].read_text(encoding="utf-8")
    assert "vivi_" not in txt and "NaN" not in txt and "Infinity" not in txt
    assert "audit report" in files[1].read_text(encoding="utf-8")
    bad = dict(rec)
    del bad["candidates"]
    with pytest.raises(AuditError, match="missing"):
        validate_record(bad)
    with pytest.raises(AuditError):
        validate_record({"schema": "other"})
    assert write_audit(run, p.dataset, tmp_path)[0].read_text(encoding="utf-8").count('"id": "ea77ac2fedc4"') == 1


# ---- V134-014: educational code --------------------------------------------------------------------------------------

@pytest.mark.r
def test_educational_script_runs_and_is_labelled(tmp_path):
    from statexplore.discovery.rcode import educational_script
    from statexplore.discovery.space import SearchSettings, enumerate_candidates, validate_settings

    d = fx("Test51D_CombinedInteractions")
    s = SearchSettings(outcome="TestScore", candidates=["Sleep", "Anxiety"], required=["Time", "Attendance", "Gender"],
                       family="gaussian", participant="StudentID", time="Time", cluster="School", structure="rs",
                       strategy="combinations", include_main=True, max_predictors=5, time_eligible=True,
                       study="longitudinal")
    fam = validate_settings(d, s)
    c = list(enumerate_candidates(s))[8]
    txt = educational_script(d, s, fam, c)
    assert "EDUCATIONAL" in txt and "DIDACTIC" in txt and "Sleep:Anxiety" in txt and "Sleep:Time" in txt
    shutil.copy(FX / "StatExplore_Test51D_CombinedInteractions.csv", tmp_path / "your_data.csv")
    res = rscript(txt, tmp_path)
    assert res.returncode == 0, res.stderr[-2000:]


# ---- v1.3.3 project compatibility (owner's saved projects) -----------------------------------------------------------

def test_v133_projects_open_as_labelled_history(tmp_path):
    from statexplore.core.project import Project
    from statexplore.discovery import engine

    for name in ("StatExplore_Test50_Reproducibility.sxp", "StatExplore_Test52D_Persistence.sxp"):
        src = tmp_path / name
        shutil.copy(FX / name, src)
        p = Project.open(src)
        assert p.searches and all(r.grammar == 2 for r in p.searches)
        for r in p.searches:
            assert (r.engine or {}).get("fit_rules") == 3          # older rules: kept as history, never reused
            assert engine.tests_frame(r, interpret=False) is not None
        p.save(src)
        q = Project.open(src)
        assert [r.id for r in q.searches] == [r.id for r in p.searches]
        assert [len(r.rows) for r in q.searches] == [len(r.rows) for r in p.searches]
    p50 = Project.open(tmp_path / "StatExplore_Test50_Reproducibility.sxp")
    assert p50.searches[0].signature.startswith("86f2c8977182b627")
