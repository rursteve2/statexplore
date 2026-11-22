"""v1.2 longitudinal discovery (M §8, §15 E-I): explicit ID/time structure, Gaussian structures (lme4 RI/RS,
nlme AR(1)/CAR(1), mmrm), binary GLMM, ordinal CLMM and GEE — each discovery statistic checked against an
independent direct R fit — workflow B structure comparison (REML), group counts vs df, refusals."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statexplore.discovery.engine import create_run, execute, fit_identity, results_frame
from statexplore.discovery.engine import tests_frame as hypotheses
from statexplore.discovery.space import SearchSettings, SearchSpaceError, count_candidates, validate_settings

from test_discovery import make_ds


@pytest.fixture(scope="module")
def long_df():
    rng = np.random.default_rng(3)
    n_id, k = 120, 5
    df = pd.DataFrame({"pid": np.repeat(np.arange(n_id), k), "t": np.tile(np.arange(k), n_id)})
    df["trt"] = np.repeat(rng.integers(0, 2, n_id), k)
    df["x"] = rng.normal(size=len(df))
    u0, u1 = np.repeat(rng.normal(0, 1, n_id), k), np.repeat(rng.normal(0, .3, n_id), k)
    df["y"] = 1 + u0 + (0.2 + u1) * df["t"] + 0.4 * df["trt"] * df["t"] + 0.3 * df["x"] + rng.normal(size=len(df))
    eta = -0.5 + u0 + 0.5 * df["trt"] * df["t"]
    df["yb"] = rng.binomial(1, 1 / (1 + np.exp(-eta)))
    df["yo"] = pd.cut(df["y"], 4, labels=False) + 1
    return df


@pytest.fixture
def long_ds(tmp_path, long_df):
    ds = make_ds(tmp_path, long_df)
    ds.set_variable_field("yo", "measure", "ordinal")
    return ds


def LS(**kw):
    base = dict(outcome="y", candidates=["trt", "x"], family="gaussian", participant="pid", time="t",
                strategy="time", workers=1, max_models=None, max_minutes=None, max_predictors=3)
    base.update(kw)
    return SearchSettings(**base)


REF = r"""
d <- utils::read.csv({path!r})
d$pid <- factor(d$pid); d$occ <- as.integer(factor(d$t)); d$tf <- factor(d$t)
d <- d[order(d$pid, d$t), ]
{body}
"""


def r_ref(r_session, tmp_path, df, body, packages=()):
    path = (tmp_path / "ref.csv").as_posix()
    df.to_csv(path, index=False)
    return r_session.worker.run(REF.format(path=path, body=body), packages=list(packages), timeout=300).result


def trt_t(tf):
    return tf[tf["label"].isin(["t × trt", "trt × t"])].iloc[0]


# ---- F: Gaussian structures against direct R fits -----------------------------------------------

@pytest.mark.r
@pytest.mark.parametrize("structure,full,reduced", [
    ("ri", "lme4::lmer(y ~ t * trt + (1 | pid), data = d, REML = FALSE)",
     "lme4::lmer(y ~ t + trt + (1 | pid), data = d, REML = FALSE)"),
    ("rs", "lme4::lmer(y ~ t * trt + (1 + t | pid), data = d, REML = FALSE)",
     "lme4::lmer(y ~ t + trt + (1 + t | pid), data = d, REML = FALSE)"),
    ("ar1", "nlme::lme(y ~ t * trt, random = ~1 | pid, correlation = nlme::corAR1(form = ~ occ | pid), data = d, method = 'ML')",
     "nlme::lme(y ~ t + trt, random = ~1 | pid, correlation = nlme::corAR1(form = ~ occ | pid), data = d, method = 'ML')"),
    ("car1", "nlme::lme(y ~ t * trt, random = ~1 | pid, correlation = nlme::corCAR1(form = ~ t | pid), data = d, method = 'ML')",
     "nlme::lme(y ~ t + trt, random = ~1 | pid, correlation = nlme::corCAR1(form = ~ t | pid), data = d, method = 'ML')"),
    ("mmrm_us", "mmrm::mmrm(y ~ t * trt + us(tf | pid), data = d, reml = FALSE)",
     "mmrm::mmrm(y ~ t + trt + us(tf | pid), data = d, reml = FALSE)"),
])
def test_F_gaussian_structures_match_direct_R_fits(long_ds, long_df, r_session, tmp_path, structure, full, reduced):
    run = execute(create_run(long_ds, LS(structure=structure)), long_ds)
    assert run.family == "lmm" and run.status == "complete"
    row = trt_t(hypotheses(run))
    ref = r_ref(r_session, tmp_path, long_df, f"m1 <- {full}; m0 <- {reduced}\n"
                "sx_result <- list(lr = 2 * (as.numeric(stats::logLik(m1)) - as.numeric(stats::logLik(m0))))",
                ["lme4", "nlme", "mmrm"])
    assert row["statistic"] == pytest.approx(ref["lr"], rel=1e-4, abs=1e-6)
    assert row["n_groups"] == 120 and row["lrt_df"] == 1 and np.isnan(row["df_den"])   # I: groups are not df
    assert row["ci_applicable"] == "same_estimand" and row["ci_scale"] == "coefficient"


@pytest.mark.r
def test_F_workflow_B_compares_structures_with_REML(long_ds, long_df, r_session, tmp_path):
    s = LS(workflow="structure", structures=["ri", "rs", "nlme_ri", "ar1", "mmrm_cs", "mmrm_us"],
           time_interactions=True, strategy="combinations", required=["t"])
    assert count_candidates(s) == 6
    run = execute(create_run(long_ds, s), long_ds)
    rows = {r["structure"]: r for r in run.rows}
    ref = r_ref(r_session, tmp_path, long_df,
                "m <- lme4::lmer(y ~ trt + x + t + t:trt + t:x + (1 | pid), data = d, REML = TRUE)\n"
                "sx_result <- list(ll = as.numeric(stats::logLik(m)))", ["lme4"])
    assert rows["ri"]["loglik"] == pytest.approx(ref["ll"], rel=1e-7)
    # equivalent models share the REML likelihood and (uniform) AIC across engines
    assert rows["nlme_ri"]["aic"] == pytest.approx(rows["ri"]["aic"], rel=1e-7)
    assert rows["mmrm_cs"]["aic"] == pytest.approx(rows["ri"]["aic"], rel=1e-6)
    tf = hypotheses(run)
    assert set(tf["test_type"]) == {"structure"}
    slope = tf[tf["label"].str.contains("random time slope")].iloc[0]
    lr = 2 * (rows["rs"]["loglik"] - rows["ri"]["loglik"])
    from scipy.stats import chi2
    assert slope["statistic"] == pytest.approx(lr) and slope["p"] == pytest.approx(0.5 * chi2.sf(lr, 1) + 0.5 * chi2.sf(lr, 2))


@pytest.mark.r
def test_F_unestimable_random_slope_is_rejected(tmp_path):
    rng = np.random.default_rng(1)
    # every participant is observed twice but at a single time point: no within-person time variation
    df = pd.DataFrame({"pid": np.repeat(np.arange(40), 2), "t": np.repeat(np.arange(40) % 3, 2).astype(float),
                       "a": rng.normal(size=80)})
    df["y"] = rng.normal(size=80)
    ds = make_ds(tmp_path, df)
    run = execute(create_run(ds, LS(candidates=["a"], structure="rs", strategy="main", required=["t"],
                                    max_predictors=2)), ds)
    assert all(r["status"] == "invalid" and "random slope is not estimable" in r["reason"] for r in run.rows)


# ---- G / H: binary GLMM, ordinal CLMM, GEE ------------------------------------------------------

@pytest.mark.r
def test_G_binary_glmm_lr_event_coding_and_odds_ratios(long_ds, long_df, r_session, tmp_path):
    run = execute(create_run(long_ds, LS(outcome="yb", family="binomial", structure="glmer_ri", event=1)), long_ds)
    assert run.family == "glmm"
    row = trt_t(hypotheses(run))
    ref = r_ref(r_session, tmp_path, long_df,
                "m1 <- lme4::glmer(yb ~ t * trt + (1 | pid), data = d, family = binomial)\n"
                "m0 <- lme4::glmer(yb ~ t + trt + (1 | pid), data = d, family = binomial)\n"
                "sx_result <- list(lr = 2 * (as.numeric(logLik(m1)) - as.numeric(logLik(m0))), b = unname(lme4::fixef(m1)['t:trt']))",
                ["lme4"])
    assert row["statistic"] == pytest.approx(ref["lr"], rel=1e-4)
    assert row["estimate"] == pytest.approx(ref["b"], rel=1e-4)
    assert row["ci_scale"].startswith("log-odds (subject-specific)")
    rec = next(t for r in run.rows for t in r["tests"] if t["label"] in ("t × trt", "trt × t"))
    assert rec["odds_ratio"] == pytest.approx(np.exp(ref["b"]), rel=1e-4) and rec["or_low"] < rec["odds_ratio"]
    flipped = execute(create_run(long_ds, LS(outcome="yb", family="binomial", structure="glmer_ri", event=0)), long_ds)
    assert trt_t(hypotheses(flipped))["estimate"] == pytest.approx(-row["estimate"], rel=1e-3)
    assert fit_identity(long_ds, LS(outcome="yb", family="binomial", structure="glmer_ri", event=0), "glmm") != \
        fit_identity(long_ds, LS(outcome="yb", family="binomial", structure="glmer_ri", event=1), "glmm")


@pytest.mark.r
def test_H_ordinal_clmm_lr_matches_direct_fit(long_ds, long_df, r_session, tmp_path):
    run = execute(create_run(long_ds, LS(outcome="yo", family="ordinal", structure="clmm_ri",
                                         candidates=["trt"], max_predictors=2)), long_ds)
    assert run.family == "clmm"
    row = trt_t(hypotheses(run))
    ref = r_ref(r_session, tmp_path, long_df,
                "d$yo <- factor(d$yo, ordered = TRUE)\n"
                "m1 <- ordinal::clmm(yo ~ t * trt + (1 | pid), data = d, link = 'logit')\n"
                "m0 <- ordinal::clmm(yo ~ t + trt + (1 | pid), data = d, link = 'logit')\n"
                "sx_result <- list(lr = 2 * (as.numeric(logLik(m1)) - as.numeric(logLik(m0))))", ["ordinal"])
    assert row["statistic"] == pytest.approx(ref["lr"], rel=1e-3)


@pytest.mark.r
def test_H_gee_uses_robust_wald_and_is_population_average(long_ds, long_df, r_session, tmp_path):
    run = execute(create_run(long_ds, LS(structure="gee_exch")), long_ds)
    assert run.family == "gee"
    tf = hypotheses(run)
    row = trt_t(tf)
    ref = r_ref(r_session, tmp_path, long_df,
                "m <- geepack::geeglm(y ~ t * trt, id = pid, waves = occ, data = d, corstr = 'exchangeable')\n"
                "s <- coef(summary(m))\n"
                "sx_result <- list(w = s['t:trt', 'Wald'], b = s['t:trt', 'Estimate'], se = s['t:trt', 'Std.err'])",
                ["geepack"])
    assert row["statistic"] == pytest.approx(ref["w"], rel=1e-6) and row["estimate"] == pytest.approx(ref["b"], rel=1e-8)
    assert row["test_method"] == "robust (sandwich) Wald test" and "population-average" in row["ci_scale"]
    assert row["se"] == pytest.approx(ref["se"], rel=1e-8)
    assert results_frame(run)["aic"].isna().all()                          # no likelihood: no AIC


# ---- E: structure validation, refusals ---------------------------------------------------------

def test_E_validation_of_id_time_and_unsupported_combinations(long_ds, tmp_path):
    validate_settings(long_ds, LS())
    with pytest.raises(SearchSpaceError, match="needs a time variable"):
        validate_settings(long_ds, LS(time=None))
    with pytest.raises(SearchSpaceError, match="need a time variable"):
        validate_settings(long_ds, LS(time=None, strategy="main", structure="rs"))
    with pytest.raises(SearchSpaceError, match="not available"):
        validate_settings(long_ds, LS(structure="glmer_ri"))                  # Gaussian outcome
    with pytest.raises(SearchSpaceError, match="cannot be ranked together"):
        validate_settings(long_ds, LS(workflow="structure", structures=["ri", "gee_exch"]))
    with pytest.raises(SearchSpaceError, match="unique in every row"):
        validate_settings(long_ds, LS(participant="x", candidates=["trt"], time=None, strategy="main"))
    with pytest.raises(SearchSpaceError, match="time variable cannot"):
        validate_settings(long_ds, LS(candidates=["trt", "t"]))
    dup = pd.DataFrame({"pid": [1, 1, 1, 2, 2, 2] * 10, "t": [0, 0, 1, 0, 1, 2] * 10,
                        "y": np.arange(60.0), "a": np.arange(60.0) % 7})
    dds = make_ds(tmp_path, dup, "dup.csv")
    with pytest.raises(SearchSpaceError, match="more than one row at the same time"):
        validate_settings(dds, LS(candidates=["a"], structure="ar1", strategy="main", max_predictors=2))


def test_multinomial_repeated_outcome_is_refused(long_ds):
    long_ds.set_variable_field("yo", "measure", "nominal")
    with pytest.raises(SearchSpaceError, match="not available in automated discovery"):
        validate_settings(long_ds, LS(outcome="yo", family="multinomial"))


def test_K_grouping_and_time_counting(long_ds):
    from statexplore.discovery.space import enumerate_candidates, full_terms
    from statexplore.discovery.modelspec import counts

    s = LS(strategy="time", candidates=["trt", "x"], covariate_mode="subsets", max_predictors=3)
    for c in enumerate_candidates(s):
        n = counts(full_terms(c))["n_predictors"]
        assert "pid" not in c.variables and n == len(set(c.variables)) <= 3        # time counts, ID never does
    assert fit_identity(long_ds, LS(structure="ri"), "lmm") != fit_identity(long_ds, LS(structure="rs"), "lmm")
    assert fit_identity(long_ds, LS(), "lmm") != fit_identity(long_ds, LS(time_type="categorical"), "lmm")


def test_structure_workflow_ignores_strategy_rules(long_ds):
    # workflow B locks the fixed part: the strategy's "two candidates" rule does not apply
    validate_settings(long_ds, LS(candidates=["trt"], strategy="focal", workflow="structure",
                                  structures=["ri", "rs"], required=["t"]))
