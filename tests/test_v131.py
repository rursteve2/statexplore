"""v1.3.1 regression tests (spec: StatExplore_v1.3.1_Developer_Prompt_Complete.md, §5.2 Tests 26-35).

  26 proportional-odds check (STAT-15)              test_26_*
  27 BIC identity across modules (STAT-16)          test_27_*
  28 model-family identity across transfers         test_28_*
  29 AR(1)/CAR(1) time index and lag diagnostics    test_29_*
  30 portable, integrity-checked exports            test_30_*
  31 bounded search resume, cache, highlighting     test_31_*
  32 UI-28 layout at 1 / 3 / 100 candidate rows     test_32_*
  33 UI-29 / UI-31 variable search, sort, grouping  test_33_*
  34 UI-30 / 32 / 33 / 34 legibility               test_34_*
  35 HRS end-to-end reference fixtures              test_35_*

Independent checks run R in a fresh `Rscript --vanilla` process on the same data file, never compare with
numbers copied from the app. Tests that need the HRS file skip when it is absent (reported as not verified).
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from statexplore.core.importers import import_file  # noqa: E402

from test_discovery import make_ds  # noqa: E402
from test_v13 import BS, OS, ordinal_df, ordinal_ds, panel_df, panel_ds  # noqa: E402,F401

WHITESPACE_FIXTURE = Path(os.environ.get("STATEXPLORE_WS_FIXTURE",
                                         r"E:\Downloads\StatExplore_v1.3.0_Whitespace_Regression_Fixture.csv"))
HRS_CSV = Path(os.environ.get("STATEXPLORE_HRS_CSV", r"E:\Downloads\hrs_least_missing_1000.csv"))


def rscript(code: str, cwd: Path, timeout: int = 600) -> subprocess.CompletedProcess:
    """Run R code in a clean session (no StatExplore worker state)."""
    from statexplore.rbridge.locate import find_rscript, r_library

    return subprocess.run([str(find_rscript()), "--vanilla", "-e", code], cwd=cwd, capture_output=True, text=True,
                          env=dict(os.environ, R_LIBS=str(r_library())), timeout=timeout)


def r_values(p: subprocess.CompletedProcess) -> list[float]:
    line = next((ln for ln in p.stdout.splitlines() if ln.startswith("@@")), None)
    assert line is not None, p.stderr[-2000:]
    return [float(x) for x in line.split()[1:]]


def table(result, prefix: str):
    t = next(t for t in result.tables if t.name.startswith(prefix))
    return [dict(zip(t.columns, r)) for r in t.rows]


# ---- 26: proportional-odds check (STAT-15) -------------------------------------------------------------

def _ordinal_reg(r_session, ds, preds, interaction=False, outcome="y"):
    from statexplore.analysis.builders import BY_KEY
    from statexplore.analysis.framework import run_analysis
    from statexplore.rbridge.codegen import alias

    internal = {}
    if interaction:
        a = [alias(ds.var(n).alias) for n in preds]
        internal["rhs"] = " + ".join(a) + " + " + ":".join(a)
    res, _ = run_analysis(r_session, ds, BY_KEY["ordinal_reg"], {"dependent": [outcome], "predictors": preds},
                          {"conf": 0.95}, internal, "PO test")
    return res, table(res, "Proportional-odds check")


def _po_invariant(rows):
    """No unexplained dash: a row is either tested with a finite p, or carries a specific reason."""
    for r in rows:
        if r["status"] == "tested":
            assert r["p"] is not None and np.isfinite(r["p"]) and r["df"] > 0, r
        else:
            assert r["reason"], r
            if r["status"] == "approximate":
                assert r["p"] is not None and "df uncertain" in r["reason"]


@pytest.mark.r
def test_26_po_check_matches_nominal_test_main_and_interaction(ordinal_ds, ordinal_df, r_session, tmp_path):
    res_m, main = _ordinal_reg(r_session, ordinal_ds, ["a", "b"])
    res_i, inter = _ordinal_reg(r_session, ordinal_ds, ["a", "b"], interaction=True)
    for rows in (main, inter):
        _po_invariant(rows)
    assert [r["term"] for r in main] == ["a", "b", "All terms (global test)"]
    assert {r["term"] for r in inter} >= {"a", "b", "a:b", "All terms (global test)"}
    # independent: ordinal::nominal_test at top level of a clean R session (where its update() works)
    ordinal_df.to_csv(tmp_path / "o.csv", index=False)
    code = ('d <- read.csv("o.csv"); d$y <- factor(d$y, ordered = TRUE); d$a <- factor(d$a); d$b <- factor(d$b);'
            'm <- ordinal::clm(y ~ a + b, data = d); n1 <- ordinal::nominal_test(m);'
            'm2 <- ordinal::clm(y ~ a * b, data = d); n2 <- ordinal::nominal_test(m2);'
            'cat("@@", sprintf("%.12g", c(n1[["Pr(>Chi)"]][2:3], n1$Df[2:3], n2[["Pr(>Chi)"]][2:4], n2$Df[2:4])), "\\n")')
    v = r_values(rscript(code, tmp_path))
    assert [main[0]["p"], main[1]["p"]] == pytest.approx(v[0:2], rel=1e-6)
    assert [main[0]["df"], main[1]["df"]] == v[2:4]
    by = {r["term"]: r for r in inter}
    for k, term in enumerate(("a", "b", "a:b")):
        assert by[term]["p"] == pytest.approx(v[4 + k], rel=1e-5) and by[term]["df"] == v[7 + k]
    assert not any("not tested" in m for m in res_m.messages)


@pytest.mark.r
def test_26_po_check_detects_non_proportional_odds(tmp_path, r_session):
    """Controlled fixture: x shifts every cut-point equally (PO) vs very differently by cut-point (non-PO)."""
    rng = np.random.default_rng(26)
    n = 1500

    def draw(betas):
        x = rng.integers(0, 2, n)
        theta = np.array([-2.0, 0.0, 2.0])
        cum = 1 / (1 + np.exp(-(theta[None, :] - np.outer(x, betas))))
        u = rng.random(n)
        return pd.DataFrame({"x": x, "y": 1 + (u[:, None] > cum).sum(axis=1)})

    po = make_ds(tmp_path, draw(np.array([1.0, 1.0, 1.0])), "po.csv")
    npo = make_ds(tmp_path, draw(np.array([0.0, 1.0, 2.0])), "npo.csv")
    for ds in (po, npo):
        ds.set_variable_field("y", "measure", "ordinal")
    _r, rows_po = _ordinal_reg(r_session, po, ["x"])
    res, rows_npo = _ordinal_reg(r_session, npo, ["x"])
    assert rows_po[0]["status"] == rows_npo[0]["status"] == "tested"
    assert rows_po[0]["p"] > 0.01 and rows_npo[0]["p"] < 1e-10
    assert any("doubtful" in m for m in res.messages)


@pytest.mark.r
def test_26_po_check_sparse_categories_gives_specific_reason(tmp_path, r_session):
    rng = np.random.default_rng(3)
    n = 300
    g = rng.integers(0, 3, n)
    y = rng.integers(1, 5, n)
    y[(g == 2) & (y == 4)] = 3                  # outcome category 4 never occurs in group 2: an empty cell
    ds = make_ds(tmp_path, pd.DataFrame({"g": g, "y": y, "z": rng.integers(0, 2, n)}))
    ds.set_variable_field("y", "measure", "ordinal")
    ds.set_variable_field("g", "measure", "nominal")
    _res, rows = _ordinal_reg(r_session, ds, ["g", "z"])
    _po_invariant(rows)
    g_row = next(r for r in rows if r["term"] == "g")
    assert "sparse: 1 of 12 outcome-by-category cells are empty" in g_row["reason"]


# ---- 27: BIC identity (STAT-16) ---------------------------------------------------------------------------

@pytest.mark.r
def test_27_bic_identical_in_discovery_full_analysis_and_base_r(panel_ds, panel_df, r_session, tmp_path):
    from statexplore.analysis import mixed
    from statexplore.analysis.descriptor import fit_stats
    from statexplore.discovery.engine import create_run, execute

    reml = execute(create_run(panel_ds, BS(workflow="structure", structures=["ri", "nlme_ri", "ar1", "car1"])),
                   panel_ds)
    ml = execute(create_run(panel_ds, BS(structure="car1", strategy="main")), panel_ds)
    rows = {f"reml_{r['structure']}": r for r in reml.rows} | {"ml_car1": ml.rows[0]}
    for key, run in (("reml", reml), ("ml", ml)):
        for row in run.rows:
            spec = mixed.spec_from_discovery(run, row)
            res, _ = mixed.run(r_session, panel_ds, spec)
            st = fit_stats(res)
            assert st["bic"] == pytest.approx(row["bic"], abs=1e-6), (key, row["structure"])
            assert res.values["identity_check"] == "verified" and "BIC" in res.messages[0]
    panel_df.to_csv(tmp_path / "p.csv", index=False)
    code = ('d <- read.csv("p.csv"); d$sex <- factor(d$sex); d$occ <- as.integer(factor(d$wave));'
            'd <- d[order(d$pid, d$wave), ]; f <- bmi ~ sex + wave;'
            'ri <- nlme::lme(f, random = ~ 1 | pid, data = d, method = "REML");'
            'ar <- nlme::lme(f, random = ~ 1 | pid, data = d, method = "REML", correlation = nlme::corAR1(form = ~ occ | pid));'
            'ca <- nlme::lme(f, random = ~ 1 | pid, data = d, method = "REML", correlation = nlme::corCAR1(form = ~ wave | pid));'
            'caml <- nlme::lme(f, random = ~ 1 | pid, data = d, method = "ML", correlation = nlme::corCAR1(form = ~ wave | pid));'
            'lm4 <- lme4::lmer(bmi ~ sex + wave + (1 | pid), data = d, REML = TRUE);'
            'cat("@@", sprintf("%.10f", c(BIC(ri), BIC(ar), BIC(ca), BIC(caml), BIC(lm4))), "\\n")')
    v = r_values(rscript(code, tmp_path))
    got = [rows["reml_nlme_ri"]["bic"], rows["reml_ar1"]["bic"], rows["reml_car1"]["bic"], rows["ml_car1"]["bic"],
           rows["reml_ri"]["bic"]]
    assert got == pytest.approx(v, abs=1e-6)             # native BIC(fit): REML nlme uses log(n - p)
    assert rows["reml_car1"]["bic_n"] == len(panel_df) - 3 and rows["ml_car1"]["bic_n"] == len(panel_df)
    assert rows["reml_car1"]["ic_basis"] == "native"


def test_27_old_rows_are_never_reused_after_the_ic_change(panel_ds):
    from statexplore.discovery.engine import FIT_RULES, _engine_id

    old = {"r_version": "4.6.2", "packages": {"nlme": "3.1"}}
    assert _engine_id(old) != _engine_id({**old, "fit_rules": FIT_RULES})


def test_27_verify_compares_bic_only_for_new_rows():
    from statexplore.analysis.descriptor import ModelDescriptor, verify

    d = ModelDescriptor(source="s", run_id="r", candidate=1, family="lmm", outcome_family="gaussian",
                        engine="nlme::lme", link="identity", outcome="y", terms=[["x"]], n=100, loglik=-50.0,
                        aic=110.0, bic=120.0, ic_basis="native")
    assert verify(d, 100, -50.0, 110.0, 120.0005)[0] is True
    ok, text = verify(d, 100, -50.0, 110.0, 120.004)        # the v1.3.0 discrepancy size
    assert ok is False and "BIC" in text
    d.ic_basis = ""                                          # a v1.3.0 row: BIC was on another convention
    assert verify(d, 100, -50.0, 110.0, 120.004)[0] is True


# ---- 28: model-family identity across transfers ------------------------------------------------------------

@pytest.mark.r
def test_28_gee_identity_and_blocked_single_level_transfer(panel_ds, r_session):
    from statexplore.analysis import mixed
    from statexplore.analysis.descriptor import TransferError
    from statexplore.analysis.models import from_candidate
    from statexplore.discovery.engine import create_run, execute

    panel_ds.set_variable_field("smk", "measure", "nominal")
    run = execute(create_run(panel_ds, BS(outcome="smk", family="binomial", workflow="structure",
                                          structures=["gee_ind", "gee_exch", "gee_ar1"])), panel_ds)
    for row in run.rows:
        spec = mixed.spec_from_discovery(run, row)
        assert (spec.engine, spec.estimator, spec.family) == ("geepack", "gee", "binomial")
        res, _ = mixed.run(r_session, panel_ds, spec)
        assert res.status.value in ("ok", "warning") and "lme4" not in res.script.split("# ---- StatExplore")[-1]
        assert "geepack::geeglm(" in res.script and f'corstr = "{spec.gee_corstr}"' in res.script
        assert row["qic"] is not None and np.isfinite(row["qic"])
        with pytest.raises(TransferError, match="not single-level"):
            from_candidate(run, row)


@pytest.mark.r
def test_28_multinomial_full_analysis_and_comparison_keep_reference(tmp_path, r_session):
    from statexplore.analysis.comparison import compare
    from statexplore.analysis.models import from_candidate
    from statexplore.discovery.detail import candidate_from_row, full_analysis
    from statexplore.discovery.engine import create_run, execute
    from statexplore.discovery.space import SearchSettings

    rng = np.random.default_rng(12)
    n = 600
    a, b = rng.integers(0, 2, n), rng.integers(0, 2, n)
    eta = np.column_stack([np.zeros(n), 0.8 * a, -0.6 * b + 0.3 * a])
    pr = np.exp(eta) / np.exp(eta).sum(axis=1, keepdims=True)
    y = np.array([rng.choice([1, 2, 3], p=p) for p in pr])
    ds = make_ds(tmp_path, pd.DataFrame({"a": a, "b": b, "y": y}))
    ds.set_variable_field("y", "measure", "nominal")
    s = SearchSettings(outcome="y", candidates=["a", "b"], family="multinomial", reference=2.0, strategy="combinations",
                       include_main=True, max_order=2, workers=1, max_models=None, max_minutes=None, max_predictors=2)
    run = execute(create_run(ds, s), ds)
    row = next(r for r in run.rows if r["inter"])
    res, _ = full_analysis(r_session, ds, run, candidate_from_row(run, row))
    assert res.values["identity_check"] == "verified", res.messages[:2]
    assert "nnet::multinom(" in res.script and 'sx_ref <- "2"' in res.script
    m = from_candidate(run, row)
    assert m.family == "multinomial" and m.reference == 2.0
    cmp, _ = compare(r_session, ds, [m, m.copy_edited([t for t in m.terms if len(t) == 1])])
    assert any("identity verified" in x for x in cmp.messages)
    joint = next(t for t in row["tests"] if t["type"] in ("joint_interaction", "interaction_term"))
    nest = table(cmp, "Nested model tests")[0]
    assert list(nest.values())[3] == pytest.approx(joint["statistic"], abs=1e-6)


@pytest.mark.r
def test_28_glmm_and_clmm_keep_family_and_predictors(panel_ds, panel_df, r_session, tmp_path):
    from statexplore.analysis import mixed
    from statexplore.discovery.engine import create_run, execute

    panel_ds.set_variable_field("smk", "measure", "nominal")
    run = execute(create_run(panel_ds, BS(outcome="smk", family="binomial", structure="glmer_ri", strategy="main")),
                  panel_ds)
    row = run.rows[0]
    spec = mixed.spec_from_discovery(run, row)
    assert (spec.engine, spec.family, spec.estimator) == ("glmer", "binomial", "mixed")
    assert spec.predictors == ["sex"] and spec.time == "wave"
    res, _ = mixed.run(r_session, panel_ds, spec)
    assert res.values["identity_check"] == "verified" and "lme4::glmer(" in res.script
    fx = table(res, "Fixed effects")
    assert any("odds" in k.lower() for k in fx[0])           # OR reported on the odds-ratio scale
    df = panel_df.assign(cat=pd.cut(panel_df["bmi"], [-np.inf, 26, 28, 30, np.inf], labels=False) + 1)
    ds = make_ds(tmp_path, df, "c.csv")
    ds.set_variable_field("cat", "measure", "ordinal")
    ds.set_variable_field("wave", "measure", "continuous")
    run = execute(create_run(ds, BS(outcome="cat", family="ordinal", structure="clmm_ri", strategy="main")), ds)
    spec = mixed.spec_from_discovery(run, run.rows[0])
    assert spec.family == "ordinal" and spec.engine == "clmm"
    res, _ = mixed.run(r_session, ds, spec)
    assert res.values["identity_check"] == "verified" and "ordinal::clmm(" in res.script
    assert len(table(res, "Thresholds")) == 3                 # 4 ordered categories -> 3 cut-points


def test_28_timeout_and_cancel_are_never_complete(qtbot, tmp_path):
    from PySide6.QtCore import QSettings

    from statexplore.gui.state import AppState, _Task
    from statexplore.rbridge.worker import RTimeoutError

    def boom():
        raise RTimeoutError("R did not respond within 1s")

    t = _Task(boom)
    got = []
    t.signals.failed.connect(got.append)
    t.run()
    assert got and got[0].startswith("Timed out: the analysis did not complete")
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(tmp_path / "settings"))
    state = AppState()
    try:
        errors, done = [], []
        state.run_r(lambda sess: sess.worker.run("Sys.sleep(60); sx_result <- 1", timeout=120), done.append, errors.append)
        qtbot.waitUntil(lambda: bool(errors or (state._r_busy and state._worker is not None and state._worker.alive)),
                        timeout=60000)
        assert not errors, errors
        qtbot.wait(500)
        state.cancel_r()
        qtbot.waitUntil(lambda: bool(errors or done), timeout=30000)
        assert not done and errors[0].startswith("Cancelled by the user")
        # no stale state: the next computation restarts R and succeeds
        state.run_r(lambda sess: sess.worker.run("sx_result <- 21 * 2").result, done.append, errors.append)
        qtbot.waitUntil(lambda: bool(done), timeout=60000)
        assert done[0] == 42 and len(errors) == 1 and not state._cancel_requested
    finally:
        state.shutdown_r()


# ---- 29: time index, gaps and lag diagnostics ----------------------------------------------------------------

@pytest.fixture
def gap_df():
    """Waves 1-4 and 6 (wave 5 absent for everyone). Pattern A skips wave 3, which pattern B has."""
    rng = np.random.default_rng(29)
    rows = []
    for pid in range(120):
        waves = [1, 2, 4, 6] if pid % 2 else [1, 2, 3, 4, 6]
        u, e, last = rng.normal(0, 1), 0.0, None
        for w in waves:
            e = (0.6 ** (w - last if last else 1)) * e + rng.normal(0, 1)
            last = w
            rows.append({"pid": pid, "wave": w, "sex": pid % 3 == 0, "y": 10 + u + 0.2 * w + e})
    return pd.DataFrame(rows).assign(sex=lambda d: d["sex"].astype(int))


@pytest.mark.r
def test_29_occasion_index_distances_match_nlme_correlation_matrices(gap_df, tmp_path, r_session):
    from statexplore.analysis import mixed
    from statexplore.discovery.engine import create_run, execute

    ds = make_ds(tmp_path, gap_df)
    ds.set_variable_field("wave", "measure", "continuous")
    s = dict(outcome="y", candidates=["sex"], required=["wave"], participant="pid", time="wave")
    run = execute(create_run(ds, BS(**s, workflow="structure", structures=["nlme_ri", "ar1", "car1"])), ds)
    by = {r["structure"]: r for r in run.rows}
    phis = {}
    for st in ("ar1", "car1"):
        spec = mixed.spec_from_discovery(run, by[st])
        res, _ = mixed.run(r_session, ds, spec)
        assert res.values["identity_check"] == "verified"
        phis[st] = table(res, "Residual autocorrelation")[0]["phi"]
        lag = table(res, "Residual autocorrelation by lag")
        assert {"successive_rows_r", "occasion_index_gap_r", "occasion_index_gap_pairs"} <= set(lag[0])
        # pattern A's wave 2 -> 4 is one successive row but an index gap of 2: fewer lag-1 index pairs
        assert lag[0]["occasion_index_gap_pairs"] < lag[0]["successive_rows_pairs"]
    gap_df.to_csv(tmp_path / "g.csv", index=False)
    code = ('d <- read.csv("g.csv"); d$sex <- factor(d$sex); d$occ <- as.integer(factor(d$wave));'
            'd <- d[order(d$pid, d$wave), ];'
            'ar <- nlme::lme(y ~ sex + wave, random = ~ 1 | pid, data = d, method = "REML",'
            '  correlation = nlme::corAR1(form = ~ occ | pid));'
            'ca <- nlme::lme(y ~ sex + wave, random = ~ 1 | pid, data = d, method = "REML",'
            '  correlation = nlme::corCAR1(form = ~ wave | pid));'
            'pa <- coef(ar$modelStruct$corStruct, unconstrained = FALSE); pc <- coef(ca$modelStruct$corStruct, unconstrained = FALSE);'
            'A <- nlme::corMatrix(ar$modelStruct$corStruct)[["1"]]; C <- nlme::corMatrix(ca$modelStruct$corStruct)[["1"]];'
            'cat("@@", sprintf("%.10f", c(pa, pc, unique(d$occ[d$pid == 1]), A[2, 3], A[3, 4], C[2, 3], C[3, 4])), "\\n")')
    v = r_values(rscript(code, tmp_path))
    pa, pc, occ = v[0], v[1], v[2:6]
    assert occ == [1, 2, 4, 5]                       # wave 5 compressed (6 -> index 5); skipped wave 3 keeps a gap
    assert phis["ar1"] == pytest.approx(pa, abs=1e-6) and phis["car1"] == pytest.approx(pc, abs=1e-6)
    a23, a34, c23, c34 = v[6:10]
    assert a23 == pytest.approx(pa ** 2, rel=1e-8)   # waves 2 -> 4: index distance 2
    assert a34 == pytest.approx(pa ** 1, rel=1e-8)   # waves 4 -> 6: index distance 1 (wave 5 compressed)
    assert c23 == pytest.approx(pc ** 2, rel=1e-8)   # actual time distance 2
    assert c34 == pytest.approx(pc ** 2, rel=1e-8)   # actual time distance 2 (wave 5 NOT compressed)


@pytest.mark.r
def test_29_per_model_sample_uses_the_models_own_occasion_index(gap_df, tmp_path, r_session):
    """Rows exist for every wave but the outcome is missing for everyone at wave 2: discovery (complete cases per
    model) and Full Analysis must build the same occasion index (v1.3.0 kept wave 2 in the discovery index)."""
    from statexplore.analysis import mixed
    from statexplore.discovery.engine import create_run, execute

    df = gap_df.copy()
    df.loc[df["wave"] == 2, "y"] = np.nan
    ds = make_ds(tmp_path, df)
    ds.set_variable_field("wave", "measure", "continuous")
    run = execute(create_run(ds, BS(outcome="y", candidates=["sex"], required=["wave"], participant="pid",
                                    time="wave", structure="ar1", strategy="main", sample="per_model")), ds)
    spec = mixed.spec_from_discovery(run, run.rows[0])
    res, _ = mixed.run(r_session, ds, spec)
    assert res.values["identity_check"] == "verified", res.messages[:2]


def test_29_wording_does_not_claim_equal_spacing_for_person_gaps(gap_df, tmp_path):
    from statexplore.analysis.descriptor import TIME_INDEX
    from statexplore.analysis.longitudinal_preflight import preflight
    from statexplore.analysis.mixed import MixedSpec

    ds = make_ds(tmp_path, gap_df)
    spec = MixedSpec(outcome="y", id="pid", time="wave", predictors=["sex"], residual="ar1")
    rep = preflight(ds, "y", "pid", "wave", ["sex"], spec)
    text = " ".join(rep.notes + rep.warnings)
    assert "global observed-occasion index" in text and "equally spaced" not in text
    assert "keeps an index gap" in TIME_INDEX["ar1"]


# ---- 30: exported scripts ---------------------------------------------------------------------------------

def _export(tmp_path, ds, run, row, r_session, comparisons=None, name="analysis"):
    from statexplore.discovery.detail import candidate_from_row
    from statexplore.discovery.rcode import candidate_script
    from statexplore.discovery.space import view
    from statexplore.export.exporters import write_portable

    v = view(ds, run.settings)
    r_session.ensure_data(v)
    script = candidate_script(r_session.load_script, v, run.settings, run.family, candidate_from_row(run, row),
                              comparisons, expected=row)
    folder = tmp_path / f"export_{name}"
    write_portable(script, folder, ds)
    return folder


def _source(folder: Path, cwd: Path, pre: str = "") -> subprocess.CompletedProcess:
    return rscript(f'{pre}source("{(folder / "analysis.R").as_posix()}")', cwd)


@pytest.mark.r
def test_30_exports_reproduce_and_verify_in_clean_r(panel_ds, ordinal_ds, r_session, tmp_path):
    from statexplore.discovery.engine import create_run, execute, structure_tests

    run = execute(create_run(panel_ds, BS(workflow="structure", structures=["nlme_ri", "ar1", "car1"])), panel_ds)
    for st in ("ar1", "car1"):
        row = next(r for r in run.rows if r["structure"] == st)
        folder = _export(tmp_path, panel_ds, run, row, r_session, structure_tests(run)[int(row["idx"])], st)
        p = _source(folder, tmp_path)                     # sourced from ANOTHER folder: relative data path
        assert p.returncode == 0, p.stderr[-2000:]
        assert "Reproduced within tolerance" in p.stdout and "comparisons reproduced" in p.stdout
        manifest = json.loads((folder / "analysis_manifest.json").read_text(encoding="utf-8"))
        assert manifest["data"]["md5"] and "nlme" in manifest["r_packages"]
        (idc,) = manifest["data"]["pseudonymized_id_columns"]
        ids = pd.read_csv(folder / "analysis_data.csv")[idc]
        assert sorted(ids.unique()) == list(range(1, ids.nunique() + 1))      # order-preserving codes 1..K
        assert "AppData" not in (folder / "analysis.R").read_text(encoding="utf-8")
        assert len(manifest["data"]["columns"]) == 4
    orun = execute(create_run(ordinal_ds, OS()), ordinal_ds)
    folder = _export(tmp_path, ordinal_ds, orun, orun.rows[-1], r_session, name="ordinal")
    p = _source(folder, tmp_path)
    assert p.returncode == 0 and "Reproduced within tolerance" in p.stdout, p.stderr[-2000:]


@pytest.mark.r
def test_30_exports_fail_clearly_on_changed_data_or_missing_packages(panel_ds, r_session, tmp_path):
    from statexplore.discovery.engine import create_run, execute

    run = execute(create_run(panel_ds, BS(structure="car1", strategy="main")), panel_ds)
    folder = _export(tmp_path, panel_ds, run, run.rows[0], r_session)
    data = folder / "analysis_data.csv"
    df = pd.read_csv(data)
    y = next(c for c in df.columns if df[c].dtype == float)
    df.loc[0, y] = np.nan                              # one analysed row removed
    df.to_csv(data, index=False)
    p = _source(folder, tmp_path)
    assert p.returncode != 0 and "MD5 differs" in p.stderr
    p = _source(folder, tmp_path, "sx_skip_checksum <- TRUE; ")
    assert p.returncode != 0 and ("n differs from the app" in p.stderr or "analysis rows differ" in p.stderr)
    script = (folder / "analysis.R").read_text(encoding="utf-8")
    (folder / "analysis.R").write_text(script.replace('sx_needed <- c(', 'sx_needed <- c("notAPackage123", ', 1),
                                       encoding="utf-8")
    p = _source(folder, tmp_path)
    assert p.returncode != 0 and "Missing R packages: notAPackage123" in p.stderr


def test_30_row_signature_guards_structure_comparisons():
    from types import SimpleNamespace

    from statexplore.discovery.engine import structure_tests

    rows = [dict(idx=0, kind="structure", structure="nlme_ri", status="ok", loglik=-100.0, n=50, n_re_params=2,
                 rows_sig="aaa"),
            dict(idx=1, kind="structure", structure="ar1", status="ok", loglik=-95.0, n=50, n_re_params=3,
                 rows_sig="aaa")]
    run = SimpleNamespace(rows=rows)
    assert structure_tests(run)[1][0]["statistic"] == pytest.approx(10.0)
    rows[1]["rows_sig"] = "bbb"                           # same n, different rows
    assert structure_tests(run) == {}
    rows[1].update(structure="car1", rows_sig="aaa")
    rows.append(dict(rows[1], idx=2, structure="ar1"))
    assert all(t["reference_structure"] != "ar1" for ts in structure_tests(run).values() for t in ts)


# ---- 31: bounded search ------------------------------------------------------------------------------------

@pytest.fixture
def four_binary(tmp_path):
    rng = np.random.default_rng(7)
    X = rng.integers(0, 2, (400, 4))
    df = pd.DataFrame(X, columns=list("abcd")).assign(y=X @ [0.3, 0.2, 0, 0] + rng.normal(size=400))
    ds = make_ds(tmp_path, df)
    for v in "abcd":
        ds.set_variable_field(v, "measure", "nominal")
    return ds


def _s112(**kw):
    from statexplore.discovery.space import SearchSettings

    return SearchSettings(**{**dict(outcome="y", candidates=list("abcd"), family="gaussian", strategy="combinations",
                                    include_main=True, max_order=2, max_predictors=4, workers=1, max_minutes=None,
                                    max_models=10), **kw})


@pytest.mark.r
def test_31_bounded_search_resume_reopen_no_duplicates(four_binary, tmp_path):
    from statexplore.core.project import Project
    from statexplore.discovery.engine import create_run, execute, results_frame, summary, tests_frame

    run = execute(create_run(four_binary, _s112()), four_binary)
    assert run.total == 112 and run.status == "partial" and run.n_done == 10
    tf = tests_frame(run, interpret=False)
    assert set(tf["search_scope"]) == {"partial search (provisional)"}
    sm = summary(run, results_frame(run, tf))
    assert sm["provisional"] and sm["not_visited"] == 102 and sm["attempted"] == 10
    run.settings.max_models = 20
    run = execute(run, four_binary)
    assert run.n_done == 30 and "run 1: 10 models" in run.budget_text() and "run 2: 20 models" in run.budget_text()
    pr = Project(four_binary)
    pr.searches.append(run)
    pr.save(tmp_path / "p.sxp")
    back = Project.open(tmp_path / "p.sxp")
    again = back.searches[0]
    again.settings.max_models = 500
    again = execute(again, back.dataset)
    keys = [r["key"] for r in again.rows]
    assert again.status == "complete" and len(keys) == len(set(keys)) == 112
    assert [r["idx"] for r in again.rows][:30] == [r["idx"] for r in run.rows]
    assert sorted(r["idx"] for r in again.rows) == list(range(112))


@pytest.mark.r
def test_31_cache_reuse_and_invalidation(four_binary):
    from statexplore.core.derive import FilterSpec
    from statexplore.discovery.engine import cache_from, create_run, execute, fit_identity, results_frame, summary, tests_frame

    first = execute(create_run(four_binary, _s112(max_models=None, max_predictors=2)), four_binary)
    second = create_run(four_binary, _s112(max_models=None, max_predictors=2))
    second = execute(second, four_binary, cache=cache_from(second, [first], four_binary))
    sm = summary(second, results_frame(second, tests_frame(second, interpret=False)))
    assert sm["reused"] == len(second.rows) > 0
    assert [r["aic"] for r in second.rows] == [r["aic"] for r in first.rows]
    ident = fit_identity(four_binary, second.settings, "gaussian")
    four_binary.set_variable_field("c", "measure", "continuous")
    assert fit_identity(four_binary, second.settings, "gaussian") != ident      # measurement override
    four_binary.set_variable_field("c", "measure", "nominal")
    four_binary.add_filter(FilterSpec("a", "==", 1))
    assert fit_identity(four_binary, second.settings, "gaussian") != ident      # row filter


# ---- 32-34: GUI --------------------------------------------------------------------------------------------

SHOTS = Path(os.environ.get("STATEXPLORE_SHOTS", "")) if os.environ.get("STATEXPLORE_SHOTS") else None


@pytest.fixture
def window(qtbot, tmp_path):
    from PySide6.QtCore import QSettings

    from statexplore.gui.main_window import MainWindow
    from statexplore.gui.state import AppState

    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(tmp_path / "settings"))
    w = MainWindow(AppState())
    yield w
    w.state.dirty = False
    w.state.shutdown_r()
    w.close()


def _load(window, tmp_path, df, name="d.csv"):
    from statexplore.core.project import Project

    window.state.set_project(Project(make_ds(tmp_path, df, name)))


@pytest.mark.r
@pytest.mark.parametrize("n_rows", [1, 3, 100])
def test_32_layout_with_1_3_100_candidate_rows(window, qtbot, tmp_path, n_rows):
    rng = np.random.default_rng(32)
    k = {1: 1, 3: 2, 100: 7}[n_rows]
    df = pd.DataFrame(rng.normal(size=(200, k)), columns=[f"x{i}" for i in range(k)]).assign(y=rng.normal(size=200))
    _load(window, tmp_path, df)
    page = window.pages["Automated Model Discovery"]
    window.sidebar.setCurrentRow(list(window.pages).index("Automated Model Discovery"))
    page.outcome.setCurrentIndex(page.outcome.findData("y"))
    page.cand_sel.set_selected([f"x{i}" for i in range(k)])
    page.strategy.setCurrentIndex(page.strategy.findData("main"))
    page.workers.setValue(1)
    page._use_suggestion()
    page._start()
    qtbot.waitUntil(lambda: page.run is not None and page.run.status == "complete", timeout=180000)
    qtbot.waitUntil(lambda: page.all_table.model.rowCount() > 0, timeout=10000)
    assert page.all_table.model.rowCount() == {1: 1, 3: 3, 100: 126}[n_rows]    # 7 predictors, at most 6
    for w, h in ((1366, 768), (1920, 1080), (2560, 1440)):
        window.resize(w, h)
        qtbot.wait(50)
        page.tabs.setCurrentWidget(page.all_table)
        page.all_table.view.selectRow(0)
        page._expand_details()
        qtbot.wait(50)
        top, bottom = page.results_split.sizes()
        if n_rows <= 3:          # few rows: the details get the space instead of an empty table
            assert bottom >= 0.4 * (top + bottom)
        assert page.all_table.view.horizontalHeader().minimumSectionSize() >= 44
        if SHOTS:
            SHOTS.mkdir(parents=True, exist_ok=True)
            window.grab().save(str(SHOTS / f"ui28_rows{n_rows}_{w}x{h}.png"))
    # dragging the divider after the search is respected (not reset by the next refresh)
    page.results_split.setSizes([500, 300])
    page.results_split.splitterMoved.emit(page.results_split.sizes()[0], 1)    # as a user drag does
    before = page.results_split.sizes()
    page._show_run()
    page.all_table.view.selectRow(min(1, page.all_table.model.rowCount() - 1))
    page._update_details()
    assert page.results_split.sizes() == before


def test_33_measurement_review_search_sort_and_saved_override(window, qtbot, tmp_path):
    from statexplore.analysis import measurement as M
    from statexplore.core.project import Project

    rng = np.random.default_rng(33)
    cols = {f"V{i:03d}": rng.normal(size=60) for i in range(165)}
    cols["R_SHLT"] = rng.integers(1, 6, 60).astype(float)
    cols["R_SHLT"][:7] = np.nan
    df = pd.DataFrame(cols)
    _load(window, tmp_path, df)
    from statexplore.gui.pages.measurement_review import COLUMNS

    override = COLUMNS.index("Override")
    review = window.pages["Dataset Explorer"].review
    review.scope.setCurrentIndex(review.scope.findData("all"))
    assert len(review._names) == 166
    review.search.setText("r_sh")
    assert review._names == ["R_SHLT"]
    M.approve(window.state.dataset, "R_SHLT", "ordinal")
    window.state.notify_metadata_changed()
    review.search.setText("saved")
    assert review._names == ["R_SHLT"] and review.table.item(0, override).text().startswith("saved: ")
    review.search.clear()
    review.sort.setCurrentIndex(review.sort.findData("miss_desc"))
    assert review._names[0] == "R_SHLT"
    review.sort.setCurrentIndex(review.sort.findData("za"))
    assert review._names[0] == "V164" and review._names[-1] == "R_SHLT"
    window.state.project.save(tmp_path / "p.sxp")        # override persists after reopening
    window.state.set_project(Project.open(tmp_path / "p.sxp"))
    review.search.setText("R_SHLT")
    assert review.table.item(0, override).text() == f"saved: {window.state.dataset.var('R_SHLT').source_measure or '—'} → ordinal" \
        or review.table.item(0, override).text().startswith("saved: ")


def test_33_selected_predictors_first_display_only(qtbot):
    from PySide6.QtCore import Qt

    from statexplore.gui.widgets.variable_selector import HEADER, VariableSelector

    names = [f"v{i:03d}" for i in range(166)]
    sel = VariableSelector(selected_first=True)
    qtbot.addWidget(sel)
    sel.set_variables(None, names)
    sel.search.setText("v15")
    for it in sel.visible_items()[:3]:
        it.setCheckState(Qt.Checked)                     # checked while a search is active
    qtbot.wait(20)
    assert sel.selected() == ["v150", "v151", "v152"]
    assert sel.list.item(0).data(HEADER) and sel.list.item(0).text() == "Selected (3)"
    assert [sel.list.item(i).data(Qt.UserRole) for i in (1, 2, 3)] == ["v150", "v151", "v152"]
    assert sel.list.item(4).text() == "Not selected (163)"
    assert sel.search.text() == "v15" and sum(not it.isHidden() for it in sel.items()) == 10
    sel.search.clear()
    sel.list.verticalScrollBar().setValue(sel.list.verticalScrollBar().maximum() // 2)
    pos = sel.list.verticalScrollBar().value()
    next(it for it in sel.items() if it.data(Qt.UserRole) == "v100").setCheckState(Qt.Checked)
    next(it for it in sel.items() if it.data(Qt.UserRole) == "v151").setCheckState(Qt.Unchecked)
    qtbot.wait(20)
    assert sel.selected() == ["v150", "v152", "v100"]       # check order kept: formula / search order unchanged
    assert sel.list.verticalScrollBar().value() == pos
    assert sel.names() == [n for n in sel.names() if n]      # headers never leak into names()
    assert sorted(sel.names()) == names


def test_34_entities_scrollbar_dropdowns_and_tables(window, qtbot, tmp_path):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QTextDocument

    from statexplore.gui.models import fmt_p
    from statexplore.gui.result_view import fmt_cell

    rng = np.random.default_rng(34)
    _load(window, tmp_path, pd.DataFrame({"y": rng.integers(1, 5, 80), "x": rng.normal(size=80)}))
    page = window.pages["Automated Model Discovery"]
    window.state.dataset.set_variable_field("y", "measure", "ordinal")
    page.outcome.setCurrentIndex(page.outcome.findData("y"))
    page._family_changed()
    label = page.family_hint
    assert label.textFormat() == Qt.RichText
    doc = QTextDocument()
    doc.setHtml(label.text())
    assert "&#x27;" not in doc.toPlainText() and "&amp;" not in doc.toPlainText()
    # UI-32: the outer section bar is wide and distinct; inner tables keep the default bar
    bar = page.findChild(type(page.all_table.view.verticalScrollBar()), "sectionScrollBar")
    assert bar is not None and "width: 16px" in bar.styleSheet() and ":pressed" in bar.styleSheet()
    assert page.all_table.view.verticalScrollBar().styleSheet() == ""
    # UI-33: dropdowns shrink safely and expose their full text
    assert page.family.minimumContentsLength() >= 10 and page.family.itemData(0, Qt.ToolTipRole)
    # UI-34 / §6.6: p never displays as 0; long formulas carry their full text as a tooltip
    assert fmt_p(0.0) == "< 1e-300" and fmt_cell(0.0, p_col=True) == "< 1e-300" and fmt_p(5e-30) == "5.00e-30"
    from statexplore.gui.models import FrameModel

    m = FrameModel(pd.DataFrame({"Formula": ["y ~ a + b + c + a:b + a:c + b:c + d"]}))
    assert m.data(m.index(0, 0), Qt.ToolTipRole).startswith("y ~ a + b")


# ---- 35: HRS end-to-end reference fixtures --------------------------------------------------------------------

@pytest.mark.r
def test_35_hrs_ordinal_candidate_and_po_check(hrs_path, r_session):
    from statexplore.analysis.comparison import compare
    from statexplore.analysis.models import from_candidate
    from statexplore.core.derive import FilterSpec
    from statexplore.discovery.detail import candidate_from_row, full_analysis
    from statexplore.discovery.engine import create_run, execute

    ds = import_file(hrs_path)
    ds.add_filter(FilterSpec("WAVE", "==", 16))
    ds.set_variable_field("R_SHLT", "measure", "ordinal")
    run = execute(create_run(ds, OS(outcome="R_SHLT", candidates=["RAGENDER", "R_DEPRES"], max_order=2)), ds)
    row = next(r for r in run.rows if r["inter"])
    assert row["n"] == 917
    res, _ = full_analysis(r_session, ds, run, candidate_from_row(run, row))
    fit = table(res, "Model fit")[0]
    assert (fit["logLik"], fit["AIC"], fit["BIC"]) == pytest.approx((-1244.243, 2502.487, 2536.234), abs=1e-3)
    assert len(table(res, "Thresholds")) == 4 and res.values["identity_check"] == "verified"
    po = {r["term"]: r for r in table(res, "Proportional-odds check")}
    assert po["RAGENDER"]["p"] == pytest.approx(0.4235, abs=1e-4) and po["R_DEPRES"]["p"] == pytest.approx(0.8084, abs=1e-4)
    assert po["RAGENDER:R_DEPRES"]["status"] == "approximate" and "empty" in po["RAGENDER:R_DEPRES"]["reason"]
    m = from_candidate(run, row)
    cmp, _ = compare(r_session, ds, [m, m.copy_edited([t for t in m.terms if len(t) == 1])])
    nest = list(table(cmp, "Nested model tests")[0].values())
    assert nest[3] == pytest.approx(0.645, abs=1e-3) and nest[4] == pytest.approx(0.4218, abs=1e-4)


@pytest.mark.r
def test_35_hrs_covariance_bic_native(hrs_path):
    from statexplore.discovery.engine import create_run, execute, structure_tests

    ds = import_file(hrs_path)
    ds.set_variable_field("WAVE", "measure", "continuous")
    run = execute(create_run(ds, BS(outcome="R_PMBMI", candidates=["RAGENDER"], required=["WAVE"],
                                    participant="HHIDPN", time="WAVE", workflow="structure",
                                    structures=["nlme_ri", "ar1", "car1"])), ds)
    by = {r["structure"]: r for r in run.rows}
    assert by["ar1"]["bic"] == pytest.approx(20550.397, abs=1e-3)
    assert by["car1"]["bic"] == pytest.approx(20577.716, abs=1e-3)
    assert by["nlme_ri"]["loglik"] == pytest.approx(-10315.171, abs=1e-3)
    st = {int(r["idx"]): r["structure"] for r in run.rows}
    lr = {(st[i], t["reference_structure"]): (t["statistic"], t["p"]) for i, ts in structure_tests(run).items()
          for t in ts}
    assert lr[("ar1", "nlme_ri")][0] == pytest.approx(129.554778, abs=1e-4)
    assert lr[("ar1", "nlme_ri")][1] == pytest.approx(5.1281518e-30, rel=1e-5)
    assert lr[("car1", "nlme_ri")][0] == pytest.approx(102.235740, abs=1e-4)
    assert lr[("car1", "nlme_ri")][1] == pytest.approx(2.4646255e-24, rel=1e-5)
    assert len(lr) == 2                                  # no AR(1) vs CAR(1) likelihood-ratio test


@pytest.mark.r
def test_35_hrs_multinomial_reference(hrs_path, r_session):
    from statexplore.analysis.builders import BY_KEY
    from statexplore.analysis.framework import run_analysis
    from statexplore.core.derive import FilterSpec

    ds = import_file(hrs_path)
    ds.add_filter(FilterSpec("WAVE", "==", 16))
    ds.set_variable_field("R_MSTAT", "measure", "nominal")
    res, _ = run_analysis(r_session, ds, BY_KEY["multinom_reg"], {"dependent": ["R_MSTAT"], "predictors": ["RAGENDER"]},
                          {"conf": 0.95}, {}, "multinomial")
    fit = table(res, "Model fit")[0]
    assert fit["n"] == 942
    assert (fit["AIC"], fit["BIC"]) == pytest.approx((1997.005, 2055.181), abs=1e-3)
    assert fit["LR_chi2"] == pytest.approx(114.4961, abs=1e-3) and fit["df"] == 6


def test_35_whitespace_fixture_and_hrs_import_counts(hrs_path):
    if not WHITESPACE_FIXTURE.is_file():
        pytest.skip(f"whitespace fixture not found at {WHITESPACE_FIXTURE}")
    from statexplore.core.variables import Measure

    ws = import_file(WHITESPACE_FIXTURE)
    assert ws.raw.shape == (12, 5)
    assert ws.var("BMI_WHITESPACE_TEST").measure is Measure.CONTINUOUS
    assert ws.missing_count("BMI_WHITESPACE_TEST") == 4 and ws.missing_count("NEG_DECIMAL_TEST") == 2
    sav = import_file(hrs_path)
    assert sav.raw.shape == (16000, 168) and int(sav.raw.isna().sum().sum()) == 767497
    if HRS_CSV.is_file():
        csv = import_file(HRS_CSV)
        assert csv.raw.shape == (16000, 168) and int(csv.raw.isna().sum().sum()) == 767497
