"""v1.3.3 regression tests (package StatExplore_v1.3.3: 01 model integrity, 02 discovery/validation, 04 Tests 39-54).

Numerical expectations come from the INDEPENDENT R reference scripts of the package (`tests/fixtures/v133/scripts`,
explicit lme4/nlme/ordinal/geepack formulas written without the application's code generator) run on the exact fixture
bytes (`tests/fixtures/v133/reference`, R 4.6.2; see SHA256SUMS.txt and R_sessionInfo.txt). The application is never
compared with itself. Tolerances follow 04 §3: logLik/AIC/BIC abs 1e-4 + rel 1e-6; p on the log10 scale (1e-3).
"""
from __future__ import annotations

import csv
import hashlib
import os
import subprocess
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from statexplore.analysis import mixed  # noqa: E402
from statexplore.core.importers import import_file  # noqa: E402

FX = Path(__file__).parent / "fixtures" / "v133"
REFD = FX / "reference"


def _ref() -> dict[tuple[str, str], float]:
    with open(REFD / "reference_metrics.csv", encoding="utf-8") as fh:
        return {(r["case"], r["metric"]): float(r["value"]) for r in csv.DictReader(fh)}


REF = _ref()


def near(a, b, atol=1e-4, rtol=1e-6) -> bool:
    return a is not None and b is not None and abs(float(a) - float(b)) <= atol + rtol * abs(float(b))


def near_p(a, b, tol=1e-3) -> bool:
    return a is not None and 0 < float(a) <= 1 and abs(np.log10(float(a)) - np.log10(float(b))) <= tol


def f43():
    return import_file(FX / "Test43B.csv")


def fx(name: str):
    return import_file(REFD / f"{name}.csv")


def fit_row(res) -> dict:
    t = next(t for t in res.tables if t.name.startswith("Model fit"))
    return dict(zip(t.columns, t.rows[0]))


def disc(d, **kw):
    from statexplore.discovery import engine
    from statexplore.discovery.space import SearchSettings

    base = dict(outcome="Y", family="gaussian", workers=1, max_models=None, max_minutes=None, study="longitudinal",
                participant="ID", time="TIME")
    base.update(kw)
    run = engine.create_run(d, SearchSettings(**base))
    engine.execute(run, d)
    return run


def rscript(code: str, cwd: Path, timeout: int = 900) -> subprocess.CompletedProcess:
    from statexplore.rbridge.locate import find_rscript

    lib = Path(__file__).resolve().parents[1] / "r" / "library"
    env = dict(os.environ, R_LIBS=str(lib), R_LIBS_USER=str(lib))
    (cwd / "oracle.R").write_text(code, encoding="utf-8")
    return subprocess.run([str(find_rscript()), "--vanilla", str(cwd / "oracle.R")], cwd=cwd, capture_output=True,
                          text=True, timeout=timeout, env=env)


# 18 base configurations (06 matrix): (random, residual) -> discovery structure key and Mixed settings
CONFIGS = {("RI", "independent"): ("ri", "intercept", "independent"),
           ("DIAG", "independent"): ("rs_uncorr", "slope_uncorrelated", "independent"),
           ("RS", "independent"): ("rs", "slope", "independent"),
           ("RI", "AR1"): ("ar1", "intercept", "ar1"), ("DIAG", "AR1"): ("diag_ar1", "slope_uncorrelated", "ar1"),
           ("RS", "AR1"): ("rs_ar1", "slope", "ar1"), ("RI", "CAR1"): ("car1", "intercept", "car1"),
           ("DIAG", "CAR1"): ("diag_car1", "slope_uncorrelated", "car1"), ("RS", "CAR1"): ("rs_car1", "slope", "car1")}
MATRIX = [(cl, re, res) for cl in (False, True) for (re, res) in CONFIGS]


def case(cl: bool, re: str, res: str) -> str:
    return f"{'cluster' if cl else 'participant'}_{re}_{res}"


# ---- fixtures ------------------------------------------------------------------------------------------------------

def test_fixture_bytes_match_manifest_and_package_hash():
    for line in (FX / "SHA256SUMS.txt").read_text().splitlines():
        h, name = line.split(maxsplit=1)
        assert hashlib.sha256((FX / name).read_bytes()).hexdigest() == h, name
    # the original 43B export is byte-identical to the archived v1.3.2 manifest hash (PACKAGE_VALIDATION.md)
    assert hashlib.sha256((FX / "Test43B_export_original.csv").read_bytes()).hexdigest() == \
        "c9427dc26c08a2178512371372428124fcf9f8574d01ac4848395b34cbb0b148"
    d = f43()
    f = d.analysis_frame(["SCHOOL", "ID", "TIME", "GROUP", "Y"])
    assert len(f) == 960 and f["ID"].nunique() == 160 and f["SCHOOL"].nunique() == 20 and not f.isna().any().any()


# ---- G1: term grammar, diff, fitted-vs-requested (pure) ------------------------------------------------------------

def test_term_presets_custom_and_hierarchy():
    s = mixed.MixedSpec(outcome="Y", id="ID", time="TIME", predictors=["GROUP", "Z"])
    assert mixed.term_set(mixed.preset_terms(s, "time")) == {frozenset(x) for x in (
        ["TIME"], ["GROUP"], ["Z"], ["TIME", "GROUP"], ["TIME", "Z"])}
    assert mixed.matching_preset(s, [["TIME"], ["GROUP"], ["Z"]]) == "none"
    custom = [["TIME"], ["GROUP"], ["Z"], ["TIME", "Z"]]
    assert mixed.matching_preset(s, custom) == "custom"
    assert not mixed.hierarchy_problems(s, custom)
    assert mixed.hierarchy_problems(s, [["TIME"], ["TIME", "Z"]])                     # Z main effect missing
    q = mixed.MixedSpec(outcome="Y", id="ID", time="TIME", time_type="quadratic", predictors=["GROUP"])
    rem = mixed.remap_time_terms(custom[:2] + [["TIME", "GROUP"]], "TIME", "quadratic")
    assert {frozenset(t) for t in rem} == {frozenset(x) for x in (["TIME"], ["TIME^2"], ["GROUP"], ["TIME", "GROUP"],
                                                                   ["TIME^2", "GROUP"])}
    assert not mixed.hierarchy_problems(q, rem)
    assert mixed.remap_time_terms(rem, "TIME", "categorical") == [["TIME"], ["GROUP"], ["TIME", "GROUP"]]
    # 'All two-way' with quadratic time never lists time x time^2 (the formula never fits it)
    assert frozenset(["TIME", "TIME^2"]) not in mixed.term_set(mixed.preset_terms(q, "all"))


def test_spec_diff_ignores_presentation_and_lists_model_changes():
    o = mixed.MixedSpec(outcome="Y", id="ID", time="TIME", predictors=["GROUP"], residual="ar1", method="ML",
                        terms=[["TIME"], ["GROUP"], ["GROUP", "TIME"]])
    same = mixed.MixedSpec(**{**o.to_dict(), "n_trajectories": 20, "terms": None, "interactions": "time"})
    assert mixed.spec_diff(o, same) == []                                        # plot setting, same term SET
    reml = mixed.MixedSpec(**{**o.to_dict(), "method": "REML"})
    assert mixed.spec_diff(o, reml) == ["estimation: 'ML' → 'REML'"]
    drop = mixed.MixedSpec(**{**o.to_dict(), "terms": [["TIME"], ["GROUP"]]})
    assert any("removed GROUP × TIME" in x for x in mixed.spec_diff(o, drop))


def test_fitted_vs_requested_fails_closed_on_injected_mismatch():
    """Deterministic injection (not a real R failure): an effective spec lacking the interaction must fail the result."""
    from statexplore.core.schema import AnalysisResult, Status

    d = f43()
    s = mixed.MixedSpec(outcome="Y", id="ID", time="TIME", predictors=["GROUP"], cluster="SCHOOL", random="slope",
                        residual="ar1", method="ML")
    ok = {"class": "lme", "fixed": ["sx_time", "GROUP", "sx_time:GROUP"], "groups": ["sx_cluster", "sx_id"],
          "blocks": [{"group": "sx_cluster", "columns": ["(Intercept)"], "structure": "diagonal"},
                     {"group": "sx_id", "columns": ["(Intercept)", "sx_time"], "structure": "correlated"}],
          "correlation": "corAR1", "method": "ML"}
    from statexplore.rbridge.codegen import alias

    ga = alias(d.var("GROUP").alias)
    ok["fixed"] = ["sx_time", ga, f"sx_time:{ga}"]
    assert mixed.effective_problems(d, s, ok) == []
    for bad, frag in ((dict(ok, fixed=["sx_time", ga]), "missing in the fit"), (dict(ok, correlation="none"),
                      "residual covariance"), (dict(ok, method="REML"), "estimation"),
                      (dict(ok, groups=["sx_id"]), "grouping"), (dict(ok, cls="x", **{"class": "lmerMod"}), "engine"),
                      (None, "could not be inspected")):
        probs = mixed.effective_problems(d, s, bad)
        assert any(frag in p for p in probs), (frag, probs)
        r = AnalysisResult(kind="mixed_model", title="t", engine=None, script="", inputs={})
        mixed.check_fitted(r, d, s, bad)
        assert r.status is Status.FAILED and r.tables[0].name == "Requested vs fitted specification"


# ---- G1: STAT-47 family through the real GUI and R -----------------------------------------------------------------

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


def _mixed_page(window, d):
    from statexplore.core.project import Project

    window.state.set_project(Project(d))
    page = window.pages["Mixed & Longitudinal Models"]
    results = []
    orig = page._done

    def done(o):
        results.append(o[0])
        orig(o)
    page._done = done
    return page, results


def _wait(qtbot, results, n, trigger, timeout=600000):
    trigger()
    qtbot.waitUntil(lambda: len(results) > n, timeout=timeout)
    return results[-1]


@pytest.mark.r
def test_stat47_transfer_refit_plot_and_reml_keep_the_model(window, qtbot):
    """44A/44C-1/2/4/44D: controls show the exact transferred model; repeated Fit, a plot-only change and ML->REML keep
    TIME x GROUP, covariance, rows and origin; REML equals the independent REML reference; return to origin works."""
    d = f43()
    page, results = _mixed_page(window, d)
    d = window.state.dataset
    run = disc(d, candidates=["GROUP"], cluster="SCHOOL", structure="rs_ar1", strategy="time")
    row = max(run.rows, key=lambda r: len(r.get("terms", [])))
    spec = mixed.spec_from_discovery(run, row)
    want = {frozenset(["TIME"]), frozenset(["GROUP"]), frozenset(["GROUP", "TIME"])}
    res = _wait(qtbot, results, 0, lambda: window.state.show_in_mixed.emit(spec))
    assert page.interactions.currentData() == "time"                           # V133-STATE-01: not "None"
    assert mixed.term_set(page.terms_editor.terms()) == want
    full = "A_cluster_RS_AR1_full"
    for step in range(3):                                                      # transfer, Fit, Fit again
        assert mixed.term_set([t.split("*") for t in res.values["model_spec"]["fixed_terms"]]) == want
        assert res.values["identity_check"] == "verified", step
        assert res.values["fit_check"].startswith("fitted model = requested")
        fr = fit_row(res)
        assert fr["observations"] == 960 and fr["parameters"] == REF[(full, "k")]
        assert near(fr["logLik"], REF[(full, "logLik")]) and near(fr["AIC"], REF[(full, "AIC")])
        if step < 2:
            res = _wait(qtbot, results, len(results), page._fit)
    page.ntraj.setValue(20)                                                    # plot-only change, then Fit
    assert "plot settings" in page.lineage.text()
    res = _wait(qtbot, results, len(results), page._fit)
    assert res.values["identity_check"] == "verified" and near(fit_row(res)["logLik"], REF[(full, "logLik")])
    page.method.setCurrentText("REML")                                         # V133-STATE-03
    assert "Modified" in page.lineage.text() and "estimation" in page.lineage.text()
    res = _wait(qtbot, results, len(results), page._fit)
    fr = fit_row(res)
    b = "B_cluster_RS_AR1"
    assert fr["method"] == "REML" and fr["parameters"] == REF[(b, "k")] and fr["BIC_n"] == 956
    assert near(fr["logLik"], REF[(b, "logLik")]) and near(fr["AIC"], REF[(b, "AIC")]) and near(fr["BIC"], REF[(b, "BIC")])
    assert res.values["identity_check"].startswith("modified") and res.values["spec_diff"] == ["estimation: 'ML' → 'REML'"]
    assert res.values["provenance"]["Origin"].startswith("discovery search")      # V133-STATE-04 lineage kept
    page.return_to_origin()
    assert page.method.currentText() == "ML" and "Unchanged" in page.lineage.text()
    res = _wait(qtbot, results, len(results), page._fit)
    assert res.values["identity_check"] == "verified"


@pytest.mark.r
def test_custom_term_subset_and_hierarchy_refusal(window, qtbot, r_session):
    """V133-STATE-05: an arbitrary subset (TIME x Z only, with W and GROUP as main effects) on >= 3 predictors fits
    exactly; the fitted term set is read back from R; a non-hierarchical set is refused, never silently changed."""
    d = fx("CustomTerms")
    from statexplore.core.variables import Measure

    d.set_variable_field("Z", "measure", Measure.CONTINUOUS)                    # Z = ID mod 3 used as a number
    page, results = _mixed_page(window, d)
    for combo, v in ((page.outcome, "Y"), (page.idv, "ID"), (page.time, "TIME"), (page.cluster, "SCHOOL")):
        combo.setCurrentIndex(combo.findData(v))
    page.pred_sel.set_selected(["GROUP", "Z", "W"])
    page.pred_sel.changed.emit()
    page.interactions.setCurrentIndex(page.interactions.findData("none"))
    terms = page.terms_editor.terms() + [["TIME", "Z"]]
    page.terms_editor.set_checked(terms)
    page.terms_editor.changed.emit()
    assert page.interactions.currentData() == "custom"
    s = page._spec()
    assert s.interactions == "custom" and mixed.term_set(s.terms) == mixed.term_set(terms)
    page.method.setCurrentText("ML")
    res = _wait(qtbot, results, len(results), page._fit)
    assert res.values["fit_check"].startswith("fitted model = requested"), res.messages
    got = mixed.effective_terms(window.state.dataset, s, [str(x) for x in res.values["effective_spec"]["fixed"]])
    assert got == mixed.term_set(terms)
    # independent R: the same model written by hand
    tmp = Path(r_session.workdir)
    out = rscript("d <- read.csv('" + (REFD / "CustomTerms.csv").as_posix() + "'); d$SCHOOL <- factor(d$SCHOOL)\n"
                  "d$PID <- interaction(d$SCHOOL, d$ID, drop = TRUE)\n"
                  "m <- lme4::lmer(Y ~ TIME + GROUP + Z + W + TIME:Z + (1 | SCHOOL) + (1 | PID), data = d, REML = FALSE)\n"
                  "cat(sprintf('%.10f', as.numeric(logLik(m))))\n", tmp)
    assert out.returncode == 0, out.stderr
    assert near(fit_row(res)["logLik"], float(out.stdout.strip().split()[-1]))
    page.terms_editor.set_checked([t for t in terms if t != ["Z"]])            # TIME x Z without Z
    page.terms_editor.changed.emit()
    assert "Hierarchy" in page.terms_editor.note.text()
    with pytest.raises(mixed.MixedInputError, match="Hierarchy"):
        mixed.validate(window.state.dataset, page._spec())


# ---- G2: time comparison keeps the covariance (V133-STAT-01, 48C) ---------------------------------------------------

@pytest.mark.r
def test_48c_time_comparison_keeps_ar1_cluster_and_terms(r_session):
    d = fx("Test46A_IrregularTime")
    base = mixed.MixedSpec(outcome="Y", id="ID", time="TIME", predictors=["GROUP"], cluster="SCHOOL", random="intercept",
                           residual="ar1", method="REML", terms=[["TIME"], ["GROUP"], ["TIME", "GROUP"]])
    res, _ = mixed.compare_specs(r_session, d, base, "time")
    t = next(t for t in res.tables if t.name.startswith("Specification"))
    rows = {r[0]: dict(zip(t.columns, r)) for r in t.rows}
    for label, key in (("Continuous (linear)", "48C_AR1_linear"), ("Continuous (quadratic)", "48C_AR1_quadratic"),
                       ("Categorical (one mean per occasion)", "48C_AR1_categorical")):
        r = rows[label]
        assert r["engine"] == "nlme" and r["estimation"] == "ML" and "residual AR(1)" in r["fitted structure"]
        assert "cluster: 1" in r["fitted structure"]
        assert r["parameters"] == REF[(key, "k")] and near(r["logLik"], REF[(key, "logLik")]), label
    assert any("ML fits; your model is REML" in m for m in res.messages)
    # the separate independent-residual control is a different model and is reproduced only when requested
    ind = mixed.MixedSpec(**{**base.to_dict(), "residual": "independent"})
    res2, _ = mixed.compare_specs(r_session, d, ind, "time")
    t2 = next(t for t in res2.tables if t.name.startswith("Specification"))
    r2 = dict(zip(t2.columns, t2.rows[0]))
    assert r2["engine"] == "lme4" and near(r2["logLik"], REF[("48C_independent_linear", "logLik")])


# ---- G2: composite participant key (V133-STAT-02 / 03) ---------------------------------------------------------------

def test_45a_preflight_composite_key_counts():
    from statexplore.analysis.longitudinal_preflight import preflight

    d = fx("Test45A_ReusedIDs")
    for pol in ("local", "global"):
        sp = mixed.MixedSpec(outcome="Y", id="ID", time="TIME", predictors=["GROUP"], cluster="SCHOOL", id_policy=pol,
                             residual="ar1")
        r = preflight(d, "Y", "ID", "TIME", ["GROUP"], sp)
        assert (r.n_participants, r.duplicates, r.n_usable, r.n_clusters, r.n_crossing) == (160, 0, 160, 20, 8)
        assert r.id_key.startswith("composite")
        assert bool(r.warnings) == (pol == "global")
    with pytest.raises(mixed.MixedInputError, match="more than one"):
        mixed.validate(d, mixed.MixedSpec(outcome="Y", id="ID", time="TIME", cluster="SCHOOL", id_policy="global"))
    dup = preflight(fx("Test46C_DuplicateTime"), "Y", "ID", "TIME", [], cluster="SCHOOL")
    assert dup.duplicates == 1 and dup.duplicate_examples[0][0].startswith("(SCHOOL 1, ID 1)")


@pytest.mark.r
def test_45a_local_ids_fit_matches_reference_and_unclustered_keeps_composite(r_session):
    d = fx("Test45A_ReusedIDs")
    s = mixed.MixedSpec(outcome="Y", id="ID", time="TIME", predictors=["GROUP"], cluster="SCHOOL", id_policy="local",
                        random="slope", residual="ar1", method="ML", interactions="time")
    res, _ = mixed.run(r_session, d, s)
    fr = fit_row(res)
    assert fr["participants"] == 160 and fr["clusters"] == 20 and near(fr["logLik"], REF[("45A_cluster_RS_AR1", "logLik")])
    off = mixed.MixedSpec(**{**s.to_dict(), "cluster_random": False, "random": "intercept", "residual": "independent",
                             "method": "REML"})
    res2, _ = mixed.run(r_session, d, off)
    assert fit_row(res2)["participants"] == 160 and res2.values["fit_check"].startswith("fitted model = requested")


# ---- G2: 18 configurations x both workflows + standalone (06 matrix, 36 obligations) ----------------------------------

@pytest.mark.r
@pytest.mark.parametrize("cl,re,res", MATRIX, ids=[case(*m) for m in MATRIX])
def test_matrix_workflow_a_focal_matches_reference(cl, re, res):
    key = case(cl, re, res)
    if (f"A_{key}_focal", "LR_raw") not in REF:
        pytest.fail(f"reference missing for {key}")
    run = disc(f43(), candidates=["GROUP"], cluster="SCHOOL" if cl else None, structure=CONFIGS[(re, res)][0],
               strategy="time")
    row = max(run.rows, key=lambda r: len(r.get("terms", [])))
    assert row["status"] in ("ok", "warning") and row["n"] == 960 and row["n_groups"] == 160
    assert row["modeled_clusters"] == (20 if cl else 0)
    assert near(row["loglik"], REF[(f"A_{key}_full", "logLik")]), key
    t = next(t for t in row["tests"] if "interaction" in t["type"])
    assert near(t["statistic"], REF[(f"A_{key}_focal", "LR_raw")], 1e-4, 1e-5), key
    assert near_p(t["p"], REF[(f"A_{key}_focal", "p")]), key


@pytest.mark.r
@pytest.mark.parametrize("cl", [False, True], ids=["participant", "cluster"])
def test_matrix_workflow_b_nine_structures_match_reference(cl):
    run = disc(f43(), candidates=["GROUP"], cluster="SCHOOL" if cl else None, workflow="structure",
               structures=[v[0] for v in CONFIGS.values()], time_interactions=True)
    by = {r["structure"]: r for r in run.rows}
    assert len(by) == 9
    for (re, res), (st, _, _) in CONFIGS.items():
        key = f"B_{case(cl, re, res)}"
        r = by[st]
        assert r["status"] in ("ok", "warning") and near(r["loglik"], REF[(key, "logLik")]), key
        assert near(r["aic"], REF[(key, "AIC")]) and near(r["bic"], REF[(key, "BIC")]), key


@pytest.mark.r
@pytest.mark.parametrize("cl,re,res", MATRIX, ids=[case(*m) for m in MATRIX])
def test_matrix_standalone_full_analysis_reml_matches_reference(r_session, cl, re, res):
    _, rnd, resid = CONFIGS[(re, res)]
    s = mixed.MixedSpec(outcome="Y", id="ID", time="TIME", predictors=["GROUP"], cluster="SCHOOL" if cl else None,
                        random=rnd, residual=resid, method="REML", interactions="time")
    out, _ = mixed.run(r_session, f43(), s)
    key = f"B_{case(cl, re, res)}"
    assert out.values["fit_check"].startswith("fitted model = requested"), out.messages
    fr = fit_row(out)
    assert fr["parameters"] == REF[(key, "k")] and near(fr["logLik"], REF[(key, "logLik")]), key
    assert near(fr["BIC"], REF[(key, "BIC")]) and fr["clusters" if cl else "participants"] == (20 if cl else 160)


# ---- G2: non-Gaussian likelihood references (V133-CMP-01) -------------------------------------------------------------

@pytest.mark.r
def test_glmm_and_clmm_match_independent_likelihoods(r_session):
    with open(REFD / "non_gaussian_likelihood.csv", encoding="utf-8") as fh:
        ref = {r["case"]: r for r in csv.DictReader(fh)}
    d = fx("NonGaussian")
    d.set_variable_field("ORD", "measure", "ordinal")
    for fam, y, key in (("binomial", "BIN", "GLMM"), ("ordinal", "ORD", "CLMM")):
        for inter, part in (("none", "reduced"), ("time", "full")):
            s = mixed.MixedSpec(outcome=y, id="PID", time="TIME", predictors=["GROUP"], family=fam, interactions=inter,
                                random="intercept")
            res, _ = mixed.run(r_session, d, s)
            fr = fit_row(res)
            r = ref[f"{key}_{part}"]
            assert res.values["fit_check"].startswith("fitted model = requested"), res.messages
            assert fr["observations"] == int(r["n"]) and near(fr["logLik"], float(r["logLik"]), 1e-3, 1e-6), (key, part)
            assert near(fr["AIC"], float(r["AIC"]), 1e-3, 1e-6)


# ---- G3: validation engine (V133-VAL-01) ----------------------------------------------------------------------------

def _val_run(d=None):
    return disc(d if d is not None else f43(), candidates=["GROUP"], required=["TIME"], cluster="SCHOOL", structure="ri", strategy="main",
                include_main=True)


@pytest.mark.r
def test_validation_splits_deterministic_disjoint_and_draw_ids_unique():
    from statexplore.discovery import validate as V

    d = f43()
    run = _val_run()
    u = V.unit_frame(d, run)
    assert len(u) == 960 and u["participant"].nunique() == 160 and u["cluster"].nunique() == 20
    for target in ("cluster", "participant_in_cluster", "participant"):
        cfg = V.ValidationConfig(target=target, modes=["holdout", "cv", "bootstrap"], k=5, repeats=2, B=5)
        a, b = V.make_splits(u, cfg), V.make_splits(u, cfg)
        assert [s["folds"] for s in a] == [s["folds"] for s in b]                   # same seed -> same assignment
        other = V.make_splits(u, V.ValidationConfig(**{**cfg.to_dict(), "seed": 7}))
        assert [s["folds"] for s in a] != [s["folds"] for s in other]
        assert a[1]["folds"] != a[2]["folds"]                                      # repeats differ
        for sp in a:
            assert V.check_split(u, sp, target) == []
            f = u.assign(fold=u["row_id"].map(sp["folds"]))
            assert (f.groupby("participant")["fold"].nunique() == 1).all()         # a participant stays together
            if sp["mode"] == "cv":
                assert set(f["fold"]) == {1, 2, 3, 4, 5}
        if target == "participant_in_cluster":
            h = u.assign(fold=u["row_id"].map(a[0]["folds"]))
            per = h[h["fold"] == 1].groupby("cluster")["participant"].nunique()
            assert len(per) == 20 and (per == 2).all()                             # ceil(20% of 8) per school
        draws = V.make_draws(u, cfg)
        for dr in draws:
            gids = [r[1] for r in dr["rows"]]
            assert len(set(gids)) == dr["units"] and len(dr["rows"]) == 6 * dr["units"]
            if target == "cluster":
                assert len({r[2] for r in dr["rows"]}) == 20                     # each drawn school = new school ID
                orig = [r[4] for r in dr["rows"]]
                assert len(set(orig)) <= 20
            else:
                assert all(r[2] == "" for r in dr["rows"])                       # schools kept
    bad = V.feasibility(d, run, V.ValidationConfig(target="rows"))
    assert any("never split at random" in p for p in bad["problems"])


@pytest.mark.r
def test_validation_predictions_match_independent_oracle_and_do_not_leak(r_session, tmp_path):
    """The app's held-out predictions for both cluster targets equal an independent lme4 fit on the same manifest; test
    outcomes changed by +10000 leave the training fit and predictions unchanged (metrics change)."""
    from statexplore.discovery import validate as V

    d = f43()
    run = _val_run()
    for target in ("cluster", "participant_in_cluster"):
        cfg = V.ValidationConfig(target=target, modes=["holdout", "cv"], pipeline="fixed", candidate=run.rows[0]["idx"],
                                 k=5)
        v = V.ValidationRunner(r_session, d, run, cfg).execute()
        assert v["status"] == "complete" and all(f["status"] == "scored" for f in v["folds"])
        assert sum(f["n_test"] for f in v["folds"] if f["split"] == "cv1") == 960
        man = tmp_path / f"man_{target}.csv"
        man.write_text(V.csv_text(v["manifest"]), encoding="utf-8")
        pred = tmp_path / f"pred_{target}.csv"
        rf = "NA" if target == "cluster" else "~(1 | SCHOOL)"
        out = rscript(f"""
d <- read.csv('{(FX / "Test43B.csv").as_posix()}'); d$row_id <- seq_len(nrow(d))
d$SCHOOL <- factor(d$SCHOOL); d$PID <- interaction(d$SCHOOL, d$ID, drop = TRUE)
m <- read.csv('{man.as_posix()}'); res <- list()
for (sp in unique(m$split)) {{ mm <- m[m$split == sp, ]; fo <- mm$fold[match(d$row_id, mm$row_id)]
  for (f in setdiff(sort(unique(fo)), 0)) {{ tr <- d[fo != f, ]; te <- d[fo == f, ]
    fit <- lme4::lmer(Y ~ GROUP + TIME + (1 | SCHOOL) + (1 | PID), data = tr, REML = FALSE)
    p <- predict(fit, newdata = te, re.form = {rf}, allow.new.levels = TRUE)
    res[[length(res) + 1]] <- data.frame(split = sp, row_id = te$row_id, oracle = p) }} }}
write.csv(do.call(rbind, res), '{pred.as_posix()}', row.names = FALSE)
""", tmp_path)
        assert out.returncode == 0, out.stderr
        import pandas as pd

        o = pd.read_csv(pred)
        a = pd.DataFrame(v["predictions"])
        mrg = a.merge(o, on=["split", "row_id"])
        assert len(mrg) == len(a) == len(o)
        assert np.max(np.abs(mrg["predicted"] - mrg["oracle"])) < 1e-6, target
    # leakage adversary: holdout test outcomes shifted -> same training model, same predictions, different metrics
    cfg = V.ValidationConfig(target="cluster", modes=["holdout"], pipeline="fixed", candidate=run.rows[0]["idx"])
    base = V.ValidationRunner(r_session, d, run, cfg).execute()
    test_rows = {p["row_id"] for p in base["predictions"]}
    import pandas as pd

    raw = pd.read_csv(FX / "Test43B.csv")
    raw.loc[[r - 1 for r in test_rows], "Y"] += 10000
    mut = tmp_path / "mut.csv"
    raw.to_csv(mut, index=False)
    d2 = import_file(mut)
    run2 = _val_run(d2)
    v2 = V.ValidationRunner(r_session, d2, run2, cfg).execute()
    p1 = {p["row_id"]: p["predicted"] for p in base["predictions"]}
    p2 = {p["row_id"]: p["predicted"] for p in v2["predictions"]}
    assert p1.keys() == p2.keys() and max(abs(p1[k] - p2[k]) for k in p1) < 1e-8
    assert base["folds"][0]["train_mean"] == pytest.approx(v2["folds"][0]["train_mean"])
    assert v2["performance"][0]["RMSE"] > 1000 > base["performance"][0]["RMSE"]


@pytest.mark.r
def test_validation_pipeline_bootstrap_failures_and_persistence(r_session, tmp_path):
    from statexplore.discovery import engine
    from statexplore.discovery import validate as V

    d = f43()
    run = disc(d, candidates=["GROUP"], cluster="SCHOOL", structure="ri", strategy="time", include_main=True)
    assert len(run.rows) >= 2
    cfg = V.ValidationConfig(target="cluster", modes=["cv", "bootstrap"], pipeline="pipeline", rule="p", k=4, B=6)
    v = V.ValidationRunner(r_session, d, run, cfg).execute()
    assert v["status"] == "complete" and len(v["bootstrap"]) == 6
    st = {s["candidate"]: s for s in v["stability"]}
    assert sum(s["selected"] for s in st.values()) == 6 and all(s["attempted"] == 6 for s in st.values())
    inter = next(s for s in st.values() if "×" in s["candidate"] or "TIME" in s["candidate"])
    assert inter["selected"] >= 5 and inter["positive_sign"] == 1.0                # the planted TIME x GROUP is stable
    # an unavailable target is refused with its reason, never replaced
    with pytest.raises(V.ValidationError, match="never split at random"):
        V.ValidationRunner(r_session, d, run, V.ValidationConfig(target="rows"))
    # partial: a stop request after the first chunk keeps the completed folds and is labelled, not complete
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] > 1
    p = V.ValidationRunner(r_session, d, run, V.ValidationConfig(target="cluster", modes=["holdout", "cv"], k=4)) \
        .execute(should_stop=stop)
    assert p["status"] == "cancelled" and {f["split"] for f in p["folds"]} == {"holdout"}
    # persistence: the run stores its validation records; export writes the evidence files
    run.validation_runs.append(v)
    back = engine.DiscoveryRun.from_meta(run.meta_dict(), run.rows)
    assert back.validation_runs[0]["manifest"] == v["manifest"] and back.validation_runs[0]["config"]["seed"] == 133
    files = V.export(v, str(tmp_path / "out dir é"))
    assert {Path(f).name for f in files} >= {"split_manifest.csv", "bootstrap_draws.csv", "predictions.csv",
                                             "folds.csv", "performance.csv", "stability.csv", "validation_code.R",
                                             "validation_manifest.json"}


# ---- G4: portable reproducibility bundles (V133-EXP-01..04, Test 44B / 52) -------------------------------------------

def _run_bundle(folder: Path, script: str) -> subprocess.CompletedProcess:
    from statexplore.rbridge.locate import find_rscript

    lib = Path(__file__).resolve().parents[1] / "r" / "library"
    env = dict(os.environ, R_LIBS=str(lib), R_LIBS_USER=str(lib))
    return subprocess.run([str(find_rscript()), "--vanilla", script], cwd=folder, capture_output=True, text=True,
                          timeout=900, env=env)


@pytest.mark.r
def test_full_analysis_bundle_runs_relocated_offline_and_fails_closed(r_session, tmp_path):
    import json
    import shutil

    from statexplore.export.exporters import fit_expectations, write_portable

    d = f43()
    run = disc(d, candidates=["GROUP"], cluster="SCHOOL", structure="rs_ar1", strategy="time")
    row = max(run.rows, key=lambda r: len(r.get("terms", [])))
    res, _ = mixed.run(r_session, d, mixed.spec_from_discovery(run, row))
    assert res.values["identity_check"] == "verified"
    first = tmp_path / "export dir é"
    files = write_portable(res.script, first, d, provenance=res.values.get("provenance"),
                           identity=res.values.get("identity"), engine=res.engine, expected=fit_expectations(res),
                           packages=sorted(res.engine.packages),
                           extra={"model_spec": res.values["model_spec"]})
    assert {p.name for p in files} == {"analysis.R", "analysis_data.csv", "analysis_manifest.json", "README.txt"}
    text = (first / "analysis.R").read_text(encoding="utf-8")
    assert "sx_tmp" not in text and "AppData" not in text and "Temp" not in text      # V133-EXP-01: no temp paths
    man = json.loads((first / "analysis_manifest.json").read_text(encoding="utf-8"))
    assert man["r_versions"]["r"] and man["provenance"] and man["identity"]["schema"] >= 2      # V133-EXP-02
    assert man["expected_results"]["n"] == 960 and set(man["r_packages"]) <= {"lme4", "lmerTest", "nlme", "performance",
                                                                               "ggplot2", "Matrix"}
    moved = tmp_path / "moved here" / "bundle ü"
    moved.parent.mkdir()
    shutil.move(str(first), str(moved))                                        # relocation; app objects not used
    out = _run_bundle(moved, "analysis.R")
    assert out.returncode == 0, out.stderr[-2000:]
    assert "Model identity reproduced" in out.stdout and "Results reproduced" in out.stdout
    # corrupted data: fails closed with an actionable message
    dp = moved / "analysis_data.csv"
    dp.write_text(dp.read_text(encoding="utf-8").replace("21.9512023136467", "21.9512023136468", 1), encoding="utf-8")
    bad = _run_bundle(moved, "analysis.R")
    assert bad.returncode != 0 and "SHA-256 differs" in bad.stderr
    # without data consent: no data file, README says where it must come from
    nod = write_portable(res.script, tmp_path / "nodata", d, include_data=False, identity=res.values.get("identity"))
    assert not (tmp_path / "nodata" / "analysis_data.csv").exists() and len(nod) == 3
    assert "NOT included" in (tmp_path / "nodata" / "README.txt").read_text(encoding="utf-8")


@pytest.mark.r
def test_amd_candidate_bundle_manifest_complete_and_reproduces(r_session, tmp_path):
    import json

    from statexplore.discovery.detail import candidate_from_row
    from statexplore.discovery.rcode import candidate_script
    from statexplore.discovery.space import view
    from statexplore.analysis.descriptor import from_discovery, provenance
    from statexplore.export.exporters import FAMILY_PACKAGES, write_portable

    d = f43()
    run = disc(d, candidates=["GROUP"], cluster="SCHOOL", structure="rs_ar1", strategy="time")
    row = max(run.rows, key=lambda r: len(r.get("terms", [])))
    v = view(d, run.settings)
    r_session.ensure_data(v)
    prov = provenance(from_discovery(run, row))
    script = candidate_script(r_session.load_script, v, run.settings, run.family, candidate_from_row(run, row), [],
                              expected=row, adjusted=[], provenance=prov)
    write_portable(script, tmp_path / "amd", d, provenance=prov, identity=row.get("identity"), engine=run.engine,
                   packages=FAMILY_PACKAGES[run.family])
    man = json.loads((tmp_path / "amd" / "analysis_manifest.json").read_text(encoding="utf-8"))
    assert man["r_versions"] and man["r_versions"]["r"] and man["provenance"] and man["identity"]
    assert set(man["r_packages"]) <= set(FAMILY_PACKAGES["lmm"])                   # V133-EXP-04: no geepack / nnet
    assert not {"geepack", "nnet", "ordinal", "detectseparation"} & set(man["r_packages"])
    out = _run_bundle(tmp_path / "amd", "analysis.R")
    assert out.returncode == 0, out.stderr[-2000:]
    assert "Model identity reproduced" in out.stdout


# ---- G2: diagnostics, optimizer fallback and LR guards (V133-DIAG-01, STAT-32 / 41) -------------------------------------

@pytest.mark.r
def test_optimizer_fallback_same_spec_structural_no_retry_all_fail(r_session):
    """Deterministic fault injection through the real attempt controller (first optimizer failure simulated, second
    optimizer a real fit). This is injection, not an independently reproduced optimizer failure."""
    from statexplore.analysis.covstruct import MIXED_R_HELPERS

    code = MIXED_R_HELPERS + f"""
d <- read.csv('{(FX / "Test43B.csv").as_posix()}'); d$sx_id <- factor(d$ID)
f <- function(opt) {{ if (opt == "nlminb") stop("nlminb problem, convergence error code = 1 (injected)")
  nlme::lme(Y ~ TIME * GROUP, random = ~ 1 | sx_id, data = d, method = "ML", control = nlme::lmeControl(opt = opt)) }}
r1 <- sx_fit_attempts(f, c("nlminb", "optim")); dg <- sx_diag(r1$fit, r1)
ref <- nlme::lme(Y ~ TIME * GROUP, random = ~ 1 | sx_id, data = d, method = "ML", control = nlme::lmeControl(opt = "optim"))
n2 <- 0; r2 <- sx_fit_attempts(function(opt) {{ n2 <<- n2 + 1; stop("invalid formula for groups") }}, c("nlminb", "optim"))
r3 <- sx_fit_attempts(function(opt) stop("iteration limit reached without convergence (injected)"), c("nlminb", "optim"))
sx_result <- list(att1 = r1$attempts$status, opt = dg$optimizer, flags = I(dg$flags), clean = dg$clean,
  ll = as.numeric(logLik(r1$fit)), ll_ref = as.numeric(logLik(ref)),
  fixed_same = identical(names(nlme::fixef(r1$fit)), names(nlme::fixef(ref))),
  n2 = n2, att2 = nrow(r2$attempts), fit2 = is.null(r2$fit), att3 = r3$attempts$status, fit3 = is.null(r3$fit),
  err3 = r3$error)
"""
    r_session.ensure_data(f43())
    out = r_session.worker.run(code, packages=["nlme"], timeout=600).result
    assert out["att1"] == ["failed", "fitted"] and out["opt"] == "optim"
    assert out["fixed_same"] and abs(out["ll"] - out["ll_ref"]) < 1e-8                  # same model, other optimizer
    assert any("optimizer fallback" in f for f in out["flags"]) and out["clean"] is False   # never a clean tier
    assert out["n2"] == 1 and out["att2"] == 1 and out["fit2"]                          # structural: no retry
    assert out["att3"] == ["failed", "failed"] and out["fit3"] and "iteration limit" in out["err3"]


def test_lr_material_negative_withheld_and_tiny_negative_clamped():
    from statexplore.analysis.lrtest import lr_test

    t = lr_test(0, -63.968673 / 2, 1, "mixture")
    assert t["status"] == "withheld" and t["p"] is None and t["LR"] is None and t["LR_raw"] == pytest.approx(-63.968673)
    t = lr_test(0, -2.094e-7 / 2, 1, "chisq")
    assert t["status"] == "tested" and t["LR"] == 0 and t["LR_raw"] == pytest.approx(-2.094e-7) and t["p"] == 1.0
    t = lr_test(0, 0.0, 1, "mixture")
    assert t["p"] == 1.0                                                             # point mass at 0
    t = lr_test(0, 400.0, 1, "chisq")                                                # far tail: log scale kept
    assert t["log_p"] is not None and t["log_p"] < -400


# ---- G3: Test 51 grammar, counts, adjustment (V133-DISC-01) -------------------------------------------------------------

def test_51_independent_enumeration_oracle_for_three_candidates():
    """Independent count oracle (itertools, not the engine) for 1-3 candidates and each grammar, plus no candidate."""
    from itertools import combinations

    from statexplore.discovery.space import SearchSettings, SearchSpaceError, count_candidates, enumerate_candidates

    names = ["A", "B", "C"]
    for k in (1, 2, 3):
        c = names[:k]
        main = SearchSettings(outcome="Y", candidates=c, family="gaussian", strategy="main", max_predictors=6)
        want = [set(x) for n in range(1, k + 1) for x in combinations(c, n)]
        got = [set(x.variables) for x in enumerate_candidates(main)]
        assert sorted(map(sorted, got)) == sorted(map(sorted, want)) and count_candidates(main) == len(want)
        focal = SearchSettings(outcome="Y", candidates=c, family="gaussian", strategy="focal", max_predictors=6,
                               max_order=3)
        want_f = [set(x) for o in (2, 3) for x in combinations(c, o)]
        got_f = [set(x.focal) for x in enumerate_candidates(focal)]
        assert sorted(map(sorted, got_f)) == sorted(map(sorted, want_f)) and count_candidates(focal) == len(want_f)
        assert len({x.index for x in enumerate_candidates(focal)}) == len(want_f)              # deterministic IDs
        assert [x.index for x in enumerate_candidates(focal)] == [x.index for x in enumerate_candidates(focal)]
    tv = SearchSettings(outcome="Y", candidates=["A", "B", "C"], family="gaussian", strategy="time", participant="ID",
                        time="T", max_predictors=6, max_order=2)
    got_t = [set(x.focal) for x in enumerate_candidates(tv)]
    assert sorted(map(sorted, got_t)) == [["A", "T"], ["B", "T"], ["C", "T"]]              # time once, never a candidate
    from statexplore.discovery.space import validate_settings

    d = f43()
    with pytest.raises(SearchSpaceError):
        validate_settings(d, SearchSettings(outcome="Y", candidates=["GROUP"], required=["TIME"], family="gaussian",
                                            strategy="time", participant="ID", time="TIME"))
    with pytest.raises(SearchSpaceError):
        validate_settings(d, SearchSettings(outcome="Y", candidates=["TIME"], family="gaussian", participant="ID",
                                            time="TIME"))


@pytest.mark.r
def test_51_bh_holm_per_family_match_p_adjust(r_session, tmp_path):
    """BH / Holm within each hypothesis family equal R's p.adjust on the same vectors (independent oracle)."""
    from statexplore.discovery import engine

    d = f43()
    run = disc(d, candidates=["GROUP"], cluster="SCHOOL", structure="ri", strategy="time", include_main=True)
    tf = engine.tests_frame(run)
    fams = {t: g for t, g in tf[tf["in_family"]].groupby("test_type")}
    assert len(fams) >= 2
    lines = []
    for t, g in fams.items():
        lines.append(f"cat({t!r}, sprintf('%.17g', p.adjust(c({', '.join(repr(float(x)) for x in g['p'])}), 'BH')), "
                     f"sprintf('%.17g', p.adjust(c({', '.join(repr(float(x)) for x in g['p'])}), 'holm')), '\\n')")
    out = rscript("\n".join(lines) + "\n", tmp_path)
    assert out.returncode == 0, out.stderr
    for line in out.stdout.strip().splitlines():
        parts = line.split()
        t, vals = parts[0], [float(x) for x in parts[1:]]
        g = fams[t]
        k = len(g)
        assert np.allclose(g["p_bh"].to_numpy(float), vals[:k], rtol=1e-12, atol=0)
        assert np.allclose(g["p_holm"].to_numpy(float), vals[k:], rtol=1e-12, atol=0)
        assert (g["family_size"] == k).all()
    assert tf["log10_p"].notna().sum() == tf["p"].notna().sum()                           # V133-UI-08


# ---- G4: historical reconstruction adversaries (V133-HIST-01, STAT-39, Test 50B-D) ---------------------------------------

@pytest.mark.r
def test_50_filter_reconstruct_960_verified_new_800_and_current_view_untouched(r_session):
    from statexplore.core.derive import FilterSpec
    from statexplore.discovery import history

    d = f43()
    run = disc(d, candidates=["GROUP"], cluster="SCHOOL", structure="rs_ar1", strategy="time")
    row = max(run.rows, key=lambda r: len(r.get("terms", [])))
    d.add_filter(FilterSpec("TIME", "<=", 4))
    before = [f.to_dict() for f in d.filters]
    ctx = history.context(d, run)
    assert ctx.state == "changed" and any("row filters" in x for x in ctx.differences)
    old, _ = mixed.run(r_session, ctx.original, mixed.spec_from_discovery(run, row))     # Refit original
    assert old.values["identity_check"] == "verified" and fit_row(old)["observations"] == 960
    assert near(fit_row(old)["logLik"], REF[("A_cluster_RS_AR1_full", "logLik")])
    assert [f.to_dict() for f in d.filters] == before                                    # current view unchanged
    spec = mixed.spec_from_discovery(run, row)                                          # New analysis, current view
    spec.descriptor = dict(spec.descriptor, new_analysis=True, new_analysis_reasons=list(ctx.differences))
    new, _ = mixed.run(r_session, d, spec)
    fr = fit_row(new)
    assert fr["observations"] == 800 and new.values["identity_check"].startswith("new analysis")
    assert near(fr["logLik"], REF[("50C_filtered", "logLik")]) and near(fr["AIC"], REF[("50C_filtered", "AIC")])


@pytest.mark.r
def test_50d_three_category_coding_change_detected_and_original_rebuilt(r_session):
    from statexplore.discovery import history

    d = fx("CustomTerms")
    run = disc(d, outcome="Y", candidates=["Z"], cluster="SCHOOL", structure="ri", strategy="time")
    row = max(run.rows, key=lambda r: len(r.get("terms", [])))
    assert d.var("Z").is_categorical and len(d.levels("Z")) == 3
    d.set_variable_field("Z", "reference_level", d.levels("Z")[2])                     # reference 0 -> 2
    ctx = history.context(d, run)
    assert ctx.state == "changed" and any("reference level" in x for x in ctx.differences)
    assert d.var("Z").reference_level == d.levels("Z")[2]                               # current coding kept
    old, _ = mixed.run(r_session, ctx.original, mixed.spec_from_discovery(run, row))
    assert old.values["identity_check"] == "verified"
    assert d.var("Z").reference_level == d.levels("Z")[2]                               # reconstruction isolated
    cur, _ = mixed.run(r_session, d, mixed.spec_from_discovery(run, row))               # same numbers, other coding
    assert cur.values["identity_check"] != "verified"                                   # semantic identity differs


@pytest.mark.r
def test_validation_rows_target_gaussian_and_binary(r_session):
    """Independent (single-level) searches: row target, glm predictions; binary metrics with the training event."""
    from statexplore.discovery import engine
    from statexplore.discovery import validate as V
    from statexplore.discovery.space import SearchSettings

    for name, fam, cands, y in (("Identity_DisjointEqualFit", "gaussian", ["X", "S"], "Y"),
                                ("NonGaussian", "binomial", ["TIME", "GROUP"], "BIN")):
        d = fx(name)
        run = engine.create_run(d, SearchSettings(outcome=y, candidates=cands, family=fam, strategy="main", workers=1,
                                                  max_models=None, max_minutes=None))
        engine.execute(run, d)
        cfg = V.ValidationConfig(target="rows", modes=["holdout", "cv"], k=5, rule="aic")
        v = V.ValidationRunner(r_session, d, run, cfg).execute()
        assert v["status"] == "complete", v["problems"]
        pooled = [p for p in v["performance"] if p["fold"] == 0 and p["split"] == "cv"][0]
        assert pooled["n"] == len(d.raw)
        if fam == "binomial":
            assert {"log_loss", "Brier", "AUC"} <= set(pooled) and 0.5 < pooled["AUC"] < 1
        else:
            assert pooled["RMSE"] > 0 and "R2_out_of_sample" in pooled
        assert all(f["selected"] for f in v["folds"])


# ---- recorded anchors that only the application computes (07_Historical_Targets.csv; 155 others = reference) -----------

def _anchors() -> dict[tuple[str, str], float]:
    out = {}
    with open(FX / "07_Historical_Targets.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                out[(r["case"], r["metric"])] = float(r["value"])
            except ValueError:
                pass
    return out


@pytest.mark.r
def test_07_application_only_anchors(r_session):
    from statexplore.discovery import engine

    A = _anchors()
    run = disc(f43(), candidates=["GROUP"], cluster="SCHOOL", structure="rs_ar1", strategy="time")
    tf = engine.tests_frame(run)
    t = tf[tf["test_type"].str.contains("interaction")].iloc[0]
    for col, key in (("p_bh", "BH"), ("p_holm", "Holm")):
        assert near_p(t[col], A[("43B_AMD", key)])
    for col, key in (("estimate", "interaction"), ("ci_low", "ci_low"), ("ci_high", "ci_high"),
                     ("effect", "delta_R2_marginal")):
        assert near(t[col], A[("43B_AMD", key)], 1e-6, 1e-5), key
    s = mixed.MixedSpec(outcome="Y", id="ID", time="TIME", predictors=["GROUP"], cluster="SCHOOL", random="intercept",
                        residual="ar1", method="REML", interactions="time")
    res, _ = mixed.run(r_session, fx("Test46B_MissingTime"), s)
    assert res.n_used == A[("46B_AR1", "n")] and res.values["n_excluded"] == A[("46B_AR1", "excluded")]
    assert near(fit_row(res)["logLik"], A[("46B_AR1", "logLik")], 1e-6, 1e-8)
    s = mixed.MixedSpec(outcome="Y", id="ID", time="TIME", predictors=["GROUP"], cluster="SCHOOL",
                        random="slope_uncorrelated", residual="independent", method="REML", interactions="time")
    res, _ = mixed.run(r_session, fx("Test49A_Singularity"), s)
    t = next(t for t in res.tables if t.name.startswith("Random effects"))
    vc = [dict(zip(t.columns, r)) for r in t.rows]
    sd = lambda pred: [float(r["sd_or_corr"]) for r in vc if pred(r)]  # noqa: E731
    assert near(sd(lambda r: "SCHOOL" in str(r["group"]))[0], A[("49A_DIAG", "school_sd")], 1e-4, 1e-4)
    assert near(sd(lambda r: r["term"] == "(Intercept)" and "SCHOOL" not in str(r["group"]) and
                   r["group"] != "Residual")[0], A[("49A_DIAG", "id_intercept_sd")], 1e-4, 1e-4)
    assert sd(lambda r: "TIME" in str(r["term"]) or "time" in str(r["term"]))[0] < 1e-4          # slope sd = 0
    assert near(sd(lambda r: r["group"] == "Residual")[0], A[("49A_DIAG", "residual_sd")], 1e-4, 1e-4)
    assert res.values["diagnostics"]["status"] != "clean"                                   # singular is flagged


@pytest.mark.r
def test_validation_gee_participant_target_marginal_mean(r_session, tmp_path):
    """GEE searches: held-out marginal-mean predictions equal an independent geeglm fit on the same manifest."""
    from statexplore.discovery import validate as V

    d = f43()
    run = disc(d, candidates=["GROUP"], structure="gee_exch", strategy="time")
    assert run.family == "gee" and run.rows[0]["status"] in ("ok", "warning")
    cfg = V.ValidationConfig(target="participant", modes=["cv"], pipeline="fixed", candidate=run.rows[0]["idx"], k=4)
    v = V.ValidationRunner(r_session, d, run, cfg).execute()
    assert v["status"] == "complete" and sum(f["n_test"] for f in v["folds"]) == 960
    man = tmp_path / "man.csv"
    man.write_text(V.csv_text(v["manifest"]), encoding="utf-8")
    out = rscript(f"""
d <- read.csv('{(FX / "Test43B.csv").as_posix()}'); d$row_id <- seq_len(nrow(d)); d <- d[order(d$ID, d$TIME), ]
m <- read.csv('{man.as_posix()}'); fo <- m$fold[match(d$row_id, m$row_id)]; res <- list()
for (f in 1:4) {{ tr <- d[fo != f, ]; te <- d[fo == f, ]
  g <- geepack::geeglm(Y ~ TIME * GROUP, id = ID, waves = TIME + 1, data = tr, corstr = "exchangeable")
  res[[f]] <- data.frame(row_id = te$row_id, oracle = as.numeric(predict(g, newdata = te, type = "response"))) }}
write.csv(do.call(rbind, res), '{(tmp_path / "o.csv").as_posix()}', row.names = FALSE)
""", tmp_path)
    assert out.returncode == 0, out.stderr
    import pandas as pd

    mrg = pd.DataFrame(v["predictions"]).merge(pd.read_csv(tmp_path / "o.csv"), on="row_id")
    assert len(mrg) == 960 and np.max(np.abs(mrg["predicted"] - mrg["oracle"])) < 1e-6


@pytest.mark.r
def test_51_cancel_kills_running_batches_and_resume_completes_exactly_once():
    """Cancellation stops the running R batches (no waiting for them); interrupted candidates are not recorded; the
    resume evaluates exactly the missing candidates with the same IDs; results equal an uninterrupted search."""
    import threading
    import time

    from statexplore.discovery import engine
    from statexplore.discovery.space import SearchSettings

    d = fx("CustomTerms")
    from statexplore.core.variables import Measure

    d.set_variable_field("Z", "measure", Measure.CONTINUOUS)
    kw = dict(outcome="Y", candidates=["GROUP", "Z", "W"], family="gaussian", participant="ID", time="TIME",
              cluster="SCHOOL", structure="rs_ar1", strategy="main", max_predictors=6, workers=2, batch_size=1,
              max_models=None, max_minutes=None, study="longitudinal")
    full = engine.create_run(d, SearchSettings(**kw))
    engine.execute(full, d)
    run = engine.create_run(d, SearchSettings(**kw))
    cancel = threading.Event()
    seen = {"t": None}

    def prog(r):
        if r.rows and seen["t"] is None:
            seen["t"] = time.perf_counter()
            cancel.set()
    engine.execute(run, d, progress=prog, cancel=cancel)
    assert run.status == "cancelled" and run.resumable
    assert time.perf_counter() - seen["t"] < 30                                  # did not wait for every batch
    done = sorted(int(r["idx"]) for r in run.rows)
    assert len(done) == len(set(done)) and len(done) < run.total
    assert all(r["status"] in ("ok", "warning") for r in run.rows)               # nothing recorded as failed
    engine.execute(run, d)
    assert run.status == "complete" and sorted(int(r["idx"]) for r in run.rows) == list(range(run.total))
    ref = {int(r["idx"]): r["loglik"] for r in full.rows}
    assert all(abs(r["loglik"] - ref[int(r["idx"])]) < 1e-6 for r in run.rows)
