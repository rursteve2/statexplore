"""v1.3.0 regression tests (spec: StatExplore_v1.3.0_Development_Prompt.md). Synthetic, privacy-safe fixtures
reproduce each root cause; the HRS fixture tests (skipped without the file) check the reference numbers.

  P0-01 model identity across modules      test_P0_01_*
  P0-02 export reproducibility             test_P0_02_*
  P0-03 hypothesis-family headline         test_P0_03_*
  P0-05 CSV whitespace / measurement       test_P0_05_*
  P1-xx inference, time index, reporting   test_P1_*
  P2-xx usability                          test_P2_*
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from statexplore.core.importers import import_file, import_report
from statexplore.core.variables import Measure, Storage

from test_discovery import make_ds


# ---- fixtures -------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def panel_df():
    """Unbalanced panel: waves 8-14 and 16 (wave 15 absent for everyone), intermittent gaps, some single-row
    participants — the BMI structure of Fixture C/D in miniature."""
    rng = np.random.default_rng(13)
    waves = [8, 9, 10, 11, 12, 13, 14, 16]
    rows = []
    for pid in range(150):
        sex = int(rng.integers(0, 2))
        u = rng.normal(0, 2)
        keep = waves if pid % 5 else [w for w in waves if rng.random() < 0.6] or [8]
        if pid % 37 == 0:
            keep = keep[:1]
        e = 0.0
        for w in keep:
            e = 0.6 * e + rng.normal(0, 1)
            rows.append({"pid": pid, "wave": w, "sex": sex, "bmi": 28 + u - 0.1 * w + 0.2 * sex + e,
                         "smk": int(rng.random() < 0.3 + 0.1 * sex)})
    return pd.DataFrame(rows)


@pytest.fixture
def panel_ds(tmp_path, panel_df):
    return make_ds(tmp_path, panel_df)


def BS(**kw):
    from statexplore.discovery.space import SearchSettings

    base = dict(outcome="bmi", candidates=["sex"], required=["wave"], family="gaussian", participant="pid",
                time="wave", workers=1, max_models=None, max_minutes=None, max_predictors=3)
    base.update(kw)
    return SearchSettings(**base)


@pytest.fixture(scope="module")
def ordinal_df():
    rng = np.random.default_rng(5)
    n = 500
    df = pd.DataFrame({"a": rng.integers(0, 2, n), "b": rng.integers(0, 2, n), "c": rng.integers(0, 2, n)})
    lat = 0.6 * df["a"] - 0.5 * df["b"] + rng.logistic(size=n)
    df["y"] = pd.cut(lat, [-np.inf, -1, 0, 1, np.inf], labels=False) + 1
    return df


@pytest.fixture
def ordinal_ds(tmp_path, ordinal_df):
    ds = make_ds(tmp_path, ordinal_df)
    ds.set_variable_field("y", "measure", "ordinal")
    return ds


def OS(**kw):
    from statexplore.discovery.space import SearchSettings

    base = dict(outcome="y", candidates=["a", "b", "c"], family="ordinal", strategy="combinations", include_main=True,
                max_order=3, workers=1, max_models=None, max_minutes=None, max_predictors=3)
    base.update(kw)
    return SearchSettings(**base)


# ---- P0-05 CSV whitespace and measurement levels (Fixture H) ---------------------------------------

FIXTURE_H = ("participant,wave,bmi,group,code,sci,blank,mixed\n"
             "001,8,30.2,A,1,1e3,,1\n"
             "001,9,   ,A,2, -2.5E-1 ,  ,2\n"
             "001,10,29.9,A,3,.5,,3\n"
             "002,8, 31.4 ,B,1,+4,,x\n"
             "002,9,,B,2,0,,5\n"
             "002,10,\t,B,3,-7,\t,6\n"
             "003,8,27.0,A,2,1.5,,7\n"
             "003,9, \t ,B,1,2,,8\n"
             "003,10,28.1,B,3,3,,9\n"
             "004,8,33.3,A,1,4,,10\n")


@pytest.fixture
def csv_h(tmp_path):
    p = tmp_path / "fixture_h.csv"   # "wave" keeps numeric/continuous (time name); "code" is a code set
    p.write_text(FIXTURE_H, encoding="utf-8")
    return p


def test_P0_05_whitespace_cells_are_missing_and_numeric_stays_numeric(csv_h):
    ds = import_file(csv_h)
    bmi = ds.var("bmi")
    assert bmi.storage is Storage.NUMERIC and bmi.measure is Measure.CONTINUOUS
    col = ds.analysis_column("bmi")
    assert col.isna().sum() == 4                      # "   ", "", tab, " \t " -> NA (never 0)
    assert not (col == 0).any()
    assert col.dropna().tolist() == [30.2, 29.9, 31.4, 27.0, 28.1, 33.3]
    sci = ds.analysis_column("sci")
    assert ds.var("sci").storage is Storage.NUMERIC
    assert sci.tolist() == [1000.0, -0.25, 0.5, 4.0, 0.0, -7.0, 1.5, 2.0, 3.0, 4.0]
    rep = ds.source.options
    assert rep["whitespace_cells"]["bmi"] == 3        # "" is already NA in the CSV reader; 3 whitespace-only cells
    assert "blank" in rep["empty_columns"]
    assert rep["text_in_numeric"]["mixed"][0] == 1 and rep["text_in_numeric"]["mixed"][1] == ["x"]
    text = " ".join(import_report(ds))
    assert "missing values (NA, not zero)" in text and "mixed is mostly numeric" in text
    raw = csv_h.read_text(encoding="utf-8")
    assert raw == FIXTURE_H                            # the source file is never modified


def test_P0_05_numeric_codes_are_not_continuous_and_ids_keep_zero_padding(csv_h):
    from statexplore.analysis.measurement import needs_review

    ds = import_file(csv_h)
    code = ds.var("code")
    assert code.storage is Storage.NUMERIC and code.measure is Measure.NOMINAL     # rule 6
    assert needs_review(code)                                                      # flagged in Measurement Review
    assert ds.var("group").storage is Storage.STRING
    assert ds.var("wave").measure is Measure.CONTINUOUS                            # time variable
    pid = ds.var("participant")
    assert pid.storage is Storage.STRING and ds.analysis_column("participant").iloc[0] == "001"


def test_P0_05_continuous_outcome_selectable_everywhere(csv_h, qtbot):
    """The BMI-like column is offered as a continuous dependent variable by the Discovery and Mixed pages."""
    from statexplore.core.project import Project
    from statexplore.gui.pages.discovery import DiscoveryPage
    from statexplore.gui.pages.mixed import MixedPage
    from statexplore.gui.state import AppState

    state = AppState()
    state.set_project(Project(import_file(csv_h)))
    for page in (DiscoveryPage(state), MixedPage(state)):
        qtbot.addWidget(page)
        items = [page.outcome.itemData(i) for i in range(page.outcome.count())]
        assert "bmi" in items


def test_P0_05_measurement_override_survives_and_propagates(csv_h, qtbot):
    from statexplore.analysis.measurement import approve, needs_review
    from statexplore.core.project import Project
    from statexplore.gui.pages.discovery import DiscoveryPage
    from statexplore.gui.state import AppState

    ds = import_file(csv_h)
    approve(ds, "code", "ordinal")
    assert ds.var("code").measure is Measure.ORDINAL and not needs_review(ds.var("code"))
    state = AppState()
    state.set_project(Project(ds))
    page = DiscoveryPage(state)
    qtbot.addWidget(page)
    label = page.outcome.itemText(page.outcome.findData("code"))
    assert "ordinal" in label


def test_P0_05_sav_metadata_unchanged(sav_labelled):
    ds = import_file(sav_labelled)
    assert ds.var("mood").measure is Measure.ORDINAL and ds.var("sex").measure is Measure.NOMINAL
    assert ds.var("score").measure is Measure.CONTINUOUS
    assert 9.0 in ds.var("mood").missing_values


def test_P0_05_hrs_csv_matches_sav(hrs_path):
    csv = hrs_path.with_suffix(".csv")
    if not csv.is_file():
        pytest.skip("HRS CSV twin not found")
    c, s = import_file(csv), import_file(hrs_path)
    assert [n for n in c.names() if c.var(n).storage is not s.var(n).storage] == []
    for n in ("R_PMBMI", "R_SHLT", "R_DEPRES", "WAVE"):
        x, y = c.analysis_column(n), s.analysis_column(n)
        assert (x.isna() == y.isna()).all() and np.allclose(x.dropna(), y.dropna())
    bmi = c.var("R_PMBMI")
    assert bmi.measure is Measure.CONTINUOUS and int(c.analysis_column("R_PMBMI").notna().sum()) == 3901


# ---- P0-03 headline by hypothesis family ----------------------------------------------------------

def _fam(t, tests, bh, holm=None):
    return {"type": t, "label": t, "tests": tests, "sig_raw": bh, "sig_adj": bh, "sig_bh": bh,
            "sig_holm": bh if holm is None else holm, "no_ci": 0, "expected_false_raw": tests * 0.05}


def test_P0_03_fixture_A_counts_never_conflated():
    from statexplore.discovery.engine import headline

    run = SimpleNamespace(settings=SimpleNamespace(alpha=0.05, correction="BH", participant=None, workflow="fixed"))
    fams = [_fam("interaction_term", 16, 0), _fam("joint_interaction", 5, 0), _fam("main_term", 9, 3),
            _fam("overall", 18, 14)]
    text = " ".join(headline(run, fams))
    assert "0 significant interactions" in text
    assert "3 of 9 main-effect term tests" in text and "14 of 18 whole-model tests" in text


def test_P0_03_no_interaction_hypotheses_and_structure_wording():
    from statexplore.discovery.engine import headline

    run = SimpleNamespace(settings=SimpleNamespace(alpha=0.05, correction="BH", participant=None, workflow="fixed"))
    text = " ".join(headline(run, [_fam("overall", 1, 1)]))
    assert "tested no interaction hypotheses" in text and "1 of 1 whole-model test " in text
    run.settings.participant, run.settings.workflow = "pid", "structure"
    text = " ".join(headline(run, [_fam("structure", 3, 2)]))
    assert "covariance-structure comparisons" in text and "interaction" not in text.split(".")[0]


@pytest.mark.r
def test_P0_03_ordinal_search_headline_matches_ranking(ordinal_ds):
    from statexplore.discovery.engine import _family_counts, create_run, execute, headline, tests_frame

    run = execute(create_run(ordinal_ds, OS()), ordinal_ds)
    tf = tests_frame(run, interpret=False)
    fams = {f["type"]: f for f in _family_counts(run)}
    for t, g in tf[tf["in_family"]].groupby("test_type"):
        assert fams[t]["tests"] == len(g) and fams[t]["sig_bh"] == int((g["p_bh"] < 0.05).sum())
    inter = sum(fams[t]["sig_bh"] for t in ("interaction_term", "joint_interaction") if t in fams)
    text = " ".join(headline(run))
    assert (f"{inter} significant interactions" in text) if inter == 0 else ("Interaction findings" in text)


# ---- P0-01 model identity -------------------------------------------------------------------------

def test_P0_01_verify_tolerances():
    from statexplore.analysis.descriptor import ModelDescriptor, verify

    d = ModelDescriptor(source="s", run_id="r", candidate=1, family="lmm", outcome_family="gaussian",
                        engine="nlme::lme", link="identity", outcome="y", terms=[["x"]], n=100, loglik=-50.0,
                        aic=110.0)
    assert verify(d, 100, -50.004, 110.005)[0] is True
    ok, text = verify(d, 99, -50.0, 110.0)
    assert ok is False and "FAILED" in text
    assert verify(d, 100, -51.0, 112.0)[0] is False
    assert verify(None, 1, 1, 1) == (None, "")


@pytest.mark.r
def test_P0_01_workflow_B_structures_survive_transfer(panel_ds, r_session):
    """Defect B/C: every structure candidate keeps its structure, engine and REML estimation, and the Mixed
    refit is verified against the search (n, logLik, AIC)."""
    from statexplore.analysis import mixed
    from statexplore.discovery.detail import candidate_from_row
    from statexplore.discovery.engine import create_run, execute
    from statexplore.discovery.rcode import candidate_args

    run = execute(create_run(panel_ds, BS(workflow="structure", structures=["nlme_ri", "ar1", "car1"])), panel_ds)
    assert {r["structure"] for r in run.rows} == {"nlme_ri", "ar1", "car1"}
    for row in run.rows:
        c = candidate_from_row(run, row)
        assert c.structure == row["structure"]
        assert f'struct = "{row["structure"]}"' in candidate_args(panel_ds, c, run.settings)
        spec = mixed.spec_from_discovery(run, row)
        assert spec.engine == "nlme" and spec.method == "REML"
        assert spec.residual == {"nlme_ri": "independent", "ar1": "ar1", "car1": "car1"}[row["structure"]]
        res, _ = mixed.run(r_session, panel_ds, spec)
        assert res.values["identity_check"] == "verified", res.messages[:2]


@pytest.mark.r
def test_P1_01_workflow_A_car1_opens_with_compatible_inference(panel_ds, r_session):
    from statexplore.analysis import mixed
    from statexplore.discovery.engine import create_run, execute

    run = execute(create_run(panel_ds, BS(structure="car1", strategy="main")), panel_ds)
    row = run.rows[0]
    spec = mixed.spec_from_discovery(run, row)
    assert spec.engine == "nlme" and spec.method == "ML" and spec.inference == "satterthwaite"
    mixed.validate(panel_ds, spec)                 # v1.2 raised the lme4-only inference error here
    res, _ = mixed.run(r_session, panel_ds, spec)
    assert res.values["identity_check"] == "verified"
    bad = mixed.MixedSpec(**{**spec.to_dict(), "inference": "lrt"})
    with pytest.raises(mixed.MixedInputError, match="implementation-specific"):
        mixed.validate(panel_ds, bad)


@pytest.mark.r
def test_P0_01_gee_keeps_working_correlation_and_reports_qic(panel_ds, r_session):
    from statexplore.analysis import mixed
    from statexplore.discovery.engine import create_run, execute

    panel_ds.set_variable_field("smk", "measure", "nominal")
    run = execute(create_run(panel_ds, BS(outcome="smk", family="binomial", workflow="structure",
                                          structures=["gee_ind", "gee_exch", "gee_ar1"])), panel_ds)
    for row in run.rows:
        assert row["qic"] is not None and np.isfinite(row["qic"]), row["flags"]       # P1-07
        spec = mixed.spec_from_discovery(run, row)
        assert spec.engine == "geepack"
        assert spec.gee_corstr == {"gee_ind": "independence", "gee_exch": "exchangeable", "gee_ar1": "ar1"}[row["structure"]]


def test_P0_01_ordinal_comparison_uses_clm_not_glm(ordinal_ds):
    from statexplore.analysis.comparison import build
    from statexplore.analysis.models import ModelSpec

    full = ModelSpec("y", [("a", "b")], "ordinal")
    mains = full.copy_edited([("a",), ("b",)])
    code = build(ordinal_ds, [full, mains], True)
    assert "ordinal::clm(" in code and "stats::glm(" not in code and "sx_event" not in code
    multi = ModelSpec("y", [("a",)], "multinomial", reference=4.0)
    code = build(ordinal_ds, [multi, multi.copy_edited([("a",), ("b",)])], True)
    assert "nnet::multinom(" in code and "ref = " in code


def test_P0_01_binary_event_is_preserved(tmp_path):
    from statexplore.analysis.comparison import ComparisonError, build
    from statexplore.analysis.models import ModelSpec

    rng = np.random.default_rng(1)
    ds = make_ds(tmp_path, pd.DataFrame({"y": rng.integers(0, 2, 80), "x": rng.normal(size=80),
                                         "z": rng.normal(size=80)}))
    a = ModelSpec("y", [("x",)], "binomial", event=0.0)
    code = build(ds, [a, a.copy_edited([("x",), ("z",)])], True)
    assert '== "0")' in code and "sx_lv[2]" not in code          # event category of the source model
    with pytest.raises(ComparisonError, match="code the outcome differently"):
        build(ds, [a, ModelSpec("y", [("x",), ("z",)], "binomial", event=1.0)], True)


def test_P0_01_teaching_refuses_non_linear_families(ordinal_ds):
    from statexplore.analysis.models import ModelCheckError, ModelSpec
    from statexplore.analysis.teaching import PlotSpec, build

    with pytest.raises(ModelCheckError, match="cannot refit faithfully"):
        build(ordinal_ds, ModelSpec("y", [("a",)], "ordinal"), PlotSpec(x="a"), "p.png")


@pytest.mark.r
def test_P0_01_ordinal_candidate_through_model_comparison(ordinal_ds, r_session):
    """Defect A: the transferred comparison reproduces the search's joint interaction LR exactly."""
    from statexplore.analysis.comparison import compare
    from statexplore.analysis.models import from_candidate
    from statexplore.discovery.engine import create_run, execute

    run = execute(create_run(ordinal_ds, OS()), ordinal_ds)
    row = next(r for r in run.rows if any(len(t) == 3 for t in r.get("inter") or []))
    m = from_candidate(run, row)
    assert m.family == "ordinal" and m.descriptor["engine"] == "ordinal::clm"
    res, _ = compare(r_session, ordinal_ds, [m, m.copy_edited([t for t in m.terms if len(t) == 1])])
    nest = next(t for t in res.tables if t.name == "Nested model tests")
    joint = next(t for t in row["tests"] if t["type"] == "joint_interaction")
    assert nest.rows[0][3] == pytest.approx(joint["statistic"], abs=1e-6)
    assert nest.rows[0][2] == joint["df1"]
    assert any("identity verified" in m_ for m_ in res.messages)


# ---- P0-02 export reproducibility -----------------------------------------------------------------

def test_P0_02_portable_script_pruning_and_wrapping():
    from statexplore.export.exporters import portable_text, wrap_r_line

    script = "\n".join([
        "# ---- StatExplore: data preparation ----", 'sx_data_file <- "C:/Users/me/AppData/Local/Temp/x/data.csv"',
        "sx_raw <- utils::read.csv(", "  sx_data_file, na.strings = \"\",", "  colClasses = c(",
        '    v001 = "numeric",', '    v002 = "character",', '    v003 = "numeric"', "  )", ")",
        "sx_names <- c(", '    v001 = "y",', '    v002 = "id",', '    v003 = "x"', ")", "dat <- sx_raw",
        "dat$v002 <- factor(dat$v002, levels = c(" + ", ".join(f'"h{i}"' for i in range(500)) + "))",
        "sx_result <- list(n_rows = nrow(dat), n_cols = ncol(dat))", "fit <- lm(v001 ~ v003, data = dat)"])
    out, used = portable_text(script, "analysis_data.csv")
    assert used == {"v001", "v003"}
    assert "v002" not in out and "AppData" not in out and 'sx_data_file <- "analysis_data.csv"' in out
    assert '    v001 = "numeric",\n    v003 = "numeric"\n  )' in out
    long = 'f(' + ", ".join(f'"a,{i}"' for i in range(60)) + ")"
    parts = wrap_r_line(long)
    assert len(parts) > 1 and all(len(p) <= 110 for p in parts) and "".join(p.strip() for p in parts).replace(
        ",\"", ", \"") == long


@pytest.mark.r
def test_P0_02_structure_export_reproduces_model_and_comparisons(panel_ds, r_session, tmp_path):
    """The exported workflow-B candidate (struct kept, comparison code included) sourced in a fresh R session from
    another folder reproduces the engine's n, logLik, AIC, likelihood parameters and covariance LR test."""
    from statexplore.discovery.detail import candidate_from_row
    from statexplore.discovery.engine import create_run, execute, structure_tests
    from statexplore.discovery.rcode import candidate_script
    from statexplore.discovery.space import view
    from statexplore.export.exporters import write_portable
    from statexplore.rbridge.locate import find_rscript, r_library

    run = execute(create_run(panel_ds, BS(workflow="structure", structures=["nlme_ri", "car1"])), panel_ds)
    row = next(r for r in run.rows if r["structure"] == "car1")
    comps = structure_tests(run)[int(row["idx"])]
    v = view(panel_ds, run.settings)
    r_session.ensure_data(v)
    script = candidate_script(r_session.load_script, v, run.settings, run.family, candidate_from_row(run, row), comps)
    assert 'struct = "car1"' in script and "sx_comparisons" in script
    folder = tmp_path / "export"
    write_portable(script, folder, panel_ds)
    code = (f'source("{(folder / "analysis.R").as_posix()}"); cat("@@", sprintf("%.12g", c(sx_result$n, sx_result$loglik, '
            'sx_result$aic, sx_result$n_lik_params, sx_comparisons$LR[1], sx_comparisons$p[1])), "\\n")')
    p = subprocess.run([str(find_rscript()), "--vanilla", "-e", code], cwd=tmp_path, capture_output=True, text=True,
                       env=dict(os.environ, R_LIBS=str(r_library())), timeout=600)
    line = next(ln for ln in p.stdout.splitlines() if ln.startswith("@@")).split()[1:]
    n, ll, aic, k, lr, pv = map(float, line)
    assert n == row["n"] and ll == pytest.approx(row["loglik"], abs=1e-6) and aic == pytest.approx(row["aic"], abs=1e-6)
    assert k == row["n_params"] + row["n_re_params"] == 6
    assert lr == pytest.approx(comps[0]["statistic"], abs=1e-6) and pv == pytest.approx(comps[0]["p"], rel=1e-6)
    header = (folder / "analysis_data.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    assert len(header) == 4                                  # outcome, sex, participant, wave only


# ---- P1 --------------------------------------------------------------------------------------------

def test_P1_02_03_preflight_reports_observed_waves_and_gap_compression(panel_ds):
    from statexplore.analysis.longitudinal_preflight import preflight

    rep = preflight(panel_ds, "bmi", "pid", "wave", ["sex"])
    assert rep.observed_times == [8, 9, 10, 11, 12, 13, 14, 16]
    assert rep.interior_absent == [] or rep.interior_absent == [15]
    assert rep.n_usable == rep.n_usable_multi + rep.n_usable_one and rep.n_usable_one >= 1
    assert rep.n_intermittent > 0
    assert any("MAR itself cannot be verified" in n for n in rep.notes)


def test_P1_02_globally_absent_wave_warns(tmp_path):
    from statexplore.analysis.longitudinal_preflight import preflight

    df = pd.DataFrame({"pid": np.repeat(np.arange(20), 4), "wave": np.tile([14, 15, 16, 17], 20)})
    df["y"] = np.where(df["wave"] == 15, np.nan, np.random.default_rng(1).normal(size=len(df)))
    ds = make_ds(tmp_path, df)
    rep = preflight(ds, "y", "pid", "wave")
    assert rep.interior_absent == [15] and rep.observed_times == [14, 16, 17]
    assert any("gap compression" in w for w in rep.warnings)


def test_P1_04_parameter_columns_and_estimation_labels():
    from statexplore.discovery.engine import estimation_label

    mk = lambda fam, wf, part="pid": SimpleNamespace(family=fam, settings=SimpleNamespace(  # noqa: E731
        participant=part, workflow=wf))
    assert estimation_label(mk("lmm", "structure")) == "REML"
    assert estimation_label(mk("lmm", "fixed")) == "ML"
    assert estimation_label(mk("gaussian", "fixed", None)) == "OLS"
    assert estimation_label(mk("gee", "structure")).startswith("GEE")


@pytest.mark.r
def test_P1_04_likelihood_parameter_count(panel_ds):
    from statexplore.discovery.engine import add_display_columns, create_run, execute, results_frame

    run = execute(create_run(panel_ds, BS(workflow="structure", structures=["car1"])), panel_ds)
    df = add_display_columns(results_frame(run), run)
    assert df["n_lik"].iloc[0] == 6 and df["estimation"].iloc[0] == "REML"     # 3 fixed + RI + residual + phi


def test_P1_06_timeouts_are_not_complete(ordinal_ds, monkeypatch):
    from statexplore.discovery import engine
    from statexplore.rbridge.worker import RTimeoutError

    class StubWorker:                      # an R worker whose every request exceeds its time limit
        def __init__(self, *a, **k):
            self.alive, self.generation = True, 0

        def start(self):
            pass

        def stop(self):
            pass

        def run(self, *a, **k):
            raise RTimeoutError("R did not respond within 1s")

        def request(self, cmd, *a, **k):
            return SimpleNamespace(result={"packages": {}, "r_version": "stub"})

    monkeypatch.setattr(engine, "RWorker", StubWorker)
    monkeypatch.setattr(engine._PooledSession, "ensure", lambda self, ds: None)
    run = engine.execute(engine.create_run(ordinal_ds, OS(strategy="main", include_main=False)), ordinal_ds)
    assert run.status == "incomplete" and "timed out" in run.stop_reason
    assert run.rows and all(r["timed_out"] and r["status"] == "failed" for r in run.rows)


def test_P1_05_interpretation_is_conditional_for_interactions():
    from statexplore.analysis.interpret import brief
    from statexplore.core.schema import AnalysisResult, EngineInfo, Table

    r = AnalysisResult(kind="lm", title="t", engine=EngineInfo("R", "4"), script="", inputs={})
    r.tables.append(Table("Coefficients", ["term", "B", "p"], [["(Intercept)", 1.0, 0.1], ["a", 0.5, 0.01],
                                                                  ["a:b", 0.2, 0.3]]))
    text = brief(r)
    assert "Strongest" not in text and "smallest p-value" in text and "conditional effects" in text


# ---- P2 --------------------------------------------------------------------------------------------

def test_P2_01_searchable_dropdown(qtbot):
    from PySide6.QtWidgets import QComboBox

    from statexplore.gui.widgets.searchable import make_searchable

    c = make_searchable(QComboBox())
    qtbot.addWidget(c)
    for name, data in (("R_PMBMI  ·  continuous", "bmi"), ("R_SHLT  ·  ordinal", "shlt")):
        c.addItem(name, data)
    c.completer().setCompletionPrefix("shlt")
    assert c.completer().completionCount() == 1
    c.lineEdit().setText("nonsense")
    c.lineEdit().editingFinished.emit()
    assert c.currentData() == "bmi" and c.lineEdit().text() == "R_PMBMI  ·  continuous"


def test_P2_02_find_bar(qtbot):
    from PySide6.QtWidgets import QPlainTextEdit, QWidget

    from statexplore.gui.widgets.find_bar import FindBar

    host = QWidget()
    qtbot.addWidget(host)
    ed = QPlainTextEdit("alpha\nbeta\ncorCAR1(form = ~ sx_time | sx_gid)\nbeta", host)
    bar = FindBar(ed, host)
    bar.open()
    bar.edit.setText("corCAR1")
    assert ed.textCursor().selectedText() == "corCAR1"
    bar.edit.setText("beta")
    bar.find_next()
    bar.find_next()          # wraps around
    assert ed.textCursor().selectedText() == "beta"
    bar.edit.setText("zzz")
    assert bar.status.text() == "not found"


def test_P2_05_budget_history_is_shown():
    from statexplore.discovery.engine import DiscoveryRun
    from statexplore.discovery.space import SearchSettings

    run = DiscoveryRun(settings=SearchSettings(outcome="y", candidates=["a"], max_models=20), family="gaussian",
                       total=112, signature="x")
    run.executions = [{"start": 0, "end": 10, "max_models": 10, "max_minutes": None},
                      {"start": 10, "end": 30, "max_models": 20, "max_minutes": None}]
    text = run.budget_text()
    assert "run 1: 10 models" in text and "run 2: 20 models" in text
    meta = run.meta_dict()
    assert DiscoveryRun.from_meta(meta, []).executions == run.executions


# ---- HRS reference fixtures (A, C/D, E) -----------------------------------------------------------

@pytest.mark.r
def test_fixture_D_E_hrs_reference_values(hrs_path, r_session):
    from statexplore.analysis import mixed
    from statexplore.discovery.engine import create_run, execute, tests_frame

    ds = import_file(hrs_path)
    ds.set_variable_field("WAVE", "measure", "continuous")
    base = dict(outcome="R_PMBMI", candidates=["RAGENDER"], required=["WAVE"], participant="HHIDPN", time="WAVE")
    run = execute(create_run(ds, BS(**base, workflow="structure", structures=["nlme_ri", "ar1", "car1"])), ds)
    by = {r["structure"]: r for r in run.rows}
    assert by["car1"]["n"] == 3901 and by["car1"]["n_groups"] == 983
    assert by["car1"]["aic"] == pytest.approx(20540.106, abs=0.01)
    assert by["car1"]["bic"] == pytest.approx(20577.720, abs=0.01)
    assert by["car1"]["loglik"] == pytest.approx(-10264.053, abs=0.01)
    assert by["ar1"]["aic"] == pytest.approx(20512.788, abs=0.01)
    assert by["nlme_ri"]["aic"] == pytest.approx(20640.34, abs=0.01)
    spec = mixed.spec_from_discovery(run, by["car1"])
    res, _ = mixed.run(r_session, ds, spec)
    assert spec.engine == "nlme" and spec.method == "REML" and res.values["identity_check"] == "verified"
    obs = res.values["observed_time_points"]
    assert "Wave 15" not in obs.split("(")[0] and "absent for everyone inside the range: Wave 15" in obs
    run_e = execute(create_run(ds, BS(**base, structure="car1", strategy="main")), ds)
    tf = tests_frame(run_e, interpret=False)
    assert tf["p"].iloc[0] == pytest.approx(0.7863, abs=1e-4)
    assert run_e.rows[0]["aic"] == pytest.approx(20532.121, abs=0.01)
    assert run_e.rows[0]["bic"] == pytest.approx(20569.735, abs=0.01)


@pytest.mark.r
def test_fixture_A_hrs_ordinal_families(hrs_path):
    from statexplore.core.derive import FilterSpec
    from statexplore.discovery.engine import _family_counts, create_run, execute, headline

    ds = import_file(hrs_path)
    ds.add_filter(FilterSpec("WAVE", "==", 16))
    ds.set_variable_field("R_SHLT", "measure", "ordinal")
    run = execute(create_run(ds, OS(outcome="R_SHLT", candidates=["RAGENDER", "R_DEPRES", "R_SMOKEV"])), ds)
    assert len(run.rows) == 18 and run.n_sample == 917
    fams = {f["type"]: (f["tests"], f["sig_bh"]) for f in _family_counts(run)}
    assert fams == {"interaction_term": (16, 0), "joint_interaction": (5, 0), "main_term": (9, 3), "overall": (18, 14)}
    text = " ".join(headline(run))
    assert "0 significant interactions" in text and "3 of 9" in text and "14 of 18" in text
    three = next(r for r in run.rows if any(len(t) == 3 for t in r.get("inter") or []))
    t3 = next(t for t in three["tests"] if t["type"] == "interaction_term")
    assert t3["estimate"] == pytest.approx(1.054, abs=1e-3) and t3["p"] == pytest.approx(0.3003, abs=1e-3)
    assert np.exp(t3["estimate"]) == pytest.approx(2.868, abs=1e-3)
