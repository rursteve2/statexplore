"""v1.1 Phase 1 acceptance: measurement levels, suggestions, analysis-specific treatments
(spec §20 A–D), using the HRS file as the integration fixture."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statexplore.analysis import measurement as M
from statexplore.analysis.builders import BY_KEY
from statexplore.analysis.framework import AnalysisInputError, run_analysis
from statexplore.core.dataset import DatasetError
from statexplore.core.importers import import_file, import_sav
from statexplore.core.project import Project
from statexplore.core.variables import Measure
from statexplore.rbridge.codegen import data_load_script


@pytest.fixture(scope="module")
def hrs():
    from tests.conftest import HRS_PATH

    if not HRS_PATH.is_file():
        pytest.skip("HRS test dataset not found")
    return import_sav(HRS_PATH)


def fresh(hrs_path):
    return import_sav(hrs_path)


def cont(name, **kw):
    return M.VariableTreatment(name, "continuous", acknowledged=True, **kw)


def coefs(res) -> dict[str, float]:
    t = next(t for t in res.tables if t.name.startswith("Coefficients"))
    i, j = t.columns.index("term"), t.columns.index("estimate")
    return {r[i]: r[j] for r in t.rows}


def ols(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    X = np.column_stack([np.ones(len(y)), X])
    return np.linalg.lstsq(X, y, rcond=None)[0]


# ---- A. variable measurement ---------------------------------------------------------------

def test_A_source_preserved_and_suggestion_not_applied(hrs):
    v = hrs.var("R_LTACTX")
    assert v.source_measure == "nominal" and v.measure is Measure.NOMINAL
    assert v.suggested_measure == "ordinal" and M.needs_review(v)
    assert "frequency" in " ".join(v.suggestion_reasons).lower()
    assert v.coding_direction == "higher codes = less frequent"
    assert v.measure_status == "source"


def test_A_suggestion_rules_on_hrs(hrs):
    s = {n: hrs.var(n).suggested_measure for n in hrs.variables}
    assert s["R_SHLT"] == "ordinal" and hrs.var("R_SHLT").coding_direction == "higher codes = worse rating"
    assert s["R_EYERT"] == "ordinal" and hrs.var("R_EYERT").coding_direction == "higher codes = better rating"
    assert s["R_LBSATHOME"] == "ordinal" and s["R_RETSAT"] == "ordinal" and s["RAEDUC"] == "ordinal"
    assert s["R_COMP"] == "ordinal" and "7 = 7.Don't do" in " ".join(hrs.var("R_COMP").suggestion_reasons)
    assert s["RAEDYRS"] == "continuous"                   # 16 values, only anchors labelled
    assert s["R_MSTAT"] == "nominal" and s["RARACEM"] == "nominal" and s["R_CANCR"] == "nominal"
    assert s["HHIDPN"] == "nominal" and s["HHID"] == "nominal"   # identifiers are not quantities
    assert s["R_BMI"] == "continuous" and s["RAGENDER"] == "binary"
    # no claim on unlabelled 4-value ordinals (five categories alone do not make a scale)
    assert s["ATTENMO"] == ""
    assert not M.needs_review(hrs.var("RAGENDER"))        # nominal vs binary is not a conflict


def test_A_approval_persists_and_is_undoable(hrs_path, tmp_path):
    ds = fresh(hrs_path)
    M.approve(ds, "R_LTACTX", "ordinal")
    v = ds.var("R_LTACTX")
    assert v.measure is Measure.ORDINAL and v.measure_status == "approved" and v.source_measure == "nominal"
    p = tmp_path / "p.sxp"
    Project(ds).save(p)
    v2 = Project.open(p).dataset.var("R_LTACTX")
    assert (v2.measure, v2.measure_status, v2.source_measure, v2.suggested_measure) == \
        (Measure.ORDINAL, "approved", "nominal", "ordinal")
    ds.undo(); ds.undo()
    assert ds.var("R_LTACTX").measure is Measure.NOMINAL and ds.var("R_LTACTX").measure_status == "source"


def test_A_category_order(hrs_path):
    ds = fresh(hrs_path)
    with pytest.raises(DatasetError):
        M.approve(ds, "R_SHLT", "ordinal", category_order=[1.0, 2.0, 3.0])
    M.approve(ds, "R_SHLT", "ordinal", category_order=[5.0, 4.0, 3.0, 2.0, 1.0])
    assert ds.levels("R_SHLT") == [5.0, 4.0, 3.0, 2.0, 1.0]
    script = data_load_script(ds, "x.csv")
    a = ds.var("R_SHLT").alias
    assert f'dat${a} <- factor(dat${a}, levels = c(5.0, 4.0, 3.0, 2.0, 1.0)' in script
    assert ds.raw["R_SHLT"].equals(fresh(hrs_path).raw["R_SHLT"])     # codes unchanged


def test_A_invalid_string_to_continuous_rejected(tmp_path):
    p = tmp_path / "s.csv"
    pd.DataFrame({"g": ["a", "b", "c", "a"], "y": [1, 2, 3, 4]}).to_csv(p, index=False)
    ds = import_file(p)
    with pytest.raises(DatasetError):
        ds.set_variable_field("g", "measure", "continuous")
    with pytest.raises(M.TreatmentError, match="text"):
        M.apply_treatments(ds, [cont("g")])
    with pytest.raises(DatasetError):
        M.approve(ds, "g", "binary")


def test_A_treatment_view_leaves_dataset_untouched(hrs):
    before = hrs.var("R_LTACTX").to_dict()
    n_changes = len(hrs.changes)
    view = M.apply_treatments(hrs, [cont("R_LTACTX", numeric_confirmed=True)])
    assert view.var("R_LTACTX").measure is Measure.CONTINUOUS
    assert hrs.var("R_LTACTX").to_dict() == before and len(hrs.changes) == n_changes
    assert view.raw is hrs.raw
    assert M.apply_treatments(hrs, []) is hrs


def test_A_acknowledgement_and_nominal_confirmation(hrs):
    with pytest.raises(M.TreatmentError, match="acknowledge"):
        M.apply_treatments(hrs, [M.VariableTreatment("R_LTACTX", "continuous")])
    with pytest.raises(M.TreatmentError, match="nominal"):        # project level is still nominal
        M.apply_treatments(hrs, [M.VariableTreatment("R_LTACTX", "continuous", acknowledged=True)])
    view = M.apply_treatments(hrs, [cont("R_LTACTX", numeric_confirmed=True)])
    assert view.var("R_LTACTX").measure is Measure.CONTINUOUS


# ---- B / C. HRS R_LTACTX and R_SHLT in real analyses -------------------------------------------

@pytest.mark.r
def test_B_ltactx_continuous_regression_uses_codes(hrs_path, r_session):
    ds = fresh(hrs_path)
    M.approve(ds, "R_LTACTX", "ordinal")              # approved ordinal -> no nominal confirmation needed
    vars_ = {"dependent": ["R_LTACTX"], "predictors": ["R_BMI"]}
    with pytest.raises(AnalysisInputError, match="continuous"):          # the checkbox is required
        run_analysis(r_session, ds, BY_KEY["lm"], vars_, {})
    with pytest.raises(AnalysisInputError, match="acknowledge"):
        run_analysis(r_session, ds, BY_KEY["lm"], vars_, {},
                     treatments=[M.VariableTreatment("R_LTACTX", "continuous")])
    res, _ = run_analysis(r_session, ds, BY_KEY["lm"], vars_, {}, treatments=[cont("R_LTACTX")])
    f = ds.analysis_frame(["R_LTACTX", "R_BMI"]).dropna()
    assert set(f["R_LTACTX"].unique()) == {1.0, 2.0, 3.0, 4.0, 5.0}
    b = ols(f["R_LTACTX"].to_numpy(), f[["R_BMI"]].to_numpy())
    est = coefs(res)
    assert est["(Intercept)"] == pytest.approx(b[0], rel=1e-8) and est["R_BMI"] == pytest.approx(b[1], rel=1e-8)
    assert res.n_used == len(f)
    rec = res.inputs["treatments"][0]
    assert rec["variable"] == "R_LTACTX" and rec["treatment"] == "continuous" and rec["acknowledged"]
    assert "1 = 1.Every day" in rec["description"] and "less frequent" in rec["description"]
    assert M.CONTINUOUS_WARNING in res.inputs["treatment_warning"]
    assert "kept as its numeric codes" in res.script
    a = ds.var("R_LTACTX").alias
    assert f"dat${a} <- factor(dat${a}" not in res.script.split("# ---- StatExplore: analysis ----")[0]
    # project + source metadata unchanged by the analysis
    v = ds.var("R_LTACTX")
    assert v.measure is Measure.ORDINAL and v.source_measure == "nominal"


@pytest.mark.r
def test_B_ltactx_ordinal_keeps_order(hrs_path, r_session):
    ds = fresh(hrs_path)
    M.approve(ds, "R_LTACTX", "ordinal")
    res, _ = run_analysis(r_session, ds, BY_KEY["anova_oneway"], {"dependent": ["R_BMI"], "factors": ["R_LTACTX"]}, {})
    assert res.status.value in ("ok", "warning")
    a = ds.var("R_LTACTX").alias
    assert (f'dat${a} <- factor(dat${a}, levels = c(1.0, 2.0, 3.0, 4.0, 5.0), labels = c("1.Every day", '
            f'"2.>1 per week", "3.1 per week", "4.l-3 per mon", "5.Never"))') in res.script


@pytest.mark.r
def test_C_shlt_continuous_after_ack_and_direction(hrs_path, r_session):
    ds = fresh(hrs_path)
    v = ds.var("R_SHLT")
    labels = dict(v.value_labels)
    with pytest.raises(AnalysisInputError, match="nominal"):
        run_analysis(r_session, ds, BY_KEY["lm"], {"dependent": ["R_BMI"], "predictors": ["R_SHLT"]}, {},
                     treatments=[M.VariableTreatment("R_SHLT", "continuous", acknowledged=True)])
    res, _ = run_analysis(r_session, ds, BY_KEY["lm"], {"dependent": ["R_BMI"], "predictors": ["R_SHLT"]}, {},
                          treatments=[cont("R_SHLT", numeric_confirmed=True)])
    f = ds.analysis_frame(["R_BMI", "R_SHLT"]).dropna()
    b = ols(f["R_BMI"].to_numpy(), f[["R_SHLT"]].to_numpy())
    slope = coefs(res)["R_SHLT"]
    assert slope == pytest.approx(b[1], rel=1e-8) and slope > 0    # higher code = worse health = higher BMI
    assert ds.var("R_SHLT").value_labels == labels and ds.var("R_SHLT").measure is Measure.NOMINAL


# ---- D. R code generation and session caching ---------------------------------------------------

@pytest.mark.r
def test_D_switching_treatments_rebuilds_encoding(hrs_path, r_session, r_worker):
    ds = fresh(hrs_path)
    M.approve(ds, "R_SHLT", "ordinal")
    a = ds.var("R_SHLT").alias
    lm = lambda tr: run_analysis(r_session, ds, BY_KEY["lm"],  # noqa: E731
                                 {"dependent": ["R_BMI"], "predictors": ["R_SHLT"]}, {}, treatments=tr)
    r1, _ = lm([cont("R_SHLT")])
    assert r_worker.run(f"sx_result <- class(dat${a})").result == "numeric"
    r2, _ = lm([])            # project level: ordinal factor -> 4 dummy coefficients
    assert r_worker.run(f"sx_result <- class(dat${a})").result == "factor"
    assert len(coefs(r2)) == 5
    r3, _ = lm([cont("R_SHLT")])
    assert coefs(r1) == coefs(r3) and len(coefs(r1)) == 2
    assert r_worker.run(f"sx_result <- class(dat${a})").result == "numeric"


@pytest.mark.r
def test_D_recorded_script_reproduces(hrs_path, r_session, r_worker):
    ds = fresh(hrs_path)
    res, _ = run_analysis(r_session, ds, BY_KEY["lm"], {"dependent": ["R_BMI"], "predictors": ["R_SHLT"]}, {},
                          treatments=[cont("R_SHLT", numeric_confirmed=True)])
    i = next(i for i, t in enumerate(res.tables) if t.name.startswith("Coefficients"))
    # Re-run the exact recorded script in a fresh local environment and compare.
    out = r_worker.run("sx_result <- local({\n" + res.script
                       + f"\nsx_result$tables[[{i + 1}]]$data$estimate\n}})").result
    assert [float(x) for x in out] == pytest.approx(list(coefs(res).values()), rel=1e-12)
