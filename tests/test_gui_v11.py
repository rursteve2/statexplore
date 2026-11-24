"""v1.1 GUI tests (offscreen): measurement review, treatment checkboxes, selectors, new pages."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402

from statexplore.gui.main_window import MainWindow  # noqa: E402
from statexplore.gui.state import AppState  # noqa: E402


@pytest.fixture
def window(qtbot, tmp_path):
    from PySide6.QtCore import QSettings

    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(tmp_path / "settings"))
    w = MainWindow(AppState())
    yield w
    w.state.dirty = False
    w.state.shutdown_r()
    w.close()


@pytest.fixture
def hrs_window(window, qtbot, hrs_path):
    window.import_path(str(hrs_path))
    qtbot.waitUntil(lambda: window.state.project is not None, timeout=30000)
    return window


def select_analysis(page, key):
    for i in range(page.tree.topLevelItemCount()):
        fam = page.tree.topLevelItem(i)
        for j in range(fam.childCount()):
            if fam.child(j).data(0, Qt.UserRole) == key:
                page.tree.setCurrentItem(fam.child(j))
                return
    raise AssertionError(key)


def test_measurement_review_accept_and_keep(hrs_window, monkeypatch):
    ds = hrs_window.state.dataset
    review = hrs_window.pages["Dataset Explorer"].review
    names = review._names
    assert "R_LTACTX" in names and "R_SHLT" in names and "RAGENDER" not in names
    row = names.index("R_LTACTX")
    from statexplore.gui.pages.measurement_review import COLUMNS

    col = COLUMNS.index
    assert review.table.item(row, col("Source (SPSS) level")).text() == "nominal"
    assert review.table.item(row, col("Suggested level")).text() == "ordinal"
    review.select("R_LTACTX")
    assert "Every day" in review.detail.toPlainText() and "less frequent" in review.detail.toPlainText()
    monkeypatch.setattr(review, "_confirm", lambda text: False)        # declined -> nothing changes
    review._accept()
    assert ds.var("R_LTACTX").measure.value == "nominal"
    monkeypatch.setattr(review, "_confirm", lambda text: True)
    review._accept()
    v = ds.var("R_LTACTX")
    assert v.measure.value == "ordinal" and v.measure_status == "approved" and v.source_measure == "nominal"
    review.select("R_SHLT")
    review._keep()
    assert ds.var("R_SHLT").measure_status == "kept" and ds.var("R_SHLT").measure.value == "nominal"
    assert hrs_window.state.dirty


@pytest.mark.r
def test_analysis_builder_treat_as_continuous(hrs_window, qtbot, r_worker):
    page = hrs_window.pages["Analysis Builder"]
    select_analysis(page, "lm")
    dv = page.slot_widgets["dependent"].combo
    idx = dv.findData("R_LTACTX")
    assert idx >= 0 and "categorical codes" in dv.itemText(idx)
    dv.setCurrentIndex(idx)
    preds = page.slot_widgets["predictors"].list
    for i in range(preds.count()):
        if preds.item(i).data(Qt.UserRole) == "R_BMI":
            preds.item(i).setCheckState(Qt.Checked)
    panel = page.treatment_panel
    assert panel.is_offered("R_LTACTX") and not panel.is_offered("R_BMI")       # v1.2: R_BMI shown, no override
    assert panel.status("R_BMI")[0] == "continuous" and "R_BMI" in panel.model_variables()
    assert panel.outcome_check.text() == "Treat outcome R_LTACTX as continuous for this analysis"
    panel.ask = False
    panel.set_treated("R_LTACTX")
    assert [t.variable for t in panel.treatments()] == ["R_LTACTX"]
    page._run()
    qtbot.waitUntil(lambda: page.current is not None, timeout=120000)
    result, _ = page.current
    assert result.status.value in ("ok", "warning"), result.messages
    assert result.inputs["treatments"][0]["variable"] == "R_LTACTX"
    assert "Variable treatment for this analysis only" in page.output.toHtml()
    assert hrs_window.state.dataset.var("R_LTACTX").measure.value == "nominal"


@pytest.mark.r
def test_eda_page_categorical_workflow(hrs_window, qtbot, r_worker):
    page = hrs_window.pages["Exploratory Data Analysis"]
    page.selector.set_selected(["R_LTACTX", "RAGENDER"])
    assert "Cross-tabulation" in page.suggestion.text()
    n_before = len(hrs_window.state.project.results)
    page._crosstab()
    qtbot.waitUntil(lambda: len(hrs_window.state.project.results) == n_before + 1, timeout=60000)
    assert "Chi-square test of independence" in page.result_view.toHtml()
    page._frequencies()
    qtbot.waitUntil(lambda: len(hrs_window.state.project.results) == n_before + 2, timeout=60000)
    assert hrs_window.state.project.results[-1].kind == "frequencies"
    page.kind.setCurrentIndex(page.kind.findData("bar_stacked100"))
    page._plot()       # two categorical variables checked: the second becomes the group
    qtbot.waitUntil(lambda: len(hrs_window.state.project.results) == n_before + 3, timeout=60000)
    r = hrs_window.state.project.results[-1]
    assert r.kind == "eda_plot" and r.inputs["group"]["name"] == "RAGENDER" and r.plots
    # treat-as-continuous inside EDA: describe R_LTACTX with psych
    page.selector.set_selected(["R_LTACTX"])
    assert page.treatment_panel.is_offered("R_LTACTX")
    page.treatment_panel.ask = False
    page.treatment_panel.set_treated("R_LTACTX")
    page._describe()
    qtbot.waitUntil(lambda: len(hrs_window.state.project.results) == n_before + 4, timeout=60000)
    r = hrs_window.state.project.results[-1]
    assert r.kind == "describe" and r.inputs["treatments"][0]["variable"] == "R_LTACTX"
    assert r.tables[0].rows[0][1] == pytest.approx(hrs_window.state.dataset.analysis_column("R_LTACTX").mean())


@pytest.mark.r
def test_analysis_builder_2x2_preview_and_run(hrs_window, qtbot, r_worker):
    page = hrs_window.pages["Analysis Builder"]
    select_analysis(page, "assoc_2x2")
    assert "Choose both variables" in page.preview.text()
    out = page.slot_widgets["outcome"].combo
    out.setCurrentIndex(out.findData("R_SMOKEN"))
    grp = page.slot_widgets["group"].combo
    grp.setCurrentIndex(grp.findData("RAGENDER"))
    ev = page.option_widgets["event"]
    assert [ev.itemText(i) for i in range(ev.count())] == ["— choose —", "0 = 0.No", "1 = 1.Yes"]
    for key, code in (("event", 1.0), ("outcome_ref", 0.0), ("exposed", 1.0), ("comparison", 2.0)):
        w = page.option_widgets[key]
        w.setCurrentIndex(w.findData(code))
    text = page.preview.text()
    assert "Event: 1.Yes" in text and "Exposed: 1.Male" in text and "Comparison: 2.Female" in text
    page._run()
    qtbot.waitUntil(lambda: page.current is not None, timeout=60000)
    result, _ = page.current
    assert result.status.value in ("ok", "warning"), result.messages
    assert result.tables[0].name == "2x2 table" and "Risk difference" in page.output.toHtml()


@pytest.mark.r
def test_discovery_page_ordinal_family_and_selectors(hrs_window, qtbot, r_worker):
    from statexplore.analysis import measurement as M
    from statexplore.core.derive import FilterSpec

    ds = hrs_window.state.dataset
    M.approve(ds, "R_SHLT", "ordinal")
    ds.add_filter(FilterSpec("WAVE", "==", 10.0))
    hrs_window.state.notify_metadata_changed()
    page = hrs_window.pages["Automated Model Discovery"]
    page.outcome.setCurrentIndex(page.outcome.findData("R_SHLT"))
    assert "Suggested: Ordinal logistic regression" in page.family_hint.text()
    page._use_suggestion()
    assert page.family.currentData() == "ordinal"
    # standard selector controls on both lists (spec §7.2: below Required Covariates too)
    for sel in (page.cand_sel, page.req_sel):
        assert set(sel.buttons) == {"check_visible", "clear_visible", "check_all", "clear_all"}
    page.cand_sel.set_filter("Big 5")
    page.cand_sel.check_visible()
    page.cand_sel.set_filter("")
    assert len(page.cand_sel.selected()) == 5
    page.req_sel.set_selected(["RAGENDER"])
    page.workers.setValue(2)
    info = page._estimate()
    assert info is not None and info["family"] == "ordinal" and "clm" in page.estimate.text()
    page._start()
    qtbot.waitUntil(lambda: page.run is not None and page.run.status == "complete", timeout=300000)
    assert page.run.settings.family == "ordinal" and page.run.coding["ordering"][0] == "1.Excellent"
    qtbot.waitUntil(lambda: "Ordinal logistic models" in page.summary_view.toPlainText(), timeout=10000)
    text = page.summary_view.toPlainText()
    assert "Ordinal logistic models" in text and "Outcome order" in text
    # gaussian needs the continuous treatment checkbox for the ordinal outcome
    page.family.setCurrentIndex(page.family.findData("gaussian"))
    assert page._estimate() is None and "Treat R_SHLT as continuous" in page.estimate.text()
    page.treatment_panel.ask = False
    page.treatment_panel.set_treated("R_SHLT")
    info = page._estimate()
    assert info is not None and info["family"] == "gaussian" and "R_SHLT: Continuous" in page.estimate.text()


@pytest.mark.r
def test_mixed_page_ltactx_workflow(hrs_window, qtbot, r_worker):
    """§23 steps 3-13: review R_LTACTX, approve ordinal, choose a model, review waves, fit."""
    from statexplore.analysis import measurement as M

    ds = hrs_window.state.dataset
    ds.set_variable_field("HHIDPN", "role", "id")
    M.approve(ds, "R_LTACTX", "ordinal")
    hrs_window.state.notify_metadata_changed()
    page = hrs_window.pages["Mixed & Longitudinal Models"]
    page.outcome.setCurrentIndex(page.outcome.findData("R_LTACTX"))
    page.time.setCurrentIndex(page.time.findData("WAVE"))
    assert page.idv.currentData() == "HHIDPN"
    assert "Suggested family: Ordinal" in page.family_hint.text()
    text = page.preflight_view.toPlainText()
    assert "Wave 1 to Wave 6" in text and "Wave 7 to Wave 16" in text
    # Gaussian needs explicit continuous treatment
    from statexplore.analysis import mixed as MX

    with pytest.raises(MX.MixedInputError, match="Treat R_LTACTX as continuous"):
        MX.validate(ds, page._spec())
    page.treatment_panel.ask = False
    page.treatment_panel.set_treated("R_LTACTX")
    page.pred_sel.set_selected(["RAGENDER"])
    page._fit()
    qtbot.waitUntil(lambda: page.current is not None, timeout=180000)
    res = page.current
    assert res.status.value in ("ok", "warning") and res.values["analysis_range"] == "Wave 7 to Wave 16"
    assert res.inputs["treatments"][0]["variable"] == "R_LTACTX"
    # GEE alternative (population-average), binary outcome
    page.outcome.setCurrentIndex(page.outcome.findData("R_DEPRES"))
    page.family.setCurrentIndex(page.family.findData("binomial"))
    page.estimator.setCurrentIndex(page.estimator.findData("gee"))
    assert "population-average" in page.engine_label.text() and not page.random.isEnabled()
    page.current = None
    page._fit()
    qtbot.waitUntil(lambda: page.current is not None, timeout=180000)
    assert page.current.title.startswith("Generalized estimating equations")


@pytest.mark.r
def test_survey_page(window, qtbot, tmp_path, r_worker, hrs_path):
    import numpy as np
    import pandas as pd

    from statexplore.core.importers import import_file
    from statexplore.core.project import Project

    rng = np.random.default_rng(3)
    n = 300
    df = pd.DataFrame({"stratum": np.repeat([1, 2, 3], 100), "psu": np.repeat(np.arange(1, 13), 25),
                       "w": rng.uniform(1, 3, n), "y": rng.normal(size=n)})
    p = tmp_path / "svy.csv"
    df.to_csv(p, index=False)
    window.state.set_project(Project(import_file(p)))
    page = window.pages["Survey Analysis"]
    for combo, name in ((page.weight, "w"), (page.psu, "psu"), (page.strata, "stratum")):
        combo.setCurrentIndex(combo.findData(name))
    page.kind.setCurrentIndex(page.kind.findData("mean"))
    page.var_sel.set_selected(["y"])
    page._run()
    qtbot.waitUntil(lambda: page.current is not None, timeout=60000)
    assert page.current.status.value == "ok" and "Weighted means" in page.output.toPlainText()
    assert abs(page.current.tables[1].rows[0][2] - np.average(df.y, weights=df.w)) < 1e-10
    window.state.dirty = False
    window.import_path(str(hrs_path))
    qtbot.waitUntil(lambda: window.state.dataset is not None and window.state.dataset.n_rows == 16000, timeout=30000)
    page._detect()
    assert "weight: none found" in page.output.toPlainText()


@pytest.mark.r
def test_r_console_page_and_package_manager(window, qtbot, tmp_path, monkeypatch, r_worker):
    import numpy as np
    import pandas as pd

    from statexplore.core.importers import import_file
    from statexplore.core.project import Project
    from statexplore.gui.pages.r_console import PackageManagerDialog
    from tests.test_console_packages import _binary_pkg

    monkeypatch.setenv("STATEXPLORE_USER_LIB_ROOT", str(tmp_path / "userroot"))
    p = tmp_path / "d.csv"
    pd.DataFrame({"x": np.arange(10.0)}).to_csv(p, index=False)
    window.state.set_project(Project(import_file(p)))
    page = window.pages["R Console"]
    page.editor.setPlainText("mean(my_data$x)\nhist(my_data$x)")
    page.run_script()
    qtbot.waitUntil(lambda: not page._busy, timeout=60000)
    assert "[1] 4.5" in page.output.toPlainText() and page.plots_layout.count() >= 1
    rec = window.state.project.results[-1]
    assert rec.kind == "r_console" and rec.plots and "mean(my_data$x)" in rec.script
    # package manager: install a local binary, then use it in a script
    dlg = PackageManagerDialog(window.state, page)
    assert dlg.install_path(str(_binary_pkg(tmp_path, "sxguipkg")), confirm=False)
    dlg.close()
    page.editor.setPlainText("library(sxguipkg); hello()")
    page.run_script()
    qtbot.waitUntil(lambda: not page._busy, timeout=60000)
    assert "hello from sxguipkg" in page.output.toPlainText()
    # persistent session + explicit import
    page.persistent.setChecked(True)
    page.editor.setPlainText("my_data$x10 <- my_data$x * 10")
    page.run_script()
    qtbot.waitUntil(lambda: not page._busy, timeout=60000)
    assert window.state.dataset.n_cols == 1                              # not applied until imported
    assert page.import_dataset(confirm=False)
    assert list(window.state.dataset.variables) == ["x", "x10"] and len(window.state.project.results) >= 3
