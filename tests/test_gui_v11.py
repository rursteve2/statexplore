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
    assert review.table.item(row, 2).text() == "nominal" and review.table.item(row, 4).text() == "ordinal"
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
    assert "R_LTACTX" in panel._boxes and "R_BMI" not in panel._boxes
    assert panel._boxes["R_LTACTX"].text() == "Treat R_LTACTX as continuous for this analysis"
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
    assert "R_LTACTX" in page.treatment_panel._boxes
    page.treatment_panel.ask = False
    page.treatment_panel.set_treated("R_LTACTX")
    page._describe()
    qtbot.waitUntil(lambda: len(hrs_window.state.project.results) == n_before + 4, timeout=60000)
    r = hrs_window.state.project.results[-1]
    assert r.kind == "describe" and r.inputs["treatments"][0]["variable"] == "R_LTACTX"
    assert r.tables[0].rows[0][1] == pytest.approx(hrs_window.state.dataset.analysis_column("R_LTACTX").mean())
