"""v1.2 GUI tests (offscreen): discovery strategies, typed hypothesis tables, highlighting toolbar,
model comparison; variable treatment manager."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
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


def load_frame(window, tmp_path, df):
    from statexplore.core.importers import import_file
    from statexplore.core.project import Project

    p = tmp_path / "d.csv"
    df.to_csv(p, index=False)
    window.state.set_project(Project(import_file(p)))


@pytest.mark.r
def test_combination_search_tables_and_highlighting(window, qtbot, tmp_path, r_worker):
    rng = np.random.default_rng(4)
    df = pd.DataFrame(rng.normal(size=(300, 3)), columns=["a", "b", "c"])
    df["y"] = 0.4 * df["a"] + df["a"] * df["b"] + rng.normal(size=300)
    load_frame(window, tmp_path, df)
    page = window.pages["Automated Model Discovery"]
    assert page.strategy.currentData() == "combinations" and page.include_main.isChecked()
    assert page.max_pred.value() == 6 and page.hl_mode.currentData() == "adjusted"
    page.outcome.setCurrentIndex(page.outcome.findData("y"))
    for i in range(page.cands.count()):
        page.cands.item(i).setCheckState(Qt.Checked)
    page.workers.setValue(1)
    page._use_suggestion()
    page.min_pred.setValue(3)
    info = page._estimate()
    assert info["total"] == 8 and "model fits" in page.estimate.text()
    page._start()
    qtbot.waitUntil(lambda: page.run is not None and page.run.status == "complete", timeout=120000)
    qtbot.waitUntil(lambda: len(page.tests) > 0, timeout=10000)
    tf = page.tests
    assert set(tf["test_type"]) == {"overall", "main_term", "interaction_term", "joint_interaction"}
    table = page.discovery_table
    frame = table.model.frame
    assert {"Test ID", "Tested effect", "Test type", "Predictors", "Model terms", "Fixed coefficients", "Likelihood parameters",
            "CI applicability", "Highlight"} <= set(frame.columns)
    assert set(frame["Predictors"]) == {"3 / 6"}
    page.tabs.setCurrentWidget(table)
    hl, shown, total = table.counts()
    assert hl >= 1 and shown == total == len(tf)
    assert "highlighted tests of" in page.hl_count.text()   # v1.3: counters name their unit
    top = frame[frame["Tested effect"] == "a × b"]
    assert len(top) >= 1
    # the hypothesis-type filter shows one family at a time
    page.view_filter.setCurrentIndex(page.view_filter.findData("test_type=interaction_term"))
    assert set(page.discovery_table.model.frame["Test type"]) == {"Individual interaction term"}
    # highlight-only + strict mode are presentation only
    page.hl_only.setChecked(True)
    assert table.proxy.rowCount() == table.counts()[0]
    page.hl_mode.setCurrentIndex(page.hl_mode.findData("strict"))
    strict_n = table.counts()[0]
    page.hl_mode.setCurrentIndex(page.hl_mode.findData("off"))
    assert table.counts()[0] == 0 and strict_n >= 1
    assert window.state.project.settings["discovery_highlight"]["mode"] == "off"
    # model comparison: one shared sample -> ΔAIC shown
    cmp_frame = page.compare_table.model.frame
    assert len(cmp_frame) == 8 and cmp_frame["ΔAIC"].min() == 0
    assert "comparable" in page.compare_note.text()


# ---- M §6 / §15 C: Variable Treatment Manager on the HRS file -------------------------------

def test_treatment_manager_shows_every_model_variable_hrs(hrs_window):
    ds = hrs_window.state.dataset
    page = hrs_window.pages["Mixed & Longitudinal Models"]
    page.outcome.setCurrentIndex(page.outcome.findData("R_SHLT"))
    preds = ["R_LTACTX", "RAGENDER", "R_BMI", "HHID", "ADVTR_MATCHED"]
    page.pred_sel.set_selected(preds)
    panel = page.treatment_panel
    # A. the outcome is always shown, with its SPSS label and levels
    assert panel.outcome == "R_SHLT" and not panel.outcome_box.isHidden()
    assert "Self-rated health" in panel.outcome_info.text() and "source file: nominal" in panel.outcome_info.text()
    assert panel.status("R_SHLT")[0] == "eligible" and not panel.outcome_check.isHidden()
    # B. every checked predictor is listed, eligible or not, with the reason
    assert panel.model_variables() == ["R_SHLT"] + preds
    assert sorted(panel.pred_sel.names()) == sorted(preds)
    st = {n: panel.status(n)[0] for n in preds}
    assert st == {"R_LTACTX": "eligible", "RAGENDER": "eligible", "R_BMI": "continuous",
                  "HHID": "unavailable", "ADVTR_MATCHED": "unavailable"}
    assert "text categories" in panel.status("HHID")[1] and "constant" in panel.status("ADVTR_MATCHED")[1]
    items = {panel.pred_sel.list.item(i).data(Qt.UserRole): panel.pred_sel.list.item(i)
             for i in range(panel.pred_sel.list.count())}
    assert not items["HHID"].flags() & Qt.ItemIsUserCheckable
    assert "already continuous" in items["R_BMI"].text()
    assert "Model variables: 6 · eligible for continuous treatment: 3" in panel.counts.text()
    # searching never unchecks hidden rows; bulk buttons only touch eligible rows
    panel.ask = False
    panel.pred_sel.search.setText("R_LT")
    panel.pred_sel.check_visible()
    assert panel.treated() == ["R_LTACTX"]
    panel.pred_sel.search.setText("")
    panel.pred_sel.check_all()
    assert panel.treated() == ["R_LTACTX", "RAGENDER"]
    panel.pred_sel.search.setText("RAG")
    panel.pred_sel.clear_visible()
    panel.pred_sel.search.setText("")
    assert panel.treated() == ["R_LTACTX"]
    panel.set_treated("R_SHLT")
    assert panel.treated() == ["R_SHLT", "R_LTACTX"] and panel.outcome_check.isChecked()
    with pytest.raises(ValueError):
        panel.set_treated("HHID")
    # changing the predictors re-renders deterministically and keeps valid decisions
    page.pred_sel.set_selected(["R_LTACTX"])
    assert panel.model_variables() == ["R_SHLT", "R_LTACTX"] and panel.treated() == ["R_SHLT", "R_LTACTX"]
    # C. preconfigured treatments apply only when the variable enters the model
    panel.view_mode.setCurrentIndex(panel.view_mode.findData("all"))
    assert "RAGENDER" in panel.pred_sel.names() and "R_BMI" not in panel.pred_sel.names()
    panel.set_treated("RAGENDER")
    assert panel.preconfigured() == ["RAGENDER"]
    assert [t.variable for t in panel.treatments()] == ["R_SHLT", "R_LTACTX"]
    page.pred_sel.set_selected(["R_LTACTX", "RAGENDER"])
    assert [t.variable for t in panel.treatments()] == ["R_SHLT", "R_LTACTX", "RAGENDER"]
    panel.view_mode.setCurrentIndex(panel.view_mode.findData("model"))
    # the analysis spec carries the treatments; the project metadata is untouched
    spec = page._spec()
    assert [t["variable"] for t in spec.treatments] == ["R_SHLT", "R_LTACTX", "RAGENDER"]
    assert ds.var("R_SHLT").measure.value == "nominal" and ds.var("R_LTACTX").source_measure == "nominal"
    from statexplore.analysis.mixed import view as mixed_view
    v = mixed_view(ds, spec)
    assert v.var("R_SHLT").measure.value == "continuous" and v.var("R_LTACTX").measure.value == "continuous"
    # an acknowledgement is dropped when the variable's coding changes (approved ordinal reclassification);
    # the variable stays eligible
    ds.set_variable_field("R_LTACTX", "measure", "ordinal")
    hrs_window.state.notify_metadata_changed()
    page.outcome.setCurrentIndex(page.outcome.findData("R_SHLT"))
    page.pred_sel.set_selected(["R_LTACTX", "RAGENDER"])
    assert "R_LTACTX" not in panel.treated() and "RAGENDER" in panel.treated()
    assert panel.status("R_LTACTX")[0] == "eligible" and "ordinal" in panel.status("R_LTACTX")[1]


def test_discovery_outcome_row_sits_below_family(hrs_window):
    page = hrs_window.pages["Automated Model Discovery"]
    page.outcome.setCurrentIndex(page.outcome.findData("R_LTACTX"))
    panel = page.treatment_panel
    assert panel.outcome == "R_LTACTX" and "Frequency of light physical activity" in panel.outcome_info.text()
    assert panel.outcome_box.parent() is not panel        # re-parented into the outcome form


# ---- M §8 / §15 E: longitudinal setup in Discovery, refit handoff ---------------------------

def test_longitudinal_setup_and_preflight_hrs(hrs_window):
    ds = hrs_window.state.dataset
    page = hrs_window.pages["Automated Model Discovery"]
    page.outcome.setCurrentIndex(page.outcome.findData("R_LTACTX"))
    assert page.study.currentData() == "independent" and not page.participant.isEnabled()
    page.cand_sel.set_selected(["RAGENDER"])
    page.study.setCurrentIndex(page.study.findData("longitudinal"))
    assert page.participant.isEnabled()
    # any structurally valid column can be the ID; the time variable is chosen explicitly
    page.participant.setCurrentIndex(page.participant.findData("HHIDPN"))
    page.time_combo.setCurrentIndex(page.time_combo.findData("WAVE"))
    txt = page.long_info.toPlainText()
    assert "Participants 1,000" in txt and "Waves 16" in txt
    assert "not observed at 6 of 16 time points" in txt          # R_LTACTX waves 1-6: reported, never zero-filled
    page.family.setCurrentIndex(page.family.findData("ordinal"))
    assert page.structure.findData("clmm_ri") >= 0 and page.structure.findData("ri") < 0
    page.time_combo.setCurrentIndex(0)                           # no time: the time strategy is refused
    page.strategy.setCurrentIndex(page.strategy.findData("time"))
    with pytest.raises(Exception, match="needs a time variable"):
        from statexplore.discovery.space import validate_settings
        ds.set_variable_field("R_LTACTX", "measure", "ordinal")
        validate_settings(ds, page._settings())


@pytest.mark.r
def test_longitudinal_candidate_refits_in_mixed_workspace(window, qtbot, tmp_path, r_worker):
    rng = np.random.default_rng(8)
    n_id, k = 60, 4
    df = pd.DataFrame({"pid": np.repeat(np.arange(n_id), k), "t": np.tile(np.arange(k), n_id)})
    df["g"] = np.repeat(rng.integers(0, 2, n_id), k)
    df["y"] = np.repeat(rng.normal(size=n_id), k) + 0.5 * df["g"] * df["t"] + rng.normal(size=len(df))
    load_frame(window, tmp_path, df)
    page = window.pages["Automated Model Discovery"]
    page.outcome.setCurrentIndex(page.outcome.findData("y"))
    page._use_suggestion()
    page.study.setCurrentIndex(page.study.findData("longitudinal"))
    page.participant.setCurrentIndex(page.participant.findData("pid"))
    page.time_combo.setCurrentIndex(page.time_combo.findData("t"))
    page.cand_sel.set_selected(["g"])
    page.strategy.setCurrentIndex(page.strategy.findData("time"))
    page.structure.setCurrentIndex(page.structure.findData("rs"))
    page.workers.setValue(1)
    info = page._estimate()
    assert info is not None and info["total"] == 2, page.estimate.text()      # main-effects model + t × g
    page._start()
    qtbot.waitUntil(lambda: page.run is not None and page.run.status == "complete", timeout=120000)
    qtbot.waitUntil(lambda: len(page.tests) > 0, timeout=10000)
    row = next(r for r in page.run.rows if r["kind"] == "focal")
    assert row["formula"].endswith("+ (1 + t | pid)") and row["n_groups"] == n_id
    mixed = window.pages["Mixed & Longitudinal Models"]
    page._to_mixed(row)
    qtbot.waitUntil(lambda: mixed.current is not None, timeout=120000)
    res = mixed.current
    assert any(m.startswith("Refit: discovery search") for m in res.messages)
    assert res.inputs["spec"]["terms"] == [["t"], ["g"], ["t", "g"]] and res.inputs["spec"]["random"] == "slope"
    assert window.stack.currentWidget() is mixed
