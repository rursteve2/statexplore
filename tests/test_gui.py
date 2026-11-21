"""GUI smoke tests (offscreen): window builds, import → explorer, edit → undo, save → reopen."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

from statexplore.gui.main_window import PLANNED, SECTIONS, MainWindow  # noqa: E402
from statexplore.gui.state import AppState  # noqa: E402


@pytest.fixture
def window(qtbot, tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings

    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(tmp_path / "settings"))
    w = MainWindow(AppState())
    yield w
    # Not registered with qtbot: pytest-qt would close it before this cleanup, and
    # closing a window with unsaved changes opens a modal "save?" prompt.
    w.state.dirty = False
    w.close()


def test_sidebar_sections(window):
    assert window.sidebar.count() == len(SECTIONS) == 10
    for i, name in enumerate(SECTIONS):
        window.sidebar.setCurrentRow(i)
        assert window.stack.currentIndex() == i
        assert window.sidebar.item(i).text().startswith(name)
    planned = [window.sidebar.item(i).text() for i in range(10) if "(Phase" in window.sidebar.item(i).text()]
    assert len(planned) == len(PLANNED)


def test_import_edit_save_reopen(window, qtbot, sav_labelled, tmp_path):
    window.import_path(str(sav_labelled))
    qtbot.waitUntil(lambda: window.state.project is not None, timeout=10000)
    explorer = window.pages["Dataset Explorer"]
    assert explorer.preview_model.rowCount() == 60
    assert explorer.overview_model.rowCount() == 5
    assert window.stack.currentIndex() == SECTIONS.index("Dataset Explorer")

    # value labels toggle shows labels without changing data
    from PySide6.QtCore import Qt

    col = list(window.state.dataset.variables).index("sex")
    explorer.preview_model.set_show_labels(True)
    shown = explorer.preview_model.data(explorer.preview_model.index(0, col), Qt.DisplayRole)
    assert shown in ("Male", "Female")

    # edit display name through the form, then undo
    explorer.var_list.setCurrentRow(col)
    explorer.f_display.setText("Gender")
    explorer._apply()
    assert window.state.dataset.var("sex").display_name == "Gender"
    assert explorer.preview_model.headerData(col, Qt.Horizontal) == "Gender"
    explorer._undo()
    assert window.state.dataset.var("sex").display_name == "sex"
    explorer.f_display.setText("Gender")
    explorer._apply()

    out = tmp_path / "gui.sxp"
    assert window._save_to(out)
    assert not window.state.dirty
    window.open_project(str(out))
    assert window.state.dataset.var("sex").display_name == "Gender"
    assert window.state.project.path == out


@pytest.mark.r
def test_analysis_builder_runs_regression(window, qtbot, sav_labelled, r_worker):
    from PySide6.QtCore import Qt

    window.import_path(str(sav_labelled))
    qtbot.waitUntil(lambda: window.state.project is not None, timeout=10000)
    page = window.pages["Analysis Builder"]
    lm_item = [page.tree.topLevelItem(i).child(j) for i in range(page.tree.topLevelItemCount())
               for j in range(page.tree.topLevelItem(i).childCount())
               if page.tree.topLevelItem(i).child(j).data(0, Qt.UserRole) == "lm"][0]
    page.tree.setCurrentItem(lm_item)
    dv = page.slot_widgets["dependent"].combo
    dv.setCurrentIndex(dv.findData("score"))
    preds = page.slot_widgets["predictors"].list
    for i in range(preds.count()):
        if preds.item(i).data(Qt.UserRole) in ("sex", "income"):
            preds.item(i).setCheckState(Qt.Checked)
    page._run()
    qtbot.waitUntil(lambda: page.current is not None, timeout=60000)
    result, artifacts = page.current
    assert result.status.value in ("ok", "warning"), result.messages
    html = page.output.toHtml()
    assert "Coefficients" in html and "sex=Female" in html
    assert len(result.plots) == 2
    window.state.shutdown_r()


@pytest.mark.r
def test_discovery_page_end_to_end(window, qtbot, tmp_path, r_worker):
    import numpy as np
    import pandas as pd
    from PySide6.QtCore import Qt

    rng = np.random.default_rng(4)
    df = pd.DataFrame(rng.normal(size=(300, 4)), columns=["a", "b", "c", "d"])
    df["y"] = df["a"] * df["b"] + rng.normal(size=300)
    p = tmp_path / "disc.csv"
    df.to_csv(p, index=False)
    from statexplore.core.importers import import_file
    from statexplore.core.project import Project

    window.state.set_project(Project(import_file(p)))
    page = window.pages["Automated Model Discovery"]
    page.outcome.setCurrentIndex(page.outcome.findData("y"))
    for i in range(page.cands.count()):
        page.cands.item(i).setCheckState(Qt.Checked)
    page.workers.setValue(1)
    info = page._estimate()
    assert info["total"] == 6 and "<b>6</b>" in page.estimate.text()
    page._start()
    qtbot.waitUntil(lambda: page.run is not None and page.run.status == "complete", timeout=120000)
    qtbot.waitUntil(lambda: page.discovery_table.model.rowCount() == 6, timeout=10000)
    assert page.discovery_table.model.frame.iloc[0]["Interaction"] == "a × b"
    assert "Complete" in page.summary_view.toPlainText()
    assert window.state.project.searches == [page.run]
    page.tabs.setCurrentWidget(page.discovery_table)
    page.discovery_table.view.selectRow(0)
    assert page._selected_row()["interaction"] == "a × b"


@pytest.mark.r
def test_teaching_and_comparison_pages(window, qtbot, tmp_path, r_worker):
    import numpy as np
    import pandas as pd

    from statexplore.analysis.models import ModelSpec
    from statexplore.core.importers import import_file
    from statexplore.core.project import Project

    rng = np.random.default_rng(6)
    df = pd.DataFrame({"x": rng.normal(size=200), "g": rng.choice(["a", "b"], 200)})
    df["y"] = df["x"] * (df["g"] == "b") + rng.normal(size=200)
    p = tmp_path / "t.csv"
    df.to_csv(p, index=False)
    window.state.set_project(Project(import_file(p)))
    m_int = ModelSpec("y", [("x", "g")], "gaussian", label="int")
    window.state.show_in_teaching.emit(m_int)
    assert window.stack.currentIndex() == window.sidebar.currentRow()
    page = window.pages["Teaching Studio"]
    assert page.x.currentData() == "x" and page.group.currentData() == "g"
    page._draw()
    qtbot.waitUntil(lambda: page.current is not None, timeout=60000)
    assert "Interpretation." in page.output.toPlainText()
    window.state.add_model(page.editor.model())          # what "Add to comparison set" does
    window.state.add_model(ModelSpec("y", [("x",), ("g",)], "gaussian", label="main"))
    comp = window.pages["Model Comparison"]
    assert comp.list.count() == 2
    comp._compare()
    qtbot.waitUntil(lambda: comp.current is not None, timeout=60000)
    assert "Nested model tests" in comp.output.toPlainText()
    out = tmp_path / "m.sxp"
    assert window._save_to(out)
    window.open_project(str(out))
    assert [m.label for m in window.state.project.models] == ["int", "main"]
    assert window.state.project.models[0].terms == [("x",), ("g",), ("x", "g")]


def test_model_editor_keeps_interaction_in_any_order(window, tmp_path):
    import pandas as pd

    from statexplore.analysis.models import ModelSpec
    from statexplore.core.importers import import_file
    from statexplore.gui.model_editor import ModelEditor

    p = tmp_path / "e.csv"
    pd.DataFrame({"y": [1.0, 2, 3, 4], "a": [1.0, 2, 3, 5], "b": [2.0, 1, 4, 3]}).to_csv(p, index=False)
    ds = import_file(p)
    ed = ModelEditor(ds)
    ed.set_model(ModelSpec("y", [("b", "a")], "gaussian"))  # reverse of the widget's variable order
    assert set(map(frozenset, ed.model().terms)) == {frozenset({"a"}), frozenset({"b"}), frozenset({"a", "b"})}


@pytest.mark.r
def test_mixed_page(window, qtbot, tmp_path, r_worker):
    import numpy as np
    import pandas as pd
    from PySide6.QtCore import Qt

    from statexplore.core.importers import import_file
    from statexplore.core.project import Project

    rng = np.random.default_rng(1)
    df = pd.DataFrame({"pid": np.repeat(np.arange(1, 31), 4), "wave": np.tile([1, 2, 3, 4], 30),
                       "grp": np.repeat(rng.choice(["a", "b"], 30), 4)})
    df["y"] = rng.normal(size=30).repeat(4) + 0.5 * df["wave"] + rng.normal(size=120)
    p = tmp_path / "long.csv"
    df.to_csv(p, index=False)
    window.state.set_project(Project(import_file(p)))
    page = window.pages["Mixed & Longitudinal Models"]
    page._detect()
    texts = [page.suggestions.item(i).text() for i in range(page.suggestions.count())]
    assert texts[0].startswith("ID: pid") and any(t.startswith("TIME: wave") for t in texts)
    page.suggestions.setCurrentRow(0)
    page._apply_suggestion()
    assert window.state.dataset.var("pid").role.value == "id" and page.idv.currentData() == "pid"
    page.outcome.setCurrentIndex(page.outcome.findData("y"))
    page.time.setCurrentIndex(page.time.findData("wave"))
    for i in range(page.preds.count()):
        if page.preds.item(i).data(Qt.UserRole) == "grp":
            page.preds.item(i).setCheckState(Qt.Checked)
    page._fit()
    qtbot.waitUntil(lambda: page.current is not None, timeout=60000)
    assert "Random effects" in page.output.toPlainText() and len(page.current.plots) == 3
    page.residual.setCurrentIndex(page.residual.findData("unstructured"))
    assert page.random.currentData() == "none" and "mmrm" in page.engine_label.text()
