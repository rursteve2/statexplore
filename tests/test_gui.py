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
