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
    qtbot.addWidget(w)
    return w


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
