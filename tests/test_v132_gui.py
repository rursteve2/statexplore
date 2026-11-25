"""v1.3.2 GUI regression tests: one interface, clustered Discovery controls, state persistence (CD-12, UI-31/41/42/
43/46/47/48/49/50/56/57). Screenshots at several window sizes and Qt scale factors are written when
STATEXPLORE_SHOTS is set (scripts in E:\\sx_v132_work run this file at QT_SCALE_FACTOR 1.0 and 1.5)."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

from statexplore.core.importers import import_file  # noqa: E402

FX = Path(__file__).parent / "fixtures" / "v132"
SHOTS = Path(os.environ["STATEXPLORE_SHOTS"]) if os.environ.get("STATEXPLORE_SHOTS") else None


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


def _load(window, name="T38"):
    from statexplore.core.project import Project

    window.state.set_project(Project(import_file(FX / f"{name}.csv")))
    return window.pages["Automated Model Discovery"]


def _configure_b(page, cluster="SCHOOL", candidates=("GROUP",)):
    page.outcome.setCurrentIndex(page.outcome.findData("Y"))
    page._use_suggestion()
    page.study.setCurrentIndex(page.study.findData("longitudinal"))
    page.participant.setCurrentIndex(page.participant.findData("ID"))
    page.time_combo.setCurrentIndex(page.time_combo.findData("TIME"))
    page.cluster_combo.setCurrentIndex(page.cluster_combo.findData(cluster))
    page.workflow.setCurrentIndex(page.workflow.findData("structure"))
    page.cand_sel.set_selected(list(candidates))
    page.workers.setValue(1)


def test_cd12_cluster_controls_one_state_and_valid_structures(window, qtbot):
    page = _load(window)
    _configure_b(page)
    s = page._settings()
    assert s.cluster == "SCHOOL" and s.id_policy == "global" and not s.cluster_compare
    # with a cluster only the Gaussian RI/DIAG/RS x residual structures are offered; the 3 x 3 table is checked
    keys = [page.structures_list.item(i).data(0x0100) for i in range(page.structures_list.count())]
    assert not any(k.startswith(("mmrm", "gee", "glmer", "clmm")) for k in keys)
    assert sorted(s.structures) == sorted(["ri", "rs_uncorr", "rs", "ar1", "diag_ar1", "rs_ar1", "car1", "diag_car1",
                                           "rs_car1"])
    page.cluster_compare.setChecked(True)
    page.workflow.setCurrentIndex(page.workflow.findData("fixed"))          # A: exploratory option disabled ...
    assert not page.cluster_compare.isEnabled() and page.cluster_compare.isChecked()   # ... but its value is kept
    assert page._settings().cluster_compare is False
    page.workflow.setCurrentIndex(page.workflow.findData("structure"))
    assert page._settings().cluster_compare is True
    # UI-56 / CD-04: time-only workflow B with no candidate predictor is accepted
    page.cand_sel.set_selected([])
    info = page._estimate()
    assert info is not None and info["total"] == 18


def test_ui50_empty_search_explained(window):
    page = _load(window)
    page.outcome.setCurrentIndex(page.outcome.findData("Y"))
    page._use_suggestion()
    page.cand_sel.set_selected(["GROUP"])
    page.strategy.setCurrentIndex(page.strategy.findData("combinations"))
    page.include_main.setChecked(False)
    page.max_pred.setValue(1)
    assert page._estimate() is None
    assert "no candidate model" in page.estimate.text()


@pytest.mark.r
def test_ui42_49_57_csv_status_reload_and_cluster_columns(window, qtbot, tmp_path):
    from PySide6.QtCore import Qt

    page = _load(window)
    _configure_b(page)
    page.structures_list.blockSignals(True)
    for i in range(page.structures_list.count()):
        it = page.structures_list.item(i)
        it.setCheckState(Qt.Checked if it.data(0x0100) in ("ri", "rs") else Qt.Unchecked)
    page.structures_list.blockSignals(False)
    page._start()
    qtbot.waitUntil(lambda: page.run is not None and page.run.status == "complete", timeout=300000)
    qtbot.waitUntil(lambda: page.all_table.model.rowCount() == 2, timeout=20000)
    csv = page.discovery_table.csv_text()
    head, *rows = csv.strip().splitlines()
    assert "Highlight" in head and rows and all(r.split(",")[0] for r in rows)       # UI-42: never blank
    page.tabs.setCurrentWidget(page.compare_table)
    cmp_csv = page.compare_table.csv_text()
    import csv as _csv
    import io

    recs = list(_csv.DictReader(io.StringIO(cmp_csv)))
    assert recs and all(float(r["Clusters (modeled)"]) == 20 for r in recs)              # UI-57
    # UI-48 / UI-49: save, reopen: form state and the latest search come back immediately
    path = tmp_path / "p.sxp"
    assert window._save_to(path)
    window.open_project(str(path))
    page = window.pages["Automated Model Discovery"]
    assert page.run is not None and page.run.settings.cluster == "SCHOOL"
    assert page.cluster_combo.currentData() == "SCHOOL" and page.workflow.currentData() == "structure"
    assert "rs" in page._checked_structures()
    assert page.summary_view.toPlainText().strip()


def test_ui43_frozen_column_stays_inside_viewport(window, qtbot):
    import pandas as pd

    from statexplore.gui.pages.discovery import DISCOVERY_COLS, ResultTable

    t = ResultTable(DISCOVERY_COLS, id_col="Test ID")
    qtbot.addWidget(t)
    df = pd.DataFrame({"test_id": [f"{i}.0" for i in range(300)], "idx": range(300), "label": ["x" * 80] * 300,
                       "p": [0.5] * 300, "status_text": [""] * 300})
    t.set_frame(df)
    t.resize(500, 300)
    t.show()
    for size in ((500, 300), (1400, 900), (420, 260)):                  # resize / maximize / shrink
        t.resize(*size)
        qtbot.wait(30)
        t.view.verticalScrollBar().setValue(t.view.verticalScrollBar().maximum())
        qtbot.wait(30)
        if t._frozen is not None and t._frozen.isVisible():
            g, vp = t._frozen.geometry(), t.view.viewport().geometry()
            assert g.bottom() <= vp.bottom() + 1                             # never over the horizontal scrollbar
            assert t._frozen.verticalScrollBar().value() == t.view.verticalScrollBar().value()


def test_ui31_selected_first_in_every_multiselect(qtbot):
    from statexplore.gui.widgets.variable_selector import VariableSelector

    ds = import_file(FX / "T38.csv")
    for single in (False, True):
        w = VariableSelector(single=single)
        qtbot.addWidget(w)
        w.set_variables(ds, ["SCHOOL", "ID", "TIME", "GROUP", "Y"])
        w.set_selected(["GROUP"])
        qtbot.wait(10)
        names = [w.list.item(i).data(0x0100) for i in range(w.list.count())]
        if single:
            assert names[0] == "SCHOOL"                                       # single lists keep their order
        else:
            assert names[1] == "GROUP"                                        # header, then the selection
        w.search.setText("o")                                                 # filtering keeps the grouping
        assert w.selected() == ["GROUP"]


def test_ui46_47_measurement_review_wording(window):
    import pyreadstat
    import pandas as pd

    from statexplore.core.project import Project

    tmp = Path(os.environ.get("TMP", ".")) / "ui47.sav"
    pyreadstat.write_sav(pd.DataFrame({"X": [1.0, 2, 99, 3, 99, None], "Y": [1.0, 2, 3, 4, 5, 6]}), str(tmp),
                         missing_ranges={"X": [99]}, variable_value_labels={"X": {99: "refused"}},
                         variable_measure={"X": "nominal", "Y": "scale"})
    window.state.set_project(Project(import_file(tmp)))
    page = window.pages["Dataset Explorer"].review
    window.state.dataset.set_variable_field("X", "measure", "ordinal")     # an edit after import
    window.state.notify_metadata_changed()
    if hasattr(page, "show_all"):
        page.show_all.setChecked(True)
    page.current = lambda: "X"                                               # select X regardless of list filters
    page._show_detail()
    text = page.detail.toPlainText()
    assert "Declared user-missing codes" in text and "Rows holding one of those codes: 2" in text
    assert "blank (system-missing) rows: 1" in text
    assert page.keep_btn.text() == "Keep current level (ordinal)"
    assert page.restore_btn.text().startswith("Restore source level (nominal")


def test_cd12_screenshots(window, qtbot):
    if SHOTS is None:
        pytest.skip("screenshots only when STATEXPLORE_SHOTS is set (evidence run, not a functional test)")
    page = _load(window)
    _configure_b(page)
    window.sidebar.setCurrentRow(list(window.pages).index("Automated Model Discovery"))
    SHOTS.mkdir(parents=True, exist_ok=True)
    for w, h in ((1366, 768), (1920, 1080), (2560, 1440)):
        window.resize(w, h)
        qtbot.wait(100)
        window.grab().save(str(SHOTS / f"discovery_cluster_{w}x{h}.png"))
    mixed = window.pages["Mixed & Longitudinal"] if "Mixed & Longitudinal" in window.pages else None
    if mixed is not None:
        window.sidebar.setCurrentRow(list(window.pages).index("Mixed & Longitudinal"))
        window.resize(1366, 768)
        qtbot.wait(100)
        window.grab().save(str(SHOTS / "mixed_1366x768.png"))
