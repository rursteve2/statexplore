"""v1.3.3 GUI regression tests (real widgets, offscreen Qt, real R): AMD validation panel (V133-VAL-01), workflow B
copy and composite-key labels (V133-UI-02/04), stale progress (UI-03), preflight labels (UI-01/05), Save CSV and exported
interpretations (UI-06/07)."""
from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

from statexplore.core.importers import import_file  # noqa: E402

FX = Path(__file__).parent / "fixtures" / "v133"


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


def _load(window):
    from statexplore.core.project import Project

    window.state.set_project(Project(import_file(FX / "Test43B.csv")))
    return window.pages["Automated Model Discovery"]


def _configure(page, workflow="fixed", strategy="main"):
    page.outcome.setCurrentIndex(page.outcome.findData("Y"))
    page._use_suggestion()
    page.study.setCurrentIndex(page.study.findData("longitudinal"))
    page.participant.setCurrentIndex(page.participant.findData("ID"))
    page.time_combo.setCurrentIndex(page.time_combo.findData("TIME"))
    page.cluster_combo.setCurrentIndex(page.cluster_combo.findData("SCHOOL"))
    page.workflow.setCurrentIndex(page.workflow.findData(workflow))
    if workflow == "fixed":
        page.structure.setCurrentIndex(page.structure.findData("ri"))
        page.strategy.setCurrentIndex(page.strategy.findData(strategy))
    page.cand_sel.set_selected(["GROUP"])
    page.workers.setValue(1)


def test_preflight_labels():
    from statexplore.analysis.longitudinal_preflight import preflight

    d = import_file(FX / "reference" / "Test46B_MissingTime.csv")
    rep = preflight(d, "Y", "ID", "TIME", ["GROUP"], cluster="SCHOOL")
    cols = rep.tables(d)[0].columns
    assert "time range with observed outcome (first to last)" in cols and "outcome range (min to max)" not in cols
    assert rep.interior_absent and any("inside the analysis range" in w for w in rep.warnings)     # UI-05
    from statexplore.analysis import mixed

    src = Path(mixed.__file__).read_text(encoding="utf-8")
    assert "are not part of the analysis range" not in src


def test_workflow_b_copy_and_composite_key_labels(window):
    page = _load(window)
    _configure(page, workflow="structure")
    page._estimate()
    txt = page.estimate.text()
    assert "comparison: none" in txt and "focal test" not in txt                       # V133-UI-02
    from statexplore.discovery.space import random_part

    s = page._settings()
    assert "composite participant key" in random_part(window.state.dataset, s, "ri", False)   # V133-UI-04
    assert "composite" not in random_part(window.state.dataset, s, "ri", True)


def test_blocked_start_resets_progress(window):
    page = _load(window)
    page.outcome.setCurrentIndex(page.outcome.findData("Y"))
    page._use_suggestion()
    page.progress.setRange(0, 10)
    page.progress.setValue(10)                                                        # a previous run's 100 %
    page.cand_sel.set_selected(["GROUP"])
    page.strategy.setCurrentIndex(page.strategy.findData("combinations"))
    page.include_main.setChecked(False)
    page.max_pred.setValue(1)
    page._start()
    assert page.progress.value() == 0 and page.progress_label.text().startswith("Not started")


@pytest.mark.r
def test_csv_save_and_interpretation(window, qtbot, tmp_path):
    page = _load(window)
    _configure(page, strategy="time")
    page._start()
    qtbot.waitUntil(lambda: page.run is not None and page.run.status == "complete", timeout=600000)
    qtbot.waitUntil(lambda: page.discovery_table.model.rowCount() >= 1, timeout=30000)
    path = page.discovery_table.save_csv(str(tmp_path / "tests é.csv"))
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    assert rows and all(r["Teaching interpretation"].strip() for r in rows)            # V133-UI-06: not blank
    assert "log10 p" in rows[0] and float(rows[0]["log10 p"]) < -10                    # V133-UI-08
    assert page.discovery_table.csv_text().count("\n") == len(rows) + 1                # same as Copy CSV


@pytest.mark.r
def test_validation_panel_run_persist_reopen(window, qtbot, tmp_path):
    page = _load(window)
    _configure(page)
    panel = page.validation_panel
    assert not panel.run_btn.isEnabled()                                               # configurable before a search
    panel.auto.setChecked(True)
    panel.target.setCurrentIndex(panel.target.findData("cluster"))
    for k, c in panel.mode_checks.items():
        c.setChecked(k == "holdout")
    page._start()
    qtbot.waitUntil(lambda: page.run is not None and page.run.status == "complete", timeout=600000)
    qtbot.waitUntil(lambda: len(page.run.validation_runs) == 1, timeout=600000)        # ran automatically
    v = page.run.validation_runs[0]
    assert v["status"] == "complete" and v["config"]["target"] == "cluster"
    assert "complete" in panel.status.text() and "Validation performance" in panel.view.toHtml()
    assert "units for this target 20" in panel.feas.text()
    # a second, different run is a new immutable record; row target is refused with its reason
    model = panel.target.model()
    assert not model.item(panel.target.findData("rows")).isEnabled()
    panel.target.setCurrentIndex(panel.target.findData("participant_in_cluster"))
    panel.preview()
    assert "disjointness checks: OK" in panel.view.toPlainText()
    panel.start()
    qtbot.waitUntil(lambda: len(page.run.validation_runs) == 2, timeout=600000)
    assert page.run.validation_runs[0]["config"]["target"] == "cluster"                 # unchanged
    from statexplore.discovery import validate as V

    files = V.export(page.run.validation_runs[1], str(tmp_path / "val"))     # v1.3.6: new folder, typed evidence
    assert any(f.endswith("split_manifest.csv") for f in files)
    # save, reopen: validation runs and their splits come back with the search
    p = tmp_path / "proj.sxp"
    assert window._save_to(p)
    window.open_project(str(p))
    page = window.pages["Automated Model Discovery"]
    assert len(page.run.validation_runs) == 2 and page.validation_panel.history.count() == 2
    assert len(page.run.validation_runs[1]["manifest"]) == 960 * len(page.run.validation_runs[1]["config"]["modes"])
