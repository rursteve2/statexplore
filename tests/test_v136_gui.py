"""v1.3.6 GUI tests (real widgets, offscreen Qt; real R where marked): Reports per-export data consent and new export
folders (AUD135-03), the code dialog refusing a historical result whose data cannot be proven (AUD135-07), the
validation panel export into a new folder with consent, and the 'unverified' title for legacy searches (AUD135-06)."""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from statexplore.core.importers import import_file  # noqa: E402

FX3 = Path(__file__).parent / "fixtures" / "v133"


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


class _Boxes:
    """Records message boxes; answers questions with `answer`."""

    def __init__(self, monkeypatch, answer):
        from PySide6.QtWidgets import QMessageBox

        self.seen: list[tuple[str, str]] = []
        monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: (self.seen.append(("question", a[2])),
                                                                                 answer)[1]))
        for kind in ("information", "warning"):
            monkeypatch.setattr(QMessageBox, kind, staticmethod(
                lambda *a, _k=kind, **k: self.seen.append((_k, a[2])) or QMessageBox.Ok))


def _project_with_results(tmp_path):
    from statexplore.core.project import Project
    from tests.test_v136 import _FakeWorker, _scores, _t_one
    from statexplore.analysis.session import RSession

    d = _scores(tmp_path)
    p = Project(d)
    sess = RSession(_FakeWorker(), tmp_path / "w")
    r1, a1 = _t_one(sess, d)
    r2, a2 = _t_one(sess, d)
    r2.values.pop("analysis_provenance")              # as saved by StatExplore 1.3.5
    p.add_result(r1, a1)
    p.add_result(r2, a2)
    return p, r1, r2


def test_reports_export_asks_consent_and_writes_a_new_folder(window, monkeypatch, tmp_path):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    p, r1, r2 = _project_with_results(tmp_path)
    window.state.set_project(p)
    page = window.pages["Reports"]
    for i in range(page.results.count()):
        page.results.item(i).setCheckState(Qt.Checked)
    out = tmp_path / "chosen"
    out.mkdir()
    (out / "data.csv").write_text("an older export\n")
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(out)))
    boxes = _Boxes(monkeypatch, QMessageBox.No)
    page._export()
    assert boxes.seen[0][0] == "question" and "Include the analysis data" in boxes.seen[0][1]
    info = boxes.seen[-1][1]
    assert "new folder" in info and "1 analysis(es) got no runnable reproduction" in info
    new = page.last_export
    assert new.parent == out and new != out and (out / "data.csv").read_text() == "an older export\n"
    assert not [f for f in new.rglob("*.csv") if "data" in f.name]               # consent declined: no data at all
    inv = json.loads((new / "bundle_inventory.json").read_text())
    assert inv["data_consent"] is False and inv["bundle_kind"] == "code-only"
    assert (new / f"One_sample_t_test_{r2.id}" / "REPRODUCTION_UNAVAILABLE.txt").exists() or \
        any(f.name == "REPRODUCTION_UNAVAILABLE.txt" and r2.id in f.parent.name for f in new.rglob("*"))
    # Yes: a second, separate folder; the provable result carries its own data
    boxes2 = _Boxes(monkeypatch, QMessageBox.Yes)
    page._export()
    new2 = page.last_export
    assert new2 != new and [f for f in new2.rglob("*_data.csv") if r1.id in f.parent.name]
    assert not [f for f in new2.rglob("*_data.csv") if r2.id in f.parent.name]
    assert boxes2.seen


def test_code_dialog_refuses_unprovable_history_and_saves_provable(window, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    from statexplore.gui.pages.eda import CodeDialog

    p, r1, r2 = _project_with_results(tmp_path)
    out = tmp_path / "bundle"
    out.mkdir()
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(out)))
    boxes = _Boxes(monkeypatch, QMessageBox.No)
    dlg = CodeDialog(r2, window, dataset=p.dataset)
    dlg._save()
    assert boxes.seen[-1][0] == "warning" and "No reproducibility bundle was written" in boxes.seen[-1][1]
    assert list(out.iterdir()) == []
    dlg.close()
    dlg2 = CodeDialog(r1, window, dataset=p.dataset)
    dlg2._save()
    assert dlg2.saved and dlg2.saved[0].parent.parent == out
    assert not [f for f in dlg2.saved if f.name.endswith("_data.csv")]           # default No: code-only
    dlg2.close()


def test_legacy_search_dialog_title_says_unverified(window, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QMessageBox

    from statexplore.core.project import Project
    from statexplore.discovery.engine import create_run
    from statexplore.discovery.space import SearchSettings
    from tests.test_v136 import SWB, WORKBOOK, _legacy

    a = import_file(WORKBOOK, sheet="A")
    run = create_run(a, SearchSettings(**SWB))
    old, _sig = _legacy(run)
    window.state.set_project(Project(a))
    page = window.pages["Automated Model Discovery"]
    page.run = old
    titles = []
    monkeypatch.setattr(QMessageBox, "exec", lambda self: titles.append(self.windowTitle()) or 0)
    assert page._historical_view("View candidate code") is None
    assert titles == ["The original data of this search cannot be verified"]


@pytest.mark.r
def test_validation_panel_export_goes_to_new_folders_with_consent(window, qtbot, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    from statexplore.core.project import Project

    window.state.set_project(Project(import_file(FX3 / "Test43B.csv")))
    page = window.pages["Automated Model Discovery"]
    page.outcome.setCurrentIndex(page.outcome.findData("Y"))
    page._use_suggestion()
    page.study.setCurrentIndex(page.study.findData("longitudinal"))
    page.participant.setCurrentIndex(page.participant.findData("ID"))
    page.time_combo.setCurrentIndex(page.time_combo.findData("TIME"))
    page.cluster_combo.setCurrentIndex(page.cluster_combo.findData("SCHOOL"))
    page.workflow.setCurrentIndex(page.workflow.findData("fixed"))
    page.structure.setCurrentIndex(page.structure.findData("ri"))
    page.strategy.setCurrentIndex(page.strategy.findData("main"))
    page.cand_sel.set_selected(["GROUP"])
    page.workers.setValue(1)
    panel = page.validation_panel
    panel.auto.setChecked(True)
    panel.target.setCurrentIndex(panel.target.findData("cluster"))
    for k, c in panel.mode_checks.items():
        c.setChecked(k == "holdout")
    page._start()
    qtbot.waitUntil(lambda: page.run is not None and len(page.run.validation_runs) == 1, timeout=900000)
    out = tmp_path / "val"
    out.mkdir()
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(out)))
    boxes = _Boxes(monkeypatch, QMessageBox.No)
    panel.export()
    f1 = panel.last_export
    assert f1.parent == out and "new folder" in boxes.seen[-1][1]
    assert (f1 / "validation_reproduce.R").exists() and not list(f1.glob("*_data.csv"))
    assert json.loads((f1 / "validation_manifest.json").read_text())["reproduction"]["bundle_kind"] == "code-only"
    _Boxes(monkeypatch, QMessageBox.Yes)
    panel.export()
    f2 = panel.last_export
    assert f2 != f1 and (f2 / "validation_reproduce_data.csv").exists()
    led = pd.read_csv(f2 / "execution_ledger.csv")
    assert led["committed"].all() and len(led) == 1
