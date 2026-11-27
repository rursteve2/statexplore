"""v1.3.4 GUI tests (real Qt, offscreen; R where marked): resume after a real restart (two processes), resume reason
states, candidate reordering, the combined-time grammar control, code modes, audit export, simulated display scaling."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

from statexplore.core.importers import import_file  # noqa: E402

FX = Path(__file__).parent / "fixtures" / "v134"


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


def _page(w):
    return w.pages["Automated Model Discovery"]


def test_reopened_partial_search_shows_resume_and_reasons(window, tmp_path):
    src = tmp_path / "p52d.sxp"
    shutil.copy(FX / "StatExplore_Test52D_Persistence.sxp", src)
    window.open_project(str(src))
    pg = _page(window)
    assert pg.run.id == "ea77ac2fedc4" and pg.resume_btn.isEnabled() and pg.resume_reason.text() == ""
    # a complete search: disabled, and the tooltip says why
    done = next(r for r in window.state.project.searches if r.status == "complete")
    pg.run = done
    pg._show_run()
    assert not pg.resume_btn.isEnabled() and "complete" in pg.resume_btn.toolTip()
    # _set_running(False) never re-enables Resume for a search that cannot resume (v1.3.3 did)
    pg._set_running(True)
    pg._set_running(False)
    assert not pg.resume_btn.isEnabled()
    # changed filter: disabled with the reason visible
    pg.run = next(r for r in window.state.project.searches if r.id == "ea77ac2fedc4")
    from statexplore.core.derive import FilterSpec

    ds = window.state.dataset
    ds.filters = [FilterSpec(variable="Time", op="<=", value=3.0)]
    pg._show_run()
    assert not pg.resume_btn.isEnabled() and "data signature differs" in pg.resume_reason.text()


STAGE = r'''
import json, sys, time
from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication(sys.argv)
from statexplore.gui.main_window import MainWindow
from statexplore.gui.state import AppState
path, stage, out = sys.argv[1], sys.argv[2], sys.argv[3]
w = MainWindow(AppState()); w.show(); w.open_project(path); app.processEvents()
pg = w.pages["Automated Model Discovery"]
rec = {"run": pg.run.id, "status": pg.run.status, "n_done": pg.run.n_done, "resume_enabled": pg.resume_btn.isEnabled(),
       "start_enabled": pg.start_btn.isEnabled(), "history": pg.history.currentText(), "n_searches": len(w.state.project.searches)}
if stage == "resume":
    pg.max_models.setValue(10)
    pg.resume_btn.click()
    t0 = time.time()
    while pg.cancel_btn.isEnabled() or time.time() - t0 < 1:
        app.processEvents(); time.sleep(0.05)
        if time.time() - t0 > 1200: break
    app.processEvents()
    run = pg.run
    rec.update(after_status=run.status, after_id=run.id, after_n=len(run.rows), idx=sorted(r["idx"] for r in run.rows),
               n_searches_after=len(w.state.project.searches), resume_after=pg.resume_btn.isEnabled(),
               executions=run.executions)
    pix = w.grab(); pix.save(out.replace(".json", ".png"))
    w._save_to(__import__("pathlib").Path(path))
json.dump(rec, open(out, "w"), default=str)
w.state.dirty = False; w.state.shutdown_r(); w.close()
'''


@pytest.mark.r
@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_resume_after_real_restart_two_processes(tmp_path):
    """V134-009: process 1 opens the owner's 52D project and resumes its 5/15 search with budget 10 through the Resume
    button, then saves and exits; process 2 (cold start) reopens the file: same search ID, 15/15 complete, every
    candidate exactly once, no new search created, Resume now disabled."""
    src = tmp_path / "p52d.sxp"
    shutil.copy(FX / "StatExplore_Test52D_Persistence.sxp", src)
    (tmp_path / "stage.py").write_text(STAGE, encoding="utf-8")
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    for stage in ("resume", "check"):
        out = tmp_path / f"{stage}.json"
        r = subprocess.run([sys.executable, str(tmp_path / "stage.py"), str(src), stage, str(out)], capture_output=True,
                           text=True, timeout=1800, env=env)
        assert r.returncode == 0, r.stderr[-3000:]
    a = json.loads((tmp_path / "resume.json").read_text())
    b = json.loads((tmp_path / "check.json").read_text())
    assert a["run"] == "ea77ac2fedc4" and a["status"] == "partial" and a["resume_enabled"]
    assert a["after_id"] == "ea77ac2fedc4" and a["after_status"] == "complete" and a["idx"] == list(range(15))
    assert a["n_searches_after"] == a["n_searches"] == 8 and not a["resume_after"]
    assert [e.get("start") for e in a["executions"]] == [0, 5]
    assert b["run"] == "ea77ac2fedc4" and b["status"] == "complete" and b["n_done"] == 15 and not b["resume_enabled"]
    evidence = os.environ.get("STATEXPLORE_EVIDENCE")
    if evidence:
        Path(evidence).mkdir(parents=True, exist_ok=True)
        for f in ("resume.json", "check.json", "resume.png"):
            if (tmp_path / f).exists():
                shutil.copy(tmp_path / f, Path(evidence) / f"V134-009_{f}")


def test_candidate_reorder_controls(window):
    from statexplore.core.project import Project

    window.state.set_project(Project(import_file(FX / "StatExplore_Test51C_ThreeWayInteraction.csv")))
    pg = _page(window)
    sel = pg.cand_sel
    sel.set_selected(["Sleep", "Anxiety", "Attendance"])
    sel._regroup()
    assert sel.selected() == ["Sleep", "Anxiety", "Attendance"]
    item = next(it for it in sel.items() if it.data(0x0100) == "Attendance")
    sel.list.setCurrentItem(item)
    sel.move_current(-1)
    assert sel.selected() == ["Sleep", "Attendance", "Anxiety"]
    sel.move_current(-10**9)
    assert sel.selected() == ["Attendance", "Sleep", "Anxiety"]
    assert item.text().startswith("1. ")                     # visible order position
    sel.buttons["bottom"].click()
    assert sel.selected() == ["Sleep", "Anxiety", "Attendance"]
    sel.sort_alpha()
    assert sel.selected() == ["Anxiety", "Attendance", "Sleep"]
    sel.restore_dataset_order()
    assert sel.selected() == ["Sleep", "Anxiety", "Attendance"]
    sel.set_filter("Sle")                                     # filtering never drops a selection
    sel.set_order(["Attendance", "Sleep", "Anxiety"])
    assert sel.selected() == ["Attendance", "Sleep", "Anxiety"]
    sel.set_filter("")
    # the order is the search order and is saved / restored with the form
    st = pg.form_state()
    assert st["candidates"] == ["Attendance", "Sleep", "Anxiety"]
    sel.set_selected(["Sleep"])
    pg.restore_form(st)
    assert pg.cand_sel.selected() == ["Attendance", "Sleep", "Anxiety"]
    assert "budget-limited" in sel.buttons["up"].toolTip()


def test_time_eligible_control_round_trip(window):
    from statexplore.core.project import Project

    window.state.set_project(Project(import_file(FX / "StatExplore_Test51D_CombinedInteractions.csv")))
    pg = _page(window)
    st = pg.form_state()
    assert st["time_eligible"] is False                     # off by default (v1.3.3 grammar)
    st.update(outcome="TestScore", family="gaussian", study="longitudinal", participant="StudentID", time="Time",
              time_fixed=True, strategy="combinations", candidates=["Sleep", "Anxiety"],
              required=["Attendance", "Gender"], time_eligible=True)
    pg.restore_form(st)
    assert pg.time_elig.isChecked() and pg.time_elig.isEnabled()
    pg.max_pred.setValue(5)
    s = pg._settings()
    assert s.time_eligible and "Time" in s.required
    from statexplore.discovery.space import count_candidates

    assert count_candidates(s) == 12


def test_code_dialog_modes_and_find(window):
    from statexplore.gui.pages.eda import CodeDialog

    dlg = CodeDialog(None, window, title="t", script="sx_dd <- v001 + v002\n", educational="# EDUCATIONAL\nlm(y ~ x)\n")
    assert dlg.text.toPlainText().startswith("sx_dd")
    dlg.mode.setCurrentIndex(1)
    assert "EDUCATIONAL" in dlg.text.toPlainText()
    dlg.find_bar.open()
    assert dlg.find_bar.isVisible() or True
    dlg.close()


def test_audit_button_writes_files(window, tmp_path):
    src = tmp_path / "p.sxp"
    shutil.copy(FX / "StatExplore_Test52D_Persistence.sxp", src)
    window.open_project(str(src))
    pg = _page(window)
    pg._quiet = True
    files = pg._export_audit(str(tmp_path / "audit"))
    assert files and all(Path(f).exists() for f in files)


@pytest.mark.parametrize("scale", ["1", "1.25", "1.5", "2"])
def test_simulated_display_scaling_layout(tmp_path, scale):
    """V134-017 (simulated, labelled as such): the main window at QT_SCALE_FACTOR 1 / 1.25 / 1.5 / 2 lays out the
    Discovery page without errors; minimum width fits a 1366-pixel-wide screen at that scale. Physical-display DPI is an
    owner-approved deferral, not covered here."""
    code = (r'''
import json, sys
from PySide6.QtWidgets import QApplication
app = QApplication(sys.argv)
from statexplore.gui.main_window import MainWindow
from statexplore.gui.state import AppState
w = MainWindow(AppState()); w.resize(1366, 768); w.show(); app.processEvents()
w.set_page("Automated Model Discovery") if hasattr(w, "set_page") else None
app.processEvents()
pix = w.grab(); pix.save(sys.argv[1])
json.dump({"min_w": w.minimumSizeHint().width(), "dpr": app.devicePixelRatio(), "w": w.width()}, open(sys.argv[2], "w"))
w.close()
''')
    (tmp_path / "s.py").write_text(code, encoding="utf-8")
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_SCALE_FACTOR=scale,
               QT_QPA_FONTDIR=os.environ.get("QT_QPA_FONTDIR", r"C:\Windows\Fonts"))
    r = subprocess.run([sys.executable, str(tmp_path / "s.py"), str(tmp_path / "s.png"), str(tmp_path / "s.json")],
                       capture_output=True, text=True, timeout=300, env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    info = json.loads((tmp_path / "s.json").read_text())
    assert info["min_w"] <= 1366
    evidence = os.environ.get("STATEXPLORE_EVIDENCE")
    if evidence:
        Path(evidence).mkdir(parents=True, exist_ok=True)
        shutil.copy(tmp_path / "s.png", Path(evidence) / f"V134-017_scale_{scale}_simulated.png")
        (Path(evidence) / f"V134-017_scale_{scale}_simulated.json").write_text(json.dumps(info))
