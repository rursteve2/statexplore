"""Gate 0: reproduce the v1.3.3 P0/P1 register items through the REAL application (offscreen Qt + R worker).
Usage: python repro_p0.py <package dir> <out json>.  PYTHONPATH selects the code under test (baseline or current)."""
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])

import statexplore  # noqa: E402
from statexplore.core.importers import import_file  # noqa: E402
from statexplore.core.project import Project  # noqa: E402
from statexplore.analysis import mixed  # noqa: E402
from statexplore.gui.main_window import MainWindow  # noqa: E402
from statexplore.gui.state import AppState  # noqa: E402

pkg = Path(sys.argv[1])
out: dict = {"version": statexplore.__version__, "code": statexplore.__file__}
F43 = pkg / "fixtures" / "Test43B.csv"
REF = pkg / "reference" / "output"


def wait(cond, ms=600000):
    import time

    t0 = time.time()
    while not cond():
        app.processEvents()
        time.sleep(0.02)
        if (time.time() - t0) * 1000 > ms:
            raise TimeoutError


def fit_row(res):
    for t in res.tables:
        if t.name.startswith("Model fit") or t.name.startswith("Model summary"):
            return dict(zip(t.columns, t.rows[0]))
    return {}


def summ(res):
    spec = res.values.get("model_spec") or {}
    return {"status": str(res.status), "fixed_terms": spec.get("fixed_terms"), "estimation": spec.get("estimation"),
            "engine": spec.get("engine"), "residual": spec.get("residual"), "random": spec.get("random"),
            "identity": res.values.get("identity_check"), "fit": {k: v for k, v in fit_row(res).items()},
            "refit_msgs": [m for m in res.messages if m.startswith("Refit")][:2]}


def step(key, fn):
    try:
        out[key] = fn()
    except Exception as e:
        out[key] = {"exception": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-1500:]}
    print(key, json.dumps(out[key], default=str)[:900], flush=True)


from PySide6.QtCore import QSettings  # noqa: E402

QSettings.setDefaultFormat(QSettings.IniFormat)
QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, tempfile.mkdtemp(dir="E:/sx_tmp"))
w = MainWindow(AppState())
ds = import_file(F43)
w.state.set_project(Project(ds))
ds = w.state.dataset
from statexplore.discovery import engine  # noqa: E402
from statexplore.discovery.space import SearchSettings  # noqa: E402

from statexplore.discovery import engine
from statexplore.discovery.space import SearchSettings

irr = import_file(pkg / "reference" / "output" / "Test46A_IrregularTime.csv")
w.state.set_project(Project(irr))
irr = w.state.dataset
for wf, st, extra_kw in (("structure", ["ar1"], dict(time_interactions=True)), ("fixed", None, dict(structure="ar1", strategy="time"))):
    kw = dict(outcome="Y", candidates=["GROUP"], family="gaussian", participant="ID", time="TIME", cluster="SCHOOL",
              workers=1, max_models=None, max_minutes=None, study="longitudinal", workflow=wf, **extra_kw)
    if st: kw["structures"] = st
    run = engine.create_run(irr, SearchSettings(**kw))
    engine.execute(run, irr)
    row = max(run.rows, key=lambda r: len(r.get("terms", [])))
    page = w.pages["Mixed & Longitudinal Models"]
    results = []
    od = page._done
    def done(o, od=od):
        results.append(o[0]); od(o)
    page._done = done
    def go(trigger):
        n = len(results); trigger(); wait(lambda: len(results) > n); w.state.wait_idle(); return results[-1]
    spec = mixed.spec_from_discovery(run, row)
    r1 = go(lambda: w.state.show_in_mixed.emit(spec))
    step(f"{wf} transfer", lambda: {**summ(r1), "ui_res": page.residual.currentData(), "ui_method": page.method.currentText(), "ui_time_type": page.time_type.currentData()})
    r2 = go(lambda: page._compare("time"))
    t = next(t for t in r2.tables if t.name.startswith("Specification"))
    step(f"{wf} compare time", lambda: [{k: v for k, v in zip(t.columns, r) if k in ("model","engine","estimation","parameters","logLik")} for r in t.rows])
w.state.dirty = False
w.state.shutdown_r()
