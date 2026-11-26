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

# AMD workflow A: TIME x GROUP focal, cluster SCHOOL, correlated slopes + AR(1), ML
s = SearchSettings(outcome="Y", candidates=["GROUP"], family="gaussian", participant="ID", time="TIME",
                   cluster="SCHOOL", structure="rs_ar1", strategy="time", workers=1, max_models=None, max_minutes=None,
                   study="longitudinal")
run = engine.create_run(ds, s)
engine.execute(run, ds)
row = next(r for r in run.rows if any("GROUP" in t and "TIME" in t for t in [list(x) for x in r.get("terms", [])])
           ) if any(len(r.get("terms", [])) > 2 for r in run.rows) else run.rows[0]
out["amd_rows"] = [{k: r.get(k) for k in ("idx", "formula", "n", "loglik", "aic", "bic", "k", "structure")} for r in run.rows]
out["amd_tests"] = [[{k: t.get(k) for k in ("type", "label", "statistic", "df", "p")} for t in r.get("tests", [])]
                    for r in run.rows]
page = w.pages["Mixed & Longitudinal Models"]
results = []
orig_done = page._done


def done(o):
    results.append(o[0])
    orig_done(o)


page._done = done


def fit_and_wait(trigger):
    n = len(results)
    trigger()
    wait(lambda: len(results) > n)
    w.state.wait_idle()
    return results[-1]


spec = mixed.spec_from_discovery(run, row)
step("44A transfer", lambda: {**summ(fit_and_wait(lambda: w.state.show_in_mixed.emit(spec))),
                              "ui_interactions": page.interactions.currentText(),
                              "ui_predictors": page.pred_sel.selected()})
step("44C-1 Fit unchanged", lambda: summ(fit_and_wait(page._fit)))
step("44C-2 Fit again", lambda: summ(fit_and_wait(page._fit)))


def plot_then_fit():
    page.ntraj.setValue(20)
    return summ(fit_and_wait(page._fit))


step("44C-2b trajectories 40->20 then Fit", plot_then_fit)


def reml():
    page.method.setCurrentText("REML")
    return summ(fit_and_wait(page._fit))


step("44C-4 ML->REML Fit", reml)

# 48C: time comparison from an AR(1) REML model on irregular time (46A fixture)
irr = import_file(REF / "Test46A_IrregularTime.csv")
base = mixed.MixedSpec(outcome="Y", id="ID", time="TIME", predictors=["GROUP"], cluster="SCHOOL", random="intercept",
                       residual="ar1", interactions="time", method="REML")
from statexplore.analysis.session import RSession  # noqa: E402
from statexplore.rbridge.worker import RWorker  # noqa: E402

rw = RWorker()
rw.start()
sess = RSession(rw, Path(tempfile.mkdtemp(prefix="sx_repro_", dir="E:/sx_tmp")))


def cmp_time():
    res, _ = mixed.compare_specs(sess, irr, base, "time")
    t = next(t for t in res.tables if t.name.startswith("Specification"))
    return [dict(zip(t.columns, r)) for r in t.rows]


step("48C compare time (API, AR1 REML base)", cmp_time)


def cmp_time_gui():
    w.state.set_project(Project(irr))
    page.outcome.setCurrentIndex(page.outcome.findData("Y"))
    page.idv.setCurrentIndex(page.idv.findData("ID"))
    page.time.setCurrentIndex(page.time.findData("TIME"))
    page.cluster.setCurrentIndex(page.cluster.findData("SCHOOL"))
    page.pred_sel.set_selected(["GROUP"])
    page.interactions.setCurrentIndex(page.interactions.findData("time"))
    page.random.setCurrentIndex(page.random.findData("intercept"))
    page.residual.setCurrentIndex(page.residual.findData("ar1"))
    page.method.setCurrentText("REML")
    res = fit_and_wait(lambda: page._compare("time"))
    t = next(t for t in res.tables if t.name.startswith("Specification"))
    return [dict(zip(t.columns, r)) for r in t.rows]


step("48C compare time (GUI)", cmp_time_gui)

# 45A: reused IDs within schools, preflight
reused = import_file(REF / "Test45A_ReusedIDs.csv")
from statexplore.analysis.longitudinal_preflight import preflight  # noqa: E402


def pre45():
    sp = mixed.MixedSpec(outcome="Y", id="ID", time="TIME", predictors=["GROUP"], cluster="SCHOOL", id_policy="local")
    r = preflight(reused, "Y", "ID", "TIME", ["GROUP"], sp)
    return {"participants": r.n_participants, "duplicates": r.duplicates, "usable": r.n_usable,
            "clusters": r.n_clusters, "warnings": r.warnings}


step("45A preflight local IDs", pre45)
rw.stop()
w.state.dirty = False
w.state.shutdown_r()
Path(sys.argv[2]).write_text(json.dumps(out, indent=1, default=str))
