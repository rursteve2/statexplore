"""Entry point: `python -m statexplore [file.sxp | dataset]`."""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def selftest() -> int:
    """Headless check used after packaging: R is found, the worker runs, psych works."""
    import tempfile

    import pandas as pd

    from .analysis import eda
    from .analysis.session import RSession
    from .core.importers import import_file
    from .rbridge.locate import find_rscript, r_library
    from .rbridge.worker import RWorker

    lines: list[str] = []

    def say(text: str) -> None:  # the windowed (frozen) build has no console: also write a report file
        lines.append(text)
        print(text)
        Path(tempfile.gettempdir(), "statexplore_selftest.txt").write_text("\n".join(lines), encoding="utf-8")

    rscript = find_rscript()
    say(f"Rscript: {rscript}\nR library: {r_library()}")
    if rscript is None:
        say("FAIL: R not found")
        return 2
    tmp = Path(tempfile.mkdtemp())
    pd.DataFrame({"x": [1.0, 2, 3, 4, 5], "y": [2.0, 1, 4, 3, 6]}).to_csv(tmp / "d.csv", index=False)
    ds = import_file(tmp / "d.csv")
    with RWorker(rscript) as w:
        res = eda.describe(RSession(w, tmp / "s"), ds, ["x", "y"])
    mean = res.tables[0].rows[0][1] if res.tables else None
    say(f"engine: {res.engine.version}; packages: {res.engine.packages}; mean(x) = {mean}")
    ok = res.status.value == "ok" and mean == 3.0
    if ok and "--full" in sys.argv:
        ok = _full_selftest(say, rscript, tmp)
    say("OK" if ok else f"FAIL: {res.messages}")
    return 0 if ok else 1


def _full_selftest(say, rscript, tmp: Path) -> bool:
    """One real analysis per workspace + exports, to prove every bundled R package loads."""
    import os

    import numpy as np
    import pandas as pd

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from .analysis import comparison, mixed, teaching, validation
    from .analysis.builders import BY_KEY
    from .analysis.framework import run_analysis
    from .analysis.models import ModelSpec
    from .analysis.session import RSession
    from .core.importers import import_file
    from .core.project import Project
    from .discovery.engine import create_run, execute
    from .discovery.space import SearchSettings
    from .export.exporters import export
    from .rbridge.worker import RWorker

    app = QApplication.instance() or QApplication([])  # needed for PDF export
    assert app is not None
    rng = np.random.default_rng(1)
    n = 120
    df = pd.DataFrame({"id": np.repeat(np.arange(1, 41), 3), "t": np.tile([0, 1, 2], 40),
                       "x": rng.normal(size=n), "g": rng.choice(["a", "b"], n)})
    df["y"] = df["x"] * (df["g"] == "b") + 0.3 * df["t"] + rng.normal(size=n)
    df["yb"] = (df["y"] > df["y"].median()).astype(int)
    df.to_csv(tmp / "full.csv", index=False)
    ds = import_file(tmp / "full.csv")
    proj = Project(ds)
    ok = True
    with RWorker(rscript) as w:
        sess = RSession(w, tmp / "full")
        checks = [
            ("linear regression", lambda: run_analysis(sess, ds, BY_KEY["lm"], {"dependent": ["y"], "predictors": ["x", "g"]},
                                                       {"interactions": "two_way"})),
            ("logistic regression", lambda: run_analysis(sess, ds, BY_KEY["logistic"], {"dependent": ["yb"], "predictors": ["x"]}, {})),
            ("factorial ANOVA", lambda: run_analysis(sess, ds, BY_KEY["anova_oneway"], {"dependent": ["y"], "factors": ["g"]}, {})),
            ("mediation", lambda: run_analysis(sess, ds, BY_KEY["mediation"], {"x": ["x"], "mediator": ["t"], "dependent": ["y"]},
                                               {"boot": 200})),
            ("Teaching Studio", lambda: teaching.run(sess, ds, ModelSpec("y", [("x", "g")], "gaussian"),
                                                     teaching.PlotSpec(x="x", group="g"))),
            ("comparison", lambda: comparison.compare(sess, ds, [ModelSpec("y", [("x",)], "gaussian"),
                                                                ModelSpec("y", [("x", "g")], "gaussian")])),
            ("mixed (lme4)", lambda: mixed.run(sess, ds, mixed.MixedSpec(outcome="y", id="id", time="t", predictors=["g"]))),
            ("mixed (nlme AR1)", lambda: mixed.run(sess, ds, mixed.MixedSpec(outcome="y", id="id", time="t", residual="ar1"))),
            ("mixed (mmrm)", lambda: mixed.run(sess, ds, mixed.MixedSpec(outcome="y", id="id", time="t", time_type="categorical",
                                                                         random="none", residual="unstructured"))),
            ("cross-validation", lambda: validation.cross_validate(sess, ds, ModelSpec("y", [("x",)], "gaussian"), "kfold", 4)),
        ]
        for name, fn in checks:
            try:
                res, art = fn()
                good = res.status.value in ("ok", "warning")
                say(f"  {'ok ' if good else 'BAD'} {name}: {res.status.value} {res.messages[:1] if not good else ''}")
                ok &= good
                proj.add_result(res, art)
            except Exception as e:  # report and continue
                say(f"  BAD {name}: {e}")
                ok = False
    try:
        run = execute(create_run(ds, SearchSettings(outcome="y", candidates=["x", "g", "t"], workers=1,
                                                    max_models=None, max_minutes=None)), ds, rscript=str(rscript))
        say(f"  {'ok ' if run.status == 'complete' else 'BAD'} discovery: {run.status}")
        ok &= run.status == "complete"
        proj.searches.append(run)
        out = export(proj, [r.id for r in proj.results], tmp / "export", {"xlsx", "docx", "pdf", "R", "py"},
                     [run.id])
        say(f"  ok  exports: {len(out)} files")
        proj.save(tmp / "p.sxp")
        Project.open(tmp / "p.sxp")
        say("  ok  project save/reopen")
    except Exception as e:
        say(f"  BAD discovery/export/project: {e}")
        ok = False
    return ok


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()
    from PySide6.QtWidgets import QApplication

    from .gui.main_window import MainWindow

    log_dir = Path.home() / "AppData" / "Local" / "StatExplore"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_dir / "statexplore.log", level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app = QApplication(sys.argv)
    app.setApplicationName("StatExplore")
    win = MainWindow()
    win.show()
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg.lower().endswith(".sxp"):
            win.open_project(arg)
        else:
            win.import_path(arg)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
