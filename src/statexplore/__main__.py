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
    say("OK" if ok else f"FAIL: {res.messages}")
    return 0 if ok else 1


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
