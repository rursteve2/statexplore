"""Entry point: `python -m statexplore [file.sxp | dataset]`."""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def main() -> int:
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
