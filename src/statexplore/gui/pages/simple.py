"""Dashboard, Settings, and not-yet-implemented section pages."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget,
)

from ...core.project import source_status
from ...rbridge.locate import PHASE1_PACKAGES, find_rscript, r_library
from ..state import AppState


def _title(text: str) -> QLabel:
    t = QLabel(text)
    t.setObjectName("pageTitle")
    return t


class DashboardPage(QWidget):
    def __init__(self, state: AppState, import_cb, open_cb):
        super().__init__()
        self.state = state
        lay = QVBoxLayout(self)
        lay.addWidget(_title("StatExplore"))
        lay.addWidget(QLabel("Exploratory statistical model discovery & teaching studio. "
                             "All data and analyses stay on this computer."))
        row = QHBoxLayout()
        b1 = QPushButton("Import dataset…")
        b1.clicked.connect(import_cb)
        b2 = QPushButton("Open project…")
        b2.clicked.connect(open_cb)
        row.addWidget(b1)
        row.addWidget(b2)
        row.addStretch()
        lay.addLayout(row)
        self.info = QLabel()
        self.info.setTextFormat(Qt.RichText)
        self.info.setWordWrap(True)
        self.info.setAlignment(Qt.AlignTop)
        lay.addWidget(self.info, 1)
        state.project_changed.connect(self.refresh)
        state.metadata_changed.connect(self.refresh)
        state.result_added.connect(lambda _r: self.refresh())
        self.refresh()

    def refresh(self) -> None:
        p = self.state.project
        if p is None:
            self.info.setText("<p>No project open. Import a CSV, SPSS (.sav) or Excel (.xlsx) file to begin "
                              "— you can also drag a file onto this window.</p>")
            return
        ds = p.dataset
        st = source_status(ds)
        src = ("original file unchanged" if st["unchanged"] else
               "original file has changed since import" if st["exists"] else "original file not found "
               "(the project keeps its own copy of the data)")
        history = "".join(f"<li>{r.created_at} — {r.title} [{r.status.value}]</li>" for r in p.results[-10:][::-1])
        self.info.setText(
            f"<h3>Current project</h3><p>{p.path or '(unsaved)'}</p>"
            f"<p><b>Data:</b> {ds.source.path}<br>{ds.n_rows:,} rows × {ds.n_cols} columns · {src}</p>"
            f"<p><b>Variable changes logged:</b> {len(ds.changes)} · <b>Analyses run:</b> {len(p.results)}</p>"
            f"<h3>Recent analyses</h3><ul>{history or '<li>none yet</li>'}</ul>"
        )


class SettingsPage(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        lay = QVBoxLayout(self)
        lay.addWidget(_title("Settings"))
        form = QFormLayout()
        self.rscript = QLineEdit(state.rscript_setting())
        self.rscript.setPlaceholderText(str(find_rscript() or "not found — install R for Windows"))
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.addWidget(self.rscript)
        row.addWidget(browse)
        form.addRow("Rscript.exe", row)
        form.addRow("R package library", QLabel(str(r_library())))
        lay.addLayout(form)
        btns = QHBoxLayout()
        save = QPushButton("Save R path")
        save.clicked.connect(self._save)
        test = QPushButton("Test R connection")
        test.clicked.connect(self._test)
        btns.addWidget(save)
        btns.addWidget(test)
        btns.addStretch()
        lay.addLayout(btns)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.status)
        lay.addStretch()

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Locate Rscript.exe", "", "Rscript (Rscript.exe)")
        if path:
            self.rscript.setText(path)

    def _save(self) -> None:
        self.state.set_rscript_setting(self.rscript.text().strip())
        self.status.setText("Saved. The R worker will restart on next use.")

    def _test(self) -> None:
        def check(session):
            return session.worker.request("versions", {"packages": PHASE1_PACKAGES}).result

        def done(info) -> None:
            pk = info.get("packages", {})
            missing = [k for k, v in pk.items() if v is None]
            lines = [info.get("r_version", "")] + [f"{k}: {v or 'NOT INSTALLED'}" for k, v in pk.items()]
            if missing:
                lines.append("Missing packages — run: Rscript r/install_packages.R")
            self.status.setText("\n".join(lines))

        self.status.setText("Starting R…")
        self.state.run_r(check, done, lambda m: self.status.setText(f"R error: {m}"))


class PlannedPage(QWidget):
    """Section not yet implemented: disabled with an explanation (§26), never a fake UI."""

    def __init__(self, title: str, phase: int, description: str):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.addWidget(_title(title))
        msg = QLabel(f"<p><b>Not yet available — planned for Phase {phase}.</b></p><p>{description}</p>")
        msg.setWordWrap(True)
        msg.setTextFormat(Qt.RichText)
        lay.addWidget(msg)
        lay.addStretch()
