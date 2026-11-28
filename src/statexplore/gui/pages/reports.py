"""Reports (§19): choose results and searches, formats, and a folder; export."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMessageBox,
    QPlainTextEdit, QPushButton, QVBoxLayout, QWidget,
)

from ...export.exporters import FORMATS, export
from ..state import AppState


class ReportsPage(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        lay = QVBoxLayout(self)
        t = QLabel("Reports")
        t.setObjectName("pageTitle")
        lay.addWidget(t)
        intro = QLabel("Every export goes to a NEW folder inside the folder you choose (nothing there is "
                       "overwritten). R format: one self-checking reproducibility bundle per analysis, written "
                       "only when the data that analysis read can be proven; the analysis data are included only "
                       "if you agree for this export. Python scripts re-run the R script and load its result.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        row = QHBoxLayout()
        rbox = QGroupBox("Analyses")
        rl = QVBoxLayout(rbox)
        self.results = QListWidget()
        rl.addWidget(self.results)
        sel = QHBoxLayout()
        for text, on in (("Select all", True), ("Select none", False)):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, v=on: self._check_all(self.results, v))
            sel.addWidget(b)
        sel.addStretch()
        rl.addLayout(sel)
        row.addWidget(rbox, 2)
        sbox = QGroupBox("Discovery searches (all attempts + settings)")
        sl = QVBoxLayout(sbox)
        self.searches = QListWidget()
        sl.addWidget(self.searches)
        row.addWidget(sbox, 1)
        lay.addLayout(row, 1)
        fbox = QGroupBox("Formats")
        fl = QHBoxLayout(fbox)
        self.formats: dict[str, QCheckBox] = {}
        for key, label in FORMATS.items():
            cb = QCheckBox(label)
            cb.setChecked(key in ("xlsx", "R"))
            self.formats[key] = cb
            fl.addWidget(cb)
        self.history = QCheckBox("Analysis history + variable change log")
        self.history.setChecked(True)
        fl.addWidget(self.history)
        lay.addWidget(fbox)
        go = QPushButton("Export to folder…")
        go.clicked.connect(self._export)
        lay.addWidget(go)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(160)
        lay.addWidget(self.log)
        state.project_changed.connect(self._reload)
        state.result_added.connect(lambda _r: self._reload())
        self._reload()

    @staticmethod
    def _check_all(lst: QListWidget, on: bool) -> None:
        for i in range(lst.count()):
            lst.item(i).setCheckState(Qt.Checked if on else Qt.Unchecked)

    @staticmethod
    def _checked(lst: QListWidget) -> list[str]:
        return [lst.item(i).data(Qt.UserRole) for i in range(lst.count()) if lst.item(i).checkState() == Qt.Checked]

    def _reload(self) -> None:
        keep_r, keep_s = set(self._checked(self.results)), set(self._checked(self.searches))
        self.results.clear()
        self.searches.clear()
        p = self.state.project
        if p is None:
            return
        for r in reversed(p.results):
            it = QListWidgetItem(f"{r.created_at} — {r.title} [{r.status.value}]")
            it.setData(Qt.UserRole, r.id)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if r.id in keep_r else Qt.Unchecked)
            self.results.addItem(it)
        for run in reversed(p.searches):
            it = QListWidgetItem(f"{run.created_at} — {run.total:,} candidates [{run.status}]")
            it.setData(Qt.UserRole, run.id)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if run.id in keep_s else Qt.Unchecked)
            self.searches.addItem(it)

    def _export(self) -> None:
        p = self.state.project
        if p is None:
            return
        rids, sids = self._checked(self.results), self._checked(self.searches)
        fmts = {k for k, cb in self.formats.items() if cb.isChecked()}
        if not rids and not sids and not self.history.isChecked():
            QMessageBox.information(self, "Reports", "Choose at least one analysis or search.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Export folder")
        if not folder:
            return
        include_data = False
        if fmts & {"R", "py"} and rids:
            # AUD135-03: explicit, per-export consent for the analysis data (default: not included)
            ans = QMessageBox.question(self, "Include the data?",
                              "Include the analysis data in the reproducibility bundles (only the columns each "
                              "analysis uses)?\n\nYes: each bundle runs on its own. No (default): scripts, manifests "
                              "and README only; the data file must be placed next to a script later and is checked by "
                              "its SHA-256. Results and metadata are exported either way and are not anonymized.",
                              QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.No)
            if ans == QMessageBox.Cancel:
                return
            include_data = ans == QMessageBox.Yes
        try:
            written = export(p, rids, Path(folder), fmts, sids, self.history.isChecked(), include_data=include_data)
        except Exception as e:  # surfaced to the user
            QMessageBox.warning(self, "Export failed", str(e))
            return
        target = written[-1].parent if written else Path(folder)
        refused = [w for w in written if w.name == "REPRODUCTION_UNAVAILABLE.txt"]
        self.log.setPlainText("\n".join(str(w) for w in written))
        QMessageBox.information(self, "Reports", f"Wrote {len(written)} file(s) to the new folder\n{target}"
                                + (f"\n\n{len(refused)} analysis(es) got no runnable reproduction: the data they analysed "
                                   "cannot be proven (see REPRODUCTION_UNAVAILABLE.txt in their folders)." if refused else ""))
        self.last_export = target
