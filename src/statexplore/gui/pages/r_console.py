"""R Console (v1.1 §14): write and run your own R code on a copy of the dataset (`my_data`)."""
from __future__ import annotations

import html
import re
import threading
from pathlib import Path

from PySide6.QtCore import QObject, QRegularExpression, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPixmap, QSyntaxHighlighter, QTextCharFormat
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QDialog, QDialogButtonBox, QFileDialog, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QSplitter, QTableWidget,
    QTableWidgetItem, QTabWidget, QTextBrowser, QVBoxLayout, QWidget,
)

from ...core.schema import AnalysisResult, EngineInfo, Status
from ...rbridge import package_manager as PM
from ...rbridge.console import SECURITY_WARNING, ConsoleResult, RConsole
from ..state import AppState

R_KEYWORDS = ("if else repeat while function for in next break TRUE FALSE NULL Inf NaN NA NA_integer_ NA_real_ "
              "NA_character_ return library require").split()
STARTER = ("# my_data is a copy of the current dataset (variable names as shown in StatExplore).\n"
           "# Changes stay in R unless you choose 'Import resulting dataset'.\n"
           "str(my_data)\nsummary(my_data[, 1:min(5, ncol(my_data))])\n")


class RHighlighter(QSyntaxHighlighter):
    def __init__(self, doc):
        super().__init__(doc)

        def fmt(color: str, bold: bool = False, italic: bool = False) -> QTextCharFormat:
            f = QTextCharFormat()
            f.setForeground(QColor(color))
            if bold:
                f.setFontWeight(QFont.Bold)
            f.setFontItalic(italic)
            return f
        self.rules = [
            (QRegularExpression(r"\b(" + "|".join(R_KEYWORDS) + r")\b"), fmt("#7b1fa2", bold=True)),
            (QRegularExpression(r"\b[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?L?\b"), fmt("#1565c0")),
            (QRegularExpression(r"<-|<<-|->|\|>|%[^%]*%|==|!=|<=|>="), fmt("#c62828")),
            (QRegularExpression(r"\b[A-Za-z.][A-Za-z0-9._]*(?=\()"), fmt("#00695c")),
            (QRegularExpression(r"\"([^\"\\\\]|\\\\.)*\"|'([^'\\\\]|\\\\.)*'"), fmt("#2e7d32")),
            (QRegularExpression(r"#[^\n]*"), fmt("#757575", italic=True)),
        ]

    def highlightBlock(self, text: str) -> None:
        for rx, f in self.rules:
            it = rx.globalMatch(text)
            while it.hasNext():
                m = it.next()
                self.setFormat(m.capturedStart(), m.capturedLength(), f)


class _Bridge(QObject):
    done = Signal(object)
    failed = Signal(str)


class PackageManagerDialog(QDialog):
    """Installed packages (core / user), local archive installation and removal (§15)."""

    def __init__(self, state: AppState, parent: QWidget | None = None, on_change=None):
        super().__init__(parent)
        self.state = state
        self.on_change = on_change
        self.setWindowTitle("R Package Manager")
        self.resize(900, 640)
        lay = QVBoxLayout(self)
        info = QLabel(f"Your packages are installed in <b>{html.escape(str(PM.user_library()))}</b>, separate from "
                      "StatExplore's core library, which always takes precedence. Installation is offline: choose a "
                      "Windows binary package (.zip) or a source package (.tar.gz) on this computer.<br><i>"
                      + html.escape(PM.SECURITY_WARNING) + "</i>")
        info.setWordWrap(True)
        lay.addWidget(info)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search installed packages…")
        self.search.textChanged.connect(self._filter)
        lay.addWidget(self.search)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Package", "Version", "Library", "Depends / Imports", "Active"])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        lay.addWidget(self.table, 1)
        row = QHBoxLayout()
        for text, fn in (("Install from local file…", self._install), ("Remove selected (your library only)", self._remove),
                         ("Refresh", self.refresh), ("Installation record", self._manifest)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch()
        lay.addLayout(row)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(170)
        lay.addWidget(self.log)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.refresh()

    def refresh(self) -> None:
        try:
            rows = PM.inventory(self.state.rscript_setting() or None)
        except Exception as e:  # noqa: BLE001
            self.log.setPlainText(f"Could not list packages: {e}")
            return
        self.table.setRowCount(len(rows))
        for i, r in enumerate(sorted(rows, key=lambda r: (r["library"] != "user", r["package"].lower()))):
            for j, val in enumerate((r["package"], r["version"], {"core": "StatExplore core", "user": "Your library",
                                                                  "R": "R base"}[r["library"]],
                                     r.get("depends") or "", "yes" if r["active"] else "masked")):
                self.table.setItem(i, j, QTableWidgetItem(str(val)))
        self.table.resizeColumnsToContents()
        self._filter(self.search.text())

    def _filter(self, text: str) -> None:
        t = text.lower()
        for i in range(self.table.rowCount()):
            self.table.setRowHidden(i, bool(t) and t not in self.table.item(i, 0).text().lower())

    def _install(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Install R package", "", "R packages (*.zip *.tar.gz *.tgz)")
        if not path:
            return
        self.install_path(path)

    def install_path(self, path: str, confirm: bool = True) -> bool:
        rs = self.state.rscript_setting() or None
        try:
            rep = PM.check_archive(path, rs)
        except PM.PackageError as e:
            QMessageBox.warning(self, "Package", str(e))
            return False
        i = rep.info
        summary = [f"{i.package} {i.version} ({i.kind}" + (f", built for R {i.built_r}" if i.built_r else "") + ")"]
        summary += [f"Problem: {p}" for p in rep.problems] + [f"Note: {n}" for n in rep.notes]
        self.log.setPlainText("\n".join(summary))
        if not rep.ok:
            QMessageBox.warning(self, "Package cannot be installed", "\n\n".join(rep.problems))
            return False
        if confirm and QMessageBox.question(self, "Install package", "\n".join(summary) + "\n\n" + PM.SECURITY_WARNING
                                            + "\n\nInstall?") != QMessageBox.Yes:
            return False
        try:
            _rep, log = PM.install(path, rs)
        except PM.PackageError as e:
            self.log.setPlainText(str(e))
            QMessageBox.warning(self, "Installation failed", str(e)[:2000])
            return False
        self.log.setPlainText(log[-6000:])
        self.refresh()
        if self.on_change:
            self.on_change()
        return True

    def _remove(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        name = self.table.item(rows[0].row(), 0).text()
        if QMessageBox.question(self, "Remove package", f"Remove {name} from your library?") != QMessageBox.Yes:
            return
        try:
            self.log.setPlainText(PM.remove(name, self.state.rscript_setting() or None))
        except PM.PackageError as e:
            QMessageBox.warning(self, "Package", str(e))
            return
        self.refresh()
        if self.on_change:
            self.on_change()

    def _manifest(self) -> None:
        entries = PM.manifest()
        self.log.setPlainText("\n".join(
            f"{e['package']} {e['version']} — {e['source_archive']} (sha256 {e['sha256'][:12]}…) installed "
            f"{e['installed_at']} into {e['library']}; depends: {', '.join(e['depends']) or 'none'}"
            for e in entries) or "No packages installed through StatExplore yet.")


class RConsolePage(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.bridge = _Bridge()
        self.bridge.done.connect(self._done)
        self.bridge.failed.connect(self._failed)
        self.console: RConsole | None = None
        self.last: ConsoleResult | None = None
        self._busy = False
        self._file: Path | None = None
        lay = QVBoxLayout(self)
        t = QLabel("R Console")
        t.setObjectName("pageTitle")
        lay.addWidget(t)
        warn = QLabel("⚠ " + html.escape(SECURITY_WARNING))
        warn.setTextFormat(Qt.RichText)
        warn.setWordWrap(True)
        warn.setStyleSheet("background:#fff3e0;border:1px solid #e0a060;padding:6px;")
        lay.addWidget(warn)
        bar = QHBoxLayout()
        self.run_btn = QPushButton("▶ Run Script")
        self.run_btn.clicked.connect(self.run_script)
        self.stop_btn = QPushButton("■ Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop)
        self.persistent = QCheckBox("Persistent session")
        self.persistent.setToolTip("Off (default): every run starts a clean R session with a fresh my_data. "
                                   "On: objects stay between runs until you restart or clear the session.")
        self.persistent.toggled.connect(self._mode_changed)
        bar.addWidget(self.run_btn)
        bar.addWidget(self.stop_btn)
        bar.addWidget(self.persistent)
        for text, fn in (("Restart session", self.restart), ("Clear workspace", self.clear_workspace),
                         ("Objects", self.show_objects), ("Loaded packages", self.show_packages),
                         ("Import resulting dataset…", self.import_dataset), ("Package Manager…", self.package_manager)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            bar.addWidget(b)
        bar.addStretch()
        lay.addLayout(bar)
        ebar = QHBoxLayout()
        for text, fn in (("Open .R…", self.open_file), ("Save .R…", self.save_file), ("Copy code", self.copy_code),
                         ("Clear editor", lambda: self.editor.clear()), ("Clear output", self.clear_output)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            ebar.addWidget(b)
        ebar.addStretch()
        self.status = QLabel()
        ebar.addWidget(self.status)
        lay.addLayout(ebar)
        self.editor = QPlainTextEdit(STARTER)
        self.editor.setStyleSheet("font-family: Consolas, monospace; font-size: 12px;")
        self.editor.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.highlighter = RHighlighter(self.editor.document())
        self.output = QTextBrowser()
        self.output.setStyleSheet("font-family: Consolas, monospace; font-size: 12px;")
        self.plots_host = QWidget()
        self.plots_layout = QVBoxLayout(self.plots_host)
        ps = QScrollArea()
        ps.setWidget(self.plots_host)
        ps.setWidgetResizable(True)
        self.out_tabs = QTabWidget()
        self.out_tabs.addTab(self.output, "Console output")
        self.out_tabs.addTab(ps, "Plots")
        split = QSplitter(Qt.Vertical)
        split.addWidget(self.editor)
        split.addWidget(self.out_tabs)
        split.setSizes([300, 420])
        lay.addWidget(split, 1)
        state.project_changed.connect(self._project_changed)
        state.metadata_changed.connect(self._project_changed)
        self._project_changed()

    # ---- session -------------------------------------------------------------------------------
    def _ensure(self) -> RConsole:
        if self.console is None:
            self.console = RConsole(self.state._workdir / "console", self.state.rscript_setting() or None,
                                    persistent=self.persistent.isChecked())
            self.console.set_dataset(self.state.dataset)
        return self.console

    def _project_changed(self) -> None:
        """A new project or changed data/metadata ends any persistent session (no stale objects)."""
        if self.console is not None:
            self.console.close()
            self.console = None
        self.status.setText("Session: clean per run" if not self.persistent.isChecked() else "Session: persistent (new)")

    def _mode_changed(self, on: bool) -> None:
        self._project_changed()

    def run_script(self) -> None:
        if self._busy:
            return
        code = self.editor.toPlainText()
        if not code.strip():
            return
        c = self._ensure()
        self._busy = True
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status.setText("Running…")
        self._code = code

        def work():
            try:
                self.bridge.done.emit(c.run(code))
            except Exception as e:  # noqa: BLE001 - shown to the user
                self.bridge.failed.emit(str(e))
        threading.Thread(target=work, daemon=True).start()

    def stop(self) -> None:
        if self.console is not None:
            self.console.cancel()

    def _finish(self) -> None:
        self._busy = False
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _done(self, res: ConsoleResult) -> None:
        self._finish()
        self.last = res
        self.status.setText(f"Done in {res.elapsed:.1f} s · {res.mode} session"
                            + (" · stopped at an error" if res.error else ""))
        self.output.append(self._html(res))
        while self.plots_layout.count():
            w = self.plots_layout.takeAt(0).widget()
            if w:
                w.setParent(None)
        for data in res.plots:
            lab = QLabel()
            pix = QPixmap()
            pix.loadFromData(data)
            lab.setPixmap(pix.scaledToWidth(min(760, pix.width()), Qt.SmoothTransformation))
            self.plots_layout.addWidget(lab)
        self.plots_layout.addStretch()
        if res.plots and not any(e.kind == "output" for e in res.entries):
            self.out_tabs.setCurrentIndex(1)
        self._record(res)

    def _failed(self, message: str) -> None:
        self._finish()
        self.status.setText("Stopped" if "stopped" in message.lower() else "Failed")
        self.output.append(f"<pre style='color:#c62828'>{html.escape(message)}</pre>")

    @staticmethod
    def _html(res: ConsoleResult) -> str:
        style = {"input": "color:#455a64;background:#eceff1", "output": "color:#000", "message": "color:#1565c0",
                 "warning": "color:#e65100", "error": "color:#c62828;font-weight:bold"}
        parts = []
        for e in res.entries:
            text = ("&gt; " + html.escape(e.text).replace("\n", "\n+ ")) if e.kind == "input" else html.escape(e.text)
            parts.append(f"<pre style='margin:0;{style.get(e.kind, '')}'>{text}</pre>")
        if res.plots:
            parts.append(f"<p style='color:#2e7d32'>[{len(res.plots)} plot(s) — see the Plots tab]</p>")
        return "".join(parts) + "<hr>"

    def _record(self, res: ConsoleResult) -> None:
        """Keep each run in the project (custom script reference, output, plots) for reports and reproducibility."""
        if self.state.project is None:
            return
        script = (res.data_script + "\n# ---- StatExplore: R Console script ----\n"
                  "my_data <- dat; my_data$sx_split <- NULL; names(my_data) <- unname(sx_names[names(my_data)])\n"
                  + self._code) if res.data_script else self._code
        result = AnalysisResult(kind="r_console", title="R Console script", engine=EngineInfo("R", "R (console)"),
                                script=script, inputs={"mode": res.mode, "code": self._code},
                                status=Status.FAILED if res.error else Status.OK,
                                messages=[e.text for e in res.entries if e.kind in ("warning", "error")],
                                elapsed_s=res.elapsed)
        result.values["notes"] = [res.text(("output", "message"))[:20000]]
        artifacts = {}
        for i, data in enumerate(res.plots):
            key = f"plots/console_{result.id}_{i}.png"
            artifacts[key] = data
            result.plots.append(key)
        self.state.add_result(result, artifacts)

    def restart(self) -> None:
        self._project_changed()
        self.output.append("<p><i>Session restarted.</i></p>")

    def clear_workspace(self) -> None:
        if not self.persistent.isChecked():
            QMessageBox.information(self, "R Console", "Clean sessions have no workspace to clear.")
            return
        try:
            self._ensure().clear_workspace()
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "R Console", str(e))
            return
        self.output.append("<p><i>Workspace cleared; my_data restored.</i></p>")

    def _session_list(self, title: str, fn, cols: list[str]) -> None:
        if not self.persistent.isChecked():
            QMessageBox.information(self, "R Console", "Switch on 'Persistent session' to inspect a workspace.")
            return
        try:
            rows = fn()
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "R Console", str(e))
            return
        self.output.append(f"<b>{title}</b><pre>" + html.escape("\n".join(
            "  ".join(str(r.get(c, "")) for c in cols) for r in rows) or "(none)") + "</pre>")

    def show_objects(self) -> None:
        self._session_list("Objects", lambda: self._ensure().objects(), ["name", "class", "size"])

    def show_packages(self) -> None:
        self._session_list("Loaded packages", lambda: self._ensure().packages(), ["package", "version", "attached"])

    def import_dataset(self, confirm: bool = True) -> bool:
        from ...core.console_import import apply_import, diff_import

        if not self.persistent.isChecked() or self.state.project is None:
            QMessageBox.information(self, "Import", "Import works from a persistent session: switch it on, run the "
                                    "script that changes my_data, then import.")
            return False
        try:
            csv, meta = self._ensure().export_my_data()
            diff, new_ds = diff_import(self.state.dataset, csv, meta)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Import", str(e))
            return False
        text = "\n".join(diff.lines())
        if confirm and QMessageBox.question(
                self, "Import resulting dataset", text + "\n\nA snapshot of the current project is saved first so it "
                "can be recovered. Apply these changes?") != QMessageBox.Yes:
            return False
        new_project, snap = apply_import(self.state.project, new_ds)
        self.state.set_project(new_project)
        self.state.dirty = True
        self.output.append(f"<p><b>Imported.</b> Snapshot of the previous project: {html.escape(str(snap))}</p>"
                           f"<pre>{html.escape(text)}</pre>")
        return True

    def package_manager(self) -> None:
        PackageManagerDialog(self.state, self, on_change=self._project_changed).exec()

    # ---- files ------------------------------------------------------------------------------------
    def open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open R script", "", "R scripts (*.R *.r);;All files (*)")
        if path:
            self.editor.setPlainText(Path(path).read_text(encoding="utf-8", errors="replace"))
            self._file = Path(path)

    def save_file(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save R script", str(self._file or "script.R"), "R scripts (*.R)")
        if path:
            Path(path).write_text(self.editor.toPlainText(), encoding="utf-8")
            self._file = Path(path)

    def copy_code(self) -> None:
        QApplication.clipboard().setText(self.editor.toPlainText())

    def clear_output(self) -> None:
        self.output.clear()
        while self.plots_layout.count():
            w = self.plots_layout.takeAt(0).widget()
            if w:
                w.setParent(None)

    def shutdown(self) -> None:
        if self.console is not None:
            self.console.cancel()
            self.console.close()
