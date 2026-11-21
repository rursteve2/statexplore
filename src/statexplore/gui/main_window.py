"""Main window: persistent left sidebar + stacked workspaces, import, project files."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout, QInputDialog, QLabel,
    QListWidget,
    QMainWindow, QMessageBox, QPlainTextEdit, QStackedWidget, QVBoxLayout, QWidget,
)

from ..core.importers import (
    CSV_DELIMITERS, CSV_ENCODINGS, CsvOptions, ImportError_, SUPPORTED_EXTENSIONS, detect_format,
    import_file, list_sheets,
)
from ..core.project import Project, ProjectError
from .pages.analysis_builder import AnalysisBuilderPage
from .pages.comparison import ComparisonPage
from .pages.dataset_explorer import DatasetExplorerPage
from .pages.discovery import DiscoveryPage
from .pages.eda import EdaPage
from .pages.simple import DashboardPage, PlannedPage, SettingsPage
from .pages.teaching import TeachingPage
from .state import AppState

SECTIONS = [
    "Dashboard", "Dataset Explorer", "Exploratory Data Analysis", "Analysis Builder",
    "Automated Model Discovery", "Model Comparison", "Mixed & Longitudinal Models",
    "Teaching Studio", "Reports", "Settings",
]
PLANNED = {
    "Mixed & Longitudinal Models": (5, "Random-intercept/slope models, AR(1) and unstructured covariance, "
                                       "longitudinal trajectories."),
    "Reports": (7, "Excel, Word, PDF and script exports."),
}
FILE_FILTER = "Datasets (*.csv *.sav *.xlsx);;CSV (*.csv);;SPSS (*.sav);;Excel (*.xlsx)"
PROJECT_FILTER = "StatExplore project (*.sxp)"

STYLE = """
QListWidget#sidebar { background: #12263f; color: #dfe9f5; border: none; font-size: 14px; }
QListWidget#sidebar::item { padding: 10px 14px; }
QListWidget#sidebar::item:selected { background: #2b5d8c; color: white; }
QListWidget#sidebar::item:disabled { color: #7d8ea3; }
QLabel#pageTitle { font-size: 20px; font-weight: 600; color: #12263f; padding: 4px 0 8px 0; }
QLabel#hint { color: #2b5d8c; }
"""


class CsvOptionsDialog(QDialog):
    def __init__(self, path: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(f"Import CSV — {Path(path).name}")
        self.resize(700, 420)
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.delim = QComboBox()
        for label, value in CSV_DELIMITERS.items():
            self.delim.addItem(label, value)
        self.enc = QComboBox()
        self.enc.addItem("Auto-detect", None)
        for e in CSV_ENCODINGS:
            self.enc.addItem(e, e)
        form.addRow("Delimiter", self.delim)
        form.addRow("Encoding", self.enc)
        lay.addLayout(form)
        preview = QPlainTextEdit()
        preview.setReadOnly(True)
        with open(path, "rb") as f:
            preview.setPlainText(f.read(4000).decode("utf-8", errors="replace"))
        lay.addWidget(QLabel("File preview (first lines):"))
        lay.addWidget(preview)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def options(self) -> CsvOptions:
        return CsvOptions(delimiter=self.delim.currentData(), encoding=self.enc.currentData())


class MainWindow(QMainWindow):
    def __init__(self, state: AppState | None = None):
        super().__init__()
        self.state = state or AppState()
        self.setWindowTitle("StatExplore")
        self.resize(1360, 860)
        self.setAcceptDrops(True)
        self.setStyleSheet(STYLE)

        self.sidebar = QListWidget()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(270)
        self.sidebar.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.sidebar.setTextElideMode(Qt.ElideRight)
        self.stack = QStackedWidget()
        pages: dict[str, QWidget] = {
            "Dashboard": DashboardPage(self.state, self.import_dialog, self.open_dialog),
            "Dataset Explorer": DatasetExplorerPage(self.state),
            "Exploratory Data Analysis": EdaPage(self.state),
            "Analysis Builder": AnalysisBuilderPage(self.state),
            "Automated Model Discovery": DiscoveryPage(self.state),
            "Model Comparison": ComparisonPage(self.state),
            "Teaching Studio": TeachingPage(self.state),
            "Settings": SettingsPage(self.state),
        }
        for name in SECTIONS:
            page = pages.get(name) or PlannedPage(name, *PLANNED[name])
            self.stack.addWidget(page)
            self.sidebar.addItem(name if name not in PLANNED else f"{name}  (Phase {PLANNED[name][0]})")
        self.sidebar.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.sidebar.setCurrentRow(0)
        self.pages = pages

        central = QWidget()
        h = QHBoxLayout(central)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(self.sidebar)
        body = QWidget()
        bl = QVBoxLayout(body)
        bl.addWidget(self.stack)
        h.addWidget(body, 1)
        self.setCentralWidget(central)

        self._build_menu()
        self.r_label = QLabel("R not started")
        self.statusBar().addPermanentWidget(self.r_label)
        self.state.r_status_changed.connect(self.r_label.setText)
        self.state.busy_changed.connect(
            lambda b: self.statusBar().showMessage("Working…" if b else "Ready", 0 if b else 3000))
        self.state.show_in_teaching.connect(
            lambda _m: self.sidebar.setCurrentRow(SECTIONS.index("Teaching Studio")))
        self.state.project_changed.connect(self._update_title)
        self.state.metadata_changed.connect(self._update_title)
        self.state.result_added.connect(lambda _r: self._update_title())

    def _build_menu(self) -> None:
        m = self.menuBar().addMenu("&File")
        for text, slot, key in [
            ("&Import Dataset…", self.import_dialog, QKeySequence("Ctrl+I")),
            ("&Open Project…", self.open_dialog, QKeySequence.Open),
            ("&Save Project", self.save_project, QKeySequence.Save),
            ("Save Project &As…", self.save_project_as, QKeySequence.SaveAs),
            ("E&xit", self.close, QKeySequence.Quit),
        ]:
            act = QAction(text, self)
            act.setShortcut(key)
            act.triggered.connect(slot)
            m.addAction(act)

    def _update_title(self) -> None:
        p = self.state.project
        name = (p.path.name if p and p.path else "Untitled project") if p else ""
        star = " *" if self.state.dirty else ""
        self.setWindowTitle(f"StatExplore — {name}{star}" if p else "StatExplore")

    # ---- import -----------------------------------------------------
    def import_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import dataset", "", FILE_FILTER)
        if path:
            self.import_path(path)

    def import_path(self, path: str) -> None:
        if not self._confirm_discard():
            return
        try:
            fmt = detect_format(path)
        except ImportError_ as e:
            QMessageBox.warning(self, "Import", str(e))
            return
        csv_opts, sheet = None, None
        if fmt == "csv":
            dlg = CsvOptionsDialog(path, self)
            if dlg.exec() != QDialog.Accepted:
                return
            csv_opts = dlg.options()
        elif fmt == "xlsx":
            try:
                sheets = list_sheets(path)
            except Exception as e:
                QMessageBox.warning(self, "Import", f"Could not read workbook: {e}")
                return
            if len(sheets) > 1:
                sheet, ok = QInputDialog.getItem(self, "Select worksheet", "Worksheet:", sheets, 0, False)
                if not ok:
                    return
        self.state.run_io(
            lambda: Project(import_file(path, csv_options=csv_opts, sheet=sheet)),
            self._imported, lambda msg: QMessageBox.warning(self, "Import failed", msg),
        )

    def _imported(self, project: Project) -> None:
        self.state.set_project(project)
        self.state.dirty = True
        self._update_title()
        self.sidebar.setCurrentRow(SECTIONS.index("Dataset Explorer"))

    def dragEnterEvent(self, e) -> None:
        urls = e.mimeData().urls()
        if len(urls) == 1 and Path(urls[0].toLocalFile()).suffix.lower() in SUPPORTED_EXTENSIONS:
            e.acceptProposedAction()

    def dropEvent(self, e) -> None:
        self.import_path(e.mimeData().urls()[0].toLocalFile())

    # ---- projects ---------------------------------------------------
    def open_dialog(self) -> None:
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open project", "", PROJECT_FILTER)
        if path:
            self.open_project(path)

    def open_project(self, path: str) -> None:
        try:
            self.state.set_project(Project.open(path))
        except ProjectError as e:
            QMessageBox.warning(self, "Open project", str(e))
            return
        self._update_title()

    def save_project(self) -> bool:
        p = self.state.project
        if p is None:
            return False
        if p.path is None:
            return self.save_project_as()
        return self._save_to(p.path)

    def save_project_as(self) -> bool:
        if self.state.project is None:
            return False
        path, _ = QFileDialog.getSaveFileName(self, "Save project", "", PROJECT_FILTER)
        if not path:
            return False
        if not path.lower().endswith(".sxp"):
            path += ".sxp"
        return self._save_to(Path(path))

    def _save_to(self, path: Path) -> bool:
        try:
            self.state.project.save(path)
        except Exception as e:
            QMessageBox.warning(self, "Save project", f"Could not save: {e}")
            return False
        self.state.dirty = False
        self._update_title()
        self.statusBar().showMessage(f"Saved {path}", 4000)
        return True

    def _confirm_discard(self) -> bool:
        if not (self.state.project and self.state.dirty):
            return True
        r = QMessageBox.question(self, "Unsaved changes", "Save the current project first?",
                                 QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if r == QMessageBox.Save:
            return self.save_project()
        return r == QMessageBox.Discard

    def closeEvent(self, e) -> None:
        if not self._confirm_discard():
            e.ignore()
            return
        self.state.wait_idle(5000)
        self.state.shutdown_r()
        e.accept()
