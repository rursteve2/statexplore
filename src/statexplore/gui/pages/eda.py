"""Exploratory Data Analysis workspace: psych descriptives and basic plots."""
from __future__ import annotations

from PySide6.QtCore import QSortFilterProxyModel, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit, QPushButton, QScrollArea,
    QSplitter, QTableView, QTabWidget, QVBoxLayout, QWidget,
)

from ...analysis import eda
from ...core.overview import suggest_plots
from ...core.schema import AnalysisResult, Status
from ...core.variables import Role, Storage
from ..models import FrameModel, table_to_frame
from ..state import AppState


class CodeDialog(QDialog):
    """Shows the exact script the R worker executed for a result (§18)."""

    def __init__(self, result: AnalysisResult, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(f"Code — {result.title}")
        self.resize(900, 650)
        lay = QVBoxLayout(self)
        pk = ", ".join(f"{k} {v}" for k, v in result.engine.packages.items())
        lay.addWidget(QLabel(f"Engine: {result.engine.version}\nPackages: {pk}"))
        text = QPlainTextEdit(result.script)
        text.setReadOnly(True)
        text.setLineWrapMode(QPlainTextEdit.NoWrap)
        text.setStyleSheet("font-family: Consolas, monospace;")
        lay.addWidget(text)
        btns = QDialogButtonBox(QDialogButtonBox.Close)
        copy = btns.addButton("Copy", QDialogButtonBox.ActionRole)
        copy.clicked.connect(lambda: (text.selectAll(), text.copy()))
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)


class EdaPage(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.last_result: AnalysisResult | None = None
        self._order: list[str] = []

        # left: controls
        left = QWidget()
        ll = QVBoxLayout(left)
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter variables…")
        self.filter.textChanged.connect(self._filter_vars)
        self.vars = QListWidget()
        self.vars.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.vars.itemSelectionChanged.connect(self._update_suggestion)
        ll.addWidget(QLabel("Variables (select in order: x, then y)"))
        ll.addWidget(self.filter)
        ll.addWidget(self.vars, 1)
        form = QFormLayout()
        self.group = QComboBox()
        self.kind = QComboBox()
        for key, label in eda.PLOT_KINDS.items():
            self.kind.addItem(label, key)
        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("(automatic)")
        form.addRow("Group by", self.group)
        form.addRow("Plot type", self.kind)
        form.addRow("Title", self.title_edit)
        ll.addLayout(form)
        self.suggestion = QLabel()
        self.suggestion.setWordWrap(True)
        self.suggestion.setObjectName("hint")
        ll.addWidget(self.suggestion)
        row = QHBoxLayout()
        self.plot_btn = QPushButton("Draw plot")
        self.plot_btn.clicked.connect(self._plot)
        self.desc_btn = QPushButton("Describe (psych)")
        self.desc_btn.clicked.connect(self._describe)
        self.code_btn = QPushButton("View Code")
        self.code_btn.setEnabled(False)
        self.code_btn.clicked.connect(self._view_code)
        for b in (self.plot_btn, self.desc_btn, self.code_btn):
            row.addWidget(b)
        ll.addLayout(row)

        # right: output
        self.out_tabs = QTabWidget()
        self.image = QLabel("Plots appear here.")
        self.image.setAlignment(Qt.AlignCenter)
        scroll = QScrollArea()
        scroll.setWidget(self.image)
        scroll.setWidgetResizable(True)
        self.out_tabs.addTab(scroll, "Plot")
        self.desc_model = FrameModel()
        proxy = QSortFilterProxyModel()
        proxy.setSourceModel(self.desc_model)
        proxy.setSortRole(Qt.UserRole)
        dv = QTableView()
        dv.setModel(proxy)
        dv.setSortingEnabled(True)
        dv.sortByColumn(-1, Qt.AscendingOrder)  # keep R's row order until the user sorts
        self.desc_note = QLabel()
        self.desc_note.setWordWrap(True)
        dw = QWidget()
        dl = QVBoxLayout(dw)
        dl.addWidget(dv)
        dl.addWidget(self.desc_note)
        self.out_tabs.addTab(dw, "Descriptives")
        self.messages = QLabel()
        self.messages.setWordWrap(True)
        self.messages.setTextInteractionFlags(Qt.TextSelectableByMouse)
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.addWidget(self.out_tabs, 1)
        rl.addWidget(self.messages)

        split = QSplitter()
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([320, 800])
        lay = QVBoxLayout(self)
        t = QLabel("Exploratory Data Analysis")
        t.setObjectName("pageTitle")
        lay.addWidget(t)
        lay.addWidget(QLabel("Exploratory graphics describe associations only; they do not establish causal relationships."))
        lay.addWidget(split, 1)

        state.project_changed.connect(self.reload)
        state.metadata_changed.connect(self.reload)
        state.busy_changed.connect(self._busy)
        self.reload()

    # ---- variable lists ---------------------------------------------
    def reload(self) -> None:
        ds = self.state.dataset
        selected = set(self._selected())
        self.vars.clear()
        self.group.clear()
        self.group.addItem("(none)", None)
        if ds is None:
            return
        for v in ds.variables.values():
            if v.role is Role.EXCLUDED or v.storage is Storage.DATETIME:
                continue
            item = QListWidgetItem(f"{v.display_name}  ·  {v.measure.value}")
            item.setData(Qt.UserRole, v.name)
            item.setToolTip(v.label or v.name)
            self.vars.addItem(item)
            item.setSelected(v.name in selected)
            if v.is_categorical and len(ds.levels(v.name)) <= 30:
                self.group.addItem(v.display_name, v.name)
        self._filter_vars(self.filter.text())

    def _filter_vars(self, text: str) -> None:
        t = text.lower()
        for i in range(self.vars.count()):
            it = self.vars.item(i)
            it.setHidden(bool(t) and t not in it.text().lower() and t not in it.toolTip().lower())

    def _selected(self) -> list[str]:
        """Selected variables in the order the user picked them (x, then y)."""
        current = {it.data(Qt.UserRole) for it in self.vars.selectedItems()}
        kept = [n for n in self._order if n in current]
        added = [it.data(Qt.UserRole) for it in self.vars.selectedItems() if it.data(Qt.UserRole) not in kept]
        self._order = kept + added
        return list(self._order)

    def _update_suggestion(self) -> None:
        ds = self.state.dataset
        names = self._selected()
        if ds is None or not names:
            self.suggestion.setText("")
            return
        measures = ["continuous" if not ds.var(n).is_categorical else "categorical" for n in names]
        kinds = suggest_plots(measures)
        if len(names) == 2 and measures.count("categorical") == 1 and kinds:
            self.suggestion.setText("Suggested: boxplot or violin — select the continuous variable and "
                                    "set the categorical one as 'Group by'.")
        elif kinds:
            self.suggestion.setText("Suggested: " + ", ".join(eda.PLOT_KINDS[k] for k in kinds))
            idx = self.kind.findData(kinds[0])
            if idx >= 0:
                self.kind.setCurrentIndex(idx)
        else:
            self.suggestion.setText("")

    # ---- actions ----------------------------------------------------
    def _busy(self, busy: bool) -> None:
        for b in (self.plot_btn, self.desc_btn):
            b.setEnabled(not busy)
        if busy:
            self.messages.setText("Running in R…")

    def _plot(self) -> None:
        ds = self.state.dataset
        if ds is None:
            return
        names = self._selected()
        req = eda.PlotRequest(kind=self.kind.currentData(), variables=names,
                              group=self.group.currentData(), title=self.title_edit.text().strip())
        # Box/violin of one continuous variable selected with a categorical one: use it as group.
        if req.kind in ("boxplot", "violin") and len(names) == 2 and req.group is None:
            cats = [n for n in names if ds.var(n).is_categorical]
            if len(cats) == 1:
                req.group = cats[0]
                req.variables = [n for n in names if n != cats[0]]
        try:
            eda._validate_plot(ds, req)
        except eda.AnalysisInputError as e:
            QMessageBox.information(self, "Plot", str(e))
            return
        self.state.run_r(lambda s: eda.plot(s, ds, req), self._plot_done, self._error)

    def _plot_done(self, out) -> None:
        result, artifacts = out
        self.state.add_result(result, artifacts)
        self._show_result(result)
        if result.plots:
            pix = QPixmap()
            pix.loadFromData(artifacts[result.plots[0]])
            self.image.setPixmap(pix)
            self.out_tabs.setCurrentIndex(0)

    def _describe(self) -> None:
        ds = self.state.dataset
        if ds is None:
            return
        names = [n for n in self._selected() if not ds.var(n).is_categorical]
        group = self.group.currentData()
        if not names:
            QMessageBox.information(self, "Describe", "Select at least one continuous variable.")
            return
        self.state.run_r(lambda s: eda.describe(s, ds, names, group), self._describe_done, self._error)

    def _describe_done(self, result: AnalysisResult) -> None:
        self.state.add_result(result, {})
        self._show_result(result)
        if result.tables:
            t = result.tables[0]
            self.desc_model.set_frame(table_to_frame(t))
            self.desc_note.setText(t.notes)
            self.out_tabs.setCurrentIndex(1)

    def _show_result(self, result: AnalysisResult) -> None:
        self.last_result = result
        self.code_btn.setEnabled(True)
        parts = [f"<b>{result.title}</b> — status: {result.status.value}"]
        if result.n_used is not None:
            parts.append(f"n used = {result.n_used:,} ({result.missing_data})")
        elif result.missing_data:
            parts.append(result.missing_data)
        parts.append(f"engine: {result.engine.version}")
        color = "#c62828" if result.status is Status.FAILED else "#8d6e00"
        msgs = "".join(f"<br><span style='color:{color}'>{m}</span>" for m in result.messages)
        self.messages.setText(" · ".join(parts) + msgs)

    def _view_code(self) -> None:
        if self.last_result:
            CodeDialog(self.last_result, self).exec()

    def _error(self, message: str) -> None:
        self.messages.setText(f"<span style='color:#c62828'>{message}</span>")
        QMessageBox.warning(self, "R engine", message)
