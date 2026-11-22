"""Exploratory Data Analysis workspace: psych descriptives and basic plots."""
from __future__ import annotations

import html

from PySide6.QtCore import QSortFilterProxyModel, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit, QPushButton, QTextBrowser,
    QSplitter, QTableView, QTabWidget, QVBoxLayout, QWidget,
)

from ...analysis import eda
from ...analysis.builders import BY_KEY
from ...analysis.framework import AnalysisInputError, run_analysis, validate
from ...analysis.measurement import TreatmentError, apply_treatments
from ...core.overview import suggest_plots
from ...core.schema import AnalysisResult, Status
from ...core.variables import Measure, Role, Storage
from ..models import FrameModel, table_to_frame
from ..result_view import render
from ..state import AppState
from ..widgets.treatment_panel import TreatmentPanel
from ..widgets.variable_selector import VariableSelector


class CodeDialog(QDialog):
    """Shows the exact script the R worker executed for a result (§18)."""

    def __init__(self, result: AnalysisResult | None, parent: QWidget | None = None, *,
                 title: str = "", script: str = "", header: str = ""):
        super().__init__(parent)
        if result is not None:
            title, script = result.title, result.script
            pk = ", ".join(f"{k} {v}" for k, v in result.engine.packages.items())
            header = f"Engine: {result.engine.version}\nPackages: {pk}"
        self.setWindowTitle(f"Code — {title}")
        self.resize(900, 650)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(header))
        text = QPlainTextEdit(script)
        text.setReadOnly(True)
        text.setLineWrapMode(QPlainTextEdit.NoWrap)
        text.setStyleSheet("font-family: Consolas, monospace;")
        lay.addWidget(text)
        btns = QDialogButtonBox(QDialogButtonBox.Close)
        copy = btns.addButton("Copy", QDialogButtonBox.ActionRole)
        copy.clicked.connect(lambda: (text.selectAll(), text.copy()))
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)


class ImageView(QLabel):
    """Shows a pixmap scaled to fit the available space (never upscaled past 100%)."""

    def __init__(self, text: str):
        super().__init__(text)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(200, 150)
        self._pix: QPixmap | None = None

    def set_image(self, pix: QPixmap) -> None:
        self._pix = pix
        self._rescale()

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._rescale()

    def _rescale(self) -> None:
        if self._pix is None or self._pix.isNull():
            return
        w = min(self.width(), self._pix.width())
        h = min(self.height(), self._pix.height())
        self.setPixmap(self._pix.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation))


class EdaPage(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.last_result: AnalysisResult | None = None

        # left: controls
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.addWidget(QLabel("Variables (check in order: x, then y)"))
        self.selector = VariableSelector()
        self.selector.changed.connect(self._selection_changed)
        ll.addWidget(self.selector, 1)
        form = QFormLayout()
        self.group = QComboBox()
        self.kind = QComboBox()
        for key, label in eda.PLOT_KINDS.items():
            self.kind.addItem(label, key)
        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("(automatic)")
        self.method = QComboBox()
        self.method.addItem("Pearson", "pearson")
        self.method.addItem("Spearman (ordinal allowed)", "spearman")
        form.addRow("Group by", self.group)
        form.addRow("Plot type", self.kind)
        form.addRow("Title", self.title_edit)
        form.addRow("Correlation", self.method)
        ll.addLayout(form)
        self.treatment_panel = TreatmentPanel()
        ll.addWidget(self.treatment_panel)
        self.suggestion = QLabel()
        self.suggestion.setWordWrap(True)
        self.suggestion.setObjectName("hint")
        ll.addWidget(self.suggestion)
        grid = QGridLayout()
        self.plot_btn = QPushButton("Draw plot")
        self.plot_btn.clicked.connect(self._plot)
        self.desc_btn = QPushButton("Describe")
        self.desc_btn.setToolTip("Continuous: psych::describe. Categorical: frequency tables (no means).")
        self.desc_btn.clicked.connect(self._describe)
        self.freq_btn = QPushButton("Frequencies")
        self.freq_btn.setToolTip("Counts, %, valid %, missing, mode and number of categories")
        self.freq_btn.clicked.connect(self._frequencies)
        self.xtab_btn = QPushButton("Cross-tabulation")
        self.xtab_btn.setToolTip("Two categorical variables (or one + Group by): counts, row/column/total %, "
                                 "expected counts, chi-square with exact-test advice")
        self.xtab_btn.clicked.connect(self._crosstab)
        self.corr_btn = QPushButton("Correlation tests")
        self.corr_btn.setToolTip("psych::corr.test: r, n, raw and Holm-adjusted p (Spearman ranks ordinal categories)")
        self.corr_btn.clicked.connect(self._corr_tests)
        self.code_btn = QPushButton("View Code")
        self.code_btn.setEnabled(False)
        self.code_btn.clicked.connect(self._view_code)
        for i, b in enumerate((self.plot_btn, self.desc_btn, self.freq_btn, self.xtab_btn, self.corr_btn, self.code_btn)):
            grid.addWidget(b, i // 3, i % 3)
        ll.addLayout(grid)

        # right: output
        self.out_tabs = QTabWidget()
        self.image = ImageView("Plots appear here.")
        self.out_tabs.addTab(self.image, "Plot")
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
        self.result_view = QTextBrowser()
        self.out_tabs.addTab(self.result_view, "Tables && tests")
        self.corr_view = self.result_view
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
        split.setSizes([360, 800])
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
        self.group.clear()
        self.group.addItem("(none)", None)
        if ds is None:
            self.selector.set_variables(None, [], keep=False)
            return
        names = [v.name for v in ds.variables.values()
                 if v.role is not Role.EXCLUDED and v.storage is not Storage.DATETIME]
        self.selector.set_variables(ds, names, decorate=lambda n: f"  ·  {ds.var(n).measure.value}")
        for v in ds.variables.values():
            if v.is_categorical and len(ds.levels(v.name)) <= 30:
                self.group.addItem(v.display_name, v.name)

    def _selected(self) -> list[str]:
        """Checked variables in the order the user checked them (x, then y)."""
        return self.selector.selected()

    def _view(self):
        """The dataset with this page's "treat as continuous" choices applied (analysis only)."""
        return apply_treatments(self.state.dataset, self.treatment_panel.treatments())

    def _selection_changed(self) -> None:
        self.treatment_panel.set_variables(self.state.dataset, self._selected())
        self._update_suggestion()

    def _update_suggestion(self) -> None:
        ds = self.state.dataset
        names = self._selected()
        if ds is None or not names:
            self.suggestion.setText("")
            return
        try:
            view = self._view()
        except TreatmentError:
            view = ds
        measures = ["continuous" if not view.var(n).is_categorical else "categorical" for n in names]
        kinds = suggest_plots(measures)
        if len(names) == 2 and measures.count("categorical") == 1 and kinds:
            self.suggestion.setText("Suggested: boxplot or violin — select the continuous variable and "
                                    "set the categorical one as 'Group by'.")
        elif len(names) == 2 and measures.count("categorical") == 2:
            self.suggestion.setText("Two categorical variables: Cross-tabulation, or a clustered / stacked / mosaic "
                                    "plot of the first with the second as 'Group by'.")
        elif kinds:
            self.suggestion.setText("Suggested: " + ", ".join(eda.PLOT_KINDS[k] for k in kinds))
            idx = self.kind.findData(kinds[0])
            if idx >= 0:
                self.kind.setCurrentIndex(idx)
        else:
            self.suggestion.setText("")

    # ---- actions ----------------------------------------------------
    def _busy(self, busy: bool) -> None:
        for b in (self.plot_btn, self.desc_btn, self.freq_btn, self.xtab_btn, self.corr_btn):
            b.setEnabled(not busy)
        if busy:
            self.messages.setText("Running in R…")

    def _checked_view(self, title: str):
        try:
            return self._view()
        except TreatmentError as e:
            QMessageBox.information(self, title, str(e))
            return None

    def _plot(self) -> None:
        ds = self.state.dataset and self._checked_view("Plot")
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
        # Grouped categorical plot of two checked categorical variables: the second is the group.
        if req.kind in eda.CATEGORICAL_PLOTS and len(names) == 2 and req.group is None:
            req.variables, req.group = names[:1], names[1]
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
            self.image.set_image(pix)
            self.out_tabs.setCurrentIndex(0)

    def _describe(self) -> None:
        ds = self.state.dataset and self._checked_view("Describe")
        if ds is None:
            return
        names = [n for n in self._selected() if not ds.var(n).is_categorical]
        cats = [n for n in self._selected() if ds.var(n).is_categorical]
        group = self.group.currentData()
        if not names and not cats:
            QMessageBox.information(self, "Describe", "Check at least one variable.")
            return
        if names:
            self.state.run_r(lambda s: eda.describe(s, ds, names, group), self._describe_done, self._error)
        if cats:
            self.state.run_r(lambda s: eda.frequencies(s, ds, cats), self._tables_done, self._error)

    def _frequencies(self) -> None:
        ds = self.state.dataset and self._checked_view("Frequencies")
        if ds is None:
            return
        cats = [n for n in self._selected() if ds.var(n).is_categorical]
        if not cats:
            QMessageBox.information(self, "Frequencies", "Check at least one categorical (nominal, ordinal or "
                                    "binary) variable.")
            return
        self.state.run_r(lambda s: eda.frequencies(s, ds, cats), self._tables_done, self._error)

    def _crosstab(self) -> None:
        ds = self.state.dataset and self._checked_view("Cross-tabulation")
        if ds is None:
            return
        cats = [n for n in self._selected() if ds.var(n).is_categorical]
        if len(cats) == 1 and self.group.currentData() and self.group.currentData() != cats[0]:
            cats.append(self.group.currentData())
        if len(cats) != 2:
            QMessageBox.information(self, "Cross-tabulation", "Check exactly two categorical variables (rows, then "
                                    "columns), or one plus a 'Group by' variable.")
            return
        self.state.run_r(lambda s: eda.crosstab(s, ds, cats[0], cats[1]), self._tables_done, self._error)

    def _corr_tests(self) -> None:
        ds = self.state.dataset and self._checked_view("Correlation tests")
        if ds is None:
            return
        method = self.method.currentData()
        names = [n for n in self._selected() if not ds.var(n).is_categorical
                 or (method == "spearman" and ds.var(n).measure is Measure.ORDINAL)
                 or (method == "pearson" and ds.var(n).measure is Measure.ORDINAL)]
        spec = BY_KEY["cor_matrix"]
        opts = {"method": method}
        treatments = self.treatment_panel.treatments()
        try:
            validate(spec, ds, {"variables": names}, opts)
        except AnalysisInputError as e:
            QMessageBox.information(self, "Correlation tests", str(e))
            return
        base = self.state.dataset
        self.state.run_r(lambda s: run_analysis(s, base, spec, {"variables": names}, opts, treatments=treatments),
                         self._corr_done, self._error)

    def _corr_done(self, out) -> None:
        result, artifacts = out
        self.state.add_result(result, artifacts)
        self._show_result(result)
        render(self.result_view, result, artifacts)
        self.out_tabs.setCurrentIndex(2)

    def _tables_done(self, result: AnalysisResult) -> None:
        self.state.add_result(result, {})
        self._show_result(result)
        render(self.result_view, result, {})
        self.out_tabs.setCurrentIndex(2)

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
        parts = [f"<b>{html.escape(result.title)}</b> — status: {result.status.value}"]
        if result.n_used is not None:
            parts.append(f"n used = {result.n_used:,} ({html.escape(result.missing_data)})")
        elif result.missing_data:
            parts.append(html.escape(result.missing_data))
        parts.append(f"engine: {html.escape(result.engine.version)}")
        for t in result.inputs.get("treatments") or []:
            parts.append(f"<br><i>{html.escape(t.get('description', ''))}</i>")
        color = "#c62828" if result.status is Status.FAILED else "#8d6e00"
        msgs = "".join(f"<br><span style='color:{color}'>{html.escape(m)}</span>" for m in result.messages)
        self.messages.setText(" · ".join(parts) + msgs)

    def _view_code(self) -> None:
        if self.last_result:
            CodeDialog(self.last_result, self).exec()

    def _error(self, message: str) -> None:
        self.messages.setText(f"<span style='color:#c62828'>{html.escape(message)}</span>")
        QMessageBox.warning(self, "R engine", message)
