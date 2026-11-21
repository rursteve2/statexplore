"""Analysis Builder: pick an analysis, assign variables to roles, set options, run."""
from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMessageBox, QPushButton, QScrollArea, QSpinBox, QSplitter,
    QTextBrowser, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from ...analysis.builders import BY_KEY, SPECS
from ...analysis.framework import KIND_LABELS, AnalysisInputError, AnalysisSpec, eligible, run_analysis, validate
from ...core.schema import AnalysisResult
from ..result_view import render
from ..state import AppState
from .eda import CodeDialog


class SlotWidget(QWidget):
    """Variable chooser for one role: a combo (single) or a checkable list (multiple)."""

    def __init__(self, slot, names: list[tuple[str, str]]):
        super().__init__()
        self.slot = slot
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.multi = slot.max > 1
        if self.multi:
            self.list = QListWidget()
            self.list.setMaximumHeight(150)
            for name, display in names:
                it = QListWidgetItem(display)
                it.setData(Qt.UserRole, name)
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                it.setCheckState(Qt.Unchecked)
                self.list.addItem(it)
            lay.addWidget(self.list)
        else:
            self.combo = QComboBox()
            self.combo.addItem("(none)" if slot.min == 0 else "— choose —", None)
            for name, display in names:
                self.combo.addItem(display, name)
            lay.addWidget(self.combo)

    def values(self) -> list[str]:
        if self.multi:
            return [self.list.item(i).data(Qt.UserRole) for i in range(self.list.count())
                    if self.list.item(i).checkState() == Qt.Checked]
        v = self.combo.currentData()
        return [v] if v else []

    def apply_filter(self, text: str) -> None:
        if self.multi:
            t = text.lower()
            for i in range(self.list.count()):
                it = self.list.item(i)
                it.setHidden(bool(t) and t not in it.text().lower() and it.checkState() != Qt.Checked)


class AnalysisBuilderPage(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.spec: AnalysisSpec | None = None
        self.slot_widgets: dict[str, SlotWidget] = {}
        self.option_widgets: dict[str, QWidget] = {}
        self.current: tuple[AnalysisResult, dict[str, bytes]] | None = None

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        families: dict[str, QTreeWidgetItem] = {}
        for spec in SPECS:
            fam = families.get(spec.family)
            if fam is None:
                fam = families[spec.family] = QTreeWidgetItem([spec.family])
                fam.setFlags(fam.flags() & ~Qt.ItemIsSelectable)
                self.tree.addTopLevelItem(fam)
            it = QTreeWidgetItem([spec.label])
            it.setData(0, Qt.UserRole, spec.key)
            fam.addChild(it)
        self.tree.expandAll()
        self.tree.currentItemChanged.connect(self._select)

        self.form_host = QWidget()
        self.form_layout = QVBoxLayout(self.form_host)
        scroll = QScrollArea()
        scroll.setWidget(self.form_host)
        scroll.setWidgetResizable(True)
        mid = QWidget()
        ml = QVBoxLayout(mid)
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter variable lists…")
        self.filter.textChanged.connect(self._apply_filter)
        ml.addWidget(self.filter)
        ml.addWidget(scroll, 1)
        row = QHBoxLayout()
        self.run_btn = QPushButton("Run analysis")
        self.run_btn.clicked.connect(self._run)
        self.code_btn = QPushButton("View Code")
        self.code_btn.setEnabled(False)
        self.code_btn.clicked.connect(lambda: self.current and CodeDialog(self.current[0], self).exec())
        row.addWidget(self.run_btn)
        row.addWidget(self.code_btn)
        ml.addLayout(row)

        right = QWidget()
        rl = QVBoxLayout(right)
        hist = QHBoxLayout()
        hist.addWidget(QLabel("Results history:"))
        self.history = QComboBox()
        self.history.activated.connect(self._show_history)
        hist.addWidget(self.history, 1)
        rl.addLayout(hist)
        self.output = QTextBrowser()
        self.output.setOpenLinks(False)
        rl.addWidget(self.output, 1)

        split = QSplitter()
        split.addWidget(self.tree)
        split.addWidget(mid)
        split.addWidget(right)
        split.setSizes([230, 380, 760])
        lay = QVBoxLayout(self)
        t = QLabel("Analysis Builder")
        t.setObjectName("pageTitle")
        lay.addWidget(t)
        lay.addWidget(split, 1)

        state.project_changed.connect(self._reload)
        state.metadata_changed.connect(self._rebuild_form)
        state.result_added.connect(lambda _r: self._refresh_history())
        state.busy_changed.connect(lambda b: self.run_btn.setEnabled(not b))
        self._reload()

    # ---- form -------------------------------------------------------
    def _apply_filter(self, text: str) -> None:
        for w in self.slot_widgets.values():
            w.apply_filter(text)

    def _reload(self) -> None:
        self._rebuild_form()
        self._refresh_history()
        self.output.clear()
        self.current = None
        self.code_btn.setEnabled(False)

    def _select(self, item: QTreeWidgetItem | None) -> None:
        key = item.data(0, Qt.UserRole) if item else None
        if key:
            self.spec = BY_KEY[key]
            self._rebuild_form()

    def _rebuild_form(self) -> None:
        old = {k: w.values() for k, w in self.slot_widgets.items()}
        while self.form_layout.count():
            w = self.form_layout.takeAt(0).widget()
            if w:
                w.setParent(None)  # detach now; deleteLater alone leaves it visible until the next loop
                w.deleteLater()
        self.slot_widgets, self.option_widgets = {}, {}
        ds = self.state.dataset
        if self.spec is None or ds is None:
            self.form_layout.addWidget(QLabel("Import a dataset and choose an analysis on the left."
                                              if ds is None else "Choose an analysis on the left."))
            self.form_layout.addStretch()
            return
        desc = QLabel(f"<b>{self.spec.label}</b><br>{self.spec.description}")
        desc.setWordWrap(True)
        self.form_layout.addWidget(desc)
        for slot in self.spec.slots:
            names = [(n, ds.var(n).display_name) for n in eligible(ds, slot)]
            box = QGroupBox(f"{slot.label} — {KIND_LABELS[slot.kind]}"
                            + ("" if slot.max == 1 else f" ({slot.min}–{slot.max})"))
            bl = QVBoxLayout(box)
            w = SlotWidget(slot, names)
            if slot.help:
                h = QLabel(slot.help)
                h.setWordWrap(True)
                h.setObjectName("hint")
                bl.addWidget(h)
            bl.addWidget(w)
            self.slot_widgets[slot.key] = w
            self._restore(w, old.get(slot.key, []))
            self.form_layout.addWidget(box)
        if self.spec.options:
            box = QGroupBox("Options")
            form = QFormLayout(box)
            for o in self.spec.options:
                form.addRow(o.label, self._option_widget(o))
            self.form_layout.addWidget(box)
        self.form_layout.addStretch()
        for w in self.slot_widgets.values():
            w.apply_filter(self.filter.text())

    @staticmethod
    def _restore(w: SlotWidget, values: list[str]) -> None:
        if w.multi:
            for i in range(w.list.count()):
                if w.list.item(i).data(Qt.UserRole) in values:
                    w.list.item(i).setCheckState(Qt.Checked)
        elif values:
            idx = w.combo.findData(values[0])
            if idx >= 0:
                w.combo.setCurrentIndex(idx)

    def _option_widget(self, o) -> QWidget:
        if o.kind == "choice":
            w = QComboBox()
            for val, label in o.choices or []:
                w.addItem(label, val)
            w.setCurrentIndex(max(0, w.findData(o.default)))
        elif o.kind == "bool":
            w = QCheckBox()
            w.setChecked(bool(o.default))
        elif o.kind == "int":
            w = QSpinBox()
            w.setRange(int(o.min if o.min is not None else -10**6), int(o.max if o.max is not None else 10**6))
            w.setValue(int(o.default))
        else:
            w = QDoubleSpinBox()
            w.setDecimals(4)
            w.setRange(o.min if o.min is not None else -1e9, o.max if o.max is not None else 1e9)
            w.setSingleStep(0.01 if (o.max or 2) <= 1 else 1)
            w.setValue(float(o.default))
        if o.help:
            w.setToolTip(o.help)
        self.option_widgets[o.key] = w
        return w

    def _values(self) -> tuple[dict[str, list[str]], dict[str, Any]]:
        vars_ = {k: w.values() for k, w in self.slot_widgets.items()}
        opts: dict[str, Any] = {}
        for k, w in self.option_widgets.items():
            if isinstance(w, QComboBox):
                opts[k] = w.currentData()
            elif isinstance(w, QCheckBox):
                opts[k] = w.isChecked()
            else:
                opts[k] = w.value()
        return vars_, opts

    # ---- run --------------------------------------------------------
    def _run(self) -> None:
        ds, spec = self.state.dataset, self.spec
        if ds is None or spec is None:
            return
        vars_, opts = self._values()
        try:
            validate(spec, ds, vars_, opts)
        except AnalysisInputError as e:
            QMessageBox.information(self, spec.label, str(e))
            return
        self.output.setHtml("<p>Running in R…</p>")
        self.state.run_r(lambda s: run_analysis(s, ds, spec, vars_, opts), self._done, self._error)

    def _done(self, out) -> None:
        result, artifacts = out
        self.state.add_result(result, artifacts)
        self._show(result, artifacts)

    def _show(self, result: AnalysisResult, artifacts: dict[str, bytes]) -> None:
        self.current = (result, artifacts)
        self.code_btn.setEnabled(True)
        render(self.output, result, artifacts)

    def _error(self, message: str) -> None:
        self.output.setHtml(f"<p style='color:#c62828'>{message}</p>")
        QMessageBox.warning(self, "Analysis failed", message)

    def _refresh_history(self) -> None:
        self.history.clear()
        p = self.state.project
        if p is None:
            return
        for r in reversed(p.results):
            if r.kind in BY_KEY:
                self.history.addItem(f"{r.created_at} — {r.title} [{r.status.value}]", r.id)

    def _show_history(self, index: int) -> None:
        p = self.state.project
        rid = self.history.itemData(index)
        if p is None or rid is None:
            return
        for r in p.results:
            if r.id == rid:
                self._show(r, p.artifacts)
                return
