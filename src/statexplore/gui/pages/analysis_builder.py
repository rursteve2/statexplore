"""Analysis Builder: pick an analysis, assign variables to roles, set options, run."""
from __future__ import annotations

import html
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMessageBox, QPushButton, QScrollArea, QSpinBox, QSplitter,
    QTextBrowser, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from ...analysis.builders import BY_KEY, SPECS
from ...analysis.framework import (
    KIND_LABELS, AnalysisInputError, AnalysisSpec, eligible, run_analysis, treatable, validate,
)
from ...analysis.measurement import TreatmentError, apply_treatments
from ...core.schema import AnalysisResult
from ..result_view import render
from ..state import AppState
from ..widgets.treatment_panel import TreatmentPanel
from ..widgets.variable_selector import VariableSelector
from .eda import CodeDialog


def fmt_code(code) -> str:
    return str(int(code)) if isinstance(code, float) and code.is_integer() else str(code)


class SlotWidget(QWidget):
    """Variable chooser for one role: a combo (single) or the standard variable selector (multiple)."""
    changed = Signal()

    def __init__(self, slot, names: list[tuple[str, str]], ds=None):
        super().__init__()
        self.slot = slot
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.multi = slot.max > 1
        if self.multi:
            self.selector = VariableSelector(max_height=230)
            labels = dict(names)
            self.selector.set_variables(ds, [n for n, _ in names], keep=False,
                                        decorate=lambda n: labels[n][len(ds.var(n).display_name):] if ds else "")
            self.list = self.selector.list
            self.selector.changed.connect(self.changed.emit)
            lay.addWidget(self.selector)
        else:
            self.combo = QComboBox()
            self.combo.addItem("(none)" if slot.min == 0 else "— choose —", None)
            for name, display in names:
                self.combo.addItem(display, name)
            self.combo.currentIndexChanged.connect(lambda _i: self.changed.emit())
            lay.addWidget(self.combo)

    def values(self) -> list[str]:
        if self.multi:
            return self.selector.selected()
        v = self.combo.currentData()
        return [v] if v else []

    def apply_filter(self, text: str) -> None:
        if self.multi:
            self.selector.set_filter(text)


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
        self.filter.setPlaceholderText("Search all variable lists…")
        self.filter.textChanged.connect(self._apply_filter)
        ml.addWidget(self.filter)
        ml.addWidget(scroll, 1)
        row = QHBoxLayout()
        self.run_btn = QPushButton("Run analysis")
        self.run_btn.clicked.connect(self._run)
        self.code_btn = QPushButton("View Code")
        self.code_btn.setEnabled(False)
        self.code_btn.clicked.connect(lambda: self.current and CodeDialog(self.current[0], self).exec())
        self.copy_btn = QPushButton("Copy tables")
        self.copy_btn.setToolTip("Copy all result tables as tab-separated text (pastes as tables into Word or Excel)")
        self.copy_btn.clicked.connect(self._copy_tables)
        row.addWidget(self.run_btn)
        row.addWidget(self.code_btn)
        row.addWidget(self.copy_btn)
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
            names = [(n, ds.var(n).display_name + ("  · categorical codes" if slot.kind == "continuous"
                                                   and treatable(ds, n) else ""))
                     for n in eligible(ds, slot, offer_treatment=True)]
            box = QGroupBox(f"{slot.label} — {KIND_LABELS[slot.kind]}"
                            + ("" if slot.max == 1 else f" ({slot.min}–{slot.max})"))
            bl = QVBoxLayout(box)
            w = SlotWidget(slot, names, ds)
            if slot.help:
                h = QLabel(slot.help)
                h.setWordWrap(True)
                h.setObjectName("hint")
                bl.addWidget(h)
            bl.addWidget(w)
            self.slot_widgets[slot.key] = w
            self._restore(w, old.get(slot.key, []))
            w.changed.connect(self._update_treatments)
            self.form_layout.addWidget(box)
        self.treatment_panel = TreatmentPanel()
        self.form_layout.addWidget(self.treatment_panel)
        self._update_treatments()
        if self.spec.options:
            box = QGroupBox("Options")
            form = QFormLayout(box)
            for o in self.spec.options:
                form.addRow(o.label, self._option_widget(o))
            self.form_layout.addWidget(box)
        self.preview = QLabel()
        self.preview.setTextFormat(Qt.RichText)
        self.preview.setWordWrap(True)
        self.preview.setVisible(self.spec.key == "assoc_2x2")
        self.form_layout.addWidget(self.preview)
        for o in self.spec.options:
            if o.kind == "level":
                self._fill_levels(o)
                self.slot_widgets[o.slot].changed.connect(lambda o=o: self._fill_levels(o))
                self.option_widgets[o.key].currentIndexChanged.connect(lambda _i: self._update_preview())
        self._update_preview()
        self.form_layout.addStretch()
        for w in self.slot_widgets.values():
            w.apply_filter(self.filter.text())

    def _fill_levels(self, o) -> None:
        """Category choices for a 'level' option follow the variable chosen in its slot."""
        w = self.option_widgets[o.key]
        ds = self.state.dataset
        keep = w.currentData()
        w.blockSignals(True)
        w.clear()
        w.addItem("— choose —", None)
        chosen = self.slot_widgets[o.slot].values()
        if ds is not None and chosen:
            for code in ds.levels(chosen[0])[:200]:
                w.addItem(f"{fmt_code(code)} = {ds.level_label(chosen[0], code)}", code)
        idx = w.findData(keep) if keep is not None else -1
        w.setCurrentIndex(max(idx, 0))
        w.blockSignals(False)
        self._update_preview()

    def _update_preview(self) -> None:
        if self.spec is None or self.spec.key != "assoc_2x2" or not hasattr(self, "preview"):
            return
        from ...analysis.builders import preview_2x2

        vars_, opts = self._values()
        tab = preview_2x2(self.state.dataset, vars_, opts) if self.state.dataset is not None else None
        if tab is None:
            self.preview.setText("<i>Choose both variables and all four categories to see the 2 × 2 table.</i>")
            return
        head = "".join(f"<th>{html.escape(c)}</th>" for c in [""] + list(tab.columns))
        rows = "".join("<tr><td><b>" + html.escape(str(i)) + "</b></td>"
                       + "".join(f"<td align='right'>{v:,}</td>" for v in r) + "</tr>"
                       for i, r in zip(tab.index, tab.to_numpy().tolist()))
        self.preview.setText("<b>2 × 2 table (complete cases)</b><table border='1' cellspacing='0' cellpadding='3'>"
                             f"<tr>{head}</tr>{rows}</table>")

    def _update_treatments(self) -> None:
        names = [n for w in self.slot_widgets.values() for n in w.values()]
        dep = self.slot_widgets.get("dependent")
        outcome = (dep.values() or [None])[0] if dep is not None else None
        self.treatment_panel.set_variables(self.state.dataset, names, outcome=outcome)

    @staticmethod
    def _restore(w: SlotWidget, values: list[str]) -> None:
        if w.multi:
            w.selector.set_selected([v for v in values if v in w.selector.names()])
        elif values:
            idx = w.combo.findData(values[0])
            if idx >= 0:
                w.combo.setCurrentIndex(idx)

    def _option_widget(self, o) -> QWidget:
        if o.kind == "level":
            w = QComboBox()
            w.addItem("— choose —", None)
        elif o.kind == "choice":
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
        treatments = self.treatment_panel.treatments()
        try:
            validate(spec, apply_treatments(ds, treatments), vars_, opts)
        except (AnalysisInputError, TreatmentError) as e:
            QMessageBox.information(self, spec.label, str(e))
            return
        self.output.setHtml("<p>Running in R…</p>")
        self.state.run_r(lambda s: run_analysis(s, ds, spec, vars_, opts, treatments=treatments),
                         self._done, self._error)

    def _done(self, out) -> None:
        result, artifacts = out
        self.state.add_result(result, artifacts)
        self._show(result, artifacts)

    def _show(self, result: AnalysisResult, artifacts: dict[str, bytes]) -> None:
        self.current = (result, artifacts)
        self.code_btn.setEnabled(True)
        render(self.output, result, artifacts)

    def _copy_tables(self) -> None:
        from PySide6.QtWidgets import QApplication

        from ...analysis.interpret import tables_tsv

        if self.current:
            QApplication.clipboard().setText(tables_tsv(self.current[0]))

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
