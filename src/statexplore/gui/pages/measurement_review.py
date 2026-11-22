"""Measurement Review (v1.1 §4.2): rule-based suggestions for measurement levels.

Nothing is reclassified automatically. Every change is an explicit, confirmed user action,
recorded in the change log (undoable) and saved with the project.
"""
from __future__ import annotations

import html

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QHeaderView, QLabel,
    QListWidget, QListWidgetItem, QMessageBox, QPushButton, QSplitter, QTableWidget, QTableWidgetItem,
    QTextBrowser, QVBoxLayout, QWidget,
)

from ...analysis import measurement as M
from ...core.dataset import DatasetError
from ...core.variables import Measure
from ..state import AppState

COLUMNS = ["Variable", "Label", "Source (SPSS) level", "Project level", "Suggested level", "Reason", "Status"]
STATUS_TEXT = {"source": "not reviewed", "approved": "approved", "kept": "reviewed, kept"}


def _fmt(code) -> str:
    return str(int(code)) if isinstance(code, float) and code.is_integer() else str(code)


class CategoryOrderDialog(QDialog):
    """Reorder categories (codes stay as they are; only their order changes)."""

    def __init__(self, ds, name: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Category order — {ds.var(name).display_name}")
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Order from lowest to highest. Numeric codes are not changed; the order is used for "
                             "ordinal analyses, tables and plots."))
        self.list = QListWidget()
        for c in ds.levels(name):
            it = QListWidgetItem(f"{_fmt(c)} = {ds.level_label(name, c)}")
            it.setData(Qt.UserRole, c)
            self.list.addItem(it)
        self.list.setDragDropMode(QAbstractItemView.InternalMove)
        lay.addWidget(self.list)
        row = QHBoxLayout()
        for text, step in (("Move up", -1), ("Move down", 1)):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, s=step: self._move(s))
            row.addWidget(b)
        rev = QPushButton("Reverse")
        rev.clicked.connect(self._reverse)
        row.addWidget(rev)
        lay.addLayout(row)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _move(self, step: int) -> None:
        r = self.list.currentRow()
        if r < 0 or not 0 <= r + step < self.list.count():
            return
        it = self.list.takeItem(r)
        self.list.insertItem(r + step, it)
        self.list.setCurrentRow(r + step)

    def _reverse(self) -> None:
        items = [self.list.takeItem(0) for _ in range(self.list.count())]
        for it in reversed(items):
            self.list.addItem(it)

    def order(self) -> list:
        return [self.list.item(i).data(Qt.UserRole) for i in range(self.list.count())]


class MeasurementReviewWidget(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        lay = QVBoxLayout(self)
        intro = QLabel("StatExplore compares each variable's declared level with its values and value labels. "
                       "Suggestions are rule-based and are <b>never applied automatically</b>: review each one. "
                       "Changes affect this project only; the source file and its declared levels are kept.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        bar = QHBoxLayout()
        self.only_flagged = QCheckBox("Show only variables that need review")
        self.only_flagged.setChecked(True)
        self.only_flagged.toggled.connect(self.refresh)
        bar.addWidget(self.only_flagged)
        self.count = QLabel()
        bar.addWidget(self.count)
        bar.addStretch()
        lay.addLayout(bar)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._show_detail)
        self.detail = QTextBrowser()

        actions = QHBoxLayout()
        self.accept_btn = QPushButton("Accept suggestion")
        self.accept_btn.clicked.connect(self._accept)
        self.keep_btn = QPushButton("Keep original classification")
        self.keep_btn.clicked.connect(self._keep)
        self.level = QComboBox()
        for m in Measure:
            self.level.addItem(m.value, m.value)
        self.set_btn = QPushButton("Set this level")
        self.set_btn.clicked.connect(self._set_level)
        self.order_btn = QPushButton("Review category order…")
        self.order_btn.clicked.connect(self._order)
        for w in (self.accept_btn, self.keep_btn, QLabel("Different level:"), self.level, self.set_btn, self.order_btn):
            actions.addWidget(w)
        actions.addStretch()

        split = QSplitter(Qt.Vertical)
        split.addWidget(self.table)
        bottom = QWidget()
        bl = QVBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.addLayout(actions)
        bl.addWidget(self.detail)
        split.addWidget(bottom)
        split.setSizes([320, 260])
        lay.addWidget(split, 1)
        self._names: list[str] = []
        self.refresh()

    # ---- data -------------------------------------------------------------------------
    def refresh(self) -> None:
        ds = self.state.dataset
        current = self.current()
        self.table.setRowCount(0)
        self._names = []
        if ds is None:
            self.count.setText("")
            self._enable(False)
            return
        rows = M.review_rows(ds, only_mismatches=self.only_flagged.isChecked())
        flagged = sum(1 for r in M.review_rows(ds, only_mismatches=True) if r["needs_review"])
        self.count.setText(f"{flagged} variable(s) with a suggestion that differs from the project level")
        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            vals = [ds.var(r["name"]).display_name, r["label"], r["source"], r["project"], r["suggested"],
                    r["reason"], STATUS_TEXT.get(r["status"], r["status"])]
            for j, val in enumerate(vals):
                it = QTableWidgetItem(str(val))
                it.setToolTip(str(val))
                if r["needs_review"] and j == 4:
                    it.setForeground(Qt.darkYellow)
                self.table.setItem(i, j, it)
            self._names.append(r["name"])
        self.table.resizeColumnsToContents()
        for j in (1, 5):
            self.table.setColumnWidth(j, min(self.table.columnWidth(j), 320))
        if current in self._names:
            self.table.selectRow(self._names.index(current))
        elif self._names:
            self.table.selectRow(0)
        self._show_detail()

    def current(self) -> str | None:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        return self._names[rows[0].row()] if rows and rows[0].row() < len(self._names) else None

    def select(self, name: str) -> None:
        if name not in self._names:
            self.only_flagged.setChecked(False)
        if name in self._names:
            self.table.selectRow(self._names.index(name))

    def _enable(self, on: bool) -> None:
        for b in (self.accept_btn, self.keep_btn, self.set_btn, self.order_btn, self.level):
            b.setEnabled(on)

    def _show_detail(self) -> None:
        ds, name = self.state.dataset, self.current()
        if ds is None or name is None:
            self.detail.clear()
            self._enable(False)
            return
        self._enable(True)
        v = ds.var(name)
        self.accept_btn.setEnabled(bool(v.suggested_measure) and v.suggested_measure != v.measure.value)
        self.order_btn.setEnabled(v.is_categorical)
        self.level.setCurrentIndex(max(0, self.level.findData(v.measure.value)))
        col = ds.analysis_column(name)
        counts = col.value_counts(dropna=True)
        raw_counts = ds.column(name).value_counts(dropna=True)
        rows = []
        for c in ds.levels(name)[:60]:
            rows.append(f"<tr><td align='right'>{html.escape(_fmt(c))}</td><td>{html.escape(ds.level_label(name, c))}"
                        f"</td><td align='right'>{int(counts.get(c, 0)):,}</td></tr>")
        missing_codes = [c for c in raw_counts.index if ds.is_user_missing_value(v, c)]
        miss = "".join(f"<tr><td align='right'>{html.escape(_fmt(c))}</td><td>"
                       f"{html.escape(v.value_labels.get(c, ''))} <i>(declared missing)</i></td>"
                       f"<td align='right'>{int(raw_counts[c]):,}</td></tr>" for c in missing_codes)
        reasons = "".join(f"<li>{html.escape(r)}</li>" for r in v.suggestion_reasons) or "<li>No suggestion.</li>"
        self.detail.setHtml(
            f"<h3>{html.escape(v.display_name)}</h3><p>{html.escape(v.label)}</p>"
            f"<p>Source file level: <b>{html.escape(v.source_measure or '—')}</b> · Project level: "
            f"<b>{v.measure.value}</b> · Suggested: <b>{html.escape(v.suggested_measure or '—')}</b> · "
            f"Status: {STATUS_TEXT.get(v.measure_status, v.measure_status)}</p>"
            + (f"<p>Coding direction: {html.escape(v.coding_direction)}</p>" if v.coding_direction else "")
            + f"<p><b>Why:</b></p><ul>{reasons}</ul>"
            f"<p><b>Original numeric codes and value labels</b> (category order as used in analyses; "
            f"missing: {int(col.isna().sum()):,})</p>"
            "<table border='1' cellspacing='0' cellpadding='3'><tr><th>Code</th><th>Label</th><th>Count</th></tr>"
            + "".join(rows) + miss + "</table>"
            + ("<p><i>Only the first 60 categories are shown.</i></p>" if len(ds.levels(name)) > 60 else ""))

    # ---- actions (each needs explicit confirmation) -----------------------------------------
    def _confirm(self, text: str) -> bool:
        return QMessageBox.question(self, "Confirm measurement level", text) == QMessageBox.Yes

    def _apply(self, name: str, measure: str, order=None) -> None:
        ds = self.state.dataset
        v = ds.var(name)
        text = (f"Change the project measurement level of {v.display_name} from {v.measure.value} to {measure}?\n\n"
                f"The source file's level ({v.source_measure or 'none'}) and the data values are not changed. "
                "You can undo this in the Variables tab.")
        if measure == "continuous" and v.is_categorical:
            text += f"\n\n{M.CONTINUOUS_WARNING} {M.EQUAL_INTERVALS}"
        if not self._confirm(text):
            return
        try:
            M.approve(ds, name, measure, order)
        except DatasetError as e:
            QMessageBox.warning(self, "Cannot change level", str(e))
            return
        self.state.notify_metadata_changed()

    def _accept(self) -> None:
        ds, name = self.state.dataset, self.current()
        if ds and name and ds.var(name).suggested_measure:
            self._apply(name, ds.var(name).suggested_measure)

    def _set_level(self) -> None:
        name = self.current()
        if name:
            self._apply(name, self.level.currentData())

    def _keep(self) -> None:
        ds, name = self.state.dataset, self.current()
        if not (ds and name):
            return
        if not self._confirm(f"Keep {ds.var(name).display_name} as {ds.var(name).measure.value}?"):
            return
        M.keep(ds, name)
        self.state.notify_metadata_changed()

    def _order(self) -> None:
        ds, name = self.state.dataset, self.current()
        if not (ds and name):
            return
        dlg = CategoryOrderDialog(ds, name, self)
        if dlg.exec() != QDialog.Accepted:
            return
        order = dlg.order()
        if order == ds.levels(name):
            return
        v = ds.var(name)
        if not self._confirm(f"Save this category order for {v.display_name}?\n"
                             + " < ".join(ds.level_label(name, c) for c in order)):
            return
        try:
            ds.set_variable_field(name, "category_order", order)
        except DatasetError as e:
            QMessageBox.warning(self, "Cannot change order", str(e))
            return
        self.state.notify_metadata_changed()
