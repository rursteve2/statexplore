"""Model Comparison workspace (§14)."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMessageBox, QPushButton, QSplitter,
    QTextBrowser, QVBoxLayout, QWidget,
)

from ...analysis import comparison
from ..model_editor import ModelDialog
from ..result_view import render
from ..state import AppState
from .eda import CodeDialog


class ComparisonPage(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.current = None
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.addWidget(QLabel("Model set — check the models to compare:"))
        self.list = QListWidget()
        ll.addWidget(self.list, 1)
        r1 = QHBoxLayout()
        for text, slot in [("New model…", self._new), ("Edit a copy…", self._edit_copy), ("Remove", self._remove),
                           ("Plot in Teaching Studio", self._to_teaching)]:
            b = QPushButton(text)
            b.clicked.connect(slot)
            r1.addWidget(b)
        ll.addLayout(r1)
        self.common = QCheckBox("Fit all models to the same rows (complete on every model's variables)")
        self.common.setChecked(True)
        ll.addWidget(self.common)
        r2 = QHBoxLayout()
        self.compare_btn = QPushButton("Compare checked models")
        self.compare_btn.clicked.connect(self._compare)
        self.code_btn = QPushButton("View Code")
        self.code_btn.setEnabled(False)
        self.code_btn.clicked.connect(lambda: self.current and CodeDialog(self.current, self, dataset=self.state.dataset).exec())
        r2.addWidget(self.compare_btn)
        r2.addWidget(self.code_btn)
        ll.addLayout(r2)
        self.output = QTextBrowser()
        split = QSplitter()
        split.addWidget(left)
        split.addWidget(self.output)
        split.setSizes([480, 880])
        lay = QVBoxLayout(self)
        t = QLabel("Model Comparison")
        t.setObjectName("pageTitle")
        lay.addWidget(t)
        note = QLabel("Nested models (one contains all terms of the other) get an F or likelihood-ratio test; "
                      "non-nested models are compared by AIC/BIC only. Editing creates a copy, so discovered "
                      "models are preserved for comparison.")
        note.setWordWrap(True)
        lay.addWidget(note)
        lay.addWidget(split, 1)
        state.project_changed.connect(self._refresh)
        state.models_changed.connect(self._refresh)
        state.busy_changed.connect(lambda b: self.compare_btn.setEnabled(not b))
        self._refresh()

    def _refresh(self) -> None:
        shown = {self.list.item(i).data(Qt.UserRole) for i in range(self.list.count())}
        checked = set(self._checked_ids())
        self.list.clear()
        p = self.state.project
        if p is None:
            return
        for m in p.models:
            it = QListWidgetItem(f"{m.label or m.id} — {m.display(p.dataset)}  [{m.source}]")
            it.setData(Qt.UserRole, m.id)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            # keep the user's choices; newly added models start checked
            it.setCheckState(Qt.Checked if m.id in checked or m.id not in shown else Qt.Unchecked)
            self.list.addItem(it)

    def _checked_ids(self) -> list[str]:
        return [self.list.item(i).data(Qt.UserRole) for i in range(self.list.count())
                if self.list.item(i).checkState() == Qt.Checked]

    def _selected_model(self):
        it = self.list.currentItem()
        p = self.state.project
        if it is None or p is None:
            QMessageBox.information(self, "Model Comparison", "Select a model first.")
            return None
        return next(m for m in p.models if m.id == it.data(Qt.UserRole))

    def _new(self) -> None:
        if self.state.dataset is None:
            return
        dlg = ModelDialog(self.state.dataset, None, self)
        if dlg.exec() == QDialog.Accepted and dlg.result_model:
            self.state.add_model(dlg.result_model)

    def _edit_copy(self) -> None:
        m = self._selected_model()
        if m is None:
            return
        dlg = ModelDialog(self.state.dataset, m, self)
        if dlg.exec() == QDialog.Accepted and dlg.result_model:
            self.state.add_model(dlg.result_model)

    def _remove(self) -> None:
        m = self._selected_model()
        if m is not None:
            self.state.remove_model(m.id)

    def _to_teaching(self) -> None:
        m = self._selected_model()
        if m is not None:
            self.state.show_in_teaching.emit(m)

    def _compare(self) -> None:
        p = self.state.project
        if p is None:
            return
        ids = self._checked_ids()
        models = [m for m in p.models if m.id in ids]
        try:
            comparison.build(p.dataset, models, self.common.isChecked())
        except comparison.ComparisonError as e:
            QMessageBox.information(self, "Model Comparison", str(e))
            return
        ds, common = p.dataset, self.common.isChecked()
        self.state.run_r(lambda s: comparison.compare(s, ds, models, common), self._done,
                         lambda msg: QMessageBox.warning(self, "Comparison failed", msg))

    def _done(self, out) -> None:
        result, artifacts = out
        self.state.add_result(result, artifacts)
        self.current = result
        self.code_btn.setEnabled(True)
        render(self.output, result, artifacts)
