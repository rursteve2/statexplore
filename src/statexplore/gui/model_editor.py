"""Widget for building a hierarchical model: outcome, predictors, interactions (up to 3-way)."""
from __future__ import annotations

from itertools import combinations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFormLayout, QGroupBox, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QVBoxLayout, QWidget,
)

from ..analysis.models import ModelSpec
from ..core.dataset import Dataset
from ..core.variables import Measure, Storage
from ..discovery.space import SearchSpaceError, outcome_family

MAX_TERMS_VARS = 6


class ModelEditor(QWidget):
    changed = Signal()

    def __init__(self, ds: Dataset | None = None):
        super().__init__()
        self.ds = ds
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        self.label = QLineEdit()
        self.label.setPlaceholderText("(optional model name)")
        self.outcome = QComboBox()
        self.outcome.currentIndexChanged.connect(self._outcome_changed)
        form.addRow("Model name", self.label)
        form.addRow("Outcome", self.outcome)
        lay.addLayout(form)
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter variables…")
        self.filter.textChanged.connect(self._filter)
        lay.addWidget(self.filter)
        box = QGroupBox(f"Predictors (up to {MAX_TERMS_VARS})")
        bl = QVBoxLayout(box)
        self.vars = QListWidget()
        self.vars.itemChanged.connect(self._vars_changed)
        bl.addWidget(self.vars)
        lay.addWidget(box, 2)
        ibox = QGroupBox("Interactions (lower-order terms are added automatically)")
        il = QVBoxLayout(ibox)
        self.inters = QListWidget()
        self.inters.itemChanged.connect(lambda _: self.changed.emit())
        il.addWidget(self.inters)
        lay.addWidget(ibox, 1)
        self.msg = QLabel()
        self.msg.setStyleSheet("color:#c62828")
        self.msg.setWordWrap(True)
        lay.addWidget(self.msg)
        self._sample_vars: list[str] = []
        self._source = "manual"
        self._selected_from: int | None = None
        self.set_dataset(ds)

    def set_dataset(self, ds: Dataset | None) -> None:
        self.ds = ds
        self.outcome.blockSignals(True)
        self.outcome.clear()
        self.outcome.addItem("— choose —", None)
        self.vars.blockSignals(True)
        self.vars.clear()
        if ds is not None:
            for v in ds.variables.values():
                if v.role.value in ("excluded", "id") or v.storage is Storage.DATETIME:
                    continue
                numeric_cont = v.measure is Measure.CONTINUOUS and v.storage is Storage.NUMERIC
                if numeric_cont or ds.analysis_frame([v.name])[v.name].dropna().nunique() == 2:
                    self.outcome.addItem(v.display_name, v.name)
                k = ds.analysis_frame([v.name])[v.name].dropna().nunique()
                if k <= 1:
                    continue  # constant: can't be a predictor
                if v.is_categorical or numeric_cont:
                    many = f"  ·  {k} categories!" if v.is_categorical and k > 50 else ""
                    it = QListWidgetItem(f"{v.display_name}  ·  {v.measure.value}{many}")
                    it.setData(Qt.UserRole, v.name)
                    it.setToolTip(v.label or v.name)
                    it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                    it.setCheckState(Qt.Unchecked)
                    self.vars.addItem(it)
        self.outcome.blockSignals(False)
        self.vars.blockSignals(False)
        self._rebuild_inters([])

    def _outcome_changed(self) -> None:
        out = self.outcome.currentData()
        for i in range(self.vars.count()):
            it = self.vars.item(i)
            if it.data(Qt.UserRole) == out:
                it.setCheckState(Qt.Unchecked)
            it.setHidden(it.data(Qt.UserRole) == out)
        self.changed.emit()

    def _filter(self, text: str) -> None:
        t = text.lower()
        out = self.outcome.currentData()
        for i in range(self.vars.count()):
            it = self.vars.item(i)
            it.setHidden(it.data(Qt.UserRole) == out or (bool(t) and t not in it.text().lower()
                                                         and it.checkState() != Qt.Checked))

    def chosen(self) -> list[str]:
        return [self.vars.item(i).data(Qt.UserRole) for i in range(self.vars.count())
                if self.vars.item(i).checkState() == Qt.Checked]

    def _vars_changed(self, *_: object) -> None:
        keep = self._checked_inters()
        self._rebuild_inters(keep)
        self.changed.emit()

    def _checked_inters(self) -> list[tuple[str, ...]]:
        return [tuple(self.inters.item(i).data(Qt.UserRole)) for i in range(self.inters.count())
                if self.inters.item(i).checkState() == Qt.Checked]

    def _rebuild_inters(self, keep: list[tuple[str, ...]]) -> None:
        self.inters.blockSignals(True)
        self.inters.clear()
        vs = self.chosen()
        if self.ds is not None and len(vs) <= MAX_TERMS_VARS:
            for k in (2, 3):
                for combo in combinations(vs, k):
                    it = QListWidgetItem(" × ".join(self.ds.var(v).display_name for v in combo))
                    it.setData(Qt.UserRole, list(combo))
                    it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                    # match as sets: (a, b) and (b, a) are the same interaction
                    it.setCheckState(Qt.Checked if frozenset(combo) in {frozenset(k) for k in keep} else Qt.Unchecked)
                    self.inters.addItem(it)
        self.inters.blockSignals(False)

    def model(self) -> ModelSpec:
        ds = self.ds
        out = self.outcome.currentData()
        if ds is None or not out:
            raise ValueError("Choose an outcome.")
        vs = self.chosen()
        if not vs:
            raise ValueError("Choose at least one predictor.")
        if len(vs) > MAX_TERMS_VARS:
            raise ValueError(f"At most {MAX_TERMS_VARS} predictors.")
        try:
            family = outcome_family(ds, out)
        except SearchSpaceError as e:
            raise ValueError(str(e)) from None
        terms = [(v,) for v in vs] + self._checked_inters()
        return ModelSpec(outcome=out, terms=terms, family=family, label=self.label.text().strip(),
                         source=self._source, sample_vars=list(self._sample_vars),
                         selected_from=self._selected_from)

    def set_model(self, m: ModelSpec) -> None:
        self.outcome.setCurrentIndex(max(0, self.outcome.findData(m.outcome)))
        self.label.setText(m.label)
        self.vars.blockSignals(True)
        for i in range(self.vars.count()):
            it = self.vars.item(i)
            it.setCheckState(Qt.Checked if it.data(Qt.UserRole) in m.variables else Qt.Unchecked)
        self.vars.blockSignals(False)
        self._rebuild_inters([t for t in m.terms if len(t) > 1])
        self._sample_vars = list(m.sample_vars)
        self._source = m.source
        self._selected_from = m.selected_from
        self.changed.emit()


class ModelDialog(QDialog):
    """Create a model, or edit a copy (the original stays in the set unchanged)."""

    def __init__(self, ds: Dataset, model: ModelSpec | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Edit a copy of the model" if model else "New model")
        self.resize(520, 700)
        lay = QVBoxLayout(self)
        self.editor = ModelEditor(ds)
        if model:
            self.editor.set_model(model)
            self.editor.label.setText(f"{model.label or 'model'} (edited)")
            self.editor._source = f"edited from {model.id}"
        lay.addWidget(self.editor)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._ok)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)
        self.result_model: ModelSpec | None = None

    def _ok(self) -> None:
        try:
            self.result_model = self.editor.model()
        except ValueError as e:
            self.editor.msg.setText(str(e))
            return
        self.accept()
