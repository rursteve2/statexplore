"""Dialogs for creating derived variables and row filters (spec-based, no free-text code)."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QPushButton, QSpinBox, QStackedWidget, QTableWidget,
    QVBoxLayout, QWidget,
)

from ..core.dataset import Dataset
from ..core.derive import DERIVE_OPS, FILTER_OPS, VALUELESS_FILTER_OPS, DerivedSpec, FilterSpec
from ..core.variables import Measure, Storage
from .models import fmt


class DeriveDialog(QDialog):
    def __init__(self, ds: Dataset, parent: QWidget | None = None):
        super().__init__(parent)
        self.ds = ds
        self.setWindowTitle("New derived variable")
        self.resize(560, 620)
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.name = QLineEdit()
        self.label = QLineEdit()
        self.label.setPlaceholderText("(optional — defaults to a description of the formula)")
        self.op = QComboBox()
        for key, text in DERIVE_OPS.items():
            self.op.addItem(text, key)
        self.measure = QComboBox()
        self.measure.addItems([m.value for m in Measure])
        form.addRow("New variable name", self.name)
        form.addRow("Label", self.label)
        form.addRow("Operation", self.op)
        form.addRow("Measurement level of result", self.measure)
        lay.addLayout(form)

        lay.addWidget(QLabel("Source variable(s) — numeric codes are used:"))
        self.sources = QListWidget()
        for v in ds.variables.values():
            if v.storage is Storage.NUMERIC:
                it = QListWidgetItem(f"{v.display_name}  ·  {v.label[:60]}" if v.label else v.display_name)
                it.setData(Qt.UserRole, v.name)
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                it.setCheckState(Qt.Unchecked)
                self.sources.addItem(it)
        lay.addWidget(self.sources, 1)

        self.params = QStackedWidget()
        self.p_none = QLabel("")
        self.min_valid = QSpinBox()
        self.min_valid.setRange(1, 100)
        mv = QWidget()
        QFormLayout(mv).addRow("Minimum valid items", self.min_valid)
        self.cutoff = QDoubleSpinBox()
        self.cutoff.setRange(-1e12, 1e12)
        self.cutoff.setDecimals(4)
        self.strict = QCheckBox("Strictly greater than cutoff (otherwise ≥)")
        bz = QWidget()
        bf = QFormLayout(bz)
        bf.addRow("Cutoff", self.cutoff)
        bf.addRow("", self.strict)
        rc = QWidget()
        rl = QVBoxLayout(rc)
        self.recode = QTableWidget(0, 2)
        self.recode.setHorizontalHeaderLabels(["Old value", "New value (blank = missing)"])
        add = QPushButton("Add row")
        add.clicked.connect(lambda: self.recode.insertRow(self.recode.rowCount()))
        self.keep_other = QCheckBox("Keep values not listed (otherwise set them missing)")
        self.keep_other.setChecked(True)
        rl.addWidget(self.recode)
        rl.addWidget(add)
        rl.addWidget(self.keep_other)
        for w in (self.p_none, mv, bz, rc):
            self.params.addWidget(w)
        lay.addWidget(self.params)
        self.op.currentIndexChanged.connect(self._op_changed)
        self._op_changed()

        self.error = QLabel()
        self.error.setStyleSheet("color:#c62828")
        self.error.setWordWrap(True)
        lay.addWidget(self.error)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)
        self.spec: DerivedSpec | None = None

    def _op_changed(self) -> None:
        op = self.op.currentData()
        self.params.setCurrentIndex({"mean_items": 1, "binarize": 2, "recode": 3}.get(op, 0))
        self.measure.setCurrentText("nominal" if op in ("binarize", "recode") else "continuous")

    def _checked(self) -> list[str]:
        return [self.sources.item(i).data(Qt.UserRole) for i in range(self.sources.count())
                if self.sources.item(i).checkState() == Qt.Checked]

    def _accept(self) -> None:
        op = self.op.currentData()
        params: dict = {}
        if op == "mean_items":
            params["min_valid"] = self.min_valid.value()
        elif op == "binarize":
            params = {"cutoff": self.cutoff.value(), "strict": self.strict.isChecked()}
        elif op == "recode":
            mapping = []
            for r in range(self.recode.rowCount()):
                old = self.recode.item(r, 0)
                new = self.recode.item(r, 1)
                if old is None or not old.text().strip():
                    continue
                try:
                    o = float(old.text())
                    n = float(new.text()) if new is not None and new.text().strip() else None
                except ValueError:
                    self.error.setText(f"Row {r + 1}: values must be numeric codes.")
                    return
                mapping.append([o, n])
            params = {"mapping": mapping, "keep_other": self.keep_other.isChecked()}
        self.spec = DerivedSpec(name=self.name.text().strip(), op=op, sources=self._checked(), params=params,
                                label=self.label.text().strip())
        self.measure_value = Measure(self.measure.currentText())
        self.accept()


class FilterDialog(QDialog):
    def __init__(self, ds: Dataset, parent: QWidget | None = None):
        super().__init__(parent)
        self.ds = ds
        self.setWindowTitle("Add row filter")
        self.resize(480, 260)
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.var = QComboBox()
        for v in ds.variables.values():
            if v.storage is not Storage.DATETIME:
                self.var.addItem(v.display_name, v.name)
        self.op = QComboBox()
        for key, text in FILTER_OPS.items():
            self.op.addItem(text, key)
        self.value = QLineEdit()
        self.value.setPlaceholderText("value, or comma-separated values for 'is one of'")
        form.addRow("Keep rows where", self.var)
        form.addRow("", self.op)
        form.addRow("Value", self.value)
        lay.addLayout(form)
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        self.hint.setObjectName("hint")
        lay.addWidget(self.hint)
        lay.addWidget(QLabel("Rows where the variable is missing are dropped by comparisons."))
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)
        self.var.currentIndexChanged.connect(self._hint)
        self.op.currentIndexChanged.connect(lambda: self.value.setEnabled(self.op.currentData() not in VALUELESS_FILTER_OPS))
        self._hint()
        self.spec: FilterSpec | None = None

    def _hint(self) -> None:
        name = self.var.currentData()
        if not name:
            return
        v = self.ds.var(name)
        if v.is_categorical:
            levels = self.ds.levels(name)[:30]
            self.hint.setText("Codes: " + ", ".join(f"{fmt(c)} = {self.ds.level_label(name, c)}" for c in levels))
        else:
            col = self.ds.analysis_column(name).dropna()
            self.hint.setText(f"Range: {fmt(col.min())} to {fmt(col.max())}" if len(col) else "")

    def _accept(self) -> None:
        name, op = self.var.currentData(), self.op.currentData()
        numeric = self.ds.var(name).storage is Storage.NUMERIC
        raw = [p.strip() for p in self.value.text().split(",") if p.strip()]
        try:
            vals = [float(x) for x in raw] if numeric else raw
        except ValueError:
            self.hint.setText("<span style='color:#c62828'>Enter numeric codes for this variable.</span>")
            return
        if op in VALUELESS_FILTER_OPS:
            value = None
        elif op in ("in", "not_in"):
            value = vals
        else:
            value = vals[0] if vals else None
        self.spec = FilterSpec(variable=name, op=op, value=value)
        self.accept()
