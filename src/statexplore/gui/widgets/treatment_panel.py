"""Per-analysis "treat as continuous" controls (v1.1 §6).

One checkbox per selected numeric categorical/ordinal variable. Checking it asks for the
§6.2 acknowledgement (and, for variables whose project level is nominal, a second explicit
confirmation that the codes are meaningful numbers). The panel only produces
VariableTreatment objects; it never changes the dataset.
"""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QGroupBox, QLabel, QVBoxLayout, QWidget,
)

from ...analysis import measurement as M
from ...core.dataset import Dataset
from ...core.variables import Storage

CHECKBOX_TEXT = "Treat {name} as continuous for this analysis"


def _fmt(code) -> str:
    return str(int(code)) if isinstance(code, float) and code.is_integer() else str(code)


def eligible_for_continuous(ds: Dataset, name: str) -> bool:
    """Numeric categorical variables with valid numeric codes (§6: only offer when valid)."""
    v = ds.var(name)
    return v.is_categorical and v.storage is Storage.NUMERIC and not M.continuous_problem(ds, name)


class AcknowledgeDialog(QDialog):
    """Shows the codes, the §6.2 warning, and the required confirmations."""

    def __init__(self, ds: Dataset, name: str, parent: QWidget | None = None):
        super().__init__(parent)
        v = ds.var(name)
        self.setWindowTitle(f"Treat {v.display_name} as continuous")
        lay = QVBoxLayout(self)
        codes = ds.levels(name)
        listing = "<br>".join(f"{_fmt(c)} = {ds.level_label(name, c)}" for c in codes[:20])
        if len(codes) > 20:
            listing += f"<br>… {len(codes) - 20} more"
        info = QLabel(
            f"<b>{v.display_name}</b> — project level: {v.measure.value}"
            + (f"; source file: {v.source_measure}" if v.source_measure else "")
            + f"<br><br>The analysis will use these numeric codes as values:<br>{listing}"
            + (f"<br><br>Direction: {v.coding_direction}." if v.coding_direction else "")
            + f"<br><br><i>{M.EQUAL_INTERVALS}</i>")
        info.setWordWrap(True)
        lay.addWidget(info)
        self.ack = QCheckBox(M.CONTINUOUS_WARNING)
        self.ack.setStyleSheet("QCheckBox { font-weight: bold }")
        lay.addWidget(self.ack)
        self.confirm = None
        if M.needs_numeric_confirmation(ds, name):
            self.confirm = QCheckBox("I confirm these codes are meaningful numbers, not arbitrary category "
                                     "identifiers. " + M.NOMINAL_NUMERIC_WARNING)
            lay.addWidget(self.confirm)
        for cb in (self.ack, self.confirm):
            if cb:
                cb.setToolTip(cb.text())
                cb.toggled.connect(self._update)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(self.buttons)
        self._update()

    def _update(self) -> None:
        ok = self.ack.isChecked() and (self.confirm is None or self.confirm.isChecked())
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(ok)


class TreatmentPanel(QGroupBox):
    changed = Signal()

    def __init__(self, parent: QWidget | None = None, title: str = "Variable treatment (this analysis only)"):
        super().__init__(title, parent)
        self.ds: Dataset | None = None
        self._lay = QVBoxLayout(self)
        self._boxes: dict[str, QCheckBox] = {}
        self._acks: dict[str, M.VariableTreatment] = {}
        self._hint = QLabel("Select a numeric categorical or ordinal variable to see its options.")
        self._hint.setObjectName("hint")
        self._hint.setWordWrap(True)
        self._lay.addWidget(self._hint)
        self.ask = True          # tests set False and call set_treated()

    def set_variables(self, ds: Dataset | None, names: list[str]) -> None:
        """Offer the checkbox for every eligible variable among `names` (keeps earlier decisions)."""
        self.ds = ds
        wanted = [n for n in dict.fromkeys(names) if ds is not None and n in ds.variables
                  and eligible_for_continuous(ds, n)]
        for n in list(self._boxes):
            if n not in wanted:
                self._boxes.pop(n).setParent(None)
        for n in wanted:
            if n not in self._boxes:
                cb = QCheckBox(CHECKBOX_TEXT.format(name=ds.var(n).display_name))
                cb.setChecked(n in self._acks)
                cb.toggled.connect(lambda on, name=n: self._toggled(name, on))
                self._boxes[n] = cb
                self._lay.addWidget(cb)
        self._hint.setVisible(not self._boxes)
        self.setVisible(True)

    def _toggled(self, name: str, on: bool) -> None:
        if on and name not in self._acks:
            if self.ask:
                dlg = AcknowledgeDialog(self.ds, name, self)
                if dlg.exec() != QDialog.Accepted:
                    self._boxes[name].blockSignals(True)
                    self._boxes[name].setChecked(False)
                    self._boxes[name].blockSignals(False)
                    return
            self._acks[name] = M.VariableTreatment(
                name, "continuous", acknowledged=True,
                numeric_confirmed=M.needs_numeric_confirmation(self.ds, name))
        elif not on:
            self._acks.pop(name, None)
        self.changed.emit()

    def set_treated(self, name: str, on: bool = True) -> None:
        """Programmatic check (the caller has collected the acknowledgement)."""
        self._boxes[name].setChecked(on)

    def treated(self) -> list[str]:
        return [n for n, cb in self._boxes.items() if cb.isChecked() and n in self._acks]

    def treatments(self) -> list[M.VariableTreatment]:
        return [self._acks[n] for n in self.treated()]

    def reset(self) -> None:
        self._acks.clear()
        for cb in self._boxes.values():
            cb.blockSignals(True)
            cb.setChecked(False)
            cb.blockSignals(False)
