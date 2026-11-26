"""Fixed-effect terms editor (v1.3.3, 01 §2): every main effect and interaction of the model is an individual, visible,
checkable term. Presets only materialize term sets; an imported (e.g. transferred) set that matches no preset shows as
Custom with each of its terms. The editor never fits anything and never changes terms on its own: the page decides."""
from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget


class FixedTermsEditor(QWidget):
    changed = Signal()          # the user (not the program) changed the checked terms

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.list = QListWidget()
        self.list.setMaximumHeight(150)
        self.list.setToolTip("The exact fixed-effect terms that will be fitted. Tick an interaction to add it; lower-order "
                             "terms an interaction needs must stay ticked (hierarchy).")
        lay.addWidget(self.list)
        row = QHBoxLayout()
        self.note = QLabel()
        self.note.setWordWrap(True)
        self.note.setObjectName("hint")
        row.addWidget(self.note, 1)
        self.undo_btn = QPushButton("Undo")
        self.undo_btn.setToolTip("Restore the terms as they were before the last preset or variable change")
        self.undo_btn.setEnabled(False)
        self.undo_btn.clicked.connect(self.undo)
        row.addWidget(self.undo_btn)
        lay.addLayout(row)
        self._undo: list[list[str]] | None = None
        self._loading = False
        self._label: Callable[[str], str] = lambda v: v
        self.list.itemChanged.connect(self._item_changed)

    # ---- content ----
    def set_universe(self, universe: list[list[str]], checked: list[list[str]], label: Callable[[str], str]) -> None:
        """Offer `universe` (main effects first), ticking exactly `checked`; programmatic, emits nothing."""
        self._label = label
        want = {frozenset(t) for t in checked}
        self._loading = True
        self.list.clear()
        seen = set()
        for t in list(universe) + [t for t in checked if frozenset(t) not in {frozenset(u) for u in universe}]:
            key = frozenset(t)
            if key in seen:
                continue
            seen.add(key)
            it = QListWidgetItem(" × ".join(label(v) for v in t))
            it.setData(Qt.UserRole, list(t))
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if key in want else Qt.Unchecked)
            self.list.addItem(it)
        self._loading = False

    def terms(self) -> list[list[str]]:
        return [list(self.list.item(i).data(Qt.UserRole)) for i in range(self.list.count())
                if self.list.item(i).checkState() == Qt.Checked]

    def set_checked(self, terms: list[list[str]], remember: bool = True) -> None:
        """Tick exactly `terms` (programmatic). With `remember`, the previous set can be restored with Undo."""
        if remember:
            self._undo = self.terms()
            self.undo_btn.setEnabled(True)
        want = {frozenset(t) for t in terms}
        self._loading = True
        for i in range(self.list.count()):
            it = self.list.item(i)
            it.setCheckState(Qt.Checked if frozenset(it.data(Qt.UserRole)) in want else Qt.Unchecked)
        self._loading = False

    def undo(self) -> None:
        if self._undo is None:
            return
        prev, self._undo = self._undo, None
        self.set_checked(prev, remember=False)
        self.undo_btn.setEnabled(False)
        self.note.setText("Previous terms restored.")
        self.changed.emit()

    def _item_changed(self, _it: QListWidgetItem) -> None:
        if not self._loading:
            self.changed.emit()
