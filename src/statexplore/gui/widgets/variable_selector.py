"""Reusable variable selector (v1.1 §7).

A searchable, checkable list with Check visible / Clear visible / Check all / Clear all and a
selected count that includes selections hidden by the current search. Searching never
changes the selection. `selected()` returns variables in the order they were checked (EDA
uses this for x-then-y plots).

`self.list` is the underlying QListWidget; item check states are the source of truth, so
code that walks the items keeps working. With `selected_first` (UI-31) checked variables are
regrouped under a "Selected" header above the others; header rows carry no variable name
(`data(Qt.UserRole)` is None) and are skipped by every method here. Regrouping only changes the
display: `selected()` keeps the check order.
"""
from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget,
)

from ...core.dataset import Dataset

ALL_LEVELS = "All measurement levels"
PIN = "★ "
HEADER = Qt.UserRole + 2      # section header rows (selected_first mode)
BASE_TEXT = Qt.UserRole + 3   # item text without the order position prefix
ORDER_TIP = ("The order of the selected candidates is the search order: it defines candidate IDs, the enumeration and "
             "the preview.\nAn exhaustive search evaluates the same set of models in any order (same results); a "
             "budget-limited, early-stopping or cancelled search evaluates a prefix, so the order can change which "
             "models are evaluated.\nReordering starts a NEW search identity; earlier results and resumable searches "
             "are never changed.")


def _pinned() -> set[str]:
    """Variables pinned by the user (remembered across sessions; shared by every selector)."""
    from PySide6.QtCore import QSettings

    v = QSettings("StatExplore", "StatExplore").value("pinned_variables") or []
    return set([v] if isinstance(v, str) else v)


def _set_pinned(names: set[str]) -> None:
    from PySide6.QtCore import QSettings

    QSettings("StatExplore", "StatExplore").setValue("pinned_variables", sorted(names))


class VariableSelector(QWidget):
    changed = Signal()

    def __init__(self, parent: QWidget | None = None, *, single: bool = False, max_height: int | None = None,
                 placeholder: str = "Search name, label or original name…", show_level_filter: bool = True,
                 selected_first: bool = True, orderable: bool = False):
        super().__init__(parent)
        self.orderable = orderable and not single      # V134-013: explicit, visible candidate order
        self.single = single
        self.selected_first = selected_first and not single   # UI-31: every multi-select list groups selections first
        self._regroup_timer = QTimer(self)
        self._regroup_timer.setSingleShot(True)
        self._regroup_timer.setInterval(0)
        self._regroup_timer.timeout.connect(self._regroup)
        self._order: list[str] = []
        self._decorate: Callable[[str], str] | None = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(3)
        top = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText(placeholder)
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._apply_filter)
        top.addWidget(self.search, 1)
        self.level = QComboBox()
        self.level.addItem(ALL_LEVELS, None)
        for m in ("continuous", "ordinal", "nominal", "binary"):
            self.level.addItem(m, m)
        self.level.setToolTip("Show only variables with this project measurement level")
        self.level.currentIndexChanged.connect(self._apply_filter)
        self.level.setVisible(show_level_filter)
        top.addWidget(self.level)
        lay.addLayout(top)
        self.list = QListWidget()
        if max_height:
            self.list.setMaximumHeight(max_height)
        self.list.itemChanged.connect(self._item_changed)
        self.list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._pin_menu)
        self.list.setToolTip("Right-click a variable to pin it to the top of every variable list.")
        lay.addWidget(self.list, 1)
        bar = QHBoxLayout()
        self.buttons: dict[str, QPushButton] = {}
        for key, text, fn in (("check_visible", "Check visible", self.check_visible),
                              ("clear_visible", "Clear visible", self.clear_visible),
                              ("check_all", "Check all", self.check_all),
                              ("clear_all", "Clear all", self.clear_all)):
            b = QPushButton(text)
            b.setAutoDefault(False)
            b.clicked.connect(fn)
            b.setVisible(not (single and key in ("check_visible", "check_all")))
            self.buttons[key] = b
            bar.addWidget(b)
        bar.addStretch()
        lay.addLayout(bar)
        if self.orderable:
            from PySide6.QtGui import QKeySequence, QShortcut
            from PySide6.QtWidgets import QAbstractItemView

            from PySide6.QtWidgets import QToolButton

            obar = QHBoxLayout()
            obar.setSpacing(2)
            obar.addWidget(QLabel("Order:"))
            for key, text, tip, fn in (
                    ("up", "▲", "Move the current selected variable up (Ctrl+Up)", lambda: self.move_current(-1)),
                    ("down", "▼", "Move the current selected variable down (Ctrl+Down)", lambda: self.move_current(1)),
                    ("top", "⤒", "Move the current selected variable to the top (Ctrl+Home)", lambda: self.move_current(-10**9)),
                    ("bottom", "⤓", "Move the current selected variable to the bottom (Ctrl+End)",
                     lambda: self.move_current(10**9)),
                    ("sort", "A–Z", "Sort the selected variables alphabetically by display name", self.sort_alpha),
                    ("restore", "Data order", "Restore the dataset's column order for the selected variables",
                     self.restore_dataset_order)):
                b = QToolButton()
                b.setText(text)
                b.setToolTip(tip + "\n\n" + ORDER_TIP)
                b.setAccessibleName(tip)
                b.clicked.connect(fn)
                self.buttons[key] = b
                obar.addWidget(b)
            obar.addStretch()
            lay.addLayout(obar)
            for seq, d in (("Ctrl+Up", -1), ("Ctrl+Down", 1), ("Ctrl+Home", -10**9), ("Ctrl+End", 10**9)):
                sc = QShortcut(QKeySequence(seq), self.list)
                sc.setContext(Qt.WidgetShortcut)
                sc.activated.connect(lambda d=d: self.move_current(d))
            self.list.setDragDropMode(QAbstractItemView.InternalMove)
            self.list.setDefaultDropAction(Qt.MoveAction)
            self.list.model().rowsMoved.connect(lambda *_: self._dropped())
            self.list.setToolTip(self.list.toolTip() + "\nDrag selected variables (or use the Order buttons) to change "
                                 "the search order.")
        self.count = QLabel()          # own line (UI-33): beside the buttons it forced a ~390 px minimum width
        self.count.setWordWrap(True)
        lay.addWidget(self.count)
        self._levels: dict[str, str] = {}
        self._update_count()

    # ---- content ----------------------------------------------------------------
    def set_variables(self, ds: Dataset | None, names: list[str], keep: bool = True,
                      decorate: Callable[[str], str] | None = None, default_checked: bool = False) -> None:
        """Fill the list. With `keep`, previously checked variables stay checked."""
        previous = [n for n in self._order if n in names] if keep else []
        self._decorate = decorate
        self.list.blockSignals(True)
        self.list.clear()
        self._levels = {}
        self._ds, self._names = ds, list(names)
        pins = _pinned()
        names = [n for n in names if n in pins] + [n for n in names if n not in pins]
        for n in names:
            v = ds.var(n) if ds is not None else None
            text = (PIN if n in pins else "") + (v.display_name if v else n) + (decorate(n) if decorate else "")
            it = QListWidgetItem(text)
            it.setData(BASE_TEXT, text)
            it.setData(Qt.UserRole, n)
            if v is not None:
                self._levels[n] = v.measure.value
                parts = [v.display_name]
                if v.name != v.display_name:
                    parts.append(f"original name: {v.name}")
                if v.label:
                    parts.append(v.label)
                parts.append(f"level: {v.measure.value}" + (f" (source: {v.source_measure})" if v.source_measure else ""))
                it.setToolTip("\n".join(parts))
                it.setData(Qt.UserRole + 1, " ".join([v.display_name, v.name, v.label]).lower())
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable | (Qt.ItemIsDragEnabled if self.orderable else Qt.NoItemFlags))
            it.setCheckState(Qt.Checked if (n in previous or (default_checked and not keep)) else Qt.Unchecked)
            self.list.addItem(it)
        self.list.blockSignals(False)
        self._order = previous + ([n for n in names if default_checked and not keep and n not in previous])
        self._base_order = names
        if self.selected_first:
            self._regroup()
        self._apply_filter()
        self.changed.emit()

    def _pin_menu(self, pos) -> None:
        from PySide6.QtWidgets import QMenu

        it = self.list.itemAt(pos)
        if it is None or it.data(HEADER):
            return
        name = it.data(Qt.UserRole)
        pins = _pinned()
        menu = QMenu(self)
        act = menu.addAction("Unpin" if name in pins else "Pin to top")
        if menu.exec(self.list.mapToGlobal(pos)) is act:
            pins.symmetric_difference_update({name})
            _set_pinned(pins)
            checked = self.selected()
            self.set_variables(self._ds, self._names, keep=False, decorate=self._decorate)
            self.set_selected(checked)

    def items(self) -> list[QListWidgetItem]:
        """Variable rows (section headers excluded)."""
        return [it for it in (self.list.item(i) for i in range(self.list.count())) if not it.data(HEADER)]

    def names(self) -> list[str]:
        return [it.data(Qt.UserRole) for it in self.items()]

    # ---- selection ----------------------------------------------------------------
    def selected(self) -> list[str]:
        checked = {it.data(Qt.UserRole) for it in self.items() if it.checkState() == Qt.Checked}
        return [n for n in self._order if n in checked] + [n for n in self.names() if n in checked and n not in self._order]

    def set_selected(self, names: list[str]) -> None:
        wanted = list(dict.fromkeys(names))
        if self.single:
            wanted = wanted[:1]
        self.list.blockSignals(True)
        for it in self.items():
            it.setCheckState(Qt.Checked if it.data(Qt.UserRole) in wanted else Qt.Unchecked)
        self.list.blockSignals(False)
        self._order = [n for n in wanted if n in self.names()]
        self._schedule_regroup()
        self._update_count()
        self.changed.emit()

    # ---- explicit order (V134-013) ------------------------------------------------------
    def set_order(self, names: list[str]) -> None:
        """Reorder the selected variables (unknown / unchecked names are ignored; none is lost)."""
        sel = self.selected()
        new = [n for n in dict.fromkeys(names) if n in sel]
        self._order = new + [n for n in sel if n not in new]
        self._regroup() if self.selected_first else None
        self._update_count()
        self.changed.emit()

    def move_current(self, delta: int) -> None:
        it = self.list.currentItem()
        if it is None or it.data(HEADER) or it.checkState() != Qt.Checked:
            return
        name = it.data(Qt.UserRole)
        order = self.selected()
        i = order.index(name)
        j = max(0, min(len(order) - 1, i + delta))
        if i == j:
            return
        order.insert(j, order.pop(i))
        self.set_order(order)
        self.list.setCurrentItem(next(x for x in self.items() if x.data(Qt.UserRole) == name))

    def sort_alpha(self) -> None:
        ds = getattr(self, "_ds", None)
        key = (lambda n: ds.var(n).display_name.casefold()) if ds is not None else str.casefold
        self.set_order(sorted(self.selected(), key=key))

    def restore_dataset_order(self) -> None:
        pos = {n: i for i, n in enumerate(getattr(self, "_names", []))}
        self.set_order(sorted(self.selected(), key=lambda n: pos.get(n, len(pos))))

    def _dropped(self) -> None:
        # the visual order of the checked rows after a drag is the new order
        QTimer.singleShot(0, lambda: self.set_order([it.data(Qt.UserRole) for it in self.items()
                                                     if it.checkState() == Qt.Checked]))

    def _set_items(self, items: list[QListWidgetItem], state: Qt.CheckState) -> None:
        self.list.blockSignals(True)
        for it in items:
            it.setCheckState(state)
        self.list.blockSignals(False)
        self._sync_order()
        self._schedule_regroup()
        self._update_count()
        self.changed.emit()

    def visible_items(self) -> list[QListWidgetItem]:
        return [it for it in self.items() if not it.isHidden()]

    def check_visible(self) -> None:
        self._set_items(self.visible_items(), Qt.Checked)

    def clear_visible(self) -> None:
        self._set_items(self.visible_items(), Qt.Unchecked)

    def check_all(self) -> None:
        self._set_items(self.items(), Qt.Checked)

    def clear_all(self) -> None:
        self._set_items(self.items(), Qt.Unchecked)

    def _item_changed(self, it: QListWidgetItem) -> None:
        if it.data(HEADER):
            return
        name = it.data(Qt.UserRole)
        if it.checkState() == Qt.Checked:
            if self.single:
                self.list.blockSignals(True)
                for i in range(self.list.count()):
                    other = self.list.item(i)
                    if other is not it and other.checkState() == Qt.Checked:
                        other.setCheckState(Qt.Unchecked)
                self.list.blockSignals(False)
                self._order = []
            if name not in self._order:
                self._order.append(name)
        elif name in self._order:
            self._order.remove(name)
        self._schedule_regroup()
        self._update_count()
        self.changed.emit()

    def _sync_order(self) -> None:
        checked = [it.data(Qt.UserRole) for it in self.items() if it.checkState() == Qt.Checked]
        self._order = [n for n in self._order if n in checked] + [n for n in checked if n not in self._order]

    # ---- filtering --------------------------------------------------------------------
    def set_filter(self, text: str) -> None:
        self.search.setText(text)

    # ---- selected-first grouping (UI-31) ---------------------------------------------------
    def _schedule_regroup(self) -> None:
        # deferred: never reorder the list inside its own itemChanged signal or under a loop over its items
        if self.selected_first:
            self._regroup_timer.start()

    def _regroup(self) -> None:
        """Checked variables first (in check order) under a 'Selected (k)' header, then the rest in their usual
        order under 'Not selected (m)'. The scroll position and current row are kept."""
        if not self.selected_first:
            return
        bar = self.list.verticalScrollBar()
        pos = bar.value() if bar else 0
        cur = self.list.currentItem()
        cur_name = cur.data(Qt.UserRole) if cur is not None else None
        self.list.blockSignals(True)
        rows = {}
        while self.list.count():
            it = self.list.takeItem(0)
            if not it.data(HEADER):
                rows[it.data(Qt.UserRole)] = it
        checked = [n for n in self.selected_order(rows) if n in rows]
        rest = [n for n in getattr(self, "_base_order", list(rows)) if n in rows and n not in checked]
        rest += [n for n in rows if n not in checked and n not in rest]
        if self.orderable:          # visible order positions of the selected candidates
            for k, n in enumerate(checked, start=1):
                rows[n].setText(f"{k}. " + (rows[n].data(BASE_TEXT) or rows[n].text()))
            for n in rest:
                rows[n].setText(rows[n].data(BASE_TEXT) or rows[n].text())
        for title, group in ((f"Selected ({len(checked)})", checked), (f"Not selected ({len(rest)})", rest)):
            h = QListWidgetItem(title)
            h.setData(HEADER, True)
            h.setFlags(Qt.ItemIsDropEnabled if self.orderable else Qt.NoItemFlags)
            f = h.font()
            f.setBold(True)
            h.setFont(f)
            h.setToolTip("Display grouping only: it does not change the predictor set, the formula or the search order.")
            self.list.addItem(h)
            for n in group:
                self.list.addItem(rows[n])
        self.list.blockSignals(False)
        if cur_name is not None and cur_name in rows:
            self.list.setCurrentItem(rows[cur_name])
        self._apply_filter()
        if bar:
            bar.setValue(pos)

    def selected_order(self, rows: dict) -> list[str]:
        checked = {n for n, it in rows.items() if it.checkState() == Qt.Checked}
        return [n for n in self._order if n in checked] + [n for n in rows if n in checked and n not in self._order]

    def _apply_filter(self, *_args) -> None:
        text = self.search.text().strip().lower()
        level = self.level.currentData()
        for i in range(self.list.count()):
            it = self.list.item(i)
            if it.data(HEADER):
                continue
            hay = it.data(Qt.UserRole + 1) or it.text().lower()
            hide = (bool(text) and text not in hay and text not in it.text().lower()) or \
                   (level is not None and self._levels.get(it.data(Qt.UserRole)) != level)
            it.setHidden(hide)
        self._update_count()

    def _update_count(self) -> None:
        sel = self.selected()
        hidden = sum(1 for it in self.items() if it.isHidden() and it.checkState() == Qt.Checked)
        txt = f"{len(sel)} of {len(self.items())} selected"
        if hidden:
            txt += f" ({hidden} hidden by the search)"
        self.count.setText(txt)
