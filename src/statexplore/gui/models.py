"""Qt table models over pandas data (virtual: only visible cells are formatted)."""
from __future__ import annotations

import math
from typing import Any

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui import QColor

from ..core.dataset import Dataset
from ..core.schema import Table


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        return f"{value:.{digits}f}"
    if value is pd.NA or value is pd.NaT:
        return ""
    return str(value)


class DataPreviewModel(QAbstractTableModel):
    """Spreadsheet-style view of the raw data; optional value-label display."""

    def __init__(self, ds: Dataset | None = None):
        super().__init__()
        self.ds = ds
        self.show_labels = False
        self._cols: list[str] = []
        self._keep = None  # row filter mask; filtered-out rows are greyed
        self._reset_columns()

    def set_dataset(self, ds: Dataset | None) -> None:
        self.beginResetModel()
        self.ds = ds
        self._reset_columns()
        self.endResetModel()

    def set_show_labels(self, on: bool) -> None:
        self.beginResetModel()
        self.show_labels = on
        self.endResetModel()

    def refresh_headers(self) -> None:
        self.headerDataChanged.emit(Qt.Horizontal, 0, max(0, len(self._cols) - 1))

    def _reset_columns(self) -> None:
        self._cols = list(self.ds.variables) if self.ds else []
        self._keep = self.ds.row_mask().to_numpy() if self.ds is not None and self.ds.filters else None

    def refresh_all(self) -> None:
        self.beginResetModel()
        self._reset_columns()
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() or self.ds is None else self.ds.n_rows

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._cols)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if self.ds is None or not index.isValid():
            return None
        name = self._cols[index.column()]
        value = self.ds.column(name).iat[index.row()]
        if role == Qt.DisplayRole:
            if pd.isna(value):
                return ""
            v = self.ds.variables[name]
            if self.show_labels and value in v.value_labels:
                return v.value_labels[value]
            return fmt(value, 6)
        if self._keep is not None and not self._keep[index.row()]:
            if role == Qt.ForegroundRole:
                return QColor("#9e9e9e")
            if role == Qt.ToolTipRole:
                return "Row excluded by the active filters"
        if role in (Qt.ForegroundRole, Qt.ToolTipRole) and not pd.isna(value):
            v = self.ds.variables[name]
            if v.has_user_missing() and self.ds.is_user_missing_value(v, value):
                return QColor("#c62828") if role == Qt.ForegroundRole else \
                    "User-missing code (treated as missing in analyses)"
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole) -> Any:
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal and self.ds:
                return self.ds.variables[self._cols[section]].display_name
            if orientation == Qt.Vertical:
                return str(section + 1)
        if role == Qt.ToolTipRole and orientation == Qt.Horizontal and self.ds:
            v = self.ds.variables[self._cols[section]]
            return f"{v.name}\n{v.label}" if v.label else v.name
        return None


class FrameModel(QAbstractTableModel):
    """Read-only, sortable view of a small DataFrame (summaries, result tables)."""

    def __init__(self, frame: pd.DataFrame | None = None, digits: int = 3):
        super().__init__()
        self.frame = frame if frame is not None else pd.DataFrame()
        self.digits = digits

    def set_frame(self, frame: pd.DataFrame) -> None:
        self.beginResetModel()
        self.frame = frame.reset_index(drop=True)
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.frame)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else self.frame.shape[1]

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if not index.isValid():
            return None
        value = self.frame.iat[index.row(), index.column()]
        if role == Qt.DisplayRole:
            return fmt(value, self.digits)
        if role == Qt.UserRole:  # raw value for sorting
            return value
        if role == Qt.TextAlignmentRole and isinstance(value, (int, float)):
            return int(Qt.AlignRight | Qt.AlignVCenter)
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole) -> Any:
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return str(self.frame.columns[section])
        return str(section + 1)


def table_to_frame(t: Table) -> pd.DataFrame:
    df = pd.DataFrame(t.rows, columns=t.columns)
    if t.row_names:
        df.insert(0, "variable", t.row_names)
    return df
