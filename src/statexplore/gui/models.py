"""Qt table models over pandas data (virtual: only visible cells are formatted)."""
from __future__ import annotations

import math
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt
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


def sort_value(value: Any) -> float | str | None:
    """A comparable sort key: finite numbers as float (numpy, nullable pandas, Decimal and numeric
    strings included), other text as text, and None for anything missing or non-finite."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating, Decimal)):
        f = float(value)
        return f if math.isfinite(f) else None
    text = str(value).strip()
    if not text:
        return None
    try:
        f = float(text)
    except ValueError:
        return text
    return f if math.isfinite(f) else None


def plain(value: Any) -> Any:
    """numpy / pandas scalars as plain Python values (Qt-safe); missing as None."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, Decimal):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class NumericSortProxy(QSortFilterProxyModel):
    """Sorts by the raw value in Qt.UserRole: numbers numerically, text case-insensitively,
    numbers before text, and missing / NaN / infinite values LAST in both directions.
    Optionally shows only the source rows in `only_rows`."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSortRole(Qt.UserRole)
        self.setFilterKeyColumn(-1)
        self.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.only_rows: set[int] | None = None

    def set_only_rows(self, rows: set[int] | None) -> None:
        self.beginFilterChange()
        self.only_rows = rows
        self.endFilterChange(QSortFilterProxyModel.Direction.Rows)

    def filterAcceptsRow(self, row: int, parent: QModelIndex) -> bool:  # noqa: N802 (Qt override)
        if self.only_rows is not None and row not in self.only_rows:
            return False
        return super().filterAcceptsRow(row, parent)

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:  # noqa: N802 (Qt override)
        a = sort_value(left.data(self.sortRole()))
        b = sort_value(right.data(self.sortRole()))
        if a is None or b is None:
            if a is None and b is None:
                return False
            # Qt sorts descending by reversing lessThan; flip so missing stays at the bottom
            return (a is None) == (self.sortOrder() == Qt.DescendingOrder)
        if isinstance(a, float) != isinstance(b, float):
            return isinstance(a, float)              # numbers before text
        if isinstance(a, str):
            return a.casefold() < b.casefold()
        return a < b


HIGHLIGHT_BG = QColor(56, 142, 60, 46)    # subdued, translucent: readable in light and dark themes
WARNING_BG = QColor(245, 166, 35, 40)


class FrameModel(QAbstractTableModel):
    """Read-only, sortable view of a DataFrame (summaries, result tables). Sorting uses the raw
    value (Qt.UserRole); formatting never changes the sort key. Optional per-row status:
    `highlight` / `warning` masks give a subdued background and a tooltip."""

    def __init__(self, frame: pd.DataFrame | None = None, digits: int = 3, p_columns: tuple[str, ...] = ()):
        super().__init__()
        self.frame = frame if frame is not None else pd.DataFrame()
        self.digits = digits
        self.p_columns = set(p_columns)
        self.highlight: np.ndarray | None = None
        self.warning: np.ndarray | None = None
        self.row_tips: list[str] | None = None
        self.status_col: str | None = None       # column whose text comes from the row status
        self.status_text: list[str] | None = None

    def set_frame(self, frame: pd.DataFrame, highlight=None, warning=None, tips: list[str] | None = None) -> None:
        self.beginResetModel()
        self.frame = frame.reset_index(drop=True)
        self.highlight = None if highlight is None else np.asarray(highlight, bool)
        self.warning = None if warning is None else np.asarray(warning, bool)
        self.row_tips = tips
        self.endResetModel()

    def set_row_status(self, highlight=None, warning=None, tips: list[str] | None = None,
                       text: list[str] | None = None) -> None:
        """Change only the row styling (no reset: sort, filter and selection are untouched)."""
        self.highlight = None if highlight is None else np.asarray(highlight, bool)
        self.warning = None if warning is None else np.asarray(warning, bool)
        self.row_tips = tips
        self.status_text = text
        if len(self.frame):
            self.dataChanged.emit(self.index(0, 0), self.index(len(self.frame) - 1, self.frame.shape[1] - 1),
                                  [Qt.BackgroundRole, Qt.ToolTipRole, Qt.DisplayRole])

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.frame)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else self.frame.shape[1]

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if not index.isValid():
            return None
        r = index.row()
        if role == Qt.BackgroundRole:
            if self.highlight is not None and r < len(self.highlight) and self.highlight[r]:
                return HIGHLIGHT_BG
            if self.warning is not None and r < len(self.warning) and self.warning[r]:
                return WARNING_BG
            return None
        if role == Qt.ToolTipRole and self.row_tips is not None and r < len(self.row_tips):
            return self.row_tips[r] or None
        if self.status_text is not None and self.frame.columns[index.column()] == self.status_col:
            if role in (Qt.DisplayRole, Qt.UserRole):
                return self.status_text[r] if r < len(self.status_text) else ""
            return None
        value = plain(self.frame.iat[r, index.column()])
        if role == Qt.DisplayRole:
            if self.frame.columns[index.column()] in self.p_columns and isinstance(value, (int, float)) \
                    and not isinstance(value, bool):
                return f"{value:.2e}" if value < 1e-4 else f"{value:.4f}"
            return fmt(value, self.digits)
        if role == Qt.UserRole:  # raw (unrounded) value for sorting
            return value
        if role == Qt.TextAlignmentRole and isinstance(value, (int, float)) and not isinstance(value, bool):
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
