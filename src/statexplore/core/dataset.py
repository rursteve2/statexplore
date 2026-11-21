"""In-memory dataset: the original data (never mutated) plus metadata and a change log."""
from __future__ import annotations

import datetime as _dt
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .variables import EDITABLE_FIELDS, Measure, Role, VariableMeta


class DatasetError(Exception):
    pass


@dataclass
class SourceInfo:
    path: str
    format: str  # "csv" | "sav" | "xlsx"
    options: dict[str, Any] = field(default_factory=dict)
    sha256: str = ""
    size_bytes: int = 0
    imported_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Change:
    op: str
    variable: str
    field: str
    old: Any
    new: Any
    at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce(field_name: str, value: Any) -> Any:
    if field_name == "measure":
        return Measure(value)
    if field_name == "role":
        return Role(value)
    return value


def _plain(value: Any) -> Any:
    """Enum -> value so the log is JSON-serialisable."""
    return value.value if isinstance(value, (Measure, Role)) else value


class Dataset:
    """Holds the raw imported frame and its variable metadata.

    The raw frame is treated as immutable: user edits only touch metadata and
    are recorded in `changes`, so every edit is reversible and the original
    values (including SPSS user-missing codes) are always preserved.
    """

    def __init__(self, raw: pd.DataFrame, variables: list[VariableMeta], source: SourceInfo):
        if list(raw.columns) != [v.name for v in variables]:
            raise DatasetError("Variable metadata does not match data columns.")
        self.raw = raw
        self.variables: dict[str, VariableMeta] = {v.name: v for v in variables}
        self.source = source
        self.changes: list[Change] = []

    # ---- basic info -------------------------------------------------
    @property
    def n_rows(self) -> int:
        return len(self.raw)

    @property
    def n_cols(self) -> int:
        return self.raw.shape[1]

    def var(self, name: str) -> VariableMeta:
        try:
            return self.variables[name]
        except KeyError:
            raise DatasetError(f"Unknown variable: {name!r}") from None

    def by_alias(self, alias: str) -> VariableMeta:
        for v in self.variables.values():
            if v.alias == alias:
                return v
        raise DatasetError(f"Unknown alias: {alias!r}")

    def names(self, include_excluded: bool = False) -> list[str]:
        return [
            v.name for v in self.variables.values()
            if include_excluded or v.role is not Role.EXCLUDED
        ]

    # ---- missing handling -------------------------------------------
    def user_missing_mask(self, name: str) -> pd.Series:
        """True where the raw value is a declared (SPSS) user-missing code."""
        v = self.var(name)
        col = self.raw[name]
        mask = pd.Series(False, index=col.index)
        if not v.has_user_missing():
            return mask
        if v.missing_values:
            mask |= col.isin(v.missing_values)
        if v.missing_ranges and pd.api.types.is_numeric_dtype(col):
            for lo, hi in v.missing_ranges:
                mask |= (col >= lo) & (col <= hi)
        return mask

    def analysis_column(self, name: str) -> pd.Series:
        """Column with user-missing codes converted to NaN (raw data untouched)."""
        col = self.raw[name]
        mask = self.user_missing_mask(name)
        return col.mask(mask) if mask.any() else col

    def analysis_frame(self, names: list[str] | None = None) -> pd.DataFrame:
        names = names if names is not None else self.names()
        return pd.DataFrame({n: self.analysis_column(n) for n in names}, index=self.raw.index)

    def missing_count(self, name: str) -> int:
        return int(self.analysis_column(name).isna().sum())

    # ---- levels ------------------------------------------------------
    def levels(self, name: str) -> list[Any]:
        """Observed non-missing codes in sorted order, plus labelled-but-unobserved codes."""
        col = self.analysis_column(name).dropna()
        v = self.var(name)
        values = set(col.unique().tolist())
        values.update(k for k in v.value_labels if not self.is_user_missing_value(v, k))
        try:
            return sorted(values)
        except TypeError:
            return sorted(values, key=str)

    @staticmethod
    def is_user_missing_value(v: VariableMeta, value: Any) -> bool:
        if value in v.missing_values:
            return True
        if isinstance(value, (int, float, np.floating)):
            return any(lo <= value <= hi for lo, hi in v.missing_ranges)
        return False

    def level_label(self, name: str, code: Any) -> str:
        v = self.var(name)
        if code in v.value_labels:
            return str(v.value_labels[code])
        return _format_code(code)

    # ---- edits -------------------------------------------------------
    def set_variable_field(self, name: str, field_name: str, value: Any) -> None:
        if field_name not in EDITABLE_FIELDS:
            raise DatasetError(f"Field {field_name!r} is not editable.")
        v = self.var(name)
        new = _coerce(field_name, value)
        if field_name == "display_name":
            new = str(new).strip()
            if not new:
                raise DatasetError("Display name cannot be empty.")
            clash = [o for o in self.variables.values() if o.name != name and o.display_name == new]
            if clash:
                raise DatasetError(f"Display name {new!r} is already used by {clash[0].name!r}.")
        if field_name == "reference_level" and new is not None and new not in self.levels(name):
            raise DatasetError(f"{new!r} is not a level of {name!r}.")
        old = getattr(v, field_name)
        if old == new:
            return
        setattr(v, field_name, new)
        self.changes.append(Change(
            op="set", variable=name, field=field_name, old=_plain(old), new=_plain(new),
            at=_dt.datetime.now().isoformat(timespec="seconds"),
        ))

    def undo(self) -> Change | None:
        if not self.changes:
            return None
        ch = self.changes.pop()
        setattr(self.var(ch.variable), ch.field, _coerce(ch.field, ch.old))
        return ch


def _format_code(code: Any) -> str:
    if isinstance(code, float) and code.is_integer():
        return str(int(code))
    return str(code)
