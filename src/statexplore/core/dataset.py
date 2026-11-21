"""In-memory dataset: the original data (never mutated) plus metadata and a change log."""
from __future__ import annotations

import datetime as _dt
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .derive import (
    DerivedSpec, DeriveError, FilterSpec, compute, filter_mask, validate_derived,
)
from .variables import EDITABLE_FIELDS, Measure, Role, Storage, VariableMeta, make_alias


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
class HoldoutSpec:
    """Rows (or whole participants/clusters) reserved before exploration (§13.1)."""
    fraction: float
    seed: int
    group: str | None = None       # split by participant/cluster when set
    evaluations: int = 0           # how often the holdout has been used for evaluation

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
        # Derived variables (in definition order) and row filters, applied in that order.
        self.derived: dict[str, DerivedSpec] = {}
        self.filters: list[FilterSpec] = []
        self.holdout: HoldoutSpec | None = None
        self._holdout_mask: pd.Series | None = None
        self._derived_values: dict[str, pd.Series] = {}

    # ---- basic info -------------------------------------------------
    @property
    def n_rows(self) -> int:
        return len(self.raw)

    @property
    def n_cols(self) -> int:
        return len(self.variables)

    def column(self, name: str) -> pd.Series:
        """Stored values: raw column, or computed values for a derived variable."""
        if name in self.derived:
            return self._derived_values[name]
        return self.raw[name]

    def is_derived(self, name: str) -> bool:
        return name in self.derived

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
        col = self.column(name)
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
        col = self.column(name)
        mask = self.user_missing_mask(name)
        return col.mask(mask) if mask.any() else col

    def row_mask(self, part: str = "train") -> pd.Series:
        """Rows kept by all filters (True = keep). With a holdout, `part` selects
        "train" (default: everything except the holdout), "holdout", or "all"."""
        keep = pd.Series(True, index=self.raw.index)
        for f in self.filters:
            keep &= filter_mask(f, self.analysis_column(f.variable))
        if self._holdout_mask is not None and part != "all":
            keep &= self._holdout_mask if part == "holdout" else ~self._holdout_mask
        return keep

    # ---- holdout -----------------------------------------------------
    def set_holdout(self, spec: HoldoutSpec | None) -> None:
        if spec is not None:
            if not 0.05 <= spec.fraction <= 0.5:
                raise DatasetError("The holdout fraction must be between 5% and 50%.")
            if spec.group is not None:
                self.var(spec.group)
        old = self.holdout.to_dict() if self.holdout else None
        self.holdout = spec
        self._holdout_mask = self._compute_holdout() if spec else None
        self.changes.append(Change(op="set_holdout", variable="", field="", old=old,
                                   new=spec.to_dict() if spec else None, at=_now()))

    def _compute_holdout(self) -> pd.Series:
        """Deterministic split: numpy Generator(seed) permutes rows, or sorted group values."""
        h = self.holdout
        rng = np.random.default_rng(h.seed)
        if h.group is None:
            n_hold = int(round(h.fraction * self.n_rows))
            chosen = rng.permutation(self.n_rows)[:n_hold]
            mask = np.zeros(self.n_rows, bool)
            mask[chosen] = True
            return pd.Series(mask, index=self.raw.index)
        col = self.analysis_column(h.group)
        groups = np.array(sorted(col.dropna().unique().tolist(), key=str), dtype=object)
        n_hold = int(round(h.fraction * len(groups)))
        held = set(groups[rng.permutation(len(groups))[:n_hold]].tolist())
        return col.isin(held) & col.notna()

    def holdout_labels(self) -> pd.Series | None:
        if self._holdout_mask is None:
            return None
        return self._holdout_mask.map({True: "holdout", False: "train"})

    @property
    def n_rows_filtered(self) -> int:
        return int(self.row_mask().sum())

    def restore_holdout(self, spec: HoldoutSpec | None) -> None:
        self.holdout = spec
        self._holdout_mask = self._compute_holdout() if spec else None

    def analysis_frame(self, names: list[str] | None = None, filtered: bool = True,
                       part: str = "train") -> pd.DataFrame:
        """User-missing -> NaN; derived variables computed; filters (and the holdout
        split) applied unless filtered=False."""
        names = names if names is not None else self.names()
        frame = pd.DataFrame({n: self.analysis_column(n) for n in names}, index=self.raw.index)
        if not filtered or (not self.filters and self._holdout_mask is None):
            return frame
        return frame[self.row_mask(part)]

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
            op="set", variable=name, field=field_name, old=_plain(old), new=_plain(new), at=_now(),
        ))

    # ---- derived variables & filters --------------------------------
    def _next_alias(self) -> str:
        used = [int(v.alias[1:]) for v in self.variables.values()]
        return make_alias(max(used, default=0))

    def _numeric_source(self, name: str) -> bool:
        v = self.variables.get(name)
        return v is not None and v.storage is Storage.NUMERIC

    def add_derived(self, spec: DerivedSpec, measure: Measure = Measure.CONTINUOUS,
                    value_labels: dict[float, str] | None = None) -> VariableMeta:
        spec.name = spec.name.strip()
        if spec.name in self.variables:
            raise DatasetError(f"A variable named {spec.name!r} already exists.")
        if any(v.display_name == spec.name for v in self.variables.values()):
            raise DatasetError(f"Display name {spec.name!r} is already used.")
        try:
            validate_derived(spec, self._numeric_source)
        except DeriveError as e:
            raise DatasetError(str(e)) from None
        meta = VariableMeta(name=spec.name, alias=self._next_alias(), storage=Storage.NUMERIC,
                            measure=measure, label=spec.label or spec.describe(),
                            value_labels=dict(value_labels or {}), source_measure="derived")
        self._install_derived(spec, meta)
        self.changes.append(Change(op="add_derived", variable=spec.name, field="",
                                   old=None, new={"spec": spec.to_dict(), "meta": meta.to_dict()},
                                   at=_now()))
        return meta

    def _install_derived(self, spec: DerivedSpec, meta: VariableMeta) -> None:
        values = compute(spec, self.analysis_column)
        self.variables[spec.name] = meta
        self.derived[spec.name] = spec
        self._derived_values[spec.name] = values

    def _dependents(self, name: str) -> list[str]:
        deps = [d.name for d in self.derived.values() if name in d.sources]
        deps += [f.variable for f in self.filters if f.variable == name]
        return deps

    def remove_derived(self, name: str) -> None:
        if name not in self.derived:
            raise DatasetError(f"{name!r} is not a derived variable.")
        if self._dependents(name):
            raise DatasetError(f"{name!r} is used by: {', '.join(self._dependents(name))}. Remove those first.")
        spec, meta = self.derived[name], self.variables[name]
        self._uninstall_derived(name)
        self.changes.append(Change(op="remove_derived", variable=name, field="",
                                   old={"spec": spec.to_dict(), "meta": meta.to_dict()}, new=None, at=_now()))

    def _uninstall_derived(self, name: str) -> None:
        del self.derived[name], self.variables[name], self._derived_values[name]

    def add_filter(self, f: FilterSpec) -> None:
        from .derive import FILTER_OPS, VALUELESS_FILTER_OPS

        self.var(f.variable)
        if f.op not in FILTER_OPS:
            raise DatasetError(f"Unknown filter operator {f.op!r}.")
        if f.op not in VALUELESS_FILTER_OPS and (f.value is None or f.value == []):
            raise DatasetError("This filter needs a value.")
        if f.op in ("in", "not_in") and not isinstance(f.value, list):
            f.value = [f.value]
        if self.n_rows and not (self.row_mask() & filter_mask(f, self.analysis_column(f.variable))).any():
            raise DatasetError("This filter would remove every row.")
        self.filters.append(f)
        self.changes.append(Change(op="add_filter", variable=f.variable, field="", old=None,
                                   new=f.to_dict(), at=_now()))

    def remove_filter(self, index: int) -> None:
        f = self.filters.pop(index)
        self.changes.append(Change(op="remove_filter", variable=f.variable, field=str(index),
                                   old=f.to_dict(), new=None, at=_now()))

    def restore_derived(self, specs: list[DerivedSpec], metas: list[VariableMeta]) -> None:
        """Rebuild derived variables when reopening a project (no change-log entries)."""
        by_name = {m.name: m for m in metas}
        for spec in specs:
            self._install_derived(spec, by_name[spec.name])

    def undo(self) -> Change | None:
        if not self.changes:
            return None
        ch = self.changes.pop()
        if ch.op == "set":
            setattr(self.var(ch.variable), ch.field, _coerce(ch.field, ch.old))
        elif ch.op == "add_derived":
            self._uninstall_derived(ch.variable)
        elif ch.op == "remove_derived":
            self._install_derived(DerivedSpec.from_dict(ch.old["spec"]), VariableMeta.from_dict(ch.old["meta"]))
        elif ch.op == "set_holdout":
            self.holdout = HoldoutSpec(**ch.old) if ch.old else None
            self._holdout_mask = self._compute_holdout() if self.holdout else None
        elif ch.op == "add_filter":
            self.filters.pop()
        elif ch.op == "remove_filter":
            self.filters.insert(int(ch.field), FilterSpec.from_dict(ch.old))
        return ch


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _format_code(code: Any) -> str:
    if isinstance(code, float) and code.is_integer():
        return str(int(code))
    return str(code)
