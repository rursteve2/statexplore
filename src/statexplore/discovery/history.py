"""Historical searches stay reproducible after the project changes (v1.3.2 STAT-39, H5).

Every new search stores a snapshot of the data view it used (source fingerprint, variable metadata, derived variables,
filters, holdout). A historical candidate is refitted only on that view: when the current project differs, the
difference is named and the user chooses between rebuilding the search's original view (possible only when the source
data are byte-identical) and an explicitly NEW analysis under the current settings, which is never labelled as the
historical model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.dataset import Dataset


def snapshot(ds: Dataset) -> dict[str, Any]:
    return {"schema": 1, "source_sha256": ds.source.sha256,
            "variables": [v.to_dict() for v in ds.variables.values()],
            "derived": [d.to_dict() for d in ds.derived.values()],
            "filters": [f.to_dict() for f in ds.filters],
            "holdout": {k: v for k, v in (ds.holdout.to_dict() if ds.holdout else {}).items() if k != "evaluations"}
            or None}


def rebuild(ds: Dataset, snap: dict[str, Any]) -> Dataset:
    """The search's original data view on the current raw data (which must be the same bytes)."""
    from ..core.dataset import HoldoutSpec, SourceInfo
    from ..core.derive import DerivedSpec, FilterSpec
    from ..core.variables import VariableMeta

    variables = [VariableMeta.from_dict(m) for m in snap["variables"]]
    derived = [DerivedSpec.from_dict(d) for d in snap.get("derived") or []]
    names = {d.name for d in derived}
    base = [v for v in variables if v.name not in names]
    out = Dataset(ds.raw, base, SourceInfo(**ds.source.to_dict()))
    out.restore_derived(derived, [v for v in variables if v.name in names])
    out.variables = {v.name: out.variables[v.name] for v in variables}
    out.filters = [FilterSpec.from_dict(f) for f in snap.get("filters") or []]
    if snap.get("holdout"):
        out.restore_holdout(HoldoutSpec(**snap["holdout"]))
    return out


@dataclass
class Context:
    state: str                      # current | changed | unrecoverable
    differences: list[str] = field(default_factory=list)
    original: Dataset | None = None

    @property
    def current(self) -> bool:
        return self.state == "current"


def _var_diffs(old: list[dict], new: dict[str, Any]) -> list[str]:
    out = []
    keys = ("measure", "reference_level", "value_labels", "missing_values", "missing_ranges", "storage", "category_order",
            "display_name")
    for m in old:
        cur = new.get(m["name"])
        if cur is None:
            out.append(f"variable {m.get('display_name') or m['name']} no longer exists")
            continue
        cd = cur.to_dict()
        changed = [k for k in keys if k in m and m.get(k) != cd.get(k)]
        if changed:
            out.append(f"{m.get('display_name') or m['name']}: " + ", ".join(k.replace("_", " ") for k in changed)
                       + " changed")
    return out


def context(ds: Dataset, run: Any) -> Context:
    """Is the current project the data view this search used?"""
    from .engine import data_signature
    from .space import SearchSpaceError, view

    try:
        if data_signature(view(ds, run.settings)) == run.signature:
            return Context("current")
    except SearchSpaceError:
        pass
    snap = getattr(run, "data_snapshot", None) or {}
    diffs: list[str] = []
    if snap:
        if snap.get("source_sha256") != ds.source.sha256:
            diffs.append("the source data file differs from the one the search used")
        if [f for f in snap.get("filters") or []] != [f.to_dict() for f in ds.filters]:
            old = "; ".join(_describe_filter(f) for f in snap.get("filters") or []) or "none"
            new = "; ".join(f.describe() for f in ds.filters) or "none"
            diffs.append(f"row filters: search used [{old}], the project now has [{new}]")
        if (snap.get("derived") or []) != [d.to_dict() for d in ds.derived.values()]:
            diffs.append("derived variables changed")
        if (snap.get("holdout") or None) != ({k: v for k, v in (ds.holdout.to_dict() if ds.holdout else {}).items()
                                              if k != "evaluations"} or None):
            diffs.append("the holdout split changed")
        diffs += _var_diffs(snap.get("variables") or [], ds.variables)
    if not diffs:
        diffs.append("the data or variable settings differ from those the search used")
    if not snap:
        return Context("unrecoverable", diffs + ["this search predates v1.3.2 and stored no snapshot of its data view, "
                                                 "so its original rows and coding cannot be rebuilt"])
    if snap.get("source_sha256") != ds.source.sha256:
        return Context("unrecoverable", diffs)
    try:
        orig = rebuild(ds, snap)
        if data_signature(view(orig, run.settings)) != run.signature:
            return Context("unrecoverable", diffs + ["rebuilding the original view did not reproduce the search's data "
                                                     "fingerprint"])
    except Exception as e:  # noqa: BLE001 - any failure means the original view is not available
        return Context("unrecoverable", diffs + [f"the original view could not be rebuilt ({e})"])
    return Context("changed", diffs, orig)


def _describe_filter(d: dict[str, Any]) -> str:
    from ..core.derive import FilterSpec

    try:
        return FilterSpec.from_dict(d).describe()
    except Exception:  # noqa: BLE001
        return str(d)
