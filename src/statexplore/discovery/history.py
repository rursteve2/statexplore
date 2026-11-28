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


SNAPSHOT_SCHEMA = 3


def snapshot(ds: Dataset) -> dict[str, Any]:
    """Schema 2 (v1.3.5, AUD134-07) adds the source file NAME (never its path), format, size and row count, so a
    historical record can describe its own data without consulting the current project. Schema 3 (v1.3.6, AUD135-06)
    adds the versioned CONTENT identity of the imported values and the parse options (sheet, delimiter, ...): a
    snapshot is rebuilt only on raw data with that exact content identity."""
    from pathlib import PureWindowsPath

    from ..core.identity import IDENTITY_SCHEMA, content_id, source_context

    return {"schema": SNAPSHOT_SCHEMA, "identity_schema": IDENTITY_SCHEMA, "content_id": content_id(ds),
            "source_options": source_context(ds).get("options") or {},
            "source_sha256": ds.source.sha256,
            "source_file": PureWindowsPath(ds.source.path or "").name, "source_format": ds.source.format,
            "source_size_bytes": ds.source.size_bytes, "source_rows": int(len(ds.raw)),
            "variables": [v.to_dict() for v in ds.variables.values()],
            "derived": [d.to_dict() for d in ds.derived.values()],
            "filters": [f.to_dict() for f in ds.filters],
            "holdout": {k: v for k, v in (ds.holdout.to_dict() if ds.holdout else {}).items() if k != "evaluations"}
            or None}


class SnapshotMismatch(ValueError):
    """The current raw data are not provably the content the snapshot was taken of (AUD135-06)."""


def verified_content(ds: Dataset, snap: dict[str, Any]) -> str:
    """'' when `ds` holds exactly the imported content `snap` was taken of (identity schema 3); otherwise why not.
    A legacy snapshot (no content identity) is never verified by the source file hash alone."""
    from ..core.identity import IDENTITY_SCHEMA, IdentityError, content_id

    if not snap:
        return "no snapshot of the data view was recorded"
    if int(snap.get("identity_schema") or 0) < IDENTITY_SCHEMA or not snap.get("content_id"):
        return ("the snapshot was recorded by StatExplore 1.3.5 or earlier without a content identity, so the imported "
                "values (for example the worksheet) cannot be verified; the same source file hash is not proof")
    try:
        cur = content_id(ds)
    except IdentityError as e:
        return f"the current data have no verifiable content identity ({e})"
    if cur != snap["content_id"]:
        same_file = snap.get("source_sha256") == ds.source.sha256
        return ("the imported data values differ from those the search used"
                + (" (same source file, different content: e.g. another worksheet, other import options or edited "
                   "values)" if same_file else " (different source file)"))
    return ""


def rebuild(ds: Dataset, snap: dict[str, Any]) -> Dataset:
    """The search's original data view on the current raw data, which must have the snapshot's exact CONTENT identity
    (v1.3.6; raises `SnapshotMismatch` otherwise - never a rebuild over different values)."""
    from ..core.dataset import HoldoutSpec, SourceInfo
    from ..core.derive import DerivedSpec, FilterSpec
    from ..core.variables import VariableMeta

    why = verified_content(ds, snap)
    if why:
        raise SnapshotMismatch(why)

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
    state: str                      # current | changed | unrecoverable | unverified (legacy identity, AUD135-06)
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
    from ..core.identity import IDENTITY_SCHEMA, IdentityError
    from .engine import LEGACY_IDENTITY, data_signature
    from .space import SearchSpaceError, view

    legacy = int(getattr(run, "identity_schema", 2) or 2) < IDENTITY_SCHEMA
    if not legacy:
        try:
            if data_signature(view(ds, run.settings)) == run.signature:
                return Context("current")
        except (SearchSpaceError, IdentityError):
            pass
    snap = getattr(run, "data_snapshot", None) or {}
    diffs: list[str] = []
    if snap:
        content = verified_content(ds, snap)
        if content and not legacy:
            diffs.append(content)
        elif snap.get("source_sha256") != ds.source.sha256:
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
    if legacy:
        return Context("unverified", diffs + [LEGACY_IDENTITY.format(schema=getattr(run, "identity_schema", 2))])
    if not diffs:
        diffs.append("the data or variable settings differ from those the search used")
    if not snap:
        return Context("unrecoverable", diffs + ["this search predates v1.3.2 and stored no snapshot of its data view, "
                                                 "so its original rows and coding cannot be rebuilt"])
    if verified_content(ds, snap):
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
