"""Historical analysis results -> the exact data view they analysed, or an explicit refusal (v1.3.6, AUD135-07).

A result carries `values["analysis_provenance"]` (captured by `RSession.run` at fit time): the content and view
identities and the SHA-256 of the exact CSV bytes the R worker read. To export a runnable reproduction of a historical
result, that CSV must be re-created and match the recorded SHA-256 - from the current project view when it is still
the analysed one, otherwise from the result's own snapshot rebuilt on the SAME imported content. Anything else
(no provenance: StatExplore 1.3.5 or earlier; other worksheet / values / holdout split / user-missing codes / R Console
replacement; a snapshot that does not rebuild) is refused with the reason: the result, its tables and report stay
exportable, but no script is paired with current data as if it were the original.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ..core.dataset import Dataset


class Unverified(ValueError):
    """The original analysed data cannot be proven for this result."""


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def input_sha256(ds: Dataset, scratch: Path) -> str:
    """SHA-256 of the full worker-input CSV `ds` produces (written to a private scratch file, then removed)."""
    from ..analysis.session import export_data

    scratch.mkdir(parents=True, exist_ok=True)
    p = scratch / f"input_{id(ds)}.csv"
    try:
        export_data(ds, p)
        return _sha(p)
    finally:
        p.unlink(missing_ok=True)


def provenance(result: Any) -> dict[str, Any]:
    return dict((getattr(result, "values", None) or {}).get("analysis_provenance") or {})


def resolve(result: Any, current: Dataset | None, scratch: Path) -> Dataset:
    """The dataset whose worker-input CSV is byte-identical to the one `result` analysed; raises `Unverified`."""
    from ..analysis.session import PROVENANCE_SCHEMA
    from ..core.identity import IDENTITY_SCHEMA, IdentityError, content_id
    from ..discovery.history import SnapshotMismatch, rebuild

    p = provenance(result)
    if current is None:
        raise Unverified("no project data are open")
    if int(p.get("schema") or 0) < PROVENANCE_SCHEMA or int(p.get("identity_schema") or 0) < IDENTITY_SCHEMA \
            or not p.get("input_sha256"):
        raise Unverified("this result was produced by StatExplore 1.3.5 or earlier and has no recorded analysis-data "
                         "provenance, so the data it analysed cannot be proven (a current data file could differ in "
                         "rows, holdout split, user-missing codes or worksheet)")
    script = getattr(result, "script", "") or ""
    if hashlib.sha256(script.encode("utf-8")).hexdigest() != p.get("script_sha256"):
        raise Unverified("the recorded script differs from the script captured with the result")
    try:
        cur = content_id(current)
    except IdentityError as e:
        raise Unverified(f"the current data have no verifiable content identity ({e})") from None
    if cur != p.get("content_id"):
        raise Unverified("the project's imported data are not the data this result analysed (different values, "
                         "worksheet, import options or an R Console replacement)")
    if input_sha256(current, scratch) == p["input_sha256"]:
        return current
    try:
        orig = rebuild(current, p.get("snapshot") or {})
    except (SnapshotMismatch, Exception) as e:  # noqa: BLE001 - any failure: not reconstructible
        raise Unverified(f"the analysed data view cannot be rebuilt ({e})") from None
    if input_sha256(orig, scratch) != p["input_sha256"]:
        raise Unverified("the data view this result analysed (rows, holdout split, user-missing codes) cannot be "
                         "reproduced exactly from the project: its recorded input SHA-256 does not match")
    return orig


def refusal_text(result: Any, why: str) -> str:
    return (f"No runnable reproduction was written for '{getattr(result, 'title', '')}' (result "
            f"{getattr(result, 'id', '')}): {why}.\n\nThe recorded results, tables, report and the exact code that "
            "ran are historical records of the analysed data. A script is never paired with today's data as if they "
            "were the original. Reopen the project with the original data (same imported content, holdout and "
            "coding) to export a runnable reproduction, or run the analysis again (a new, separately labelled "
            "result).\n")
