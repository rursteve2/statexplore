"""Transactional, isolated exports with explicit data consent (v1.3.6, AUD135-03).

Every export action writes ONE new folder that did not exist before: files are staged in a hidden sibling folder
(`.<name>.incomplete-<random>`) and the stage is renamed to the final name only after every file was written and the
inventory check passed. Nothing in the chosen parent folder is ever overwritten or deleted, so an old bundle (its
scripts, data or manifests) can never be mistaken for part of the new export, and an interrupted export never looks
complete (its stage is removed on any error; a process killed mid-export leaves only a clearly named `.incomplete-`
folder).

The transaction's registry is the only way to put files into a bundle:
- `write_text` / `write_bytes` / `register` (files written by a library, e.g. xlsx) with a declared kind;
- analysis DATA only through `write_data`, and only when the user consented for this export (`include_data=True`).
  `exporters.export_data_csv` refuses to write into the stage of a no-consent transaction, whoever calls it.
At `finish` the stage must contain exactly the registered files (an unregistered file fails the export), no symbolic
links, and - without consent - no data file at all (declared or not: `data.csv` / `*_data.csv` are refused by name too).
A `bundle_inventory.json` lists every file with its SHA-256, kind, the consent and the bundle kind:
  results-only   results, reports, metadata and refusal reasons; no runnable reproduction;
  code-only      runnable code that needs the matching data file (verified by SHA-256) placed next to it;
  runnable       code plus the consented analysis data.
Hash scratch for no-consent exports lives in a private temporary folder outside the export.
"""
from __future__ import annotations

import contextvars
import datetime as _dt
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

INVENTORY = "bundle_inventory.json"
INVENTORY_SCHEMA = 1
KINDS = ("data", "script", "manifest", "results", "report", "plot", "metadata", "readme")
BUNDLE_KINDS = {"results-only": "results, reports and metadata only; no runnable reproduction",
                "code-only": "runnable code that needs the matching data file (checked by SHA-256), not included",
                "runnable": "runnable code with the analysis data included (with the user's consent)"}
_DATA_NAMES = ("data.csv",)


class ExportError(RuntimeError):
    pass


class ConsentError(ExportError):
    """Analysis data would be written without the user's consent for this export."""


_ACTIVE: contextvars.ContextVar[tuple["ExportTransaction", ...]] = contextvars.ContextVar("sx_export_tx", default=())


def _looks_like_data(rel: str) -> bool:
    name = rel.rsplit("/", 1)[-1].lower()
    return name in _DATA_NAMES or name.endswith("_data.csv")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def unique_child(parent: Path, name: str) -> Path:
    """A child of `parent` that does not exist yet: name_<timestamp>, then _2, _3, ..."""
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"{name}_{stamp}"
    cand = parent / base
    i = 2
    while cand.exists() or (parent / f".{cand.name}.reserved").exists():
        cand = parent / f"{base}_{i}"
        i += 1
    return cand


class ExportTransaction:
    def __init__(self, parent: str | os.PathLike, name: str, include_data: bool):
        if type(include_data) is not bool:
            raise ConsentError("explicit data consent (True / False) is required for every export")
        self.parent = Path(parent)
        self.parent.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.include_data = include_data
        self.stage = Path(tempfile.mkdtemp(prefix=f".{name}.incomplete-", dir=self.parent))
        self.final: Path | None = None
        self.files: dict[str, str] = {}          # relative posix path -> kind
        self.bundle_kind = "results-only"
        self.notes: list[str] = []
        self._scratch: Path | None = None
        self.done = False

    # ---- registry ----
    def _rel(self, rel: str) -> str:
        rel = Path(rel).as_posix().lstrip("/")
        if not rel or rel.startswith("../") or "/../" in rel or rel == "..":
            raise ExportError(f"invalid bundle path {rel!r}")
        return rel

    def path(self, rel: str, kind: str) -> Path:
        """Register `rel` (kind) and return its staging path (parent folders created)."""
        rel = self._rel(rel)
        if kind not in KINDS:
            raise ExportError(f"unknown artifact kind {kind!r}")
        if kind == "data" and not self.include_data:
            raise ConsentError(f"{rel}: analysis data are not included in this export (no consent)")
        if kind != "data" and _looks_like_data(rel):
            raise ExportError(f"{rel}: a data file must be written as data (with consent), not as {kind}")
        if rel in self.files and self.files[rel] != kind:
            raise ExportError(f"{rel} registered twice with different kinds")
        self.files[rel] = kind
        p = self.stage / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def register(self, rel: str, kind: str) -> Path:
        return self.path(rel, kind)

    def write_text(self, rel: str, text: str, kind: str) -> Path:
        p = self.path(rel, kind)
        p.write_text(text, encoding="utf-8", newline="\n")
        return p

    def write_bytes(self, rel: str, data: bytes, kind: str) -> Path:
        p = self.path(rel, kind)
        p.write_bytes(data)
        return p

    def write_data(self, rel: str, writer: Any) -> Path:
        """`writer(path)` writes the analysis data file; only with consent."""
        p = self.path(rel, "data")
        writer(p)
        return p

    def scratch(self) -> Path:
        """Private temporary folder OUTSIDE the export (e.g. to hash data that are not included)."""
        if self._scratch is None:
            self._scratch = Path(tempfile.mkdtemp(prefix="sx_export_scratch_"))
        return self._scratch

    def in_stage(self, path: Path) -> bool:
        try:
            Path(path).resolve().relative_to(self.stage.resolve())
            return True
        except ValueError:
            return False

    def final_path(self, rel: str) -> Path:
        if self.final is None:
            raise ExportError("the export has not been published")
        return self.final / self._rel(rel)

    # ---- publish ----
    def _check(self) -> list[dict[str, Any]]:
        inv = []
        on_disk = set()
        for p in sorted(self.stage.rglob("*")):
            if p.is_symlink():
                raise ExportError(f"symbolic link in the export: {p.name}")
            if p.is_file():
                on_disk.add(p.relative_to(self.stage).as_posix())
        unregistered = sorted(on_disk - set(self.files))
        if unregistered:
            raise ExportError("files written outside the export registry: " + ", ".join(unregistered[:5]))
        missing = sorted(set(self.files) - on_disk)
        if missing:
            raise ExportError("registered files were not written: " + ", ".join(missing[:5]))
        if not self.include_data:
            bad = sorted(r for r in on_disk if self.files.get(r) == "data" or _looks_like_data(r))
            if bad:
                raise ConsentError("analysis data in an export without data consent: " + ", ".join(bad))
        for rel in sorted(on_disk):
            p = self.stage / rel
            inv.append({"path": rel, "kind": self.files[rel], "bytes": p.stat().st_size, "sha256": _sha256(p)})
        return inv

    def finish(self) -> Path:
        from .. import __version__

        inv = self._check()
        has_data = any(f["kind"] == "data" for f in inv)
        if self.bundle_kind == "runnable" and not has_data:
            self.bundle_kind = "code-only"
        if has_data and self.bundle_kind != "runnable":
            raise ExportError("data were written into an export that is not a runnable bundle")
        meta = {"inventory_schema": INVENTORY_SCHEMA, "statexplore": __version__,
                "created": _dt.datetime.now().isoformat(timespec="seconds"), "bundle_kind": self.bundle_kind,
                "bundle_kind_label": BUNDLE_KINDS[self.bundle_kind], "data_included": has_data,
                "data_consent": self.include_data, "notes": self.notes, "files": inv,
                "policy": "new folder per export; nothing in the parent folder was overwritten or deleted"}
        (self.stage / INVENTORY).write_text(json.dumps(meta, indent=1), encoding="utf-8", newline="\n")
        final = unique_child(self.parent, self.name)
        os.rename(self.stage, final)       # same parent folder: an atomic rename on one volume
        self.final = final
        self.done = True
        return final

    def abort(self) -> None:
        shutil.rmtree(self.stage, ignore_errors=True)

    def close_scratch(self) -> None:
        if self._scratch is not None:
            shutil.rmtree(self._scratch, ignore_errors=True)
            self._scratch = None

    def published(self, rels: list[str] | None = None) -> list[Path]:
        """Final paths of the registered files (all, or `rels` in that order) plus the inventory."""
        keys = list(self.files) if rels is None else rels
        return [self.final_path(r) for r in keys] + [self.final / INVENTORY]


@contextmanager
def export_transaction(parent: str | os.PathLike, name: str, include_data: bool) -> Iterator[ExportTransaction]:
    tx = ExportTransaction(parent, name, include_data)
    token = _ACTIVE.set(_ACTIVE.get() + (tx,))
    try:
        yield tx
        tx.finish()
    except BaseException:
        tx.abort()
        raise
    finally:
        _ACTIVE.reset(token)
        tx.close_scratch()


def guard_data_write(path: Path) -> None:
    """Called by every analysis-data writer: refuse a data file inside the stage of a no-consent export, and any data
    file inside a stage that was not registered as data."""
    for tx in _ACTIVE.get():
        if tx.in_stage(path):
            rel = Path(path).resolve().relative_to(tx.stage.resolve()).as_posix()
            if not tx.include_data:
                raise ConsentError(f"{rel}: analysis data are not included in this export (no consent)")
            if tx.files.get(rel) != "data":
                raise ExportError(f"{rel}: analysis data must be registered with write_data")


def new_id() -> str:
    return uuid.uuid4().hex[:8]
