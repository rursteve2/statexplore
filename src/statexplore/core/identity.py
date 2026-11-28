"""Versioned data identities (v1.3.6, AUD135-06).

Three identities, never confused with each other:

1. Source file SHA-256 (`SourceInfo.sha256`): provenance of the file that was opened. It is NOT proof of the analysed
   content: one workbook holds several sheets, and a parser change can read the same bytes differently.
2. Content identity (`content_id`): SHA-256 over the imported, normalised values themselves - column names and order,
   each column's canonical kind, every value bit-exactly (little-endian float64 bits, int64 datetimes, length-prefixed
   UTF-8 text), the null mask, the row count and index - plus the parse context (source format, sheet / delimiter /
   encoding / reshape options and, conservatively, the source file SHA-256). Same sheet re-imported -> same identity;
   another sheet, an edited value, a reordered row -> another identity.
3. View identity (`view_id`): content identity + variable metadata (measures, codes, labels, user-missing, order,
   treatments) + derived-variable and filter specifications + holdout specification + the REALIZED filter mask, holdout
   assignment and derived values. This is what a search or an analysis actually saw; `discovery.engine.data_signature`
   is this identity.

Schema `IDENTITY_SCHEMA` is recorded with every search, snapshot and analysis result. Records with an older schema
(v1.3.5 and earlier: source hash + metadata only) are UNVERIFIED: they stay viewable, but never qualify for cache reuse,
resume, rebuilding or a runnable reproduction.

The raw frame of a Dataset is immutable by contract (every import, console import, reshape and project reopen builds a
new Dataset). The value digest is cached per raw OBJECT (a live weak reference, never a bare address) with a
shape / column / dtype guard, so it is computed once per imported revision, not once per candidate or per refresh.
"""
from __future__ import annotations

import hashlib
import json
import math
import weakref
from typing import Any

import numpy as np
import pandas as pd

IDENTITY_SCHEMA = 3
_TAG = "statexplore-identity-3"


class IdentityError(ValueError):
    """The content cannot be given a verified identity (unsupported storage): callers refuse rather than guess."""


# ---- canonical JSON (strict: no default=str) -------------------------------------------------------------------------

def _plain(x: Any) -> Any:
    if x is None or isinstance(x, (bool, str)):
        return x
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        f = float(x)
        if math.isnan(f):
            return {"$float": "nan"}
        if math.isinf(f):
            return {"$float": "inf" if f > 0 else "-inf"}
        return {"$float": f.hex()} if not f.is_integer() or abs(f) >= 2 ** 53 else f
    if isinstance(x, dict):
        return {"$map": sorted([[canonical(k), _plain(v)] for k, v in x.items()])}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if hasattr(x, "value") and type(x).__module__.startswith("statexplore"):   # enums
        return _plain(x.value)
    raise IdentityError(f"value of type {type(x).__name__} has no canonical identity encoding")


def canonical(x: Any) -> str:
    """Deterministic JSON text of `x` (typed floats, sorted maps; raises on unsupported types)."""
    return json.dumps(_plain(x), ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(x: Any) -> str:
    return hashlib.sha256(canonical(x).encode("utf-8")).hexdigest()


# ---- value digest of a raw frame --------------------------------------------------------------------------------------

def _kind(col: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(col):
        return "bool"
    if pd.api.types.is_datetime64_any_dtype(col):
        return "datetime"
    if pd.api.types.is_numeric_dtype(col):
        return "number"
    if pd.api.types.is_string_dtype(col) or col.dtype == object:
        return "text"
    raise IdentityError(f"column storage {col.dtype} has no verified identity encoding")


def _column_bytes(col: pd.Series, h: "hashlib._Hash") -> str:
    kind = _kind(col)
    n = len(col)
    if kind in ("number", "bool"):
        arr = col.to_numpy(dtype="float64", na_value=np.nan).copy()
        null = np.isnan(arr)
        arr[null] = 0.0
        h.update(np.packbits(null).tobytes())
        h.update(arr.astype("<f8").tobytes())
        return kind
    if kind == "datetime":
        s = col
        if getattr(s.dt, "tz", None) is not None:
            raise IdentityError("time-zone aware datetimes need a separately versioned encoding")
        null = s.isna().to_numpy()
        try:
            vals = s.dt.as_unit("ns").to_numpy(dtype="datetime64[ns]").astype("<i8").copy()
        except (OverflowError, ValueError, pd.errors.OutOfBoundsDatetime):
            text = [("" if m else pd.Timestamp(v).isoformat()) for v, m in zip(s, null)]
            h.update(b"iso")
            h.update(np.packbits(null).tobytes())
            enc = [t.encode("utf-8") for t in text]
            h.update(np.array([len(e) for e in enc], dtype="<u4").tobytes())
            h.update(b"".join(enc))
            return kind
        vals[null] = 0
        h.update(np.packbits(null).tobytes())
        h.update(vals.tobytes())
        return kind
    # text (pandas string dtypes or an object column of str): exact UTF-8, missing separate from ""
    vals = col.to_numpy(dtype=object)
    null = np.fromiter((v is None or v is pd.NA or (isinstance(v, float) and math.isnan(v)) for v in vals),
                       dtype=bool, count=n)
    enc = []
    for v, m in zip(vals, null):
        if m:
            enc.append(b"")
        elif isinstance(v, str):
            enc.append(v.encode("utf-8"))
        else:
            raise IdentityError(f"text column holds a {type(v).__name__} value")
    h.update(np.packbits(null).tobytes())
    h.update(np.array([len(e) for e in enc], dtype="<u4").tobytes())
    h.update(b"".join(enc))
    return kind


def _guard(raw: pd.DataFrame) -> tuple:
    return (raw.shape, tuple(map(str, raw.columns)), tuple(str(d) for d in raw.dtypes))


_VALUES: dict[int, tuple[weakref.ref, tuple, str]] = {}


def values_digest(raw: pd.DataFrame) -> str:
    """SHA-256 of the canonical typed values of `raw` (cached per live frame object)."""
    key = id(raw)
    hit = _VALUES.get(key)
    g = _guard(raw)
    if hit is not None and hit[0]() is raw and hit[1] == g:
        return hit[2]
    if raw.columns.has_duplicates:
        raise IdentityError("duplicate column names")
    if not raw.index.is_unique:
        raise IdentityError("duplicate row index")
    h = hashlib.sha256()
    idx = raw.index
    if isinstance(idx, pd.RangeIndex):
        ix = ["range", int(idx.start), int(idx.stop), int(idx.step)]
    elif pd.api.types.is_integer_dtype(idx):
        ix = ["int", hashlib.sha256(idx.to_numpy().astype("<i8").tobytes()).hexdigest()]
    else:
        raise IdentityError("row index of this type has no verified identity encoding")
    kinds = []
    for name in raw.columns:
        sub = hashlib.sha256()
        kinds.append([str(name), _column_bytes(raw[name], sub), sub.hexdigest()])
    h.update(canonical([_TAG, "values", len(raw), ix, kinds]).encode("utf-8"))
    out = h.hexdigest()
    try:
        ref = weakref.ref(raw, lambda _r, k=key: _VALUES.pop(k, None))
    except TypeError:          # not weak-referenceable: never cached
        return out
    _VALUES[key] = (ref, g, out)
    return out


# ---- dataset identities ----------------------------------------------------------------------------------------------

_VOLATILE_SOURCE = ("path", "imported_at")


def source_context(ds: Any) -> dict[str, Any]:
    """The parse context that, with the values, defines the imported content (never the file path or time)."""
    s = ds.source.to_dict()
    return {k: v for k, v in s.items() if k not in _VOLATILE_SOURCE}


def content_id(ds: Any) -> str:
    """Identity of the imported content (values + parse context), schema IDENTITY_SCHEMA."""
    return digest([_TAG, "content", values_digest(ds.raw), source_context(ds)])


def _holdout_spec(ds: Any) -> dict[str, Any] | None:
    return {k: v for k, v in (ds.holdout.to_dict() if ds.holdout else {}).items() if k != "evaluations"} or None


def view_metadata(ds: Any) -> list[Any]:
    """Everything of a view that is not a value: variable metadata (incl. treatments), derived / filter / holdout specs."""
    return [[v.to_dict() for v in ds.variables.values()], [d.to_dict() for d in ds.derived.values()],
            [f.to_dict() for f in ds.filters], _holdout_spec(ds)]


_REALIZED: dict[tuple[str, str], str] = {}


def _realized(ds: Any, cid: str, meta_text: str) -> str:
    """Digest of what the specifications produced on these values: the filter mask, the holdout assignment and the
    derived values (deterministic for (content, metadata) within one environment; recorded so that a different numpy /
    parser result can never pass as the same view)."""
    key = (cid, hashlib.sha256(meta_text.encode("utf-8")).hexdigest())
    hit = _REALIZED.get(key)
    if hit is not None:
        return hit
    h = hashlib.sha256()
    h.update(np.packbits(ds.row_mask("all").to_numpy(dtype=bool)).tobytes())
    hm = getattr(ds, "_holdout_mask", None)
    h.update(b"H" + (np.packbits(hm.to_numpy(dtype=bool)).tobytes() if hm is not None else b"none"))
    for name in ds.derived:
        sub = hashlib.sha256()
        _column_bytes(ds.column(name), sub)
        h.update(name.encode("utf-8") + b"\0" + sub.digest())
    out = h.hexdigest()
    if len(_REALIZED) > 4096:
        _REALIZED.clear()
    _REALIZED[key] = out
    return out


def view_id(ds: Any) -> str:
    """Identity of the analysed data view (content + metadata + realized masks / derived values)."""
    cid = content_id(ds)
    meta_text = canonical(view_metadata(ds))
    return digest([_TAG, "view", cid, meta_text, _realized(ds, cid, meta_text)])


def file_sha256(path: Any) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
