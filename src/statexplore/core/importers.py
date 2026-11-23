"""File importers for CSV, SPSS (.sav) and Excel (.xlsx).

Imported files are opened read-only and never modified. All content is treated
as untrusted: names and labels are stored as plain data and only ever reach R
as escaped string literals (see rbridge.codegen).
"""
from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import io
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .dataset import Dataset, SourceInfo
from .variables import Measure, Storage, VariableMeta, make_alias

SUPPORTED_EXTENSIONS = {".csv": "csv", ".sav": "sav", ".xlsx": "xlsx"}
CSV_ENCODINGS = ["utf-8-sig", "cp1252", "latin-1", "utf-16"]
SMALL_CODE_SET = 9          # non-negative integer columns with at most this many distinct values look like codes
# time variables (wave 1-4, visit 0-3) are numbers even with few integer values
TIME_NAME = re.compile(r"(^|[_\W])(wave|waves|time|visit|occasion|period|year|month|day|t)($|[_\W\d])", re.I)
CSV_DELIMITERS = {"Auto-detect": None, "Comma (,)": ",", "Semicolon (;)": ";", "Tab": "\t", "Pipe (|)": "|"}


class ImportError_(Exception):
    """Raised for any failure to import a file (message is user-facing)."""


@dataclass
class CsvOptions:
    delimiter: str | None = None  # None = sniff
    encoding: str | None = None  # None = try CSV_ENCODINGS in order


def detect_format(path: str | os.PathLike) -> str:
    ext = Path(path).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ImportError_(f"Unsupported file type {ext!r}. Supported: .csv, .sav, .xlsx")
    return SUPPORTED_EXTENSIONS[ext]


def file_sha256(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _source(path: str | os.PathLike, fmt: str, options: dict[str, Any]) -> SourceInfo:
    p = Path(path)
    return SourceInfo(
        path=str(p.resolve()), format=fmt, options=options,
        sha256=file_sha256(p), size_bytes=p.stat().st_size,
        imported_at=_dt.datetime.now().isoformat(timespec="seconds"),
    )


def import_file(path: str | os.PathLike, *, csv_options: CsvOptions | None = None,
                sheet: str | int | None = None) -> Dataset:
    fmt = detect_format(path)
    if not Path(path).is_file():
        raise ImportError_(f"File not found: {path}")
    try:
        if fmt == "csv":
            return import_csv(path, csv_options or CsvOptions())
        if fmt == "xlsx":
            return import_xlsx(path, sheet)
        return import_sav(path)
    except ImportError_:
        raise
    except Exception as e:  # pragma: no cover - message passthrough
        raise ImportError_(f"Could not read {Path(path).name}: {e}") from e


# ---- CSV -----------------------------------------------------------------

def _decode(data: bytes, encoding: str | None) -> tuple[str, str]:
    encodings = [encoding] if encoding else CSV_ENCODINGS
    last: Exception | None = None
    for enc in encodings:
        try:
            return data.decode(enc), enc
        except (UnicodeDecodeError, LookupError) as e:
            last = e
    raise ImportError_(f"Could not decode file with encodings {encodings}: {last}")


def sniff_delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def import_csv(path: str | os.PathLike, options: CsvOptions) -> Dataset:
    data = Path(path).read_bytes()
    text, encoding = _decode(data, options.encoding)
    delimiter = options.delimiter or sniff_delimiter(text[:64_000])
    # every cell is read as text and typed by _type_text_cells (blanks, padded numbers, zero-padded codes)
    df = pd.read_csv(io.StringIO(text), sep=delimiter, dtype=str, low_memory=False)
    if df.shape[1] == 0:
        raise ImportError_("No columns found.")
    df = _normalise_frame(df)
    df, typing = _type_text_cells(df)
    decimal = set(typing.pop("_decimal", []))
    variables = [_infer_meta(i, name, df[name], codes_possible=name not in decimal) for i, name in enumerate(df.columns)]
    return _annotated(Dataset(df, variables, _source(path, "csv", {"delimiter": delimiter, "encoding": encoding,
                                                                   **typing})))


# ---- Excel ---------------------------------------------------------------

def list_sheets(path: str | os.PathLike) -> list[str]:
    with pd.ExcelFile(path, engine="openpyxl") as xf:
        return list(xf.sheet_names)


def import_xlsx(path: str | os.PathLike, sheet: str | int | None = None) -> Dataset:
    sheets = list_sheets(path)
    if not sheets:
        raise ImportError_("Workbook has no sheets.")
    if sheet is None:
        sheet = sheets[0]
    if isinstance(sheet, str) and sheet not in sheets:
        raise ImportError_(f"Sheet {sheet!r} not found. Available: {sheets}")
    df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    df = _normalise_frame(df)
    if df.shape[1] == 0:
        raise ImportError_(f"Sheet {sheet!r} is empty.")
    df, typing = _type_text_cells(df)
    typing.pop("_decimal", None)
    variables = [_infer_meta(i, name, df[name]) for i, name in enumerate(df.columns)]
    sheet_name = sheet if isinstance(sheet, str) else sheets[sheet]
    return _annotated(Dataset(df, variables, _source(path, "xlsx", {"sheet": sheet_name, **typing})))


# ---- SPSS ----------------------------------------------------------------

_SPSS_MEASURE = {"scale": Measure.CONTINUOUS, "ordinal": Measure.ORDINAL, "nominal": Measure.NOMINAL}


def import_sav(path: str | os.PathLike) -> Dataset:
    import pyreadstat

    # user_missing=True keeps user-missing codes as values; definitions go to metadata
    # and are applied (-> NA) only in Dataset.analysis_column.
    df, meta = pyreadstat.read_sav(str(path), user_missing=True, apply_value_formats=False)
    df = _normalise_frame(df)
    labels = meta.column_names_to_labels or {}
    value_labels = meta.variable_value_labels or {}
    missing_ranges = meta.missing_ranges or {}
    measures = meta.variable_measure or {}
    variables: list[VariableMeta] = []
    for i, name in enumerate(df.columns):
        inferred = _infer_meta(i, name, df[name])
        discrete: list[Any] = []
        ranges: list[tuple[float, float]] = []
        for r in missing_ranges.get(name, []):
            lo, hi = r["lo"], r["hi"]
            if lo == hi:
                discrete.append(lo)
            else:
                ranges.append((float(lo), float(hi)))
        declared = measures.get(name, "unknown")
        measure = _SPSS_MEASURE.get(declared, inferred.measure)
        if inferred.storage is Storage.STRING and measure is Measure.CONTINUOUS:
            measure = Measure.NOMINAL
        vl = {(float(k) if isinstance(k, (int, float)) else k): str(v)
              for k, v in value_labels.get(name, {}).items()}
        variables.append(VariableMeta(
            name=name, alias=inferred.alias, storage=inferred.storage, measure=measure,
            label=labels.get(name) or "", value_labels=vl,
            missing_values=discrete, missing_ranges=ranges, source_measure=declared,
        ))
    return _annotated(Dataset(df, variables, _source(path, "sav", {})))


# ---- shared helpers ------------------------------------------------------

def _annotated(ds: Dataset) -> Dataset:
    """Attach measurement-level suggestions (§4). Metadata only: nothing is reclassified."""
    from ..analysis.measurement import annotate

    annotate(ds)
    return ds


def _normalise_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Unique string column names; integer columns stored as float (NaN-capable);
    mixed object columns stored as strings."""
    names: list[str] = []
    seen: dict[str, int] = {}
    for i, c in enumerate(df.columns):
        base = str(c).strip() or f"column_{i + 1}"
        name = base
        while name in seen:
            seen[base] += 1
            name = f"{base}_{seen[base]}"
        seen.setdefault(name, 0)
        names.append(name)
    df = df.copy()
    df.columns = names
    for c in df.columns:
        col = df[c]
        if pd.api.types.is_bool_dtype(col):
            df[c] = col.astype("float64")
        elif pd.api.types.is_integer_dtype(col):
            df[c] = col.astype("float64")
        elif pd.api.types.is_datetime64_any_dtype(col) or pd.api.types.is_numeric_dtype(col):
            continue
        else:
            df[c] = col.map(lambda x: x if pd.isna(x) else str(x)).astype("string")
    return df


def _type_text_cells(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Typed working representation of untyped (CSV/Excel) text cells (v1.3 P0-05). The source file is
    never changed. Empty and whitespace-only cells are missing (NA, never 0); a text column whose every
    non-missing cell is a number once surrounding whitespace is trimmed becomes numeric (signs, decimals
    and scientific notation kept). Returns the frame and a report for the source options:
      whitespace_cells  {column: number of blank/whitespace-only cells turned into NA}
      empty_columns     columns with no non-missing value
      text_in_numeric   {column: [count, examples]} mostly numeric columns kept as text because some
                        cells are not numbers (never coerced silently)."""
    whitespace: dict[str, int] = {}
    empty: list[str] = []
    decimal: list[str] = []
    mixed: dict[str, list[Any]] = {}
    df = df.copy()
    for c in df.columns:
        col = df[c]
        if pd.api.types.is_numeric_dtype(col) or pd.api.types.is_datetime64_any_dtype(col):
            if col.notna().sum() == 0:
                empty.append(c)
            continue
        stripped = col.astype("string").str.strip()
        blank = stripped.notna() & (stripped == "")
        if blank.any():
            whitespace[c] = int(blank.sum())
            stripped = stripped.mask(blank)
        present = stripped.dropna()
        if present.empty:
            df[c] = pd.Series(np.nan, index=df.index, dtype="float64")
            empty.append(c)
            continue
        num = pd.to_numeric(present, errors="coerce")
        # "inf"/"nan" spelled out are text, not measurements
        bad = num.isna() | ~np.isfinite(num.astype("float64"))
        if not bad.any():
            if present.str.match(r"^[+-]?0\d+$").any():
                df[c] = stripped        # zero-padded codes ("007", household IDs) are labels, as in SPSS
            else:
                df[c] = pd.to_numeric(stripped, errors="coerce").astype("float64")
                if present.str.contains(".", regex=False).any():
                    decimal.append(c)   # written with decimals ("3.0"): stored as quantities, not codes
            continue
        low = present.str.lower()
        if low.isin(["true", "false"]).all():
            df[c] = stripped.str.lower().map({"true": 1.0, "false": 0.0}).astype("float64")
            continue
        if blank.any():
            df[c] = col.mask(blank)
        if (~bad).sum() >= 0.8 * len(present):
            mixed[c] = [int(bad.sum()), sorted(set(present[bad].astype(str)))[:3]]
    report: dict[str, Any] = {}
    if whitespace:
        report["whitespace_cells"] = whitespace
    if empty:
        report["empty_columns"] = empty
    if mixed:
        report["text_in_numeric"] = mixed
    if decimal:
        report["_decimal"] = decimal       # internal: consumed by the CSV importer, not stored
    return df, report


def import_report(ds: Dataset) -> list[str]:
    """User-facing notes on how untyped cells were normalised during import (empty when nothing changed)."""
    o = ds.source.options or {}
    out: list[str] = []
    ws = o.get("whitespace_cells") or {}
    if ws:
        out.append(f"{sum(ws.values()):,} empty or whitespace-only cells in {len(ws):,} column(s) were read as "
                   "missing values (NA, not zero); numbers with surrounding spaces were read as numbers. "
                   "The source file was not changed.")
    if o.get("empty_columns"):
        cols = o["empty_columns"]
        out.append(f"{len(cols):,} column(s) have no values at all: " + ", ".join(cols[:8])
                   + (" …" if len(cols) > 8 else "") + ".")
    for c, (n, ex) in (o.get("text_in_numeric") or {}).items():
        out.append(f"{c} is mostly numeric but was kept as text because {n:,} cell(s) are not numbers "
                   f"(e.g. {', '.join(repr(x) for x in ex)}). Recode or declare those values missing to analyse it "
                   "as a number.")
    return out


def _storage(col: pd.Series) -> Storage:
    if pd.api.types.is_datetime64_any_dtype(col):
        return Storage.DATETIME
    if pd.api.types.is_numeric_dtype(col):
        return Storage.NUMERIC
    return Storage.STRING


def _infer_meta(index: int, name: str, col: pd.Series, codes_possible: bool = True) -> VariableMeta:
    storage = _storage(col)
    if storage is Storage.NUMERIC:
        values = col.dropna()
        n_unique = values.nunique()
        # numeric storage alone is not a measurement level (P0-05 rule 6): a few integer codes are treated as
        # categories until the user reviews them (Measurement Review offers continuous / ordinal)
        coded = (codes_possible and n_unique <= SMALL_CODE_SET and bool((values == values.round()).all())
                 and bool((values >= 0).all()) and not TIME_NAME.search(name))
        measure = Measure.NOMINAL if n_unique <= 2 or coded else Measure.CONTINUOUS
    elif storage is Storage.DATETIME:
        measure = Measure.CONTINUOUS
    else:
        measure = Measure.NOMINAL
    return VariableMeta(name=name, alias=make_alias(index), storage=storage, measure=measure)

