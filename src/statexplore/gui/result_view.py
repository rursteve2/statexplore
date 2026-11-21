"""Render an AnalysisResult (tables, notes, plots, messages) into a QTextBrowser.

All text from results is HTML-escaped: variable names and labels come from
user files and must never be interpreted as markup.
"""
from __future__ import annotations

import html
import math
from typing import Any

from PySide6.QtCore import QUrl
from PySide6.QtGui import QImage, QTextDocument
from PySide6.QtWidgets import QTextBrowser

from ..core.schema import AnalysisResult, Status, Table

_STATUS_COLOR = {Status.OK: "#2e7d32", Status.WARNING: "#8d6e00", Status.FAILED: "#c62828",
                 Status.INVALID: "#c62828", Status.CANCELLED: "#616161"}


def is_p_column(name: str) -> bool:
    n = name.lower()
    return n in ("p", "p.value", "pr(>f)") or n.startswith("p_") or n.endswith("_p")


def fmt_cell(value: Any, p_col: bool = False) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        v = float(value)
        if math.isnan(v):
            return "—"
        if math.isinf(v):
            return "∞" if v > 0 else "−∞"
        if p_col:
            return f"{v:.2e}" if v < 1e-4 else f"{v:.4f}"
        if v.is_integer() and abs(v) < 1e12:
            return f"{int(v):,}" if abs(v) >= 10_000 else str(int(v))
        if abs(v) >= 1e6 or abs(v) < 1e-3:
            return f"{v:.3e}"
        return f"{v:.3f}"
    return html.escape(str(value))


def table_html(t: Table) -> str:
    p_cols = [is_p_column(c) for c in t.columns]
    head = "".join(f"<th>{html.escape(c)}</th>" for c in ([""] if t.row_names else []) + t.columns)
    body = []
    for i, row in enumerate(t.rows):
        cells = []
        if t.row_names:
            cells.append(f"<td><b>{html.escape(t.row_names[i] if i < len(t.row_names) else '')}</b></td>")
        for v, pc in zip(row, p_cols):
            align = "right" if isinstance(v, (int, float)) and not isinstance(v, bool) else "left"
            cells.append(f"<td align='{align}'>{fmt_cell(v, pc)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    notes = f"<p class='note'>{html.escape(t.notes)}</p>" if t.notes else ""
    return (f"<h3>{html.escape(t.name)}</h3><table cellspacing='0' cellpadding='4' border='1'>"
            f"<tr>{head}</tr>{''.join(body)}</table>{notes}")


def render(view: QTextBrowser, result: AnalysisResult, artifacts: dict[str, bytes]) -> None:
    doc = view.document()
    parts = [
        "<style>h2{color:#12263f;margin-bottom:2px} h3{color:#2b5d8c;margin:14px 0 4px 0}"
        "th{background:#e8eef5} table{border-color:#c5d0dc} .note{color:#555;font-size:small}</style>",
        f"<h2>{html.escape(result.title)}</h2>",
        f"<p><span style='color:{_STATUS_COLOR[result.status]}'><b>{result.status.value.upper()}</b></span>"
        f" &nbsp;·&nbsp; {html.escape(result.engine.version)}",
    ]
    if result.n_used is not None:
        excl = result.values.get("n_excluded")
        parts.append(f" &nbsp;·&nbsp; n used = {result.n_used:,}"
                     + (f" ({excl:,} excluded)" if excl else ""))
    parts.append("</p>")
    if result.missing_data:
        parts.append(f"<p class='note'>Missing data: {html.escape(result.missing_data)}</p>")
    filters = result.inputs.get("filters") or []
    if filters:
        parts.append(f"<p class='note'>Active filters: {html.escape('; '.join(filters))}</p>")
    for m in result.messages:
        color = "#c62828" if result.status is Status.FAILED else "#8d6e00"
        parts.append(f"<p style='color:{color}'>{html.escape(m)}</p>")
    for n in result.values.get("notes") or []:
        parts.append(f"<p class='note'>{html.escape(n)}</p>")
    for t in result.tables:
        parts.append(table_html(t))
    width = max(300, min(760, view.viewport().width() - 40))
    for i, key in enumerate(result.plots):
        data = artifacts.get(key)
        if not data:
            continue
        img = QImage.fromData(data)
        url = QUrl(f"sxplot://{result.id}/{i}")
        doc.addResource(QTextDocument.ImageResource, url, img)
        w = min(width, img.width())
        parts.append(f"<p><img src='{url.toString()}' width='{w}'></p>")
    view.setHtml("".join(parts))
