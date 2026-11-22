"""Exports (§19): Excel, PDF, Word, R scripts, Python launcher scripts, search settings and
analysis history. Exported R scripts are self-contained: they read the exported data.csv and
write their result to JSON so the analysis can be checked outside StatExplore (Test O)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from ..core.dataset import Dataset
from ..core.project import Project
from ..core.schema import AnalysisResult, Table

FORMATS = {"xlsx": "Excel workbook", "pdf": "PDF report", "docx": "Word document",
           "R": "R script (+ data.csv)", "py": "Python launcher script"}

_PLOT_PATH = re.compile(r'"[^"\n]*/plots/([0-9a-f]{12}\.(?:png|svg|pdf))"')
_DATA_FILE = re.compile(r'^sx_data_file <- "[^"\n]*"$', re.M)


def slug(text: str, n: int = 40) -> str:
    return (re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_") or "result")[:n]


def table_frame(t: Table) -> pd.DataFrame:
    df = pd.DataFrame(t.rows, columns=t.columns)
    if t.row_names:
        df.insert(0, "row", t.row_names)
    return df


# ---- data + R scripts ------------------------------------------------------------------

def export_data_csv(ds: Dataset, path: Path) -> None:
    """The same data file the R worker reads (base variables by alias, user-missing as NA)."""
    from ..analysis.session import export_data

    export_data(ds, path)


def portable_script(result: AnalysisResult, data_name: str = "data.csv") -> str:
    """Rewrite temp paths to relative ones and append a JSON dump of the result."""
    script = _DATA_FILE.sub(f'sx_data_file <- "{data_name}"', result.script)
    script = _PLOT_PATH.sub(lambda m: f'"plots/{m.group(1)}"', script)
    header = (f"# StatExplore analysis: {result.title}\n# Result id: {result.id}  ·  run {result.created_at}\n"
              f"# Engine: {result.engine.version}\n"
              + "".join(f"#   {k} {v}\n" for k, v in result.engine.packages.items())
              + "# Run from this folder:  Rscript " + f"{slug(result.title)}_{result.id}.R\n"
              "dir.create(\"plots\", showWarnings = FALSE)\n")
    footer = (f'\n# ---- write the result so it can be compared with StatExplore ----\n'
              f'jsonlite::write_json(sx_result, "result_{result.id}.json", auto_unbox = TRUE, digits = I(17), '
              f'null = "null", na = "null", force = TRUE)\n')
    return header + script + footer


def python_launcher(result: AnalysisResult, r_name: str) -> str:
    return f'''"""StatExplore analysis: {result.title} (result {result.id}).

All statistics in StatExplore are computed in R (one engine per analysis). This script
re-runs the exported R script and loads its result, so the analysis can be reproduced and
inspected from Python. Requires R with the packages listed in {r_name}.
"""
import json
import shutil
import subprocess
from pathlib import Path

here = Path(__file__).resolve().parent
rscript = shutil.which("Rscript") or r"C:\\Program Files\\R\\R-4.6.2\\bin\\x64\\Rscript.exe"
subprocess.run([rscript, "--vanilla", "{r_name}"], cwd=here, check=True)
result = json.loads((here / "result_{result.id}.json").read_text(encoding="utf-8"))
print(json.dumps(result, indent=2)[:4000])
'''


# ---- Excel ------------------------------------------------------------------------------

def _summary_rows(result: AnalysisResult) -> list[tuple[str, Any]]:
    rows = [("Title", result.title), ("Status", result.status.value), ("Run", result.created_at),
            ("Engine", result.engine.version),
            ("Packages", ", ".join(f"{k} {v}" for k, v in result.engine.packages.items())),
            ("n used", result.n_used), ("Missing data", result.missing_data)]
    if result.values.get("interpretation"):
        rows.append(("Interpretation", result.values["interpretation"]))
    rows += [("Variable treatment (this analysis only)", t.get("description", "")) for t in
             result.inputs.get("treatments") or []]
    if result.inputs.get("treatment_warning"):
        rows.append(("Treatment warning", result.inputs["treatment_warning"]))
    rows += [("Message", m) for m in result.messages]
    rows += [("Note", n) for n in result.values.get("notes") or []]
    if result.seed is not None:
        rows.append(("Random seed", result.seed))
    return rows


def write_xlsx(results: list[AnalysisResult], path: Path, extra_sheets: dict[str, pd.DataFrame] | None = None) -> None:
    used: set[str] = set()

    def sheet(name: str) -> str:
        base = re.sub(r"[\[\]:*?/\\]", "_", name)[:28] or "Sheet"
        out, i = base, 1
        while out.lower() in used:
            i += 1
            out = f"{base[:25]}_{i}"
        used.add(out.lower())
        return out

    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        index = pd.DataFrame([{"result": r.id, "title": r.title, "status": r.status.value, "run": r.created_at,
                               "n_used": r.n_used} for r in results])
        index.to_excel(xw, sheet_name=sheet("Index"), index=False)
        for r in results:
            pd.DataFrame(_summary_rows(r), columns=["field", "value"]).to_excel(
                xw, sheet_name=sheet(f"{r.id} summary"), index=False)
            for t in r.tables:
                table_frame(t).to_excel(xw, sheet_name=sheet(f"{r.id} {t.name}"), index=False)
            pd.DataFrame({"R code": r.script.splitlines()}).to_excel(xw, sheet_name=sheet(f"{r.id} code"), index=False)
        for name, df in (extra_sheets or {}).items():
            df.to_excel(xw, sheet_name=sheet(name), index=False)


# ---- Word -------------------------------------------------------------------------------

def write_docx(results: list[AnalysisResult], artifacts: dict[str, bytes], path: Path,
               include_code: bool = True) -> None:
    import io

    from docx import Document
    from docx.shared import Inches, Pt

    from ..gui.result_view import fmt_cell, is_p_column

    doc = Document()
    doc.add_heading("StatExplore report", 0)
    doc.add_paragraph("Exploratory analyses. Statistical significance in exploratory searches is not reliable "
                      "evidence; see each result's notes.")
    for r in results:
        doc.add_heading(r.title, 1)
        doc.add_paragraph(f"Status: {r.status.value} · {r.engine.version} · n used = {r.n_used} · run {r.created_at}")
        if r.values.get("interpretation"):
            p = doc.add_paragraph()
            p.add_run("Interpretation. ").bold = True
            p.add_run(r.values["interpretation"])
        for t in r.inputs.get("treatments") or []:
            p = doc.add_paragraph(style="List Bullet")
            p.add_run("Variable treatment for this analysis only: ").bold = True
            p.add_run(t.get("description", ""))
        if r.inputs.get("treatment_warning"):
            doc.add_paragraph(r.inputs["treatment_warning"]).italic = True
        for m in r.messages:
            doc.add_paragraph(m, style="List Bullet")
        for key in r.plots:
            data = artifacts.get(key)
            if data and data[:4] == b"\x89PNG":
                doc.add_picture(io.BytesIO(data), width=Inches(6))
        for t in r.tables:
            doc.add_heading(t.name, 2)
            df = table_frame(t)
            rows = df.head(60)
            table = doc.add_table(rows=1, cols=len(rows.columns))
            table.style = "Light Grid Accent 1"
            for i, c in enumerate(rows.columns):
                table.rows[0].cells[i].text = str(c)
            pcols = [is_p_column(str(c)) for c in rows.columns]
            for rec in rows.itertuples(index=False):
                cells = table.add_row().cells
                for i, v in enumerate(rec):
                    cells[i].text = fmt_cell(v, pcols[i]).replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
            if len(df) > 60:
                doc.add_paragraph(f"(first 60 of {len(df)} rows; the Excel export has all rows)")
            if t.notes:
                doc.add_paragraph(t.notes).italic = True
        if include_code:
            doc.add_heading("R code", 2)
            para = doc.add_paragraph()
            run = para.add_run(r.script)
            run.font.name = "Consolas"
            run.font.size = Pt(7)
    doc.save(path)


# ---- PDF --------------------------------------------------------------------------------

def write_pdf(results: list[AnalysisResult], artifacts: dict[str, bytes], path: Path) -> None:
    """Rendered with Qt (no extra dependency); same HTML as the on-screen view."""
    from PySide6.QtGui import QPageSize, QPdfWriter, QTextDocument

    from ..gui.result_view import build_html

    doc = QTextDocument()
    parts = ["<h1>StatExplore report</h1><p>Exploratory analyses; see each result's notes and messages.</p>"]
    for i, r in enumerate(results):
        if i:
            parts.append("<p style='page-break-before: always'></p>")
        parts.append(build_html(doc, r, artifacts, width=620, plots_first=None))
    doc.setHtml("".join(parts))
    writer = QPdfWriter(str(path))
    writer.setPageSize(QPageSize(QPageSize.A4))
    writer.setResolution(110)
    doc.setPageSize(writer.pageLayout().paintRectPixels(writer.resolution()).size().toSizeF())
    doc.print_(writer)


# ---- orchestration ----------------------------------------------------------------------

def export(project: Project, result_ids: Iterable[str], folder: Path, formats: set[str],
           search_ids: Iterable[str] = (), include_history: bool = True) -> list[Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    ids = list(result_ids)
    results = [r for r in project.results if r.id in ids]
    written: list[Path] = []
    extra: dict[str, pd.DataFrame] = {}
    for sid in search_ids:
        from ..discovery.engine import results_frame, summary

        run = next(r for r in project.searches if r.id == sid)
        df = results_frame(run)
        keep = [c for c in df.columns if c not in ("focal", "covariates", "key")]
        extra[f"search {run.id}"] = df[keep]
        sm = summary(run, df) if len(df) else {}
        settings = {**run.settings.to_dict(), **{k: v for k, v in sm.items()}, "status": run.status,
                    "warnings": " | ".join(run.warnings)}
        extra[f"search {run.id} settings"] = pd.DataFrame({"setting": list(settings), "value": [str(v) for v in settings.values()]})
        (folder / f"search_{run.id}_settings.json").write_text(json.dumps(run.meta_dict(), indent=2, default=str))
        written.append(folder / f"search_{run.id}_settings.json")
    if "xlsx" in formats and (results or extra):
        p = folder / "statexplore_results.xlsx"
        write_xlsx(results, p, extra)
        written.append(p)
    if "docx" in formats and results:
        p = folder / "statexplore_report.docx"
        write_docx(results, project.artifacts, p)
        written.append(p)
    if "pdf" in formats and results:
        p = folder / "statexplore_report.pdf"
        write_pdf(results, project.artifacts, p)
        written.append(p)
    if formats & {"R", "py"} and results:
        export_data_csv(project.dataset, folder / "data.csv")
        written.append(folder / "data.csv")
        for r in results:
            name = f"{slug(r.title)}_{r.id}.R"
            (folder / name).write_text(portable_script(r), encoding="utf-8")
            written.append(folder / name)
            if "py" in formats:
                pyn = f"{slug(r.title)}_{r.id}.py"
                (folder / pyn).write_text(python_launcher(r, name), encoding="utf-8")
                written.append(folder / pyn)
    plots_dir = folder / "plots"
    for r in results:
        for key in r.plots:
            if key in project.artifacts:
                plots_dir.mkdir(exist_ok=True)
                (plots_dir / Path(key).name).write_bytes(project.artifacts[key])
                written.append(plots_dir / Path(key).name)
    if include_history:
        hist = pd.DataFrame([{"id": r.id, "run": r.created_at, "kind": r.kind, "title": r.title,
                              "status": r.status.value, "n_used": r.n_used, "engine": r.engine.version,
                              "messages": " | ".join(r.messages)} for r in project.results])
        hist.to_csv(folder / "analysis_history.csv", index=False)
        changes = pd.DataFrame([c.to_dict() for c in project.dataset.changes])
        changes.to_csv(folder / "variable_changes.csv", index=False)
        written += [folder / "analysis_history.csv", folder / "variable_changes.csv"]
    return written
