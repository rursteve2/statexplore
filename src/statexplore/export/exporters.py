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

def export_data_csv(ds: Dataset, path: Path, aliases: Iterable[str] | None = None) -> None:
    """The same data file the R worker reads (base variables by alias, user-missing as NA); with `aliases`,
    only those columns (portable exports carry the analysis columns only)."""
    from ..analysis.session import export_data

    export_data(ds, path, set(aliases) if aliases is not None else None)


_ALIAS = re.compile(r"\bv\d{3,}\b")
_PREP_END = "sx_result <- list(n_rows = nrow(dat), n_cols = ncol(dat))"
_ELEMENT = re.compile(r"^\s+(v\d{3,}) = ")
_PER_VAR = re.compile(r"^dat\$(v\d{3,}) <- ")


def used_aliases(script: str) -> set[str] | None:
    """Columns an analysis script needs: those its analysis part uses, plus the inputs of derived variables
    it uses and of every row filter. None if the script has no StatExplore data-preparation block."""
    head, sep, body = script.partition(_PREP_END)
    if not sep:
        return None
    used = set(_ALIAS.findall(body))
    lines = head.split("\n")
    for line in lines:
        if line.startswith("dat <- dat[which("):
            used |= set(_ALIAS.findall(line))
    derived, section = [], ""
    for line in lines:
        if line.startswith("# "):
            section = line
        elif section.startswith("# Derived variables") and _PER_VAR.match(line):
            derived.append(line)
    changed = True
    while changed:
        changed = False
        for line in derived:
            if _PER_VAR.match(line).group(1) in used:
                new = set(_ALIAS.findall(line)) - used
                if new:
                    used |= new
                    changed = True
    return used


def prune_script(script: str, used: set[str]) -> str:
    """Drop the data-preparation lines of columns the analysis does not use (column classes, display names,
    factor codings, unused derived variables); fixes the trailing comma of each shortened vector."""
    head, sep, body = script.partition(_PREP_END)
    out: list[str] = []
    for line in head.split("\n"):
        m = _ELEMENT.match(line) or _PER_VAR.match(line)
        if m and m.group(1) not in used:
            continue
        if line.strip() == ")" and out and _ELEMENT.match(out[-1]) and out[-1].endswith(","):
            out[-1] = out[-1][:-1]
        out.append(line)
    return "\n".join(out) + sep + body


def wrap_r_line(line: str, width: int = 100) -> list[str]:
    """Break a long R line after top-level-inside-parentheses commas (never inside strings or comments)."""
    if len(line) <= width or line.lstrip().startswith("#"):
        return [line]
    indent = " " * (len(line) - len(line.lstrip()) + 4)
    pieces, cur, depth, quote, i = [], "", 0, "", 0
    while i < len(line):
        ch = line[i]
        cur += ch
        if quote:
            if ch == "\\" and i + 1 < len(line):
                cur += line[i + 1]
                i += 1
            elif ch == quote:
                quote = ""
        elif ch in "\"'`":
            quote = ch
        elif ch == "#":
            cur += line[i + 1:]
            break
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth >= 1 and len(cur) >= width - len(indent) - 20:
            pieces.append(cur.rstrip())
            cur = ""
            if i + 1 < len(line) and line[i + 1] == " ":
                i += 1
        i += 1
    pieces.append(cur)
    return [pieces[0]] + [indent + p.lstrip() for p in pieces[1:] if p.strip()]


def wrap_script(script: str, width: int = 100) -> str:
    return "\n".join(part for line in script.split("\n") for part in wrap_r_line(line, width))


DATA_LOCATOR = """sx_data_file <- "{name}"  # exported with this script (analysis columns only); edit if you move it
if (!file.exists(sx_data_file)) {{  # source("analysis.R") from another folder: look next to the script
  sx_here <- tryCatch(dirname(normalizePath(sys.frame(1)$ofile)), error = function(e) NA_character_)
  if (!is.na(sx_here) && file.exists(file.path(sx_here, sx_data_file))) sx_data_file <- file.path(sx_here, sx_data_file)
}}
if (!file.exists(sx_data_file)) stop("Data file not found: put ", sx_data_file, " next to this script or set sx_data_file.")"""


def portable_text(script: str, data_name: str = "data.csv", data_check: str = "") -> tuple[str, set[str] | None]:
    """A shareable copy of a recorded script: relative data reference (no temporary or personal paths), only
    the analysis columns, manageable line lengths. `data_check` is R code run right after the data file is located.
    Returns (script, columns the data file must contain)."""
    used = used_aliases(script)
    if used is not None:
        script = prune_script(script, used)
    script = _DATA_FILE.sub(lambda _m: DATA_LOCATOR.format(name=data_name) + ("\n" + data_check if data_check else ""),
                            script)
    script = _PLOT_PATH.sub(lambda m: f'"plots/{m.group(1)}"', script)
    return wrap_script(script), used


def portable_script(result: AnalysisResult, data_name: str = "data.csv") -> str:
    """Rewrite temp paths to relative ones and append a JSON dump of the result."""
    script, _used = portable_text(result.script, data_name)
    header = (f"# StatExplore analysis: {result.title}\n# Result id: {result.id}  ·  run {result.created_at}\n"
              f"# Engine: {result.engine.version}\n"
              + "".join(f"#   {k} {v}\n" for k, v in result.engine.packages.items())
              + ("# Model provenance:\n" + "".join(f"#   {k}: {v}\n" for k, v in
                                                     (result.values.get("provenance") or {}).items())
                 if result.values.get("provenance") else "")
              + "# Run from this folder:  Rscript " + f"{slug(result.title)}_{result.id}.R"
              + "   (or source() it from anywhere)\n"
              "dir.create(\"plots\", showWarnings = FALSE)\n")
    footer = (f'\n# ---- write the result so it can be compared with StatExplore ----\n'
              f'jsonlite::write_json(sx_result, "result_{result.id}.json", auto_unbox = TRUE, digits = I(17), '
              f'null = "null", na = "null", force = TRUE)\n')
    return header + script + footer


_BASE_R = {"base", "stats", "utils", "tools", "graphics", "grDevices", "methods", "grid", "parallel", "splines"}
_PKG = re.compile(r"\b([A-Za-z][A-Za-z0-9.]*[A-Za-z0-9])::")
_ID_COLUMN = re.compile(r"\bsx_(?:gid|id) <- factor\(sx_dd?\$(v\d{3,})\)")


def required_packages(script: str) -> list[str]:
    """Non-base packages a script calls with pkg:: (comment lines ignored)."""
    code = "\n".join(line for line in script.split("\n") if not line.lstrip().startswith("#"))
    return sorted({m for m in _PKG.findall(code)} - _BASE_R)


def package_check(packages: list[str]) -> str:
    if not packages:
        return ""
    return ("# ---- required R packages (checked before anything runs) ----\n"
            f"sx_needed <- c({', '.join(json.dumps(p) for p in packages)})\n"
            "sx_missing <- sx_needed[!vapply(sx_needed, requireNamespace, logical(1), quietly = TRUE)]\n"
            "if (length(sx_missing)) stop(\"Missing R packages: \", paste(sx_missing, collapse = \", \"), \". Install them \",\n"
            "  \"(offline: install.packages('<file>.zip', repos = NULL)) and run this script again.\", call. = FALSE)\n")


def _md5(path: Path) -> str:
    import hashlib

    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pseudonymize(path: Path, columns: list[str]) -> list[str]:
    """Replace numeric participant IDs by their rank (1, 2, ...). Ranks keep R's factor order, so grouping, row order
    and every fitted number are unchanged. Text IDs are kept (R's locale-dependent sort could not be matched)."""
    if not columns:
        return []
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    done = []
    for c in columns:
        if c not in frame:
            continue
        num = pd.to_numeric(frame[c].replace("", None), errors="coerce")
        if num.notna().sum() != (frame[c] != "").sum():
            continue
        codes = num.rank(method="dense")
        frame[c] = codes.map(lambda x: "" if pd.isna(x) else str(int(x)))
        done.append(c)
    if done:
        frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    return done


def write_portable(script: str, folder: Path, ds: Dataset, name: str = "analysis",
                   pseudonymize_ids: bool = True) -> list[Path]:
    """Save a script, the data file it reads (analysis columns only) and a manifest as a self-contained set.
    The script checks its packages first and refuses a data file whose MD5 differs from the exported one
    (P1-EXPORT). Numeric participant IDs are replaced by order-preserving codes unless `pseudonymize_ids` is off."""
    import datetime as _dt

    from .. import __version__

    folder.mkdir(parents=True, exist_ok=True)
    data_name = f"{name}_data.csv"
    used = used_aliases(script)
    data_path = folder / data_name
    export_data_csv(ds, data_path, used)
    ids = _pseudonymize(data_path, sorted(set(_ID_COLUMN.findall(script)))) if pseudonymize_ids else []
    md5 = _md5(data_path)
    check = (f'sx_data_md5 <- "{md5}"  # MD5 of {data_name} as exported (see {name}_manifest.json)\n'
             "if (!isTRUE(get0(\"sx_skip_checksum\")) && !identical(unname(tools::md5sum(sx_data_file)), sx_data_md5))\n"
             "  stop(\"The data file \", sx_data_file, \" is not the file exported with this script (MD5 differs). \",\n"
             "       \"Use the exported file, or set sx_skip_checksum <- TRUE before source() to run anyway.\", call. = FALSE)")
    text, used = portable_text(script, data_name, check)
    packages = required_packages(text)
    text = package_check(packages) + text
    if ids:
        text = ("# Participant IDs in the data file are order-preserving codes (1, 2, ...), not the original IDs; "
                "grouping, row order and results are unchanged.\n" + text)
    (folder / f"{name}.R").write_text(text, encoding="utf-8")
    rows = sum(1 for _ in open(data_path, encoding="utf-8")) - 1
    manifest = {"created": _dt.datetime.now().isoformat(timespec="seconds"), "statexplore": __version__,
                "script": f"{name}.R", "data": {"file": data_name, "md5": md5, "rows": rows,
                                                "columns": sorted(used) if used is not None else None,
                                                "pseudonymized_id_columns": ids},
                "r_packages": packages,
                "notes": "Paths are relative to this folder. Run: source(\"" + name + ".R\") or Rscript " + name + ".R"}
    (folder / f"{name}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return [folder / f"{name}.R", data_path, folder / f"{name}_manifest.json"]


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
        keep = [c for c in df.columns if c not in ("focal", "covariates", "key", "inter")]
        extra[f"search {run.id}"] = df[keep]
        from ..discovery.engine import tests_frame

        tf = tests_frame(run)
        if len(tf):   # every tested hypothesis with raw and adjusted p, its family and CI metadata (numeric)
            tcols = [c for c in ("test_id", "idx", "model_type", "formula", "reference", "test_type", "test_label", "label",
                                 "test_method", "statistic", "df1", "df_den", "lrt_df", "p", "p_adj", "p_bh",
                                 "p_holm", "family_size", "adj_method", "effect", "estimate", "se", "ci_low",
                                 "ci_high", "ci_scale", "ci_null", "ci_level", "ci_method", "ci_applicable",
                                 "odds_ratio", "or_low", "or_high",
                                 "n", "n_groups", "n_predictors", "n_terms", "n_params", "status", "flags",
                                 "search_scope", "candidate_id", "model_id") if c in tf.columns]
            extra[f"search {run.id} hypotheses"] = tf[tcols]
        sm = summary(run, df) if len(df) else {}
        settings = {**run.settings.to_dict(), **{k: v for k, v in sm.items()}, "status": run.status,
                    "warnings": " | ".join(run.warnings),
                    # presentation metadata only — never part of the inferential calculations
                    "presentation: highlight": json.dumps(project.settings.get("discovery_highlight")
                                                          or {"mode": "adjusted", "alpha": 0.05, "only": False})}
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
        cols: set[str] | None = set()
        for r in results:
            u = used_aliases(r.script)
            cols = None if u is None or cols is None else cols | u
        export_data_csv(project.dataset, folder / "data.csv", cols)
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
