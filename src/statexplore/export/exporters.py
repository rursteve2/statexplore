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
           "R": "R reproducibility bundle per analysis", "py": "Python launcher script"}

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
    from .bundle import guard_data_write

    guard_data_write(Path(path))     # AUD135-03: never into a no-consent export, never unregistered
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
    footer = (identity_check(result.values.get("identity"))
              + f'\n# ---- write the result so it can be compared with StatExplore ----\n'
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


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def identity_check(identity: dict | None) -> str:
    """R lines that stop unless the re-run analysed the same rows, outcome values / coding and fixed design as the
    recorded model (identity schema 2, SHA-256; STAT-22 / STAT-36)."""
    if not identity or int(identity.get("schema") or 0) < 2:
        return ""
    parts = [(k, identity[k]) for k in ("rows", "response", "design", "groups") if identity.get(k)]
    if not parts:
        return ""
    return ("\n# ---- identity check: same analysed rows, outcome coding and design as the recorded model ----\n"
            "stopifnot(\n" + ",\n".join(f'  "{k} differ from the recorded model" = identical(sx_result$identity${k}, '
                                       f'"{v}")' for k, v in parts) + ")\n"
            'cat("Model identity reproduced (rows, outcome, design, grouping).\\n")\n')


def results_check(expected: dict | None) -> str:
    """R lines that stop unless the re-run reproduces the recorded fit (n and logLik / AIC / BIC within the documented
    tolerance abs 1e-4 + rel 1e-6), read from the fitted object itself (V133-EXP-03)."""
    if not expected:
        return ""
    checks = []
    for key, expr in (("n", "stats::nobs(sx_fit)"), ("logLik", "as.numeric(stats::logLik(sx_fit))"),
                      ("AIC", "stats::AIC(sx_fit)"), ("BIC", "stats::BIC(sx_fit)")):
        v = expected.get(key)
        if v is None or (isinstance(v, float) and v != v):
            continue
        if key == "n":
            checks.append(f'  "n differs from the recorded model" = isTRUE(all.equal({expr}, {int(v)}))')
        else:
            checks.append(f'  "{key} differs from the recorded model" = isTRUE(abs({expr} - ({float(v)!r})) <= 1e-4 + '
                          f'1e-6 * abs({float(v)!r}))')
    if not checks:
        return ""
    return ("\n# ---- results check: the fitted model reproduces the recorded fit (abs 1e-4 + rel 1e-6) ----\n"
            "if (exists(\"sx_fit\") && !inherits(sx_fit, \"geeglm\")) {\n  stopifnot(\n" + ",\n".join(checks)
            + ")\n  cat(\"Results reproduced (n, logLik, AIC, BIC).\\n\")\n}\n")


def _engine_versions(engine: Any) -> dict | None:
    if engine is None:
        return None
    if isinstance(engine, dict):
        return {"r": engine.get("r_version") or engine.get("version"), **(engine.get("packages") or {})}
    return {"r": getattr(engine, "version", None), **(getattr(engine, "packages", {}) or {})}


def build_id() -> str:
    """Application build identity: version plus the source commit when known (a frozen build records it at build
    time in statexplore/_build.txt)."""
    from .. import __version__

    here = Path(__file__).resolve().parents[1]
    for cand in (here / "_build.txt",):
        if cand.is_file():
            return f"{__version__} ({cand.read_text(encoding='utf-8').strip()})"
    try:
        import subprocess

        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=here, capture_output=True, text=True,
                             timeout=5).stdout.strip()
        return f"{__version__} (commit {sha})" if sha else __version__
    except Exception:
        return __version__


def write_portable(script: str, folder: Path, ds: Dataset, name: str = "analysis",
                   pseudonymize_ids: bool = True, provenance: dict | None = None, identity: dict | None = None,
                   engine: Any = None, expected: dict | None = None, include_data: bool = False,
                   packages: list[str] | None = None, extra: dict | None = None) -> list[Path]:
    """Save a portable reproducibility bundle in a NEW folder inside `folder` (AUD135-03: never into a reused folder;
    the returned paths name the actual location): script, the data file it reads (analysis columns only; only with
    the user's explicit consent for this export), a schema-versioned manifest, a README and the bundle inventory. See
    `portable_into`."""
    from .bundle import export_transaction

    with export_transaction(folder, name, include_data) as tx:
        rels = portable_into(tx, "", script, ds, name=name, pseudonymize_ids=pseudonymize_ids, provenance=provenance,
                             identity=identity, engine=engine, expected=expected, packages=packages, extra=extra)
    return tx.published(rels)


def portable_into(tx: Any, sub: str, script: str, ds: Dataset, name: str = "analysis",
                  pseudonymize_ids: bool = True, provenance: dict | None = None, identity: dict | None = None,
                  engine: Any = None, expected: dict | None = None, packages: list[str] | None = None,
                  extra: dict | None = None, readme: str = "README.txt", result_json: str | None = None) -> list[str]:
    """Write a portable bundle into the export transaction `tx` (sub-folder `sub`): the script checks its packages
    first, refuses a data file whose SHA-256 differs from the exported one, and (when known) stops unless the re-run
    analysed the same rows, outcome coding, design and grouping (identity) and reproduced n / logLik / AIC / BIC
    (expected). Paths are relative and output folders are created, so the folder can be moved and run in a clean,
    offline R session. `packages` restricts the declared dependencies to those the executed model family needs
    (V133-EXP-04). Without data consent the data file is written only to private scratch (to compute its SHA-256)
    and the bundle is code-only. Returns the registered relative paths (script, [data], manifest, readme)."""
    import datetime as _dt

    from .. import __version__

    pre = (sub.strip("/") + "/") if sub else ""
    include_data = tx.include_data
    data_name = f"{name}_data.csv"
    used = used_aliases(script)
    if include_data:
        data_path = tx.path(pre + data_name, "data")
    else:
        data_path = tx.scratch() / f"{name}_{id(script)}_data.csv"
    export_data_csv(ds, data_path, used)
    ids = _pseudonymize(data_path, sorted(set(_ID_COLUMN.findall(script)))) if pseudonymize_ids else []
    md5, sha = _md5(data_path), _sha256(data_path)
    rows = sum(1 for _ in open(data_path, encoding="utf-8")) - 1
    if not include_data:
        data_path.unlink()
    check = (f'sx_data_sha256 <- "{sha}"  # SHA-256 of {data_name} as exported (see {name}_manifest.json)\n'
             "if (!isTRUE(get0(\"sx_skip_checksum\")) && !identical(unname(tools::sha256sum(sx_data_file)), sx_data_sha256))\n"
             "  stop(\"The data file \", sx_data_file, \" is not the file exported with this script (SHA-256 differs). \",\n"
             "       \"Use the exported file, or set sx_skip_checksum <- TRUE before source() to run anyway (the run is then \",\n"
             "       \"UNVERIFIED and is not a reproduction).\", call. = FALSE)\n"
             "if (isTRUE(get0(\"sx_skip_checksum\"))) message(\"WARNING: the data checksum was skipped - this run is UNVERIFIED.\")")
    text, used = portable_text(script, data_name, check)
    found = required_packages(text)
    if packages is not None:
        found = sorted(set(found) & set(packages))
    text = package_check(found) + 'dir.create("plots", showWarnings = FALSE)  # output folder for any plots\n' + text
    text += identity_check(identity)
    text += results_check(expected)
    if result_json:
        text += (f'\n# ---- write the result so it can be compared with StatExplore ----\n'
                 f'jsonlite::write_json(sx_result, "{result_json}", auto_unbox = TRUE, digits = I(17), '
                 f'null = "null", na = "null", force = TRUE)\n')
    if ids:
        text = ("# Participant IDs in the data file are order-preserving codes (1, 2, ...), not the original IDs; "
                "grouping, row order and results are unchanged.\n" + text)
    tx.write_text(pre + f"{name}.R", text, "script")
    versions = _engine_versions(engine)
    tx.bundle_kind = "runnable"      # code-only when the data are withheld (decided by the transaction)
    manifest = {"manifest_schema": 3, "created": _dt.datetime.now().isoformat(timespec="seconds"),
                "statexplore": __version__, "build": build_id(),
                "bundle_kind": "runnable" if include_data else "code-only",
                "script": f"{name}.R", "data": {"file": data_name, "included": bool(include_data), "sha256": sha,
                                                "md5": md5, "rows": rows,
                                                "columns": sorted(used) if used is not None else None,
                                                "pseudonymized_id_columns": ids,
                                                "source_dataset": getattr(ds, "source_path", None) and str(ds.source_path)},
                "r_packages": found,
                "r_versions": versions,
                "provenance": provenance or None, "identity": identity or None,
                "expected_results": expected or None,
                "tolerance": {"logLik/AIC/BIC": "abs 1e-4 + rel 1e-6", "n, identity fingerprints": "exact"},
                **({"model": extra} if extra else {}),
                "notes": "Paths are relative to this folder. Run: source(\"" + name + ".R\") or Rscript " + name + ".R"}
    tx.write_text(pre + f"{name}_manifest.json", json.dumps(manifest, indent=2, default=str), "manifest")
    lines = [f"StatExplore reproducibility bundle ({build_id()})", "",
             f"Run in R (no network needed):  Rscript {name}.R   (from this folder), or source(\"{name}.R\") from anywhere.",
             f"Data: {data_name} " + ("(included: analysis columns only, exported with your consent)" if include_data else
                                     "(NOT included - code-only bundle: place the analysis-columns file here; its "
                                     f"SHA-256 must be {sha})"),
             f"R packages used by this script: {', '.join(found) or 'base R only'}"
             + (f" (recorded versions: R {versions.get('r')})" if versions else ""),
             "The script stops with a message if the data file differs from the exported one, if the analysed rows, "
             "outcome coding, design or grouping differ from the recorded model, or if n / logLik / AIC / BIC are "
             "not reproduced within abs 1e-4 + rel 1e-6.",
             "Results and metadata in this folder are not anonymized." + (
                 " Treat the data file with the same care as the original data." if include_data else "")]
    tx.write_text(pre + readme, "\n".join(lines) + "\n", "readme")
    out = [pre + f"{name}.R"] + ([pre + data_name] if include_data else []) + [pre + f"{name}_manifest.json", pre + readme]
    return out


FAMILY_PACKAGES = {   # packages a model family's executed code can need (anything else is not declared)
    "lmm": ["lme4", "lmerTest", "nlme", "mmrm", "Matrix"], "glmm": ["lme4", "Matrix"], "clmm": ["ordinal"],
    "gee": ["geepack"], "gaussian": [], "binomial": ["detectseparation"], "ordinal": ["ordinal"],
    "multinomial": ["nnet"],
}


def fit_expectations(result: AnalysisResult) -> dict | None:
    """n / logLik / AIC / BIC of a result's 'Model fit' table at full precision (None if absent)."""
    t = next((t for t in result.tables if t.name.startswith("Model fit")), None)
    if t is None or not t.rows:
        return None
    r = dict(zip(t.columns, t.rows[0]))
    out = {"n": r.get("observations"), "logLik": r.get("logLik"), "AIC": r.get("AIC"), "BIC": r.get("BIC")}
    return {k: v for k, v in out.items() if isinstance(v, (int, float))} or None


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
           search_ids: Iterable[str] = (), include_history: bool = True, include_data: bool = False) -> list[Path]:
    """Reports export into a NEW folder inside `folder` (AUD135-03; the returned paths name it). R / Python formats
    write one self-checking reproducibility bundle PER ANALYSIS (its own script and, only with `include_data` consent,
    its own data file) and only when the data that analysis actually read can be proven (AUD135-07,
    `historical.resolve`); otherwise the analysis gets REPRODUCTION_UNAVAILABLE.txt with the reason and its recorded
    code as a labelled, non-runnable record. Tables, reports, plots, search settings and audits are always written."""
    from .bundle import export_transaction

    ids = list(result_ids)
    results = [r for r in project.results if r.id in ids]
    with export_transaction(folder, "statexplore_export", include_data) as tx:
        rels = _export_into(tx, project, results, formats, search_ids, include_history)
    return tx.published(rels)


def _export_into(tx: Any, project: Project, results: list[AnalysisResult], formats: set[str],
                 search_ids: Iterable[str], include_history: bool) -> list[str]:
    written: list[str] = []
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
                                 "p_holm", "log_p", "log10_p", "log_p_adj", "log_p_bh", "log_p_holm", "log_p_source",
                                 "test_status", "test_reason", "reference_flags", "reference_source", "family_size", "adj_method", "effect", "estimate", "se", "ci_low",
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
        rel = f"search_{run.id}_settings.json"
        tx.write_text(rel, json.dumps(run.meta_dict(), indent=2, default=str), "metadata")
        written.append(rel)
        from .audit import audit_into      # V134-015: versioned machine-readable audit + human-readable report

        written += audit_into(tx, "", run, project.dataset)
    if "xlsx" in formats and (results or extra):
        rel = "statexplore_results.xlsx"
        write_xlsx(results, tx.path(rel, "results"), extra)
        written.append(rel)
    if "docx" in formats and results:
        rel = "statexplore_report.docx"
        write_docx(results, project.artifacts, tx.path(rel, "report"))
        written.append(rel)
    if "pdf" in formats and results:
        rel = "statexplore_report.pdf"
        write_pdf(results, project.artifacts, tx.path(rel, "report"))
        written.append(rel)
    if formats & {"R", "py"} and results:
        from .historical import Unverified, refusal_text, resolve

        runnable = 0
        for r in results:
            sub = f"{slug(r.title)}_{r.id}"
            try:
                view = resolve(r, project.dataset, tx.scratch())
            except Unverified as e:
                written.append(tx.write_text(f"{sub}/REPRODUCTION_UNAVAILABLE.txt", refusal_text(r, str(e)),
                                             "metadata").relative_to(tx.stage).as_posix())
                written.append(tx.write_text(
                    f"{sub}/recorded_code_not_runnable.R",
                    "# HISTORICAL RECORD - NOT A RUNNABLE REPRODUCTION: " + str(e).replace("\n", " ") + "\n"
                    "# The exact code StatExplore ran for this result (its data file is not part of this export).\n"
                    + r.script, "metadata").relative_to(tx.stage).as_posix())
                tx.notes.append(f"result {r.id}: reproduction refused ({e})")
                continue
            vals = r.values
            name = sub
            written += portable_into(
                tx, sub, r.script, view, name=name, provenance=vals.get("provenance"), identity=vals.get("identity"),
                engine=r.engine, expected=fit_expectations(r),
                packages=sorted((getattr(r.engine, "packages", None) or {}).keys()) or None,
                extra={"result_id": r.id, "title": r.title, "analysis_provenance": vals.get("analysis_provenance")},
                result_json=f"result_{r.id}.json")
            runnable += 1
            if "py" in formats:
                rel = f"{sub}/{name}.py"
                tx.write_text(rel, python_launcher(r, f"{name}.R"), "script")
                written.append(rel)
        if not runnable:
            tx.bundle_kind = "results-only"
    for r in results:
        for key in r.plots:
            if key in project.artifacts:
                rel = f"plots/{Path(key).name}"
                tx.write_bytes(rel, project.artifacts[key], "plot")
                written.append(rel)
    if include_history:
        hist = pd.DataFrame([{"id": r.id, "run": r.created_at, "kind": r.kind, "title": r.title,
                              "status": r.status.value, "n_used": r.n_used, "engine": r.engine.version,
                              "messages": " | ".join(r.messages)} for r in project.results])
        hist.to_csv(tx.path("analysis_history.csv", "metadata"), index=False)
        changes = pd.DataFrame([c.to_dict() for c in project.dataset.changes])
        changes.to_csv(tx.path("variable_changes.csv", "metadata"), index=False)
        written += ["analysis_history.csv", "variable_changes.csv"]
    return list(dict.fromkeys(written))
