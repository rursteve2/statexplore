# StatExplore

Offline Windows desktop app for exploratory statistical model discovery and teaching
(Python/PySide6 front end, R statistical engine). Build plan: `tasks/todo.md`.
Architecture: `docs/architecture.md`.

**Status: Phase 1 (desktop foundation).**

## Requirements
- Windows 10/11
- Python 3.11+
- R 4.4+ for Windows (tested target: 4.6.2)

## Install
```bat
cd statexplore
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
"C:\Program Files\R\R-4.6.2\bin\Rscript.exe" r\install_packages.R
```
`install_packages.R` puts jsonlite, psych and ggplot2 into `r\library` (project-local).
StatExplore finds R automatically (PATH, registry, standard install folders). You can also set
the path under **Settings** or with the `STATEXPLORE_RSCRIPT` environment variable.

## Run
```bat
run.bat                       :: or: .venv\Scripts\python -m statexplore
run.bat path\to\data.sav      :: open a dataset directly
```

## Test
```bat
.venv\Scripts\python -m pytest -q
```
R tests skip automatically if R or the Phase 1 packages are missing. The HRS fixture tests look for
`E:\Downloads\hrs_least_missing_1000.sav` (override with `STATEXPLORE_HRS`).

## Phase 1 features
- Import CSV (delimiter/encoding auto-detect or manual), SPSS .sav (variable labels, value labels,
  user-missing definitions, measurement levels) and Excel .xlsx (sheet selection); drag-and-drop.
- Dataset Explorer: overview (rows, columns, missing cells, types, measurement levels, unique
  values, mean/SD/min/max, Tukey outlier counts), spreadsheet preview with value-label toggle,
  variable manager (display name, measurement level, role, reference level) with change log + undo.
- EDA: `psych::describe` / `describeBy`; histogram, density, boxplot, violin, scatter, bar,
  `psych::pairs.panels`, correlation heatmap (ggplot2); type-based plot suggestions; **View Code**
  shows the exact R script executed.
- Projects: single `.sxp` file (SQLite) with a copy of the data, metadata, change log, results
  and plots. The original data file is never modified.

## Known limitations (Phase 1)
- Derived variables, row filters, and wide-to-long reshaping aren't implemented yet (Phases 2/5).
- Date/time columns are imported and shown but excluded from analyses.
- Empty strings in text columns are treated as missing.
- Ordinal variables are coded as unordered factors in R (avoids polynomial contrasts by default).
- The R runtime isn't bundled yet; the offline installer is built in Phase 7.
