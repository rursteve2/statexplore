# StatExplore — Developer Guide

Technical details for building, testing and extending StatExplore. For using the program, see the
main [README](../README.md).

- Plan, decisions (D1–D12) and per-phase reviews: [`tasks/todo.md`](../tasks/todo.md)
- Lessons learned: [`tasks/lessons.md`](../tasks/lessons.md)
- Architecture: [`architecture.md`](architecture.md)
- Discovery candidate grammar: [`formula-grammar.md`](formula-grammar.md)

## Stack
Python 3.11 + PySide6 (GUI, import, data management, orchestration, exports). R 4.6 is the single
statistical engine (decision D1). It runs as persistent `Rscript` workers that speak JSON lines (see
`src/statexplore/r/worker.R`). Every analysis is an R script generated in `rbridge/codegen.py` and the
`analysis/` modules; the exact script is stored with each result ("View Code").

## Set up a development environment
```bat
cd statexplore
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m pip install statsmodels python-docx pyinstaller pyflakes
"C:\Program Files\R\R-4.6.2\bin\x64\Rscript.exe" r\install_packages.R
```
`install_packages.R` installs the R packages (CRAN binaries) into the project-local `r\library`.
Optional and GitHub-only: `remotes::install_github("dustinfife/flexplot", lib = "r/library")`.

## Run
```bat
run.bat                                  :: GUI
.venv\Scripts\python -m statexplore      :: same
.venv\Scripts\python -m statexplore --selftest
```

## Test
```bat
.venv\Scripts\python -m pytest -q
.venv\Scripts\python -m pyflakes src tests
```
- 144 tests. Statistical results are checked against independent implementations (scipy,
  statsmodels, numpy) rather than against R itself.
- Tests that need R are marked `r` and skip automatically when R or its packages are missing.
- HRS-based tests look for `E:\Downloads\hrs_least_missing_1000.sav` (override with the
  `STATEXPLORE_HRS` environment variable) and skip if it is absent.
- GUI tests run offscreen (`QT_QPA_PLATFORM=offscreen`).

## Build the offline distribution and installer
```bat
.venv\Scripts\python packaging\build.py
:: -> dist\StatExplore\  (StatExplore.exe, _internal\, R\ = bundled R, rlib\ = bundled packages)
dist\StatExplore\StatExplore.exe --selftest
:: exit code 0 = OK; report in %TEMP%\statexplore_selftest.txt
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" packaging\statexplore.iss
:: -> dist\installer\StatExplore-Setup.exe
```
The bundled library folder is `rlib`, not `r\library`: Windows paths are case-insensitive, so
`r\library` would merge into R's own `R\library`. Exact R and package versions are in
`r/packages.lock.json`.

## Source layout
```
src/statexplore/
  core/        importers, dataset (derived variables, filters, holdout), variables, derive,
               overview, missing, structure (ID/time detection, wide->long), schema, project (.sxp)
  rbridge/     locate (find R), worker (persistent Rscript, JSON lines), codegen (all R text)
  analysis/    session, framework + builders (Analysis Builder specs), models (ModelSpec),
               teaching, comparison, mixed, validation, eda
  discovery/   space (grammar, enumeration, counts), rcode, engine (worker pool, modes,
               adjustments, rankings), detail
  export/      exporters (xlsx, docx, pdf, R, py, history)
  gui/         main_window, state, models, result_view, model_editor, derive_dialogs, pages/
  r/worker.R
packaging/     build.py, launcher.py, statexplore.spec, statexplore.iss
r/             install_packages.R, packages.lock.json (library/ is git-ignored)
tests/         pytest suite
```

## Safety rules for contributors
- Build R text only through `rbridge/codegen.py` helpers: `r_str` for any user-derived text and
  `alias()` for identifiers. Variable names from files never become R identifiers.
- Never pass two strings where R expects one (`notes = "a", "b"` binds "b" to the next argument);
  use `paste0`.
- Every statistical feature needs a test against an independent reference.
