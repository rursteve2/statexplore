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
- 152 tests. Statistical results are checked against independent implementations (scipy,
  statsmodels, numpy) rather than against R itself.
- Tests that need R are marked `r` and skip automatically when R or its packages are missing.
- HRS-based tests look for `E:\Downloads\hrs_least_missing_1000.sav` (override with the
  `STATEXPLORE_HRS` environment variable) and skip if it is absent.
- GUI tests run offscreen (`QT_QPA_PLATFORM=offscreen`).

## Build the offline distribution, installer and release
```bat
.venv\Scripts\python packaging\build.py
:: dist\StatExplore\        StatExplore.exe, _internal\, R\ (bundled R), rlib\ (bundled packages)
:: dist\release\           StatExplore-<version>-portable.zip, StatExplore-<version>-Setup.exe
dist\StatExplore\StatExplore.exe --selftest --full
:: exit code 0 = OK; report in %TEMP%\statexplore_selftest.txt
gh release create v1.0.0 dist\release\* --title "StatExplore 1.0.0" --notes-file docs\RELEASE_NOTES_v1.0.0.md
```
`build.py` finds Inno Setup in Program Files (x86) or `%LOCALAPPDATA%\Programs`. The version comes
from `src/statexplore/__init__.py` (keep `pyproject.toml` and `packaging/statexplore.iss` in step).
The bundled library folder is `rlib`, not `r\library`: Windows paths are case-insensitive, so
`r\library` would merge into R's own `R\library`. Exact R and package versions are in
`r/packages.lock.json`.

## Source layout
```
src/statexplore/
  core/        importers, dataset (derived variables, filters, holdout), variables, derive,
               overview, missing, structure (ID/time detection, wide->long), schema, project (.sxp),
               console_import (R Console -> project, with diff and snapshot)
  rbridge/     locate (find R), worker (persistent Rscript, JSON lines), codegen (all R text),
               console (user R code, separate process), package_manager (offline local installs)
  analysis/    session, framework + builders (Analysis Builder specs), models (ModelSpec),
               teaching, comparison, mixed (Gaussian + glmer/clmm/mblogit/GEE), validation, eda,
               measurement (suggestions, approval, per-analysis treatments), longitudinal_preflight,
               survey, interpret (brief interpretations), teaching_finder
  discovery/   space (grammar, enumeration, counts), rcode, engine (worker pool, modes,
               adjustments, rankings), detail
  export/      exporters (xlsx, docx, pdf, R, py, history)
  gui/         main_window, state, models, result_view, model_editor, derive_dialogs, pages/
               (+ measurement_review, survey, r_console), widgets/ (variable_selector, treatment_panel)
  r/worker.R, r/console_worker.R
packaging/     build.py, launcher.py, statexplore.spec, statexplore.iss
r/             install_packages.R, packages.lock.json (library/ is git-ignored)
tests/         pytest suite
```

## v1.1 design notes
- **Treatments are views.** `measurement.apply_treatments(ds, treatments)` returns a shallow copy of the
  dataset whose `VariableMeta` objects carry the effective measure for one analysis. Validators, builders
  and the R load script read the view, so the project is never modified and the R session's metadata
  signature rebuilds the R-side coding whenever the encoding differs. Pass treatments, never mutate `measure`.
- **Two-level R cache.** `RSession.ensure_data` re-reads the CSV only when the data signature changes and
  re-runs the cheap "prepare" part (coding, derived variables, filters) when metadata or treatments change.
- **User R code never runs in the analysis workers** (`r/console_worker.R`, separate process).
- **Project format 2** is additive; `Project.save` copies a v1 file to `*.v1-backup.sxp` before the first
  overwrite. `tests/fixtures/v1_0_project.sxp` was written by the real v1.0.0 code
  (`tests/fixtures/make_v1_0_fixture.py`).

## Safety rules for contributors
- Build R text only through `rbridge/codegen.py` helpers: `r_str` for any user-derived text and
  `alias()` for identifiers. Variable names from files never become R identifiers.
- Never pass two strings where R expects one (`notes = "a", "b"` binds "b" to the next argument);
  use `paste0`.
- Every statistical feature needs a test against an independent reference.
