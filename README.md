# StatExplore

Offline Windows desktop app for exploratory statistical model discovery and teaching
(Python/PySide6 front end, R statistical engine). Build plan: `tasks/todo.md`.
Architecture: `docs/architecture.md`.

**Status: Phases 1–3 complete (foundation, core statistical engine, automated model discovery).**

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
`install_packages.R` puts the required R packages (jsonlite, psych, ggplot2, car, afex, emmeans,
effectsize, ppcor, lmtest, detectseparation and their dependencies) into `r\library` (project-local).
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

## Phase 2 features
- Analysis Builder: one-sample / Student / Welch / paired t-tests; Pearson, Spearman, partial correlation,
  correlation matrices (raw + Holm/BH/Bonferroni p); one-way, Welch, factorial (≤3 factors, Type III),
  repeated-measures, mixed-design ANOVA and ANCOVA (afex) with EMMs and adjusted pairwise comparisons;
  linear regression (interactions, polynomial/spline term, standardized betas, nested term tests,
  conditional slopes, diagnostics + plots); hierarchical regression (R² change); binary logistic regression
  (ORs with profile CIs, LR tests, predicted probabilities + plot, separation detection).
- Derived variables (item mean/sum, recode, log, sqrt, z-score, center, square, binarize) and row filters —
  computed again in R, so View Code reproduces them from the source file.
- Warnings when a participant-ID variable shows repeated rows (non-independent observations).

## Phase 3 features
- Automated Model Discovery: bounded exhaustive search for two- and three-way interactions over up to 39
  candidates (≤ 6 predictors per model), focal or all-subset covariate adjustment, shared or per-model
  complete cases. Exact search-size estimate before running; modes A/B/C/D, Fast/Balanced/Thorough profiles,
  parallel R workers, progress + ETA, cancel and resume.
- Every attempt is recorded (invalid models with reasons). Raw, BH and Holm p-values, provisional when
  incomplete. Discovery-significance and statistical-support rankings (transparent tiers).
- Any candidate opens as a full regression analysis on the same rows; View Code shows the exact R call.
- Candidate grammar: `docs/formula-grammar.md`.

## Known limitations
- Wide-to-long reshaping arrives in Phase 5. Welch ANOVA has no post hoc test yet (Games-Howell planned).
- z-scores / centering of derived variables use all rows (before filters).
- Date/time columns are imported and shown but excluded from analyses.
- Empty strings in text columns are treated as missing.
- Ordinal variables are coded as unordered factors in R (avoids polynomial contrasts by default).
- The R runtime isn't bundled yet; the offline installer is built in Phase 7.
