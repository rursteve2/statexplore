# StatExplore

Offline Windows desktop app for exploratory statistical model discovery and teaching
(Python/PySide6 front end, R statistical engine). Build plan: `tasks/todo.md`.
Architecture: `docs/architecture.md`.

**Status: all seven phases built and tested (see `tasks/todo.md`).**

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
`install_packages.R` (in development) puts the required R packages (jsonlite, psych, ggplot2, car, afex, emmeans,
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

## Phase 4 features
- Teaching Studio: interaction plots computed from the fitted model (two- and three-way, categorical and
  continuous, logistic probability curves) with CI bands, raw data, panels, labels; PNG/SVG/PDF; ggplot2 or
  base R. Conditional slopes / simple effects, exact Johnson-Neyman interval, and a 3–6 sentence
  template interpretation under every graph.
- Model Comparison: build a model set (from discovery, the Teaching Studio or by hand), edit copies,
  compare AIC/BIC/R² on a common sample, with nested F / likelihood-ratio tests only where valid.
- Optional flexplot backend: `remotes::install_github("dustinfife/flexplot", lib = "r/library")`
  (GitHub-only package; the option is disabled with an explanation when it is not installed).

## Phase 5 features
- Mixed & Longitudinal Models: structure detection (ID / time / cluster), wide → long restructuring;
  lme4/lmerTest random intercepts and slopes (nested clusters), nlme AR(1) / continuous-time AR(1),
  mmrm unstructured marginal models; continuous, quadratic or categorical time; variance components,
  marginal/conditional R², singularity and convergence checks; population, individual and conditional
  trajectory plots; time / random-effect / residual-structure comparisons with the correct ML/REML rules.

## Phase 6 features
- Validation & Uncertainty: holdout reserved before exploring (rows or whole participants), holdout
  evaluation with an interaction replication test, k-fold / grouped / temporal cross-validation with
  leakage checks, discovery stability by resampling, full-procedure cross-validation (selection inside
  training folds), confidence vs prediction intervals, calibration.
- Missing Data panel: missingness by variable, patterns, usable n per model.

## Phase 7 features
- Reports: Excel, Word, PDF; self-contained R scripts (+ data.csv) that reproduce each analysis outside the
  app; Python launcher scripts; discovery settings, analysis history, variable change log.

## Build an offline installer
```bat
.venv\Scripts\python packaging\build.py        :: dist\StatExplore\ with R + packages bundled
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" packaging\statexplore.iss   :: dist\installer\StatExplore-Setup.exe
dist\StatExplore\StatExplore.exe --selftest     :: writes %TEMP%\statexplore_selftest.txt; exit code 0 = OK
```

## Known limitations
- Binary mixed models and mixed models inside discovery are not available yet. Welch ANOVA has no post hoc test yet (Games-Howell planned).
- z-scores / centering of derived variables use all rows (before filters).
- Date/time columns are imported and shown but excluded from analyses.
- Empty strings in text columns are treated as missing.
- Ordinal variables are coded as unordered factors in R (avoids polynomial contrasts by default).
- The installer has not yet been tested on a clean Windows 10/11 machine (user step).
