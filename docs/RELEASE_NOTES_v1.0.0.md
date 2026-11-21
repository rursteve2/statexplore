# StatExplore 1.0.0

First release of StatExplore: a Windows desktop program for exploring datasets, searching them for
statistical interactions, and teaching what those results do and don't mean. It runs entirely
offline; R and every statistical package are included.

## Download (choose one)
- **`StatExplore-1.0.0-Setup.exe`**: installer (recommended). Double-click, click through, then start
  StatExplore from the Start menu.
- **`StatExplore-1.0.0-portable.zip`**: no installation needed. Unzip anywhere (e.g. Documents) and
  double-click `StatExplore.exe` inside the folder.

**Windows may say "Windows protected your PC".** The program isn't code-signed (signing certificates
are paid), so Windows warns about every new program downloaded from the internet. Click
**More info ▸ Run anyway**.

Requirements: Windows 10 or 11 (64-bit), about 1.5 GB of free disk space. No internet connection,
Python or R installation needed.

Checksums (SHA-256): Setup `731aa36cf172b2c219e539baa32fe578a49a76b3b5a33f45e82eedcc5a16d225` · portable zip `5cbbbf11ec667343dea75667aa91eb07ea17135401add19386a6b81054ac9c85`.

## What's in it
- Import CSV, Excel and SPSS (labels and missing-value codes kept); variable management with undo;
  derived variables and filters.
- Exploratory analysis: psych descriptives, plots, correlation tests, missing-data views.
- Analysis Builder: t-tests, correlations, ANOVA/ANCOVA (incl. repeated measures), linear,
  hierarchical and logistic regression, mediation, Cronbach's alpha.
- **Automated Model Discovery**: bounded exhaustive search for two- and three-way interactions, with
  raw and multiple-testing-adjusted p-values, a significance ranking and a statistical-support
  ranking, and every attempt recorded. Supports repeated measures via mixed models.
- Teaching Studio: model-based interaction plots with plain-English interpretations.
- Mixed and longitudinal models (random intercepts/slopes, AR(1), unstructured covariance).
- Validation: holdout, grouped/temporal cross-validation, resampling stability, full-procedure CV.
- Reports: Excel, Word, PDF, and R scripts that reproduce each analysis.

See the README for a step-by-step guide and `docs/spec-compliance.md` for the full requirement checklist.

## Known limitations
- Not yet tested on a freshly installed Windows machine outside the development computer. Please
  report problems via GitHub Issues.
- The optional flexplot plotting backend isn't included (ggplot2 and base R plots are).
- Not included: multiple imputation, logistic mixed models, Games-Howell post hoc, robust standard
  errors, factor analysis.
- StatExplore is for exploration and teaching, not confirmatory research.
