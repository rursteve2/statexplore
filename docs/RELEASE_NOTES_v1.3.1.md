# StatExplore 1.3.1 — release notes

1.3.1 is a maintenance release. It fixes the two open reporting defects from v1.3.0 testing (STAT-15, STAT-16)
and two defects found while auditing for this release. It adds integrity checks to exported scripts and
completes the open usability items UI-28 to UI-34. All v1.3.0 regression fixtures still pass, and the engines'
numbers are unchanged. The one deliberate change is that discovery BIC now equals R's native `BIC()`.

## Fixed

- **Proportional-odds check no longer shows only dashes (STAT-15).**
  - Root cause: `ordinal::nominal_test()` refits each term with `try(update(...), silent = TRUE)`. `update()`
    re-evaluates the stored call outside the scope that holds the data, so every refit failed silently and the
    table showed NA. Main-effects models were affected too.
  - StatExplore now fits each category-specific model explicitly (`clm(..., nominal = ~term)`, the same test
    `nominal_test` performs). It reports df, LR χ² and p, plus a global test.
  - Each row has a status: *tested*; *approximate* (the likelihood converged but some parameters are not
    uniquely determined, so df is uncertain); or *not tested*. Every row that is not *tested* states its specific
    reason: an empty outcome-by-category cell, rank deficiency, non-convergence or optimizer failure.
  - A missing test is never reported as "proportional odds holds".
  - HRS Wave 16, n = 917: gender p = .4235 and depression p = .8084, identical to `nominal_test` run in plain R.
    The gender × depression row is *approximate* because one of its 20 cells is empty.
- **BIC is identical across Discovery, Full Analysis, exports and R (STAT-16).**
  - Root cause: Discovery recomputed BIC as −2ℓ + log(n)·k for every mixed engine. R's `BIC()` for a REML
    `nlme` fit uses log(n − p), which explains the 0.004 differences.
  - Discovery now takes AIC/BIC from the fitted object (native `AIC()`/`BIC()`). HRS: AR(1) 20550.397 and
    CAR(1) 20577.716, equal to independently sourced `nlme`.
  - mmrm is the one exception. Its native REML criteria count only covariance parameters and use n = subjects,
    so mmrm is put on nlme's convention instead, and the *AIC/BIC basis* field says so.
  - Identity checks now also compare BIC, with a tolerance of 0.0015.
  - Cached fits made by v1.3.0 are refitted, never reused.
- **AR(1) occasion index with per-model samples (found in the P1-TIME audit).** With *complete cases per model*,
  discovery built the occasion index before dropping rows. A wave where the outcome is missing for everyone
  therefore still took an index step, while Full Analysis compressed it, so the two could fit different models.
  The index is now built on each model's own rows.
- **Wording about time spacing.** The old note said discrete AR(1) "treats consecutive observed occasions as
  equally spaced". That is not what the model does:
  - The index is global: a time value nobody has is compressed away.
  - A participant who skips an occasion that others have keeps an index gap, so their correlation is φ².
  - The residual-lag table now shows two lag definitions, successive observed rows and global index gaps. It
    explains why empirical lag correlations of normalized residuals can be near zero or negative while φ > 0,
    and that this is not by itself a model failure.
- **Literal `p = 0`.** An underflowed p-value now displays as `< 1e-300`. Exports keep the full value.
- **HTML entities (UI-30).** Labels that hold escaped text are now rendered as rich text, so `&#x27;` displays as
  an apostrophe.

## Exports (P1-EXPORT)

- *View candidate code* scripts verify themselves with `stopifnot()`: n, the analysis-row signature, logLik,
  AIC, BIC, and every test statistic and p-value against the values the app recorded. Tolerances are stated in
  the script. Workflow-B comparisons check LR and p, and require the reference model to have the same row
  signature, not just the same n.
- The analysis-row signature is an MD5 of the row identities in fitting order plus every factor's levels. It is
  recorded with each candidate and shown in the provenance.
- *Save script + data…* now also writes `analysis_manifest.json` (file MD5, rows, columns, R packages). The
  script checks the packages first and stops with a clear message if any are missing. It refuses a data file
  whose MD5 differs; setting `sx_skip_checksum <- TRUE` overrides that.
- Numeric participant IDs are replaced by order-preserving codes 1…K. Results are unchanged.
- Adjusted p-values are listed with their family, method and scope (complete or provisional).

## Search (P1-SEARCH)

- The summary separates fitted, reused from an identical earlier fit, invalid, failed, timed out and not yet
  visited.
- Structure LR tests require identical row signatures.
- AR(1) and CAR(1) are documented as not nested and are never LR-tested against each other.
- Verified on the 112-candidate fixture: budget 10, then 20, then save, reopen and finish gives 112 unique fits
  in order, with no duplicates.

## Long computations (P1-MODEL)

- **Stop R** in the status bar stops a long Full Analysis fit (for example a full-sample CLMM). The result is
  reported as *Cancelled by the user*, and R restarts cleanly on next use.
- A worker timeout is reported as *Timed out: the analysis did not complete*, never as a result.

## Usability (UI-28 to UI-34)

- **UI-28:** settings columns no longer overflow sideways at 1366 px or at 125/150 % scaling. The variable
  selector's count moved to its own line. A divider the user drags is never overridden by the automatic
  few-rows layout.
- **UI-29:** Measurement Review has search (source name, display name, label, levels, saved override), a scope
  selector (needs review / with a suggestion / all variables) and sorting (name A–Z / Z–A, level, missingness).
  New columns: Source name, Override (for example *saved: nominal → ordinal*) and Missing.
- **UI-31:** in Discovery and Mixed, checked predictors appear first under *Selected (k)*, in the order they were
  checked. This is display only: the predictor set, formula and search order are unchanged. Search text and
  scroll position are kept.
- **UI-32:** the whole-section scroll bar is 16 px wide, high-contrast, and has hover and pressed states. Inner
  tables keep their own bars.
- **UI-33:** dropdowns shrink with the panel and show their full text as a tooltip. Long labels wrap, and form
  rows wrap when narrow.
- **UI-34:** discovery tables have a minimum column width and a header menu: *Fit columns to contents*, *Reset
  column layout*, *Freeze the # column*, row density and font size. Long cells show a tooltip with their full
  text, and a hint appears when more columns are off-screen. Column layouts persist.

## Added (staged after the fixes)

- **Model provenance** in every refit and export: origin, family/link, fixed part, engine/estimation, coding,
  structure and time index, sample, row signature, dataset fingerprint, AIC/BIC basis and R/package versions.
- **Specification differences**: the first table of every Model Comparison lists the terms added or removed,
  family, estimation, sample, and whether a likelihood-ratio test is valid (nested) or only AIC/BIC applies.

## Not done in 1.3.1

See `docs/spec-compliance-v1.3.1.md` §4: participant × wave heatmap, autosave/undo for settings, retrying timed-out
candidates on resume, and the clean-machine install.
