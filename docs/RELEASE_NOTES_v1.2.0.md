# StatExplore 1.2.0

Reliability, discovery and longitudinal upgrade of StatExplore 1.1. Everything still runs offline; R and
every statistical package are included. Projects saved with 1.0 or 1.1 open unchanged, and their saved
searches keep their results (older mixed-model searches show their participant count as *Groups*, not
as a degrees-of-freedom column).

## Download (choose one)
- **`StatExplore-1.2.0-Setup.exe`**: installer (recommended).
- **`StatExplore-1.2.0-portable.zip`**: no installation; unzip and run `StatExplore.exe`.

Windows may say "Windows protected your PC" (the program is not code-signed): **More info ▸ Run anyway**.
Requirements: Windows 10 or 11 (64-bit), about 1.5 GB of free disk space.

Checksums (SHA-256): Setup `d8e6b1b56e3cf88588b42f1f3f00524a1fee03ba3fd3e4cd17f6ed52f389670c` · portable zip `35523a62734df5c46321e8cdc9d4af98bc1fbf0a41764bde84b66cfa48908255`.

## Fixed
- **Sorting** of p-values, adjusted p-values, confidence limits, estimates, AIC and every other number in
  the Discovery tables (and the other result tables) is now numeric. Missing values always sort last,
  in both directions. In 1.1, clicking *p (adjusted)* produced an essentially random order.
- **Cached fits are never reused after a coding change.** In 1.1 a search could reuse fits from an
  earlier search with a different binary event category or multinomial reference. The cache identity
  now covers the data and metadata as the search sees them (treatments, filters, missing codes,
  category order), the resolved outcome coding, the participant / time structure, the random-effect
  or covariance structure, the sample, and the R and package versions. Changing the significance level,
  correction or highlighting never refits.
- **Group counts are no longer shown as degrees of freedom.** Mixed-model searches report *Groups
  (participants)* separately; likelihood-ratio tests carry their chi-square df.
- **The treatment panel shows every model variable.** The outcome is always shown (directly below the
  outcome family), and so is every checked predictor — with *Already continuous — no override needed* or
  *Unavailable* and the reason, instead of silently disappearing.

## New in Automated Model Discovery
- **All eligible interaction combinations** (new default): every hierarchical set of interaction terms on
  every subset of predictors — main effects only, one interaction, several interactions at once, all
  two-way interactions and (when enabled) three-way interactions. Also: *Full factorial*, *Main effects
  only*, and the 1.1 *Single focal interaction* grammar.
- **Six distinct predictors**, not six terms: tables show *Predictors* (e.g. `3 / 6`), *Model terms* and
  *Estimated parameters* (from R's design matrix). Equivalent formulas are deduplicated.
- **Every p-value names its hypothesis**: individual interaction term, joint test of all interactions,
  main-effect term, whole model, focal interaction, or covariance-structure comparison — each its own
  multiple-testing family, never ranked together. Each row shows the reference model and the test.
- **Highlight findings**: rows with adjusted p < .05 are highlighted by default (subdued background and a
  ★ text marker). Options: raw p, a *strict* rule that also requires the confidence interval of the
  same coefficient to exclude its null value (0, or 1 for odds ratios; joint tests never qualify), off;
  α presets or a custom value; *Show highlighted only*; counts. Saved with the project. Screening
  warnings are flagged separately (⚠), never as non-significance.
- **Model comparison** tab (AIC / BIC / R² / adjusted R², ΔAIC only when all models used the same rows),
  **Compare with main-effects model**, odds-ratio columns for logistic models, and filters by hypothesis
  type, model type, diagnostics and CI availability.
- **Workload estimate** before starting (models and model fits), minimum predictors and minimum complete
  rows per model.
- Large result tables (hundreds of thousands of hypotheses) sort in a fraction of a second.

## Longitudinal discovery
- **Study structure**: *Independent observations* or *Repeated measures / longitudinal* with an explicit
  participant ID (any column with repeated values; ID-role variables suggested first) and time variable.
  A preflight shows rows, participants, waves and observed outcomes by wave (unobserved waves are
  reported, never filled).
- **Workflow A — fixed-effect discovery** with one locked structure: lme4 random intercept / random slope
  (correlated or uncorrelated), nlme random effects with AR(1) or continuous-time AR(1) residuals, mmrm
  marginal models (unstructured, compound symmetry, AR(1)), binary logistic mixed models (glmer), ordinal
  mixed models (clmm), and GEE (population-average, robust Wald tests). Strategy *Time × each predictor*.
- **Workflow B — covariance-structure comparison** with the fixed part locked: REML fits, uniformly
  computed AIC/BIC, boundary-corrected likelihood-ratio tests for nested variance components, QIC for GEE.
- Longitudinal candidates open in **Mixed & Longitudinal Models**, refitted with the same formula,
  structure, ML estimation and coding (clearly labelled as a refit), with population and participant
  trajectories.

## Manual models
- Adjustable **complexity limits**: regressions default to 6 predictors (raise to 10, 15 or more), factorial
  ANOVA to 3 factors (up to 5), ANCOVA 2 factors + 4 covariates, Mixed & Longitudinal 6 fixed-effect
  predictors. A preflight shows distinct predictors, terms, fixed coefficients after dummy coding, complete
  rows, residual df and factor cells.

## Exports
- Excel exports include every tested hypothesis with raw and adjusted p, its family, CI type and scale,
  and the highlight settings as presentation metadata.

## Known limitations
See README §10. Not included: four-way interactions in searches; repeated multinomial outcomes in
discovery; selection-adjusted confidence intervals; the optional research extras of the upgrade
specification §13 beyond those listed above.
