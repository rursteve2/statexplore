# StatExplore 1.3.0 — release notes

1.3.0 fixes defects found in 17 stages of structured user testing. The common theme is that a model or a
claim changed when it moved between modules, into exported code, or into a summary sentence. The statistical
engines were already correct. On the reference HRS data, every engine statistic (Fixtures A, C, D, E) reproduces
exactly before and after the fixes. The fixes are in model transfer, export, reporting and import.

## Fixed (release blockers)

- **Models keep their identity across modules (P0-01).** Each discovery candidate now carries a model descriptor:
  family, engine, link, outcome coding, terms, ID/time, structure and time-index meaning, ML/REML, sample, fit
  statistics and parameter counts. Every downstream action is built from it.
  - Model Comparison and *Compare with main-effects model* now fit ordinal models with `ordinal::clm` and
    multinomial models with `nnet::multinom`. They previously refitted them silently as a binary `glm` of
    "category 2 vs the rest" (Defect A).
  - Workflow-B structure candidates no longer lose their structure. CAR(1), AR(1) and GEE candidates previously
    opened as an lme4 random-intercept ML model (Defects B and C).
  - Binary models keep the search's event category.
  - Every refit is checked against the search (n, log-likelihood, AIC) and labelled *Model identity verified*.
    If the check fails, the result says so.
  - Destinations that cannot represent a model say so specifically. The Teaching Studio draws only linear and
    logistic models. Longitudinal candidates are not sent to the single-level comparison.
- **Exports reproduce the model and its comparisons (P0-02).**
  - The candidate's own `struct` (for example `"car1"`) is exported, never `""`.
  - Workflow-B exports refit the reference structures and recompute the reported likelihood-ratio tests,
    including the boundary χ² mixture.
  - *Save script + data…* writes `analysis.R` plus `analysis_data.csv` holding only the analysis columns. The
    script finds its data next to itself, so `source("analysis.R")` works from any folder.
  - No temporary or personal paths, no unused factor-level vectors (such as 1,000 household IDs), and wrapped
    lines.
  - Reference check: the exported HRS CAR(1) candidate reproduces n 3901, AIC 20540.106, logLik −10264.053,
    6 likelihood parameters and the covariance LR test, from both the `.sav` and the `.csv`.
- **Headlines no longer conflate hypothesis families (P0-03).** Interaction findings are counted only from
  interaction tests, so Fixture A now reads "0 significant interactions". Main-effect, whole-model and
  covariance-structure results are reported as what they are. Each family shows tested / raw / BH / Holm counts
  and its own chance expectation.
- **CSV whitespace no longer hides numeric outcomes (P0-05).**
  - Empty and whitespace-only cells (spaces, tabs, mixed) are missing values. Numbers with surrounding spaces,
    signs, decimals and scientific notation are parsed.
  - The number of normalised cells is reported, empty columns and mostly-numeric text columns are reported, and
    zero-padded codes stay text.
  - Numeric storage alone no longer implies continuous: a few unlabelled integer codes are read as categories
    and flagged for review.
  - The HRS CSV now imports with the same storage types and values as the `.sav` for all 168 columns
    (`R_PMBMI` was text before).
- **ML/REML and comparison guards (P0-04).** Workflow B is labelled REML everywhere. An *Estimation* column
  shows ML / REML / OLS / GEE. Model Comparison refuses models with different families or outcome coding, and
  runs LR tests only for nested models on identical rows.

## Fixed (statistical reporting)

- **Full Analysis of `nlme` models (P1-01).** Discovery models open with the inference their engine supports
  (nlme containment df, mmrm Satterthwaite, LRT for ML lme4). The message about the Kenward–Roger /
  likelihood-ratio setting now explains that it is implementation-specific.
- **Time index (P1-02).**
  - AR(1) is labelled *observed-occasion index* and CAR(1) *actual time differences*.
  - The preflight lists the observed time points and any time point absent for everyone inside the range, with a
    gap-compression warning.
  - Lags in residual autocorrelation tables are described as observed-occasion steps.
- **Missingness (P1-03).**
  - The preflight now gives usable participants (2+ rows, 1 row) and participants with intermittent gaps.
  - The MAR wording is qualified.
  - Trajectory predictions at times without observed outcomes are marked model-based.
- **Parameter counts (P1-04).** *Fixed coefficients*, *Covariance parameters* and *Likelihood parameters* (the k
  in AIC) are separate columns, with header definitions.
- **Interpretations (P1-05).** "Strongest term" became "term with the smallest p-value". Models with
  interactions get a note that main effects are conditional.
- **Diagnostics and timeouts (P1-06).**
  - Worker timeouts are marked *timed out*, and the search is *Incomplete*, not *Complete*.
  - When `ordinal::nominal_test` cannot run, the analysis says why.
- **GEE QIC (P1-07).** QIC was always missing in discovery: `geepack::QIC` could not re-evaluate the model call.
  It is now computed. If it cannot be, the reason is recorded.
- **Multinomial (P1-08).**
  - Every coefficient row names its comparison and reference categories.
  - Categories absent from the analysed rows are reported.
  - Model fit shows logLik and BIC.

## Usability

- Searchable dependent-variable dropdowns in every module. Variables can be pinned to the top of every list
  (right-click). Variables not offered as outcomes are listed with the reason.
- The Mixed form shows a transferred model's predictors as checked; v1.2 showed none.
- Ctrl+F find bar in code listings and Mixed results.
- **UI-28:**
  - Discovery has a resizable, collapsible *selected-model details* pane under the result tables. It shows the
    model identity, fit, parameters and every hypothesis test, and double-clicking a row opens it.
  - The configuration panel is collapsible.
  - Table columns can be resized, reordered and hidden (right-click the header), and the rows are compact.
  - *Copy CSV* is available on every results table.
  - Pane sizes and column layouts are remembered.
  - The Mixed preflight sits in a resizable pane with *Copy preflight*.
- Filter scope (P2-03): a line under the filters says which filters apply to the visible tab, and whether it lists
  hypothesis tests or models. Counters say "tests" or "models".
- Strict highlighting (P2-04): when significant tests are not strictly highlighted, the reason is stated.
- Resume (P2-05): the summary lists each execution's budget and candidate range. Not-yet-visited and timed-out
  candidates are counted.

## Compatibility

- Projects from 1.0–1.2 open unchanged. New fields (model coding, descriptor, executions) have defaults.
- Stored measurement levels are never re-inferred. Only new CSV/Excel imports use the new typing rules.
- Model Comparison's *parameters* column now counts likelihood parameters. For linear models this includes the
  residual variance, so it is one higher than in 1.2.

See [spec-compliance-v1.3.md](spec-compliance-v1.3.md) for the defect-to-fix traceability, test evidence and known
limitations.

## Downloads (SHA-256)

| File | SHA-256 |
|---|---|
| `StatExplore-1.3.0-Setup.exe` | `f792eb30781ba61b5bd8c024a67f5d466b05ca9e669674b0224a03d6fdc15a7b` |
| `StatExplore-1.3.0-portable.zip` | `28774e3f4a11a6fbd5f07b0e7afa84d6d97b6fc3bf489880e707955eebf31a84` |

Verified before release: 345/345 automated tests. The frozen build's `--selftest --full` passes (every engine,
discovery, exports, project save/reopen, R console).
