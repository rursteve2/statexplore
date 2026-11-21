# Release audit — StatExplore 1.0.0 (2026-11-21)

Checked against `StatExplore_Final_Master_Prompt.pdf` before publishing.

## 1. §28 Final acceptance workflow — executed end to end on the HRS file
Script: one pass through all 20 steps with real data (16,000 rows × 168 variables; CES-D 8 derived;
wave 10 only; 14 candidate predictors; up to 6 predictors; two- and three-way interactions).

| # | Step | Result |
|---|---|---|
| 1 | Opens on Windows | PASS: window "StatExplore 1.0.0", 11 sections |
| 2 | Imports CSV/SAV/XLSX | PASS: SAV 16,000 × 168 (CSV/XLSX covered by tests) |
| 3 | Descriptive statistics and variable information | PASS |
| 4 | One DV, up to 39 candidates | PASS |
| 5 | Max six predictors per model | PASS |
| 6 | Two-way and optional three-way | PASS |
| 7 | Bounded search space calculated | PASS: 455 candidates, shared n = 662 |
| 8 | Analysis initiated | PASS: 9 s with 4 R workers |
| 9 | Eligible candidates evaluated, all attempts recorded | PASS: 239 tested, 216 invalid (with reasons) |
| 10 | Significant interactions identified | PASS: 24 raw p < .05 (≈ 12 expected by chance), 2 after FDR |
| 11 | Raw and adjusted p-values | PASS: raw, BH, Holm |
| 12 | Uncertainty and diagnostics | PASS: CIs, effect sizes, screening flags |
| 13 | Separate discovery and support rankings | PASS |
| 14 | Select an interesting model | PASS: self-rated health × drinking × neuroticism |
| 15 | Interaction graph | PASS |
| 16 | Brief interpretation | PASS: 5 sentences; notes selection from 455 candidates |
| 17 | Compare with alternatives | PASS: nested F-test vs main-effects model |
| 18 | Reproducible R code | PASS |
| 19 | Results and graphs exported | PASS: xlsx, docx, pdf, R, py, png |
| 20 | Save and reopen | PASS |

## 2. Section-by-section compliance
See [`spec-compliance.md`](spec-compliance.md): every section of the PDF with status and evidence.
Items found missing during the audit and added before release: model pre-checks (user-reported
error), mediation (§9.8), missing-data plot (§8/§17), validation-status and teaching-interpretation
columns (§10.8), user-set held values (§16.3), random-effects / autocorrelation / comparison plots
(§15.5/§16.1), all-two-way candidate models (§10.3), caching (§21), Test C (§25), Cronbach's alpha (§5).

## 3. Automated tests (§25)
152 tests pass, including Tests A–O. Statistical results are verified against independent
implementations (scipy, statsmodels, numpy).

## 4. Packaged build (§3.1, Phase 7)
`StatExplore.exe --selftest --full` runs one real analysis per workspace (regression, logistic,
ANOVA, mediation, Teaching Studio, comparison, lme4 / nlme / mmrm mixed models, cross-validation,
discovery) plus every export format and a project save/reopen, using **only** the bundled R and
package library. The result is recorded in section 6.

## 5. Repository checks before going public
- No credentials or tokens (scan); no data files tracked (HRS data is not redistributed).
- MIT license; third-party notices for the bundled R, R packages, Python and Qt.
- The commit author email (`rursteve2@yahoo.com`) is visible in the public history.

## 6. Packaged-build self-test and release files
- `dist\StatExplore\StatExplore.exe --selftest --full`: **OK** (bundled R 4.6.2 at `R\`, packages at `rlib\`;
  linear, logistic, ANOVA, mediation, Teaching Studio, comparison, lme4, nlme AR(1), mmrm,
  cross-validation, discovery, 46 export files, project save/reopen). The lme4 "warning" is a correct
  singular-fit notice on the synthetic test data.
- **Installer tested**: silent per-user install into a temporary folder → `--selftest --full` from the
  installed copy (using the installed R) → **OK** → uninstall → nothing left behind.
- Portable zip: 11,279 entries, integrity verified; contains the program, R, packages, LICENSE and notices.

| File | Size | SHA-256 |
|---|---|---|
| StatExplore-1.0.0-Setup.exe | 222 MB | `731aa36cf172b2c219e539baa32fe578a49a76b3b5a33f45e82eedcc5a16d225` |
| StatExplore-1.0.0-portable.zip | 303 MB | `5cbbbf11ec667343dea75667aa91eb07ea17135401add19386a6b81054ac9c85` |

## Known limitations (also in the README and the release notes)
- Not yet tested on a freshly installed Windows 10/11 machine; the release notes ask users to report
  problems via Issues.
- flexplot is optional and not bundled; R packages are pinned via a lockfile instead of renv; R
  components are tested via pytest instead of testthat.
- Not included: multiple imputation, logistic mixed models, Games-Howell, robust SEs, factor analysis.
