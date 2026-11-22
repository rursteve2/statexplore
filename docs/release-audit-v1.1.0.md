# Release audit — StatExplore 1.1.0 (2026-11-22)

Checked against *StatExplore v1.1 Existing-Codebase Upgrade Specification* before publishing.
Section-by-section evidence: [spec-compliance-v1.1.md](spec-compliance-v1.1.md).

## 1. §23 Final user acceptance workflow — executed through the GUI on the HRS file
`tests/test_acceptance_workflow.py` (part of the regression suite; run with `-s` for this evidence).
Data: `hrs_least_missing_1000.sav` (16,000 rows × 168 variables, 1,000 people × 16 waves).

| # | Step | Result |
|---|---|---|
| 1 | Open StatExplore | PASS: window "StatExplore 1.1.0", 13 sections |
| 2 | Import the HRS SAV | PASS: 16,000 × 168 |
| 3 | Open Dataset Explorer | PASS |
| 4 | Find R_LTACTX | PASS: listed in Measurement Review (26 variables flagged) |
| 5 | Review nominal SPSS metadata and suggested ordinal level | PASS: SPSS level nominal, suggested ordinal; codes, labels, counts; direction "higher codes = less frequent" |
| 6 | Approve ordinal | PASS: status approved; source level still nominal |
| 7 | Open Mixed & Longitudinal Models | PASS |
| 8 | Select R_LTACTX as outcome | PASS |
| 9 | Choose an ordinal model (or treat as continuous for Gaussian) | PASS: ordinal family (ordinal::clmm); Gaussian requires the "treat as continuous" checkbox |
| 10 | Participant ID, time, predictors | PASS: HHIDPN, WAVE, RAGENDER |
| 11 | Review missing observations by wave | PASS: unobserved Wave 1 to Wave 6; analysis range Wave 7 to Wave 16 |
| 12 | Fit a statistically appropriate longitudinal model | PASS: ordinal logistic mixed model, n = 9,929 observations |
| 13 | Coefficients, uncertainty, diagnostics, plots | PASS: ORs with 95 % CIs and p; thresholds; random-effect variance; convergence / Hessian checks; availability table |
| 14 | Compare alternative specifications | PASS: Gaussian model on codes 1–5 (5 plots) and random-effects comparison |
| 15 | 2 × 2 OR, RR, RD in Analysis Builder | PASS: wave 10, smoking × gender: OR 0.78, RR 0.79, RD −1.5 percentage points |
| 16 | Discovery with a categorical outcome | PASS: R_SHLT (ordinal family), 10 candidates, complete |
| 17 | Standard variable-selection controls | PASS: search "Big 5" → Check visible (5); required covariates have the same controls |
| 18 | Open R Console | PASS: security warning shown |
| 19 | Custom script on a copy of the data | PASS: table + barplot of my_data$R_SHLT, plot captured |
| 20 | Install a local R package | PASS: binary package installed offline into the user library |
| 21 | Use the new package in a script | PASS |
| 22 | Survey Analysis | PASS: no sampling weight, PSU or strata in this extract (R_WEIGHT is body weight); nothing invented |
| 23 | Export results and reproducible code | PASS: 24 files (Excel, Word, PDF, R scripts, Python launchers) |
| 24 | Save and reopen | PASS: results, search and approved levels kept |
| — | Original SAV unchanged | PASS: SHA-256 identical before and after |
| — | Works without internet | PASS by construction: no network calls; packages install with `repos = NULL` |

## 2. Automated tests
- Baseline v1.0.0: 152/152.
- v1.1.0: **247/247** (`pytest`, R 4.6.2, all dependencies present; no skips).
- Numerical references: statsmodels (OLS, Logit, OrderedModel, MNLogit, MixedLM, GEE, Table2x2, proportion CIs),
  scipy (chi-square, Fisher, odds ratio, Spearman, Cramér's V), pbkrtest, closed-form survey linearization,
  and independent numpy Gauss-Hermite maximum-likelihood fits for glmer and clmm.
- Backward compatibility: a project written by the real v1.0.0 code opens, reproduces its stored numbers
  (1e-9), re-runs its search identically, exports, and is backed up before its first v1.1 save.

## 3. Packaging and installer (development PC, Windows 11)
- `python packaging/build.py`: PyInstaller app + bundled R 4.6.2 + `rlib\` (now including ordinal, nnet,
  mclogit, geepack, survey, vcd, pbkrtest and their dependencies; versions in `rlib\packages.lock.json`),
  `r\console_worker.R` bundled next to `worker.R`.
- Frozen build: `dist\StatExplore\StatExplore.exe --selftest --full` → **OK** using only the bundled `R\`
  and `rlib\` (22 checks: every v1.0 workspace, 2×2 association, ordinal / multinomial regression,
  continuous treatment, glmer, clmm, GEE, survey, discovery, 66 exported files, project round-trip, R Console).
- Installer: `StatExplore-1.1.0-Setup.exe /VERYSILENT /CURRENTUSER /DIR=<temp>` → installed copy's
  `--selftest --full` → **OK** → `unins000.exe /VERYSILENT` → folder and uninstall registry entry removed.
- Artifacts (SHA-256):
  - `StatExplore-1.1.0-Setup.exe` (229 MB): `8a634fbee48d73fa1c8f6c15126a062f230ef9f6e4ba8ea68f929b10950d55c9`
  - `StatExplore-1.1.0-portable.zip` (311 MB): `02ad56583f64a7fd8a42273efebbf03f6b53974941e7bf057e9c9ae73cedf3bc`

## 4. Open items
- ⏳ Install on a clean Windows 10/11 machine with networking off (user step, as for 1.0).
- Limitations listed in README §10 and spec-compliance-v1.1.md (◐ rows).
