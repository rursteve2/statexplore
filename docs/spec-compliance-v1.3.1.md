# StatExplore 1.3.1 — spec compliance, traceability and test evidence

Spec: `StatExplore_v1.3.1_Developer_Prompt_Complete.md` (§ numbers below). Historical inventory:
`StatExplore_v1.2.0_to_v1.3.0_Test_Inventory.md`. Baseline: v1.3.0 (`a7bae5e`), 345/345 tests on a clean
worktree. Environment: Windows 11, R 4.6.2, ordinal 2026.7.26, nlme/lme4/mmrm/geepack/nnet as in
`r/packages.lock.json`, Python 3 venv, PySide6.

Evidence folder (kept out of the repo because it contains health data): `E:\StatExplore_v1.3.1_Testing\`.
- `logs/`: before/after reproductions, HRS cross-module run, HRS export runs, full pytest log.
- `exports/`: each HRS export with its `source_log.txt`.
- `screenshots/`: before (v1.3.0) and after (v1.3.1) at 1366×768, 1920×1080 and 2560×1440; after also at scale
  1.25 and 1.5.
- `tests/`: driver scripts.

## 1. Defect → root cause → fix → test

| ID | Root cause (code level) | Fix (module) | Test(s) | Status |
|---|---|---|---|---|
| STAT-15 | `nominal_test()` → `try(update())` re-evaluates the call outside the data's scope → silent NA | `rbridge/codegen.py` `sx_po_check`; `analysis/builders.py` (ordinal_reg, likert) | 26 ×3, 35 (HRS) | pass |
| STAT-16 | discovery recomputed BIC with log(n); REML `nlme` `BIC()` uses log(n − p) | `codegen.sx_ic`; `discovery/rcode.py`; `discovery/engine.py` (no override, `FIT_RULES`); `analysis/mixed.py`; `descriptor.verify` (BIC) | 27 ×3, 35 | pass |
| P1-TIME a | per-model sample: `sx_occ` built before dropping incomplete rows | `discovery/rcode.py` (index rebuilt per model) | 29 per-model | pass |
| P1-TIME b | wording claimed equal spacing; lag basis undefined | `longitudinal_preflight.py`, `descriptor.TIME_INDEX`, `mixed.py` lag table | 29 ×3 | pass |
| P1-EXPORT | printed app values next to new ones; only `n` compared; no data/package checks | `rcode.verification_script`, `sx_rows_sig`; `export/exporters.write_portable` | 30 ×3, HRS export runs | pass |
| P1-SEARCH | status counts merged reuse / not visited; structure LR on same n only | `engine.summary`, `structure_tests` (row signature) | 31 ×2, 30 signature | pass |
| P1-MODEL | no user cancel for long fits; timeout text generic | `gui/state.py` (`cancel_r`, timeout label), `main_window` Stop R | 28 timeout/cancel | pass |
| UI-28 | selector button bar forced ~390 px; auto-sizing overrode user drags | `widgets/variable_selector.py`, `pages/discovery.py` | 32 ×3, screenshots | pass |
| UI-29 | no search/sort/scope in Measurement Review | `pages/measurement_review.py`, `measurement.review_rows` | 33, acceptance | pass |
| UI-30 | QLabel auto text format showed escaped text literally | `pages/discovery.py`, `pages/r_console.py` | 34 | pass |
| UI-31 | checked items not grouped | `variable_selector` (`selected_first`) | 33 selector | pass |
| UI-32 | default thin outer scroll bar | `widgets/layout.section_scroll` | 34 | pass |
| UI-33 | dropdowns / labels fixed width | `widgets/layout.make_responsive` | 34, screenshots | pass |
| UI-34 | no minimum width, density, freeze, tooltips | `pages/discovery.ResultTable`, `gui/models` | 34, 32 | pass |
| §6.6 | `p = 0` shown after underflow | `gui/models.fmt_p` | 34 | pass |

## 2. §5.1 historical regression (v1.2.0 → v1.3.0)

| Stage | Regression | Fixture | Command / test | Expected | Actual | Status |
|---|---|---|---|---|---|---|
| 1–9 | single-level discovery, config, comparison, UI | synthetic suites | full pytest | as in v1.3.0 | pass | pass. Individual per-test labels for stages 2–5 and 7–9 are not recoverable and are not asserted |
| 6B | interaction LR df 4 | — | not re-run | χ² 14.057 | — | **not verified** (configuration not recorded) |
| 10 | glmer identity, OR, predictors in UI | synthetic panel | test_28_glmm…; `test_P0_*` | engine glmer, verified | verified | pass (synthetic). HRS n ≈ 9874 configuration not recorded: **not verified** |
| 11 | clmm identity, thresholds; timeout/cancel | synthetic panel | test_28_glmm_and_clmm, test_28_timeout_and_cancel, test_P1_06 | verified; never Complete | as expected | pass. HRS reduced panel (n 3968, AIC 8427.137): **not verified** (panel selection not recorded) |
| 12 | multinomial R_MSTAT ~ RAGENDER, W16 | HRS .sav | test_35_hrs_multinomial_reference; export run | n 942, AIC 1997.005, BIC 2055.181, χ²(6) 114.4961 | identical (1e-3) | pass |
| 13 | ordinal clm candidate, compare with main effects | HRS .sav | test_35_hrs_ordinal_candidate_and_po_check | n 917, ℓ −1244.243, AIC 2502.487, BIC 2536.234, 4 thresholds, LR .645, p .4218 | identical | pass |
| 14 | 112 candidates, budget 10 → 20, resume | synthetic 4 binary | test_31_*; `logs/P1-SEARCH_112_candidates_v1.3.1.txt` | 112; no duplicates; resumable after reopen | as expected | pass. The "cap 5 screens 16 of first 20" figure depends on the original HRS predictors (not recorded): **not verified**. With binary predictors all of the first 20 have ≤ 5 parameters |
| 15 | RI / AR(1) / CAR(1) REML, 3901 rows, 983 people | HRS .sav + .csv | test_35_hrs_covariance_bic_native; `test_fixture_D_E…`; cross-module; export runs | AIC 20640.342 / 20512.787 / 20540.106; BIC 20550.397 / 20577.716; LR 129.554778 (p 5.128e-30), 102.235740 (p 2.465e-24) | identical | pass |
| 16 | Workflow A CAR(1) ML | HRS | test_fixture_D_E; stat16 log | AIC 20532.121, BIC 20569.735, p .7863 | identical | pass |
| 17 | family counts | HRS W16 | test_fixture_A_hrs_ordinal_families | 0/16, 0/5, 3/9, 14/18 | identical | pass |
| 18–19 | import, whitespace | HRS .sav/.csv; 12×5 fixture | test_35_whitespace…, test_P0_05_* | 16,000×168, 767,497 missing; BMI 4, neg/dec 2 missing | identical | pass |
| 21–23 | transfers keep identity | HRS | `logs/cross_module_hrs_v1.3.1.txt` | verified | 7/7 verified (RI, AR1, CAR1, CAR1-ML, GEE ×3) | pass |
| 22 | export reproduction | HRS .sav and .csv | `logs/export_reproducibility_hrs_{sav,csv}.txt` | self-verifying scripts pass in clean R | 5/5 + 5/5 ALL VERIFIED | pass |
| 24 | resize / reorder kept | GUI | test_32, `test_P2_*`, screenshots | splitters, column move | kept | pass |

## 3. §5.2 Tests 26–35 (`tests/test_v131.py`, 28 tests)

| Test | Configuration / fixture | Tolerance | Result |
|---|---|---|---|
| 26 | synthetic ordinal (main, interaction); PO vs non-PO (n 1500); sparse 3×4 with an empty cell; checked independently with `nominal_test` in a clean R session | p rel 1e-6 (main), 1e-5 (interaction) | pass |
| 27 | synthetic panel: lme4 RI, nlme RI/AR1/CAR1 (REML), CAR1 (ML) vs base `BIC()` in clean R; verify() unit test | 1e-6 | pass |
| 28 | GEE ×3, multinomial (ref = 2), glmer, clmm; timeout + Stop R + restart | identity tolerances | pass |
| 29 | waves 1–4, 6 (5 absent globally); pattern A skips wave 3. φ and `corMatrix` checked against φ^distance in clean R; per-model index; wording | 1e-6 / 1e-8 | pass |
| 30 | AR1, CAR1, ordinal exports sourced from another folder; changed data → MD5 stop; `sx_skip_checksum` → n / row check stop; missing package stop; row-signature guard | stated in script | pass |
| 31 | 112 candidates; budget, reopen, resume; cache reuse and invalidation (measurement, filter) | exact | pass |
| 32 | 1, 3, 126 candidate rows × 1366×768, 1920×1080, 2560×1440; user drag kept | — | pass |
| 33 | 166 variables: search, saved override, sort, persistence after reopen; selector with 166 items | exact | pass |
| 34 | entity rendering, section scroll bar, dropdown tooltips, p display, long-cell tooltip | exact | pass |
| 35 | HRS end to end (ordinal, PO, covariance BIC/LR, multinomial, import counts) | 1e-3 / 1e-4 | pass |

DPI: offscreen captures at QT_SCALE_FACTOR 1, 1.25 and 1.5 (`screenshots/after_v1.3.1_scale*`). Real Windows
display scaling on physical monitors has not been checked by a person: **not verified**.

## 4. Unsupported or not done

- Participant × wave coverage heatmap (§6.3): not added. Per-wave availability tables and observed-wave lists
  already exist.
- Autosave/undo of settings (§6.7), performance telemetry beyond elapsed time (§6.8): not added.
- Resume does not retry timed-out candidates. They stay *timed out*, and the run is *incomplete* with
  provisional adjusted p-values.
- Text participant IDs are not pseudonymized in exports, because R's locale-dependent sort could reorder the
  rows. Only numeric IDs are replaced.
- Row signatures include factor coding: the same rows from .sav (labelled) and .csv (codes) have different
  signatures by design.
- Settings use the Windows registry (`QSettings` native format), also during tests.
- Clean-machine install of the 1.3.1 build: user step.

## 5. Run, install, rollback

- Tests: `TEMP=E:\sx_tmp TMP=E:\sx_tmp .venv\Scripts\python -m pytest -q --basetemp=E:\sx_tmp\t`.
- Build: `set STATEXPLORE_BUILD_DIR=E:\statexplore_build` then `.venv\Scripts\python packaging\build.py`. Output
  goes to `E:\statexplore_build\dist\release`. Then run `StatExplore.exe --selftest --full`.
- Install: run `StatExplore-1.3.1-Setup.exe`. It installs over 1.3.0, and projects open unchanged.
- Rollback: uninstall, then install `StatExplore-1.3.0-Setup.exe` from the v1.3.0 GitHub release. Project files
  saved by 1.3.1 add only new row fields, which 1.3.0 should ignore. This was not tested: keep a copy of the .sxp
  before rolling back.
