# StatExplore v1.2 — specification compliance and feature matrix

Sources: **M** = *StatExplore v1.2.0 — Reliability, Discovery & Longitudinal Upgrade* (master prompt);
**X** = *Exhaustive Model Combination and Predictor Complexity Enhancement*.
Baseline: local `main` at `1673df9` (= GitHub release v1.1.0 commit `99aeb93` + one docs commit; no local
`v1.1.0` tag). Baseline tests on a clean worktree of that commit: **247/247 passed**.

Status: **Implemented+Tested** · **Partial** (implemented with a stated limitation) · **Deferred** · **Blocked/User**.

## Design decisions
Recorded in `tasks/todo.md` (F1–F10): one model row with many typed hypothesis rows; multiplicity family =
hypothesis type; canonical term-set model specs; four strategies with "all eligible interaction
combinations" as the default (X §16 recommended default; M's three targets are covered by
main-effects-only, focal, and focal + covariates/combinations); typed CI metadata; a fit-identity hash for
caching; `n_groups` instead of `df2`; explicit longitudinal workflows A/B; a treatment manager built on
`VariableSelector`. Where M and X differ (default strategy), X's explicit default was followed and M's
modes remain selectable.

## Feature matrix

| Spec | Requirement | Status | Evidence (code · tests) |
|---|---|---|---|
| M §2 | Baseline audit, tests, code map | Implemented+Tested | `tasks/todo.md` v1.2 section; 247/247 on `E:\sx_baseline` worktree |
| M §3 | Numeric sorting (p, CI, estimates, AIC…), missing last both directions, numpy / nullable / Decimal / text numbers, unrounded keys, stable ties | Implemented+Tested | `gui/models.py` `sort_value`, `NumericSortProxy`, `ResultFrameModel` · `test_v12_results_table.py` (5 value types × 2 directions, signed CI, ties, text, filters). v1.1 bug reproduced on the baseline (numpy scalars unsortable by Qt). |
| M §3 | Sort column / direction, search, selection (stable ID) kept across refresh; data never mutated | Implemented+Tested | `ResultTable.set_frame/select_id` · `test_result_table_keeps_sort_search_and_selection` |
| M §3 | Large tables without freezing | Implemented+Tested (benchmark) | Vectorised sort/search, lazy interpretations: 311,286 hypotheses → sort 0.09 s, search 1.3 s, table load 0.5 s, frame build ≈ 5 s (`E:\sx_tmp\perf.py`) |
| M §4.1 | Highlight toolbar: adjusted p < α (default on), raw, strict, off; α presets + custom; highlighted-only; N of M / K counters; warnings separate; non-colour ★/⚠ markers; persisted | Implemented+Tested | `discovery.py` toolbar; `engine.highlight_mask/warning_mask` · tests: alpha boundary, raw vs adjusted, warnings independent, GUI test with counts and persistence |
| M §4.2 | Strict criterion only with same-estimand CI; correct null (0 / 1); joint tests never qualify; missing / non-finite never | Implemented+Tested | `engine.ci_excludes_null`, typed `ci_applicable/ci_scale/ci_null/ci_level/ci_method/ci_effect_id/p_test_id` · `test_strict_mode_needs_same_estimand_ci_excluding_null` (all M §15 A cases) |
| M §4.2 | p/CI method mismatch noted; partial searches provisional | Implemented+Tested | `ci_note` (LRT p vs Wald CI); `search_scope` column · `test_partial_search_scope_is_labelled_provisional` |
| M §4.3 | Columns: tested effect, scale, CI applicability, test type, adjusted-p method, significance status, diagnostics status, search scope; counts incl. joint tests without CI | Implemented+Tested | `_TEST_COLS`; summary "Hypotheses by type" |
| M §5 | Fit identity covering data/metadata/treatments/filters, coding (event, reference, levels), structure, sample, pool, fitting settings, R/package versions; presentation changes never refit; adjusted p recomputed | Implemented+Tested | `engine.fit_identity`, `cache_from`, `_engine_id` · `test_B_fit_identity_changes_with_every_fit_defining_setting`, `test_B_event_change_never_reuses_fits_but_alpha_change_does`, longitudinal event/structure/time-type identity tests. Found and fixed: identity missed the event for longitudinal logistic engines. |
| M §5 | Provenance: run_id, candidate_id, model_id, signatures, timestamp, engine, cache reason; resume kept; legacy caches invalidated safely | Implemented+Tested | row fields; legacy runs never match the v1.2 engine identity · `test_L_*`, `test_project_round_trip_keeps_typed_tests` |
| M §6 | Treatment manager: outcome always shown below the family with label / levels / storage / status; every predictor listed with status and reason; search, check/clear visible, check all eligible, clear all, counts; hidden rows keep state; all-eligible preconfiguration applied only in the model; acknowledgements invalidated when coding changes | Implemented+Tested | `gui/widgets/treatment_panel.py`, `measurement.treatment_status/coding_fingerprint` · `test_treatment_manager_shows_every_model_variable_hrs` (R_SHLT, R_LTACTX, RAGENDER, R_BMI, HHID, ADVTR_MATCHED), `test_discovery_outcome_row_sits_below_family` |
| M §6.2 | Propagation to analyses, R code, results, exports; source metadata untouched | Implemented+Tested | dataset-view design (v1.1 E1); spec / view assertions in the HRS test; v1.1 codegen tests |
| M §7 / X §3–6 | Main-effects-only, focal, focal + covariates, all combinations, full factorial; hierarchy; dedup; six distinct predictors; exact counts; resume without duplicates | Implemented+Tested | `discovery/space.py`, `discovery/modelspec.py` · X Tests A–H, E (32,768), M Test D, count/key/resume matrix (8 settings) |
| M §7.2 / X §8 | Declared hypothesis per model; distinct test types; families by type | Implemented+Tested | `candidate_tests`; `tests_frame` · X Tests I/J against statsmodels (joint F, 1-df F = t², coefficient CI, whole-model F, main-term p, BH per family) |
| M §8.1 | Study structure selector; any ID column; explicit time; validation (unique IDs, duplicates, numeric time); role link; rows / participants / waves / availability by wave | Implemented+Tested | `_build_study_box`, `space._check_longitudinal`, preflight · `test_E_validation_*`, `test_longitudinal_setup_and_preflight_hrs` (1,000 participants, 16 waves, R_LTACTX unobserved in 6 waves) |
| M §8.2 | Workflow A (fixed-effect discovery) and B (structure comparison), separate rankings | Implemented+Tested | `workflow` setting; `structure_tests` · `test_F_workflow_B_compares_structures_with_REML` |
| M §8.3 | lme4 RI / RS / uncorrelated RS; nlme RI/RS, AR(1), CAR(1); mmrm US/CS/AR(1); ML for fixed-effect LRTs; REML only for B | Implemented+Tested | `rcode.sx_fit_lmm` · `test_F_gaussian_structures_match_direct_R_fits` (5 structures vs direct R fits), lme4 vs nlme agreement |
| M §8.3 | Satterthwaite / KR as a discovery inference option | Deferred | Discovery uses ML LRTs; lmerTest S/KR remain in the Mixed workspace |
| M §8.4 | glmer binary (event coding, OR, CIs, LRT, diagnostics); clmm ordinal (LRT); GEE robust Wald, population-average labels, QIC only within GEE | Implemented+Tested | `test_G_binary_glmm_*`, `test_H_ordinal_clmm_*`, `test_H_gee_uses_robust_wald_*` (vs direct R) |
| M §8.4 | Repeated multinomial outcomes | Implemented (refused with reason) | `longitudinal_family` · `test_multinomial_repeated_outcome_is_refused` |
| M §8.5 | Time × each predictor, all two-way, selected three-way, main only, interactions + covariates; budgets; no structure × subset explosion | Implemented+Tested | strategy `time` (+ subsets, order 3); one structure per run (A) |
| M §8.6 | Preflight / missingness; grouped CV for longitudinal searches | Partial | Preflight in Discovery; stability / procedure CV refuse longitudinal searches (grouped holdout and grouped CV exist in Validation for single models) |
| M §8.7 | Discovery → plots with exact model | Partial | Longitudinal: refit in Mixed workspace with exact terms, structure, ML, coding, labelled "Refit" (`spec_from_discovery`) · `test_longitudinal_candidate_refits_in_mixed_workspace`. mmrm CS/AR(1) not offered there. Single-level: Teaching Studio with exact terms (`from_candidate`) · drill-down test |
| M §9 | n_groups vs df; LRT df; estimate/SE/CI with scale; OR + exponentiated CI; joint-test statement; CI method stated | Implemented+Tested | `_test_record`; legacy mapping · `test_legacy_mixed_rows_report_groups_not_df`, mixed discovery test, OR columns |
| M §9 | Conditional slopes / contrasts for multi-df interactions | Partial | Available in Teaching Studio (single-level) and Mixed workspace; not in the discovery table |
| M §10 | Manual limits: regressions 6 (10/15/custom), Mixed 6, factorial ANOVA 3 (4–5 with cell preflight), ANCOVA 2 + 4; preflight counts; one limit for GUI and backend | Implemented+Tested | `analysis/complexity.py`, `MixedSpec.max_fixed` · `test_v12_manual_limits.py` |
| M §11 | Preflight (treatment, pool, caps, candidates, sample, ID/time, family, test family, workload, incomplete projection); filters (hypothesis/model type, diagnostics, CI, significance via highlight, text); formula + hypothesis + reference model; compare with main-effects model; 3–6 sentence interpretations; exports keep raw/adjusted p | Implemented+Tested | estimate panel, filters, `reference` column, `_compare_main`, `interpret_row`, Excel hypotheses sheet · GUI + export tests |
| M §11 | Highlight/filter status in PDFs | Partial | Searches are exported to Excel/JSON (highlight settings included as presentation metadata); PDF reports contain analyses, not searches |
| M §12 | v1.1 features preserved | Implemented+Tested | full suite (all v1.0/v1.1 tests; four v1.1 assertions updated for deliberate v1.2 changes: df2 → n_groups, hypothesis table column, default strategy, ordinal repeated outcomes now supported) |
| M §13 | Optional research extras | Partial / Deferred | Done: study-design selector, coding audit in the preflight, chance-findings counts per family, performance. Deferred: missing-data sensitivity screen, longitudinal selection stability, marginaleffects prediction surfaces, simulated multiple-testing lesson, one-click reproducibility bundle |
| M §14 | v1.1.1 milestone build | Deferred | Milestone-1 content is commit `3ef839a`; not built or tagged separately (one release build at 1.2.0) |
| M §15 K | Clean Windows 10/11 offline install | Blocked/User | Requires a clean machine |
| M §16 | Statistical safeguards | Implemented | exploratory warnings, families, provisional labels, no cross-family IC ranking, no silent family/coding/structure changes |
| X §7 | Eligibility screening (rank, df, sparse cells per interaction term, collinearity, events, separation, participants for slopes) | Implemented+Tested | `sx_discover_one` · invalid-reason tests |
| X §9 | Model comparison incl. CV metrics | Partial | AIC/BIC/R²/adj R²/nested tests in Discovery; CV metrics in Validation & Uncertainty |
| X §11 | Controls (max/min predictors, order, strategy, model/time budgets, workers, min observations, max parameters, cancel, resume, progress, history) | Implemented+Tested | Discovery settings |
| X §13 | Grouping variables not counted; random-effect parameters tracked separately | Implemented+Tested | `modelspec.random_parameters`, `n_re_params` · `test_K_*` |
| X §15 E | Execution of all 32,768 six-predictor models | Partial | Enumeration tested without fitting (as the spec allows); execution covered by smaller controlled searches and the full self-test |
| X §15 L | Interrupted searches labelled incomplete; resume without duplicates | Implemented+Tested | `test_L_interrupted_search_resumes_without_duplicates` |

## Bugs found during v1.2 work (all fixed, with tests)
1. v1.1: numeric table columns did not sort (numpy scalars / NaN) — the reported bug.
2. v1.1: discovery cache ignored the binary event / multinomial reference.
3. v1.1: participant count stored in `df2`.
4. v1.1: treatment panel silently dropped the outcome and ineligible predictors.
5. v1.2 (during development): reduced-model cache reused across resampled rows (caught by the stability test);
   fit identity missed the event for GLMM/GEE; mmrm AIC used a different parameter convention; workflow B was
   subject to strategy rules.
