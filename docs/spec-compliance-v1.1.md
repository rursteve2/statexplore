# StatExplore v1.1 — specification compliance

Source: *StatExplore v1.1 Existing-Codebase Upgrade Specification* (24 sections).
Baseline: repository commit `eb80db6` (= GitHub release v1.0.0). No `statexplore-1.0.0.zip` archive
was supplied, so the repository is the baseline; the supplied `hrs_least_missing_1000 (1).sav` is
byte-identical to the v1.0 test fixture. Baseline tests: **152/152 passed** before any change.

Status: ✅ implemented and tested · ◐ implemented with a documented limitation · ⏳ user step.

| § | Requirement | Status | Evidence (code · tests) | Notes / deviations |
|---|---|---|---|---|
| 2 | Inspect codebase, run baseline tests | ✅ | tasks/todo.md "Baseline" | 152/152 |
| 3.1 | Source / project / analysis levels kept separate | ✅ | `core/variables.py` (`source_measure`, `measure`, `measure_status`), `analysis/measurement.py` (`VariableTreatment`, `apply_treatments` view) · test_measurement A | Treatments never touch the dataset: an analysis sees a view with the effective measure. |
| 3.2 | Metadata: source, approved, suggested level, reasons, codes, labels, order, direction, approval | ✅ | `VariableMeta` new fields; `Dataset.levels()` honours `category_order` · test_measurement | `Measure.BINARY` added. |
| 3 | Fix at data-preparation / validation level, not just GUI | ✅ | `rbridge/codegen.py` emits numeric codes for continuous treatment; validators read the view | |
| 4 | Flag possibly misclassified variables (rules + limitations, no auto-reclassification) | ✅ | `measurement.suggest/annotate/needs_review` · test_A_suggestion_rules_on_hrs | Label-pattern scales (frequency, health, agreement, degree, education); special codes reported (R_COMP 7, R_DRINKN 99); identifiers kept nominal; 5 categories alone never imply ordinal. |
| 4.2 | Measurement Review panel (accept / keep / other level / order / labels / codes; explicit approval; saved) | ✅ | `gui/pages/measurement_review.py` · test_gui_v11::test_measurement_review_accept_and_keep | Every change asks for confirmation; logged + undoable. |
| 5.1 | R_LTACTX: nominal kept, ordinal suggested, continuous with codes 1–5, warning, direction note | ✅ | test_B_* (OLS on raw codes == R to 1e-8) | Direction: "higher codes = less frequent". |
| 5.2 | R_SHLT: ordinal outcome/predictor, continuous with acknowledgement, direction kept | ✅ | test_C_*, test_spearman_uses_category_order | |
| 5.3 | Overrides in model config, result metadata, R code, reports, projects | ✅ | `measurement.record` (inputs), load-script comment block, exporters (Excel/Word/PDF), ModelSpec/MixedSpec/SearchSettings `.treatments` · test_exports_include_v11_result_kinds | |
| 6 | "Treat as continuous for this analysis" checkbox (Analysis Builder, Discovery, Mixed, EDA/correlation); nominal needs extra confirmation | ✅ | `gui/widgets/treatment_panel.py` · GUI tests | |
| 6.2 | Warning text; acknowledgement recorded | ✅ | `CONTINUOUS_WARNING`; `acknowledged` in result inputs | |
| 6.3 | Codegen: treatment maps, ordered factors, events, no `as.numeric(factor)`, cache invalidation | ✅ | read/prepare split + metadata-signature cache in `analysis/session.py` · test_D_* | Ordinal *predictors* keep dummy coding (as in v1.0) so existing results do not change. |
| 7 | Reusable variable selector (search, check/clear visible, check/clear all, count incl. hidden, level filter, label search, tooltips) | ✅ | `gui/widgets/variable_selector.py` · test_E_* | Used in Analysis Builder, EDA, Discovery candidates **and required covariates**, Mixed predictors, Survey, replicate weights. |
| 8.1–8.2 | psych::describe kept; frequencies (counts, %, valid %, missing, mode, categories; ordinal order) without nominal means | ✅ | `analysis/eda.frequencies` · test_F_frequencies_match_pandas, test_F_ordered_categories_stay_ordered | Base R (janitor not needed). |
| 8.3 | Frequency / % / horizontal / clustered / stacked / 100 % / mosaic / ordinal distribution plots, grouped | ✅ | `eda._categorical_plot` (vcd mosaic) · test_F_plots_match_frequency_tables, test_F_grouped_plots | Each plot returns the counts it drew. |
| 8.4 | Cross-tabs with row/col/total %, observed, expected, chi-square / exact advice | ✅ | `eda.crosstab` · test_F_crosstab_matches_scipy | |
| 8.5 | Spearman for ordinal without reclassification; Pearson needs treatment; no Pearson on nominal codes | ✅ | `builders._cor_check`, `_rank_ordinal` · test_spearman_* | Spearman ranks by category order (explicit label map). |
| 9 | Categorical Association: chi-square, Fisher, OR, RR, RD, phi, Cramér's V | ✅ | `builders._assoc_table`, `_assoc_2x2` · test_association | |
| 9.1 | 2×2: event / reference / exposed / comparison chosen; table shown before statistics | ✅ | level options + live preview in the form · test_analysis_builder_2x2_preview_and_run | |
| 9.2–9.4 | OR (Woolf + conditional exact), RR (Katz), RD (Newcombe), labelled CIs, % points | ✅ | vs statsmodels Table2x2 / confint_proportions_2indep, scipy odds_ratio | |
| 9.5 | Zero cells, sparse tables, missing, reference changes, no silent continuity correction; unadjusted vs adjusted distinguished | ✅ | test_G_zero_cells, test_G_reversing_event_and_groups, test_G_chisq_options_are_reported | Haldane–Anscombe only when chosen, reported. |
| 10.1 | Explicit, confirmed outcome model family with suggestion | ✅ | `discovery/space.suggest_family`, `SearchSettings.family`; GUI requires confirmation · test_H_family_validation_and_suggestions | API keeps v1.0's automatic choice when `family=""` (old projects). |
| 10.2 | lm / glm / ordinal::clm / nnet::multinom | ✅ | `discovery/rcode.py` | |
| 10.3 | Family-specific focal tests, joint test of multi-level interactions | ✅ | F (lm), LR (glm, clm, multinom joint over contrasts) · LR == statsmodels OrderedModel / MNLogit | |
| 10.4 | Results: family, treatment, coding, event/reference, ordering, convergence | ✅ | `DiscoveryRun.coding`, flags | |
| 10.5 | Continuous overrides used consistently (candidates, prep, tests, plots, scripts) | ✅ | search view in engine, detail, validation · test_H_continuous_override_in_hrs_search | |
| 10.6 | 40 variables / 6 predictors kept; stricter parameter counts for ordinal / multinomial; incomplete runs labelled | ✅ | `rcode` parameter count includes thresholds / (K−1)×p | |
| 11.1 | Mixed family selector; treat-as-continuous checkbox | ✅ | `MixedSpec.family/estimator`; Mixed page | Family is never changed silently; a suggestion is shown. |
| 11.2 | Gaussian lme4 / nlme / mmrm features retained | ✅ | v1.0 tests unchanged and passing | |
| 11.3 | lmerTest inference: Satterthwaite (default), Kenward–Roger, LRT; method shown; not applied to GLMMs | ✅ | `_lme4_inference` · test_I_* (balanced exact df; KR == pbkrtest::KRmodcomp; LRT == statsmodels ML) | KR requires REML (enforced). |
| 11.4 | Binary GLMM (glmer): ORs, CIs, tests, variance, convergence, singularity, conditional vs population-average probabilities | ✅ | `_build_generalized` · test_J_* (== independent numpy Gauss-Hermite MLE, R nAGQ = 25, to 2e-4) | Population-average by simulation (4,000 seeded draws). |
| 11.5 | Ordinal GLMM (clmm) with order, direction, PO note | ✅ | test_K_clmm_matches_independent_ghq, test_K_ordinal_order_reversal_flips_sign | Joint Wald term tests (LR refits were too slow on HRS). |
| 11.6 | Multinomial GLMM (mclogit::mblogit, PQL) with disclosure | ◐ | test_K_multinomial_mixed_recovers_parameters | Approximate method, labelled; recovery-by-simulation validation (no exact reference exists); random intercept only. |
| 11.7 | GEE (geepack) with working correlations; estimand distinction | ✅ | test_gee_matches_statsmodels | |
| 12 | Longitudinal preflight: records, participants, time points, availability, 1 vs 2+ obs, duplicates, missing waves, analysis range, slope / spacing checks, no zero-filling | ✅ | `analysis/longitudinal_preflight.py`; availability also computed in R inside every longitudinal script · test_longitudinal_preflight | HRS R_LTACTX: waves 1–6 unobserved, range 7–16. |
| 13 | Survey design (weight, PSU, strata, FPC, replicate weights), no invented variables, explanation of what can be modelled | ✅ | `analysis/survey.py`, Survey page · test_M_hrs_has_no_design_variables | HRS has no design variables (R_WEIGHT is body weight). srvyr not needed. |
| 13.2 | Weighted means / proportions / tables, Rao–Scott cross-tabs, svyglm linear / logistic | ✅ | test_M_* (== closed-form stratified Taylor linearization, 1e-10) | |
| 13.3 | Safeguards: raw n vs weighted, design df, invalid weights, lonely PSU policy, longitudinal weights | ✅ | `check_design` · test_M_invalid_weights_and_design_checks | |
| 14 | R Console: editor + highlighting, run / stop / clear / open / save / copy, output / warnings / errors / plots | ✅ | `gui/pages/r_console.py`, `r/console_worker.R`, `rbridge/console.py` · test_N_* | |
| 14.2 | `my_data` copy; explicit import with diff, confirmation, recoverable snapshot | ✅ | `core/console_import.py` · test_N_internal_sessions_unaffected_and_explicit_import | Import creates a new project (history carried over) from the data; the previous project is saved as a snapshot. |
| 14.3–14.4 | Clean session per run; optional persistent session; restart / clear / packages / objects; isolation from analysis workers; cancellation | ✅ | test_N_clean_vs_persistent_and_project_isolation, test_N_cancellation | |
| 14.5 | Security warning; no shell; explicit Run only; offline | ✅ | banner on the page; JSON over stdin | Not described as a sandbox. |
| 15 | Package Manager: inventory, search, local install (.zip / .tar.gz), dependencies, logs, removal, use in scripts | ✅ | `rbridge/package_manager.py`, dialog · test_O_* | |
| 15.2–15.5 | Separate user library, core first, no CRAN, compatibility checks, warning, manifest | ✅ | `%LOCALAPPDATA%\StatExplore\R\4.6\user-library`; `statexplore-user-packages.json` | Core-package names refused (no shadowing). |
| 16 | Package updates: ordinal, nnet, mclogit, geepack, survey, vcd, pbkrtest added | ✅ | `r/install_packages.R`, `locate.V11_PACKAGES`, self-test | Not added (not needed by implemented features): janitor, srvyr, gtsummary, marginaleffects, interactions, sandwich, clubSandwich, mice, lavaan, mediation, glmmTMB. |
| 17.1 | Structured output tables; copy into documents | ✅ | **Copy tables** (tab-separated) in Analysis Builder, Mixed, Survey | |
| 17.2 | Brief interpretation | ✅ | `analysis/interpret.py` · test_brief_interpretation_and_tsv | |
| 17.3 | Likert sensitivity (Gaussian vs ordinal) | ✅ | `likert_sensitivity` · test_likert_sensitivity_matches_references | Scales declared non-comparable; AIC flagged not comparable. |
| 17.4 | Diagnostics | ✅ | existing + convergence, singularity, separation, sparsity, PO check, residual autocorrelation | |
| 17.5 | Teaching Example Finder from real results only | ✅ | `analysis/teaching_finder.py` · test_teaching_finder_uses_real_results | |
| 18 | Persistence + migration with backup; v1.0 projects open | ✅ | `PROJECT_VERSION = 2`; `*.v1-backup.sxp` on first overwrite · test_backward_compat (genuine v1.0.0 fixture) | |
| 19 | Phases 1–9 | ✅ | tasks/todo.md reviews | |
| 20 | Acceptance tests A–P | ✅ | tests/test_measurement, test_eda_categorical, test_association, test_discovery_families, test_mixed_families, test_longitudinal_preflight, test_survey, test_console_packages, test_research_tools, test_backward_compat, test_gui_v11 | |
| 21 | Release quality (tests, HRS, GUI, R scripts, survey refs, old projects, offline installer, packages, docs) | ◐ | full suite, `--selftest --full`, release audit | Installer on a clean machine: ⏳ user step. |
| 22 | Per-phase reporting | ✅ | tasks/todo.md | |
| 23 | Final user acceptance workflow | ✅ | docs/release-audit-v1.1.0.md | |
