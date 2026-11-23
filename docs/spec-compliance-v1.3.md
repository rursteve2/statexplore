# StatExplore 1.3.0 — spec compliance, defect traceability and test evidence

Spec: `StatExplore_v1.3.0_Development_Prompt.md`. Baseline: v1.2.0 (`a72ddeb`); 315/315 tests pass on a clean
worktree. Reference data:
- `hrs_least_missing_1000.sav` (byte-identical to the v1.0–1.2 fixture).
- `hrs_least_missing_1000.csv` (the same 16,000 × 168 records; blanks written as a single space).

Both files were used read-only.

The reproduction package lives outside the repository so that no health data is committed:
`E:\StatExplore_v1.3.0_Testing\`
- `reproduction_steps/repro_p0*.py` and `*_prefix.txt`: pre-fix evidence.
- `tests/export_reproducibility_hrs.py` and `tests/cross_module_hrs.py`: dataset-backed end-to-end checks.
- `ui28/`: screenshots.

The repository tests use synthetic fixtures. HRS-backed tests skip when the file is absent.

Status key: **verified** = numerical reproduction or automated test; **implemented** = behaviour changed and
covered by a GUI/unit test; **partial** = see limitations.

## 1. Defect-to-fix traceability

| ID | Root cause (found before editing) | Changed modules | Before → after | Regression tests | Status |
|---|---|---|---|---|---|
| P0-01 A ordinal → binary | `comparison.build` knew only lm/glm. Any other family became `glm(y == level[2])`. | `analysis/comparison.py`, `models.py` (event/reference/descriptor), `teaching.py`, `gui/pages/discovery.py` | **Before:** Fixture A three-way vs main effects was fitted as binomial glm (repro shows `stats::glm(sx_event ~ …)`). **After:** `ordinal::clm` gives LR 4.0900, df 4, p .3940, identical to the search's joint test, and identity is verified. | `test_P0_01_ordinal_comparison_uses_clm_not_glm`, `test_P0_01_ordinal_candidate_through_model_comparison`, `test_P0_01_binary_event_is_preserved`, `test_P0_01_teaching_refuses_non_linear_families` | verified |
| P0-01 B CAR(1) → lme4 RI | `detail.candidate_from_row` dropped `structure`; `spec_from_discovery` then fell back to `s.structure` ("ri") with `method="ML"`. | `discovery/detail.py`, `analysis/mixed.py`, new `analysis/descriptor.py` | **Before:** all three workflow-B HRS candidates opened as lme4 RI ML (repro). **After:** `nlme` REML with the candidate's correlation. The refit matches the search (CAR(1) AIC 20540.106) and says "Model identity verified". | `test_P0_01_workflow_B_structures_survive_transfer`, `test_fixture_D_E_hrs_reference_values` | verified |
| P0-01 C GEE → lme4 | same dropped field | same | **After:** geeglm with independence / exchangeable / AR(1) working correlation; identity verified. | `test_P0_01_gee_keeps_working_correlation_and_reports_qic` | verified |
| P0-01 refit identity | No comparison of a refit with its origin. | `analysis/descriptor.py` (`verify`, `attach_check`), used in Mixed, Full analysis, Model Comparison | Every refit is labelled *verified* (n exact, ΔlogLik ≤ 0.01, ΔAIC ≤ 0.02) or *FAILED* with a warning status. | `test_P0_01_verify_tolerances` and the transfer tests above | verified |
| P0-02 `struct = ""`, `tests = list()` | Same dropped field. Structure comparisons are computed in Python after the run, so the export never contained them. | `discovery/rcode.py` (`comparison_script`, `CANDIDATE_SUMMARY`, `n_lik_params`), `engine.structure_tests` | The export carries `struct = "car1"`, refits the reference structure, recomputes the LR test (χ² or boundary mixture) and prints n / AIC / BIC / logLik / likelihood parameters. | `test_P0_02_structure_export_reproduces_model_and_comparisons` | verified |
| P0-02 portability | Absolute temp path, personal source path, every column and every factor level in the load script. | `rbridge/codegen.py` (one column per line, basename only), `export/exporters.py` (`portable_text`, `write_portable`, pruning, wrapping, data locator), `gui/pages/eda.py` (Save script + data) | HRS CAR(1) export: 4 data columns; no AppData path; `source()` works from another folder. It reproduces n 3901, AIC 20540.106, logLik −10264.053 and 6 parameters from both `.sav` and `.csv`. | `test_P0_02_portable_script_pruning_and_wrapping`, the export test above, `tests/export_reproducibility_hrs.py` | verified |
| P0-03 headline | The headline counted each model's *primary* test, whatever its type, and always called them interactions. | `discovery/engine.py` (`headline`, family counts with BH and Holm), `gui/pages/discovery.py` | **Before:** "4 interaction(s) … 4 after BH" (Fixture A). **After:** "0 significant interactions…; 3 of 9 main-effect term tests…; 14 of 18 whole-model tests…", plus a per-family table (tests / raw / BH / Holm / chance / no CI). | `test_P0_03_*`, `test_fixture_A_hrs_ordinal_families` | verified |
| P0-04 ML/REML, guards | Workflow-B summary said "LR tests on ML fits". Comparison accepted mixed families. | `discovery/engine.py` (`estimation_label`), discovery summary, `comparison.py` | Estimation column (ML/REML/OLS/GEE). Workflow B labelled REML. Comparison refuses different families or outcome coding; LR only for nested models on identical rows. | `test_P1_04_parameter_columns_and_estimation_labels`, `test_P0_01_binary_event_is_preserved` | verified |
| P0-05 CSV whitespace | `pd.read_csv` keeps `" "`, so whole columns became object/text/nominal. 144 HRS columns differed from the `.sav`. | `core/importers.py` (`_type_text_cells`, `import_report`, inference rule), `analysis/measurement.py` (review flag), `gui/widgets/searchable.py` | **After:** all 168 HRS columns have the same storage as the `.sav`, with identical values. R_PMBMI is numeric and continuous (3,901 observed). Normalised cells are counted (767,497 in 135 columns). | `test_P0_05_*` (Fixture H, override, selectors, `.sav` unchanged, HRS CSV = SAV) | verified |
| P0-06 process | — | this document, `tests/test_v13.py`, reproduction package | Each P0 defect has a pre-fix reproduction, a fix and a regression test. | — | verified |
| P1-01 nlme Full Analysis | `spec_from_discovery` forced `inference="lrt"`, which validation rejects for nlme/mmrm. | `analysis/mixed.py`, `gui/pages/mixed.py` | Fixture E CAR(1) ML opens as `nlme` with engine-default df; identity verified. The message explains that KR and the automatic LR refits are implementation-specific. | `test_P1_01_workflow_A_car1_opens_with_compatible_inference` | verified |
| P1-02 time index | Labels said "AR(1) equally spaced occasions". No observed-wave set, no gap-compression warning. | `discovery/space.py` labels, `analysis/descriptor.TIME_INDEX`, `longitudinal_preflight.py`, `mixed.py` (lag note) | AR(1) = observed-occasion index; CAR(1) = actual time differences. The preflight lists observed waves and waves absent inside the range, with a warning. Lag tables say "observed-occasion steps". Duplicate ID/time pairs were already refused. | `test_P1_02_*` | verified |
| P1-03 missingness | The range alone hid Wave 15. MAR wording was unconditional. | `longitudinal_preflight.py`, `mixed.py`, `gui/pages/mixed.py` | Usable participants (2+ / 1 row), intermittent gaps, observed time points, qualified MAR text, model-based labels on trajectory predictions. HRS: 983 usable participants, 973 with 2+ rows, 10 with one; observed waves 8–14 and 16. | `test_P1_02_03_preflight_reports_observed_waves_and_gap_compression`, `test_fixture_D_E_hrs_reference_values` | verified |
| P1-04 parameters | A single "Estimated parameters" column. | engine display columns, discovery tables, header tooltips | Fixed coefficients / Covariance parameters / Likelihood parameters. CAR(1) BMI model = 6. | `test_P1_04_likelihood_parameter_count` | verified |
| P1-05 conditional effects | "Strongest term" claim. | `analysis/interpret.py`, discovery details pane | "Term with the smallest p-value", plus a conditional-effects note when interactions are present. Fixture A three-way coefficient 1.054, OR 2.868, p .3003. | `test_P1_05_*`, `test_fixture_A_hrs_ordinal_families` | verified |
| P1-06 diagnostics / timeouts | A timeout was recorded as a plain failure and the run as Complete. A missing PO table had no explanation. | `rbridge/worker.py` (`RTimeoutError`), `discovery/engine.py`, `analysis/builders.py` | Rows are marked *timed out*; the run status is *Incomplete*. When `nominal_test` is unavailable, the reason is stated. | `test_P1_06_timeouts_are_not_complete` | verified |
| P1-07 GEE QIC | `geepack::QIC` re-evaluates the stored call, whose `f` was out of scope, so it was always NA. | `discovery/rcode.py` | QIC is computed; when it cannot be, the reason is flagged. | `test_P0_01_gee_keeps_working_correlation_and_reports_qic` | verified |
| P1-08 multinomial | Coefficient rows had a bare `category`. | `analysis/builders.py`, `interpret.py` | `comparison_category` and `reference_category` on every row; absent categories reported; logLik and BIC in Model fit. | `test_analyses` multinomial tests + interpretation | implemented |
| P2-01 selectors | Plain combos. Hidden outcomes unexplained. Loaded Mixed model showed 0 predictors. | `gui/widgets/searchable.py`, `variable_selector.py` (pins), discovery / mixed / survey / builder / model editor | Type-to-search everywhere; right-click pin; excluded outcomes listed with reasons; transferred predictors checked. | `test_P2_01_searchable_dropdown`, `test_P0_05_continuous_outcome_selectable_everywhere` | implemented |
| P2-02 panes, tables, find | — | `gui/widgets/find_bar.py`, `eda.CodeDialog`, `pages/mixed.py`, `pages/discovery.py` | Ctrl+F; resizable Preflight with Copy; Copy CSV; movable / hideable / persistent columns; double-click opens details. | `test_P2_02_find_bar` | implemented |
| P2-03 filter scope | — | discovery scope line, unit-aware counter | "Rows are hypothesis tests…" / "Rows are candidate models…"; counters say tests or models. | GUI smoke | implemented |
| P2-04 strict | — | discovery scope line, column tooltip | States how many significant tests lack a same-estimand CI. The four modes are unchanged. | existing v1.2 highlight tests | implemented |
| P2-05 resume | The budget display was ambiguous across continuations. | `DiscoveryRun.executions`, `budget_text` | Each execution's budget and candidate range are shown; not-yet-visited and timed-out candidates are counted. | `test_P2_05_budget_history_is_shown` | implemented |
| P2-06 tables | — | header tooltips, Estimation column | definitions on hover | — | implemented |
| UI-28 | — | discovery details pane and splitters, compact rows, `QSettings` layout, minimum width reduced from 1,483 to 1,146 px | screenshots `ui28/discovery_few_rows.png` (1366×768, one model) and `discovery_many_rows.png` (1920×1000, 1,100 models / 5,170 tests) | GUI suite | implemented |

## 2. Reference results (HRS, R 4.6.2)

| Fixture | Statistic | Spec | StatExplore 1.3.0 |
|---|---|---|---|
| A | n; families (interaction terms / joint / main / whole-model, BH-significant) | 917; 16/0, 5/0, 9/3, 18/14 | 917; 16/0, 5/0, 9/3, 18/14 |
| A | three-way coefficient, OR, LR χ²(1), p | 1.054, 2.868, 1.073, .3003 | 1.0537, 2.868, 1.073, .3003 |
| D | AR(1) logLik / AIC / BIC | −10250.394 / 20512.788 / 20550.401 | −10250.394 / 20512.787 / 20550.401 |
| D | CAR(1) logLik / AIC / BIC, parameters | −10264.053 / 20540.106 / 20577.720, 6 | −10264.053 / 20540.106 / 20577.720, 6 |
| D | independent logLik / AIC / BIC | −10315.171 / 20640.34 / 20671.68 | −10315.171 / 20640.342 / 20671.687 |
| E | gender p; full AIC / BIC | .7863; 20532.121 / 20569.735 | .78629; 20532.121 / 20569.735 |
| Export | CAR(1) script sourced independently (.sav and .csv) | n 3901, AIC 20540.11, logLik −10264.05, k 6 | identical to 1e-7 (R's own REML BIC 20577.7157; see limitations) |

Tolerances: identities (family, engine, structure, n, parameter counts, test identity) must match exactly.
logLik is held to 1e-6 in export tests and 0.01 in identity checks. AIC is held to 0.02. Reference values from the
spec are held to their printed precision.

## 3. Test execution

- Python 3.12 (Windows venv), R 4.6.2 x64, packages from `r/packages.lock.json`. Offline.
- Baseline v1.2.0: 315 passed.
- v1.3.0: **345 passed, 0 failed, 0 skipped** (9 min 19 s), with R and the HRS fixture present. That includes
  `tests/test_v13.py`: 30 tests, synthetic fixtures plus HRS fixtures A / D / E.
- Dataset-backed end-to-end scripts (outside the repo) pass on the final code. Export reproducibility matches for
  all three structures, from `.sav` and `.csv`. Cross-module transfers all report *Model identity verified*.
- Old assertions updated for deliberate changes:
  - `test_v12_gui`: column names "Fixed coefficients" / "Likelihood parameters"; counter "highlighted tests of".
- Deliberate test-visible changes:
  - Numeric CSV columns with ≤ 9 non-negative integer codes, no decimals and no time-like name are read as nominal
    and flagged for review (P0-05 rule 6).
  - Model Comparison's *parameters* column counts likelihood parameters.

## 4. Unsupported or unresolved (intentional limitations vs open items)

Intentional limitations:
- Model Comparison compares single-level models (lm / glm / clm / multinom). Longitudinal candidates are compared
  in Discovery's model-comparison tab (identical rows, labelled ML/REML) or the Mixed workspace's *Compare*
  buttons. They are refused with a reason, never substituted.
- The Teaching Studio draws linear and binary logistic models only; other families are refused with a reason.
- Mixed workspace cannot refit mmrm compound-symmetry / AR(1) candidates, or unstructured covariance with
  continuous fixed time. These are refused; *View candidate code* reproduces them.
- BIC convention: StatExplore uses log(n observations) for every engine. R's `BIC()` for REML `nlme` fits uses
  log(n − p). Exported scripts print R's value, so the BIC may differ in the second decimal; AIC and logLik agree.
- Kenward–Roger df: lme4 only (pbkrtest). nlme reports containment df; mmrm reports Satterthwaite df.
- The CSV time-name exemption is a heuristic: a code column named, for example, `day_type` stays numeric. Measurement
  Review shows every inferred level.

Open items (not done in 1.3.0):
- Resuming does not retry timed-out candidates. They stay marked *timed out*; start a new search with a longer
  limit.
- Boundary tests use the 50:50 χ² mixture where it is valid. No parametric bootstrap is implemented.
- End-to-end GUI automation of every button path is covered by unit, integration and GUI smoke tests. Full
  click-through scripts for Tests 18–25 are left to the user's structured testing.
- Clean-machine install of the 1.3.0 build has not been done yet (user step).
