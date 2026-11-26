# AMD validation: methods and estimands (v1.3.3)

Implementation: `src/statexplore/discovery/validate.py` (engine), `src/statexplore/gui/widgets/validation_panel.py`
(Automated Model Discovery ▸ **Validation** tab). Requirement: 02_Discovery_and_Validation.md §2–3, register V133-VAL-01.

## What it answers

Validation estimates how well a model, or the whole selection procedure, predicts data that played no part in fitting
or choosing it, **for a declared prediction target**. It is separate from the discovery p-values: it is not a
confirmatory test, it does not remove selection bias from the discovery CIs, and a bootstrap selection frequency is not
the probability that a hypothesis is true.

## Configuration

| Setting | Values | Default |
|---|---|---|
| Prediction target | New independent observations (rows; only without repeated measures) · New participants · New clusters · New participants in existing clusters | new clusters if the search has a cluster, else new participants |
| Procedure | Whole selection pipeline (the search's candidates refitted inside each training part, then the prespecified rule picks one) · Fixed candidate (conditional / post-selection assessment) | pipeline |
| Selection rule (pipeline) | smallest training p-value of each candidate's primary test (the discovery ranking) · lowest training AIC (required for workflow B) | p |
| Modes | holdout · grouped K-fold CV (repeats) · bootstrap stability (any combination) | holdout + CV |
| Numbers | holdout 20 %, K = 5, repeats = 1, B = 200, seed = 133 | as listed (no claim of statistical adequacy) |
| Budget | fits = candidates × (splits + B); refused above 20,000 before anything runs | — |

Unavailable targets are disabled with the reason (e.g. rows of a participant are never split at random; no cluster →
no cluster targets). Families without a correct out-of-sample prediction here (GLMM, CLMM, multinomial / ordinal mixed,
marginal mmrm) are refused with the reason; the inverse link of the fixed part is **not** used as a substitute for a
marginal prediction.

## Splits (made before any fit)

- Units: rows; participants (composite `(cluster, ID)` keys when a cluster exists); clusters; or participants within
  clusters. Rows without an ID / cluster have no unit (they are not in any model).
- Generator: numpy `PCG64([seed, stream])`; stream 0 = holdout, 1 + r = CV repeat r, 1000 + b = bootstrap draw b. Same
  seed → same assignment; a different seed gives a different split. Splits never depend on fitted performance.
- Holdout: `ceil(fraction × units)` units to the test part. Within-cluster target: per cluster
  `min(n − 1, ceil(fraction × n))` participants, so every evaluated cluster keeps training participants; clusters with
  fewer than 2 participants are training-only and listed as ineligible.
- K-fold: units permuted, assigned round-robin to folds 1..K (balanced); within-cluster target per cluster.
- Manifest: every analysed row (stable source-row number = R rowname) × split → fold (0 = training only), with its unit
  keys. Disjointness is asserted in Python on the manifest **and** again in R on the fitted rows: no row or participant
  in both parts; for new clusters no shared cluster; for new participants in existing clusters every test cluster is in
  training.

## Fitting and prediction

Each training part is passed to the search's own R helpers (same formulas, structure, estimator, grouping and helper
code as the search). Test outcomes are never used: predictions use test covariates/time only.

| Target | lme4 | nlme | GEE / lm / glm |
|---|---|---|---|
| New participants / new clusters | `predict(re.form = NA)` (fixed effects; new random effects = 0) | `predict(level = 0)` | `predict(type = "response")` (marginal mean) |
| New participants in existing clusters | `predict(re.form = ~(1 | cluster))`: cluster BLUP from training participants, new participant effect 0 | `predict(level = 1)` | — |

Test rows incomplete on the selected model's variables, or with a category level never seen in training, are excluded
and **counted** per fold (`excluded_incomplete`, `excluded_unseen_levels`, `excluded_unpredicted`). Transformations that
use whole-sample statistics (z-score / centred derived variables) are refused because they would leak test data.

## Metrics

Gaussian: RMSE, MAE, out-of-sample R² = 1 − SSE / Σ(y − training-part mean)² (may be negative; NA when the denominator
is 0). Binary (single-level): log loss, Brier, AUC (NA with one class). Per fold, then pooled per mode:
observation-weighted (all held-out rows) and group-weighted (each participant, or each cluster for the new-cluster
target, counts once). No t-based interval across correlated folds is reported.

## Bootstrap stability

New clusters: clusters drawn with replacement, complete nested histories copied, every drawn cluster and participant
gets new IDs (`dc<i>`, `d<j>`). New participants in existing clusters: clusters kept, participants drawn with
replacement within each cluster, new participant IDs. New participants: participants drawn with replacement. Original IDs
are kept in the draw map. Each draw reruns the selection pipeline (or refits the fixed candidate) and records the
selected candidate, the primary tested coefficient, its sign and the diagnostics. Reported: B requested / attempted /
successful, selection frequency over attempted **and** over successful draws, sign stability and percentiles among draws
that selected the candidate (an unselected coefficient is never replaced by 0), flagged fits.

## Status, failures, persistence, export

Statuses: not run · running · complete · partial (a fold or draw failed, or fewer than planned) · cancelled (completed
chunks kept) · failed. Failed folds/draws are listed with their R error and never scored as 0. Every run is an immutable
record stored with its search (`DiscoveryRun.validation_runs`, saved in the project). Export writes
`validation_manifest.json`, `split_manifest.csv`, `bootstrap_draws.csv`, `predictions.csv`, `folds.csv`,
`performance.csv`, `stability.csv`, `bootstrap_results.csv` and `validation_code.R` (the exact R code of each chunk).

## Evidence

`tests/test_v133.py`: deterministic and disjoint splits, unique draw IDs, new cluster IDs per drawn school
(`test_validation_splits_deterministic_disjoint_and_draw_ids_unique`); held-out predictions for both cluster targets equal
an independent hand-written lme4 fit on the same manifest (max |Δ| < 1e-6), and +10,000 on the test outcomes leaves
the training fit and predictions unchanged while the metrics change
(`test_validation_predictions_match_independent_oracle_and_do_not_leak`); pipeline bootstrap selects the planted
TIME × GROUP with stable sign, refused targets, cancellation keeps completed folds, persistence and export
(`test_validation_pipeline_bootstrap_failures_and_persistence`). `tests/test_v133_gui.py::test_validation_panel_run_persist_reopen`
drives the real panel (automatic run after the search, second immutable run, preview, save / reopen).

## Limits

No external validation set, no repeated-history (future-occasion) forecasting target in this panel (the older
procedure-CV "temporal" target on the Validation page remains), no nonlinear-mixed marginal predictions, no
inner-loop tuning (the selection rules have no tuning parameter).
