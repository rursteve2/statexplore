# StatExplore 1.3.3 — release notes

Reliability release: one source of truth for a transferred model, comparisons that keep the model, composite
participant identity everywhere, **AMD validation** (holdout, grouped cross-validation, bootstrap stability), and
portable reproducibility bundles for every analysis view. Developer package: `StatExplore_v1.3.3` (MASTER_PROMPT,
01 model integrity, 02 discovery and validation, 03 register, 04 regression protocol, 05 acceptance). Baseline: v1.3.2
(`ff7b4f3`).

## P0 fixes

| ID | Change |
|---|---|
| STAT-47, V133-STATE-01…04 | **Full Analysis keeps the exact model.** A transferred model becomes an editable *draft* with an immutable *origin*. The form shows every fixed term in a new **Fixed terms** list (presets None / Time × each / All two-way only fill the list; any other set shows as **Custom** with each term; **Undo** restores the previous terms). Fit uses the draft, never widget defaults, so an unchanged Fit, a plot-only change (trajectories) and a repeated Fit reproduce the transferred model and stay **verified**. Changing only ML → REML changes only the estimation (independent REML reference: k 10, AIC 3324.0788, BIC 3372.7064, BIC n 956) and the result is labelled **modified from the origin** with the field-level difference; lineage is kept, identity is never claimed. **Return to origin** and **Revert to last fit**; a "Model origin and state" box shows unchanged / modified and unfitted edits. |
| (01 §1) | **Fitted = requested, checked on every fit.** The fitted R object is read back (class, fixed term labels, grouping factors, random-effect blocks, residual correlation, estimation, GEE working correlation) and compared with the request; any difference fails the result closed and shows both. |
| V133-STAT-01 | **Time-form comparisons keep the model.** Linear / quadratic / categorical variants keep every fixed term (time terms follow the time form), random effects, residual correlation (AR(1) / CAR(1)), cluster and ID policy; the REML → ML change is stated before and in the result; each row shows the *fitted structure* read back from R. On v1.3.2 the interaction was dropped after a transfer (reproduced); the reported AR(1) → independent lme4 substitution did not reproduce on v1.3.2 code in the tested routes and is now impossible without a fail-closed error. |

## P1 fixes and completed requirements

| ID | Change |
|---|---|
| V133-STAT-02 | Preflight counts participants, duplicates, gaps and cluster sizes on the composite (cluster, ID) key (45A: 160 participants, 0 duplicates; was 8 / 48). |
| V133-STAT-03, V133-UI-04 | Exploratory cluster omission stays opt-in; cluster-free fits keep and label composite participant keys; Full Analysis has an explicit cluster random-intercept switch. |
| V133-VAL-01 | **AMD ▸ Validation tab** (configurable before the search, optional automatic run when it finishes): targets *new independent observations*, *new participants*, *new clusters*, *new participants in existing clusters*; holdout, grouped K-fold (repeats) and bootstrap stability; whole-selection-pipeline or fixed-candidate (labelled post-selection) mode; deterministic split manifests saved with the search; per-fold tables with failures and exclusions; observation- and group-weighted metrics; selection and sign stability with both denominators; export of manifests, predictions and R code. See `docs/v1.3.3/validation_methods.md`. |
| V133-EXP-01…04 | **Save reproducibility bundle…** in View Code of every analysis view (Full Analysis included): relative-path script, data only with consent, manifest (schema 2: app build + commit, R / package versions, data SHA-256, provenance, identity, expected results, tolerance, effective spec), README; the script stops unless data hash, identity and n / logLik / AIC / BIC reproduce. AMD candidate manifests now carry versions, provenance and identity; declared packages are only those the model family uses. |
| V133-CMP-01, V133-DIAG-01, V133-HIST-01, V133-DISC-01 | Verified with new evidence: GLMM / CLMM likelihoods vs independent lme4 / ordinal fits; optimizer fallback (same spec, flagged, never clean), structural errors not retried; material negative LR withheld; filter / 3-category coding reconstruction; independent enumeration oracle and `p.adjust` per family. |

## P2 interface items

V133-UI-01 preflight "time range with observed outcome" (was "outcome range"); UI-02 workflow B no longer shows a
focal-test description; UI-03 a blocked start resets the progress bar; UI-05 an interior absent wave is described as
inside the range with no data; UI-06 teaching interpretations are exported (were blank in CSV); UI-07 **Save CSV…**
beside Copy CSV; UI-08 **log10 p** column from R's log tail; UI-09 TIME counts once with the Time × each strategy (and
cannot also be required).

## Evidence

Tests: 425 (v1.3.2) → **493** (492 pass, 1 screenshot-only skip, 0 fail) on Windows with real Qt and R; developer
runner 97 gates PASS, 0 FAIL; all 36 workflow obligations of the compatibility matrix plus 18 standalone fits equal the
independent R references (`tests/fixtures/v133/reference`, the package's scripts run with R 4.6.2 and reproduced
1994 / 1994). Cancelling a running search now returns in about 2 s (was minutes). Details:
`docs/v1.3.3/evidence_report_v1.3.3.md`, `docs/v1.3.3/disposition_v1.3.3.csv`. Owner steps: offline clean-machine install, physical 150 % DPI (`docs/v1.3.3/manual_acceptance_v1.3.3.md`).

## Compatibility

Projects from 1.0–1.3.2 open unchanged (new fields default; old Mixed form states load as drafts without an origin;
old searches have no validation runs). No project-format migration is needed.
