# StatExplore 1.3.2 — release notes

Statistical-integrity release plus **Gaussian clustered Automated Model Discovery** (participants nested in one
higher-level cluster, in both Discovery workflows). Developer package: `StatExplore_v1.3.2_Developer_Package`
(master prompt 01, register 02, specifications 03/04, protocol 05). Baseline: v1.3.1 (`9153b96`).

## New: clustered Gaussian Discovery (04, CD-01…CD-12)

- **Study structure → Higher-level cluster** (e.g. school, hospital) next to participant ID and time. Default *none*
  keeps the earlier behaviour. The cluster is a grouping level (random intercept), never a predictor or an
  interaction term.
- **Participant IDs**: *unique across clusters* (checked: an ID in two clusters is refused as a crossed / moving
  design) or *numbered within each cluster* (the participant is the (cluster, ID) pair; locally reused IDs never
  merge). Rows with a missing cluster are excluded and counted.
- **Workflow A** (fixed effects): every candidate and every reduced model keeps the locked hierarchy, random part and
  residual structure; ML fits.
- **Workflow B** (covariance structures): the full 3 × 3 table — random intercept / uncorrelated slope (DIAG) /
  correlated slope (RS) × independent / AR(1) / CAR(1) — with or without the cluster (18 combinations), REML with the
  fixed part locked. B no longer needs a candidate predictor (time-only fixed part).
- **Exploratory cluster comparison** (B, off by default): each structure is also fitted without the cluster variance
  on the same rows and composite participants; the cluster variance test uses the boundary reference. The cluster is
  never removed automatically.
- **Validation for longitudinal searches** (Validation ▸ Full-procedure CV): prediction target *new organizations*
  (clusters held out), *new participants* or *future visits* (rolling origin); the whole search runs inside each
  training fold; too few groups gives "unavailable", never a row split.
- Preflight shows outcome availability **and** complete-model rows with their own time range, clusters, participants
  per cluster, excluded rows and IDs seen in several clusters.

## Corrected defects (by register ID)

| ID | Change |
|---|---|
| STAT-22 (P0), STAT-39, STAT-40 | **Model identity is decided by fingerprints, not fit statistics.** Every fit records SHA-256 fingerprints of the analysed rows, outcome values / coding, fixed design and grouping (schema 2). A refit is *verified* only if they match; *mismatch* names what differs; older results without fingerprints are *unverifiable*; failed refits say so. Equal n / logLik / AIC never verify (Tests 35A/35C reproduced and fixed). Searches store a snapshot of their data view: after a filter or coding change, a historical candidate is refitted on the rebuilt original view or run as an explicitly labelled **new analysis**. Duplicate value labels use the exact (de-duplicated) factor level in every generated comparison. |
| STAT-21 | One covariance-structure builder for the Mixed workspace, its comparisons, Discovery and exports: clustered nlme random slopes are `list(cluster = pdDiag(~1), id = pdDiag/pdSymm(~1 + time))` with correlation grouping `~ t | cluster/id`. All clustered RI/DIAG/RS × AR(1)/CAR(1) models now fit and match independent R references (e.g. three-level RS + AR(1): k = 9, logLik −1652.158826). |
| STAT-17, STAT-23, STAT-41 | One likelihood-ratio service (Python + R twin): the reference distribution follows the hypothesis (variance at the boundary → 50:50 mixture; covariance / signed AR(1) φ → χ²; CAR(1) φ = 0 → boundary mixture; AR(1) vs CAR(1) and simultaneous changes → no LR). Negligible negative LR (> −1e‑6) → 0 with the raw value kept; material negative, non-finite LR or df ≤ 0 → no p-value, reason given. DIAG→RS is withheld when a variance is at the boundary. |
| STAT-18, STAT-32 | Diagnostics per fit: optimizer and every attempt (nlminb → optim only on optimizer errors, never on structural errors), optimizer / lme4 check codes and messages (identical text once), lme4 singularity, nlme near-boundary covariance per grouping level (scaled eigenvalue or ratio < 1e‑6, \|r\| > 0.999), apVar status reported separately. Status: clean / warned / near-boundary / singular. Tests involving a flagged fit are *provisional* (support tier C, never clean). |
| STAT-19, STAT-25, STAT-27, STAT-33, STAT-34 | Family-aware comparisons: GLMM/CLMM use ML (Laplace) likelihoods, never "REML"; GEE has no likelihood test or AIC — mean structures by robust Wald test, working correlations by QIC / QICu / CIC; random-effect comparison is refused for GEE and all comparisons for PQL multinomial models, with reasons. All-failed or single-success comparisons are reported as failed / incomplete (no Inf, no winner). GEE refits are no longer described as ML. |
| STAT-20 | Preflight reports the complete-model rows and their time range separately from outcome availability. |
| STAT-28, STAT-29, STAT-30, STAT-35 | Missing-data text distinguishes likelihood-based models (ignorable / MAR, an assumption) from unweighted GEE (MCAR or covariate-dependent missingness). GLMM odds ratios: time-invariant predictors compare participants, not within-person change. CLMM: the ordinary clm nominal test is only a marginal auxiliary check. Lag diagnostics say what they are (pooled within-participant pairs, not re-centred). |
| STAT-36 | Exported script + data bundles: relative paths, output folders created, SHA-256 data check, package check, manifest with versions, provenance and identity; the re-run stops unless rows, outcome, design and grouping are identical. |
| STAT-42, STAT-43, STAT-46, UI-57 | nlme F-test p-values recomputed from full precision (log10 kept); nlme covariance table lists every level's variances, covariance and correlation (DIAG covariance marked "constrained to 0"); group counts (participants, clusters observed / modeled, group sizes) in fits, results and exports. |
| STAT-44, STAT-45, STAT-38 | Uncorrelated lme4 slopes no longer fail in post-processing; optional plots can never fail a fit (plot-specific notes instead). R² is a direct variance decomposition over every grouping level (or "unavailable" with the reason) — the misleading fixed + residual fallback is gone (37A RS: .2144 / .8157). |
| STAT-24 | Native BIC sample sizes are labelled; mixed lme4 / nlme REML bases are not ranked natively — a harmonized BIC (n − p) is shown separately. |
| STAT-26 | A category-specific interaction test that spans the global alternative is labelled so. |
| UI-31, 39–43, 46–52, 54–56 | Selected-first grouping in every multi-select list; filter dialogs explain current vs historical data; "Holm and Holm" wording fixed; reliable copy feedback; Highlight column filled in CSV copies; frozen Test ID column stays inside the viewport after maximize / scroll; "Keep current level" vs "Restore source level"; SPSS missing-code definitions vs row counts; form settings saved with the project and restored on open; the latest search is shown on reopen; empty searches are refused with the reason and remedy; effect / interaction wording matches the tested hypothesis; random-effect comparisons include the uncorrelated slope; workflow B needs no candidate predictor. |

## Compatibility and migration

- Projects from 1.0–1.3.1 open unchanged (read-only open; the first save keeps a backup). Their discovery rows have
  no identity fingerprints: refits from them are reported **unverifiable**, never verified. Their searches have no
  data snapshot: after a project change they can only be rerun as new analyses.
- Discovery cache rules changed (fit rules 3: identity, diagnostics, shared structure builder, nlme msMaxIter 200):
  earlier cached fits are refitted, not reused.
- Clustered GEE, mmrm, binary and ordinal mixed models are **not** part of this release: they are refused with a
  reason, never fitted without the cluster.

## Known limitations

- The developer could not run the packaged build in a clean offline Windows VM or on a physical 150 % display;
  these are owner steps (see `docs/v1.3.2/manual_acceptance_v1.3.2.md`).
- The independent reference fit `CD_Irregular` two-level RS + AR(1) fails in R with both optimizers; the
  application reports the same model as failed with its error (no substitute).
- Simulation evidence is deliberately small (`docs/v1.3.2/simulation`, seed 132, 200 replicates per scenario,
  balanced 20 clusters × 8 participants × 6 occasions, through the shipped structure builders): Workflow A GROUP LR
  type I error 6.5 % (Monte Carlo SE 1.7 %), TIME 95 % CI coverage 99.0 % (conservative), cluster-variance boundary
  test type I error 3.0 % (SE 1.2 %; conservative, as expected with few clusters), LR = 0 in 57 % of null replicates,
  100 % convergence. Unbalanced designs, few clusters and boundary alternatives were not simulated.
- R² with residual autocorrelation uses the marginal residual variance σ²; this is a documented convention, not a
  theoretical validation.
