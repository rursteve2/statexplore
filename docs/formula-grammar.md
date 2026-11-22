# Discovery candidate grammar (v1.2)

Defined before any search runs. Implemented in `discovery/space.py` (enumeration, test plans) and
`discovery/modelspec.py` (canonical terms, counting); verified in `tests/test_discovery.py`,
`tests/test_v12_discovery.py` and `tests/test_v12_longitudinal.py`.

**Inputs:** outcome *Y*; candidate predictors *P* (ordered as selected, ≤ 39); required covariates *R*;
maximum distinct predictors *k* (1–6, default 6; required covariates and a fixed time effect count toward it;
the participant ID never does); minimum distinct predictors (default 1); maximum interaction order (2 or 3).

## Canonical model specification

A model's fixed part is a set of **terms**; a term is a set of variables ({X1} main effect, {X1, X2} two-way).
Specs are **hierarchically closed** (strong hierarchy): every lower-order term of an interaction is present.
Two formulas are the same model exactly when their closed term sets are equal, so `Y ~ X1*X2 + X1 + X3`,
`Y ~ X3 + X2:X1 + X1 + X2` and `Y ~ (X1 + X2)^2 + X3` share one canonical key (`X1 + X2 + X3 + X1:X2`).

Complexity is counted three ways, never from formula text:

| measure | example `Y ~ X1 + X2 + X3 + X1:X2 + X2:X3` (continuous) |
|---|---|
| distinct predictors (the limit applies here) | 3 of 6 |
| model terms (main + interaction) | 5 |
| fixed coefficients (design-matrix columns incl. intercept; R reports the exact count) | 6 |

A k-level categorical variable counts as one predictor but contributes k − 1 columns to each term it enters.
Random-effect variances/covariances and residual covariance parameters are counted separately.

## Strategies

| strategy | candidates |
|---|---|
| **All eligible interaction combinations** (default) | every predictor subset S (min ≤ \|S\| ≤ k − \|R\|) × every hierarchical set of interaction terms on S (two-way; with three-way enabled, any three-way term whose three two-way terms are present). Includes main-effects-only models. For \|S\| = 6 with two-way terms: 2^15 = 32,768 models. |
| **Single focal interaction** (v1.0/v1.1) | (focal interaction F, covariate set C); Focal mode C = ∅, Subsets mode C = S \ F. Unchanged so old runs reproduce. |
| **Full factorial** | per subset S (\|S\| ≥ 2) the full hierarchy up to the maximum order. Not the same as every subset of its terms. |
| **Main effects only** | every subset S as main effects. |
| **Time × each predictor** (longitudinal) | focal sets that always contain the time variable: time × X (and time × X × Z with three-way). |

"Include main-effects-only models" adds the main-effects block to focal / factorial / time strategies.
Exact counts are computed before fitting (sum over subset sizes of C(p, s) × number of interaction sets);
resuming skips whole subsets by count, so a resumed search never re-enumerates finished candidates.

## Hypotheses (test plans)

Each candidate's full model is fitted once; each hypothesis is a nested comparison with a reduced model
(also hierarchical). The first test listed is the model's declared focal hypothesis.

| type | reduced model | when |
|---|---|---|
| focal interaction | full minus the focal term | focal strategy |
| individual interaction term | full minus one *maximal* interaction term (a two-way term inside a three-way term is never dropped alone) | every interaction model |
| joint interaction | main effects of S + R only | two or more interaction terms |
| main-effect term | full minus one main effect | main-effects-only models with \|S\| ≥ 2 (main effects inside interactions are not tested: marginality) |
| whole model | R only (or intercept only) | every model |
| covariance structure | the nested smaller structure | workflow B (longitudinal) |

Tests: nested F (linear), likelihood-ratio χ² (logistic, ordinal, multinomial, ML mixed models, GLMM,
CLMM), robust sandwich Wald (GEE). Multi-level factors give multi-df tests; multinomial tests are joint
over all outcome contrasts.

**Multiplicity families** are the hypothesis types within a run: BH and Holm are computed separately for
each type, never across types. On partial / stopped runs adjusted p-values are labelled provisional.

**Confidence intervals** carry typed metadata: a CI qualifies for the strict highlight only when it is the
interval of the single tested coefficient (`ci_applicable = same_estimand`), finite, and excludes its null
(0 for coefficients / log-odds; 1 for odds ratios). Joint tests have no single-coefficient CI.

## Longitudinal searches

Study structure *repeated measures* requires an explicit participant ID (any column with repeated values)
and, for slopes / correlation structures / the time strategy, a time variable (numeric codes; labels are
mapped back). **Workflow A** locks one structure and searches fixed effects with ML fits; every full and
reduced model shares the grouping, structure, rows and estimator. **Workflow B** locks the fixed part and
compares structures (REML for Gaussian; AIC computed uniformly as −2 log L + 2 × (fixed + covariance
parameters); boundary-corrected LR tests for nested variance components; QIC for GEE working correlations).

| structure | engine |
|---|---|
| random intercept / + time slope (correlated, uncorrelated) | `lme4::lmer` |
| random intercept / slope, independent residuals; AR(1) over occasions; continuous-time AR(1) | `nlme::lme` |
| unstructured / compound symmetry / AR(1) marginal covariance (no random effects) | `mmrm::mmrm` |
| binary logistic mixed model, random intercept / slope | `lme4::glmer` |
| ordinal cumulative-logit mixed model, random intercept / slope | `ordinal::clmm` |
| GEE, exchangeable / AR(1) / independence working correlation (population-average) | `geepack::geeglm` |

Repeated multinomial outcomes are refused in discovery (no valid LR test for the PQL approximation).

## Screening

A candidate is *invalid* (recorded, not tested) when it exceeds the parameter limit, has fewer complete rows
than the minimum, leaves no residual df, has an empty factor cell within an interaction term, has a
rank-deficient design, or (longitudinal) has fewer than 10 participants with 2+ time points for a random
slope. *Flags* (tested, excluded from support tiers A/B, shown separately from significance): sparse cells
(< 5), |r| > .9 between continuous predictors, separation, non-convergence (full or reduced model),
singular fits, ill-conditioned Hessians, < 10 events per parameter, R warnings.
