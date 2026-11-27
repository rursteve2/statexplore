# StatExplore v1.3.5 — methods, compatibility and historical-results policy

This document supplements `docs/v1.3.4/methods_v1.3.4.md`. Everything there still holds unless a section below changes
it. v1.3.5 is a corrective patch: it adds no statistical method.

## 1. Selection on log tails, in every engine (AUD134-01, AUD134-02)

**One score.** Every path that picks a model by its p-value uses one R helper, `sx_primary_logp(r)`. It returns the
natural-log upper tail of the candidate's primary test, `tests[[1]]$log_p`. That value comes from the test distribution
itself: `pf`, `pchisq` or the boundary mixture, with `log.p = TRUE`. The helper returns `NA` in three cases:

- the fit is not `ok` or `warning`;
- the test was withheld (for example, a negative LR);
- the test has no p.

It never returns `log(0)` and never substitutes a value. `sx_select_min(score)` takes the smallest score. An exact tie
goes to the lower candidate position (the declared secondary key), and `NA` never wins.

**Where the score is used.** AMD validation (`sx_val_select`) and the legacy Validation Workspace both use it:

- discovery stability: the argmin per resample, and the share with `log p < log α`;
- single-level full-procedure CV;
- longitudinal full-procedure CV.

The ordinary p stays in these tables for display. The tables add log10 columns and a per-resample selection trace.

**A known limit.** The rule ranks each candidate's own primary test. In a combinations search those tests can be of
different types (a joint test, a single interaction term, or the whole model). The legacy and AMD engines document
this the same way. It is the search's own discovery ranking, not a cross-family inference.

**Rankings and sorting.** `discovery_ranking` sorts each probability column by its log twin:

| Column | Sort key |
|---|---|
| raw p, log10 p | log p |
| selected adjusted p | log p_adj |
| BH | log p_bh |
| Holm | log p_holm |

Adjusted p-values belong to their test-type family. An adjusted ranking is therefore listed within each family, in the
fixed family order. The raw ranking stays one list, as in v1.3.4. Ties break by raw log p, then candidate index, then
test id. Missing values sort last. The support-tier ranking is unchanged.

**Rows from older searches.** A p of 0 with no recorded tail sorts first at `-inf`. Its source column says "magnitude not
recorded", so the table does not imply an exact zero or a precise tie.

**Verified by** independent R `pf` / `pchisq` calls (`log.p = TRUE`) and fresh `lm` fits on fixture 55L. The log-domain
BH / Holm oracle is a separate loop written in the test file.

## 2. Resume compatibility and the historical-results policy (AUD134-03)

**When a search resumes in place.** All of the following must hold:

- it is partial or cancelled;
- its candidate grammar reproduces the declared order (`grammar_problem`);
- every retained row was fitted under the current fit rules (4), with recorded R / package versions;
- its fit identity is recorded and matches the current data view;
- its data signature matches;
- its saved progress is consistent: unique indices inside the space, every candidate before the continuation point has
  a row, and every row has a key;
- each retained key equals the key the grammar produces at that index (checked in `execute`).

`execute` then starts the workers and queries the versions of every worker. A worker that differs from another, or
from the versions that fitted the retained rows, is refused. Nothing is committed before these checks.

**On refusal.** `ResumeRefused` leaves status, executions and rows exactly as they were. Missing provenance is never
treated as compatible.

**Incompatible searches** stay viewable and exportable with their original caveats. "New search from this one"
(`restart_from`) creates a NEW search with a new ID. It uses the parent's declared settings under the current grammar
and fit rules, and every candidate is fitted again. The new search records `parent_id` and a `lineage` block. Nothing is
carried over from the parent, and no fit-rules-3 row can enter its cache or its multiplicity families. A parent from a
newer grammar is refused: its specification cannot be expressed by this version.

**Saved 52D** (grammar 2, fit rules 3, 5 / 15). Resume is disabled with the reason; a new linked search is offered.

## 3. Grammar 4: the minimum of candidate predictors (AUD134-04)

The minimum counts **candidate** predictors. Required covariates, including time, are never counted. The maximum counts
distinct predictors, with each required variable counted once. Grammar 4 applies the minimum to the single-focal-
interaction strategy:

| Covariate mode | How a model's candidate count is measured |
|---|---|
| Focal | the focal variables only: a pair is 2, a triple 3 |
| Subsets | the subset size |

Counting, preview, enumeration, keys, execution, validation and generated code all share the same enumerator.
Enumeration skips whole blocks and subsets by their exact counts, so a resume or an indexed lookup never walks the
skipped part of the space.

Workflow B uses one locked fixed part. There the minimum, strategy, order and main-effects controls are disabled with a
tip; they are not silently ignored.

**Compatibility of older searches.** A focal search of grammar 3 or earlier with a minimum above 2 now has a different
candidate list. It is not resumed, its whole-procedure validation is refused, and its fixed candidates are validated from
their saved specification. Searches whose list is unchanged stay compatible.

## 4. Fixed-candidate validation uses the saved specification (AUD134-05, AUD134-08)

`fixed_candidate(run, i)` rebuilds candidate *i* from its saved row: focal variables, covariates, kind, interaction
terms and structure. It then checks the rebuilt key against the saved key. It never indexes the current enumerator, and
it never enumerates at all, so the cost is the same for a 15-candidate space and a 147,713,444-candidate space.

The validation refuses when:

- the row is missing;
- the key is missing;
- the rebuilt key differs from the saved key.

In R, `sx_val_check_terms` compares the fitted object's fixed terms with the candidate's terms and fails closed if they
differ. The key travels into folds, draws, coefficients, the trace and the exports.

Whole-procedure validation repeats the declared search. It is refused when the grammar changed, or when a saved key
differs from the key at the same position in the repeated pool. It never silently runs a different search.

A fixed candidate fitted under older fit rules is refitted under fit rules 4, and the validation record states this.
That refit is not a reproduction of the original fit.

## 5. Validation pool scope (AUD134-06)

The pool comes only from declared quantities: the grammar's exact count, each execution's declared model budget, or an
explicit prefix. It never depends on the rows evaluated, their statuses or p-values, the stop index, or the number of
significant findings.

| Scope | Mode D (budget-limited) | Other modes |
|---|---|---|
| cumulative (default) | sum of the declared budgets, capped at the total | the whole declared space |
| original (v1.3.4 "declared") | the first execution's budget | the whole declared space |
| prefix | the first N in the declared order (RESTRICTED) | same |

**Disclosure.** Continuation budgets are chosen after seeing earlier results, so cumulative validation of a multi-
execution search is labelled ADAPTIVE. It is a conditional validation of the frozen prefix, not an unbiased validation of
the adaptive procedure. An execution that stopped on time or by cancellation before its budget is noted; time and
cancellation cannot be replayed, so its declared budget is used. An execution with no model budget (time only) leaves the
pool undefined, and an explicit prefix is then required.

Modes A and B still replay the stopping rule on the training data only. The same scope is used everywhere: feasibility,
execution, manifest, the trace, the reproduction script and the UI.

## 6. Workload bound and hard fit budget (AUD134-10)

**Unit.** `max_fits` counts model-fitting invocations:

- every call of a model fitter: full, reduced, null and selected refits;
- each optimizer retry (nlme: nlminb, then optim);
- each auxiliary fit (the binary separation check, the GEE QIC independence refit).

Cache hits cost nothing and are counted separately.

**The bound.** `workload()` assumes no cache reuse. It uses the exact per-candidate plan (full model plus one reduced
model per planned test) for a materialized pool, or for a pool of at most 20,000 candidates. Larger pools use the
grammar's analytic maximum per candidate. On top of that it adds:

- the retry factor: 2 for nlme structures;
- auxiliary fits;
- null models for binary, ordinal and multinomial families;
- one selected refit per training part.

The bound, not the estimate, is compared with the budget. The estimate (the v1.3.4 average over the first candidates) is
still shown, labelled as an estimate.

**Hard enforcement in R.** `sx_budget_take()` counts every invocation. When the remaining budget of a chunk is used up,
it raises the condition class `sx_budget`. That condition aborts the training part; it is not a failed candidate. The
part is recorded with status `budget` and is never scored. The remaining parts are not run, and the result is `partial`.
The record reports used, retries, auxiliary fits, cache hits and misses, the requested bound and the estimate.

**Check.** On 55L, counted invocations plus cache hits equal the uncached plan exactly.

## 7. Cancellation, deadline and cached batches (AUD134-09)

**Checks.** Before every batch, cached or not, and after every wait, conditions are checked in this order:

1. cancellation;
2. time budget (monotonic clock);
3. model budget (candidate attempts, cached or fitted);
4. significance target (modes A / B);
5. exhaustion.

**What is committed.** Cached rows are never committed after an acknowledged cancel or deadline. A cancel before the
workers start, or after they start but before any batch, commits nothing; the status is `cancelled`. Running batches are
killed and not recorded; the continuation cursor is the contiguous prefix. A cancel or deadline acknowledged after the
last candidate was committed leaves the search `complete`. Progress is reported during cache-only stretches.

**Tests.** A fake clock and fake workers cover all-cached, mixed, pre-cancel, mid-cache, deadline, in-flight and
completion-race cases. The 147,713,444-candidate space is tested with a budget of 100 and a cancel near 50,000 cached
rows; the whole space is never executed.

## 8. Historical data identity in audits and bundles (AUD134-07)

**Audit schema 2.**

- `data.historical` is built from the search's own record: the snapshot, the analysed-content signature, the fit
  identity, filters, derived variables, holdout and treatments. It defines its fingerprints: the source-file SHA-256 is
  not the analysed-row fingerprint. Metadata that was never recorded is `unknown`; it is never filled from the current
  project.
- `data.current_context` is optional and labelled "export-time project context (NOT the analysed data unless
  matches_analysis is true)", with the differences listed.
- Hypotheses carry `p_underflow` and `log_p_unknown_magnitude`, so a stored 0.0 is never mistaken for an exact zero.

Schema-1 records stay readable, and their report notes that the file line may describe the export-time project.
Snapshot schema 2 (new searches) adds the source file name (never a path), format, size and row count.

**Validation export.** `bundle_data` writes a runnable reproduction bundle only when the data view reproduces the
validated search's data signature: the current view when the project is unchanged, or the original view rebuilt from
the snapshot when the source bytes are identical. Otherwise it writes the results, manifests and code, and
`REPRODUCTION_UNAVAILABLE.txt` with the reason. It never writes a data file. The manifest records `reproduction.runnable`.
Row-level data are written only with consent and never in the audit.

## 9. Standalone reproduction checks (AUD134-11)

`validation_reproduce.R` replays every chunk, including its budget guard, from the exported folder. It then verifies the
items below against the app's exported CSVs:

| Item | Keys and comparison |
|---|---|
| split manifest | split, row |
| bootstrap draw manifest | draw, draw unit, row |
| predictions | observed, predicted, participant, cluster |
| folds | status, selected index **and key**, realized sizes, exclusions, visited / scored, training score and mean |
| performance | recomputed in R from the reproduced predictions with an independent implementation of the aggregation |
| bootstrap draws | status, selection and key, diagnostics, n, groups, estimate |
| selection frequencies | recomputed |
| bootstrap coefficients | draw, candidate, coefficient |
| candidate trace | part, candidate: status, row fingerprint, log p, statistic, log-likelihood, AIC, score |

**Key and file rules.**

- Keys must be unique on both sides, and the key sets must be equal, so a duplicate or an extra row fails.
- Missing selections compare equal only together with equal statuses: two failed draws match; a draw that failed on
  only one side fails the check.
- The expected evidence files and their row counts are embedded in the script. A missing or truncated required file
  fails. An optional file that does not apply to the configured modes is reported NOT APPLICABLE.

**Tolerances.**

| Quantity | Tolerance |
|---|---|
| predictions, observed values, metrics | abs 1e-6 + rel 1e-6 |
| log p, statistics, log-likelihood, AIC, estimates | abs 1e-6 + rel 1e-6 (mixed models: rel 1e-5) |
| counts, statuses, indices, keys | exact |

**Output.** The results go to `validation_check_results.csv`. Any FAIL exits with status 1. A partial export is reported
as partial and never as a complete reproduction. The educational script is unchanged and separate.

## 10. Version identifiers

| Identifier | v1.3.4 | v1.3.5 |
|---|---|---|
| Candidate grammar (`GRAMMAR_VERSION`) | 3 | **4** |
| Fit rules (`FIT_RULES`) | 4 | 4 (no stored statistic changed definition) |
| Validation grammar | validation-2 | **validation-3** |
| Audit schema | 1 | **2** |
| Data snapshot schema | 1 | **2** |

Project files need no migration. New fields have defaults (`parent_id`, `lineage`, validation `pool`, `pool_scope`,
`provenance`, `fit_budget`), and older files open unchanged.
