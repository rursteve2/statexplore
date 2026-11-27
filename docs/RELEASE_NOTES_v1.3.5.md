# StatExplore 1.3.5 — corrective patch

v1.3.5 corrects 11 findings (AUD134-01 to AUD134-11) from an independent source audit of v1.3.4. It is a targeted
patch: it adds no statistical method and is not a new architecture. Every v1.3.4 correction and regression obligation
is preserved.

**Details:** `docs/v1.3.5/` contains the evidence report, defect register, methods and compatibility notes, the
benchmark and the owner acceptance tests 55–61.

## What changed

**Selection on exact log tails.** This applies in the legacy Validation Workspace as well as in AMD validation (01).
Stability and full-procedure validation used to pick the smallest ordinary p. Two p-values below the double range were
then both 0 and artificially tied. Every selection path now uses the natural-log tail of the test distribution. Exact
ties go to the lower candidate position, and a failed or withheld test never wins. Tables show log10 p beside p.

**Rankings by adjusted p sort on their log tails** (02). The same holds for BH, Holm and the log10 column, within each
test-type family. Teaching examples drawn from searches are ordered the same way.

**Resume only compatible searches** (03). A partial search resumes in place only when every retained result was fitted
under the current fit rules and grammar, on the same data, with recorded R / package versions. The versions of every R
worker are checked before anything is written.

- A refusal changes nothing in the search.
- *New search from this one* starts an explicitly new, linked search that refits every candidate.
- The saved 52D project (fit rules 3) is now refused with the reason, which is the intended behaviour.

**The minimum of candidate predictors applies to the single-focal-interaction strategy** (04, grammar 4). In Focal
covariate mode a pair counts 2 candidates. In Subsets mode the subset size counts. The control is disabled for
workflow B, which has a locked fixed part.

**Fixed-candidate validation validates the saved model** (05, 08). The candidate is rebuilt from its saved specification
and verified by its key; it is not looked up by its position in the current grammar. This takes constant time even in
a 147-million-candidate space. R checks the fitted terms.

**The validation pool scope is explicit** (06).

| Scope | Meaning |
|---|---|
| Complete declared procedure (default) | all execution budgets; for 5 + 10 that is 15 candidates |
| Original execution | the first execution's budget |
| Explicit first N | restricted to the first N candidates |

Adaptive continuations and truncated executions are disclosed.

**Historical identity is authoritative in audits and bundles** (07).

- The audit's data block describes the analysed data from the search's own record.
- The current project appears only in a separately labelled block.
- A validation export writes a runnable reproduction bundle only when the data reproduce the analysed view. Otherwise
  it writes results only and explains why.

**Cancellation and time limits apply to cached results** (09). Cached rows are never committed after an acknowledged
cancel or deadline.

**A conservative validation workload bound and a hard fit budget** (10).

- The feasibility check uses an uncached upper bound, not an optimistic estimate.
- R counts every model-fitting invocation, including retries and auxiliary fits, and stops at the budget.
- A part stopped by the budget is never scored.

**The standalone reproduction script checks everything it reproduces** (11). That covers manifests, predictions, folds
and sizes, selected keys, bootstrap coefficients, the candidate trace, and recomputed metrics and selection frequencies.

- Keys must be unique on both sides.
- Expected files and row counts are embedded in the script.
- Failed draws are compared without NA errors.
- Results go to `validation_check_results.csv`, and the script exits non-zero on any mismatch.

## Compatibility

Projects from every earlier version open unchanged; new fields have defaults.

| Item | v1.3.4 | v1.3.5 |
|---|---|---|
| Candidate grammar | 3 | 4 |
| Fit rules | 4 | 4 (unchanged) |
| Validation grammar | validation-2 | validation-3 |
| Audit schema | 1 | 2 (schema-1 records stay readable) |

Validation configs saved with pool `declared` keep their v1.3.4 meaning (original execution).

## Verification

**Tests and baseline.**

- New tests: `tests/test_v135.py` (red on v1.3.4 dd7313a, green here).
- The full inherited suite was run on the same machine.
- Independent R oracles were used: `pf` / `pchisq` with `log.p = TRUE`, and fresh `lm` fits.
- Gate-0 reproductions of every finding on the released baseline are in `docs/v1.3.5/evidence/`.

**Environment-dependent steps.**

| Step | Status |
|---|---|
| Physical display scaling | not run |
| Clean-machine offline install | owner-approved deferral |
