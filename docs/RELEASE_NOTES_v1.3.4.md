# StatExplore 1.3.4 — release notes (corrective release)

1.3.4 corrects statistical and functional defects found in the owner's v1.3.3 acceptance campaign (Tests 43–54). It
adds no statistical families or methods. Every issue below has a failing test on v1.3.3 and a passing test on 1.3.4;
the evidence is in `docs/v1.3.4/` (defect register, evidence report, methods, benchmark).

## Results that may change compared with 1.3.3

- **Bootstrap validation (AMD ▸ Validation)** (V134-001). v1.3.3 reused one resample's reduced-model fits for later
  resamples in the same 10-draw chunk, because every resample was relabelled b1…bn and the cache was keyed by row
  names. Selection frequencies, stability counts and "failed" draws from 1.3.3 bootstrap runs may be wrong. Example:
  51H3 gave 8 / 3 / 9 selections of #13 / #16 / #17 in 1.3.3, against 0 / 0 / 20 from fresh independent fits (1.3.4
  matches the fresh fits). Holdout and CV runs were not affected by this mechanism, but they now use the corrected
  candidate pool (see below). Rerun earlier bootstrap validations.
- **GEE workflow B** (V134-005). Every structure was fitted with the first structure's working correlation, so all
  QICs were equal. Each structure now uses its own working correlation, read back from the fitted object
  (independence / exchangeable / AR(1) for 48D: QIC 1146.365 / 1147.731 / 1149.216). Rerun earlier GEE structure
  comparisons.
- **Validation candidate pool** (V134-004). The pool is now the declared search space, in its declared order, with the
  search's own stopping rule replayed on each training part. It was previously the full-data run's evaluated,
  non-invalid candidates. Whole-pipeline holdout / CV results of early-stopped or partial searches can change.
- **Extreme p-values** (V134-002). Ranking, BH / Holm, highlighting, early stopping and validation selection now use
  R's native log upper tail. Ties caused by p underflowing to 0 are gone, and the tables show
  `< 1e-300 (log10 p = −485.20)`.

Searches saved by 1.3.3 open unchanged as history. Their rows were fitted under fit rules 3, so they are never reused
as cached fits; a new search refits them (fit rules 4).

## Fixes and refinements

| ID | What changed |
|---|---|
| V134-001 | Fit caches keyed by a fit key built from the content of every model column, the row identities, the family, structure, cluster, ML/REML, GEE working correlation, fit rules and the sample context. Each training part and draw gets a fresh cache. Hits and misses are counted, and each reference fit keeps its own diagnostics and provenance. |
| V134-002 | Native log tails for LR, F and Wald tests (and the Python test service, via a continued fraction); log-scale BH / Holm; sorting by log p; `log_p`, `log10_p`, `log_p_bh`, `log_p_holm` and `log_p_adj` in tables and exports. |
| V134-003 | Bootstrap counts by candidate identity (index + key), never by display label. Labels carry `[#index]`. Counts sum to the successful draws. |
| V134-004 | The training-only pipeline. Workload is checked from exact counts before any list is built (the 147,713,444-candidate space is refused in under a second). An explicit "restricted pool" is labelled as such. |
| V134-005 | `geeglm` gets the structure's own `corstr`; readback of corstr, alpha, scale, clusters and robust SEs. QIC / QICu / CIC are kept in their own fields and columns (never in the AIC column). AIC-based validation is refused for GEE. |
| V134-006 | AIC/BIC ranks, deltas and "best" only when every model used identical analysed rows (SHA-256 row fingerprint); equal n is not enough. Otherwise the models are listed in candidate order. |
| V134-007 | Opt-in **Include time in eligible interactions** (combinations / factorial): predictor × predictor and time × predictor terms in one model, with time counted once. The time strategy now honours the minimum number of candidates. Grammar version 3 is recorded with each search. |
| V134-008 | Validation manifests carry the real search identity, the validation ID, the data hash and requested vs realized test sizes. A portable `validation_reproduce.R` re-runs every chunk outside StatExplore and checks the predictions and selections. |
| V134-009 | Resume shows why it is unavailable (complete, data changed, grammar, running). It is never re-enabled blindly after a run. A real two-process restart test covers resuming 52D from 5/15 to 15/15 under the same search ID. The owner's greyed-out Resume did not reproduce from the saved project (see the register). |
| V134-010 | A reference (reduced) fit's warnings flag only the tests that use it (test status *provisional*). Warnings are captured per fit and travel with cached fits. |
| V134-011 | Every coefficient of a multi-df primary test is reported, e.g. the 51H3 triple coefficient (≈ 1.35–1.62) instead of its parent. Aliases are mapped to display names. Wording: "decreases by 5.9 percentage points"; time ranges are labelled with the time variable. |
| V134-012 | No cache reuse from runs without their own fit identity, or from older fit rules. |
| V134-013 | Candidate **Order** controls (▲ ▼ ⤒ ⤓, A–Z, data order), drag-and-drop, Ctrl+arrow keys and visible positions. The order is the search order, saved with the form. |
| V134-014 | View Code offers **Educational R code** (short, original names, clearly labelled as didactic) beside the **Complete reproducible script**. Ctrl+F, Copy and Save bundle are unchanged. |
| V134-015 | **Export audit trail…** writes a versioned JSON record (schema `statexplore-audit` v1) and a Markdown report per search. No row-level data and no personal paths are included. Project exports include it. |
| V134-016 | Validation workload checks without materializing candidates; the progress bar is safe above 2³¹. Cancel latency is about 2 s with no orphaned R processes (benchmark). |
| V134-017 | Simulated display-scaling layout tests (100 / 125 / 150 / 200 %); frozen build self-test. Physical-display DPI and a clean-machine install are owner-approved deferrals (not run). |

## Compatibility

- Projects from 1.3.3 and earlier open unchanged; saving keeps the project format (version 2).
- Saved forms and earlier searches keep the 1.3.3 grammar: the time-eligible option is off unless ticked.
- Old time-strategy searches with a minimum above 1 cannot be resumed under grammar 3; the reason is shown and a new
  search uses the corrected grammar.
- The CSV / XLSX exports gain additive columns (log-p fields, test status, reference flags). Existing columns are
  unchanged.

## Known limitations

- The exact 47C replay of counts [1,0,1,6,1,0] needs the original 47C search settings, which are not in the handoff.
  The duplicate-label mechanism is reproduced and fixed.
- Optimizer-path differences remain for flagged, non-converged lme4 fits when another program writes the terms in a
  different order (51H3 #11 / #12, |ΔlogLik| ≤ 0.134). Selections are unaffected and the fits are flagged.
- Not run (owner-approved deferrals): physical-display DPI at 150 %, and a clean-machine install.
