# StatExplore 1.3.4 — numerical, identity, validation and reporting rules

The rules below are what the code does (file / symbol given), not aspirations. They complete `docs/formula-grammar.md`
(grammar 3) and `docs/v1.3.3/validation_methods.md`.

## 1. Log-domain inference (V134-002)

- **Source of truth:** the natural-log upper tail from the reference distribution itself, never `log(p)`:
  - LR tests: `pchisq(LR, df, lower.tail = FALSE, log.p = TRUE)`.
  - Boundary mixtures: the log-sum-exp of the component log tails (`sx_lr_test`).
  - Nested OLS: `pf(F, df1, df2, lower.tail = FALSE, log.p = TRUE)`.
  - GEE robust Wald: `pchisq(W, df, …, log.p = TRUE)` (`discovery/rcode.py sx_compare / sx_wald`).
- The Python service (`analysis/lrtest.py`) uses `log(chi2.sf)` while `sf > 1e-280`. Below that it uses a log-space
  continued fraction / series for the regularized upper incomplete gamma (`log_upper_gamma_q`). It is checked against R
  and against `log 2 + log Φ(−√x)` for 1 df (54B: −27,048.436 / −11,691.989).
- **Stored fields:**
  - `p` is the ordinary double, which may be 0 after underflow.
  - `log_p`, `log10_p = log_p / ln 10` and `log_p_source` (`native log tail`, `log of p`, or `p underflowed;
    magnitude not recorded` for an older row with p = 0 and no tail).
  - A withheld test has no p and no log p, and is not in its family.
- **Adjustment, on the log scale, within each test-type family** (`engine.log_bh / log_holm`):
  - BH: log p sorted ascending, backward cumulative minimum of log p₍ᵢ₎ + log m − log i, clipped at 0.
  - Holm: forward cumulative maximum of log p₍ᵢ₎ + log(m − i + 1), clipped at 0.
  - The result is restored to the input order. Exact ties keep candidate / test order, the documented secondary key.
  - Agrees with `p.adjust` / statsmodels to 1e-12 for representable p.
- **Uses:** ranking, alpha comparisons (`log p < log α`), strict highlighting, the early-stop count, support tiers,
  validation selection and exports all use log p. The effect-first support ranking is unchanged.
- **Display:** below 1e-300, `< 1e-300 (log10 p = −485.20)` (`gui/models.fmt_p`); interpretations use the same text.
- **Negative LR:** a negative LR within 1e-6 is set to 0 and the raw value is kept. A materially negative LR withholds
  the test with its reason. For OLS, a negative F beyond 1e-8 · RSS₀ withholds the test.

## 2. Fit identity and caches (V134-001 / V134-012)

**Fit key** (R `sx_fit_key`). A reduced or null fit is reused only under a key that contains:

- the sample context (`search`, `<split>:fold<k>` or `boot<b>`), family, structure, cluster flag, ML/REML, the GEE
  family and working correlation, and the fit-rules tag
- the canonical term labels and the row count
- an MD5 of each used column (response, design variables, `sx_gid`, `sx_cluster`, `sx_time`, `sx_timef`, `sx_occ`),
  computed over its row names, factor levels and values

Each column is hashed once per sample (cached on identical row names and contents).

**Per-sample reset** (`sx_cache_reset`). Validation resets the cache for every training part and every bootstrap draw.
Bootstrap rows are renamed `b<draw>.<k>`, so row identities are draw-specific; the source row is kept in the manifest.

**Diagnostics travel with the fit.** A cached reference keeps its own convergence / singularity / warning text
(`sx_fit_capture`). A cache hit reports `reference_source = cache` and the candidate that produced the fit
(`reference_fitted_for`).

**Python-level reuse between searches** (`engine.cache_from`) requires all of:

- the same data signature
- a recorded fit identity (never reconstructed from current data)
- the same fit rules (now 4)
- the same R / package versions

Rows from v1.3.3 and earlier (fit rules ≤ 3) are kept as labelled history and never reused.

Presentation settings (alpha, adjustment, highlight, sort) never enter an identity.

## 3. Comparability of criteria (V134-006)

- AIC / BIC deltas and AIC ordering need identical analysed rows (equal SHA-256 row-identity fingerprints, or row
  signatures for older rows); equal n is not enough. Otherwise each model's criterion is shown in candidate order, with
  no delta and no "best".
- Mixed native BIC bases (n vs n − p) keep the harmonized-BIC column from v1.3.2.
- GEE has no likelihood: no AIC / BIC / LRT. QIC, QICu and CIC (geepack definitions from the actual fit) are compared
  only on identical rows:
  - QIC: working correlation and mean model
  - QICu: mean models under one working correlation
  - CIC: working correlation
- Validation's AIC rule refuses GEE searches and per-model complete cases. Within a training part, candidates fitted
  to different rows stop that part with a reason.

## 4. Validation pipeline (V134-004 / 008 / 011)

- **Pool:** the declared search space in its declared order. For mode D it is the declared model budget (an
  outcome-independent prefix). An explicit restriction to the first N candidates (`pool = prefix`) is labelled
  RESTRICTED. The full-data run's statuses, rows, stop point or speculative results never define the pool.
- **Stopping:** the search's own outcome-dependent stopping rule (mode A: first candidate with raw p < α; mode B: the
  target count) is replayed in candidate order on each training part. Selection uses the visited candidates only.
- **Selection:** the smallest training log p of the primary test, or the lowest training AIC on common rows. Exact
  ties go to the lower candidate index.
- **Workload:** pool × training parts × (1 + planned reduced fits), from exact counts and the average test plan of
  the first 500 candidates, checked against the budget before any list is built.
- **Records:**
  - per part / draw: `candidate_trace` (every candidate's logLik, LR, df, log p, reference logLik, flags)
  - per draw and coefficient: `bootstrap_estimands` (every coefficient of the primary test; sign and percentiles
    conditional on selection and on that coefficient existing)
  - `realized_sizes`: requested vs realized test units and rows; rule ceil(fraction × units); rows are not balanced
  - the real search identity (`fit_id`), validation ID, data SHA-256 and grammar
- **Portable reproduction:** `validation_reproduce.R` contains the data load, setup, every chunk with its exact
  manifest, and self-checks against `predictions.csv` / `bootstrap_results.csv` (1e-6). It is exported with a relative
  data file (with consent), a SHA-256 data check and a package check.

## 5. Diagnostics attribution (V134-010)

- A model row's status and flags come from its full fit (plus batch-level R warnings).
- Each test carries `reference_flags` / `reference_source` from the reduced fit it used. A flagged reference makes
  that test *provisional* (support tier not A/B) without flagging the model or its other tests.
- Retry with another optimizer only on optimizer errors, on the same model; this is unchanged from v1.3.3.

## 6. Teaching definitions (V134-011)

- Marginal R² changes are reported in percentage points, and decreases are called decreases: a direct variance
  decomposition need not increase when terms are added.
- OLS ΔR² is in percentage points.
- A test without a p-value says why and that it is outside its family.
- Bootstrap selection frequencies and sign stability are not probabilities that a hypothesis is true, and the
  percentiles are not post-selection confidence intervals.
- Time ranges in the preflight are labelled with the time variable.

## 7. R code modes (V134-014)

- **Complete reproducible script** is the exact code StatExplore ran, with stopifnot checks of n, rows, identity,
  logLik / AIC / BIC and tests; it is unchanged from v1.3.3 and can be saved as a portable bundle.
- **Educational R code** is short and annotated, with original variable names. For a Discovery candidate it shows the
  engine call, the full and each reduced model, and the test. Elsewhere it is the exact script with aliases written
  as names. It is labelled DIDACTIC and states what it does not reproduce (rows, filters, coding, composite keys,
  optimizer fallback, adjustment).

## 8. Audit trail schema (V134-015)

`export/audit.py`, schema `statexplore-audit`, version 1. The top-level keys are all required:

- `schema`, `schema_version`, `generated_utc`, `application`
- `search`: id, fit identity, data signature, status, executions, resume state, provisional flag, fit rules of rows,
  historical flag
- `settings`, `environment` (R / packages, Python, OS)
- `data`: file name only, SHA-256, snapshot of metadata; no rows
- `grammar`: version, exact space size, order, budgets, counts
- `candidates`: identity, diagnostics, cache provenance, GEE readback
- `hypotheses`: raw / log / adjusted p, family, status, reference flags
- `validation`: identities, pool, stop rule, realized sizes, performance, stability, estimands
- `limitations`

Serialization is canonical JSON (sorted keys, `allow_nan = False`, NaN / Inf → null) written atomically.
`read_audit` refuses another schema, a newer version, missing fields or NaN. Timestamps never enter semantic
identities.

## 9. Migration and offline operation

- Project format: unchanged (version 2).
- Searches gain `grammar` (missing = 2) and the new row fields; older rows keep theirs.
- v1.3.3 projects open, save and reopen with the same searches and rows (`test_v133_projects_open_as_labelled_history`).
- No network access at startup or during analysis. The bundled R (4.6.2) and packages are installed from the build's
  local library.
- Exports use relative paths and check package versions without installing anything.
