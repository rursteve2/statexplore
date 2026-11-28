# StatExplore v1.3.6 — methods of the corrective changes

v1.3.6 changes no statistical method, model family, estimator or tolerance. It changes how data are identified, how
exports are written, how validation work is accounted for, and how exported evidence is verified.

## 1. Data identities (AUD135-06) — `core/identity.py`

| Identity | What it covers | Used for |
|---|---|---|
| Source file SHA-256 | the bytes of the file that was opened | provenance only (never proof of the analysed content) |
| Content identity (schema 3) | the imported values: column names and order; per column its kind (number, text, datetime, bool) and every value bit-exactly (little-endian float64 bits, int64 nanoseconds, length-prefixed UTF-8 text) with a separate null mask; row count and index; and the parse context (format, sheet, delimiter, encoding, reshape options, R Console origin, typing report, source file SHA-256) | snapshots, history rebuild, analysis provenance |
| View identity (schema 3) | content identity + variable metadata (measures, codes, labels, user-missing, category order, treatments) + derived / filter / holdout specifications + the REALIZED filter mask, holdout assignment and derived values | search data signature, fit identity, cache, resume, validation bundles |

- Same worksheet re-imported → same content identity. Another worksheet, an edited value, a reordered row, a column
  rename, `"001"` vs `"1"`, missing vs empty text, or another delimiter → another identity.
- Metadata changes alter only the view identity.
- The canonical encoding refuses unsupported types (time-zone-aware datetimes, mixed objects) instead of guessing.
  There is no `default=str` anywhere in an identity.
- **Cost.** The value digest is cached per raw frame object. The cache holds a weak reference, never a bare address,
  plus a shape / column / dtype guard. A Dataset's raw frame is immutable by contract: import, console import,
  reshape and reopen all build new Datasets. The realized-mask digest is cached per (content, metadata).

### Legacy policy

Searches recorded before v1.3.6 have identity schema 2 (source file hash + metadata). They are **unverified**:

- they stay viewable as history;
- they are never used for cache reuse, resume, snapshot rebuild or a runnable reproduction;
- *New search from this one* creates a linked schema-3 search.

Nothing is backfilled from the current data.

## 2. Analysis provenance and historical exports (AUD135-07)

`RSession.run` records `analysis_provenance` (schema 1) for every result, failed ones included. It is captured right
after the worker loaded the data and contains:

- the content and view identities;
- the SHA-256 and row count of the exact CSV bytes R read, before any column projection or pseudonymization;
- the SHA-256 of the load script and of the full script;
- the data-view snapshot and the treatments.

A historical result is exported as a runnable bundle only if `export/historical.resolve` re-creates a CSV with that
same SHA-256:

1. from the current project view, else
2. from the result's snapshot, rebuilt on the same content identity.

Otherwise the export refuses, with the reason. Reports writes one bundle per analysis (its own script and, with
consent, its own projected data file with its own SHA-256). The data are never shared through one `data.csv`.

## 3. Transactional exports and consent (AUD135-03) — `export/bundle.py`

Every export (validation, portable bundle, Reports, audit, code dialog):

1. stages its files in `.<name>.incomplete-<random>` inside the chosen folder;
2. checks the inventory;
3. renames the stage to a new `<name>_<timestamp>[_n]` folder.

Nothing in the chosen folder is overwritten or deleted. A failed export removes its stage.

The registry rules:

- Files enter a bundle only through the transaction.
- Analysis data enter only through `write_data`, and only with the user's explicit consent for this export (the
  default is No in every dialog).
- `export_data_csv` refuses to write into the stage of a no-consent export.
- The inventory refuses unregistered files, symbolic links and data files without consent (by declaration and by
  name).

`bundle_inventory.json` lists each file with its SHA-256 and kind, and gives the bundle kind: results-only, code-only
or runnable. To compute the SHA-256 of withheld data, the data are written to a private temporary folder outside the
export.

## 4. Execution ledger and fit budget (AUD135-04, AUD135-05)

**Plan.** A validation has an immutable plan:

- parts: each fold of each split, and each bootstrap draw;
- chunks: one per split, and `chunk` draws per bootstrap chunk.

**Commit.** A chunk is committed only after its response was received and validated: every dispatched part returned
exactly once, and the counters are well-formed integers within the remaining budget. A lost or invalid response
leaves its parts *uncommitted*, and the fit counters become a lower bound.

**R budget.**

- `sx_budget_take` counts REFUSED fits (`denied`) apart from reaching the cap (`at_cap`).
- A part during which any fit was refused becomes a budget stop, even if a wrapper swallowed the refusal. It keeps no
  predictions and no coefficients.
- Once a fit has been denied, later parts of that chunk are marked "not run: budget used up".

**Final status** follows the ledger:

| Status | When |
|---|---|
| complete | every planned part committed successfully and no fit refused (used = cap is complete) |
| cancelled | a cancel request stopped the run (a cancel that arrives after the last part is recorded; the run stays complete) |
| failed | a chunk failed or was lost |
| partial | work was truncated by the budget, or a part failed |

Python does not dispatch a chunk when no fit is left and work remains.

**Reproduction scope.** The reproduction script:

- replays the committed chunks only, with their exact guards;
- embeds the pending plan (splits and draws not committed) and the ledger;
- verifies the planned manifests (committed parts from the replayed code, pending parts from the embedded plan);
- checks that the replayed parts and statuses equal the committed ones;
- prints the coverage. A partial run is never called a complete reproduction.

## 5. Typed, complete verification (AUD135-01, AUD135-02)

### Writer (Python)

`EVIDENCE` is a schema for every evidence file. It declares:

- the keys;
- each column's type: `int`, `num`, `text` or `bool`;
- flags: `na` (may be missing), `inf` (signed infinity allowed), `need=<status>` (required when the reproduced row has
  that status), `tol=<class>`.

The Python writer fixes the columns and their order. It writes `repr()` floats, `Inf` / `-Inf`, `""` for missing, and
exact integers. It refuses undeclared columns and wrong types. A file is written with its header even when it has no
rows.

### Reader (generated R)

- Every cell is read as text: `colClasses = "character"` and `na.strings = character(0)`.
- Each value is parsed by its declared type: finite, `+Inf`, `-Inf`, missing or malformed.
- Malformed text, NaN, decimal overflow and silent underflow to 0 are FAIL.
- Integers and text compare exactly, so `"001"` ≠ `"1"`.
- Signed infinities match only the same sign.
- Missing equals missing only where the field allows it.
- Keys are length-prefixed UTF-8 and must be unique on both sides.
- A check with an NA or vector result is FAIL. NOT APPLICABLE appears only where it is declared.

### What R recomputes

**From replayed code:**

- predictions, folds, bootstrap draws, coefficients and the candidate trace (including df, reference log-likelihood,
  flags and row signatures).

**Independently in R, from those reproduced values:**

- **performance** (as in v1.3.5);
- **stability**, over the DECLARED candidate pool (index / key / label from the setup, never from `stability.csv`):
  - attempted / successful denominators (committed / fitted draws);
  - sign = share of estimates > 0;
  - percentiles: `quantile(type = 7)` (the same as pandas' linear);
  - flagged fits;
  - coefficient display labels, from the embedded alias map;
- **estimands**, plus a coefficient identity check (draw, selected candidate, declared key, one term, `n_coef` rows);
- **realized sizes**, from the verified plan and R's grouping.

**Metadata** (verified against the embedded plan, not statistics):

- manifests, by unit-key partitions (same key ⇔ same R group) and cross-file original ids;
- the ledger.

### Tolerances (unchanged)

| Quantity | Tolerance |
|---|---|
| Predictions, observed values, metrics | abs 1e-6 + rel 1e-6 |
| Fit quantities | abs 1e-6 + rel 1e-6 (rel 1e-5 for longitudinal) |
| Exactly computed fractions | abs 1e-12 + rel 1e-12 |
| Counts, keys, text | exact |
