# StatExplore 1.3.6 — corrective patch

v1.3.6 corrects seven findings (AUD135-01 to AUD135-07) from the independent audit of v1.3.5. It is a targeted patch:
no new statistical method, model family or estimator, and no tolerance was widened. Every earlier correction and
regression obligation is preserved.

**Details:** `docs/v1.3.6/` has the evidence report, the defect register, the traceability matrix, methods,
migration notes, the benchmark and the owner acceptance guide. The separate **acceptance kit** runs the automatable
part of owner tests 55–61 unattended and writes one evidence ZIP.

## What changed

**Data identity is the imported content, not the file hash** (06). Two worksheets of one workbook have the same file
hash. Until now they also had the same search identity, so:

- a search on sheet B could reuse sheet A's fitted rows;
- a sheet-A search could be resumed on sheet B.

Searches, fits, caches, history and exports now identify the analysed data by a versioned content identity. It covers
the typed values, columns, rows and import options, including the worksheet, and it is computed once per import.

Searches saved by 1.3.5 or earlier cannot prove which values they analysed:

- they stay viewable;
- they are not reused, resumed or exported as runnable reproductions;
- *New search from this one* refits them as a new, linked search.

**Historical results export their own data or nothing** (07). Every result now records a SHA-256 of the exact data R
read. A Reports or code-dialog bundle of an older result carries that analysis's own data view, rebuilt and checked
against the hash, or it states why it cannot. Until now, after a holdout change, an exported t-test bundle silently
reproduced the new split (mean 10.3) instead of the recorded one (9.7). Each analysis gets its own bundle; there is no
shared `data.csv`.

**Exports never reuse a folder, and data need your consent each time** (03). Every export creates a new folder and
names it. Nothing already in the chosen folder is overwritten or deleted, so an old script or data file can no longer
sit next to a new refusal. Analysis data are included only when you answer *Yes* to "Include the data?". *No* is the
default, Reports asks too, and `bundle_inventory.json` lists every file. An interrupted export leaves no
complete-looking folder.

**Validation accounting: planned, committed and not run** (04). A validation now keeps:

- its plan;
- what was dispatched;
- what was committed after a validated response.

A cancelled or failed validation's reproduction replays exactly the committed parts, checks the rest of the plan
against the embedded plan, and says how much was reproduced. It never calls a partial run complete.

**Using the last allowed fit is not a failure** (05). A validation that finishes all its work with exactly the fit
budget is complete. Until now it was labelled partial with "remaining parts were not run". A fit that the budget
refuses always stops its part, even if the refusal was caught inside a model wrapper.

**Exported evidence is checked field by field, with types** (01, 02). The reproduction script:

- reads every value as text and parses it by its declared type: `"001"` ≠ `"1"`; +Inf, −Inf, missing and malformed
  text are all distinct;
- recomputes stability, estimands and realized sizes in R from the reproduced fits and the declared candidate pool;
- compares every field.

Altering one value, at the same row count, now fails a named check. In 1.3.5, ten such edits were all accepted.

## Also fixed (found during the work)

- **V136-D1:** the portable-bundle function defaulted to including the data. It now requires an explicit choice.
- **V136-D2:** a budget refusal swallowed inside a model wrapper could leave a part "scored" over fewer candidates.
- **V136-D3:** the R session no longer keys its loaded data by an object's memory address.
- **V136-D4:** a holdout split that changes after reopening (another numpy) is detected, not used silently.
- **V136-D5:** validation runs saved by 1.3.5 are exported as results only. Their old scripts lacked the new checks.

## Compatibility

- Projects from 1.3.4 and 1.3.5 open unchanged and keep their results.
- Identity schema 3; snapshot schema 3; audit schema 3; validation grammar `validation-4` with evidence schema 1.
- Fit rules stay 4 and grammar 4: no statistic changed definition.
- The project file format is unchanged.

## Known limitations

- Physical 150 % display scaling: NOT RUN (simulated only).
- Clean-machine install: owner-approved deferral.
- Identical values imported from a different file count as different content. This is conservative by design.
