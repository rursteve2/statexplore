# StatExplore v1.3.6 — implementation and release report

**Proposal: GO for independent audit** (a developer proposal, not acceptance). The owner alone accepts v1.3.6 as
the stable baseline. Do not start v1.4.0 before that.

## 1. Baseline and provenance

**Baseline.** Tag `v1.3.5` = `b7610a0feb9b19f44fcd178a0eaf437eced72e9b`.

- The handoff's `statexplore-1.3.5-source.zip` (SHA-256 `20376a44…e765`, ZIP comment `b7610a0…`) is byte-identical to
  the tag's tree: 339 of 339 files.
- The embedded build ID `21f698a` differs from `b7610a0` only in docs and `tasks/` (no `src`, `tests` or packaging
  changes).
- The locally kept v1.3.5 `Setup.exe` and portable ZIP match the published SHA-256 values (`6138447…`, `0ac6cfd…`).
  The auditor could verify only the source.

**Candidate.**

- Branch `release/v1.3.6`.
- Code commits `c2308df` (the fixes and tests) and `090d120` (acceptance-runner correction, §6). The release build is
  from `090d120` (embedded `_build.txt` = "commit 090d120e691f", clean tree). Later commits are docs and developer
  tools only.

**Environment.**

| Item | Version |
|---|---|
| OS | Windows 10 |
| Python | 3.11.1 |
| pandas | 3.1.1 |
| numpy | 2.4.6 |
| pyarrow | 26.0.1 |
| PySide6 / Qt | as in v1.3.5 (unchanged) |
| R | 4.6.2 (2026-10-29 ucrt), bundled in the build |
| CPU / RAM | 4 cores; about 3–8 GB RAM free |

Package versions are in the acceptance `environment.json`.

## 2. Findings: baseline red → candidate green

The kind of evidence is given for every result:

| Kind | Meaning |
|---|---|
| real R | the application ran real R fits |
| script-R | a generated script was run in a fresh Rscript |
| Python | controlled Python / fake worker |
| owner R | the owner's RStudio run |

| ID | Sev | v1.3.5 (b7610a0) red evidence | v1.3.6 green evidence |
|---|---|---|---|
| AUD135-01 | P2 | **Owner R:** 9 / 20 (NUM-04..09, ID-01 FAIL). **script-R:** a `0`-prefixed participant id and 9 other edits accepted, exit 0 (`G0_red_tamper_v135.json`). | **script-R:** the 13 owner NUM / ID / SEL cases + 17 extra controls PASS on the generated verifier; CSV round trip; 23 single-field tampers each exit 1 naming the check; v1.3.6 rejects all 10 tampers (`G3_green_tamper_v136.json`). |
| AUD135-02 | P2 | **Owner R:** SUM-04..07 accepted. **script-R:** stability median / frequency / sign / key, estimand median, realized sizes, trace df / flags and draw ids: 10 of 10 accepted. | **script-R:** SUM-01..07 correct through the full generated script; complete 55L reproduction (189 checks, 0 FAIL); stability / estimands / realized sizes recomputed in R; tampers rejected. |
| AUD135-03 | P1 | **Python:** `adversarial.py` 13 pass / 4 fail (stale script / data in a reused folder; portable no-consent kept old data). | **Python + GUI:** a new folder per export, consent default No, registry / inventory negative controls; ported adversarial assertions pass; GUI Reports, code dialog and validation panel tests. |
| AUD135-04 | P2 | **real R / script-R:** a cancelled validation's export fails its own reproduction (`split manifest: same rows`, exit 1). | **script-R:** the same case exits 0 and reports "1 of 3 planned parts committed", not complete; editing the pending plan or promoting the ledger fails; controlled: lost worker / invalid response → uncommitted, counters a lower bound. |
| AUD135-05 | P2 | **real R:** cap = the 7 fits needed → `partial`, "remaining parts were not run", although the part was scored (`G0_budget_v135.json`). | **real R:** cap 7 → complete (used = cap, `at_cap`, not exhausted); cap 6 → partial with a budget part; both reproduce; controlled 0 / 1 / exact / exact − 1. |
| AUD135-06 | P1 | **Python:** workbook 9 / 7. **real R:** the sheet-B search admitted sheet A's 3 cached rows and reported A's p-values as B's. | **Python:** workbook 16 / 16. **real R:** 0 rows admitted, B's p-values equal a fresh B search; lm on A then B (and reversed) in one worker = each sheet's own `lm()` oracle. |
| AUD135-07 | P1 | **Python:** 6 / 2. **real R:** the seed-1 t-test bundle, exported after the seed change, reproduced mean 10.3 instead of the recorded 9.7. | **real R:** the bundle reproduces 9.7 (n 15, t, CI equal to an independent `t.test` on the seed-1 rows); seed-2 result separately bound; R Console replacement / no provenance → refused with the reason. |

Severities are as proposed by the owner; the evidence did not justify a change. AUD135-06 was reached at the API
level. No GUI path was found that puts sheet A's search next to sheet B's data in one project: a new import creates a
new project, and an R Console import changes the source hash. The P1 is kept because the wrong worksheet's fits were
reported.

**Red test files.** The new tests run against the v1.3.5 source (non-R subset): 26 failed, 1 passed (the fixture-hash
check) — `G0_red_v136_tests_on_v135.txt`. The v1.3.5 baseline suite itself: 747 passed, 1 skipped
(`G0_baseline_v135_full_suite.txt`), the same as the developer record.

## 3. Additional defects found during the work

| ID | Sev | What | Correction |
|---|---|---|---|
| V136-D1 | P2 | `write_portable` defaulted to including the data | explicit consent required; default No |
| V136-D2 | P2 | a budget refusal swallowed inside a model wrapper could leave a part "scored" over fewer candidates (source-confirmed; not reproduced at runtime) | a denial makes its part a budget stop |
| V136-D3 | P3 | R session keys used `id(ds.raw)` (address reuse) | content identity + CSV SHA-256 |
| V136-D4 | P3 | a holdout split recomputed differently after reopening (numpy change) would pass unnoticed | realized masks are part of the view identity |
| V136-D5 | P2 | v1.3.5 validation records would be re-exported runnable with the old verifier | results-only, with the reason |
| V136-D6 | P3 | introduced and caught during development: the Reports label widened the window's minimum width to 1580 px | word wrap; simulated-DPI test passes |

All are corrected and covered by named tests (register). No P0 or P1 is open.

## 4. Supplement: adopted, modified, replaced

| Supplement item | Disposition | Why |
|---|---|---|
| `staged_bundle` / `publish_export` | **Adopted and extended** (`export/bundle.py`) | Adds a central registry with a data choke point (`export_data_csv` refuses a no-consent stage), kinds, name-based data detection, scratch outside the bundle, readable `<name>_<timestamp>` folders instead of a uuid, and inventory with bundle kind. |
| `typed` / `imported_id` / `view_id` | **Modified** (`core/identity.py`) | Vectorized per-column hashing (binary64 bits, int64 datetimes, length-prefixed UTF-8, null masks) instead of per-row JSON. That makes 100k rows practical. Cached per raw object via a weak reference. View identity adds realized filter / holdout masks and derived values. |
| `capture_provenance` / `verified_view` / `verify_input_file` | **Adopted with changes** (`RSession.provenance`, `export/historical.resolve`) | Verification by re-creating the full worker CSV and comparing its SHA-256, first on the current view, then on the snapshot rebuild (which itself requires the content identity). |
| `ExecutionLedger` | **Replaced** by the ledger inside `ValidationRunner` | It works per part and chunk on the real R responses: returned parts validated, denial counters, uncommitted chunks. |
| R `sx_budget_take` denied flag | **Adopted** as a counter | Plus the per-part rule (a denial since the part began makes it a budget stop). |
| `sx_read_typed` with `v:` / `m:` wire encoding | **Not adopted** | Readable CSVs are kept (spreadsheet users). Typing comes from a declared schema embedded in the script, with `""` as the only missing token, text read verbatim and malformed tokens rejected. The supplement lists this as the preferable alternative. |
| `sx_parse_num`, `sx_num_equal`, `sx_text_equal`, `sx_exact_integer`, `sx_key2`, `sx_compare_table` | **Adopted with changes** | Integrated as `sx_parse_num` / `sx_num_ok` / `sx_int_ok` / `sx_text_ok` / `sx_key` / `sx_compare`, with per-field flags (`na`, `inf`, `need=`, `tol=`). The owner diagnostic's `sx_num_ok(a, b, abs, rel)` call form is kept. |
| `sx_stability2` / `sx_estimands2` | **Adopted with changes** | The pool comes from the setup's `sx_val_index` / `sx_val_keys` / `sx_val_labels`, labels from an embedded alias map, and a separate coefficient identity check. Semantics are identical to the Python aggregation (denominators, sign, type-7). |
| Supplement R tests | **Superseded** | They were never executed by the author. Equivalent and wider cases run in the generated verifier (§2); the 15 Python helper tests are not used as evidence. |

## 5. Test results

| Run | Result | Log |
|---|---|---|
| v1.3.5 baseline suite (b7610a0) | 747 passed, 1 skipped | `G0_baseline_v135_full_suite*` |
| v1.3.6 first full run | 812 passed, 6 failed, 1 skipped | `G3_full_suite_v136*` (failures preserved; see §6) |
| v1.3.6 final full run (c2308df) | **818 passed, 1 skipped, 0 failed** (38.6 min) | `G3_final_full_suite_v136*` |
| After the runner fix (090d120): non-R suite + acceptance-runner tests | 444 passed, 1 skipped; 2 passed | `G3_final_nonr_090d120*`, `G3_acceptance_tests*` |
| Frozen build (090d120) `--selftest --full` | OK (30 workspace checks + base check) | `G3_frozen_selftest_v136.txt` |
| Installed: v1.3.5 → v1.3.6 in-place upgrade, `--selftest --full` | OK | `G3_install_upgrade_v136.txt` |
| Acceptance kit, installed executable (090d120, after the upgrade) | **131 PASS, 0 FAIL, 0 BLOCKED** (6.7 min) | `acceptance/` |

The one skip is the screenshot evidence run (`STATEXPLORE_SHOTS`). It is not a functional test.

## 6. Intermediate failures and how they were resolved

The first full run had 6 failures:

1. **`test_v134.py::test_validation_manifest_identity_and_relocated_reproduction`** — a real gap in my schema.
   Fixed-candidate validation has no training score, but the new schema required `train_score` on scored parts. The
   field is now optional; missing still equals only missing. The tampers were re-verified.
2. **`test_v134.py::test_52d_resume_continues_same_search_to_15`** and
   **`test_v135.py::test_52d_historical_resume_is_refused_with_reason`** — the resume refusal named the new legacy
   identity before the inherited "fit rules 3" reason. The order was restored (the more specific inherited reason
   first).
3. **Three new trace-lock tests** — test-side mistakes (a missing `status` field; quote escaping in a source-string
   check).

After the first build (c2308df), reading the kit's own results showed a runner bug. Four checks passed their tolerance
text positionally into the `ok` slot, so they could not fail. Their actual values were all true. The options are now
keyword-only, a test covers this, and the kit was rebuilt (090d120) and rerun. The c2308df kit result (131 PASS) is
superseded; its ZIP was not kept (only the summary is quoted here).

During development:

- The simulated-DPI test caught V136-D6 (fixed).
- Fifteen inherited tests needed adaptation for intended changes (§7).
- The auditor scripts were ported, not edited. The originals are kept unchanged in the handoff.

## 7. Inherited tests changed (intended behaviour, assertions kept)

- **New export folders** (paths taken from the returned files): `test_export.py` O / xlsx / docx; `test_v13`, `v131`,
  `v132`, `v133`, `v134`, `v135` bundle and validation exports.
- **Explicit `include_data=True`** where a test needs the data file. The API default is now No (V136-D1).
- **`test_v135` audit test:** the v1.3.4 project's search is identity schema 2, so its export-time context is
  "unverified" (no longer "matches"). Audit schema is 3.
- **`test_v135` validation-export tests:** the legacy search is refused; a current-version search on the same data is
  runnable; 50E is refused.
- **`test_v135` NA-safe comparison:** ported from `sx_match` to `sx_compare` with the same assertions.
- **Check name:** "selection frequencies (recomputed)" → "stability (recomputed from the reproduced draws and the
  declared pool)".
- **`test_v134` cache test:** the "good" fixture run carries identity schema 3.

## 8. Migration and compatibility

A v1.3.5 project (partial and complete searches, a validation run, a t-test) was written by v1.3.5 and opened by
v1.3.6 (`G3_migration_v135_project_in_v136.json`):

- the searches are identity schema 2 and unverified: Resume refused with the reason, no cache rows, context
  "unverified";
- the historical rows are unchanged;
- the validation export is results-only, with the reason;
- the t-test export is `REPRODUCTION_UNAVAILABLE` plus a labelled non-runnable record and the xlsx;
- *New search from this one* gives schema 3, partial 2 / 3 → save → reopen → resumable → complete 3 / 3.

v1.3.4 projects (50, 52D) behave the same in the suite and in acceptance case 57.

Versions: identity schema 3, snapshot schema 3, audit schema 3, validation grammar `validation-4` with evidence schema
1, provenance schema 1. Fit rules 4 and grammar 4 are unchanged. The project file format is unchanged; old fields are
read as before.

## 9. Performance

See `benchmark_v1.3.6.md`. No workload changed by more than ±5 % against v1.3.5 (largest: +4 % on 52A reused, 0.77 →
0.80 s). The identity hash of 100k rows costs about 10 ms once per import, and about 0.6 ms per later call.

## 10. Windows, offline, installation

- **Frozen build** at `E:\statexplore_build\dist\StatExplore` (bundled R 4.6.2 and all packages): `--selftest --full`
  OK.
- **Installer / upgrade:** a silent per-user install of 1.3.5 (`/CURRENTUSER`, no admin), then 1.3.6 over it. Both
  self-tests OK; the embedded build ID is c2308df.
- **Offline:** offline by construction (`test_console_packages.py::test_O_offline_by_construction`). No code path opens
  a network connection, and the bundled R installs nothing. The network was not physically disabled on this machine.
- **Not run:** the clean-machine install (owner-approved deferral, standing approval 2026-11-28) and physical 150 %
  scaling (NOT RUN; simulated scaling tested; permitted by the master prompt).

## 11. Known limitations and remaining verification

- **Conservative identity:** identical values from a different file are different content.
- **Legacy searches and results** (≤ 1.3.5) cannot be resumed or reproduced as verified. They must be restarted, by
  design.
- **The bijection checks** verify the app's composite unit keys against R's grouping. The key text itself is
  application metadata.
- **A killed export** leaves a `.incomplete-` folder that has to be deleted by hand. There is no secure erasure.
- **Owner steps:** the GUI look (`manual_acceptance_v1.3.6.md`), the installed upgrade with the owner's own project,
  and physical DPI.
- **Not claimed:** no complete independent semantic audit of all 92 requirements is claimed by this report. The
  traceability gives the exact test nodes (`traceability_v1.3.6.csv`).

## 12. Deliverables

| Deliverable | Location |
|---|---|
| Source ZIP, Setup.exe, portable ZIP, acceptance kit, `SHA256SUMS-1.3.6.txt` | GitHub release v1.3.6 |
| Build manifest | `evidence/G3_build_manifest_v136.txt` |
| Defect register | `defect_register_v1.3.6.csv` |
| Traceability with exact test ids | `traceability_v1.3.6.csv` |
| Raw logs and JUnit | `evidence/` |
| Real-R red / green | `evidence/G0_*`, `evidence/G3_*` |
| Benchmark | `benchmark_v1.3.6.md` + raw JSON |
| Migration | §8 + JSON |
| Owner guide | `manual_acceptance_v1.3.6.md` |
| Acceptance kit and its evidence ZIP from the installed build | `acceptance/` |
| Methods | `methods_v1.3.6.md` |
| Release notes | `../RELEASE_NOTES_v1.3.6.md` |
