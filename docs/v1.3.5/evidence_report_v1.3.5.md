# StatExplore v1.3.5 — evidence report and release decision

**Inputs**

- **Baseline:** v1.3.4 = `dd7313aebfaf8991e4b921feb42a3a37cf355fe6`. The tag was verified, and the source archive
  SHA-256 `860b62cf…de56` matches the prompt (`evidence/G0_source_archive_sha256.txt`).
- **Branch:** `release/v1.3.5`. The code commit is `21f698a`, which the build embeds (`_build.txt`).
- **Handoff:** `StatExplore_v1.3.4_Developer_Handoff`, reused. Its 50 files are hashed in
  `evidence/G0_handoff_SHA256.txt`.
- **New fixtures** (`tests/fixtures/v135`, with SHA256SUMS):
  - 53A, copied byte-identical from the handoff;
  - 55L, a documented synthetic single-level underflow fixture, with its generator `make_55l.py`.

## Gates

| Gate | Result | Evidence |
|---|---|---|
| G0 baseline suite on dd7313a | 537 passed, 1 skipped (screenshot-only), 0 failed. Executed; this matches the reported baseline. | `G0_baseline_v134_full_suite.txt` + JUnit |
| G0 reproductions on dd7313a | 9 findings reproduced in Python (02–10), 01 in real R, 11 by source | `G0_reproductions_v134.jsonl`, `G0_repro_g0.py`, `G0_red_v135_tests_on_v134.txt` |
| G0 red tests | `tests/test_v135.py` on dd7313a: **71 failed**, 139 passed (the passes are unchanged strategies, see below) | `G0_red_v135_tests_on_v134*` |
| G1 + G2 implementation | 01–11 + 3 discovered issues | commit `21f698a` |
| G3 full suite (inherited + new) | **747 passed, 1 skipped, 0 failed** (27 min 56 s, real R, offscreen Qt) | `G3_final_full_suite_v135.txt` + JUnit |
| G3 benchmarks | no regression > 20 % (largest +1.6 %) | `benchmark_v1.3.5.md` |
| G3 build + frozen self-test | Setup.exe + portable zip built with bundled R 4.6.2; `StatExplore.exe --selftest --full` **OK** | `G3_build_v135.txt`, `G3_frozen_selftest_v135.txt` |
| G4 register | 11/11 audit IDs PASS; 3 discovered PASS; 17 v1.3.4 issue groups PASS; 92 Appendix-A requirements PASS (suite) | `defect_register_v1.3.5.csv`, `evidence/gates_v1.3.5.csv` |

**The 139 tests that pass on v1.3.4** are expected to pass there:

- brute-force grammar oracle cases for strategies whose grammar did not change (they guard against side effects);
- the large-count preservation test;
- the existing GUI log-twin sort;
- the changed-data refusal;
- the unknown-magnitude label;
- the fixture manifest.

**An intermediate full run had 12 false failures.** My first full run had 12 `ModuleNotFoundError: tests` errors. The
helper runner called `pytest.main()` without the repository root on `sys.path`. Separately, one updated GUI test was
genuinely wrong: its fake rows were incomplete. Both were fixed, and the final run above is clean
(`G3_full_suite_v135.txt` keeps the intermediate run; the lesson is recorded in `tasks/lessons.md`).

## Traceability matrix

| ID | Sev. | Baseline (dd7313a), actually executed | Correction | Tests (all PASS) | Disposition |
|---|---|---|---|---|---|
| AUD134-01 | P1 | 55L, real R: v1.3.4 stability ranked x1:x3 (#0, the weaker) first in 5 / 5 resamples (`ranked_first` #1 = 0). Both p = 0. | shared `sx_primary_logp` / `sx_select_min` in legacy stability, both procedure-CV paths and AMD | `test_shared_selector_*`, `test_55l_legacy_stability_*`, `test_55l_legacy_procedure_cv_*`, `test_hard_fit_budget_*` (AMD selects #1) | confirmed, fixed / tested |
| AUD134-02 | P1 | raw order [1, 0], adjusted order [0, 1] | log keys for every p column; adjusted ranking within family | `test_adjusted_ranking_*` (both corrections, both input orders), `test_gui_table_sorts_*`, v1.3.4 51C −419.54 / −485.20 targets | confirmed, fixed / tested |
| AUD134-03 | P1 | 52D (grammar 2, fit rules 3, 5 / 15) `resumable = True` | compatibility gate, all-worker version check, refusal without mutation, linked restart | 11-case matrix, 52D refusal, changed data / coding, GEE fit rules 3, same-version 5 + 10 → 15 (= fresh exhaustive), 3-process GUI restart + resume | confirmed, fixed / tested |
| AUD134-04 | P1 | focal-only 3 (expected 0), subsets 6 (expected 3) | grammar 4 | audit example, 154-case brute-force oracle over 5 strategies, 53A 435 / 53B 147,713,444, 147,539,008, 174,436 preserved, grammar detection, GUI disabled for B | confirmed, fixed / tested |
| AUD134-05 | P1 | historical #0 `Time:X01` validated as `Time:X01 + X02` | saved-spec retrieval verified by key; R term check; pipeline refusal | historical grammar example, refusals (missing row / key, other spec, reorder), pool key mismatch | confirmed, fixed / tested |
| AUD134-06 | P1 | mode D 5 + 10 → pool 5 | scopes cumulative (default) / original / prefix; adaptive and truncation disclosure | scope tests incl. > total, time-limited, time-only, mode C; v1.3.4 leakage tests | confirmed, fixed / tested |
| AUD134-07 | P1 | 50 + 50E: audit top-level SHA-256 = 50E (current); snapshot = 43B | audit schema 2 (historical vs current_context); `bundle_data` refusal | replacement, filter (rebuilt view), wrong view, schema-1 snapshot unknown, tail round trip | confirmed, fixed / tested |
| AUD134-08 | P1 | 8,478-candidate space, #0 → 8,478 visited | no enumeration (saved spec); bounded block skipping | zero enumeration calls at #0 / middle / last of 147,713,444; 60F timing | confirmed, fixed / tested |
| AUD134-09 | P2 | pre-cancel, all cached → 10 rows committed, status complete | checks before every batch / cached commit; injectable clock; completion race | 7 deterministic fake-clock / fake-worker tests incl. 147M budget 100 and cancel at about 50,000 | confirmed, fixed / tested |
| AUD134-10 | P2 | 8,478 × 6 parts: estimate 203,478 vs plan 289,062; cap 250,000 accepted | conservative bound; hard R budget; budget parts never scored | bound ≥ brute-force plan; cap boundaries 0 / exact / −1; nlme retry factor; R: used + hits = plan, cap 10 enforced | confirmed, fixed / tested |
| AUD134-11 | P2 | source: coefficients unchecked, missing CSVs skipped, NA compare | full verification, embedded expectations, check CSV, exit 1 | 55L relocated (space + Unicode) fresh Rscript; 10 tamper cases each fail (prediction, coefficient, selection, status, duplicate key, extra draw, trace, metric, missing file, data SHA-256); NA-safe unit test; 51C pipeline bootstrap reproduced | confirmed (source), fixed / tested |
| V135-D1 | P2 | teaching examples ordered by ordinary p | log-p ordering | suite | fixed |
| V135-D2 | P2 | non-Gaussian nested structure tests had no log_p | stored | suite | fixed |
| V135-D3 | P2 | progress count on ordinary p | log-scale `_qualifies` | suite | fixed |

**Preserved distinctions**

- **47C** stays PARTIAL, as in v1.3.4: the exact historical settings are not in the handoff.
- **52D in v1.3.4:** the original failure was not reproduced in two source processes. v1.3.5 now refuses 52D by design,
  which is a safe refusal, not a regression.
- **51B** remains reconstructed.
- **Reconstructed fixtures** are not claimed to be byte-for-byte equivalent.

**Source-process vs installed-executable evidence.** All tests above ran in the source process (with offscreen GUI). The
installed executable is verified only by its frozen full self-test.

## Environment-dependent items

| Item | Status |
|---|---|
| Physical display scaling | **NOT RUN.** No VM; only the v1.3.4 simulated-scaling evidence applies. |
| Clean-machine offline installation | **Owner-approved deferral**, given explicitly by the owner in the v1.3.5 session on 2026-11-27. |
| In-place upgrade of the installed app on this machine | **NOT RUN by the developer.** Not done on the owner's working installation; part of owner test 57 / 61 in `manual_acceptance_v1.3.5.md`. Saved-project compatibility is covered in the source process: the 52D and 50 projects, v1.0 project, and backward-compatibility tests. |

## Release decision (proposed)

**GO for v1.3.5.**

- Every P1 and P2 finding is fixed and tested.
- No mandatory correctness gate failed.
- The only open items are the environment-dependent ones above: one owner-approved deferral and two NOT RUN items that
  do not affect statistical correctness.

The owner accepts the stable baseline after tests 55–61. v1.4.0 development does not start before that acceptance.
