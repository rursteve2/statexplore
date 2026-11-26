# StatExplore 1.3.3 — evidence report and release decision

Package: `StatExplore_v1.3.3` (SHA256SUMS verified). Baseline: tag v1.3.2 = `ff7b4f3`. Environment: Windows 10 19045,
AMD Ryzen 7 5800X, Python 3.11.1, PySide6 (offscreen Qt for automated GUI tests), R 4.6.2 ucrt with the project library
(`evidence/R_sessionInfo.txt`; versions in `r/packages.lock.json`).

## Gate summary (developer gates of 05_Acceptance_Checklist.md)

| Gate | Evidence | Result |
|---|---|---|
| G0 baseline | v1.3.2 suite on a clean worktree: 420 pass + 1 skip, 4 failures caused by Windows memory exhaustion during a concurrent run ("paging file is too small"), all 4 pass when rerun → **424 pass + 1 skip**. P0s reproduced on v1.3.2 through the real Qt + R application: `evidence/gate0_repro_v132_baseline.txt`, `gate0_repro_48c_v132_baseline.txt` | PASS |
| G1 STAT-47 family | `test_stat47_transfer_refit_plot_and_reml_keep_the_model` (real GUI: transfer → Fit ×2 → plot change → ML→REML vs independent REML reference → return to origin), custom subset + hierarchy refusal vs a hand-written lme4 fit, spec diff, fail-closed injection; after-fix reproduction `evidence/gate0_repro_v133.txt` | PASS |
| G2 matrix | 18 configurations × workflows A and B + 18 standalone Full Analysis fits vs the independent references: `compatibility_matrix_v1.3.3.csv` (36 / 36 obligations PASS) | PASS |
| G2 composite IDs, time comparison | 45A 160 / 0, global refusal, cluster-free composite; 48C AR(1) linear / quadratic / categorical equal the reference (k 8 / 10 / 16) with the fitted structure read back; independent control separate | PASS |
| G2 families, diagnostics | GLMM / CLMM likelihoods vs independent lme4 / ordinal; optimizer fallback injection (same spec, flagged, not clean), structural error not retried, all-fail; LR −63.968673 withheld, −2.094e−7 clamped with raw kept | PASS |
| G3 validation | splits, independent prediction oracle (lme4, GEE) on the same manifest (|Δ| < 1e−6), +10,000 test-outcome leakage adversary, pipeline bootstrap, cancellation, persistence, export, GUI panel; methods `validation_methods.md` | PASS |
| G3 Test 51 | independent enumeration oracle, `p.adjust` per family, log10 p, cancel → resume exactly once | PASS |
| G4 Test 52 | Full Analysis and AMD bundles run unmodified in a clean `Rscript --vanilla` after relocation to paths with spaces / non-ASCII, corrupted data fails closed, manifest fields populated; save / reopen of searches with validation runs; historical reconstruction (960 verified vs new 800, 3-category reference change) | PASS (physical 150 % DPI: owner) |
| G4 Test 53 | `benchmark_v1.3.3.md`: no throughput regression; cancellation 368.7 s → 2.3 s; no orphan R processes | PASS (100k-row tier not run) |
| G5 Test 54 | full suite on Windows with real Qt + R: **492 pass, 1 skip** (screenshot-only evidence test), 0 fail; runner `evidence/gates.csv`: **97 PASS, 0 FAIL, 0 BLOCKED**, 5 not applicable (wording-only, inherited), 3 owner-only; frozen build `--selftest --full` OK (below) | PASS except owner steps |

Reference reproducibility: the four package reference scripts re-executed by the runner reproduce the committed
reference output 1994 / 1994 (`evidence/133_reference_reproducibility.csv`); committed fixtures are byte-identical to
the package; v1.3.2 historical targets 353 PASS / 101 documentary; 07 anchors: 155 / 166 equal the fresh references,
the other 11 (BH, Holm, CI, ΔR², exclusions, variance SDs) are application outputs checked by
`test_07_application_only_anchors` (`evidence/historical_targets_audit.csv`).

## Build

Built with `packaging/build.py` (STATEXPLORE_BUILD_DIR on E:) from commit `df249322ad24` (recorded in the frozen
build's `_internal/statexplore/_build.txt`; later commits only add this section). Frozen `StatExplore.exe --selftest
--full`: all checks OK, exit 0 (fits of every engine, clustered RS + AR(1), discovery A / B / clustered B, transfer
identity verified, 68 export files, project save / reopen, R console). New modules present in the frozen archive
(`discovery.validate`, `gui.widgets.validation_panel`, `gui.widgets.terms_editor`).

| File | SHA-256 |
|---|---|
| StatExplore-1.3.3-Setup.exe | `240966342ad70fda29553d1bb0dacc33ddc052bb3a98f64189f17ed4d3598414` |
| StatExplore-1.3.3-portable.zip | `a1c381f600bcfa32537685afdfe85e01a3df45b80fe546faf807129a2674f842` |
| statexplore-1.3.3-source.zip (`git archive` of df249322) | `80363c07cf741a4bb1a26498c0f4ed628de79422345a185765ee41c07a0d298b` |

Reproducible build: `py -3.11 -m venv .venv`, `pip install -e ".[dev]"` + `pyinstaller`, R 4.6.2 with
`r\install_packages.R` (versions in `r/packages.lock.json`), Inno Setup 6, then `set STATEXPLORE_BUILD_DIR=E:\statexplore_build`
and `.venv\Scripts\python packaging\build.py` (docs/DEVELOPER.md).

## Issue dispositions

Machine-readable: `disposition_v1.3.3.csv` (all 92 register IDs, original IDs kept; root cause, code location, gate
status and tests of this run). Counts by priority × gate: P0 6 PASS; P1 26 PASS + 1 owner step (V133-QA-03 clean
offline machine); P2 49 PASS + 2 not applicable (UI-39, UI-52: wording, owner check); P3 5 PASS + 3 not applicable
(UI-40, UI-53, UI-55: wording, owner check). No P0 / P1 item is open in developer scope.

Root causes found (new in 1.3.3): STAT-47 family — the Full Analysis form rebuilt every refit from widget defaults; the
time comparison inherited the same defect (interaction dropped after a transfer); the reported AR(1) → independent lme4
substitution did not reproduce on ff7b4f3 in any tested route (manual form; workflow A and B transfers on the 46A
irregular-time fixture) and is now prevented by the fail-closed read-back; preflight grouped on the raw ID; AMD
manifests lacked provenance because the candidate dialog was created without a result; the shared diagnostics helper
declared `ordinal` for every model; cancellation waited for running batches (also in 1.3.2: 368.7 s).

## Deviations and residual limitations (disclosed)

- Validation predictions for GLMM / CLMM / multinomial-mixed / mmrm searches are refused with the reason (no correct
  marginal prediction implemented); no external validation set; no inner-loop tuning.
- Benchmark large tier (100k rows, 200 schools) not run; medium tier covers batching, memory and cancellation.
- Owner-only: clean offline Windows machine install / uninstall (Test 54), physical 150 % display scaling — steps M1 and
  M11 of `manual_acceptance_v1.3.3.md`.
- The v1.3.2 package's single failing reference fit (CD_Irregular two-level RS + AR(1), both optimizers) is unchanged
  and is excluded from tests as before.

## Decision

Developer gates: all mandatory developer-executable gates pass. **GO for release as 1.3.3**, with the owner steps M1
(clean offline install) and M11 (physical 150 % DPI) still to be performed; any P0 / P1 mismatch found there stops
acceptance. This is a teaching / exploratory tool release, not a production or research certification.
