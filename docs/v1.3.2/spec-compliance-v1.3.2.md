# StatExplore 1.3.2 — compliance and evidence summary

Package: `StatExplore_v1.3.2_Developer_Package` (SHA256SUMS verified). Baseline: tag v1.3.1 =
`9153b96dc50f6678083753de43e207e091ab5dce` (matches the archive commit marker; the source archive itself was not
supplied, so its SHA-256 could not be checked).

## Gates (master prompt §4)

| Phase | Result | Evidence |
|---|---|---|
| 0 Baseline | v1.3.1 suite 373/373 pass on a clean worktree with real R 4.6.2; all target defects reproduced through the app API (STAT-17/18/21/22/44/45) | `E:\sx_v132_work\logs\baseline_v131_suite.txt`, `repro\before_v131.json` |
| 1 Identity | 35C (disjoint rows, equal fit) and 35A (reversed ordinal) → MISMATCH; same origin → verified; legacy → unverifiable; changed filter → rebuilt original or labelled new analysis | `tests/test_v132.py::test_identity_*`, `test_cd09_*` |
| 2 Construction | all 18 Gaussian structures (2/3 levels) match independent R references (logLik 1e-4, AIC/BIC 1e-3, k exact, BIC n exact) | `test_stat21_*`, `test_cd01_*`, `test_cd02_*` |
| 3 Diagnostics | near-boundary 37A/37B detected, 36C clean; 37D gradient warning kept once and makes the tests provisional (1 clean, 2 provisional); DIAG lme4 fits; direct R² | `test_diag_*` |
| 4 Clustered Discovery | CD-01…CD-12 automated (CD-12 display partly owner) | `test_cd*`, `tests/test_v132_gui.py` |
| 5 Reporting/UI | STAT-24…46 and UI items | register below |
| 6 Release | full suite 424 pass + 1 screenshot-only skip; 68/68 mandatory gates PASS; fixtures byte-identical to fresh generation; historical audit 353/353; frozen `--selftest --full` OK | `docs/v1.3.2/evidence/` |

Owner steps (NOT RUN by the developer, release notes say so): clean offline Windows VM install of the packaged
build; physical 150 % display. Developer proxies: frozen self-test on the build machine; Qt scale-factor 1.0/1.5
screenshots at 1366×768, 1920×1080, 2560×1440 (`E:\sx_v132_work\screenshots`).

## Files

- `defect_register_v1.3.2.csv` — every register ID with disposition, change location and tests.
- `compatibility_matrix_v1.3.2.csv` — package matrix plus the actual 1.3.2 status of every row.
- `evidence/gates.csv`, `evidence/summary.json` — runner output (`tools/sx_regress.py`, run 20261125_030050).
- `evidence/historical_reference_audit.csv` — independent references vs historical targets.
- `simulation/` — predeclared small simulation (seed 132).
- `manual_acceptance_v1.3.2.md` — owner checklist (10 steps).

## Independent reference discrepancies

None against historical targets (353 PASS, 101 documentary). One independent reference fit fails in R itself
(CD_Irregular two-level RS + AR(1): nlminb convergence error, then optim "Coefficient matrix not invertible");
the application reports the same failure and fits no substitute.

## Performance

Cross-sectional search, 155 candidates, 2,000 rows, 1 worker: v1.3.1 3.45 s, v1.3.2 4.11 s median of 3 (+19 %,
≈ 4 ms per candidate for identity fingerprints and diagnostics; under the 20 % investigation trigger after
optimization from an initial +268 %). Single longitudinal fit: 0.57 → 0.72 s.
