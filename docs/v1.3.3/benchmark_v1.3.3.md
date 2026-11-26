# Benchmark (Test 53) — v1.3.3 vs v1.3.2

Tool: `tools/sx_bench.py` (developer-only), 3 repeats per workload, recorded seeds, declared candidate spaces; no time
limits are asserted. Hardware: AMD Ryzen 7 5800X (8 cores / 16 threads), 32 GB RAM (≈ 3–4 GB free during the runs:
browser and emulator processes were running), Windows 10 19045, Python 3.11.1, R 4.6.2. Discovery used 4 R workers.
Peak memory = working set of the Python process plus every Rscript.exe, sampled with `tasklist` every 0.25 s
(approximate). Raw output: `E:\sx_v133_bench\baseline`, `E:\sx_v133_bench\current2` (bench.csv, bench_summary.json).

| Workload (rows, candidates) | v1.3.2 wall s (3 runs) | v1.3.3 wall s (3 runs) | first result s (1.3.2 / 1.3.3) | peak MB (1.3.2 / 1.3.3) |
|---|---|---|---|---|
| 43B, workflow A, RS + AR(1), Time × each (960, 1) | 2.09 / 2.00 / 2.01 | 1.91 / 1.90 / 1.92 | 0.95 / 0.87 | 427 / 382 |
| 43B, workflow B, 9 structures (960, 9) | 4.00 / 4.02 / 3.99 | 3.72 / 3.79 / 3.75 | 0.94 / 0.86 | 568 / 560 |
| synthetic 16k (2,000 participants × 8, 100 schools), A, RI, main effects, cap 3 (16,000, 175) | 27.5 / 29.0 / 26.9 | 24.3 / 24.4 / 24.2 | 1.79 / 1.78 | 1530 / 1291 |
| synthetic 16k, A, AR(1), Time × each, 10 predictors (16,000, 10) | 34.4 / 34.6 / 34.6 | 34.6 / 34.6 / 34.4 | 1.82 / 1.73 | 549 / 549 |
| 43B validation: new clusters, holdout + 5-fold CV + 20 bootstrap draws, pipeline (960, 2) | n/a (no AMD validation) | 7.16 (cold) / 5.49 / 5.42 | — | — / 386 |

Synthetic data: `synthetic(seed = 5301)` in the tool (planted X1 × TIME and X2 effects).

## Cancellation and cleanup

Medium search (16k rows, RS + AR(1), main effects up to 4 of 10 predictors = 385 candidates, 4 workers, batches of
40) cancelled 3 s after the start:

| | v1.3.2 | v1.3.3 |
|---|---|---|
| time from cancel to return | **368.7 s** (waited for every running 40-candidate batch) | **2.3 s** (running batches' R workers stopped) |
| status / evaluated | cancelled / 160 of 385 | cancelled / 0 of 385 (the interrupted batches stay pending) |
| Rscript processes after cancel | 0 | 0 |

The interrupted candidates are never recorded or cached as results; a resume evaluates exactly the missing candidates
with the same IDs (`tests/test_v133.py::test_51_cancel_kills_running_batches_and_resume_completes_exactly_once`).

## Reading

No throughput regression: v1.3.3 is equal or faster on every common workload (differences of a few percent are within
the run-to-run noise of a machine with other load). The new per-fit read-back of the fitted object (J4) is only done in
Full Analysis and its comparisons, not inside discovery batches. Not measured here: the 100k-row tier and 200-school
designs (the medium tier already exercises batching, memory and cancellation; a large tier is left to the owner if
needed).
