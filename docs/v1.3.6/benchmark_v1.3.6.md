# Benchmark: v1.3.6 vs v1.3.5 vs v1.3.4

All three versions were benchmarked on the same machine in one session, one after another on an otherwise idle
machine.

**Method**

- **Tools:**
  - `tools/sx_bench134.py`: the owner workloads 52A, 53A, 53B, 53C, 54B;
  - `tools/sx_bench135.py`: 51H3 and 60F;
  - `tools/sx_bench136.py`: FP (identity fingerprints) and VAL (55L AMD validation with export).
- **Repeats:** 3 per workload; the first run is cold.
- **Workers:** 4 R workers for searches; 1 for validation.
- **Baselines:** clean worktrees with `PYTHONPATH` set to their `src`:
  - v1.3.4 = `dd7313a`
  - v1.3.5 = `b7610a0`
  - v1.3.6 = `090d120` (source)
- **Fixtures:** the acceptance kit's (SHA-256 in `fixtures_SHA256SUMS.txt`).

**Machine**

- Windows 10, 4 cores, Python 3.11, R 4.6.2.
- About 3–8 GB RAM free.

**Peak memory** is the working set of Python plus every Rscript.exe, sampled with `tasklist`, so it is approximate.

**Raw output** is in `evidence/bench_<version>_<tool>.json`.

## Results

| Workload | v1.3.4 median s [range] | v1.3.5 | v1.3.6 | Δ 1.3.6 vs 1.3.5 | Δ vs 1.3.4 | Peak MB 4 / 5 / 6 |
|---|---|---|---|---|---|---|
| 52A: 15 pairwise, fresh | 3.07 | 3.03 | 2.96 | −2 % | −4 % | — |
| 52A: same 15, reused | 0.77 | 0.77 | 0.80 | +4 % | +4 % | — |
| 53A: 435 candidates | 13.05 [12.90–13.85] | 12.69 [12.66–12.69] | 12.44 [12.32–12.53] | −2 % | −5 % | 1115 / 1117 / 1113 |
| 53B: count + preview of 147,713,444 (no fits) | 0.67 | 0.69 | 0.65 | −5 % | −3 % | 91 / 90 / 92 |
| 53C: model budget 100 | 5.50 [5.41–5.50] | 5.41 [5.38–6.51] | 5.35 [5.32–6.47] | −1 % | −3 % | 905 / 906 / 906 |
| 53C: cancel 3 s after start (latency) | 1.52–1.55 | 1.59–1.64 | 1.54–1.61 | — | — | — |
| 54B: 100k rows, 15 candidates | 45.17 [45.09–45.98] | 45.60 [44.60–46.17] | 45.44 [45.26–45.66] | −0.4 % | +0.6 % | 1003 / 997 / 974 |
| 51H3: bootstrap B = 20, 18 candidates (pipeline) | 63.21 [62.90–64.11] | 63.34 [63.30–64.38] | 63.20 [63.17–64.04] | −0.2 % | 0 % | 435 / 439 / 435 |
| 60F: fixed-candidate setup on 147,713,444 | 0.69–0.73 (no saved-spec retrieval) | 0.67 | 0.67 | 0 % | — | 108–109 |
| FP: first signature + 1,000 signature / fit-identity calls + save / reopen (100k rows) | 0.79 | 0.81 | 0.79 | −4 % | −1 % | 181 / 181 / 180 |
| VAL: 55L validation (holdout + CV2 + B3) | 2.22 | 2.27 | 2.20 | −3 % | −1 % | 278 / 282 / 282 |
| VAL: export of that validation | 0.28 | 0.31 | 0.30 | −3 % | +7 % | — |

The 53C cancellations left 0 Rscript processes and 0 recorded rows in all versions. Attempted, reused and BH-significant
counts are identical across versions. 54B: 15 candidates, 2 BH-significant. 51H3: #17 is selected in 20 / 20 draws.

## Identity fingerprint overhead (AUD135-06)

The content identity of the 54B file (100,000 rows × 10 numeric columns) was measured directly
(`evidence/bench_v136_fingerprint_direct.txt`):

| Measurement | Time |
|---|---|
| Cold value digest | 9.8 ms |
| Cold full data signature | 6.8 ms |
| Cached signature | 0.62 ms per call |

In the installed acceptance run, case 61 measured 0.01 s cold and 0.7 ms warm.

**Disclosure.** The "cold" field inside the FP workload reads 0.000 s in all versions. For v1.3.6 that cannot be the
first hash of 100k rows, so that field is not used here: the direct measurement above is reported instead. The FP
totals (1,000 warm calls + save / reopen) are comparable across versions.

**Signatures stay stable after save / reopen in all versions.** In v1.3.6 this is a verified content identity across
the Parquet round trip.

## Reading

**No regression above 20 %.** No workload changed by more than ±5 % against v1.3.5. The largest change is +4 % on
52A reused (0.77 → 0.80 s, within run-to-run variation at that scale).

**Cost of the corrective changes.** The content hashing is computed once per imported revision; it is never repeated
per candidate. The typed evidence writer and the transactional export (VAL export 0.30 s) have no measurable effect on
fitting workloads.

**What is not measured:**

- the installed-executable runtime under the GUI (the acceptance kit records the installed-build timings for 55, 61
  and the cancellation latency in its `results.csv`);
- memory beyond the sampled working sets.
