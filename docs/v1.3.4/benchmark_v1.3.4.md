# Benchmark — v1.3.4 vs v1.3.3 (same machine, same session)

**Tool:** `tools/sx_bench134.py` (the owner workloads of master prompt §7), 3 repeats per workload, the first one
cold. The baseline is run from the clean v1.3.3 worktree (`PYTHONPATH=<v1.3.3 src>`).

**Fixtures:** the owner's handoff files (51F Null_13, 53A, 54B).

**Machine and settings:**

- AMD Ryzen 7 5800X, 32 GB RAM, with about 3–6 GB free (other applications running)
- Windows 10, Python 3.11, R 4.6.2
- 4 R workers

Peak memory is the working set of Python plus every Rscript.exe, sampled with `tasklist` every 0.25 s
(approximate). Raw output: `evidence/bench_v133.json`, `evidence/bench_v134.json`.

| Workload | Owner v1.3.3 (other machine) | v1.3.3 median s [range] | v1.3.4 median s [range] | Δ | peak MB 1.3.3 / 1.3.4 |
|---|---|---|---|---|---|
| 52A: 15 pairwise candidates, n 960, fresh | 2.9 | 3.34 [3.30–3.37] | 3.42 [3.34–3.44] | +2.4 % | 542 / 543 |
| 52A: the same 15 reused from the previous search | 0.4 | 0.87 [0.85–0.87] | 0.88 [0.87–0.96] | +1 % | — |
| 53A: 30 predictors, 435 candidates (11 BH) | 13.7 | 13.56 [12.63–14.09] | 13.85 [13.75–14.22] | +2.1 % | 1118 / 1121 |
| 53B: count + preview of 147,713,444 candidates (no fits) | fast | 0.76 | 0.80 | — | 94 / 96 |
| 53C: same space, budget 100 (100 attempted) | 4.9 | 6.16 [6.13–6.17] | 6.23 [6.21–6.40] | +1.1 % | 904 / 908 |
| 53C: cancel 3 s after start, latency | not isolated | 1.40–2.07 | 1.82–1.96 | — | 0 Rscript left in both |
| 54B: 100,000 rows, 10,000 participants, 100 schools, 15 candidates (2 BH) | 43.7 | 47.23 [46.94–47.75] | 49.92 [48.78–50.32] | +5.7 % | 922 / 958 |
| 54B: CSV load | — | 1.54 | 1.43 | — | — |

**Result counts.** Every workload produced identical attempted, reused and BH-significant counts in both versions.

**Reading.**

- No common workload regressed by more than 20 % in time or memory.
- The small increase comes from the new per-fit content key (an MD5 of each used column, hashed once per sample) and
  the per-fit warning capture.
- The cancelled 53C runs recorded no interrupted candidate. They left 0 Rscript processes.

**Correctness-driven slowdown (not a regression in the §7 sense).** The 51H3 bootstrap (B = 20, 18 candidates)
took 48.5 s on v1.3.3 and 79.2 s on v1.3.4. v1.3.3 was faster only because it reused draw 1's reduced fits for draws
2–10 and 12–20: those were the wrong fits (V134-001). v1.3.4 fits every candidate and reference on its own draw:
20 × 18 full fits + one baseline and the needed references per draw, as the owner's independent oracle does. No
safe way exists to share fits across independent resamples. This is reported for owner review, not hidden.

**Not measured.**

- UI latency in milliseconds: the GUI was not instrumented. The owner reported the GUI responsive during 53B/C on
  v1.3.3.
- 1 vs 4 workers for the owner workloads: discovery results are keyed by candidate and identical by construction.
  Validation runs in one worker.
