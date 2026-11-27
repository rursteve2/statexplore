# Benchmark: v1.3.5 vs v1.3.4

Both versions were benchmarked on the same machine in the same session.

**Method**

- **Tools:** `tools/sx_bench134.py` runs the owner workloads. `tools/sx_bench135.py` adds 51H3 and 60F.
- **Repeats:** 3 per workload; the first run is cold.
- **Baseline:** v1.3.4, run from the clean worktree with `PYTHONPATH=E:\sx_v134_base\src` (dd7313a).
- **Fixtures:** the handoff files.

**Machine:** the same one as the v1.3.4 report.

- Windows 10, Python 3.11, R 4.6.2.
- 4 R workers; validation uses 1.
- About 3–8 GB of RAM free.

**Peak memory** is the working set of Python plus every Rscript.exe, sampled with `tasklist`, so it is approximate.

**Raw output** is in `evidence/`: `bench_v134_owner.json`, `bench_v135_owner.json`, `bench_v134_extra.json` and
`bench_v135_extra.json`.

## Results

| Workload | v1.3.4 median s [range] | v1.3.5 median s [range] | Δ | Peak MB 1.3.4 / 1.3.5 |
|---|---|---|---|---|
| 52A: 15 pairwise, fresh | 4.11 [4.03–4.22] | 3.99 [3.99–3.99] | −3 % | 540 / 540 |
| 52A: same 15, reused | 0.85 [0.82–0.87] | 0.84 [0.83–0.84] | −1 % | 372 / 373 |
| 53A: 435 candidates | 15.16 [15.03–15.97] | 14.83 [14.24–14.90] | −2 % | 1113 / 1116 |
| 53B: count + preview of 147,713,444 (no fits) | 0.76 | 0.77 | — | 90 / 92 |
| 53C: budget 100 | 6.57 [6.54–6.74] | 6.68 [6.56–6.73] | +1.6 % | 903 / 906 |
| 53C: cancel 3 s after start (latency) | 1.33–1.75 | 1.46–1.78 | — | — |
| 54B: 100k rows, 15 candidates | 53.37 [52.35–53.75] | 53.48 [53.08–53.83] | +0.2 % | 975 / 982 |
| 51H3: bootstrap B = 20, 18 candidates (pipeline) | 72.44 [72.22–75.06] | 73.26 [71.30–73.55] | +1.1 % | 418 / 406 |
| 51H3: setup | 0.98–1.05 | 0.97–1.09 | — | — |
| 60F: fixed-candidate setup at #0 / middle / last of 147,713,444 | n/a (no saved-spec retrieval) | 0.74 / 0.74 / 0.74 | — | — |

The 53C cancellation runs left 0 Rscript processes and 0 recorded rows in both versions.

## Reading

**Counts.** Attempted, reused and BH-significant counts are identical in both versions. In 51H3, both versions select
#17 in 20 / 20 draws.

**Regressions.** No workload changed by more than 20 % in time or memory. The fit-budget instrumentation, cancellation
checks and compatibility gate cost nothing measurable.

**51H3 fit budget.** Fitting invocations: 680 used, 660 cache hits, 0 retries. The conservative bound was 1,340, which
equals used + hits.

**60F (fixed-candidate setup).** Setup takes constant time at the first, middle and last index of the
147-million-candidate space (0.74 s, dominated by building the unit frame).

- v1.3.4 resolved a fixed candidate inside `ValidationRunner` by enumerating from the requested index to the end of the
  space (`G0_reproductions_v134.jsonl`: 8,478 candidates visited for #0).
- On v1.3.4 the 60F script measures only `feasibility`. That function never enumerated, so its times there are not the
  setup cost.
- Late indices of the 147M space were deliberately not run on v1.3.4 (unbounded).

**Comparison with the v1.3.4 report.** Absolute times are 10–20 % higher than in `docs/v1.3.4/benchmark_v1.3.4.md` for
both versions, because the machine was more heavily loaded in this session. The comparison uses only same-session pairs.
