# StatExplore 1.3.2 — owner manual check (about 30 minutes)

Developer automation already covers the numbers (see `tools/sx_regress.py` evidence and
`docs/v1.3.2/spec-compliance-v1.3.2.md`). These checks cover what automation cannot: the packaged build on a clean
offline machine, a real 150 % display, and how the warnings read. Fixtures: `tests/fixtures/v132/*.csv` (synthetic;
SHA-256 in `SHA256SUMS.txt`). Record PASS / FAIL / BLOCKED / NOT RUN for each step.

| Step | Do | Expect |
|---|---|---|
| 1 Offline install (M13) | Network off. Install `StatExplore-1.3.2-Setup.exe` (and unzip the portable build elsewhere). Start both; Help ▸ About shows 1.3.2. Run `StatExplore.exe --selftest --full`. | Starts without internet; self-test exit code 0 (report in `%TEMP%\statexplore_selftest.txt`). |
| 2 Wrong-origin guard (M01) | Import `T35C_Identity.csv`. Filter FILTER = A. Discovery: outcome Y, candidate X, main effects; run. Change the filter to B. Open full analysis of the candidate. | A dialog names the filter change and offers "Refit the original" or "New analysis". Original → *identity check: verified*. New analysis → labelled new analysis, never verified. |
| 3 Clustered standalone (M06) | Import `T38.csv`. Mixed & Longitudinal: Y, ID, TIME, cluster SCHOOL, predictor GROUP, interactions none, correlated slope, AR(1), REML. Fit; repeat with CAR(1). | n 960, 160 participants, 20 clusters, 9 parameters, logLik −1652.159; the covariance table shows school variance, slope variance, covariance and correlation. CAR(1): apVar warning, not "singular". |
| 4 Workflow B (M08/M09) | Discovery: Y, longitudinal, ID, TIME (fixed), cluster SCHOOL, workflow B, no candidates. Calculate, run. Then tick the exploratory cluster option and run again. | 9 then 18 structures; headline says how many significant comparisons are clean vs provisional; "Clusters (modeled)" 20 or 0. Default run never drops SCHOOL. |
| 5 Workflow A + transfer (M07) | Workflow A, structure "correlated slope + AR(1)", candidate GROUP. Run; open full analysis; View candidate code ▸ Save script + data; close the app; run the saved script with Rscript from another folder. | Full analysis *verified*; the script prints "Reproduced within tolerance" and "Model identity reproduced". |
| 6 Reused IDs (M09) | Import `CD_ReusedID.csv`; workflow B with cluster SCHOOL, IDs unique (default) → refused with an explanation; switch to "numbered within each cluster". | Runs; 160 participants. |
| 7 Warnings (M05) | `T37D_3702.csv`, workflow B (RI / uncorrelated / correlated slope, lme4), fixed GROUP + TIME. | RS flagged with one gradient message; 1 clean and 2 provisional significant comparisons; RI group count 80. |
| 8 Non-Gaussian (M11) | `T28_29_Synthetic.csv`: Mixed, outcome BINARY (nominal), family binary, compare random effects; GEE outcome Y, compare residual structure; Discovery with cluster for a binary outcome. | GLMM comparisons labelled ML (Laplace); GEE shows QIC/QICu/CIC, no AIC/LRT; the clustered binary search is refused with a reason. |
| 9 Display (CD-12, UI-33/43) | Windows display scaling 150 %: Discovery results at maximized and restored window sizes; scroll the tests table sideways and down. | No clipped labels; the frozen Test ID column stays aligned and never covers the scroll bar. |
| 10 Persistence (M12) | Save the project from step 4, change settings, reopen it. | The cluster, workflow and structures come back; the latest search is shown immediately. |

Rollback: v1.3.1 can be reinstalled over 1.3.2; projects saved by 1.3.2 open in 1.3.1 (extra fields are ignored),
and the first 1.3.2 save of an older project keeps a backup copy.
