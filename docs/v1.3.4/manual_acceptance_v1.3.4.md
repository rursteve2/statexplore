# StatExplore 1.3.4 — minimal owner acceptance checklist

Everything below was also run automatically (see `evidence_report_v1.3.4.md`). These are the few checks worth seeing
with your own eyes in the installed program; each takes a few minutes. Items marked **deferred** were approved as
deferrals on 2026-11-27 and were not run by the developer.

| # | Check | Expected |
|---|---|---|
| M1 | 51H3 bootstrap: open 51C data; AMD with required Time + Gender, candidates Sleep, Anxiety, Attendance, all combinations up to three-way, random slope, School cluster; Validation ▸ new clusters, bootstrap B = 20, seed 133, whole pipeline | #17 selected in 20 of 20 draws; estimand table lists Sleep × Anxiety × Attendance (≈ 1.35–1.62) |
| M2 | 48D GEE: workflow B with GEE independence, exchangeable, AR(1), time × X locked | QIC 1146.365 / 1147.731 / 1149.216 (different per structure); Model comparison shows QIC, ΔQIC, QICu, CIC and no AIC |
| M3 | 51D: tick *Include time in eligible interactions*; Sleep, Anxiety optional; Time, Attendance, Gender required; max 5; combinations | 12 candidates; one is Sleep + Anxiety + Sleep:Anxiety + Time:Sleep |
| M4 | 52D restart: open `StatExplore_Test52D_Persistence.sxp`, pick the partial 5/15 search, set model budget 10, Resume | Same search ID; 15 / 15 complete; no new search in the list; Resume then disabled with "the search is complete" |
| M5 | Log-domain display: any search with an extreme test (e.g. 54B or 51C) | p shows `< 1e-300 (log10 p = …)`; sorting by p puts −485 before −419 |
| M6 | Candidate order: in AMD select three candidates and use ▲ / ▼ / ⤒ / ⤓ / A–Z / Data order | Positions 1., 2., 3. update; the order is kept after Save / reopen |
| M7 | View candidate code ▸ Educational R code; then Export audit trail… | Educational view labelled DIDACTIC; audit JSON + Markdown written, no personal paths |
| M8 | Validation ▸ Export with data, move the folder, run `Rscript validation_reproduce.R` | "Predictions reproduced" / "Bootstrap selections reproduced" |
| M9 | **deferred** Physical display at 150 % Windows scaling | No clipped controls (simulated 100–200 % layout tests pass; physical not run) |
| M10 | **deferred** Clean-machine install (installer and portable zip) offline | Starts, loads a CSV, fits, exports, reopens a project |
