# StatExplore 1.3.6 — consolidated acceptance kit

One unattended run covers the automatable part of owner tests 55–61 plus the owner's R diagnostic (NUM/ID/SEL/SUM)
against the verifier that this build generates. It replaces the earlier "configure, export, upload, write R code,
compare" loop: run it once and upload one ZIP.

## Run

1. Install StatExplore 1.3.6 (or keep the portable copy).
2. Extract this kit anywhere (for example `D:\SX136 kit`).
3. Double-click **`Run_Acceptance.bat`**. It finds the installed `StatExplore.exe` (Program Files or your user
   Programs folder). For another copy: `Run_Acceptance.bat "D:\path\StatExplore.exe"`.
   `Run_Acceptance.bat "" --quick` shortens the 51C bootstrap (3 draws instead of 20; about 15 minutes).
4. Wait (about 30–60 minutes on a 4-core machine). Progress: `results\acceptance_live_*.log`.
5. Upload `results\StatExplore_v1.3.6_Acceptance_Results_<timestamp>.zip`.

The run uses the R that ships with StatExplore. It installs nothing, needs no network, uploads nothing, and works on
copies of the fixtures in a temporary folder (the files in `fixtures\` are never changed). All fixtures are synthetic
(SHA-256 in `fixtures_SHA256SUMS.txt`); no HRS or other private data.

Source checkout alternative: `python -m statexplore --acceptance --fixtures <kit>\fixtures --out <dir> [--only 55,57]`.

## What the ZIP contains

| File | Content |
|---|---|
| `summary.md` | PASS / FAIL / BLOCKED per case and every check that did not pass |
| `results.csv`, `results.json` | one row per check: case, check, expected, actual, tolerance, kind, status |
| `environment.json` | StatExplore version and build, executable SHA-256, Python / numpy / pandas, R and package versions, OS, CPU, fixture SHA-256 |
| `run.log`, `logs\` | the run's log and the raw output of every Rscript run |
| `artifacts\` | every export, reproduction script and `validation_check_results.csv` produced |
| `SHA256SUMS.txt` | SHA-256 of every file in the ZIP |

Execution kinds (column `kind`): **app+R** the application ran real R fits (same code as the GUI, no window);
**oracle-R** an independent R calculation written in the kit (lm/anova/pf, geepack, t.test); **app** application
logic without R; **script-R** a script exported by the application, run in a fresh Rscript. A check that could not run
is BLOCKED, never PASS.

## Cases

| Case | Inherited objective | Automated here |
|---|---|---|
| D | Owner R diagnostic (v1.3.5: 11/20 failed) | NUM-01..09, ID-01, SEL-01..03, 17 extra controls, SUM-01..07 on the generated verifier |
| 55 | Extreme tails, bootstrap, reproduction | 55L log10 p vs lm+pf oracle, log-tail ranking, AMD validation export → fresh Rscript (exit 0), 3 field tampers (exit 1 naming the check), cancelled run → partial replay; 51C log10 p −485.200773 / −419.542529, 18/48, bootstrap #17 in 20/20, reproduction |
| 56 | 48D GEE and QIC | geepack oracle vs owner QIC (1146.365433 / 1147.731438 / 1149.21553), fitted working correlation per structure (both orders), QICu/CIC, no AIC, transfer keeps exchangeable, same-view cache reuse |
| 57 | 52D refusal, linked new search, save / restart / resume | fit-rules-3 refusal, unverified legacy identity, restart (parent link) 5/15 → save → reopen → resume → 15/15 (budgets 5, 10), original unchanged; workbook sheet A vs B: no resume, no cache |
| 58 | Null13 minimum and cumulative scope | min-3 focal refused, subsets = 3 models, 5 + 10 = 15 cumulative, original 5, prefix 7, ADAPTIVE label, training trace, ledger, reproduction |
| 59 | 50 / 50E historical audit and export | historical block unchanged after the data replacement, legacy context unverified, 50E differs, validation export results-only; reused folder + consent; holdout seed 1 → 2 with real t-tests: each bundle reproduces its own n / mean / t / CI (t.test oracle) |
| 60 | 53A space, cancellation, budgets | 147,713,444 without materializing, budget 100, fixed #50 feasibility, cancellation, pre-cancelled cached run; exact cap complete / cap−1 partial (real R) + reproduction |
| 61 | 54B 100k and sorting | identity fingerprint cold / warm time, real 15-candidate search, 2 BH-significant, adjusted sort by log tail, no literal 0 |

## Still manual (installed GUI only)

See `manual_acceptance_v1.3.6.md`: the GUI look of the same results (sorting clicks, dialogs, consent prompt, new export
folder message), the installer upgrade from 1.3.5 with your own copy of a project, and physical 150 % display scaling.
Everything else listed above is automated and need not be repeated by hand.
