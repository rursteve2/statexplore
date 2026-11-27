# v1.3.5 targeted owner acceptance (tests 55–61)

The developer ran the automated and real-R coverage first (`evidence_report_v1.3.5.md`). These GUI checks complement
it; you do not need to repeat older tests.

**Before you start**

- Use the installed StatExplore 1.3.5. In the title bar, click Help ▸ About and check the version.
- Datasets are in the v1.3.4 handoff, `Synthetic_Datasets\`. 55L and 53A are also in `tests\fixtures\v135\`.
- Projects are in `Saved Projects\`. **Always open a copy.**

**Record each test as** PASS, FAIL, PARTIAL, BLOCKED or NOT RUN, and note anything unexpected.

**Optional exports.** Use *Export audit trail…* (Automated Model Discovery) or *Export…* (Validation panel) into a new,
empty folder, and keep the folder.

---

## 55 — Extreme tails and complete reproduction

Two parts: **55a** in the legacy workspace, **55b** in AMD validation.

**Why two datasets.** `StatExplore_Test51C_ThreeWayInteraction.csv` is longitudinal, and the legacy Validation
Workspace supports single-level searches only. Part 55a therefore uses the synthetic fixture
`StatExplore_Test55L_LegacyUnderflow.csv` (n 6,000; generator `make_55l.py`).

### 55a — legacy workspace (55L)

**Settings** (Automated Model Discovery)

| Setting | Value |
|---|---|
| Outcome | y |
| Family | Gaussian |
| Candidates, in this order | x1, x3, x2 |
| Strategy | Single focal interaction |
| Covariate adjustment | Focal |
| Max distinct predictors | 2 |
| Min candidate predictors | 1 |
| Mode | C |
| Workers | 1 |

**Steps**

1. Start the search.
2. Open Validation Workspace ▸ Discovery stability with resamples 5, fraction 0.8, seed 11.
3. Open Validation Workspace ▸ Full-procedure CV with folds 3, seed 5.

**Expected**

- After step 1:
  - Ranking by raw p and by adjusted p: #1 (x1 × x2) is first and #0 (x1 × x3) second. Both show "< 1e-300 (log10 p = …)"; the
    log10 p of #1 is about −885 and that of #0 about −555 (±1).
  - The *log10 p* column sorts the same way.
- After step 2: in the stability table, #1 is ranked first in 100% of resamples. `p_full_data` is 0 for #0 and #1, but
  `log10_p_full_data` differs.
- After step 3: every fold has `selected_index` = 1.

### 55b — AMD validation (51C)

**Settings** (Automated Model Discovery)

| Setting | Value |
|---|---|
| Outcome | TestScore |
| Candidates | Sleep, Anxiety, Attendance |
| Required | Time, Gender |
| Repeated measures | ID StudentID, Time, cluster School |
| Structure | random intercept + time slope (lme4) |
| Strategy | All eligible interaction combinations |
| Include main-effects models | on |
| Interaction order | two- and three-way |
| Max predictors | 5 |
| Workers | 2 |

**Steps**

1. Start the search. It has 18 candidates.
2. In the Validation panel, set:

   | Setting | Value |
   |---|---|
   | Target | New clusters |
   | Procedure | Whole selection pipeline |
   | Candidate pool | Complete declared procedure |
   | Modes | Bootstrap stability only |
   | B | 20 |
   | Seed | 133 |
   | Fit budget | 100,000 |

3. Run the validation.
4. Export with *Include the data* = Yes, into a folder whose path has a space and a non-ASCII letter (e.g. `D:\test ü`).
5. In that folder, run `Rscript validation_reproduce.R`.

**Expected**

- After step 1: #17 has whole-model log10 p ≈ −485.20 and joint log10 p ≈ −419.54 (±0.05), in different families.
- After step 3: selection #17 in 20 / 20 draws; status complete.
- After step 5:
  - The script ends with "All N checks passed" and "The complete validation was reproduced".
  - It writes `validation_check_results.csv` with no FAIL.
  - If you edit one value in `predictions.csv` and run it again, the run stops with "VERIFICATION FAILED" (exit code 1).

## 56 — GEE structures (48D)

**Settings**

| Setting | Value |
|---|---|
| Dataset | `StatExplore_Test48D_GEE.csv` |
| Outcome | Y |
| Candidates | X |
| Required | TIME |
| Repeated measures | ID ID, Time TIME |
| Workflow | B |
| Structures | GEE independence, exchangeable, AR(1) |
| Family | Gaussian |

**Steps**

1. Start the search.
2. Open the Model comparison tab.
3. Transfer the exchangeable model to Mixed & Longitudinal.

**Expected**

- QIC: independence ≈ 1146.3654, exchangeable ≈ 1147.7314, AR(1) ≈ 1149.2155.
- QICu and CIC appear in their own columns; there is no AIC.
- The fitted working correlation of each row matches its structure.
- The transferred model keeps its working correlation.
- No value changed from v1.3.4.

## 57 — Historical resume refusal (52D)

**Input:** a copy of `StatExplore_Test52D_Persistence.sxp`.

**Steps**

1. Open the project. Select search `ea77ac2fedc4` (partial, 5 of 15).
2. Click *New search from this one* and confirm.
3. Save. Close StatExplore. Reopen the project.
4. Select the new search. Set the model budget to 10 and click Resume.
5. Save, reopen, and select the original search again.

**Expected**

- After step 1:
  - Resume is disabled.
  - The reason names fit rules 3 versus 4 and suggests a new search.
  - *New search from this one* is enabled.
- After step 2:
  - A new search ID appears, partial 5 / 15, with budget 5 as declared.
  - The original stays partial with its 5 rows unchanged.
- After step 4:
  - Resume is enabled with no reason shown.
  - The search becomes complete at 15 / 15, every candidate exactly once.
  - The execution budgets read 5 then 10.
- After step 5:
  - The original is unchanged.
  - The audit export of the new search shows `parent_id` = ea77ac2fedc4.

## 58 — Minimum count and validation scope (51F Null_13)

**Dataset:** `StatExplore_Test51F_Null_13.csv`

### 58a — minimum count

**Settings**

| Setting | Value |
|---|---|
| Outcome | Y |
| Candidates | X01, X02, X03 |
| Required | Time |
| Repeated measures | ID StudentID, Time Time |
| Strategy | Single focal interaction |
| Max predictors | 4 |
| Min candidate predictors | 3 |
| Interaction order | two-way |

**Steps**

1. Click *Calculate search space* with covariate adjustment set to *Focal*.
2. Switch covariate adjustment to *Subsets* and click *Calculate search space* again.

**Expected**

- With *Focal*: refused, with the reason (a focal pair has 2 candidate predictors).
- With *Subsets*: exactly 3 candidate models, each containing X01, X02 and X03.

### 58b — validation scope

**Settings:** X01 … X06, Min 1, Max 3, Focal covariate adjustment, mode D, model budget 5.

**Steps**

1. Start the search.
2. Set the model budget to 10 and click Resume. The search completes at 15.
3. In the Validation panel, set:

   | Setting | Value |
   |---|---|
   | Target | New participants |
   | Procedure | Pipeline |
   | Mode | Holdout |
   | Candidate pool | Complete declared procedure |

4. Preview the validation, then switch the pool to *Original execution only*.
5. Switch the pool to *Explicit first N* with N = 7.

**Expected**

- After step 4:
  - *Complete declared procedure*: "CUMULATIVE declared model budgets: the first 15 of 15 (5 + 10)", with the ADAPTIVE
    note.
  - *Original execution only*: "ORIGINAL … first 5".
- After step 5: "RESTRICTED pool … first 7".

## 59 — Historical audit and reproduction identity (50 / 50E)

**Inputs**

- a copy of `StatExplore_Test50_Reproducibility.sxp`
- `StatExplore_Test50E_ModifiedValues.csv`

**Steps**

1. Open the project. Export the audit trail into folder A.
2. Replace the data with the 50E file (Dataset Explorer ▸ replace / re-import). Export the audit trail into folder B.
3. If the search has a validation run, export it with data.

**Expected**

- In A: `data.historical.source_sha256` = 3586bd71…; `data.current_context.matches_analysis` = true.
- In B:
  - `data.historical` is identical to A.
  - `current_context.matches_analysis` = false, with SHA-256 8b7d2a4f…
  - The Markdown report says "DIFFERS from the analysed data".
- From step 3:
  - No runnable bundle and no data file are written.
  - `REPRODUCTION_UNAVAILABLE.txt` gives the reason.
  - The results CSVs are still written.

## 60 — Large-space fixed model, cache-only cancellation, budgets (53A)

**Dataset:** `StatExplore_Test53A_30Predictors.csv`

**Settings**

| Setting | Value |
|---|---|
| Outcome | Y |
| Candidates | X01 … X30 |
| Required | Time |
| Repeated measures | ID StudentID, Time Time |
| Strategy | All eligible interaction combinations |
| Include main-effects models | on |
| Max predictors | 6 |

**Steps**

1. Click *Calculate search space*.
2. Start with mode D and a model budget of 100.
3. In the Validation panel, choose Fixed candidate, pick #50, and look at the preview / feasibility line.
4. Start a second search with the same settings, mode D and a model budget of 100. Cancel it within a second.

**Expected**

- After step 1:
  - 147,713,444 candidates, shown immediately.
  - The workload line reads "about … (estimate; at most … without reuse)".
- After step 2: 100 attempted; status partial (model budget reached).
- After step 3:
  - The line appears immediately; the GUI does not freeze.
  - It reads "at most N model-fitting invocations (conservative bound …)".
- After step 4: the status is cancelled, and no rows were added after the cancel was acknowledged.

## 61 — Large data and extreme adjusted sorting (54B)

**Dataset:** `StatExplore_Test54B_100K.csv`

**Settings**

| Setting | Value |
|---|---|
| Outcome | Y |
| Candidates | X01 … X06 |
| Required | Time |
| Repeated measures | ID StudentID, Time Time, cluster School |
| Strategy | Single focal interaction |
| Max predictors | 3 |
| Min | 2 |
| Workers | 4 |

**Steps**

1. Start the search.
2. Sort the Discovery ranking by *p (adjusted)*, then by *log10 p*.

**Expected**

- After step 1: completes in about 50 s on a machine like the developer's; 15 candidates; 2 BH-significant.
- After step 2:
  - The two planted interactions are first in both sorts, ordered by their log10 p (both below −300, for example
    LRT 54085.5 and 23373.5).
  - Neither shows a literal 0.

---

## Environment-dependent steps

| Item | Status |
|---|---|
| Physical display scaling (150%) | **NOT RUN** — no VM required. Only simulated scaling was checked (`docs/v1.3.4/evidence/V134-017_*`). |
| Clean-machine offline install | **Owner-approved deferral** (2026-11-27, v1.3.5 session). The in-place upgrade, the frozen self-test and offline operation on the developer machine were run (`evidence_report_v1.3.5.md`). |
