# v1.3.6 targeted owner acceptance (tests 55–61)

**Run the automated kit first.** `StatExplore-1.3.6-Acceptance-Kit.zip` → `Run_Acceptance.bat`, then upload the one
ZIP it writes.

The kit automates every numerical, identity, export and reproduction check of tests 55–61. It also runs the owner R
diagnostic (NUM / ID / SEL / SUM) against this build's generated verifier, so **no RStudio session is needed**.

What remains below is what only a person at the installed GUI can see. Each step says what to compare with the kit's
results. You do not need to repeat any v1.3.5-or-earlier test that the kit covers.

**Before you start**

- Install `StatExplore-1.3.6-Setup.exe` over 1.3.5 (this is the upgrade check, step U).
- Open Help ▸ About and check that it says 1.3.6.
- The datasets are in the kit's `fixtures\` folder. **Always open a copy of a project.**
- Record each step as PASS, FAIL, PARTIAL, BLOCKED or NOT RUN.

---

## U — Upgrade from 1.3.5 (installed)

1. With 1.3.5 installed, save a project with one search (any small one, e.g. 55L below) and one t-test.
2. Install 1.3.6 over it. Start StatExplore from the Start menu and open that project.

**Expected**

- The project opens and the results are shown unchanged.
- The search's Resume button is disabled. The reason starts "this search was recorded with data identity schema 2".
- *New search from this one* is enabled.
- Reports ▸ export of the t-test with format R writes `REPRODUCTION_UNAVAILABLE.txt` for it, not a script with data.
  It was saved by 1.3.5 and has no recorded data provenance.

## 55 — Extreme tails in the GUI

**Dataset:** `StatExplore_Test55L_LegacyUnderflow.csv`.

**Settings:** y; Gaussian; candidates x1, x3, x2 in this order; single focal interaction; covariates focal; max 2;
mode C; 1 worker.

**Expected**

- The ranking shows #1 (x1 × x2) first and #0 second, by raw p and by adjusted p.
- Both show "< 1e-300 (log10 p = …)": about −885 for #1 and −555 for #0.
- The *log10 p* column sorts the same way.

The kit checked the same numbers against an independent lm + pf in R. This step only checks the display and the
sorting clicks.

## 56 — GEE (48D) in the GUI

Run workflow B with GEE independence, exchangeable and AR(1) (as in 1.3.5).

**Expected**

- The comparison table has QIC, QICu and CIC columns and no AIC.
- The exchangeable transfer to Mixed & Longitudinal shows the working correlation "exchangeable".

The kit checked all values against geepack.

## 57 — Resume after restart (installed)

1. Open a copy of `StatExplore_Test52D_Persistence.sxp`. Select search `ea77ac2fedc4`.
   - **Expected:** Resume is disabled; the reason names fit rules 3.
2. Click *New search from this one* and confirm. Save, **close StatExplore**, start it again and reopen.
3. Set the model budget to 10 and click Resume.
   - **Expected:** complete, 15 / 15.
   - **Expected:** the original search is unchanged.

## 59 — Exports: new folder and consent (installed)

1. In Reports, tick a result, choose format R and export into a folder that already contains files.
2. Answer **No** to "Include the data?".
3. Export again and answer **Yes**.

**Expected**

- Each export creates a **new** sub-folder (the message names it). The old files are untouched.
- The *No* folder has no `*_data.csv` file. Its `bundle_inventory.json` says `"bundle_kind": "code-only"`.
- The *Yes* folder has one sub-folder per analysis, each with its own `*_data.csv`.
- The same happens for Validation ▸ Export and Code ▸ Save reproducibility bundle.

## 60 — Large space and cancellation (installed)

**Dataset:** `StatExplore_Test53A_30Predictors.csv`. Use the 1.3.5 settings.

**Expected**

- *Calculate search space* shows 147,713,444 immediately, and the window does not freeze.
- Cancel within a second of starting a mode-D search with budget 2,000: the status is cancelled.

The kit measured the cancellation latency.

## 61 — 100k data (installed)

**Dataset:** `StatExplore_Test54B_100K.csv`. Use the 1.3.5 settings.

**Expected**

- The search completes. The kit's timing is in its `results.csv`.
- Sorting by *p (adjusted)* puts the two planted interactions first. Neither shows a literal 0.

---

## Environment-dependent steps

| Item | Status |
|---|---|
| Physical display scaling (150 %) | NOT RUN. Simulated scaling is tested (`tests/test_v134_gui.py::test_simulated_display_scaling_layout`). The master prompt allows it to remain NOT RUN. |
| Clean-machine offline install | Owner-approved deferral (standing approval, 2026-11-28). The installed and frozen builds were checked offline on the developer machine (`evidence_report_v1.3.6.md`). |
| In-place upgrade from 1.3.5 | Run by the developer on the developer machine (see the evidence report); repeated above as step U with your own project. |
