# StatExplore — Build Plan

Source spec: `StatExplore_Final_Master_Prompt.pdf` (v1.0, Nov 2026).
Test dataset: `E:\Downloads\hrs_least_missing_1000.sav` (HRS, long format, 1000 persons × 16 waves = 16,000 rows, 168 vars, ~29% cells missing).

## Decisions (agreed 2026-11-20)

| # | Decision | Why |
|---|----------|-----|
| D1 | **R is the single model-fitting engine.** Python = GUI, import, data management, orchestration, project storage. Python stats only for cheap overview numbers (counts, missingness, IQR outliers) and validated cross-checks in tests. | Satisfies "one default implementation per analysis", no silent engine drift, exported code is always R. |
| D2 | **Persistent R worker pool** (long-lived `Rscript` processes, JSON-lines over stdin/stdout). Dataset loaded into each worker once; requests carry only commands/specs. | Launching Rscript per model costs ~0.3–1s; unusable for discovery. |
| D3 | **Generated script = executed script.** Every analysis is rendered to an R script string; the worker `eval`s exactly that string; "View Code" shows that same string. | Guarantees §18 by construction. |
| D4 | **Internal safe aliases** (`v001`, `v002`, …) for all variables in R; display names/labels mapped in Python. Exported code includes the alias→name map and renames back. | Removes formula/shell injection risk (§23) and R-hostile names. |
| D5 | Python→R data transfer: CSV of codes + JSON schema (type, levels, value labels, reference level, missing codes). R rebuilds factors from schema. | Preserves types/labels/missing; zero extra R deps. Can swap to Arrow later if speed requires. |
| D6 | Project file = **single SQLite file** (`.sxp`) holding metadata, transformation log, analysis history, results, and a project-local Parquet copy of the data + source path + SHA-256. | One file to save/move; original file never touched. |
| D7 | **Discovery default unit = focal interaction + fixed/required covariate set**, default max predictors = 3 (user may raise to 6 with warning about search size). Full "all covariate subsets" enumeration is an explicit opt-in. | 39C6 × 15 pairs ≈ 49M models; the covariate-subset family makes adjusted p-values meaningless. |
| D8 | **Shared analysis sample** in discovery by default: rows complete on all selected variables (rows dropped shown up front). Per-model complete-case available as an option, flagged. | Comparable models; valid AIC/LRT; coherent multiple-testing family. |
| D9 | **Focal interaction test = nested model comparison** (with vs without the focal term): F-test (lm), LRT (glm, ML mixed). Effect size = ΔR² / partial η² (lm), ΔDeviance + OR where 1-df (glm). | Defined for multi-df categorical interactions; Type II/III agree for highest-order term in hierarchical models. |
| D10 | Outcomes in v1: continuous + binary. Count/ordinal deferred (post-Phase 7). | Scope. |
| D11 | Develop with Windows Python 3.11 venv (`.venv`, Windows-native), R 4.6.x for Windows. Project-local R library `r/library` for Phase 1–6; renv lockfile + bundled portable R in Phase 7. | Target is native Windows. |
| D12 | `flexplot` is GitHub-only → vendor a pinned source tarball. Add `detectseparation` for logistic separation checks. | Offline packaging. |

---

## Phase 1 — Desktop Foundation  ✅ **complete (2026-11-21)**

- [x] Project scaffold, `pyproject.toml`, git init
- [x] `docs/architecture.md`
- [x] Standardized result schema (`core/schema.py`)
- [x] Variable metadata model (name, alias, label, value labels, user-missing, measure, role, reference)
- [x] Importers: CSV (sniffing + encoding fallback/manual), SAV (labels, value labels, user-missing, measures), XLSX (sheet selection)
- [x] Python↔R bridge: R locator, persistent worker, JSON-lines protocol, timeout→kill→auto-restart, structured errors, version capture
- [x] R side: `worker.R`, generated data-load script, `psych::describe` / `describeBy`, ggplot2 + pairs.panels plots
- [x] GUI: main window, 10-section sidebar (unbuilt sections disabled with "Planned for Phase N"), drag-and-drop, CSV options dialog, sheet chooser
- [x] Dataset Explorer: overview, spreadsheet preview (value-label toggle, user-missing in red), variable inspector/editor
- [x] Variable management subset: display name, measure, role, reference level — logged + undo
- [x] EDA: 8 plot types, type-based suggestions, View Code (exact executed script)
- [x] Background execution (R pool = 1 thread, IO pool)
- [x] Project save/open (`.sxp` SQLite, atomic save, read-only open)
- [x] Acceptance tests: **55/55 pass** (import, labels, missing, project round-trip, source unchanged, codegen/injection, GUI smoke, R engine: describe vs independent reference to 1e-10, all 8 plots, factor coding, script reproducibility, timeout/restart) — stable over 3 consecutive runs, no orphan R processes
- [x] R 4.6.2 installed; jsonlite 2.0.0, psych 2.6.9, ggplot2 4.0.4 in `r/library`
- [x] End-to-end GUI drive on HRS (describeBy, violin, correlation heatmap) — results OK, renders checked
- Moved to Phase 2: derived variables, row filters (§6.4)

## Phase 2 — Core Statistical Engine  ✅ **complete (2026-11-21)**

### Design (defaults are recorded in every result)
- **Analyses are declarative specs** (`analysis/specs.py`): role slots (dependent, factors, covariates, predictors, id, within…) + options. The Analysis Builder form is generated from the spec; each spec has a pure `build(ds, params) -> R body` + `parse(raw) -> tables`. GUI-independent and unit-testable.
- **Shared R helpers** (`sx_table`, `sx_shapiro`, …) are defined in the data-load script header, so View Code stays self-contained.
- **Missing data:** complete cases on the analysis variables (shown: n used / n excluded). RM/mixed ANOVA: participants missing any within-cell are dropped by afex (reported).
- **Derived variables & filters** are specs, not free text: computed in Python (preview/overview) AND rendered to R in the load script (so View Code reproduces from the source file). A test asserts R == Python for every operation.
  - Ops: row mean/sum of items (with min-valid count), recode/collapse codes, log/sqrt/z-score/center/square, binarize at cutoff.
  - Filters: `var op value` rows, AND-combined; NA comparisons drop the row (as dplyr::filter).
- **Engine defaults (one per analysis):**
  | Analysis | R implementation | Effect size | Assumption checks |
  |---|---|---|---|
  | t-tests | `stats::t.test` (Student / Welch / paired / one-sample) | `effectsize::cohens_d`, `hedges_g` (+CI) | Shapiro-Wilk (n≤5000), `car::leveneTest` (median) |
  | Correlation | `stats::cor.test` (Pearson CI = Fisher z; Spearman no CI), `ppcor::pcor.test`, `psych::corr.test` matrix (raw + Holm) | r | — |
  | ANOVA/ANCOVA | `afex::aov_car` Type III, sum contrasts; ANCOVA covariates mean-centred; Welch = `stats::oneway.test` | partial η² + generalized η² | Levene; Mauchly + GG correction (RM) |
  | Post hoc / EMMs | `emmeans` pairwise, Holm (default) or Tukey | — | — |
  | Linear regression | `stats::lm`, treatment contrasts (reference level shown); hierarchical = nested `anova()` ΔR²/F-change; interactions also tested by nested model comparison (D9) | standardized β (`effectsize::standardize_parameters`, refit), R²/adj R², f² for blocks | residual plots, QQ, BP (`lmtest::bptest`), VIF (`car::vif`), Cook's D > 4/n |
  | Logistic | `stats::glm(binomial)`, profile-likelihood CIs | OR (+CI), Tjur R² | `detectseparation`, convergence, events/parameter |

### Tasks
- [x] R helpers in load script; generic result parser (tables/plots/values)
- [x] Derived variables + row filters (dataset model, Python compute, R codegen, project save, UI dialogs, undo)
- [x] Specs + R builders: t-tests ×4, correlations (pair, partial, matrix), ANOVA ×6 (one-way, Welch, factorial ≤3 factors, RM, mixed, ANCOVA), linear (simple/multiple/interaction/polynomial/spline), hierarchical, logistic
- [x] Diagnostics plots (lm: residual vs fitted, QQ; logistic: predicted probability curve)
- [x] Analysis Builder page: analysis picker, generated form, results viewer (tables, plots, messages, View Code)
- [x] EDA: `psych::corr.test` matrix (raw + adjusted p)
- [x] Tests vs independent references (scipy/statsmodels): A (t-test, d), B (2×2 interaction, Type III F), D (continuous interaction + conditional slopes), E (logistic OR, predicted p), J (missing → n), plus Welch, paired, one-way, Welch ANOVA, RM ANOVA, ANCOVA, correlations, partial r, hierarchical ΔR², separation detection, derived/filter parity
- [x] HRS end-to-end check in the GUI

## Phase 3 — Automated Model Discovery  ✅ **complete (2026-11-21)**

### Design
- **Candidate grammar** (`docs/formula-grammar.md`): a candidate = (focal interaction, covariate set).
  Full model = focal variables' complete hierarchy (all lower-order terms) + covariates as main effects + required covariates.
  Reduced model = full model minus the focal term only. Focal test = nested comparison (F for linear, LR chi² for logistic) — D9.
  - *Focal mode* (default, D7): covariates = required covariates only. Count = C(p,2) [+ C(p,3)].
  - *Subset mode* (opt-in): for every subset S of candidates with |S| ≤ max predictors, every focal pair/triple inside S, other S members as covariates. Count = Σ_s C(p,s)·[C(s,2) + C(s,3)]. Each (focal, covariate set) is a distinct candidate analysis (§10.3).
  - Required covariates count toward the max-predictor limit. Unique by construction; canonical key = sorted term set (tested).
- **Deterministic order**: lexicographic over candidate-predictor index (subset, then focal). Recorded with the run.
- **Sample**: shared complete-case sample on outcome + all candidates + required (D8, default; n shown before run) or per-model complete cases (flagged).
- **Screening (R, per candidate)**: parameter count vs limit, residual df, design rank, empty focal cells (invalid); sparse cells < 5, |r| > .9 between continuous predictors, logistic separation / non-convergence / fitted 0-1 warnings (flags).
- **Execution**: pool of persistent R workers (each loads data once), batches of candidates; R function `sx_discover_one()` lives in the load-script helpers, so a candidate's View Code is exactly what ran.
- **Modes**: A first significant, B target count, C exhaustive, D budget (models / minutes / Cancel). A/B stop deterministically at the first qualifying index in enumeration order; results computed speculatively past it are kept but marked and excluded from the testing family.
- **Profiles**: Fast (1 worker, 2,000 models, 2 min), Balanced (default; up to 4 workers, 50,000 models, 30 min), Thorough (cores-1 workers, no model cap, no time cap).
- **Multiplicity**: family = all candidates actually tested (valid) in the run. BH (FDR) and Holm reported; provisional when the run is incomplete.
- **Rankings**: Discovery = by raw or adjusted p (never labelled "best"). Support = transparent tiers within the run's model family: A clean + adjusted-significant, B clean + raw-significant only, C flagged, D not significant; within tier by effect size (ΔR² or |log OR|), then fewer parameters.
- **Storage/resume**: results kept as a DataFrame per run, saved in the project; a stopped run can be resumed (same settings + data signature) and continues from the indices not yet done.

### Tasks
- [x] `discovery/space.py`: settings, enumeration, exact counts, keys, formulas, shared-sample size
- [x] R `sx_discover_one` + batch runner; worker pool (threads, one persistent R process each)
- [x] `discovery/engine.py`: modes, budgets, cancel, progress, resume, adjustments, rankings
- [x] Detail view: re-run candidate as full Phase 2 analysis on the same sample (p and n match exactly)
- [x] Discovery workspace UI (config, search-size estimate, progress/ETA, summary + 3 tables, drill-down, candidate code)
- [x] Project persistence of search runs (`searches` table; older projects still open)
- [x] Tests (24): M enumeration/count/uniqueness/hierarchy/resume-tail, spec §10.3 example, 39C6 count, BH/Holm vs statsmodels, planted interaction (linear + logistic, subset mode), K null data, invalid models recorded with reasons, partial budget + resume == full, mode A deterministic with 2 workers, 1 vs 2 workers identical, cancel, detail p/n == search, project round-trip, dependence warning, GUI end-to-end
- [x] HRS end-to-end
- [x] Added: dependence (repeated-ID) warning for searches

### Review (Phase 3)
- 101/101 tests pass; no orphan R processes.
- HRS wave 10 (one row per person), CES-D 8 ~ 14 candidates, focal two-way: 91 candidates in 6.8 s (4 workers); 41 invalid (8-level marital status → empty cells; years of education stored as a 16-level nominal → > 40 parameters); 50 tested; 3 raw p < .05 vs ≈ 2.5 expected by chance; 0 after FDR. An honest null.
- Deferred: "teaching interpretation" column and interaction graphs → Phase 4; validation status → Phase 6; mixed-model discovery → Phase 5; "all two-way terms in one model" variant not included (documented in grammar).
- Teaching note for the user: set `RAEDYRS` to continuous before searching; declared nominal it adds 15 dummy parameters.

## Phase 4 — Interaction Visualization & Teaching Studio  ✅ **complete (2026-11-21)**
- [x] `ModelSpec` (hierarchically closed terms, focal interaction, shared-sample variables, provenance)
- [x] Interaction plots from model predictions on an explicit grid (x varies; group/panel at displayed values; others held at mean / reference level and printed); 2-way, 3-way (panels), categorical × categorical (points + error bars), logistic probability curves
- [x] Conditional slopes (emtrends) / simple effects (emmeans), exact Johnson-Neyman interval
- [x] Graph controls: x, lines, panels, moderator values (±1 SD / percentiles), CI, raw data, labels, title, format (PNG/SVG/PDF; PNG preview always)
- [x] Backends: ggplot2 (default), base R; flexplot optional → disabled with explanation when not installed (GitHub-only; install blocked here — user action)
- [x] Template interpretations (3–6 sentences: what, whether, direction with CIs, J-N, limitation; selection-from-search caveat)
- [x] Model Comparison workspace: model set (from discovery, Teaching Studio, or new), edit-a-copy (original preserved), common-sample option, AIC/BIC/R², nested F/LR tests only for nested pairs on the same rows, different-n warning
- [x] Persisted model set in projects
- [x] Tests (13): predictions & CI bands == statsmodels get_prediction (1e-9), held values, moderator values, interaction p, slopes, J-N bounds hit t-crit exactly, reference levels for held/x factors, logistic curves, SVG/PDF/PNG for both backends, flexplot graceful failure, code reproduces predictions; comparison AIC/R²/nested F vs statsmodels, non-nested not tested, different-n warning, mixed outcomes rejected; GUI teaching + comparison + save/reopen; editor interaction-order bug

### Review (Phase 4)
- 115/115 tests pass; no orphan R processes.
- Bugs found and fixed: (1) a model's interaction was dropped when loaded into the editor if its variable order differed from the list order → matched as sets (+ test); (2) new models in the comparison set came in unchecked → default checked (found by GUI test); (3) display: plots buried under 240-row prediction tables → teaching graphs first, long tables truncated in view; internal columns hidden; "held at: none" subtitle removed.
- Convention noted: linear-model AIC/BIC count the residual variance (R convention; statsmodels differs by 2).
- HRS demo (wave 10): CES-D 8 ~ neuroticism × gender, F(1, 666) = 5.97, p = .015; slopes 0.73 (men) vs 1.27 (women). Interpretation flags it was selected from 91 candidates.

## Phase 5 — Advanced Mixed and Longitudinal Models  ✅ **complete (2026-11-21)** (mixed models in discovery deferred → see Remaining)
- [x] Assisted ID / time / cluster detection (suggestions + one-click role); wide → long restructuring into a new project
- [x] lme4/lmerTest: random intercept, correlated / uncorrelated random slopes, nested clusters; REML/ML; Satterthwaite df; Type III F
- [x] nlme: AR(1) (occasion order) and continuous-time AR(1); nlminb with optim fallback (recorded)
- [x] mmrm: unstructured marginal model (no random effects; clearly labelled)
- [x] Continuous / quadratic / categorical time; time × predictor interactions; labelled time codes handled
- [x] Variance components, R² marginal/conditional (Nakagawa), singular-fit and convergence checks, lag-1..3 residual autocorrelation
- [x] Plots: population-level trajectory with CI over sampled observed trajectories (seeded), conditional participant-specific predictions, residuals vs fitted
- [x] Specification comparisons: time (ML), random effects (REML, boundary-corrected p), residual structure (REML, all in nlme)
- [x] Workspace UI
- [x] Tests (13): F vs statsmodels MixedLM, G random slope variances/cov vs statsmodels, H time×group trajectory = fixed effects (with missing occasions), I AR(1) log-likelihood == independent numpy MVN computation, unstructured log-likelihood == numpy, singular fit, spec validation, duplicate occasions rejected, HRS structure detection, wide→long, comparisons vs statsmodels ML LRT + boundary formula, labelled time == numeric time; GUI test

### Review (Phase 5)
- 128/128 tests pass; no orphan R processes.
- Bugs found and fixed: (1) nlme `optim` failed on unbalanced data ("coefficient matrix not invertible") → nlminb default + fallback; (2) labelled time (SPSS "Wave 3") became NA as continuous time → explicit label→code map (+ regression test); (3) multi-string check message split again (lessons.md rule applied); (4) spurious ggplot/model.frame warnings; (5) internal names sx_time/sx_id shown in tables.
- HRS (waves 2–16, 1,000 people, 9,786 rows): CES-D rises 0.053/wave; women +0.58; no time × gender (p = .36) or time × vigorous activity (p = .28); random intercept SD 1.98; AR(1) improves fit (ΔAIC 34). lme4 convergence gradient 0.028 flagged.

## Phase 6 — Validation & Uncertainty  ✅ **complete (2026-11-21)**
- [x] Holdout reserved before exploring (rows or whole participants/clusters; seeded, deterministic, undoable, saved); every analysis, search and EDA uses training rows only; R load script splits explicitly
- [x] Holdout evaluation (fit on training, score on holdout) + interaction replication test in the holdout; reuse counter with warning
- [x] Cross-validation: k-fold, grouped (participants), temporal (expanding window); per-fold + summary + in-sample optimism; participant-overlap (leakage) table and warning; unseen factor levels handled
- [x] Discovery stability by resampling (participants when an ID exists): selection frequency, top-rank frequency
- [x] Full-procedure CV: whole search inside each training fold; selected model scored and re-tested on the test fold
- [x] Prediction uncertainty: CI vs prediction interval bands (linear), calibration plot (logistic), predicted vs observed
- [x] Missing Data panel (Dataset Explorer tab): by variable, patterns, usable n per model / search
- [x] Validation & Uncertainty workspace (11th sidebar section)
- [x] Tests (10): holdout split deterministic/grouped/undo/round-trip, holdout rows never reach analyses or discovery, holdout RMSE/R² vs numpy, reuse warning, Test N grouped CV zero overlap vs naive leakage, temporal splits, logistic CV metrics, stability + procedure CV recover planted effect, prediction intervals vs statsmodels, missing summaries; GUI test

### Review (Phase 6)
- 138/138 tests pass; no orphan R processes.
- Bugs found and fixed: helper names failed alias validation (good: the guard works) → explicit split column; complete-case helper dropped the ID/time columns CV needs.
- HRS wave-10 lead (light activity × education, p = .005, q = .31): significant in 80% of subsamples but ranked first in 40%; full-procedure CV picks different interactions across folds and out-of-sample R² ≈ 0 — a clear "significant but unstable" teaching example.

## Phase 7 — Export & Packaging  ✅ **complete (2026-11-21)** — clean-VM install test is a user step
- [x] Excel (index, per-result summary/tables/code sheets, discovery attempts + settings), Word (python-docx: interpretation, plots, tables, code), PDF (Qt, same HTML as screen)
- [x] R script export: self-contained (reads exported data.csv, relative plot paths, writes result JSON); Python launcher scripts (re-run R, load result — all statistics are R by design, D1)
- [x] Search settings JSON, analysis history CSV, variable change log CSV
- [x] Reports page (all 11 sidebar sections now implemented)
- [x] Packaging: PyInstaller spec + `packaging/build.py` (bundles R + r/library, writes r/packages.lock.json), Inno Setup script, frozen-mode R lookup, `--selftest`
- [x] Tests: Test O (exported R scripts reproduce lm / Teaching Studio / mixed results to 1e-9 in a fresh Rscript), xlsx values, docx/pdf content, full project round-trip
- [ ] User: compile `packaging/statexplore.iss` with Inno Setup; install on a clean Windows 10 and 11 machine with networking off; run the app and `StatExplore.exe --selftest`
### Review (Phase 7)
- 143/143 tests pass (+1 mixed-discovery test). Test O: exported R scripts reproduce lm / Teaching Studio / mixed results to 1e-9 in a fresh Rscript. One Test O failure occurred once while a PyInstaller build saturated the machine; not reproduced in 3 reruns (likely subprocess timing).
- Frozen build verified: `dist\StatExplore\StatExplore.exe --selftest` passes using ONLY the bundled `R\` and `rlib\`; the GUI launches (window "StatExplore"). Bundle 637 MB.
- Packaging bugs found by running the build: relative imports in the frozen entry point (→ launcher.py); `r/library` merged into R's own `R/library` on case-insensitive Windows (→ `rlib/`); a crashed windowed exe held the file and blocked the rebuild (→ self-test timeout).
- Deviation: R package versions are pinned by `r/packages.lock.json` (exact versions from the bundled library) instead of renv — the bundled library itself is the reproducible environment.

## Remaining from earlier phases
- [x] Mixed models inside the discovery engine: random intercept per participant, ML likelihood-ratio focal test, Δ marginal R², singular/convergence flags; LR statistic verified against statsmodels MixedLM (2026-11-21)
- [ ] Binary (logistic) mixed models (glmer) — not started
- [ ] Drill-down / Teaching Studio / stability for mixed-model discovery candidates (currently: View candidate code + refit in the Mixed workspace)

## Later
- Synthetic teaching dataset generator; mice; mediation workflow (lavaan); count/ordinal outcomes; psychometrics module (alpha/omega/fa).

---

## Review (Phase 1)
- Real HRS file: 16,000 × 168 imported; 82 value-labelled vars and missing-cell count match a direct pyreadstat read; project round-trip is lossless.
- GUI checked via offscreen renders with HRS loaded; fixed EDA layout stretch, default sort order, sidebar scrollbar.
- R verification found and fixed 3 real bugs:
  1. jsonlite `digits = NA` = 15 significant digits → numbers lost precision. Now `I(17)` (exact round trip).
  2. `R/bin/Rscript.exe` is a launcher (cmd.exe → bin/x64/Rscript.exe); killing it orphaned R. Locator now uses bin/x64 directly; kill uses `taskkill /T`.
  3. Old stdout reader thread pushed its EOF into the restarted worker's queue → false "worker exited". Each process now has its own queue.
- UI fixes: plot scales to panel; correlation subtitle no longer implies a single n for pairwise rs.
- Data note (HRS): `R_CESDM` = number of *missing* CES-D items, not a depression score; no CES-D total in file (needs derived variable, Phase 2). Activity frequency vars are declared nominal in the .sav but are ordinal; `R_COMP` code 7 "Don't do" isn't declared user-missing.
