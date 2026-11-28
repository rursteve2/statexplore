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

## Spec compliance pass (2026-11-21, after user report)
User hit: `Interaction plot: HHIDPN ~ HHID * PN + HHID * ADVTR_MATCHED * …` → raw R error (constant factor; ID variables used as predictors).
- [x] A. Pre-fit model checks everywhere (constant variables, one-category factors, >50-category "identifier-like" factors, ID-role variables as outcome/predictor, too few rows) + friendly translation of R errors; hide constant variables from predictor lists
- [x] B. §9.8 Mediation workflow (single mediator, bootstrap CI of indirect effect, non-causal note)
- [x] C. §8/§17 Missing-data visualization (EDA plot)
- [x] D. §10.8 Discovery table: teaching interpretation + validation status columns (validation status filled by stability runs)
- [x] E. §16.3 Teaching Studio: user-set values for held predictors
- [x] F. §15.5/§16.1 Random-effect (caterpillar) plot, residual autocorrelation plot, model-comparison plot
- [x] G. §10.3 "all eligible two-way terms" candidate model; §21 caching of already-fitted candidates
- [x] H. §25 Test C: planted three-way interaction found by discovery
- [x] I. `docs/spec-compliance.md`: every PDF section → status / evidence / deviation
- [x] Also: Cronbach's alpha (§5 optional psychometrics); README corrected (no installer exists yet; how to build/publish it)
- Result: 152/152 tests pass. Remaining open items are listed at the end of docs/spec-compliance.md.

## Release 1.0.0 (2026-11-21)
- [x] §28 20-step acceptance workflow executed on HRS: all PASS (docs/release-audit-v1.0.0.md)
- [x] Packaged full self-test OK; installer silent-install → self-test → uninstall OK
- [x] MIT license + third-party notices; repo scanned for secrets; made public; GitHub release v1.0.0

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

---

# v1.1 Upgrade (spec: `E:\Downloads\StatExplore_v1.1_Upgrade_Prompt.pdf`, started 2026-11-21)

Baseline: local repo at tag-equivalent commit `eb80db6` (= GitHub v1.0.0; no source zip supplied, so the repo is the baseline).
The supplied `hrs_least_missing_1000 (1).sav` is byte-identical to the v1.0 fixture.

## Design decisions (v1.1)
| # | Decision | Why |
|---|----------|-----|
| E1 | **Analysis treatments = a dataset *view*.** `measurement.apply_treatments(ds, treatments)` returns a shallow Dataset copy whose `VariableMeta` objects carry the *effective* measure for that analysis. Raw data, source metadata and project metadata are untouched. | Every existing validator, builder and the R load script already key off `measure`; the view makes them honour the treatment with no per-module special cases. |
| E2 | The R session signature already hashes variable metadata → a treatment change rebuilds R preparation state automatically. Split into two cached steps: CSV read (data signature) and coding (metadata signature). | §6.3 cache invalidation, without re-reading the CSV on every treatment switch. |
| E3 | Continuous treatment uses the stored numeric codes (no `factor()` line is emitted), never factor indexes. Load script lists every override as a comment block. | §5.1, §6.3, Test D. |
| E4 | Ordinal predictors keep treatment (dummy) contrasts as in v1.0; ordered factors are built only for ordinal *outcomes* (clm/clmm). | v1.0 results must not silently change (Test P). |
| E5 | Project measurement level stays in `VariableMeta.measure` (user-approved); `source_measure` is the file's level; new fields: `suggested_measure`, `suggestion_reasons`, `measure_status`, `category_order`, `coding_direction`. New `Measure.BINARY`. Old projects load with defaults. | §3.2, §18. |
| E6 | Continuous treatment of a project-*nominal* variable needs a second explicit confirmation ("codes are meaningful numbers"); strings can never be continuous. | §6 nominal rule, Test A. |
| E7 | R Console runs in its own Rscript process(es) (`console_worker.R`), never the analysis workers. Clean mode = fresh process per run. | §14.3/14.4 isolation. |
| E8 | User R packages go to `%LOCALAPPDATA%\StatExplore\R\user-library`; console libPaths = core library first, then user library; installing a package that shadows a core package is refused. Install from local .zip only (`repos = NULL`). | §15.2 core preservation, offline. |
| E9 | Multinomial mixed models: `mclogit::mblogit` (PQL), labelled approximate. GEE: `geepack::geeglm`. KR df: `lmerTest` + `pbkrtest`. | §11. |
| E10 | Packages added: ordinal, nnet, mclogit, geepack, survey, vcd, pbkrtest. Not added (base R suffices / not implemented): janitor, srvyr, gtsummary, marginaleffects, interactions, sandwich, clubSandwich. | §16: only ship what is implemented and tested. |

## Phases
- [x] P1 Measurement architecture: metadata fields + suggestions, approval, treatments view, codegen, R cache, project serialization; Measurement Review panel
- [x] P2 Variable selector widget (Analysis Builder, EDA; Discovery/Mixed/Survey in their phases) + categorical EDA (frequencies, crosstabs, 8 categorical plots, Spearman/Pearson rules)
- [x] P3 Categorical Association analyses (χ², Fisher, OR/RR/RD, phi, Cramér's V, reference controls)
- [x] P4 Discovery: explicit outcome family (gaussian/binomial/ordinal/multinomial), treatments, LR focal tests
- [x] P5 Mixed: glmer, clmm, mblogit, GEE, lmerTest inference setting
- [x] P6 Longitudinal preflight + Survey Analysis page
- [x] P7 R Console
- [x] P8 R Package Manager
- [x] P9 Research extras (Likert sensitivity, teaching finder), migrations, docs, packaging, release audit
- [x] Published: pushed main + GitHub release v1.1.0 (2026-11-22)
- [ ] User: install on a clean Windows 10/11 machine (networking off)

### Baseline (2026-11-21): 152/152 v1.0 tests pass.

### Review (P1 + P2)
- 187/187 tests pass (152 baseline + 35 new: test_measurement 12, test_eda_categorical 20, test_gui_v11 3).
- New: `analysis/measurement.py` (suggestion rules, approval, `VariableTreatment`, `apply_treatments` view), `gui/widgets/{treatment_panel,variable_selector}.py`, `gui/pages/measurement_review.py`.
- Changed: `core/variables.py` (fields, Measure.BINARY), `core/dataset.py` (category order, measure guards), `core/importers.py` (annotate), `rbridge/codegen.py` (read/prepare split + treatment comments), `analysis/session.py` (two-level cache), framework/models/mixed/teaching/validation/comparison (treatments), builders (rankable slots, Pearson rule), eda (frequencies, crosstab, categorical plots), EDA + Analysis Builder + Dataset Explorer pages.
- Numerical checks: R_LTACTX/R_SHLT continuous-treatment regressions == numpy OLS on raw codes (1e-8); frequencies == pandas; crosstab chi²/expected/Cramér's V == scipy (1e-10); Spearman on ordinal == scipy; reversed category order flips rho; recorded script re-run == result (1e-12).
- HRS suggestion audit: 25 activity/health/satisfaction/agreement/difficulty/education scales flagged ordinal (with direction); special codes called out (R_COMP 7 "Don't do", R_DRINKN 99 "All day"); identifiers kept nominal; RAEDYRS/RAMEDUC/RAFEDUC flagged continuous.
- Deviation: ordinal predictors keep dummy coding (E4) so v1.0 results do not change.

### Review (P3 + P4)
- 205/205 tests pass (+8 test_association, +8 test_discovery_families, +2 GUI).
- P3: `assoc_table` (r×c χ² / Yates on request / Fisher / Fisher Monte Carlo; Cramér's V, phi) and `assoc_2x2` (event/reference/exposed/comparison chosen explicitly; live 2×2 preview in the form; sample OR + Woolf CI, conditional MLE OR + exact CI, RR + Katz CI, RD + Newcombe hybrid-score CI, phi; zero cells reported, Haldane only on request). Verified vs statsmodels Table2x2 / confint_proportions_2indep, scipy fisher_exact / odds_ratio / association. Conditional MLE OR agrees with scipy to ~1e-5 (both root-finders; R uniroot tol 1.2e-4).
- P4: SearchSettings.family/event/reference/treatments; family suggested but must be confirmed (GUI); ordinal = ordinal::clm, multinomial = nnet::multinom, focal = LR test of nested models (joint over contrasts for multinomial); parameter limit counts thresholds / (K-1)×p. Run records `coding` (family, focal test, event/reference, outcome order, treatments). New Analysis Builder regressions `ordinal_reg` (with nominal_test PO check) and `multinom_reg` (car::Anova LR); logistic gets an optional event category. Drill-down maps every family. LR statistics == statsmodels OrderedModel / MNLogit (1e-5 / 1e-4).
- Bugs found: `self.event` attribute shadowed QWidget.event() → access-violation crash (lessons.md); AppState and discovery never deleted their temp folders (hundreds of `statexplore_*` dirs filled %TEMP%) → cleaned on close / after runs.
- Limitation: resampling stability / procedure CV not offered for ordinal/multinomial searches yet (clear message).
- Environment: drive C: is full (≈0.6 GB free). Old v1.0 `dist/` (1.2 GB) is regenerated by the v1.1 build.

### Review (P5–P8)
- 237/237 tests pass (+12 test_mixed_families, +3 test_longitudinal_preflight, +6 test_survey, +9 test_console_packages, +4 GUI).
- P5: MixedSpec.family/estimator/inference/event/reference/nagq/gee_corstr. glmer (OR, Wald + LR term tests, conditional vs population-average predicted probabilities, singular/convergence), clmm (cumulative logit, direction, joint Wald terms), mblogit (PQL, disclosed as approximate, joint Wald), GEE (robust SE, working correlation, QIC, estimand warning), lmerTest inference = Satterthwaite / Kenward-Roger / LRT (ML refits). Unsupported combinations rejected with reasons. glmer and clmm match independent numpy Gauss-Hermite ML fits (R nAGQ = 25) to 2e-4; KR == pbkrtest::KRmodcomp; balanced-design df exact; LRT == statsmodels MixedLM ML; GEE == statsmodels GEE.
- P6: `analysis/longitudinal_preflight.py` (records, participants, availability by time, 1 vs 2+ observations, duplicates, unobserved waves → analysis range; slope/spacing notes) shown live on the Mixed page; availability table also computed in R inside every longitudinal script (reproducible). HRS: R_LTACTX unobserved in waves 1–6, range Wave 7–16. `analysis/survey.py` + Survey page: svydesign / svrepdesign, explicit lonely-PSU policy, domain analysis by subset(), means, proportions, crosstab + Rao-Scott, svyglm linear/logistic with design F tests; weights checked; longitudinal pooling refused without explicit confirmation; design variables only suggested from labels (HRS: none — R_WEIGHT is body weight). Means, proportions, regression SEs == closed-form stratified Taylor linearization (1e-10).
- P7: `r/console_worker.R` + `rbridge/console.py` + R Console page: highlighted editor, run/stop, clean vs persistent sessions, output/messages/warnings/errors, PNG plots, open/save .R, objects, loaded packages, clear/restart, explicit Import Resulting Dataset (diff → confirm → snapshot .sxp → new project with history). Each run recorded in the project.
- P8: `rbridge/package_manager.py` + dialog (R Console and Settings): inventory core/user/R-base, local .zip/.tar.gz only (repos = NULL), DESCRIPTION checks (platform, R version, dependencies with versions, core shadowing, compiler for source), manifest with SHA-256, removal guarded by reverse dependencies.
- Bugs found and fixed: preflight info changed v1.0 models' status (now notes); Python-computed tables broke script reproducibility (moved to R); single plot JSON-unboxed to a string; console error call showed the console's own eval(); no zip tool on Windows for test packages (zip in Python).

### Review (P9 / release 1.1.0)
- 247/247 tests (adds test_research_tools 4, test_backward_compat 5, test_acceptance_workflow 1).
- §17: `likert_sensitivity` (== statsmodels OLS / OrderedModel), `analysis/interpret.py` (brief interpretation on every Analysis Builder / mixed result; Copy tables), `analysis/teaching_finder.py` (+ Teaching Studio button).
- §18: PROJECT_VERSION 2; first save over a v1 file keeps `<name>.v1-backup.sxp`; genuine v1.0.0 fixture generated from a worktree of eb80db6; old results reproduce to 1e-9; status-bar notices.
- Exports show per-analysis treatments (Excel, Word; PDF via shared HTML).
- §23 workflow automated (24 steps, real GUI, HRS) — all PASS; docs/release-audit-v1.1.0.md.
- Packaging: console_worker.R was missing from the PyInstaller spec (caught before building); self-test extended to every new engine + R Console; frozen build and silent-installed copy both pass `--selftest --full`; clean uninstall.
- Docs: README (new sections, 4A measurement levels, limitations), spec-compliance-v1.1.md, RELEASE_NOTES_v1.1.0.md, DEVELOPER.md v1.1 notes, release audit.
- Not published: GitHub release/push is outward-facing → awaiting the user's decision.

---

# v1.2.0 Upgrade (specs: `StatExplore_v1.2.0_Upgrade_Master_Prompt.md` = **M**, `StatExplore_Exhaustive_Interaction_Combination_Enhancement.md` = **X**; started 2026-11-22)

Baseline: local main `1673df9` (= GitHub v1.1.0 release commit 99aeb93 + a docs commit; no local v1.1.0 tag).

## Design decisions (v1.2)
| # | Decision | Why |
|---|----------|-----|
| F1 | **One model row, many hypothesis rows.** Each fitted candidate keeps its v1.1 model row (status, n, params, AIC…) and gains `tests`: a list of typed hypotheses (`focal_interaction`, `interaction_term`, `joint_interaction`, `main_term`, `overall`, `structure`). Ranking tables list hypotheses; "All attempts"/"Model comparison" list models. Old rows get one synthesized focal test. | M §4.3/§7.2/§11, X §8/§10: never one unexplained p per multi-interaction model; never mix hypothesis types. |
| F2 | **Multiplicity family = test type within the run** (BH/Holm per family; provisional when partial). | M §7.2 "define multiplicity families by hypothesis type"; X §12. |
| F3 | **Canonical model spec** (`discovery/modelspec.py`): terms = frozensets, hierarchically closed, + required; key = sorted terms; parse/display formulas; counts: distinct predictors, main / 2-way / 3-way terms, total terms; fixed coefficients from R's design matrix; random-effect parameters separately. | X §2/§14; M §7.1 dedup `X1*X2 + X1 + X3` → `X1*X2 + X3`. |
| F4 | **Strategies** (single choice): `combinations` = every hierarchical set of interaction terms (order ≤ max order) on every predictor subset (default, X §3/§16); `focal` = v1.1 grammar (Focal / Subsets covariate modes kept); `factorial` = full hierarchical factorial per subset; `main` = main effects only. `include_main` adds main-effects-only models (all subsets) to focal/factorial and can drop them from combinations. Max interaction order 2 or 3 (4-way deferred). Default max predictors 6 (both specs). | Both specs; X Test C/E. |
| F5 | **Strict CI criterion uses typed CI metadata** per test: `ci_applicable` (same_estimand / joint / unavailable), `ci_scale` (coefficient / odds ratio), `ci_null` (0/1), `ci_method`, `ci_level`, `ci_effect_id` = `p_test_id`. Logistic/ordinal CIs stored on the OR scale. Only 1-df single-coefficient tests can qualify. | M §4.2, §9. |
| F6 | **Fit identity hash** replaces `CACHE_FIELDS`: view data signature (metadata, treatments, filters, derived, holdout) + family + resolved event/reference + participant/time/time type/structure + sample strategy + variable pool + max_params + conf + kind; cached rows also require identical R/package versions. Highlight/sort/alpha/correction never refit. | M §5 (v1.1 ignored event/reference). |
| F7 | `df2` for mixed fits → `n_groups`; LRT rows carry `lrt_df`; legacy lmm rows renamed on load. | M §9. |
| F8 | Longitudinal discovery = explicit **Study structure** (independent / repeated) + ID + Time + one locked structure per search (Workflow A) or a structure comparison with a locked fixed formula (Workflow B). Engines: lme4 RI / RS / uncorrelated RS, nlme RI / RS / AR(1) / CAR(1), mmrm (marginal, labelled), glmer RI/RS, clmm RI/RS, GEE (gaussian/binomial; robust Wald). Fixed-effect tests: ML LRT (Gaussian, GLMM, CLMM), Wald (GEE, robust). Multinomial repeated outcomes: refused with reason (deferred). | M §8. |
| F9 | Treatment panel shows **every** model variable with a status (eligible checkbox / already continuous / unavailable + reason), outcome always first; view toggle for all eligible dataset variables (preconfigured, applied only when in the model); acknowledgements invalidated when the variable's coding fingerprint changes. Built on `VariableSelector`. | M §6. |
| F10 | Milestone 1 (sorting, highlight, cache, labels, treatment visibility) gets a local `v1.1.1` tag; one installer build at v1.2.0. Publishing (push/release) waits for the user. | M §14; outward-facing. |

## Phases (gates = tests pass before moving on)
- [x] 0 Baseline: 247/247 on a clean worktree of 1673df9 (E:\sx_baseline); v1.1 sort bug reproduced (numpy scalars + NaN unsortable by Qt)
- [x] 1 Numeric sort proxy (missing last both directions), sort/filter/selection preserved; highlight toolbar (adj p default, raw, strict, off; α presets/custom; highlighted-only; counters; warnings separate); project persistence of highlight prefs
- [x] 2 Fit identity + provenance (run_id, candidate_id, model_id, signatures, cache reason); typed CI/test metadata; n_groups vs df; tests: event/reference/treatment/structure changes never reuse
- [x] 3 Variable Treatment Manager (outcome section, all predictors with status, all-eligible view, ack fingerprint) in Discovery + Mixed (+ any other users); HRS R_LTACTX/R_SHLT tests
- [x] 4 Canonical spec + counts; strategies (combinations/focal/factorial/main); R multi-test evaluation; families by type; result views (hypothesis-type filter, model comparison tab, columns Predictors "3 / 6", Model terms, Estimated parameters); X Tests A–L; M Test D
- [x] 5 Longitudinal setup UI (study structure, ID/time validation + preflight) and Gaussian structures (lme4 RI/RS, nlme AR1/CAR1, mmrm); Workflow B structure comparison
- [x] 6 glmer / clmm / GEE discovery; R reference checks
- [x] 7 Discovery → Teaching Studio / Mixed handoff (exact terms, refit labelled); "Compare with main-effects model"; adjustable manual-model limits; exports carry new schema
- [x] 8 Regression suite, docs (README, DEVELOPER, release notes, spec-compliance-v1.2, feature matrix), version 1.2.0, build + frozen self-test

### Review (v1.2.0)
- Tests: 247 baseline → **315** (new: test_v12_results_table 16, test_v12_discovery 28, test_v12_longitudinal 14,
  test_v12_gui 5, test_v12_manual_limits 5). Four v1.1 assertions updated for deliberate changes (df2 → n_groups,
  hypothesis-table column, default strategy, ordinal repeated outcomes now supported) + version-agnostic title.
- Reference checks: X Tests I/J vs statsmodels (joint F, 1-df F = t², CI, whole-model F, per-type BH); every
  longitudinal engine vs a direct R fit (lme4 RI/RS, nlme AR1/CAR1, mmrm US, glmer, clmm, geepack Wald);
  workflow B REML logLik vs lme4.
- Bugs found and fixed: v1.1 sort (numpy/NaN), v1.1 cache ignoring event/reference, v1.1 df2 = participants,
  v1.1 panel dropping outcome; during v1.2: R reduced-fit cache reused across resampled rows, fit identity missing
  the GLMM/GEE event, mmrm AIC convention, workflow B validated against strategy rules, interpretation crash on
  unlabelled terms, empty-cell check pre-empting the R-side failure message.
- Performance: 311k hypotheses — sort 0.09 s (was 100 s), search 1.3 s (was 239 s), frames ≈ 5 s (was 48 s).
- Deferred (documented in docs/spec-compliance-v1.2.md): discovery Satterthwaite/KR option; stability / procedure
  CV for longitudinal searches; M §13 extras beyond the study-structure selector and chance counts; four-way
  interactions; separate v1.1.1 build (milestone-1 content = commit 3ef839a).
- User steps: clean-machine install; publish decision (push + GitHub release).

---

# v1.3.0 (spec: `E:\Downloads\StatExplore_v1.3.0_Development_Prompt.md`, started 2026-11-23)

Baseline: main `a72ddeb` (= GitHub v1.2.0). Fixtures: `hrs_least_missing_1000 (2).sav` (byte-identical to the v1.0–1.2
fixture) and `hrs_least_missing_1000.csv` (same 16,000 × 168 data; blanks are single spaces).

## Root causes found before editing (P0)
| Defect | Root cause |
|---|---|
| P0-01 A ordinal → binary | `analysis/comparison.build` only knows lm/glm: any non-Gaussian ModelSpec is refitted as `glm(y == level[2])`. Ordinal/multinomial candidates reach it via Add to Comparison / Compare with main-effects. |
| P0-01 B CAR(1) → lme4 RI ML | `discovery/detail.candidate_from_row` drops the row's `structure`; `spec_from_discovery` then falls back to `s.structure` ("ri") and forces `method="ML"`. |
| P0-01 C GEE → lme4 | same as B (workflow-B GEE rows lose their working correlation). |
| P0-02 `struct = ""` | same dropped field; `candidate_tests()` has no structure tests (they are computed in Python after the run), so the export has `tests = list()`. |
| P0-03 "N interactions" | headline counts `sig_raw/sig_adj` over each model's *primary* test, whatever its type, and always calls them interactions. |
| P0-05 CSV blanks | `pd.read_csv` keeps `" "` as a string → whole column object → STRING/NOMINAL. 144 HRS columns differ from the .sav. |
| P1-01 nlme blocked | `spec_from_discovery` sets `inference="lrt"` for every Gaussian model; validation rejects lrt for nlme/mmrm. |

## Design decisions (v1.3)
| # | Decision | Why |
|---|----------|-----|
| G1 | **`ModelDescriptor`** (`analysis/descriptor.py`): one serializable record built from a discovery row + run (family, engine, link, outcome levels/order/event/reference, terms, required, ID, time, time type, structure + time-index semantics, estimator ML/REML, sample vars, n, n_groups, logLik/AIC/BIC, parameter counts, warnings, tests). Every handoff (Full Analysis, Add to Comparison, Compare with main-effects, Mixed, View code, reports) is built from it, and the refit is checked against it (n, logLik, AIC within tolerance) with a visible "identity verified / differs" note. | P0-01 one authoritative descriptor; refits disclosed and verified. |
| G2 | Model Comparison fits each family with its own engine (lm / glm with the descriptor's event / ordinal::clm / nnet::multinom); refuses mixed families, different event/reference/order; LR tests only for nested models on identical rows. Longitudinal candidates are not sent to the single-level comparison: a specific message points to Discovery's model-comparison tab (same rows) or Mixed. | P0-01 A, P0-04. |
| G3 | Workflow-B candidate export = setup + fits of every structure the reported comparisons need + explicit LR/mixture test code; struct always the candidate's own id. Exported data reference = relative `data/<name>.csv` written next to the script with only the analysis columns (+ instructions); no AppData paths, no level vectors of ID columns. | P0-02. |
| G4 | Headline counts are per hypothesis family; "interactions" = individual + joint + focal interaction families only; covariance-structure wording; Holm count shown too. | P0-03. |
| G5 | CSV/XLSX: whitespace-only cells → NA, numbers trimmed; count of normalised cells recorded per column in the source options and shown. Measurement inference: numeric with ≤ 2 values → nominal (as before); integer-valued with 3–9 distinct values and no labels → nominal + flagged for review (rule 6: numeric coding alone is not continuous); else continuous. Stored project levels are never re-inferred. | P0-05. |
| G6 | Mixed handoff: inference default per engine (lme4 Gaussian → LRT on ML; nlme/mmrm → model's own df with ML/REML kept from the origin); validation messages explain KR is implementation-specific. | P1-01. |
| G7 | UI-28: discovery results area becomes a vertical splitter (tables / collapsible selected-model details pane); configuration and preflight in a horizontal splitter; sizes, collapsed states and column widths saved with QSettings; Ctrl+F find bar in code and report views; double-click opens details; copy table as CSV. | P2-02, UI-28. |

## Phases (v1.3)
- [x] 1 Baseline (315/315 on clean worktree E:\sx_v120_base) + pre-fix reproductions on HRS (Fixtures A, D, E; P0-01 A/B, P0-02, P0-03, P0-05) — E:\StatExplore_v1.3.0_Testing\reproduction_steps
- [x] 2 P0: descriptor + identity checks (G1), per-family Model Comparison (G2), structure-preserving exports with comparisons + portable script/data (G3), family headline (G4), CSV typing (G5)
- [x] 3 P1: nlme inference default (G6), time-index labels + observed waves + gap compression, usable/intermittent counts, MAR wording, model-based predictions, parameter columns + Estimation, interpretation wording, timeouts, nominal_test reason, GEE QIC, multinomial categories
- [x] 4 P2/UI-28: searchable dropdowns, pins, excluded-outcome reasons, find bar, Copy CSV, column menu/persistence, details pane + splitters, Mixed preflight pane, filter scope, strict note, budget history (G7)
- [x] 5 Tests (tests/test_v13.py), docs (README 5A, RELEASE_NOTES_v1.3.0, spec-compliance-v1.3), version 1.3.0
- [x] 6 Build installer + frozen self-test (pass); published v1.3.0 (push + GitHub release) on the user's go-ahead

### Review (v1.3.0)
- Tests: 315 baseline → **345/345** (tests/test_v13.py: 30; HRS fixtures A, D, E included). Two v1.2 GUI assertions updated for deliberate wording/column changes.
- Pre-fix reproductions confirmed every P0 root cause on HRS (repro_p0*.txt); the engines' numbers were already right (Fixture D AR(1) 20512.788 / CAR(1) 20540.106 / indep 20640.342; Fixture E p .7863; Fixture A 16/0, 5/0, 9/3, 18/14) — the defects were in transfer, export, reporting and import.
- Post-fix: every transfer verified against its origin (n, logLik, AIC); CAR(1) export sourced independently reproduces n 3901 / AIC 20540.106 / logLik −10264.053 / k 6 + the covariance LR test from .sav and .csv; HRS CSV now = SAV for all 168 columns.
- Found along the way: GEE QIC always NA in discovery (scope bug), holdout column broke the one-per-line load script (caught by tests), import heuristic needed time-name / negative / decimal exceptions.
- Limitations and open items: docs/spec-compliance-v1.3.md §4.

---

# v1.3.1 maintenance (spec: `E:\Downloads\StatExplore_v1.3.1_Developer_Prompt_Complete.md` = **S**, inventory
`StatExplore_v1.2.0_to_v1.3.0_Test_Inventory.md`; started 2026-11-23)

Baseline: main `a7bae5e` (= v1.3.0 + build-dir commit). Clean worktree `E:\sx_v130_base`. Fixtures: HRS `.sav`/`.csv`
byte-identical to v1.3.0's; `StatExplore_v1.3.0_Whitespace_Regression_Fixture.csv` (12×5, CRLF, tab cell).
Work/evidence folder: `E:\sx_v131_work` (kept out of the repo: health data).

## Root causes found before editing
| Defect | Root cause |
|---|---|
| STAT-15 dashes | `ordinal::nominal_test()` refits each term by `try(update(object, nominal = ...), silent = TRUE)`; `update()` re-evaluates the stored call (`data = sx_d`) outside our scope → "object not found", swallowed → NA rows. Same bug class as GEE QIC (lessons.md). Main-effects models fail too. Plain top-level R works (HRS W16 n917: RAGENDER p .4235, R_DEPRES p .8084). |
| STAT-16 BIC .004 | Discovery `_row()` recomputes BIC as −2ℓ + log(**n**)·k for every mixed engine; `BIC()` for REML `nlme` uses `attr(logLik, "nobs")` = n − p. 6·(log 3901 − log 3898) = 0.0046. |

## Phases
- [x] 1 Baseline suite on clean worktree (345 expected) + pre-fix reproductions (STAT-15/16)
- [x] 2 STAT-15 explicit per-term PO tests with reasons + global test; STAT-16 native nobs; P1-MODEL/TIME/EXPORT/SEARCH audits + fixes
- [x] 3 UI-28/29/30/31/32/33/34
- [x] 4 Provenance panel, p-value display, other §6 items as time allows (staged separately)
- [x] 5 Tests 26–35 (tests/test_v131.py), historical regression (HRS), traceability matrix, screenshots, docs, version, build on E:, frozen self-test

### Review (v1.3.1)
- Baseline 345/345 (clean worktree `E:\sx_v130_base`). New `tests/test_v131.py`: 28 tests (Tests 26-35), all pass incl. HRS.
- Root causes confirmed before editing (logs in `E:\StatExplore_v1.3.1_Testing\logs\*_before_v1.3.0.txt`): STAT-15 7/7 PO rows NA; STAT-16 BIC 20550.401 / 20577.720 vs native .397 / .716.
- Found during audits: per-model AR(1) index built before row filtering; Measurement-Review-independent selector bar forcing ~390 px (settings overflow at 1366 px, also in 1.3.0); auto layout overriding user divider drags; header min width lost on restoreState from older saved layouts.
- HRS evidence: cross-module 7/7 identity verified; exports from .sav and .csv 5/5 each self-verify in clean R (RI, AR1, CAR1 + LR, ordinal, multinomial).
- Not verified (configs not recorded): stage 6B, glmer n≈9874, clmm reduced panel AIC 8427.137, "cap 5 screens 16 of 20". Real-monitor DPI check: user.
- Docs: RELEASE_NOTES_v1.3.1, spec-compliance-v1.3.1 (traceability, §5.1/5.2 tables, limitations, run/install/rollback), README §10.
- Not committed / not published: awaiting the user.

---

# v1.3.2 (package `E:\Downloads\StatExplore_v1.3.2_Developer_Package`, master = 01; started 2026-11-25)

Baseline: tag v1.3.1 = 9153b96dc50f6678083753de43e207e091ab5dce (= archive commit marker; source archive not supplied,
so its SHA-256 cannot be checked). Clean worktree `E:\sx_v131_base` (r/library junction). Work/evidence: `E:\sx_v132_work`
(package copy in `pkg\`, SHA256SUMS verified OK). Independent references: `pkg\reference_run_001` (R 4.6.2; 353/353
selected historical targets PASS, 101 documentary; 1 reference fit fails in both optimizers: CD_Irregular 2L RS AR1).

## Design decisions (v1.3.2)
| # | Decision | Why |
|---|----------|-----|
| H1 | **One covariance-structure builder** (R helpers in the load-script header: `sx_re_lme4`, `sx_re_nlme`, `sx_cor_nlme`, fresh pd/cor objects per call) used by standalone Mixed, its comparisons, Discovery A/B and every export. Random type RI/DIAG/RS × residual Independent/AR1/CAR1 × optional cluster; clustered nlme = `list(sx_cluster = pdDiag(~1), sx_id = pd*(...))`, correlation grouping `~ t | sx_cluster/sx_id`. | STAT-21 root cause: three generators, slopes dropped the cluster. |
| H2 | Discovery structure keys stay (`ri`, `rs`, `rs_uncorr`, `ar1`, `car1`, `nlme_*`, …) + new `diag_ar1`, `diag_car1`, `rs_ar1`, `rs_car1`, `nlme_diag`; a candidate also carries `clustered` (bool). Settings gain `cluster`, `id_policy` (global / local), `cluster_compare` (B only, exploratory, off). Old keys/IDs unchanged → old projects and candidate IDs stay valid. | Deterministic IDs; 18 required combinations. |
| H3 | Participant key: clustered → composite `sx_id` = length-prefixed `(cluster, ID)` factor in (cluster, ID) order; global policy refuses an ID in 2+ clusters (crossing/migration) with the offending IDs; local policy declares nested labels. Missing cluster = excluded row (counted). Cluster-off exploratory fits keep the same composite IDs and rows. | 04 participant identity. |
| H4 | **Model identity v2** (`sx_identity()` in R, SHA-256 via `tools::sha256sum`, schema version recorded): ordered row IDs (source row numbers), row set, response values+levels, fixed design matrix (canonical column names, %.17g), grouping codes, factor coding; plus a canonical spec record. Verification = semantic first (any fingerprint difference → `mismatch`, listing which), then numeric (logLik/AIC/BIC) separately; missing fingerprints → `unverifiable`; refit error → `refit failed`. n / IC equality is never enough. | STAT-22 P0. |
| H5 | Historical runs snapshot the dataset view (variable metadata, derived, filters, holdout, treatments). A transfer whose current view differs is blocked with the differences named; the user may (a) rebuild the original view from the snapshot (same source bytes only) and refit, or (b) run an explicitly *new* analysis under current settings (labelled, never "verified"). | STAT-39 / CD-09. |
| H6 | **One LR-test service** (Python `analysis/lrtest.py` + R twin `sx_lr_test` in helpers, cross-checked by tests): reference distribution chosen by the hypothesis (RI→DIAG 0:1 mixture; RI→RS 1:2 mixture; DIAG→RS χ²1 only with interior variances; indep→AR1 χ²1; indep→CAR1 0:1 mixture; cluster off→on 0:1; AR1 vs CAR1 and simultaneous changes: no LR, IC only); |LR| ≤ 1e-6 negative → 0 with raw kept; material negative / non-finite / df ≤ 0 → no p, reason; either fit flagged → provisional. | STAT-17/23/41. |
| H7 | **One diagnostics extractor** `sx_diag(fit)`: optimizer + attempts, opt code, lme4 check code/messages, warnings (dedup identical text only), lme4 isSingular, nlme per-block relative covariance (min eigenvalue of G/σ² < 1e-6 or eigen ratio < 1e-6 or |r| > 0.999 → near-boundary; non-PD → singular), apVar status, Hessian messages. Status set: clean / warned / singular / near_boundary / failed / unknown. Retry nlme with optim only on optimizer errors; structural errors are not retried. | STAT-18/32. |
| H8 | R² = direct Gaussian decomposition (var(Xβ), Σ levels mean diag(ZGZ'), σ²) or NA with reason; never the package fallback that drops random variance. | STAT-45. |
| H9 | Family-aware comparison capability matrix (Gaussian ML/REML rules; GLMM/CLMM ML Laplace LR; GEE robust Wald (geepack anova) + QIC/QICu/CIC, never LR/AIC; mblogit none). Unsupported → disabled with reason. All-failed / single-success comparisons are explicit. | STAT-19/25/33/34. |
| H10 | Clustered Discovery validation (CD-11): holdout by organization (default when ≥ 10 clusters), participant, or forward time; whole search on training rows; holdout untouched; too few groups → unavailable, no row split. | CD-11. |
| H11 | Generated code: export bundle = script + consented analysis-columns CSV + manifest (SHA-256, package versions, spec, provenance) with relative paths and data/row assertions; refuses to claim reproducibility when the data file is absent. | STAT-36. |
| H12 | Developer runner `tools/sx_regress.py` (no user mode): runs references + app adapters, writes machine-readable results with PASS/FAIL/BLOCKED/SKIPPED/NOT RUN/NOT APPLICABLE, nonzero exit on any mandatory non-pass. | 05 runner. |

## Phases (v1.3.2)
- [x] 0 Baseline: suite on clean worktree, references (done), reproduce STAT-21/22/44/45/17/18 on v1.3.1 with fixtures
- [x] 1 Identity: STAT-22/39/40 (H4, H5), migration of old rows → unverifiable
- [x] 2 Construction: STAT-21 (H1–H3), STAT-17/19/20/33 (H6, H9)
- [x] 3 Diagnostics: STAT-18/32/34/41/44/45, STAT-23 (H6–H8), plot isolation
- [x] 4 Clustered Discovery A/B (04, matrix 12), UI-54/56, CD-01..12
- [x] 5 Reporting/UI: remaining STAT-24..30/35/36/38/42/43/46, UI-28..57
- [x] 6 Regression runner, full suite, references vs app, build on E:, frozen self-test, docs, ledger, release

### Review (v1.3.2)
- Baseline 373/373 (clean worktree). Final: 424 pass + 1 screenshot-only skip; runner 68/68 mandatory gates PASS
  (5 wording items N/A → manual checklist; clean VM and physical 150 % DPI = owner steps). Frozen self-test OK.
- Independent references regenerated on identical bytes; 353/353 historical targets reproduced.
- Root causes: STAT-21 three divergent random-part generators; STAT-22 identity = n/IC; STAT-44 postVar list;
  STAT-45 package R² fallback; STAT-46 RI count overwritten by an unmapped tests frame; UI-42 status text not in the
  frame; UI-31 grouping only on 2 of 8 lists.
- Performance: identity fingerprints first cost +268 % on cross-sectional searches (R's sha256 is slow); fixed by
  per-variable caching → +19 %.
- Evidence: docs/v1.3.2/, E:\sx_v132_work, E:\sx_v132_runs.

---

# v1.3.3 (package `E:\Downloads\StatExplore_v1.3.3` = MASTER_PROMPT + 01..07; started 2026-11-25)

Baseline: tag v1.3.2 = ff7b4f3c9b38a63e5c67f033eb4454872f99e6e7 (HEAD 93bc432 = + README only). Clean worktree
`E:\sx_v132_base` (r/library junction). Work/evidence: `E:\sx_v133_work` (package copy `pkg\`, SHA256SUMS OK).
Independent references (`pkg\reference\*.R`, R 4.6.2, all four scripts exit 0; 8 flagged fits = apVar / singular,
expected): 43B ML logLik −1647.9521623673, LR 63.5209122009, p 1.5867e−15; REML AIC 3324.0788 — match 07 anchors.

## Gate 0 reproductions on v1.3.2 (real Qt + R, `repro\repro_p0.py`, `repro\repro_48c.py`)
| Item | Result on v1.3.2 | Root cause |
|---|---|---|
| STAT-47 / V133-STATE-01 | after transfer the Interactions combo shows **None** | `fit_spec` puts the candidate's exact terms in `MixedSpec.terms` with `interactions="none"`; the form has no term editor |
| V133-STATE-02 | Fit unchanged / trajectories 40→20 + Fit → AIC 3377.425 = **reduced** model, identity none | `_fit` rebuilds the spec from widgets (`_spec()`): `terms`, `descriptor`, `source`, `sample_vars` dropped |
| V133-STATE-03 | ML→REML + Fit → REML reduced model | same |
| V133-STATE-04 | provenance/identity gone after any refit | same |
| V133-STAT-01 | time comparison after a transfer drops TIME×GROUP (k 7 vs 8); AR(1) kept in both tested routes (manual form, A and B transfers on 46A). The reported "independent lme4" substitution did **not** reproduce on ff7b4f3 (logged with inputs); variants are now verified from the fitted objects anyway | `_compare` also uses `_spec()`; with explicit terms, time-form variants would not remap time terms |
| V133-STAT-02 | 45A local IDs: preflight 8 participants / 48 duplicates | `preflight()` groups on the raw ID column |

## Design decisions (v1.3.3)
| # | Decision | Why |
|---|---|---|
| J1 | **ModelDraft in the Mixed workspace**: the page owns `origin` (immutable MixedSpec + descriptor from a transfer), `draft` (MixedSpec) and `fitted` (last FittedSnapshot spec). Widgets are a projection of the draft (populated under a binding guard, no handler mutates it). `_spec()` = draft + widget edits; terms always explicit. | 01 §1–2, STAT-47 |
| J2 | **Fixed-terms editor**: checkable list of every main effect and every eligible 2-/3-way term; presets (None / Time × each / All two-way) materialize term sets with a preview of additions/removals; an imported set shows "Custom". Hierarchy enforced (warn + block a missing lower-order term). | V133-STATE-05 |
| J3 | **Spec diff + lineage**: Fit compares the requested spec with the origin's canonical spec (plot settings excluded). Equal → origin kept, verified against the descriptor. Different → status "modified from origin" with the field-level diff, lineage kept, never "verified". "Return to origin" restores the origin draft. | 01 §1 identity |
| J4 | **Fitted-vs-requested check**: R returns the fitted object's actual fixed term labels, engine/class, random/grouping, correlation class, estimation method; Python compares with the requested canonical spec; mismatch → result FAILED with both shown. | 01 §1 "inspect actual fitted object" |
| J5 | **Comparisons keep the model**: every variant keeps terms (time terms remapped to the time form), random part, residual correlation, cluster, ID policy; the method change (REML→ML) is stated before and in the result; each variant row shows the fitted engine/random/residual/cluster read back from R and fails closed if it differs from the request. | V133-STAT-01 |
| J6 | **Composite participant key in preflight**: with a cluster, participant = (cluster, ID) in both policies; counts, duplicates, gaps and sizes on that key after complete-row restriction; label "composite (cluster, ID)". | V133-STAT-02, UI-04 |
| J7 | **AMD validation engine** (`discovery/validate.py`): Python builds a deterministic split manifest (numpy PCG64(seed), stable row IDs + composite group keys) for holdout / grouped K-fold (repeats) / bootstrap; targets independent rows, new participant, new cluster, new participant within existing cluster; fixed-candidate or full-pipeline (train-only search + prespecified selection rule); R fits with the search's own model helpers, predicts per target (population / cluster-conditional), returns per-row predictions and per-fold metrics; Python aggregates (observation- and group-weighted), selection/sign stability with both denominators. Stored with the run (project), exported as CSVs + R script. | V133-VAL-01, 02 §2 |
| J8 | **Portable Full Analysis bundle**: script with relative paths + consented data CSV + manifest (versions, provenance, identity, effective spec) + README; the script asserts data hash, rows, design columns and results; AMD manifest gets the same fields. | V133-EXP-01..04 |

## Phases (v1.3.3)
- [x] G0 baseline worktree, references run, P0 reproductions
- [x] G0 baseline suite result recorded (424 + 1 skip after rerunning 4 memory-exhaustion failures)
- [x] G1 J1–J4 model draft / term editor / diff / fitted-vs-requested (STAT-47 family) + tests
- [x] G2 J5 comparisons, J6 preflight, 18×2 matrix vs references, GLMM/CLMM/GEE vs references, diagnostics injection
- [x] G3 J7 validation engine + AMD UI, Test 51 grammar/adjustment/cancel-resume tests
- [x] G4 J8 bundles, UI items (V133-UI-01..09), persistence, benchmarks (+ J9 prompt cancellation)
- [x] G5 runner + gates, full suite, build on E:, frozen self-test, docs, disposition report, release

| J9 | Cancel kills the R workers of running discovery batches; interrupted candidates are not recorded; n_done = contiguous prefix; resume skips existing rows. | Benchmark found 369 s cancel latency (also in 1.3.2). |

### Review (v1.3.3)
- Tests 425 → 493: final runner 492 pass + 1 screenshot skip, 0 fail; 97 gates PASS (3 owner-only NOT RUN, 5 N/A
  wording). New: tests/test_v133.py (matrix 36 obligations + 18 standalone vs independent R; STAT-47 GUI; 48C; 45A;
  validation oracle/leakage/bootstrap/GEE/rows; bundles; history; diagnostics; Test 51), tests/test_v133_gui.py.
- Root causes: Full Analysis refit from widget defaults (STAT-47 + time comparison); raw-ID preflight; AMD dialog
  without result (null manifest); sx_diag declared ordinal for every model; cancel waited for batches.
- Found during the work: nlme corAR1 → corARMA(1,0) with index gaps; lme4::nobars deprecation warning downgraded
  clean fits; 'All two-way' + quadratic listed time × time²; TIME could be required and enter through Time × each.
- Not reproduced: 48C "AR(1) → independent lme4" (all tested routes kept AR(1) on ff7b4f3); hardened anyway.
- Evidence: docs/v1.3.3/, E:\sx_v133_work, E:\sx_v133_runs, E:\sx_v133_bench. Owner steps: clean offline VM,
  physical 150 % DPI (docs/v1.3.3/manual_acceptance_v1.3.3.md).

---

# v1.3.4 (handoff `E:\Downloads\StatExplore_v1.3.4_Developer_Handoff`; started 2026-11-27)

Baseline: tag v1.3.3 = 90305ec (spec names df24932: the code is identical, 90305ec only adds evidence text to
docs/v1.3.3/evidence_report_v1.3.3.md). Branch `release/v1.3.4`. Clean worktree `E:\sx_v133_base` (r/library
junction). Work/evidence: `E:\sx_v134_work` (handoff copy `pkg\`, `handoff_SHA256.txt`). Fixture hashes: 43B MD5
15b557b5… = owner record; Null_13 sha256 2975c585… = 52D project source. Owner decisions (2026-11-27): clean-VM install
and physical DPI deferred as owner steps (simulated DPI + same-machine frozen/offline evidence, labelled); 47A–D draws
regenerated with the deterministic v1.3.3 splitter and checked against the owner's per-draw evidence (labelled
reconstructed). 2026-11-27 amendment to V134-017: no VM required; physical DPI and clean-machine install are
separate owner-approved deferrals if unavailable; never reported as PASS; they do not block other gates.

## Gate 0 runtime reproductions on v1.3.3 (E:\sx_v134_work\repro, logs)
- 51H3: regenerated draw manifest == owner export (32,000 rows); baseline selections == owner app [17,13,13,...]
- 47B3: config = participants-in-clusters, rs_ar1, local IDs, seed 133 (draw-1 logLik -1671.83698173322 vs owner
  -1671.83698173324); baseline selections and 7 failed draws == owner app

## Root causes found in Gate 0 (source; runtime reproductions in docs/v1.3.4/evidence)
| ID | Root cause |
|---|---|
| V134-001 | `sx_cache_for` keys the reduced/null cache by rownames only; `sx_val_boot` renames every draw b1..bn, so a 10-draw chunk reuses draw 1's reduced fits |
| V134-002 | `sx_lr_test` computes log p but `sx_compare` keeps only `p`; F / Wald tests never compute it; BH/Holm/ranking/selection on ordinary p |
| V134-003 | `stability_table` groups by display label; labels are not unique |
| V134-004 | validation pool = `run.rows` non-invalid (outcome- and stop-dependent); early stop not replayed |
| V134-005 | `sx_fit_model` GEE uses global `sx_disc_corstr` (first structure) for every structure |
| V134-010 | reduced-model flags appended to the row; batch-level warning handler attributes any fit's warning to the row; a cache hit emits no warning |
| V134-011 | `sx_val_coef` returns `cols[1]` |

## Plan
- [x] G0 baseline suite + red reproductions (001/002/003/004/005/009 + 007 counts)
- [x] G1 fit context + content-keyed, per-sample caches; log-domain inference/adjustment/ranking; GEE corstr + readback + QIC/QICu/CIC; comparability; FIT_RULES 4; legacy-identity no-reuse
- [x] G2 candidate identity in stability; training-only pipeline with replayed stopping; time-eligible grammar (opt-in); validation manifest identity/realized sizes/portable script; per-fit diagnostic attribution; multi-coefficient estimands; workload caps
- [x] G3 resume after reopen (reason states); Educational vs Complete R code; audit JSON + report; migration
- [x] G4 candidate reordering; teaching wording; code Ctrl+F; DPI simulation
- [x] G5 full suite + references + benchmarks + build + frozen self-test
- [x] G6 docs (register, evidence report, release notes), release

### Review (v1.3.4)
- Tests 493 -> 538 (537 pass + 1 screenshot skip); 44 new tests, 37 red on v1.3.3. Register 111 rows: 103 PASS,
  5 N/A, 3 NOT RUN (owner-approved deferrals). Frozen self-test OK. Benchmarks within 20 % (54B +5.7 %); 51H3
  bootstrap 48 -> 79 s because wrong cross-draw reuse is gone.
- Not reproduced: V134-009 greyed Resume (saved project reopens resumable on v1.3.3 too). Partial: 47C exact counts
  (settings not supplied). Evidence: docs/v1.3.4/, E:\sx_v134_work.

---

# v1.3.5 corrective patch (prompt `E:\Downloads\StatExplore_v1.3.5_CORRECTIVE_PATCH_MASTER_PROMPT.md`; started 2026-11-27)

Baseline: tag v1.3.4 = dd7313aebfaf8991e4b921feb42a3a37cf355fe6 (verified); source zip SHA-256 860b62cf… matches the
prompt. Branch `release/v1.3.5`. Clean worktree `E:\sx_v134_base` (r/library junction). Work/evidence `E:\sx_v135_work`.
Handoff reused: `E:\Downloads\StatExplore_v1.3.4_Developer_Handoff` (50 files hashed in logs/G0_handoff_SHA256.txt).
Owner decision 2026-11-27: clean-machine install deferral APPROVED for v1.3.5 (explicitly, in this session); physical
DPI recorded NOT RUN (no VM required).

## Design decisions (v1.3.5)
| # | Decision | Issue |
|---|---|---|
| K1 | One R primary-score helper `sx_primary_logp(r)` (log tail of tests[[1]], NA when withheld / not ok) used by the legacy stability + procedure CV and the AMD selector; legacy thresholds/argmin on log p; p kept for display + log10 column | 01 |
| K2 | `discovery_ranking` maps every p column to its log twin (p, p_adj, p_bh, p_holm); adjusted rankings are within test-type family; ties: raw log p, idx, test id; -inf (unknown magnitude) labelled; GUI log10 column sorts -inf first | 02 |
| K3 | `DiscoveryRun.compatibility()` = one resume gate: fit rules, engine provenance, fit identity, grammar, data signature, row keys/indices/progress; worker versions of ALL workers checked in `execute` before any mutation; refusal leaves the run untouched; `restart_from(run)` = new search (new ID, parent link) refitting everything | 03 |
| K4 | Minimum CANDIDATE predictors applies to the focal strategy (pairs/triples/subsets); GRAMMAR 4; grammar_problem covers it; label says "candidate (required not counted)"; disabled for workflow B | 04 |
| K5 | Fixed-candidate validation resolves the saved row (focal/covariates/kind/inter/structure) and verifies its key; no enumeration; R verifies the fitted terms; pipeline validation refuses on grammar change or pool-key mismatch | 05, 08 |
| K6 | Validation pool scopes: original / cumulative (default for mode D) / space / prefix; from declared execution budgets; adaptive + truncated-continuation disclosure; legacy `declared` = original | 06 |
| K7 | Audit: `historical` block from the run's snapshot (never current file); optional labelled `current_context`; validation/portable export refuses runnable bundles unless the current view reproduces the search signature (or rebuilt original) — else metadata-only | 07 |
| K8 | Dispatch checks cancel/deadline before every batch (cached or not) and before committing cached rows; injectable monotonic clock; precedence cancel > deadline > budget > exhaustion | 09 |
| K9 | Validation workload = conservative uncached bound (exact per-candidate plan for a materialized pool, analytic bound otherwise) + retry/aux factors; hard R-side fit-invocation cap with a budget condition; counts used/retries/aux/cache hits | 10 |
| K10 | Reproduction script binds and checks predictions, folds (sizes), selections+keys, bootstrap coefs, trace; per-check CSV + nonzero exit; required-file presence by mode; NA-safe failed-draw comparison | 11 |

## Plan
- [x] G0 baseline suite on dd7313a + red reproducers for 01–11 (E:\sx_v135_work\repro)
- [x] G1 K1 K2 K3 K4 K5 + tests
- [x] G2 K6 K7 K8 K9 K10 + tests, GUI messages
- [x] G3 full suite, independent R refs, benchmarks, build, frozen self-test
- [x] G4 docs/v1.3.5 (register, traceability, evidence, methods, acceptance 55–61), release

### Review (v1.3.5)
- Baseline 537 + 1 skip; red: tests/test_v135.py 71 fail on dd7313a. Final 747 pass + 1 screenshot skip, 0 fail.
  Register: 11 AUD + 3 discovered PASS; 17 V134 groups + 92 inherited PASS. Benchmarks within ±3 %. Frozen self-test OK.
- 52D is now refused (fit rules 3) with 'New search from this one'; v1.3.4 tests expecting its resume were updated.
- 55L fixture (n 6000) for legacy underflow: 51C/54B are longitudinal (legacy workspace is single-level).
- Open: physical DPI NOT RUN; clean install = owner-approved deferral; in-place upgrade = owner step.

---

# v1.3.6 corrective patch (handoff `E:\Downloads\StatExplore_v1.3.6_Developer_Handoff.zip` + supplement; started 2026-11-28)

Baseline: tag v1.3.5 = b7610a0 (source zip 20376a44… = tag tree, 339 files byte-identical; 21f698a→b7610a0 docs only).
Branch `release/v1.3.6`. Clean worktree `E:\sx_v135_base`. Work/evidence `E:\sx_v136_work` (pkg\ = extracted handoff).
Owner decisions 2026-11-28: clean-machine install = owner-approved deferral (standing, never ask again); release as a
normal "Latest" GitHub release. Physical DPI stays NOT RUN (no new approval).

## Design decisions (v1.3.6)
| # | Decision | Finding |
|---|---|---|
| M1 | `core/identity.py`: content identity schema 3 = SHA-256 over canonical typed columns (kind, little-endian float64 bits / int64 datetimes / length-prefixed UTF-8 text, null mask), column names+order, row count/index, source format + parse options (sheet, delimiter, encoding, reshape) + source file SHA (conservative). Cached per Dataset ONLY for the same raw object (`is`) + shape/dtype guard; raw is immutable by contract (all writers make new Datasets) | 06 |
| M2 | View identity = content id + variable metadata + derived specs + filters + holdout spec + REALIZED filter/holdout masks + derived values; `data_signature` = view identity (schema 3); `fit_identity` identity_schema 3; runs record `identity_schema` 3; legacy (<3) runs are UNVERIFIED: no cache, resume, rebuild or runnable reproduction; history viewable; "New search from this one" | 06 |
| M3 | Snapshot schema 3 adds `content_id`; `rebuild` only after the current content id equals it; `context` compares content ids, never only the source file hash | 06 |
| M4 | RSession: no `id(ds.raw)`; worker data keyed by content id + CSV SHA-256 of the bytes written; every result gets `analysis_provenance` (schema 1: content id, view id, input CSV SHA-256, rows, script SHA-256, snapshot) | 06/07 |
| M5 | Historical export resolver: regenerate the full worker-input CSV from the current view, else from the snapshot rebuild (content id must match), and compare its SHA-256 with the provenance; else refuse (results/report only + reason). One sub-bundle per result (own script/data), never one shared data.csv | 07 |
| M6 | `export/bundle.py` transaction: every export goes to a NEW child folder (never reused), staged in a hidden sibling and renamed on success; cleanup on failure; a central registry is the only writer API; analysis data only via `tx.write_data` and only with explicit consent; inventory check: no unregistered or undeclared data file; hash scratch outside the bundle; manifest `bundle_kind` = results-only / code-only / runnable | 03 |
| M7 | Reports: explicit per-export data consent (default No); R/Py scripts per result in their own sub-bundle via the portable writer | 03/07 |
| M8 | Validation ledger: immutable plan (parts), attempts log, committed chunks only after a validated response; uncertain chunks after errors; counters lower-bound/unknown; R budget `denied` flag (any denial in a part → that part `budget`); status from evidence: complete iff every planned part committed and succeeded and no denial (used == cap is complete); stop before dispatch when remaining = 0 and work remains | 04/05 |
| M9 | Reproduction: replay committed chunks only; split/draw manifests verified per scope: committed vs replay, pending vs embedded plan; `execution_ledger.csv` verified exactly; coverage reported (never "complete" for partial) | 04 |
| M10 | Typed evidence: Python writes every evidence CSV from a declared schema (fixed columns, round-trip floats, Inf/-Inf, '' = missing), header-only when empty; generated R reads all as character, parses per declared type (int/num/text/bool), signed infinities, malformed → FAIL, missing only where the schema/reproduced status allows; length-prefixed composite keys; `sx_add_chk` NA → FAIL, NOT APPLICABLE only when declared | 01 |
| M11 | Independent recomputation in R: stability + estimands from reproduced draws/coefs + declared pool (index/key/label from setup) + embedded alias map for coefficient labels; realized sizes from manifest + R grouping; ID partitions (app participant/cluster keys ↔ R groups) bijective; trace df/ref loglik/flags/rows; folds/boot reasons, flags | 02 |

## Plan
- [x] G0 baseline suite on b7610a0 (747 + 1 skip) + red reproductions 01–07 (auditor scripts 9/7, 6/2, 13/4; real R 04/05/06/07, script-R 01/02)
- [x] G1 M1–M7 (identity, provenance, export transactions, consent) + tests incl. real R, GUI, migration
- [x] G2 M8–M11 (ledger, budget, typed verifier, independent recomputation) + tests incl. real-R tamper suite
- [x] G3 full suite (818 + 1 skip), build 090d120, frozen + installed upgrade checks, acceptance kit 131/131, benchmarks
- [x] G4 docs/v1.3.6 (register, traceability, evidence, methods, acceptance 55–61), release

### Review (v1.3.6)
- First full run 812 pass / 6 fail (schema required train_score for fixed candidates; reason order; 3 test mistakes);
  final 818 pass + 1 screenshot skip. Register 138 rows all PASS, 607 exact test-node links (12 unlinked + 5 ambiguous
  inherited requirements resolved). Real-R red on v1.3.5 for 04/05/06/07 and 10/10 tampers accepted (01/02).
- Found: V136-D1..D6 (portable consent default, swallowed budget denial, id(raw) keys, holdout realized split, legacy
  validation records, a DPI regression caught in development). Runner bug (vacuous pass) caught by reading results.
- Legacy policy: identity schema 2 searches / results unverified (no cache, resume, rebuild, runnable bundle).
