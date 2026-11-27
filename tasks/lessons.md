# Lessons

## R on Windows
- `R/bin/Rscript.exe` is a launcher that spawns `bin/x64/Rscript.exe` through cmd.exe. Always run the x64 binary and kill process trees, or timeouts leave orphaned R processes (they also lock `worker.R`).
- `jsonlite::toJSON(digits = NA)` gives only 15 significant digits. Use `digits = I(17)` when numbers must round-trip exactly.

## Process/thread hygiene
- Reader threads must capture their own queue or connection at start. Never look up `self.<queue>`, because a restart replaces it and stale events leak into the new worker.

## Verification
- Tests that skip (missing R) are not passing tests. Run the full suite with every dependency present before calling a phase done.
- Read the variable label before interpreting a variable (HRS `R_CESDM` is a missing-item count, not a score).

## R code generation
- Never pass two strings where R expects one: `f(notes = "a", "b")` silently binds "b" to the next positional arg. Use paste0() for multi-part text, and make parsers reject shape mismatches so tests catch it.
- R's multi-model `anova(m1, m2, m3)` tests each step against the LARGEST model's residual variance. For hierarchical F-change, compare consecutive pairs.

## Tests
- pytest-qt closes `qtbot.addWidget` widgets BEFORE fixture teardown; a dirty MainWindow then blocks on a modal dialog. Own the window's lifetime in the fixture.
- Count overlapping missing rows as a set when writing expected n.

## Model identity
- Interactions are sets, not tuples: compare `frozenset(term)` whenever terms come from different sources (UI order vs stored order).

## UI defaults
- When a list refreshes, keep the user's check states AND default newly added items to checked; otherwise "add then compare" silently compares fewer items.

## Labelled codes
- Categorical numeric variables arrive in R as labelled factors; never `as.numeric(as.character(f))` them. Map labels back to codes explicitly.

## Optimizers
- Prefer the package-default optimizer; add fallbacks explicitly and record when they are used.

## Packaging
- A PyInstaller entry script runs as top-level `__main__`: relative imports fail. Use a tiny launcher with absolute imports.
- Windows paths are case-insensitive: `r/library` and `R/library` are the same folder. Never name bundled folders so they differ only by case.
- Verify the frozen build by running it (`--selftest`), not just by building it.

## Flaky vs real
- A test that fails once under heavy load must be rerun and investigated; report it even if it doesn't reproduce.

## Commands I give the user (WSL terminal)
- Never give `cmd.exe /c "\"C:\path with spaces\prog.exe\" args"`: the escaped quotes don't survive. Give the WSL path instead: `"/mnt/c/Program Files/.../prog.exe" args`, or `cmd.exe /c "winget ..."` only for commands without inner quotes. (Mistake made twice: Rscript, gh auth login.)

## Qt attribute names
- Never name a widget attribute after a QWidget/QObject virtual (`event`, `show`, `close`, `update`, `layout`...): `self.event = QComboBox()` shadows `QWidget.event()` and Qt crashes with an access violation on the next event. Use suffixed names (`event_combo`).

## JSON from R
- `jsonlite::toJSON(auto_unbox = TRUE)` turns a length-1 vector into a scalar: a list of one plot path arrived as a string and Python iterated its characters. Wrap vectors that are semantically lists in `I()`, and make the Python side accept a scalar too.

## Disk space
- Drive C: is nearly full on this machine; E: has the space. Run tests with `--basetemp` on E: and put builds/scratch on E:. A full C: made R workers exit at startup with no error text.

## Caches in R workers
- A per-worker cache of fitted statistics must be keyed to the exact rows, not just the formula: other code (stability, procedure CV) re-points `sx_dd` at resampled rows. Validate the cache against `rownames(d)` before reuse. (v1.2: caught by test_stability_and_procedure_cv.)

## pytest naming
- Never import a helper whose name starts with `test` (e.g. `tests_frame`) into a test module: pytest collects it. Alias it.

## Test runs vs live edits
- The editable install imports the working tree: a long pytest run started before edits silently mixes old and new code. Run baselines from a clean worktree (PYTHONPATH to its src, r/library junction), and rerun the suite after the last edit.

## R calls that re-evaluate themselves
- Functions like `geepack::QIC`, `update()`, `anova()` re-evaluate the stored model call: any local variable named in it (`f`, `d`) must be reachable (`QIC(m, env = ...)`). A swallowed error here looked like "QIC not available" for two releases.

## Heuristics on imported data
- A measurement-level heuristic (e.g. "few integer codes = categorical") must be checked against time variables, negative values and decimal-written floats before shipping; run the whole suite, which is full of small synthetic CSVs.

## Drive C: vs E: (correction, 2026-11-23)
- The repo is ON C:. A build or tool that writes inside the repo (PyInstaller dist/build, release zips) writes to C: even when TEMP points at E:. Before any build, set `STATEXPLORE_BUILD_DIR=E:\statexplore_build`; before any test run, `--basetemp` and TEMP/TMP on E:. Check every output path, not just temp.

## R functions that swallow their own errors (v1.3.1, STAT-15)
- `ordinal::nominal_test()` wraps each refit in `try(update(...), silent = TRUE)`: inside a function scope every refit fails and the table is silently NA. Same class as `geepack::QIC`. Never call R helpers that refit via `update()` on a model fitted to local data; refit explicitly and report the reason per row.

## Native statistics
- Take AIC/BIC from the fitted object. Re-deriving them in Python with one "convention" (log n) silently disagreed with R's REML log(n − p) by 0.004. When a convention must differ (mmrm), label it on every row.

## Index columns built before row filtering
- Any derived index (occasion rank, factor levels) must be rebuilt after the rows of THIS model are chosen; otherwise per-model samples and Full Analysis disagree.

## Screenshots
- Offscreen Qt has no fonts on Windows: set `QT_QPA_FONTDIR=C:\Windows\Fonts`, or every glyph is a box.

## QSettings in tests
- `QSettings("Org", "App")` ignores `setDefaultFormat(IniFormat)` and uses the Windows registry: tests read the user's real saved layouts. Re-apply invariants (e.g. minimum section size) after `restoreState`, and don't trust test isolation for settings.

## Inserting into generated code
- Insert lines before wrapping/formatting, never after: the wrapper had split the anchor statement across two lines and the inserted check broke the script.

## cmd.exe quoting (third time)
- Paths with spaces as program + args: write a .bat and run `cmd.exe /c 'E:\path\x.bat'`. Never `cmd.exe /c '"C:\...\python.exe" ...'`.

## Source snapshots for test runs (v1.3.2)
- A copy of `src/` on E: needs the `r\library` junction next to it (the package finds R packages relative to the
  repository root): without it every R test errors at the fixture. Use a .bat with `mklink /J` (cmd quoting).

## Identity fingerprints and speed (v1.3.2)
- `tools::sha256sum` is ~20 MB/s: hashing a whole design matrix per discovery candidate tripled search time. Hash
  each variable once per sample (cached) and the per-model design as (canonical column names + variable hashes).
  Profile with Rprof before guessing; per-element `writeBin` loops were the other cost.

## Labels in generated R
- Factor levels in R are de-duplicated value labels ("Label [2]"). Any R comparison with a category must use
  `factor_label()`, never `ds.level_label()` — a duplicated event label silently made a binary outcome constant.

## Tests that index list rows
- Don't index `QListWidget.item(i)` in tests when the widget may insert header rows; use the selector's `items()`.

## Ask "nested?" before testing structures
- Two structures differing in more than one component (e.g. RI-independent vs RS-AR(1)) have no LR test; a test
  fixture that expects one is wrong, not the app.

## Forms that hold a transferred model (v1.3.3, STAT-47)
- Never rebuild a model from widgets on Fit. Keep the loaded spec as a draft, populate controls under a binding guard,
  and build the request as draft + explicit edits; compare the request with the fitted R object before showing results.

## nlme converts corAR1 (v1.3.3)
- `corAR1` with gaps in the integer covariate is silently turned into `corARMA(p = 1, q = 0)` (same model). Any
  read-back of the fitted correlation class must accept that form, and only that form.

## Deprecation warnings change result status
- `lme4::nobars` now warns (moved to reformulas): an R warning downgrades a clean result to WARNING. Use
  `formula(fit, fixed.only = TRUE)` for merMod and look up optional helpers with `getExportedValue()`.

## Declared dependencies (v1.3.3, V133-EXP-04)
- Scanning a script for `pkg::` declares every package a shared helper mentions. Declare what the analysis loaded
  (result.engine.packages / family list) and write optional helper calls as `getExportedValue("pkg", "fun")`.

## Cancellation must not wait for running work (v1.3.3)
- A cancel flag checked only between batches waited ~6 minutes for 40-candidate nlme batches. Kill the workers of
  in-flight batches, record nothing for them, set the contiguous done prefix, and make resume skip existing rows.

## Low memory on this machine
- Chrome / BlueStacks leave ~2-3 GB free: parallel pytest + repro runs failed with "paging file is too small" and
  OpenBLAS allocation errors. Run heavy jobs one at a time and set OPENBLAS_NUM_THREADS=1 in helper .bat files.

## pytest.main() from a helper script (v1.3.5)
- `python runner.py` -> `pytest.main()` does NOT put the repo root on sys.path (`python -m pytest` does): every test
  that imports `tests.<module>` errored with "No module named 'tests'" (12 false failures). Insert the repo root into
  sys.path in any runner script, and compare a new runner's result with a `-m pytest` baseline before trusting it.

## Fixtures for "below the double range" (v1.3.5)
- Tune an underflow fixture with the EXACT test distribution in R (`pf(..., log.p = TRUE)`), not a chi-square
  approximation (it overstated the tail by ~30 %), and check the margin on the resampled sizes (80 % subsamples) too.
