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
