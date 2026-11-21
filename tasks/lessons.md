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
