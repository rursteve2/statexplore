# StatExplore — Technical Architecture

```
┌──────────────────────── Python process (PySide6) ────────────────────────┐
│ gui/            MainWindow · sidebar · pages · Qt table models           │
│   state.py      AppState: project, signals, background pools             │
│                   ├─ io pool (2 threads): file import                    │
│                   └─ R pool (1 thread): every R request, serialised      │
│ analysis/       eda.py (describe, plots) → session.py (RSession)         │
│ rbridge/        codegen.py (only place R text is built) · worker.py      │
│ core/           importers · dataset · variables · overview · schema ·    │
│                 project (.sxp SQLite)                                     │
└───────────────┬───────────────────────────────────────────────────────────┘
                │ stdin: JSON lines  {"id","cmd","args"}
                │ stdout: "@@SX@@ {json}" lines; other output = log
┌───────────────▼──── Rscript --vanilla r/worker.R (persistent) ───────────┐
│ .sx_env: `dat` (loaded once per data/metadata change), `sx_names`        │
│ run: eval(script) → sx_result → jsonlite::toJSON(digits = NA)            │
│ warnings captured · errors returned (worker survives) · package versions │
└──────────────────────────────────────────────────────────────────────────┘
```

## Data flow
1. **Import** (`core/importers.py`): CSV (delimiter sniffing, encoding fallback), SAV (pyreadstat with
   `user_missing=True`), XLSX (sheet choice). Source file opened read-only; SHA-256 recorded.
2. **Dataset** (`core/dataset.py`): `raw` frame is never mutated. Metadata edits (display name,
   measure, role, reference level) go through `set_variable_field` and are logged in `changes`
   (undoable). SPSS user-missing codes stay in `raw`; `analysis_column()` maps them to NA.
3. **To R** (`analysis/session.py`): analysis frame written as CSV with alias headers
   (`v001…`), then `codegen.data_load_script()` builds explicit R code (read.csv with colClasses,
   `factor(levels = codes, labels = value labels)`, `relevel`). Re-run only when data, metadata
   or the worker process changes.
4. **Analysis**: Python renders an R body; worker evaluates it; Python wraps the response in an
   `AnalysisResult`. `result.script` = load script + body — exactly what ran (§18).
5. **Project** (`core/project.py`): one SQLite file: meta, variables, change log, Parquet data copy,
   results (JSON), artifacts (PNG bytes). Atomic save (temp file + replace); opened read-only.

## Safety rules
- Variable names/labels/levels from files reach R only via `codegen.r_str()` (escaped, ASCII-only).
- Identifiers in R are internal aliases validated by `codegen.alias()` (`^v\d{3,}$`).
- R is started with an argument list (no shell). Requests are JSON on stdin.
- A request timeout kills the worker; the next request restarts it and reloads data
  (tracked by `RWorker.generation`).

## Result schema (`core/schema.py`)
`AnalysisResult{id, kind, title, engine{engine, version, packages}, script, inputs, status
(ok|warning|invalid|failed|cancelled), n_used, missing_data, tables[Table], values, plots,
messages, seed, elapsed_s, created_at, schema_version}`. Phase 3 search candidates will reuse it,
adding family-specific numbers in `values`.

## Engine policy
R is the single statistical engine (decision D1 in `tasks/todo.md`). Python computes only
overview counts (missingness, uniques, Tukey outlier counts) and is used in tests as an
independent reference implementation.
