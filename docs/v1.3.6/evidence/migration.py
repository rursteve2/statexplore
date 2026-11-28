"""Migration evidence. `make <dir>` (run with v1.3.5 on PYTHONPATH) builds a v1.3.5 project with a partial search, a
holdout t-test result and an AMD validation run. `check <dir>` (run with v1.3.6) opens it, records what v1.3.6 does
with each historical record, saves, reopens and continues with a linked new search."""
import json, sys, tempfile
from pathlib import Path
import pandas as pd
import statexplore
from statexplore.core.importers import import_file
from statexplore.core.project import Project
from statexplore.core.dataset import HoldoutSpec
from statexplore.discovery import engine, validate as V
from statexplore.discovery.space import SearchSettings
from statexplore.analysis.session import RSession
from statexplore.analysis.builders import BY_KEY
from statexplore.analysis.framework import run_analysis
from statexplore.rbridge.worker import RWorker
mode, d = sys.argv[1], Path(sys.argv[2]); fx = Path(sys.argv[3])
S = dict(outcome="y", candidates=["x1", "x3", "x2"], family="gaussian", strategy="focal", covariate_mode="focal",
         max_predictors=2, workers=1, mode="D", max_models=2, max_minutes=None)
if mode == "make":
    ds = import_file(fx / "StatExplore_Test55L_LegacyUnderflow.csv")
    p = Project(ds)
    run = engine.create_run(ds, SearchSettings(**S)); engine.execute(run, ds); p.searches.append(run)
    full = engine.create_run(ds, SearchSettings(**{**S, "mode": "C", "max_models": None})); engine.execute(full, ds); p.searches.append(full)
    with RWorker() as w:
        s = RSession(w, Path(tempfile.mkdtemp()))
        v = V.ValidationRunner(s, ds, full, V.ValidationConfig(target="rows", modes=["holdout"], seed=3, max_fits=10000)).execute()
        full.validation_runs.append(v)
        r, a = run_analysis(s, ds, BY_KEY["t_one_sample"], {"dependent": ["y"]}, {})
        p.add_result(r, a)
    p.save(d / "v135_project.sxp")
    print(json.dumps({"made_by": statexplore.__version__, "partial": run.id, "complete": full.id, "result": r.id}))
else:
    out = {"opened_by": statexplore.__version__}
    p = Project.open(d / "v135_project.sxp")
    part, full = p.searches
    out["identity_schema"] = [part.identity_schema, full.identity_schema]
    out["partial_resume_reason"] = part.resume_reason(p.dataset)
    from statexplore.discovery.history import context
    out["context_state"] = context(p.dataset, full).state
    out["cache_rows_from_legacy"] = len(engine.cache_from(engine.create_run(p.dataset, full.settings), p.searches, p.dataset))
    t = Path(tempfile.mkdtemp())
    fv = V.export(full.validation_runs[0], str(t / "val"), ds=p.dataset, include_data=True, run=full)
    out["validation_export_files"] = sorted(Path(x).name for x in fv)
    out["validation_reproduction"] = json.loads((Path(fv[0]).parent / "validation_manifest.json").read_text())["reproduction"]
    from statexplore.export.exporters import export
    fr = export(p, [p.results[0].id], t / "rep", {"R", "xlsx"}, include_history=False, include_data=True)
    out["result_export_files"] = sorted(Path(x).name for x in fr)
    out["historical_rows_unchanged"] = [len(part.rows), len(full.rows)]
    new = engine.restart_from(part, p.dataset); p.searches.append(new); engine.execute(new, p.dataset)
    out["restart"] = {"id_new": new.id != part.id, "parent": new.parent_id == part.id, "status": new.status, "n_done": new.n_done,
                      "identity_schema": new.identity_schema}
    p.save(t / "v136_saved.sxp"); q = Project.open(t / "v136_saved.sxp")
    n2 = next(r for r in q.searches if r.id == new.id)
    out["after_resave"] = {"resumable": n2.resumable, "reason": n2.resume_reason(q.dataset), "legacy_still_legacy": q.searches[0].identity_schema,
                           "result_provenance_legacy": "analysis_provenance" not in q.results[0].values}
    n2.settings.max_models = 5; engine.execute(n2, q.dataset)
    out["after_resume"] = {"status": n2.status, "n_done": n2.n_done, "total": n2.total}
    print(json.dumps(out, indent=1, default=str))
