"""G0 red evidence (real R) on whichever StatExplore is on PYTHONPATH: exact-cap and cap-1 validation status."""
import json, sys, tempfile
from pathlib import Path
from statexplore.core.importers import import_file
from statexplore.discovery import engine, validate as V
from statexplore.discovery.space import SearchSettings
from statexplore.analysis.session import RSession
from statexplore.rbridge.worker import RWorker
import statexplore
fx = Path(sys.argv[1])
d = import_file(fx / "StatExplore_Test55L_LegacyUnderflow.csv")
run = engine.create_run(d, SearchSettings(outcome="y", candidates=["x1", "x3", "x2"], family="gaussian", strategy="focal",
                        covariate_mode="focal", max_predictors=2, workers=1, max_models=None, max_minutes=None))
engine.execute(run, d)
out = {"version": statexplore.__version__}
with RWorker() as w:
    s = RSession(w, Path(tempfile.mkdtemp()))
    cfg = V.ValidationConfig(target="rows", modes=["holdout"], seed=5, max_fits=100_000)
    full = V.ValidationRunner(s, d, run, cfg).execute()
    U = full["fit_budget"]["used"]
    out["ample"] = {"status": full["status"], "used": U, "fold": full["folds"][0]["status"]}
    for cap in (U, U - 1):
        r = V.ValidationRunner(s, d, run, cfg); r.budget["cap"] = cap
        v = r.execute()
        out[f"cap_{cap}"] = {"status": v["status"], "used": v["fit_budget"]["used"], "fold_status": v["folds"][0]["status"],
                             "selected": v["folds"][0].get("selected_index"), "problems": v["problems"]}
print(json.dumps(out, indent=1, default=str))
