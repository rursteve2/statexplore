"""Build tests/fixtures/v1_0_project.sxp with StatExplore 1.0.0. Run from a worktree of tag v1.0.0 (commit eb80db6):
    git worktree add <dir> eb80db6; copy this file there; set STATEXPLORE_R_LIB to the project r/library;
    python make_v1_0_fixture.py <out.sxp> <tmpdir>
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
import numpy as np, pandas as pd
from statexplore import __version__
from statexplore.core.importers import import_file
from statexplore.core.project import Project
from statexplore.core.derive import DerivedSpec, FilterSpec
from statexplore.analysis.builders import BY_KEY
from statexplore.analysis.framework import run_analysis
from statexplore.analysis import mixed
from statexplore.analysis.models import ModelSpec
from statexplore.analysis.session import RSession
from statexplore.discovery.engine import create_run, execute
from statexplore.discovery.space import SearchSettings
from statexplore.rbridge.worker import RWorker
assert __version__ == "1.0.0", __version__
out = Path(sys.argv[1]); tmp = Path(sys.argv[2]); tmp.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(2026)
n = 40
df = pd.DataFrame({"pid": np.repeat(np.arange(1, n + 1), 3), "wave": np.tile([1, 2, 3], n),
                   "sex": np.repeat(rng.choice(["f", "m"], n), 3), "x": rng.normal(size=3 * n),
                   "mood": rng.choice([1, 2, 3, 4], 3 * n)})
df["y"] = 0.5 * df["x"] + 0.3 * df["wave"] + (df["sex"] == "m") * 0.4 + rng.normal(size=3 * n)
df["yb"] = (df["y"] > df["y"].median()).astype(int)
df.to_csv(tmp / "v1src.csv", index=False)
ds = import_file(tmp / "v1src.csv")
ds.set_variable_field("pid", "role", "id")
ds.set_variable_field("mood", "measure", "ordinal")
ds.add_derived(DerivedSpec("x_c", "center", ["x"]))
ds.add_filter(FilterSpec("x", ">", -3))
w = RWorker(); w.start(); s = RSession(w, tmp / "rs")
proj = Project(ds)
for args in [("lm", {"dependent": ["y"], "predictors": ["x_c", "sex"]}, {"interactions": "two_way"}),
             ("logistic", {"dependent": ["yb"], "predictors": ["x"]}, {}),
             ("anova_oneway", {"dependent": ["y"], "factors": ["mood"]}, {})]:
    r, a = run_analysis(s, ds, BY_KEY[args[0]], args[1], args[2]); proj.add_result(r, a)
r, a = mixed.run(s, ds, mixed.MixedSpec(outcome="y", id="pid", time="wave", predictors=["sex"])); proj.add_result(r, a)
run = execute(create_run(ds, SearchSettings(outcome="y", candidates=["x", "sex", "mood"], workers=1,
                                            max_models=None, max_minutes=None)), ds, workdir=tmp / "disc")
proj.searches.append(run)
proj.models.append(ModelSpec("y", [("x", "sex")], "gaussian", label="v1 model"))
w.stop()
proj.save(out)
print("saved", out, len(proj.results), run.status)
