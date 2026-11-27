"""Gate 0: minimal reproducers of AUD134-02..10 on the RELEASED v1.3.4 source (PYTHONPATH=E:\\sx_v134_base\\src).
AUD134-01 and AUD134-11 need R: see the red run of tests/test_v135.py on the baseline. Prints JSON lines."""
import json, math, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import statexplore
from statexplore.core.importers import import_file
from statexplore.core.project import Project
from statexplore.discovery import engine, space
from statexplore.discovery import validate as V
from statexplore.discovery.space import SearchSettings, count_candidates, enumerate_candidates, candidate_key, Candidate

REPO = Path(r"C:\Users\Stephen\Desktop\My stuff\statexplore\tests\fixtures")
out = {"statexplore_file": statexplore.__file__, "version": statexplore.__version__}

def emit(k, v):
    print(json.dumps({k: v}, default=str)); sys.stdout.flush()

emit("baseline", out)

# AUD134-02: adjusted ranking with two underflowed p in one family
rows = []
for i, l10 in enumerate([-434.0, -651.0]):
    lp = l10 * math.log(10)
    rows.append({"idx": i, "key": f"k{i}", "focal": ["a", "b"], "covariates": [], "kind": "focal", "status": "ok",
                 "reason": "", "flags": "", "speculative": False, "formula": f"y ~ m{i}", "interaction": f"t{i}",
                 "p": 0.0, "log_p": lp, "tests": [{"test_id": 0, "type": "focal_interaction", "label": f"t{i}", "p": 0.0,
                                                    "log_p": lp, "df1": 1.0, "statistic": 1.0}]})
run = engine.DiscoveryRun(settings=SearchSettings(outcome="y", candidates=["a", "b"]), family="gaussian", total=2,
                          signature="x")
run.rows, run.status = rows, "complete"
tf = engine.tests_frame(run, interpret=False)
emit("AUD134-02", {"raw_order": list(engine.discovery_ranking(tf, by="p")["idx"]),
                   "adjusted_order": list(engine.discovery_ranking(tf, by="p_adj")["idx"]), "expected": [1, 0]})

# AUD134-03: saved 52D project resumable despite fit rules 3
p = Project.open(REPO / "v134" / "StatExplore_Test52D_Persistence.sxp")
r52 = next(r for r in p.searches if r.id == "ea77ac2fedc4")
emit("AUD134-03", {"id": r52.id, "grammar": r52.grammar, "fit_rules": (r52.engine or {}).get("fit_rules"),
                   "current_fit_rules": engine.FIT_RULES, "n_done": r52.n_done, "total": r52.total,
                   "resumable": r52.resumable, "resume_reason": r52.resume_reason(p.dataset)})

# AUD134-04: focal strategy ignores the minimum
s = SearchSettings(outcome="Y", candidates=["X01", "X02", "X03"], required=["Time"], strategy="focal",
                   covariate_mode="focal", min_predictors=3, max_predictors=4, max_order=2)
a = count_candidates(s)
s.covariate_mode = "subsets"
b = count_candidates(s)
emit("AUD134-04", {"focal_required_only": a, "expected_focal": 0, "subsets": b, "expected_subsets": 3,
                   "subset_models": [(c.focal, c.covariates) for c in enumerate_candidates(s)]})

# AUD134-05: historical grammar-2 time candidate #0 vs current enumerator
d13 = import_file(REPO / "v134" / "StatExplore_Test51F_Null_13.csv")
st = SearchSettings(outcome="Y", candidates=["X01", "X02"], participant="StudentID", time="Time", strategy="time",
                    covariate_mode="subsets", max_predictors=3, min_predictors=2, study="longitudinal", family="gaussian")
rt = engine.create_run(d13, st)
rt.grammar = 2
old0 = Candidate(0, ("Time", "X01"), (), ())
rt.rows = [{"idx": 0, "key": candidate_key(old0), "focal": ["Time", "X01"], "covariates": [], "kind": "focal",
            "inter": [], "status": "ok"}]
rt.status, rt.n_done = "complete", rt.total
r5 = V.ValidationRunner(None, d13, rt, V.ValidationConfig(target="participant", modes=["holdout"], pipeline="fixed",
                                                          candidate=0))
emit("AUD134-05", {"historical_key": candidate_key(old0), "validated_key": candidate_key(r5.cands[0]),
                   "validated_model": [r5.cands[0].focal, r5.cands[0].covariates], "same": candidate_key(old0) ==
                   candidate_key(r5.cands[0]), "grammar_problem": rt.grammar_problem()})

# AUD134-06: 5 + 10 continuation validated with a pool of 5
s6 = SearchSettings.from_dict(r52.settings.to_dict())
r6 = engine.create_run(d13, s6)
r6.executions = [{"start": 0, "max_models": 5, "end": 5}, {"start": 5, "max_models": 10, "end": 15}]
r6.status = "complete"
emit("AUD134-06", {"mode": s6.mode, "executions": r6.executions, "pool": V.pool_size(r6, V.ValidationConfig()),
                   "label": V.pool_label(r6, V.ValidationConfig()), "expected_cumulative": 15})

# AUD134-07: audit top-level data identity after replacing the data with 50E
from statexplore.export.audit import audit_record
p50 = Project.open(REPO / "v134" / "StatExplore_Test50_Reproducibility.sxp")
r50 = p50.searches[0]
before = audit_record(r50, p50.dataset)["data"]
d50e = import_file(REPO / "v134" / "StatExplore_Test50E_ModifiedValues.csv")
p50.dataset.raw = d50e.raw
p50.dataset.source.sha256, p50.dataset.source.path = d50e.source.sha256, d50e.source.path
after = audit_record(r50, p50.dataset)["data"]
emit("AUD134-07", {"snapshot_sha": r50.data_snapshot.get("source_sha256"), "top_level_sha_before": before["sha256"],
                   "top_level_sha_after": after["sha256"], "top_level_file_after": after["file"],
                   "top_level_describes_current_data": after["sha256"] == d50e.source.sha256,
                   "snapshot_still_historical": after["snapshot"]["source_sha256"] == r50.data_snapshot["source_sha256"]})

# AUD134-08: fixed candidate #0 of a 7,300-candidate space visits the whole suffix
s8 = SearchSettings(outcome="Y", candidates=[f"X{i:02d}" for i in range(1, 9)], strategy="combinations",
                    include_main=True, max_predictors=4, max_order=3)
tot8 = count_candidates(s8)
calls = {"yield": 0}
real = space.enumerate_candidates
def counted(*a, **k):
    for c in real(*a, **k):
        calls["yield"] += 1
        yield c
import statexplore.discovery.space as sp
sp.enumerate_candidates = counted
first = next(real(s8))
cands = [c for c in counted(s8, start=0) if c.index == 0][:1]          # the v1.3.4 expression in ValidationRunner
sp.enumerate_candidates = real
emit("AUD134-08", {"declared": tot8, "requested_index": 0, "candidates_visited": calls["yield"]})

# AUD134-09: pre-cancelled, all-cached batches are still committed
d53 = import_file(Path(r"E:\Downloads\StatExplore_v1.3.4_Developer_Handoff\Synthetic_Datasets\StatExplore_Test53A_30Predictors.csv"))
s9 = SearchSettings(outcome="Y", candidates=[f"X{i:02d}" for i in range(1, 11)], strategy="main", max_predictors=1,
                    batch_size=2, max_models=None, max_minutes=None, family="gaussian")
r9 = engine.create_run(d53, s9)
r9.status = "running"
cache = {candidate_key(c): {"status": "ok", "key": candidate_key(c), "p": 0.5, "_engine": (), "_from": "old"}
         for c in enumerate_candidates(s9)}
class FP:
    session = type("S", (), {"worker": type("W", (), {"kill": lambda self: None})()})()
    def run_batch(self, *a):
        raise AssertionError("no fresh fits expected")
cancel = threading.Event(); cancel.set()
with ThreadPoolExecutor(1) as ex:
    engine._dispatch(r9, d53, [FP()], ex, cancel, None, time.perf_counter(), 0.0, cache)
emit("AUD134-09", {"cancel_set_before_dispatch": True, "rows_committed": len(r9.rows), "status": r9.status,
                   "stop_reason": r9.stop_reason, "expected": "cancelled, 0 rows"})

# AUD134-10: estimate vs uncached planned fits
from statexplore.discovery.space import candidate_tests, estimate_fits
plan = sum(1 + len(candidate_tests(c)) for c in enumerate_candidates(s8))
cfg = V.ValidationConfig(target="rows", modes=["holdout", "cv"], k=5, max_fits=250_000)
r10 = engine.DiscoveryRun(settings=s8, family="gaussian", total=tot8, signature="x")
s8.outcome = "y"
info = V.feasibility(import_file(REPO / "v135" / "StatExplore_Test55L_LegacyUnderflow.csv"), r10, cfg)
emit("AUD134-10", {"candidates": tot8, "samples": 6, "estimated_fits": info.get("fits"),
                   "uncached_planned_fits": 6 * plan + 6, "cap": 250_000,
                   "accepted": not [x for x in info["problems"] if "budget" in x]})
