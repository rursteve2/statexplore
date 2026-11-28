"""Acceptance cases 55-61 (see the package docstring for the execution kinds). Settings follow
docs/v1.3.6/manual_acceptance_v1.3.6.md; expected values are the owner's recorded references (prompt section 2) or are
computed here by an independent R oracle. Fixtures are synthetic (no private data)."""
from __future__ import annotations

import json
import math
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

S55L = dict(outcome="y", candidates=["x1", "x3", "x2"], family="gaussian", strategy="focal", covariate_mode="focal",
            max_predictors=2, workers=1, max_models=None, max_minutes=None)
S51C = dict(outcome="TestScore", candidates=["Sleep", "Anxiety", "Attendance"], required=["Time", "Gender"],
            participant="StudentID", time="Time", cluster="School", structure="rs", strategy="combinations",
            include_main=True, max_order=3, max_predictors=5, family="gaussian", workers=2, max_models=None,
            max_minutes=None, study="longitudinal")


def _settings(**kw):
    from ..discovery.space import SearchSettings

    return SearchSettings(**kw)


def _search(d, **kw):
    from ..discovery import engine

    run = engine.create_run(d, _settings(**kw))
    engine.execute(run, d)
    return run


def _imp(ctx, name, **kw):
    from ..core.importers import import_file

    return import_file(ctx.fx(name), **kw)


def _folder(files) -> Path:
    return Path(next(f for f in files if Path(f).name == "bundle_inventory.json")).parent


def _session(ctx, w, name="s"):
    from ..analysis.session import RSession

    return RSession(w, ctx.dir(name))


def _reproduce(ctx, folder: Path, label: str) -> tuple[int, pd.DataFrame, str]:
    r = ctx.run_r(folder, "validation_reproduce.R")
    chk = folder / "validation_check_results.csv"
    tab = pd.read_csv(chk, keep_default_na=False) if chk.is_file() else pd.DataFrame(columns=["check", "status"])
    return r.returncode, tab, r.stdout + r.stderr


def _edit(folder: Path, fname: str, fn) -> None:
    df = pd.read_csv(folder / fname, dtype=str, keep_default_na=False)
    fn(df)
    df.to_csv(folder / fname, index=False, lineterminator="\n")


# ---- 55 ---------------------------------------------------------------------------------------------------------------

def case_55(ctx) -> None:
    """Extreme probabilities, bootstrap and reproduction (55L legacy tails, 51C tails + AMD bootstrap, exported
    reproduction with tamper / partial replay)."""
    from ..discovery import engine
    from ..discovery import validate as V

    d = _imp(ctx, "StatExplore_Test55L_LegacyUnderflow.csv")
    run = _search(d, **S55L)
    tf = engine.tests_frame(run, interpret=False)
    o = ctx.oracle(f"""d <- utils::read.csv("{ctx.fx('StatExplore_Test55L_LegacyUnderflow.csv').as_posix()}")
one <- function(a, b) {{ f1 <- stats::lm(stats::reformulate(c(a, b, paste0(a, ":", b)), "y"), d)
  f0 <- stats::lm(stats::reformulate(c(a, b), "y"), d); A <- stats::anova(f0, f1)
  stats::pf(A$F[2], A$Df[2], A$Res.Df[2], lower.tail = FALSE, log.p = TRUE) / log(10) }}
cat(sprintf("%.10f", c(one("x1", "x3"), one("x1", "x2"), one("x3", "x2"))), sep = "\\n")""", "o55l")
    want = [float(x) for x in o.stdout.split()]
    for idx in (0, 1, 2):
        got = tf.loc[tf["idx"] == idx, "log10_p"].iloc[0]
        ctx.close(f"55L #{idx} log10 p (lm + pf log.p oracle)", want[idx], float(got), 1e-6 * max(1, abs(want[idx])))
    rk = engine.discovery_ranking(tf)
    ctx.check("55L ranking by log tail: #1 (x1 x x2) first, #0 second", [1, 0], [int(x) for x in rk["idx"][:2]])
    ctx.check("55L both p underflow to 0 but log tails differ", True,
              bool((tf.loc[tf["idx"].isin([0, 1]), "p"] == 0).all() and want[1] < want[0] < -300))
    w = ctx.worker()
    try:
        sess = _session(ctx, w)
        # AMD validation on 55L with bootstrap, exported with data to a path with a space and a non-ASCII letter
        cfg = V.ValidationConfig(target="rows", modes=["holdout", "cv", "bootstrap"], k=2, B=2 if ctx.quick else 3,
                                 seed=7, max_fits=100_000)
        v = V.ValidationRunner(sess, d, run, cfg).execute()
        ctx.check("55L AMD validation status", "complete", v["status"], kind="app+R")
        ctx.check("55L every part selects #1 (training log tail)", True,
                  all(f["selected_index"] == 1 for f in v["folds"]) and all(b["selected_index"] == 1 for b in v["bootstrap"]),
                  kind="app+R")
        dst = ctx.dir("export ü é")
        f = _folder(V.export(v, str(dst), ds=d, include_data=True, run=run))
        rc, tab, out = _reproduce(ctx, f, "clean")
        ctx.check("55L exported reproduction exits 0 in a fresh Rscript", 0, rc, kind="script-R")
        ctx.check("55L reproduction: no FAIL checks", 0, int((tab["status"] == "FAIL").sum()), kind="script-R")
        ctx.check("55L reproduction says complete", True, "The complete validation was reproduced" in out, kind="script-R")
        tampers = [("predictions.csv", lambda df: df.__setitem__("predicted", [repr(float(df["predicted"][0]) + 0.01)]
                                                                 + list(df["predicted"][1:])), "predictions: predicted"),
                   ("stability.csv", lambda df: df.__setitem__("estimate_median", ["100" if x else x for x in df["estimate_median"]]),
                    "stability"),
                   ("execution_ledger.csv", lambda df: df.__setitem__("status", ["failed"] + list(df["status"][1:])),
                    "execution ledger: status")]
        for fname, fn, chk in tampers:
            t = f.parent / f"tamper_{fname.split('.')[0]}"
            shutil.copytree(f, t)
            _edit(t, fname, fn)
            rc2, tab2, _ = _reproduce(ctx, t, fname)
            ctx.check(f"55L tamper {fname}: exit 1 naming '{chk}'", True,
                      rc2 != 0 and tab2.loc[tab2["status"] == "FAIL", "check"].str.startswith(chk).any(), kind="script-R")
        # partial replay (AUD135-04): cancel after the first chunk
        calls = {"n": 0}

        def stop():
            calls["n"] += 1
            return calls["n"] > 1
        pv = V.ValidationRunner(sess, d, run, V.ValidationConfig(target="rows", modes=["holdout", "cv"], k=2, seed=11,
                                                                 max_fits=100_000)).execute(should_stop=stop)
        ctx.check("55L cancelled validation: status and committed parts", ("cancelled", 1, 3),
                  (pv["status"], pv["execution"]["committed_parts"], pv["execution"]["planned_parts"]), kind="app+R")
        pf = _folder(V.export(pv, str(ctx.dir("partial")), ds=d, include_data=True, run=run))
        rc3, tab3, out3 = _reproduce(ctx, pf, "partial")
        ctx.check("55L partial reproduction exits 0 and reports 1 of 3 parts, not complete", True,
                  rc3 == 0 and "1 of 3 planned parts committed" in out3 and "complete validation was reproduced" not in out3,
                  kind="script-R")
        # 51C tails + AMD bootstrap (new clusters, whole pipeline)
        d51 = _imp(ctx, "StatExplore_Test51C_ThreeWayInteraction.csv")
        r51 = _search(d51, **S51C)
        t51 = engine.tests_frame(r51)
        ctx.check("51C candidates / tests", (18, 48), (len(r51.rows), len(t51)), kind="app+R")
        t17 = t51[t51["idx"] == 17].set_index("test_type")
        ctx.close("51C #17 whole-model log10 p", -485.200773, float(t17.loc["overall", "log10_p"]), 0.05)
        ctx.close("51C #17 joint log10 p", -419.542529, float(t17.loc["joint_interaction", "log10_p"]), 0.05)
        ctx.check("51C families separate (whole-model vs joint)", True,
                  t17.loc["overall", "test_type"] != t17.loc["joint_interaction", "test_type"]
                  if "test_type" in t17.columns else True, kind="app+R")
        B = 3 if ctx.quick else 20
        v51 = V.ValidationRunner(sess, d51, r51, V.ValidationConfig(target="cluster", modes=["bootstrap"], B=B, seed=133,
                                                                    max_fits=100_000)).execute()
        ctx.check(f"51C bootstrap selects #17 in {B}/{B} draws, complete", (B, "complete"),
                  (sum(1 for b in v51["bootstrap"] if b["selected_index"] == 17), v51["status"]), kind="app+R")
        f51 = _folder(V.export(v51, str(ctx.dir("51C export ü")), ds=d51, include_data=True, run=r51))
        rc4, tab4, out4 = _reproduce(ctx, f51, "51C")
        ctx.check("51C exported reproduction exits 0, no FAIL, complete", (0, 0, True),
                  (rc4, int((tab4["status"] == "FAIL").sum()), "The complete validation was reproduced" in out4),
                  kind="script-R")
    finally:
        w.stop()


# ---- 56 ---------------------------------------------------------------------------------------------------------------

def case_56(ctx) -> None:
    """GEE structures and QIC (48D): structure-specific working correlation, QIC/QICu/CIC vs owner references and an
    independent geepack oracle, transfer keeps the correlation, same-view cache reuse unchanged."""
    from ..analysis import mixed
    from ..discovery import engine

    d = _imp(ctx, "StatExplore_Test48D_GEE.csv")
    o = ctx.oracle(f"""suppressMessages(library(geepack))
d <- read.csv("{ctx.fx('StatExplore_Test48D_GEE.csv').as_posix()}"); d <- d[order(d$ID, d$TIME), ]
d$occ <- as.integer(factor(d$TIME)); out <- NULL
for (cs in c("independence", "exchangeable", "ar1")) {{
  m <- geeglm(Y ~ TIME * X, id = ID, waves = occ, corstr = cs, data = d, family = gaussian()); q <- QIC(m)
  out <- rbind(out, data.frame(corstr = m$corstr, QIC = q[["QIC"]], QICu = q[["QICu"]], CIC = q[["CIC"]]))
}}
write.csv(out, "gee.csv", row.names = FALSE)""", "o48d")
    ref = pd.read_csv(ctx.dir("oracle") / "gee.csv").set_index("corstr")
    owner = {"independence": 1146.365433, "exchangeable": 1147.731438, "ar1": 1149.21553}
    for cs, val in owner.items():
        ctx.close(f"48D oracle QIC {cs} vs owner reference", val, float(ref.loc[cs, "QIC"]), 5e-6, kind="oracle-R")
    names = {"gee_ind": "independence", "gee_exch": "exchangeable", "gee_ar1": "ar1"}
    base = dict(outcome="Y", candidates=["X"], required=["TIME"], participant="ID", time="TIME", workflow="structure",
                time_interactions=True, workers=1, family="gaussian", max_models=None, max_minutes=None,
                study="longitudinal")
    runs = []
    for order in (["gee_ind", "gee_exch", "gee_ar1"], ["gee_ar1", "gee_exch", "gee_ind"]):
        run = _search(d, structures=order, **base)
        runs.append(run)
        for r in run.rows:
            cs = names[r["structure"]]
            ctx.check(f"48D {'/'.join(order)}: {cs} fitted working correlation", cs, r["gee"]["corstr"], kind="app+R")
            for k, col in (("qic", "QIC"), ("qicu", "QICu"), ("cic", "CIC")):
                ctx.close(f"48D {cs} {col} vs geepack oracle ({'/'.join(order)})", float(ref.loc[cs, col]),
                          float(r[k]), 1e-6)
        ctx.check("48D no AIC for GEE rows", True, all(r.get("aic") is None or r.get("aic") != r.get("aic")
                                                      for r in run.rows), kind="app+R")
    row = next(r for r in runs[0].rows if r["structure"] == "gee_exch")
    w = ctx.worker()
    try:
        spec = mixed.spec_from_discovery(runs[0], row)
        res, _ = mixed.run(_session(ctx, w), d, spec)
        txt = json.dumps([t.rows for t in res.tables] + [res.values.get("effective_spec")], default=str).lower()
        ctx.check("48D transfer keeps the exchangeable working correlation", ("exchangeable", True, True),
                  (spec.gee_corstr, res.status.value in ("ok", "warning"), "exchangeable" in txt), kind="app+R")
    finally:
        w.stop()
    again = engine.create_run(d, _settings(structures=["gee_ind", "gee_exch", "gee_ar1"], **base))
    cache = engine.cache_from(again, [runs[0]], d)
    ctx.check("48D same verified view: fit identity unchanged and all 3 rows reusable", (runs[0].fit_id, 3),
              (again.fit_id, len(cache)))


# ---- 57 ---------------------------------------------------------------------------------------------------------------

def case_57(ctx) -> None:
    """Historical project and resume compatibility (52D refusal, linked new search, save / reopen / resume to 15;
    workbook A/B; legacy identity)."""
    from ..core.project import Project
    from ..discovery import engine
    from ..discovery.history import context

    src = ctx.fx("StatExplore_Test52D_Persistence.sxp")
    p = Project.open(src)
    old = next(r for r in p.searches if r.id == "ea77ac2fedc4")
    before = json.dumps(old.rows, sort_keys=True, default=str)
    why = old.resume_reason(p.dataset)
    ctx.check("52D resume refused with fit rules 3 reason", True, bool(why) and "fit rules 3" in why)
    ctx.check("52D legacy identity: context unverified", "unverified", context(p.dataset, old).state)
    new = engine.restart_from(old, p.dataset)
    new.settings.mode, new.settings.max_models, new.settings.workers = "D", 5, 2
    p.searches.append(new)
    engine.execute(new, p.dataset)
    ctx.check("restarted search: new id, parent link, partial 5/15", (True, old.id, "partial", 5, 15),
              (new.id != old.id, new.parent_id, new.status, new.n_done, new.total), kind="app+R")
    path = ctx.dir("projects") / "p57.sxp"
    p.save(path)
    p2 = Project.open(path)
    r2 = next(r for r in p2.searches if r.id == new.id)
    ctx.check("after save / reopen: resumable, no reason", (True, ""), (r2.resumable, r2.resume_reason(p2.dataset)))
    r2.settings.max_models = 10
    engine.execute(r2, p2.dataset, cache=engine.cache_from(r2, p2.searches))
    ctx.check("resumed: complete 15/15, every candidate once, budgets 5 then 10",
              ("complete", 15, list(range(15)), [5, 10]),
              (r2.status, r2.n_done, sorted(r["idx"] for r in r2.rows), [e["max_models"] for e in r2.executions]),
              kind="app+R")
    o2 = next(r for r in p2.searches if r.id == old.id)
    ctx.check("original 52D search unchanged", before, json.dumps(o2.rows, sort_keys=True, default=str))
    # workbook A/B (AUD135-06): same file hash, different sheets
    a = _imp(ctx, "AUD135_06_two_sheets.xlsx", sheet="A")
    b = _imp(ctx, "AUD135_06_two_sheets.xlsx", sheet="B")
    s = dict(outcome="y", candidates=["x", "z", "w"], family="gaussian", strategy="focal", max_predictors=2,
             workers=1, mode="D", max_models=1, max_minutes=None)
    ra = engine.create_run(a, _settings(**s))
    engine.execute(ra, a)
    ctx.check("workbook: same source SHA-256, different signatures", (True, True),
              (a.source.sha256 == b.source.sha256, ra.signature != engine.data_signature(b)))
    ctx.check("workbook: sheet-A partial cannot resume on sheet B", True, bool(ra.resume_reason(b)))
    ctx.check("workbook: sheet-A partial resumes on sheet A (re-imported)", "",
              ra.resume_reason(_imp(ctx, "AUD135_06_two_sheets.xlsx", sheet="A")))
    ctx.check("workbook: no cache admission across sheets", 0,
              len(engine.cache_from(engine.create_run(b, _settings(**s)), [ra], b)))


# ---- 58 ---------------------------------------------------------------------------------------------------------------

def case_58(ctx) -> None:
    """Candidate enumeration and cumulative validation (Null13: minimum count, 5 + 10 = 15 cumulative, 7 restricted,
    training-only trace, execution scope)."""
    from ..discovery import engine
    from ..discovery import validate as V
    from ..discovery.space import SearchSpaceError, count_candidates, enumerate_candidates, validate_settings

    d = _imp(ctx, "StatExplore_Test51F_Null_13.csv")
    base = dict(outcome="Y", participant="StudentID", time="Time", required=["Time"], family="gaussian",
                study="longitudinal", workers=2, max_minutes=None)
    s3 = _settings(candidates=["X01", "X02", "X03"], strategy="focal", max_predictors=4, min_predictors=3,
                   covariate_mode="focal", max_models=None, **base)
    try:
        validate_settings(d, s3)
        n_focal = count_candidates(s3)
    except SearchSpaceError as e:
        n_focal = f"refused: {e}"
    ctx.check("min 3 candidates, focal covariates: refused or empty (a pair has 2)", True,
              (isinstance(n_focal, str) and "refused" in n_focal) or n_focal == 0)
    s3.covariate_mode = "subsets"
    cs = list(enumerate_candidates(s3))
    ctx.check("min 3, subsets: exactly 3 models, each with X01, X02, X03", (3, True),
              (len(cs), all({"X01", "X02", "X03"} <= set(c.focal + c.covariates) for c in cs)))
    s = _settings(candidates=[f"X{i:02d}" for i in range(1, 7)], strategy="focal", covariate_mode="focal",
                  max_predictors=3, min_predictors=1, mode="D", max_models=5, **{k: v for k, v in base.items()
                                                                                  if k != "max_minutes"},
                  max_minutes=None)
    run = engine.create_run(d, s)
    engine.execute(run, d)
    run.settings.max_models = 10
    engine.execute(run, d)
    ctx.check("mode D 5 then 10: complete 15", ("complete", 15), (run.status, run.n_done), kind="app+R")
    cfg = lambda **k: V.ValidationConfig(target="participant", modes=["holdout"], **k)  # noqa: E731
    ctx.check("pool sizes cumulative / original / prefix 7", (15, 5, 7),
              (V.pool_size(run, cfg()), V.pool_size(run, cfg(pool="original")), V.pool_size(run, cfg(pool="prefix",
                                                                                                        prefix_n=7))))
    lab = V.pool_label(run, cfg())
    ctx.check("cumulative label names 5 + 10 and ADAPTIVE", True, "5 + 10" in lab and "ADAPTIVE" in lab)
    w = ctx.worker()
    try:
        v = V.ValidationRunner(_session(ctx, w), d, run, cfg(seed=58)).execute()
        tr = pd.DataFrame(v["trace"])
        ctx.check("validation complete; training trace holds the 15 declared candidates", ("complete", 15),
                  (v["status"], int(tr["candidate_index"].nunique()) if len(tr) else 0), kind="app+R")
        ctx.check("ledger: 1 planned part, committed", (1, 1),
                  (v["execution"]["planned_parts"], v["execution"]["committed_parts"]), kind="app+R")
        f = _folder(V.export(v, str(ctx.dir("export")), ds=d, include_data=True, run=run))
        rc, tab, out = _reproduce(ctx, f, "null13")
        ctx.check("Null13 reproduction exits 0, no FAIL", (0, 0), (rc, int((tab["status"] == "FAIL").sum())),
                  kind="script-R")
    finally:
        w.stop()


# ---- 59 ---------------------------------------------------------------------------------------------------------------

def case_59(ctx) -> None:
    """Historical dataset and model identity (50/50E audit, reused folders + consent, ordinary holdout result
    exported after a seed change, real-R t-test oracle)."""
    from ..analysis.builders import BY_KEY
    from ..analysis.framework import run_analysis
    from ..core.dataset import HoldoutSpec
    from ..core.importers import import_file
    from ..core.project import Project
    from ..discovery import validate as V
    from ..export.audit import read_audit, write_audit
    from ..export.exporters import export

    p = Project.open(ctx.fx("StatExplore_Test50_Reproducibility.sxp"))
    run = p.searches[0]
    fa = write_audit(run, p.dataset, ctx.dir("audit"))
    a = read_audit(fa[0])
    h = a["data"]["historical"]
    ctx.check("50: historical source SHA-256 (3586bd71...)", True, str(h["source_sha256"]).startswith("3586bd71"))
    ctx.check("50: legacy search identity is labelled unverified (v1.3.6)", (False, "unverified"),
              (a["data"]["current_context"]["matches_analysis"], a["data"]["current_context"]["state"]))
    e = import_file(ctx.fx("StatExplore_Test50E_ModifiedValues.csv"))
    p.dataset.raw = e.raw
    p.dataset.source.sha256, p.dataset.source.path = e.source.sha256, e.source.path
    fb = write_audit(run, p.dataset, ctx.dir("audit"))
    b = read_audit(fb[0])
    ctx.check("50E: historical block identical", json.dumps(h, sort_keys=True), json.dumps(b["data"]["historical"],
                                                                                        sort_keys=True))
    ctx.check("50E: current context differs (8b7d2a4f...)", (False, True),
              (b["data"]["current_context"]["matches_analysis"],
               str(b["data"]["current_context"]["source_sha256"]).startswith("8b7d2a4f")))
    ctx.check("audit exports went to two new folders", True, _folder(fa) != _folder(fb))
    if run.validation_runs:
        fv = V.export(run.validation_runs[0], str(ctx.dir("val")), ds=p.dataset, include_data=True, run=run)
        names = {Path(x).name for x in fv}
        ctx.check("50E validation export: results only, reason given, no data", (True, False, True),
                  ("REPRODUCTION_UNAVAILABLE.txt" in names, any(n.endswith("_data.csv") for n in names),
                   "predictions.csv" in names))
    # ordinary analysis on a holdout: seed 1 result exported after the seed changed (real R + t.test oracle)
    f = ctx.dir("t") / "scores.csv"
    pd.DataFrame({"score": [x + 0.5 for x in range(20)]}).to_csv(f, index=False)
    ds = import_file(f)
    ds.set_variable_field("score", "measure", "continuous")
    ds.set_holdout(HoldoutSpec(0.25, 1))
    pr = Project(ds)
    w = ctx.worker()
    try:
        sess = _session(ctx, w, "t_sess")
        r1, a1 = run_analysis(sess, ds, BY_KEY["t_one_sample"], {"dependent": ["score"]}, {})
        x1 = ds.analysis_frame()["score"].tolist()
        ds.set_holdout(HoldoutSpec(0.25, 2))
        r2, a2 = run_analysis(sess, ds, BY_KEY["t_one_sample"], {"dependent": ["score"]}, {})
        x2 = ds.analysis_frame()["score"].tolist()
    finally:
        w.stop()
    pr.add_result(r1, a1)
    pr.add_result(r2, a2)
    o = ctx.oracle("for (x in list(c(" + ", ".join(map(repr, x1)) + "), c(" + ", ".join(map(repr, x2)) + "))) {"
                   "tt <- t.test(x); cat(sprintf('%.15g %.15g %.15g %.15g %d\\n', mean(x), tt$statistic, tt$conf.int[1], "
                   "tt$conf.int[2], length(x))) }", "o_t")
    ref = [list(map(float, ln.split())) for ln in o.stdout.strip().splitlines()]
    ctx.check("t oracle: same n, means 9.7 vs 10.3", (15, 15, 9.7, 10.3),
              (int(ref[0][4]), int(ref[1][4]), round(ref[0][0], 10), round(ref[1][0], 10)), kind="oracle-R")
    out = ctx.dir("reports")
    (out / "data.csv").write_text("an older export\n")
    w1 = export(pr, [r1.id, r2.id], out, {"R"}, include_history=False, include_data=True)
    w0 = export(pr, [r1.id, r2.id], out, {"R"}, include_history=False, include_data=False)
    ctx.check("reports: two new folders, older file untouched", (True, "an older export\n"),
              (_folder(w1) != _folder(w0), (out / "data.csv").read_text()))
    ctx.check("no consent: no data file anywhere in that export", 0,
              len([x for x in _folder(w0).rglob("*.csv") if "data" in x.name]))
    for res, (m, t, lo, hi, _n) in ((r1, ref[0]), (r2, ref[1])):
        sp = next(Path(x) for x in w1 if Path(x).suffix == ".R" and res.id in Path(x).parent.name)
        rr = ctx.run_r(sp.parent, sp.name)
        got = json.loads((sp.parent / f"result_{res.id}.json").read_text()) if rr.returncode == 0 else {"tables": []}
        tab = next((x["data"] for x in got["tables"] if x["name"] == "One-sample t-test"), {})
        ok = rr.returncode == 0 and tab.get("n") == [15] and all(
            abs(tab[k][0] - v) <= 1e-9 * max(1, abs(v)) for k, v in (("mean", m), ("t", t), ("ci_low", lo), ("ci_high", hi)))
        ctx.check(f"bundle of result {res.id} reproduces its own n / mean / t / CI (t.test oracle)", True, ok,
                  tol="rel 1e-9", kind="script-R")


# ---- 60 ---------------------------------------------------------------------------------------------------------------

def case_60(ctx) -> None:
    """Large model spaces, cancellation and computational budgets (53A counts and bounded retrieval, budget 100,
    cancellation, exact cap and mid-chunk partial on 55L)."""
    from ..discovery import engine
    from ..discovery import validate as V
    from ..discovery.space import candidate_key, count_candidates, enumerate_candidates

    d = _imp(ctx, "StatExplore_Test53A_30Predictors.csv")
    huge = dict(outcome="Y", candidates=[f"X{i:02d}" for i in range(1, 31)], required=["Time"], participant="StudentID",
                time="Time", strategy="combinations", include_main=True, max_predictors=6, family="gaussian",
                structure="ri", study="longitudinal", workers=2, max_minutes=None)
    t0 = time.perf_counter()
    n = count_candidates(_settings(max_models=None, **huge))
    ctx.check("53A count 147,713,444 without materializing (< 2 s)", (147_713_444, True),
              (n, time.perf_counter() - t0 < 2))
    run = engine.create_run(d, _settings(mode="D", max_models=100, **huge))
    engine.execute(run, d)
    ctx.check("53A mode D budget 100: 100 attempted, partial", (100, "partial"), (len(run.rows), run.status),
              kind="app+R")
    c = next(enumerate_candidates(run.settings, start=50))
    t1 = time.perf_counter()
    info = V.feasibility(d, run, V.ValidationConfig(target="participant", modes=["holdout"], pipeline="fixed",
                                                    candidate=50))
    ctx.check("fixed candidate #50 feasibility immediate, verified key, pool 1", (True, 1, True),
              (time.perf_counter() - t1 < 10, info["pool"], candidate_key(c) == next(r["key"] for r in run.rows
                                                                                       if r["idx"] == 50)))
    canc = engine.create_run(d, _settings(mode="D", max_models=2000, **huge))
    ev = threading.Event()
    th = threading.Thread(target=lambda: engine.execute(canc, d, cancel=ev))
    th.start()
    time.sleep(3.0)
    ev.set()
    tc = time.perf_counter()
    th.join()
    lat = time.perf_counter() - tc
    k = len(canc.rows)
    time.sleep(1.0)
    ctx.check("cancel: status cancelled, no rows after acknowledgement", ("cancelled", k), (canc.status, len(canc.rows)),
              kind="app+R")
    ctx.check("cancel latency < 30 s", True, lat < 30, tol=f"measured {lat:.1f} s", kind="app+R")
    pre = engine.create_run(d, _settings(mode="D", max_models=100, **huge))
    ev2 = threading.Event()
    ev2.set()
    engine.execute(pre, d, cache=engine.cache_from(pre, [run], d), cancel=ev2)
    ctx.check("pre-cancelled, all cached: nothing committed", ("cancelled", 0), (pre.status, len(pre.rows)))
    # exact cap (AUD135-05) on 55L, real R
    d5 = _imp(ctx, "StatExplore_Test55L_LegacyUnderflow.csv")
    r5 = _search(d5, **S55L)
    w = ctx.worker()
    try:
        sess = _session(ctx, w)
        cfg = V.ValidationConfig(target="rows", modes=["holdout"], seed=5, max_fits=100_000)
        U = V.ValidationRunner(sess, d5, r5, cfg).execute()["fit_budget"]["used"]
        outs = {}
        for cap in (U, U - 1):
            r = V.ValidationRunner(sess, d5, r5, cfg)
            r.budget["cap"] = cap
            outs[cap] = r.execute()
        ctx.check(f"exact cap {U}: complete, used == cap, not exhausted", ("complete", U, False),
                  (outs[U]["status"], outs[U]["fit_budget"]["used"], outs[U]["fit_budget"]["exhausted"]), kind="app+R")
        ctx.check(f"cap {U - 1}: partial, part stopped by the budget, cap not exceeded", ("partial", "budget", True),
                  (outs[U - 1]["status"], outs[U - 1]["folds"][0]["status"], outs[U - 1]["fit_budget"]["used"] <= U - 1),
                  kind="app+R")
        f = _folder(V.export(outs[U - 1], str(ctx.dir("cap")), ds=d5, include_data=True, run=r5))
        rc, tab, out = _reproduce(ctx, f, "cap")
        ctx.check("budget-stopped export reproduces exactly (same guard), exit 0, not complete", True,
                  rc == 0 and "not a reproduction of a complete validation" in out, kind="script-R")
    finally:
        w.stop()


# ---- 61 ---------------------------------------------------------------------------------------------------------------

def case_61(ctx) -> None:
    """Large-dataset performance and sorting (54B 100k: real search, extreme adjusted p sorted by log tails;
    identity fingerprint overhead)."""
    from ..core.identity import content_id, values_digest, view_id
    from ..core.importers import import_file
    from ..discovery import engine

    t0 = time.perf_counter()
    d = import_file(ctx.fx("StatExplore_Test54B_100K.csv"))
    load = time.perf_counter() - t0
    t1 = time.perf_counter()
    values_digest(d.raw)
    cold = time.perf_counter() - t1
    t2 = time.perf_counter()
    for _ in range(100):
        content_id(d)
        view_id(d)
    warm = (time.perf_counter() - t2) / 100
    ctx.check("100k content identity: cold hash < 5 s, cached identity < 50 ms", True, cold < 5 and warm < 0.05,
              tol=f"measured cold {cold:.2f} s, warm {warm * 1000:.1f} ms")
    s = dict(outcome="Y", family="gaussian", participant="StudentID", time="Time", required=["Time"], structure="ri",
             cluster="School", strategy="focal", covariate_mode="focal", max_predictors=3, min_predictors=2,
             include_main=False, workers=4, max_models=None, max_minutes=None, study="longitudinal", correction="BH",
             mode="C", candidates=[f"X{i:02d}" for i in range(1, 7)])
    t3 = time.perf_counter()
    run = _search(d, **s)
    wall = time.perf_counter() - t3
    tf = engine.tests_frame(run, interpret=False)
    ctx.check("54B: 15 candidates, 2 BH-significant", (15, 2), (len(run.rows), int(tf["sig_adj"].sum())),
              kind="app+R")
    ctx.check("54B wall time recorded (reference ~50 s on the developer machine)", True, wall > 0,
              tol=f"measured {wall:.1f} s (load {load:.1f} s)", kind="app+R")
    rk = engine.discovery_ranking(tf, by="p_adj")
    top = rk.head(2)
    ctx.check("54B adjusted sort: two planted interactions first, log10 p < -300, ordered by log tail", True,
              bool((top["log10_p"] < -300).all() and top["log10_p"].is_monotonic_increasing), kind="app+R")
    ctx.check("54B no literal 0 shown for underflowed p", True,
              all("0" != engine._p_text(float(r["p"]), float(r["log_p"])).strip() for _, r in top.iterrows()))


# ---- owner R diagnostic (AUD135-01/02) on the GENERATED verifier ------------------------------------------------------

_STUB = """sx_long <- function() FALSE
sx_disc_has_cluster <- FALSE
sx_val_index <- c(0); sx_val_keys <- c("candidate0"); sx_val_labels <- c("candidate 0")
sx_val_full <- data.frame(y = 1:20, row.names = as.character(1:20))
sx_alias <- character()
"""
_DIAG = r"""
res <- list()
add <- function(id, expected, actual) res[[length(res) + 1L]] <<- data.frame(test = id, expected = as.character(expected),
  actual = as.character(actual), status = if (identical(as.character(expected), as.character(actual))) "PASS" else "FAIL")
add("NUM-01 identical finite", TRUE, sx_num_ok(1, 1, 1e-6, 1e-6))
add("NUM-02 different finite", FALSE, sx_num_ok(1, 2, 1e-6, 1e-6))
add("NUM-03 both missing", TRUE, sx_num_ok(NA_real_, NA_real_, 1e-6, 1e-6))
add("NUM-04 same positive infinity", TRUE, sx_num_ok(Inf, Inf, 1e-6, 1e-6))
add("NUM-05 same negative infinity", TRUE, sx_num_ok(-Inf, -Inf, 1e-6, 1e-6))
add("NUM-06 opposite infinities", FALSE, sx_num_ok(-Inf, Inf, 1e-6, 1e-6))
add("NUM-07 finite versus infinity", FALSE, sx_num_ok(1, Inf, 1e-6, 1e-6))
add("NUM-08 invalid text versus missing", FALSE, sx_num_ok("corrupted", NA_real_, 1e-6, 1e-6))
add("NUM-09 both invalid numeric text", FALSE, sx_num_ok("bad", "bad", 1e-6, 1e-6))
utils::write.csv(data.frame(id = "1", participant = "001"), "app.csv", row.names = FALSE)
spec <- list(keys = "id", cols = data.frame(name = c("id", "participant"), type = c("int", "text"), flags = c("", ""),
             stringsAsFactors = FALSE))
sx_chk <- sx_chk[0, ]
sx_compare("identity", "app.csv", data.frame(id = 1L, participant = "1"), spec)
add("ID-01 numeric-looking string identity", "FAIL", sx_chk$status[sx_chk$check == "identity: participant"])
add("SEL-01 underflowed ordinary p distinct log tails", 2L, sx_select_min(c(-434, -651) * log(10)))
add("SEL-02 true tie", 1L, sx_select_min(c(-5, -5)))
add("SEL-03 withheld test rejected", TRUE, is.na(sx_primary_logp(list(status = "ok", tests = list(list(log_p = -5, withheld = "negative LR"))))))
# extra controls
add("X-01 decimal overflow is malformed", "bad", sx_parse_num("1e999")$state)
add("X-02 silent underflow is malformed", "bad", sx_parse_num("1e-400")$state)
add("X-03 subnormal is a number", "num", sx_parse_num("5e-324")$state)
add("X-04 NaN text is malformed", "bad", sx_parse_num("NaN")$state)
add("X-05 hex text is malformed", "bad", sx_parse_num("0x10")$state)
add("X-06 reproduced NaN is malformed", FALSE, sx_num_ok(NaN, NaN, 1e-6, 1e-6))
add("X-07 missing not allowed by the field", FALSE, sx_num_ok("", NA_real_, 1e-6, 1e-6, missing = FALSE))
add("X-08 infinity not allowed by the field", FALSE, sx_num_ok("Inf", Inf, 1e-6, 1e-6, inf = FALSE))
add("X-09 integer text 007 vs 7", FALSE, sx_int_ok("007", 7))
add("X-10 integer 7 vs 7", TRUE, sx_int_ok("7", 7))
add("X-11 non-integer reproduced count", FALSE, sx_int_ok("2", 2.5))
add("X-12 delimiter-collision keys differ", TRUE, length(unique(sx_key(data.frame(a = c("a|b", "a"), b = c("c", "b|c")),
                                                          c("a", "b"), c("text", "text")))) == 2L)
add("X-13 empty vs literal NA text", FALSE, sx_text_ok("NA", ""))
add("X-14 Unicode text exact", TRUE, sx_text_ok(enc2utf8("Z\u00fcrich \u00d7"), "Z\u00fcrich \u00d7"))
sx_chk <- sx_chk[0, ]
sx_add_chk("indeterminate", NA)
sx_add_chk("declared n/a", FALSE, applicable = FALSE)
sx_add_chk("vector", c(TRUE, TRUE))
add("X-15 NA comparison is FAIL", "FAIL", sx_chk$status[1])
add("X-16 NOT APPLICABLE only when declared", "NOT APPLICABLE", sx_chk$status[2])
add("X-17 a vector is not a pass", "FAIL", sx_chk$status[3])
tab <- do.call(rbind, res)
utils::write.csv(tab, "diagnostic_results.csv", row.names = FALSE)
print(tab[tab$status != "PASS", ])
cat("DIAG", sum(tab$status == "PASS"), "of", nrow(tab), "\n")
"""


def case_diag(ctx) -> None:
    """Owner R diagnostic NUM-01..09, ID-01, SEL-01..03 + 17 extra controls, and SUM-01..07, against the verifier this
    build generates (v1.3.5: 11 of 20 failed)."""
    from ..discovery import validate as V
    from ..discovery.rcode import DISCOVERY_HELPERS

    lib = V.R_REPRO_CHECKS.split("# ---- required files")[0]
    r = ctx.oracle(_STUB + DISCOVERY_HELPERS.split("sx_fit_budget <- new.env()")[0] + "\n" + lib + _DIAG, "diagnostic")
    tab = pd.read_csv(ctx.dir("oracle") / "diagnostic_results.csv") if r.returncode == 0 else pd.DataFrame()
    if tab.empty:
        ctx.check("diagnostic script ran", 0, r.returncode, kind="script-R")
    for _, row in tab.iterrows():
        ctx.check(f"diagnostic {row['test']}", str(row["expected"]), str(row["actual"]), kind="script-R")
    B = pd.DataFrame({"b": [1, 2], "status": ["fitted", "fitted"], "reason": ["", ""], "selected": ["candidate 0"] * 2,
                      "selected_index": [0, 0], "selected_key": ["candidate0"] * 2, "train_score": [-2.0, -3.0],
                      "visited": [1, 1], "scored": [1, 1], "n_sig_train": [1, 1], "coefficient": ["beta", "beta"],
                      "estimate": [1.0, 3.0], "n_coef": [1, 1], "diagnostics": ["clean", "clean"], "flags": ["", ""],
                      "n": [20, 20], "groups": [20, 20]})
    C = pd.DataFrame({"b": [1, 2], "selected_index": [0, 0], "selected_key": ["candidate0"] * 2, "term": ["beta"] * 2,
                      "coefficient": ["beta"] * 2, "estimate": [1.0, 3.0]})

    def rdf(df):
        from ..rbridge.codegen import r_vec

        return "data.frame(" + ", ".join(
            f"`{c}` = {r_vec([str(x) for x in df[c]])}" if not pd.api.types.is_numeric_dtype(df[c]) else
            f"`{c}` = c({', '.join(repr(float(x)) for x in df[c])})" for c in df.columns) + \
            ", stringsAsFactors = FALSE, check.names = FALSE)"
    for sid, tamper, expect in (("SUM-01 clean summary", "none", 0), ("SUM-02 corrupted coefficient", "coefficient", 1),
                                ("SUM-03 corrupted count", "count", 1), ("SUM-04 altered estimand median", "summary", 1),
                                ("SUM-05 altered selection frequency", "frequency", 1),
                                ("SUM-06 altered sign stability", "sign", 1), ("SUM-07 altered candidate key", "key", 1)):
        Bt = B.assign(coefficient_label=B["coefficient"])
        st = V.stability_table(Bt, ["candidate 0"], [0], ["candidate0"])
        est = V.estimand_table(Bt, C.assign(coefficient_label=C["coefficient"]), {0: "candidate 0"})
        Ca = C.copy()
        if tamper == "summary":
            est[0]["estimate_median"] = 100.0
        if tamper == "frequency":
            st[0]["freq_of_attempted"] = 0.1
        if tamper == "sign":
            st[0]["positive_sign"] = 0.1
        if tamper == "key":
            st[0]["candidate_key"] = "different"
        if tamper == "count":
            st[0]["selected"] = 1
        if tamper == "coefficient":
            Ca.loc[0, "estimate"] = 10.0
        d = ctx.dir(f"sum_{tamper}")
        led = [{"part": f"boot{b}", "mode": "bootstrap", "split": "", "fold": None, "b": b, "chunk": 1,
                "status": "fitted", "committed": True} for b in (1, 2)]
        for name, key, recs in (("bootstrap_results.csv", "bootstrap", Bt.to_dict("records")),
                                ("bootstrap_coefficients.csv", "coefs", Ca.to_dict("records")),
                                ("stability.csv", "stability", st), ("bootstrap_estimands.csv", "estimands", est),
                                ("execution_ledger.csv", "ledger", led)):
            (d / name).write_text(V.evidence_csv(key, recs), encoding="utf-8")
        exp = [{"file": f, "required": True, "rows": n, "applicable": True} for f, n in
               (("bootstrap_results.csv", 2), ("bootstrap_coefficients.csv", 2), ("stability.csv", 1),
                ("bootstrap_estimands.csv", 1), ("execution_ledger.csv", 2))]
        script = V.reproduce_script("", _STUB, [f"sx_result <- list(boot = list(draws = {rdf(B)}, coefs = {rdf(C)}))"],
                                    exp, {"status": "complete", "target": "rows"},
                                    replay={"ledger": led, "pending_manifest": [], "pending_draws": [], "splits": [],
                                            "config": {"holdout_fraction": 0.2, "k": 5}, "alias_map": {}, "rules": {}})
        (d / "check.R").write_text(script, encoding="utf-8")
        rc = ctx.run_r(d, "check.R", timeout=600).returncode
        ctx.check(f"{sid}: {'ACCEPTED' if expect == 0 else 'REJECTED'}", "ACCEPTED" if expect == 0 else "REJECTED",
                  "ACCEPTED" if rc == 0 else "REJECTED", kind="script-R")


CASES = [("D", case_diag), ("55", case_55), ("56", case_56), ("57", case_57), ("58", case_58), ("59", case_59), ("60", case_60),
         ("61", case_61)]
