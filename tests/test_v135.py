"""StatExplore v1.3.5 corrective patch: tests per audit finding AUD134-01 ... AUD134-11.

Oracles are independent of the code under test: direct R calls (stats::pf / pchisq with log.p = TRUE, lm), brute-force
enumeration written here, a log-domain BH / Holm written here, hand-built historical rows, and deterministic fake clocks
and R workers for the execution-control paths. Run on v1.3.4 (dd7313a) these fail (red), on v1.3.5 they pass.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from statexplore.core.importers import import_file

FX = Path(__file__).parent / "fixtures" / "v135"
FX4 = Path(__file__).parent / "fixtures" / "v134"
LN10 = math.log(10)


def fx(name: str, folder: Path = FX):
    return import_file(folder / f"StatExplore_{name}.csv")


def rscript(code: str, cwd: Path, timeout: int = 1800, script: str = "oracle.R") -> subprocess.CompletedProcess:
    from statexplore.rbridge.locate import find_rscript

    lib = Path(__file__).resolve().parents[1] / "r" / "library"
    env = dict(os.environ, R_LIBS=str(lib), R_LIBS_USER=str(lib), OPENBLAS_NUM_THREADS="1")
    (cwd / script).write_text(code, encoding="utf-8")
    return subprocess.run([str(find_rscript()), "--vanilla", str(cwd / script)], cwd=cwd, capture_output=True,
                          text=True, timeout=timeout, env=env)


def run_r(cwd: Path, script: str, timeout: int = 1800) -> subprocess.CompletedProcess:
    from statexplore.rbridge.locate import find_rscript

    lib = Path(__file__).resolve().parents[1] / "r" / "library"
    env = dict(os.environ, R_LIBS=str(lib), R_LIBS_USER=str(lib), OPENBLAS_NUM_THREADS="1")
    return subprocess.run([str(find_rscript()), "--vanilla", script], cwd=cwd, capture_output=True, text=True,
                          timeout=timeout, env=env)


def helpers() -> str:
    from statexplore.analysis.covstruct import MIXED_R_HELPERS
    from statexplore.discovery.rcode import DISCOVERY_HELPERS
    from statexplore.rbridge.codegen import IDENTITY_R_HELPERS, R_HELPERS

    return "\n".join([R_HELPERS, IDENTITY_R_HELPERS, MIXED_R_HELPERS, DISCOVERY_HELPERS])


S55L = dict(outcome="y", candidates=["x1", "x3", "x2"], family="gaussian", strategy="focal", covariate_mode="focal",
            max_predictors=2, workers=1, max_models=None, max_minutes=None)


def test_fixture_bytes_match_manifest():
    for line in (FX / "SHA256SUMS.txt").read_text().splitlines():
        h, name = line.split(maxsplit=1)
        assert hashlib.sha256((FX / name.lstrip("./")).read_bytes()).hexdigest() == h, name


# ---- AUD134-01: log-tail selection in the legacy Validation Workspace ------------------------------------------------

@pytest.mark.r
def test_shared_selector_orders_underflowed_tails_and_ties(tmp_path):
    """Ordinary p of both is 0 (tie); the log tails (log10 -434 and -651) decide, in either input order; genuine ties
    take the lower position; NA (failed / withheld) never wins; the primary score is NA for withheld or failed fits."""
    code = helpers() + r"""
a <- c(-434, -651) * log(10)
stopifnot(identical(exp(a), c(0, 0)))
stopifnot(sx_select_min(a) == 2L, sx_select_min(rev(a)) == 1L)
stopifnot(sx_select_min(c(-5, -5, -1)) == 1L, sx_select_min(c(NA, -1, NA)) == 2L, is.na(sx_select_min(c(NA, NA))))
stopifnot(sx_select_min(c(NA, -Inf, -3)) == 2L)
ok <- list(status = "ok", tests = list(list(log_p = -3)))
stopifnot(sx_primary_logp(ok) == -3)
stopifnot(is.na(sx_primary_logp(list(status = "ok", tests = list(list(log_p = -3, withheld = "negative LR"))))))
stopifnot(is.na(sx_primary_logp(list(status = "invalid", tests = NULL))), is.na(sx_primary_logp(NULL)))
stopifnot(is.na(sx_primary_logp(list(status = "ok", tests = list(list(p = 0))))))
# ordinary representable values: the shared tails are R's own log upper tails
stopifnot(isTRUE(all.equal(stats::pchisq(30, 2, lower.tail = FALSE, log.p = TRUE), log(stats::pchisq(30, 2, lower.tail = FALSE)))))
cat("OK\n")
"""
    res = rscript(code, tmp_path)
    assert res.returncode == 0 and "OK" in res.stdout, res.stderr[-2000:]


def _legacy_run(d, **kw):
    from statexplore.discovery import engine
    from statexplore.discovery.space import SearchSettings

    run = engine.create_run(d, SearchSettings(**{**S55L, **kw}))
    engine.execute(run, d)
    return run


def _oracle_55l(tmp_path: Path, rows_expr: str = "TRUE") -> pd.DataFrame:
    """Independent fresh lm fits (no StatExplore code): F test log10 p of each focal pair on the given rows."""
    csv = (FX / "StatExplore_Test55L_LegacyUnderflow.csv").as_posix()
    code = f"""d <- utils::read.csv("{csv}")
d <- d[{rows_expr}, , drop = FALSE]
one <- function(a, b) {{ f1 <- stats::lm(stats::reformulate(c(a, b, paste0(a, ":", b)), "y"), d)
  f0 <- stats::lm(stats::reformulate(c(a, b), "y"), d); A <- stats::anova(f0, f1)
  c(F = A$F[2], log10_p = stats::pf(A$F[2], A$Df[2], A$Res.Df[2], lower.tail = FALSE, log.p = TRUE) / log(10)) }}
out <- rbind(one("x1", "x3"), one("x1", "x2"), one("x3", "x2"))
utils::write.csv(data.frame(candidate = 0:2, out), "oracle.csv", row.names = FALSE)
"""
    res = rscript(code, tmp_path)
    assert res.returncode == 0, res.stderr[-2000:]
    return pd.read_csv(tmp_path / "oracle.csv")


@pytest.mark.r
@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_55l_legacy_stability_selects_by_log_tail(r_session, tmp_path):
    """Both strong interactions underflow (p == 0 in double precision). v1.3.4 tied them and picked the first (#0,
    x1:x3); the log tails put x1:x2 (#1) first in every resample. Full-data log10 p equals an independent lm + pf."""
    from statexplore.analysis import validation

    d = fx("Test55L_LegacyUnderflow")
    run = _legacy_run(d)
    oracle = _oracle_55l(tmp_path)
    assert (oracle["log10_p"][:2] < -330).all() and oracle["log10_p"][1] < oracle["log10_p"][0]
    res, _ = validation.stability(r_session, d, run, validation.ResampleSpec(resamples=5, fraction=0.8, seed=11))
    t = res.tables[0]
    tab = pd.DataFrame(t.rows, columns=t.columns).set_index("candidate")
    assert tab.loc[1, "ranked_first"] == 1.0 and tab.loc[0, "ranked_first"] == 0.0
    assert float(tab.loc[0, "p_full_data"]) == 0.0 and float(tab.loc[1, "p_full_data"]) == 0.0
    for j in (0, 1, 2):
        assert tab.loc[j, "log10_p_full_data"] == pytest.approx(oracle.loc[j, "log10_p"], abs=1e-6 * max(1, abs(
            oracle.loc[j, "log10_p"])))
    assert list(tab.index[:2]) == [1, 0]            # ordered by significance share, then the exact log tail
    trace = res.tables[1]
    tr = pd.DataFrame(trace.rows, columns=trace.columns)
    assert len(tr) == 15 and tr.loc[tr["selected"].astype(bool), "candidate"].eq(1).all()
    assert run.validation[1].startswith("stability") and "ranked first in 100%" in run.validation[1]


@pytest.mark.r
@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_55l_legacy_procedure_cv_selects_by_log_tail(r_session, tmp_path):
    """Full-procedure CV: every training fold selects x1:x2 (#1) by its log tail; the training log10 p of the selected
    candidate equals an independent fresh lm fit on the same training rows (fold assignment replayed from the seed)."""
    from statexplore.analysis import validation

    d = fx("Test55L_LegacyUnderflow")
    run = _legacy_run(d)
    res, _ = validation.procedure_cv(r_session, d, run, validation.ResampleSpec(folds=3, seed=5))
    t = pd.DataFrame(res.tables[0].rows, columns=res.tables[0].columns)
    assert len(t) == 3 and t["selected_index"].eq(1).all()
    assert (t["p_training"] == 0).all() and (t["log10_p_training"] < -300).all()
    idv = validation._id_var(d)
    group = f"as.character(d0${idv})" if idv else "rownames(d0)"
    csv = (FX / "StatExplore_Test55L_LegacyUnderflow.csv").as_posix()
    code = f"""d0 <- utils::read.csv("{csv}"); rownames(d0) <- seq_len(nrow(d0))
set.seed(5); g <- {group}; u <- unique(g); gf <- sample(rep(seq_len(3), length.out = length(u))); fold <- gf[match(g, u)]
out <- sapply(1:3, function(f) {{ d <- d0[fold != f, ]
  f1 <- stats::lm(y ~ x1 + x2 + x1:x2, d); f0 <- stats::lm(y ~ x1 + x2, d); A <- stats::anova(f0, f1)
  stats::pf(A$F[2], A$Df[2], A$Res.Df[2], lower.tail = FALSE, log.p = TRUE) / log(10) }})
utils::write.csv(data.frame(fold = 1:3, log10_p = out), "oracle.csv", row.names = FALSE)
"""
    res2 = rscript(code, tmp_path)
    assert res2.returncode == 0, res2.stderr[-2000:]
    o = pd.read_csv(tmp_path / "oracle.csv")
    np.testing.assert_allclose(t.sort_values("fold")["log10_p_training"], o["log10_p"], rtol=1e-6)


# ---- AUD134-02: adjusted-p ranking on log tails ----------------------------------------------------------------------

def _rows(log10s: list[float | None], kind: str = "focal") -> list[dict]:
    out = []
    for i, l10 in enumerate(log10s):
        lp = None if l10 is None else l10 * LN10
        p = None if lp is None else math.exp(lp)
        out.append({"idx": i, "key": f"k{i}", "focal": ["a", "b"], "covariates": [], "kind": kind, "status": "ok",
                    "reason": "", "flags": "", "speculative": False, "formula": f"y ~ m{i}", "interaction": f"t{i}",
                    "p": p, "log_p": lp,
                    "tests": [{"test_id": 0, "type": "focal_interaction", "label": f"t{i}", "p": p, "log_p": lp,
                               "df1": 1.0, "statistic": 1.0, "ci_applicable": "unavailable"}]})
    return out


def _fake_run(rows, correction="BH"):
    from statexplore.discovery.engine import DiscoveryRun
    from statexplore.discovery.space import SearchSettings

    run = DiscoveryRun(settings=SearchSettings(outcome="y", candidates=["a", "b"], correction=correction),
                       family="gaussian", total=len(rows), signature="x")
    run.rows, run.status = rows, "complete"
    return run


def _oracle_log_adjust(lp: np.ndarray, method: str) -> np.ndarray:
    """Independent log-domain BH / Holm (plain loops, no shared code)."""
    m = len(lp)
    order = sorted(range(m), key=lambda i: (lp[i], i))
    adj = [0.0] * m
    if method == "BH":
        run = 0.0
        for r in range(m, 0, -1):
            i = order[r - 1]
            run = min(run, lp[i] + math.log(m) - math.log(r)) if r < m else min(0.0, lp[i])
            adj[i] = min(run, 0.0)
    else:
        run = -math.inf
        for r, i in enumerate(order, start=1):
            run = max(run, lp[i] + math.log(m - r + 1))
            adj[i] = min(run, 0.0)
    return np.array(adj)


@pytest.mark.parametrize("correction", ["BH", "holm"])
@pytest.mark.parametrize("reverse", [False, True])
def test_adjusted_ranking_uses_log_tails(correction, reverse):
    """Two underflowed candidates (log10 -651 and -434) plus representable, tied and missing ones: raw, adjusted, BH and
    Holm rankings follow the log tails in any input order; ties by raw log p then index; missing last."""
    from statexplore.discovery import engine

    l10 = [-434.0, -651.0, -3.0, -3.0, None, -0.2]
    rows = _rows(l10)
    if reverse:
        rows = rows[::-1]
    tf = engine.tests_frame(_fake_run(rows, correction), interpret=False)
    fam = tf[tf["in_family"]].sort_values("idx")
    for meth, col in (("BH", "log_p_bh"), ("holm", "log_p_holm")):
        np.testing.assert_allclose(fam[col].to_numpy(), _oracle_log_adjust(fam["log_p"].to_numpy(), meth), rtol=1e-12)
    assert (fam.loc[fam["idx"].isin([0, 1]), "p_adj"] == 0).all()        # the ordinary adjusted p underflows too
    for by in ("p", "log10_p", "p_adj", "p_bh", "p_holm", "log_p_adj"):
        order = list(engine.discovery_ranking(tf, by=by)["idx"])
        assert order[:2] == [1, 0], (by, order)
        assert order[2:4] == [2, 3] and order[-1] == 5 and 4 not in order       # tie by index; missing not in family


def test_adjusted_ranking_is_within_family():
    from statexplore.discovery import engine

    rows = _rows([-500.0, -400.0])
    rows[0]["tests"][0]["type"] = "overall"
    tf = engine.tests_frame(_fake_run(rows), interpret=False)
    rk = engine.discovery_ranking(tf, by="p_adj")
    # families in their fixed order (focal interaction before whole-model), never one cross-family list
    assert list(rk["test_type"]) == ["focal_interaction", "overall"]
    assert list(engine.discovery_ranking(tf, by="p")["idx"]) == [0, 1]


def test_unknown_magnitude_is_labelled_not_zero():
    from statexplore.discovery import engine

    rows = _rows([-2.0, -1.0])
    for r in rows:            # an older search: p underflowed, no log tail recorded
        r["log_p"] = r["tests"][0]["log_p"] = None
    rows[0]["p"] = rows[0]["tests"][0]["p"] = 0.0
    tf = engine.tests_frame(_fake_run(rows), interpret=False)
    r0 = tf[tf["idx"] == 0].iloc[0]
    assert r0["log_p"] == -np.inf and "magnitude not recorded" in r0["log_p_source"]
    assert list(engine.discovery_ranking(tf, by="p_adj")["idx"]) == [0, 1]


def test_gui_table_sorts_adjusted_p_and_log10_by_log_tail(qtbot):
    from PySide6.QtCore import Qt

    from statexplore.discovery import engine
    from statexplore.gui.pages.discovery import DISCOVERY_COLS, ResultTable

    rows = _rows([-434.0, -651.0, -3.0, None])
    rows[3]["log_p"] = rows[3]["tests"][0]["log_p"] = None      # p missing
    rows[3]["p"] = rows[3]["tests"][0]["p"] = None
    tf = engine.tests_frame(_fake_run(rows), interpret=False)
    t = ResultTable(DISCOVERY_COLS)
    qtbot.addWidget(t)
    t.set_frame(tf)
    cols = list(t.model.frame.columns)
    for name in ("p (adjusted)", "log10 p", "p"):
        t.model.sort(cols.index(name), Qt.AscendingOrder)
        shown = [int(t.model.frame.iloc[t.model.source_row(i)]["#"]) for i in range(t.model.rowCount())]
        assert shown[:3] == [1, 0, 2], (name, shown)
    t.model.sort(cols.index("p (adjusted)"), Qt.AscendingOrder)
    csv = t.csv_text()
    assert csv.index("y ~ m1") < csv.index("y ~ m0") < csv.index("y ~ m2")


# ---- AUD134-03: resume only when every retained result is compatible -------------------------------------------------

def _52d():
    from statexplore.core.project import Project

    p = Project.open(FX4 / "StatExplore_Test52D_Persistence.sxp")
    return p, next(r for r in p.searches if r.id == "ea77ac2fedc4")


def _fingerprint(run) -> str:
    return hashlib.sha256(json.dumps([run.meta_dict(), run.rows], sort_keys=True, default=str).encode()).hexdigest()


def test_52d_historical_resume_is_refused_with_reason():
    """Saved 52D: grammar 2, fit rules 3, 5 of 15 done. v1.3.4 allowed the resume (mixing fit rules 3 and 4 in one
    multiplicity family); v1.3.5 refuses with the reason and offers a new linked search. The original stays unchanged."""
    from statexplore.discovery import engine

    p, run = _52d()
    before = _fingerprint(run)
    assert run.engine.get("fit_rules") == 3 and run.grammar == 2 and run.n_done == 5 and len(p.dataset.raw) == 960
    assert not run.resumable
    why = run.resume_reason(p.dataset)
    assert "fit rules 3" in why and "new search" in why
    with pytest.raises(engine.ResumeRefused, match="fit rules 3"):
        engine.execute(run, p.dataset)
    assert _fingerprint(run) == before
    new = engine.restart_from(run, p.dataset)
    assert new.id != run.id and new.parent_id == run.id and new.rows == [] and new.total == 15
    assert new.lineage["parent_fit_rules"] == 3 and new.lineage["fit_rules"] == engine.FIT_RULES
    assert engine.cache_from(new, p.searches) == {}            # no fit-rules-3 row can enter the new search
    assert _fingerprint(run) == before
    back = engine.DiscoveryRun.from_meta(json.loads(json.dumps(new.meta_dict())), [])
    assert back.parent_id == run.id and back.lineage["parent"] == run.id


def _compatible_partial(ds, n_done=5, total=15):
    """A v1.3.5-compatible partial search built without R (engine record of this version)."""
    from statexplore.discovery import engine
    from statexplore.discovery.space import SearchSettings, candidate_key, enumerate_candidates

    p, old = _52d()
    s = SearchSettings.from_dict(old.settings.to_dict())
    run = engine.create_run(ds, s)
    run.status, run.n_done = "partial", n_done
    run.engine = {"r_version": "4.6.2", "packages": {k: "1" for k in engine.ENGINE_PACKAGES}, "fit_rules": engine.FIT_RULES}
    run.rows = [{"idx": c.index, "key": candidate_key(c), "status": "ok"} for c in enumerate_candidates(s)
                if c.index < n_done]
    return run


@pytest.mark.parametrize("mutation, expect", [
    ("none", ""),
    ("old_rules", "fit rules 3"),
    ("no_rules", "fit rules (not recorded)"),
    ("no_engine", "no recorded R / package versions"),
    ("grammar", "grammar"),
    ("bad_key", "key differs"),
    ("dup_idx", "duplicate"),
    ("gap", "have no result"),
    ("no_key", "no candidate key"),
    ("no_fit_id", "no recorded fit identity"),
    ("out_of_range", "outside the search space"),
])
def test_resume_compatibility_matrix(mutation, expect):
    p, _ = _52d()
    run = _compatible_partial(p.dataset)
    if mutation == "old_rules":
        run.engine["fit_rules"] = 3
    elif mutation == "no_rules":
        del run.engine["fit_rules"]
    elif mutation == "no_engine":
        run.engine = {}
    elif mutation == "grammar":
        run.grammar, run.settings.strategy, run.settings.min_predictors = 2, "time", 2
        run.settings.required = []
    elif mutation == "bad_key":
        run.rows[2]["key"] = "focal|something else"
    elif mutation == "dup_idx":
        run.rows.append(dict(run.rows[0]))
    elif mutation == "gap":
        del run.rows[1]
    elif mutation == "no_key":
        run.rows[0]["key"] = ""
    elif mutation == "no_fit_id":
        run.fit_id = ""
    elif mutation == "out_of_range":
        run.rows[-1]["idx"] = 99
    why = run.compatibility(p.dataset, deep=True)
    assert (why == "") if not expect else (expect in why), why
    assert run.resumable == (not expect) or mutation in ("bad_key",)      # the key check is deep (execute) only


def test_resume_refused_on_changed_data_and_coding():
    p, _ = _52d()
    run = _compatible_partial(p.dataset)
    assert run.resume_reason(p.dataset) == ""
    ds = p.dataset
    ds.raw.loc[ds.raw.index[0], "Y"] += 1.0          # same n, different data
    ds.source.sha256 = "0" * 64
    assert "data signature differs" in run.resume_reason(ds)
    p, _ = _52d()
    run = _compatible_partial(p.dataset)
    p.dataset.set_variable_field("X01", "display_name", "Renamed X01")      # changed coding / metadata
    assert "data signature differs" in run.resume_reason(p.dataset)


@pytest.mark.r
@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_same_version_partial_resumes_to_exact_15_and_engine_mismatch_refused(tmp_path):
    """A fresh v1.3.5 partial (budget 5), saved and reopened, resumes with budget 10 to exactly 15 rows - no duplicates,
    identical to a fresh exhaustive search. A different recorded R / package version is refused by the worker check
    before anything is committed."""
    from statexplore.core.project import Project
    from statexplore.discovery import engine
    from statexplore.discovery.space import SearchSettings

    p, old = _52d()
    s = SearchSettings.from_dict(old.settings.to_dict())
    s.mode, s.max_models, s.workers = "D", 5, 2
    run = engine.create_run(p.dataset, s)
    p.searches.append(run)
    engine.execute(run, p.dataset)
    assert run.status == "partial" and run.n_done == 5
    path = tmp_path / "fresh.sxp"
    p.save(path)
    p2 = Project.open(path)
    r2 = next(r for r in p2.searches if r.id == run.id)
    assert r2.resumable and r2.resume_reason(p2.dataset) == ""
    # engine mismatch: refused at the worker check, run unchanged
    saved_engine = dict(r2.engine)
    r2.engine = {**saved_engine, "r_version": "0.0.0"}
    before = _fingerprint(r2)
    with pytest.raises(engine.ResumeRefused, match="R 0.0.0"):
        engine.execute(r2, p2.dataset)
    assert _fingerprint(r2) == before
    r2.engine = saved_engine
    r2.settings.max_models = 10
    engine.execute(r2, p2.dataset, cache=engine.cache_from(r2, p2.searches))
    assert r2.status == "complete" and r2.n_done == 15 and sorted(r["idx"] for r in r2.rows) == list(range(15))
    s3 = SearchSettings.from_dict(old.settings.to_dict())
    s3.workers, s3.mode, s3.max_models = 2, "C", None          # the fresh exhaustive reference
    ref = engine.create_run(p2.dataset, s3)
    engine.execute(ref, p2.dataset)
    a = {r["idx"]: (r["loglik"], r["log_p"]) for r in r2.rows}
    b = {r["idx"]: (r["loglik"], r["log_p"]) for r in ref.rows}
    for i in range(15):
        assert a[i][0] == pytest.approx(b[i][0], abs=1e-6) and a[i][1] == pytest.approx(b[i][1], abs=1e-6)
    assert len(r2.executions) == 2 and r2.executions[1]["start"] == 5


def test_historical_gee_search_with_old_rules_is_refused():
    """A GEE search fitted under fit rules 3 (one working correlation for every structure, V134-005) is a known
    affected fit: it must not be continued."""
    from statexplore.discovery import engine
    from statexplore.discovery.space import SearchSettings, candidate_key, enumerate_candidates

    d = import_file(FX4 / "StatExplore_Test48D_GEE.csv")
    s = SearchSettings(outcome="Y", candidates=["X"], required=["TIME"], participant="ID", time="TIME",
                       study="longitudinal", workflow="structure", structures=["gee_ind", "gee_exch", "gee_ar1"],
                       family="gaussian", max_predictors=2)
    run = engine.create_run(d, s)
    assert run.family == "gee" and run.total == 3
    run.status, run.n_done = "partial", 1
    run.engine = {"r_version": "4.6.2", "packages": {}, "fit_rules": 3}
    run.rows = [{"idx": 0, "key": candidate_key(next(enumerate_candidates(s))), "status": "ok"}]
    assert not run.resumable and "fit rules 3" in run.resume_reason(d)


# ---- AUD134-04: minimum CANDIDATE predictors in every strategy -------------------------------------------------------

def _oracle_space(s) -> set[str]:
    """Brute-force grammar oracle (written independently of space.py): every model a strategy may produce, as a
    canonical set of terms, filtered by the minimum / maximum rules stated in the documentation."""
    cands, req, t = list(s.candidates), list(s.required), s.time
    room = s.max_predictors - len(req)
    lo = max(1, s.min_predictors)
    out = set()

    def canon(terms) -> str:
        return "+".join(sorted(":".join(sorted(x)) for x in {tuple(sorted(x)) for x in terms}))

    orders = [o for o in (2, 3) if o <= s.max_order and o <= room]
    if s.strategy in ("main",) or (s.include_main and s.strategy in ("focal", "factorial", "time")):
        for k in range(lo, min(room, len(cands)) + 1):
            for sub in combinations(cands, k):
                out.add("main|" + canon([(v,) for v in sub] + [(v,) for v in req]))
    if s.strategy == "focal":
        sizes = [o for o in orders] if s.covariate_mode == "focal" else range(2, room + 1)
        for size in sizes:
            if size < lo:
                continue
            for sub in combinations(cands, size):
                for o in orders:
                    if o > size or (s.covariate_mode == "focal" and o != size):
                        continue
                    for f in combinations(sub, o):
                        hier = [c for k in range(1, len(f) + 1) for c in combinations(f, k)]
                        out.add("focal|" + canon(hier + [(v,) for v in sub if v not in f] + [(v,) for v in req]))
    if s.strategy == "time":
        for k in range(2, room + 1):
            if k - 1 < lo:
                continue
            for rest in combinations(cands, k - 1):
                for o in orders:
                    if o > k or (s.covariate_mode == "focal" and k != o):
                        continue
                    for f in combinations(rest, o - 1):
                        ff = (t,) + f
                        hier = [c for j in range(1, len(ff) + 1) for c in combinations(ff, j)]
                        out.add("focal|" + canon(hier + [(v,) for v in rest if v not in f] + [(v,) for v in req]))
    return out


def _space_keys(s) -> set[str]:
    from statexplore.discovery.space import enumerate_candidates, full_terms, model_terms

    out = []
    for c in enumerate_candidates(s):
        terms = full_terms(c) if c.kind in ("main", "combo", "factorial") else model_terms(c)[0]
        kind = "main" if c.kind == "main" else "focal"
        out.append(kind + "|" + "+".join(sorted(":".join(sorted(x)) for x in {tuple(sorted(x)) for x in terms})))
    assert len(out) == len(set(out)), "duplicate candidates"
    return set(out)


def test_focal_minimum_example_from_the_audit():
    """3 optional candidates, required Time, minimum 3 candidates, maximum 4 distinct predictors (Time counted once):
    required-only focal mode cannot form a 3-candidate model (refused); subset mode gives exactly 3 models, each with
    all 3 candidates (v1.3.4: 3 and 6)."""
    from statexplore.discovery.space import (
        SearchSettings, SearchSpaceError, count_candidates, enumerate_candidates, validate_settings,
    )

    s = SearchSettings(outcome="Y", candidates=["X01", "X02", "X03"], required=["Time"], strategy="focal",
                       covariate_mode="focal", min_predictors=3, max_predictors=4, max_order=2)
    assert count_candidates(s) == 0 and list(enumerate_candidates(s)) == []
    d = import_file(FX4 / "StatExplore_Test51F_Null_13.csv")
    with pytest.raises(SearchSpaceError, match="Subsets"):
        validate_settings(d, s)
    s.covariate_mode = "subsets"
    cs = list(enumerate_candidates(s))
    assert count_candidates(s) == len(cs) == 3
    assert all(set(c.focal) | set(c.covariates) == {"X01", "X02", "X03"} for c in cs)
    assert all("Time" not in c.focal + c.covariates and c.required == ("Time",) for c in cs)


GRAMMAR_CASES = [(st, cm, lo, hi, o) for st in ("focal", "time", "main", "combinations", "factorial")
                 for cm in ("focal", "subsets") for lo in (1, 2, 3, 4) for hi in (3, 4, 5) for o in (2, 3)
                 if lo <= hi and (cm == "focal" or st in ("focal", "time"))]


@pytest.mark.parametrize("strategy, cov_mode, lo, hi, order", GRAMMAR_CASES)
def test_counts_and_enumeration_match_bruteforce_oracle(strategy, cov_mode, lo, hi, order):
    from statexplore.discovery.space import SearchSettings, count_candidates, enumerate_candidates

    req = [] if strategy == "time" else ["R1"]
    s = SearchSettings(outcome="Y", candidates=["A", "B", "C", "D"], required=req, strategy=strategy,
                       covariate_mode=cov_mode, min_predictors=lo, max_predictors=hi, max_order=order,
                       include_main=strategy in ("focal", "combinations"), participant="ID" if strategy == "time" else
                       None, time="T" if strategy == "time" else None)
    cs = list(enumerate_candidates(s))
    assert count_candidates(s) == len(cs) and [c.index for c in cs] == list(range(len(cs)))
    from statexplore.discovery.space import candidate_key

    assert len({candidate_key(c) for c in cs}) == len(cs)
    for c in cs:            # every model honours the minimum and the maximum (required counted once, never minimum)
        n_cand = len(set(c.focal + c.covariates) - {"T"})
        assert lo <= n_cand and n_cand + len(set(c.required)) + (1 if "T" in c.focal else 0) <= hi
    if strategy in ("focal", "time", "main"):
        assert _space_keys(s) == _oracle_space(s)
    for start in range(0, len(cs), 3):               # resuming from any index gives the same tail (bounded skip)
        assert [c.index for c in enumerate_candidates(s, start=start)] == list(range(start, len(cs)))


def test_large_counts_preserved_without_materializing():
    from statexplore.discovery.space import SearchSettings, count_candidates

    base = dict(outcome="Y", candidates=[f"X{i:02d}" for i in range(1, 31)], participant="StudentID", time="Time")
    assert count_candidates(SearchSettings(**base, strategy="focal", covariate_mode="focal", max_predictors=2)) == 435
    huge = SearchSettings(**base, required=["Time"], strategy="combinations", include_main=True, max_predictors=6)
    assert count_candidates(huge) == 147_713_444
    huge.include_main = False
    assert count_candidates(huge) == 147_539_008
    main = SearchSettings(**base, required=["Time"], strategy="main", max_predictors=6)
    assert count_candidates(main) == 174_436


def test_grammar4_changes_are_detected_for_historical_focal_searches():
    from statexplore.discovery.engine import DiscoveryRun
    from statexplore.discovery.space import GRAMMAR_VERSION, SearchSettings

    assert GRAMMAR_VERSION == 4
    s = SearchSettings(outcome="Y", candidates=["A", "B", "C"], strategy="focal", min_predictors=3, max_predictors=3)
    run = DiscoveryRun(settings=s, family="gaussian", total=3, signature="x", grammar=3)
    assert "grammar 4" in run.grammar_problem()
    s.min_predictors = 2
    assert run.grammar_problem() == ""         # unchanged list: still compatible


def test_min_control_disabled_for_workflow_b(qtbot):
    from statexplore.gui.main_window import MainWindow
    from statexplore.gui.state import AppState

    w = MainWindow(AppState())
    try:
        pg = w.pages["Automated Model Discovery"]
        pg.study.setCurrentIndex(pg.study.findData("longitudinal"))
        pg.workflow.setCurrentIndex(pg.workflow.findData("structure"))
        pg._study_changed()
        assert not pg.min_pred.isEnabled() and not pg.strategy.isEnabled()
        pg.workflow.setCurrentIndex(pg.workflow.findData("fixed"))
        pg._study_changed()
        assert pg.min_pred.isEnabled() and pg.strategy.isEnabled()
    finally:
        w.state.dirty = False
        w.state.shutdown_r()
        w.close()


# ---- AUD134-05 / AUD134-08: fixed candidates from their saved specification --------------------------------------------

def _time_grammar2_run(ds):
    """Historical grammar-2 time-strategy search: #0 = Time:X01 WITHOUT X02 (grammar 2 ignored the minimum); grammar 3+
    puts Time:X01 WITH X02 at #0."""
    from statexplore.discovery import engine
    from statexplore.discovery.space import Candidate, SearchSettings, candidate_key, enumerate_candidates

    s = SearchSettings(outcome="Y", candidates=["X01", "X02"], participant="StudentID", time="Time", strategy="time",
                       covariate_mode="subsets", max_predictors=3, min_predictors=2, study="longitudinal",
                       family="gaussian")
    run = engine.create_run(ds, s)
    run.grammar = 2
    old0 = Candidate(0, ("Time", "X01"), (), ())
    cur0 = next(enumerate_candidates(s))
    assert cur0.covariates == ("X02",) and candidate_key(cur0) != candidate_key(old0)
    run.rows = [{"idx": 0, "key": candidate_key(old0), "focal": ["Time", "X01"], "covariates": [], "kind": "focal",
                 "inter": [], "structure": "ri", "status": "ok", "model_id": "m0"}]
    run.status, run.n_done = "complete", run.total
    return run, old0


def test_historical_fixed_candidate_is_the_saved_model_not_the_current_index():
    from statexplore.discovery import validate as V
    from statexplore.discovery.space import candidate_key

    d = import_file(FX4 / "StatExplore_Test51F_Null_13.csv")
    run, old0 = _time_grammar2_run(d)
    c = V.fixed_candidate(run, 0)
    assert c.focal == ("Time", "X01") and c.covariates == () and candidate_key(c) == candidate_key(old0)
    r = V.ValidationRunner(None, d, run, V.ValidationConfig(target="participant", modes=["holdout"], pipeline="fixed",
                                                            candidate=0))
    assert [candidate_key(x) for x in r.cands] == [candidate_key(old0)] and r.keys == [candidate_key(old0)]
    assert "sx_val_keys" in r._setup() and candidate_key(old0) in r._setup()
    # whole-procedure validation of the changed grammar is refused (never a silently different search)
    cfg = V.ValidationConfig(target="participant", modes=["holdout"], pipeline="pipeline")
    assert any("grammar 3" in p for p in V.feasibility(d, run, cfg)["problems"])
    with pytest.raises(V.ValidationError, match="grammar"):
        V.ValidationRunner(None, d, run, cfg)


@pytest.mark.parametrize("damage", ["missing_row", "missing_key", "other_spec", "reordered"])
def test_fixed_candidate_refusals(damage):
    from statexplore.discovery import validate as V

    d = import_file(FX4 / "StatExplore_Test51F_Null_13.csv")
    run, _ = _time_grammar2_run(d)
    row = run.rows[0]
    if damage == "missing_row":
        run.rows = []
    elif damage == "missing_key":
        row["key"] = ""
    elif damage == "other_spec":
        row["covariates"] = ["X02"]                 # same label, different key
    elif damage == "reordered":
        run.settings.candidates = ["X02", "X01"]    # display order changed: the saved key still decides
        run.grammar = 4
        c = V.fixed_candidate(run, 0)
        assert c.focal == ("Time", "X01")
        return
    with pytest.raises(V.ValidationError):
        V.fixed_candidate(run, 0)


def test_pipeline_pool_with_mismatching_saved_keys_is_refused():
    from statexplore.discovery import engine
    from statexplore.discovery import validate as V
    from statexplore.discovery.space import SearchSettings, candidate_key, enumerate_candidates

    d = import_file(FX4 / "StatExplore_Test51F_Null_13.csv")
    s = SearchSettings(outcome="Y", candidates=["X01", "X02", "X03"], participant="StudentID", time="Time",
                       required=["Time"], strategy="focal", max_predictors=3, study="longitudinal", family="gaussian")
    run = engine.create_run(d, s)
    run.rows = [{"idx": c.index, "key": candidate_key(c), "status": "ok"} for c in enumerate_candidates(s)]
    cfg = V.ValidationConfig(target="participant", modes=["holdout"])
    assert V.ValidationRunner(None, d, run, cfg).provenance["verified_saved_keys"] == len(run.rows)
    run.rows[1]["key"] = "focal|X01:X03|other"
    with pytest.raises(V.ValidationError, match="not the candidates"):
        V.ValidationRunner(None, d, run, cfg)


def test_fixed_candidate_retrieval_never_scans_the_space(monkeypatch):
    """7,300-candidate and 147,713,444-candidate spaces: a fixed candidate (first, middle, late, missing) is resolved
    with ZERO enumeration calls and in constant time (v1.3.4 visited the whole suffix)."""
    import time as _t

    from statexplore.discovery import space
    from statexplore.discovery import validate as V
    from statexplore.discovery.engine import DiscoveryRun
    from statexplore.discovery.space import SearchSettings, candidate_key, count_candidates, enumerate_candidates

    s = SearchSettings(outcome="Y", candidates=[f"X{i:02d}" for i in range(1, 31)], required=["Time"],
                       participant="StudentID", time="Time", strategy="combinations", include_main=True,
                       max_predictors=6)
    total = count_candidates(s)
    picks = [0, total // 2, total - 1]
    rows = []
    for i in picks:
        c = next(enumerate_candidates(s, start=i))
        rows.append({"idx": i, "key": candidate_key(c), "focal": list(c.focal), "covariates": list(c.covariates),
                     "kind": c.kind, "inter": [list(t) for t in c.inter], "status": "ok"})
    run = DiscoveryRun(settings=s, family="lmm", total=total, signature="x")
    run.rows = rows
    calls = {"n": 0}
    real = space.enumerate_candidates

    def counted(*a, **k):
        calls["n"] += 1
        return real(*a, **k)
    monkeypatch.setattr(space, "enumerate_candidates", counted)
    for i in picks:
        t0 = _t.perf_counter()
        c = V.fixed_candidate(run, i)
        assert _t.perf_counter() - t0 < 0.05 and candidate_key(c) == next(r["key"] for r in rows if r["idx"] == i)
    with pytest.raises(V.ValidationError):
        V.fixed_candidate(run, 12345)
    t0 = _t.perf_counter()
    info = V.feasibility(fx("Test53A_30Predictors"), run, V.ValidationConfig(target="participant", modes=["holdout"],
                                                                              pipeline="fixed", candidate=picks[1]))
    assert _t.perf_counter() - t0 < 10 and info["pool"] == 1 and not [p for p in info["problems"] if "Candidate" in p]
    assert calls["n"] == 0


# ---- AUD134-06: declared validation scope ------------------------------------------------------------------------------

def _null13_run(executions, mode="D"):
    from statexplore.discovery import engine
    from statexplore.discovery.space import SearchSettings

    d = import_file(FX4 / "StatExplore_Test51F_Null_13.csv")
    p, old = _52d()
    s = SearchSettings.from_dict(old.settings.to_dict())
    s.mode = mode
    run = engine.create_run(d, s)
    run.executions = executions
    run.status = "complete"
    return d, run


def test_scope_original_cumulative_and_prefix():
    from statexplore.discovery import validate as V

    d, run = _null13_run([{"start": 0, "max_models": 5, "max_minutes": 30, "end": 5},
                          {"start": 5, "max_models": 10, "max_minutes": 30, "end": 15}])
    cfg = lambda **k: V.ValidationConfig(target="participant", modes=["holdout"], **k)  # noqa: E731
    assert V.pool_size(run, cfg(pool="original")) == 5
    assert V.pool_size(run, cfg(pool="declared")) == 5          # the v1.3.4 name keeps its meaning
    assert V.pool_size(run, cfg()) == 15 and V.ValidationConfig().pool == "cumulative"
    assert V.pool_size(run, cfg(pool="prefix", prefix_n=7)) == 7
    sc = V.pool_scope(run, cfg())
    assert sc["adaptive"] and "ADAPTIVE" in V.pool_label(run, cfg())
    info = V.feasibility(d, run, cfg())
    assert info["pool"] == 15 and info["adaptive"] and any("conditional validation" in w for w in info["warnings"])
    r = V.ValidationRunner(None, d, run, cfg())
    assert [c.index for c in r.cands] == list(range(15))
    # outcome-independent: rows, statuses and significance never change the pool
    for rows in ([], [{"idx": i, "status": "invalid", "key": ""} for i in range(15)]):
        run.rows = rows
        assert V.pool_size(run, cfg()) == 15 and V.pool_size(run, cfg(pool="original")) == 5


def test_scope_edge_cases():
    from statexplore.discovery import validate as V

    cfg = V.ValidationConfig(target="participant", modes=["holdout"])
    d, run = _null13_run([{"start": 0, "max_models": 5, "end": 5}, {"start": 5, "max_models": 20, "end": 15}])
    assert V.pool_size(run, cfg) == 15 and "capped" in V.pool_label(run, cfg)
    d, run = _null13_run([{"start": 0, "max_models": 5, "end": 5}, {"start": 5, "max_models": 10, "end": 9}])
    assert V.pool_size(run, cfg) == 15 and "cannot be replayed" in V.pool_label(run, cfg)
    d, run = _null13_run([{"start": 0, "max_models": None, "max_minutes": 1.0, "end": 4}])
    assert V.pool_size(run, cfg) == 0 and any("explicit number" in p for p in V.feasibility(d, run, cfg)["problems"])
    assert V.pool_size(run, V.ValidationConfig(pool="prefix", prefix_n=4)) == 4
    d, run = _null13_run([{"start": 0, "max_models": 5, "end": 5}, {"start": 5, "max_models": 10, "end": 15}],
                         mode="C")
    assert V.pool_size(run, cfg) == V.pool_size(run, V.ValidationConfig(pool="original")) == 15


# ---- AUD134-07: historical audit and reproduction identity ---------------------------------------------------------------

def _p50():
    from statexplore.core.project import Project

    p = Project.open(FX4 / "StatExplore_Test50_Reproducibility.sxp")
    return p, p.searches[0]


def _swap_to_50e(ds):
    d2 = import_file(FX4 / "StatExplore_Test50E_ModifiedValues.csv")
    assert len(d2.raw) == len(ds.raw) == 960
    ds.raw = d2.raw
    ds.source.sha256, ds.source.path = d2.source.sha256, d2.source.path


def test_audit_historical_identity_survives_data_replacement(tmp_path):
    from statexplore.export.audit import SCHEMA_VERSION, read_audit, write_audit

    p, run = _p50()
    orig_sha = run.data_snapshot["source_sha256"]
    a = read_audit(write_audit(run, p.dataset, tmp_path / "a")[0])
    assert SCHEMA_VERSION == 2 and a["schema_version"] == 2
    h = a["data"]["historical"]
    assert h["source_sha256"] == orig_sha and h["analysis_signature"] == run.signature
    assert a["data"]["current_context"]["matches_analysis"] is True
    _swap_to_50e(p.dataset)
    b = read_audit(write_audit(run, p.dataset, tmp_path / "b")[0])
    assert b["data"]["historical"] == {**h}                       # unchanged, never the current file
    cur = b["data"]["current_context"]
    assert cur["matches_analysis"] is False and cur["source_sha256"] != orig_sha and "NOT the analysed data" in cur["label"]
    assert [c["key"] for c in b["candidates"]] == [c["key"] for c in a["candidates"]]
    report = (tmp_path / "b" / f"search_{run.id}_audit_report.md").read_text(encoding="utf-8")
    assert "DIFFERS from the analysed data" in report


def test_audit_missing_historical_metadata_is_unknown_and_tails_round_trip(tmp_path):
    from statexplore.export.audit import read_audit, write_audit

    p, run = _52d()
    rec = read_audit(write_audit(run, p.dataset, tmp_path)[0])
    h = rec["data"]["historical"]
    assert h["source_file"].startswith("unknown") and h["snapshot_schema"] == 1      # schema-1 snapshot: no file name
    assert rec["data"]["current_context"]["source_file"] == "StatExplore_Test51F_Null_13.csv"
    assert "fit rules 3" in rec["search"]["resume"] and rec["grammar"]["stored_total"] == 15
    rows = _rows([-434.0, -2.0])
    fake = _fake_run(rows)
    fake.id, fake.fit_id = "fake", "f"
    rec = read_audit(write_audit(fake, None, tmp_path / "x")[0])
    hy = {h["idx"]: h for h in rec["hypotheses"]}
    assert hy[0]["p"] == 0.0 and hy[0]["p_underflow"] is True and hy[0]["log10_p"] == pytest.approx(-434.0)
    assert hy[1]["p_underflow"] is False and "current_context" not in rec["data"]


def _validation_record(run) -> dict:
    return {"grammar": "validation-3", "schema": 2, "validation_id": "v1", "config": {"modes": ["holdout"], "pipeline":
            "fixed", "holdout_fraction": 0.2, "k": 5, "repeats": 1, "B": 1, "seed": 1}, "status": "complete",
            "problems": [], "search_id": run.id, "search_identity": run.fit_id, "search_signature": run.signature,
            "target_label": "New participants", "target_estimand": "x", "pipeline_label": "Fixed", "rule_label": "",
            "predictions": [{"split": "holdout", "fold": 1, "row_id": 1, "observed": 1.0, "predicted": 1.0}],
            "scripts": ["# chunk"], "reproduce_script": "dat <- utils::read.csv(\"x.csv\")\n# body\n"}


def test_validation_export_refuses_runnable_bundle_over_other_data(tmp_path):
    from statexplore.discovery import validate as V

    p, run = _p50()
    v = _validation_record(run)
    ok = tmp_path / "ok"
    V.export(v, str(ok), ds=p.dataset, include_data=True, run=run)
    man = json.loads((ok / "validation_manifest.json").read_text())
    assert man["reproduction"]["runnable"] is True and (ok / "validation_reproduce.R").exists()
    assert (ok / "validation_reproduce_data.csv").exists()
    _swap_to_50e(p.dataset)
    bad = tmp_path / "bad"
    V.export(v, str(bad), ds=p.dataset, include_data=True, run=run)
    man = json.loads((bad / "validation_manifest.json").read_text())
    assert man["reproduction"]["runnable"] is False and (bad / "REPRODUCTION_UNAVAILABLE.txt").exists()
    assert not (bad / "validation_reproduce.R").exists() and not list(bad.glob("*_data.csv"))
    assert (bad / "predictions.csv").exists()            # historical results stay exportable
    # a v1.3.4 caller handing a wrong VIEW is refused as well (signature check without the run)
    from statexplore.discovery.space import view

    old = tmp_path / "old"
    V.export(v, str(old), ds=view(p.dataset, run.settings), include_data=True)
    assert (old / "REPRODUCTION_UNAVAILABLE.txt").exists()


def test_validation_export_filter_change_uses_rebuilt_original_view(tmp_path):
    from statexplore.core.derive import FilterSpec
    from statexplore.discovery import validate as V

    p, run = _p50()
    p.dataset.filters = [FilterSpec(variable="TIME", op="<=", value=1e9)]
    out = tmp_path / "f"
    V.export(_validation_record(run), str(out), ds=p.dataset, include_data=True, run=run)
    man = json.loads((out / "validation_manifest.json").read_text())
    assert man["reproduction"]["runnable"] is True            # rebuilt from the snapshot on byte-identical source data


# ---- AUD134-09: cancellation, deadline and cached batches ------------------------------------------------------------------

class _FakeWorker:
    def __init__(self):
        self.killed = 0

    def kill(self):
        self.killed += 1


class _FakePooled:
    """Stands in for an R worker: returns a valid 'ok' result per candidate (no R)."""

    def __init__(self, delay: float = 0.0, gate: threading.Event | None = None):
        self.session = type("S", (), {"worker": _FakeWorker()})()
        self.delay, self.gate, self.batches = delay, gate, 0

    def run_batch(self, ds, script, n):
        self.batches += 1
        if self.gate is not None:
            self.gate.wait(5)
        return [{"i": i, "status": "ok", "flags": "", "p": 0.5, "log_p": math.log(0.5), "tests": []}
                for i in json.loads(script)], {}


class _Clock:
    def __init__(self, step: float = 0.0):
        self.t, self.step = 0.0, step

    def __call__(self):
        self.t += self.step
        return self.t


def _dispatch_run(total_cands=10, **kw):
    from statexplore.discovery import engine
    from statexplore.discovery.space import SearchSettings

    d = fx("Test53A_30Predictors")
    s = SearchSettings(outcome="Y", candidates=[f"X{i:02d}" for i in range(1, total_cands + 1)], strategy="main",
                       max_predictors=1, batch_size=2, max_models=None, max_minutes=None, family="gaussian", **kw)
    run = engine.create_run(d, s)
    run.status = "running"
    return d, run


def _cache_for(run):
    from statexplore.discovery.space import candidate_key, enumerate_candidates

    return {candidate_key(c): {"status": "ok", "key": candidate_key(c), "p": 0.5, "log_p": math.log(0.5), "idx": c.index,
                               "_from": "old", "_engine": ()} for c in enumerate_candidates(run.settings)}


@pytest.fixture
def fake_batches(monkeypatch):
    from statexplore.discovery import engine

    monkeypatch.setattr(engine, "batch_script", lambda ds, batch, s=None: json.dumps([c.index for c in batch]))


def _go(run, d, pooled, cancel, cache, clock=None, progress=None):
    from statexplore.discovery import engine

    clock = clock or _Clock()
    with ThreadPoolExecutor(max_workers=2) as ex:
        engine._dispatch(run, d, pooled, ex, cancel, progress, clock(), 0.0, cache, clock=clock)
    return run


def test_precancelled_all_cached_commits_nothing(fake_batches):
    d, run = _dispatch_run()
    cancel = threading.Event()
    cancel.set()
    _go(run, d, [_FakePooled()], cancel, _cache_for(run))
    assert run.rows == [] and run.status == "cancelled" and run.n_done == 0


def test_cancel_mid_cache_stops_at_exact_cursor(fake_batches):
    d, run = _dispatch_run()
    cancel = threading.Event()

    def prog(r):
        if len(r.rows) >= 4:
            cancel.set()
    _go(run, d, [_FakePooled()], cancel, _cache_for(run), clock=_Clock(step=1.0), progress=prog)
    idx = [r["idx"] for r in run.rows]
    assert run.status == "cancelled" and len(idx) == len(set(idx)) and idx == list(range(len(idx)))
    assert run.n_done == len(idx) < 10


def test_deadline_before_dispatch_with_fake_clock(fake_batches):
    d, run = _dispatch_run()
    run.settings.max_minutes = 1.0
    clock = _Clock()
    pooled = _FakePooled()

    def prog(r):
        clock.t += 61.0          # one minute passes after the first batch
    _go(run, d, [pooled], threading.Event(), {}, clock=clock, progress=prog)
    assert run.status == "partial" and run.stop_reason == "time budget reached" and len(run.rows) == 2
    assert pooled.batches == 1 and run.n_done == 2


def test_mixed_cached_and_fresh_budget_counts_attempts(fake_batches):
    d, run = _dispatch_run()
    run.settings.max_models = 5
    cache = {k: v for i, (k, v) in enumerate(_cache_for(run).items()) if i % 2 == 0}
    _go(run, d, [_FakePooled()], threading.Event(), cache)
    assert run.status == "partial" and run.stop_reason == "model budget reached" and len(run.rows) == 5
    assert sorted(r["idx"] for r in run.rows) == list(range(5)) and run.n_done == 5
    assert sum(r["cached"] for r in run.rows) == 3


def test_cancel_with_inflight_batch_kills_and_records_nothing(fake_batches):
    d, run = _dispatch_run()
    gate = threading.Event()
    pooled = _FakePooled(gate=gate)
    cancel = threading.Event()
    t = threading.Timer(0.3, cancel.set)
    t.start()
    try:
        _go(run, d, [pooled], cancel, {})
    finally:
        gate.set()
    assert run.status == "cancelled" and pooled.session.worker.killed == 1 and run.rows == [] and run.n_done == 0


def test_cancel_after_completion_keeps_complete(fake_batches):
    d, run = _dispatch_run()
    cancel = threading.Event()

    def prog(r):
        if len(r.rows) == 10:
            cancel.set()
    _go(run, d, [_FakePooled()], cancel, {}, progress=prog)
    assert run.status == "complete" and len(run.rows) == 10


def test_53c_budget_and_cancel_on_huge_declared_space(fake_batches):
    """147,713,444 declared candidates: a 100-model budget and a cancel at ~50,000 cached rows (never the whole space)."""
    from statexplore.discovery import engine
    from statexplore.discovery.space import SearchSettings, candidate_key, enumerate_candidates

    d = fx("Test53A_30Predictors")
    s = SearchSettings(outcome="Y", candidates=[f"X{i:02d}" for i in range(1, 31)], required=["Time"],
                       participant="StudentID", time="Time", strategy="combinations", include_main=True,
                       max_predictors=6, batch_size=500, max_models=100, max_minutes=None, family="gaussian",
                       study="longitudinal")
    run = engine.DiscoveryRun(settings=s, family="lmm", total=147_713_444, signature="x", fit_id="f")
    _go(run, d, [_FakePooled()], threading.Event(), {})
    assert len(run.rows) == 100 and run.status == "partial" and run.stop_reason == "model budget reached"
    run2 = engine.DiscoveryRun(settings=s, family="lmm", total=147_713_444, signature="x", fit_id="f")
    s.max_models = None
    from itertools import islice

    cache = {candidate_key(c): {"status": "ok", "key": candidate_key(c), "_engine": (), "_from": "old"}
             for c in islice(enumerate_candidates(s), 60_000)}
    cancel = threading.Event()

    def prog(r):
        if len(r.rows) >= 50_000:
            cancel.set()
    _go(run2, d, [_FakePooled()], cancel, cache, clock=_Clock(step=1.0), progress=prog)
    idx = [r["idx"] for r in run2.rows]
    assert run2.status == "cancelled" and 50_000 <= len(idx) < 60_000 and len(idx) == len(set(idx))
    assert run2.n_done == len(idx)


# ---- AUD134-10: conservative workload bound and hard budget --------------------------------------------------------------

def _heterogeneous_7300():
    """A heterogeneous plan of ~7,300 candidates (combinations strategy: main-effects, single- and multi-interaction
    models with very different numbers of tests), early candidates simple - the case where averaging the first
    candidates underestimates."""
    from statexplore.discovery.space import SearchSettings, count_candidates

    s = SearchSettings(outcome="y", candidates=[f"X{i:02d}" for i in range(1, 9)], strategy="combinations",
                       include_main=True, max_predictors=4, max_order=3)
    return s, count_candidates(s)


def test_workload_bound_covers_uncached_plan():
    from statexplore.discovery import validate as V
    from statexplore.discovery.engine import DiscoveryRun
    from statexplore.discovery.space import candidate_tests, enumerate_candidates

    s, total = _heterogeneous_7300()
    assert 5_000 < total < 10_000
    run = DiscoveryRun(settings=s, family="gaussian", total=total, signature="x")
    cfg = V.ValidationConfig(target="rows", modes=["holdout", "cv"], k=5, max_fits=10**9)
    exact = sum(1 + len(candidate_tests(c)) for c in enumerate_candidates(s))     # brute force, written here
    wl = V.workload(run, cfg)
    samples = 1 + 5
    assert wl["bound"] == samples * (exact + 1) and wl["bound"] >= samples * exact
    assert wl["estimate"] < wl["bound"]                    # the old average is optimistic - never used as the cap check
    # a pool too large to enumerate gets the analytic maximum, which still bounds the exact plan
    import statexplore.discovery.validate as VV

    old = VV._EXACT_LIMIT
    VV._EXACT_LIMIT = 10
    try:
        VV._exact_cache.clear()
        assert V.workload(run, cfg)["bound"] >= samples * (exact + 1)
    finally:
        VV._EXACT_LIMIT = old
    cfg.max_fits = samples * (exact + 1)
    assert not [p for p in V.feasibility(fx("Test55L_LegacyUnderflow"), run, cfg)["problems"] if "budget" in p]
    cfg.max_fits -= 1
    assert any("exceed the budget" in p for p in V.feasibility(fx("Test55L_LegacyUnderflow"), run, cfg)["problems"])
    cfg.max_fits = 0
    assert any("at least 1" in p for p in V.feasibility(fx("Test55L_LegacyUnderflow"), run, cfg)["problems"])


def test_retry_factor_for_nlme_structures():
    from statexplore.discovery import validate as V
    from statexplore.discovery.engine import DiscoveryRun
    from statexplore.discovery.space import SearchSettings

    s = SearchSettings(outcome="Y", candidates=["A"], participant="ID", time="T", structure="ar1", strategy="main")
    run = DiscoveryRun(settings=s, family="lmm", total=1, signature="x")
    assert V.workload(run, V.ValidationConfig(modes=["holdout"]))["retry_factor"] == 2
    s.structure = "ri"
    assert V.workload(run, V.ValidationConfig(modes=["holdout"]))["retry_factor"] == 1


@pytest.mark.r
@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_hard_fit_budget_counts_and_stops(r_session):
    """55L (Gaussian, 3 focal candidates, rows target): with an ample budget the counted invocations plus cache hits equal
    the uncached plan exactly (full + reduced per candidate, + the selected refit) - an independent identity check of
    the instrumentation; with a cap of 10 the validation stops, never exceeds it and never scores a budget-stopped part."""
    from statexplore.discovery import validate as V

    d = fx("Test55L_LegacyUnderflow")
    run = _legacy_run(d)
    cfg = V.ValidationConfig(target="rows", modes=["holdout", "cv"], k=2, seed=3, max_fits=10_000)
    v = V.ValidationRunner(r_session, d, run, cfg).execute()
    b = v["fit_budget"]
    assert v["status"] == "complete" and b["used"] + b["cache_hits"] == b["requested_bound"] == 3 * (3 * 2 + 1)
    assert all(f["selected_index"] == 1 for f in v["folds"])          # log-tail selection inside AMD too
    # the conservative bound exceeds a cap of 10: refused up front ...
    small = V.ValidationConfig(**{**cfg.to_dict(), "max_fits": 10})
    assert any("exceed the budget" in p for p in V.feasibility(d, run, small)["problems"])
    with pytest.raises(V.ValidationError, match="exceed the budget"):
        V.ValidationRunner(r_session, d, run, small)
    # ... and the hard cap still holds when a run is forced past the check (the bound is not trusted blindly)
    v2 = V.ValidationRunner(r_session, d, run, cfg)
    v2.budget["cap"] = 10
    out = v2.execute()
    assert out["status"] == "partial" and out["fit_budget"]["used"] <= 10
    assert any("fit budget reached" in p for p in out["problems"])
    assert all(f["status"] in ("scored", "budget") for f in out["folds"])
    assert not any(f["status"] == "scored" and "budget" in str(f.get("reason")) for f in out["folds"])


# ---- AUD134-11: complete standalone reproduction checks --------------------------------------------------------------------

def _repro_export(r_session, tmp_path: Path) -> tuple[Path, dict]:
    from statexplore.discovery import validate as V

    d = fx("Test55L_LegacyUnderflow")
    run = _legacy_run(d)
    cfg = V.ValidationConfig(target="rows", modes=["holdout", "cv", "bootstrap"], k=2, B=3, seed=7, max_fits=100_000)
    v = V.ValidationRunner(r_session, d, run, cfg).execute()
    assert v["status"] == "complete" and v["coefs"] and v["trace"]
    out = tmp_path / "export"
    V.export(v, str(out), ds=d, include_data=True, run=run)
    moved = tmp_path / "moved folder ü é"
    shutil.copytree(out, moved)
    return moved, v


@pytest.mark.r
@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_reproduction_verifies_everything_and_catches_tampering(r_session, tmp_path):
    moved, v = _repro_export(r_session, tmp_path)
    res = run_r(moved, "validation_reproduce.R")
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    assert "All" in res.stdout and "The complete validation was reproduced" in res.stdout
    chk = pd.read_csv(moved / "validation_check_results.csv")
    assert (chk["status"] != "FAIL").all() and len(chk) > 40
    for name in ("predictions: predicted", "folds: selected_key", "bootstrap coefficients: estimate",
                 "candidate trace: log_p", "selection frequencies (recomputed): selected",
                 "performance (recomputed in R from the reproduced predictions): RMSE", "split manifest: fold",
                 "bootstrap draw manifest: same rows"):
        assert (chk.loc[chk["check"] == name, "status"] == "PASS").all() and (chk["check"] == name).any(), name

    def tamper(label, fn):
        t = tmp_path / f"t_{label}"
        shutil.copytree(moved, t)
        fn(t)
        r = run_r(t, "validation_reproduce.R")
        assert r.returncode != 0, label
        return r

    def edit(fname, f):
        def go(t):
            df = pd.read_csv(t / fname)
            f(df)
            df.to_csv(t / fname, index=False)
        return go
    tamper("pred", edit("predictions.csv", lambda df: df.__setitem__("predicted", df["predicted"] + np.r_[0.01, np.zeros(len(df) - 1)])))
    tamper("coef", edit("bootstrap_coefficients.csv", lambda df: df.__setitem__("estimate", df["estimate"] * 1.001)))
    tamper("sel", edit("bootstrap_results.csv", lambda df: df.__setitem__("selected_index", (df["selected_index"] + 1) % 3)))
    tamper("status", edit("bootstrap_results.csv", lambda df: df.__setitem__("status", ["failed"] + list(df["status"][1:]))))
    tamper("dup", lambda t: (t / "folds.csv").write_text((t / "folds.csv").read_text() + (t / "folds.csv").read_text()
                                                          .splitlines()[1] + "\n"))
    tamper("extra_draw", edit("bootstrap_results.csv", lambda df: df.loc.__setitem__(
        len(df), pd.Series(df.iloc[0].to_dict() | {"b": 99}))))
    tamper("trace", edit("candidate_trace.csv", lambda df: df.__setitem__("log_p", df["log_p"] - 1)))
    tamper("metric", edit("performance.csv", lambda df: df.__setitem__("RMSE", df["RMSE"] + 1e-3)))
    tamper("missing", lambda t: (t / "predictions.csv").unlink())
    r = tamper("data", lambda t: (t / "validation_reproduce_data.csv").write_text(
        (t / "validation_reproduce_data.csv").read_text().replace(",1,", ",2,", 1)))
    assert "SHA-256" in r.stderr


@pytest.mark.r
def test_reproduction_na_safe_failed_draw_comparison(tmp_path):
    """Matched missing selections of failed draws are equal only when the statuses agree; no NA comparison crash."""
    from statexplore.discovery.validate import R_REPRO_CHECKS

    lib = R_REPRO_CHECKS.split("sx_tol <- c(1e-6, 1e-6)")[0]
    code = "sx_long <- function() FALSE\n" + lib + r"""
app <- data.frame(b = 1:3, status = c("fitted", "failed", "fitted"), selected_index = c(1, NA, 2))
rep <- app
sx_match("draws", app, rep, "b", exact = c("status", "selected_index"))
stopifnot(all(sx_chk$status == "PASS"))
rep$status[2] <- "fitted"; rep$selected_index[2] <- 1
sx_chk <- sx_chk[0, ]
sx_match("draws", app, rep, "b", exact = c("status", "selected_index"))
stopifnot(sum(sx_chk$status == "FAIL") == 2)
sx_chk <- sx_chk[0, ]
sx_match("draws", app, rbind(rep, rep[1, ]), "b")
stopifnot(any(sx_chk$status == "FAIL"))
cat("OK\n")
"""
    res = rscript(code, tmp_path)
    assert res.returncode == 0 and "OK" in res.stdout, res.stderr[-2000:]


@pytest.mark.r
@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_51c_pipeline_bootstrap_reproduces_in_fresh_r(r_session, tmp_path):
    """51C (18 candidates, three-way grammar, clustered random slopes): pipeline holdout + 2 bootstrap draws exported and
    reproduced in a fresh Rscript in a relocated folder with spaces and Unicode; the log-tail selection keeps #17."""
    from statexplore.discovery import engine
    from statexplore.discovery import validate as V
    from statexplore.discovery.space import SearchSettings

    d = import_file(FX4 / "StatExplore_Test51C_ThreeWayInteraction.csv")
    s = SearchSettings(outcome="TestScore", candidates=["Sleep", "Anxiety", "Attendance"], required=["Time", "Gender"],
                       participant="StudentID", time="Time", cluster="School", structure="rs", strategy="combinations",
                       include_main=True, max_order=3, max_predictors=5, family="gaussian", workers=2,
                       max_models=None, max_minutes=None, study="longitudinal")
    run = engine.create_run(d, s)
    engine.execute(run, d)
    cfg = V.ValidationConfig(target="cluster", modes=["holdout", "bootstrap"], B=2, seed=51, max_fits=10**6)
    v = V.ValidationRunner(r_session, d, run, cfg).execute()
    assert v["status"] == "complete"
    assert all(b["selected_index"] == 17 for b in v["bootstrap"]) and v["folds"][0]["selected_index"] == 17
    out = tmp_path / "e"
    V.export(v, str(out), ds=d, include_data=True, run=run)
    moved = tmp_path / "relocated ä folder"
    shutil.copytree(out, moved)
    res = run_r(moved, "validation_reproduce.R", timeout=3000)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    assert "The complete validation was reproduced" in res.stdout
