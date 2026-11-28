"""StatExplore v1.3.6 corrective patch: tests per audit finding AUD135-01 ... AUD135-07.

Oracles are independent of the code under test: values computed here with numpy / pandas from the imported frames,
direct R calls (stats::t.test, quantile type 7, lm), hand-built records, controlled fake R sessions for the execution
paths, and fresh Rscript runs of the generated reproduction scripts. Run on v1.3.5 (b7610a0) the finding tests fail
(red evidence in docs/v1.3.6/evidence), on v1.3.6 they pass.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from statexplore.core.importers import import_file

FX = Path(__file__).parent / "fixtures" / "v136"
FX5 = Path(__file__).parent / "fixtures" / "v135"
FX4 = Path(__file__).parent / "fixtures" / "v134"
WORKBOOK = FX / "AUD135_06_two_sheets.xlsx"
WORKBOOK_SHA = "b22ed8f4b06dc3964b96413ca2bfa88863ed168db2bcd24747b5c204731ce02e"


def rscript(code: str, cwd: Path, timeout: int = 1800, script: str = "oracle.R") -> subprocess.CompletedProcess:
    from statexplore.rbridge.locate import find_rscript

    lib = Path(__file__).resolve().parents[1] / "r" / "library"
    env = dict(os.environ, R_LIBS=str(lib), R_LIBS_USER=str(lib), OPENBLAS_NUM_THREADS="1")
    (cwd / script).write_text(code, encoding="utf-8")
    return subprocess.run([str(find_rscript()), "--vanilla", str(cwd / script)], cwd=cwd, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=timeout, env=env)


def run_r(cwd: Path, script: str, timeout: int = 1800) -> subprocess.CompletedProcess:
    from statexplore.rbridge.locate import find_rscript

    lib = Path(__file__).resolve().parents[1] / "r" / "library"
    env = dict(os.environ, R_LIBS=str(lib), R_LIBS_USER=str(lib), OPENBLAS_NUM_THREADS="1")
    return subprocess.run([str(find_rscript()), "--vanilla", script], cwd=cwd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout, env=env)


def fx5(name: str):
    return import_file(FX5 / f"StatExplore_{name}.csv")


def one(paths, name: str) -> Path:
    hits = [Path(p) for p in paths if Path(p).name == name]
    assert len(hits) == 1, (name, [Path(p).name for p in paths])
    return hits[0]


def folder_of(paths) -> Path:
    """The new export folder (the inventory sits at its root)."""
    return one(paths, "bundle_inventory.json").parent


S55L = dict(outcome="y", candidates=["x1", "x3", "x2"], family="gaussian", strategy="focal", covariate_mode="focal",
            max_predictors=2, workers=1, max_models=None, max_minutes=None)
SWB = dict(outcome="y", candidates=["x", "z", "w"], family="gaussian", strategy="focal", max_predictors=2)


def test_fixture_bytes():
    assert hashlib.sha256(WORKBOOK.read_bytes()).hexdigest() == WORKBOOK_SHA


# ---- AUD135-06: imported-content identity -------------------------------------------------------------------------------

def _sheets():
    return import_file(WORKBOOK, sheet="A"), import_file(WORKBOOK, sheet="A"), import_file(WORKBOOK, sheet="B")


def test_workbook_sheets_have_distinct_identities_everywhere():
    """Port of the auditor's Round 3 workbook reproducer (baseline 9 pass / 7 fail): every consumer of the data identity
    tells the two worksheets apart, and an unchanged same-sheet re-import still matches (positive controls)."""
    from statexplore.discovery.engine import FIT_RULES, cache_from, create_run
    from statexplore.discovery.history import context
    from statexplore.discovery.space import SearchSettings, candidate_key, enumerate_candidates
    from statexplore.discovery.validate import bundle_data
    from statexplore.export.audit import current_context

    a, a2, b = _sheets()
    # independent descriptive witnesses (values from the imported frames)
    assert np.corrcoef(a.raw.y, a.raw.x * a.raw.z)[0, 1] == pytest.approx(0.9991717923620217, abs=1e-12)
    assert np.corrcoef(b.raw.y, b.raw.x * b.raw.z)[0, 1] == pytest.approx(0.07341307696951486, abs=1e-12)
    assert a.source.sha256 == b.source.sha256 == WORKBOOK_SHA and not a.raw.equals(b.raw)
    assert [v.to_dict() for v in a.variables.values()] == [v.to_dict() for v in b.variables.values()]
    s = SearchSettings(**SWB)
    r1, r1b, r2 = create_run(a, s), create_run(a2, s), create_run(b, s)
    c = next(enumerate_candidates(s))
    k = candidate_key(c)
    r1.rows = [{"idx": 0, "key": k, "status": "ok", "p": 0.001}]
    r1.n_done, r1.status = 1, "partial"
    r1.engine = {"fit_rules": FIT_RULES, "r_version": "synthetic-control", "packages": {}}
    # positive controls: same sheet re-imported
    assert r1.signature == r1b.signature and r1.fit_id == r1b.fit_id
    assert context(a2, r1).state == "current"
    assert k in cache_from(r1b, [r1], a2)
    assert r1.resume_reason(a2) == ""
    assert bundle_data({"search_signature": r1.signature}, a2, r1)[0] is not None
    # the other worksheet
    assert r1.signature != r2.signature and r1.fit_id != r2.fit_id
    ctx = context(b, r1)
    assert ctx.state == "unrecoverable" and any("different content" in d for d in ctx.differences)
    assert k not in cache_from(r2, [r1], b)
    assert r1.resume_reason(b)
    assert current_context(r1, b)["matches_analysis"] is False
    view, why = bundle_data({"search_signature": r1.signature}, b, r1)
    assert view is None and why
    assert r1.data_snapshot["content_id"] and r1.data_snapshot["source_options"]["sheet"] == "A"


def test_content_identity_definition():
    """Same values -> same identity; any value, row order, column order, text-vs-number, leading zero, missing-vs-empty
    or parse option change -> another identity; metadata changes the VIEW identity only."""
    from statexplore.core.dataset import Dataset, HoldoutSpec, SourceInfo
    from statexplore.core.derive import FilterSpec
    from statexplore.core.identity import content_id, view_id
    from statexplore.core.variables import Measure, Storage, VariableMeta

    def ds(frame, opts=None):
        metas = [VariableMeta(name=c, alias=f"v{i + 1:03d}",
                              storage=Storage.STRING if frame[c].dtype != "float64" else Storage.NUMERIC,
                              measure=Measure.NOMINAL if frame[c].dtype != "float64" else Measure.CONTINUOUS)
                 for i, c in enumerate(frame.columns)]
        return Dataset(frame, metas, SourceInfo(path="x.csv", format="csv", options=opts or {"delimiter": ","},
                                                sha256="f" * 64))
    base = pd.DataFrame({"y": [1.0, 2.0, 3.0, np.nan], "id": pd.array(["001", "002", "", None], dtype="string")})
    d0 = ds(base)
    same = ds(base.copy())
    assert content_id(d0) == content_id(same) and view_id(d0) == view_id(same)
    variants = {
        "value": base.assign(y=[1.0, 2.0, 3.5, np.nan]),
        "row order": base.iloc[[1, 0, 2, 3]].reset_index(drop=True),
        "column order": base[["id", "y"]],
        "leading zero": base.assign(id=pd.array(["1", "002", "", None], dtype="string")),
        "missing vs empty": base.assign(id=pd.array(["001", "002", None, None], dtype="string")),
        "signed zero": base.assign(y=[1.0, 2.0, 3.0, -0.0]),
    }
    ids = {k: content_id(ds(v)) for k, v in variants.items()}
    assert len(set(ids.values()) | {content_id(d0)}) == len(ids) + 1, ids
    assert content_id(ds(base, {"delimiter": ";"})) != content_id(d0)          # parse context
    d1 = ds(base.copy())
    d1.set_variable_field("y", "measure", "nominal")
    assert content_id(d1) == content_id(d0) and view_id(d1) != view_id(d0)
    d2 = ds(base.copy())
    d2.add_filter(FilterSpec(variable="y", op=">", value=1.0))
    assert view_id(d2) != view_id(d0)
    d3, d4 = ds(base.copy()), ds(base.copy())
    d3.set_holdout(HoldoutSpec(0.25, 1))
    d4.set_holdout(HoldoutSpec(0.25, 2))
    assert len({view_id(d0), view_id(d3), view_id(d4)}) == 3


def test_identity_survives_project_save_reopen_and_reimport(tmp_path):
    """Unchanged same-sheet re-import and project (Parquet) round trips keep the verified identity: the search stays
    current, resumable state and cache admission are preserved, and the rebuilt snapshot reproduces the signature."""
    from statexplore.core.derive import FilterSpec
    from statexplore.core.project import Project
    from statexplore.discovery.engine import create_run, data_signature
    from statexplore.discovery.history import context
    from statexplore.discovery.space import SearchSettings, view

    a, _a2, _b = _sheets()
    run = create_run(a, SearchSettings(**SWB))
    p = Project(a)
    p.searches.append(run)
    p.save(tmp_path / "p.sxp")
    q = Project.open(tmp_path / "p.sxp")
    r = q.searches[0]
    assert r.identity_schema == 3 and r.data_snapshot["content_id"] == run.data_snapshot["content_id"]
    assert data_signature(view(q.dataset, r.settings)) == r.signature
    assert context(q.dataset, r).state == "current"
    q.dataset.add_filter(FilterSpec(variable="w", op="<=", value=1e9))
    ctx = context(q.dataset, r)
    assert ctx.state == "changed" and ctx.original is not None          # rebuilt from the snapshot on the same content
    assert data_signature(view(ctx.original, r.settings)) == r.signature


def test_value_digest_is_computed_once_per_revision(monkeypatch):
    """Performance contract: identities of many views / candidates hash the values once per imported revision."""
    from statexplore.core import identity
    from statexplore.discovery.engine import data_signature, fit_identity
    from statexplore.discovery.space import SearchSettings, view

    d = fx5("Test55L_LegacyUnderflow")
    calls = {"n": 0}
    orig = identity._column_bytes

    def count(col, h):
        calls["n"] += 1
        return orig(col, h)
    monkeypatch.setattr(identity, "_column_bytes", count)
    s = SearchSettings(**S55L)
    for _ in range(50):
        data_signature(view(d, s))
        fit_identity(d, s, "gaussian")
    assert calls["n"] <= len(d.raw.columns) + 5, calls          # values once; realized parts once per metadata
    d.raw = d.raw.copy()                                         # a new raw revision is hashed again
    data_signature(d)
    assert calls["n"] > len(d.raw.columns)


def test_session_data_keys_do_not_use_object_addresses():
    """RSession reload keys bind the content identity: two worksheets with the same file hash and metadata never share
    the worker's loaded data (the v1.3.5 key used id(ds.raw), which can be reused by another frame)."""
    from statexplore.analysis.session import RSession

    a, a2, b = _sheets()
    assert RSession._data_signature_of(a) == RSession._data_signature_of(a2)
    assert RSession._data_signature_of(a) != RSession._data_signature_of(b)
    assert RSession._metadata_signature(a) != RSession._metadata_signature(b)
    import inspect

    src = inspect.getsource(RSession)
    assert "id(ds.raw)" not in src


def _legacy(run):
    meta = run.meta_dict()
    meta.pop("identity_schema")
    snap = dict(meta["data_snapshot"])
    for k in ("content_id", "identity_schema", "source_options"):
        snap.pop(k, None)
    snap["schema"] = 2
    meta["data_snapshot"] = snap
    from statexplore.discovery.engine import DiscoveryRun, legacy_data_signature
    from statexplore.discovery.space import view

    old = DiscoveryRun.from_meta(meta, list(run.rows))
    return old, legacy_data_signature


def test_legacy_identity_is_unverified_everywhere(tmp_path):
    """v1.3.5 searches (identity schema 2) stay viewable but are never cached from, resumed, rebuilt or exported as a
    runnable reproduction; 'New search from this one' works and records the reason (explicit legacy policy)."""
    from statexplore.discovery import validate as V
    from statexplore.discovery.engine import FIT_RULES, ResumeRefused, cache_from, create_run, execute, restart_from
    from statexplore.discovery.history import context
    from statexplore.discovery.space import SearchSettings, candidate_key, enumerate_candidates, view
    from statexplore.export.audit import audit_record

    a, a2, _b = _sheets()
    s = SearchSettings(**SWB)
    run = create_run(a, s)
    k = candidate_key(next(enumerate_candidates(s)))
    run.rows = [{"idx": 0, "key": k, "status": "ok", "p": 0.01}]
    run.n_done, run.status = 1, "partial"
    run.engine = {"fit_rules": FIT_RULES, "r_version": "x", "packages": {}}
    old, legacy_sig = _legacy(run)
    old.signature = legacy_sig(view(a, s))           # what v1.3.5 recorded
    assert old.identity_schema == 2
    why = old.compatibility(a2)
    assert "identity schema 2" in why and "worksheet" in why
    assert not old.resumable and old.resume_reason(a2)
    with pytest.raises(ResumeRefused, match="identity schema 2"):      # (after the more specific v1.3.5 reasons)
        execute(old, a2)
    fresh = create_run(a2, s)
    assert k not in cache_from(fresh, [old], a2)
    ctx = context(a2, old)
    assert ctx.state == "unverified" and ctx.original is None
    assert V.bundle_data({"search_signature": old.signature}, a2, old)[0] is None
    rec = audit_record(old, a2)
    assert rec["data"]["historical"]["identity_verifiable"] is False
    assert "unverified" in rec["data"]["historical"]["fingerprints"]["analysis_signature"]
    new = restart_from(old, a2)
    assert new.identity_schema == 3 and new.parent_id == old.id and "identity schema 2" in new.lineage["reason"]


def test_console_import_and_reshape_change_the_content_identity(tmp_path):
    from statexplore.core.console_import import diff_import
    from statexplore.core.identity import content_id
    from statexplore.core.structure import ReshapeSpec, wide_to_long

    d = fx5("Test55L_LegacyUnderflow")
    csv = tmp_path / "console.csv"
    frame = pd.read_csv(FX5 / "StatExplore_Test55L_LegacyUnderflow.csv")
    frame.to_csv(csv, index=False)
    meta = {"columns": [{"name": c, "class": "numeric"} for c in frame.columns]}
    _diff, same = diff_import(d, csv, meta)
    frame.loc[0, "y"] = frame.loc[0, "y"] + 1
    frame.to_csv(csv, index=False)
    _diff, changed = diff_import(d, csv, meta)
    assert content_id(same) != content_id(d)                      # another source (console), conservatively distinct
    assert content_id(changed) != content_id(same)
    w = pd.DataFrame({"id": [1.0, 2.0], "a1": [1.0, 2.0], "a2": [3.0, 4.0]})
    w.to_csv(tmp_path / "w.csv", index=False)
    wd = import_file(tmp_path / "w.csv")
    long = wide_to_long(wd, ReshapeSpec(id_var="id", measures={"a": ["a1", "a2"]}, time_values=[1, 2], time_name="t"))
    assert content_id(long) != content_id(wd)


# ---- AUD135-03: transactional, isolated exports with consent ---------------------------------------------------------------

def test_transaction_policy_negative_controls(tmp_path):
    """The central registry catches: data without consent (declared or disguised), unregistered files, symbolic links;
    a failed export leaves no folder and no staging data behind; nothing in the parent folder is touched."""
    from statexplore.export.bundle import ConsentError, ExportError, export_transaction
    from statexplore.export.exporters import export_data_csv

    parent = tmp_path / "out"
    parent.mkdir()
    (parent / "old_data.csv").write_text("secret\n")
    (parent / "notes.txt").write_text("mine\n")
    before = sorted(p.name for p in parent.iterdir())
    d = fx5("Test55L_LegacyUnderflow")
    with pytest.raises(ConsentError):
        with export_transaction(parent, "x", False) as tx:
            tx.write_data("a_data.csv", lambda p: export_data_csv(d, p))
    with pytest.raises(ExportError):
        with export_transaction(parent, "x", False) as tx:
            tx.write_text("data.csv", "y\n1\n", "results")             # data disguised as results
    with pytest.raises(ConsentError):
        with export_transaction(parent, "x", False) as tx:
            export_data_csv(d, tx.stage / "hidden.csv")                # the choke point refuses an unregistered write
    with pytest.raises(ExportError, match="outside the export registry"):
        with export_transaction(parent, "x", True) as tx:
            tx.write_text("a.txt", "x", "metadata")
            (tx.stage / "b.txt").write_text("unregistered")
    with pytest.raises(RuntimeError, match="boom"):
        with export_transaction(parent, "x", True) as tx:
            tx.write_data("a_data.csv", lambda p: export_data_csv(d, p))
            raise RuntimeError("boom")
    with pytest.raises(ConsentError):
        export_transaction(parent, "x", None).__enter__()           # consent must be explicit
    assert sorted(p.name for p in parent.iterdir()) == before          # nothing new, nothing removed, no staging left
    assert (parent / "old_data.csv").read_text() == "secret\n"


def test_portable_exports_never_reuse_a_folder(tmp_path):
    """Consent true -> false into the same parent: two separate new folders; the second has no data file, its manifest
    and inventory say code-only, and the first bundle and unrelated files are untouched."""
    from statexplore.export.exporters import write_portable

    d = fx5("Test55L_LegacyUnderflow")
    script = 'sx_data_file <- "C:/tmp/data.csv"\ndat <- utils::read.csv(sx_data_file)\n'
    parent = tmp_path / "chosen"
    parent.mkdir()
    (parent / "analysis.R").write_text("# an older script\n")
    (parent / "analysis_data.csv").write_text("old,data\n")
    w1 = write_portable(script, parent, d, include_data=True)
    w2 = write_portable(script, parent, d, include_data=False)
    f1, f2 = folder_of(w1), folder_of(w2)
    assert f1 != f2 and f1.parent == f2.parent == parent
    assert (f1 / "analysis_data.csv").exists() and not list(f2.glob("*data*.csv"))
    inv1, inv2 = (json.loads((f / "bundle_inventory.json").read_text()) for f in (f1, f2))
    assert inv1["bundle_kind"] == "runnable" and inv1["data_included"] is True
    assert inv2["bundle_kind"] == "code-only" and inv2["data_included"] is False and inv2["data_consent"] is False
    man2 = json.loads((f2 / "analysis_manifest.json").read_text())
    assert man2["data"]["included"] is False and man2["bundle_kind"] == "code-only"
    assert {x["path"] for x in inv2["files"]} == {p.name for p in f2.iterdir()} - {"bundle_inventory.json"}
    assert (parent / "analysis.R").read_text() == "# an older script\n"
    assert (parent / "analysis_data.csv").read_text() == "old,data\n"
    assert not [p for p in parent.iterdir() if ".incomplete-" in p.name]
    # the withheld data's hash scratch never touched the export folder; the code-only script checks that SHA-256
    assert man2["data"]["sha256"] == hashlib.sha256((f1 / "analysis_data.csv").read_bytes()).hexdigest()


def test_audit_export_goes_to_a_new_folder(tmp_path):
    from statexplore.discovery.engine import create_run
    from statexplore.discovery.space import SearchSettings
    from statexplore.export.audit import read_audit, write_audit

    a = import_file(WORKBOOK, sheet="A")
    run = create_run(a, SearchSettings(**SWB))
    f1 = write_audit(run, a, tmp_path)
    f2 = write_audit(run, a, tmp_path)
    assert folder_of(f1) != folder_of(f2)
    rec = read_audit(one(f1, f"search_{run.id}_audit.json"))
    assert rec["schema_version"] == 3 and rec["data"]["historical"]["identity_verifiable"] is True


# ---- AUD135-07: historical ordinary analyses export their own data view -----------------------------------------------------

class _FakeWorker:
    """Controlled R worker (no R): records the scripts, answers with a fixed analysis response."""
    alive = True
    generation = 1

    def __init__(self):
        self.calls = []

    def run(self, code, **kw):
        self.calls.append(code)
        return SimpleNamespace(versions={"r_version": "CONTROL ONLY", "packages": {}}, warnings=[], elapsed=0.0,
                               result={"n_used": 15, "n_excluded": 0, "tables": []})

    def request(self, *a, **k):
        return SimpleNamespace(result={"r_version": "CONTROL ONLY", "packages": {}})


def _scores(tmp_path: Path):
    from statexplore.core.dataset import HoldoutSpec

    f = tmp_path / "input.csv"
    pd.DataFrame({"score": [x + 0.5 for x in range(20)]}).to_csv(f, index=False)
    d = import_file(f)
    d.set_variable_field("score", "measure", "continuous")
    d.set_holdout(HoldoutSpec(0.25, 1))
    return d


def _t_one(session, d):
    from statexplore.analysis.builders import BY_KEY
    from statexplore.analysis.framework import run_analysis

    return run_analysis(session, d, BY_KEY["t_one_sample"], {"dependent": ["score"]}, {})


def test_historical_report_exports_the_original_view_or_refuses(tmp_path):
    """Port of the auditor's AUD135-07 reproducer (baseline 6 pass / 2 fail): after the holdout seed changes, the Reports
    export carries the ORIGINAL training rows (rebuilt from the result's snapshot and proven by the input SHA-256) - or,
    when the original data cannot be proven, refuses with a reason. Never the current split with the old script."""
    from statexplore.analysis.session import RSession
    from statexplore.core.dataset import HoldoutSpec
    from statexplore.core.project import Project
    from statexplore.export.exporters import export

    d = _scores(tmp_path)
    pr = Project(d)
    sess = RSession(_FakeWorker(), tmp_path / "worker")
    res, art = _t_one(sess, d)
    pr.add_result(res, art)
    prov = res.values["analysis_provenance"]
    old = pd.read_csv(tmp_path / "worker" / "data.csv")
    alias = d.var("score").alias
    assert prov["input_sha256"] == hashlib.sha256((tmp_path / "worker" / "data.csv").read_bytes()).hexdigest()
    assert int((old["sx_split"] == "train").sum()) == 15
    assert old.loc[old.sx_split == "train", alias].mean() == pytest.approx(9.7)
    original = res.to_dict()
    w1 = export(pr, [res.id], tmp_path / "before", {"R"}, include_history=False, include_data=True)
    b1 = pd.read_csv([p for p in w1 if p.name.endswith("_data.csv")][0])
    assert b1.equals(old[[alias, "sx_split"]])                          # unchanged view: exact data
    d.set_holdout(HoldoutSpec(0.25, 2))
    cur = d.analysis_frame()["score"]
    assert len(cur) == 15 and cur.mean() == pytest.approx(10.3)          # same n, different membership
    assert res.to_dict() == original
    w2 = export(pr, [res.id], tmp_path / "after", {"R"}, include_history=False, include_data=True)
    data2 = [p for p in w2 if p.name.endswith("_data.csv")]
    assert len(data2) == 1
    b2 = pd.read_csv(data2[0])
    assert b2.equals(old[[alias, "sx_split"]])                          # the ORIGINAL seed-1 rows, not seed 2
    assert b2.loc[b2.sx_split == "train", alias].mean() == pytest.approx(9.7)
    # the same result after an R Console replacement of the data: refused, no runnable script, results kept
    from statexplore.core.console_import import diff_import

    frame = pd.read_csv(tmp_path / "input.csv")
    frame.loc[0, "score"] = 99.0
    frame.to_csv(tmp_path / "console.csv", index=False)
    _diff, new_ds = diff_import(d, tmp_path / "console.csv", {"columns": [{"name": "score", "class": "numeric"}]})
    pr2 = Project(new_ds)
    pr2.results = list(pr.results)
    w3 = export(pr2, [res.id], tmp_path / "console_out", {"R", "xlsx"}, include_history=False, include_data=True)
    names = [p.name for p in w3]
    assert "REPRODUCTION_UNAVAILABLE.txt" in names and "statexplore_results.xlsx" in names
    assert not [n for n in names if n.endswith("_data.csv") or (n.endswith(".R") and "not_runnable" not in n)]
    inv = json.loads(one(w3, "bundle_inventory.json").read_text())
    assert inv["bundle_kind"] == "results-only" and inv["data_included"] is False


def test_multiple_results_get_separately_bound_bundles(tmp_path):
    """Two results analysed under different holdout seeds -> two bundles, each with its OWN data (never one shared
    data.csv); declined consent -> code-only bundles with no data anywhere."""
    from statexplore.analysis.session import RSession
    from statexplore.core.dataset import HoldoutSpec
    from statexplore.core.project import Project
    from statexplore.export.exporters import export

    d = _scores(tmp_path)
    pr = Project(d)
    sess = RSession(_FakeWorker(), tmp_path / "worker")
    r1, a1 = _t_one(sess, d)
    s1 = pd.read_csv(tmp_path / "worker" / "data.csv")["sx_split"].tolist()
    d.set_holdout(HoldoutSpec(0.25, 2))
    r2, a2 = _t_one(sess, d)
    s2 = pd.read_csv(tmp_path / "worker" / "data.csv")["sx_split"].tolist()
    assert s1 != s2
    pr.add_result(r1, a1)
    pr.add_result(r2, a2)
    w = export(pr, [r1.id, r2.id], tmp_path / "e", {"R", "py"}, include_history=False, include_data=True)
    data = sorted((p for p in w if p.name.endswith("_data.csv")), key=lambda p: p.parent.name)
    assert len(data) == 2 and data[0].parent != data[1].parent
    got = {p.parent.name.rsplit("_", 1)[-1]: pd.read_csv(p)["sx_split"].tolist() for p in data}
    assert got == {r1.id: s1, r2.id: s2}
    assert not [p for p in w if p.name == "data.csv"]
    w0 = export(pr, [r1.id, r2.id], tmp_path / "e", {"R"}, include_history=False, include_data=False)
    assert folder_of(w0) != folder_of(w)
    assert not [p for p in folder_of(w0).rglob("*") if p.suffix == ".csv" and "data" in p.name]
    assert json.loads(one(w0, "bundle_inventory.json").read_text())["bundle_kind"] == "code-only"


def test_result_without_provenance_is_refused(tmp_path):
    """A result saved by v1.3.5 (no analysis provenance) is never paired with today's data as a runnable bundle."""
    from statexplore.analysis.session import RSession
    from statexplore.core.project import Project
    from statexplore.export.exporters import export
    from statexplore.export.historical import Unverified, resolve

    d = _scores(tmp_path)
    pr = Project(d)
    res, art = _t_one(RSession(_FakeWorker(), tmp_path / "worker"), d)
    res.values.pop("analysis_provenance")
    pr.add_result(res, art)
    with pytest.raises(Unverified, match="1.3.5 or earlier"):
        resolve(res, d, tmp_path / "s")
    w = export(pr, [res.id], tmp_path / "e", {"R"}, include_history=False, include_data=True)
    assert "REPRODUCTION_UNAVAILABLE.txt" in [p.name for p in w]
    res.values["analysis_provenance"] = {"schema": 1, "identity_schema": 3, "input_sha256": "0" * 64,
                                         "script_sha256": "x"}
    with pytest.raises(Unverified, match="recorded script differs"):
        resolve(res, d, tmp_path / "s")


def test_failed_result_keeps_provenance_and_stays_failed(tmp_path):
    from statexplore.analysis.session import RSession
    from statexplore.core.schema import Status
    from statexplore.rbridge.worker import RScriptError

    class Failing(_FakeWorker):
        def run(self, code, **kw):
            if "t.test" in code:
                raise RScriptError("boom")
            return super().run(code, **kw)
    d = _scores(tmp_path)
    res, _ = _t_one(RSession(Failing(), tmp_path / "w"), d)
    assert res.status is Status.FAILED and res.values["analysis_provenance"]["input_sha256"]


# ---- AUD135-01: typed verification in the GENERATED R code -----------------------------------------------------------------

def _verifier_lib() -> str:
    """The helper part of the candidate's generated verifier (exactly as embedded in every reproduction script)."""
    from statexplore.discovery.validate import R_REPRO_CHECKS

    return R_REPRO_CHECKS.split("# ---- required files")[0]


_STUB = """sx_long <- function() FALSE
sx_disc_has_cluster <- FALSE
sx_val_index <- c(0); sx_val_keys <- c("candidate0"); sx_val_labels <- c("candidate 0")
sx_val_full <- data.frame(y = 1:20, row.names = as.character(1:20))
sx_alias <- character()
"""


@pytest.mark.r
def test_owner_diagnostic_num_id_sel_cases_on_generated_verifier(tmp_path):
    """The owner's R diagnostic NUM-01..09, ID-01 and SEL-01..03 (v1.3.5: 7 FAIL) against the v1.3.6 generated helpers,
    plus malformed / overflow / underflow text, NaN, integer keys, delimiter-collision keys and the NA -> FAIL rule."""
    from statexplore.discovery.rcode import DISCOVERY_HELPERS

    code = _STUB + DISCOVERY_HELPERS.split("sx_fit_budget <- new.env()")[0] + "\n" + _verifier_lib() + r"""
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
    res = rscript(code, tmp_path)
    assert res.returncode == 0, res.stderr[-3000:]
    tab = pd.read_csv(tmp_path / "diagnostic_results.csv")
    assert (tab["status"] == "PASS").all(), tab[tab["status"] != "PASS"].to_string()
    assert len(tab) == 30


def _summary_fixture():
    B = pd.DataFrame({"b": [1, 2], "status": ["fitted", "fitted"], "reason": ["", ""], "selected": ["candidate 0"] * 2,
                      "selected_index": [0, 0], "selected_key": ["candidate0"] * 2, "train_score": [-2.0, -3.0],
                      "visited": [1, 1], "scored": [1, 1], "n_sig_train": [1, 1], "coefficient": ["beta", "beta"],
                      "estimate": [1.0, 3.0], "n_coef": [1, 1], "diagnostics": ["clean", "clean"], "flags": ["", ""],
                      "n": [20, 20], "groups": [20, 20]})
    C = pd.DataFrame({"b": [1, 2], "selected_index": [0, 0], "selected_key": ["candidate0"] * 2, "term": ["beta"] * 2,
                      "coefficient": ["beta"] * 2, "estimate": [1.0, 3.0]})
    return B, C


def _r_df(df: pd.DataFrame) -> str:
    from statexplore.rbridge.codegen import r_vec

    cols = []
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            cols.append(f"`{c}` = {r_vec([str(x) for x in df[c]])}")
        else:
            cols.append(f"`{c}` = c({', '.join(repr(float(x)) for x in df[c])})")
    return "data.frame(" + ", ".join(cols) + ", stringsAsFactors = FALSE, check.names = FALSE)"


def _summary_case(tmp_path: Path, tamper: str) -> tuple[int, pd.DataFrame]:
    """SUM-01..07 of the owner diagnostic, end to end through the generated reproduction script: the app files are
    written by the app's Python writer and summaries (stability / estimands) by the app's Python aggregation; the R
    verifier recomputes them independently from the reproduced draws (synthetic reproduced rows, no model fit)."""
    from statexplore.discovery import validate as V

    B, C = _summary_fixture()
    Bt = B.assign(coefficient_label=B["coefficient"])
    st = V.stability_table(Bt, ["candidate 0"], [0], ["candidate0"])
    est = V.estimand_table(Bt, C.assign(coefficient_label=C["coefficient"]), {0: "candidate 0"})
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
    Ca = C.copy()
    if tamper == "coefficient":
        Ca.loc[0, "estimate"] = 10.0
    d = tmp_path / tamper
    d.mkdir()
    ledger = [{"part": f"boot{b}", "mode": "bootstrap", "split": "", "fold": None, "b": b, "chunk": 1,
               "status": "fitted", "committed": True} for b in (1, 2)]
    files = {"bootstrap_results.csv": V.evidence_csv("bootstrap", Bt.to_dict("records")),
             "bootstrap_coefficients.csv": V.evidence_csv("coefs", Ca.to_dict("records")),
             "stability.csv": V.evidence_csv("stability", st), "bootstrap_estimands.csv": V.evidence_csv("estimands", est),
             "execution_ledger.csv": V.evidence_csv("ledger", ledger)}
    for k, t in files.items():
        (d / k).write_text(t, encoding="utf-8")
    expected = [{"file": "bootstrap_results.csv", "required": True, "rows": 2, "applicable": True},
                {"file": "bootstrap_coefficients.csv", "required": True, "rows": 2, "applicable": True},
                {"file": "stability.csv", "required": True, "rows": 1, "applicable": True},
                {"file": "bootstrap_estimands.csv", "required": True, "rows": 1, "applicable": True},
                {"file": "execution_ledger.csv", "required": True, "rows": 2, "applicable": True}]
    chunk = f"sx_result <- list(boot = list(draws = {_r_df(B)}, coefs = {_r_df(C)}))"
    script = V.reproduce_script("", _STUB, [chunk], expected, {"status": "complete", "target": "rows"},
                                replay={"ledger": ledger, "pending_manifest": [], "pending_draws": [], "splits": [],
                                        "config": {"holdout_fraction": 0.2, "k": 5}, "alias_map": {}, "rules": {}})
    (d / "check.R").write_text(script, encoding="utf-8")
    r = run_r(d, "check.R", timeout=600)
    return r.returncode, pd.read_csv(d / "validation_check_results.csv")


@pytest.mark.r
@pytest.mark.parametrize("tamper, failing", [
    ("none", None), ("coefficient", "bootstrap coefficients: estimate"), ("count", "stability"),
    ("summary", "bootstrap estimands (recomputed from the reproduced coefficients): estimate_median"),
    ("frequency", "stability (recomputed from the reproduced draws and the declared pool): freq_of_attempted"),
    ("sign", "stability (recomputed from the reproduced draws and the declared pool): positive_sign"),
    ("key", "stability (recomputed from the reproduced draws and the declared pool): candidate_key")])
def test_owner_summary_cases_sum01_to_sum07(tmp_path, tamper, failing):
    rc, chk = _summary_case(tmp_path, tamper)
    bad = chk[chk["status"] == "FAIL"]
    if failing is None:
        assert rc == 0 and bad.empty, bad.to_string()
        names = set(chk["check"])
        for f in ("estimate_median", "freq_of_attempted", "positive_sign", "candidate_key", "estimate_p2.5"):
            assert f"stability (recomputed from the reproduced draws and the declared pool): {f}" in names
    else:
        assert rc != 0 and bad["check"].str.startswith(failing).any(), bad.to_string()


@pytest.mark.r
def test_evidence_csv_round_trip_into_r(tmp_path):
    """The Python writer and the R reader agree exactly on leading zeros, Unicode, delimiters / quotes / newlines in
    text, the literal NA, empty text, missing numbers, signed infinities and extreme magnitudes."""
    from statexplore.discovery import validate as V

    texts = ["001", "Zürich × ü", 'a,"b"', "line1\nline2", "NA", "", "m:", " padded "]
    recs = [{"part": f"p{i}", "candidate_index": i, "status": "ok", "reason": t, "loglik": [-1.5, None, 1e-300,
             5e-324, -1e300, 0.1, 3.0, 2.5][i], "statistic": [math.inf, -math.inf, None, 1.0, 2.0, 3.0, 4.0, 5.0][i],
             "df": 1.0, "log_p": [-1e300 * 10 if False else -4.6e5, None, -math.inf, -1.0, -2.0, -3.0, -4.0, -5.0][i],
             "reference_loglik": None, "aic": None, "score": None, "flags": t, "reference_flags": "", "rows_sig": "x"}
            for i, t in enumerate(texts)]
    (tmp_path / "candidate_trace.csv").write_text(V.evidence_csv("trace", recs), encoding="utf-8")
    from statexplore.rbridge.codegen import r_vec

    code = _STUB + _verifier_lib() + f"""
a <- sx_read("candidate_trace.csv")
stopifnot(identical(a$reason, enc2utf8({r_vec(texts)})), identical(a$flags, a$reason))
p <- sx_parse_num(a$statistic); stopifnot(identical(p$state[1:3], c("inf", "-inf", "na")))
q <- sx_parse_num(a$loglik); stopifnot(identical(q$state[1:5], c("num", "na", "num", "num", "num")), q$value[3] == 1e-300,
  q$value[4] > 0, q$value[5] == -1e300)
stopifnot(identical(sx_parse_num(a$log_p)$state[3], "-inf"), identical(a$candidate_index, as.character(0:7)))
stopifnot(all(sx_num_ok(a$loglik, c(-1.5, NA, 1e-300, 5e-324, -1e300, 0.1, 3, 2.5), 1e-12, 1e-12)))
cat("ROUNDTRIP OK\\n")
"""
    res = rscript(code, tmp_path)
    assert res.returncode == 0 and "ROUNDTRIP OK" in res.stdout, res.stderr[-3000:]
    with pytest.raises(V.EvidenceError):
        V.evidence_csv("trace", [dict(recs[0], candidate_index=1.5)])           # wrong type at the writer boundary
    with pytest.raises(V.EvidenceError):
        V.evidence_csv("trace", [dict(recs[0], loglik=math.inf)])               # infinity not declared for loglik
    with pytest.raises(V.EvidenceError):
        V.evidence_csv("trace", [dict(recs[0], extra=1)])                       # undeclared column


# ---- AUD135-04 / 05: execution ledger and exact budget (controlled session) ------------------------------------------------

class _ChunkSession:
    """Controlled R session for ValidationRunner: every fold / draw costs `cost` fits; a chunk whose remaining budget runs
    out inside returns the remaining parts with status 'budget' and denied fits (as the R code does). `fail_on` raises
    for that chunk title (a lost worker); `dup` returns one part twice."""
    load_script = ""

    def __init__(self, cost=3, fail_on="", dup=False):
        self.cost, self.fail_on, self.dup, self.titles = cost, fail_on, dup, []

    def run(self, v, *, kind, title, body, packages, inputs, timeout):
        import re

        self.titles.append(title)
        if self.fail_on and self.fail_on in title:
            raise RuntimeError("R worker lost")
        rem = int(re.search(r"sx_budget_reset\((\d+)\)", body).group(1))
        used, denied = 0, 0

        def take():
            nonlocal used, denied
            if used + self.cost <= rem:
                used += self.cost
                return True
            denied += 1
            used = rem
            return False
        if "sx_val_split(" in body:
            sid = re.search(r'sx_val_split\("([^"]+)"', body).group(1)
            folds = sorted({int(x) for x in re.search(r"sx_vfold <- c\(([^)]*)\)", body).group(1).split(", ")} - {0})
            recs = []
            for f in folds:
                ok = denied == 0 and take()
                recs.append({"split": sid, "fold": f, "status": "scored" if ok else "budget", "reason": "",
                             "selected": "", "selected_index": 1 if ok else None, "selected_key": "k",
                             "train_score": -1.0 if ok else None, "n_train": 5, "n_test": 1 if ok else 0})
            if self.dup:
                recs.append(dict(recs[0]))
            raw = {"folds": recs, "predictions": None, "trace": None}
        else:
            bs = [int(x) for x in re.search(r"b = c\(([^)]*)\)", body).group(1).split(", ")]
            recs = []
            for b in sorted(set(bs)):
                ok = denied == 0 and take()
                recs.append({"b": b, "status": "fitted" if ok else "budget", "selected_index": 1 if ok else None,
                             "selected_key": "k", "estimate": 1.0 if ok else None, "coefficient": "",
                             "diagnostics": "clean"})
            raw = {"boot": {"draws": recs, "coefs": None, "trace": None}}
        raw["budget"] = {"used": used, "retries": 0, "aux": 0, "cap": rem, "at_cap": used >= rem, "denied": denied,
                         "cache_hits": 0, "cache_misses": 0}
        return SimpleNamespace(messages=[]), raw


def _runner(session, modes=("holdout",), B=2, cap=10_000, **kw):
    from statexplore.discovery import engine
    from statexplore.discovery import validate as V
    from statexplore.discovery.space import SearchSettings

    d = fx5("Test55L_LegacyUnderflow")
    run = engine.create_run(d, SearchSettings(**S55L))
    cfg = V.ValidationConfig(target="rows", modes=list(modes), k=2, B=B, seed=3, max_fits=cap, pipeline="fixed",
                             candidate=0, **kw)
    from statexplore.discovery.space import candidate_key, enumerate_candidates

    c = next(enumerate_candidates(run.settings))
    run.rows = [{"idx": c.index, "key": candidate_key(c), "status": "ok", "focal": list(c.focal),
                 "covariates": list(c.covariates), "kind": c.kind, "inter": [list(t) for t in c.inter]}]
    return V.ValidationRunner(session, d, run, cfg)


@pytest.mark.parametrize("cap, status, used, denied", [
    (3, "complete", 3, 0),           # exactly the needed fits: COMPLETE although used == cap (AUD135-05)
    (2, "partial", 2, 1),            # one fit short: the holdout part is a budget stop, never scored
    (100, "complete", 3, 0)])
def test_exact_budget_boundary_holdout(cap, status, used, denied):
    r = _runner(_ChunkSession(cost=3))
    r.budget["cap"] = cap
    v = r.execute()
    b = v["fit_budget"]
    assert v["status"] == status and b["used"] == used and b["denied"] == denied and b["used"] <= cap
    assert b["at_cap"] is (used >= cap) and b["exhausted"] is (status == "partial")
    assert v["execution"]["planned_parts"] == 1 and v["execution"]["committed_parts"] == 1


def test_budget_exactly_exhausted_with_work_remaining_is_partial_not_dispatched():
    s = _ChunkSession(cost=3)
    r = _runner(s, modes=("holdout", "cv"))
    r.budget["cap"] = 3                       # the holdout needs 3; the CV split cannot get a single fit
    v = r.execute()
    assert v["status"] == "partial" and v["fit_budget"]["used"] == 3 and len(s.titles) == 1
    led = {p["part"]: p["status"] for p in v["execution"]["parts"]}
    assert led == {"holdout:fold1": "scored", "cv1:fold1": "not_run", "cv1:fold2": "not_run"}
    assert any("2 planned part(s) not run" in p for p in v["problems"])


def test_zero_budget_runs_nothing():
    s = _ChunkSession()
    r = _runner(s)
    r.budget["cap"] = 0
    v = r.execute()
    assert v["status"] == "partial" and not s.titles and v["execution"]["committed_parts"] == 0


def test_cancel_before_and_after_first_chunk_and_late_cancel():
    s = _ChunkSession()
    v = _runner(s, modes=("holdout", "cv")).execute(should_stop=lambda: True)
    assert v["status"] == "cancelled" and not s.titles
    calls = {"n": 0}

    def after_first():
        calls["n"] += 1
        return calls["n"] > 1
    s2 = _ChunkSession()
    v2 = _runner(s2, modes=("holdout", "cv")).execute(should_stop=after_first)
    assert v2["status"] == "cancelled" and len(s2.titles) == 1
    assert [p["status"] for p in v2["execution"]["parts"]] == ["scored", "not_run", "not_run"]
    # the replay covers the committed chunk only; the rest of the plan is embedded for verification
    assert v2["reproduce_script"].count("# ---- committed chunk") == 1
    assert "sx_pend_M <- data.frame(" in v2["reproduce_script"]
    flag = {"done": False}
    s3 = _ChunkSession()
    r3 = _runner(s3)
    orig = s3.run

    def run_then_cancel(*a, **k):
        out = orig(*a, **k)
        flag["done"] = True
        return out
    s3.run = run_then_cancel
    v3 = r3.execute(should_stop=lambda: flag["done"])
    assert v3["status"] == "complete" and any("arrived after the last planned part" in p for p in v3["problems"])


def test_lost_worker_and_invalid_response_are_uncommitted_and_counters_unknown():
    s = _ChunkSession(fail_on="cv1")
    v = _runner(s, modes=("holdout", "cv")).execute()
    assert v["status"] == "failed"
    led = {p["part"]: p["status"] for p in v["execution"]["parts"]}
    assert led == {"holdout:fold1": "scored", "cv1:fold1": "uncommitted", "cv1:fold2": "uncommitted"}
    assert v["fit_budget"]["counters_known"] is False and any("lower bound" in p for p in v["problems"])
    assert v["reproduce_script"].count("# ---- committed chunk") == 1
    v2 = _runner(_ChunkSession(dup=True)).execute()
    assert v2["status"] == "failed" and v2["execution"]["committed_parts"] == 0
    assert "uncommitted" in {p["status"] for p in v2["execution"]["parts"]}


def test_partial_bootstrap_chunks_and_budget_inside_a_chunk():
    s = _ChunkSession(cost=2)
    r = _runner(s, modes=("bootstrap",), B=5)
    r.chunk = 2
    r.budget["cap"] = 5                  # draws 1, 2 fit (4 fits); draw 3 is denied inside chunk 2; chunk 3 not run
    v = r.execute()
    led = {p["part"]: p["status"] for p in v["execution"]["parts"]}
    assert led == {"boot1": "fitted", "boot2": "fitted", "boot3": "budget", "boot4": "budget", "boot5": "not_run"}
    assert v["status"] == "partial" and v["fit_budget"]["used"] == 5 and v["fit_budget"]["denied"] >= 1
    assert v["execution"]["chunks_committed"] == 2


# ---- real R, end to end: fits, exported scripts, fresh Rscript replay ------------------------------------------------------

def _search55(**kw):
    from statexplore.discovery import engine
    from statexplore.discovery.space import SearchSettings

    d = fx5("Test55L_LegacyUnderflow")
    run = engine.create_run(d, SearchSettings(**{**S55L, **kw}))
    engine.execute(run, d)
    return d, run


def _edit(folder: Path, fname: str, fn) -> None:
    """Edit an exported CSV as TEXT (no re-typing of untouched cells)."""
    df = pd.read_csv(folder / fname, dtype=str, keep_default_na=False)
    fn(df)
    df.to_csv(folder / fname, index=False, lineterminator="\n")


@pytest.fixture(scope="module")
def complete55(tmp_path_factory, r_worker):
    """55L pipeline validation (holdout + 2-fold CV + 3 bootstrap draws) exported with data and moved to a folder
    with spaces and Unicode; reproduced once in a fresh Rscript."""
    from statexplore.analysis.session import RSession
    from statexplore.discovery import validate as V

    tmp = tmp_path_factory.mktemp("c55")
    d, run = _search55()
    cfg = V.ValidationConfig(target="rows", modes=["holdout", "cv", "bootstrap"], k=2, B=3, seed=7, max_fits=100_000)
    v = V.ValidationRunner(RSession(r_worker, tmp / "rs"), d, run, cfg).execute()
    files = V.export(v, str(tmp / "export"), ds=d, include_data=True, run=run)
    moved = tmp / "moved folder ü é"
    shutil.copytree(folder_of(files), moved)
    res = run_r(moved, "validation_reproduce.R")
    return SimpleNamespace(v=v, moved=moved, res=res, tmp=tmp, d=d, run=run)


@pytest.mark.r
@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_complete_reproduction_verifies_every_field(complete55):
    c = complete55
    assert c.v["status"] == "complete" and c.v["execution"]["committed_parts"] == c.v["execution"]["planned_parts"] == 6
    assert c.res.returncode == 0, c.res.stdout[-3000:] + c.res.stderr[-3000:]
    assert "The complete validation was reproduced" in c.res.stdout and "6 of 6 planned parts" in c.res.stdout
    chk = pd.read_csv(c.moved / "validation_check_results.csv", keep_default_na=False)
    assert (chk["status"] != "FAIL").all() and len(chk) > 150
    names = set(chk["check"])
    for n in ("predictions: predicted", "predictions: participant", "folds: selected_key", "folds: reason",
              "bootstrap draws: train_score", "bootstrap coefficients: estimate",
              "bootstrap coefficients: identity (draw, selected candidate, declared key, term, n_coef)",
              "stability (recomputed from the reproduced draws and the declared pool): estimate_median",
              "bootstrap estimands (recomputed from the reproduced coefficients): estimate_p97.5",
              "realized sizes (recomputed from the verified plan and the analysed units): test_units",
              "split manifest: fold", "split manifest: participant keys identify the analysed participants",
              "bootstrap draw manifest: original keys equal the split manifest's keys of the same rows",
              "candidate trace: df", "candidate trace: reference_loglik", "candidate trace: flags",
              "execution ledger: status", "replay scope: the replayed parts are exactly the committed parts"):
        assert n in names and (chk.loc[chk["check"] == n, "status"] == "PASS").all(), n
    # independent oracle: stability recomputed here from the exported (verified) draws with numpy type-7 quantiles
    B = pd.DataFrame(c.v["bootstrap"])
    st = pd.DataFrame(c.v["stability"])
    for _, r in st.iterrows():
        e = B.loc[(B["status"] == "fitted") & (B["selected_index"] == r["candidate_index"]), "estimate"].dropna()
        assert r["selected"] == len(e) and r["attempted"] == len(B)
        if len(e):
            assert r["estimate_median"] == pytest.approx(float(np.quantile(e, 0.5)), abs=1e-12)
            assert r["estimate_p2.5"] == pytest.approx(float(np.quantile(e, 0.025)), abs=1e-12)


TAMPERS = [
    ("pred", "predictions.csv", lambda df: df.__setitem__("predicted", [repr(float(df["predicted"][0]) + 0.01)]
                                                          + list(df["predicted"][1:])), "predictions: predicted"),
    ("id", "predictions.csv", lambda df: df.__setitem__("participant", ["0" + df["participant"][0]]
                                                        + list(df["participant"][1:])), "predictions: participant"),
    ("malformed", "predictions.csv", lambda df: df.__setitem__("observed", ["1.2.3"] + list(df["observed"][1:])),
     "predictions: observed"),
    ("reason", "folds.csv", lambda df: df.__setitem__("reason", ["edited"] + list(df["reason"][1:])), "folds: reason"),
    ("score_inf", "bootstrap_results.csv", lambda df: df.__setitem__("train_score", ["-Inf"] + list(df["train_score"][1:])),
     "bootstrap draws: train_score"),
    ("coef", "bootstrap_coefficients.csv", lambda df: df.__setitem__("estimate", [repr(float(x) * 1.001) for x in df["estimate"]]),
     "bootstrap coefficients: estimate"),
    ("coef_key", "bootstrap_coefficients.csv", lambda df: df.__setitem__("selected_key", ["other"] + list(df["selected_key"][1:])),
     "bootstrap coefficients"),
    ("median", "stability.csv", lambda df: df.__setitem__("estimate_median", ["100" if x else x for x in df["estimate_median"]]),
     "stability (recomputed from the reproduced draws and the declared pool): estimate_median"),
    ("freq", "stability.csv", lambda df: df.__setitem__("freq_of_attempted", ["0.5"] * len(df)),
     "stability (recomputed from the reproduced draws and the declared pool): freq_of_attempted"),
    ("sign", "stability.csv", lambda df: df.__setitem__("positive_sign", ["0.25" if x else x for x in df["positive_sign"]]),
     "stability (recomputed from the reproduced draws and the declared pool): positive_sign"),
    ("cand_key", "stability.csv", lambda df: df.__setitem__("candidate_key", ["x"] + list(df["candidate_key"][1:])),
     "stability (recomputed from the reproduced draws and the declared pool): candidate_key"),
    ("universe", "stability.csv", lambda df: df.drop(index=len(df) - 1, inplace=True),
     "stability (recomputed from the reproduced draws and the declared pool): same rows"),
    ("estimand", "bootstrap_estimands.csv", lambda df: df.__setitem__("estimate_median", ["9"] * len(df)),
     "bootstrap estimands (recomputed from the reproduced coefficients): estimate_median"),
    ("realized", "realized_sizes.csv", lambda df: df.__setitem__("test_units", [str(int(df["test_units"][0]) + 1)]
                                                                 + list(df["test_units"][1:])),
     "realized sizes (recomputed from the verified plan and the analysed units): test_units"),
    ("fold", "split_manifest.csv", lambda df: df.__setitem__("fold", ["0" if df["fold"][0] != "0" else "1"]
                                                             + list(df["fold"][1:])), "split manifest: fold"),
    ("merge_ids", "split_manifest.csv", lambda df: df.__setitem__("participant", [df["participant"][1]]
                                                                  + list(df["participant"][1:])),
     "split manifest: participant keys identify the analysed participants"),
    ("orig_id", "bootstrap_draws.csv", lambda df: df.__setitem__("original_participant", ["999999"]
                                                                 + list(df["original_participant"][1:])),
     "bootstrap draw manifest"),
    ("trace_df", "candidate_trace.csv", lambda df: df.__setitem__("df", [repr(float(df["df"][0]) + 1)]
                                                                  + list(df["df"][1:])), "candidate trace: df"),
    ("trace_flags", "candidate_trace.csv", lambda df: df.__setitem__("flags", ["x"] + list(df["flags"][1:])),
     "candidate trace: flags"),
    ("ledger", "execution_ledger.csv", lambda df: df.__setitem__("status", ["failed"] + list(df["status"][1:])),
     "execution ledger: status"),
    ("metric", "performance.csv", lambda df: df.__setitem__("RMSE", [repr(float(x) + 1e-3) for x in df["RMSE"]]),
     "performance (recomputed in R from the reproduced predictions): RMSE"),
    ("dup", "folds.csv", lambda df: df.loc.__setitem__(len(df), df.iloc[0]), "folds: unique keys (app)"),
    ("column", "candidate_trace.csv", lambda df: df.drop(columns="rows_sig", inplace=True), "candidate trace: columns"),
]


@pytest.mark.r
@pytest.mark.slow
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("label, fname, fn, check", TAMPERS, ids=[t[0] for t in TAMPERS])
def test_each_field_family_tamper_fails_with_a_named_check(complete55, tmp_path, label, fname, fn, check):
    """AUD135-01/02: altering ONE field family at the same row count (or a key / column / universe) makes the generated
    script exit nonzero and names the failing check."""
    t = tmp_path / f"t {label}"
    shutil.copytree(complete55.moved, t)
    for f in t.glob("validation_check_results.csv"):
        f.unlink()
    _edit(t, fname, fn)
    r = run_r(t, "validation_reproduce.R")
    assert r.returncode != 0, label
    chk = pd.read_csv(t / "validation_check_results.csv", keep_default_na=False)
    bad = chk[chk["status"] == "FAIL"]
    assert bad["check"].str.startswith(check).any(), (label, bad.to_string()[:3000])


@pytest.mark.r
@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_missing_file_and_data_checksum(complete55, tmp_path):
    t = tmp_path / "missing"
    shutil.copytree(complete55.moved, t)
    (t / "stability.csv").unlink()
    r = run_r(t, "validation_reproduce.R")
    chk = pd.read_csv(t / "validation_check_results.csv", keep_default_na=False)
    assert r.returncode != 0 and (chk.loc[chk["check"] == "file stability.csv", "status"] == "FAIL").all()
    t2 = tmp_path / "data"
    shutil.copytree(complete55.moved, t2)
    p = t2 / "validation_reproduce_data.csv"
    p.write_text(p.read_text().replace(",1,", ",2,", 1))
    r2 = run_r(t2, "validation_reproduce.R")
    assert r2.returncode != 0 and "SHA-256" in r2.stderr


@pytest.mark.r
@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_cancelled_validation_replays_committed_scope_only(r_session, tmp_path):
    """AUD135-04 (real R): cancel after the holdout chunk -> the export's reproduction replays exactly that chunk, verifies
    the pending CV plan against the embedded plan, reports partial coverage and exits 0; presenting the pending part as
    committed, or editing the pending plan, fails."""
    from statexplore.discovery import validate as V

    d, run = _search55()
    cfg = V.ValidationConfig(target="rows", modes=["holdout", "cv"], k=2, seed=11, max_fits=100_000)
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] > 1
    v = V.ValidationRunner(r_session, d, run, cfg).execute(should_stop=stop)
    assert v["status"] == "cancelled"
    led = {p["part"]: (p["status"], p["committed"]) for p in v["execution"]["parts"]}
    assert led == {"holdout:fold1": ("scored", True), "cv1:fold1": ("not_run", False), "cv1:fold2": ("not_run", False)}
    files = V.export(v, str(tmp_path / "e"), ds=d, include_data=True, run=run)
    f = folder_of(files)
    man = pd.read_csv(f / "split_manifest.csv")
    assert set(man["split"]) == {"holdout", "cv1"}                       # the PLAN stays complete and labelled
    r = run_r(f, "validation_reproduce.R")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "1 of 3 planned parts committed" in r.stdout and "CANCELLED" in r.stdout
    assert "The complete validation was reproduced" not in r.stdout
    for label, fname, fn in (("pend", "split_manifest.csv", lambda df: df.__setitem__(
            "fold", [("1" if x == "2" else "2") if s == "cv1" else x for s, x in zip(df["split"], df["fold"])])),
                             ("promote", "execution_ledger.csv", lambda df: df.__setitem__(
            "committed", ["TRUE"] * len(df)))):
        t = tmp_path / label
        shutil.copytree(f, t)
        _edit(t, fname, fn)
        assert run_r(t, "validation_reproduce.R").returncode != 0, label


@pytest.mark.r
@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_exact_budget_complete_and_one_short_partial_real_r(r_session, tmp_path):
    """AUD135-05 (real R): with the cap equal to the fits a complete holdout validation needs, the status is COMPLETE
    (used == cap); one fit fewer -> PARTIAL with the part stopped by the budget, cap never exceeded; both exports
    reproduce with identical guards in a fresh Rscript."""
    from statexplore.discovery import validate as V

    d, run = _search55()
    cfg = V.ValidationConfig(target="rows", modes=["holdout"], seed=5, max_fits=100_000)
    full = V.ValidationRunner(r_session, d, run, cfg).execute()
    U = full["fit_budget"]["used"]
    assert full["status"] == "complete" and U > 1
    out = {}
    for cap in (U, U - 1):
        r = V.ValidationRunner(r_session, d, run, cfg)
        r.budget["cap"] = cap
        v = r.execute()
        out[cap] = v
        assert v["fit_budget"]["used"] <= cap
        f = folder_of(V.export(v, str(tmp_path / f"cap{cap}"), ds=d, include_data=True, run=run))
        res = run_r(f, "validation_reproduce.R")
        assert res.returncode == 0, (cap, res.stdout[-3000:] + res.stderr[-3000:])
    ok, short = out[U], out[U - 1]
    assert ok["status"] == "complete" and ok["fit_budget"]["used"] == U and ok["fit_budget"]["at_cap"] is True
    assert ok["fit_budget"]["exhausted"] is False and not any("budget" in p for p in ok["problems"])
    assert ok["folds"][0]["selected_index"] == full["folds"][0]["selected_index"]
    assert short["status"] == "partial" and short["folds"][0]["status"] == "budget"
    assert short["fit_budget"]["denied"] >= 1 and any("fit budget reached" in p for p in short["problems"])


@pytest.mark.r
@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_workbook_sheets_fit_differently_in_one_worker_both_orders(r_worker, tmp_path):
    """AUD135-06 (real R): lm(y ~ x * z) on sheet A then B (and reversed) in ONE worker gives each sheet's own
    coefficients (independent R oracle per sheet); a discovery search on B never reuses A's fits and A's partial search
    cannot be resumed on B."""
    from statexplore.analysis.builders import BY_KEY
    from statexplore.analysis.framework import run_analysis
    from statexplore.analysis.session import RSession
    from statexplore.discovery import engine
    from statexplore.discovery.space import SearchSettings

    a, _a2, b = _sheets()
    oracle = {}
    for sh, dd in (("A", a), ("B", b)):
        dd.raw.to_csv(tmp_path / f"{sh}.csv", index=False)
        r = rscript(f'd <- read.csv("{sh}.csv"); cat(sprintf("%.15g", coef(lm(y ~ x * z, d))[["x:z"]]))', tmp_path,
                    script=f"o{sh}.R")
        oracle[sh] = float(r.stdout.strip())
    assert abs(oracle["A"] - oracle["B"]) > 0.5
    for order in (("A", "B"), ("B", "A")):
        sess = RSession(r_worker, tmp_path / ("w" + "".join(order)))
        for sh in order:
            dd = {"A": a, "B": b}[sh]
            res, _ = run_analysis(sess, dd, BY_KEY["lm"], {"dependent": ["y"], "predictors": ["x", "z"]},
                                  {"interactions": "two_way"})
            t = next(t for t in res.tables if t.name.startswith("Coefficients"))
            row = next(r for r in t.rows if "×" in str(r[0]) or ":" in str(r[0]))
            est = float(row[t.columns.index("estimate")] if "estimate" in t.columns else row[1])
            assert est == pytest.approx(oracle[sh], rel=1e-9), (order, sh)
            assert res.values["analysis_provenance"]["content_id"] != ("" if sh == "A" else None)
    s = SearchSettings(**{**SWB, "workers": 1, "max_models": None, "max_minutes": None})
    ra = engine.create_run(a, s)
    engine.execute(ra, a)
    rb = engine.create_run(b, s)
    cache = engine.cache_from(rb, [ra], b)
    assert cache == {}
    engine.execute(rb, b, cache=cache)
    assert not any(r.get("cached") for r in rb.rows)
    pa = {r["key"]: r["p"] for r in ra.rows}
    pb = {r["key"]: r["p"] for r in rb.rows}
    assert any(abs(math.log10(max(pa[k], 1e-300)) - math.log10(max(pb[k], 1e-300))) > 1 for k in pa)
    ra2 = engine.create_run(a, SearchSettings(**{**SWB, "workers": 1, "max_models": 1, "max_minutes": None}))
    engine.execute(ra2, a)
    assert ra2.status == "partial" and ra2.resume_reason(b)
    with pytest.raises(engine.ResumeRefused):
        engine.execute(ra2, b)


@pytest.mark.r
@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_historical_one_sample_t_real_r_seed_change(r_worker, tmp_path):
    """AUD135-07 (real R): one-sample t on the 15 training rows of seed 1, then seed 2 (same n, other rows). The seed-1
    result's Reports bundle (exported AFTER the seed change) carries the seed-1 rows, and its script re-run in a fresh
    Rscript reproduces n, mean, t and CI of an independent t.test on those rows; seed 2 differs."""
    from statexplore.analysis.session import RSession
    from statexplore.core.dataset import HoldoutSpec
    from statexplore.core.project import Project
    from statexplore.export.exporters import export

    d = _scores(tmp_path)
    pr = Project(d)
    sess = RSession(r_worker, tmp_path / "w")
    r1, a1 = _t_one(sess, d)
    train1 = d.analysis_frame()["score"].tolist()
    d.set_holdout(HoldoutSpec(0.25, 2))
    r2, a2 = _t_one(sess, d)
    train2 = d.analysis_frame()["score"].tolist()
    pr.add_result(r1, a1)
    pr.add_result(r2, a2)
    assert len(train1) == len(train2) == 15 and np.mean(train1) == pytest.approx(9.7) and np.mean(train2) == pytest.approx(10.3)
    o = rscript(f"for (x in list(c({', '.join(map(repr, train1))}), c({', '.join(map(repr, train2))}))) {{"
                "tt <- t.test(x); cat(sprintf('%.15g %.15g %.15g %.15g\\n', mean(x), tt$statistic, tt$conf.int[1], "
                "tt$conf.int[2])) }", tmp_path)
    (m1, t1, l1, h1), (m2, t2, _l2, _h2) = [list(map(float, line.split())) for line in o.stdout.strip().splitlines()]
    assert r1.n_used == r2.n_used == 15
    w = export(pr, [r1.id, r2.id], tmp_path / "rep", {"R"}, include_history=False, include_data=True)
    scripts = {p.parent.name.rsplit("_", 1)[-1]: p for p in w if p.suffix == ".R"}
    for rid, (m, t, lo, hi) in ((r1.id, (m1, t1, l1, h1)), (r2.id, (m2, t2, _l2, _h2))):
        sp = scripts[rid]
        rr = run_r(sp.parent, sp.name)
        assert rr.returncode == 0, rr.stdout[-2000:] + rr.stderr[-2000:]
        got = json.loads((sp.parent / f"result_{rid}.json").read_text())
        tab = next(x for x in got["tables"] if x["name"] == "One-sample t-test")["data"]
        assert tab["n"] == [15]
        for k, want in (("mean", m), ("t", t), ("ci_low", lo), ("ci_high", hi)):
            assert tab[k][0] == pytest.approx(want, rel=1e-12, abs=1e-12), (rid, k)
    assert m1 == pytest.approx(9.7) and m2 == pytest.approx(10.3) and t1 != pytest.approx(t2)
    # the stored (historical) table of result 1 equals the independent seed-1 oracle too
    t_tab = next(t for t in r1.tables if t.name == "One-sample t-test")
    assert t_tab.rows[0][t_tab.columns.index("mean")] == pytest.approx(m1, rel=1e-12)


def test_adversarial_round2_port_reused_folders_and_exact_cap(tmp_path):
    """Port of the auditor's Round 2 adversarial.py (baseline 13 pass / 4 fail) to the v1.3.6 interfaces: exports to a
    reused destination now land in NEW folders - the stale script / data of the earlier export are not part of the
    refusal export - and the exact-cap case is complete (assertions adapted to the unique-subfolder design)."""
    from statexplore.discovery import engine
    from statexplore.discovery import validate as V
    from statexplore.discovery.space import SearchSettings
    from statexplore.export.exporters import write_portable

    d = fx5("Test55L_LegacyUnderflow")
    run = engine.create_run(d, SearchSettings(**{k: v for k, v in S55L.items() if k not in ("workers", "max_models",
                                                                                           "max_minutes")}))
    v = {"grammar": "validation-4", "schema": 2, "validation_id": "audit1", "config": {"modes": ["holdout"],
         "pipeline": "fixed", "holdout_fraction": .2, "k": 5, "repeats": 1, "B": 1, "seed": 1}, "status": "complete",
         "problems": [], "search_id": run.id, "search_identity": run.fit_id, "search_signature": run.signature,
         "target_label": "rows", "target_estimand": "synthetic", "pipeline_label": "Fixed", "rule_label": "",
         "predictions": [{"split": "holdout", "fold": 1, "row_id": 1, "observed": 1., "predicted": 1.,
                          "participant": "1", "cluster": ""}],
         "execution": {"parts": [{"part": "holdout:fold1", "mode": "holdout", "split": "holdout", "fold": 1, "b": None,
                                  "chunk": 1, "status": "scored", "committed": True}]},
         "scripts": ["# chunk"], "reproduce_script": 'dat <- utils::read.csv("x.csv")\n# synthetic test\n'}
    a = tmp_path / "export"
    f1 = folder_of(V.export(v, str(a), ds=d, include_data=True))
    assert json.loads((f1 / "validation_manifest.json").read_text())["reproduction"]["runnable"] is True
    assert (f1 / "validation_reproduce_data.csv").exists()
    wrong = dict(v, validation_id="audit2", search_signature="different_signature")
    f2 = folder_of(V.export(wrong, str(a), ds=d, include_data=False))           # the SAME destination, reused
    assert f2 != f1
    assert json.loads((f2 / "validation_manifest.json").read_text())["reproduction"]["runnable"] is False
    assert not (f2 / "validation_reproduce.R").exists() and not list(f2.glob("*_data.csv"))
    assert (f1 / "validation_reproduce.R").exists()                             # the earlier bundle is untouched
    c = tmp_path / "consent"
    w1 = write_portable(v["reproduce_script"], c, d, include_data=True)
    w2 = write_portable(v["reproduce_script"], c, d, include_data=False)
    assert not list(folder_of(w2).glob("*_data.csv")) and list(folder_of(w1).glob("*_data.csv"))
    r = _runner(_ChunkSession(cost=1))
    r.budget["cap"] = 1
    assert r.execute()["status"] == "complete"                                   # exact_cap_complete_status
