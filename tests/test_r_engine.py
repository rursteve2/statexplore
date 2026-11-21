"""R engine integration (skipped automatically when R or Phase 1 packages are missing)."""
from __future__ import annotations

import numpy as np
import pytest

from statexplore.analysis import eda
from statexplore.core.importers import import_file
from statexplore.core.schema import Status
from statexplore.rbridge.worker import REngineError, RScriptError

pytestmark = pytest.mark.r


def psych_reference(x: np.ndarray) -> dict[str, float]:
    """Independent implementation of psych::describe() (type 3 skew/kurtosis)."""
    x = x[~np.isnan(x)]
    n = len(x)
    m = x.mean()
    sd = x.std(ddof=1)
    dev = x - m
    m2 = np.mean(dev ** 2)
    skew = np.mean(dev ** 3) / m2 ** 1.5 * ((n - 1) / n) ** 1.5
    kurt = np.mean(dev ** 4) / m2 ** 2 * ((n - 1) / n) ** 2 - 3
    return {"n": n, "mean": m, "sd": sd, "median": np.median(x), "min": x.min(), "max": x.max(),
            "range": x.max() - x.min(), "skew": skew, "kurtosis": kurt, "se": sd / np.sqrt(n)}


def test_ping_and_versions(r_worker):
    assert r_worker.ping()
    info = r_worker.request("versions", {"packages": ["psych"]}).result
    assert info["r_version"].startswith("R version")
    assert info["packages"]["psych"]


def test_script_error_keeps_worker_alive(r_worker):
    with pytest.raises(RScriptError, match="boom"):
        r_worker.run('stop("boom")')
    assert r_worker.ping()


def test_warnings_are_captured(r_worker):
    resp = r_worker.run('warning("careful"); sx_result <- list(ok = TRUE)')
    assert resp.warnings == ["careful"] and resp.result == {"ok": True}


def test_missing_sx_result_is_error(r_worker):
    with pytest.raises(RScriptError, match="sx_result"):
        r_worker.run("x <- 1")


def test_timeout_kills_and_restarts(r_worker):
    with pytest.raises(REngineError):
        r_worker.run("Sys.sleep(30); sx_result <- 1", timeout=2)
    assert r_worker.ping()  # auto-restarted


def test_numeric_precision_round_trip(r_worker):
    resp = r_worker.run("sx_result <- list(x = 1/3, big = 1e300, tiny = 5e-324)")
    assert resp.result["x"] == 1 / 3 and resp.result["big"] == 1e300 and resp.result["tiny"] == 5e-324


def test_unicode_round_trip(r_worker):
    from statexplore.rbridge.codegen import r_str

    s = 'Zürich "quoted" \\ 🙂\nline'
    assert r_worker.run(f"sx_result <- list(s = {r_str(s)})").result["s"] == s


def test_describe_matches_reference(r_session, sav_labelled):
    ds = import_file(sav_labelled)
    res = eda.describe(r_session, ds, ["score", "income"])
    assert res.status is Status.OK, res.messages
    t = res.tables[0]
    for row_name, name in zip(t.row_names, ["score", "income"]):
        ref = psych_reference(ds.analysis_column(name).to_numpy(float))
        row = dict(zip(t.columns, t.rows[t.row_names.index(row_name)]))
        for k, v in ref.items():
            assert row[k] == pytest.approx(v, rel=1e-10, abs=1e-12), (name, k)
    # user-missing (-99) excluded from income
    assert dict(zip(t.columns, t.rows[1]))["min"] >= 10
    assert res.engine.engine == "R" and "psych" in res.engine.packages
    assert "psych::describe" in res.script and "read.csv" in res.script


def test_describe_by_group(r_session, sav_labelled):
    ds = import_file(sav_labelled)
    res = eda.describe(r_session, ds, ["score"], group="sex")
    t = res.tables[0]
    assert t.row_names == ["score | Male", "score | Female"]
    frame = ds.analysis_frame(["score", "sex"]).dropna()
    for i, code in enumerate([1.0, 2.0]):
        ref = psych_reference(frame.loc[frame["sex"] == code, "score"].to_numpy(float))
        row = dict(zip(t.columns, t.rows[i]))
        assert row["mean"] == pytest.approx(ref["mean"], rel=1e-10)
        assert row["n"] == ref["n"]


def test_factor_coding_and_reference_level(r_session, sav_labelled):
    ds = import_file(sav_labelled)
    ds.set_variable_field("sex", "reference_level", 2.0)
    r_session.ensure_data(ds)
    a, m = ds.var("sex").alias, ds.var("mood").alias
    out = r_session.worker.run(
        f"sx_result <- list(sex = levels(dat${a}), mood = levels(dat${m}), mood_na = sum(is.na(dat${m})))").result
    assert out["sex"] == ["Female", "Male"]
    assert out["mood"] == ["Low", "Medium", "High"]
    assert out["mood_na"] == ds.missing_count("mood")


@pytest.mark.parametrize("kind,vars_,group", [
    ("histogram", ["score"], "sex"), ("density", ["score"], None), ("boxplot", ["score"], "sex"),
    ("violin", ["score"], "sex"), ("scatter", ["score", "income"], None), ("bar", ["mood"], "sex"),
    ("pairs", ["score", "income", "id"], None), ("correlation", ["score", "income", "id"], None),
])
def test_plots_render(r_session, sav_labelled, kind, vars_, group):
    ds = import_file(sav_labelled)
    res, artifacts = eda.plot(r_session, ds, eda.PlotRequest(kind, vars_, group))
    assert res.status in (Status.OK, Status.WARNING), res.messages
    assert artifacts[res.plots[0]][:8] == b"\x89PNG\r\n\x1a\n"
    expected = (len(ds.analysis_frame(vars_ + ([group] if group else [])).dropna()))
    assert res.n_used == expected


def test_recorded_script_reproduces_result(r_worker, r_session, sav_labelled):
    """§18: the recorded script, run in a fresh environment, gives the same numbers."""
    ds = import_file(sav_labelled)
    res = eda.describe(r_session, ds, ["score"])
    fresh = r_worker.run("local({\n" + res.script + "\nsx_result\n}) -> sx_result")
    assert fresh.result["table"]["mean"] == res.tables[0].rows[0][1]


def test_hrs_describe(r_session, hrs_path):
    ds = import_file(hrs_path)
    names = ["R_CESDM", "R_BMI", "H_ITOT"]
    res = eda.describe(r_session, ds, names)
    assert res.status is Status.OK, res.messages
    for i, name in enumerate(names):
        ref = psych_reference(ds.analysis_column(name).to_numpy(float))
        row = dict(zip(res.tables[0].columns, res.tables[0].rows[i]))
        assert row["n"] == ref["n"]
        assert row["mean"] == pytest.approx(ref["mean"], rel=1e-10)
        assert row["skew"] == pytest.approx(ref["skew"], rel=1e-8)
