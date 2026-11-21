"""Phase 1 acceptance: metadata edits are logged and reversible; projects reopen identically."""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from statexplore.core.dataset import DatasetError
from statexplore.core.importers import file_sha256, import_file
from statexplore.core.overview import summarize, suggest_plots, tukey_outliers
from statexplore.core.project import Project, ProjectError
from statexplore.core.schema import AnalysisResult, EngineInfo, Status, Table
from statexplore.core.variables import Measure, Role


def test_edits_logged_and_undoable(sav_labelled):
    ds = import_file(sav_labelled)
    raw_before = ds.raw.copy()
    ds.set_variable_field("sex", "display_name", "Gender")
    ds.set_variable_field("sex", "reference_level", 2.0)
    ds.set_variable_field("id", "role", "id")
    ds.set_variable_field("mood", "measure", "nominal")
    assert [c.field for c in ds.changes] == ["display_name", "reference_level", "role", "measure"]
    assert ds.var("id").role is Role.ID
    ds.undo()
    assert ds.var("mood").measure is Measure.ORDINAL
    ds.undo(); ds.undo(); ds.undo()
    assert ds.var("sex").display_name == "sex" and ds.var("sex").reference_level is None
    assert ds.undo() is None
    pd.testing.assert_frame_equal(ds.raw, raw_before)  # data never touched


def test_invalid_edits_rejected(sav_labelled):
    ds = import_file(sav_labelled)
    with pytest.raises(DatasetError):
        ds.set_variable_field("sex", "reference_level", 7.0)
    with pytest.raises(DatasetError):
        ds.set_variable_field("sex", "display_name", "mood")  # clashes
    with pytest.raises(DatasetError):
        ds.set_variable_field("sex", "label", "x")  # not editable
    with pytest.raises(ValueError):
        ds.set_variable_field("sex", "measure", "interval")
    assert ds.changes == []


def test_project_round_trip(sav_labelled, tmp_path):
    ds = import_file(sav_labelled)
    ds.set_variable_field("sex", "display_name", "Gender")
    ds.set_variable_field("sex", "reference_level", 2.0)
    proj = Project(ds)
    res = AnalysisResult(kind="describe", title="t", engine=EngineInfo("R", "R 4.6", {"psych": "2.5"}),
                         script="sx_result <- 1", inputs={"variables": ["score"]}, status=Status.WARNING,
                         tables=[Table("describe", ["n"], [[58.0]], ["score"])], plots=["plots/a.png"])
    proj.add_result(res, {"plots/a.png": b"\x89PNG fake"})
    out = tmp_path / "p.sxp"
    proj.save(out)

    again = Project.open(out)
    pd.testing.assert_frame_equal(again.dataset.raw, ds.raw)
    assert [v.to_dict() for v in again.dataset.variables.values()] == [v.to_dict() for v in ds.variables.values()]
    assert [c.to_dict() for c in again.dataset.changes] == [c.to_dict() for c in ds.changes]
    assert again.dataset.source == ds.source
    assert again.results[0].to_dict() == res.to_dict()
    assert again.artifacts == {"plots/a.png": b"\x89PNG fake"}
    # user-missing semantics survive the round trip
    assert again.dataset.missing_count("mood") == ds.missing_count("mood")
    # undo still works after reopening
    again.dataset.undo()
    assert again.dataset.var("sex").reference_level is None


def test_save_does_not_touch_source_and_open_is_read_only(sav_labelled, tmp_path):
    h = file_sha256(sav_labelled)
    proj = Project(import_file(sav_labelled))
    out = tmp_path / "p.sxp"
    proj.save(out)
    proj.save(out)  # overwrite atomically
    h_proj = file_sha256(out)
    Project.open(out)
    assert file_sha256(sav_labelled) == h
    assert file_sha256(out) == h_proj


def test_open_rejects_bad_files(tmp_path):
    bad = tmp_path / "bad.sxp"
    bad.write_bytes(b"not sqlite at all")
    with pytest.raises(ProjectError):
        Project.open(bad)
    other = tmp_path / "other.sxp"
    con = sqlite3.connect(other)
    con.execute("CREATE TABLE meta (key TEXT, value TEXT)")
    con.execute("INSERT INTO meta VALUES ('format', '\"something-else\"')")
    con.commit(); con.close()
    with pytest.raises(ProjectError):
        Project.open(other)


def test_hrs_project_round_trip(hrs_path, tmp_path):
    ds = import_file(hrs_path)
    out = tmp_path / "hrs.sxp"
    Project(ds).save(out)
    again = Project.open(out)
    pd.testing.assert_frame_equal(again.dataset.raw, ds.raw)


def test_overview_numbers(sav_labelled):
    ds = import_file(sav_labelled)
    s = summarize(ds)
    score = next(v for v in s.variables if v.name == "score")
    col = ds.raw["score"].dropna()
    assert score.n_valid == 58 and score.n_missing == 2
    assert score.mean == pytest.approx(col.mean())
    assert score.sd == pytest.approx(col.std(ddof=1))
    assert s.n_rows == 60 and s.n_cols == 5
    mood = next(v for v in s.variables if v.name == "mood")
    assert mood.n_user_missing_codes == mood.n_missing
    assert mood.n_outliers is None  # not computed for categorical


def test_tukey_and_suggestions():
    assert tukey_outliers(pd.Series([1, 2, 3, 4, 5, 100]).to_numpy(float)) == 1
    assert suggest_plots(["continuous", "continuous"]) == ["scatter"]
    assert suggest_plots(["continuous", "categorical"]) == ["boxplot", "violin"]
    assert suggest_plots(["continuous"] * 4) == ["pairs", "correlation"]


def test_derived_filters_roundtrip_and_undo(sav_labelled, tmp_path):
    from statexplore.core.derive import DerivedSpec, FilterSpec

    ds = import_file(sav_labelled)
    raw_before = ds.raw.copy()
    m = ds.add_derived(DerivedSpec("avg", "mean_items", ["score", "income"], {"min_valid": 1}))
    ds.add_derived(DerivedSpec("hi", "binarize", ["avg"], {"cutoff": 30}), measure=Measure.NOMINAL)
    ds.add_filter(FilterSpec("sex", "in", [1.0]))
    assert m.alias == "v006" and ds.var("hi").alias == "v007"
    n_kept = ds.n_rows_filtered
    assert n_kept == int((ds.raw["sex"] == 1).sum())
    # dependency protection
    with pytest.raises(DatasetError):
        ds.remove_derived("avg")
    out = tmp_path / "d.sxp"
    Project(ds).save(out)
    again = Project.open(out).dataset
    pd.testing.assert_series_equal(again.column("avg"), ds.column("avg"))
    assert list(again.variables) == list(ds.variables)
    assert again.n_rows_filtered == n_kept
    pd.testing.assert_frame_equal(again.raw, raw_before)  # source data untouched
    # undo in reverse order restores everything
    again.undo()
    assert again.filters == []
    again.undo()
    assert "hi" not in again.variables
    again.remove_derived("avg")
    again.undo()  # undo the removal
    assert "avg" in again.variables


def test_derived_validation(sav_labelled):
    from statexplore.core.derive import DerivedSpec, FilterSpec

    ds = import_file(sav_labelled)
    bad = [
        DerivedSpec("x", "mean_items", ["score"], {"min_valid": 1}),           # needs 2+ items
        DerivedSpec("x", "mean_items", ["score", "id"], {"min_valid": 3}),     # min_valid too big
        DerivedSpec("x", "log", ["score", "id"]),                             # single source
        DerivedSpec("score", "log", ["id"]),                                  # name clash
        DerivedSpec("x", "recode", ["mood"], {"mapping": [[1, 2], [1, 3]]}),  # duplicate old value
        DerivedSpec("x", "nope", ["score"]),
    ]
    for spec in bad:
        with pytest.raises(DatasetError):
            ds.add_derived(spec)
    with pytest.raises(DatasetError):
        ds.add_filter(FilterSpec("score", ">", 1e9))  # would remove every row
    assert ds.changes == [] and ds.derived == {}
