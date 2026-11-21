"""Phase 1 acceptance: import supported formats, preserve SPSS metadata, handle missing values."""
from __future__ import annotations

import numpy as np
import pytest

from statexplore.core.importers import CsvOptions, ImportError_, file_sha256, import_file, list_sheets
from statexplore.core.variables import Measure, Storage


def test_csv_sniffs_delimiter_and_encoding(csv_semicolon_cp1252):
    ds = import_file(csv_semicolon_cp1252)
    assert ds.source.options == {"delimiter": ";", "encoding": "cp1252"}
    assert list(ds.raw.columns) == ["id", "city", "score", "group"]
    assert ds.raw["city"].tolist()[:2] == ["Montréal", "Zürich"]
    assert ds.var("score").storage is Storage.NUMERIC
    assert ds.var("city").storage is Storage.STRING
    assert ds.missing_count("score") == 1
    assert ds.missing_count("city") == 1
    assert ds.missing_count("group") == 1  # literal "NA" is missing


def test_csv_explicit_options(csv_semicolon_cp1252):
    ds = import_file(csv_semicolon_cp1252, csv_options=CsvOptions(delimiter=";", encoding="cp1252"))
    assert ds.n_rows == 4 and ds.n_cols == 4


def test_csv_wrong_encoding_reports_error(csv_semicolon_cp1252):
    with pytest.raises(ImportError_):
        import_file(csv_semicolon_cp1252, csv_options=CsvOptions(encoding="utf-8"))


def test_csv_duplicate_and_blank_headers(tmp_path):
    p = tmp_path / "dup.csv"
    p.write_text("a,a,\n1,2,3\n", encoding="utf-8")
    ds = import_file(p)
    assert len(set(ds.raw.columns)) == 3


def test_xlsx_sheet_selection(xlsx_two_sheets):
    assert list_sheets(xlsx_two_sheets) == ["First", "Second"]
    first = import_file(xlsx_two_sheets)
    assert list(first.raw.columns) == ["a"]
    second = import_file(xlsx_two_sheets, sheet="Second")
    assert second.source.options["sheet"] == "Second"
    assert second.missing_count("x") == 1 and second.missing_count("label") == 1
    with pytest.raises(ImportError_):
        import_file(xlsx_two_sheets, sheet="Nope")


def test_sav_preserves_labels_measures_and_codes(sav_labelled):
    ds = import_file(sav_labelled)
    sex = ds.var("sex")
    assert sex.label == "Sex of respondent"
    assert sex.value_labels == {1.0: "Male", 2.0: "Female"}
    assert sex.measure is Measure.NOMINAL
    assert ds.var("mood").measure is Measure.ORDINAL
    assert ds.var("score").measure is Measure.CONTINUOUS
    # underlying numeric codes are kept (labels are display-only)
    assert set(ds.raw["sex"].dropna().unique()) <= {1.0, 2.0}


def test_sav_user_missing_preserved_but_excluded(sav_labelled):
    ds = import_file(sav_labelled)
    mood, income = ds.var("mood"), ds.var("income")
    assert mood.missing_values == [9.0]
    assert income.missing_ranges == [(-99.0, -1.0)]
    # raw data still holds the codes...
    n9 = int((ds.raw["mood"] == 9).sum())
    assert n9 > 0
    # ...but the analysis view treats them as missing
    assert ds.missing_count("mood") == n9
    assert ds.missing_count("income") == int((ds.raw["income"] == -99).sum())
    assert 9.0 not in ds.levels("mood")
    # system-missing values stay missing
    assert ds.missing_count("score") == 2


def test_unsupported_extension(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("hi")
    with pytest.raises(ImportError_):
        import_file(p)


def test_import_never_modifies_source(sav_labelled, csv_semicolon_cp1252, xlsx_two_sheets):
    for p in (sav_labelled, csv_semicolon_cp1252, xlsx_two_sheets):
        before = (file_sha256(p), p.stat().st_mtime_ns)
        ds = import_file(p)
        assert ds.source.sha256 == before[0]
        assert (file_sha256(p), p.stat().st_mtime_ns) == before


def test_hrs_dataset(hrs_path):
    import pyreadstat

    ds = import_file(hrs_path)
    assert (ds.n_rows, ds.n_cols) == (16000, 168)
    _, meta = pyreadstat.read_sav(str(hrs_path), metadataonly=True)
    labelled = [v for v in ds.variables.values() if v.value_labels]
    assert len(labelled) == len(meta.variable_value_labels) == 82
    assert ds.var("RAGENDER").label == "Gender"
    assert ds.var("RAGENDER").measure is Measure.NOMINAL
    assert ds.var("WAVE").measure is Measure.ORDINAL
    # missing cells match a direct pyreadstat read
    direct, _ = pyreadstat.read_sav(str(hrs_path))
    assert int(ds.analysis_frame(list(ds.variables)).isna().sum().sum()) == int(direct.isna().sum().sum())
    np.testing.assert_allclose(ds.raw["R_CESDM"].mean(), direct["R_CESDM"].mean())
