"""Generated R code: escaping, aliases, and data-load script structure (no R needed)."""
from __future__ import annotations

import re

import pandas as pd
import pytest

from statexplore.analysis import eda
from statexplore.core.dataset import Dataset, SourceInfo
from statexplore.core.importers import import_file
from statexplore.core.variables import Measure, Storage, VariableMeta
from statexplore.rbridge.codegen import alias, data_load_script, r_num, r_str, unique_labels

HOSTILE = 'x"); system("calc"); #\n`rm -rf`\\ é 🙂'


def test_r_str_escapes_everything():
    s = r_str(HOSTILE)
    assert s.isascii()
    assert "\n" not in s
    assert s.startswith('"') and s.endswith('"')
    body = s[1:-1]
    # every double quote inside the literal is escaped
    i = 0
    while (i := body.find('"', i)) != -1:
        assert body[i - 1] == "\\"
        i += 1
    assert "\\U{0001f642}" in s and "\\u00e9" in s


def test_r_num():
    assert r_num(float("nan")) == "NA_real_"
    assert r_num(float("inf")) == "Inf"
    assert r_num(0.1) == "0.1"
    assert float(r_num(1 / 3)) == 1 / 3  # full precision


def test_alias_validation():
    assert alias("v001") == "v001"
    for bad in ["x", "v1; q()", "v001)", "", "V001"]:
        with pytest.raises(ValueError):
            alias(bad)


def test_unique_labels():
    assert unique_labels(["a", "b", "a", "a"]) == ["a", "b", "a [1]", "a [2]"]


def _hostile_dataset() -> Dataset:
    raw = pd.DataFrame({HOSTILE: [1.0, 2.0, 1.0], "g": pd.array(["a", 'b"', None], dtype="string")})
    vs = [
        VariableMeta(name=HOSTILE, alias="v001", storage=Storage.NUMERIC, measure=Measure.NOMINAL,
                     value_labels={1.0: HOSTILE, 2.0: "two"}),
        VariableMeta(name="g", alias="v002", storage=Storage.STRING, measure=Measure.NOMINAL),
    ]
    return Dataset(raw, vs, SourceInfo(path="C:/data/evil\nfile.csv", format="csv"))


def test_load_script_contains_user_text_only_in_literals():
    ds = _hostile_dataset()
    script = data_load_script(ds, "C:/tmp/data.csv")
    assert script.isascii()
    for line in script.splitlines():
        # the hostile payload never appears unescaped
        assert 'system("calc")' not in line
    assert "dat$v001 <- factor(dat$v001, levels = c(1.0, 2.0)" in script
    assert "dat$v002 <- factor(dat$v002, levels = c(\"a\", \"b\\\"\")" in script


def test_load_script_reference_level(sav_labelled):
    ds = import_file(sav_labelled)
    ds.set_variable_field("sex", "reference_level", 2.0)
    script = data_load_script(ds, "C:/tmp/data.csv")
    a = ds.var("sex").alias
    assert f'dat${a} <- factor(dat${a}, levels = c(1.0, 2.0), labels = c("Male", "Female"))' in script
    assert f'dat${a} <- stats::relevel(dat${a}, ref = "Female")' in script
    # user-missing code 9 is not a factor level
    m = ds.var("mood").alias
    assert f'dat${m} <- factor(dat${m}, levels = c(1.0, 2.0, 3.0), labels = c("Low", "Medium", "High"))' in script


def test_plot_validation(sav_labelled):
    ds = import_file(sav_labelled)
    with pytest.raises(eda.AnalysisInputError):
        eda._validate_plot(ds, eda.PlotRequest("histogram", ["sex"]))
    with pytest.raises(eda.AnalysisInputError):
        eda._validate_plot(ds, eda.PlotRequest("scatter", ["score"]))
    with pytest.raises(eda.AnalysisInputError):
        eda._validate_plot(ds, eda.PlotRequest("violin", ["score"]))
    with pytest.raises(eda.AnalysisInputError):
        eda._validate_plot(ds, eda.PlotRequest("histogram", ["score"], group="income"))
    eda._validate_plot(ds, eda.PlotRequest("boxplot", ["score"], group="sex"))


@pytest.mark.parametrize("kind,vars_,group", [
    ("histogram", ["score"], "sex"), ("density", ["score"], None), ("boxplot", ["score"], None),
    ("violin", ["score"], "sex"), ("scatter", ["score", "income"], "sex"), ("bar", ["mood"], "sex"),
    ("pairs", ["score", "income", "id"], None), ("correlation", ["score", "income", "id"], None),
])
def test_plot_bodies_are_ascii_and_parse_shape(sav_labelled, kind, vars_, group):
    ds = import_file(sav_labelled)
    body, packages = eda.plot_body(ds, eda.PlotRequest(kind, vars_, group, title=HOSTILE), "C:/t/p.png")
    assert body.isascii()
    assert "sx_result <- list(plot = sx_plot_file, n = sx_n)" in body
    assert packages in (["ggplot2"], ["psych"])
    code = re.sub(r'"(?:\\.|[^"\\])*"', '""', body)  # drop string literals
    assert code.count("(") == code.count(")")
    assert "calc" not in code
