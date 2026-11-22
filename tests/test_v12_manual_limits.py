"""v1.2 M §10: configurable complexity limits for manual models — one limit used by the form and the
backend, preflight counts (distinct predictors, terms, coefficients, rows, residual df, cells)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statexplore.analysis.builders import BY_KEY
from statexplore.analysis.complexity import preflight
from statexplore.analysis.framework import AnalysisInputError, validate

from test_discovery import make_ds


@pytest.fixture
def wide(tmp_path):
    rng = np.random.default_rng(2)
    n = 400
    df = pd.DataFrame(rng.normal(size=(n, 8)), columns=[f"x{i}" for i in range(1, 9)])
    df["g"] = rng.choice(["a", "b", "c", "d"], n)
    for f in ("f1", "f2", "f3", "f4"):
        df[f] = rng.choice(["p", "q"], n)
    df["y"] = df["x1"] + rng.normal(size=n)
    return make_ds(tmp_path, df)


def test_regression_default_limit_is_six_and_can_be_raised(wide):
    spec = BY_KEY["lm"]
    seven = {"dependent": ["y"], "predictors": [f"x{i}" for i in range(1, 8)]}
    opt = next(o for o in spec.options if o.key == "max_predictors")
    assert opt.default == 6 and next(s for s in spec.slots if s.key == "predictors").max == 30
    with pytest.raises(AnalysisInputError, match="complexity limit is 6"):
        validate(spec, wide, seven, {})
    assert validate(spec, wide, seven, {"max_predictors": 10})["max_predictors"] == 10
    for key in ("logistic", "ordinal_reg", "multinom_reg", "likert_sensitivity"):
        assert any(o.key == "max_predictors" and o.default == 6 for o in BY_KEY[key].options)


def test_preflight_counts_coefficients_after_dummy_coding(wide):
    pre = preflight("lm", wide, {"dependent": ["y"], "predictors": ["g", "x1", "x2"]}, {"interactions": "two_way"})
    # intercept + g (3) + x1 + x2 + g:x1 (3) + g:x2 (3) + x1:x2 = 13
    assert pre == {"predictors": 3, "terms": 6, "coefficients": 13, "n": 400, "residual_df": 387}


def test_factorial_anova_expands_to_five_factors_with_cell_check(wide):
    spec = BY_KEY["anova_factorial"]
    four = {"dependent": ["y"], "factors": ["f1", "f2", "f3", "f4"]}
    with pytest.raises(AnalysisInputError, match="complexity limit is 3"):
        validate(spec, wide, four, {})
    assert validate(spec, wide, four, {"max_factors": 4})
    pre = preflight("anova_factorial", wide, four, {})
    assert pre["cells"] == 16 and pre["empty_cells"] == 0 and pre["terms"] == 15
    sparse = {"dependent": ["y"], "factors": ["f1", "f2", "g"]}
    wide.raw.loc[(wide.raw["f1"] == "p") & (wide.raw["f2"] == "p") & (wide.raw["g"] == "a"), "y"] = np.nan
    pre = preflight("anova_factorial", wide, sparse, {})
    assert pre["cells"] == 16 and pre["empty_cells"] == 1              # shown before fitting


def test_ancova_default_two_factors_four_covariates(wide):
    spec = BY_KEY["ancova"]
    ok = {"dependent": ["y"], "factors": ["f1", "f2"], "covariates": ["x1", "x2", "x3", "x4"]}
    validate(spec, wide, ok, {})
    with pytest.raises(AnalysisInputError, match="complexity limit is 4"):
        validate(spec, wide, {**ok, "covariates": ["x1", "x2", "x3", "x4", "x5"]}, {})


def test_mixed_fixed_effect_limit(tmp_path):
    from statexplore.analysis.mixed import MixedInputError, MixedSpec, validate as mvalidate

    rng = np.random.default_rng(4)
    df = pd.DataFrame(rng.normal(size=(400, 6)), columns=[f"x{i}" for i in range(1, 7)])
    df["pid"], df["t"] = np.repeat(np.arange(100), 4), np.tile(np.arange(4), 100)
    df["y"] = rng.normal(size=400)
    ds = make_ds(tmp_path, df)
    s = MixedSpec(outcome="y", id="pid", time="t", predictors=[f"x{i}" for i in range(1, 7)])
    with pytest.raises(MixedInputError, match="complexity limit of 6"):       # 6 predictors + time = 7
        mvalidate(ds, s)
    s.max_fixed = 10
    mvalidate(ds, s)
