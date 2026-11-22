"""v1.2 Phase 1 (M §3, §4, §15 A): numeric sorting with missing values last, significance
highlighting (adjusted / raw / strict same-estimand CI), warnings kept separate, and table state
(sort, search, selection) preserved across refreshes."""
from __future__ import annotations

import os
from decimal import Decimal

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402

from statexplore.discovery import engine  # noqa: E402
from statexplore.discovery.engine import DiscoveryRun  # noqa: E402
from statexplore.discovery.space import SearchSettings  # noqa: E402
from statexplore.gui.models import FrameModel, NumericSortProxy, sort_value  # noqa: E402

P_VALUES = [0.105, None, 0.000001, 0.8, 0.012, float("nan"), 0.049, 0.0008]
SORTED = [0.000001, 0.0008, 0.012, 0.049, 0.105, 0.8]


def proxy_for(frame: pd.DataFrame):
    model = FrameModel(p_columns=("p",))
    model.set_frame(frame)
    proxy = NumericSortProxy()
    proxy.setSourceModel(model)
    return model, proxy


def column(proxy, col=0):
    return [proxy.index(r, col).data(Qt.UserRole) for r in range(proxy.rowCount())]


def finite(values):
    return [v for v in values if v is not None]


@pytest.mark.parametrize("make", [
    lambda v: pd.Series(v, dtype=object),                                         # Python floats + None
    lambda v: pd.Series([np.float64(x) if x is not None else np.nan for x in v]),  # numpy float64
    lambda v: pd.Series([x if x is None or x == x else None for x in v], dtype="Float64"),  # nullable
    lambda v: pd.Series([None if x is None or x != x else f"{x:.6e}" for x in v], dtype=object),  # text
    lambda v: pd.Series([None if x is None or x != x else Decimal(repr(x)) for x in v], dtype=object),
])
def test_p_values_sort_numerically_missing_last_both_directions(qapp, make):
    model, proxy = proxy_for(pd.DataFrame({"p": make(P_VALUES)}))
    proxy.sort(0, Qt.AscendingOrder)
    asc = column(proxy)
    assert [float(x) for x in asc[:6]] == pytest.approx(SORTED)
    assert all(sort_value(x) is None for x in asc[6:])
    proxy.sort(0, Qt.DescendingOrder)
    desc = column(proxy)
    assert [float(x) for x in desc[:6]] == pytest.approx(SORTED[::-1])
    assert all(sort_value(x) is None for x in desc[6:])


def test_display_never_rounds_the_sort_key(qapp):
    model, proxy = proxy_for(pd.DataFrame({"p": [0.00004, 0.00002, 0.00003]}))
    proxy.sort(0, Qt.AscendingOrder)
    assert column(proxy) == [0.00002, 0.00003, 0.00004]
    assert proxy.index(0, 0).data(Qt.DisplayRole) == "2.00e-05"     # not "0.0000"


def test_signed_values_ties_and_text(qapp):
    frame = pd.DataFrame({"ci_low": [-0.5, 0.2, -1.5, 0.2, np.inf, -0.1],
                          "name": ["b", "A", "c", "a", "z", "B"],
                          "id": [1, 2, 3, 4, 5, 6]})
    model, proxy = proxy_for(frame)
    proxy.sort(0, Qt.AscendingOrder)
    assert column(proxy, 0) == [-1.5, -0.5, -0.1, 0.2, 0.2, None]      # inf is non-finite -> last
    assert column(proxy, 2)[3:5] == [2, 4]                              # stable for ties
    proxy.sort(1, Qt.AscendingOrder)
    assert [x.casefold() for x in column(proxy, 1)] == ["a", "a", "b", "b", "c", "z"]
    assert list(frame["ci_low"]) == [-0.5, 0.2, -1.5, 0.2, np.inf, -0.1]  # source data untouched


def test_numbers_before_text_and_filter(qapp):
    model, proxy = proxy_for(pd.DataFrame({"v": ["n/a", 3, 1, None], "flag": ["x", "y", "Y", "x"]}))
    proxy.sort(0, Qt.AscendingOrder)
    assert column(proxy) == [1, 3, "n/a", None]
    proxy.setFilterFixedString("y")                                     # case-insensitive filter
    assert sorted(column(proxy)) == [1, 3]


# ---- highlighting rules -------------------------------------------------------------------

def hl_frame(**cols):
    n = len(next(iter(cols.values())))
    base = {"in_family": [True] * n, "status": ["ok"] * n, "flags": [""] * n,
            "p": [0.01] * n, "p_adj": [0.01] * n, "ci_low": [np.nan] * n, "ci_high": [np.nan] * n,
            "ci_null": [0.0] * n, "ci_applicable": ["same_estimand"] * n}
    base.update(cols)
    return pd.DataFrame(base)


def test_alpha_boundary_and_adjusted_default():
    df = hl_frame(p=[0.05, 0.049, 0.01, 0.2], p_adj=[0.05, 0.049, 0.08, 0.3])
    assert list(engine.highlight_mask(df)) == [False, True, False, False]          # p = .05 is not < .05
    assert list(engine.highlight_mask(df, "raw")) == [False, True, True, False]    # raw-only not in default
    assert not engine.highlight_mask(df, "off").any()


def test_strict_mode_needs_same_estimand_ci_excluding_null():
    df = hl_frame(p_adj=[0.01] * 7,
                  ci_low=[-0.5, -0.1, 1.2, 0.8, 1.2, np.nan, 0.0],
                  ci_high=[-0.1, 0.2, 2.0, 1.3, 2.0, np.nan, 0.4],
                  ci_null=[0, 0, 1, 1, 1, 0, 0],
                  ci_applicable=["same_estimand"] * 4 + ["joint", "same_estimand", "same_estimand"])
    # beta [-0.5,-0.1] passes; [-0.1,0.2] fails; OR [1.2,2.0] passes; OR [0.8,1.3] fails;
    # a joint multi-df test cannot pass; a missing CI cannot pass; a CI touching the null fails
    assert list(engine.highlight_mask(df, "strict")) == [True, False, True, False, False, False, False]


def test_warnings_are_independent_of_significance():
    df = hl_frame(p_adj=[0.01, 0.5], status=["warning", "warning"], flags=["singular fit", "separation"])
    assert list(engine.highlight_mask(df)) == [True, False]
    assert list(engine.warning_mask(df)) == [True, True]


def test_untested_rows_never_highlight():
    df = hl_frame(in_family=[False, False], status=["invalid", "ok"], p=[np.nan, 0.001], p_adj=[np.nan, 0.001])
    assert not engine.highlight_mask(df, "raw").any()


# ---- legacy rows become typed hypotheses ---------------------------------------------------

def legacy_run(family="gaussian", rows=None, status="complete"):
    s = SearchSettings(outcome="y", candidates=["a", "b", "c"], family=family)
    run = DiscoveryRun(settings=s, family=family, total=len(rows), signature="x", status=status)
    run.rows = rows
    return run


def legacy_row(i, p, **kw):
    r = {"idx": i, "key": f"k{i}", "focal": ["a", "b"], "covariates": [], "kind": "focal", "order": 2,
         "interaction": "a × b", "formula": "y ~ a * b", "cached": False, "status": "ok", "reason": "",
         "flags": "", "speculative": False, "n": 100.0, "n_params": 4.0, "statistic": 5.0, "df1": 1.0,
         "df2": 96.0, "p": p, "effect": 0.02, "partial_eta2": 0.03, "estimate": 0.4, "se": 0.1,
         "ci_low": 0.2, "ci_high": 0.6, "r2": 0.1, "aic": 10.0, "bic": 12.0}
    r.update(kw)
    return r


def test_legacy_rows_get_one_focal_test_with_typed_ci():
    run = legacy_run(rows=[legacy_row(0, 0.001), legacy_row(1, 0.3, ci_low=None, ci_high=None, df1=2.0),
                           legacy_row(2, None, status="invalid", reason="empty cell")])
    tf = engine.tests_frame(run)
    assert list(tf["test_type"]) == ["focal_interaction"] * 2
    assert list(tf["ci_applicable"]) == ["same_estimand", "joint"]
    assert list(tf["df_den"]) == [96.0, 96.0]
    mf = engine.results_frame(run)
    assert len(mf) == 3 and mf.loc[2, "interpretation"].startswith("Not tested")


def test_legacy_mixed_rows_report_groups_not_df():
    run = legacy_run("lmm", rows=[legacy_row(0, 0.01, df2=250.0)])
    run.settings.participant = "id"
    tf = engine.tests_frame(run)
    assert tf.loc[0, "n_groups"] == 250 and np.isnan(tf.loc[0, "df_den"]) and tf.loc[0, "lrt_df"] == 1


def test_partial_search_scope_is_labelled_provisional():
    tf = engine.tests_frame(legacy_run(rows=[legacy_row(0, 0.01)], status="partial"))
    assert tf.loc[0, "search_scope"].startswith("partial")


# ---- the Discovery tables keep their state -------------------------------------------------

def test_result_table_keeps_sort_search_and_selection(qapp):
    from statexplore.gui.pages.discovery import DISCOVERY_COLS, ResultTable

    rows = [legacy_row(i, p) for i, p in enumerate([0.3, 0.001, 0.04, None, 0.0002])]
    rows[3]["status"] = "invalid"
    run = legacy_run(rows=rows)
    tf = engine.add_display_columns(engine.tests_frame(run), run)
    table = ResultTable(DISCOVERY_COLS, id_col="Test ID")
    table.set_frame(tf)
    pcol = list(table.model.frame.columns).index("p (adjusted)")
    table.view.sortByColumn(pcol, Qt.DescendingOrder)
    table.select_id("2.0")
    table.search.setText("a")
    table.set_frame(tf)                                   # live refresh
    assert table.proxy.sortColumn() == pcol and table.proxy.sortOrder() == Qt.DescendingOrder
    assert table.selected_id() == "2.0"
    assert table.search.text() == "a"
    got = [table.proxy.index(r, pcol).data(Qt.UserRole) for r in range(table.proxy.rowCount())]
    assert got == sorted(got, reverse=True)
    hl = engine.highlight_mask(tf).to_numpy()
    table.set_status(hl)
    table.set_highlight_only(True)
    assert table.counts() == (int(hl.sum()), int(hl.sum()), len(tf))
