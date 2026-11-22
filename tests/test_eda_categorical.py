"""v1.1 Phase 2: variable selector (§20 E), categorical EDA (§20 F), ordinal correlations (§8.5)."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import scipy.stats as st  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402

from statexplore.analysis import eda  # noqa: E402
from statexplore.analysis import measurement as M  # noqa: E402
from statexplore.analysis.builders import BY_KEY  # noqa: E402
from statexplore.analysis.framework import AnalysisInputError, run_analysis  # noqa: E402
from statexplore.core.importers import import_sav  # noqa: E402


@pytest.fixture(scope="module")
def hrs():
    from tests.conftest import HRS_PATH

    if not HRS_PATH.is_file():
        pytest.skip("HRS test dataset not found")
    return import_sav(HRS_PATH)


def table(res, prefix):
    t = next(t for t in res.tables if t.name.startswith(prefix))
    return pd.DataFrame(t.rows, columns=t.columns, index=t.row_names or None)


# ---- E. variable selector ---------------------------------------------------------------------

@pytest.fixture
def selector(qtbot, hrs):
    from statexplore.gui.widgets.variable_selector import VariableSelector

    w = VariableSelector()
    qtbot.addWidget(w)
    w.set_variables(hrs, ["R_SHLT", "R_LTACTX", "R_VGACTX", "R_BMI", "RAGENDER", "R_HSWRKF"])
    return w


def visible(w):
    return [it.data(Qt.UserRole) for it in w.visible_items()]


def test_E_check_and_clear_visible_only(selector):
    selector.set_filter("activity")          # label search: two activity-frequency variables
    assert visible(selector) == ["R_LTACTX", "R_VGACTX"]
    selector.check_visible()
    assert selector.selected() == ["R_LTACTX", "R_VGACTX"]
    selector.set_filter("")
    selector.set_selected(["R_BMI", "R_LTACTX", "R_VGACTX"])
    selector.set_filter("light")
    selector.clear_visible()
    assert selector.selected() == ["R_BMI", "R_VGACTX"]


def test_E_search_never_changes_selection_and_count_includes_hidden(selector):
    selector.set_selected(["R_SHLT", "R_BMI"])
    selector.set_filter("gender")
    assert selector.selected() == ["R_SHLT", "R_BMI"]
    assert selector.count.text() == "2 of 6 selected (2 hidden by the search)"
    selector.clear_all()                     # clears hidden selections too
    assert selector.selected() == []
    selector.set_filter("")
    assert selector.count.text() == "0 of 6 selected"
    selector.check_all()
    assert len(selector.selected()) == 6


def test_E_check_order_and_level_filter(selector):
    selector.list.item(3).setCheckState(Qt.Checked)      # R_BMI first
    selector.list.item(0).setCheckState(Qt.Checked)      # then R_SHLT
    assert selector.selected() == ["R_BMI", "R_SHLT"]
    selector.level.setCurrentIndex(selector.level.findData("continuous"))
    assert visible(selector) == ["R_BMI"]
    selector.set_filter("R_HSWRKF")                       # original name search still respects the level
    assert visible(selector) == []


def test_E_single_mode(qtbot, hrs):
    from statexplore.gui.widgets.variable_selector import VariableSelector

    w = VariableSelector(single=True)
    qtbot.addWidget(w)
    w.set_variables(hrs, ["R_SHLT", "R_BMI"])
    w.list.item(0).setCheckState(Qt.Checked)
    w.list.item(1).setCheckState(Qt.Checked)
    assert w.selected() == ["R_BMI"]


# ---- F. categorical EDA -------------------------------------------------------------------------

@pytest.mark.r
def test_F_frequencies_match_pandas(hrs, r_session):
    res = eda.frequencies(r_session, hrs, ["R_LTACTX", "RAGENDER"])
    col = hrs.analysis_column("R_LTACTX")
    f = table(res, "Frequencies: R_LTACTX")
    body = f.iloc[:5]
    counts = col.value_counts()
    assert list(body["category"]) == ["1.Every day", "2.>1 per week", "3.1 per week", "4.l-3 per mon", "5.Never"]
    assert list(body["code"]) == ["1", "2", "3", "4", "5"]
    assert list(body["n"]) == [counts[c] for c in (1.0, 2.0, 3.0, 4.0, 5.0)]
    n, valid = len(col), col.notna().sum()
    assert body["percent"].tolist() == pytest.approx([100 * counts[c] / n for c in (1.0, 2.0, 3.0, 4.0, 5.0)], rel=1e-12)
    assert body["valid_percent"].tolist() == pytest.approx([100 * counts[c] / valid for c in (1.0, 2.0, 3.0, 4.0, 5.0)])
    miss = f[f["category"] == "Missing"].iloc[0]
    assert miss["n"] == col.isna().sum() and miss["percent"] == pytest.approx(100 * col.isna().mean())
    s = table(res, "Categorical summary").set_index("variable")
    assert s.loc["R_LTACTX", "n_missing"] == col.isna().sum() and s.loc["R_LTACTX", "categories"] == 5
    assert s.loc["R_LTACTX", "mode"] == hrs.level_label("R_LTACTX", counts.idxmax())
    assert "mean" not in s.columns and "mean" not in f.columns     # no means for categorical codes
    assert "cumulative_valid_percent" not in f.columns              # project level nominal: not ordered


@pytest.mark.r
def test_F_ordered_categories_stay_ordered(hrs_path, r_session):
    ds = import_sav(hrs_path)
    M.approve(ds, "R_SHLT", "ordinal", category_order=[5.0, 4.0, 3.0, 2.0, 1.0])
    ds.set_variable_field("R_SHLT", "reference_level", 3.0)    # modelling reference must not reorder tables
    res = eda.frequencies(r_session, ds, ["R_SHLT"])
    f = table(res, "Frequencies")
    assert list(f["category"][:5]) == ["5.Poor", "4.Fair", "3.Good", "2.Very good", "1.Excellent"]
    col = ds.analysis_column("R_SHLT")
    valid = col.dropna()
    cum = np.cumsum([100 * (valid == c).mean() for c in (5.0, 4.0, 3.0, 2.0, 1.0)])
    assert f["cumulative_valid_percent"][:5].tolist() == pytest.approx(cum.tolist())
    s = table(res, "Categorical summary")
    order = [5.0, 4.0, 3.0, 2.0, 1.0]
    med = next(c for c, cp in zip(order, cum) if cp >= 50)
    assert s["median_category"][0] == ds.level_label("R_SHLT", med)


@pytest.mark.r
@pytest.mark.parametrize("kind", ["bar", "bar_percent", "bar_horizontal", "ordinal_dist"])
def test_F_plots_match_frequency_tables(hrs_path, r_session, kind):
    ds = import_sav(hrs_path)
    M.approve(ds, "R_LTACTX", "ordinal")
    res, art = eda.plot(r_session, ds, eda.PlotRequest(kind, ["R_LTACTX"]))
    assert res.status.value in ("ok", "warning") and res.plots and art[res.plots[0]][:4] == b"\x89PNG"
    plotted = table(res, "Plotted counts")
    counts = ds.analysis_column("R_LTACTX").value_counts()
    assert list(plotted["n"]) == [counts[c] for c in (1.0, 2.0, 3.0, 4.0, 5.0)]
    assert plotted["percent"].sum() == pytest.approx(100)


@pytest.mark.r
@pytest.mark.parametrize("kind", ["bar_clustered", "bar_stacked", "bar_stacked100", "mosaic", "ordinal_dist"])
def test_F_grouped_plots(hrs_path, r_session, kind):
    ds = import_sav(hrs_path)
    M.approve(ds, "R_SHLT", "ordinal")
    res, art = eda.plot(r_session, ds, eda.PlotRequest(kind, ["R_SHLT"], group="RAGENDER"))
    assert res.status.value in ("ok", "warning"), res.messages
    assert art[res.plots[0]][:4] == b"\x89PNG"
    p = table(res, "Plotted counts")
    f = ds.analysis_frame(["R_SHLT", "RAGENDER"]).dropna()
    ref = f.groupby(["RAGENDER", "R_SHLT"]).size()
    got = {(r["g"], r["x"]): r["n"] for _, r in p.iterrows()}
    for (g, x), n in ref.items():
        assert got[(ds.level_label("RAGENDER", g), ds.level_label("R_SHLT", x))] == n
    for g, part in p.groupby("g"):
        assert part["percent"].sum() == pytest.approx(100)


def test_F_plot_validation(hrs):
    with pytest.raises(eda.AnalysisInputError, match="grouping"):
        eda._validate_plot(hrs, eda.PlotRequest("mosaic", ["R_SHLT"]))
    with pytest.raises(eda.AnalysisInputError, match="not ordinal"):
        eda._validate_plot(hrs, eda.PlotRequest("ordinal_dist", ["R_SHLT"]))
    with pytest.raises(eda.AnalysisInputError, match="not categorical"):
        eda.frequencies(None, hrs, ["R_BMI"])


@pytest.mark.r
def test_F_crosstab_matches_scipy(hrs, r_session):
    res = eda.crosstab(r_session, hrs, "R_SHLT", "RAGENDER")
    f = hrs.analysis_frame(["R_SHLT", "RAGENDER"]).dropna()
    obs = pd.crosstab(f["R_SHLT"], f["RAGENDER"])
    o = table(res, "Observed counts")
    assert o.iloc[:5, :2].to_numpy().tolist() == obs.to_numpy().tolist()
    assert o.loc["Total", "Total"] == len(f)
    chi2, p, dof, expected = st.chi2_contingency(obs.to_numpy(), correction=False)
    e = table(res, "Expected counts")
    assert e.to_numpy() == pytest.approx(expected, rel=1e-10)
    t = table(res, "Chi-square").iloc[0]
    assert t["statistic"] == pytest.approx(chi2, rel=1e-10) and t["p"] == pytest.approx(p, rel=1e-8)
    assert t["df"] == dof
    assert t["cramers_v"] == pytest.approx(np.sqrt(chi2 / (len(f) * (min(obs.shape) - 1))), rel=1e-10)
    rp = table(res, "Row percentages")
    assert rp.iloc[:5, :2].to_numpy() == pytest.approx((100 * obs.div(obs.sum(axis=1), axis=0)).to_numpy())
    cp = table(res, "Column percentages")
    assert cp.iloc[:5, :2].to_numpy() == pytest.approx((100 * obs / obs.sum(axis=0)).to_numpy())
    assert "adequate" in table(res, "Chi-square").attrs.get("notes", "") or "adequate" in res.tables[-1].notes


@pytest.mark.r
def test_F_crosstab_sparse_suggests_fisher(tmp_path, r_session):
    from statexplore.core.importers import import_file

    p = tmp_path / "s.csv"
    pd.DataFrame({"a": ["x"] * 8 + ["y"] * 4, "b": ["p"] * 7 + ["q"] + ["q"] * 3 + ["p"]}).to_csv(p, index=False)
    res = eda.crosstab(r_session, import_file(p), "a", "b")
    assert "Fisher" in res.tables[-1].notes


# ---- §8.5 correlation rules ----------------------------------------------------------------------

@pytest.mark.r
def test_spearman_accepts_ordinal_pearson_needs_treatment(hrs_path, r_session):
    ds = import_sav(hrs_path)
    M.approve(ds, "R_SHLT", "ordinal")
    M.approve(ds, "R_LTACTX", "ordinal")
    vars_ = {"x": ["R_SHLT"], "y": ["R_LTACTX"]}
    res, _ = run_analysis(r_session, ds, BY_KEY["cor_pair"], vars_, {"method": "spearman"})
    f = ds.analysis_frame(["R_SHLT", "R_LTACTX"]).dropna()
    rho = st.spearmanr(f["R_SHLT"], f["R_LTACTX"]).statistic
    assert table(res, "Spearman")["r"][0] == pytest.approx(rho, rel=1e-10)
    assert ds.var("R_SHLT").measure.value == "ordinal"           # no reclassification needed
    with pytest.raises(AnalysisInputError, match="Pearson"):
        run_analysis(r_session, ds, BY_KEY["cor_pair"], vars_, {"method": "pearson"})
    tr = [M.VariableTreatment(n, "continuous", acknowledged=True) for n in ("R_SHLT", "R_LTACTX")]
    res, _ = run_analysis(r_session, ds, BY_KEY["cor_pair"], vars_, {"method": "pearson"}, treatments=tr)
    assert table(res, "Pearson")["r"][0] == pytest.approx(st.pearsonr(f["R_SHLT"], f["R_LTACTX"]).statistic, rel=1e-10)
    # nominal codes are never ranked or correlated
    with pytest.raises(AnalysisInputError, match="continuous or ordinal"):
        run_analysis(r_session, ds, BY_KEY["cor_pair"], {"x": ["R_SHLT"], "y": ["R_MSTAT"]}, {"method": "spearman"})


@pytest.mark.r
def test_spearman_uses_category_order(hrs_path, r_session):
    ds = import_sav(hrs_path)
    M.approve(ds, "R_SHLT", "ordinal", category_order=[5.0, 4.0, 3.0, 2.0, 1.0])
    res, _ = run_analysis(r_session, ds, BY_KEY["cor_matrix"], {"variables": ["R_SHLT", "R_BMI"]},
                          {"method": "spearman"})
    f = ds.analysis_frame(["R_SHLT", "R_BMI"]).dropna()
    rho = st.spearmanr(f["R_SHLT"], f["R_BMI"]).statistic
    assert table(res, "Correlation matrix")["r"][0] == pytest.approx(-rho, rel=1e-8)   # reversed order
