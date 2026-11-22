"""v1.1 Phase 3 (§9, §20 G): categorical association analyses vs independent references
(scipy, statsmodels). 2x2 cells: a = exposed & event, b = exposed & reference,
c = comparison & event, d = comparison & reference."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.stats as st
from scipy.stats.contingency import association, odds_ratio
from statsmodels.stats.contingency_tables import Table2x2
from statsmodels.stats.proportion import confint_proportions_2indep

from statexplore.analysis.builders import BY_KEY, preview_2x2
from statexplore.analysis.framework import AnalysisInputError, run_analysis, validate
from statexplore.core.importers import import_file, import_sav

pytestmark = pytest.mark.r


def make(tmp_path, a, b, c, d, extra=0):
    """Outcome y (1 = event 'sick', 2 = 'well'), group g (1 = 'treated', 2 = 'control')."""
    rows = [(1, 1)] * a + [(2, 1)] * b + [(1, 2)] * c + [(2, 2)] * d + [(3, 1)] * extra
    df = pd.DataFrame(rows, columns=["y", "g"])
    p = tmp_path / "t.csv"
    df.to_csv(p, index=False)
    ds = import_file(p)
    ds.set_variable_field("y", "measure", "nominal")
    ds.set_variable_field("g", "measure", "nominal")
    return ds


def run2x2(r_session, ds, event=1.0, ref=2.0, exposed=1.0, comparison=2.0, **opts):
    o = {"event": event, "outcome_ref": ref, "exposed": exposed, "comparison": comparison, **opts}
    res, _ = run_analysis(r_session, ds, BY_KEY["assoc_2x2"], {"outcome": ["y"], "group": ["g"]}, o)
    assert res.status.value in ("ok", "warning"), res.messages
    eff = next(t for t in res.tables if t.name.startswith("Effect"))
    m = {r[0]: dict(zip(eff.columns, r)) for r in eff.rows}
    return res, m


def test_G_estimates_and_cis_match_references(tmp_path, r_session):
    a, b, c, d = 30, 70, 15, 85
    ds = make(tmp_path, a, b, c, d)
    res, m = run2x2(r_session, ds)
    t = Table2x2(np.array([[a, b], [c, d]]))
    assert m["Odds ratio (sample)"]["estimate"] == pytest.approx(t.oddsratio, rel=1e-12)
    assert [m["Odds ratio (sample)"]["ci_low"], m["Odds ratio (sample)"]["ci_high"]] == \
        pytest.approx(list(t.oddsratio_confint(0.05)), rel=1e-10)
    assert m["Risk ratio"]["estimate"] == pytest.approx(t.riskratio, rel=1e-12)
    assert [m["Risk ratio"]["ci_low"], m["Risk ratio"]["ci_high"]] == pytest.approx(list(t.riskratio_confint(0.05)), rel=1e-10)
    rd_ci = confint_proportions_2indep(a, a + b, c, c + d, method="newcomb", compare="diff")
    assert m["Risk difference"]["estimate"] == pytest.approx(a / (a + b) - c / (c + d), rel=1e-12)
    assert [m["Risk difference"]["ci_low"], m["Risk difference"]["ci_high"]] == pytest.approx(list(rd_ci), rel=1e-8)
    cond = odds_ratio(np.array([[a, b], [c, d]]), kind="conditional")
    # both are root-finder solutions (R uniroot tol ~1.2e-4): agree to ~1e-5
    assert m["Odds ratio (conditional MLE)"]["estimate"] == pytest.approx(cond.statistic, rel=1e-4)
    ci = cond.confidence_interval(0.95)
    assert [m["Odds ratio (conditional MLE)"]["ci_low"], m["Odds ratio (conditional MLE)"]["ci_high"]] == \
        pytest.approx([ci.low, ci.high], rel=2e-4)       # root-finder tolerance, as above
    phi = (a * d - b * c) / np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    assert m["Phi coefficient"]["estimate"] == pytest.approx(phi, rel=1e-12)
    test = next(t for t in res.tables if t.name == "Test").rows[0]
    assert test[0].startswith("Fisher") and test[3] == pytest.approx(st.fisher_exact([[a, b], [c, d]]).pvalue, rel=1e-10)
    assert m["Risk difference"]["ci_method"].startswith("Newcombe") and m["Risk ratio"]["ci_method"] == "Katz log Wald"
    assert "Unadjusted" in next(t for t in res.tables if t.name.startswith("Effect")).notes
    assert "percentage points" in res.values["notes"][0]
    shown = next(t for t in res.tables if t.name == "2x2 table")
    assert shown.row_names == ["Exposed: 1", "Comparison: 2", "Total"]
    assert [r[:3] for r in shown.rows] == [[a, b, a + b], [c, d, c + d], [a + c, b + d, a + b + c + d]]


def test_G_chisq_options_are_reported(tmp_path, r_session):
    a, b, c, d = 30, 70, 15, 85
    ds = make(tmp_path, a, b, c, d)
    for test, corr in (("chisq", False), ("chisq_yates", True)):
        res, _ = run2x2(r_session, ds, test=test)
        row = next(t for t in res.tables if t.name == "Test").rows[0]
        chi2, p, _, _ = st.chi2_contingency([[a, b], [c, d]], correction=corr)
        assert row[1] == pytest.approx(chi2, rel=1e-10) and row[3] == pytest.approx(p, rel=1e-8)
        assert ("Yates" in row[0]) == corr


def test_G_reversing_event_and_groups(tmp_path, r_session):
    ds = make(tmp_path, 30, 70, 15, 85)
    _, m = run2x2(r_session, ds)
    _, mr = run2x2(r_session, ds, event=2.0, ref=1.0)                 # event <-> reference
    assert mr["Odds ratio (sample)"]["estimate"] == pytest.approx(1 / m["Odds ratio (sample)"]["estimate"], rel=1e-12)
    assert mr["Risk difference"]["estimate"] == pytest.approx(-m["Risk difference"]["estimate"], rel=1e-12)
    assert mr["Risk ratio"]["estimate"] == pytest.approx((70 / 100) / (85 / 100), rel=1e-12)
    _, mg = run2x2(r_session, ds, exposed=2.0, comparison=1.0)         # exposed <-> comparison
    assert mg["Risk ratio"]["estimate"] == pytest.approx(1 / m["Risk ratio"]["estimate"], rel=1e-12)
    assert mg["Phi coefficient"]["estimate"] == pytest.approx(-m["Phi coefficient"]["estimate"], rel=1e-12)


def test_G_zero_cells(tmp_path, r_session):
    ds = make(tmp_path, 0, 20, 6, 14)
    res, m = run2x2(r_session, ds)
    assert m["Odds ratio (sample)"]["estimate"] == 0 and m["Odds ratio (sample)"]["ci_low"] is None
    assert "zero" in m["Odds ratio (sample)"]["note"]
    assert m["Risk ratio"]["ci_low"] is None and "No events" in m["Risk ratio"]["note"]
    assert m["Odds ratio (conditional MLE)"]["ci_low"] == 0                 # exact CI still defined
    rd_ci = confint_proportions_2indep(0, 20, 6, 20, method="newcomb", compare="diff")
    assert [m["Risk difference"]["ci_low"], m["Risk difference"]["ci_high"]] == pytest.approx(list(rd_ci), rel=1e-8)
    assert any("zero cell" in msg for msg in res.messages)
    # Haldane-Anscombe only when the user selects it, and it is reported
    _, mh = run2x2(r_session, ds, zero_cells="haldane")
    # closed form: 0.5 added to every cell
    a, b, c, d = 0.5, 20.5, 6.5, 14.5
    orh, se = a * d / (b * c), np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    assert mh["Odds ratio (sample)"]["estimate"] == pytest.approx(orh, rel=1e-12)
    assert mh["Odds ratio (sample)"]["ci_low"] == pytest.approx(orh * np.exp(-st.norm.ppf(0.975) * se), rel=1e-10)
    assert "Haldane" in mh["Odds ratio (sample)"]["note"]


def test_G_other_categories_excluded_and_reference_required(tmp_path, r_session):
    ds = make(tmp_path, 30, 70, 15, 85, extra=9)       # 9 rows with outcome code 3
    res, _ = run2x2(r_session, ds)
    assert res.n_used == 200
    assert "9 complete rows in other categories are excluded" in res.tables[0].notes
    with pytest.raises(AnalysisInputError, match="choose a category"):
        validate(BY_KEY["assoc_2x2"], ds, {"outcome": ["y"], "group": ["g"]},
                 {"event": 1.0, "exposed": 1.0, "comparison": 2.0})
    with pytest.raises(AnalysisInputError, match="must differ"):
        validate(BY_KEY["assoc_2x2"], ds, {"outcome": ["y"], "group": ["g"]},
                 {"event": 1.0, "outcome_ref": 1.0, "exposed": 1.0, "comparison": 2.0})
    tab = preview_2x2(ds, {"outcome": ["y"], "group": ["g"]},
                      {"event": 1.0, "outcome_ref": 2.0, "exposed": 1.0, "comparison": 2.0})
    assert tab.to_numpy().tolist() == [[30, 70, 100], [15, 85, 100], [45, 155, 200]]


def test_G_rxc_table_cramers_v(hrs_path, r_session):
    ds = import_sav(hrs_path)
    res, _ = run_analysis(r_session, ds, BY_KEY["assoc_table"], {"row": ["R_SHLT"], "col": ["RARACEM"]}, {})
    f = ds.analysis_frame(["R_SHLT", "RARACEM"]).dropna()
    obs = pd.crosstab(f["R_SHLT"], f["RARACEM"]).to_numpy()
    chi2, p, dof, _ = st.chi2_contingency(obs, correction=False)
    test = next(t for t in res.tables if t.name == "Test of independence").rows[0]
    assert test[1] == pytest.approx(chi2, rel=1e-10) and test[2] == dof and test[3] == pytest.approx(p, rel=1e-8)
    meas = next(t for t in res.tables if t.name == "Association measures").rows
    assert meas[0][1] == pytest.approx(association(obs, method="cramer"), rel=1e-10)


def test_G_fisher_rxc(tmp_path, r_session):
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"r": rng.choice(["a", "b", "c"], 40), "c": rng.choice(["x", "y"], 40)})
    p = tmp_path / "f.csv"
    df.to_csv(p, index=False)
    ds = import_file(p)
    res, _ = run_analysis(r_session, ds, BY_KEY["assoc_table"], {"row": ["r"], "col": ["c"]}, {"test": "fisher"})
    rows = next(t for t in res.tables if t.name == "Test of independence").rows
    assert rows[0][0] == "Fisher's exact test" and 0 <= rows[0][3] <= 1
    # 2x2: Fisher p equals scipy
    df2 = pd.DataFrame({"r": ["a"] * 7 + ["b"] * 9, "c": ["x"] * 5 + ["y"] * 2 + ["x"] * 2 + ["y"] * 7})
    df2.to_csv(p, index=False)
    res, _ = run_analysis(r_session, import_file(p), BY_KEY["assoc_table"], {"row": ["r"], "col": ["c"]}, {"test": "fisher"})
    rows = next(t for t in res.tables if t.name == "Test of independence").rows
    assert rows[0][3] == pytest.approx(st.fisher_exact([[5, 2], [2, 7]]).pvalue, rel=1e-10)
    assert any("Sparse" in m for m in res.messages)


def test_G_unadjusted_vs_adjusted_are_separate(tmp_path):
    spec2, logit = BY_KEY["assoc_2x2"], BY_KEY["logistic"]
    assert spec2.family == "Categorical Association" and logit.family == "Regression"
    assert "adjusted" in spec2.description and "logistic" in spec2.description.lower()
