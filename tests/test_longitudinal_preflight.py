"""v1.1 Phase 6 (§12, §20 L): longitudinal data availability and missingness."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statexplore.analysis import mixed as MX
from statexplore.analysis.longitudinal_preflight import preflight
from statexplore.core.importers import import_file, import_sav


def test_L_hrs_ltactx_unobserved_waves(hrs_path):
    ds = import_sav(hrs_path)
    rep = preflight(ds, "R_LTACTX", "HHIDPN", "WAVE")
    f = ds.analysis_frame(["R_LTACTX", "HHIDPN", "WAVE"])
    assert rep.n_records == 16000 and rep.n_participants == 1000 and len(rep.time_points) == 16
    assert rep.unobserved_times == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert rep.analysis_range == (7.0, 16.0)
    by = rep.by_time
    expected = f.groupby("WAVE")["R_LTACTX"].count()
    assert by["observed"].tolist() == expected.tolist()
    assert by["missing"].tolist() == (1000 - expected).tolist()
    assert (by.loc[["Wave 1", "Wave 6"], "percent_missing"] == 100).all()
    assert any("not observed at 6 of 16 time points (Wave 1 to Wave 6)" in w and "not treated as zeros" in w
               and "Wave 7 to Wave 16" in w for w in rep.warnings)
    per = f.dropna(subset=["R_LTACTX"]).groupby("HHIDPN").size().reindex(f.HHIDPN.unique(), fill_value=0)
    assert rep.n_one_obs == (per == 1).sum() and rep.n_multi_obs == (per >= 2).sum() and rep.n_no_obs == (per == 0).sum()
    assert rep.duplicates == 0


def test_L_duplicates_spacing_slopes_and_no_zero_filling(tmp_path):
    rows = []
    for pid in range(1, 21):
        for t in ([1, 2, 4, 8] if pid % 2 else [1, 2, 3]):
            rows.append((pid, t, np.nan if (t == 4) else pid + t))
    rows.append((1, 2, 5.0))                                        # duplicate participant-time
    rows.append((99, 1, 3.0))                                       # a participant with one observation
    df = pd.DataFrame(rows, columns=["id", "t", "y"])
    p = tmp_path / "l.csv"
    df.to_csv(p, index=False)
    ds = import_file(p)
    rep = preflight(ds, "y", "id", "t", spec=MX.MixedSpec("y", "id", time="t", random="slope", residual="ar1"))
    assert rep.duplicates == 1 and rep.duplicate_examples[0][:2] == (1.0, 2.0)
    assert rep.by_time.loc["4", "observed"] == 0 and rep.unobserved_times == [4.0]
    assert rep.n_one_obs == 1
    assert any("Unequal spacing" in n and "continuous-time AR(1)" in n for n in rep.notes)        # information
    assert any(n.startswith("Random time slopes: 20 of 21") for n in rep.notes)   # enough participants: a note

    assert any("more than one row" in w for w in rep.warnings)
    # the analysis data never contains a filled-in value for the unobserved occasion
    assert ds.analysis_frame(["y", "t"]).query("t == 4")["y"].isna().all()


@pytest.mark.r
def test_L_model_result_reports_analysis_range(hrs_path, r_session):
    from statexplore.analysis import measurement as M

    ds = import_sav(hrs_path)
    ds.set_variable_field("HHIDPN", "role", "id")
    tr = [M.VariableTreatment("R_LTACTX", "continuous", acknowledged=True, numeric_confirmed=True).to_dict()]
    res, _ = MX.run(r_session, ds, MX.MixedSpec("R_LTACTX", "HHIDPN", time="WAVE", predictors=["RAGENDER"],
                                                treatments=tr))
    assert res.values["analysis_range"] == "Wave 7 to Wave 16"
    assert any("Wave 1 to Wave 6" in m for m in res.messages)
    t = next(t for t in res.tables if t.name == "Outcome availability by time point")   # computed in R
    assert t.rows[0][0] == "Wave 1" and t.rows[0][2] == 0 and t.rows[6][2] == 998
    fx = next(t for t in res.tables if t.name == "Fixed effects")
    assert fx.rows and res.n_used == ds.analysis_frame(["R_LTACTX", "WAVE", "RAGENDER"]).dropna().shape[0]
