from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HRS_PATH = Path(os.environ.get("STATEXPLORE_HRS", r"E:\Downloads\hrs_least_missing_1000.sav"))


@pytest.fixture
def csv_semicolon_cp1252(tmp_path: Path) -> Path:
    p = tmp_path / "semi.csv"
    text = "id;city;score;group\n1;Montréal;3.5;a\n2;Zürich;;b\n3;Lyon;4.25;a\n4;;5;NA\n"
    p.write_bytes(text.encode("cp1252"))
    return p


@pytest.fixture
def xlsx_two_sheets(tmp_path: Path) -> Path:
    p = tmp_path / "book.xlsx"
    with pd.ExcelWriter(p, engine="openpyxl") as xw:
        pd.DataFrame({"a": [1, 2, 3]}).to_excel(xw, sheet_name="First", index=False)
        pd.DataFrame({"x": [1.5, None, 3.0], "label": ["p", "q", None]}).to_excel(xw, sheet_name="Second", index=False)
    return p


@pytest.fixture
def sav_labelled(tmp_path: Path) -> Path:
    """SPSS file with variable labels, value labels, user-missing codes and measures."""
    import pyreadstat

    rng = np.random.default_rng(1)
    n = 60
    df = pd.DataFrame({
        "id": np.arange(1, n + 1, dtype=float),
        "sex": rng.choice([1.0, 2.0], n),
        "mood": rng.choice([1.0, 2.0, 3.0, 9.0], n),        # 9 = user-missing ("refused")
        "score": rng.normal(50, 10, n).round(3),
        "income": rng.choice([10.0, 20.0, 30.0, -99.0], n),  # -99..-1 = user-missing range
    })
    df.loc[[3, 7], "score"] = np.nan                         # system-missing
    p = tmp_path / "labelled.sav"
    pyreadstat.write_sav(
        df, str(p),
        column_labels={"id": "Participant ID", "sex": "Sex of respondent", "mood": "Mood today",
                       "score": "Test score", "income": "Income band"},
        variable_value_labels={"sex": {1: "Male", 2: "Female"},
                               "mood": {1: "Low", 2: "Medium", 3: "High", 9: "Refused"}},
        missing_ranges={"mood": [9], "income": [{"lo": -99, "hi": -1}]},
        variable_measure={"id": "scale", "sex": "nominal", "mood": "ordinal", "score": "scale",
                          "income": "scale"},
    )
    return p


@pytest.fixture
def hrs_path() -> Path:
    if not HRS_PATH.is_file():
        pytest.skip(f"HRS test dataset not found at {HRS_PATH}")
    return HRS_PATH


# ---- R ----------------------------------------------------------------

@pytest.fixture(scope="session")
def r_worker():
    from statexplore.rbridge.locate import PHASE1_PACKAGES, find_rscript
    from statexplore.rbridge.worker import RWorker

    if find_rscript() is None:
        pytest.skip("R is not installed")
    w = RWorker()
    w.start()
    vers = w.request("versions", {"packages": PHASE1_PACKAGES}).result["packages"]
    missing = [k for k, v in vers.items() if v is None]
    if missing:
        w.stop()
        pytest.skip(f"R packages missing: {missing} (run Rscript r/install_packages.R)")
    yield w
    w.stop()


@pytest.fixture
def r_session(r_worker, tmp_path):
    from statexplore.analysis.session import RSession

    return RSession(r_worker, tmp_path / "rsession")
