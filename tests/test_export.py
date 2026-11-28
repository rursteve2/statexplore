"""Phase 7 acceptance: exported values match analysis results; exported code reproduces the
fitted models in a fresh R process (Test O); saved projects reopen correctly."""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np
import pandas as pd
import pytest

from statexplore.analysis import mixed, teaching
from statexplore.analysis.builders import BY_KEY
from statexplore.analysis.framework import run_analysis
from statexplore.analysis.models import ModelSpec
from statexplore.core.derive import DerivedSpec, FilterSpec
from statexplore.core.importers import import_file
from statexplore.core.project import Project
from statexplore.discovery.engine import create_run, execute
from statexplore.discovery.space import SearchSettings
from statexplore.export import exporters
from statexplore.rbridge.locate import find_rscript, r_library


@pytest.fixture
def project_with_results(tmp_path, r_session):
    rng = np.random.default_rng(10)
    n = 150
    df = pd.DataFrame({"id": np.repeat(np.arange(1, 51), 3), "t": np.tile([0, 1, 2], 50),
                       "x": rng.normal(size=n), "g": rng.choice(["a", "b"], n)})
    df["y"] = df["x"] * (df["g"] == "b") + 0.3 * df["t"] + rng.normal(size=n)
    p = tmp_path / "src.csv"
    df.to_csv(p, index=False)
    ds = import_file(p)
    ds.add_derived(DerivedSpec("x_c", "center", ["x"]))
    ds.add_filter(FilterSpec("x", ">", -3))
    proj = Project(ds)
    for res, art in [
        run_analysis(r_session, ds, BY_KEY["lm"], {"dependent": ["y"], "predictors": ["x_c", "g"]},
                     {"interactions": "two_way"}),
        teaching.run(r_session, ds, ModelSpec("y", [("x_c", "g")], "gaussian"), teaching.PlotSpec(x="x_c", group="g")),
        mixed.run(r_session, ds, mixed.MixedSpec(outcome="y", id="id", time="t", predictors=["g"])),
    ]:
        proj.add_result(res, art)
    run = execute(create_run(ds, SearchSettings(outcome="y", candidates=["x", "g", "t"], workers=1,
                                                max_models=None, max_minutes=None)), ds)
    proj.searches.append(run)
    return proj


def _numbers(table_rows) -> list[float]:
    out = []
    for row in table_rows:
        out += [float(v) for v in row if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v]
    return out


@pytest.mark.r
def test_O_exported_r_scripts_reproduce_results(project_with_results, tmp_path):
    proj = project_with_results
    # v1.3.6 (AUD135-03/07): a NEW export folder with one self-checking bundle per analysis (own data, with consent)
    written = exporters.export(proj, [r.id for r in proj.results], tmp_path / "export", {"R", "py"}, include_data=True)
    new = written[-1].parent
    env = dict(os.environ, R_LIBS=str(r_library()))
    rscript = str(find_rscript())
    for r in proj.results:
        sub = f"{exporters.slug(r.title)}_{r.id}"
        folder = new / sub
        script = folder / f"{sub}.R"
        text = script.read_text(encoding="utf-8")
        assert script in written and f'sx_data_file <- "{sub}_data.csv"' in text and (folder / f"{sub}_data.csv") in written
        assert "statexplore_" not in text.replace("# Source file", "")  # no temp-folder paths left
        done = subprocess.run([rscript, "--vanilla", script.name], cwd=folder, env=env, capture_output=True,
                              text=True, timeout=300)
        assert done.returncode == 0, done.stderr[-2000:]
        out = json.loads((folder / f"result_{r.id}.json").read_text(encoding="utf-8"))
        fresh = {t["name"]: t for t in out["tables"]}
        for t in r.tables:
            ft = fresh[next(k for k in fresh if exporters.slug(k) == exporters.slug(t.name) or k == t.name
                            or k.startswith(t.name.split(":")[0]))]
            cols = [c for c in ft["columns"]]
            rows = list(zip(*[ft["data"][c] if isinstance(ft["data"][c], list) else [ft["data"][c]] for c in cols]))
            np.testing.assert_allclose(_numbers(rows), _numbers(t.rows), rtol=1e-9, err_msg=f"{r.title}: {t.name}")
    assert (folder / f"{sub}_data.csv").is_file() and any(p.suffix == ".py" for p in written)


def test_xlsx_values_match(project_with_results, tmp_path):
    proj = project_with_results
    folder = exporters.export(proj, [r.id for r in proj.results], tmp_path / "x", {"xlsx"},
                              search_ids=[proj.searches[0].id])[-1].parent          # v1.3.6: a new export folder
    book = pd.read_excel(folder / "statexplore_results.xlsx", sheet_name=None)
    r = proj.results[0]
    coef = next(t for t in r.tables if t.name == "Coefficients")
    sheet = next(v for k, v in book.items() if k.startswith(f"{r.id} Coeff"))
    np.testing.assert_allclose(sheet["estimate"].to_numpy(float), [row[1] for row in coef.rows], rtol=1e-12)
    search = next(v for k, v in book.items() if k == f"search {proj.searches[0].id}"[:28])
    assert len(search) == 3 and "p_bh" in search.columns
    assert (folder / "analysis_history.csv").is_file() and (folder / f"search_{proj.searches[0].id}_settings.json").is_file()


def test_docx_and_pdf(project_with_results, tmp_path, qtbot):
    from docx import Document

    proj = project_with_results
    folder = exporters.export(proj, [r.id for r in proj.results], tmp_path / "d", {"docx", "pdf"})[-1].parent
    doc = Document(str(folder / "statexplore_report.docx"))
    text = "\n".join(p.text for p in doc.paragraphs)
    assert "Interpretation." in text and "Linear regression" in text
    cells = [c.text for t in doc.tables for row in t.rows for c in row.cells]
    est = next(t for t in proj.results[0].tables if t.name == "Coefficients").rows[0][1]
    assert f"{est:.3f}" in cells
    assert len(doc.inline_shapes) >= 1
    pdf = (folder / "statexplore_report.pdf").read_bytes()
    assert pdf[:4] == b"%PDF" and len(pdf) > 20_000


@pytest.mark.r
def test_project_with_everything_reopens(project_with_results, tmp_path):
    proj = project_with_results
    proj.models.append(ModelSpec("y", [("x", "g")], "gaussian", label="m"))
    out = tmp_path / "all.sxp"
    proj.save(out)
    again = Project.open(out)
    assert [r.id for r in again.results] == [r.id for r in proj.results]
    assert again.searches[0].meta_dict() == proj.searches[0].meta_dict()
    assert again.models[0].terms == proj.models[0].terms
    assert set(again.artifacts) == set(proj.artifacts)
    assert again.dataset.filters[0].to_dict() == proj.dataset.filters[0].to_dict()
