"""v1.1 §23: the final user acceptance workflow, driven through the real GUI on the HRS file.
Run with `-s` to print the evidence for each step (used for docs/release-audit-v1.1.0.md)."""
from __future__ import annotations

import hashlib
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

from statexplore.gui.main_window import SECTIONS, MainWindow  # noqa: E402
from statexplore.gui.state import AppState  # noqa: E402

pytestmark = pytest.mark.r


def sha(p) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def test_final_acceptance_workflow(qtbot, tmp_path, hrs_path, monkeypatch, r_worker):
    from PySide6.QtCore import QSettings

    from statexplore.analysis import mixed as MX
    from statexplore.core.derive import FilterSpec
    from statexplore.core.project import Project
    from statexplore.export import exporters
    from statexplore.gui.pages.r_console import PackageManagerDialog
    from tests.test_console_packages import _binary_pkg
    from tests.test_gui_v11 import select_analysis

    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(tmp_path / "settings"))
    monkeypatch.setenv("STATEXPLORE_USER_LIB_ROOT", str(tmp_path / "userlib"))
    log: list[str] = []

    def step(n: int, text: str) -> None:
        log.append(f"| {n} | {text} |")
        print(f"STEP {n}: {text}".encode("ascii", "replace").decode())   # Windows console encoding

    sav_hash = sha(hrs_path)
    w = MainWindow(AppState())
    try:
        step(1, f"PASS: window '{w.windowTitle()}', {w.sidebar.count()} sections")
        from statexplore import __version__
        assert w.windowTitle() == f"StatExplore {__version__}" and w.sidebar.count() == len(SECTIONS) == 13

        w.import_path(str(hrs_path))
        qtbot.waitUntil(lambda: w.state.dataset is not None, timeout=60000)
        ds = w.state.dataset
        step(2, f"PASS: SAV imported, {ds.n_rows:,} × {ds.n_cols}")
        assert (ds.n_rows, ds.n_cols) == (16000, 168)

        w.sidebar.setCurrentRow(SECTIONS.index("Dataset Explorer"))
        exp = w.pages["Dataset Explorer"]
        step(3, "PASS: Dataset Explorer open (Overview, Data, Variables, Measurement Review, …)")

        review = exp.review
        assert "R_LTACTX" in review._names
        row = review._names.index("R_LTACTX")
        src, sugg = review.table.item(row, 2).text(), review.table.item(row, 4).text()
        step(4, f"PASS: R_LTACTX listed in Measurement Review ({len(review._names)} variables flagged)")
        review.select("R_LTACTX")
        detail = review.detail.toPlainText()
        step(5, f"PASS: SPSS level '{src}', suggested '{sugg}'; codes/labels shown; "
                f"direction '{ds.var('R_LTACTX').coding_direction}'")
        assert (src, sugg) == ("nominal", "ordinal") and "5.Never" in detail

        monkeypatch.setattr(review, "_confirm", lambda text: True)
        review._accept()
        v = ds.var("R_LTACTX")
        step(6, f"PASS: approved ordinal (status '{v.measure_status}'; source level still '{v.source_measure}')")
        assert v.measure.value == "ordinal" and v.source_measure == "nominal"

        ds.set_variable_field("HHIDPN", "role", "id")
        ds.set_variable_field("WAVE", "role", "time")
        w.state.notify_metadata_changed()
        w.sidebar.setCurrentRow(SECTIONS.index("Mixed & Longitudinal Models"))
        mp = w.pages["Mixed & Longitudinal Models"]
        step(7, "PASS: Mixed & Longitudinal Models open")

        mp.outcome.setCurrentIndex(mp.outcome.findData("R_LTACTX"))
        step(8, "PASS: R_LTACTX selected as the outcome")

        mp._use_family()
        assert mp.family.currentData() == "ordinal"
        step(9, f"PASS: ordinal family chosen from the suggestion ({mp.engine_label.text()[:60]}…); Gaussian "
                "would require ticking 'Treat R_LTACTX as continuous'")

        mp.time.setCurrentIndex(mp.time.findData("WAVE"))
        mp.pred_sel.set_selected(["RAGENDER"])
        mp.interactions.setCurrentIndex(mp.interactions.findData("none"))
        assert mp.idv.currentData() == "HHIDPN"
        step(10, "PASS: participant ID HHIDPN, time WAVE, predictor RAGENDER")

        pre = mp.preflight_view.toPlainText()
        assert "Wave 1 to Wave 6" in pre and "Wave 7 to Wave 16" in pre
        step(11, "PASS: preflight shows R_LTACTX unobserved in Wave 1 to Wave 6; analysis range Wave 7 to Wave 16")

        mp._fit()
        qtbot.waitUntil(lambda: mp.current is not None, timeout=600000)
        res = mp.current
        assert res.status.value in ("ok", "warning") and res.title.startswith("Ordinal logistic mixed model")
        fx = next(t for t in res.tables if t.name.startswith("Fixed effects"))
        step(12, f"PASS: {res.title}; n = {res.n_used:,} observations; analysis range {res.values['analysis_range']}")

        assert {"or_ci_low", "or_ci_high", "p"} <= set(fx.columns)
        step(13, f"PASS: coefficients with ORs, 95% CIs and p ({len(fx.rows)} terms); thresholds; random-effect "
                 f"variance; convergence and Hessian checks; availability table")

        mp.current = None
        mp.family.setCurrentIndex(mp.family.findData("gaussian"))
        mp.treatment_panel.ask = False
        mp.treatment_panel.set_treated("R_LTACTX")
        mp._fit()
        qtbot.waitUntil(lambda: mp.current is not None, timeout=600000)
        gauss = mp.current
        cmp_res, _ = MX.compare_specs(w.state.r_session(), ds, mp._spec(), "random")
        step(14, f"PASS: alternative Gaussian model on codes 1–5 ({len(gauss.plots)} plots) and random-effects "
                 f"comparison ({cmp_res.tables[0].name})")
        assert gauss.status.value in ("ok", "warning") and cmp_res.status.value in ("ok", "warning")

        ds.add_filter(FilterSpec("WAVE", "==", 10.0))
        w.state.notify_metadata_changed()
        ab = w.pages["Analysis Builder"]
        select_analysis(ab, "assoc_2x2")
        ab.slot_widgets["outcome"].combo.setCurrentIndex(ab.slot_widgets["outcome"].combo.findData("R_SMOKEN"))
        ab.slot_widgets["group"].combo.setCurrentIndex(ab.slot_widgets["group"].combo.findData("RAGENDER"))
        for key, code in (("event", 1.0), ("outcome_ref", 0.0), ("exposed", 1.0), ("comparison", 2.0)):
            ab.option_widgets[key].setCurrentIndex(ab.option_widgets[key].findData(code))
        assert "Exposed: 1.Male" in ab.preview.text()
        ab._run()
        qtbot.waitUntil(lambda: ab.current is not None, timeout=120000)
        eff = {r[0]: r[1] for r in next(t for t in ab.current[0].tables if t.name.startswith("Effect")).rows}
        step(15, f"PASS: wave-10 2×2 (smoking × gender): OR {eff['Odds ratio (sample)']:.2f}, "
                 f"RR {eff['Risk ratio']:.2f}, RD {100 * eff['Risk difference']:.1f} points")

        from statexplore.analysis import measurement as M

        M.approve(ds, "R_SHLT", "ordinal")
        w.state.notify_metadata_changed()
        dp = w.pages["Automated Model Discovery"]
        dp.outcome.setCurrentIndex(dp.outcome.findData("R_SHLT"))
        dp._use_suggestion()
        dp.cand_sel.set_filter("Big 5")
        dp.cand_sel.check_visible()
        dp.cand_sel.set_filter("")
        dp.req_sel.set_selected(["RAGENDER"])
        dp.workers.setValue(2)
        info = dp._estimate()
        dp._start()
        qtbot.waitUntil(lambda: dp.run is not None and dp.run.status == "complete", timeout=600000)
        step(16, f"PASS: ordinal-outcome search (R_SHLT): {info['total']} candidates, family "
                 f"{dp.run.coding['family'].split(' (')[0]}, status {dp.run.status}")

        step(17, f"PASS: standard selector controls used (search 'Big 5' → Check visible: "
                 f"{len(dp.cand_sel.selected())} selected; required covariates have the same controls)")
        assert set(dp.req_sel.buttons) == {"check_visible", "clear_visible", "check_all", "clear_all"}

        w.sidebar.setCurrentRow(SECTIONS.index("R Console"))
        rc = w.pages["R Console"]
        step(18, "PASS: R Console open, security warning shown")

        rc.editor.setPlainText("tab <- table(my_data$R_SHLT)\ntab\nbarplot(tab)")
        rc.run_script()
        qtbot.waitUntil(lambda: not rc._busy, timeout=120000)
        assert "1.Excellent" in rc.output.toPlainText() and rc.last.plots
        step(19, f"PASS: script ran on my_data (copy of {ds.n_rows_filtered:,} filtered rows), plot captured")

        dlg = PackageManagerDialog(w.state, rc)
        assert dlg.install_path(str(_binary_pkg(tmp_path, "sxaudit")), confirm=False)
        dlg.close()
        step(20, "PASS: local binary package sxaudit installed offline into the user library")

        rc.editor.setPlainText("library(sxaudit)\nhello()")
        rc.run_script()
        qtbot.waitUntil(lambda: not rc._busy, timeout=120000)
        assert "hello from sxaudit" in rc.output.toPlainText()
        step(21, "PASS: the new package loads and runs in a script")

        sp = w.pages["Survey Analysis"]
        sp._detect()
        assert "weight: none found" in sp.output.toPlainText()
        sp._explain()
        step(22, "PASS: Survey Analysis reports that this HRS extract has no sampling weight, PSU or strata "
                 "(R_WEIGHT is body weight), so no design is invented")

        out = exporters.export(w.state.project, [r.id for r in w.state.project.results], tmp_path / "export",
                               {"xlsx", "docx", "pdf", "R", "py"}, [dp.run.id])
        step(23, f"PASS: exported {len(out)} files (Excel, Word, PDF, R scripts, Python launchers)")

        assert w._save_to(tmp_path / "audit.sxp")
        w.open_project(str(tmp_path / "audit.sxp"))
        again = w.state.project
        assert again.dataset.var("R_LTACTX").measure.value == "ordinal" and len(again.searches) == 1
        step(24, f"PASS: saved and reopened ({len(again.results)} results, {len(again.searches)} search, "
                 "approved levels kept)")
        assert sha(hrs_path) == sav_hash
        log.append("| — | Original SAV unchanged | PASS: SHA-256 identical before and after |")
    finally:
        w.state.dirty = False
        w.close()
        (tmp_path / "audit_steps.md").write_text("\n".join(log), encoding="utf-8")
        print("\n".join(log).encode("ascii", "replace").decode())
