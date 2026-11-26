"""Build docs/v1.3.3/disposition_v1.3.3.csv: every ID of the v1.3.3 register (03_Defect_Register.csv, original IDs
kept) with root cause, code location, regression tests, the ACTUAL gate status of a runner run (gates.csv) and the
disposition. Usage: python make_disposition.py <03_Defect_Register.csv> <runner gates.csv> <out.csv> <runner pytest_junit.xml>"""
import csv
import sys

# new / reworked items: (disposition, root cause, code location)
NEW = {
    "STAT-47": ("fixed", "Full Analysis rebuilt every refit from widget defaults (`MixedPage._spec`): the transferred "
                "model's explicit term list, descriptor, source and shared sample were dropped; the Interactions combo had "
                "no way to show an arbitrary term set", "gui/pages/mixed.py (load_spec/_spec/_fit), "
                "gui/widgets/terms_editor.py, analysis/mixed.py (spec_diff, attach_lineage)"),
    "V133-STATE-01": ("fixed", "fit_spec set interactions='none' with the terms only in MixedSpec.terms",
                      "gui/pages/mixed.py load_spec + FixedTermsEditor"),
    "V133-STATE-02": ("fixed", "_fit called _spec() which ignored the draft's terms", "gui/pages/mixed.py _spec"),
    "V133-STATE-03": ("fixed", "same as STATE-02 (method change refit from widgets)", "gui/pages/mixed.py _spec"),
    "V133-STATE-04": ("fixed", "descriptor/source not carried into refits; no field-level diff",
                      "analysis/mixed.py spec_diff/attach_lineage; MixedSpec.origin"),
    "V133-STATE-05": ("fixed", "preset-only editor (None / Time x each / All two-way)",
                      "gui/widgets/terms_editor.py; analysis/mixed.py term_universe/hierarchy_problems"),
    "V133-STAT-01": ("fixed (hardened)", "Reproduced on ff7b4f3: after a transfer the time comparison dropped "
                     "TIME x GROUP (same _spec root cause); AR(1) itself was kept in every route tested (manual form, "
                     "workflow A and B transfers, 46A). Variants now keep terms (remapped per time form), random part, "
                     "residual, cluster and ID policy, state the REML->ML change, and are read back from the fitted R "
                     "object (fail closed on any difference)", "analysis/mixed.py comparison_variants/compare_specs/"
                     "effective_problems; analysis/covstruct.py sx_effective"),
    "V133-STAT-02": ("fixed", "preflight grouped on the raw ID column", "analysis/longitudinal_preflight.py (PID_KEY); "
                     "discovery/space.py longitudinal_summary"),
    "V133-STAT-03": ("verified (no defect)", "default-off exploratory omission already existed; composite IDs and "
                     "labels now explicit; Full Analysis exposes the cluster random-intercept switch",
                     "gui/pages/mixed.py cluster_random_chk; discovery/space.py random_part"),
    "V133-VAL-01": ("implemented", "no AMD validation UI; the older procedure CV (Validation page) had no within-cluster "
                    "target, bootstrap, fixed mode or saved manifest", "discovery/validate.py, "
                    "gui/widgets/validation_panel.py, DiscoveryRun.validation_runs"),
    "V133-EXP-01": ("fixed", "Full Analysis View Code had no dataset, hence no portable bundle (only raw code with temp "
                    "paths)", "gui/pages/*.py CodeDialog(dataset=...), export/exporters.py write_portable"),
    "V133-EXP-02": ("fixed", "AMD candidate CodeDialog was created without result, so provenance / identity / versions "
                    "were never passed to the manifest", "gui/pages/discovery.py _view_code; exporters.write_portable"),
    "V133-EXP-03": ("fixed", "bundles checked data hash and identity but not the fitted numbers",
                    "export/exporters.py results_check"),
    "V133-EXP-04": ("fixed", "dependencies were every pkg:: in the script text; shared helpers mention ordinal "
                    "(sx_diag) and all discovery families", "export/exporters.py FAMILY_PACKAGES / loaded packages; "
                    "analysis/covstruct.py getExportedValue for optional helpers"),
    "V133-UI-01": ("fixed", "preflight column 'outcome range' held a time range", "analysis/longitudinal_preflight.py"),
    "V133-UI-02": ("fixed", "workflow B estimate printed the focal-test text", "discovery/space.py coding; "
                   "gui/pages/discovery.py _estimate"),
    "V133-UI-03": ("fixed", "blocked start left the previous progress bar", "gui/pages/discovery.py _start"),
    "V133-UI-04": ("fixed", "cluster-free formula showed the raw ID", "discovery/space.py random_part"),
    "V133-UI-05": ("fixed", "availability note said an absent interior wave is not part of the analysis range",
                   "analysis/mixed.py availability notes"),
    "V133-UI-06": ("fixed", "interpretations are lazy (display only) and were blank in CSV",
                   "gui/pages/discovery.py ResultTable.csv_text"),
    "V133-UI-07": ("fixed", "only clipboard copy", "gui/pages/discovery.py ResultTable.save_csv"),
    "V133-UI-08": ("fixed", "no log-tail column", "discovery/engine.py tests_frame log10_p"),
    "V133-UI-09": ("fixed", "time could be listed as required AND enter through the Time x each strategy (double "
                   "count)", "discovery/space.py validate_settings; gui/pages/discovery.py _settings/_estimate"),
    "V133-CMP-01": ("verified", "GLMM / CLMM likelihoods equal independent lme4 / ordinal fits; GEE robust Wald / QIC "
                    "path unchanged and covered by v1.3.2 tests", "tests/test_v133.py, tests/test_v132.py"),
    "V133-DIAG-01": ("verified", "attempt controller retries optimizer errors only and flags fallback; injection tests "
                     "added; material negative LR withheld", "analysis/covstruct.py sx_fit_attempts; analysis/lrtest.py"),
    "V133-HIST-01": ("verified", "adversaries added: filter TIME<=4 (960 verified vs new 800), 3-category reference "
                     "change, current view unchanged", "discovery/history.py (unchanged)"),
    "V133-DISC-01": ("verified", "independent count oracle, p.adjust per family, time counted once",
                     "discovery/space.py, discovery/engine.py"),
    "V133-QA-01": ("done", "36 workflow obligations + 18 standalone fits vs independent R", "tests/test_v133.py matrix"),
    "V133-QA-02": ("done (partial: owner DPI)", "benchmark tool and report; GUI persistence tests",
                   "tools/sx_bench.py, docs/v1.3.3/benchmark_v1.3.3.md"),
    "V133-QA-03": ("owner step", "clean offline Windows machine install (Test 54)", "docs/v1.3.3/manual_acceptance_v1.3.3.md"),
}


# items without a runner gate: the tests that cover them, evaluated from the same run's JUnit file
EXTRA = {"UI-51": ["test_51_independent_enumeration_oracle_for_three_candidates", "test_ui50_empty_search_explained"],
         "V133-QA-02": ["test_validation_panel_run_persist_reopen", "test_51_cancel_kills_running_batches_and_resume_completes_exactly_once",
                        "test_ui42_49_57_csv_status_reload_and_cluster_columns"]}
NEW["UI-51"] = ("verified", "candidate vs total predictor count: time counts once with Time x each (V133-UI-09) and is "
                "refused as a required duplicate; empty / too-small spaces are explained before launch",
                "discovery/space.py validate_settings")


def main(register: str, gates: str, out: str, junit: str) -> None:
    import xml.etree.ElementTree as ET

    g = {r["id"]: r for r in csv.DictReader(open(gates, encoding="utf-8"))}
    res = {}
    for tc in ET.parse(junit).getroot().iter("testcase"):
        res[tc.get("name")] = "FAIL" if tc.find("failure") is not None or tc.find("error") is not None else \
            "SKIPPED" if tc.find("skipped") is not None else "PASS"
    for gid, names in EXTRA.items():
        st = [res.get(n, "NOT RUN") for n in names]
        g[gid] = {"status": "PASS" if all(x == "PASS" for x in st) else "/".join(st), "tests": f"{len(names)} tests",
                  "detail": "; ".join(names)}
    rows = []
    for r in csv.DictReader(open(register, encoding="utf-8")):
        i = r["issue_id"]
        disp, cause, loc = NEW.get(i, ("regression lock re-verified" if i in g else "carried (wording / manual check)",
                                       "inherited from v1.3.2 (see docs/v1.3.2/defect_register_v1.3.2.csv)", ""))
        gate = g.get(i, {})
        rows.append({"issue_id": i, "priority": r["priority"], "status_at_handoff": r["status_at_handoff"],
                     "requirement": r["requirement"], "disposition": disp, "root_cause": cause, "code_location": loc,
                     "gate_status": gate.get("status", "NO GATE"), "regression_tests": gate.get("tests", ""),
                     "gate_detail": gate.get("detail", ""), "acceptance": r["acceptance"]})
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    by = {}
    for r in rows:
        by[(r["priority"], r["gate_status"])] = by.get((r["priority"], r["gate_status"]), 0) + 1
    for k in sorted(by):
        print(k, by[k])


if __name__ == "__main__":
    main(*sys.argv[1:5])
