"""StatExplore developer regression runner (v1.3.2 05 "Developer runner"; v1.3.3 gates and reference re-run).

Developer-only: no user-facing mode. One invocation produces an immutable evidence folder:

  1. independent references: 07_Generate_Test_Fixtures.R and 08_Independent_R_Reference_Models.R in a fresh R
     session (Rscript --vanilla), then Verify_Reference_Results.py (historical numeric audit);
  2. the application suite (pytest, real R, offscreen Qt) with JUnit XML;
  3. a machine-readable gate table: every defect-register / acceptance ID -> named tests -> status
     PASS / FAIL / BLOCKED / SKIPPED / NOT RUN / NOT APPLICABLE.

Exit code 0 only when every mandatory gate is PASS. Skipped, missing or blocked mandatory tests are failures.

    python tools/sx_regress.py --package E:\\path\\to\\StatExplore_v1.3.2_Developer_Package --out E:\\sx_runs
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# register / acceptance ID -> test node prefixes (file::name) that must all PASS. Rows marked "NOT APPLICABLE" carry
# the reason instead; owner-only checks (clean VM, physical DPI) are reported as NOT RUN by this runner.
GATES: dict[str, list[str] | str] = {
    "STAT-15": ["tests/test_v131.py::test_26_po_check"],
    "STAT-16": ["tests/test_v131.py::test_27_bic_identical", "tests/test_v132.py::test_cd01_workflow_b_18_combinations"],
    "STAT-17": ["tests/test_v132.py::test_cmp_residual_boundary_and_not_nested",
                "tests/test_v132.py::test_lr_python_service_negative_and_boundary_rules",
                "tests/test_v132.py::test_cd03_b_tests_match_reference"],
    "STAT-18": ["tests/test_v132.py::test_diag_nlme_near_boundary_detected_and_controls_clean"],
    "STAT-19": ["tests/test_v132.py::test_cmp_family_aware_glmm_clmm_gee"],
    "STAT-20": ["tests/test_v132.py::test_report_preflight_complete_rows_range",
                "tests/test_v132.py::test_cd07_irregular_time"],
    "STAT-21": ["tests/test_v132.py::test_stat21_clustered_mixed_matches_reference",
                "tests/test_v132.py::test_stat21_random_comparison_keeps_cluster",
                "tests/test_v132.py::test_cd06_reused_ids_local_policy"],
    "STAT-22": ["tests/test_v132.py::test_identity_35c", "tests/test_v132.py::test_identity_35a",
                "tests/test_v132.py::test_identity_legacy_origin_is_unverifiable",
                "tests/test_v132.py::test_identity_shifted_outcome", "tests/test_v132.py::test_identity_cross_module",
                "tests/test_v13.py::test_P0_01_verify_tolerances"],
    "STAT-23": ["tests/test_v132.py::test_lr_contrast_reference_follows_hypothesis",
                "tests/test_v132.py::test_stat21_random_comparison_keeps_cluster"],
    "STAT-24": ["tests/test_v132.py::test_cd01_workflow_b_18_combinations"],
    "STAT-25": ["tests/test_v132.py::test_cmp_family_aware_glmm_clmm_gee"],
    "STAT-26": ["tests/test_v131.py::test_26_po_check"],
    "STAT-27": ["tests/test_v131.py::test_28_gee_identity"],
    "STAT-28": ["tests/test_v13.py::test_P1_02_03_preflight"],
    "STAT-29": ["tests/test_v131.py::test_28_glmm_and_clmm_keep_family"],
    "STAT-30": ["tests/test_v131.py::test_28_glmm_and_clmm_keep_family"],
    "STAT-31": ["tests/test_v132.py::test_stat31_cancel_long_task_no_orphan",
                "tests/test_v131.py::test_28_timeout_and_cancel_are_never_complete"],
    "STAT-32": ["tests/test_v132.py::test_diag_37d_gradient_warning"],
    "STAT-33": ["tests/test_v132.py::test_cmp_family_aware_glmm_clmm_gee"],
    "STAT-34": ["tests/test_v132.py::test_cmp_all_failed_is_explicit"],
    "STAT-35": ["tests/test_v131.py::test_29_wording_does_not_claim_equal_spacing"],
    "STAT-36": ["tests/test_v132.py::test_cd08_exported_candidate_code", "tests/test_v131.py::test_30_exports"],
    "STAT-37": ["tests/test_v131.py::test_31_bounded_search_resume"],
    "STAT-38": ["tests/test_v132.py::test_diag_lme4_uncorrelated_slope_survives_postprocessing"],
    "STAT-39": ["tests/test_v132.py::test_cd09_history_context_rebuild_and_new_analysis"],
    "STAT-40": ["tests/test_v132.py::test_stat40_duplicate_labels"],
    "STAT-41": ["tests/test_v132.py::test_lr_python_service_negative_and_boundary_rules",
                "tests/test_v132.py::test_lr_r_twin_matches_python"],
    "STAT-42": ["tests/test_v132.py::test_report_nlme_tiny_f_probabilities"],
    "STAT-43": ["tests/test_v132.py::test_stat21_clustered_mixed_matches_reference"],
    "STAT-44": ["tests/test_v132.py::test_diag_lme4_uncorrelated_slope_survives_postprocessing"],
    "STAT-45": ["tests/test_v132.py::test_diag_direct_r2_not_misleading_fallback"],
    "STAT-46": ["tests/test_v132.py::test_diag_37d_gradient_warning"],
    "UI-28": ["tests/test_v131.py::test_32_layout"], "UI-29": ["tests/test_v131.py::test_33_measurement_review"],
    "UI-30": ["tests/test_v131.py::test_34_entities"], "UI-31": ["tests/test_v132_gui.py::test_ui31_selected_first",
                                                             "tests/test_v131.py::test_33_selected_predictors_first"],
    "UI-32": ["tests/test_v131.py::test_34_entities"], "UI-33": ["tests/test_v131.py::test_34_entities"],
    "UI-34": ["tests/test_v131.py::test_34_entities"],
    "UI-41": ["tests/test_v132_gui.py::test_ui42_49_57"], "UI-42": ["tests/test_v132_gui.py::test_ui42_49_57"],
    "UI-43": ["tests/test_v132_gui.py::test_ui43_frozen_column"],
    "UI-46": ["tests/test_v132_gui.py::test_ui46_47"], "UI-47": ["tests/test_v132_gui.py::test_ui46_47"],
    "UI-48": ["tests/test_v132_gui.py::test_ui42_49_57"], "UI-49": ["tests/test_v132_gui.py::test_ui42_49_57"],
    "UI-50": ["tests/test_v132_gui.py::test_ui50_empty_search_explained"],
    "UI-54": ["tests/test_v132.py::test_stat21_random_comparison_keeps_cluster"],
    "UI-56": ["tests/test_v132.py::test_cd04_time_only_b", "tests/test_v132_gui.py::test_cd12_cluster_controls"],
    "UI-57": ["tests/test_v132_gui.py::test_ui42_49_57", "tests/test_v132.py::test_stat21_clustered_mixed_matches"],
    "CD-01": ["tests/test_v132.py::test_cd01_workflow_b_18_combinations"],
    "CD-02": ["tests/test_v132.py::test_cd02_workflow_a_ml_tests_every_structure"],
    "CD-03": ["tests/test_v132.py::test_cd03_b_tests_match_reference"],
    "CD-04": ["tests/test_v132.py::test_cd04_time_only_b"],
    "CD-05": ["tests/test_v132.py::test_cd05_explicit_cluster_on_off"],
    "CD-06": ["tests/test_v132.py::test_cd06_reused_ids", "tests/test_v132.py::test_cd06_discovery_local_ids"],
    "CD-07": ["tests/test_v132.py::test_cd07_irregular_time"],
    "CD-08": ["tests/test_v132.py::test_cd08_exported_candidate_code", "tests/test_v132.py::test_identity_cross_module"],
    "CD-09": ["tests/test_v132.py::test_cd09_history_context", "tests/test_v132.py::test_cd09_cache_identity",
              "tests/test_v132.py::test_project_round_trip"],
    "CD-10": ["tests/test_v132.py::test_cd10_unsupported_clustered_families_refused",
              "tests/test_v132.py::test_diag_nlme_near_boundary"],
    "CD-11": ["tests/test_v132.py::test_cd11_validation_targets"],
    "CD-12": ["tests/test_v132_gui.py::test_cd12_cluster_controls", "tests/test_v132.py::test_stat31_cancel"],
    "UI-39": "NOT APPLICABLE: wording-only change (filter context notes); owner manual check M01/M12",
    "UI-40": "NOT APPLICABLE: wording-only change; owner manual check",
    "UI-52": "NOT APPLICABLE: wording-only change; owner manual check",
    "UI-53": "NOT APPLICABLE: plot subtitle text; owner manual check M04",
    "UI-55": "NOT APPLICABLE: wording-only change; owner manual check M08",
    "UI-35": ["tests/test_v13.py"], "UI-36": ["tests/test_v13.py"], "UI-37": ["tests/test_v13.py"],
    "UI-38": ["tests/test_v13.py"], "UI-44": ["tests/test_v131.py::test_33"], "UI-45": ["tests/test_v131.py::test_33"],
}
T133 = "tests/test_v133.py::"
GATES.update({   # v1.3.3 register (03_Defect_Register.csv): new IDs and re-verified inherited ones
    "STAT-47": [T133 + "test_stat47_transfer_refit_plot_and_reml_keep_the_model",
                T133 + "test_spec_diff_ignores_presentation", T133 + "test_fitted_vs_requested_fails_closed"],
    "V133-STATE-01": [T133 + "test_stat47_transfer_refit_plot_and_reml_keep_the_model"],
    "V133-STATE-02": [T133 + "test_stat47_transfer_refit_plot_and_reml_keep_the_model"],
    "V133-STATE-03": [T133 + "test_stat47_transfer_refit_plot_and_reml_keep_the_model"],
    "V133-STATE-04": [T133 + "test_stat47_transfer_refit_plot_and_reml_keep_the_model"],
    "V133-STATE-05": [T133 + "test_custom_term_subset_and_hierarchy_refusal", T133 + "test_term_presets_custom"],
    "V133-STAT-01": [T133 + "test_48c_time_comparison_keeps_ar1_cluster_and_terms"],
    "V133-STAT-02": [T133 + "test_45a_preflight_composite_key_counts"],
    "V133-STAT-03": [T133 + "test_45a_local_ids_fit_matches_reference", "tests/test_v132.py::test_cd05_explicit_cluster"],
    "V133-VAL-01": [T133 + "test_validation_splits_deterministic", T133 + "test_validation_predictions_match",
                    T133 + "test_validation_pipeline_bootstrap", "tests/test_v133_gui.py::test_validation_panel"],
    "V133-EXP-01": [T133 + "test_full_analysis_bundle_runs_relocated"],
    "V133-EXP-02": [T133 + "test_amd_candidate_bundle_manifest_complete"],
    "V133-EXP-03": [T133 + "test_full_analysis_bundle_runs_relocated"],
    "V133-EXP-04": [T133 + "test_amd_candidate_bundle_manifest_complete"],
    "V133-UI-01": ["tests/test_v133_gui.py::test_preflight_labels"],
    "V133-UI-02": ["tests/test_v133_gui.py::test_workflow_b_copy"],
    "V133-UI-03": ["tests/test_v133_gui.py::test_blocked_start_resets_progress"],
    "V133-UI-04": ["tests/test_v133_gui.py::test_workflow_b_copy"],
    "V133-UI-05": ["tests/test_v133_gui.py::test_preflight_labels"],
    "V133-UI-06": ["tests/test_v133_gui.py::test_csv_save_and_interpretation"],
    "V133-UI-07": ["tests/test_v133_gui.py::test_csv_save_and_interpretation"],
    "V133-UI-08": [T133 + "test_51_bh_holm_per_family_match_p_adjust"],
    "V133-UI-09": [T133 + "test_51_independent_enumeration_oracle"],
    "V133-CMP-01": [T133 + "test_glmm_and_clmm_match_independent", "tests/test_v132.py::test_cmp_family_aware"],
    "V133-DIAG-01": [T133 + "test_optimizer_fallback_same_spec", T133 + "test_lr_material_negative_withheld"],
    "V133-HIST-01": [T133 + "test_50_filter_reconstruct", T133 + "test_50d_three_category_coding_change"],
    "V133-DISC-01": [T133 + "test_51_independent_enumeration_oracle", T133 + "test_51_bh_holm_per_family"],
    "V133-QA-01": [T133 + "test_matrix_workflow_a_focal", T133 + "test_matrix_workflow_b_nine",
                   T133 + "test_matrix_standalone_full_analysis"],
    "STAT-22": GATES["STAT-22"] + [T133 + "test_stat47_transfer_refit_plot_and_reml_keep_the_model"],
    "STAT-36": GATES["STAT-36"] + [T133 + "test_full_analysis_bundle_runs_relocated"],
    "STAT-39": GATES["STAT-39"] + [T133 + "test_50_filter_reconstruct"],
    "STAT-31": GATES["STAT-31"] + [T133 + "test_51_cancel_kills_running_batches"],
    "STAT-37": GATES["STAT-37"] + [T133 + "test_51_cancel_kills_running_batches"],
    "V133-HIST-ANCHORS": [T133 + "test_07_application_only_anchors", T133 + "test_fixture_bytes_match_manifest"],
})
GATES["V133-VAL-01"] = GATES["V133-VAL-01"] + [T133 + "test_validation_rows_target", T133 + "test_validation_gee"]
OWNER_ONLY = {"PACKAGED-OFFLINE-VM": "clean Windows VM with the network off (installer + portable): owner step M13",
              "PHYSICAL-DPI-150": "real monitor at 150% Windows scaling: owner step M12 (Qt scale-factor screenshots "
                                  "are the developer proxy)",
              "V133-QA-03": "offline final acceptance on a clean Windows machine (Test 54): owner step in "
                            "docs/v1.3.3/manual_acceptance_v1.3.3.md"}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run(cmd: list[str], log: Path, cwd: Path, env: dict | None = None, timeout: int = 7200) -> int:
    with log.open("w", encoding="utf-8") as fh:
        p = subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT, env=env, timeout=timeout)
    return p.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", type=Path, default=None, help="v1.3.2 developer package (07/08 scripts)")
    ap.add_argument("--package133", type=Path, default=None,
                    help="v1.3.3 developer package (reference/*.R, qa/compare_metrics.py)")
    ap.add_argument("--out", required=True, type=Path, help="parent folder; a fresh run folder is created")
    ap.add_argument("--rscript", default=None, help="Rscript executable (default: the one StatExplore finds)")
    ap.add_argument("--tests", nargs="*", default=["tests"], help="pytest targets")
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    from statexplore import __version__
    from statexplore.rbridge.locate import find_rscript, r_library

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = a.out / f"run_{stamp}"
    out.mkdir(parents=True, exist_ok=False)            # immutable evidence: never reuse a run folder
    rscript = a.rscript or str(find_rscript())
    env = dict(os.environ, R_LIBS=str(r_library()), PYTHONIOENCODING="utf-8", QT_QPA_PLATFORM="offscreen")
    summary: dict = {"run": stamp, "statexplore": __version__, "rscript": rscript, "steps": {}}
    ref = out / "reference"
    gen = fit = audit = 0
    if a.package is not None:
        pkg = a.package.resolve()
        gen = run([rscript, "--vanilla", str(pkg / "07_Generate_Test_Fixtures.R"), str(ref)], out / "07_generation.log",
                  pkg, env)
        fit = run([rscript, "--vanilla", str(pkg / "08_Independent_R_Reference_Models.R"), str(ref)],
                  out / "08_reference.log", pkg, env) if gen == 0 else None
        audit = run([sys.executable, str(pkg / "Verify_Reference_Results.py"), str(ref)], out / "09_audit.log", pkg,
                    env) if (ref / "actual_metrics.csv").exists() else None
    if a.package133 is not None:
        # v1.3.3: re-execute the independent reference scripts in a copy of the package, then check the committed
        # reference output (what tests/test_v133.py compares the application with) against the fresh one using the
        # package's own comparator, and the committed fixtures against the package bytes
        import shutil

        p133 = out / "package133"
        shutil.copytree(a.package133.resolve(), p133, ignore=shutil.ignore_patterns("output"))
        steps = {}
        for name in ("generate_fixtures", "reference_fits", "validation_reference", "non_gaussian_reference"):
            steps[name] = run([rscript, "--vanilla", f"reference/{name}.R"], out / f"133_{name}.log", p133, env)
        committed = ROOT / "tests" / "fixtures" / "v133" / "reference" / "reference_metrics.csv"
        steps["compare_committed_vs_fresh"] = run(
            [sys.executable, str(p133 / "qa" / "compare_metrics.py"), str(p133 / "reference" / "output" /
             "reference_metrics.csv"), str(committed), str(out / "133_reference_reproducibility.csv")],
            out / "133_compare.log", p133, env)
        same133 = {}
        for f in sorted((ROOT / "tests" / "fixtures" / "v133").glob("*.csv")):
            src = p133 / "fixtures" / f.name if (p133 / "fixtures" / f.name).exists() else p133 / f.name
            same133[f.name] = src.exists() and sha256(f) == sha256(src)
        for f in sorted((ROOT / "tests" / "fixtures" / "v133" / "reference").glob("Test*.csv")):
            g = p133 / "reference" / "output" / f.name
            same133[f"reference/{f.name}"] = g.exists() and sha256(g) == sha256(f)
        steps["fixtures_identical"] = 0 if all(same133.values()) else 1
        summary["v133_reference"] = steps
        summary["v133_fixture_bytes"] = same133
        if any(v not in (0, None) for v in steps.values()):
            audit = audit or 1
    summary["steps"]["fixtures"] = gen
    summary["steps"]["references"] = fit            # 2 = some reference fit failed in R (listed in the log)
    summary["steps"]["historical_audit"] = audit
    # the shipped fixtures must be the bytes the references were computed from
    shipped = ROOT / "tests" / "fixtures" / "v132"
    same = {}
    if a.package is not None:
        for f in sorted(shipped.glob("*.csv")):
            g = ref / "fixtures" / f.name
            same[f.name] = g.exists() and sha256(g) == sha256(f)
    summary["fixtures_identical_to_fresh_generation"] = same
    junit = out / "pytest_junit.xml"
    rc = run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={junit}",
              f"--basetemp={out / 'pytest_tmp'}", *a.tests], out / "pytest.log", ROOT, env)
    summary["steps"]["pytest"] = rc
    results: dict[str, str] = {}
    if junit.exists():
        for tc in ET.parse(junit).getroot().iter("testcase"):
            node = f"{tc.get('file') or tc.get('classname', '').replace('.', '/') + '.py'}::{tc.get('name')}"
            node = node.replace("\\", "/")
            status = "PASS"
            if tc.find("failure") is not None or tc.find("error") is not None:
                status = "FAIL"
            elif tc.find("skipped") is not None:
                status = "SKIPPED"
            results[node] = status
    rows = []
    for gid, spec in GATES.items():
        if isinstance(spec, str):
            rows.append({"id": gid, "status": "NOT APPLICABLE", "tests": "", "detail": spec})
            continue
        matched = {n: s for n, s in results.items() if any(n.startswith(p) for p in spec)}
        missing = [p for p in spec if not any(n.startswith(p) for n in results)]
        if missing:
            status = "NOT RUN"
        elif any(s == "FAIL" for s in matched.values()):
            status = "FAIL"
        elif any(s == "SKIPPED" for s in matched.values()):
            status = "SKIPPED"
        else:
            status = "PASS"
        rows.append({"id": gid, "status": status, "tests": f"{len(matched)} tests",
                     "detail": "missing: " + "; ".join(missing) if missing else
                     "; ".join(f"{n}={s}" for n, s in matched.items() if s != "PASS")})
    for gid, why in OWNER_ONLY.items():
        rows.append({"id": gid, "status": "NOT RUN", "tests": "", "detail": why})
    with (out / "gates.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "status", "tests", "detail"])
        w.writeheader()
        w.writerows(rows)
    counts: dict[str, int] = {}
    for s in results.values():
        counts[s] = counts.get(s, 0) + 1
    summary["tests_by_status"] = counts
    summary["gates_by_status"] = {s: sum(r["status"] == s for r in rows) for s in
                                  ("PASS", "FAIL", "BLOCKED", "SKIPPED", "NOT RUN", "NOT APPLICABLE")}
    mandatory_bad = [r["id"] for r in rows if r["status"] in ("FAIL", "BLOCKED", "SKIPPED") or
                     (r["status"] == "NOT RUN" and r["id"] not in OWNER_ONLY)]
    summary["mandatory_not_passed"] = mandatory_bad
    summary["owner_steps_not_run"] = list(OWNER_ONLY)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    ok = not mandatory_bad and gen == 0 and audit == 0 and rc == 0
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
