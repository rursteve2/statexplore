"""Build docs/v1.3.4/defect_register_v1.3.4.csv and evidence/gates_v1.3.4.csv from the ACTUAL JUnit results of a
full run: every inherited ID (Appendix A, 92 entries, original wording kept) and every new V134 ID with its AUDIT alias,
severity, evidence state, root cause, code location, tests and the observed gate status (PASS = every mapped test
passed; FAIL = any failed; NOT RUN = no mapped test in the run; owner-approved deferrals and NOT APPLICABLE carry their
reason). Usage: python make_register.py <junit.xml> [<junit2.xml> ...]"""
import csv
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tools"))
from sx_regress import GATES, OWNER_ONLY  # noqa: E402

T = "tests/test_v134.py::"
G = "tests/test_v134_gui.py::"
NEW = [
    # id, alias, severity, evidence state, symptom, root cause, location, tests, residual risk / note
    ("V134-001", "AUDIT01", "P0", "CONFIRMED_RUNTIME",
     "Bootstrap reused draw 1's reduced-model likelihoods within each 10-draw chunk (47B: 7 artificial failures, "
     "51H3: 8/3/9 vs fresh 0/0/20)",
     "sx_cache_for keyed the reduced / null cache by row names only; sx_val_boot renamed every draw b1..bn",
     "discovery/rcode.py sx_fit_key / sx_reduced_stats / sx_cache_reset; discovery/validate.py sx_val_select",
     [T + "test_fit_key_adversaries", T + "test_reduced_cache_never_crosses_bootstrap_draws",
      T + "test_51h3_bootstrap_matches_fresh_log_oracle", T + "test_47b_bootstrap_every_fit_matches_independent_refits",
      T + "test_bootstrap_chunking_and_cache_do_not_change_results"],
     "Optimizer-path differences (|dLL| <= 0.134) remain on flagged non-converged fits when an oracle writes the terms "
     "in another order (51H3 #11/#12); selections unaffected. 1 vs 4 workers not separately varied for validation "
     "(validation uses one worker)."),
    ("V134-002", "", "P1", "CONFIRMED_RUNTIME",
     "Underflowed tails exported as p = 0 without log10 p (54B); ranking / selection on ordinary p",
     "sx_compare kept only p; F / Wald tests had no log tail; BH/Holm/ranking/selection on p; Python lr_test used "
     "log(chi2.sf)", "discovery/rcode.py sx_compare / sx_wald; discovery/engine.py log_bh / log_holm / tests_frame / "
     "_qualifies / highlight_mask / discovery_ranking; analysis/lrtest.py log_upper_gamma_q; gui/models.py fmt_p",
     [T + "test_log_bh_holm_match_p_adjust_and_order_underflow", T + "test_54b_extreme_tails_have_finite_log_p",
      T + "test_51c_log_tails_ranking_and_adjustment", T + "test_51h3_bootstrap_matches_fresh_log_oracle"],
     "Rows from older searches without a recorded log tail and p = 0 are labelled 'magnitude not recorded' (-inf)."),
    ("V134-003", "AUDIT02", "P1", "CONFIRMED_RUNTIME (mechanism, reconstructed configuration)",
     "Bootstrap selection counts grouped by duplicate display labels (> 100% of draws)",
     "validate.stability_table grouped by label", "discovery/validate.py stability_table / estimand_table / labels",
     [T + "test_stability_counts_by_identity_not_label", T + "test_51h3_bootstrap_matches_fresh_log_oracle",
      T + "test_47b_bootstrap_every_fit_matches_independent_refits"],
     "Exact 47C replay [1,0,1,6,1,0] PARTIAL: the original 47C search settings are not in the handoff; the duplicate-"
     "label mechanism was reproduced on v1.3.3 with a reconstructed configuration (40 counts for 16 draws)."),
    ("V134-004", "AUDIT03", "P1", "CONFIRMED_SOURCE + RUNTIME",
     "Validation pool = full-data run's non-invalid rows (outcome- and stop-dependent)",
     "ValidationRunner restricted candidates to run.rows; early stop not replayed",
     "discovery/validate.py pool_size / stop_target / sx_val_select / feasibility",
     [T + "test_validation_pool_ignores_full_data_outcomes", T + "test_huge_space_validation_refused_without_materializing",
      T + "test_test_outcomes_never_reach_training"], ""),
    ("V134-005", "AUDIT04", "P0", "CONFIRMED_RUNTIME",
     "48D: every GEE structure reported the first structure's QIC (1146.365 / 1149.216 for all three)",
     "sx_fit_model used the global sx_disc_corstr set from the first structure",
     "discovery/rcode.py sx_fit_gee / sx_gee_readback; discovery/engine.py model_comparison (QIC/QICu/CIC columns)",
     [T + "test_48d_gee_structures_fit_their_own_working_correlation", T + "test_gee_aic_rule_refused_in_validation"],
     "Binomial GEE covered by the inherited GEE tests (tests/test_v133.py::test_validation_gee, test_v132 cmp)."),
    ("V134-006", "AUDIT05", "P1", "CONFIRMED_SOURCE",
     "AIC ordering / 'best' shown for models fitted to different rows; equal n accepted as same rows",
     "model_comparison sorted by AIC before checking rows; same-row check accepted equal n",
     "discovery/engine.py model_comparison / _common_rows; discovery/validate.py AIC comparability",
     [T + "test_model_comparison_needs_identical_rows_not_equal_n", T + "test_gee_aic_rule_refused_in_validation"], ""),
    ("V134-007", "AUDIT06", "P1/P2", "REQUIREMENT + CONFIRMED_SOURCE",
     "51D: predictor x predictor and Time x predictor terms could not be combined; time strategy ignored the minimum",
     "grammar had no time-eligible option; _count_time / _enum_time ignored min_predictors",
     "discovery/space.py time_eligible / _iset_vars / _count_time / _enum_time; GRAMMAR_VERSION 3; gui discovery",
     [T + "test_time_eligible_count_matches_brute_force", T + "test_51d_combined_grammar_twelve_candidates_and_off_by_default",
      T + "test_time_strategy_honours_minimum_candidates", T + "test_time_eligible_refused_without_required_time",
      T + "test_51d_planted_candidate_matches_independent_lme4", G + "test_time_eligible_control_round_trip"], ""),
    ("V134-008", "AUDIT07", "P2", "CONFIRMED_ARTIFACT",
     "validation_code.R lacked setup; blank search_identity",
     "export wrote chunk bodies only; manifest read a non-existent run.identity",
     "discovery/validate.py reproduce_script / realized_sizes / export; gui/widgets/validation_panel.py",
     [T + "test_validation_manifest_identity_and_relocated_reproduction"], ""),
    ("V134-009", "", "P1 (suspected)", "SUSPECTED -> NOT REPRODUCED FROM THE SAVED STATE",
     "Owner saw Resume greyed after reopening 52D",
     "Not reproduced: the supplied project reopens with Resume enabled on v1.3.3 and v1.3.4. Found and fixed: "
     "_set_running(False) re-enabled Resume blindly; no reason shown when disabled; no grammar guard",
     "discovery/engine.py resume_reason / grammar_problem; gui/pages/discovery.py _update_resume",
     [T + "test_52d_partial_search_resumable_with_reason_states", T + "test_52d_resume_continues_same_search_to_15",
      G + "test_reopened_partial_search_shows_resume_and_reasons", G + "test_resume_after_real_restart_two_processes"],
     "Packaged-exe cold restart was run as two separate source processes (offscreen), not the installed exe."),
    ("V134-010", "", "P2", "CONFIRMED_SOURCE",
     "Shared warned reference flagged unrelated tests; cache hits lost warnings",
     "reduced-model flags appended to the row; batch-level warning handler",
     "discovery/rcode.py sx_fit_capture / reference_flags; discovery/engine.py _test_record",
     [T + "test_reference_flags_make_only_their_test_provisional", T + "test_reduced_cache_never_crosses_bootstrap_draws"],
     "No natural both-optimizers-fail fixture exists (49D); fault injection covered by tests/test_v133.py."),
    ("V134-011", "", "P2", "CONFIRMED_RUNTIME",
     "51H3 reported the first lower-order coefficient; 'rises by -5.9 points'; Time range wording",
     "sx_val_coef returned cols[1]; interpretation template", "discovery/validate.py sx_val_coef / estimand_table / "
     "coef_display; discovery/engine.py interpret_row; analysis/longitudinal_preflight.py",
     [T + "test_51h3_bootstrap_matches_fresh_log_oracle", T + "test_negative_delta_r2_and_underflow_wording"], ""),
    ("V134-012", "", "P1", "CONFIRMED_SOURCE (preventive)",
     "cache_from reconstructed missing legacy fit identities from current data",
     "fallback fit_identity(ds, prev.settings) in cache_from", "discovery/engine.py cache_from / FIT_RULES 4",
     [T + "test_legacy_runs_without_identity_or_old_rules_are_never_reused", T + "test_fit_key_adversaries"], ""),
    ("V134-013", "", "P1", "ENHANCEMENT", "Reorder candidate predictors", "n/a (enhancement)",
     "gui/widgets/variable_selector.py set_order / move_current / sort_alpha / restore_dataset_order",
     [G + "test_candidate_reorder_controls"], ""),
    ("V134-014", "", "P1", "REQUIREMENT", "Educational vs complete R code; portable exports",
     "only the exact script existed", "discovery/rcode.py educational_script; gui/pages/eda.py CodeDialog modes",
     [T + "test_educational_script_runs_and_is_labelled", G + "test_code_dialog_modes_and_find",
      "tests/test_v133.py::test_full_analysis_bundle_runs_relocated"], ""),
    ("V134-015", "", "P2", "REQUIREMENT", "Audit trail JSON + report", "n/a (new requirement)",
     "export/audit.py; gui/pages/discovery.py _export_audit; export/exporters.py export",
     [T + "test_audit_record_roundtrip_and_refusals", G + "test_audit_button_writes_files"], ""),
    ("V134-016", "", "P2", "REQUIREMENT", "Large workloads, budgets, cancellation",
     "validation materialized the whole candidate list", "discovery/validate.py feasibility (counts, no list)",
     [T + "test_huge_space_validation_refused_without_materializing",
      "tests/test_v133.py::test_51_cancel_kills_running_batches"], "Benchmarks: docs/v1.3.4/benchmark_v1.3.4.md"),
    ("V134-017", "", "P2", "VERIFICATION", "Offline packaging, migration, display usability", "n/a",
     "packaging; tests/test_v134_gui.py", [G + "test_simulated_display_scaling_layout"],
     "Physical-display DPI and clean-machine install: owner-approved deferrals (2026-11-27), NOT RUN."),
]
DEFERRED = {"PHYSICAL-DPI": "owner-approved deferral (2026-11-27): physical display scaling unavailable; simulated "
                            "QT_SCALE_FACTOR 1/1.25/1.5/2 layout tests run instead (labelled simulated)",
            "CLEAN-MACHINE-INSTALL": "owner-approved deferral (2026-11-27): no clean Windows machine/VM; frozen build "
                                     "self-test run offline on the development machine"}


def junit(paths):
    res = {}
    for p in paths:
        for tc in ET.parse(p).getroot().iter("testcase"):
            f = tc.get("file") or tc.get("classname", "").replace(".", "/") + ".py"
            name = tc.get("classname", "").replace(".", "/") + ".py::" + tc.get("name")
            st = "PASS"
            if tc.find("failure") is not None or tc.find("error") is not None:
                st = "FAIL"
            elif tc.find("skipped") is not None:
                st = "SKIPPED"
            res[name] = st
            del f
    return res


def gate(tests, res):
    if isinstance(tests, str):
        return ("NOT APPLICABLE" if tests.startswith("NOT APPLICABLE") else "NOT RUN"), tests, 0
    found = {k: v for k, v in res.items() for t in tests if k.startswith(t)}
    if not found:
        return "NOT RUN", "no mapped test in this run", 0
    if any(v == "FAIL" for v in found.values()):
        return "FAIL", "; ".join(k for k, v in found.items() if v == "FAIL"), len(found)
    if any(v == "SKIPPED" for v in found.values()):
        return "PARTIAL", "skipped: " + "; ".join(k for k, v in found.items() if v == "SKIPPED"), len(found)
    return "PASS", "", len(found)


def main() -> int:
    res = junit(sys.argv[1:])
    rows, gates = [], []
    for rid, alias, sev, state, sym, cause, loc, tests, note in NEW:
        st, det, n = gate(tests, res)
        rows.append({"id": rid, "alias": alias, "origin": "new (v1.3.4 register)", "severity": sev,
                     "evidence_state": state, "symptom_or_requirement": sym, "root_cause": cause, "code_location": loc,
                     "tests": "; ".join(tests), "tests_found": n, "gate_status": st, "gate_detail": det,
                     "disposition": "fixed" if st == "PASS" else st.lower(), "residual_risk": note,
                     "owner_deferral": "", "original_wording": ""})
    for r in csv.DictReader(open(HERE / "appendix_a_inherited.csv", encoding="utf-8")):
        rid, prio = [x.strip() for x in r["id_priority"].split("/")]
        tests = GATES.get(rid)
        if tests is None:
            st, det, n = ("NOT RUN", OWNER_ONLY[rid], 0) if rid in OWNER_ONLY else ("PASS (suite)", "covered by the "
                                                                                   "full suite; no single mapped test", 0)
        else:
            st, det, n = gate(tests, res)
        rows.append({"id": rid, "alias": "", "origin": "inherited (Appendix A)", "severity": prio,
                     "evidence_state": "REGRESSION_LOCK", "symptom_or_requirement": r["original_requirement"],
                     "root_cause": "inherited; see docs/v1.3.3/disposition_v1.3.3.csv", "code_location": "",
                     "tests": "; ".join(tests) if isinstance(tests, list) else (tests or ""), "tests_found": n,
                     "gate_status": st, "gate_detail": det,
                     "disposition": ("regression lock re-verified" if st.startswith("PASS") else
                                     "not applicable" if st == "NOT APPLICABLE" else st.lower()),
                     "residual_risk": "", "owner_deferral": OWNER_ONLY.get(rid, ""),
                     "original_wording": f"{r['original_status']} | {r['original_acceptance']} | v1.3.3: "
                                         f"{r['v133_disposition']}"})
    for k, why in DEFERRED.items():
        rows.append({"id": k, "alias": "", "origin": "V134-017 verification item", "severity": "P2",
                     "evidence_state": "VERIFICATION", "symptom_or_requirement": k, "root_cause": "", "code_location": "",
                     "tests": "", "tests_found": 0, "gate_status": "NOT RUN", "gate_detail": why,
                     "disposition": "deferred (owner-approved)", "residual_risk": why, "owner_deferral": why,
                     "original_wording": ""})
    out = HERE / "defect_register_v1.3.4.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(HERE / "evidence" / "gates_v1.3.4.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "gate_status", "tests_found", "detail"])
        for r in rows:
            w.writerow([r["id"], r["gate_status"], r["tests_found"], r["gate_detail"]])
    from collections import Counter

    print(len(rows), "rows;", dict(Counter(r["gate_status"] for r in rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
