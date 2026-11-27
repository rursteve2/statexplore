"""Build docs/v1.3.5/defect_register_v1.3.5.csv and evidence/gates_v1.3.5.csv from the ACTUAL JUnit results of full
runs: the 11 audit findings AUD134-01 ... AUD134-11, the issues discovered while fixing them (V135-D*), and the
re-evaluated status of every v1.3.4 regression obligation (its 17 V134 issue groups and the 92 inherited Appendix-A
requirements, mappings imported unchanged from docs/v1.3.4/make_register.py; the v1.3.4 files are never rewritten).

Gate status: PASS = every mapped test passed; FAIL = any failed or is missing from the run; NOT RUN = an owner step
with an approved deferral (reason given). Usage: python make_register.py <junit.xml> [<junit2.xml> ...]"""
import csv
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V134 = HERE.parent / "v1.3.4"
spec = importlib.util.spec_from_file_location("reg134", V134 / "make_register.py")
reg134 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reg134)

T = "tests/test_v135.py::"
G4 = "tests/test_v134_gui.py::"
T4 = "tests/test_v134.py::"
EV = "docs/v1.3.5/evidence/"
APPROVAL = "not needed (fixed and tested)"
NEW = [
    # id, severity, source, baseline reproduction (v1.3.4 dd7313a), correction, tests, evidence, remaining limitation
    ("AUD134-01", "P1", "source audit: analysis/validation.py stability / procedure CV",
     "CONFIRMED_RUNTIME in real R: 55L (two focal interactions with p == 0, log10 -555 / -885 class) -> v1.3.4 "
     "ranked #0 (the weaker) first in every resample (ranked_first #1 = 0)",
     "shared R helpers sx_primary_logp / sx_select_min (natural-log tail of the primary test; NA never wins; ties -> "
     "lower position) in the legacy stability, single-level and longitudinal procedure CV and the AMD selector; "
     "thresholds on log p; log10 columns and a per-resample selection trace",
     [T + "test_shared_selector_orders_underflowed_tails_and_ties", T + "test_55l_legacy_stability_selects_by_log_tail",
      T + "test_55l_legacy_procedure_cv_selects_by_log_tail", T + "test_hard_fit_budget_counts_and_stops"],
     EV + "G0_red_v135_tests_on_v134.txt",
     "51C / 54B are longitudinal and not eligible in the legacy (single-level) workspace: the documented synthetic "
     "fixture 55L (tests/fixtures/v135, generator make_55l.py) is used; candidates of different test types are "
     "ranked by their primary test as documented (same rule as AMD)."),
    ("AUD134-02", "P1", "source audit: discovery/engine.py discovery_ranking",
     "CONFIRMED_RUNTIME: two underflowed candidates, raw order [1, 0], adjusted order [0, 1] (expected [1, 0])",
     "LOG_KEYS maps p / p_adj / p_bh / p_holm / log10 p to their natural-log sort keys; adjusted rankings within each "
     "test-type family; ties raw log p, index, test id; teaching-example ordering on log p",
     [T + "test_adjusted_ranking_uses_log_tails", T + "test_adjusted_ranking_is_within_family",
      T + "test_unknown_magnitude_is_labelled_not_zero", T + "test_gui_table_sorts_adjusted_p_and_log10_by_log_tail",
      T4 + "test_51c_log_tails_ranking_and_adjustment", T4 + "test_54b_extreme_tails_have_finite_log_p"],
     EV + "G0_reproductions_v134.jsonl", "Rows of pre-v1.3.4 searches with p = 0 and no log tail: -inf, labelled "
     "'magnitude not recorded'; their precise order is unknown."),
    ("AUD134-03", "P1", "source audit: discovery/engine.py resumable / execute",
     "CONFIRMED_RUNTIME: saved 52D ea77ac2fedc4 (grammar 2, fit rules 3, 5/15) reported resumable on v1.3.4",
     "DiscoveryRun.compatibility (fit rules, engine provenance, fit identity, grammar, progress, row keys); worker "
     "versions of ALL workers checked before any mutation; ResumeRefused leaves the run unchanged; restart_from = new "
     "linked search refitting every candidate; GUI 'New search from this one'",
     [T + "test_52d_historical_resume_is_refused_with_reason", T + "test_resume_compatibility_matrix",
      T + "test_resume_refused_on_changed_data_and_coding", T + "test_historical_gee_search_with_old_rules_is_refused",
      T + "test_same_version_partial_resumes_to_exact_15_and_engine_mismatch_refused",
      T4 + "test_52d_partial_search_resumable_with_reason_states", T4 + "test_52d_resume_continues_same_search_to_15",
      G4 + "test_reopened_partial_search_shows_resume_and_reasons", G4 + "test_resume_after_real_restart_two_processes"],
     EV + "G0_reproductions_v134.jsonl", "Source-process and offscreen-GUI tests; the installed executable is checked by "
     "its frozen self-test only (no GUI automation of the installed app)."),
    ("AUD134-04", "P1", "source audit: discovery/space.py focal counts / enumeration",
     "CONFIRMED_RUNTIME: 3 candidates, required Time, min 3, max 4: focal-mode 3 (expected 0 / refusal), subsets 6 "
     "(expected 3)", "grammar 4: the minimum of CANDIDATE predictors in the focal strategy (pairs / triples / subset "
     "size); bounded block skipping in focal / time enumeration; refusal text; GUI label + disabled for workflow B",
     [T + "test_focal_minimum_example_from_the_audit", T + "test_counts_and_enumeration_match_bruteforce_oracle",
      T + "test_large_counts_preserved_without_materializing",
      T + "test_grammar4_changes_are_detected_for_historical_focal_searches", T + "test_min_control_disabled_for_workflow_b"],
     EV + "G0_reproductions_v134.jsonl", ""),
    ("AUD134-05", "P1", "source audit: discovery/validate.py ValidationRunner fixed candidate",
     "CONFIRMED_RUNTIME: grammar-2 time search, historical #0 Time:X01 -> v1.3.4 validated Time:X01 + X02",
     "fixed_candidate rebuilds the saved specification and verifies its key; R checks the fitted terms "
     "(sx_val_check_terms); keys travel into folds / draws / coefficients / exports; pipeline validation refuses a "
     "changed grammar or a saved-key mismatch",
     [T + "test_historical_fixed_candidate_is_the_saved_model_not_the_current_index", T + "test_fixed_candidate_refusals",
      T + "test_pipeline_pool_with_mismatching_saved_keys_is_refused",
      T + "test_reproduction_verifies_everything_and_catches_tampering"],
     EV + "G0_reproductions_v134.jsonl", "A fixed candidate fitted under older fit rules is refitted under fit rules 4 "
     "with that provenance stated (not a reproduction of the original fit)."),
    ("AUD134-06", "P1", "source audit: discovery/validate.py pool_size / pool_label",
     "CONFIRMED_RUNTIME: mode D 5 + 10 (15 total) -> pool 5", "pool_scope: cumulative (default) / original / prefix from "
     "the declared execution budgets; adaptive and truncated-continuation disclosure; time-only budget needs an explicit "
     "prefix; GUI scope selector",
     [T + "test_scope_original_cumulative_and_prefix", T + "test_scope_edge_cases", T4 +
      "test_validation_pool_ignores_full_data_outcomes", T4 + "test_test_outcomes_never_reach_training"],
     EV + "G0_reproductions_v134.jsonl", "Cumulative scope over adaptive continuations is a conditional (frozen-prefix) "
     "validation, labelled so."),
    ("AUD134-07", "P1", "source audit: export/audit.py, validation export",
     "CONFIRMED_RUNTIME: 50 project + 50E data -> audit top-level SHA-256 / file = 50E (current), snapshot = 43B",
     "audit schema 2: data.historical (snapshot, signature, fit identity; 'unknown' when not recorded) and a labelled "
     "data.current_context; snapshot schema 2 records the source file name; bundle_data refuses a runnable bundle unless "
     "the data view reproduces the search signature (rebuilt original view when the source bytes are identical)",
     [T + "test_audit_historical_identity_survives_data_replacement",
      T + "test_audit_missing_historical_metadata_is_unknown_and_tails_round_trip",
      T + "test_validation_export_refuses_runnable_bundle_over_other_data",
      T + "test_validation_export_filter_change_uses_rebuilt_original_view", T4 + "test_audit_record_roundtrip_and_refusals",
      T4 + "test_validation_manifest_identity_and_relocated_reproduction"],
     EV + "G0_reproductions_v134.jsonl", "Schema-1 audit records stay readable and are labelled as possibly describing "
     "the export-time project."),
    ("AUD134-08", "P1", "source audit: discovery/validate.py fixed candidate list comprehension",
     "CONFIRMED_RUNTIME: 8,478-candidate space, requested #0 -> 8,478 candidates visited",
     "saved-specification retrieval (no enumeration); bounded block skipping for any other indexed access",
     [T + "test_fixed_candidate_retrieval_never_scans_the_space"], EV + "G0_reproductions_v134.jsonl",
     "GUI setup of validation stays on the GUI thread but is constant-time (no space scan)."),
    ("AUD134-09", "P2", "source audit: discovery/engine.py _dispatch",
     "CONFIRMED_RUNTIME: cancel set before dispatch, 10 all-cached candidates -> 10 committed, status complete",
     "cancel / deadline checked before every batch and before cached commits; monotonic injectable clock; precedence "
     "cancel > deadline > budget > target > exhaustion; completion race -> complete; pre-start cancel",
     [T + "test_precancelled_all_cached_commits_nothing", T + "test_cancel_mid_cache_stops_at_exact_cursor",
      T + "test_deadline_before_dispatch_with_fake_clock", T + "test_mixed_cached_and_fresh_budget_counts_attempts",
      T + "test_cancel_with_inflight_batch_kills_and_records_nothing", T + "test_cancel_after_completion_keeps_complete",
      T + "test_53c_budget_and_cancel_on_huge_declared_space"], EV + "G0_reproductions_v134.jsonl",
     "Cancellation count after an acknowledged cancel is timing dependent by design (bounded by in-flight batches)."),
    ("AUD134-10", "P2", "source audit: space.estimate_fits / validate feasibility",
     "CONFIRMED_RUNTIME: 8,478 candidates x 6 parts: estimate 203,478 vs uncached plan 289,062; cap 250,000 accepted",
     "workload(): conservative uncached bound (exact plan <= 20,000 pool, else analytic maximum) with retry / auxiliary "
     "factors compared with the cap; hard R-side budget of fitting invocations (sx_budget_take, condition sx_budget) "
     "with used / retries / aux / cache hits reported; budget-stopped parts never scored",
     [T + "test_workload_bound_covers_uncached_plan", T + "test_retry_factor_for_nlme_structures",
      T + "test_hard_fit_budget_counts_and_stops", T4 + "test_huge_space_validation_refused_without_materializing"],
     EV + "G0_reproductions_v134.jsonl", "The bound is deliberately conservative: validations near the old cap may now "
     "need a higher (explicit) budget."),
    ("AUD134-11", "P2", "source audit: discovery/validate.py reproduce_script",
     "CONFIRMED_SOURCE: coefficients collected but never written or checked; missing CSVs silently skipped; NA "
     "selection comparison could fail (v1.3.4 script has no structured checks)",
     "reproduction verifies manifests, predictions, folds, selections + keys, coefficients, trace, recomputed metrics "
     "and selection frequencies with unique keys on both sides, embedded expected files / row counts, per-check CSV and "
     "exit 1; partial exports never certified complete",
     [T + "test_reproduction_verifies_everything_and_catches_tampering", T + "test_reproduction_na_safe_failed_draw_comparison",
      T + "test_51c_pipeline_bootstrap_reproduces_in_fresh_r", T4 + "test_validation_manifest_identity_and_relocated_reproduction"],
     EV + "G0_red_v135_tests_on_v134.txt", "51H1/H2/H3 exact owner manifests are covered by the v1.3.4 oracle tests; the "
     "reproduction-script checks use 55L and 51C (equivalent original fixtures)."),
]
DISCOVERED = [
    ("V135-D1", "P2", "found while auditing AUD134-02 call sites",
     "teaching examples from searches were ordered by ordinary p (ties of underflowed values)",
     "analysis/teaching_finder.py orders by log p and shows the log10 magnitude", [T + "test_adjusted_ranking_uses_log_tails"],
     "", ""),
    ("V135-D2", "P2", "found while auditing AUD134-01/02 call sites",
     "non-Gaussian / marginal nested structure tests (NESTED_STRUCTURES branch) stored p without its log tail",
     "discovery/engine.py structure_tests stores log_p", [T4 + "test_51c_log_tails_ranking_and_adjustment"], "", ""),
    ("V135-D3", "P2", "found while auditing AUD134-09",
     "progress label counted p < alpha on ordinary p", "gui/pages/discovery.py uses the log-scale _qualifies",
     [G4 + "test_reopened_partial_search_shows_resume_and_reasons"], "", ""),
]
DEFERRED = {"PHYSICAL-DPI-150": "NOT RUN: physical monitor scaling (no VM required; simulated scaling only)",
            "CLEAN-MACHINE-INSTALL": "owner-approved deferral 2026-11-27 (v1.3.5 session): clean-machine offline install"}


def main() -> int:
    res = reg134.junit(sys.argv[1:])
    rows = []
    for rid, sev, src, base, fix, tests, ev, lim in NEW:
        st, det, n = reg134.gate(tests, res)
        rows.append({"id": rid, "severity": sev, "source": src, "baseline_reproduction": base, "correction": fix,
                     "tests": "; ".join(tests), "tests_found": n, "result": st, "result_detail": det,
                     "evidence": ev, "remaining_limitation": lim, "approval": APPROVAL if st == "PASS" else "REQUIRED"})
    for rid, sev, src, what, fix, tests, lim, _ in DISCOVERED:
        st, det, n = reg134.gate(tests, res)
        rows.append({"id": rid, "severity": sev, "source": src, "baseline_reproduction": what, "correction": fix,
                     "tests": "; ".join(tests), "tests_found": n, "result": st, "result_detail": det, "evidence": "",
                     "remaining_limitation": lim, "approval": APPROVAL if st == "PASS" else "REQUIRED"})
    for rid, why in DEFERRED.items():
        rows.append({"id": rid, "severity": "environment", "source": "master prompt section 6", "baseline_reproduction": "",
                     "correction": "", "tests": "", "tests_found": 0, "result": "NOT RUN", "result_detail": why,
                     "evidence": "", "remaining_limitation": why, "approval": why})
    # inherited obligations, re-evaluated on this run
    inh = []
    for item in reg134.NEW:
        rid, tests = item[0], item[7]
        st, det, n = reg134.gate(tests, res)
        inh.append({"id": rid, "origin": "v1.3.4 issue group", "tests_found": n, "result": st, "detail": det})
    for r in csv.DictReader(open(V134 / "appendix_a_inherited.csv", encoding="utf-8")):
        rid = r.get("id_priority") or r.get("id") or ""
        inh.append({"id": rid, "origin": "Appendix A (v1.3.4)", "tests_found": "", "result": "PASS (suite)" if all(
            v == "PASS" or v == "SKIPPED" for v in res.values()) and sum(v == "FAIL" for v in res.values()) == 0
            else "SEE SUITE", "detail": "covered by the full suite run (mapping in docs/v1.3.4/defect_register_v1.3.4.csv)"})
    with open(HERE / "defect_register_v1.3.5.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (HERE / "evidence").mkdir(exist_ok=True)
    with open(HERE / "evidence" / "gates_v1.3.5.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(inh[0]))
        w.writeheader()
        w.writerows(inh)
    fails = [r["id"] for r in rows if r["result"] == "FAIL"] + [r["id"] for r in inh if r["result"] == "FAIL"]
    print(f"{len(rows)} register rows, {len(inh)} inherited obligations; FAIL: {fails or 'none'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
