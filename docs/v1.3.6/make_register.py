"""Build the v1.3.6 defect register and requirement traceability from the ACTUAL JUnit results of full runs.

Outputs (the v1.3.4 / v1.3.5 registers are never rewritten):
  defect_register_v1.3.6.csv   7 AUD135 findings, V136-D* found during the work, 11 AUD134 + 3 V135-D (re-gated),
                               17 V134 groups + 92 Appendix-A requirements (re-gated), environment items
  traceability_v1.3.6.csv      one row per (requirement, EXACT JUnit node id incl. parameters) with that node's status
Links: every mapped name is resolved against the run's node ids; a shortened name (a function prefix, a module) is
EXPANDED to all the exact nodes it matches and they are all listed - no prefix stands in for a test. Inherited mappings
come from the independent audit's round-3 reconciliation (evidence/round3_traceability_input.csv) with explicit v1.3.6
overrides for the 12 requirements that had no named test and the 5 with ambiguous shorthand.

Gate: PASS = every exact node passed; FAIL = any failed / errored or nothing matched; NOT RUN = an environment item
without a test (owner step; approval stated). Usage: python make_register.py <junit.xml> [<junit2.xml> ...]"""
import csv
import importlib.util
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("reg135", HERE.parent / "v1.3.5" / "make_register.py")
reg135 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reg135)

T = "tests/test_v136.py::"
G = "tests/test_v136_gui.py::"
TR = "tests/test_v136_trace.py::"
EV = "docs/v1.3.6/evidence/"
OK = "not needed (fixed and tested)"

NEW = [
    ("AUD135-01", "P2", "independent audit round 2 (owner R diagnostic 9/20)",
     "R-CONFIRMED (owner, R 4.6.2): NUM-04..09 and ID-01 FAIL; real-R red on the v1.3.5 GENERATED script: a "
     "'0'-prefixed participant id and other same-row-count edits accepted (exit 0) - " + EV + "G0_red_tamper_v135.json",
     "typed evidence schema 1: Python writes every evidence CSV from a declared schema (fixed columns, round-trip "
     "floats, Inf/-Inf, header-only when empty); the generated R reads text only and parses per declared type "
     "(finite / +Inf / -Inf / missing / malformed distinct, signed infinities, exact integers and text, missing only "
     "where the field / reproduced status allows), length-prefixed keys, NA -> FAIL, NOT APPLICABLE only when declared",
     [T + "test_owner_diagnostic_num_id_sel_cases_on_generated_verifier", T + "test_evidence_csv_round_trip_into_r",
      T + "test_each_field_family_tamper_fails_with_a_named_check",
      "tests/test_v135.py::test_reproduction_na_safe_failed_draw_comparison"],
     EV + "G0_red_tamper_v135.json; " + EV + "G0_owner_R_results_v135.csv",
     "Legacy (v1.3.5 or earlier) validation records are exported results-only: their scripts used the old verifier."),
    ("AUD135-02", "P2", "independent audit round 2 (SUM-04..07 R-confirmed)",
     "R-CONFIRMED: SUM-04..07 accepted; real-R red: altered stability median / frequency / sign / key, estimand median, "
     "realized sizes, trace df / flags and draw ids all accepted by the v1.3.5 generated script (10 of 10 exit 0)",
     "the reproduction recomputes stability (declared pool from the setup, both denominators, sign, type-7 "
     "percentiles), estimands, realized sizes (verified plan + R grouping) and performance in R from the reproduced fits; "
     "compares every declared field of predictions, folds, draws, coefficients (+ identity check), trace (df, reference "
     "loglik, flags, rows), manifests (unit-key partitions, cross-file ids) and the ledger",
     [T + "test_owner_summary_cases_sum01_to_sum07", T + "test_complete_reproduction_verifies_every_field",
      T + "test_each_field_family_tamper_fails_with_a_named_check", T + "test_missing_file_and_data_checksum",
      "tests/test_v135.py::test_reproduction_verifies_everything_and_catches_tampering",
      "tests/test_v135.py::test_51c_pipeline_bootstrap_reproduces_in_fresh_r",
      "tests/test_v134.py::test_validation_manifest_identity_and_relocated_reproduction"],
     EV + "G0_red_tamper_v135.json", "Partition (bijection) checks verify the app's composite unit keys against R's "
     "grouping; the keys' text form itself is the app's (metadata, not a statistic)."),
    ("AUD135-03", "P1", "independent audit round 2 (adversarial.py 13/4)",
     "PYTHON-CONFIRMED: reused folder kept the old runnable script and data after a refusal; portable export without "
     "consent kept an older data file (adversarial.py on b7610a0: 13 pass / 4 fail)",
     "export/bundle.py: every export (validation, portable, Reports, audit, code dialog) writes a NEW folder, staged "
     "hidden and renamed on success; central registry with data only through write_data and explicit per-export consent "
     "(default No in every dialog); inventory check (no unregistered / undeclared data, no symlinks); hash scratch "
     "outside the bundle; bundle_kind results-only / code-only / runnable; actual path reported",
     [T + "test_transaction_policy_negative_controls", T + "test_portable_exports_never_reuse_a_folder",
      T + "test_audit_export_goes_to_a_new_folder", T + "test_adversarial_round2_port_reused_folders_and_exact_cap",
      T + "test_multiple_results_get_separately_bound_bundles",
      G + "test_reports_export_asks_consent_and_writes_a_new_folder",
      G + "test_code_dialog_refuses_unprovable_history_and_saves_provable",
      G + "test_validation_panel_export_goes_to_new_folders_with_consent"],
     EV + "G0_adversarial_v135.json", "Normal error cleanup is not secure erasure; a killed process leaves a clearly "
     "named .incomplete- folder (never a complete-looking bundle)."),
    ("AUD135-04", "P2", "independent audit round 2 (controlled scope witness)",
     "REAL-R CONFIRMED on b7610a0: a cancelled validation's exported reproduction exits 1 ('split manifest: same rows': "
     "plan compared with the replayed subset) - " + EV + "G0_red_real_r_v135.json",
     "execution ledger: immutable plan (parts / chunks), attempts log, chunks committed only after a validated response "
     "(parts returned exactly once, counters well formed); lost responses -> uncommitted, counters a lower bound; the "
     "script replays committed chunks only, verifies the pending plan against the embedded plan, checks the ledger and "
     "replay scope, and reports partial coverage",
     [T + "test_cancel_before_and_after_first_chunk_and_late_cancel",
      T + "test_lost_worker_and_invalid_response_are_uncommitted_and_counters_unknown",
      T + "test_partial_bootstrap_chunks_and_budget_inside_a_chunk",
      T + "test_cancelled_validation_replays_committed_scope_only",
      T + "test_budget_exactly_exhausted_with_work_remaining_is_partial_not_dispatched"],
     EV + "G0_red_real_r_v135.json", "Exact replay of a chunk stopped by the budget relies on the identical guard (the "
     "same remaining budget is embedded); a chunk without a validated response is never replayed."),
    ("AUD135-05", "P2", "independent audit round 2 (controlled exact-cap failure)",
     "REAL-R CONFIRMED on b7610a0: cap = the 7 fits needed -> status partial with 'remaining parts were not run' although "
     "the only part was scored - " + EV + "G0_budget_v135.json",
     "R counts fits REFUSED by the cap (denied) apart from reaching it (at_cap); any denial in a part makes it a budget "
     "stop (never scored); Python stops before dispatch when no fit is left and work remains; final status from the "
     "ledger: complete iff every planned part committed successfully and nothing was denied (used == cap is complete)",
     [T + "test_exact_budget_boundary_holdout", T + "test_zero_budget_runs_nothing",
      T + "test_exact_budget_complete_and_one_short_partial_real_r",
      T + "test_adversarial_round2_port_reused_folders_and_exact_cap",
      "tests/test_v135.py::test_hard_fit_budget_counts_and_stops"],
     EV + "G0_budget_v135.json", "Cache-only continuation at a used-up cap is not offered (each part needs fresh fits)."),
    ("AUD135-06", "P1", "independent audit round 3 (workbook 9/7)",
     "PYTHON-CONFIRMED (workbook reproducer 9 pass / 7 fail) and REAL-R CONFIRMED on b7610a0: a sheet-B search admitted "
     "sheet A's 3 cached rows and reported A's p-values as B's - " + EV + "G0_red_real_r_v135.json",
     "core/identity.py identity schema 3: content identity (typed values, columns, rows, parse options incl. sheet, "
     "source hash) and view identity (+ metadata, filters, derived, holdout and their realized masks); data_signature = "
     "view identity; fit identity schema 3; snapshot schema 3 with content id (rebuild only on that content); RSession "
     "keys by content id + CSV SHA-256; legacy (< 3) searches unverified: no cache / resume / rebuild / runnable bundle",
     [T + "test_workbook_sheets_have_distinct_identities_everywhere", T + "test_content_identity_definition",
      T + "test_identity_survives_project_save_reopen_and_reimport", T + "test_value_digest_is_computed_once_per_revision",
      T + "test_session_data_keys_do_not_use_object_addresses", T + "test_legacy_identity_is_unverified_everywhere",
      T + "test_console_import_and_reshape_change_the_content_identity",
      T + "test_workbook_sheets_fit_differently_in_one_worker_both_orders",
      G + "test_legacy_search_dialog_title_says_unverified",
      "tests/test_v134.py::test_legacy_runs_without_identity_or_old_rules_are_never_reused"],
     EV + "G0_workbook_v135.json; " + EV + "G0_red_real_r_v135.json",
     "Conservative: the same values imported from a different file (other source hash) are a different content "
     "identity; v1.3.5-and-earlier searches must be restarted ('New search from this one') to be resumed or reproduced."),
    ("AUD135-07", "P1", "independent audit (additional historical export finding, 6/2)",
     "PYTHON-CONFIRMED (6 pass / 2 fail) and REAL-R CONFIRMED on b7610a0: after a holdout seed change the Reports bundle "
     "of the seed-1 t-test reproduced mean 10.3 instead of the recorded 9.7 - " + EV + "G0_red_real_r_v135.json",
     "every result records analysis_provenance at fit time (content / view identity, SHA-256 of the exact CSV R read, "
     "script SHA-256, snapshot); Reports, code dialog and portable exports re-create that CSV from the current view or "
     "the snapshot rebuild and require the same SHA-256, else refuse (REPRODUCTION_UNAVAILABLE + labelled non-runnable "
     "record); one bundle per result",
     [T + "test_historical_report_exports_the_original_view_or_refuses",
      T + "test_multiple_results_get_separately_bound_bundles", T + "test_result_without_provenance_is_refused",
      T + "test_failed_result_keeps_provenance_and_stays_failed", T + "test_historical_one_sample_t_real_r_seed_change",
      G + "test_reports_export_asks_consent_and_writes_a_new_folder",
      G + "test_code_dialog_refuses_unprovable_history_and_saves_provable"],
     EV + "G0_historical_report_v135.json; " + EV + "G0_red_real_r_v135.json",
     "Results saved by v1.3.5 or earlier carry no provenance: their tables / reports export, never a runnable bundle."),
]
DISCOVERED = [
    ("V136-D1", "P2", "found while auditing AUD135-03 call sites",
     "write_portable() defaulted to include_data=True: any caller that omitted the flag exported the analysis data "
     "without asking", "default False; consent must be an explicit bool (None refused)",
     [T + "test_portable_exports_never_reuse_a_folder", T + "test_transaction_policy_negative_controls"], ""),
    ("V136-D2", "P2", "found while auditing AUD135-05 (budget wrappers)",
     "SOURCE-CONFIRMED: a budget refusal swallowed by a wrapper (reduced-model fits run inside sx_discover_one's "
     "tryCatch) recorded a failed candidate and the part stayed 'scored' with a selection over fewer candidates; not "
     "reproduced at runtime in the tested configurations (the denial propagated)",
     "R counts denials; any denial during a part turns it into a budget stop (no predictions / coefficients kept)",
     [T + "test_partial_bootstrap_chunks_and_budget_inside_a_chunk",
      T + "test_exact_budget_complete_and_one_short_partial_real_r"], ""),
    ("V136-D3", "P3", "found while auditing AUD135-06 (RSession)",
     "RSession reload keys used id(ds.raw): a process-local address can be reused by another frame (stale worker data);"
     " no GUI path found (the session is invalidated on project change)", "content identity + CSV SHA-256",
     [T + "test_session_data_keys_do_not_use_object_addresses"], ""),
    ("V136-D4", "P3", "found while auditing AUD135-06 (holdout)",
     "the holdout split is recomputed from numpy's generator on reopen: a different numpy could change the split while "
     "the data signature stayed equal", "the realized holdout / filter masks are part of the view identity (a changed "
     "split is detected and refused, never used silently)", [T + "test_content_identity_definition"], ""),
    ("V136-D5", "P2", "found while auditing AUD135-01/02 (legacy records)",
     "a validation run saved by v1.3.5 would be re-exported with its old (weak) reproduction script as runnable",
     "v1.3.5-and-earlier validation records export results only, with the reason",
     ["tests/test_v135.py::test_validation_export_refuses_runnable_bundle_over_other_data"], ""),
    ("V136-D6", "P3", "introduced and caught during v1.3.6 development (not in any release)",
     "the new Reports explanation label widened the main window's minimum width to 1580 px (simulated DPI test FAIL)",
     "label word-wrapped; minimum width back to <= 1366", ["tests/test_v134_gui.py::test_simulated_display_scaling_layout"],
     ""),
]
# 12 inherited requirements without a named test and 5 with ambiguous shorthand (round 3): explicit exact links
OVERRIDES = {
    "UI-35": ["tests/test_v12_gui.py::test_longitudinal_candidate_refits_in_mixed_workspace",
              "tests/test_v133.py::test_spec_diff_ignores_presentation_and_lists_model_changes",
              "tests/test_v131.py::test_27_bic_identical_in_discovery_full_analysis_and_base_r",
              "tests/test_v132.py::test_cd09_history_context_rebuild_and_new_analysis"],
    "UI-39": [TR + "test_ui39_filter_context_note_names_historical_views",
              "tests/test_v132.py::test_cd09_history_context_rebuild_and_new_analysis"],
    "UI-40": [TR + "test_ui40_no_duplicated_holm_wording"],
    "UI-51": ["tests/test_v135.py::test_focal_minimum_example_from_the_audit",
              "tests/test_v135.py::test_min_control_disabled_for_workflow_b"],
    "UI-52": [TR + "test_ui52_whole_model_text_is_not_an_interaction_claim"],
    "UI-53": [TR + "test_ui53_random_intercept_plot_subtitle_follows_available_intervals"],
    "UI-55": [TR + "test_ui55_structure_comparison_text_matches_its_estimand"],
    "V133-QA-02": ["tests/test_v133_gui.py::test_validation_panel_run_persist_reopen",
                   "tests/test_v134_gui.py::test_resume_after_real_restart_two_processes"],
    "V133-QA-03": ["tests/test_console_packages.py::test_O_offline_by_construction",
                   "tests/test_v133.py::test_full_analysis_bundle_runs_relocated_offline_and_fails_closed"],
}
for k in ("UI-36", "UI-37", "UI-38"):
    OVERRIDES[k] = OVERRIDES["UI-35"]
NOTE = {"V133-QA-02": "plus benchmark evidence benchmark_v1.3.6.md", "V133-QA-03": "plus frozen / installed offline "
        "evidence (evidence_report_v1.3.6.md); clean-machine install = owner-approved deferral",
        "UI-35": "regression lock of an earlier GUI fix; semantic GUI look remains an owner check"}
ENV = {"PHYSICAL-DPI-150": ("NOT RUN: physical 150 % monitor scaling (simulated scaling tested)",
                            "permitted to remain NOT RUN by the v1.3.6 master prompt (G3: earlier owner permission)"),
       "CLEAN-MACHINE-INSTALL": ("NOT RUN: clean-machine offline install", "owner-approved deferral (standing approval "
                                 "given 2026-11-28 in the v1.3.6 session)")}


def junit(paths):
    res = {}
    for p in paths:
        for tc in ET.parse(p).getroot().iter("testcase"):
            name = tc.get("classname", "").replace(".", "/") + ".py::" + tc.get("name")
            st = "PASS"
            if tc.find("failure") is not None or tc.find("error") is not None:
                st = "FAIL"
            elif tc.find("skipped") is not None:
                st = "SKIPPED"
            res[name] = st
    return res


def resolve(links, res):
    out = {}
    for t in links:
        t = t.strip()
        if not t:
            continue
        hit = {k: v for k, v in res.items() if k == t or k.startswith(t + "[") or (k.startswith(t) and "::" not in t)}
        if not hit:
            hit = {k: v for k, v in res.items() if k.startswith(t)}   # a shortened function name: expand all
        out.update(hit)
    return out


def gate(links, res):
    nodes = resolve(links, res)
    if not nodes:
        return "FAIL", "no mapped test in this run", nodes
    bad = [k for k, v in nodes.items() if v == "FAIL"]
    if bad:
        return "FAIL", "; ".join(bad), nodes
    sk = [k for k, v in nodes.items() if v == "SKIPPED"]
    if sk:
        return "PARTIAL", "skipped: " + "; ".join(sk), nodes
    return "PASS", "", nodes


def main() -> int:
    res = junit(sys.argv[1:])
    reg, trace = [], []

    def add(rid, sev, origin, src, base, fix, links, ev, lim, note=""):
        st, det, nodes = gate(links, res)
        reg.append({"id": rid, "severity": sev, "origin": origin, "source": src, "baseline_reproduction": base,
                    "correction": fix, "mapped_links": "; ".join(links), "exact_tests": len(nodes), "result": st,
                    "result_detail": det, "evidence": ev, "remaining_limitation": lim + (f" ({note})" if note else ""),
                    "approval": OK if st == "PASS" else "REQUIRED"})
        for k, v in sorted(nodes.items()):
            trace.append({"requirement": rid, "origin": origin, "test_node": k, "status": v})
    for rid, sev, src, base, fix, links, ev, lim in NEW:
        add(rid, sev, "v1.3.6 finding", src, base, fix, links, ev, lim)
    for rid, sev, src, base, fix, links, lim in DISCOVERED:
        add(rid, sev, "v1.3.6 discovered", src, base, fix, links, "", lim)
    for rid, sev, src, base, fix, links, ev, lim in reg135.NEW:
        add(rid, sev, "AUD134 (v1.3.5 fix)", src, base, fix, links, ev, lim)
    for rid, sev, src, what, fix, links, lim, _ in reg135.DISCOVERED:
        add(rid, sev, "V135-D (v1.3.5 fix)", src, what, fix, links, "", lim)
    for r in csv.DictReader(open(HERE / "evidence" / "round3_traceability_input.csv", encoding="utf-8")):
        if not (r["origin"].startswith("inherited") or r["origin"].startswith("new (v1.3.4")):
            continue
        rid = r["id"]
        links = OVERRIDES.get(rid) or [x for x in r["developer_mapping"].split(";") if x.strip().startswith("tests/")]
        origin = "Appendix A (92 inherited)" if r["origin"].startswith("inherited") else "V134 issue group (17)"
        add(rid, r["priority"], origin, "inherited", "", r["requirement"], links, "", "", NOTE.get(rid, ""))
    for rid, (why, appr) in ENV.items():
        reg.append({"id": rid, "severity": "environment", "origin": "environment", "source": "master prompt G3",
                    "baseline_reproduction": "", "correction": "", "mapped_links": "", "exact_tests": 0,
                    "result": "NOT RUN", "result_detail": why, "evidence": "", "remaining_limitation": why,
                    "approval": appr})
    with open(HERE / "defect_register_v1.3.6.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(reg[0]))
        w.writeheader()
        w.writerows(reg)
    with open(HERE / "traceability_v1.3.6.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(trace[0]))
        w.writeheader()
        w.writerows(trace)
    from collections import Counter

    by = Counter((r["origin"], r["result"]) for r in reg)
    for k in sorted(by):
        print(f"{k[0]:32s} {k[1]:8s} {by[k]}")
    fails = [r["id"] for r in reg if r["result"] in ("FAIL", "PARTIAL")]
    print(f"{len(reg)} register rows, {len(trace)} exact test links; FAIL/PARTIAL: {fails or 'none'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
