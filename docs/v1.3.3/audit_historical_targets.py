"""Audit 07_Historical_Targets.csv (recorded anchors, some rounded) against the fresh INDEPENDENT reference output.
A secondary check of the oracle itself; application values are compared with the same reference by tests/test_v133.py.
Usage: python audit_historical_targets.py <07_Historical_Targets.csv> <reference_metrics.csv> <out.csv>"""
import csv
import math
import sys

CASE = {"43B_AMD": "A_cluster_RS_AR1_full", "43B_R_independent": "A_cluster_RS_AR1_full", "44D_REML": "B_cluster_RS_AR1",
        "45A_composite": "45A_cluster_RS_AR1", "45E_unclustered": "45E_participant_RI", "46A_AR1": "46A_AR1",
        "46A_CAR1": "46A_CAR1", "46B_AR1": "46B_AR1", "48A_DIAG": "B_cluster_DIAG_independent",
        "48A_RI": "B_cluster_RI_independent", "48A_RS": "B_cluster_RS_independent", "48A_DIAG_RS": "48A_DIAG_RS",
        "48A_RI_DIAG": "48A_RI_DIAG", "48A_RI_RS": "48A_RI_RS", "48B_independent": "48B_nlme_independent",
        "48C_correct_AR1_linear": "48C_AR1_linear", "48C_wrong_independent_linear": "48C_independent_linear",
        "48C_wrong_independent_quadratic": "48C_independent_quadratic",
        "48C_wrong_independent_categorical": "48C_independent_categorical", "49A_DIAG": "49A_DIAG", "49B_RI": "49B_RI",
        "49B_RS": "49B_RS", "50B_D_historical": "A_cluster_RS_AR1_full", "50C_new": "50C_filtered"}
METRIC = {"interaction": "beta:TIME:GROUP1", "intercept": "beta:(Intercept)", "time": "beta:TIME", "group": "beta:GROUP1",
          "se_interaction": "se:TIME:GROUP1", "se": "se:TIME:GROUP1", "p_interaction": "wald_p:TIME:GROUP1",
          "school_variance": "cov:SCHOOL:(Intercept):variance", "id_intercept_variance": "cov:PID:(Intercept):variance",
          "id_slope_variance": "cov:PID:TIME:variance", "clusters": "modeled_clusters"}
SPECIAL = {("43B_AMD", "LR"): ("A_cluster_RS_AR1_focal", "LR"), ("43B_AMD", "p"): ("A_cluster_RS_AR1_focal", "p"),
           ("43B_R_independent", "LR"): ("A_cluster_RS_AR1_focal", "LR"),
           ("43B_R_independent", "reduced_logLik"): ("A_cluster_RS_AR1_reduced", "logLik"),
           ("43B_R_independent", "reduced_AIC"): ("A_cluster_RS_AR1_reduced", "AIC"),
           ("43B_R_independent", "reduced_BIC"): ("A_cluster_RS_AR1_reduced", "BIC")}


def main(targets: str, ref: str, out: str) -> int:
    R = {(r["case"], r["metric"]): float(r["value"]) for r in csv.DictReader(open(ref, encoding="utf-8"))}
    rows = []
    for t in csv.DictReader(open(targets, encoding="utf-8")):
        case, metric, val = t["case"], t["metric"], t["value"]
        key = SPECIAL.get((case, metric)) or (CASE.get(case, ""), METRIC.get(metric, metric))
        e = R.get(key)
        if e is None and case.startswith(("48A_", "49A", "49B")) and metric in ("LR", "p"):
            e = R.get((key[0], "LR_raw" if metric == "LR" else metric))
        try:
            v = float(val)
        except ValueError:
            rows.append({**t, "reference_key": "|".join(key), "reference": "", "status": "DOCUMENTARY"})
            continue
        if e is None:
            rows.append({**t, "reference_key": "|".join(key), "reference": "", "status": "NOT IN REFERENCE"})
            continue
        dec = len(val.split(".")[1]) if "." in val and "e" not in val.lower() else None
        if metric in ("p", "p_interaction", "BH", "Holm") and v > 0 and e > 0:
            ok = abs(math.log10(v) - math.log10(e)) <= 1e-3
        elif t["precision"] == "full":
            ok = abs(v - e) <= 1e-4 + 1e-6 * abs(e) or (metric.startswith(("se", "R2", "phi", "id_", "school", "residual"))
                                                       and abs(v - e) <= 1e-4 + 1e-4 * abs(e))
        else:
            ok = abs(v - e) <= (0.5 * 10 ** -dec if dec else 0.5) + 1e-9
        rows.append({**t, "reference_key": "|".join(key), "reference": e, "status": "PASS" if ok else "DIFFERS"})
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    by = {}
    for r in rows:
        by[r["status"]] = by.get(r["status"], 0) + 1
    print(by)
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
