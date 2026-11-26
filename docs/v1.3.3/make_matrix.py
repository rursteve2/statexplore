"""Fill 06_Compatibility_Matrix.csv execution_status from a runner JUnit file (actual results, no assumption).
Usage: python make_matrix.py <06_Compatibility_Matrix.csv> <pytest_junit.xml> <out.csv>"""
import csv
import sys
import xml.etree.ElementTree as ET


def main(matrix: str, junit: str, out: str) -> None:
    res = {}
    for tc in ET.parse(junit).getroot().iter("testcase"):
        st = "FAIL" if tc.find("failure") is not None or tc.find("error") is not None else \
            "SKIPPED" if tc.find("skipped") is not None else "PASS"
        res[tc.get("name")] = st
    rows = []
    for r in csv.DictReader(open(matrix, encoding="utf-8")):
        cfg, wf = r["configuration"], r["workflow"]
        hier, rnd, resid = cfg.split("_")
        cl = hier == "cluster"
        pid = f"{'cluster' if cl else 'participant'}_{rnd}_{resid}"
        tests = ([f"test_matrix_workflow_a_focal_matches_reference[{pid}]"] if wf == "A" else
                 [f"test_matrix_workflow_b_nine_structures_match_reference[{'cluster' if cl else 'participant'}]",
                  f"test_matrix_standalone_full_analysis_reml_matches_reference[{pid}]"])
        st = [res.get(t, "NOT RUN") for t in tests]
        r["execution_status"] = ("PASS" if all(s == "PASS" for s in st) else "/".join(st)) + " — " + "; ".join(
            f"tests/test_v133.py::{t}" for t in tests)
        rows.append(r)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(sum(r["execution_status"].startswith("PASS") for r in rows), "of", len(rows), "obligations PASS")


if __name__ == "__main__":
    main(*sys.argv[1:4])
