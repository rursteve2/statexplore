"""v1.3.2 regression tests (developer package `StatExplore_v1.3.2_Developer_Package`).

Every numerical expectation comes from the INDEPENDENT R reference (`tests/fixtures/v132/reference`, produced by
08_Independent_R_Reference_Models.R on the same serialized fixture bytes, see SHA256SUMS.txt); the application is
never compared with itself. Tolerances follow 05_Regression_Test_Protocol.md:
  logLik / LR 1e-4 absolute, AIC / BIC 1e-3, variance components / phi 1e-4 abs + 1e-4 rel, p on the natural-log scale
  1e-3, counts exact.

  identity (STAT-22/39/40, N-01..N-03)          test_identity_*
  construction (STAT-21, CD-01..CD-07)           test_stat21_*, test_cd0*
  comparisons / LR (STAT-17/19/23/33/34/41)       test_lr_*, test_cmp_*
  diagnostics (STAT-18/32/44/45, N-06/N-08)       test_diag_*
  reporting (STAT-20/42/43/46)                    test_report_*
  validation, persistence (CD-08..CD-11)          test_cd08_* .. test_cd11_*
"""
from __future__ import annotations

import csv
import math
import os
import subprocess
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

from statexplore.core.derive import FilterSpec  # noqa: E402
from statexplore.core.importers import import_file  # noqa: E402

FX = Path(__file__).parent / "fixtures" / "v132"
pytestmark = pytest.mark.r


def ref() -> dict[tuple[str, str, str], float]:
    with (FX / "reference" / "actual_metrics.csv").open(newline="") as fh:
        return {(r["fixture"], r["model"], r["metric"]): float(r["value"]) for r in csv.DictReader(fh)
                if r["value"] not in ("", "NA")}        # NA = not estimable in the reference (kept out)


REF = ref()


def near(a, b, atol, rtol=0.0) -> bool:
    return a is not None and b is not None and abs(float(a) - float(b)) <= atol + rtol * abs(float(b))


def near_p(a, b, log_atol=1e-3) -> bool:
    return a is not None and b is not None and a > 0 and b > 0 and abs(math.log(a) - math.log(b)) <= log_atol


def ds(name: str):
    return import_file(FX / f"{name}.csv")


def table(result, prefix: str) -> list[dict]:
    t = next(t for t in result.tables if t.name.startswith(prefix))
    return [dict(zip(t.columns, r)) for r in t.rows]


def disc(d, **kw):
    from statexplore.discovery import engine
    from statexplore.discovery.space import SearchSettings

    base = dict(outcome="Y", family="gaussian", workers=2, max_models=None, max_minutes=None)
    base.update(kw)
    run = engine.create_run(d, SearchSettings(**base))
    engine.execute(run, d)
    return run


def rscript(code: str, cwd: Path, timeout: int = 600) -> subprocess.CompletedProcess:
    from statexplore.rbridge.locate import find_rscript, r_library

    return subprocess.run([str(find_rscript()), "--vanilla", "-e", code], cwd=cwd, capture_output=True, text=True,
                          env=dict(os.environ, R_LIBS=str(r_library())), timeout=timeout)


# ---- fixtures integrity -----------------------------------------------------------------------------------------------

def test_fixture_bytes_match_manifest():
    import hashlib

    for line in (FX / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        assert hashlib.sha256((FX / name.strip("* ")).read_bytes()).hexdigest() == digest, name


# ---- LR service (STAT-17 / 23 / 41, N-04) -----------------------------------------------------------------------------

def test_lr_python_service_negative_and_boundary_rules():
    from statexplore.analysis.lrtest import lr_test

    t = lr_test(0, -7.008e-7 / 2, 1, "chisq")             # 36A: negligible negative -> 0, raw kept
    assert t["status"] == "tested" and t["LR"] == 0 and t["LR_raw"] < 0 and t["p"] == pytest.approx(1.0)
    t = lr_test(0, -9.45e-10 / 2, 1, "mixture")           # 37A RI -> DIAG
    assert t["status"] == "tested" and t["p"] == pytest.approx(1.0)
    t = lr_test(0, -63.968673 / 2, 1, "mixture")          # 37B material negative: no p, never abs()
    assert t["status"] == "withheld" and t["p"] is None and t["LR"] is None
    assert lr_test(0, float("nan"), 1)["status"] == "withheld"
    assert lr_test(0, 1, 0)["status"] == "withheld"
    # 36B ordinary and CAR(1) boundary probabilities (reference values)
    lr = REF[("T36B", "RI_Independent_to_CAR1", "LR_raw")]
    assert near_p(lr_test(0, lr / 2, 1, "chisq")["p"], REF[("T36B", "RI_Independent_to_CAR1", "p_ordinary")])
    assert near_p(lr_test(0, lr / 2, 1, "mixture")["p"], REF[("T36B", "RI_Independent_to_CAR1", "p_boundary")])
    # tiny tails survive (log scale), 1:2 mixture
    t = lr_test(0, 233.740449 / 2, 2, "mixture")
    assert near_p(t["p"], REF[("T38", "3L_RI_to_RS_Independent", "p_boundary")])


def test_lr_r_twin_matches_python(r_session):
    from statexplore.analysis.covstruct import MIXED_R_HELPERS
    from statexplore.analysis.lrtest import lr_test

    cases = [(0.0, -3.504e-7, 1, "chisq"), (0.0, 8.476, 1, "mixture"), (0.0, 116.87, 2, "mixture"),
             (0.0, 3.6, 1, "chisq"), (0.0, -31.98, 1, "mixture"), (0.0, 400.0, 2, "mixture")]
    code = MIXED_R_HELPERS + "sx_result <- lapply(list(" + ", ".join(
        f"c({a}, {b}, {d})" for a, b, d, _ in cases) + "), function(x) NULL)\n"
    code += "sx_ref <- c(" + ", ".join(f'"{r}"' for *_, r in cases) + ")\n"
    code += ("sx_result <- lapply(seq_along(sx_result), function(i) { x <- list(" + ", ".join(
        f"c({a}, {b}, {d})" for a, b, d, _ in cases) + ")[[i]]; t <- sx_lr_test(x[1], x[2], x[3], sx_ref[i]); "
             "list(p = if (is.na(t$p)) -1 else t$p, status = t$status) })\n")
    out = r_session.worker.run(code).result
    for (a, b, d, refk), r in zip(cases, out):
        py = lr_test(a, b, d, refk)
        assert r["status"] == py["status"]
        if py["p"] is not None:
            assert r["p"] == pytest.approx(py["p"], rel=1e-10, abs=1e-300)


def test_lr_contrast_reference_follows_hypothesis():
    from statexplore.analysis.covstruct import Structure, contrast

    S = Structure
    assert contrast(S("ri", "independent"), S("diag", "independent")).reference == "mixture"
    assert contrast(S("ri", "independent"), S("rs", "independent")).df == 2
    c = contrast(S("diag", "ar1"), S("rs", "ar1"))
    assert c.reference == "chisq" and c.needs_interior
    assert contrast(S("ri", "independent"), S("ri", "ar1")).reference == "chisq"      # signed phi: interior
    assert contrast(S("ri", "independent"), S("ri", "car1")).reference == "mixture"   # phi >= 0: boundary
    assert not contrast(S("ri", "ar1"), S("ri", "car1")).nested                       # not nested
    assert not contrast(S("ri", "independent"), S("rs", "ar1")).nested                # simultaneous changes
    assert contrast(S("rs", "ar1", False), S("rs", "ar1", True)).reference == "mixture"


# ---- STAT-21: clustered construction in the Mixed workspace ---------------------------------------------------------

@pytest.mark.parametrize("random,residual,key,k", [
    ("intercept", "ar1", "3L_RI_AR1", 7), ("slope_uncorrelated", "ar1", "3L_DIAG_AR1", 8), ("slope", "ar1", "3L_RS_AR1", 9),
    ("intercept", "car1", "3L_RI_CAR1", 7), ("slope_uncorrelated", "car1", "3L_DIAG_CAR1", 8),
    ("slope", "car1", "3L_RS_CAR1", 9), ("slope", "independent", "3L_RS_Independent", 8),
    ("slope_uncorrelated", "independent", "3L_DIAG_Independent", 7), ("intercept", "independent", "3L_RI_Independent", 6)])
def test_stat21_clustered_mixed_matches_reference(r_session, random, residual, key, k):
    from statexplore.analysis import mixed

    d = ds("T38")
    res, _ = mixed.run(r_session, d, mixed.MixedSpec(outcome="Y", id="ID", time="TIME", predictors=["GROUP"],
                                                     cluster="SCHOOL", random=random, residual=residual,
                                                     interactions="none", method="REML"))
    assert res.status.value in ("ok", "warning"), res.messages
    fit = table(res, "Model fit")[0]
    assert fit["parameters"] == k and fit["observations"] == 960 and fit["participants"] == 160 and fit["clusters"] == 20
    assert near(fit["logLik"], REF[("T38", key, "logLik")], 1e-4)
    assert near(fit["AIC"], REF[("T38", key, "AIC")], 1e-3) and near(fit["BIC"], REF[("T38", key, "BIC")], 1e-3)
    assert fit["BIC_n"] == REF[("T38", key, "BIC_n")]
    groups = {g["level"]: g for g in table(res, "Grouping levels")}
    assert groups[next(n for n in groups if "cluster" in n or n == "SCHOOL")]["groups"] == 20
    vc = table(res, "Random effects")
    cl = next(r for r in vc if r["group"] in ("SCHOOL", "sx_cluster") and r["term"] == "(Intercept)")
    assert near(cl["variance_or_cov"], REF[("T38", key, "SCHOOL.var_(Intercept)")], 1e-4, 1e-4)
    if random == "slope" and residual != "independent":
        cov = next(r for r in vc if r["with"])                       # STAT-43: covariance / correlation reported
        assert near(cov["variance_or_cov"], REF[("T38", key, "ID.cov_intercept_TIME")], 1e-4, 1e-4)
        assert near(cov["sd_or_corr"], REF[("T38", key, "ID.corr_intercept_TIME")], 1e-4, 1e-4)
    if random == "slope_uncorrelated" and residual != "independent":
        cov = next(r for r in vc if r["with"])
        assert cov["sd_or_corr"] == 0 and "constrained" in cov["estimated"]
    if residual != "independent":
        phi = table(res, "Residual autocorrelation")[0]["phi"]
        assert near(phi, REF[("T38", key, "phi")], 1e-4, 1e-4)
    assert near(fit["R2_marginal"], REF[("T38", key, "R2_direct_m")], 1e-3)
    assert near(fit["R2_conditional"], REF[("T38", key, "R2_direct_c")], 1e-3)
    if key == "3L_RS_CAR1":    # apVar not positive definite: reported, not called singular, not clean
        assert fit["apVar"] != "positive definite" and fit["singular"] is False and fit["diagnostics"] == "warned"
    elif residual != "independent":
        assert fit["diagnostics"] == "clean"


def test_stat21_random_comparison_keeps_cluster(r_session):
    from statexplore.analysis import mixed

    base = mixed.MixedSpec(outcome="Y", id="ID", time="TIME", predictors=["GROUP"], cluster="SCHOOL", random="slope",
                           residual="ar1", interactions="none")
    res, _ = mixed.compare_specs(r_session, ds("T38"), base, "random")
    fits = table(res, "Specification comparison")
    assert [f["parameters"] for f in fits] == [7, 8, 9] and all(f["status"] == "fitted" for f in fits)
    rows = table(res, "Likelihood-ratio")
    assert len(rows) == 3 and all(r["status"] == "tested" for r in rows)
    ri_rs = next(r for r in rows if r["null_hypothesis"].startswith("participant time-slope variance = 0 and"))
    assert near(ri_rs["LR"], REF[("T38", "3L_RI_to_RS_AR1", "LR_raw")], 1e-4)
    assert near_p(ri_rs["p"], REF[("T38", "3L_RI_to_RS_AR1", "p_boundary")])
    diag_rs = next(r for r in table(res, "Likelihood-ratio") if r["null_hypothesis"] == "intercept-slope covariance = 0")
    assert near_p(diag_rs["p"], REF[("T38", "3L_DIAG_to_RS_AR1", "p_ordinary")]) and diag_rs["status"] == "tested"


def test_cd06_reused_ids_local_policy_and_global_refusal(r_session):
    from statexplore.analysis import mixed

    d = ds("CD_ReusedID")
    spec = mixed.MixedSpec(outcome="Y", id="ID", time="TIME", predictors=["GROUP"], cluster="SCHOOL", random="slope",
                           residual="ar1", interactions="none", id_policy="local")
    res, _ = mixed.run(r_session, d, spec)
    fit = table(res, "Model fit")[0]
    assert fit["participants"] == 160 and fit["clusters"] == 20
    assert near(fit["logLik"], REF[("T38", "3L_RS_AR1", "logLik")], 1e-4)        # same model as global IDs
    with pytest.raises(mixed.MixedInputError, match="more than one"):
        mixed.validate(d, mixed.MixedSpec(**{**spec.to_dict(), "id_policy": "global"}))
    with pytest.raises(mixed.MixedInputError, match="more than one"):          # CD_Crossing under the global policy
        mixed.validate(ds("CD_Crossing"), mixed.MixedSpec(**{**spec.to_dict(), "id_policy": "global"}))


# ---- diagnostics (STAT-18 / 32 / 44 / 45) -----------------------------------------------------------------------------

def test_diag_lme4_uncorrelated_slope_survives_postprocessing(r_session):
    from statexplore.analysis import mixed

    for f in ("T37A", "T36C"):                                               # singular and nonsingular DIAG
        res, art = mixed.run(r_session, ds(f), mixed.MixedSpec(outcome="Y", id="ID", time="TIME",
                                                               random="slope_uncorrelated", interactions="none"))
        assert res.status.value in ("ok", "warning"), res.messages
        fit = table(res, "Model fit")[0]
        assert near(fit["logLik"], REF[(f, "lmer_DIAG", "logLik")], 1e-4)
        assert any("random" in k or k.startswith("plots/") for k in art)        # optional plots still drawn
        assert not any("incorrect number of dimensions" in m for m in res.messages)


def test_diag_direct_r2_not_misleading_fallback(r_session):
    from statexplore.analysis import mixed

    for f in ("T37A", "T37B"):
        res, _ = mixed.run(r_session, ds(f), mixed.MixedSpec(outcome="Y", id="ID", time="TIME", random="slope",
                                                              interactions="none"))
        fit = table(res, "Model fit")[0]
        assert near(fit["R2_marginal"], REF[(f, "lmer_RS", "R2_direct_m")], 1e-3)
        assert near(fit["R2_conditional"], REF[(f, "lmer_RS", "R2_direct_c")], 1e-3)
        assert fit["singular"] is True and fit["diagnostics"] == "singular"


def test_diag_nlme_near_boundary_detected_and_controls_clean(r_session):
    from statexplore.analysis import mixed

    cases = [("T37A", "slope_uncorrelated", "near_boundary"), ("T37A", "slope", "near_boundary"),
             ("T37B", "intercept", "near_boundary"), ("T36C", "slope", "clean"), ("T36C", "intercept", "clean")]
    for f, random, want in cases:
        res, _ = mixed.run(r_session, ds(f), mixed.MixedSpec(outcome="Y", id="ID", time="TIME", random=random,
                                                              residual="ar1", interactions="none"))
        fit = table(res, "Model fit")[0]
        assert fit["diagnostics"] == want, (f, random, fit["diagnostics"], res.messages)
        if want != "clean":
            assert any("near-boundary" in m for m in res.messages) and fit["singular"] is False


def test_diag_37d_gradient_warning_kept_per_model_and_deduplicated(r_session):
    run = disc(ds("T37D_3702"), candidates=["GROUP"], required=["TIME"], participant="ID", time="TIME",
               workflow="structure", structures=["ri", "rs_uncorr", "rs"])
    by = {r["structure"]: r for r in run.rows}
    assert by["ri"]["status"] == by["rs_uncorr"]["status"] == "ok"
    flags = by["rs"]["flags"]
    assert by["rs"]["status"] == "warning" and "max|grad|" in flags
    assert flags.count("Model failed to converge") == 1                       # identical messages once
    for st, key in (("ri", "RI"), ("rs_uncorr", "DIAG"), ("rs", "RS")):
        assert near(by[st]["loglik"], REF[("T37D_3702", key, "logLik")], 1e-4)
        assert by[st]["n_groups"] == 80                                         # STAT-46: RI count present
    from statexplore.discovery import engine

    tf = engine.tests_frame(run)
    tiers = dict(zip(tf["label"].str.split(":").str[-1].str.strip(), tf["tier"]))
    assert sorted(tiers.values()) == ["A", "C", "C"]                           # 1 clean, 2 provisional
    head = " ".join(engine.headline(run))
    assert "1 with clean diagnostics, 2 provisional" in head
    df = engine.results_frame(run, tf)
    assert df.loc[df["structure"] == "ri", "n_groups"].iloc[0] == 80


# ---- comparisons (STAT-17 / 19 / 33 / 34) -----------------------------------------------------------------------------

def test_cmp_residual_boundary_and_not_nested(r_session):
    from statexplore.analysis import mixed

    res, _ = mixed.compare_specs(r_session, ds("T36B"), mixed.MixedSpec(outcome="Y", id="ID", time="TIME",
                                                                          interactions="none"), "residual")
    rows = table(res, "Likelihood-ratio")
    ar1 = next(r for r in rows if r["null_hypothesis"].startswith("AR(1)"))
    car1 = next(r for r in rows if r["null_hypothesis"].startswith("continuous-time"))
    assert near(ar1["LR"], REF[("T36B", "RI_Independent_to_AR1", "LR_raw")], 1e-4)
    assert near_p(ar1["p"], REF[("T36B", "RI_Independent_to_AR1", "p_ordinary")])
    assert near_p(car1["p"], REF[("T36B", "RI_Independent_to_CAR1", "p_boundary")]) and "mixture" in car1["reference"]
    assert any(r["status"] == "not tested" and "not nested" in r["reason"] for r in rows)


def test_cmp_family_aware_glmm_clmm_gee(r_session):
    from statexplore.analysis import mixed

    d = ds("T28_29_Synthetic")
    for col in ("BINARY", "ORDINAL", "GROUP"):
        d.set_variable_field(col, "measure", "ordinal" if col == "ORDINAL" else "nominal")
    g = mixed.MixedSpec(outcome="BINARY", id="ID", time="TIME", predictors=["GROUP"], family="binomial",
                        interactions="none", method="ML")
    res, _ = mixed.compare_specs(r_session, d, g, "random")
    fits = table(res, "Specification comparison")
    assert all(f["estimation"].startswith("ML (Laplace)") for f in fits if f["status"] == "fitted")
    with pytest.raises(mixed.MixedInputError):
        mixed.comparison_variants(g, "residual")
    o = mixed.MixedSpec(outcome="ORDINAL", id="ID", time="TIME", predictors=["GROUP"], family="ordinal",
                        interactions="none", method="ML")
    res, _ = mixed.compare_specs(r_session, d, o, "time")                         # STAT-33: no missing sx_singular
    assert res.status.value in ("ok", "warning") and len(table(res, "Specification comparison")) == 3
    gee = mixed.MixedSpec(outcome="Y", id="ID", time="TIME", predictors=["GROUP"], estimator="gee", interactions="none")
    with pytest.raises(mixed.MixedInputError, match="no random"):
        mixed.comparison_variants(gee, "random")
    res, _ = mixed.compare_specs(r_session, d, gee, "residual")
    fits = table(res, "Specification comparison")
    assert "QIC" in fits[0] and "logLik" not in fits[0] and "AIC" not in fits[0]   # no likelihood fiction
    assert near(next(f for f in fits if "exchangeable" in f["model"].lower())["QIC"],
                REF[("T28_29_Synthetic", "GEE_exchangeable", "QIC")], 1e-3)


def test_cmp_all_failed_is_explicit(r_session):
    from statexplore.analysis import mixed

    d = ds("T36B")
    # an uncorrelated + correlated slope comparison on a dataset whose time has one value per ID cannot fit slopes
    d.add_filter(FilterSpec("TIME", "==", 0))
    base = mixed.MixedSpec(outcome="Y", id="ID", time="TIME", interactions="none")
    try:
        res, _ = mixed.compare_specs(r_session, d, base, "residual")
    except mixed.MixedInputError:
        return                                                                  # refused before fitting: fine
    assert res.values.get("comparison_state", "").startswith(("failed", "incomplete"))
    assert not any(str(v) in ("inf", "-inf") for r in table(res, "Specification comparison") for v in r.values())


# ---- identity (STAT-22 / 39 / 40) -------------------------------------------------------------------------------------

def test_identity_35c_disjoint_rows_with_equal_fit_never_verified(r_session):
    from statexplore.discovery import detail

    d = ds("T35C_Identity")
    d.add_filter(FilterSpec("FILTER", "==", "A"))
    run = disc(d, candidates=["X"], strategy="main", workers=1)
    row = run.rows[0]
    assert near(row["loglik"], REF[("T35C_Identity", "A", "logLik")], 1e-6)
    ok, _ = detail.full_analysis(r_session, d, run, detail.candidate_from_row(run, row))
    assert ok.values["identity_check"] == "verified"                           # same origin: verified
    d.remove_filter(0)
    d.add_filter(FilterSpec("FILTER", "==", "B"))                             # disjoint rows, identical n / logLik / AIC
    bad, _ = detail.full_analysis(r_session, d, run, detail.candidate_from_row(run, row))
    assert near(table(bad, "Model fit")[0]["AIC"], row["aic"], 1e-6)          # the statistics agree ...
    assert bad.values["identity_check"] == "MISMATCH"                          # ... identity does not
    assert any("analysed rows" in m for m in bad.messages)
    assert not any(m.startswith("Model identity verified") for m in bad.messages)


def test_identity_35a_reversed_ordinal_coding(r_session):
    from statexplore.discovery import detail

    d = ds("T35A_Ordinal")
    d.set_variable_field("Y", "measure", "ordinal")
    run = disc(d, outcome="Y", family="ordinal", candidates=["X"], strategy="main", workers=1)
    row = run.rows[0]
    d.set_variable_field("Y", "category_order", [5.0, 4.0, 3.0, 2.0, 1.0])     # reversed ordering
    res, _ = detail.full_analysis(r_session, d, run, detail.candidate_from_row(run, row))
    assert res.values["identity_check"] == "MISMATCH"
    assert any("outcome values or outcome coding" in m for m in res.messages)


def test_identity_legacy_origin_is_unverifiable():
    from statexplore.analysis.descriptor import ModelDescriptor, verify

    d = ModelDescriptor(source="old", run_id="x", candidate=0, family="gaussian", outcome_family="gaussian",
                        engine="stats::lm", link="identity", outcome="Y", terms=[["X"]], n=40, loglik=-88.95,
                        aic=183.89)
    status, text = verify(d, 40, -88.95, 183.89, identity={"schema": 2, "rows": "a", "response": "b", "design": "c"})
    assert status == "unverifiable" and "not certified" in text


def test_identity_shifted_outcome_same_rows_is_mismatch(r_session, tmp_path):
    import pandas as pd

    from statexplore.discovery import detail

    src = pd.read_csv(FX / "T35C_Identity.csv")
    src.to_csv(tmp_path / "a.csv", index=False)
    run_ds = import_file(tmp_path / "a.csv")
    run = disc(run_ds, candidates=["X"], strategy="main", workers=1)
    shifted = src.assign(Y=src["Y"] + 10)
    shifted.to_csv(tmp_path / "b.csv", index=False)
    d2 = import_file(tmp_path / "b.csv")
    res, _ = detail.full_analysis(r_session, d2, run, detail.candidate_from_row(run, run.rows[0]))
    assert res.values["identity_check"] == "MISMATCH"


def test_identity_cross_module_lmm_verified_and_cluster_mismatch(r_session):
    from statexplore.analysis import mixed

    d = ds("T38")
    run = disc(d, candidates=["GROUP"], required=["TIME"], participant="ID", time="TIME", cluster="SCHOOL",
               structure="rs_ar1", strategy="main")
    row = run.rows[0]
    res, _ = mixed.run(r_session, d, mixed.spec_from_discovery(run, row))
    assert res.values["identity_check"] == "verified"
    spec = mixed.spec_from_discovery(run, row)
    spec.cluster = None                                                        # not the historical model
    res2, _ = mixed.run(r_session, d, spec)
    # v1.3.3 (01 §1): an explicitly edited transferred model is "modified" (lineage kept, never verified)
    assert res2.values["identity_check"].startswith("modified") and res2.values["spec_diff"]


def test_cd09_history_context_rebuild_and_new_analysis(r_session):
    from statexplore.discovery import detail, history

    d = ds("T35C_Identity")
    d.add_filter(FilterSpec("FILTER", "==", "A"))
    run = disc(d, candidates=["X"], strategy="main", workers=1)
    d.remove_filter(0)
    d.add_filter(FilterSpec("FILTER", "==", "B"))
    ctx = history.context(d, run)
    assert ctx.state == "changed" and any("row filters" in x for x in ctx.differences)
    res, _ = detail.full_analysis(r_session, ctx.original, run, detail.candidate_from_row(run, run.rows[0]))
    assert res.values["identity_check"] == "verified"                          # the rebuilt original view
    res2, _ = detail.full_analysis(r_session, d, run, detail.candidate_from_row(run, run.rows[0]), ctx.differences)
    assert res2.values["identity_check"].startswith("new analysis")
    run.data_snapshot = {}                                                    # legacy search: cannot be rebuilt
    assert history.context(d, run).state == "unrecoverable"


def test_cd09_cache_identity_changes_with_cluster_and_policy():
    from statexplore.discovery import engine
    from statexplore.discovery.space import SearchSettings

    d = ds("T38")
    base = dict(outcome="Y", family="gaussian", candidates=["GROUP"], required=["TIME"], participant="ID", time="TIME",
                structure="rs_ar1")
    ids = {engine.fit_identity(d, SearchSettings(**base, **kw), "lmm")
           for kw in ({}, {"cluster": "SCHOOL"}, {"cluster": "SCHOOL", "id_policy": "local"})}
    assert len(ids) == 3


# ---- clustered Discovery (CD-01 .. CD-05, CD-07, CD-10) -----------------------------------------------------------------

B9 = ["ri", "rs_uncorr", "rs", "ar1", "diag_ar1", "rs_ar1", "car1", "diag_car1", "rs_car1"]
KEYS = {"ri": ("RI", "Independent"), "rs_uncorr": ("DIAG", "Independent"), "rs": ("RS", "Independent"),
        "ar1": ("RI", "AR1"), "diag_ar1": ("DIAG", "AR1"), "rs_ar1": ("RS", "AR1"), "car1": ("RI", "CAR1"),
        "diag_car1": ("DIAG", "CAR1"), "rs_car1": ("RS", "CAR1")}
K_T38 = {("2L", "RI"): 4, ("2L", "DIAG"): 5, ("2L", "RS"): 6}


@pytest.mark.parametrize("clustered", [False, True])
def test_cd01_workflow_b_18_combinations(clustered):
    run = disc(ds("T38"), candidates=["GROUP"], required=["TIME"], participant="ID", time="TIME",
               workflow="structure", structures=B9, cluster="SCHOOL" if clustered else None)
    lvl = "3L" if clustered else "2L"
    assert run.status == "complete" and len(run.rows) == 9
    for r in run.rows:
        rnd, res = KEYS[r["structure"]]
        key = f"{lvl}_{rnd}_{res}"
        assert r["status"] in ("ok", "warning"), (key, r["reason"])
        assert near(r["loglik"], REF[("T38", key, "logLik")], 1e-4), key
        assert near(r["aic"], REF[("T38", key, "AIC")], 1e-3) and near(r["bic"], REF[("T38", key, "BIC")], 1e-3)
        assert r["bic_n"] == REF[("T38", key, "BIC_n")]
        assert r["n_lik_params"] == REF[("T38", key, "parameters")]
        assert r["n"] == 960 and r["n_groups"] == 160
        assert r["modeled_clusters"] == (20 if clustered else None if not clustered else 0) or not clustered
        assert r["identity"]["rows"] == run.rows[0]["identity"]["rows"]        # identical rows for every structure
    if clustered:
        assert all(r["modeled_clusters"] == 20 and r["observed_clusters"] == 20 for r in run.rows)


def test_cd02_workflow_a_ml_tests_every_structure():
    for clustered in (False, True):
        lvl = "3L" if clustered else "2L"
        for st in B9:
            rnd, res = KEYS[st]
            key = f"{lvl}_{rnd}_{res}_A_GROUP"
            if (("T38", key, "LR_raw")) not in REF:
                continue                                                       # reference fit failed (reported)
            run = disc(ds("T38"), candidates=["GROUP"], required=["TIME"], participant="ID", time="TIME",
                       structure=st, cluster="SCHOOL" if clustered else None, strategy="main", workers=1)
            row = run.rows[0]
            t = row["tests"][0]
            assert t["type"] in ("overall", "main_term")
            assert near(t["statistic"], REF[("T38", key, "LR_raw")], 1e-4), key
            assert near_p(t["p"], REF[("T38", key, "p_ordinary")]), key
            assert near(row["loglik"], REF[("T38", f"{lvl}_{rnd}_{res}_A_full", "logLik")], 1e-4)


def test_cd03_b_tests_match_reference_and_flag_car1_apvar():
    from statexplore.discovery import engine

    run = disc(ds("T38"), candidates=["GROUP"], required=["TIME"], participant="ID", time="TIME",
               workflow="structure", structures=B9, cluster="SCHOOL")
    tf = engine.tests_frame(run)
    lab = lambda s: tf[tf["label"].str.startswith(s)]  # noqa: E731
    ri_rs = lab("Random intercept + correlated time slope + AR(1) residuals (occasion index) vs Random intercept + AR(1)")
    assert near(ri_rs["statistic"].iloc[0], REF[("T38", "3L_RI_to_RS_AR1", "LR_raw")], 1e-4)
    assert near_p(ri_rs["p"].iloc[0], REF[("T38", "3L_RI_to_RS_AR1", "p_boundary")])
    rs_car = next(r for r in run.rows if r["structure"] == "rs_car1")
    assert rs_car["status"] == "warning" and "apVar" in rs_car["flags"]
    assert set(tf.loc[tf["idx"] == rs_car["idx"], "test_status"]) <= {"provisional", "withheld"}


def test_cd04_time_only_b_without_candidates():
    run = disc(ds("T38"), candidates=[], required=["TIME"], participant="ID", time="TIME", workflow="structure",
               structures=B9, cluster="SCHOOL")
    assert run.status == "complete" and all(r["status"] in ("ok", "warning") for r in run.rows)
    rs = next(r for r in run.rows if r["structure"] == "rs_ar1")
    assert near(rs["loglik"], REF[("T38", "3L_RS_AR1_B_TIME_only", "logLik")], 1e-4)


def test_cd05_explicit_cluster_on_off_same_rows():
    from statexplore.discovery import engine

    run = disc(ds("T38"), candidates=["GROUP"], required=["TIME"], participant="ID", time="TIME",
               workflow="structure", structures=["rs_ar1", "ri"], cluster="SCHOOL", cluster_compare=True)
    assert len(run.rows) == 4
    on = {r["structure"]: r for r in run.rows if r["clustered"]}
    off = {r["structure"]: r for r in run.rows if not r["clustered"]}
    assert on["rs_ar1"]["identity"]["rows"] == off["rs_ar1"]["identity"]["rows"]
    assert near(off["rs_ar1"]["loglik"], REF[("T38", "2L_RS_AR1", "logLik")], 1e-4)
    assert off["rs_ar1"]["modeled_clusters"] == 0 and off["rs_ar1"]["observed_clusters"] == 20
    tf = engine.tests_frame(run)
    add = tf[tf["label"].str.contains("without the cluster", regex=False) & tf["label"].str.contains("cluster variance")]
    row = add[add["label"].str.startswith("Random intercept + correlated")].iloc[0]
    assert near(row["statistic"], REF[("T38", "Add_cluster_RS_AR1", "LR_raw")], 1e-4)
    assert near_p(row["p"], REF[("T38", "Add_cluster_RS_AR1", "p_boundary")])
    default = disc(ds("T38"), candidates=["GROUP"], required=["TIME"], participant="ID", time="TIME",
                   workflow="structure", structures=["rs_ar1"], cluster="SCHOOL")
    assert len(default.rows) == 1 and default.rows[0]["clustered"] is True      # never removed by default


def test_cd06_discovery_local_ids_equal_global():
    g = disc(ds("T38"), candidates=["GROUP"], required=["TIME"], participant="ID", time="TIME",
             workflow="structure", structures=["rs_ar1"], cluster="SCHOOL")
    loc = disc(ds("CD_ReusedID"), candidates=["GROUP"], required=["TIME"], participant="ID", time="TIME",
               workflow="structure", structures=["rs_ar1"], cluster="SCHOOL", id_policy="local")
    assert loc.rows[0]["n_groups"] == 160 and near(loc.rows[0]["loglik"], g.rows[0]["loglik"], 1e-6)
    from statexplore.discovery.space import SearchSettings, SearchSpaceError, validate_settings

    with pytest.raises(SearchSpaceError, match="more than one"):
        validate_settings(ds("CD_ReusedID"), SearchSettings(outcome="Y", family="gaussian", candidates=["GROUP"],
                                                           required=["TIME"], participant="ID", time="TIME",
                                                           cluster="SCHOOL", structure="rs_ar1"))


def test_cd07_irregular_time_ar1_vs_car1_and_missing_predictor_rows():
    run = disc(ds("CD_Irregular"), candidates=["GROUP"], required=["TIME"], participant="ID", time="TIME",
               workflow="structure", structures=["ar1", "car1", "rs_car1"], cluster="SCHOOL")
    by = {r["structure"]: r for r in run.rows}
    assert by["ar1"]["n"] == 938 and by["car1"]["n"] == 938
    assert near(by["ar1"]["loglik"], REF[("CD_Irregular", "3L_RI_AR1", "logLik")], 1e-4)
    assert near(by["car1"]["loglik"], REF[("CD_Irregular", "3L_RI_CAR1", "logLik")], 1e-4)
    assert not near(by["ar1"]["loglik"], by["car1"]["loglik"], 1e-3)           # occasion index vs actual time
    from statexplore.analysis.longitudinal_preflight import preflight

    rep = preflight(ds("CD_Missing"), "Y", "ID", "TIME", ["GROUP"], cluster="SCHOOL")
    assert rep.n_complete == 943 and rep.n_clusters == 20
    mrun = disc(ds("CD_Missing"), candidates=["GROUP"], required=["TIME"], participant="ID", time="TIME",
                workflow="structure", structures=["ri", "rs_ar1"], cluster="SCHOOL")
    assert {r["n"] for r in mrun.rows} == {943}
    assert near(next(r for r in mrun.rows if r["structure"] == "rs_ar1")["loglik"],
                REF[("CD_Missing", "3L_RS_AR1", "logLik")], 1e-4)
    from statexplore.discovery.space import SearchSettings, SearchSpaceError, validate_settings

    with pytest.raises(SearchSpaceError, match="same time point"):
        validate_settings(ds("CD_DuplicateTime"), SearchSettings(outcome="Y", family="gaussian", candidates=["GROUP"],
                                                                 participant="ID", time="TIME", cluster="SCHOOL",
                                                                 workflow="structure", structures=["ar1"]))


def test_cd10_unsupported_clustered_families_refused():
    from statexplore.discovery.space import SearchSettings, SearchSpaceError, validate_settings

    d = ds("T38")
    for st in ("mmrm_us", "gee_exch"):
        with pytest.raises(SearchSpaceError, match="never dropped silently"):
            validate_settings(d, SearchSettings(outcome="Y", family="gaussian", candidates=["GROUP"],
                                                participant="ID", time="TIME", cluster="SCHOOL", structure=st))


def test_report_preflight_complete_rows_range():
    from statexplore.analysis.longitudinal_preflight import preflight

    rep = preflight(ds("T28_29_Synthetic"), "Y", "ID", "TIME", ["X"])
    assert rep.analysis_range == (0, 5) and rep.complete_range == (1, 5)
    assert any("complete on every model variable" in w for w in rep.warnings)


def test_report_nlme_tiny_f_probabilities(r_session):
    from statexplore.analysis import mixed

    res, _ = mixed.run(r_session, ds("T38"), mixed.MixedSpec(outcome="Y", id="ID", time="TIME", predictors=["GROUP"],
                                                              cluster="SCHOOL", random="intercept", residual="ar1",
                                                              interactions="none"))
    terms = {r["term"]: r for r in table(res, "Fixed-effect term tests")}
    assert near_p(terms["sx_time"]["p"] if "sx_time" in terms else next(v for k, v in terms.items() if "TIME" in k)["p"],
                  REF[("T38", "3L_RI_AR1", "F_p.TIME")])
    assert terms["(Intercept)"]["p"] > 0


# ---- CD-08 transfer, exported code in a fresh R session; CD-11 validation; persistence ----------------------------

def test_cd08_exported_candidate_code_reproduces_in_fresh_r(r_session, tmp_path):
    from statexplore.discovery import engine
    from statexplore.export.exporters import write_portable
    from statexplore.discovery.detail import candidate_from_row
    from statexplore.discovery.rcode import candidate_script
    from statexplore.discovery.space import view

    d = ds("T38")
    run = disc(d, candidates=["GROUP"], required=["TIME"], participant="ID", time="TIME", workflow="structure",
               structures=["ar1", "rs_ar1"], cluster="SCHOOL")
    row = next(r for r in run.rows if r["structure"] == "rs_ar1")
    comps = engine.structure_tests(run).get(int(row["idx"]), [])
    v = view(d, run.settings)
    r_session.ensure_data(v)
    script = candidate_script(r_session.load_script, v, run.settings, run.family, candidate_from_row(run, row),
                              comps, expected=row)
    written = write_portable(script, tmp_path / "bundle", v, name="candidate")
    moved = tmp_path / "relocated"                                            # relocation: paths are relative
    (tmp_path / "bundle").rename(moved)
    p = rscript('source("candidate.R")', moved)
    assert p.returncode == 0, p.stderr[-3000:]
    assert "Reproduced within tolerance" in p.stdout and "reproduced within tolerance" in p.stdout
    assert written[0].name == "candidate.R"
    # a changed data file is refused, not silently analysed
    data = moved / "candidate_data.csv"
    data.write_text(data.read_text().replace("\n1,", "\n2,", 1))
    p2 = rscript('source("candidate.R")', moved)
    assert p2.returncode != 0 and "SHA-256 differs" in p2.stderr


def test_cd11_validation_targets(r_session):
    from statexplore.analysis import validation

    d = ds("T38")
    run = disc(d, candidates=["GROUP"], required=["TIME"], participant="ID", time="TIME", structure="ri",
               cluster="SCHOOL", strategy="main")
    for target in ("cluster", "participant", "temporal"):
        res, _ = validation.procedure_cv(r_session, d, run, validation.ResampleSpec(folds=5, seed=3, target=target))
        assert res.status.value in ("ok", "warning"), (target, res.messages)
        rows = table(res, "Full-procedure validation")
        assert rows and all(r["n_test"] > 0 for r in rows)
        if target == "cluster":
            assert sum(r["test_units"] for r in rows) == 20                    # every cluster held out once
    small = ds("T38")
    small.add_filter(FilterSpec("SCHOOL", "<=", 4))
    run2 = disc(small, candidates=["GROUP"], required=["TIME"], participant="ID", time="TIME", structure="ri",
                cluster="SCHOOL", strategy="main")
    res, _ = validation.procedure_cv(r_session, small, run2, validation.ResampleSpec(folds=5, target="cluster"))
    assert res.status.value == "failed" and any("unavailable" in m for m in res.messages)


def test_project_round_trip_keeps_identity_diagnostics_snapshot(tmp_path):
    from statexplore.core.project import Project

    d = ds("T38")
    run = disc(d, candidates=["GROUP"], required=["TIME"], participant="ID", time="TIME", workflow="structure",
               structures=["rs_car1"], cluster="SCHOOL")
    p = Project(d)
    p.searches.append(run)
    p.save(tmp_path / "c.sxp")
    q = Project.open(tmp_path / "c.sxp")
    r0, r1 = run.rows[0], q.searches[0].rows[0]
    assert r1["identity"] == r0["identity"] and r1["diagnostics"]["status"] == "warned"
    assert q.searches[0].settings.cluster == "SCHOOL" and q.searches[0].data_snapshot["source_sha256"]


# ---- STAT-40 duplicate category labels; STAT-31 cancellation of a long analysis ----------------------------------

def test_stat40_duplicate_labels_keep_codes_and_event(r_session, tmp_path):
    import numpy as np
    import pandas as pd

    from statexplore.discovery import detail

    rng = np.random.default_rng(40)
    x = rng.normal(size=300)
    df = pd.DataFrame({"X": x, "Y": (rng.random(300) < 1 / (1 + np.exp(-x))).astype(int),
                       "G": rng.integers(1, 4, 300)})
    import pyreadstat

    pyreadstat.write_sav(df.astype(float), str(tmp_path / "d.sav"),
                         variable_value_labels={"Y": {0: "Same", 1: "Same"}, "G": {1: "A", 2: "A", 3: "B"}},
                         variable_measure={"X": "scale", "Y": "nominal", "G": "nominal"})   # duplicated labels
    d = import_file(tmp_path / "d.sav")
    run = disc(d, outcome="Y", family="binomial", event=1, candidates=["X", "G"], strategy="main", workers=1)
    row = next(r for r in run.rows if r["status"] in ("ok", "warning") and len(r["covariates"]) == 2)
    assert row["n"] == 300
    res, _ = detail.full_analysis(r_session, d, run, detail.candidate_from_row(run, row))
    assert res.status.value in ("ok", "warning"), res.messages
    assert res.values["identity_check"] == "verified"                         # same event coding and levels


def test_stat31_cancel_long_task_no_orphan(r_worker):
    import threading
    import time as _t

    from statexplore.rbridge.worker import REngineError, RScriptError, RTimeoutError

    gen = r_worker.generation
    err: list[Exception] = []

    def long():
        try:
            r_worker.run("Sys.sleep(60); sx_result <- 1", timeout=120)
        except (REngineError, RScriptError, RTimeoutError) as e:
            err.append(e)

    t = threading.Thread(target=long)
    t.start()
    _t.sleep(2)
    pid = r_worker._proc.pid if getattr(r_worker, "_proc", None) else None
    r_worker.kill()                                                              # what the Stop R button does
    t.join(30)
    assert not t.is_alive() and err                                              # the task reports failure
    if pid:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout
        assert str(pid) not in out                                               # no orphan R process
    r_worker.start()
    assert r_worker.generation > gen and r_worker.ping()
