"""The authoritative identity of a fitted discovery model (v1.3 P0-01).

A `ModelDescriptor` is built once from a discovery row and its run, and every downstream action (Full
Analysis, Add to Comparison, Compare with main-effects model, Mixed workspace, View candidate code, reports)
is derived from it instead of from whatever UI controls happen to be selected. Destinations that refit the
model call `verify()` with the refit's statistics: a refit that differs from the origin in n, log-likelihood
or AIC is reported, never shown as if it were the same model.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

# run family -> (outcome family, engine, link)
ENGINES = {
    "gaussian": ("gaussian", "stats::lm", "identity"),
    "binomial": ("binomial", "stats::glm", "logit"),
    "ordinal": ("ordinal", "ordinal::clm", "cumulative logit"),
    "multinomial": ("multinomial", "nnet::multinom", "baseline-category logit"),
    "glmm": ("binomial", "lme4::glmer", "logit"),
    "clmm": ("ordinal", "ordinal::clmm", "cumulative logit"),
}
from .covstruct import GAUSSIAN_STRUCTURES as _GS  # noqa: E402

LMM_ENGINES = {**{k: ("lme4::lmer" if v[2] == "lme4" else "nlme::lme") for k, v in _GS.items()},
               "mmrm_us": "mmrm::mmrm", "mmrm_cs": "mmrm::mmrm", "mmrm_ar1": "mmrm::mmrm"}
GEE_CORSTR = {"gee_exch": "exchangeable", "gee_ar1": "ar1", "gee_ind": "independence"}
TIME_INDEX = {
    **{k: "global observed-occasion index (rank of the time values present in the analysed rows)"
       for k, v in _GS.items() if v[1] == "ar1" and k != "ar1"},
    **{k: "actual time values (correlation decays with the numeric time difference)"
       for k, v in _GS.items() if v[1] == "car1" and k != "car1"},
    "ar1": "global observed-occasion index (sx_occ = rank of the time values present in the analysed rows; a time "
           "value no analysed row has is compressed; a participant who skips an occasion others have keeps an index "
           "gap, correlation phi^gap)",
    "car1": "actual time values (correlation decays with the numeric time difference)",
    "gee_ar1": "observed-occasion index (waves = rank of the observed time values)",
    "mmrm_ar1": "observed occasions as categories",
}

# Tolerances for a refit to count as the same model (optimizer-dependent statistics).
TOL_LOGLIK = 0.01
TOL_AIC = 0.02
TOL_BIC = 0.0015   # STAT-16: the same model and estimator must show the same BIC to three decimals


class TransferError(ValueError):
    """User-facing reason why a destination cannot represent this model faithfully."""


@dataclass
class ModelDescriptor:
    source: str
    run_id: str
    candidate: int
    family: str                     # run family: gaussian | binomial | ordinal | multinomial | lmm | glmm | clmm | gee
    outcome_family: str             # gaussian | binomial | ordinal | multinomial
    engine: str
    link: str
    outcome: str
    terms: list[list[str]]
    required: list[str] = field(default_factory=list)
    event: Any = None
    reference: Any = None
    ordering: list[str] = field(default_factory=list)
    participant: str | None = None
    time: str | None = None
    time_type: str = ""
    structure: str = ""
    time_index: str = ""
    estimation: str = ""            # ML | REML | OLS | GEE
    workflow: str = ""
    sample: str = ""                # shared | per model
    sample_vars: list[str] = field(default_factory=list)
    n: int | None = None
    n_groups: int | None = None
    loglik: float | None = None
    aic: float | None = None
    bic: float | None = None
    qic: float | None = None
    ic_basis: str = ""              # "native" or the convention used (v1.3.1+; empty = older rows)
    rows_sig: str = ""              # MD5 of the analysed row identities + factor levels (v1.3.1+)
    dataset_signature: str = ""     # data signature of the searched dataset (metadata, filters, coding)
    n_fixed: int | None = None      # fixed coefficients (+ thresholds / contrasts) counted by the engine
    n_cov: int | None = None        # random-effect + residual covariance / correlation parameters
    flags: str = ""
    treatments: list[dict[str, Any]] = field(default_factory=list)
    # ---- v1.3.2: identity fingerprints (schema 2, SHA-256) and the canonical structure (H4) ----
    identity: dict[str, Any] = field(default_factory=dict)   # rows / response / design / groups of the origin fit
    cluster: str | None = None
    id_policy: str = ""
    clustered: bool = False         # this candidate's random part includes the cluster (B exploratory may drop it)
    random: str = ""                # ri | diag | rs (Gaussian longitudinal)
    residual: str = ""              # independent | ar1 | car1
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def longitudinal(self) -> bool:
        return bool(self.participant)

    @property
    def n_likelihood(self) -> int | None:
        if self.n_fixed is None:
            return None
        return self.n_fixed + (self.n_cov or 0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelDescriptor":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})

    def summary(self) -> str:
        bits = [f"{self.engine} ({self.outcome_family}, {self.link})"]
        if self.structure:
            bits.append(f"structure {self.structure}")
        if self.time_index:
            bits.append(f"time index: {self.time_index.split(' (')[0]}")
        if self.estimation:
            bits.append(self.estimation)
        if self.n is not None:
            bits.append(f"n = {self.n:,}" + (f", {self.n_groups:,} groups" if self.n_groups else ""))
        if self.loglik is not None:
            bits.append(f"logLik = {self.loglik:.3f}")
        if self.aic is not None:
            bits.append(f"AIC = {self.aic:.3f}")
        return "; ".join(bits)


def _num(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _int(x: Any) -> int | None:
    v = _num(x)
    return None if v is None else int(round(v))


def candidate_structure(run: Any, row: dict[str, Any]) -> str:
    """The structure this candidate was fitted with: its own (workflow B) or the search's locked one."""
    s = run.settings
    if not s.participant:
        return ""
    return str(row.get("structure") or "") or (s.structures[0] if s.workflow == "structure" and s.structures
                                               else s.structure) or "ri"


def _gauss(st: str) -> tuple[str, str]:
    from .covstruct import GAUSSIAN_STRUCTURES

    v = GAUSSIAN_STRUCTURES.get(st)
    return (v[0], v[1]) if v else ("", "")


def from_discovery(run: Any, row: dict[str, Any]) -> ModelDescriptor:
    from ..discovery.detail import candidate_from_row
    from ..discovery.space import full_terms

    s = run.settings
    if row.get("status") in ("invalid", "failed"):
        raise TransferError(f"This candidate was not fitted: {row.get('reason') or row.get('status')}")
    c = candidate_from_row(run, row)
    fam = run.family
    st = candidate_structure(run, row)
    if fam == "lmm":
        out_fam, engine, link = "gaussian", LMM_ENGINES.get(st, "lme4::lmer"), "identity"
    elif fam == "gee":
        out_fam = s.family or "gaussian"
        engine, link = "geepack::geeglm", "logit" if out_fam == "binomial" else "identity"
    else:
        out_fam, engine, link = ENGINES[fam]
    workflow = s.workflow if s.participant else ""
    if fam == "lmm":
        estimation = "REML" if workflow == "structure" else "ML"
    elif fam == "gee":
        estimation = "GEE (robust sandwich SE)"
    elif fam == "gaussian":
        estimation = "OLS (= ML for the coefficients)"
    elif fam in ("glmm", "clmm"):
        estimation = "ML (Laplace)"
    else:
        estimation = "ML"
    coding = run.coding or {}
    sample_vars = list(dict.fromkeys([s.outcome] + list(s.candidates) + list(s.required)
                                     + ([s.participant] if s.participant else []) + ([s.time] if s.time else [])
                                     + ([s.cluster] if s.participant and getattr(s, "cluster", None) else []))) \
        if s.sample == "shared" else []
    return ModelDescriptor(
        source=f"discovery search {run.id}, candidate #{row['idx']}", run_id=run.id, candidate=int(row["idx"]),
        family=fam, outcome_family=out_fam, engine=engine, link=link, outcome=s.outcome,
        terms=[list(t) for t in full_terms(c)], required=list(s.required),
        event=s.event if out_fam == "binomial" else None,
        reference=s.reference if out_fam == "multinomial" else None,
        ordering=[str(x) for x in coding.get("ordering") or []],
        participant=s.participant, time=s.time if s.participant else None,
        time_type=s.time_type if s.participant else "", structure=st,
        time_index=TIME_INDEX.get(st, ""), estimation=estimation, workflow=workflow,
        sample="shared" if s.sample == "shared" else "per model", sample_vars=sample_vars,
        n=_int(row.get("n")), n_groups=_int(row.get("n_groups")), loglik=_num(row.get("loglik")),
        aic=_num(row.get("aic")), bic=_num(row.get("bic")), qic=_num(row.get("qic")),
        ic_basis=str(row.get("ic_basis") or ""),
        rows_sig=str(row.get("rows_sig") or ""), dataset_signature=str(getattr(run, "signature", "") or ""),
        n_fixed=_int(row.get("n_params")), n_cov=_int(row.get("n_re_params")),
        flags=str(row.get("flags") or ""), treatments=[dict(t) for t in getattr(s, "treatments", [])],
        identity=dict(row.get("identity") or {}), cluster=getattr(s, "cluster", None) if s.participant else None,
        id_policy=getattr(s, "id_policy", "global") if getattr(s, "cluster", None) else "",
        clustered=bool(row.get("clustered")) if s.participant else False,
        random=_gauss(st)[0], residual=_gauss(st)[1], diagnostics=dict(row.get("diagnostics") or {}))


IDENTITY_PARTS = {"rows": "analysed rows (source row identities)", "response": "outcome values or outcome coding",
                  "design": "fixed-effect design (predictor values, coding, reference levels or terms)",
                  "groups": "participant / cluster grouping"}
STATUS_TEXT = {"verified": "verified", "mismatch": "MISMATCH", "unverifiable": "unverifiable",
               "refit failed": "refit failed", "new analysis": "new analysis (not the historical model)"}


def parse_identity(raw: Any) -> dict[str, Any]:
    """The `identity` list returned by R's sx_identity(), flattened (length-1 vectors -> scalars)."""
    if not isinstance(raw, dict):
        return {}
    out = {}
    for k, v in raw.items():
        out[k] = v[0] if isinstance(v, list) and len(v) == 1 and k != "design_columns" else v
    return out


def structural_check(origin: dict[str, Any] | None, refit: dict[str, Any] | None) -> tuple[str, list[str]]:
    """('verified' | 'mismatch' | 'unverifiable', differences). Semantic identity is decided by fingerprints of what
    the engine actually used: equal n or equal fit statistics never establish it (STAT-22)."""
    if not origin or int(origin.get("schema") or 0) < 2 or not origin.get("rows"):
        return "unverifiable", ["the origin has no identity fingerprints (created before v1.3.2 or not recorded)"]
    if not refit or int(refit.get("schema") or 0) < 2 or not refit.get("rows"):
        return "unverifiable", ["the refit returned no identity fingerprints"]
    diffs = [IDENTITY_PARTS[k] for k in ("rows", "response", "design", "groups")
             if origin.get(k) and refit.get(k) and origin[k] != refit[k]]
    if origin.get("n") is not None and refit.get("n") is not None and int(origin["n"]) != int(refit["n"]):
        diffs.insert(0, f"n ({int(refit['n']):,} vs {int(origin['n']):,})")
    return ("mismatch" if diffs else "verified"), diffs


def verify(d: ModelDescriptor | dict[str, Any] | None, n: Any = None, loglik: Any = None,
           aic: Any = None, bic: Any = None, identity: dict[str, Any] | None = None,
           failed: bool = False) -> tuple[str | None, str]:
    """Compare a refit with its origin: structural identity first (fingerprints), numerical agreement separately.
    Returns (status, note); status is None when there is nothing to compare."""
    if d is None:
        return None, ""
    if isinstance(d, dict) and d.get("new_analysis"):
        return "new analysis", ("NEW analysis under the current data and settings (the project changed since the "
                                "search: " + "; ".join(d.get("new_analysis_reasons") or []) + "); no identity with "
                                "the historical model is claimed.")
    if isinstance(d, dict):
        d = ModelDescriptor.from_dict(d)
    if failed:
        return "refit failed", (f"Refit FAILED for {d.source}: the original model could not be reproduced here. The "
                                "stored search result is unchanged; nothing in this result describes that model.")
    struct, diffs = structural_check(d.identity, identity)
    checks, ndiffs = [], []
    if d.loglik is not None and _num(loglik) is not None:
        checks.append("log-likelihood")
        if abs(_num(loglik) - d.loglik) > TOL_LOGLIK:
            ndiffs.append(f"logLik {_num(loglik):.3f} vs {d.loglik:.3f}")
    # rows from before v1.3.1 (no ic_basis) computed mmrm's AIC and every BIC on another convention: skip those
    if d.aic is not None and _num(aic) is not None and (d.engine != "mmrm::mmrm" or d.ic_basis):
        checks.append("AIC")
        if abs(_num(aic) - d.aic) > TOL_AIC:
            ndiffs.append(f"AIC {_num(aic):.3f} vs {d.aic:.3f}")
    if d.bic is not None and _num(bic) is not None and d.ic_basis:
        checks.append("BIC")
        if abs(_num(bic) - d.bic) > TOL_BIC:
            ndiffs.append(f"BIC {_num(bic):.3f} vs {d.bic:.3f}")
    numeric = ("not compared" if not checks else "agree (" + ", ".join(checks) + ")" if not ndiffs
               else "differ: " + "; ".join(ndiffs))
    what = f"({d.engine}, {d.estimation}{', structure ' + d.structure if d.structure else ''})"
    if struct == "mismatch":
        return "mismatch", (f"Model identity MISMATCH against {d.source}: " + "; ".join(diffs) + " differ. Numerical "
                            f"check: {numeric}. Equal fit statistics do not make it the same model: this refit is NOT "
                            "the model the search reported; do not interpret it as that model.")
    if struct == "unverifiable":
        return "unverifiable", (f"Model identity UNVERIFIABLE against {d.source}: {diffs[0]}. Numerical check: "
                                f"{numeric}. Matching statistics alone cannot show that the rows, coding and design "
                                "are the same, so this refit is not certified as the original model.")
    if ndiffs:
        return "mismatch", (f"Structural identity verified against {d.source} (same rows, outcome, design and grouping "
                            f"{what}), but the numerical results differ: {numeric}. The same model reached a different "
                            "optimum (optimizer / package version); treat the two results as different estimates.")
    return "verified", (f"Model identity verified against {d.source}: structural (rows, outcome, design, grouping "
                        f"fingerprints, SHA-256) and numerical ({numeric}) {what}.")


def fit_stats(result: Any) -> dict[str, Any]:
    """n, logLik and AIC from a result's tables (first 'Model fit'-like table holding them)."""
    out: dict[str, Any] = {"n": result.n_used}
    tables = sorted(result.tables, key=lambda t: t.name != "Model fit")
    for t in tables:
        cols = {c.lower(): i for i, c in enumerate(t.columns)}
        if ("loglik" in cols or "aic" in cols) and t.rows:
            for key in ("loglik", "aic", "bic"):
                if key in cols:
                    out[key] = t.rows[0][cols[key]]
            if "n" in cols:
                out["n"] = t.rows[0][cols["n"]]
            break
    return out


def provenance(d: ModelDescriptor, result: Any = None) -> dict[str, str]:
    """The immutable identity of a transferred model, shown with every refit and export (v1.3.1 §6.1)."""
    terms = " + ".join(":".join(t) for t in d.terms) or "1"
    out = {
        "Origin": d.source,
        "Outcome / family / link": f"{d.outcome} / {d.outcome_family} / {d.link}",
        "Fixed part": f"{d.outcome} ~ {terms}" + (f" (required: {', '.join(d.required)})" if d.required else ""),
        "Engine / estimation": f"{d.engine} / {d.estimation}",
    }
    if d.event is not None:
        out["Event category"] = str(d.event)
    if d.reference is not None:
        out["Reference category"] = str(d.reference)
    if d.ordering:
        out["Outcome order"] = " < ".join(d.ordering)
    if d.participant:
        out["Participants / time"] = f"{d.participant} / {d.time or '—'} ({d.time_type or '—'})"
    if d.structure:
        out["Covariance structure"] = d.structure + (f"; time index: {d.time_index}" if d.time_index else "")
    out["Sample"] = f"{d.sample}; n = {d.n if d.n is not None else '—'}" + (
        f", groups = {d.n_groups}" if d.n_groups else "")
    if d.rows_sig:
        out["Analysis-row signature"] = d.rows_sig
    if d.dataset_signature:
        out["Dataset fingerprint"] = d.dataset_signature[:16]
    if d.ic_basis:
        out["AIC/BIC basis"] = d.ic_basis
    eng = getattr(result, "engine", None)
    if eng is not None and getattr(eng, "packages", None):
        out["R / packages"] = f"{eng.version}; " + ", ".join(f"{k} {v}" for k, v in eng.packages.items() if v)
    return out


def attach_check(result: Any, d: ModelDescriptor | dict[str, Any] | None) -> None:
    """Record the identity check on a refit result (message + values) and downgrade its status unless verified."""
    from ..core.schema import Status

    if d is None:
        return
    if isinstance(d, dict) and d.get("new_analysis"):
        desc = ModelDescriptor.from_dict(d)
        result.values["origin_model"] = desc.summary()
        result.values["provenance"] = provenance(desc, result)
        result.values["identity_check"] = "new analysis (not the historical model)"
        result.messages.insert(0, f"Check: NEW analysis under the current data and settings, started from {desc.source}. "
                                  "The project changed since that search (" + "; ".join(d.get("new_analysis_reasons") or [])
                               + "), so this is not the historical model and no identity with it is claimed.")
        if result.status is Status.OK:
            result.status = Status.WARNING
        return
    failed = result.status is Status.FAILED
    st = fit_stats(result) if not failed else {}
    status, text = verify(d, st.get("n"), st.get("loglik"), st.get("aic"), st.get("bic"),
                          identity=result.values.get("identity"), failed=failed)
    desc = d if isinstance(d, ModelDescriptor) else ModelDescriptor.from_dict(d)
    result.values["origin_model"] = desc.summary()
    result.values["provenance"] = provenance(desc, result)
    result.values["identity_check"] = STATUS_TEXT.get(status or "", "not available")
    if text:
        result.messages.insert(0, text if status == "verified" else f"Check: {text}")
    if status in ("mismatch", "unverifiable") and result.status is Status.OK:
        result.status = Status.WARNING
