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
LMM_ENGINES = {"ri": "lme4::lmer", "rs": "lme4::lmer", "rs_uncorr": "lme4::lmer", "nlme_ri": "nlme::lme",
               "nlme_rs": "nlme::lme", "ar1": "nlme::lme", "car1": "nlme::lme", "mmrm_us": "mmrm::mmrm",
               "mmrm_cs": "mmrm::mmrm", "mmrm_ar1": "mmrm::mmrm"}
GEE_CORSTR = {"gee_exch": "exchangeable", "gee_ar1": "ar1", "gee_ind": "independence"}
TIME_INDEX = {
    "ar1": "observed-occasion index (sx_occ = rank of the observed time values; globally absent occasions are "
           "compressed, so lag 1 = the next observed occasion)",
    "car1": "actual time values (correlation decays with the numeric time difference)",
    "gee_ar1": "observed-occasion index (waves = rank of the observed time values)",
    "mmrm_ar1": "observed occasions as categories",
}

# Tolerances for a refit to count as the same model (optimizer-dependent statistics).
TOL_LOGLIK = 0.01
TOL_AIC = 0.02


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
    n_fixed: int | None = None      # fixed coefficients (+ thresholds / contrasts) counted by the engine
    n_cov: int | None = None        # random-effect + residual covariance / correlation parameters
    flags: str = ""
    treatments: list[dict[str, Any]] = field(default_factory=list)

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
                                     + ([s.participant] if s.participant else []) + ([s.time] if s.time else []))) \
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
        n_fixed=_int(row.get("n_params")), n_cov=_int(row.get("n_re_params")),
        flags=str(row.get("flags") or ""), treatments=[dict(t) for t in getattr(s, "treatments", [])])


def verify(d: ModelDescriptor | dict[str, Any] | None, n: Any = None, loglik: Any = None,
           aic: Any = None) -> tuple[bool | None, str]:
    """Compare a refit with its origin. Returns (True, note) when n, logLik and AIC agree within tolerance,
    (False, reason) when any differs, (None, '') when there is nothing to compare."""
    if d is None:
        return None, ""
    if isinstance(d, dict):
        d = ModelDescriptor.from_dict(d)
    checks, diffs = [], []
    if d.n is not None and _int(n) is not None:
        checks.append("n")
        if _int(n) != d.n:
            diffs.append(f"n {_int(n):,} vs {d.n:,} in the search")
    if d.loglik is not None and _num(loglik) is not None:
        checks.append("log-likelihood")
        if abs(_num(loglik) - d.loglik) > TOL_LOGLIK:
            diffs.append(f"logLik {_num(loglik):.3f} vs {d.loglik:.3f}")
    # mmrm's own AIC uses a different parameter count from the search's single convention: compare logLik only
    if d.aic is not None and _num(aic) is not None and d.engine != "mmrm::mmrm":
        checks.append("AIC")
        if abs(_num(aic) - d.aic) > TOL_AIC:
            diffs.append(f"AIC {_num(aic):.3f} vs {d.aic:.3f}")
    if not checks:
        return None, ""
    if diffs:
        return False, (f"Model identity check FAILED against {d.source}: " + "; ".join(diffs)
                       + ". This refit is not the model the search reported; do not interpret it as that model.")
    return True, (f"Model identity verified against {d.source}: {', '.join(checks)} agree "
                  f"({d.engine}, {d.estimation}{', structure ' + d.structure if d.structure else ''}).")


def fit_stats(result: Any) -> dict[str, Any]:
    """n, logLik and AIC from a result's tables (first 'Model fit'-like table holding them)."""
    out: dict[str, Any] = {"n": result.n_used}
    tables = sorted(result.tables, key=lambda t: t.name != "Model fit")
    for t in tables:
        cols = {c.lower(): i for i, c in enumerate(t.columns)}
        if ("loglik" in cols or "aic" in cols) and t.rows:
            for key in ("loglik", "aic"):
                if key in cols:
                    out[key] = t.rows[0][cols[key]]
            if "n" in cols:
                out["n"] = t.rows[0][cols["n"]]
            break
    return out


def attach_check(result: Any, d: ModelDescriptor | dict[str, Any] | None) -> None:
    """Record the identity check on a refit result (message + values) and downgrade its status on failure."""
    from ..core.schema import Status

    if d is None:
        return
    st = fit_stats(result)
    ok, text = verify(d, st.get("n"), st.get("loglik"), st.get("aic"))
    desc = d if isinstance(d, ModelDescriptor) else ModelDescriptor.from_dict(d)
    result.values["origin_model"] = desc.summary()
    result.values["identity_check"] = {True: "verified", False: "FAILED", None: "not available"}[ok]
    if text:
        result.messages.insert(0, text if ok else f"Check: {text}")
    if ok is False:
        result.status = Status.WARNING
