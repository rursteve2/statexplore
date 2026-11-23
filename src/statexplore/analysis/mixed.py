"""Mixed-effects and longitudinal models (§9.6, §9.7, §15).

One default engine per covariance structure, recorded with every result:
  independent residuals  -> lme4/lmerTest (Satterthwaite df)
  AR(1) / CAR(1)         -> nlme::lme (random effects + residual correlation)
  unstructured           -> mmrm (marginal model, no random effects)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..core.dataset import Dataset
from ..core.schema import AnalysisResult, Status
from ..core.variables import Measure, Storage
from ..rbridge.codegen import alias, level_labels, r_num, r_str, r_vec
from .session import RSession

TIME_TYPES = {"continuous": "Continuous (linear)", "quadratic": "Continuous (quadratic)",
              "categorical": "Categorical (one mean per occasion)"}
RANDOM = {"intercept": "Random intercept", "slope": "Random intercept + time slope (correlated)",
          "slope_uncorrelated": "Random intercept + time slope (uncorrelated)", "none": "None (marginal model)"}
RESIDUAL = {"independent": "Independent", "ar1": "AR(1) — observed-occasion index (gaps compressed)",
            "car1": "Continuous-time AR(1) — actual time values", "unstructured": "Unstructured (marginal, mmrm)"}
INTERACTIONS = {"none": "None", "time": "Time × each predictor", "all": "All two-way"}
FAMILIES = {"gaussian": "Gaussian (continuous)", "binomial": "Binary logistic", "ordinal": "Ordinal logistic (cumulative logit)",
            "multinomial": "Multinomial nominal (baseline-category logit)"}
ESTIMATORS = {"mixed": "Mixed-effects model (subject-specific)", "gee": "GEE (population-average, geepack)"}
INFERENCE = {"satterthwaite": "Satterthwaite df (lmerTest; default)", "kenward_roger": "Kenward-Roger df (lmerTest + pbkrtest; REML)",
             "lrt": "Likelihood-ratio tests (ML refits)"}
GEE_CORSTR = {"exchangeable": "Exchangeable", "ar1": "AR(1) (occasion order)", "independence": "Independence",
              "unstructured": "Unstructured"}
ENGINE_TITLES = {
    "lme4": "Linear mixed model", "nlme": "Linear mixed model with residual autocorrelation",
    "mmrm": "Marginal repeated-measures model (unstructured)",
    "glmer": "Binary logistic mixed model (lme4::glmer)", "clmm": "Ordinal logistic mixed model (ordinal::clmm)",
    "mblogit": "Multinomial logistic mixed model (mclogit::mblogit, PQL)",
    "geepack": "Generalized estimating equations (geepack::geeglm)",
}
ENGINE_PACKAGES = {"lme4": ["lme4", "lmerTest", "performance"], "nlme": ["nlme", "performance"], "mmrm": ["mmrm", "car"],
                   "glmer": ["lme4", "MASS"], "clmm": ["ordinal"], "mblogit": ["mclogit"], "geepack": ["geepack"]}


class MixedInputError(ValueError):
    pass


STRUCTURE_SPECS = {   # discovery structure -> Mixed workspace settings (None: not available here)
    "ri": dict(random="intercept"), "rs": dict(random="slope"), "rs_uncorr": dict(random="slope_uncorrelated"),
    "nlme_ri": dict(random="intercept", force_nlme=True), "nlme_rs": dict(random="slope", force_nlme=True),
    "ar1": dict(random="intercept", residual="ar1"), "car1": dict(random="intercept", residual="car1"),
    "mmrm_us": dict(random="none", residual="unstructured"), "mmrm_cs": None, "mmrm_ar1": None,
    "glmer_ri": dict(random="intercept"), "glmer_rs": dict(random="slope"),
    "clmm_ri": dict(random="intercept"), "clmm_rs": dict(random="slope"),
    "gee_exch": dict(estimator="gee", gee_corstr="exchangeable"), "gee_ar1": dict(estimator="gee", gee_corstr="ar1"),
    "gee_ind": dict(estimator="gee", gee_corstr="independence"),
}


def spec_from_discovery(run: Any, row: dict[str, Any]) -> "MixedSpec":
    """The exact model of a longitudinal discovery candidate for the Mixed workspace, built from its
    ModelDescriptor (P0-01): same formula, structure, estimator (ML for workflow A, REML for workflow B
    Gaussian structure comparisons), coding and shared sample. The refit is checked against the search's
    n / log-likelihood / AIC when it runs; a structure the workspace cannot represent is refused with a reason."""
    from ..discovery.space import STRUCTURES
    from .descriptor import TransferError, from_discovery

    s = run.settings
    try:
        d = from_discovery(run, row)
    except TransferError as e:
        raise MixedInputError(str(e)) from None
    st = d.structure
    label = STRUCTURES.get(st, (st,))[0]
    extra = STRUCTURE_SPECS.get(st)
    if extra is None:
        raise MixedInputError(f"The Mixed workspace cannot fit this candidate's structure ({label}; {d.engine}). "
                              "It was not replaced by another model: the search result stays available, and View "
                              "candidate code reproduces the exact fit.")
    if st == "mmrm_us" and s.time_type != "categorical":
        raise MixedInputError("This candidate uses an unstructured covariance over occasions with continuous time in "
                              "the fixed part; the Mixed workspace's unstructured model needs categorical time, so it "
                              "cannot refit the same model. View candidate code reproduces the exact fit.")
    terms = [list(t) for t in d.terms]
    preds = list(dict.fromkeys(v for t in terms for v in t if v != s.time))
    fam = s.family or "gaussian"
    method = "REML" if d.estimation == "REML" else "ML"
    spec = MixedSpec(outcome=s.outcome, id=s.participant, time=s.time, time_type=s.time_type, predictors=preds,
                     interactions="none", method=method, family=fam, event=s.event, reference=s.reference,
                     treatments=[dict(t) for t in s.treatments], terms=terms,
                     max_fixed=max(6, len(preds) + (1 if s.time else 0)),
                     sample_vars=[v for v in d.sample_vars if v not in [s.outcome, s.participant, s.time] + preds],
                     descriptor=d.to_dict(),
                     source=f"{d.source} (refitted here with the same formula, structure {label}, {method} "
                            "estimation, coding and rows)", **extra)
    # inference the model's engine supports (P1-01): term LR tests on ML lme4 fits; nlme / mmrm report their own df
    spec.inference = "lrt" if spec.engine == "lme4" and method == "ML" and fam == "gaussian" else "satterthwaite"
    return spec


@dataclass
class MixedSpec:
    outcome: str
    id: str
    time: str | None = None
    time_type: str = "continuous"
    predictors: list[str] = field(default_factory=list)
    interactions: str = "time"
    random: str = "intercept"
    residual: str = "independent"
    cluster: str | None = None          # higher-level grouping; participants nested within clusters
    method: str = "REML"
    conf: float = 0.95
    n_trajectories: int = 40
    seed: int = 20261121
    force_nlme: bool = False            # comparisons of residual structures fit every variant in nlme
    treatments: list[dict[str, Any]] = field(default_factory=list)   # analysis-specific encodings (v1.1)
    # ---- v1.1 (§11): outcome family, estimator and inference ----
    family: str = "gaussian"            # gaussian | binomial | ordinal | multinomial
    estimator: str = "mixed"            # mixed (subject-specific) | gee (population-average, geepack)
    inference: str = "satterthwaite"    # Gaussian lme4: satterthwaite | kenward_roger | lrt
    event: Any = None                   # binomial: event category code (None = second category)
    reference: Any = None               # multinomial: baseline category code (None = first category)
    nagq: int = 1                       # GLMM: 1 = Laplace; > 1 = adaptive Gauss-Hermite (one random intercept)
    gee_corstr: str = "exchangeable"    # GEE working correlation
    # ---- v1.2: exact fixed terms (a discovery candidate refitted here); overrides `interactions` ----
    terms: list[list[str]] | None = None
    source: str = ""                    # e.g. "discovery run abc #12 (refit)"
    max_fixed: int = 6                  # complexity limit: distinct fixed-effect predictors incl. time (M §10)
    # ---- v1.3: identity of a transferred model ----
    sample_vars: list[str] = field(default_factory=list)   # extra variables defining a shared complete-case sample
    descriptor: dict[str, Any] | None = None               # origin ModelDescriptor (refit is verified against it)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def engine(self) -> str:
        if self.estimator == "gee":
            return "geepack"
        if self.family != "gaussian":
            return {"binomial": "glmer", "ordinal": "clmm", "multinomial": "mblogit"}[self.family]
        if self.residual == "unstructured":
            return "mmrm"
        if self.residual in ("ar1", "car1") or self.force_nlme:
            return "nlme"
        return "lme4"


def view(ds: Dataset, s: MixedSpec) -> Dataset:
    """The dataset as this model sees it (analysis-specific treatments applied)."""
    from .measurement import TreatmentError, apply_treatments, as_treatments

    try:
        return apply_treatments(ds, as_treatments(s.treatments))
    except TreatmentError as e:
        raise MixedInputError(str(e)) from None


def _validate_family(ds: Dataset, s: MixedSpec) -> None:
    """Model-family-specific outcome and structure rules (§11), replacing v1.0's continuous-only rule."""
    v = ds.var(s.outcome)
    dn = v.display_name
    k = ds.analysis_frame([s.outcome])[s.outcome].dropna().nunique()
    if s.family not in FAMILIES or s.estimator not in ESTIMATORS or s.inference not in INFERENCE:
        raise MixedInputError("Invalid option.")
    if s.family == "gaussian":
        if not (v.measure is Measure.CONTINUOUS and v.storage is Storage.NUMERIC):
            hint = (f" It is {v.measure.value}: choose the ordinal or multinomial family, or tick \"Treat {dn} as "
                    "continuous for this analysis\" to model its numeric codes." if v.is_categorical else "")
            raise MixedInputError(f"A Gaussian model needs a continuous outcome.{hint}")
    elif s.family == "binomial":
        if k != 2:
            raise MixedInputError(f"Binary logistic models need exactly two outcome values; {dn} has {k}.")
        if s.event is not None and s.event not in ds.levels(s.outcome):
            raise MixedInputError(f"{s.event!r} is not a value of {dn}.")
    else:
        if not v.is_categorical or k < 3:
            raise MixedInputError(f"{FAMILIES[s.family]} models need a categorical outcome with 3+ categories.")
        if s.family == "ordinal" and v.measure is not Measure.ORDINAL:
            raise MixedInputError(f"{dn} is {v.measure.value}: ordinal models need ordered categories (approve an ordinal "
                                  "level in Dataset Explorer ▸ Measurement Review), or choose the multinomial family.")
        if s.family == "multinomial" and s.reference is not None and s.reference not in ds.levels(s.outcome):
            raise MixedInputError(f"{s.reference!r} is not a category of {dn}.")
    if s.estimator == "gee":
        if s.family not in ("gaussian", "binomial"):
            raise MixedInputError("GEE is available for continuous and binary outcomes.")
        if s.gee_corstr not in GEE_CORSTR:
            raise MixedInputError("Invalid working correlation.")
        if s.gee_corstr in ("ar1", "unstructured") and not s.time:
            raise MixedInputError("AR(1) and unstructured working correlations need a time variable (occasion order).")
        if s.cluster:
            raise MixedInputError("GEE here clusters on participants only; remove the higher-level cluster.")
        return
    if s.family != "gaussian":
        if s.residual != "independent":
            raise MixedInputError("Residual correlation structures (AR(1), unstructured) belong to Gaussian models. "
                                  f"{FAMILIES[s.family]} mixed models use random effects only; for serial dependence in "
                                  "a binary outcome consider GEE with an AR(1) working correlation.")
        if s.random == "none":
            raise MixedInputError("A mixed model needs at least a random intercept (or choose GEE).")
        if s.family == "multinomial" and (s.random != "intercept" or s.cluster):
            raise MixedInputError("Multinomial mixed models (mclogit::mblogit) are supported with a random intercept per "
                                  "participant only (no random slopes, no higher-level cluster).")
        if s.nagq > 1 and (s.random != "intercept" or s.cluster or s.family == "multinomial"):
            raise MixedInputError("Adaptive Gauss-Hermite quadrature (nAGQ > 1) needs a single random intercept; use "
                                  "the Laplace approximation (nAGQ = 1) for this structure.")
        if s.inference != "satterthwaite":
            raise MixedInputError("Satterthwaite / Kenward-Roger denominator df and the inference option apply to "
                                  "Gaussian lme4 models only; GLMMs report Wald z tests and likelihood-ratio term tests.")
    elif s.inference != "satterthwaite":
        if s.engine != "lme4":
            raise MixedInputError("This inference setting is implementation-specific: Kenward-Roger df (pbkrtest) and the "
                                  "automatic term-wise likelihood-ratio refits are implemented here for lme4 models with "
                                  "independent residuals. nlme and mmrm models report their own df (nlme containment, "
                                  "mmrm Satterthwaite): choose 'Satterthwaite df (default)' to fit this model. "
                                  "Likelihood-ratio tests between nested nlme ML fits remain valid (Compare structures / "
                                  "Discovery workflow A).")
        if s.inference == "kenward_roger" and s.method != "REML":
            raise MixedInputError("Kenward-Roger df are defined for REML fits: set estimation to REML.")


def validate(ds: Dataset, s: MixedSpec) -> None:
    ds = view(ds, s)
    _validate_family(ds, s)
    v = ds.var(s.outcome)
    names = [s.outcome, s.id] + ([s.time] if s.time else []) + ([s.cluster] if s.cluster else []) + s.predictors
    if len(set(names)) != len(names):
        raise MixedInputError("Each variable can have only one role.")
    n_fixed = len(s.predictors) + (1 if s.time else 0)
    if n_fixed > s.max_fixed:
        raise MixedInputError(f"{n_fixed} distinct fixed-effect predictors (time included) exceed the complexity limit "
                              f"of {s.max_fixed}. Raise 'Max fixed-effect predictors' to fit a larger model.")
    from .models import variable_problem

    frame = ds.analysis_frame(list(dict.fromkeys(names))).dropna()
    if len(frame) < 10:
        raise MixedInputError(f"Only {len(frame)} rows have no missing values on these variables.")
    for n in [s.outcome] + s.predictors:
        why = variable_problem(ds, n, frame)
        if why:
            raise MixedInputError(why)
    for n in s.predictors:
        pv = ds.var(n)
        if not (pv.is_categorical or (pv.measure is Measure.CONTINUOUS and pv.storage is Storage.NUMERIC)):
            raise MixedInputError(f"{pv.display_name} cannot be used as a predictor.")
    if s.time:
        tv = ds.var(s.time)
        if s.time_type in ("continuous", "quadratic") and tv.storage is not Storage.NUMERIC:
            raise MixedInputError("Continuous time needs a numeric time variable.")
    needs_time = (s.random in ("slope", "slope_uncorrelated") and s.estimator != "gee") or s.residual != "independent"
    if needs_time and not s.time:
        raise MixedInputError("This random-effects or residual structure needs a time variable.")
    if s.residual == "unstructured":
        if s.random != "none":
            raise MixedInputError("Unstructured covariance is a marginal model: set random effects to 'None'.")
        if s.time_type != "categorical":
            raise MixedInputError("Unstructured covariance needs categorical time (one parameter per occasion).")
        if s.cluster:
            raise MixedInputError("Clusters are not supported with the unstructured marginal model.")
    elif s.random == "none" and s.estimator != "gee":
        raise MixedInputError("Random-effects models need at least a random intercept "
                              "(choose unstructured covariance for a marginal model).")
    if s.method not in ("REML", "ML") or s.time_type not in TIME_TYPES or s.random not in RANDOM \
            or s.residual not in RESIDUAL or s.interactions not in INTERACTIONS:
        raise MixedInputError("Invalid option.")


def _fixed_rhs(ds: Dataset, s: MixedSpec) -> str:
    a = lambda n: alias(ds.var(n).alias)  # noqa: E731
    if s.terms:
        name = lambda n: "sx_time" if n == s.time else a(n)  # noqa: E731
        return " + ".join(":".join(name(v) for v in t) for t in s.terms)
    preds = [a(p) for p in s.predictors]
    terms: list[str] = []
    if s.time:
        t = "sx_time"
        terms.append(t if s.time_type != "quadratic" else "poly(sx_time, 2, raw = TRUE)")
    tt = terms[0] if terms else None
    if s.interactions == "all" and preds:
        body = f"({' + '.join(([tt] if tt else []) + preds)})^2"
        return body
    terms += preds
    if s.interactions == "time" and tt:
        terms += [f"{tt}:{p}" for p in preds]
    return " + ".join(terms) if terms else "1"


def build(ds: Dataset, s: MixedSpec, plot_paths: list[str]) -> str:
    ds = view(ds, s)
    validate(ds, s)
    if s.family != "gaussian" or s.estimator == "gee":
        return _build_generalized(ds, s, plot_paths)
    a = lambda n: alias(ds.var(n).alias)  # noqa: E731
    y = a(s.outcome)
    conf = r_num(s.conf)
    reml = "TRUE" if s.method == "REML" else "FALSE"
    L = _prep(ds, s) + [f"sx_fixed <- {y} ~ {_fixed_rhs(ds, s)}"]
    return _build_gaussian(ds, s, plot_paths, L, y, conf, reml)


def _prep(ds: Dataset, s: MixedSpec) -> list[str]:
    """Shared data preparation: complete cases, participant / cluster factors, time coding, ordering."""
    a = lambda n: alias(ds.var(n).alias)  # noqa: E731
    y, idv = a(s.outcome), a(s.id)
    used = [y, idv] + ([a(s.time)] if s.time else []) + ([a(s.cluster)] if s.cluster else []) + [a(p) for p in s.predictors]
    used += [a(v) for v in s.sample_vars if a(v) not in used]   # shared sample of the search this model came from
    L = [f"sx_used <- {r_vec(used)}"]
    if s.time:
        L += [
            "# outcome availability by time point (all rows; unobserved occasions stay missing, never zero-filled)",
            f"sx_av_t <- dat${a(s.time)}; sx_av_t <- if (is.factor(sx_av_t)) sx_av_t else factor(sx_av_t)",
            f"sx_av <- data.frame(time = levels(sx_av_t), records = as.vector(table(sx_av_t)),",
            f"  observed = as.vector(tapply(!is.na(dat${y}), sx_av_t, sum)))",
            "sx_av$observed[is.na(sx_av$observed)] <- 0",
            "sx_av$missing <- sx_av$records - sx_av$observed; sx_av$percent_missing <- 100 * sx_av$missing / sx_av$records",
        ]
    L += [
        "sx_d <- sx_complete(dat, sx_used)",
        f"sx_d$sx_id <- factor(sx_d${idv})",
    ]
    if s.cluster:
        L.append(f"sx_d$sx_cluster <- factor(sx_d${a(s.cluster)})")
    if s.time:
        t = a(s.time)
        tv = ds.var(s.time)
        if tv.is_categorical and tv.storage is Storage.NUMERIC:
            # labelled factor in R: recover the numeric codes (e.g. "Wave 3" -> 3) for continuous time
            levels = ds.levels(s.time)
            L.append(f"sx_codes <- stats::setNames({r_vec(levels)}, {r_vec(level_labels(ds, s.time, levels))})")
            num = f"unname(sx_codes[as.character(sx_d${t})])"
        elif tv.is_categorical:
            num = f"as.numeric(sx_d${t})  # string categories: occasion order"
        else:
            num = f"sx_d${t}"
        L.append(f"sx_d$sx_tnum <- {num}")
        if s.time_type == "categorical":
            L.append(f"sx_d$sx_time <- if (is.factor(sx_d${t})) droplevels(sx_d${t}) else factor(sx_d${t})")
        else:
            L.append("sx_d$sx_time <- sx_d$sx_tnum")
        L += [
            "sx_d$sx_tidx <- as.integer(factor(sx_d$sx_tnum))  # occasion order (AR(1) assumes equal spacing)",
            "sx_d <- sx_d[order(sx_d$sx_id, sx_d$sx_tnum), ]",
        ]
        if s.residual != "independent" or (s.estimator == "gee" and s.gee_corstr in ("ar1", "unstructured")):
            L += ["if (anyDuplicated(sx_d[, c(\"sx_id\", \"sx_tnum\")])) stop(\"Some participants have more than one row "
                  "at the same time point; residual correlation structures need one row per participant per occasion.\")"]
    else:
        L.append("sx_d <- sx_d[order(sx_d$sx_id), ]")
    L += [
        "sx_counts <- table(sx_d$sx_id)",
        "if (length(sx_counts) < 3) stop(\"Fewer than 3 participants with complete data.\")",
    ]
    return L


def _build_gaussian(ds: Dataset, s: MixedSpec, plot_paths: list[str], L: list[str], y: str, conf: str,
                    reml: str) -> str:
    tslope = "sx_time" if s.time_type == "continuous" else "sx_tnum"
    grp = "sx_cluster/sx_id" if s.cluster else "sx_id"
    if s.engine == "lme4":
        re = {"intercept": f"(1 | {grp})", "slope": f"(1 + {tslope} | {grp})",
              "slope_uncorrelated": f"(1 + {tslope} || {grp})"}[s.random]
        if s.cluster and s.random != "intercept":
            re = "(1 | sx_cluster) + " + re.replace(grp, "sx_id")
        L += [
            f"sx_fit <- lmerTest::lmer(stats::update(sx_fixed, . ~ . + {re}), data = sx_d, REML = {reml})",
            *_lme4_inference(s),
            "sx_vc <- as.data.frame(lme4::VarCorr(sx_fit))",
            "sx_vc_tab <- data.frame(group = sx_vc$grp, term = ifelse(is.na(sx_vc$var1), \"(residual)\", sx_vc$var1),",
            "  with = ifelse(is.na(sx_vc$var2), \"\", sx_vc$var2), variance_or_cov = sx_vc$vcov, sd_or_corr = sx_vc$sdcor)",
            "sx_singular <- lme4::isSingular(sx_fit)",
            "sx_conv <- unlist(sx_fit@optinfo$conv$lme4$messages)",
            "sx_conv <- sx_conv[!grepl(\"singular\", sx_conv)]  # reported by the singular-fit check",
            "sx_beta <- lme4::fixef(sx_fit); sx_V <- as.matrix(stats::vcov(sx_fit))",
            f"sx_df_note <- {r_str(INFERENCE_NOTES[s.inference])}",
            "sx_fitted <- stats::fitted(sx_fit); sx_resid <- stats::residuals(sx_fit)",
        ]
    elif s.engine == "nlme":
        rnd = {"intercept": f"~ 1 | {grp}",
               "slope": f"~ 1 + {tslope} | sx_id",
               "slope_uncorrelated": f"list(sx_id = nlme::pdDiag(~ 1 + {tslope}))"}[s.random]
        cor = {"ar1": "nlme::corAR1(form = ~ sx_tidx | sx_id)", "car1": "nlme::corCAR1(form = ~ sx_tnum | sx_id)",
               "independent": "NULL"}[s.residual]
        if s.cluster:
            cor = cor.replace("| sx_id)", "| sx_cluster/sx_id)")
        L += [
            "# nlminb (nlme default) first; optim as a fallback if it fails to converge",
            f"sx_lme <- function(opt) nlme::lme(sx_fixed, random = {rnd}, correlation = {cor}, data = sx_d,",
            f"  method = {r_str(s.method)}, control = nlme::lmeControl(opt = opt, msMaxIter = 200))",
            "sx_opt <- \"nlminb\"",
            "sx_fit <- tryCatch(sx_lme(\"nlminb\"), error = function(e) { sx_opt <<- \"optim\"; sx_lme(\"optim\") })",
            "sx_tt <- as.data.frame(summary(sx_fit)$tTable)",
            "sx_fx <- data.frame(term = rownames(sx_tt), estimate = sx_tt$Value, se = sx_tt$Std.Error, df = sx_tt$DF,",
            "  t = sx_tt[[\"t-value\"]], p = sx_tt[[\"p-value\"]])",
            "sx_at <- as.data.frame(stats::anova(sx_fit, type = \"marginal\"))",
            "sx_terms <- data.frame(term = rownames(sx_at), F = sx_at[[\"F-value\"]], df1 = sx_at$numDF, df2 = sx_at$denDF,",
            "  p = sx_at[[\"p-value\"]])",
            "sx_vcm <- nlme::VarCorr(sx_fit)",
            "sx_vc_tab <- data.frame(term = rownames(sx_vcm), variance = suppressWarnings(as.numeric(sx_vcm[, 1])),",
            "  sd = suppressWarnings(as.numeric(sx_vcm[, 2])))",
            "sx_phi <- if (is.null(sx_fit$modelStruct$corStruct)) NA_real_ else",
            "  unname(stats::coef(sx_fit$modelStruct$corStruct, unconstrained = FALSE))",
            "sx_singular <- FALSE",
            "sx_conv <- if (sx_opt != \"nlminb\") \"nlminb failed; estimated with optim (check the results)\" else character()",
            "sx_beta <- nlme::fixef(sx_fit); sx_V <- as.matrix(stats::vcov(sx_fit))",
            "sx_df_note <- \"nlme containment degrees of freedom; marginal (Type III) F-tests.\"",
            "sx_fitted <- stats::fitted(sx_fit); sx_resid <- stats::residuals(sx_fit, type = \"normalized\")",
        ]
    else:
        L += [
            f"sx_fit <- mmrm::mmrm(stats::update(sx_fixed, . ~ . + us(sx_time | sx_id)), data = sx_d, reml = {reml})",
            "sx_co <- as.data.frame(coef(summary(sx_fit)))",
            "sx_fx <- data.frame(term = rownames(sx_co), estimate = sx_co$Estimate, se = sx_co[[\"Std. Error\"]],",
            "  df = sx_co$df, t = sx_co[[\"t value\"]], p = sx_co[[\"Pr(>|t|)\"]])",
            "sx_terms <- tryCatch({ a <- as.data.frame(car::Anova(sx_fit, type = \"III\"))",
            "  data.frame(term = rownames(a), F = a[[\"F Statistic\"]], df1 = a[[\"Num.Df\"]], df2 = a[[\"Denom.Df\"]], p = a[[\"Pr(>=F)\"]]) },",
            "  error = function(e) NULL)",
            "sx_cov <- mmrm::VarCorr(sx_fit)",
            "sx_vc_tab <- data.frame(occasion = rownames(sx_cov), as.data.frame(unclass(sx_cov), check.names = FALSE), check.names = FALSE)",
            "sx_singular <- FALSE; sx_conv <- character()",
            "sx_beta <- stats::coef(sx_fit); sx_V <- as.matrix(stats::vcov(sx_fit))",
            "sx_df_note <- \"Satterthwaite degrees of freedom (mmrm); Type III F-tests.\"",
            "sx_fitted <- stats::fitted(sx_fit); sx_resid <- stats::residuals(sx_fit, type = \"normalized\")",
        ]
    L += [
        "sx_crit <- stats::qt(1 - (1 - " + conf + ") / 2, pmax(sx_fx$df, 1))",
        "sx_fx$ci_low <- sx_fx$estimate - sx_crit * sx_fx$se; sx_fx$ci_high <- sx_fx$estimate + sx_crit * sx_fx$se",
        "sx_tabs <- list(sx_table(\"Fixed effects\", sx_fx, notes = paste0(sx_df_note, \" Treatment coding; time is \",",
        f"  {r_str(TIME_TYPES[s.time_type] if s.time else 'not modelled')}, \".\")))",
        "if (!is.null(sx_terms)) sx_tabs[[length(sx_tabs) + 1]] <- sx_table(\"Fixed-effect term tests\", sx_terms)",
    ]
    if s.engine == "mmrm":
        L.append("sx_tabs[[length(sx_tabs) + 1]] <- sx_table(\"Residual covariance (unstructured)\", sx_vc_tab, "
                 "notes = \"Marginal model: no random effects. Each occasion has its own variance and every pair of "
                 "occasions its own covariance.\")")
    else:
        L.append("sx_tabs[[length(sx_tabs) + 1]] <- sx_table(\"Random effects (variance components)\", sx_vc_tab, "
                 "notes = \"Variance of participant-specific deviations; correlations between random effects where estimated.\")")
    if s.engine == "nlme" and s.residual != "independent":
        L.append("sx_tabs[[length(sx_tabs) + 1]] <- sx_table(\"Residual autocorrelation\", data.frame(structure = "
                 + r_str(RESIDUAL[s.residual]) + ", phi = sx_phi), notes = \"phi = correlation between residuals one "
                 "occasion (AR1) or one time unit (CAR1) apart, after random effects. Random effects and residual "
                 "autocorrelation are different things.\")")
    # model fit + R2
    L += [
        "sx_r2 <- if (" + ("FALSE" if s.engine == "mmrm" else "TRUE") + ") tryCatch(suppressWarnings(performance::r2_nakagawa(sx_fit)),",
        "  error = function(e) NULL) else NULL",
        "sx_fit_tab <- data.frame(observations = nrow(sx_d), participants = nlevels(sx_d$sx_id),",
        "  obs_per_participant_min = min(sx_counts[sx_counts > 0]), obs_per_participant_max = max(sx_counts),",
        f"  method = {r_str(s.method)}, logLik = as.numeric(stats::logLik(sx_fit)), AIC = stats::AIC(sx_fit), BIC = stats::BIC(sx_fit),",
        "  R2_marginal = if (is.null(sx_r2)) NA_real_ else as.numeric(sx_r2$R2_marginal),",
        "  R2_conditional = if (is.null(sx_r2)) NA_real_ else as.numeric(sx_r2$R2_conditional),",
        "  singular = sx_singular)",
        "sx_tabs[[length(sx_tabs) + 1]] <- sx_table(\"Model fit\", sx_fit_tab, notes = paste0(\"Marginal R2: variance explained \",",
        "  \"by fixed effects; conditional R2: by fixed + random effects (Nakagawa). AIC/BIC from \", "
        + r_str(s.method) + ", \" fits; compare fixed effects only with ML fits.\"))",
        "sx_checks <- character()",
        "if (isTRUE(sx_singular)) sx_checks <- c(sx_checks, paste0(\"Singular fit: a random-effect variance is (near) \",",
        "  \"zero or a correlation is +/-1. Simplify the random-effects structure.\"))",
        "if (length(sx_conv)) sx_checks <- c(sx_checks, paste(\"Convergence:\", sx_conv))",
        "# residual autocorrelation within participants (lags 1-3)",
        "sx_rd <- data.frame(id = sx_d$sx_id, r = as.numeric(sx_resid))",
        "sx_lagcor <- vapply(1:3, function(k) { x <- unlist(lapply(split(sx_rd$r, sx_rd$id), function(v) if (length(v) > k) head(v, -k) else NULL))",
        "  y <- unlist(lapply(split(sx_rd$r, sx_rd$id), function(v) if (length(v) > k) tail(v, -k) else NULL))",
        "  if (length(x) > 3) stats::cor(x, y) else NA_real_ }, 0)",
        "sx_tabs[[length(sx_tabs) + 1]] <- sx_table(\"Residual autocorrelation by lag\", data.frame(lag = 1:3, correlation = sx_lagcor),",
        "  notes = paste0(\"Lag = observed-occasion steps within a participant (k-th next observed row), not actual \",",
        "  \"time units: across a gap or an occasion absent for everyone, lag 1 spans more than one time unit. \",",
        "  \"Normalized residuals for nlme/mmrm. Large values suggest residual autocorrelation.\"))",
    ]
    if s.time:
        L.append("sx_tabs[[length(sx_tabs) + 1]] <- sx_table(\"Outcome availability by time point\", sx_av, notes = "
                 "\"Observed = non-missing outcome. Time points with 0 observed values are structurally unobserved and "
                 "are not part of the analysis range.\")")
    L += _trajectory(ds, s, plot_paths[:3] if s.time else [])
    L += _extra_plots(s, plot_paths[3:])
    L.append("sx_result <- list(tables = sx_tabs, checks = sx_checks, n_used = nrow(sx_d), n_excluded = nrow(dat) - nrow(sx_d),")
    L.append("  notes = c(\"Mixed and marginal models use every available observation: participants with missing \",")
    L += ["    \"occasions still contribute (no imputation). This is valid when missingness is ignorable given the model \",",
          "    \"(MAR given the modelled outcomes and covariates, correct model); MAR is an assumption, not a guarantee.\"))"]
    return "\n".join(L) + "\n"


# ---- generalized outcomes: GLMM (glmer / clmm / mblogit) and GEE (geepack) — v1.1 §11.4-11.7 -------

GLMM_HELPERS = r"""# Joint Wald test of each model term (all its coefficients) from a coefficient vector and covariance.
sx_wald_terms <- function(beta, V, assign, labels) {
  out <- lapply(seq_along(labels), function(j) {
    idx <- which(assign == j); idx <- idx[idx <= length(beta)]
    if (!length(idx)) return(NULL)
    b <- beta[idx]; W <- tryCatch(as.numeric(t(b) %*% solve(V[idx, idx, drop = FALSE]) %*% b), error = function(e) NA_real_)
    data.frame(term = labels[j], df = length(idx), Wald_chi2 = W, p = stats::pchisq(W, length(idx), lower.tail = FALSE))
  })
  do.call(rbind, out)
}
# Likelihood-ratio tests of droppable terms (marginality respected) by refitting without each.
sx_lr_terms <- function(fit, fixed, refit) {
  sc <- stats::drop.scope(fixed)
  do.call(rbind, lapply(sc, function(tl) {
    f0 <- stats::update(fixed, stats::as.formula(paste(". ~ . -", tl)))
    m0 <- tryCatch(refit(f0), error = function(e) NULL)
    if (is.null(m0)) return(data.frame(term = tl, df = NA_real_, LR_chi2 = NA_real_, p = NA_real_))
    lr <- 2 * (as.numeric(stats::logLik(fit)) - as.numeric(stats::logLik(m0)))
    d <- attr(stats::logLik(fit), "df") - attr(stats::logLik(m0), "df")
    data.frame(term = tl, df = d, LR_chi2 = lr, p = stats::pchisq(max(lr, 0), d, lower.tail = FALSE))
  }))
}
"""


def _outcome_lines(ds: Dataset, s: MixedSpec, y: str) -> list[str]:
    """sx_y: explicit outcome coding for the chosen family (event / ordered levels / baseline)."""
    from .eda import category_codes

    v = ds.var(s.outcome)
    if s.family == "gaussian":
        return [f"sx_d$sx_y <- sx_d${y}"]
    if s.family == "binomial":
        observed = [c for c in ds.levels(s.outcome) if c in set(ds.analysis_column(s.outcome).dropna().unique())]
        if v.is_categorical and v.reference_level in observed:
            observed = [v.reference_level] + [c for c in observed if c != v.reference_level]
        event = s.event if s.event is not None else observed[1]
        ref = next(c for c in observed if c != event)
        ev, rf = ds.level_label(s.outcome, event), ds.level_label(s.outcome, ref)
        cmp = f"as.character(sx_d${y}) == {r_str(ev)}" if v.is_categorical else f"sx_d${y} == {r_num(event)}"
        return [f"sx_d$sx_y <- as.integer({cmp})",
                f"sx_event <- {r_str(ev)}; sx_nonevent <- {r_str(rf)}  # modelled event vs reference"]
    _codes, labels = category_codes(ds, s.outcome)
    if s.family == "ordinal":
        return [f"sx_d$sx_y <- droplevels(factor(as.character(sx_d${y}), levels = {r_vec(labels)}, ordered = TRUE))",
                "if (nlevels(sx_d$sx_y) < 3) stop(\"Fewer than three outcome categories are observed.\")"]
    ref = s.reference if s.reference is not None else ds.levels(s.outcome)[0]
    return [f"sx_d$sx_y <- droplevels(factor(as.character(sx_d${y}), levels = {r_vec(labels)}))",
            f"sx_ref <- {r_str(ds.level_label(s.outcome, ref))}",
            "if (!(sx_ref %in% levels(sx_d$sx_y))) stop(\"The reference category is not observed in the analysed rows.\")",
            "sx_d$sx_y <- stats::relevel(sx_d$sx_y, ref = sx_ref)",
            "if (nlevels(sx_d$sx_y) < 3) stop(\"Fewer than three outcome categories are observed.\")"]


def _build_generalized(ds: Dataset, s: MixedSpec, plot_paths: list[str]) -> str:
    a = lambda n: alias(ds.var(n).alias)  # noqa: E731
    y = a(s.outcome)
    conf = r_num(s.conf)
    L = _prep(ds, s) + [GLMM_HELPERS] + _outcome_lines(ds, s, y) + [
        f"sx_fixed <- sx_y ~ {_fixed_rhs(ds, s)}",
        f"sx_z <- stats::qnorm(1 - (1 - {conf}) / 2)",
        "sx_checks <- character()",
    ]
    tslope = "sx_time" if s.time_type == "continuous" else "sx_tnum"
    grp = "sx_cluster/sx_id" if s.cluster else "sx_id"
    re = {"intercept": f"(1 | {grp})", "slope": f"(1 + {tslope} | {grp})",
          "slope_uncorrelated": f"(1 + {tslope} || {grp})", "none": ""}[s.random]
    if s.cluster and s.random not in ("intercept", "none"):
        re = "(1 | sx_cluster) + " + re.replace(grp, "sx_id")
    eng = s.engine
    if eng == "geepack":
        fam = "stats::binomial()" if s.family == "binomial" else "stats::gaussian()"
        waves = ", waves = sx_tidx" if s.time else ""
        L += [
            f"sx_fit <- geepack::geeglm(sx_fixed, id = sx_id, data = sx_d, family = {fam}, corstr = {r_str(s.gee_corstr)}{waves})",
            "sx_co <- as.data.frame(summary(sx_fit)$coefficients)",
            "sx_b <- sx_co$Estimate; sx_se <- sx_co[[\"Std.err\"]]",
            "sx_fx <- data.frame(term = rownames(sx_co), estimate = sx_b, robust_se = sx_se, Wald_chi2 = sx_co$Wald,",
            "  p = sx_co[[\"Pr(>|W|)\"]], ci_low = sx_b - sx_z * sx_se, ci_high = sx_b + sx_z * sx_se)",
        ]
        if s.family == "binomial":
            L += ["sx_fx$odds_ratio <- exp(sx_fx$estimate); sx_fx$or_ci_low <- exp(sx_fx$ci_low); sx_fx$or_ci_high <- exp(sx_fx$ci_high)"]
        L += [
            "sx_X <- stats::model.matrix(sx_fit)",
            "sx_terms <- sx_wald_terms(stats::coef(sx_fit), as.matrix(sx_fit$geese$vbeta), attr(sx_X, \"assign\"),",
            "  attr(stats::terms(sx_fixed), \"term.labels\"))",
            "sx_corr <- summary(sx_fit)$corr",
            "sx_qic <- tryCatch(geepack::QIC(sx_fit), error = function(e) NULL)",
            "sx_tabs <- list(",
            "  sx_table(\"Coefficients (population-average)\", sx_fx, notes = paste0(\"GEE estimates average (marginal) effects \",",
            "    \"over the population of participants, with robust (sandwich) standard errors that remain valid if the \",",
            "    \"working correlation is wrong. They answer a different question from mixed-model (subject-specific) \",",
            "    \"coefficients and are not interchangeable with them.\"" + (", \" Modelled event: \", sx_event, \".\"" if s.family == "binomial" else "") + ")),",
            "  sx_table(\"Term tests (robust Wald)\", sx_terms, notes = \"Joint Wald chi-square of each term's coefficients with the robust covariance.\"),",
            f"  sx_table(\"Working correlation\", data.frame(structure = {r_str(GEE_CORSTR[s.gee_corstr])},",
            "    parameter = if (is.null(sx_corr) || !nrow(sx_corr)) NA_character_ else rownames(sx_corr),",
            "    estimate = if (is.null(sx_corr) || !nrow(sx_corr)) NA_real_ else sx_corr$Estimate),",
            "    notes = \"Nuisance correlation used to weight observations within a participant; inference relies on the sandwich SE.\"),",
            "  sx_table(\"Model fit\", data.frame(observations = nrow(sx_d), participants = nlevels(droplevels(sx_d$sx_id)),",
            "    clusters_max_size = max(sx_counts), QIC = if (is.null(sx_qic)) NA_real_ else unname(sx_qic[[\"QIC\"]]),",
            "    scale = as.numeric(summary(sx_fit)$dispersion$Estimate[1]))))",
            "sx_beta <- stats::coef(sx_fit); sx_V <- as.matrix(sx_fit$geese$vbeta)",
            "if (nlevels(droplevels(sx_d$sx_id)) < 40) sx_checks <- c(sx_checks, paste0(\"Only \", nlevels(droplevels(sx_d$sx_id)),",
            "  \" participants: robust (sandwich) standard errors can be too small with few clusters.\"))",
        ]
        inference = "Robust (sandwich) Wald tests; population-average estimand."
    elif eng == "glmer":
        L += [
            f"sx_fit <- lme4::glmer(stats::update(sx_fixed, . ~ . + {re}), data = sx_d, family = stats::binomial(), nAGQ = {int(s.nagq)})",
            "sx_co <- as.data.frame(coef(summary(sx_fit)))",
            "sx_b <- sx_co[[1]]; sx_se <- sx_co[[2]]",
            "sx_fx <- data.frame(term = rownames(sx_co), log_odds = sx_b, se = sx_se, z = sx_co[[3]], p = sx_co[[4]],",
            "  odds_ratio = exp(sx_b), or_ci_low = exp(sx_b - sx_z * sx_se), or_ci_high = exp(sx_b + sx_z * sx_se))",
            "sx_terms <- sx_lr_terms(sx_fit, sx_fixed, function(f0) lme4::glmer(stats::update(f0, . ~ . + "
            + re + f"), data = sx_d, family = stats::binomial(), nAGQ = {int(s.nagq)}))",
            "sx_vc <- as.data.frame(lme4::VarCorr(sx_fit))",
            "sx_vc_tab <- data.frame(group = sx_vc$grp, term = sx_vc$var1, with = ifelse(is.na(sx_vc$var2), \"\", sx_vc$var2),",
            "  variance_or_cov = sx_vc$vcov, sd_or_corr = sx_vc$sdcor)",
            "sx_singular <- lme4::isSingular(sx_fit)",
            "sx_conv <- unlist(sx_fit@optinfo$conv$lme4$messages); sx_conv <- sx_conv[!grepl(\"singular\", sx_conv)]",
            "if (isTRUE(sx_singular)) sx_checks <- c(sx_checks, \"Singular fit: a random-effect variance is (near) zero or a correlation is +/-1. Simplify the random effects.\")",
            "if (length(sx_conv)) sx_checks <- c(sx_checks, paste(\"Convergence:\", sx_conv))",
            "sx_beta <- lme4::fixef(sx_fit); sx_V <- as.matrix(stats::vcov(sx_fit))",
            "sx_events <- sum(sx_d$sx_y); sx_ne <- nrow(sx_d) - sx_events",
            "sx_tabs <- list(",
            "  sx_table(\"Fixed effects (log-odds and odds ratios)\", sx_fx, notes = paste0(\"Modelled event: \", sx_event,",
            "    \" (vs \", sx_nonevent, \"). Odds ratios are conditional (subject-specific): they compare observations of the \",",
            "    \"same participant, holding the random effects fixed. They are not population-average odds ratios and not \",",
            "    \"risk ratios. Wald z tests and CIs; estimation: \", " + r_str("Laplace approximation" if s.nagq == 1 else f"adaptive Gauss-Hermite quadrature ({s.nagq} points)") + ", \".\")),",
            "  sx_table(\"Term tests (likelihood-ratio)\", sx_terms, notes = \"Each term dropped in turn (marginality respected) and the model refitted.\"),",
            "  sx_table(\"Random effects (variance components, logit scale)\", sx_vc_tab),",
            "  sx_table(\"Model fit\", data.frame(observations = nrow(sx_d), participants = nlevels(droplevels(sx_d$sx_id)),",
            "    events = sx_events, non_events = sx_ne, logLik = as.numeric(stats::logLik(sx_fit)), AIC = stats::AIC(sx_fit),",
            "    BIC = stats::BIC(sx_fit), singular = sx_singular, converged = !length(sx_conv))))",
        ]
        L += _glmm_probabilities(ds, s, plot_paths[:1])
        inference = "Wald z tests (Laplace/AGQ likelihood); likelihood-ratio term tests. No denominator-df procedure (lmerTest does not apply to GLMMs)."
    elif eng == "clmm":
        L += [
            f"sx_fit <- ordinal::clmm(stats::update(sx_fixed, . ~ . + {re}), data = sx_d, link = \"logit\", nAGQ = {int(s.nagq)})",
            "sx_co <- as.data.frame(summary(sx_fit)$coefficients)",
            "sx_isb <- rownames(sx_co) %in% names(sx_fit$beta)",
            "sx_bb <- sx_co[sx_isb, , drop = FALSE]; sx_th <- sx_co[!sx_isb, , drop = FALSE]",
            "sx_fx <- data.frame(term = rownames(sx_bb), log_odds = sx_bb[[1]], se = sx_bb[[2]], z = sx_bb[[3]], p = sx_bb[[4]],",
            "  odds_ratio = exp(sx_bb[[1]]), or_ci_low = exp(sx_bb[[1]] - sx_z * sx_bb[[2]]), or_ci_high = exp(sx_bb[[1]] + sx_z * sx_bb[[2]]))",
            "sx_Xc <- stats::model.matrix(sx_fixed, sx_d)",
            "sx_Vc <- as.matrix(stats::vcov(sx_fit))[names(sx_fit$beta), names(sx_fit$beta), drop = FALSE]",
            "sx_terms <- sx_wald_terms(sx_fit$beta, sx_Vc, attr(sx_Xc, \"assign\")[-1], attr(stats::terms(sx_fixed), \"term.labels\"))",
            "sx_vcl <- ordinal::VarCorr(sx_fit)",
            "sx_vc_tab <- do.call(rbind, lapply(names(sx_vcl), function(g) { m <- as.matrix(sx_vcl[[g]])",
            "  data.frame(group = g, term = if (is.null(rownames(m))) \"(Intercept)\" else rownames(m), variance = diag(m),",
            "    sd = sqrt(diag(m))) }))",
            "sx_conv_ok <- identical(as.integer(sx_fit$optRes$convergence), 0L)",
            "if (!sx_conv_ok) sx_checks <- c(sx_checks, \"The optimizer reported a convergence problem.\")",
            "sx_condH <- summary(sx_fit)$condHess",
            "if (isTRUE(sx_condH > 1e4)) sx_checks <- c(sx_checks, sprintf(\"Large Hessian condition number (%.3g): estimates may be unstable.\", sx_condH))",
            "if (any(sx_vc_tab$variance < 1e-6)) sx_checks <- c(sx_checks, \"A random-effect variance is (near) zero.\")",
            "sx_small <- min(table(sx_d$sx_y))",
            "if (sx_small < 10) sx_checks <- c(sx_checks, sprintf(\"The smallest outcome category has only %d rows.\", sx_small))",
            "sx_tabs <- list(",
            "  sx_table(\"Fixed effects (cumulative logit)\", sx_fx, notes = paste0(\"Category order: \", paste(levels(sx_d$sx_y), collapse = \" < \"),",
            "    \". A positive coefficient (OR > 1) shifts responses towards higher categories, for the same participant \",",
            "    \"(conditional on the random effects). Proportional-odds model: one OR for every cut-point; the assumption is \",",
            "    \"not tested inside the mixed model (check it with Ordinal logistic regression, which runs a nominal test).\""
            + (", " + r_str(" Coding direction: " + ds.var(s.outcome).coding_direction + ".") if ds.var(s.outcome).coding_direction else "") + ")),",
            "  sx_table(\"Thresholds (cut-points)\", data.frame(cut = rownames(sx_th), estimate = sx_th[[1]], se = sx_th[[2]])),",
            "  sx_table(\"Term tests (joint Wald)\", sx_terms, notes = \"Joint Wald chi-square of each term's coefficients.\"),",
            "  sx_table(\"Random effects (variance components, latent logit scale)\", sx_vc_tab),",
            "  sx_table(\"Model fit\", data.frame(observations = nrow(sx_d), participants = nlevels(droplevels(sx_d$sx_id)),",
            "    categories = nlevels(sx_d$sx_y), logLik = as.numeric(stats::logLik(sx_fit)), AIC = stats::AIC(sx_fit),",
            "    BIC = stats::BIC(sx_fit), condition_hessian = sx_condH, converged = sx_conv_ok)))",
        ]
        inference = "Wald z tests and joint Wald term tests (" + ("Laplace approximation" if s.nagq == 1 else f"AGQ, {s.nagq} points") + "). Not a Gaussian model."
    else:  # mblogit
        L += [
            "sx_fit <- mclogit::mblogit(sx_fixed, random = ~ 1 | sx_id, data = sx_d, method = \"PQL\")",
            "sx_s <- summary(sx_fit)",
            "sx_co <- as.data.frame(sx_s$coefficients)",
            "sx_parts <- do.call(rbind, strsplit(rownames(sx_co), \"~\", fixed = TRUE))",
            "sx_fx <- data.frame(category = sx_parts[, 1], term = sx_parts[, 2], B = sx_co[[1]], se = sx_co[[2]], z = sx_co[[3]],",
            "  p = sx_co[[4]], RRR = exp(sx_co[[1]]), ci_low = exp(sx_co[[1]] - sx_z * sx_co[[2]]), ci_high = exp(sx_co[[1]] + sx_z * sx_co[[2]]))",
            "sx_Vb <- as.matrix(sx_s$vcov.coef)",
            "sx_tl <- attr(stats::terms(sx_fixed), \"term.labels\")",
            "sx_Xa <- attr(stats::model.matrix(sx_fixed, sx_d), \"assign\")",
            "sx_assign <- sx_Xa[match(sx_parts[, 2], colnames(stats::model.matrix(sx_fixed, sx_d)))]",
            "sx_terms <- sx_wald_terms(sx_co[[1]], sx_Vb, sx_assign, sx_tl)",
            "sx_vcm <- as.matrix(sx_fit$VarCov[[1]])",
            "sx_vc_tab <- data.frame(category_contrast = rownames(sx_vcm), variance = diag(sx_vcm), sd = sqrt(diag(sx_vcm)))",
            "if (!isTRUE(sx_fit$converged)) sx_checks <- c(sx_checks, \"The PQL algorithm did not converge.\")",
            "sx_small <- min(table(sx_d$sx_y))",
            "if (sx_small < 10) sx_checks <- c(sx_checks, sprintf(\"The smallest outcome category has only %d rows.\", sx_small))",
            "sx_tabs <- list(",
            "  sx_table(\"Fixed effects (baseline-category logits)\", sx_fx, notes = paste0(\"Each category is compared with the \",",
            "    \"reference '\", sx_ref, \"'. RRR = conditional relative risk ratio. Wald z tests and CIs.\")),",
            "  sx_table(\"Term tests (joint Wald)\", sx_terms, notes = paste0(\"Joint Wald test of each term over all outcome \",",
            "    \"contrasts. Likelihood-ratio tests are not available: PQL maximizes a quasi-likelihood.\")),",
            "  sx_table(\"Random intercepts (variance per category contrast)\", sx_vc_tab),",
            "  sx_table(\"Model fit\", data.frame(observations = nrow(sx_d), participants = nlevels(droplevels(sx_d$sx_id)),",
            "    categories = nlevels(sx_d$sx_y), deviance = sx_fit$deviance, iterations = sx_fit$iter, converged = isTRUE(sx_fit$converged),",
            "    estimation = \"PQL (penalized quasi-likelihood)\")))",
        ]
        inference = ("Approximate: penalized quasi-likelihood (PQL, mclogit::mblogit). PQL can underestimate random-effect "
                     "variances and bias fixed effects towards zero, especially with few observations per participant or "
                     "rare categories. Wald tests only.")
    if s.time:
        L.append("sx_tabs[[length(sx_tabs) + 1]] <- sx_table(\"Outcome availability by time point\", sx_av, notes = "
                 "\"Observed = non-missing outcome. Time points with 0 observed values are structurally unobserved and "
                 "are not part of the analysis range.\")")
    L += [
        f"sx_inference <- {r_str(inference)}",
        "sx_result <- list(tables = sx_tabs, checks = sx_checks, n_used = nrow(sx_d), n_excluded = nrow(dat) - nrow(sx_d),",
        "  notes = c(sx_inference, \" Every available observation is used (participants with missing occasions still \",",
        "    \"contribute); estimates assume data are missing at random given the model.\"))",
    ]
    if eng == "glmer":
        L += _extra_plots(s, plot_paths[1:2])
    return "\n".join(L) + "\n"


def _glmm_probabilities(ds: Dataset, s: MixedSpec, paths: list[str]) -> list[str]:
    """Predicted probabilities across time (or the first predictor): conditional on a typical participant
    (random effects = 0) and population-average (averaged over simulated random effects)."""
    a = lambda n: alias(ds.var(n).alias)  # noqa: E731
    xvar = "sx_time" if s.time else (a(s.predictors[0]) if s.predictors else None)
    if xvar is None:
        return []
    others = [a(p) for p in s.predictors if a(p) != xvar]
    xlab = r_str(ds.var(s.time).display_name) if s.time else f"sx_names[[{r_str(xvar)}]]"
    L = [
        "# ---- predicted probabilities (conditional vs population-average) ----",
        f"sx_xv <- sx_d${xvar}",
        "sx_grid <- if (is.factor(sx_xv)) factor(levels(sx_xv), levels = levels(sx_xv)) else seq(min(sx_xv), max(sx_xv), length.out = 40)",
        f"sx_held <- lapply(sx_d[, {r_vec(others) if others else 'character(0)'}, drop = FALSE],",
        "  function(v) if (is.factor(v)) factor(levels(v)[1], levels = levels(v)) else mean(v))",
        f"sx_nd <- do.call(data.frame, c(list({xvar} = sx_grid), sx_held))",
    ]
    if s.time:
        L.append("sx_nd$sx_tnum <- if (is.factor(sx_nd$sx_time)) as.numeric(sx_nd$sx_time) else sx_nd$sx_time")
    L += [
        "sx_tt <- stats::delete.response(stats::terms(sx_fixed))",
        "sx_xlev <- lapply(Filter(is.factor, sx_d[, intersect(names(sx_d), all.vars(sx_tt)), drop = FALSE]), levels)",
        "sx_X <- stats::model.matrix(sx_tt, stats::model.frame(sx_tt, sx_nd, xlev = sx_xlev))[, names(sx_beta), drop = FALSE]",
        "sx_eta <- as.vector(sx_X %*% sx_beta); sx_eta_se <- sqrt(rowSums((sx_X %*% sx_V) * sx_X))",
        f"set.seed({int(s.seed)})",
        "sx_tot <- matrix(0, 4000, nrow(sx_nd))",
        "for (sx_g in names(lme4::VarCorr(sx_fit))) {",
        "  sx_S <- as.matrix(lme4::VarCorr(sx_fit)[[sx_g]])",
        "  sx_B <- matrix(MASS::mvrnorm(4000, rep(0, ncol(sx_S)), sx_S), nrow = 4000)",
        "  sx_Z <- sapply(colnames(sx_S), function(cn) if (cn == \"(Intercept)\") rep(1, nrow(sx_nd)) else as.numeric(sx_nd[[cn]]))",
        "  sx_tot <- sx_tot + sx_B %*% t(matrix(sx_Z, nrow = nrow(sx_nd)))",
        "}",
        "sx_pa <- colMeans(stats::plogis(sweep(sx_tot, 2, sx_eta, \"+\")))",
        "sx_pred <- data.frame(x = sx_nd[[1]], conditional = stats::plogis(sx_eta),",
        "  ci_low = stats::plogis(sx_eta - sx_z * sx_eta_se), ci_high = stats::plogis(sx_eta + sx_z * sx_eta_se),",
        "  population_average = sx_pa)",
        f"names(sx_pred)[1] <- {r_str(xvar)}",
        "sx_tabs[[length(sx_tabs) + 1]] <- sx_table(\"Predicted probabilities\", sx_pred[unique(round(seq(1, nrow(sx_pred), length.out = min(nrow(sx_pred), 8)))), ],",
        f"  notes = paste0(\"P(\", sx_event, \"). conditional = for a participant with random effects at 0 (CI from the fixed effects); \",",
        f"    \"population_average = averaged over 4,000 simulated participants (seed {int(s.seed)}). Other predictors held at their mean or first category.\"))",
    ]
    if paths:
        L += [
            "sx_pl <- rbind(data.frame(x = sx_pred[[1]], p = sx_pred$conditional, type = \"Conditional (random effects = 0)\"),",
            "  data.frame(x = sx_pred[[1]], p = sx_pred$population_average, type = \"Population-average\"))",
            "sx_p <- ggplot2::ggplot(sx_pl, ggplot2::aes(x = x, y = p, colour = type, group = type)) +",
            "  (if (is.factor(sx_pl$x)) ggplot2::geom_point(size = 2.5) else NULL) + ggplot2::geom_line(linewidth = 1.1) +",
            "  (if (is.factor(sx_pred[[1]])) NULL else ggplot2::geom_ribbon(data = cbind(sx_pred, sx_xx = sx_pred[[1]]),",
            "    inherit.aes = FALSE, ggplot2::aes(x = sx_xx, ymin = ci_low, ymax = ci_high), alpha = 0.15)) +",
            "  ggplot2::coord_cartesian(ylim = c(0, 1)) +",
            f"  ggplot2::labs(title = paste0(\"Predicted probability of \", sx_event), x = {xlab},",
            "    y = \"Probability\", colour = NULL, subtitle = \"Conditional and population-average predictions differ in a logistic mixed model\") +",
            "  ggplot2::theme_minimal() + ggplot2::theme(legend.position = \"bottom\")",
            f"ggplot2::ggsave({r_str(paths[0])}, sx_p, width = 7, height = 4.8, dpi = 120)",
        ]
    return L


INFERENCE_NOTES = {
    "satterthwaite": "Inference: Satterthwaite degrees of freedom (lmerTest); Type III F-tests of fixed effects.",
    "kenward_roger": "Inference: Kenward-Roger degrees of freedom and small-sample adjusted covariance (lmerTest via "
                     "pbkrtest; REML); Type III F-tests of fixed effects.",
    "lrt": "Inference: coefficient t tests use Satterthwaite df; term tests are likelihood-ratio tests of ML refits "
           "(each term dropped in turn, marginality respected). These are model comparisons, not Type III F-tests.",
}


def _lme4_inference(s: MixedSpec) -> list[str]:
    """Coefficient table and term tests for lmerTest fits under the chosen inference method (§11.3)."""
    ddf = "\"Kenward-Roger\"" if s.inference == "kenward_roger" else "\"Satterthwaite\""
    L = [
        f"sx_co <- as.data.frame(coef(summary(sx_fit, ddf = {ddf})))",
        "sx_fx <- data.frame(term = rownames(sx_co), estimate = sx_co$Estimate, se = sx_co[[\"Std. Error\"]],",
        "  df = sx_co$df, t = sx_co[[\"t value\"]], p = sx_co[[\"Pr(>|t|)\"]])",
    ]
    if s.inference == "lrt":
        L += [
            "sx_ml <- methods::as(if (lme4::isREML(sx_fit)) stats::update(sx_fit, REML = FALSE) else sx_fit, \"lmerMod\")",
            "sx_at <- as.data.frame(stats::drop1(sx_ml, test = \"Chisq\"))[-1, , drop = FALSE]",
            "sx_terms <- data.frame(term = rownames(sx_at), df = sx_at$npar, LR_chi2 = sx_at$LRT, p = sx_at[[\"Pr(Chi)\"]])",
        ]
    else:
        L += [
            f"sx_at <- as.data.frame(stats::anova(sx_fit, type = 3, ddf = {ddf}))",
            "sx_terms <- data.frame(term = rownames(sx_at), F = sx_at[[\"F value\"]], df1 = sx_at$NumDF, df2 = sx_at$DenDF,",
            "  p = sx_at[[\"Pr(>F)\"]])",
        ]
    return L


def _extra_plots(s: MixedSpec, paths: list[str]) -> list[str]:
    """Random-effect (caterpillar) plot and residual autocorrelation plot (§15.5, §16.1)."""
    L: list[str] = []
    if len(paths) >= 1 and s.engine != "mmrm":
        L += [
            "# random intercepts with approximate 95% intervals (lme4: conditional SDs)",
            "sx_re <- if (inherits(sx_fit, \"merMod\")) {",
            "  r <- lme4::ranef(sx_fit, condVar = TRUE)$sx_id; sdv <- sqrt(attr(r, \"postVar\")[1, 1, ])",
            "  data.frame(id = rownames(r), value = r[[1]], lo = r[[1]] - 1.96 * sdv, hi = r[[1]] + 1.96 * sdv)",
            "} else { r <- nlme::ranef(sx_fit); r <- if (is.list(r) && !is.data.frame(r)) r[[length(r)]] else r",
            "  data.frame(id = rownames(r), value = r[[1]], lo = NA_real_, hi = NA_real_) }",
            "sx_re <- sx_re[order(sx_re$value), ]; sx_re$rank <- seq_len(nrow(sx_re))",
            "sx_p4 <- ggplot2::ggplot(sx_re, ggplot2::aes(x = rank, y = value)) +",
            "  ggplot2::geom_hline(yintercept = 0, linetype = 2) +",
            "  (if (all(is.na(sx_re$lo))) NULL else ggplot2::geom_errorbar(ggplot2::aes(ymin = lo, ymax = hi), colour = \"grey70\", width = 0)) +",
            "  ggplot2::geom_point(size = 0.8, colour = \"#2166ac\") +",
            "  ggplot2::labs(title = \"Random intercepts (participant deviations from the average)\",",
            "    subtitle = \"Sorted; bars = approximate 95% intervals\", x = \"Participants (sorted)\", y = \"Deviation\") +",
            "  ggplot2::theme_minimal()",
            f"ggplot2::ggsave({r_str(paths[0])}, sx_p4, width = 7, height = 4.5, dpi = 120)",
        ]
    if len(paths) >= 2:
        L += [
            "sx_acf <- data.frame(lag = 1:3, correlation = sx_lagcor); sx_acf <- sx_acf[!is.na(sx_acf$correlation), ]",
            "sx_p5 <- ggplot2::ggplot(sx_acf, ggplot2::aes(x = factor(lag), y = correlation)) +",
            "  ggplot2::geom_col(fill = \"#b2182b\", width = 0.5) + ggplot2::geom_hline(yintercept = 0) +",
            "  ggplot2::coord_cartesian(ylim = c(-1, 1)) +",
            "  ggplot2::labs(title = \"Residual autocorrelation within participants\", x = \"Lag (occasions apart)\",",
            "    y = \"Correlation\") + ggplot2::theme_minimal()",
            f"ggplot2::ggsave({r_str(paths[1])}, sx_p5, width = 6, height = 4, dpi = 120)",
        ]
    return L


def _trajectory(ds: Dataset, s: MixedSpec, paths: list[str]) -> list[str]:
    """Population-level trajectory (fixed effects) with CI, observed individual trajectories,
    and subject-specific (conditional) predictions for a few participants."""
    if not s.time or not paths:
        return []
    a = lambda n: alias(ds.var(n).alias)  # noqa: E731
    y = a(s.outcome)
    group = next((p for p in s.predictors if ds.var(p).is_categorical), None)
    g = a(group) if group else None
    others = [a(p) for p in s.predictors if p != group]
    cont = s.time_type != "categorical"
    L = [
        "# ---- trajectories ----",
        "sx_tv <- if (is.factor(sx_d$sx_time)) factor(levels(sx_d$sx_time), levels = levels(sx_d$sx_time)) else",
        "  seq(min(sx_d$sx_time), max(sx_d$sx_time), length.out = 50)",
        "sx_grid <- list(sx_time = sx_tv)",
    ]
    if g:
        L.append(f"sx_grid${g} <- factor(levels(sx_d${g}), levels = levels(sx_d${g}))")
    L += [
        f"sx_held <- lapply(sx_d[, {r_vec(others) if others else 'character(0)'}, drop = FALSE],",
        "  function(v) if (is.factor(v)) factor(levels(v)[1], levels = levels(v)) else mean(v))",
        "sx_nd <- do.call(expand.grid, c(sx_grid, sx_held, list(KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)))",
        "sx_nd$sx_tnum <- as.numeric(sx_nd$sx_time)",
        "sx_tt <- stats::delete.response(stats::terms(sx_fixed))",
        "sx_xlev <- lapply(Filter(is.factor, sx_d[, intersect(names(sx_d), all.vars(sx_tt)), drop = FALSE]), levels)",
        "sx_X <- stats::model.matrix(sx_tt, stats::model.frame(sx_tt, sx_nd, xlev = sx_xlev))",
        "sx_X <- sx_X[, names(sx_beta), drop = FALSE]",
        "sx_nd$predicted <- as.vector(sx_X %*% sx_beta)",
        "sx_se <- sqrt(rowSums((sx_X %*% sx_V) * sx_X))",
        f"sx_z <- stats::qnorm(1 - (1 - {r_num(s.conf)}) / 2)",
        "sx_nd$ci_low <- sx_nd$predicted - sx_z * sx_se; sx_nd$ci_high <- sx_nd$predicted + sx_z * sx_se",
        "# P1-03: predictions between or at time points without observed outcomes are model-based, not data",
        "sx_obs_t <- unique(as.numeric(sx_d$sx_tnum))",
        "sx_nd$basis <- if (is.factor(sx_d$sx_time)) \"at an observed time point\" else",
        "  ifelse(vapply(sx_nd$sx_tnum, function(x) any(abs(x - sx_obs_t) < 1e-8), TRUE),",
        "    \"at an observed time point\", \"model-based (no observed outcomes at this time)\")",
        "sx_held_txt <- if (length(sx_held)) paste(sprintf(\"%s = %s\", sx_names[names(sx_held)],",
        "  vapply(sx_held, function(v) if (is.factor(v)) as.character(v) else format(signif(v, 4)), \"\")), collapse = \"; \") else \"\"",
        "sx_tabs[[length(sx_tabs) + 1]] <- sx_table(\"Population-level trajectory\", sx_nd[, setdiff(names(sx_nd), \"sx_tnum\"), drop = FALSE],",
        "  notes = paste0(\"Fixed-effects (marginal, population-average) predictions with normal-approximation CI. \",",
        "    \"Rows marked model-based lie between observed time points or at a time with no observed outcome \",",
        "    \"(e.g. a wave missing for everyone): they are model predictions, not observed data\",",
        "    if (nzchar(sx_held_txt)) paste0(\"; other predictors held at: \", sx_held_txt) else \"\", \".\"))",
        f"set.seed({int(s.seed)})",
        f"sx_ids <- sample(levels(droplevels(sx_d$sx_id)), min({int(s.n_trajectories)}, nlevels(droplevels(sx_d$sx_id))))",
        "sx_obs <- sx_d[sx_d$sx_id %in% sx_ids, ]",
        f"sx_p <- ggplot2::ggplot() + ggplot2::geom_line(data = sx_obs, ggplot2::aes(x = sx_time, y = {y}, group = sx_id),",
        "  colour = \"grey60\", alpha = 0.45)",
    ]
    fill_g = f", fill = {g}, group = {g}" if g else ", group = 1"
    col_g = f", colour = {g}, group = {g}" if g else ", group = 1"
    if cont:
        L += [f"sx_p <- sx_p + ggplot2::geom_ribbon(data = sx_nd, ggplot2::aes(x = sx_time, ymin = ci_low, ymax = ci_high{fill_g}), alpha = 0.2, colour = NA) +",
              f"  ggplot2::geom_line(data = sx_nd, ggplot2::aes(x = sx_time, y = predicted{col_g}), linewidth = 1.3)"]
    else:
        L += [f"sx_p <- sx_p + ggplot2::geom_errorbar(data = sx_nd, ggplot2::aes(x = sx_time, ymin = ci_low, ymax = ci_high{col_g}), width = 0.15) +",
              f"  ggplot2::geom_line(data = sx_nd, ggplot2::aes(x = sx_time, y = predicted{col_g}), linewidth = 1.2) +",
              f"  ggplot2::geom_point(data = sx_nd, ggplot2::aes(x = sx_time, y = predicted{col_g}), size = 2.5)"]
    L += [
        "sx_p <- sx_p + ggplot2::labs(title = \"Population-level trajectory (fixed effects)\",",
        f"  subtitle = paste0(\"Grey: observed trajectories of \", length(sx_ids), \" randomly sampled participants (seed {int(s.seed)})\"),",
        f"  x = sx_names[[\"{a(s.time)}\"]], y = sx_names[[\"{y}\"]]"
        + (f", colour = sx_names[[\"{g}\"]], fill = sx_names[[\"{g}\"]]" if g else "") + ") +",
        "  ggplot2::theme_minimal(base_size = 12) + ggplot2::theme(legend.position = \"bottom\")",
        f"ggplot2::ggsave({r_str(paths[0])}, sx_p, width = 7.5, height = 5.2, dpi = 120)",
    ]
    if len(paths) > 1:
        L += [
            "# conditional (participant-specific) predictions for 6 participants",
            "sx_six <- head(sx_ids, 6)",
            "sx_cd <- data.frame(sx_id = sx_d$sx_id, sx_time = sx_d$sx_time, observed = sx_d[[sx_used[1]]], conditional = as.numeric(sx_fitted))",
            "sx_cd <- sx_cd[sx_cd$sx_id %in% sx_six, ]",
            "sx_p2 <- ggplot2::ggplot(sx_cd, ggplot2::aes(x = sx_time, group = sx_id)) +",
            "  ggplot2::geom_point(ggplot2::aes(y = observed), colour = \"grey30\") +",
            "  ggplot2::geom_line(ggplot2::aes(y = conditional), colour = \"#b2182b\", linewidth = 1) +",
            "  ggplot2::facet_wrap(~ sx_id) +",
            "  ggplot2::labs(title = \"Conditional (participant-specific) predictions\",",
            "    subtitle = " + r_str("Red: fitted values including each participant's random effects" if s.engine != "mmrm"
                                     else "Red: marginal fitted values (mmrm has no random effects)") + ",",
            f"    x = sx_names[[\"{a(s.time)}\"]], y = sx_names[[\"{y}\"]]) + ggplot2::theme_minimal(base_size = 11)",
            f"ggplot2::ggsave({r_str(paths[1])}, sx_p2, width = 7.5, height = 5.2, dpi = 120)",
        ]
    if len(paths) > 2:
        L += [
            "sx_p3 <- ggplot2::ggplot(data.frame(fitted = as.numeric(sx_fitted), residual = as.numeric(sx_resid)),",
            "  ggplot2::aes(fitted, residual)) + ggplot2::geom_point(alpha = 0.3) + ggplot2::geom_hline(yintercept = 0, linetype = 2) +",
            "  (if (length(unique(round(as.numeric(sx_fitted), 8))) > 20) ggplot2::geom_smooth(method = \"loess\", formula = y ~ x,",
            "    se = FALSE, colour = \"#b2182b\") else NULL) +",
            "  ggplot2::labs(title = \"Residuals vs fitted\") + ggplot2::theme_minimal()",
            f"ggplot2::ggsave({r_str(paths[2])}, sx_p3, width = 6.5, height = 4.5, dpi = 120)",
        ]
    return L


def _internal_names(ds: Dataset, s: MixedSpec) -> dict[str, str]:
    out = {"sx_id": ds.var(s.id).display_name}
    if s.time:
        out["sx_time"] = ds.var(s.time).display_name
        out["sx_tnum"] = ds.var(s.time).display_name
    if s.cluster:
        out["sx_cluster"] = ds.var(s.cluster).display_name
    return out


def run(session: RSession, ds: Dataset, s: MixedSpec) -> tuple[AnalysisResult, dict[str, bytes]]:
    from .framework import _alias_map, _as_list, _parse_table
    from .measurement import record

    from .longitudinal_preflight import preflight

    ds = view(ds, s)
    pre = preflight(ds, s.outcome, s.id, s.time, s.predictors, s)
    paths = [session.new_plot_path() for _ in range(5)]  # 3 trajectory/diagnostic + random effects + ACF
    body = build(ds, s, [p.as_posix() for p in paths])
    pk = ENGINE_PACKAGES[s.engine] + (["pbkrtest"] if s.inference == "kenward_roger" else [])
    title = ENGINE_TITLES[s.engine]
    result, raw = session.run(ds, kind="mixed_model", title=title, body=body, packages=pk + ["ggplot2"],
                              inputs={"spec": s.to_dict(), "filters": [f.describe() for f in ds.filters]})
    result.missing_data = "All available observations (complete on the model variables within each row); MAR assumed."
    result.seed = s.seed
    if s.source:
        result.messages.append(f"Refit: {s.source}.")
    record(result, ds)
    artifacts: dict[str, bytes] = {}
    if raw is None:
        return result, artifacts
    names = _alias_map(ds)
    extra = _internal_names(ds, s)
    for t in _as_list(raw.get("tables")):
        result.tables.append(_parse_table(t, names, extra))
    result.n_used = raw.get("n_used")
    result.values["n_excluded"] = raw.get("n_excluded")
    result.values["notes"] = ["".join(str(x) for x in _as_list(raw.get("notes")))]
    if pre.analysis_range is not None and s.time:
        result.values["analysis_range"] = pre._range_text(ds)
        # P1-03: the min-max range alone can hide a wave absent for everyone
        result.values["observed_time_points"] = (pre.observed_text(ds) + (
            f" (absent for everyone inside the range: "
            f"{', '.join(ds.level_label(s.time, c) for c in pre.interior_absent)})" if pre.interior_absent else ""))
    for w in pre.warnings:
        result.messages.append(f"Check: {w}")
        if result.status is Status.OK:
            result.status = Status.WARNING
    result.values.setdefault("notes", [])
    result.values["notes"] = list(result.values["notes"]) + pre.notes
    from .interpret import brief

    result.values["interpretation"] = brief(result)
    result.values["engine_choice"] = s.engine
    result.values["family"] = FAMILIES[s.family]
    result.values["estimator"] = ESTIMATORS[s.estimator]
    result.values["inference"] = (INFERENCE_NOTES[s.inference] if s.engine == "lme4" else
                                  result.values["notes"][0] if s.family != "gaussian" or s.estimator == "gee" else
                                  "nlme containment df" if s.engine == "nlme" else "mmrm Satterthwaite df")
    checks = [str(c) for c in _as_list(raw.get("checks"))]
    # an R warning that a check already reports (e.g. lme4 convergence) is shown once
    result.messages = [m for m in result.messages if not any(m.removeprefix("R warning: ") in c for c in checks)]
    for c in checks:
        result.messages.append(f"Check: {c}")
        if result.status is Status.OK:
            result.status = Status.WARNING
    for p in paths:
        if p.is_file():
            key = f"plots/{p.name}"
            artifacts[key] = p.read_bytes()
            result.plots.append(key)
    if s.descriptor:
        from .descriptor import attach_check

        attach_check(result, s.descriptor)
    return result, artifacts


# ---- specification comparisons (§15.3, §14) -------------------------------------------------

COMPARISONS = {
    "time": "Time specifications (linear / quadratic / categorical) — ML fits",
    "random": "Random effects (intercept vs intercept + slope) — REML fits",
    "residual": "Residual structure (independent / AR(1) / CAR(1)) — REML fits in nlme",
}


def comparison_variants(base: MixedSpec, which: str) -> list[tuple[str, MixedSpec]]:
    from dataclasses import replace

    if which == "time":
        if base.residual == "unstructured":
            raise MixedInputError("Unstructured models need categorical time; time specifications cannot be compared.")
        return [(TIME_TYPES[t], replace(base, time_type=t, method="ML"))
                for t in ("continuous", "quadratic", "categorical")]
    if which == "random":
        if base.residual == "unstructured":
            raise MixedInputError("The unstructured marginal model has no random effects.")
        return [(RANDOM[r], replace(base, random=r, method="REML")) for r in ("intercept", "slope")]
    if which == "residual":
        if base.random == "none":
            raise MixedInputError("Choose a random-effects model to compare residual structures.")
        return [(RESIDUAL[r], replace(base, residual=r, method="REML", force_nlme=True))
                for r in ("independent", "ar1", "car1")]
    raise MixedInputError(which)


# (smaller, larger, boundary) nested pairs by position in comparison_variants
NESTED = {"time": [(0, 1, False), (0, 2, False), (1, 2, False)], "random": [(0, 1, True)],
          "residual": [(0, 1, False), (0, 2, False)]}


def compare_specs(session: RSession, ds: Dataset, base: MixedSpec, which: str) -> tuple[AnalysisResult, dict]:
    from .framework import _alias_map, _as_list, _parse_table
    from .measurement import record

    ds = view(ds, base)
    variants = comparison_variants(base, which)
    for _, v in variants:
        validate(ds, v)
    L = ["sx_cmp <- list()"]
    for k, (label, v) in enumerate(variants, start=1):
        L.append(f"# ---- variant {k}: {label} ----")
        L.append("sx_cmp[[" + str(k) + "]] <- tryCatch({")
        L.append(build(ds, v, []).replace("sx_result <- list(", "sx_unused <- list("))
        L.append(f"  list(label = {r_str(label)}, engine = {r_str(v.engine)}, method = {r_str(v.method)},")
        L.append("    logLik = as.numeric(stats::logLik(sx_fit)), df = attr(stats::logLik(sx_fit), \"df\"),")
        L.append("    AIC = stats::AIC(sx_fit), BIC = stats::BIC(sx_fit), n = nrow(sx_d), singular = isTRUE(sx_singular))")
        L.append(f"}}, error = function(e) list(label = {r_str(label)}, error = conditionMessage(e)))")
    pairs = [(a, b, bd) for a, b, bd in NESTED[which] if a < len(variants) and b < len(variants)]
    L += [
        "sx_ok <- vapply(sx_cmp, function(x) is.null(x$error), TRUE)",
        "sx_tab <- do.call(rbind, lapply(sx_cmp[sx_ok], function(x) data.frame(model = x$label, engine = x$engine,",
        "  estimation = x$method, parameters = x$df, logLik = x$logLik, AIC = x$AIC, BIC = x$BIC, n = x$n, singular = x$singular)))",
        "sx_tab$delta_AIC <- sx_tab$AIC - min(sx_tab$AIC)",
        "sx_tabs <- list(sx_table(\"Specification comparison\", sx_tab, notes = " + r_str(COMPARISONS[which]) + "))",
        "sx_lrt <- list()",
    ]
    for a, b, boundary in pairs:
        L += [
            f"if (sx_ok[{a + 1}] && sx_ok[{b + 1}] && sx_cmp[[{a + 1}]]$n == sx_cmp[[{b + 1}]]$n && sx_cmp[[{b + 1}]]$df > sx_cmp[[{a + 1}]]$df) {{",
            f"  sx_chi <- 2 * (sx_cmp[[{b + 1}]]$logLik - sx_cmp[[{a + 1}]]$logLik); sx_df <- sx_cmp[[{b + 1}]]$df - sx_cmp[[{a + 1}]]$df",
            "  sx_p <- stats::pchisq(sx_chi, sx_df, lower.tail = FALSE)",
            "  sx_lrt[[length(sx_lrt) + 1]] <- data.frame(smaller = sx_cmp[[" + str(a + 1) + "]]$label, larger = sx_cmp[[" + str(b + 1) + "]]$label,",
            "    chi2 = sx_chi, df = sx_df, p = sx_p, p_boundary_corrected = "
            + ("0.5 * sx_p + 0.5 * stats::pchisq(sx_chi, sx_df - 1, lower.tail = FALSE)" if boundary else "NA_real_") + ")",
            "}",
        ]
    note = ("Likelihood-ratio tests for nested pairs on the same rows. " +
            ("Variance tests sit on the boundary of the parameter space: the naive p is conservative; "
             "p_boundary_corrected uses the 50:50 chi-square mixture." if which == "random" else
             "Fixed-effect comparisons use ML fits (REML likelihoods are not comparable across fixed effects)."
             if which == "time" else "AR(1) and CAR(1) are not nested in each other: compare them by AIC."))
    L += [
        "if (length(sx_lrt)) sx_tabs[[2]] <- sx_table(\"Likelihood-ratio tests\", do.call(rbind, sx_lrt), notes = " + r_str(note) + ")",
        "sx_errors <- vapply(sx_cmp[!sx_ok], function(x) paste0(x$label, \": \", x$error), \"\")",
        "sx_result <- list(tables = sx_tabs, checks = sx_errors, n_used = max(sx_tab$n), n_excluded = nrow(dat) - max(sx_tab$n))",
    ]
    body = "\n".join(L) + "\n"
    result, raw = session.run(ds, kind="mixed_comparison", title=f"Compare {which} specifications", body=body,
                              packages=["lme4", "lmerTest", "nlme", "mmrm"], inputs={"base": base.to_dict(), "which": which})
    record(result, ds)
    if raw is None:
        return result, {}
    names = _alias_map(ds)
    for t in _as_list(raw.get("tables")):
        result.tables.append(_parse_table(t, names, _internal_names(ds, base)))
    result.n_used = raw.get("n_used")
    for c in _as_list(raw.get("checks")):
        result.messages.append(f"Check: could not fit {c}")
        result.status = Status.WARNING
    return result, {}
