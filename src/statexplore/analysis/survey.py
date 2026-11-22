"""Survey-weighted analysis (v1.1 §13) with the `survey` package.

The design is always declared by the user (weights, PSUs, strata, FPC, or replicate weights);
StatExplore only *suggests* candidate columns from their labels and never invents design
variables. The design is built on every row with valid design information and analyses use
`subset()` of the design, so domain (subpopulation) standard errors are correct.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from ..core.dataset import Dataset
from ..core.schema import AnalysisResult, Status
from ..core.variables import Measure, Role, Storage
from ..rbridge.codegen import alias, r_str, r_vec
from .session import RSession

LONELY_PSU = {
    "fail": "Stop with an error (survey default) — no hidden adjustment",
    "adjust": "Centre the single PSU at the grand mean (conservative)",
    "average": "Use the average variance of the other strata",
    "certainty": "Treat single-PSU strata as certainty units (no variance contribution)",
    "remove": "Ignore single-PSU strata in the variance",
}
REP_TYPES = {"JK1": "Jackknife (JK1)", "JKn": "Stratified jackknife (JKn)", "BRR": "Balanced repeated replication",
             "Fay": "Fay's BRR", "bootstrap": "Bootstrap"}
KINDS = {
    "mean": "Weighted means (continuous variables)",
    "proportion": "Weighted proportions / frequency table (categorical)",
    "crosstab": "Survey-adjusted cross-tabulation + Rao-Scott chi-square",
    "linear": "Survey-weighted linear regression (svyglm)",
    "logistic": "Survey-weighted logistic regression (svyglm, quasibinomial)",
}


class SurveyInputError(ValueError):
    pass


@dataclass
class SurveyDesign:
    weight: str | None = None
    psu: str | None = None
    strata: str | None = None
    fpc: str | None = None
    nest: bool = True
    lonely_psu: str = "fail"
    replicate_weights: list[str] = field(default_factory=list)   # advanced: replicate-weight columns
    replicate_type: str = "JK1"
    pooled_waves_ack: bool = False    # user confirms the weight is valid for rows pooled across waves

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SurveyDesign":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def variables(self) -> list[str]:
        return [v for v in (self.weight, self.psu, self.strata, self.fpc) if v] + list(self.replicate_weights)


# ---- design suggestions (labels only; never assumed) ---------------------------------------------

_W = re.compile(r"(sampl\w*|survey|analysis|design|person[- ]level|household|respondent|final|cross-?sectional|"
                r"longitudinal|expansion|base)\W+(\w+\W+)?weight|\bweight\b\W*\((sampl|survey)|\bwgt\b|\bwtresp\b", re.I)
_PSU = re.compile(r"\b(psu|secu|primary sampling unit|sampling error comput\w* unit|cluster)\b", re.I)
_STRATA = re.compile(r"\b(strat(a|um|ification)|sampling error stratum|vstrat)\b", re.I)


def suggest_design(ds: Dataset) -> dict[str, list[str]]:
    """Candidate design columns found from names and labels. A 'Weight' column holding body
    weight is not a sampling weight: labels must say so."""
    out: dict[str, list[str]] = {"weight": [], "psu": [], "strata": []}
    for v in ds.variables.values():
        text = f"{v.name} {v.label}"
        if v.storage is Storage.NUMERIC and _W.search(text):
            out["weight"].append(v.name)
        if _PSU.search(text):
            out["psu"].append(v.name)
        if _STRATA.search(text):
            out["strata"].append(v.name)
    return out


def explain_design(ds: Dataset, d: SurveyDesign) -> list[str]:
    """Which design features can and cannot be modelled with the declared variables (§13.1)."""
    lines = []
    if d.replicate_weights:
        lines.append(f"Replicate-weight design ({REP_TYPES.get(d.replicate_type, d.replicate_type)}, "
                     f"{len(d.replicate_weights)} replicates): variances from the replicates.")
    if d.weight:
        lines.append(f"Sampling weight: {ds.var(d.weight).display_name} — estimates are weighted.")
    else:
        lines.append("No sampling weight: every row counts equally (an unweighted design-based analysis).")
    if d.psu:
        lines.append(f"Clustering: PSUs = {ds.var(d.psu).display_name} — standard errors account for clustering.")
    else:
        lines.append("No PSU/cluster variable: rows are treated as independently sampled; if the survey used "
                     "cluster sampling, standard errors will be too small.")
    if d.strata:
        lines.append(f"Stratification: {ds.var(d.strata).display_name}. Single-PSU strata policy: "
                     f"{LONELY_PSU[d.lonely_psu]}.")
    else:
        lines.append("No strata variable: stratification (which usually reduces variance) cannot be modelled.")
    if d.fpc:
        lines.append(f"Finite population correction from {ds.var(d.fpc).display_name}.")
    sugg = suggest_design(ds)
    if not any(sugg.values()) and not d.weight:
        lines.append("No variable in this dataset is labelled as a sampling weight, PSU or stratum. A column named "
                     "'Weight' that measures body weight is not a survey weight. Design-based inference is only "
                     "possible if the data provider supplies the design variables.")
    return lines


def check_design(ds: Dataset, d: SurveyDesign) -> list[str]:
    """Raise SurveyInputError for an unusable design; return warnings."""
    warnings: list[str] = []
    for role, name in (("weight", d.weight), ("PSU", d.psu), ("strata", d.strata), ("FPC", d.fpc)):
        if name and name not in ds.variables:
            raise SurveyInputError(f"Unknown {role} variable {name!r}.")
    if d.lonely_psu not in LONELY_PSU:
        raise SurveyInputError("Choose a policy for strata with a single PSU.")
    if not d.weight and not d.psu and not d.strata and not d.replicate_weights:
        raise SurveyInputError("Declare at least a sampling weight, PSU or strata (or replicate weights).")
    if d.weight:
        v = ds.var(d.weight)
        if v.storage is not Storage.NUMERIC:
            raise SurveyInputError(f"{v.display_name} is not numeric; weights must be numbers.")
        w = ds.analysis_frame([d.weight])[d.weight]
        neg = int((w < 0).sum())
        if neg:
            raise SurveyInputError(f"{v.display_name} has {neg} negative values; sampling weights cannot be negative.")
        if int(w.isna().sum()):
            warnings.append(f"{int(w.isna().sum()):,} rows have a missing weight and are excluded from the design.")
        if int((w == 0).sum()):
            warnings.append(f"{int((w == 0).sum()):,} rows have weight 0: they are in the design but do not "
                            "contribute to estimates.")
    for r in d.replicate_weights:
        if ds.var(r).storage is not Storage.NUMERIC:
            raise SurveyInputError(f"Replicate weight {ds.var(r).display_name} is not numeric.")
    if d.replicate_type not in REP_TYPES:
        raise SurveyInputError("Unknown replication method.")
    # longitudinal data: a cross-sectional weight is not automatically valid for pooled waves
    ids = [v.name for v in ds.variables.values() if v.role is Role.ID]
    if ids and d.psu != ids[0]:
        f = ds.analysis_frame([ids[0]])
        if f[ids[0]].duplicated().any() and not d.pooled_waves_ack:
            raise SurveyInputError(
                f"Rows repeat participants ({ds.var(ids[0]).display_name}): this looks like longitudinal data. A single "
                "cross-sectional weight is not automatically valid across waves. Filter to one wave (Dataset Explorer ▸ "
                "Derived & Filters), or set the participant ID as the PSU and confirm that the weight is appropriate "
                "for the pooled rows.")
    if d.strata and d.psu:
        f = ds.analysis_frame([d.strata, d.psu]).dropna()
        per = f.groupby(d.strata)[d.psu].nunique()
        single = int((per == 1).sum())
        if single:
            msg = f"{single} strata contain a single PSU; policy: {LONELY_PSU[d.lonely_psu]}."
            if d.lonely_psu == "fail":
                raise SurveyInputError(msg + " Choose an explicit policy for single-PSU strata.")
            warnings.append(msg)
    return warnings


# ---- R code ----------------------------------------------------------------------------------------

def _design_lines(ds: Dataset, d: SurveyDesign, used: list[str]) -> list[str]:
    a = lambda n: alias(ds.var(n).alias)  # noqa: E731
    dv = [a(n) for n in d.variables()]
    L = [
        f"options(survey.lonely.psu = {r_str(d.lonely_psu)})  # explicit policy for strata with a single PSU",
        f"sx_dv <- {r_vec(dv)}",
        "sx_dd <- dat[stats::complete.cases(dat[, sx_dv, drop = FALSE]), , drop = FALSE]  # rows with design information",
    ]
    if d.weight:
        L.append(f"if (any(sx_dd${a(d.weight)} < 0)) stop(\"Negative sampling weights.\")")
    L.append(f"for (v in {r_vec([a(n) for n in used])}) if (is.factor(sx_dd[[v]])) sx_dd[[v]] <- droplevels(sx_dd[[v]])")
    if d.replicate_weights:
        L.append(f"sx_des <- survey::svrepdesign(data = sx_dd, repweights = sx_dd[, {r_vec([a(r) for r in d.replicate_weights])}],"
                 f" weights = {('~' + a(d.weight)) if d.weight else 'NULL'}, type = {r_str(d.replicate_type)}, combined.weights = TRUE)")
    else:
        L.append(f"sx_des <- survey::svydesign(ids = ~{a(d.psu) if d.psu else '1'}, strata = "
                 f"{('~' + a(d.strata)) if d.strata else 'NULL'}, weights = {('~' + a(d.weight)) if d.weight else 'NULL'}, "
                 f"fpc = {('~' + a(d.fpc)) if d.fpc else 'NULL'}, nest = {'TRUE' if d.nest else 'FALSE'}, data = sx_dd)")
    L += [
        f"sx_used <- {r_vec([a(n) for n in used])}",
        "sx_keep <- stats::complete.cases(sx_dd[, sx_used, drop = FALSE])",
        "sx_sub <- subset(sx_des, sx_keep)  # domain analysis: the full design is kept for the variance",
        "sx_info <- data.frame(rows_in_design = nrow(sx_dd), rows_analysed = sum(sx_keep),",
        "  sum_of_weights = sum(stats::weights(sx_sub, \"sampling\")),",
        "  PSUs = if (inherits(sx_sub, \"svyrep.design\")) NA_real_ else length(unique(sx_sub$cluster[[1]])),",
        "  strata = if (inherits(sx_sub, \"svyrep.design\")) NA_real_ else length(unique(sx_sub$strata[[1]])),",
        "  design_df = survey::degf(sx_sub))",
    ]
    return L


def build(ds: Dataset, d: SurveyDesign, kind: str, variables: list[str], outcome: str | None = None,
          conf: float = 0.95) -> str:
    a = lambda n: alias(ds.var(n).alias)  # noqa: E731
    if kind not in KINDS:
        raise SurveyInputError("Unknown survey analysis.")
    used = list(dict.fromkeys(([outcome] if outcome else []) + variables))
    if not used:
        raise SurveyInputError("Choose variables.")
    L = _design_lines(ds, d, used)
    z = f"stats::qnorm(1 - (1 - {conf!r}) / 2)"
    info_note = ("rows_analysed = raw (unweighted) sample size; sum_of_weights estimates the population size only if "
                 "the weights are expansion weights; design_df = degrees of freedom of the design (PSUs - strata).")
    if kind == "mean":
        bad = [ds.var(n).display_name for n in variables if ds.var(n).is_categorical]
        if bad:
            raise SurveyInputError(f"Means are for continuous variables: {', '.join(bad)} is categorical "
                                   "(use proportions).")
        L += [
            "sx_rows <- lapply(sx_used, function(v) {",
            "  m <- survey::svymean(stats::as.formula(paste0(\"~\", v)), sx_sub, deff = TRUE)",
            "  se <- unname(survey::SE(m)); est <- unname(stats::coef(m))",
            "  data.frame(variable = sx_names[[v]], n = sum(!is.na(sx_dd[sx_keep, v])), mean = est, se = se,",
            f"    ci_low = est - {z} * se, ci_high = est + {z} * se, design_effect = unname(survey::deff(m))) }})",
            "sx_tabs <- list(sx_table(\"Weighted means\", do.call(rbind, sx_rows), notes = paste0(\"Design-based (Taylor \",",
            "  \"linearization or replicate) standard errors. design_effect = variance relative to simple random sampling \",",
            "  \"of the same size.\")))",
        ]
    elif kind == "proportion":
        if not all(ds.var(n).is_categorical for n in variables):
            raise SurveyInputError("Proportions are for categorical variables.")
        L += ["sx_tabs <- list()", "for (v in sx_used) {",
              "  m <- survey::svymean(stats::as.formula(paste0(\"~\", v)), sx_sub)",
              "  tt <- survey::svytable(stats::as.formula(paste0(\"~\", v)), sx_sub)",
              "  raw <- table(sx_dd[sx_keep, v])  # all categories of the design rows (zeros kept)",
              "  est <- unname(stats::coef(m)); se <- unname(survey::SE(m))",
              "  sx_tabs[[length(sx_tabs) + 1]] <- sx_table(paste0(\"Weighted proportions: \", sx_names[[v]]), data.frame(",
              "    category = levels(sx_dd[[v]]), raw_n = as.numeric(raw),",
              "    weighted_count = as.numeric(tt), proportion = est, se = se,",
              f"    ci_low = pmax(0, est - {z} * se), ci_high = pmin(1, est + {z} * se)),",
              "    notes = \"raw_n = unweighted rows; weighted_count = sum of weights (not the number of independent observations); Wald CIs.\")",
              "}"]
    elif kind == "crosstab":
        if len(variables) != 2 or not all(ds.var(n).is_categorical for n in variables):
            raise SurveyInputError("Choose exactly two categorical variables.")
        r, c = a(variables[0]), a(variables[1])
        L += [
            f"sx_f <- ~ {r} + {c}",
            "sx_wt <- survey::svytable(sx_f, sx_sub)",
            "sx_obs <- as.data.frame.matrix(sx_wt)",
            f"sx_rawt <- as.data.frame.matrix(table(droplevels(sx_dd[sx_keep, \"{r}\"]), droplevels(sx_dd[sx_keep, \"{c}\"])))",
            "sx_rowp <- as.data.frame.matrix(100 * prop.table(sx_wt, 1))",
            "sx_chi <- survey::svychisq(sx_f, sx_sub, statistic = \"F\")",
            "sx_tabs <- list(",
            "  sx_table(\"Weighted counts\", sx_obs, row_names = rownames(sx_obs), notes = \"Sums of weights, not numbers of respondents.\"),",
            "  sx_table(\"Raw (unweighted) counts\", sx_rawt, row_names = rownames(sx_rawt)),",
            "  sx_table(\"Weighted row percentages\", sx_rowp, row_names = rownames(sx_rowp)),",
            "  sx_table(\"Design-adjusted test of independence\", data.frame(test = \"Rao-Scott second-order (F)\",",
            "    F = unname(sx_chi$statistic), df1 = unname(sx_chi$parameter[1]), df2 = unname(sx_chi$parameter[2]),",
            "    p = sx_chi$p.value), notes = \"Pearson chi-square corrected for the survey design (survey::svychisq).\"))",
        ]
    else:
        if not outcome:
            raise SurveyInputError("Choose an outcome.")
        preds = [n for n in variables if n != outcome]
        if not preds:
            raise SurveyInputError("Choose at least one predictor.")
        yv = ds.var(outcome)
        if kind == "linear" and not (yv.measure is Measure.CONTINUOUS and yv.storage is Storage.NUMERIC):
            raise SurveyInputError(f"{yv.display_name} is not continuous.")
        k = ds.analysis_frame([outcome])[outcome].dropna().nunique()
        if kind == "logistic" and k != 2:
            raise SurveyInputError(f"Logistic regression needs exactly two outcome values; {yv.display_name} has {k}.")
        y = a(outcome)
        rhs = " + ".join(a(p) for p in preds)
        if kind == "logistic":
            L += [f"sx_lv <- if (is.factor(sx_dd${y})) levels(droplevels(sx_dd${y}[sx_keep])) else sort(unique(sx_dd${y}[sx_keep]))",
                  f"sx_sub <- stats::update(sx_sub, sx_event = as.integer({y} == sx_lv[2]))",
                  f"sx_fit <- survey::svyglm(sx_event ~ {rhs}, design = sx_sub, family = stats::quasibinomial())"]
        else:
            L.append(f"sx_fit <- survey::svyglm({y} ~ {rhs}, design = sx_sub)")
        L += [
            "sx_co <- as.data.frame(summary(sx_fit)$coefficients)",
            "sx_ci <- stats::confint(sx_fit, level = " + repr(conf) + ")",
            "sx_tab <- data.frame(term = rownames(sx_co), estimate = sx_co[[1]], se = sx_co[[2]], t = sx_co[[3]], p = sx_co[[4]],",
            "  ci_low = sx_ci[, 1], ci_high = sx_ci[, 2])",
        ]
        if kind == "logistic":
            L.append("sx_tab$odds_ratio <- exp(sx_tab$estimate); sx_tab$or_ci_low <- exp(sx_tab$ci_low); sx_tab$or_ci_high <- exp(sx_tab$ci_high)")
        note = ("Design-based standard errors; t tests on the design degrees of freedom. "
                + ("Event = second category (\" , sx_lv[2], \"). " if kind == "logistic" else ""))
        L += [
            f"sx_tabs <- list(sx_table(\"Survey-weighted coefficients\", sx_tab, notes = paste0(\"{note}\", "
            "\"Survey-weighted estimates describe the population the weights represent; they are not a substitute for "
            "modelling repeated measurements (see Mixed & Longitudinal Models), and vice versa.\")))",
            "sx_terms <- do.call(rbind, lapply(attr(stats::terms(sx_fit), \"term.labels\"), function(tl) {",
            "  w <- survey::regTermTest(sx_fit, stats::as.formula(paste0(\"~\", tl)))",
            "  data.frame(term = tl, F = as.numeric(w$Ftest), df1 = w$df, df2 = w$ddf, p = as.numeric(w$p)) }))",
            "sx_tabs[[2]] <- sx_table(\"Term tests (design-based Wald F)\", sx_terms)",
        ]
    L += [
        f"sx_tabs <- c(list(sx_table(\"Survey design\", sx_info, notes = {r_str(info_note)})), sx_tabs)",
        "sx_result <- list(tables = sx_tabs, n_used = sum(sx_keep), n_excluded = nrow(dat) - sum(sx_keep))",
    ]
    return "\n".join(L) + "\n"


def run(session: RSession, ds: Dataset, d: SurveyDesign, kind: str, variables: list[str],
        outcome: str | None = None, conf: float = 0.95) -> tuple[AnalysisResult, dict[str, bytes]]:
    from .framework import _alias_map, _as_list, _parse_table

    warnings = check_design(ds, d)
    body = build(ds, d, kind, variables, outcome, conf)
    result, raw = session.run(ds, kind="survey", title=f"Survey analysis: {KINDS[kind]}", body=body,
                              packages=["survey"], inputs={"design": d.to_dict(), "analysis": kind,
                                                           "variables": variables, "outcome": outcome,
                                                           "filters": [f.describe() for f in ds.filters]})
    result.missing_data = "Domain analysis: rows missing the analysed variables are excluded via subset() of the full design."
    result.values["design"] = explain_design(ds, d)
    for w in warnings:
        result.messages.append(f"Check: {w}")
        if result.status is Status.OK:
            result.status = Status.WARNING
    if raw is not None:
        names = _alias_map(ds)
        for t in _as_list(raw.get("tables")):
            result.tables.append(_parse_table(t, names))
        result.n_used = raw.get("n_used")
        result.values["n_excluded"] = raw.get("n_excluded")
        result.values["notes"] = explain_design(ds, d)
    return result, {}
