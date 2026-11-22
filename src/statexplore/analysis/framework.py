"""Declarative analysis specs: role slots + options -> R body -> AnalysisResult.

A spec knows nothing about Qt. The Analysis Builder generates its form from
`slots` and `options`, and tests call `run_analysis()` directly.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.dataset import Dataset
from ..core.schema import AnalysisResult, Status, Table
from ..core.variables import Measure, Role, Storage
from ..rbridge.codegen import alias, r_num, r_str, r_vec
from .session import RSession

KIND_LABELS = {
    "continuous": "continuous",
    "categorical": "categorical (nominal/ordinal)",
    "binary": "exactly two values",
    "any": "continuous or categorical",
    "rankable": "continuous or ordinal",
    "ordinal": "ordinal (ordered categories)",
}


class AnalysisInputError(ValueError):
    """User-facing message explaining why an analysis cannot run."""


@dataclass
class Slot:
    key: str
    label: str
    kind: str                 # continuous | categorical | binary | any
    min: int = 1
    max: int = 1
    help: str = ""


@dataclass
class Option:
    key: str
    label: str
    kind: str                 # float | int | choice | bool | level (a category of the variable in `slot`)
    default: Any
    choices: list[tuple[str, str]] | None = None   # (value, label)
    min: float | None = None
    max: float | None = None
    help: str = ""
    slot: str = ""            # kind "level": the slot whose variable's categories are offered
    required: bool = True     # kind "level": False = may be left unset (the builder's default applies)


@dataclass
class AnalysisSpec:
    key: str
    family: str
    label: str
    description: str
    slots: list[Slot]
    options: list[Option]
    packages: list[str]
    build: Callable[["Ctx"], str]
    n_plots: int = 0
    missing_data: str = "Complete cases on the analysis variables (listwise deletion)."
    # Optional extra validation (ds, variables, cleaned options) -> raises AnalysisInputError.
    check: Callable[[Dataset, dict[str, list[str]], dict[str, Any]], None] | None = None


@dataclass
class Ctx:
    """What a builder sees: the dataset, validated parameters, and plot file paths."""
    ds: Dataset
    vars: dict[str, list[str]]           # slot key -> variable names
    opts: dict[str, Any]
    plot_paths: list[str] = field(default_factory=list)

    def a(self, name: str) -> str:
        return alias(self.ds.var(name).alias)

    def aliases(self, slot: str) -> list[str]:
        return [self.a(n) for n in self.vars.get(slot, [])]

    def one(self, slot: str) -> str:
        return self.aliases(slot)[0]

    def used(self, *slots: str) -> list[str]:
        out: list[str] = []
        for s in slots:
            for a in self.aliases(s):
                if a not in out:
                    out.append(a)
        return out

    def is_cat(self, name_or_alias: str) -> bool:
        v = self.ds.by_alias(name_or_alias) if re.fullmatch(r"v\d{3,}", name_or_alias) else self.ds.var(name_or_alias)
        return v.is_categorical

    def num(self, key: str) -> str:
        return r_num(self.opts[key])

    def str_(self, key: str) -> str:
        return r_str(self.opts[key])

    def vec(self, items: list[Any]) -> str:
        return r_vec(items)


# ---- validation -------------------------------------------------------------

def _matches(ds: Dataset, name: str, kind: str) -> bool:
    v = ds.var(name)
    if v.storage is Storage.DATETIME:
        return False
    if kind == "continuous":
        return v.measure is Measure.CONTINUOUS and v.storage is Storage.NUMERIC
    if kind == "categorical":
        return v.is_categorical
    if kind == "binary":
        return ds.analysis_frame([name])[name].dropna().nunique() == 2
    if kind == "any":
        return v.is_categorical or (v.measure is Measure.CONTINUOUS and v.storage is Storage.NUMERIC)
    if kind == "ordinal":
        return v.measure is Measure.ORDINAL
    if kind == "rankable":
        return v.measure is Measure.ORDINAL or (v.measure is Measure.CONTINUOUS and v.storage is Storage.NUMERIC)
    raise ValueError(kind)


def treatable(ds: Dataset, name: str) -> bool:
    """A numeric categorical variable that may be treated as continuous for one analysis (§6)."""
    from .measurement import continuous_problem

    v = ds.var(name)
    return v.is_categorical and v.storage is Storage.NUMERIC and not continuous_problem(ds, name)


def eligible(ds: Dataset, slot: Slot, offer_treatment: bool = False) -> list[str]:
    """Variables offered for a slot: right kind, not constant, and not the ID (except in ID slots).
    With `offer_treatment`, continuous slots also list numeric categorical variables, which
    become usable once the user ticks "treat as continuous for this analysis"."""
    from ..core.variables import Role

    out = []
    for n in ds.names():
        v = ds.var(n)
        if v.role is Role.ID and slot.key != "id":
            continue
        ok = _matches(ds, n, slot.kind) or (offer_treatment and slot.kind in ("continuous", "rankable")
                                            and treatable(ds, n))
        if ok and ds.analysis_frame([n])[n].dropna().nunique() > 1:
            out.append(n)
    return out


def validate(spec: AnalysisSpec, ds: Dataset, vars_: dict[str, list[str]], opts: dict[str, Any]) -> dict[str, Any]:
    seen: dict[str, str] = {}
    for slot in spec.slots:
        chosen = vars_.get(slot.key, [])
        if len(chosen) < slot.min:
            need = "at least one variable" if slot.min == 1 else f"at least {slot.min} variables"
            raise AnalysisInputError(f"{slot.label}: choose {need}.")
        if len(chosen) > slot.max:
            raise AnalysisInputError(f"{slot.label}: choose at most {slot.max}.")
        for n in chosen:
            if n in seen:
                raise AnalysisInputError(f"{ds.var(n).display_name} is used in both "
                                         f"'{seen[n]}' and '{slot.label}'.")
            seen[n] = slot.label
            if not _matches(ds, n, slot.kind):
                hint = ""
                if slot.kind in ("continuous", "rankable") and treatable(ds, n):
                    hint = (f" It is {ds.var(n).measure.value} in this project: tick \"Treat "
                            f"{ds.var(n).display_name} as continuous for this analysis\" to use its numeric codes.")
                raise AnalysisInputError(f"{slot.label}: {ds.var(n).display_name} must be "
                                         f"{KIND_LABELS[slot.kind]}.{hint}")
    from .models import variable_problem

    chosen_all = [n for names in vars_.values() for n in names]
    if chosen_all:
        frame = ds.analysis_frame(list(dict.fromkeys(chosen_all))).dropna()
        for slot in spec.slots:
            for n in vars_.get(slot.key, []):
                why = variable_problem(ds, n, frame, allow_many_levels=slot.key in ("id", "variables"))
                if why:
                    raise AnalysisInputError(why)
    clean: dict[str, Any] = {}
    for o in spec.options:
        val = opts.get(o.key, o.default)
        if o.kind == "float":
            val = float(val)
        elif o.kind == "int":
            val = int(val)
        elif o.kind == "bool":
            val = bool(val)
        elif o.kind == "choice" and val not in [c[0] for c in o.choices or []]:
            raise AnalysisInputError(f"{o.label}: invalid choice {val!r}.")
        elif o.kind == "level":
            chosen = vars_.get(o.slot, [])
            if val is None and o.required:
                raise AnalysisInputError(f"{o.label}: choose a category.")
            if val is not None and chosen and val not in ds.levels(chosen[0]):
                raise AnalysisInputError(f"{o.label}: {val!r} is not a category of {ds.var(chosen[0]).display_name}.")
        if o.min is not None and val < o.min or o.max is not None and val > o.max:
            raise AnalysisInputError(f"{o.label} must be between {o.min} and {o.max}.")
        clean[o.key] = val
    if spec.check is not None:
        spec.check(ds, vars_, clean)
    return clean


# ---- running & parsing --------------------------------------------------------

def run_analysis(session: RSession, ds: Dataset, spec: AnalysisSpec, vars_: dict[str, list[str]],
                 opts: dict[str, Any] | None = None, internal: dict[str, Any] | None = None,
                 title: str | None = None, treatments: list[Any] | None = None,
                 ) -> tuple[AnalysisResult, dict[str, bytes]]:
    """`internal` holds program-built options that never come from the UI (e.g. an explicit
    model formula from the discovery engine); values must be built from validated aliases.
    `treatments` are analysis-specific encodings (measurement.VariableTreatment): this
    analysis sees a view of the dataset; the project metadata is not changed."""
    from .measurement import TreatmentError, apply_treatments, as_treatments, record

    try:
        ds = apply_treatments(ds, as_treatments(treatments))
    except TreatmentError as e:
        raise AnalysisInputError(str(e)) from None
    clean = validate(spec, ds, vars_, opts or {})
    clean.update(internal or {})
    plot_paths = [session.new_plot_path() for _ in range(spec.n_plots)]
    ctx = Ctx(ds=ds, vars={k: list(v) for k, v in vars_.items()}, opts=clean,
              plot_paths=[p.as_posix() for p in plot_paths])
    body = spec.build(ctx)
    inputs = {
        "analysis": spec.key,
        "variables": {k: [{"name": n, "alias": ds.var(n).alias} for n in v] for k, v in ctx.vars.items()},
        "options": clean,
        "filters": [f.describe() for f in ds.filters],
    }
    dependence = independence_check(ds, spec, ctx.vars)
    result, raw = session.run(ds, kind=spec.key, title=title or spec.label, body=body,
                              packages=spec.packages, inputs=inputs)
    result.missing_data = spec.missing_data
    record(result, ds)
    if dependence and raw is not None:
        result.messages.append(f"Check: {dependence}")
        if result.status is Status.OK:
            result.status = Status.WARNING
    artifacts: dict[str, bytes] = {}
    if raw is None:
        return result, artifacts
    names = _alias_map(ds)
    for t in raw.get("tables") or []:
        result.tables.append(_parse_table(t, names))
    result.n_used = _scalar(raw.get("n_used"))
    excluded = _scalar(raw.get("n_excluded"))
    if excluded:
        result.values["n_excluded"] = excluded
    result.values["notes"] = [_rename(str(n), names) for n in _as_list(raw.get("notes"))]
    checks = [_rename(str(c), names) for c in _as_list(raw.get("checks"))]
    if checks:
        result.messages.extend(f"Check: {c}" for c in checks)
        if result.status is Status.OK:
            result.status = Status.WARNING
    for p in plot_paths:
        if p.is_file():
            key = f"plots/{p.name}"
            artifacts[key] = p.read_bytes()
            result.plots.append(key)
    result.messages = [_rename(m, names) for m in result.messages]
    return result, artifacts


# Analyses that model repeated measurements themselves.
HANDLES_REPEATED = {"anova_rm", "anova_mixed"}


def independence_check(ds: Dataset, spec: AnalysisSpec, vars_: dict[str, list[str]]) -> str:
    """If a participant-ID variable is set and the analysed rows repeat IDs, the analysis's
    independence assumption is violated (§7.2 effective sample size)."""
    if spec.key in HANDLES_REPEATED:
        return ""
    ids = [v.name for v in ds.variables.values() if v.role is Role.ID]
    used = [n for names in vars_.values() for n in names]
    if not ids or not used:
        return ""
    frame = ds.analysis_frame(list(dict.fromkeys(used + ids[:1]))).dropna(subset=used)
    n_rows, n_ids = len(frame), frame[ids[0]].nunique()
    if n_ids >= n_rows:
        return ""
    return (f"The {n_rows:,} analysed rows come from only {n_ids:,} participants "
            f"({ds.var(ids[0]).display_name}). Rows from the same participant are not independent, "
            "so standard errors and p-values here are too small. Analyse one row per participant "
            "(e.g. filter to one wave) or use a mixed model (Phase 5).")


def _as_list(x: Any) -> list[Any]:
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def _scalar(x: Any) -> Any:
    if isinstance(x, list):
        return x[0] if x else None
    return x


def _alias_map(ds: Dataset) -> dict[str, str]:
    return {v.alias: v.display_name for v in ds.variables.values()}


_ALIAS_TOKEN = re.compile(r"(?<![A-Za-z0-9_.])v\d{3,4}")
_SEPARATORS = set(":*() ,|-~+^=.[]")


def _rename(text: str, names: dict[str, str]) -> str:
    """Replace internal aliases with display names. A factor coefficient such as
    'v014Female' becomes 'RAGENDER=Female'."""
    out: list[str] = []
    pos = 0
    for m in _ALIAS_TOKEN.finditer(text):
        token = m.group(0)
        # prefer the longest alias that is actually defined (v1000 vs v100 + "0")
        while token not in names and len(token) > 4:
            token = token[:-1]
        if token not in names:
            continue
        end = m.start() + len(token)
        out.append(text[pos:m.start()])
        out.append(names[token])
        if end < len(text) and text[end] not in _SEPARATORS:
            out.append("=")
        pos = end
    out.append(text[pos:])
    return "".join(out)


def _parse_table(t: dict[str, Any], names: dict[str, str], extra: dict[str, str] | None = None) -> Table:
    """`extra` maps internal non-alias tokens (e.g. sx_time) to display names."""
    if extra:
        base = names

        def names_fn(text: str) -> str:
            text = _rename(text, base)
            for k, v in extra.items():
                text = text.replace(k, v)
            return text
        return _parse_table_with(t, names_fn)
    return _parse_table_with(t, lambda text: _rename(text, names))


def _parse_table_with(t: dict[str, Any], rename) -> Table:
    columns = [str(c) for c in _as_list(t.get("columns"))]
    data = t.get("data") or {}
    cols = [_as_list(data.get(c)) for c in columns]
    n = max((len(c) for c in cols), default=0)
    rows = []
    for i in range(n):
        row = []
        for c in cols:
            v = c[i] if i < len(c) else None
            row.append(rename(v) if isinstance(v, str) else v)
        rows.append(row)
    row_names = [rename(str(r)) for r in _as_list(t.get("row_names"))]
    if row_names and len(row_names) != n:
        raise ValueError(f"Table {t.get('name')!r}: {len(row_names)} row names for {n} rows")
    return Table(
        name=rename(str(t.get("name", ""))),
        columns=[rename(c) for c in columns],
        rows=rows,
        row_names=row_names,
        notes=rename(str(t.get("notes") or "")),
    )


# ---- shared option sets -------------------------------------------------------

CONF = Option("conf", "Confidence level", "float", 0.95, min=0.5, max=0.999)
ALTERNATIVE = Option("alternative", "Alternative hypothesis", "choice", "two.sided",
                     [("two.sided", "Two-sided"), ("less", "Less"), ("greater", "Greater")])
ADJUST = Option("adjust", "Post hoc p-value adjustment", "choice", "holm",
                [("holm", "Holm"), ("tukey", "Tukey"), ("bonferroni", "Bonferroni"), ("none", "None")])
INTERACTIONS = Option("interactions", "Interactions among predictors", "choice", "none",
                      [("none", "None (main effects)"), ("two_way", "All two-way"),
                       ("full", "Full factorial (up to 4 predictors)")])
