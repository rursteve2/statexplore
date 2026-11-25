"""Measurement levels (v1.1 §3–§6).

Three separate concepts:
  A. source level      VariableMeta.source_measure   (as declared by the file; never changed)
  B. project level     VariableMeta.measure          (user-approved; logged + undoable)
  C. analysis treatment VariableTreatment            (one analysis only; never stored on the dataset)

Suggestions (`suggest`) are transparent rules over value labels, storage and distinct values.
They are shown for review and never applied automatically.

Treatments are applied by `apply_treatments`, which returns a *view* of the dataset whose
variable metadata carries the effective measure for that analysis. The raw data, the
source metadata and the project metadata are not modified; every validator, model builder
and the R load script then honour the treatment without special cases, and the R session
rebuilds its prepared data whenever the effective encoding changes (its cache key hashes
the metadata).
"""
from __future__ import annotations

import copy
import dataclasses
import re
from dataclasses import dataclass, field
from typing import Any

from ..core.dataset import Dataset, DatasetError
from ..core.variables import (
    STATUS_APPROVED, STATUS_KEPT, STATUS_SOURCE, Measure, Role, Storage, VariableMeta,
)

CONTINUOUS_WARNING = (
    "Treating this ordinal or categorically coded variable as continuous assumes its numeric values can be "
    "interpreted quantitatively. Results may differ from categorical or ordinal models."
)
NOMINAL_NUMERIC_WARNING = (
    "This variable is classified as nominal (unordered categories). Its codes are identifiers, not "
    "quantities, unless you know otherwise. Confirm only if the numeric codes are meaningful numbers."
)
CODED_REASON = "Numeric codes without labels from a file that has no measurement levels: "
EQUAL_INTERVALS = "The intervals between ordered response categories are not guaranteed to be equal."

TREATMENTS = {
    "continuous": "Continuous (numeric codes)",
    "ordinal": "Ordered categories",
    "nominal": "Unordered categories",
}


class TreatmentError(ValueError):
    """User-facing reason why a treatment cannot be applied."""


# ---- suggestions ---------------------------------------------------------------------------

# Each scale: (name, reason, (text when higher rank has higher codes, text otherwise),
#              [(pattern, rank), ...]) — patterns are tried in order; the first match wins,
#              so more specific patterns come first. Rank = position on the scale's dimension.
_SCALES: list[tuple[str, str, tuple[str, str], list[tuple[str, float]]]] = [
    ("frequency", "Ordered frequency categories", ("more frequent", "less frequent"), [
        (r"lt 1 per mon|less than once a month|rarely|seldom|hardly ever|a little of the time", 2),
        (r"every ?day|daily|always|all of the time", 6),
        (r"3\+ per week|more than once|>\s*1 per|several times a week|most of the time|often|usually", 5),
        (r"1-2 per week|once a week|1 per week|once per week|weekly|sometimes|some of the time", 4),
        (r"[1l]-3 (per|a) mon|per mon|monthly|a month", 3),
        (r"never|none of the time", 1),
    ]),
    ("health", "Ordered health / quality ratings", ("better rating", "worse rating"), [
        (r"excellent", 5), (r"very good", 4), (r"\bgood\b", 3), (r"\bfair\b", 2), (r"\bpoor\b", 1),
    ]),
    ("agreement", "Ordered agreement categories", ("more agreement", "less agreement"), [
        (r"strongly disagree", 1), (r"strongly agree", 5), (r"disagree", 2),
        (r"neither|neutral|undecided", 3), (r"\bagree", 4),
    ]),
    ("degree", "Ordered degree / intensity categories", ("more of the attribute", "less of the attribute"), [
        (r"did not happen", 0), (r"not at all", 1), (r"not very", 2), (r"\bnot\b", 1.5),
        (r"a little|slightly|mild", 2), (r"somewhat|moderately|moderate", 3),
        (r"completely|extremely", 5), (r"\bvery\b|a lot|severe", 4),
    ]),
    ("education", "Ordered education levels", ("more education", "less education"), [
        (r"lt high|less than high|no degree", 1), (r"hs/ged|high-?school grad|^hs\b", 3), (r"\bged\b", 2),
        (r"some college|\baa\b|lt ba", 4), (r"college and above|\bba\b|bachelor", 5),
        (r"\bma\b|\bms\b|master|mba", 6), (r"law|\bmd\b|phd|doctor", 7),
    ]),
]

_ID_RE = re.compile(r"identif|\bid\b|\bid number|person number", re.I)
_PREFIX = re.compile(r"^\s*-?\d+(\.\d+)?\s*[.)]?\s*(\d+(\.\d+)?\s*[.)]\s*)?")


def _clean(label: str) -> str:
    return _PREFIX.sub("", str(label)).strip().lower()


def _code(c: Any) -> str:
    return str(int(c)) if isinstance(c, float) and c.is_integer() else str(c)


@dataclass
class Suggestion:
    measure: str                          # "" = no suggestion
    reasons: list[str] = field(default_factory=list)
    direction: str = ""                   # e.g. "higher codes = less frequent"


def _ordered_scale(codes: list[Any], labels: dict[Any, str]) -> tuple[str, str, list[Any]] | None:
    """(reason, direction, unmatched codes) when the labelled codes follow one known ordered scale."""
    best = None
    for _name, reason, (up, down), patterns in _SCALES:
        ranks: list[tuple[Any, float]] = []
        unmatched = []
        for c in codes:
            text = _clean(labels[c])
            rank = next((r for p, r in patterns if re.search(p, text)), None)
            if rank is None:
                unmatched.append(c)
            else:
                ranks.append((c, rank))
        values = [r for _, r in ranks]
        if len(set(values)) < 3 or len(ranks) < len(codes) / 2:
            continue
        inc = all(a <= b for a, b in zip(values, values[1:]))
        dec = all(a >= b for a, b in zip(values, values[1:]))
        if not (inc or dec):
            continue
        if best is None or len(ranks) > best[0]:
            best = (len(ranks), reason, f"higher codes = {up if inc else down}", unmatched)
    return None if best is None else best[1:]


def suggest(ds: Dataset, name: str) -> Suggestion:
    """Rule-based measurement-level suggestion for one variable (never applied automatically)."""
    v = ds.var(name)
    if v.storage is Storage.DATETIME or ds.is_derived(name):
        return Suggestion("")
    col = ds.analysis_column(name).dropna()
    k = int(col.nunique())
    if k == 0:
        return Suggestion("")
    if v.role in (Role.ID, Role.CLUSTER) or _ID_RE.search(f"{v.label} {v.name}"):
        return Suggestion(Measure.NOMINAL.value, ["Looks like an identifier (role, name or label): codes are "
                                                  "labels for units, not quantities."])
    if v.storage is Storage.STRING:
        return Suggestion(Measure.BINARY.value if k == 2 else Measure.NOMINAL.value,
                          ["Text values cannot be numbers."] + (["Exactly two observed categories."] if k == 2 else []))
    if k == 2:
        return Suggestion(Measure.BINARY.value, ["Exactly two observed values."])
    observed = sorted(col.unique().tolist())
    labelled = [c for c in observed if c in v.value_labels]
    if len(labelled) >= 3 and len(labelled) >= len(observed) / 2:
        hit = _ordered_scale(labelled, v.value_labels)
        if hit:
            reason, direction, unmatched = hit
            reasons = [f"{reason}: " + "; ".join(f"{_code(c)} = {v.value_labels[c]}" for c in labelled
                                                  if c not in unmatched) + ".",
                       "Rule-based on the value labels; check the questionnaire wording."]
            if unmatched:
                reasons.append("Not part of the ordered scale: " + "; ".join(
                    f"{_code(c)} = {v.value_labels[c]}" for c in unmatched)
                    + ". Declare such codes missing (or recode) before ordinal or numeric analyses.")
            return Suggestion(Measure.ORDINAL.value, reasons, direction)
        return Suggestion(Measure.NOMINAL.value if v.measure is not Measure.ORDINAL else Measure.ORDINAL.value,
                          ["Labelled categories without a recognised order; keep the declared level unless the "
                           "categories are ordered."])
    if (not labelled and v.measure is Measure.NOMINAL and v.source_measure in ("", "unknown") and k <= 9
            and all(float(c).is_integer() and c >= 0 for c in observed)):
        return Suggestion("", [CODED_REASON + ", ".join(_code(c) for c in observed) + ". They were read as "
                               "unordered categories. If they are quantities (counts, scores), set the level to "
                               "continuous; if they are ordered categories, set ordinal."])
    if k >= 10 or (v.source_measure == "scale" and not labelled):
        reasons = [f"Numeric quantitative data: {k} distinct values"
                   + ("" if labelled else ", no category labels") + "."]
        if labelled:
            reasons.append("Some values carry labels (anchors or special codes): " + "; ".join(
                f"{_code(c)} = {v.value_labels[c]}" for c in labelled)
                + ". Check that these codes are real quantities before numeric analyses.")
        return Suggestion(Measure.CONTINUOUS.value, reasons)
    return Suggestion("")


def annotate(ds: Dataset) -> None:
    """Fill suggestion fields on every variable (metadata only; the measure is never changed)."""
    for n in ds.variables:
        s = suggest(ds, n)
        v = ds.var(n)
        v.suggested_measure, v.suggestion_reasons = s.measure, s.reasons
        if s.direction and not v.coding_direction:
            v.coding_direction = s.direction


def needs_review(v: VariableMeta) -> bool:
    """A suggestion that differs from the project level in a way that changes analyses, or a few unlabelled
    integer codes from an untyped file whose level nobody has reviewed yet."""
    if v.measure_status == STATUS_SOURCE and v.suggestion_reasons and v.suggestion_reasons[0].startswith(CODED_REASON):
        return True
    if not v.suggested_measure or v.suggested_measure == v.measure.value:
        return False
    return not {v.suggested_measure, v.measure.value} <= {Measure.NOMINAL.value, Measure.BINARY.value}


def review_rows(ds: Dataset, only_mismatches: bool = True, include_all: bool = False) -> list[dict[str, Any]]:
    """Rows of the Measurement Review table: variables needing review (default), every variable with a suggestion
    (`only_mismatches=False`) or every variable (`include_all`, UI-29). `override` says whether a project level
    different from the source file's is saved."""
    rows = []
    for v in ds.variables.values():
        if not include_all and (not (v.suggested_measure or needs_review(v))
                                or (only_mismatches and not needs_review(v) and v.measure_status == STATUS_SOURCE)):
            continue
        rows.append({"name": v.name, "label": v.label, "source": v.source_measure or "—",
                     "project": v.measure.value, "suggested": v.suggested_measure,
                     "reason": " ".join(v.suggestion_reasons), "status": v.measure_status,
                     "needs_review": needs_review(v), "direction": v.coding_direction,
                     "override": (f"saved: {_source_level(v) or '—'} → {v.measure.value}"
                                  if v.measure.value != _source_level(v) and v.measure_status != STATUS_SOURCE else
                                  "reviewed, kept" if v.measure_status == STATUS_KEPT else "")})
    return rows


def approve(ds: Dataset, name: str, measure: str | Measure, category_order: list[Any] | None = None) -> None:
    """Set the project measurement level after an explicit user decision (logged, undoable)."""
    v = ds.var(name)
    m = Measure(measure)
    if m is Measure.CONTINUOUS and v.storage is not Storage.NUMERIC:
        raise DatasetError(f"{v.display_name} holds text; text values cannot be continuous.")
    if m is Measure.BINARY and ds.analysis_column(name).dropna().nunique() > 2:
        raise DatasetError(f"{v.display_name} has more than two categories; it cannot be binary.")
    if category_order is not None:
        levels = ds.levels(name)
        if sorted(map(str, category_order)) != sorted(map(str, levels)):
            raise DatasetError("The category order must list every category exactly once.")
        ds.set_variable_field(name, "category_order", list(category_order))
    ds.set_variable_field(name, "measure", m)
    status = STATUS_KEPT if m.value == _source_level(v) else STATUS_APPROVED
    ds.set_variable_field(name, "measure_status", status)


def keep(ds: Dataset, name: str) -> None:
    """Mark the variable reviewed with its current level kept."""
    ds.set_variable_field(name, "measure_status", STATUS_KEPT)


def _source_level(v: VariableMeta) -> str:
    return {"scale": "continuous"}.get(v.source_measure, v.source_measure)


def source_level(v: VariableMeta) -> str:
    """The level the source file declared, in project terms ('' when the file declared none)."""
    lvl = _source_level(v)
    return "" if lvl in ("", "unknown", None) else lvl


# ---- analysis-specific treatment --------------------------------------------------------------

@dataclass
class VariableTreatment:
    """How one variable is encoded for one analysis (§3.1 C). Stored with the analysis, never
    on the dataset."""
    variable: str
    treatment: str                       # continuous | ordinal | nominal
    acknowledged: bool = False           # §6.2 warning acknowledged (continuous)
    numeric_confirmed: bool = False      # §6: nominal codes confirmed to be meaningful numbers
    ordered_levels: list[Any] | None = None
    reference_level: Any = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "VariableTreatment":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def describe(self, ds: Dataset) -> str:
        v = ds.var(self.variable)
        txt = f"{v.display_name}: {TREATMENTS.get(self.treatment, self.treatment)} for this analysis " \
              f"(project level: {v.measure.value}"
        if v.source_measure:
            txt += f"; source: {v.source_measure}"
        txt += ")"
        if self.treatment == "continuous":
            codes = [c for c in ds.levels(self.variable) if c in v.value_labels]
            if codes:
                txt += ". Numeric codes used as values: " + "; ".join(
                    f"{_code(c)} = {v.value_labels[c]}" for c in codes)
            if v.coding_direction:
                txt += f". Direction: {v.coding_direction}"
            txt += ". Warning acknowledged."
        elif self.ordered_levels:
            txt += ". Order: " + " < ".join(ds.level_label(self.variable, c) for c in self.ordered_levels)
        return txt


def continuous_problem(ds: Dataset, name: str) -> str:
    """Why a variable can never be treated as continuous ('' if it can, given acknowledgement)."""
    v = ds.var(name)
    if v.storage is not Storage.NUMERIC:
        return f"{v.display_name} holds text values; text cannot be treated as numbers."
    if ds.analysis_column(name).dropna().nunique() < 2:
        return f"{v.display_name} has fewer than two distinct values."
    return ""


def treatment_status(ds: Dataset, name: str) -> tuple[str, str]:
    """(status, explanation) for the 'treat as continuous' control of one variable (M §6.1):
    eligible | continuous (already continuous — no override needed) | unavailable (with the reason)."""
    from ..core.variables import Role

    v = ds.var(name)
    if v.role is Role.ID:
        return "unavailable", f"Unavailable: {v.display_name} is an identifier (ID role), not a measurement."
    if v.storage is Storage.DATETIME:
        return "unavailable", "Unavailable: dates and times cannot be treated as numeric codes here."
    if v.storage is not Storage.NUMERIC:
        return "unavailable", "Unavailable: text categories have no numeric codes."
    if v.measure is Measure.CONTINUOUS:
        return "continuous", "Already continuous — no override needed."
    k = ds.analysis_column(name).dropna().nunique()
    if k == 0:
        return "unavailable", "Unavailable: no valid values in the analysed rows."
    if k < 2:
        return "unavailable", "Unavailable: constant in the analysed rows (after filters and missing codes)."
    codes = sorted(ds.analysis_column(name).dropna().unique().tolist())
    lo, hi = _code(codes[0]), _code(codes[-1])
    extra = " (needs a second confirmation: nominal codes)" if needs_numeric_confirmation(ds, name) else ""
    return "eligible", f"Eligible: {k} numeric {v.measure.value} codes from {lo} to {hi}{extra}."


def coding_fingerprint(ds: Dataset, name: str) -> str:
    """Changes whenever the variable's values or coding change; an acknowledgement is only valid
    for the exact coding it was given for (M §6.2)."""
    import hashlib
    import json

    v = ds.var(name)
    col = ds.analysis_column(name).dropna()
    payload = [v.measure.value, v.storage.value, sorted(map(str, col.unique().tolist())),
               sorted((str(k), str(lab)) for k, lab in v.value_labels.items()),
               [str(x) for x in v.missing_values], [list(map(str, r)) for r in v.missing_ranges]]
    return hashlib.sha1(json.dumps(payload).encode()).hexdigest()


def needs_numeric_confirmation(ds: Dataset, name: str) -> bool:
    v = ds.var(name)
    return v.measure in (Measure.NOMINAL, Measure.BINARY) and ds.analysis_column(name).dropna().nunique() > 2


def check_treatment(ds: Dataset, t: VariableTreatment) -> None:
    v = ds.var(t.variable)
    if t.treatment not in TREATMENTS:
        raise TreatmentError(f"Unknown treatment {t.treatment!r}.")
    if t.treatment == "continuous":
        why = continuous_problem(ds, t.variable)
        if why:
            raise TreatmentError(why)
        if v.measure is not Measure.CONTINUOUS:
            if not t.acknowledged:
                raise TreatmentError(f"{v.display_name}: acknowledge the warning before treating it as "
                                     f"continuous. {CONTINUOUS_WARNING}")
            if needs_numeric_confirmation(ds, t.variable) and not t.numeric_confirmed:
                raise TreatmentError(f"{v.display_name}: {NOMINAL_NUMERIC_WARNING}")
    else:
        k = ds.analysis_column(t.variable).dropna().nunique()
        if v.storage is Storage.DATETIME:
            raise TreatmentError(f"{v.display_name} is a date/time and cannot be categorical.")
        if k > 50:
            raise TreatmentError(f"{v.display_name} has {k} distinct values; too many to treat as categories.")
        levels = ds.levels(t.variable)
        if t.ordered_levels is not None and sorted(map(str, t.ordered_levels)) != sorted(map(str, levels)):
            raise TreatmentError(f"{v.display_name}: the category order must list every category once.")
        if t.reference_level is not None and t.reference_level not in levels:
            raise TreatmentError(f"{v.display_name}: {t.reference_level!r} is not a category.")


def effective(ds: Dataset, t: VariableTreatment) -> VariableMeta:
    v = ds.var(t.variable)
    measure = Measure.CONTINUOUS if t.treatment == "continuous" else Measure(t.treatment)
    changes: dict[str, Any] = {"measure": measure}
    if t.ordered_levels is not None:
        changes["category_order"] = list(t.ordered_levels)
    if t.reference_level is not None:
        changes["reference_level"] = t.reference_level
    return dataclasses.replace(v, **changes)


def is_noop(ds: Dataset, t: VariableTreatment) -> bool:
    v = ds.var(t.variable)
    e = effective(ds, t)
    return (e.measure, e.category_order, e.reference_level) == (v.measure, v.category_order, v.reference_level)


def apply_treatments(ds: Dataset, treatments: list[VariableTreatment] | None) -> Dataset:
    """Dataset view with analysis-specific encodings. Returns `ds` itself when nothing changes.

    The view shares the raw data and every other structure with `ds`; only the treated
    variables' metadata objects are replaced (copies), so the project is never modified.
    """
    active = [t for t in (treatments or []) if not is_noop(ds, t)]
    for t in active:
        check_treatment(ds, t)
    if not active:
        return ds
    if len({t.variable for t in active}) != len(active):
        raise TreatmentError("A variable can have only one treatment per analysis.")
    view = copy.copy(ds)
    view.variables = dict(ds.variables)
    for t in active:
        view.variables[t.variable] = effective(ds, t)
    view.treatments = [(t, t.describe(ds)) for t in active]
    view.changes = []   # a view is read-only; edits must go to the project dataset
    return view


def treatment_records(ds: Dataset) -> list[dict[str, Any]]:
    """Recorded with results (inputs) so reports and saved projects show every override."""
    return [{**t.to_dict(), "description": d} for t, d in getattr(ds, "treatments", [])]


def treatment_notes(ds: Dataset) -> list[str]:
    notes = [f"Analysis-specific treatment — {d}" for _t, d in getattr(ds, "treatments", [])]
    if any(t.treatment == "continuous" for t, _ in getattr(ds, "treatments", [])):
        notes.append(f"{CONTINUOUS_WARNING} {EQUAL_INTERVALS}")
    return notes



def as_treatments(items: list[Any] | None) -> list[VariableTreatment]:
    """Accept VariableTreatment objects or their saved dicts."""
    return [t if isinstance(t, VariableTreatment) else VariableTreatment.from_dict(t) for t in items or []]


def record(result: Any, view: Dataset) -> None:
    """Store the analysis-specific treatments with a result (inputs → saved project, reports)."""
    recs = treatment_records(view)
    if recs:
        result.inputs["treatments"] = recs
        result.inputs["treatment_warning"] = f"{CONTINUOUS_WARNING} {EQUAL_INTERVALS}" if any(
            r["treatment"] == "continuous" for r in recs) else ""
