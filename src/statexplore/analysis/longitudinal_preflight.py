"""Longitudinal Data Preflight (v1.1 §12): what the data can support before a longitudinal model is fitted.

Pure pandas on the analysed rows (filters and holdout applied), so it runs instantly and can be
shown before anything is sent to R. Structurally unobserved occasions are reported as missing —
never filled with zeros or imputed — and the actual analysis range is stated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..core.dataset import Dataset
from ..core.schema import Table


@dataclass
class PreflightReport:
    outcome: str
    id: str
    time: str | None
    n_records: int
    n_participants: int
    time_points: list[Any]
    by_time: pd.DataFrame
    n_one_obs: int
    n_multi_obs: int
    n_no_obs: int
    duplicates: int
    duplicate_examples: list[tuple[Any, Any]]
    unobserved_times: list[Any]
    analysis_range: tuple[Any, Any] | None
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # ---- v1.3 (P1-02 / P1-03) ----
    observed_times: list[Any] = field(default_factory=list)   # time points with at least one observed outcome
    interior_absent: list[Any] = field(default_factory=list)  # absent for everyone but inside the observed range
    n_usable: int = 0                 # participants with >= 1 row complete on the model variables
    n_usable_multi: int = 0           # ... with >= 2 such rows
    n_usable_one: int = 0
    n_intermittent: int = 0           # participants with a gap inside their own observed span (not global gaps)

    def observed_text(self, ds: Dataset) -> str:
        if not self.time:
            return "—"
        lab = lambda c: ds.level_label(self.time, c)  # noqa: E731
        return ", ".join(lab(c) for c in self.observed_times) or "none"

    def tables(self, ds: Dataset) -> list[Table]:
        t = self.by_time
        rows = t.reset_index().to_numpy().tolist()
        summary = Table(
            name="Longitudinal data preflight",
            columns=["outcome", "participant ID", "time", "records", "participants", "time points",
                     "participants with 1 observation", "participants with 2+ observations",
                     "participants with no observation", "participants with usable rows",
                     "usable: 2+ rows", "usable: 1 row", "intermittent gaps", "duplicate participant-time rows",
                     "analysis range (min to max)", "observed time points", "absent inside range"],
            rows=[[ds.var(self.outcome).display_name, ds.var(self.id).display_name,
                   ds.var(self.time).display_name if self.time else "—", self.n_records, self.n_participants,
                   len(self.time_points), self.n_one_obs, self.n_multi_obs, self.n_no_obs, self.n_usable,
                   self.n_usable_multi, self.n_usable_one, self.n_intermittent, self.duplicates,
                   self._range_text(ds), self.observed_text(ds),
                   ", ".join(ds.level_label(self.time, c) for c in self.interior_absent) if self.time else "—"]],
            notes="Usable = complete on the outcome, time and predictors (the rows a model can use). Intermittent = "
                  "a participant missing an occasion inside their own observed span that others did observe; "
                  "occasions absent for everyone are listed separately.")
        return [summary, Table(name="Outcome availability by time point", columns=["time"] + list(t.columns),
                               rows=rows, notes="Observed = non-missing outcome. A time point with 0 observed "
                                                "values is structurally unobserved for this outcome.")]

    def _range_text(self, ds: Dataset) -> str:
        if self.analysis_range is None:
            return "—"
        lo, hi = self.analysis_range
        lab = (lambda c: ds.level_label(self.time, c)) if self.time else str
        return f"{lab(lo)} to {lab(hi)}"


def _label(ds: Dataset, name: str, code: Any) -> str:
    return ds.level_label(name, code)


def preflight(ds: Dataset, outcome: str, id_var: str, time: str | None = None,
              predictors: list[str] | None = None, spec: Any = None) -> PreflightReport:
    """Availability, missingness and structure checks for (outcome, participant, time)."""
    predictors = [p for p in (predictors or []) if p not in (outcome, id_var, time)]
    cols = list(dict.fromkeys([outcome, id_var] + ([time] if time else []) + predictors))
    f = ds.analysis_frame(cols)
    f = f[f[id_var].notna()]
    obs = f[outcome].notna()
    complete = f[[outcome] + predictors + ([time] if time else [])].notna().all(axis=1)
    per_id = obs.groupby(f[id_var]).sum()
    warnings: list[str] = []
    notes: list[str] = []
    if time:
        tcol = f[time]
        times = sorted(tcol.dropna().unique().tolist(), key=lambda x: (isinstance(x, str), x))
        g = pd.DataFrame({"t": tcol, "obs": obs, "complete": complete})
        by = g.groupby("t", sort=False).agg(records=("obs", "size"), observed=("obs", "sum"),
                                            complete_for_model=("complete", "sum"))
        by = by.reindex(times)
        by["missing"] = by["records"] - by["observed"]
        by["percent_missing"] = 100 * by["missing"] / by["records"]
        by = by[["records", "observed", "missing", "percent_missing", "complete_for_model"]]
        by.index = [_label(ds, time, c) for c in times]
        unobserved = [c for c, n in zip(times, by["observed"]) if n == 0]
        seen = [c for c, n in zip(times, by["observed"]) if n > 0]
        rng = (seen[0], seen[-1]) if seen else None
        dup_mask = f.duplicated([id_var, time], keep=False) & tcol.notna()
        dups = int(f[dup_mask].groupby([id_var, time]).ngroups) if dup_mask.any() else 0
        examples = list(f.loc[dup_mask, [id_var, time]].drop_duplicates().head(5).itertuples(index=False, name=None))
    else:
        times, unobserved, rng, dups, examples = [], [], None, 0, []
        by = pd.DataFrame({"records": [len(f)], "observed": [int(obs.sum())], "missing": [int((~obs).sum())],
                           "percent_missing": [100 * float((~obs).mean()) if len(f) else 0.0],
                           "complete_for_model": [int(complete.sum())]}, index=["all rows"])
    dn = ds.var(outcome).display_name
    if unobserved and time:
        lab = lambda c: _label(ds, time, c)  # noqa: E731
        warnings.append(
            f"{dn} is not observed at {len(unobserved)} of {len(times)} time points ({_runs(unobserved, lab)}). "
            f"These occasions are left out — not treated as zeros and not imputed. Actual analysis range: "
            f"{lab(rng[0])} to {lab(rng[1])}." if rng else f"{dn} is never observed.")
    if dups:
        warnings.append(f"{dups} participant-time combinations occur in more than one row (e.g. "
                        + ", ".join(f"{a} at {_label(ds, time, b)}" for a, b in examples[:3])
                        + "). Repeated-measures structures need one row per participant per occasion.")
    n_one, n_multi, n_none = int((per_id == 1).sum()), int((per_id >= 2).sum()), int((per_id == 0).sum())
    per_usable = complete.groupby(f[id_var]).sum()
    n_usable, n_usable_multi = int((per_usable >= 1).sum()), int((per_usable >= 2).sum())
    seen_times = [c for c in times if c not in unobserved] if time else []
    interior = [c for c in unobserved if rng and rng[0] < c < rng[1]] if time else []
    n_inter = 0
    if time and seen_times:
        pos = {c: i for i, c in enumerate(seen_times)}
        idx = f.loc[complete, time].map(pos)
        span = idx.groupby(f.loc[complete, id_var]).agg(["min", "max", "nunique"])
        n_inter = int(((span["max"] - span["min"] + 1) > span["nunique"]).sum())
    if interior and time:
        lab = lambda c: _label(ds, time, c)  # noqa: E731
        warnings.append(
            f"Observed time points: {', '.join(lab(c) for c in seen_times)}. {', '.join(lab(c) for c in interior)} "
            f"lies inside the analysis range but has no observed {dn} for anyone. An observed-occasion index (AR(1) "
            "over occasions, GEE AR(1), occasion lags) treats the occasions on either side as adjacent (gap "
            "compression); continuous-time AR(1) uses the actual time differences. Predictions at absent time "
            "points are model-based, not observed data.")
    if n_one:
        notes.append(f"{n_one} participants contribute a single observation: they inform the averages but not "
                     "within-person change.")
    if spec is not None and time:
        w, n = _model_checks(ds, f, complete, outcome, id_var, time, spec)
        warnings += w
        notes += n
    if n_inter:
        notes.append(f"{n_inter} participants have intermittent gaps (an occasion missing inside their own observed "
                     "span); they are retained with their available rows.")
    notes.append("Incomplete histories are retained (no complete-panel requirement, no imputation). Estimates are "
                 "unbiased only if the missingness is ignorable given the model — missing at random given the "
                 "observed outcomes and covariates in the model, with a correctly specified model; MAR itself cannot "
                 "be verified from the data.")
    return PreflightReport(outcome, id_var, time, len(f), int(f[id_var].nunique()), times, by, n_one, n_multi,
                           n_none, dups, examples, unobserved, rng, warnings, notes, seen_times, interior,
                           n_usable, n_usable_multi, n_usable - n_usable_multi, n_inter)


def _runs(codes: list[Any], lab) -> str:
    """'Wave 1 to Wave 6' style summary of consecutive codes."""
    if all(isinstance(c, (int, float)) for c in codes):
        runs, start, prev = [], codes[0], codes[0]
        for c in codes[1:]:
            if c != prev + 1:
                runs.append((start, prev))
                start = c
            prev = c
        runs.append((start, prev))
        return ", ".join(lab(a) if a == b else f"{lab(a)} to {lab(b)}" for a, b in runs)
    return ", ".join(lab(c) for c in codes)


def _model_checks(ds: Dataset, f: pd.DataFrame, complete: pd.Series, outcome: str, id_var: str, time: str,
                  spec: Any) -> tuple[list[str], list[str]]:
    """(warnings, notes): warnings are data problems for this model; notes are information."""
    out: list[str] = []
    notes: list[str] = []
    d = f[complete]
    if d.empty:
        return ["No rows are complete on the outcome, time and predictors."], []
    n_times = d.groupby(id_var)[time].nunique()
    if getattr(spec, "random", "") in ("slope", "slope_uncorrelated") and getattr(spec, "estimator", "mixed") != "gee":
        k = int((n_times >= 2).sum())
        msg = (f"Random time slopes: {k} of {len(n_times)} participants have 2+ distinct time points "
               f"({int((n_times >= 3).sum())} have 3+).")
        (out if k < 10 else notes).append(msg + (" Too few to estimate slope variation reliably." if k < 10 else ""))
    if getattr(spec, "residual", "") in ("ar1", "car1") or getattr(spec, "gee_corstr", "") == "ar1" \
            and getattr(spec, "estimator", "") == "gee":
        codes = pd.to_numeric(d[time], errors="coerce")
        if codes.notna().all():
            gaps = (d.assign(_t=codes).sort_values([id_var, "_t"]).groupby(id_var)["_t"].diff().dropna())
            ug = sorted(set(np.round(gaps.to_numpy(), 8)))
            if len(ug) > 1:
                notes.append(f"Unequal spacing: within-participant gaps between observed occasions take {len(ug)} "
                           f"different values ({', '.join(f'{x:g}' for x in ug[:6])}). Discrete-time AR(1) treats "
                           "consecutive observed occasions as equally spaced; continuous-time AR(1) uses the actual "
                           "time values.")
    v = ds.var(outcome)
    if getattr(spec, "family", "gaussian") == "gaussian" and v.is_categorical:
        out.append(f"{v.display_name} is {v.measure.value}: a Gaussian residual structure is only possible with "
                   "an explicit 'treat as continuous' choice.")
    return out, notes
