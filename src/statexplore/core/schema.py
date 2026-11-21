"""Standardized result schema shared by every analysis (§26).

Every analysis, from descriptives to discovery-search candidates, produces an
AnalysisResult. Fields are stable and JSON-serialisable; family-specific
content lives in `tables` / `values` rather than in new top-level fields.
"""
from __future__ import annotations

import datetime as _dt
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

SCHEMA_VERSION = 1


class Status(str, Enum):
    OK = "ok"                # completed without problems
    WARNING = "warning"      # completed; warnings must be shown with the result
    INVALID = "invalid"      # rejected before fitting (screening), reason in `messages`
    FAILED = "failed"        # engine error
    CANCELLED = "cancelled"


@dataclass
class Table:
    name: str
    columns: list[str]
    rows: list[list[Any]]
    row_names: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class EngineInfo:
    engine: str                       # "R" | "Python"
    version: str                      # e.g. "R version 4.6.2 (...)"
    packages: dict[str, str] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    kind: str                                   # e.g. "describe", "eda_plot"
    title: str
    engine: EngineInfo
    script: str                                 # exact code executed (§18)
    inputs: dict[str, Any]                      # variables (names + aliases), options
    status: Status = Status.OK
    n_used: int | None = None
    missing_data: str = ""                      # missing-data approach applied
    tables: list[Table] = field(default_factory=list)
    values: dict[str, Any] = field(default_factory=dict)
    plots: list[str] = field(default_factory=list)  # artifact keys (PNG files / blobs)
    messages: list[str] = field(default_factory=list)  # warnings / errors / screening reasons
    seed: int | None = None
    elapsed_s: float | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: str = field(default_factory=lambda: _dt.datetime.now().isoformat(timespec="seconds"))
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AnalysisResult":
        d = dict(d)
        d["status"] = Status(d["status"])
        d["engine"] = EngineInfo(**d["engine"])
        d["tables"] = [Table(**t) for t in d["tables"]]
        return cls(**d)
