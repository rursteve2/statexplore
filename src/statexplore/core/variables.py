"""Variable metadata: names, safe aliases, labels, measurement levels, roles."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Measure(str, Enum):
    CONTINUOUS = "continuous"
    ORDINAL = "ordinal"
    NOMINAL = "nominal"


class Storage(str, Enum):
    NUMERIC = "numeric"
    STRING = "string"
    DATETIME = "datetime"


class Role(str, Enum):
    NONE = "none"
    ID = "id"
    TIME = "time"
    CLUSTER = "cluster"
    EXCLUDED = "excluded"


# Fields a user may change through Dataset.set_variable_field (all logged + undoable).
EDITABLE_FIELDS = ("display_name", "measure", "role", "reference_level")


@dataclass
class VariableMeta:
    """Metadata for one column. `name` is the immutable original column name.

    `alias` (v001, v002, ...) is the only identifier ever placed in R code, so
    user-supplied names never reach a formula or a shell.
    """

    name: str
    alias: str
    storage: Storage
    measure: Measure
    display_name: str = ""
    label: str = ""
    # Code -> label. Keys are floats for numeric variables, str for strings.
    value_labels: dict[Any, str] = field(default_factory=dict)
    # SPSS user-missing definitions: discrete values and inclusive (lo, hi) ranges.
    missing_values: list[Any] = field(default_factory=list)
    missing_ranges: list[tuple[float, float]] = field(default_factory=list)
    role: Role = Role.NONE
    reference_level: Any = None
    source_measure: str = ""  # measurement level as declared by the source file, if any

    def __post_init__(self) -> None:
        if not self.display_name:
            self.display_name = self.name

    @property
    def is_categorical(self) -> bool:
        return self.measure in (Measure.NOMINAL, Measure.ORDINAL)

    def has_user_missing(self) -> bool:
        return bool(self.missing_values or self.missing_ranges)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["storage"] = self.storage.value
        d["measure"] = self.measure.value
        d["role"] = self.role.value
        # JSON object keys must be strings; keep the original type alongside.
        d["value_labels"] = [[k, v] for k, v in self.value_labels.items()]
        d["missing_ranges"] = [list(r) for r in self.missing_ranges]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "VariableMeta":
        d = dict(d)
        d["storage"] = Storage(d["storage"])
        d["measure"] = Measure(d["measure"])
        d["role"] = Role(d["role"])
        d["value_labels"] = {k: v for k, v in d["value_labels"]}
        d["missing_ranges"] = [tuple(r) for r in d["missing_ranges"]]
        return cls(**d)


def make_alias(index: int) -> str:
    return f"v{index + 1:03d}"
