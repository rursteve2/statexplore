"""Type-to-search for variable dropdowns (v1.3 P2-01)."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QCompleter


def make_searchable(combo: QComboBox) -> QComboBox:
    """Let the user type part of a variable name or label to find it. The list itself is unchanged: typed text
    never creates an item, and the selection only changes when an existing item is picked; leaving the field
    with unmatched text restores the current item."""
    combo.setEditable(True)
    combo.setInsertPolicy(QComboBox.NoInsert)
    comp = QCompleter(combo.model(), combo)
    comp.setFilterMode(Qt.MatchContains)
    comp.setCaseSensitivity(Qt.CaseInsensitive)
    comp.setCompletionMode(QCompleter.PopupCompletion)
    combo.setCompleter(comp)
    comp.activated[str].connect(lambda text: combo.setCurrentIndex(max(combo.findText(text), 0)))
    line = combo.lineEdit()
    line.setPlaceholderText("type to search…")

    def restore() -> None:
        if combo.findText(line.text()) < 0:
            line.setText(combo.itemText(combo.currentIndex()))

    line.editingFinished.connect(restore)
    combo.setToolTip((combo.toolTip() + "\n" if combo.toolTip() else "") + "Type part of a name or label to search.")
    return combo


def excluded_outcomes_hint(ds, offered: set[str]) -> tuple[str, str]:
    """(short text, tooltip) naming the variables NOT offered as outcomes and why (P2-01: never hide silently)."""
    from ...core.variables import Measure, Role, Storage

    if ds is None:
        return "", ""
    reasons: list[str] = []
    for v in ds.variables.values():
        if v.name in offered or v.role is Role.EXCLUDED:
            continue
        if v.storage is Storage.DATETIME:
            why = "date/time"
        elif v.storage is Storage.STRING and v.measure is Measure.CONTINUOUS:
            why = "text cannot be continuous"
        else:
            k = ds.analysis_column(v.name).dropna().nunique()
            if k <= 1:
                why = "constant or empty in the analysed rows"
            elif v.is_categorical:
                why = (f"{v.measure.value} with {k:,} categories (outcome models allow 2–15; identifiers are not "
                       "outcomes)")
            else:
                why = f"{v.measure.value} {v.storage.value}"
        reasons.append(f"{v.display_name}: {why}")
    if not reasons:
        return "", ""
    text = (f"{len(reasons):,} variable(s) are not offered as outcomes (hover for reasons). Change a level in "
            "Dataset Explorer ▸ Measurement Review if it was read wrongly.")
    tip = "\n".join(reasons[:60]) + (f"\n… and {len(reasons) - 60:,} more" if len(reasons) > 60 else "")
    return text, tip
