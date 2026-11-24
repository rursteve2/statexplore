"""Layout helpers shared by the pages (v1.3.1 UI-32 / UI-33)."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QScrollArea, QWidget

# The whole-section scroll bar: wider, higher contrast, a large drag handle, distinct hover / pressed states.
# Set on the scroll area's OWN vertical bar, so tables and lists inside the section keep their normal bars.
SECTION_SCROLLBAR = """
QScrollBar:vertical { width: 16px; background: #e3e8ef; border-left: 1px solid #c5cfdb; margin: 0; }
QScrollBar::handle:vertical { background: #8796a8; min-height: 48px; border-radius: 5px; margin: 2px 3px; }
QScrollBar::handle:vertical:hover { background: #5d7290; }
QScrollBar::handle:vertical:pressed { background: #2b5d8c; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
"""


def section_scroll(widget: QWidget, min_width: int | None = None) -> QScrollArea:
    """A resizable scroll area for a page's configuration column, with the wide section scroll bar."""
    scroll = QScrollArea()
    scroll.setObjectName("sectionScroll")
    scroll.setWidget(widget)
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    bar = scroll.verticalScrollBar()
    bar.setObjectName("sectionScrollBar")
    bar.setStyleSheet(SECTION_SCROLLBAR)
    bar.setToolTip("Scrolls the whole settings section")
    if min_width:
        scroll.setMinimumWidth(min_width)
    make_responsive(widget)
    return scroll


def fit_combo(combo: QComboBox, min_chars: int = 14) -> QComboBox:
    """A dropdown that shrinks with its panel instead of clipping or pushing the layout (UI-33): the closed box
    shows at least `min_chars` characters, the open list is as wide as its longest entry, and every entry (and the
    box itself) carries its full text as a tooltip."""
    combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(min_chars)

    def sync() -> None:
        fm = combo.view().fontMetrics()
        widest = 0
        for i in range(combo.count()):
            text = combo.itemText(i)
            if not combo.itemData(i, Qt.ToolTipRole):
                combo.setItemData(i, text, Qt.ToolTipRole)
            widest = max(widest, fm.horizontalAdvance(text))
        combo.view().setMinimumWidth(widest + 40)
        combo.setToolTip(combo.currentText())

    sync()
    model = combo.model()
    model.rowsInserted.connect(lambda *_: sync())
    model.modelReset.connect(sync)
    combo.currentIndexChanged.connect(lambda *_: combo.setToolTip(combo.currentText()))
    return combo


def make_responsive(widget: QWidget) -> None:
    """Width-aware settings (UI-33): every dropdown shrinks safely and shows its full text as a tooltip, long
    explanatory labels wrap instead of clipping, and form rows put the field under its label when narrow."""
    from PySide6.QtWidgets import QFormLayout, QLabel

    for combo in widget.findChildren(QComboBox):
        if not combo.property("sx_fit"):
            fit_combo(combo)
            combo.setProperty("sx_fit", True)
    for label in widget.findChildren(QLabel):
        if len(label.text()) > 60 and not label.wordWrap():
            label.setWordWrap(True)
    for form in widget.findChildren(QFormLayout):
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
