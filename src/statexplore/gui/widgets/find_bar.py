"""Ctrl+F search for long code listings and reports (v1.3 P2-02)."""
from __future__ import annotations

from PySide6.QtGui import QKeySequence, QShortcut, QTextDocument
from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton, QTextEdit, QWidget


class FindBar(QWidget):
    """Hidden until Ctrl+F (scoped to `host`); Enter / F3 = next, Shift+Enter / Shift+F3 = previous, Esc = close."""

    def __init__(self, editor: QPlainTextEdit | QTextEdit, host: QWidget):
        super().__init__(host)
        self.editor = editor
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.addWidget(QLabel("Find:"))
        self.edit = QLineEdit()
        self.edit.setPlaceholderText("text in this listing (Enter = next, Shift+Enter = previous)")
        self.edit.setClearButtonEnabled(True)
        lay.addWidget(self.edit, 1)
        prev, nxt, close = QPushButton("Previous"), QPushButton("Next"), QPushButton("×")
        close.setFixedWidth(28)
        lay.addWidget(prev)
        lay.addWidget(nxt)
        self.status = QLabel("")
        lay.addWidget(self.status)
        lay.addWidget(close)
        self.edit.returnPressed.connect(self.find_next)
        self.edit.textChanged.connect(lambda: self.find(from_start=True))
        nxt.clicked.connect(self.find_next)
        prev.clicked.connect(self.find_prev)
        close.clicked.connect(self.hide)
        for seq, fn in ((QKeySequence.Find, self.open), (QKeySequence.FindNext, self.find_next),
                        (QKeySequence.FindPrevious, self.find_prev)):
            sc = QShortcut(seq, host)
            sc.activated.connect(fn)
        esc = QShortcut(QKeySequence("Esc"), self.edit)
        esc.activated.connect(self.hide)
        shift_enter = QShortcut(QKeySequence("Shift+Return"), self.edit)
        shift_enter.activated.connect(self.find_prev)
        self.hide()

    def open(self) -> None:
        self.show()
        sel = self.editor.textCursor().selectedText()
        if sel and " " not in sel:
            self.edit.setText(sel)
        self.edit.setFocus()
        self.edit.selectAll()

    def find_next(self) -> None:
        self.find()

    def find_prev(self) -> None:
        self.find(backward=True)

    def find(self, backward: bool = False, from_start: bool = False) -> bool:
        text = self.edit.text()
        if not text:
            self.status.setText("")
            return False
        flags = QTextDocument.FindBackward if backward else QTextDocument.FindFlag(0)
        if from_start:
            cur = self.editor.textCursor()
            cur.movePosition(cur.MoveOperation.Start)
            self.editor.setTextCursor(cur)
        found = self.editor.find(text, flags)
        if not found:   # wrap around
            cur = self.editor.textCursor()
            cur.movePosition(cur.MoveOperation.End if backward else cur.MoveOperation.Start)
            self.editor.setTextCursor(cur)
            found = self.editor.find(text, flags)
        self.status.setText("" if found else "not found")
        return bool(found)
