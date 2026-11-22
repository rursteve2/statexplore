"""Teaching Studio (§16): interaction plots with controls and brief interpretations."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPushButton, QScrollArea, QSplitter, QTextBrowser, QVBoxLayout, QWidget,
)

from ...analysis import teaching
from ...analysis.models import ModelSpec
from ..model_editor import ModelEditor
from ..result_view import render
from ..state import AppState
from .eda import CodeDialog


class TeachingPage(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.current = None
        self.flexplot_available: bool | None = None

        left = QWidget()
        ll = QVBoxLayout(left)
        src = QHBoxLayout()
        src.addWidget(QLabel("Load model:"))
        self.saved = QComboBox()
        self.saved.activated.connect(self._load_saved)
        src.addWidget(self.saved, 1)
        ll.addLayout(src)
        self.editor = ModelEditor()
        self.editor.changed.connect(self._model_changed)
        ll.addWidget(self.editor, 3)

        pbox = QGroupBox("Graph")
        pf = QFormLayout(pbox)
        self.x = QComboBox()
        self.group = QComboBox()
        self.panel = QComboBox()
        self.modvals = QComboBox()
        self.modvals.addItem("Mean and ±1 SD", "sd")
        self.modvals.addItem("10th, 50th, 90th percentiles", "quantiles")
        self.ci = QCheckBox("Show confidence bands")
        self.ci.setChecked(True)
        self.data_pts = QCheckBox("Show raw data")
        self.backend = QComboBox()
        for k, v in teaching.BACKENDS.items():
            self.backend.addItem(v, k)
        self.backend.currentIndexChanged.connect(self._backend_changed)
        self.fmt = QComboBox()
        for f in teaching.FORMATS:
            self.fmt.addItem(f.upper(), f)
        self.title = QLineEdit()
        self.title.setPlaceholderText("(default)")
        self.xlab = QLineEdit()
        self.xlab.setPlaceholderText("(variable name)")
        self.ylab = QLineEdit()
        self.ylab.setPlaceholderText("(outcome name)")
        for label, w in [("X axis", self.x), ("Lines (group)", self.group), ("Panels", self.panel),
                         ("Continuous moderator values", self.modvals), ("", self.ci), ("", self.data_pts),
                         ("Plotting backend", self.backend), ("Export format", self.fmt), ("Title", self.title),
                         ("X label", self.xlab), ("Y label", self.ylab)]:
            pf.addRow(label, w)
        ll.addWidget(pbox)
        self.held_box = QGroupBox("Hold the other predictors at")
        self.held_form = QFormLayout(self.held_box)
        self.held_widgets: dict[str, QComboBox] = {}
        ll.addWidget(self.held_box)
        for w in (self.x, self.group, self.panel):
            w.currentIndexChanged.connect(self._rebuild_held)
        row = QHBoxLayout()
        self.draw_btn = QPushButton("Draw")
        self.draw_btn.clicked.connect(self._draw)
        self.code_btn = QPushButton("View Code")
        self.code_btn.setEnabled(False)
        self.code_btn.clicked.connect(lambda: self.current and CodeDialog(self.current[0], self).exec())
        self.save_btn = QPushButton("Save graph…")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save_graph)
        self.add_btn = QPushButton("Add to comparison set")
        self.add_btn.clicked.connect(self._add_to_set)
        for b in (self.draw_btn, self.code_btn, self.save_btn, self.add_btn):
            row.addWidget(b)
        ll.addLayout(row)
        self.finder_btn = QPushButton("Find teaching examples in this project…")
        self.finder_btn.setToolTip("Lists completed analyses that illustrate common teaching points (significant "
                                   "interactions, adjustment, instability, assumption problems, model disagreement)")
        self.finder_btn.clicked.connect(self.show_examples)
        ll.addWidget(self.finder_btn)
        scroll = QScrollArea()
        scroll.setWidget(left)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(440)

        self.output = QTextBrowser()
        split = QSplitter()
        split.addWidget(scroll)
        split.addWidget(self.output)
        split.setSizes([460, 900])
        lay = QVBoxLayout(self)
        t = QLabel("Teaching Studio")
        t.setObjectName("pageTitle")
        lay.addWidget(t)
        note = QLabel("Lines are model predictions: the x variable varies, lines and panels show chosen values, "
                      "and all other predictors are held at the values printed under the graph. "
                      "Significance comes from the statistical test, not from how the lines look.")
        note.setWordWrap(True)
        lay.addWidget(note)
        lay.addWidget(split, 1)

        state.project_changed.connect(self._reload)
        state.metadata_changed.connect(self._reload)
        state.models_changed.connect(self._refresh_saved)
        state.show_in_teaching.connect(self.show_model)
        state.busy_changed.connect(lambda b: self.draw_btn.setEnabled(not b))
        self._reload()

    def _reload(self) -> None:
        keep = None
        try:
            keep = self.editor.model()
        except ValueError:
            pass
        self.editor.set_dataset(self.state.dataset)
        if keep is not None and self.state.dataset is not None and all(
                v in self.state.dataset.variables for v in [keep.outcome] + keep.variables):
            self.editor.set_model(keep)
        self._refresh_saved()
        self._model_changed()

    def _refresh_saved(self) -> None:
        self.saved.clear()
        self.saved.addItem("— from the comparison set —", None)
        p = self.state.project
        if p is None:
            return
        for m in p.models:
            self.saved.addItem(f"{m.label or m.id}: {m.display(p.dataset)}", m.id)

    def _load_saved(self, index: int) -> None:
        mid = self.saved.itemData(index)
        p = self.state.project
        m = next((m for m in p.models if m.id == mid), None) if p and mid else None
        if m:
            self.show_model(m)

    def show_model(self, m: ModelSpec) -> None:
        self.editor.set_model(m)
        self._model_changed()
        if m.focal:  # x = first focal variable, lines = second, panels = third
            self.x.setCurrentIndex(max(0, self.x.findData(m.focal[0])))
            self.group.setCurrentIndex(max(0, self.group.findData(m.focal[1])))
            if len(m.focal) > 2:
                self.panel.setCurrentIndex(max(0, self.panel.findData(m.focal[2])))

    def _model_changed(self) -> None:
        ds = self.state.dataset
        vs = self.editor.chosen() if ds else []
        for combo, none in ((self.x, False), (self.group, True), (self.panel, True)):
            keep = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            if none:
                combo.addItem("(none)", None)
            for v in vs:
                combo.addItem(ds.var(v).display_name, v)
            combo.setCurrentIndex(max(0, combo.findData(keep)))
            combo.blockSignals(False)
        if self.group.currentData() is None and len(vs) > 1:
            self.group.setCurrentIndex(2 if self.x.currentIndex() == 0 else 1)
        self._rebuild_held()

    def _backend_changed(self) -> None:
        """flexplot is optional (GitHub-only): check once, and disable it with a reason if missing."""
        if self.backend.currentData() != "flexplot" or self.flexplot_available:
            return
        if self.flexplot_available is False:
            self.backend.setCurrentIndex(0)
            return

        def checked(ok) -> None:
            self.flexplot_available = bool(ok)
            if not ok:
                idx = self.backend.findData("flexplot")
                item = self.backend.model().item(idx)
                item.setEnabled(False)
                item.setToolTip("flexplot is not installed. It is distributed on GitHub only; see README.")
                self.backend.setCurrentIndex(0)
                QMessageBox.information(self, "flexplot", "The flexplot package is not installed, so this backend "
                                        "is disabled. ggplot2 and base R are available. See the README for how "
                                        "to install flexplot.")

        self.state.run_r(lambda s: s.worker.run("sx_result <- requireNamespace('flexplot', quietly = TRUE)").result,
                         checked, lambda m: checked(False))

    def _rebuild_held(self) -> None:
        """One control per non-plotted predictor: default = mean / reference level (blank)."""
        ds = self.state.dataset
        keep = {n: w.currentText() for n, w in self.held_widgets.items()}
        while self.held_form.rowCount():
            self.held_form.removeRow(0)
        self.held_widgets = {}
        if ds is None:
            return
        shown = {self.x.currentData(), self.group.currentData(), self.panel.currentData()}
        for n in self.editor.chosen():
            if n in shown:
                continue
            v = ds.var(n)
            c = QComboBox()
            c.setEditable(not v.is_categorical)
            if v.is_categorical:
                c.addItem("(reference level)")
                c.addItems([ds.level_label(n, code) for code in ds.levels(n)])
            else:
                c.addItem("(mean)")
                c.setToolTip("Type a number to hold this predictor at that value.")
            if keep.get(n):
                c.setCurrentText(keep[n])
            self.held_widgets[n] = c
            self.held_form.addRow(v.display_name, c)
        self.held_box.setVisible(bool(self.held_widgets))

    def _held_values(self) -> dict:
        return {n: w.currentText() for n, w in self.held_widgets.items()
                if w.currentText() and not w.currentText().startswith("(")}

    def _plotspec(self) -> teaching.PlotSpec:
        return teaching.PlotSpec(
            held=self._held_values(),
            x=self.x.currentData(), group=self.group.currentData(), panel=self.panel.currentData(),
            ci=self.ci.isChecked(), show_data=self.data_pts.isChecked(), backend=self.backend.currentData(),
            fmt=self.fmt.currentData(), title=self.title.text().strip(), xlab=self.xlab.text().strip(),
            ylab=self.ylab.text().strip(), moderator_values=self.modvals.currentData())

    def _draw(self) -> None:
        ds = self.state.dataset
        if ds is None:
            return
        try:
            m = self.editor.model()
            p = self._plotspec()
            if not p.x:
                raise ValueError("Choose the x-axis variable.")
            teaching.validate(ds, m, p)
        except (ValueError, teaching.TeachingInputError) as e:
            QMessageBox.information(self, "Teaching Studio", str(e))
            return
        self.output.setHtml("<p>Drawing in R…</p>")
        self.state.run_r(lambda s: teaching.run(s, ds, m, p), self._done,
                         lambda msg: (self.output.setHtml(f"<p style='color:#c62828'>{msg}</p>")))

    def _done(self, out) -> None:
        result, artifacts = out
        self.state.add_result(result, artifacts)
        self.current = (result, artifacts)
        self.code_btn.setEnabled(True)
        self.save_btn.setEnabled(bool(result.plots))
        render(self.output, result, artifacts)

    def _save_graph(self) -> None:
        if not self.current:
            return
        result, artifacts = self.current
        key = result.values.get("export") or result.plots[0]
        ext = Path(key).suffix
        path, _ = QFileDialog.getSaveFileName(self, "Save graph", f"interaction{ext}", f"*{ext}")
        if path:
            Path(path).write_bytes(artifacts[key])

    def _add_to_set(self) -> None:
        try:
            m = self.editor.model()
        except ValueError as e:
            QMessageBox.information(self, "Teaching Studio", str(e))
            return
        self.state.add_model(m)
        QMessageBox.information(self, "Teaching Studio", "Added to the Model Comparison set.")

    def show_examples(self) -> list:
        """Teaching Example Finder (§17.5): only completed results in the project are listed."""
        from PySide6.QtWidgets import QDialog, QDialogButtonBox, QHeaderView, QTableWidget, QTableWidgetItem

        from ...analysis.teaching_finder import find

        if self.state.project is None:
            return []
        examples = find(self.state.project)
        dlg = QDialog(self)
        dlg.setWindowTitle("Teaching examples found in this project")
        dlg.resize(1000, 520)
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel(f"{len(examples)} example(s) from analyses and searches already run. Nothing here is "
                             "fitted or invented: run more analyses to find more."))
        tab = QTableWidget(len(examples), 4)
        tab.setHorizontalHeaderLabels(["Teaching point", "Analysis", "Detail", "Source"])
        for i, e in enumerate(examples):
            for j, v in enumerate((e.category, e.title, e.detail, e.source)):
                tab.setItem(i, j, QTableWidgetItem(v))
        tab.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        tab.horizontalHeader().setStretchLastSection(True)
        tab.resizeColumnsToContents()
        lay.addWidget(tab)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        self._examples_dialog = dlg
        dlg.show()
        return examples
