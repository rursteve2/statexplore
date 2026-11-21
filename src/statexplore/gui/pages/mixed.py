"""Mixed & Longitudinal Models workspace (§15)."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMessageBox, QPushButton, QScrollArea, QSpinBox, QSplitter, QTextBrowser, QVBoxLayout, QWidget,
)

from ...analysis import mixed
from ...core.project import Project
from ...core.structure import ReshapeSpec, suggest_structure, wide_to_long
from ...core.variables import Measure, Role, Storage
from ..result_view import render
from ..state import AppState
from .eda import CodeDialog

ENGINE_TEXT = {
    "lme4": "lme4 / lmerTest (random effects, independent residuals, Satterthwaite df)",
    "nlme": "nlme (random effects + correlated residuals)",
    "mmrm": "mmrm (marginal model — no random effects; unstructured covariance)",
}


class ReshapeDialog(QDialog):
    """Assisted wide-to-long restructuring; produces a new project, the current one is untouched."""

    def __init__(self, state: AppState, parent: QWidget | None = None):
        super().__init__(parent)
        self.state = state
        ds = state.dataset
        self.setWindowTitle("Restructure wide → long")
        self.resize(560, 640)
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.idv = QComboBox()
        self.idv.addItem("(number the rows)", None)
        for v in ds.variables.values():
            self.idv.addItem(v.display_name, v.name)
        self.time_name = QLineEdit("time")
        self.time_values = QLineEdit()
        self.time_values.setPlaceholderText("e.g. 1, 2, 3  (one value per occasion)")
        form.addRow("Participant ID", self.idv)
        form.addRow("New time variable", self.time_name)
        form.addRow("Time values", self.time_values)
        lay.addLayout(form)
        lay.addWidget(QLabel("Check the columns of ONE repeated measure in time order, name it, then 'Add measure'."))
        self.cols = QListWidget()
        for v in ds.variables.values():
            if v.storage is Storage.NUMERIC:
                it = QListWidgetItem(v.display_name)
                it.setData(Qt.UserRole, v.name)
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                it.setCheckState(Qt.Unchecked)
                self.cols.addItem(it)
        lay.addWidget(self.cols, 1)
        row = QHBoxLayout()
        self.measure_name = QLineEdit()
        self.measure_name.setPlaceholderText("name of the long variable")
        add = QPushButton("Add measure")
        add.clicked.connect(self._add)
        row.addWidget(self.measure_name)
        row.addWidget(add)
        lay.addLayout(row)
        self.summary = QLabel("No measures yet.")
        self.summary.setWordWrap(True)
        lay.addWidget(self.summary)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._ok)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)
        self.measures: dict[str, list[str]] = {}
        self.result_dataset = None

    def _add(self) -> None:
        cols = [self.cols.item(i).data(Qt.UserRole) for i in range(self.cols.count())
                if self.cols.item(i).checkState() == Qt.Checked]
        name = self.measure_name.text().strip()
        if not name or len(cols) < 2:
            QMessageBox.information(self, "Restructure", "Give a name and check at least two columns.")
            return
        self.measures[name] = cols
        for i in range(self.cols.count()):
            self.cols.item(i).setCheckState(Qt.Unchecked)
        self.measure_name.clear()
        self.summary.setText("; ".join(f"{k} ← {', '.join(v)}" for k, v in self.measures.items()))

    def _ok(self) -> None:
        raw = [t.strip() for t in self.time_values.text().split(",") if t.strip()]
        try:
            values = [float(t) for t in raw]
        except ValueError:
            values = raw
        try:
            self.result_dataset = wide_to_long(self.state.dataset, ReshapeSpec(
                id_var=self.idv.currentData(), measures=self.measures, time_values=values,
                time_name=self.time_name.text().strip() or "time"))
        except (ValueError, KeyError) as e:
            QMessageBox.warning(self, "Restructure", str(e))
            return
        self.accept()


class MixedPage(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.current = None
        left = QWidget()
        ll = QVBoxLayout(left)

        sbox = QGroupBox("Data structure (suggestions — confirm before use)")
        sl = QVBoxLayout(sbox)
        self.suggestions = QListWidget()
        self.suggestions.setMaximumHeight(130)
        sl.addWidget(self.suggestions)
        srow = QHBoxLayout()
        detect = QPushButton("Detect structure")
        detect.clicked.connect(self._detect)
        apply_btn = QPushButton("Set selected role")
        apply_btn.clicked.connect(self._apply_suggestion)
        reshape = QPushButton("Wide → long…")
        reshape.clicked.connect(self._reshape)
        for b in (detect, apply_btn, reshape):
            srow.addWidget(b)
        sl.addLayout(srow)
        ll.addWidget(sbox)

        mbox = QGroupBox("Model")
        mf = QFormLayout(mbox)
        self.outcome, self.idv, self.time, self.cluster = QComboBox(), QComboBox(), QComboBox(), QComboBox()
        self.time_type = self._choice(mixed.TIME_TYPES)
        self.interactions = self._choice(mixed.INTERACTIONS, "time")
        self.random = self._choice(mixed.RANDOM)
        self.residual = self._choice(mixed.RESIDUAL)
        self.method = QComboBox()
        self.method.addItems(["REML", "ML"])
        self.ntraj = QSpinBox()
        self.ntraj.setRange(0, 500)
        self.ntraj.setValue(40)
        for label, w in [("Outcome (continuous)", self.outcome), ("Participant ID", self.idv), ("Time", self.time),
                         ("Time as", self.time_type), ("Cluster (optional)", self.cluster),
                         ("Interactions", self.interactions), ("Random effects", self.random),
                         ("Residual covariance", self.residual), ("Estimation", self.method),
                         ("Trajectories to draw", self.ntraj)]:
            mf.addRow(label, w)
        ll.addWidget(mbox)
        self.preds = QListWidget()
        self.preds.setMaximumHeight(170)
        pbox = QGroupBox("Fixed-effect predictors (besides time)")
        QVBoxLayout(pbox).addWidget(self.preds)
        ll.addWidget(pbox)
        self.engine_label = QLabel()
        self.engine_label.setWordWrap(True)
        self.engine_label.setObjectName("hint")
        ll.addWidget(self.engine_label)
        for w in (self.residual, self.random):
            w.currentIndexChanged.connect(self._update_engine)
        row = QHBoxLayout()
        self.fit_btn = QPushButton("Fit model")
        self.fit_btn.clicked.connect(self._fit)
        self.code_btn = QPushButton("View Code")
        self.code_btn.setEnabled(False)
        self.code_btn.clicked.connect(lambda: self.current and CodeDialog(self.current, self).exec())
        row.addWidget(self.fit_btn)
        row.addWidget(self.code_btn)
        ll.addLayout(row)
        crow = QHBoxLayout()
        crow.addWidget(QLabel("Compare:"))
        for key, text in (("time", "Time specs"), ("random", "Random effects"), ("residual", "Residual structure")):
            b = QPushButton(text)
            b.setToolTip(mixed.COMPARISONS[key])
            b.clicked.connect(lambda _=False, k=key: self._compare(k))
            crow.addWidget(b)
        ll.addLayout(crow)
        ll.addStretch()
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
        t = QLabel("Mixed & Longitudinal Models")
        t.setObjectName("pageTitle")
        lay.addWidget(t)
        note = QLabel("Data must be in long format (one row per participant per occasion). Random effects model "
                      "differences between participants; residual covariance models correlation between a "
                      "participant's occasions. They are different things.")
        note.setWordWrap(True)
        lay.addWidget(note)
        lay.addWidget(split, 1)
        state.project_changed.connect(self._reload)
        state.metadata_changed.connect(self._reload)
        state.busy_changed.connect(lambda b: self.fit_btn.setEnabled(not b))
        self._reload()

    @staticmethod
    def _choice(options: dict[str, str], default: str | None = None) -> QComboBox:
        c = QComboBox()
        for k, v in options.items():
            c.addItem(v, k)
        if default:
            c.setCurrentIndex(c.findData(default))
        return c

    def _reload(self) -> None:
        ds = self.state.dataset
        keep = {w: w.currentData() for w in (self.outcome, self.idv, self.time, self.cluster)}
        checked = {self.preds.item(i).data(Qt.UserRole) for i in range(self.preds.count())
                   if self.preds.item(i).checkState() == Qt.Checked}
        for w in (self.outcome, self.idv, self.time, self.cluster):
            w.clear()
        self.preds.clear()
        self.time.addItem("(none)", None)
        self.cluster.addItem("(none)", None)
        if ds is None:
            return
        role_default = {}
        for v in ds.variables.values():
            if v.storage is Storage.DATETIME or v.role is Role.EXCLUDED:
                continue
            if v.measure is Measure.CONTINUOUS and v.storage is Storage.NUMERIC:
                self.outcome.addItem(v.display_name, v.name)
            self.idv.addItem(v.display_name, v.name)
            self.cluster.addItem(v.display_name, v.name)
            if v.storage is Storage.NUMERIC or v.is_categorical:
                self.time.addItem(v.display_name, v.name)
            if v.role in (Role.ID, Role.TIME, Role.CLUSTER):
                role_default[v.role] = v.name
            if v.role is Role.NONE and (v.is_categorical or (v.measure is Measure.CONTINUOUS and v.storage is Storage.NUMERIC)):
                it = QListWidgetItem(f"{v.display_name}  ·  {v.measure.value}")
                it.setData(Qt.UserRole, v.name)
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                it.setCheckState(Qt.Checked if v.name in checked else Qt.Unchecked)
                self.preds.addItem(it)
        for w, role in ((self.idv, Role.ID), (self.time, Role.TIME), (self.cluster, Role.CLUSTER)):
            target = keep.get(w) or role_default.get(role)
            w.setCurrentIndex(max(0, w.findData(target)))
        self.outcome.setCurrentIndex(max(0, self.outcome.findData(keep.get(self.outcome))))
        self._update_engine()

    def _spec(self) -> mixed.MixedSpec:
        return mixed.MixedSpec(
            outcome=self.outcome.currentData(), id=self.idv.currentData(), time=self.time.currentData(),
            time_type=self.time_type.currentData(), cluster=self.cluster.currentData(),
            predictors=[self.preds.item(i).data(Qt.UserRole) for i in range(self.preds.count())
                        if self.preds.item(i).checkState() == Qt.Checked],
            interactions=self.interactions.currentData(), random=self.random.currentData(),
            residual=self.residual.currentData(), method=self.method.currentText(), n_trajectories=self.ntraj.value())

    def _update_engine(self) -> None:
        if self.residual.currentData() == "unstructured" and self.random.currentData() != "none":
            self.random.setCurrentIndex(self.random.findData("none"))
            self.time_type.setCurrentIndex(self.time_type.findData("categorical"))
        engine = mixed.MixedSpec(outcome="", id="", residual=self.residual.currentData()).engine
        self.engine_label.setText(f"Engine: {ENGINE_TEXT[engine]}")

    def _validated(self) -> mixed.MixedSpec | None:
        ds = self.state.dataset
        if ds is None:
            return None
        try:
            s = self._spec()
            if not s.outcome or not s.id:
                raise mixed.MixedInputError("Choose an outcome and a participant ID.")
            mixed.validate(ds, s)
        except mixed.MixedInputError as e:
            QMessageBox.information(self, "Mixed models", str(e))
            return None
        return s

    def _fit(self) -> None:
        s = self._validated()
        if s is None:
            return
        ds = self.state.dataset
        self.output.setHtml("<p>Fitting in R…</p>")
        self.state.run_r(lambda sess: mixed.run(sess, ds, s), self._done,
                         lambda m: self.output.setHtml(f"<p style='color:#c62828'>{m}</p>"))

    def _compare(self, which: str) -> None:
        s = self._validated()
        if s is None:
            return
        ds = self.state.dataset
        try:
            mixed.comparison_variants(s, which)
        except mixed.MixedInputError as e:
            QMessageBox.information(self, "Mixed models", str(e))
            return
        self.output.setHtml("<p>Fitting the alternative specifications in R…</p>")
        self.state.run_r(lambda sess: mixed.compare_specs(sess, ds, s, which), self._done,
                         lambda m: self.output.setHtml(f"<p style='color:#c62828'>{m}</p>"))

    def _done(self, out) -> None:
        result, artifacts = out
        self.state.add_result(result, artifacts)
        self.current = result
        self.code_btn.setEnabled(True)
        render(self.output, result, artifacts, plots_first=False)

    def _detect(self) -> None:
        ds = self.state.dataset
        if ds is None:
            return
        self.suggestions.clear()
        for sgg in suggest_structure(ds)[:12]:
            it = QListWidgetItem(f"{sgg.role.upper()}: {ds.var(sgg.variable).display_name} — {sgg.reason}")
            it.setData(Qt.UserRole, (sgg.variable, sgg.role))
            self.suggestions.addItem(it)
        if not self.suggestions.count():
            self.suggestions.addItem("No repeated-measures structure detected (is the data in long format?).")

    def _apply_suggestion(self) -> None:
        it = self.suggestions.currentItem()
        ds = self.state.dataset
        if it is None or ds is None or not it.data(Qt.UserRole):
            return
        name, role = it.data(Qt.UserRole)
        ds.set_variable_field(name, "role", role)
        self.state.notify_metadata_changed()

    def _reshape(self) -> None:
        if self.state.dataset is None:
            return
        dlg = ReshapeDialog(self.state, self)
        if dlg.exec() == QDialog.Accepted and dlg.result_dataset is not None:
            if self.state.dirty and QMessageBox.question(
                    self, "Restructure", "This opens the long dataset as a new, unsaved project. Continue? "
                    "(Save the current project first if needed.)") != QMessageBox.Yes:
                return
            self.state.set_project(Project(dlg.result_dataset))
            self.state.dirty = True
