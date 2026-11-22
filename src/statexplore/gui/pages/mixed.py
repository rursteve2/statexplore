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
from ...analysis.measurement import TreatmentError, apply_treatments
from ..result_view import render
from ..state import AppState
from ..widgets.treatment_panel import TreatmentPanel
from ..widgets.variable_selector import VariableSelector
from .eda import CodeDialog

ENGINE_TEXT = {
    "lme4": "lme4 / lmerTest (random effects, independent residuals)",
    "nlme": "nlme (random effects + correlated residuals)",
    "mmrm": "mmrm (marginal model — no random effects; unstructured covariance)",
    "glmer": "lme4::glmer (binary logistic mixed model; Laplace or adaptive quadrature)",
    "clmm": "ordinal::clmm (cumulative-logit mixed model)",
    "mblogit": "mclogit::mblogit (multinomial mixed model, PQL approximation — see the limitations in the results)",
    "geepack": "geepack::geeglm (GEE: population-average effects, robust SEs; no random effects)",
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
        self.family = self._choice(mixed.FAMILIES)
        self.estimator = self._choice(mixed.ESTIMATORS)
        self.inference = self._choice(mixed.INFERENCE)
        self.gee_corstr = self._choice(mixed.GEE_CORSTR)
        self.event_combo, self.reference_combo = QComboBox(), QComboBox()
        self.nagq = QSpinBox()
        self.nagq.setRange(1, 25)
        self.nagq.setToolTip("1 = Laplace approximation (default). More points = adaptive Gauss-Hermite quadrature; "
                             "only with a single random intercept.")
        self.family_hint = QLabel()
        self.family_hint.setWordWrap(True)
        self.family_hint.setObjectName("hint")
        self.use_family = QPushButton("Use suggested family")
        self.use_family.clicked.connect(self._use_family)
        fam_box = QWidget()
        fam_row = QHBoxLayout(fam_box)
        fam_row.setContentsMargins(0, 0, 0, 0)
        fam_row.addWidget(self.family_hint, 1)
        fam_row.addWidget(self.use_family)
        for label, w in [("Outcome", self.outcome), ("Outcome family", self.family), ("", fam_box),
                         ("Estimator", self.estimator), ("Participant ID", self.idv), ("Time", self.time),
                         ("Time as", self.time_type), ("Cluster (optional)", self.cluster),
                         ("Interactions", self.interactions), ("Random effects", self.random),
                         ("Residual covariance", self.residual), ("Estimation", self.method),
                         ("Inference (Gaussian lme4)", self.inference), ("Event category (binary)", self.event_combo),
                         ("Reference category (multinomial)", self.reference_combo), ("Quadrature points (nAGQ)", self.nagq),
                         ("GEE working correlation", self.gee_corstr), ("Trajectories to draw", self.ntraj)]:
            if label:
                mf.addRow(label, w)
            else:
                mf.addRow(w)
        ll.addWidget(mbox)
        self.pred_sel = VariableSelector(max_height=230)
        self.preds = self.pred_sel.list
        self.pred_sel.changed.connect(self._update_treatments)
        pbox = QGroupBox("Fixed-effect predictors (besides time)")
        QVBoxLayout(pbox).addWidget(self.pred_sel)
        ll.addWidget(pbox)
        self.treatment_panel = TreatmentPanel()
        self.treatment_panel.changed.connect(self._update_engine)
        ll.addWidget(self.treatment_panel)
        pre_box = QGroupBox("Longitudinal data preflight (before fitting)")
        pl = QVBoxLayout(pre_box)
        self.preflight_view = QTextBrowser()
        self.preflight_view.setMinimumHeight(180)
        pl.addWidget(self.preflight_view)
        pre_btn = QPushButton("Review missing observations by time point")
        pre_btn.clicked.connect(self._preflight)
        pl.addWidget(pre_btn)
        ll.addWidget(pre_box)
        self.engine_label = QLabel()
        self.engine_label.setWordWrap(True)
        self.engine_label.setObjectName("hint")
        ll.addWidget(self.engine_label)
        for w in (self.residual, self.random, self.family, self.estimator, self.inference):
            w.currentIndexChanged.connect(self._update_engine)
        self.outcome.currentIndexChanged.connect(self._outcome_changed)
        for w in (self.idv, self.time):
            w.currentIndexChanged.connect(self._preflight)
        row = QHBoxLayout()
        self.fit_btn = QPushButton("Fit model")
        self.fit_btn.clicked.connect(self._fit)
        self.code_btn = QPushButton("View Code")
        self.code_btn.setEnabled(False)
        self.code_btn.clicked.connect(lambda: self.current and CodeDialog(self.current, self).exec())
        self.copy_btn = QPushButton("Copy tables")
        self.copy_btn.clicked.connect(self._copy_tables)
        row.addWidget(self.fit_btn)
        row.addWidget(self.code_btn)
        row.addWidget(self.copy_btn)
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
        for w in (self.outcome, self.idv, self.time, self.cluster):
            w.blockSignals(True)
            w.clear()
        self.time.addItem("(none)", None)
        self.cluster.addItem("(none)", None)
        preds: list[str] = []
        role_default = {}
        if ds is not None:
            for v in ds.variables.values():
                if v.storage is Storage.DATETIME or v.role is Role.EXCLUDED:
                    continue
                k = ds.analysis_column(v.name).dropna().nunique() if v.is_categorical else 0
                if (v.measure is Measure.CONTINUOUS and v.storage is Storage.NUMERIC) or (v.is_categorical and 2 <= k <= 15):
                    self.outcome.addItem(f"{v.display_name}  ·  {v.measure.value}", v.name)
                self.idv.addItem(v.display_name, v.name)
                self.cluster.addItem(v.display_name, v.name)
                if v.storage is Storage.NUMERIC or v.is_categorical:
                    self.time.addItem(v.display_name, v.name)
                if v.role in (Role.ID, Role.TIME, Role.CLUSTER):
                    role_default[v.role] = v.name
                if v.role is Role.NONE and (v.is_categorical or (v.measure is Measure.CONTINUOUS and v.storage is Storage.NUMERIC)):
                    preds.append(v.name)
            for w, role in ((self.idv, Role.ID), (self.time, Role.TIME), (self.cluster, Role.CLUSTER)):
                target = keep.get(w) or role_default.get(role)
                w.setCurrentIndex(max(0, w.findData(target)))
            self.outcome.setCurrentIndex(max(0, self.outcome.findData(keep.get(self.outcome))))
        for w in (self.outcome, self.idv, self.time, self.cluster):
            w.blockSignals(False)
        self.pred_sel.set_variables(ds, preds, decorate=(lambda n: f"  ·  {ds.var(n).measure.value}") if ds else None)
        self._outcome_changed()

    def _outcome_changed(self, *_args) -> None:
        ds, out = self.state.dataset, self.outcome.currentData()
        for combo, first in ((self.event_combo, "(second category)"), (self.reference_combo, "(first category)")):
            combo.clear()
            combo.addItem(first, None)
            if ds is not None and out:
                for code in ds.levels(out)[:15]:
                    c = str(int(code)) if isinstance(code, float) and code.is_integer() else str(code)
                    combo.addItem(f"{c} = {ds.level_label(out, code)}", code)
        self._update_treatments()

    def _update_treatments(self) -> None:
        out = self.outcome.currentData()
        names = ([out] if out else []) + self.pred_sel.selected()
        self.treatment_panel.set_variables(self.state.dataset, names)
        self._update_engine()

    def _suggested_family(self) -> tuple[str, str]:
        from ...discovery.space import suggest_family

        ds, out = self.state.dataset, self.outcome.currentData()
        if ds is None or not out:
            return "", ""
        try:
            v = apply_treatments(ds, self.treatment_panel.treatments())
        except TreatmentError:
            v = ds
        return suggest_family(v, out)

    def _use_family(self) -> None:
        key, _ = self._suggested_family()
        if key:
            self.family.setCurrentIndex(self.family.findData(key))

    def _spec(self) -> mixed.MixedSpec:
        return mixed.MixedSpec(
            outcome=self.outcome.currentData(), id=self.idv.currentData(), time=self.time.currentData(),
            time_type=self.time_type.currentData(), cluster=self.cluster.currentData(),
            predictors=self.pred_sel.selected(),
            interactions=self.interactions.currentData(), random=self.random.currentData(),
            residual=self.residual.currentData(), method=self.method.currentText(), n_trajectories=self.ntraj.value(),
            treatments=[t.to_dict() for t in self.treatment_panel.treatments()],
            family=self.family.currentData(), estimator=self.estimator.currentData(),
            inference=self.inference.currentData(), event=self.event_combo.currentData(),
            reference=self.reference_combo.currentData(), nagq=self.nagq.value(),
            gee_corstr=self.gee_corstr.currentData())

    def _update_engine(self, *_args) -> None:
        if self.residual.currentData() == "unstructured" and self.random.currentData() != "none":
            self.random.setCurrentIndex(self.random.findData("none"))
            self.time_type.setCurrentIndex(self.time_type.findData("categorical"))
        fam, est = self.family.currentData(), self.estimator.currentData()
        gauss_lme4 = fam == "gaussian" and est == "mixed" and self.residual.currentData() == "independent"
        self.inference.setEnabled(gauss_lme4)
        self.event_combo.setEnabled(fam == "binomial")
        self.reference_combo.setEnabled(fam == "multinomial")
        self.nagq.setEnabled(fam in ("binomial", "ordinal") and est == "mixed")
        self.gee_corstr.setEnabled(est == "gee")
        for w in (self.random, self.residual, self.method):
            w.setEnabled(est == "mixed")
        engine = mixed.MixedSpec(outcome="", id="", residual=self.residual.currentData(), family=fam, estimator=est).engine
        txt = f"Engine: {ENGINE_TEXT[engine]}"
        if gauss_lme4:
            txt += f" · inference: {mixed.INFERENCE[self.inference.currentData()]}"
        if est == "gee":
            txt += (" · GEE estimates population-average effects; mixed models estimate subject-specific effects. "
                    "They are different estimands.")
        self.engine_label.setText(txt)
        key, why = self._suggested_family()
        if key:
            self.family_hint.setText(f"Suggested family: {mixed.FAMILIES[key]}. {why}"
                                     + ("" if key == fam else " (The selected family is kept until you change it.)"))
        else:
            self.family_hint.setText(why)
        self.use_family.setEnabled(bool(key) and key != fam)
        self._preflight()

    def _preflight(self, *_args) -> None:
        """Availability by time point for the current outcome / participant / time (§12), before fitting."""
        import html as _html

        from ...analysis.longitudinal_preflight import preflight

        ds = self.state.dataset
        out, idv = self.outcome.currentData(), self.idv.currentData()
        if ds is None or not out or not idv or out == idv:
            self.preflight_view.setHtml("<i>Choose an outcome and a participant ID.</i>")
            return
        try:
            view = apply_treatments(ds, self.treatment_panel.treatments())
            rep = preflight(view, out, idv, self.time.currentData(), self.pred_sel.selected(), self._spec())
        except Exception as e:  # an incomplete selection must never break the form
            self.preflight_view.setHtml(f"<i>{_html.escape(str(e))}</i>")
            return
        parts = [f"<p><b>{rep.n_records:,}</b> records · <b>{rep.n_participants:,}</b> participants · "
                 f"{len(rep.time_points)} time points · {rep.n_multi_obs:,} participants with 2+ observations, "
                 f"{rep.n_one_obs:,} with one, {rep.n_no_obs:,} with none · duplicate participant-time rows: "
                 f"{rep.duplicates}</p>"]
        parts += [f"<p style='color:#8d6e00'>⚠ {_html.escape(w)}</p>" for w in rep.warnings]
        parts += [f"<p class='note'>{_html.escape(n)}</p>" for n in rep.notes]
        from ..result_view import table_html

        parts.append(table_html(rep.tables(view)[1]))
        self.preflight_view.setHtml("".join(parts))

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

    def _copy_tables(self) -> None:
        from PySide6.QtWidgets import QApplication

        from ...analysis.interpret import tables_tsv

        if self.current is not None:
            QApplication.clipboard().setText(tables_tsv(self.current))
