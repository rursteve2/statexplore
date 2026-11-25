"""Validation & Uncertainty workspace (§12, §13): holdout, cross-validation, stability,
full-procedure CV, prediction intervals."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox, QDialog, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QSpinBox, QSplitter, QTextBrowser, QVBoxLayout, QWidget,
)

from ...analysis import validation
from ...core.dataset import DatasetError, HoldoutSpec
from ...core.variables import Role
from ..model_editor import ModelDialog
from ..result_view import render
from ..widgets.layout import section_scroll
from ..state import AppState
from .eda import CodeDialog


class ValidationPage(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.current = None
        left = QWidget()
        ll = QVBoxLayout(left)

        hbox = QGroupBox("Holdout (reserve data BEFORE exploring)")
        hf = QFormLayout(hbox)
        self.h_frac = QDoubleSpinBox()
        self.h_frac.setRange(0.05, 0.5)
        self.h_frac.setSingleStep(0.05)
        self.h_frac.setValue(0.2)
        self.h_seed = QSpinBox()
        self.h_seed.setRange(1, 10**9)
        self.h_seed.setValue(20261121)
        self.h_group = QComboBox()
        hf.addRow("Fraction reserved", self.h_frac)
        hf.addRow("Seed", self.h_seed)
        hf.addRow("Split by", self.h_group)
        hrow = QHBoxLayout()
        b1 = QPushButton("Reserve holdout")
        b1.clicked.connect(self._reserve)
        b2 = QPushButton("Remove holdout")
        b2.clicked.connect(self._remove_holdout)
        hrow.addWidget(b1)
        hrow.addWidget(b2)
        hf.addRow(hrow)
        self.h_status = QLabel()
        self.h_status.setWordWrap(True)
        hf.addRow(self.h_status)
        ll.addWidget(hbox)

        mbox = QGroupBox("Model (from the comparison set)")
        mf = QFormLayout(mbox)
        self.model = QComboBox()
        newm = QPushButton("New model…")
        newm.clicked.connect(self._new_model)
        mrow = QHBoxLayout()
        mrow.addWidget(self.model, 1)
        mrow.addWidget(newm)
        mf.addRow(mrow)
        self.scheme = QComboBox()
        for k, v in validation.SCHEMES.items():
            self.scheme.addItem(v, k)
        self.k = QSpinBox()
        self.k.setRange(2, 20)
        self.k.setValue(5)
        self.cv_seed = QSpinBox()
        self.cv_seed.setRange(1, 10**9)
        self.cv_seed.setValue(1)
        mf.addRow("Cross-validation", self.scheme)
        mf.addRow("Folds", self.k)
        mf.addRow("Seed", self.cv_seed)
        brow = QHBoxLayout()
        for text, slot in (("Cross-validate", self._cv), ("Evaluate on holdout", self._holdout_eval),
                           ("Prediction uncertainty", self._prediction)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            brow.addWidget(b)
        mf.addRow(brow)
        ll.addWidget(mbox)

        sbox = QGroupBox("Discovery stability (searches with ≤ 2,000 candidates)")
        sf = QFormLayout(sbox)
        self.search = QComboBox()
        self.B = QSpinBox()
        self.B.setRange(5, 500)
        self.B.setValue(20)
        self.frac = QDoubleSpinBox()
        self.frac.setRange(0.5, 0.95)
        self.frac.setValue(0.8)
        self.folds = QSpinBox()
        self.folds.setRange(2, 10)
        self.folds.setValue(5)
        sf.addRow("Search", self.search)
        sf.addRow("Resamples", self.B)
        sf.addRow("Subsample fraction", self.frac)
        sf.addRow("Folds (full-procedure CV)", self.folds)
        self.target = QComboBox()
        self.target.addItem("Automatic (new organizations when a cluster has 10+ groups, else new participants)", "auto")
        for key, text in validation.TARGETS.items():
            self.target.addItem(text.split(":")[0], key)
        self.target.setToolTip("Longitudinal searches: what the validation predicts. The folds follow this target; "
                               "individual repeated rows are never split at random.\n\n"
                               + "\n".join(validation.TARGETS.values()))
        sf.addRow("Prediction target (longitudinal)", self.target)
        srow = QHBoxLayout()
        for text, slot in (("Resampling stability", self._stability), ("Full-procedure CV", self._procedure)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            srow.addWidget(b)
        sf.addRow(srow)
        ll.addWidget(sbox)
        crow = QHBoxLayout()
        self.code_btn = QPushButton("View Code")
        self.code_btn.setEnabled(False)
        self.code_btn.clicked.connect(lambda: self.current and CodeDialog(self.current, self).exec())
        crow.addWidget(self.code_btn)
        crow.addStretch()
        ll.addLayout(crow)
        ll.addStretch()
        scroll = section_scroll(left, 440)
        self.output = QTextBrowser()
        split = QSplitter()
        split.addWidget(scroll)
        split.addWidget(self.output)
        split.setSizes([460, 900])
        lay = QVBoxLayout(self)
        t = QLabel("Validation & Uncertainty")
        t.setObjectName("pageTitle")
        lay.addWidget(t)
        note = QLabel("Searching many models makes some look better than they are. These tools estimate how well "
                      "a discovery holds up in data that played no part in finding it.")
        note.setWordWrap(True)
        lay.addWidget(note)
        lay.addWidget(split, 1)
        for sig in (state.project_changed, state.metadata_changed, state.models_changed):
            sig.connect(self._reload)
        state.result_added.connect(lambda _r: self._reload_searches())
        self._reload()

    def _reload(self) -> None:
        ds, p = self.state.dataset, self.state.project
        keep = self.model.currentData()
        self.model.clear()
        self.h_group.clear()
        self.h_group.addItem("(individual rows)", None)
        if ds is None or p is None:
            self.h_status.setText("")
            return
        for v in ds.variables.values():
            if v.role in (Role.ID, Role.CLUSTER):
                self.h_group.addItem(f"{v.display_name} (whole {v.role.value}s)", v.name)
        if self.h_group.count() > 1:
            self.h_group.setCurrentIndex(1)
        for m in p.models:
            self.model.addItem(f"{m.label or m.id}: {m.display(ds)}", m.id)
        self.model.setCurrentIndex(max(0, self.model.findData(keep)))
        h = ds.holdout
        if h:
            n_hold = len(ds.analysis_frame([next(iter(ds.variables))], part="holdout"))
            self.h_status.setText(f"<b>{n_hold:,}</b> rows reserved ({h.fraction:.0%} of "
                                  f"{'participants' if h.group else 'rows'}, seed {h.seed}); all other analyses use "
                                  f"the remaining rows. Evaluations so far: <b>{h.evaluations}</b>.")
        else:
            self.h_status.setText("No holdout reserved.")
        self._reload_searches()

    def _reload_searches(self) -> None:
        p = self.state.project
        self.search.clear()
        if p is None:
            return
        for run in reversed(p.searches):
            if run.total <= 2000 and run.status in ("complete", "partial", "stopped_early", "cancelled"):
                self.search.addItem(f"{run.created_at} — {run.total} candidates [{run.status}]", run.id)

    def _reserve(self) -> None:
        ds = self.state.dataset
        if ds is None:
            return
        if self.state.project and (self.state.project.searches or self.state.project.results):
            if QMessageBox.question(self, "Holdout", "Analyses have already been run on these rows. A holdout is "
                                    "only independent if it was reserved before exploring. Reserve anyway?") \
                    != QMessageBox.Yes:
                return
        try:
            ds.set_holdout(HoldoutSpec(fraction=self.h_frac.value(), seed=self.h_seed.value(),
                                       group=self.h_group.currentData()))
        except DatasetError as e:
            QMessageBox.warning(self, "Holdout", str(e))
            return
        self.state.notify_metadata_changed()

    def _remove_holdout(self) -> None:
        ds = self.state.dataset
        if ds is not None and ds.holdout is not None:
            ds.set_holdout(None)
            self.state.notify_metadata_changed()

    def _new_model(self) -> None:
        if self.state.dataset is None:
            return
        dlg = ModelDialog(self.state.dataset, None, self)
        if dlg.exec() == QDialog.Accepted and dlg.result_model:
            self.state.add_model(dlg.result_model)
            self.model.setCurrentIndex(self.model.count() - 1)

    def _selected_model(self):
        p = self.state.project
        mid = self.model.currentData()
        m = next((m for m in p.models if m.id == mid), None) if p else None
        if m is None:
            QMessageBox.information(self, "Validation", "Add a model to the comparison set first (or use New model…).")
        return m

    def _run(self, fn) -> None:
        self.output.setHtml("<p>Running in R…</p>")
        self.state.run_r(fn, self._done, lambda msg: self.output.setHtml(f"<p style='color:#c62828'>{msg}</p>"))

    def _cv(self) -> None:
        m, ds = self._selected_model(), self.state.dataset
        if m is None:
            return
        scheme, k, seed = self.scheme.currentData(), self.k.value(), self.cv_seed.value()
        if scheme == "grouped" and not any(v.role is Role.ID for v in ds.variables.values()):
            QMessageBox.information(self, "Validation", "Grouped CV needs a variable with the participant-ID role.")
            return
        if scheme == "temporal" and not any(v.role is Role.TIME for v in ds.variables.values()):
            QMessageBox.information(self, "Validation", "Temporal validation needs a variable with the time role.")
            return
        self._run(lambda s: validation.cross_validate(s, ds, m, scheme, k, seed))

    def _holdout_eval(self) -> None:
        m, ds = self._selected_model(), self.state.dataset
        if m is None:
            return
        if ds.holdout is None:
            QMessageBox.information(self, "Validation", "Reserve a holdout first.")
            return
        self._run(lambda s: validation.holdout_evaluate(s, ds, m))

    def _prediction(self) -> None:
        m, ds = self._selected_model(), self.state.dataset
        if m is not None:
            self._run(lambda s: validation.prediction_uncertainty(s, ds, m))

    def _run_for_search(self, fn) -> None:
        p, ds = self.state.project, self.state.dataset
        rid = self.search.currentData()
        run = next((r for r in p.searches if r.id == rid), None) if p else None
        if run is None:
            QMessageBox.information(self, "Validation", "Run a discovery search (≤ 2,000 candidates) first.")
            return
        rs = validation.ResampleSpec(resamples=self.B.value(), fraction=self.frac.value(), folds=self.folds.value(),
                                     seed=self.cv_seed.value(), target=self.target.currentData() or "auto")
        self._run(lambda s: fn(s, ds, run, rs))

    def _stability(self) -> None:
        self._run_for_search(validation.stability)

    def _procedure(self) -> None:
        self._run_for_search(validation.procedure_cv)

    def _done(self, out) -> None:
        result, artifacts = out
        self.state.add_result(result, artifacts)
        self.current = result
        self.code_btn.setEnabled(True)
        render(self.output, result, artifacts, plots_first=False)
        self._reload()
