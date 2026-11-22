"""Survey Analysis workspace (v1.1 §13): declare the design, then run design-based analyses."""
from __future__ import annotations

import html

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QMessageBox, QPushButton, QScrollArea,
    QSplitter, QTextBrowser, QVBoxLayout, QWidget,
)

from ...analysis import survey as S
from ...core.variables import Storage
from ..result_view import render
from ..state import AppState
from ..widgets.variable_selector import VariableSelector
from .eda import CodeDialog


class SurveyPage(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.current = None
        left = QWidget()
        ll = QVBoxLayout(left)
        dbox = QGroupBox("Survey design (declare it — StatExplore never invents design variables)")
        df = QFormLayout(dbox)
        self.weight, self.psu, self.strata, self.fpc = QComboBox(), QComboBox(), QComboBox(), QComboBox()
        self.lonely = QComboBox()
        for k, v in S.LONELY_PSU.items():
            self.lonely.addItem(v, k)
        self.nest = QCheckBox("PSU codes are only unique within strata (nest)")
        self.nest.setChecked(True)
        self.pooled = QCheckBox("I confirm this weight is valid for rows pooled across waves")
        self.pooled.setToolTip("Only for longitudinal data: a single cross-sectional weight is not automatically "
                               "valid across waves.")
        for label, w in (("Sampling weight", self.weight), ("PSU / cluster", self.psu), ("Strata", self.strata),
                         ("Finite population correction", self.fpc), ("Single-PSU strata", self.lonely)):
            df.addRow(label, w)
        df.addRow(self.nest)
        df.addRow(self.pooled)
        ll.addWidget(dbox)
        rbox = QGroupBox("Advanced: replicate weights")
        rl = QVBoxLayout(rbox)
        self.rep_type = QComboBox()
        for k, v in S.REP_TYPES.items():
            self.rep_type.addItem(v, k)
        rl.addWidget(self.rep_type)
        self.rep_sel = VariableSelector(max_height=140, show_level_filter=False)
        rl.addWidget(self.rep_sel)
        ll.addWidget(rbox)
        row = QHBoxLayout()
        detect = QPushButton("Look for design variables")
        detect.clicked.connect(self._detect)
        explain = QPushButton("What can this design model?")
        explain.clicked.connect(self._explain)
        row.addWidget(detect)
        row.addWidget(explain)
        ll.addLayout(row)
        abox = QGroupBox("Analysis")
        af = QFormLayout(abox)
        self.kind = QComboBox()
        for k, v in S.KINDS.items():
            self.kind.addItem(v, k)
        self.outcome = QComboBox()
        af.addRow("Analysis", self.kind)
        af.addRow("Outcome (regression)", self.outcome)
        ll.addWidget(abox)
        self.var_sel = VariableSelector(max_height=220)
        vbox = QGroupBox("Variables (means / proportions / two for a cross-tab / predictors)")
        QVBoxLayout(vbox).addWidget(self.var_sel)
        ll.addWidget(vbox)
        brow = QHBoxLayout()
        self.run_btn = QPushButton("Run survey analysis")
        self.run_btn.clicked.connect(self._run)
        self.code_btn = QPushButton("View Code")
        self.code_btn.setEnabled(False)
        self.code_btn.clicked.connect(lambda: self.current and CodeDialog(self.current, self).exec())
        brow.addWidget(self.run_btn)
        brow.addWidget(self.code_btn)
        ll.addLayout(brow)
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
        t = QLabel("Survey Analysis")
        t.setObjectName("pageTitle")
        lay.addWidget(t)
        note = QLabel("Design-based estimates for complex samples (survey package). Weighted counts are sums of weights, "
                      "not numbers of independent observations. Survey weighting is not a substitute for modelling "
                      "repeated measurements, and mixed models are not a substitute for design-based inference.")
        note.setWordWrap(True)
        lay.addWidget(note)
        lay.addWidget(split, 1)
        state.project_changed.connect(self._reload)
        state.metadata_changed.connect(self._reload)
        state.busy_changed.connect(lambda b: self.run_btn.setEnabled(not b))
        self._reload()

    def _reload(self) -> None:
        ds = self.state.dataset
        combos = (self.weight, self.psu, self.strata, self.fpc, self.outcome)
        keep = {w: w.currentData() for w in combos}
        for w in combos:
            w.clear()
            w.addItem("(none)", None)
        names = []
        if ds is not None:
            for v in ds.variables.values():
                if v.storage is Storage.DATETIME:
                    continue
                for w in (self.psu, self.strata, self.outcome):
                    w.addItem(v.display_name, v.name)
                if v.storage is Storage.NUMERIC:
                    self.weight.addItem(f"{v.display_name}  —  {v.label}"[:80], v.name)
                    self.fpc.addItem(v.display_name, v.name)
                names.append(v.name)
            for w in combos:
                w.setCurrentIndex(max(0, w.findData(keep[w])))
        self.var_sel.set_variables(ds, names, decorate=(lambda n: f"  ·  {ds.var(n).measure.value}") if ds else None)
        self.rep_sel.set_variables(ds, [n for n in names if ds.var(n).storage is Storage.NUMERIC] if ds else [])

    def design(self) -> S.SurveyDesign:
        return S.SurveyDesign(weight=self.weight.currentData(), psu=self.psu.currentData(),
                              strata=self.strata.currentData(), fpc=self.fpc.currentData(),
                              nest=self.nest.isChecked(), lonely_psu=self.lonely.currentData(),
                              replicate_weights=self.rep_sel.selected(), replicate_type=self.rep_type.currentData(),
                              pooled_waves_ack=self.pooled.isChecked())

    def _detect(self) -> None:
        ds = self.state.dataset
        if ds is None:
            return
        sugg = S.suggest_design(ds)
        lines = [f"<b>{k}</b>: " + (", ".join(html.escape(ds.var(n).display_name) for n in v) or "none found")
                 for k, v in sugg.items()]
        self.output.setHtml("<h3>Candidate design variables (from labels)</h3><p>" + "<br>".join(lines)
                            + "</p><p>Candidates are only suggestions; check the data documentation before declaring "
                              "a design. Nothing is selected automatically.</p>")

    def _explain(self) -> None:
        ds = self.state.dataset
        if ds is None:
            return
        self.output.setHtml("<h3>What this design can model</h3><ul>"
                            + "".join(f"<li>{html.escape(x)}</li>" for x in S.explain_design(ds, self.design())) + "</ul>")

    def _run(self) -> None:
        ds = self.state.dataset
        if ds is None:
            return
        d, kind = self.design(), self.kind.currentData()
        variables, outcome = self.var_sel.selected(), self.outcome.currentData()
        try:
            S.check_design(ds, d)
            S.build(ds, d, kind, variables, outcome if kind in ("linear", "logistic") else None)
        except S.SurveyInputError as e:
            QMessageBox.information(self, "Survey analysis", str(e))
            return
        self.output.setHtml("<p>Running in R…</p>")
        out = outcome if kind in ("linear", "logistic") else None
        self.state.run_r(lambda s: S.run(s, ds, d, kind, variables, out), self._done,
                         lambda m: self.output.setHtml(f"<p style='color:#c62828'>{html.escape(m)}</p>"))

    def _done(self, out) -> None:
        result, artifacts = out
        self.state.add_result(result, artifacts)
        self.current = result
        self.code_btn.setEnabled(True)
        render(self.output, result, artifacts)
