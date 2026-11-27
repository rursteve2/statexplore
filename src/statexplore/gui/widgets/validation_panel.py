"""AMD validation panel (v1.3.3, 02 §2-3; V133-VAL-01): configure before or after a search, preview the split and its
feasibility, run / cancel, inspect fold, performance and stability tables, export the evidence. Every run is a new
immutable record stored with its search (saved in the project)."""
from __future__ import annotations

import html
from typing import Any, Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QSpinBox, QTextBrowser, QVBoxLayout, QWidget,
)

from ...core.schema import Table
from ...discovery import validate as V


class ValidationPanel(QWidget):
    def __init__(self, state: Any, get_run: Callable[[], Any], get_dataset: Callable[[], Any],
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.state, self.get_run, self.get_dataset = state, get_run, get_dataset
        self._stop = False
        self._running = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        intro = QLabel("Validation checks how well the selected model — or the whole selection procedure — predicts data "
                       "that played no part in fitting or choosing it, for an explicit prediction target. It is separate "
                       "from the discovery p-values and is not a confirmatory test.")
        intro.setWordWrap(True)
        intro.setObjectName("hint")
        lay.addWidget(intro)
        form = QFormLayout()
        self.target = QComboBox()
        for k, (label, _) in V.TARGETS.items():
            self.target.addItem(label, k)
        self.pipeline = QComboBox()
        for k, label in V.PIPELINES.items():
            self.pipeline.addItem(label, k)
        self.candidate = QComboBox()
        # AUD134-06: which declared candidates the training parts search (explicit; adaptive history is disclosed)
        self.scope = QComboBox()
        for k, label in V.POOL_SCOPES.items():
            self.scope.addItem(label, k)
        self.prefix_n = QSpinBox()
        self.prefix_n.setRange(1, 2_000_000_000)
        self.prefix_n.setValue(100)
        self.prefix_n.setToolTip("Restricted pool: the first N candidates in the declared order")
        self.max_fits = QSpinBox()
        self.max_fits.setRange(1, 2_000_000_000)
        self.max_fits.setValue(V.ValidationConfig().max_fits)
        self.max_fits.setToolTip("Hard budget of model-fitting invocations (full, reduced, null and selected refits, "
                                 "optimizer retries, auxiliary fits). A validation whose conservative bound exceeds it is "
                                 "refused; if it is reached anyway, the remaining parts are not run and the result is "
                                 "partial.")
        scope_row = QWidget()
        sl = QHBoxLayout(scope_row)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.addWidget(self.scope, 1)
        sl.addWidget(QLabel("N"))
        sl.addWidget(self.prefix_n)
        sl.addWidget(QLabel("fit budget"))
        sl.addWidget(self.max_fits)
        self.rule = QComboBox()
        for k, label in V.RULES.items():
            self.rule.addItem(label, k)
        modes = QWidget()
        ml = QHBoxLayout(modes)
        ml.setContentsMargins(0, 0, 0, 0)
        self.mode_checks = {}
        for k, label in V.MODES.items():
            c = QCheckBox(label)
            c.setChecked(k in ("holdout", "cv"))
            self.mode_checks[k] = c
            ml.addWidget(c)
        nums = QWidget()
        nl = QHBoxLayout(nums)
        nl.setContentsMargins(0, 0, 0, 0)
        self.fraction = QDoubleSpinBox()
        self.fraction.setRange(0.05, 0.5)
        self.fraction.setSingleStep(0.05)
        self.fraction.setValue(0.2)
        self.k = QSpinBox()
        self.k.setRange(2, 50)
        self.k.setValue(5)
        self.repeats = QSpinBox()
        self.repeats.setRange(1, 50)
        self.B = QSpinBox()
        self.B.setRange(1, 5000)
        self.B.setValue(200)
        self.seed = QSpinBox()
        self.seed.setRange(1, 2_000_000_000)
        self.seed.setValue(133)
        for lab, w in (("holdout", self.fraction), ("K", self.k), ("repeats", self.repeats), ("B", self.B),
                       ("seed", self.seed)):
            nl.addWidget(QLabel(lab))
            nl.addWidget(w)
        nl.addStretch()
        for label, w in (("Prediction target", self.target), ("Procedure", self.pipeline),
                         ("Candidate (fixed mode)", self.candidate), ("Candidate pool (pipeline)", scope_row),
                         ("Selection rule (pipeline)", self.rule),
                         ("Modes", modes), ("Settings", nums)):
            form.addRow(label, w)
        lay.addLayout(form)
        self.auto = QCheckBox("Run this validation automatically when the search finishes")
        lay.addWidget(self.auto)
        self.feas = QLabel()
        self.feas.setWordWrap(True)
        self.feas.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.feas)
        row = QHBoxLayout()
        self.preview_btn = QPushButton("Preview split")
        self.preview_btn.clicked.connect(self.preview)
        self.run_btn = QPushButton("Run validation")
        self.run_btn.clicked.connect(self.start)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel)
        self.history = QComboBox()
        self.history.setToolTip("Earlier validation runs of this search (each is immutable)")
        self.history.currentIndexChanged.connect(self._show_history)
        self.export_btn = QPushButton("Export…")
        self.export_btn.setToolTip("Write the split manifest, bootstrap draw map, predictions, fold / performance / "
                                   "stability tables and the exact R code to a folder")
        self.export_btn.clicked.connect(self.export)
        for w in (self.preview_btn, self.run_btn, self.cancel_btn):
            row.addWidget(w)
        row.addWidget(QLabel("Runs:"))
        row.addWidget(self.history, 1)
        row.addWidget(self.export_btn)
        lay.addLayout(row)
        self.status = QLabel("Validation status: not run.")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        self.view = QTextBrowser()
        self.view.setMinimumHeight(320)
        lay.addWidget(self.view, 1)
        for w in (self.target, self.pipeline, self.candidate, self.rule, self.scope):
            w.currentIndexChanged.connect(self.refresh)
        self.prefix_n.valueChanged.connect(self.refresh)
        self.max_fits.valueChanged.connect(self.refresh)
        for c in self.mode_checks.values():
            c.toggled.connect(self.refresh)
        for w in (self.k, self.repeats, self.B, self.seed):
            w.valueChanged.connect(self.refresh)
        self.fraction.valueChanged.connect(self.refresh)
        self.current: dict[str, Any] | None = None
        self.refresh()

    # ---- configuration ----
    def config(self) -> V.ValidationConfig:
        return V.ValidationConfig(
            target=self.target.currentData(), modes=[k for k, c in self.mode_checks.items() if c.isChecked()],
            pipeline=self.pipeline.currentData(), candidate=self.candidate.currentData(), rule=self.rule.currentData(),
            holdout_fraction=round(self.fraction.value(), 4), k=self.k.value(), repeats=self.repeats.value(),
            B=self.B.value(), seed=self.seed.value(), pool=self.scope.currentData() or "cumulative",
            prefix_n=self.prefix_n.value() if self.scope.currentData() == "prefix" else None,
            max_fits=self.max_fits.value())

    def set_config(self, cfg: V.ValidationConfig) -> None:
        for combo, val in ((self.target, cfg.target), (self.pipeline, cfg.pipeline), (self.rule, cfg.rule),
                           (self.candidate, cfg.candidate)):
            i = combo.findData(val)
            if i >= 0:
                combo.setCurrentIndex(i)
        for k, c in self.mode_checks.items():
            c.setChecked(k in cfg.modes)
        i = self.scope.findData("original" if cfg.pool == "declared" else cfg.pool)
        if i >= 0:
            self.scope.setCurrentIndex(i)
        if cfg.prefix_n:
            self.prefix_n.setValue(int(cfg.prefix_n))
        self.max_fits.setValue(int(cfg.max_fits))
        self.fraction.setValue(cfg.holdout_fraction)
        self.k.setValue(cfg.k)
        self.repeats.setValue(cfg.repeats)
        self.B.setValue(cfg.B)
        self.seed.setValue(cfg.seed)

    def set_run(self, run: Any) -> None:
        """A search was selected / finished: candidates, allowed targets, its saved validation runs."""
        keep = self.candidate.currentData()
        self.candidate.blockSignals(True)
        self.candidate.clear()
        if run is not None:
            for r in run.rows:
                if r.get("status") in ("ok", "warning"):
                    self.candidate.addItem(f"#{r['idx']}: {r.get('formula') or r.get('label') or ''}", int(r["idx"]))
            i = self.candidate.findData(keep)
            self.candidate.setCurrentIndex(max(i, 0))
        self.candidate.blockSignals(False)
        if run is not None:
            allowed = V.allowed_targets(run)
            model = self.target.model()
            for i in range(self.target.count()):
                why = allowed.get(self.target.itemData(i), "")
                item = model.item(i)
                item.setEnabled(not why)
                item.setToolTip(why)
            if allowed.get(self.target.currentData()):
                self.target.setCurrentIndex(self.target.findData(V.default_target(run)))
            if run.settings.workflow == "structure":
                self.rule.setCurrentIndex(self.rule.findData("aic"))
        self._fill_history(run)
        self.refresh()

    def _fill_history(self, run: Any, select_last: bool = True) -> None:
        self.history.blockSignals(True)
        self.history.clear()
        runs = list(getattr(run, "validation_runs", []) or []) if run is not None else []
        for i, v in enumerate(runs, start=1):
            c = v["config"]
            self.history.addItem(f"{i}: {v['target_label']} · {', '.join(c['modes'])} · {v['status']}", i - 1)
        self.history.blockSignals(False)
        if runs and select_last:
            self.history.setCurrentIndex(len(runs) - 1)
            self._show(runs[-1])
        elif not runs:
            self.current = None
            self.view.setHtml("")
            self.status.setText("Validation status: not run.")

    def _show_history(self, *_args) -> None:
        run = self.get_run()
        i = self.history.currentData()
        if run is not None and i is not None and i < len(run.validation_runs):
            self._show(run.validation_runs[i])

    def refresh(self, *_args) -> None:
        run, ds = self.get_run(), self.get_dataset()
        fixed = self.pipeline.currentData() == "fixed"
        self.candidate.setEnabled(fixed)
        self.rule.setEnabled(not fixed)
        self.scope.setEnabled(not fixed)
        self.prefix_n.setEnabled(not fixed and self.scope.currentData() == "prefix")
        if run is None or ds is None:
            self.feas.setText("Configure now; validation runs on a completed search (or automatically when it finishes "
                              "if ticked above).")
            self.run_btn.setEnabled(False)
            self.preview_btn.setEnabled(False)
            return
        try:
            info = V.feasibility(ds, run, self.config())
        except Exception as e:  # incomplete selections must never break the panel
            self.feas.setText(html.escape(str(e)))
            self.run_btn.setEnabled(False)
            return
        parts = [f"Rows {info.get('rows', '—')} · participants {info.get('participants', '—')} · clusters "
                 f"{info.get('clusters', '—')} · units for this target {info.get('units', '—')}"
                 + (f" · eligible clusters {info['eligible_clusters']}" if "eligible_clusters" in info else "")
                 + (f" (ineligible: {len(info['ineligible_clusters'])})" if info.get("ineligible_clusters") else "")
                 + f" · at most {info.get('fits', 0):,} model-fitting invocations (conservative bound; estimate about "
                 f"{info.get('fits_estimate', 0):,}; budget {self.max_fits.value():,})."]
        if info.get("pool_label") and not fixed:
            parts.append("Candidate pool: " + html.escape(info["pool_label"]))
        parts += [f"<span style='color:#c62828'>✖ {html.escape(p)}</span>" for p in info["problems"]]
        parts += [f"<span style='color:#8d6e00'>⚠ {html.escape(w)}</span>" for w in info["warnings"]]
        self.feas.setText("<br>".join(parts))
        ok = not info["problems"] and not self._running
        self.run_btn.setEnabled(ok)
        self.preview_btn.setEnabled(not info["problems"])

    def preview(self) -> None:
        run, ds = self.get_run(), self.get_dataset()
        if run is None or ds is None:
            return
        cfg = self.config()
        u = V.unit_frame(ds, run)
        lines = []
        for sp in V.make_splits(u, cfg):
            f = u.assign(fold=u["row_id"].map(sp["folds"]))
            col = "cluster" if cfg.target == "cluster" else "participant"
            for k in sorted(set(f["fold"]) - {0}):
                t = f[f["fold"] == k]
                lines.append(f"{sp['id']} fold {k}: {len(t)} test rows, {t[col].nunique() if cfg.target != 'rows' else len(t)}"
                             f" test units" + (f", {t['cluster'].nunique()} clusters" if (u['cluster'] != '').any() else ""))
            bad = V.check_split(u, sp, cfg.target)
            lines.append("  disjointness checks: " + ("OK" if not bad else "; ".join(bad)))
        if "bootstrap" in cfg.modes:
            lines.append(f"bootstrap: {cfg.B} draws of the {('clusters' if cfg.target == 'cluster' else 'participants' if cfg.target != 'rows' else 'rows')}"
                         " with replacement; each drawn unit gets a new ID.")
        self.view.setPlainText("Split preview (no model fitted)\n" + "\n".join(lines))

    # ---- execution ----
    def start(self, *_args, run: Any = None) -> None:
        run = run or self.get_run()
        ds = self.get_dataset()
        if run is None or ds is None or self._running:
            return
        cfg = self.config()
        try:
            runner = V.ValidationRunner(self.state.r_session(), ds, run, cfg)
        except V.ValidationError as e:
            QMessageBox.information(self, "Validation", str(e))
            return
        self._running, self._stop = True, False
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.status.setText("Validation status: RUNNING…")
        self.view.setHtml("<p>Fitting in R…</p>")
        self.state.run_r(lambda _s: runner.execute(should_stop=lambda: self._stop), lambda v: self._done(run, v),
                         self._error)

    def cancel(self) -> None:
        self._stop = True
        self.status.setText("Validation status: cancelling (completed folds / draws are kept)…")
        self.state.cancel_r()

    def _done(self, run: Any, v: dict[str, Any]) -> None:
        self._running = False
        self.cancel_btn.setEnabled(False)
        names = {}
        ds = self.get_dataset()
        if ds is not None:
            from ...rbridge.codegen import alias

            names = {alias(x.alias): x.display_name for x in ds.variables.values()}
        for rec in v.get("stability", []) + v.get("bootstrap", []):
            c = str(rec.get("coefficient") or "")
            for a, n in sorted(names.items(), key=lambda x: -len(x[0])):
                if c.startswith(a):
                    rec["coefficient"] = n + c[len(a):]
                    break
        run.validation_runs.append(v)
        self.state.dirty = True
        self._fill_history(run)
        self.refresh()

    def _error(self, msg: str) -> None:
        self._running = False
        self.cancel_btn.setEnabled(False)
        self.status.setText(f"Validation status: FAILED — {html.escape(msg)}")
        self.refresh()

    def _show(self, v: dict[str, Any]) -> None:
        from ..result_view import table_html

        self.current = v
        col = {"complete": "#2e7d32", "partial": "#8d6e00", "cancelled": "#8d6e00"}.get(v["status"], "#c62828")
        self.status.setText(f"<span style='color:{col}'><b>Validation status: {html.escape(v['status'])}</b></span> — "
                            f"{html.escape(v['target_label'])}; {html.escape(v['pipeline_label'])}")
        parts = ["<pre style='white-space:pre-wrap'>" + html.escape(V.summary_text(v)) + "</pre>"]
        for name, cols, rows, notes in V.tables(v):
            parts.append(f"<h4>{html.escape(name)}</h4>" + table_html(Table(name=name, columns=cols, rows=rows,
                                                                             notes=notes)))
        self.view.setHtml("".join(parts))

    def export(self) -> None:
        if self.current is None:
            QMessageBox.information(self, "Validation", "Run or select a validation first.")
            return
        d = QFileDialog.getExistingDirectory(self, "Export validation to folder")
        if d:
            ans = QMessageBox.question(
                self, "Include the data?", "Include the analysis data (only the columns the validation uses) so that "
                "validation_reproduce.R runs on its own?\n\nNo: the script, manifests and results are written; the "
                "data file must be placed in the folder later and is checked by its SHA-256.",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            if ans == QMessageBox.Cancel:
                return
            run = self.get_run()
            ds = self.get_dataset()
            # AUD134-07: the export verifies that the project's data are the analysed data (or rebuilds them from the
            # search's snapshot); otherwise it writes results and metadata only
            files = V.export(self.current, d, ds=ds, include_data=ans == QMessageBox.Yes, run=run)
            from pathlib import Path

            note = ""
            if (Path(d) / "REPRODUCTION_UNAVAILABLE.txt").exists():
                note = ("\n\nNo runnable reproduction was written: the project's data are not the data this validation "
                        "analysed (see REPRODUCTION_UNAVAILABLE.txt). The historical results and code were exported.")
            QMessageBox.information(self, "Validation", f"{len(files)} files written to {d}.{note}")
