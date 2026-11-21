"""Automated Model Discovery workspace (§10, §11, §20.3–20.5)."""
from __future__ import annotations

import html
import threading
from typing import Any

import pandas as pd
from PySide6.QtCore import QObject, QSortFilterProxyModel, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QProgressBar, QPushButton, QScrollArea, QSpinBox,
    QSplitter, QTableView, QTabWidget, QTextBrowser, QVBoxLayout, QWidget,
)

from ...core.variables import Measure, Storage
from ...analysis.models import from_candidate
from ...discovery import engine
from ...discovery.detail import candidate_from_row, full_analysis
from ...discovery.rcode import candidate_script
from ...discovery.space import PROFILES, SearchSettings, SearchSpaceError, describe_space
from ..models import FrameModel
from ..result_view import render
from ..state import AppState
from .eda import CodeDialog

P_COLS = ("p", "p (adjusted)", "p (BH)", "p (Holm)")
DISCOVERY_COLS = {"idx": "#", "interaction": "Interaction", "formula": "Model", "statistic": "Statistic",
                  "df1": "df1", "df2": "df2", "p": "p", "p_adj": "p (adjusted)", "effect": "Effect",
                  "estimate": "Estimate", "ci_low": "CI low", "ci_high": "CI high", "n": "n",
                  "n_params": "Parameters", "status": "Status", "flags": "Flags"}
SUPPORT_COLS = {"tier": "Tier", "idx": "#", "interaction": "Interaction", "formula": "Model", "effect": "Effect",
                "p": "p", "p_adj": "p (adjusted)", "estimate": "Estimate", "ci_low": "CI low",
                "ci_high": "CI high", "n_params": "Parameters", "aic": "AIC", "r2": "R² (full model)",
                "flags": "Flags"}
ALL_COLS = {"idx": "#", "interaction": "Interaction", "formula": "Model", "status": "Status", "reason": "Reason",
            "flags": "Flags", "n": "n", "n_params": "Parameters", "p": "p", "p_bh": "p (BH)",
            "p_holm": "p (Holm)", "speculative": "After stop point"}


class _Bridge(QObject):
    progress = Signal(object)


class ResultTable(QWidget):
    """Searchable, sortable table over a results frame; remembers candidate indices."""

    def __init__(self, columns: dict[str, str]):
        super().__init__()
        self.columns = columns
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search (variable names, status, flags…)")
        lay.addWidget(self.search)
        self.model = FrameModel(p_columns=P_COLS)
        self.proxy = QSortFilterProxyModel()
        self.proxy.setSourceModel(self.model)
        self.proxy.setSortRole(Qt.UserRole)
        self.proxy.setFilterKeyColumn(-1)
        self.proxy.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.search.textChanged.connect(self.proxy.setFilterFixedString)
        self.view = QTableView()
        self.view.setModel(self.proxy)
        self.view.setSortingEnabled(True)
        self.view.sortByColumn(-1, Qt.AscendingOrder)
        self.view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.view.setSelectionMode(QAbstractItemView.SingleSelection)
        lay.addWidget(self.view)

    def set_frame(self, df: pd.DataFrame) -> None:
        cols = [c for c in self.columns if c in df.columns]
        self.model.set_frame(df[cols].rename(columns=self.columns) if len(df) else pd.DataFrame(columns=list(self.columns.values())))
        if 0 < len(df) <= 5000:  # sizing to contents scans every row; skip for very large searches
            self.view.resizeColumnsToContents()
            for c in range(self.model.columnCount()):
                self.view.setColumnWidth(c, min(self.view.columnWidth(c), 320))

    def selected_idx(self) -> int | None:
        sel = self.view.selectionModel().selectedRows()
        if not sel:
            return None
        src = self.proxy.mapToSource(sel[0])
        return int(self.model.frame.iloc[src.row()]["#"])


class DiscoveryPage(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.run: engine.DiscoveryRun | None = None
        self.frame = pd.DataFrame()
        self.cancel_event: threading.Event | None = None
        self.bridge = _Bridge()
        self.bridge.progress.connect(self._on_progress)
        self.estimate = QLabel()  # created first: settings widgets clear it when they change
        self.estimate.setWordWrap(True)
        self.estimate.setTextFormat(Qt.RichText)

        # ---------- configuration ----------
        cfg = QWidget()
        cl = QVBoxLayout(cfg)
        form = QFormLayout()
        self.outcome = QComboBox()
        self.outcome.currentIndexChanged.connect(self._outcome_changed)
        form.addRow("Dependent variable", self.outcome)
        self.participant = QComboBox()
        self.participant.setToolTip("Long data with several rows per participant: fit linear mixed models with a "
                                    "random intercept per participant (continuous outcomes).")
        form.addRow("Repeated measures", self.participant)
        cl.addLayout(form)
        self.var_filter = QLineEdit()
        self.var_filter.setPlaceholderText("Filter variables…")
        self.var_filter.textChanged.connect(self._filter_lists)
        cl.addWidget(self.var_filter)
        cand_box = QGroupBox("Candidate predictors (up to 39)")
        cb = QVBoxLayout(cand_box)
        self.cands = QListWidget()
        self.cands.itemChanged.connect(self._selection_changed)
        cb.addWidget(self.cands)
        row = QHBoxLayout()
        b_all = QPushButton("Check visible")
        b_all.clicked.connect(lambda: self._check_visible(self.cands, True))
        b_none = QPushButton("Clear")
        b_none.clicked.connect(lambda: self._check_visible(self.cands, False, all_items=True))
        self.n_sel = QLabel()
        row.addWidget(b_all)
        row.addWidget(b_none)
        row.addWidget(self.n_sel)
        row.addStretch()
        cb.addLayout(row)
        cl.addWidget(cand_box, 2)
        req_box = QGroupBox("Required covariates (in every model; count toward the maximum)")
        rb = QVBoxLayout(req_box)
        self.reqs = QListWidget()
        self.reqs.itemChanged.connect(self._selection_changed)
        rb.addWidget(self.reqs)
        cl.addWidget(req_box, 1)

        opts = QGroupBox("Search settings")
        of = QFormLayout(opts)
        self.max_pred = QSpinBox()
        self.max_pred.setRange(2, 6)
        self.max_pred.setValue(3)
        self.order = QComboBox()
        self.order.addItem("Two-way", 2)
        self.order.addItem("Two- and three-way (advanced)", 3)
        self.cov_mode = QComboBox()
        self.cov_mode.addItem("Focal: required covariates only (recommended)", "focal")
        self.cov_mode.addItem("Subsets: every covariate combination (exhaustive)", "subsets")
        self.sample = QComboBox()
        self.sample.addItem("Shared complete-case sample (recommended)", "shared")
        self.sample.addItem("Complete cases per model (sample sizes differ)", "per_model")
        self.alpha = QDoubleSpinBox()
        self.alpha.setDecimals(4)
        self.alpha.setRange(0.0001, 0.2)
        self.alpha.setSingleStep(0.005)
        self.alpha.setValue(0.05)
        self.correction = QComboBox()
        self.correction.addItem("Benjamini-Hochberg (FDR)", "BH")
        self.correction.addItem("Holm (family-wise)", "holm")
        self.mode = QComboBox()
        for key, text in [("C", "C: Exhaustive (bounded)"), ("A", "A: Stop at first significant"),
                          ("B", "B: Stop at target number significant"), ("D", "D: Budget-limited")]:
            self.mode.addItem(text, key)
        self.target = QSpinBox()
        self.target.setRange(1, 100000)
        self.target.setValue(10)
        self.profile = QComboBox()
        for key in ("fast", "balanced", "thorough"):
            self.profile.addItem(key.capitalize() + (" (default)" if key == "balanced" else ""), key)
        self.profile.setCurrentIndex(1)
        self.profile.currentIndexChanged.connect(self._apply_profile)
        self.max_models = QSpinBox()
        self.max_models.setRange(0, 100_000_000)
        self.max_models.setSpecialValueText("no limit")
        self.max_minutes = QDoubleSpinBox()
        self.max_minutes.setRange(0, 100000)
        self.max_minutes.setSpecialValueText("no limit")
        self.workers = QSpinBox()
        self.workers.setRange(0, 64)
        self.workers.setSpecialValueText("all cores - 1")
        self.max_params = QSpinBox()
        self.max_params.setRange(2, 500)
        self.max_params.setValue(40)
        for label, w in [("Max predictors per model", self.max_pred), ("Interaction order", self.order),
                         ("Covariate adjustment", self.cov_mode), ("Missing data", self.sample),
                         ("Significance threshold (α)", self.alpha), ("Multiple-testing adjustment", self.correction),
                         ("Search mode", self.mode), ("Target significant models (mode B)", self.target),
                         ("Search profile", self.profile), ("Model budget", self.max_models),
                         ("Time budget (minutes)", self.max_minutes), ("Parallel R workers", self.workers),
                         ("Max parameters per model", self.max_params)]:
            of.addRow(label, w)
        for w in (self.max_pred, self.order, self.cov_mode, self.sample, self.mode, self.max_models):
            sig = getattr(w, "valueChanged", None) or w.currentIndexChanged
            sig.connect(self._invalidate_estimate)
        cl.addWidget(opts)
        self._apply_profile()

        btns = QHBoxLayout()
        self.estimate_btn = QPushButton("Calculate search space")
        self.estimate_btn.clicked.connect(self._estimate)
        self.start_btn = QPushButton("Start search")
        self.start_btn.clicked.connect(self._start)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        self.resume_btn = QPushButton("Resume")
        self.resume_btn.setEnabled(False)
        self.resume_btn.clicked.connect(self._resume)
        for b in (self.estimate_btn, self.start_btn, self.cancel_btn, self.resume_btn):
            btns.addWidget(b)
        cl.addLayout(btns)
        cl.addWidget(self.estimate)
        self.progress = QProgressBar()
        self.progress_label = QLabel()
        cl.addWidget(self.progress)
        cl.addWidget(self.progress_label)
        scroll = QScrollArea()
        scroll.setWidget(cfg)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(430)

        # ---------- results ----------
        right = QWidget()
        rl = QVBoxLayout(right)
        top = QHBoxLayout()
        top.addWidget(QLabel("Search:"))
        self.history = QComboBox()
        self.history.activated.connect(self._pick_history)
        top.addWidget(self.history, 1)
        rl.addLayout(top)
        self.tabs = QTabWidget()
        self.summary_view = QTextBrowser()
        self.discovery_table = ResultTable(DISCOVERY_COLS)
        self.support_table = ResultTable(SUPPORT_COLS)
        self.all_table = ResultTable(ALL_COLS)
        self.tabs.addTab(self.summary_view, "Summary")
        self.tabs.addTab(self.discovery_table, "Discovery significance ranking")
        self.tabs.addTab(self.support_table, "Statistical support ranking")
        self.tabs.addTab(self.all_table, "All attempts")
        rl.addWidget(self.tabs, 1)
        sort_row = QHBoxLayout()
        sort_row.addWidget(QLabel("Discovery ranking order:"))
        self.rank_by = QComboBox()
        self.rank_by.addItem("Raw p-value", "p")
        self.rank_by.addItem("Adjusted p-value", "p_adj")
        self.rank_by.currentIndexChanged.connect(self._refresh_tables)
        sort_row.addWidget(self.rank_by)
        sort_row.addStretch()
        self.open_btn = QPushButton("Open full analysis")
        self.open_btn.clicked.connect(self._open_detail)
        self.code_btn = QPushButton("View candidate code")
        self.code_btn.clicked.connect(self._view_code)
        self.teach_btn = QPushButton("Plot in Teaching Studio")
        self.teach_btn.clicked.connect(self._to_teaching)
        self.set_btn = QPushButton("Add to comparison")
        self.set_btn.clicked.connect(self._to_set)
        for b in (self.open_btn, self.code_btn, self.teach_btn, self.set_btn):
            sort_row.addWidget(b)
        rl.addLayout(sort_row)

        split = QSplitter()
        split.addWidget(scroll)
        split.addWidget(right)
        split.setSizes([460, 900])
        lay = QVBoxLayout(self)
        t = QLabel("Automated Model Discovery")
        t.setObjectName("pageTitle")
        lay.addWidget(t)
        warn = QLabel("Exploratory: models are searched for statistically significant interactions. A significant "
                      "result here is a lead to examine, not reliable evidence that the effect is real.")
        warn.setWordWrap(True)
        lay.addWidget(warn)
        lay.addWidget(split, 1)

        state.project_changed.connect(self._reload)
        state.metadata_changed.connect(self._reload_lists)
        self._reload()

    # ---------- variable lists ----------
    def _reload(self) -> None:
        self.run = None
        self._reload_lists()
        self._refresh_history()
        self._show_run()

    def _reload_lists(self) -> None:
        ds = self.state.dataset
        keep_outcome = self.outcome.currentData()
        keep_c, keep_r = set(self._checked(self.cands)), set(self._checked(self.reqs))
        for w in (self.cands, self.reqs):
            w.blockSignals(True)
            w.clear()
        keep_part = self.participant.currentData()
        self.participant.clear()
        self.participant.addItem("None (one row per participant)", None)
        self.outcome.blockSignals(True)
        self.outcome.clear()
        self.outcome.addItem("— choose —", None)
        if ds is not None:
            for v in ds.variables.values():
                if v.role.value in ("excluded",) or v.storage is Storage.DATETIME:
                    continue
                n_values = ds.analysis_frame([v.name])[v.name].dropna().nunique()
                if n_values == 2 or (v.measure is Measure.CONTINUOUS and v.storage is Storage.NUMERIC):
                    self.outcome.addItem(v.display_name, v.name)
                if v.role.value == "id":
                    self.participant.addItem(f"Random intercept per {v.display_name}", v.name)
                if v.role.value in ("id",) or (not v.is_categorical and v.storage is not Storage.NUMERIC):
                    continue
                for lst, keep in ((self.cands, keep_c), (self.reqs, keep_r)):
                    it = QListWidgetItem(f"{v.display_name}  ·  {v.measure.value}")
                    it.setToolTip(v.label or v.name)
                    it.setData(Qt.UserRole, v.name)
                    it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                    it.setCheckState(Qt.Checked if v.name in keep else Qt.Unchecked)
                    lst.addItem(it)
        self.participant.setCurrentIndex(max(0, self.participant.findData(keep_part)))
        idx = self.outcome.findData(keep_outcome)
        self.outcome.setCurrentIndex(max(0, idx))
        self.outcome.blockSignals(False)
        for w in (self.cands, self.reqs):
            w.blockSignals(False)
        self._outcome_changed()
        self._filter_lists(self.var_filter.text())

    def _outcome_changed(self) -> None:
        out = self.outcome.currentData()
        for lst in (self.cands, self.reqs):
            for i in range(lst.count()):
                it = lst.item(i)
                is_out = it.data(Qt.UserRole) == out
                it.setHidden(is_out)
                if is_out:
                    it.setCheckState(Qt.Unchecked)
        self._selection_changed()

    def _filter_lists(self, text: str) -> None:
        t = text.lower()
        out = self.outcome.currentData()
        for lst in (self.cands, self.reqs):
            for i in range(lst.count()):
                it = lst.item(i)
                hide = it.data(Qt.UserRole) == out or (bool(t) and t not in it.text().lower()
                                                       and t not in it.toolTip().lower()
                                                       and it.checkState() != Qt.Checked)
                it.setHidden(hide)

    @staticmethod
    def _checked(lst: QListWidget) -> list[str]:
        return [lst.item(i).data(Qt.UserRole) for i in range(lst.count()) if lst.item(i).checkState() == Qt.Checked]

    def _check_visible(self, lst: QListWidget, on: bool, all_items: bool = False) -> None:
        lst.blockSignals(True)
        for i in range(lst.count()):
            it = lst.item(i)
            if all_items or not it.isHidden():
                it.setCheckState(Qt.Checked if on else Qt.Unchecked)
        lst.blockSignals(False)
        self._selection_changed()

    def _selection_changed(self, *_: Any) -> None:
        n = len(self._checked(self.cands))
        self.n_sel.setText(f"{n} selected" + (" — too many (max 39)" if n > 39 else ""))
        self._invalidate_estimate()

    def _invalidate_estimate(self, *_: Any) -> None:
        self.estimate.setText("")

    def _apply_profile(self) -> None:
        p = PROFILES[self.profile.currentData()]
        self.max_models.setValue(p["max_models"] or 0)
        self.max_minutes.setValue(p["max_minutes"] or 0)
        self.workers.setValue(p["workers"])

    # ---------- settings ----------
    def _settings(self) -> SearchSettings:
        outcome = self.outcome.currentData()
        if not outcome:
            raise SearchSpaceError("Choose a dependent variable.")
        return SearchSettings(
            outcome=outcome, candidates=[n for n in self._checked(self.cands) if n != outcome],
            required=[n for n in self._checked(self.reqs) if n != outcome],
            max_predictors=self.max_pred.value(), max_order=self.order.currentData(),
            covariate_mode=self.cov_mode.currentData(), alpha=self.alpha.value(),
            correction=self.correction.currentData(), mode=self.mode.currentData(), target=self.target.value(),
            profile=self.profile.currentData(), max_models=self.max_models.value() or None,
            max_minutes=self.max_minutes.value() or None, workers=self.workers.value(),
            sample=self.sample.currentData(), max_params=self.max_params.value(),
            participant=self.participant.currentData(),
        )

    def _estimate(self) -> dict[str, Any] | None:
        ds = self.state.dataset
        if ds is None:
            return None
        try:
            info = describe_space(ds, self._settings())
        except SearchSpaceError as e:
            self.estimate.setText(f"<span style='color:#c62828'>{html.escape(str(e))}</span>")
            return None
        fam = {"gaussian": "linear regression (F-tests)", "binomial": "logistic regression (LR tests)",
               "lmm": "linear mixed models, random intercept (LR tests, ML)"}[info["family"]]
        warn = "".join(f"<br><span style='color:#8d6e00'>⚠ {html.escape(w)}</span>" for w in info["warnings"])
        self.estimate.setText(f"<b>{info['total']:,}</b> candidate models · {fam} · shared complete-case "
                              f"sample: <b>{info['n_shared']:,}</b> of {info['n_rows']:,} rows{warn}")
        return info

    # ---------- running ----------
    def _start(self) -> None:
        ds = self.state.dataset
        if ds is None:
            return
        info = self._estimate()
        if info is None:
            return
        if info["total"] >= 1_000_000 and QMessageBox.question(
                self, "Large search", f"This search space has {info['total']:,} candidate models. Start anyway?"
        ) != QMessageBox.Yes:
            return
        run = engine.create_run(ds, self._settings())
        if self.state.project is not None:
            self.state.project.searches.append(run)
            self.state.dirty = True
        self.run = run
        self._refresh_history()
        self._launch(run)

    def _resume(self) -> None:
        if self.run is not None and self.run.resumable:
            self.run.settings.max_models = self.max_models.value() or None
            self.run.settings.max_minutes = self.max_minutes.value() or None
            self._launch(self.run)

    def _launch(self, run: engine.DiscoveryRun) -> None:
        ds = self.state.dataset
        self.cancel_event = threading.Event()
        cancel = self.cancel_event
        rscript = self.state.rscript_setting() or None
        self._set_running(True)
        self.progress.setRange(0, max(1, run.total))
        self.state.run_io(
            lambda: engine.execute(run, ds, rscript=rscript, cancel=cancel,
                                   progress=lambda r: self.bridge.progress.emit(r)),
            self._finished, self._failed)

    def _cancel(self) -> None:
        if self.cancel_event:
            self.cancel_event.set()
            self.progress_label.setText("Cancelling — finishing the batches already running…")

    def _set_running(self, on: bool) -> None:
        for w in (self.start_btn, self.resume_btn, self.estimate_btn):
            w.setEnabled(not on)
        self.cancel_btn.setEnabled(on)

    def _on_progress(self, run: engine.DiscoveryRun) -> None:
        done = len(run.rows)
        self.progress.setValue(min(done, run.total))
        rate = done / run.elapsed_s if run.elapsed_s > 0 else 0
        remaining = (run.total - run.n_done) / rate if rate > 0 else None
        sig = sum(1 for r in run.rows if r["status"] in ("ok", "warning") and r.get("p") is not None
                  and r["p"] < run.settings.alpha)
        eta = f" · about {remaining / 60:.1f} min left at this rate" if remaining is not None else ""
        self.progress_label.setText(f"{done:,} of {run.total:,} evaluated · {sig:,} with raw p < "
                                    f"{run.settings.alpha:g} · {run.elapsed_s:.0f} s{eta}")

    def _finished(self, run: engine.DiscoveryRun) -> None:
        self._set_running(False)
        self.state.dirty = True
        self._on_progress(run)
        self._show_run()

    def _failed(self, message: str) -> None:
        self._set_running(False)
        self._show_run()
        QMessageBox.warning(self, "Discovery failed", message)

    # ---------- results ----------
    def _refresh_history(self) -> None:
        self.history.clear()
        p = self.state.project
        if p is None:
            return
        for run in reversed(p.searches):
            ds = p.dataset
            name = ds.var(run.settings.outcome).display_name if run.settings.outcome in ds.variables else "?"
            self.history.addItem(f"{run.created_at} — {name}: {run.total:,} candidates [{run.status}]", run.id)
        if self.run is not None:
            self.history.setCurrentIndex(max(0, self.history.findData(self.run.id)))

    def _pick_history(self, index: int) -> None:
        p = self.state.project
        rid = self.history.itemData(index)
        if p is None:
            return
        self.run = next((r for r in p.searches if r.id == rid), None)
        self._show_run()

    def _show_run(self) -> None:
        run = self.run
        self.resume_btn.setEnabled(bool(run and run.resumable))
        if run is None:
            self.summary_view.setHtml("<p>Configure a search on the left, calculate the search space, then start.</p>")
            for t in (self.discovery_table, self.support_table, self.all_table):
                t.set_frame(pd.DataFrame())
            return
        self.frame = engine.results_frame(run)
        self.summary_view.setHtml(self._summary_html(run, self.frame))
        self._refresh_tables()
        self._refresh_history()

    def _refresh_tables(self) -> None:
        if self.run is None:
            return
        df = self.frame
        self.discovery_table.set_frame(engine.discovery_ranking(df, by=self.rank_by.currentData()) if len(df) else df)
        self.support_table.set_frame(engine.support_ranking(df) if len(df) else df)
        self.all_table.set_frame(df)

    def _summary_html(self, run: engine.DiscoveryRun, df: pd.DataFrame) -> str:
        sm = engine.summary(run, df) if len(df) else None
        s = run.settings
        ds = self.state.dataset
        e = html.escape
        color = "#2e7d32" if run.status == "complete" else "#8d6e00"
        parts = [f"<h2>Search {e(run.id)}</h2>",
                 f"<p><b style='color:{color}'>{e(engine.STATUS_LABELS.get(run.status, run.status))}</b>"
                 + (f" — {e(run.stop_reason)}" if run.stop_reason else "") + "</p>"]
        for w in run.warnings:
            parts.append(f"<p style='color:#c62828'><b>⚠ {e(w)}</b></p>")
        if sm:
            corr = "Benjamini-Hochberg (FDR)" if s.correction == "BH" else "Holm"
            parts.append(
                f"<p>{sm['attempted']:,} of {sm['total']:,} candidate models attempted · {sm['family_size']:,} tested "
                f"(the multiple-testing family) · {sm['invalid']:,} invalid · {sm['failed']:,} failed · "
                f"{sm['warning']:,} flagged by screening"
                + (f" · {sm['speculative']:,} evaluated after the stop point (excluded)" if sm['speculative'] else "")
                + "</p>")
            parts.append(
                f"<p><b>{sm['sig_raw']:,}</b> interaction(s) with raw p &lt; {s.alpha:g}. "
                f"With {sm['family_size']:,} tests, about <b>{sm['expected_false_raw']:.1f}</b> would be expected "
                f"by chance alone if no interaction were real. <b>{sm['sig_adj']:,}</b> remain significant after "
                f"{corr} adjustment.</p>")
            if sm["provisional"]:
                parts.append("<p style='color:#8d6e00'>Adjusted p-values are <b>provisional</b>: they are "
                             "conditional on the candidates evaluated, not the full search space.</p>")
        parts.append(
            "<p class='note'>Multiple-testing adjustment controls false discoveries among the tests run; it does "
            "not remove post-selection bias. Effect sizes and confidence intervals of the top-ranked models are "
            "optimistic. Confirm promising interactions on new data or a holdout sample (Phase 6).</p>")
        fam = {"gaussian": "Linear models, focal F-test (nested comparison)",
               "binomial": "Logistic models, focal likelihood-ratio test",
               "lmm": "Linear mixed models with a random intercept per participant; focal likelihood-ratio "
                      "test on ML fits (df2 column = participants)"}[run.family]
        effect = {"gaussian": "ΔR² (variance uniquely explained by the interaction)",
                  "binomial": "ΔMcFadden R² (deviance reduction / null deviance)",
                  "lmm": "Δ marginal R² (variance explained by fixed effects)"}[run.family]
        name = lambda n: e(ds.var(n).display_name) if ds and n in ds.variables else e(n)  # noqa: E731
        parts.append("<h3>Settings</h3><table cellpadding='3'>" + "".join(
            f"<tr><td><b>{k}</b></td><td>{v}</td></tr>" for k, v in [
                ("Outcome", name(s.outcome)), ("Model family", fam), ("Effect", effect),
                ("Candidates", ", ".join(name(n) for n in s.candidates)),
                ("Required covariates", ", ".join(name(n) for n in s.required) or "none"),
                ("Max predictors / order", f"{s.max_predictors} / {'two-way' if s.max_order == 2 else 'up to three-way'}"),
                ("Covariate adjustment", "focal (required only)" if s.covariate_mode == "focal" else "all subsets"),
                ("Sample", f"shared complete cases (n = {run.n_sample:,})" if run.n_sample else "complete cases per model"),
                ("α / adjustment", f"{s.alpha:g} / {s.correction}"), ("Mode", s.mode),
                ("Budgets", f"{s.max_models or 'no'} models, {s.max_minutes or 'no'} minutes"),
                ("Order", "deterministic: lexicographic by candidate position (see docs/formula-grammar.md)"),
                ("Elapsed", f"{run.elapsed_s:.1f} s"),
                ("Engine", e(str(run.engine.get('r_version', '')))),
            ]) + "</table>")
        parts.append("<h3>Support tiers</h3><ul>" + "".join(f"<li><b>{k}</b>: {v}</li>" for k, v in engine.TIERS.items())
                     + "</ul><p class='note'>Within a tier, models are ordered by effect size, then by fewer "
                       "parameters. No single score is computed; the lowest p-value is not labelled 'best'.</p>")
        return "".join(parts)

    # ---------- drill-down ----------
    def _selected_row(self) -> dict[str, Any] | None:
        table = self.tabs.currentWidget()
        if not isinstance(table, ResultTable) or self.run is None:
            QMessageBox.information(self, "Discovery", "Select a model in one of the result tables.")
            return None
        idx = table.selected_idx()
        if idx is None:
            QMessageBox.information(self, "Discovery", "Select a model in the table first.")
            return None
        return next(r for r in self.run.rows if r["idx"] == idx)

    def _open_detail(self) -> None:
        row = self._selected_row()
        if row is None:
            return
        if row["status"] in ("invalid", "failed"):
            QMessageBox.information(self, "Discovery", f"This candidate was not fitted: {row['reason']}")
            return
        ds, run = self.state.dataset, self.run
        if run.family == "lmm":
            QMessageBox.information(self, "Discovery", "For mixed-model candidates use the Mixed & Longitudinal "
                                    "Models workspace with the same predictors; View candidate code shows the exact fit.")
            return
        cand = candidate_from_row(run, row)
        self.state.run_r(lambda s: full_analysis(s, ds, run, cand), self._detail_done,
                         lambda m: QMessageBox.warning(self, "Analysis failed", m))

    def _detail_done(self, out) -> None:
        result, artifacts = out
        self.state.add_result(result, artifacts)
        dlg = QDialog(self)
        dlg.setWindowTitle(result.title)
        dlg.resize(1000, 800)
        lay = QVBoxLayout(dlg)
        view = QTextBrowser()
        lay.addWidget(view)
        code = QPushButton("View Code")
        code.clicked.connect(lambda: CodeDialog(result, dlg).exec())
        lay.addWidget(code)
        dlg.show()
        render(view, result, artifacts)

    def _candidate_model(self):
        row = self._selected_row()
        if row is None:
            return None
        if self.run.family == "lmm":
            QMessageBox.information(self, "Discovery", "Mixed-model candidates are refitted in the Mixed & "
                                    "Longitudinal Models workspace (same outcome, participant ID and predictors).")
            return None
        if row["status"] in ("invalid", "failed"):
            QMessageBox.information(self, "Discovery", f"This candidate was not fitted: {row['reason']}")
            return None
        return from_candidate(self.run, row)

    def _to_teaching(self) -> None:
        m = self._candidate_model()
        if m is not None:
            self.state.show_in_teaching.emit(m)

    def _to_set(self) -> None:
        m = self._candidate_model()
        if m is not None:
            self.state.add_model(m)
            QMessageBox.information(self, "Discovery", f"{m.label} added to the Model Comparison set.")

    def _view_code(self) -> None:
        row = self._selected_row()
        if row is None:
            return
        ds, run = self.state.dataset, self.run
        cand = candidate_from_row(run, row)

        def build(session):
            session.ensure_data(ds)
            return candidate_script(session.load_script, ds, run.settings, run.family, cand)

        self.state.run_r(build, lambda script: CodeDialog(
            None, self, title=f"Candidate #{cand.index}", script=script,
            header="The search evaluated this candidate with exactly this function and arguments "
                   "(inside sx_discover_batch, which also records warnings).").exec(),
            lambda m: QMessageBox.warning(self, "Code", m))
