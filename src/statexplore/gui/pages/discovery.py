"""Automated Model Discovery workspace (§10, §11, §20.3–20.5)."""
from __future__ import annotations

import html
import threading
from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QProgressBar, QPushButton, QScrollArea, QSpinBox,
    QSplitter, QTableView, QTabWidget, QTextBrowser, QVBoxLayout, QWidget,
)

from ...core.variables import Measure, Storage
from ...analysis.models import from_candidate
from ...discovery import engine
from ...discovery.detail import candidate_from_row, full_analysis
from ...discovery.rcode import candidate_script
from ...discovery.space import view as search_view
from ...discovery.space import (
    FAMILIES, MAX_PREDICTORS_LIMIT, PROFILES, STRATEGIES, STRUCTURES, SearchSettings, SearchSpaceError,
    describe_space, structures_for, suggest_family,
)
from ...analysis.measurement import TreatmentError, apply_treatments
from ..widgets.treatment_panel import TreatmentPanel
from ..widgets.variable_selector import VariableSelector
from ..models import FrameModel, NumericSortProxy
from ..result_view import render
from ..state import AppState
from .eda import CodeDialog

P_COLS = ("p", "p (adjusted)", "p (BH)", "p (Holm)")
STATUS_COL = "Highlight"
_TEST_COLS = {"status_text": STATUS_COL, "test_id": "Test ID", "idx": "#", "model_type": "Model type",
              "label": "Tested effect", "test_label": "Test type", "formula": "Model", "statistic": "Statistic",
              "df1": "df1", "df_den": "df (denominator)", "p": "p", "p_adj": "p (adjusted)",
              "adj_method": "Adjusted p method", "effect": "Effect", "estimate": "Estimate", "ci_low": "CI low",
              "ci_high": "CI high", "ci_scale": "Effect scale", "ci_status": "CI applicability",
              "predictors_txt": "Predictors", "n_terms": "Model terms", "n_params": "Estimated parameters",
              "n": "n", "n_groups": "Groups (participants)", "diag_text": "Diagnostics status", "flags": "Flags",
              "search_scope": "Search complete/partial", "validation": "Validation status",
              "interpretation": "Teaching interpretation"}
DISCOVERY_COLS = dict(_TEST_COLS)
SUPPORT_COLS = {"status_text": STATUS_COL, "tier": "Tier", **{k: v for k, v in _TEST_COLS.items() if k != "status_text"},
                "aic": "AIC", "r2": "R² (full model)"}
ALL_COLS = {"idx": "#", "model_type": "Model type", "interaction": "Interaction", "formula": "Model",
            "status": "Status", "reason": "Reason", "flags": "Flags", "predictors_txt": "Predictors",
            "n_terms": "Model terms", "n_params": "Estimated parameters", "n_tests": "Hypotheses tested",
            "n": "n", "n_groups": "Groups (participants)", "label": "Focal hypothesis", "p": "p", "p_bh": "p (BH)",
            "p_holm": "p (Holm)", "aic": "AIC", "bic": "BIC", "r2": "R²", "adj_r2": "Adjusted R²",
            "cached": "Reused fit", "speculative": "After stop point", "interpretation": "Teaching interpretation"}
COMPARE_COLS = {"idx": "#", "model_type": "Model type", "formula": "Model", "predictors_txt": "Predictors",
                "n_main": "Main-effect terms", "n_two_way": "Two-way terms", "n_three_way": "Three-way terms",
                "n_terms": "Model terms", "n_params": "Estimated parameters", "n_re_params": "Random-effect parameters",
                "n": "n", "n_groups": "Groups (participants)", "aic": "AIC", "d_aic": "ΔAIC", "bic": "BIC",
                "d_bic": "ΔBIC", "r2": "R²", "adj_r2": "Adjusted R²", "loglik": "log-likelihood",
                "diag_text": "Diagnostics status"}
ALPHA_PRESETS = ("0.05", "0.01", "0.001")


class _Bridge(QObject):
    progress = Signal(object)


class ResultTable(QWidget):
    """Searchable, numerically sortable table over a results frame. Keeps the sort column and
    direction, the search text and the selected row (by its stable ID) when the data refresh."""

    def __init__(self, columns: dict[str, str], id_col: str = "#"):
        super().__init__()
        self.columns = columns
        self.id_col = id_col
        self.highlight_only = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search (variable names, status, flags…)")
        lay.addWidget(self.search)
        self.model = FrameModel(p_columns=P_COLS)
        self.model.status_col = STATUS_COL
        self.proxy = NumericSortProxy()
        self.proxy.setSourceModel(self.model)
        self.search.textChanged.connect(self.proxy.setFilterFixedString)
        self.view = QTableView()
        self.view.setModel(self.proxy)
        self.view.setSortingEnabled(True)
        self.view.sortByColumn(-1, Qt.AscendingOrder)
        self.view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.view.setSelectionMode(QAbstractItemView.SingleSelection)
        lay.addWidget(self.view)
        self._highlight = np.zeros(0, bool)

    def set_frame(self, df: pd.DataFrame, highlight=None, warning=None, text: list[str] | None = None) -> None:
        keep = self.selected_id()
        cols = [c for c in self.columns if c in df.columns]
        frame = df[cols].rename(columns=self.columns) if len(df) else pd.DataFrame(columns=list(self.columns.values()))
        self.model.set_frame(frame)
        self.set_status(highlight, warning, text)
        if 0 < len(df) <= 5000 and not getattr(self, "_sized", False):
            # sizing scans every row: only once, and never for very large searches
            self.view.resizeColumnsToContents()
            for c in range(self.model.columnCount()):
                self.view.setColumnWidth(c, min(self.view.columnWidth(c), 320))
            self._sized = True
        if keep is not None:
            self.select_id(keep)

    def set_status(self, highlight=None, warning=None, text: list[str] | None = None) -> None:
        n = len(self.model.frame)
        self._highlight = np.zeros(n, bool) if highlight is None else np.asarray(highlight, bool)
        self.model.set_row_status(self._highlight, warning, text=text,
                                  tips=None if text is None else [t or None for t in text])
        self._apply_only()

    def set_highlight_only(self, on: bool) -> None:
        self.highlight_only = on
        self._apply_only()

    def _apply_only(self) -> None:
        self.proxy.set_only_rows(set(np.flatnonzero(self._highlight).tolist()) if self.highlight_only else None)

    def counts(self) -> tuple[int, int, int]:
        """(highlighted among shown rows, shown rows, total rows)."""
        shown = [self.proxy.mapToSource(self.proxy.index(r, 0)).row() for r in range(self.proxy.rowCount())]
        hl = int(sum(self._highlight[i] for i in shown if i < len(self._highlight)))
        return hl, len(shown), len(self.model.frame)

    def selected_id(self):
        sel = self.view.selectionModel().selectedRows() if self.view.selectionModel() else []
        if not sel or self.id_col not in self.model.frame.columns:
            return None
        return self.model.frame.iloc[self.proxy.mapToSource(sel[0]).row()][self.id_col]

    def select_id(self, value) -> bool:
        if self.id_col not in self.model.frame.columns:
            return False
        hits = np.flatnonzero((self.model.frame[self.id_col] == value).to_numpy())
        if not len(hits):
            return False
        idx = self.proxy.mapFromSource(self.model.index(int(hits[0]), 0))
        if not idx.isValid():
            return False
        self.view.selectRow(idx.row())
        self.view.scrollTo(idx)
        return True

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
        self.family = QComboBox()
        self.family.addItem("— confirm the outcome model family —", "")
        for key, label in FAMILIES.items():
            self.family.addItem(label, key)
        self.family.setToolTip("Required: StatExplore suggests a family from the outcome's measurement level and "
                               "treatment, but never chooses one silently.")
        self.family.currentIndexChanged.connect(self._family_changed)
        form.addRow("Outcome model family", self.family)
        self.family_hint = QLabel()
        self.family_hint.setWordWrap(True)
        self.family_hint.setObjectName("hint")
        self.use_suggestion = QPushButton("Use suggested family")
        self.use_suggestion.clicked.connect(self._use_suggestion)
        hint_box = QWidget()
        hint_row = QHBoxLayout(hint_box)
        hint_row.setContentsMargins(0, 0, 0, 0)
        hint_row.addWidget(self.family_hint, 1)
        hint_row.addWidget(self.use_suggestion)
        form.addRow(hint_box)
        self.treatment_panel = TreatmentPanel()
        form.addRow(self.treatment_panel.outcome_box)     # outcome treatment directly below the family (M §6.1)
        self.event_combo = QComboBox()
        self.event_combo.setToolTip("Binary logistic: the category whose probability is modelled")
        self.event_combo.currentIndexChanged.connect(self._invalidate_estimate)
        form.addRow("Event category (binary)", self.event_combo)
        self.reference_combo = QComboBox()
        self.reference_combo.setToolTip("Multinomial: every other category is compared with this baseline")
        self.reference_combo.currentIndexChanged.connect(self._invalidate_estimate)
        form.addRow("Reference category (multinomial)", self.reference_combo)
        cl.addLayout(form)
        self._build_study_box(cl)
        cand_box = QGroupBox("Candidate predictors (up to 39)")
        cb = QVBoxLayout(cand_box)
        self.cand_sel = VariableSelector()
        self.cand_sel.setMinimumHeight(220)
        self.cands = self.cand_sel.list
        self.cand_sel.changed.connect(self._selection_changed)
        cb.addWidget(self.cand_sel)
        self.n_sel = QLabel()
        cb.addWidget(self.n_sel)
        cl.addWidget(cand_box, 2)
        req_box = QGroupBox("Required covariates (in every model; count toward the maximum)")
        rb = QVBoxLayout(req_box)
        self.req_sel = VariableSelector()
        self.req_sel.setMinimumHeight(170)
        self.reqs = self.req_sel.list
        self.req_sel.changed.connect(self._selection_changed)
        rb.addWidget(self.req_sel)
        cl.addWidget(req_box, 1)
        self.treatment_panel.changed.connect(self._treatments_changed)
        cl.addWidget(self.treatment_panel)

        opts = QGroupBox("Search settings")
        of = QFormLayout(opts)
        self.strategy = QComboBox()
        for key, text in STRATEGIES.items():
            self.strategy.addItem(text, key)
        self.strategy.setToolTip(
            "All eligible interaction combinations: every hierarchical set of two-way (and, if enabled, "
            "three-way)\ninteractions on every subset of predictors — including models with several "
            "interactions at once.\nSingle focal interaction: the v1.1 grammar (one tested interaction, "
            "optional covariates).\nFull factorial: one model with every interaction per predictor subset.\n"
            "Main effects only: no interactions.")
        self.include_main = QCheckBox("Include main-effects-only models")
        self.include_main.setChecked(True)
        self.include_main.setToolTip("Adds Y ~ X1 + X2 + … models (every predictor subset) with a whole-model "
                                     "test and main-effect term tests, in their own multiple-testing families.")
        self.max_pred = QSpinBox()
        self.max_pred.setRange(1, MAX_PREDICTORS_LIMIT)
        self.max_pred.setValue(MAX_PREDICTORS_LIMIT)
        self.max_pred.setToolTip("Distinct predictors per model (a categorical variable counts once; interaction "
                                 "terms do not count extra).\nRequired covariates count toward it.")
        self.min_pred = QSpinBox()
        self.min_pred.setRange(1, MAX_PREDICTORS_LIMIT)
        self.min_pred.setValue(1)
        self.min_obs = QSpinBox()
        self.min_obs.setRange(0, 10_000_000)
        self.min_obs.setSpecialValueText("no minimum")
        self.order = QComboBox()
        self.order.addItem("Two-way", 2)
        self.order.addItem("Two- and three-way (advanced)", 3)
        self.cov_mode = QComboBox()
        self.cov_mode.addItem("Focal: required covariates only (recommended)", "focal")
        self.cov_mode.addItem("Subsets: every covariate combination (exhaustive)", "subsets")
        self.cov_mode.setToolTip("Single-focal-interaction strategy only. Subsets mode adds extra covariates to "
                                 "each focal interaction;\nit does not by itself add main-effects-only models "
                                 "(use the checkbox above).")
        self.all2 = QComboBox()
        self.all2.addItem("No", False)
        self.all2.addItem("Yes: also test each subset's model with all two-way terms jointly", True)
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
        for label, w in [("Search strategy", self.strategy), ("", self.include_main),
                         ("Max distinct predictors per model", self.max_pred),
                         ("Min distinct predictors per model", self.min_pred),
                         ("Maximum interaction order", self.order),
                         ("Covariate adjustment (focal strategy)", self.cov_mode),
                         ("All-two-way models (subset mode)", self.all2), ("Missing data", self.sample),
                         ("Significance threshold (α)", self.alpha), ("Multiple-testing adjustment", self.correction),
                         ("Search mode", self.mode), ("Target significant models (mode B)", self.target),
                         ("Search profile", self.profile), ("Model budget", self.max_models),
                         ("Time budget (minutes)", self.max_minutes), ("Parallel R workers", self.workers),
                         ("Max parameters per model", self.max_params),
                         ("Minimum complete rows per model", self.min_obs)]:
            if label:
                of.addRow(label, w)
            else:
                of.addRow(w)
        for w in (self.max_pred, self.min_pred, self.order, self.cov_mode, self.sample, self.mode, self.max_models,
                  self.strategy, self.min_obs):
            sig = getattr(w, "valueChanged", None) or w.currentIndexChanged
            sig.connect(self._invalidate_estimate)
        self.include_main.toggled.connect(self._invalidate_estimate)
        self.strategy.currentIndexChanged.connect(self._strategy_changed)
        self._strategy_changed()
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
        hl = QHBoxLayout()
        hl.addWidget(QLabel("<b>Highlight findings:</b>"))
        self.hl_mode = QComboBox()
        for key, text in engine.HIGHLIGHT_MODES.items():
            self.hl_mode.addItem(text, key)
        self.hl_mode.setToolTip("Presentation only: highlighting never refits a model or changes a p-value.\n"
                                "Strict mode needs a confidence interval for the SAME coefficient the p-value "
                                "tests;\njoint (multi-df) tests have none and are never highlighted in strict mode.")
        hl.addWidget(self.hl_mode, 1)
        hl.addWidget(QLabel("α"))
        self.hl_alpha = QComboBox()
        self.hl_alpha.setEditable(True)
        self.hl_alpha.addItems(list(ALPHA_PRESETS))
        self.hl_alpha.setToolTip("Highlight threshold (0 < α < 1); type a custom value")
        self.hl_alpha.setMaximumWidth(90)
        hl.addWidget(self.hl_alpha)
        self.hl_only = QCheckBox("Show highlighted only")
        hl.addWidget(self.hl_only)
        self.hl_count = QLabel()
        hl.addWidget(self.hl_count)
        rl.addLayout(hl)
        frow = QHBoxLayout()
        frow.addWidget(QLabel("Show:"))
        self.view_filter = QComboBox()
        self.view_filter.setToolTip("Hypotheses of different types are never ranked together as if they tested "
                                    "the same thing; each type is its own multiple-testing family.")
        self.view_filter.addItem("All hypotheses", None)
        frow.addWidget(self.view_filter, 1)
        rl.addLayout(frow)
        self.tabs = QTabWidget()
        self.summary_view = QTextBrowser()
        self.discovery_table = ResultTable(DISCOVERY_COLS, id_col="Test ID")
        self.support_table = ResultTable(SUPPORT_COLS, id_col="Test ID")
        self.all_table = ResultTable(ALL_COLS)
        self.compare_table = ResultTable(COMPARE_COLS)
        self.compare_note = QLabel()
        self.compare_note.setWordWrap(True)
        self.compare_note.setObjectName("hint")
        compare_box = QWidget()
        cbl = QVBoxLayout(compare_box)
        cbl.setContentsMargins(0, 0, 0, 0)
        cbl.addWidget(self.compare_note)
        cbl.addWidget(self.compare_table, 1)
        self.tests = pd.DataFrame()
        self._table_frames: dict[int, pd.DataFrame] = {}
        self.hl_mode.currentIndexChanged.connect(self._highlight_changed)
        self.hl_alpha.currentTextChanged.connect(self._highlight_changed)
        self.hl_only.toggled.connect(self._highlight_changed)
        self.view_filter.currentIndexChanged.connect(self._refresh_tables)
        self.tabs.addTab(self.summary_view, "Summary")
        self.tabs.addTab(self.discovery_table, "Discovery significance ranking")
        self.tabs.addTab(self.support_table, "Statistical support ranking")
        self.tabs.addTab(compare_box, "Model comparison")
        self.tabs.addTab(self.all_table, "All attempts")
        rl.addWidget(self.tabs, 1)
        self.tabs.currentChanged.connect(self._update_counts)
        for t in (self.discovery_table, self.support_table, self.all_table, self.compare_table):
            t.search.textChanged.connect(self._update_counts)
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
        self.teach_btn.setToolTip("Longitudinal candidates open in the Mixed & Longitudinal workspace (exact formula "
                                  "and structure, refitted), which draws population and participant trajectories.")
        self.teach_btn.clicked.connect(self._to_teaching)
        self.set_btn = QPushButton("Add to comparison")
        self.set_btn.clicked.connect(self._to_set)
        self.main_btn = QPushButton("Compare with main-effects model")
        self.main_btn.setToolTip("Adds this model and the same model without any interaction (same rows, same "
                                 "variables) to the Model Comparison set.")
        self.main_btn.clicked.connect(self._compare_main)
        for b in (self.open_btn, self.code_btn, self.teach_btn, self.set_btn, self.main_btn):
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

        self._fill_structures()
        self._study_changed()
        state.project_changed.connect(self._reload)
        state.metadata_changed.connect(self._reload_lists)
        state.result_added.connect(lambda r: self._show_run() if r.kind == "discovery_stability" and self.run else None)
        self._reload()

    # ---------- study structure (M §8.1) ----------
    def _build_study_box(self, layout: QVBoxLayout) -> None:
        box = QGroupBox("Study structure")
        f = QFormLayout(box)
        self.study = QComboBox()
        self.study.addItem("Independent observations — no within-participant modelling", "independent")
        self.study.addItem("Repeated measures / longitudinal (participant ID + time)", "longitudinal")
        self.study.setToolTip("Independent observations treats every row as a separate unit. It does NOT turn long "
                              "data into one row per participant:\nwith repeated IDs, p-values are too small. Choose "
                              "repeated measures to model the dependence.")
        f.addRow(self.study)
        self.participant = QComboBox()
        self.participant.setToolTip("Any column with repeated values can identify participants; ID-role variables "
                                    "are listed first. The ID is a grouping variable, never a predictor.")
        self.time_combo = QComboBox()
        self.time_combo.setToolTip("Needed for random slopes, residual correlation, marginal models and the time "
                                   "strategy. Numeric codes are used (labels mapped back).")
        self.time_type = QComboBox()
        self.time_type.addItem("Continuous (linear trend)", "continuous")
        self.time_type.addItem("Categorical (one effect per occasion)", "categorical")
        self.time_fixed = QCheckBox("Include time as a fixed effect in every model (counts as a predictor)")
        self.time_fixed.setChecked(True)
        self.workflow = QComboBox()
        self.workflow.addItem("A: Fixed-effect discovery (lock one structure)", "fixed")
        self.workflow.addItem("B: Compare random-effect / covariance structures (lock the fixed part)", "structure")
        self.workflow.setToolTip("A searches main effects and interactions with one locked structure (ML fits).\n"
                                 "B fits the candidates as main effects (+ optional time interactions) under each "
                                 "structure and compares fit (REML for Gaussian).\nThe two rankings are kept separate.")
        self.structure = QComboBox()
        self.structures_list = QListWidget()
        self.structures_list.setMaximumHeight(140)
        self.time_inter = QCheckBox("Locked fixed part includes time × each candidate")
        self.role_btn = QPushButton("Mark this column as the participant ID in the project")
        self.role_btn.setAutoDefault(False)
        self.role_btn.clicked.connect(self._mark_id_role)
        self.long_info = QTextBrowser()
        self.long_info.setMaximumHeight(170)
        self._long_widgets = [self.participant, self.time_combo, self.time_type, self.time_fixed, self.workflow,
                              self.structure, self.structures_list, self.time_inter, self.role_btn, self.long_info]
        for label, w in [("Participant ID", self.participant), ("", self.role_btn), ("Time", self.time_combo),
                         ("Time enters as", self.time_type), ("", self.time_fixed), ("Workflow", self.workflow),
                         ("Structure (workflow A)", self.structure), ("Structures (workflow B)", self.structures_list),
                         ("", self.time_inter), ("Preflight", self.long_info)]:
            if label:
                f.addRow(label, w)
            else:
                f.addRow(w)
        layout.addWidget(box)
        for w in (self.study, self.participant, self.time_combo, self.time_type, self.workflow, self.structure):
            w.currentIndexChanged.connect(self._study_changed)
        self.time_fixed.toggled.connect(self._study_changed)
        self.time_inter.toggled.connect(self._invalidate_estimate)
        self.structures_list.itemChanged.connect(self._invalidate_estimate)

    def _fill_structures(self) -> None:
        fam = self.family.currentData() or "gaussian"
        keep, keep_list = self.structure.currentData(), set(self._checked_structures())
        self.structure.blockSignals(True)
        self.structure.clear()
        self.structures_list.blockSignals(True)
        self.structures_list.clear()
        for key in structures_for(fam):
            label = STRUCTURES[key][0]
            self.structure.addItem(label, key)
            it = QListWidgetItem(label)
            it.setData(Qt.UserRole, key)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            default = key in ("ri", "rs", "glmer_ri", "glmer_rs", "clmm_ri", "clmm_rs") if not keep_list else key in keep_list
            it.setCheckState(Qt.Checked if default else Qt.Unchecked)
            self.structures_list.addItem(it)
        self.structure.setCurrentIndex(max(0, self.structure.findData(keep)))
        self.structure.blockSignals(False)
        self.structures_list.blockSignals(False)

    def _checked_structures(self) -> list[str]:
        lst = self.structures_list
        return [lst.item(i).data(Qt.UserRole) for i in range(lst.count()) if lst.item(i).checkState() == Qt.Checked]

    def _study_changed(self, *_: Any) -> None:
        long = self.study.currentData() == "longitudinal"
        for w in self._long_widgets:
            w.setEnabled(long)
        if long:
            wf = self.workflow.currentData()
            self.structure.setEnabled(wf == "fixed")
            self.structures_list.setEnabled(wf == "structure")
            self.time_inter.setEnabled(wf == "structure")
        self._update_long_info()
        self._invalidate_estimate()

    def _mark_id_role(self) -> None:
        ds, pid = self.state.dataset, self.participant.currentData()
        if ds is None or not pid:
            return
        ds.set_variable_field(pid, "role", "id")
        self.state.notify_metadata_changed()

    def _update_long_info(self) -> None:
        """Rows, participants, waves and outcome availability by wave on the analysed rows (M §8.1, §8.6)."""
        ds, out = self.state.dataset, self.outcome.currentData()
        if self.study.currentData() != "longitudinal":
            dep = ""
            if ds is not None and out:
                try:
                    from ...discovery.space import dependence_warning
                    dep = dependence_warning(ds, SearchSettings(outcome=out, candidates=self.cand_sel.selected()))
                except Exception:   # an incomplete selection must never break the form
                    dep = ""
            self.long_info.setHtml(f"<p style='color:#c62828'>⚠ {html.escape(dep)}</p>" if dep else "")
            return
        pid, time = self.participant.currentData(), self.time_combo.currentData()
        if ds is None or not out or not pid:
            self.long_info.setHtml("<i>Choose an outcome and a participant ID.</i>")
            return
        try:
            from ...analysis.longitudinal_preflight import preflight
            from ..result_view import table_html

            v = self._search_view()
            rep = preflight(v, out, pid, time, [n for n in self.cand_sel.selected() if n not in (out, pid, time)])
            parts = [f"<p><b>Rows</b> {rep.n_records:,} · <b>Participants</b> {rep.n_participants:,} · "
                     f"<b>Waves</b> {len(rep.time_points) if time else '—'} · participants with 2+ observations "
                     f"{rep.n_multi_obs:,} · duplicate participant-time rows {rep.duplicates}</p>"]
            parts += [f"<p style='color:#8d6e00'>⚠ {html.escape(w)}</p>" for w in rep.warnings]
            parts += [f"<p class='note'>{html.escape(n)}</p>" for n in rep.notes]
            if time:
                parts.append(table_html(rep.tables(v)[1]))
            self.long_info.setHtml("".join(parts))
        except Exception as e:  # an incomplete selection must never break the form
            self.long_info.setHtml(f"<i>{html.escape(str(e))}</i>")

    # ---------- variable lists ----------
    def _reload(self) -> None:
        self.run = None
        self._load_highlight_prefs()
        self._reload_lists()
        self._refresh_history()
        self._show_run()

    def _reload_lists(self) -> None:
        ds = self.state.dataset
        keep_outcome = self.outcome.currentData()
        keep_part, keep_time = self.participant.currentData(), self.time_combo.currentData()
        self.participant.clear()
        self.time_combo.clear()
        self.time_combo.addItem("(none)", None)
        ids, others, times, tothers = [], [], [], []
        self.outcome.blockSignals(True)
        self.outcome.clear()
        self.outcome.addItem("— choose —", None)
        self._pool: list[str] = []
        if ds is not None:
            for v in ds.variables.values():
                if v.role.value in ("excluded",) or v.storage is Storage.DATETIME:
                    continue
                n_values = ds.analysis_frame([v.name])[v.name].dropna().nunique()
                if n_values == 2 or (v.measure is Measure.CONTINUOUS and v.storage is Storage.NUMERIC) or \
                        (v.is_categorical and 3 <= n_values <= 15):
                    self.outcome.addItem(f"{v.display_name}  ·  {v.measure.value}", v.name)
                (ids if v.role.value == "id" else others).append(v)
                if v.storage is Storage.NUMERIC:
                    (times if v.role.value == "time" else tothers).append(v)
                if v.role.value in ("id",) or (not v.is_categorical and v.storage is not Storage.NUMERIC):
                    continue
                if n_values <= 1:
                    continue  # constant in the analysed rows
                self._pool.append(v.name)
        # any structurally valid column may be the ID or time; configured roles are suggested first (M §8.1)
        for v in ids:
            self.participant.addItem(f"{v.display_name}  (ID role)", v.name)
        for v in others:
            self.participant.addItem(v.display_name, v.name)
        for v in times:
            self.time_combo.addItem(f"{v.display_name}  (time role)", v.name)
        for v in tothers:
            self.time_combo.addItem(v.display_name, v.name)
        self.participant.setCurrentIndex(max(0, self.participant.findData(keep_part)))
        self.time_combo.setCurrentIndex(max(0, self.time_combo.findData(keep_time)))
        idx = self.outcome.findData(keep_outcome)
        self.outcome.setCurrentIndex(max(0, idx))
        self.outcome.blockSignals(False)
        self._outcome_changed()

    def _outcome_changed(self) -> None:
        ds = self.state.dataset
        out = self.outcome.currentData()
        names = [n for n in getattr(self, "_pool", []) if n != out]
        deco = (lambda n: f"  ·  {ds.var(n).measure.value}") if ds is not None else None
        for sel in (self.cand_sel, self.req_sel):
            sel.blockSignals(True)
            sel.set_variables(ds, names, decorate=deco)
            sel.blockSignals(False)
        self._fill_outcome_levels()
        self._selection_changed()

    def _fill_outcome_levels(self) -> None:
        ds = self.state.dataset
        out = self.outcome.currentData()
        for combo, first in ((self.event_combo, "(second category)"), (self.reference_combo, "(first category)")):
            keep = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(first, None)
            if ds is not None and out:
                for code in ds.levels(out)[:15]:
                    c = str(int(code)) if isinstance(code, float) and code.is_integer() else str(code)
                    combo.addItem(f"{c} = {ds.level_label(out, code)}", code)
            combo.setCurrentIndex(max(0, combo.findData(keep)) if keep is not None else 0)
            combo.blockSignals(False)
        self._family_changed()

    def _search_view(self):
        ds = self.state.dataset
        try:
            return apply_treatments(ds, self.treatment_panel.treatments())
        except TreatmentError:
            return ds

    def _suggested(self) -> tuple[str, str]:
        ds, out = self.state.dataset, self.outcome.currentData()
        if ds is None or not out:
            return "", ""
        return suggest_family(self._search_view(), out)

    def _family_changed(self, *_: Any) -> None:
        fam = self.family.currentData()
        if hasattr(self, "structure"):
            self._fill_structures()
        self.event_combo.setEnabled(fam == "binomial")
        self.reference_combo.setEnabled(fam == "multinomial")
        key, why = self._suggested()
        if key:
            label = FAMILIES[key].split(" (")[0]
            status = "" if fam else " <b>Confirm a family to continue.</b>"
            self.family_hint.setText(f"Suggested: {label}. {html.escape(why)}{status}")
        else:
            self.family_hint.setText(html.escape(why))
        self.use_suggestion.setEnabled(bool(key) and key != fam)
        self._invalidate_estimate()

    def _use_suggestion(self) -> None:
        key, _ = self._suggested()
        if key:
            self.family.setCurrentIndex(self.family.findData(key))

    def _treatments_changed(self) -> None:
        self._family_changed()

    @staticmethod
    def _checked(lst: QListWidget) -> list[str]:
        return [lst.item(i).data(Qt.UserRole) for i in range(lst.count()) if lst.item(i).checkState() == Qt.Checked]

    def _selection_changed(self, *_: Any) -> None:
        n = len(self.cand_sel.selected())
        self.n_sel.setText("Too many candidates (maximum 39)." if n > 39 else "")
        out = self.outcome.currentData()
        self.treatment_panel.set_model(self.state.dataset, out, self.cand_sel.selected() + self.req_sel.selected())
        if hasattr(self, "long_info"):
            self._update_long_info()
        self._invalidate_estimate()

    def _invalidate_estimate(self, *_: Any) -> None:
        self.estimate.setText("")

    def _strategy_changed(self, *_: Any) -> None:
        st = self.strategy.currentData()
        self.cov_mode.setEnabled(st == "focal")
        self.all2.setEnabled(st == "focal")
        self.include_main.setEnabled(st != "main")

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
        family = self.family.currentData()
        if not family:
            key, why = self._suggested()
            raise SearchSpaceError("Confirm the outcome model family" + (
                f" (suggested: {FAMILIES[key].split(' (')[0]} — {why})" if key else "") + ".")
        long = self.study.currentData() == "longitudinal"
        time = self.time_combo.currentData() if long else None
        required = [n for n in self.req_sel.selected() if n != outcome]
        if long and time and self.time_fixed.isChecked() and self.strategy.currentData() != "time" \
                and time not in required:
            required.append(time)          # time as a fixed effect: counts toward the predictor limit
        return SearchSettings(
            outcome=outcome, candidates=[n for n in self.cand_sel.selected() if n not in (outcome, time)],
            required=required,
            family=family,
            event=self.event_combo.currentData() if family == "binomial" else None,
            reference=self.reference_combo.currentData() if family == "multinomial" else None,
            treatments=[t.to_dict() for t in self.treatment_panel.treatments()],
            max_predictors=self.max_pred.value(), max_order=self.order.currentData(),
            covariate_mode=self.cov_mode.currentData(), alpha=self.alpha.value(),
            correction=self.correction.currentData(), mode=self.mode.currentData(), target=self.target.value(),
            profile=self.profile.currentData(), max_models=self.max_models.value() or None,
            max_minutes=self.max_minutes.value() or None, workers=self.workers.value(),
            sample=self.sample.currentData(), max_params=self.max_params.value(),
            participant=self.participant.currentData() if long else None,
            study="longitudinal" if long else "independent",
            time=self.time_combo.currentData() if long else None, time_type=self.time_type.currentData(),
            structure=self.structure.currentData() or "ri", workflow=self.workflow.currentData() if long else "fixed",
            structures=self._checked_structures(), time_interactions=self.time_inter.isChecked(),
            all_two_way=bool(self.all2.currentData()) and self.cov_mode.currentData() == "subsets"
            and self.strategy.currentData() == "focal",
            strategy=self.strategy.currentData(), include_main=self.include_main.isChecked(),
            min_predictors=self.min_pred.value(), min_obs=self.min_obs.value(),
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
        c = info["coding"]
        fam = f"{html.escape(c['family'])}; focal test: {html.escape(c['focal_test'])}"
        extra = []
        for key, label in (("event", "event"), ("reference", "reference"), ("ordering", "order")):
            if c.get(key):
                val = " < ".join(c[key]) if isinstance(c[key], list) else c[key]
                extra.append(f"{label}: {html.escape(str(val))}")
        extra += [html.escape(t) for t in c.get("treatments", [])]
        coding_txt = ("<br>" + " · ".join(extra)) if extra else ""
        warn = "".join(f"<br><span style='color:#8d6e00'>⚠ {html.escape(w)}</span>" for w in info["warnings"])
        blocks = " + ".join(f"{n:,} {name}" for name, n in info.get("blocks", [])) if len(info.get("blocks", [])) > 1 else ""
        work = (f" · workload: up to <b>{info['fits']:,}</b> model fits"
                + (f" ({blocks})" if blocks else "")) if info.get("fits") else ""
        self.estimate.setText(f"<b>{info['total']:,}</b> candidate models{work} · {fam} · shared complete-case "
                              f"sample: <b>{info['n_shared']:,}</b> of {info['n_rows']:,} rows{coding_txt}{warn}")
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
        previous = list(self.state.project.searches) if self.state.project else []
        cache = engine.cache_from(run, previous, ds)
        self._set_running(True)
        self.progress.setRange(0, max(1, run.total))
        self.state.run_io(
            lambda: engine.execute(run, ds, rscript=rscript, cancel=cancel, cache=cache,
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
            self.frame, self.tests = pd.DataFrame(), pd.DataFrame()
            for t in (self.discovery_table, self.support_table, self.all_table, self.compare_table):
                t.set_frame(pd.DataFrame())
            self._update_counts()
            return
        self.frame = engine.add_display_columns(engine.results_frame(run), run)
        self.tests = engine.add_display_columns(engine.tests_frame(run), run)
        self.summary_view.setHtml(self._summary_html(run, self.frame))
        self._fill_view_filter()
        self._refresh_tables()
        self._refresh_history()

    def _fill_view_filter(self) -> None:
        keep = self.view_filter.currentData()
        self.view_filter.blockSignals(True)
        self.view_filter.clear()
        self.view_filter.addItem("All hypotheses", None)
        df = self.tests
        if len(df):
            for t in df["test_type"].dropna().unique():
                self.view_filter.addItem(f"Hypothesis: {engine.TEST_TYPES.get(t, t)}", f"test_type={t}")
            for m in df["model_type"].dropna().unique():
                self.view_filter.addItem(f"Models: {m}", f"model_type={m}")
        self.view_filter.setCurrentIndex(max(0, self.view_filter.findData(keep)))
        self.view_filter.blockSignals(False)

    def _filtered(self, df: pd.DataFrame) -> pd.DataFrame:
        f = self.view_filter.currentData()
        if not len(df) or not f:
            return df
        col, val = f.split("=", 1)
        return df[df[col] == val] if col in df else df

    def _refresh_tables(self) -> None:
        if self.run is None:
            return
        tf = self._filtered(self.tests)
        mf = self._filtered(self.frame) if str(self.view_filter.currentData()).startswith("model_type=") else self.frame
        cmp_df, note = engine.model_comparison(mf, self.run) if len(mf) else (mf, "")
        self.compare_note.setText(note)
        frames = {id(self.discovery_table): engine.discovery_ranking(tf, by=self.rank_by.currentData()) if len(tf) else tf,
                  id(self.support_table): engine.support_ranking(tf) if len(tf) else tf,
                  id(self.all_table): mf, id(self.compare_table): cmp_df}
        self._table_frames = frames
        for table in (self.discovery_table, self.support_table, self.all_table, self.compare_table):
            df = frames[id(table)]
            hl, warn, text = self._status(df)
            table.set_frame(df, hl, warn, text)
            table.set_highlight_only(self.hl_only.isChecked())
        self._update_counts()

    def _alpha(self) -> float:
        try:
            a = float(self.hl_alpha.currentText())
        except ValueError:
            return 0.05
        return a if 0 < a < 1 else 0.05

    def _status(self, df: pd.DataFrame):
        if not len(df):
            return None, None, None
        mode, alpha = self.hl_mode.currentData(), self._alpha()
        hl = engine.highlight_mask(df, mode, alpha).to_numpy()
        warn = engine.warning_mask(df).to_numpy()
        tag = {"adjusted": f"adj p < {alpha:g}", "raw": f"p < {alpha:g}",
               "strict": f"adj p < {alpha:g} + CI excludes null"}.get(mode, "")
        text = [" · ".join(x for x in ((f"★ {tag}" if h else ""), ("⚠ flagged" if w else "")) if x)
                for h, w in zip(hl, warn)]
        return hl, warn, text

    def _highlight_changed(self, *_: Any) -> None:
        for table in (self.discovery_table, self.support_table, self.all_table, self.compare_table):
            df = self._table_frames.get(id(table))
            if df is not None and len(df):
                table.set_status(*self._status(df))
            table.set_highlight_only(self.hl_only.isChecked())
        self._update_counts()
        p = self.state.project
        if p is not None:
            prefs = {"mode": self.hl_mode.currentData(), "alpha": self._alpha(), "only": self.hl_only.isChecked()}
            if p.settings.get("discovery_highlight") != prefs:
                p.settings["discovery_highlight"] = prefs
                self.state.dirty = True

    def _load_highlight_prefs(self) -> None:
        p = self.state.project
        prefs = (p.settings.get("discovery_highlight") if p is not None else None) or {}
        for w in (self.hl_mode, self.hl_alpha, self.hl_only):
            w.blockSignals(True)
        self.hl_mode.setCurrentIndex(max(0, self.hl_mode.findData(prefs.get("mode", "adjusted"))))
        self.hl_alpha.setCurrentText(f"{prefs.get('alpha', 0.05):g}")
        self.hl_only.setChecked(bool(prefs.get("only", False)))
        for w in (self.hl_mode, self.hl_alpha, self.hl_only):
            w.blockSignals(False)

    def _current_table(self) -> "ResultTable | None":
        w = self.tabs.currentWidget()
        if w is not None and not isinstance(w, ResultTable):
            w = w.findChild(ResultTable)
        return w if isinstance(w, ResultTable) else None

    def _update_counts(self) -> None:
        table = self._current_table()
        if isinstance(table, ResultTable):
            hl, shown, total = table.counts()
            self.hl_count.setText(f"{hl:,} highlighted of {shown:,} shown / {total:,} total")
        else:
            self.hl_count.setText("")

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
            fams = sm.get("families") or []
            if len(fams) > 1 or (fams and fams[0]["type"] != "focal_interaction"):
                parts.append("<p><b>Hypotheses by type</b> (each type is its own multiple-testing family; "
                             "different types are never ranked together):</p><ul>" + "".join(
                                 f"<li>{e(f['label'])}: {f['tests']:,} tests · {f['sig_raw']:,} raw p &lt; "
                                 f"{s.alpha:g} (≈{f['expected_false_raw']:.1f} expected by chance) · "
                                 f"{f['sig_adj']:,} after adjustment"
                                 + (f" · {f['no_ci']:,} joint/without a comparable single-coefficient CI"
                                    if f["no_ci"] else "") + "</li>" for f in fams) + "</ul>")
            if sm["provisional"]:
                parts.append("<p style='color:#8d6e00'>Adjusted p-values are <b>provisional</b>: they are "
                             "conditional on the candidates evaluated, not the full search space.</p>")
        parts.append(
            "<p class='note'>Multiple-testing adjustment controls false discoveries among the tests run; it does "
            "not remove post-selection bias. Effect sizes and confidence intervals of the top-ranked models are "
            "optimistic. Confirm promising interactions on new data or a holdout sample (Phase 6).</p>")
        fam = {"gaussian": "Linear models, focal F-test (nested comparison)",
               "binomial": "Logistic models, focal likelihood-ratio test",
               "ordinal": "Ordinal logistic models (ordinal::clm), focal likelihood-ratio test",
               "multinomial": "Multinomial logistic models (nnet::multinom), focal likelihood-ratio test "
                              "(joint over all outcome contrasts)",
               "lmm": "Gaussian longitudinal models (structure below); likelihood-ratio tests on ML fits "
                      "(Groups column = participants; never a denominator df)",
               "glmm": "Binary logistic mixed models (lme4::glmer, Laplace ML); likelihood-ratio tests; "
                       "subject-specific log-odds",
               "clmm": "Ordinal cumulative-logit mixed models (ordinal::clmm); likelihood-ratio tests; "
                       "subject-specific log-odds; proportional odds assumed",
               "gee": "GEE (geepack::geeglm): population-average effects, robust sandwich Wald tests; no likelihood "
                      "(QIC instead of AIC)"}[run.family]
        effect = {"gaussian": "ΔR² (variance uniquely explained by the interaction)",
                  "binomial": "ΔMcFadden R² (deviance reduction / null deviance)",
                  "ordinal": "ΔMcFadden pseudo-R² (LR statistic / -2 log-likelihood of the null model)",
                  "multinomial": "ΔMcFadden pseudo-R² (deviance reduction / null deviance)",
                  "lmm": "Δ marginal R² (random-intercept lme4 fits only; otherwise not computed)",
                  "glmm": "not computed (use the LR statistic and odds ratios)",
                  "clmm": "not computed (use the LR statistic and odds ratios)",
                  "gee": "not computed (use the Wald statistic and coefficients)"}[run.family]
        coding_rows = []
        c = run.coding or {}
        if c.get("event"):
            coding_rows.append(("Event / reference", f"{e(c['event'])} vs {e(c.get('reference', ''))} "
                                                     f"({e(c.get('event_source', ''))})"))
        elif c.get("reference"):
            coding_rows.append(("Reference category", e(c["reference"])))
        if c.get("ordering"):
            coding_rows.append(("Outcome order", e(" < ".join(c["ordering"]))
                                + (f" ({e(c['direction'])})" if c.get("direction") else "")))
        for t in c.get("treatments", []):
            coding_rows.append(("Variable treatment", e(t)))
        for key, label in (("workflow", "Longitudinal workflow"), ("participant", "Participant ID"),
                           ("time", "Time"), ("structure", "Random / covariance structure"),
                           ("estimand", "Estimand")):
            if c.get(key):
                coding_rows.append((label, e(c[key])))
        if c.get("predictor_coding"):
            coding_rows.append(("Category coding", e(c["predictor_coding"])))
        name = lambda n: e(ds.var(n).display_name) if ds and n in ds.variables else e(n)  # noqa: E731
        parts.append("<h3>Settings</h3><table cellpadding='3'>" + "".join(
            f"<tr><td><b>{k}</b></td><td>{v}</td></tr>" for k, v in [
                ("Outcome", name(s.outcome)), ("Model family", fam), ("Effect", effect), *coding_rows,
                ("Candidates", ", ".join(name(n) for n in s.candidates)),
                ("Required covariates", ", ".join(name(n) for n in s.required) or "none"),
                ("Strategy", e(STRATEGIES.get(getattr(s, "strategy", "focal"), "")) + (
                    " + main-effects-only models" if getattr(s, "include_main", False) and s.strategy != "main" else "")),
                ("Max predictors / order", f"{s.max_predictors} distinct / {'two-way' if s.max_order == 2 else 'up to three-way'}"),
                ("Covariate adjustment", ("focal (required only)" if s.covariate_mode == "focal" else "all subsets")
                 if getattr(s, "strategy", "focal") == "focal" else "—"),
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
        table = self._current_table()
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
        if run.settings.participant:
            self._to_mixed(row)
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
        if self.run.settings.participant:
            self._to_mixed(row)
            return None
        if row["status"] in ("invalid", "failed"):
            QMessageBox.information(self, "Discovery", f"This candidate was not fitted: {row['reason']}")
            return None
        return from_candidate(self.run, row)

    def _to_mixed(self, row: dict[str, Any]) -> None:
        """Longitudinal candidates: refit the exact formula and structure in the Mixed workspace, which draws
        population-average and participant-specific trajectories (M §8.7). The result is labelled a refit."""
        from ...analysis.mixed import MixedInputError, spec_from_discovery

        if row["status"] in ("invalid", "failed"):
            QMessageBox.information(self, "Discovery", f"This candidate was not fitted: {row['reason']}")
            return
        try:
            spec = spec_from_discovery(self.run, row)
        except MixedInputError as e:
            QMessageBox.information(self, "Discovery", str(e))
            return
        self.state.show_in_mixed.emit(spec)

    def _to_teaching(self) -> None:
        m = self._candidate_model()
        if m is not None:
            self.state.show_in_teaching.emit(m)

    def _to_set(self) -> None:
        m = self._candidate_model()
        if m is not None:
            self.state.add_model(m)
            QMessageBox.information(self, "Discovery", f"{m.label} added to the Model Comparison set.")

    def _compare_main(self) -> None:
        """Interaction model vs its main-effects-only version on the same sample (M §11). Hierarchy is kept:
        the interaction model contains every main effect, so the two are nested."""
        row = self._selected_row()
        if row is None:
            return
        if self.run.settings.participant:
            QMessageBox.information(
                self, "Discovery", "For longitudinal searches, the 'Joint test of all interaction terms' (or the "
                "single interaction test) in this search already compares the model with its main-effects version "
                "under the same random-effect / covariance structure, ML fits and rows.")
            return
        m = self._candidate_model()
        if m is None:
            return
        if not any(len(t) > 1 for t in m.terms):
            QMessageBox.information(self, "Discovery", "This model has no interaction terms.")
            return
        mains = m.copy_edited([t for t in m.terms if len(t) == 1], label=f"{m.label} — main effects only")
        mains.focal = None
        self.state.add_model(m)
        self.state.add_model(mains)
        QMessageBox.information(self, "Discovery", f"{m.label} and its main-effects-only version were added to the "
                                "Model Comparison set (same rows). The nested F / likelihood-ratio test there asks "
                                "whether the interaction terms improve the model jointly.")

    def _view_code(self) -> None:
        row = self._selected_row()
        if row is None:
            return
        ds, run = self.state.dataset, self.run
        cand = candidate_from_row(run, row)

        def build(session):
            v = search_view(ds, run.settings)
            session.ensure_data(v)
            return candidate_script(session.load_script, v, run.settings, run.family, cand)

        self.state.run_r(build, lambda script: CodeDialog(
            None, self, title=f"Candidate #{cand.index}", script=script,
            header="The search evaluated this candidate with exactly this function and arguments "
                   "(inside sx_discover_batch, which also records warnings).").exec(),
            lambda m: QMessageBox.warning(self, "Code", m))
