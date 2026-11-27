"""Automated Model Discovery workspace (§10, §11, §20.3–20.5)."""
from __future__ import annotations

from pathlib import Path

import html
import threading
from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QProgressBar, QPushButton, QSpinBox,
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
from ..widgets.layout import section_scroll
from ..widgets.treatment_panel import TreatmentPanel
from ..widgets.variable_selector import VariableSelector
from ..models import ResultFrameModel
from ..result_view import render
from ..state import AppState
from .eda import CodeDialog
from ..widgets.searchable import make_searchable

P_COLS = ("p", "p (adjusted)", "p (BH)", "p (Holm)")
STATUS_COL = "Highlight"
_TEST_COLS = {"status_text": STATUS_COL, "test_id": "Test ID", "idx": "#", "model_type": "Model type",
              "label": "Tested effect", "test_label": "Test type", "formula": "Model", "statistic": "Statistic",
              "reference": "Reference model", "test_method": "Test method",
              "df1": "df1", "df_den": "df (denominator)", "p": "p", "log10_p": "log10 p", "p_adj": "p (adjusted)",
              "adj_method": "Adjusted p method", "effect": "Effect", "estimate": "Estimate", "ci_low": "CI low",
              "ci_high": "CI high", "ci_scale": "Effect scale", "odds_ratio": "Odds ratio", "or_low": "OR CI low",
              "or_high": "OR CI high", "ci_status": "CI applicability",
              "predictors_txt": "Predictors", "n_terms": "Model terms", "n_params": "Fixed coefficients",
              "n_lik": "Likelihood parameters", "estimation": "Estimation",
              "n": "n", "n_groups": "Groups (participants)", "observed_clusters": "Clusters (observed)",
              "modeled_clusters": "Clusters (modeled)", "diag_text": "Diagnostics status", "flags": "Flags",
              "test_status": "Test status", "reference_distribution": "Reference distribution",
              "search_scope": "Search complete/partial", "validation": "Validation status",
              "interpretation": "Teaching interpretation"}
DISCOVERY_COLS = dict(_TEST_COLS)
SUPPORT_COLS = {"status_text": STATUS_COL, "tier": "Tier", **{k: v for k, v in _TEST_COLS.items() if k != "status_text"},
                "aic": "AIC", "r2": "R² (full model)"}
ALL_COLS = {"idx": "#", "model_type": "Model type", "interaction": "Interaction", "formula": "Model",
            "status": "Status", "reason": "Reason", "flags": "Flags", "predictors_txt": "Predictors",
            "n_terms": "Model terms", "n_params": "Fixed coefficients", "n_re_params": "Covariance parameters",
            "n_lik": "Likelihood parameters", "estimation": "Estimation", "n_tests": "Hypotheses tested",
            "n": "n", "n_groups": "Groups (participants)", "label": "Focal hypothesis", "p": "p", "p_bh": "p (BH)",
            "p_holm": "p (Holm)", "aic": "AIC", "bic": "BIC", "r2": "R²", "adj_r2": "Adjusted R²",
            "cached": "Reused fit", "speculative": "After stop point", "interpretation": "Teaching interpretation"}
COMPARE_COLS = {"idx": "#", "model_type": "Model type", "formula": "Model", "predictors_txt": "Predictors",
                "n_main": "Main-effect terms", "n_two_way": "Two-way terms", "n_three_way": "Three-way terms",
                "n_terms": "Model terms", "n_params": "Fixed coefficients", "n_re_params": "Covariance parameters",
                "n_lik": "Likelihood parameters", "estimation": "Estimation", "structure_label": "Structure",
                "n": "n", "n_groups": "Groups (participants)", "observed_clusters": "Clusters (observed)",
                "modeled_clusters": "Clusters (modeled)", "aic": "AIC", "d_aic": "ΔAIC", "bic": "BIC",
                "bic_n": "BIC n", "d_bic": "ΔBIC", "bic_harmonized": "BIC (harmonized, n − p)",
                "d_bic_harmonized": "ΔBIC (harmonized)", "qic": "QIC", "d_qic": "ΔQIC", "qicu": "QICu",
                "d_qicu": "ΔQICu", "cic": "CIC", "r2": "R²", "adj_r2": "Adjusted R²",
                "loglik": "log-likelihood", "diag_text": "Diagnostics status"}
ALPHA_PRESETS = ("0.05", "0.01", "0.001")
COLUMN_TIPS = {
    "log10 p": "log10 of the p-value at full precision (from R's log upper tail when recorded): orders p-values below "
               "the smallest double too. Blank = no p-value (withheld or not applicable), not p = 0.",
    "Fixed coefficients": "Fixed-effect coefficients including the intercept; ordinal models add the K-1 thresholds, "
                          "multinomial models count (K-1) x coefficients. Used for the configured parameter limit.",
    "Covariance parameters": "Random-effect variances/covariances plus residual variance and correlation parameters "
                             "(longitudinal models); GEE: working-correlation parameters.",
    "Likelihood parameters": "Every parameter of the fitted likelihood (fixed + thresholds + covariance; linear "
                             "models include the residual variance). This is the k in AIC = -2 logLik + 2k.",
    "Estimation": "ML, REML (covariance-structure comparisons with a locked fixed part), OLS or GEE. Never compare "
                  "REML likelihoods of models with different fixed effects.",
    "Groups (participants)": "Participants contributing rows to the fit (not a denominator df).",
    "Test type": "Hypothesis family; multiple-testing adjustment is applied within each family.",
    "p (adjusted)": "Adjusted within this test type's family only (BH or Holm as selected).",
    "Highlight": "Strict highlights also need a same-estimand single-coefficient CI that excludes the null; joint and "
                 "covariance tests never qualify, so zero strict highlights does not mean zero significant tests.",
    "Structure": "Random-effect / residual-correlation structure. AR(1) uses the observed-occasion index; CAR(1) uses "
                 "actual time differences.",
    "log-likelihood": "From the estimation method in the Estimation column (ML or REML).",
    "Clusters (observed)": "Higher-level clusters in the analysed rows.",
    "Clusters (modeled)": "Clusters with a random intercept in this model (0 = exploratory fit without the cluster).",
    "Test status": "tested = clean; provisional = a compared fit is flagged (not clean evidence); withheld = no valid "
                   "reference distribution or an unreliable likelihood (no p-value).",
    "Reference distribution": "The null distribution used for this test: chi-square, or a 50:50 mixture when a variance "
                              "or a CAR(1) correlation is tested on the boundary of its range.",
    "QIC": "GEE quasi-likelihood information criterion (Pan 2001; geepack::QIC of the actual fit): compares working "
           "correlations and mean models on the same rows. Not an AIC.",
    "QICu": "QIC with the penalty 2p: for comparing mean models under the same working correlation.",
    "CIC": "Correlation information criterion (trace term of QIC): compares working correlations.",
    "BIC n": "Sample size in the native BIC: n for lme4, n − p for REML nlme fits.",
    "BIC (harmonized, n − p)": "Every model's BIC recomputed on one basis (n − p) when native BICs use different n.",
}


class _Bridge(QObject):
    progress = Signal(object)


class ResultTable(QWidget):
    """Searchable, numerically sortable table over a results frame. Sorting and searching are vectorised
    (fast for hundreds of thousands of rows). Keeps the sort column and direction, the search text and
    the selected row (by its stable ID) when the data refresh."""

    selected = Signal()          # selection changed
    activated = Signal()         # double-click: expand the selected model's details

    def __init__(self, columns: dict[str, str], id_col: str = "#", name: str = ""):
        super().__init__()
        self.columns = columns
        self.id_col = id_col
        self.name = name
        self.highlight_only = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search (variable names, status, flags…)")
        top.addWidget(self.search, 1)
        copy = QPushButton("Copy CSV")
        copy.setToolTip("Copy the rows shown (current filter, search and sort; visible columns) as CSV")
        copy.clicked.connect(self.copy_csv)
        top.addWidget(copy)
        save = QPushButton("Save CSV…")     # V133-UI-07: a file, not only the clipboard (full precision)
        save.setToolTip("Save the rows shown (current filter, search and sort; visible columns) as a CSV file at full "
                        "numeric precision")
        save.clicked.connect(lambda: self.save_csv())
        top.addWidget(save)
        self.copy_note = QLabel("")         # UI-41: say what was copied (or that copying failed)
        self.copy_note.setObjectName("hint")
        top.addWidget(self.copy_note)
        lay.addLayout(top)
        self.model = ResultFrameModel(p_columns=P_COLS)
        self.model.header_tips = COLUMN_TIPS
        self.model.status_col = STATUS_COL
        self.proxy = self.model            # the model sorts and filters itself (kept name for callers)
        self.search.textChanged.connect(self._query)
        self.view = QTableView()
        self.view.setModel(self.model)
        self.view.setSortingEnabled(True)
        self.view.horizontalHeader().setSortIndicator(-1, Qt.AscendingOrder)
        self.view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.view.setSelectionMode(QAbstractItemView.SingleSelection)
        # UI-28: compact rows, reorderable / resizable / hideable columns, per-pixel horizontal scrolling
        self.view.setWordWrap(False)
        self.view.verticalHeader().setDefaultSectionSize(self.view.fontMetrics().height() + 6)
        self.view.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        hdr = self.view.horizontalHeader()
        hdr.setSectionsMovable(True)
        hdr.setContextMenuPolicy(Qt.CustomContextMenu)
        hdr.customContextMenuRequested.connect(self._column_menu)
        hdr.sectionResized.connect(lambda *_: self._save_header())
        hdr.sectionMoved.connect(lambda *_: self._save_header())
        self.view.doubleClicked.connect(lambda *_: self.activated.emit())
        hdr.setMinimumSectionSize(44)                      # UI-34: no column collapses to an unreadable sliver
        lay.addWidget(self.view)
        # UI-34: hint when columns continue off-screen; frozen identification column; density / font preference
        self.scroll_hint = QLabel("More columns to the right → scroll sideways, or right-click a column header to "
                                  "choose columns, fit them to their contents or change the row density.")
        self.scroll_hint.setObjectName("hint")
        self.scroll_hint.setWordWrap(True)
        self.scroll_hint.setVisible(False)
        lay.addWidget(self.scroll_hint)
        self.view.horizontalScrollBar().rangeChanged.connect(
            lambda _lo, hi: self.scroll_hint.setVisible(hi > 0 and self.model.rowCount() > 0))
        self._frozen: QTableView | None = None
        self._highlight = np.zeros(0, bool)
        self._restored = False
        self._apply_density()

    def _settings(self):
        from PySide6.QtCore import QSettings

        return QSettings("StatExplore", "StatExplore")

    def _layout_key(self) -> str:
        """Column layouts are stored per set of columns (longitudinal searches show extra columns)."""
        import hashlib

        sig = hashlib.sha1("|".join(map(str, self.model.frame.columns)).encode()).hexdigest()[:10]
        return f"discovery/columns/{self.name}/{sig}"

    def _save_header(self) -> None:
        if self.name and self._restored:
            self._settings().setValue(self._layout_key(), self.view.horizontalHeader().saveState())

    def _restore_header(self) -> None:
        cols = tuple(self.model.frame.columns)
        if cols != getattr(self, "_layout_cols", None):
            self._layout_cols, self._restored = cols, False
        if self._restored or not self.name or self.model.columnCount() == 0:
            return
        state = self._settings().value(self._layout_key())
        self._restored = True
        if state is not None:
            self.view.horizontalHeader().restoreState(state)
            self.view.horizontalHeader().setMinimumSectionSize(44)    # layouts saved by older versions had 16
            self._sized = True

    def _column_menu(self, pos) -> None:
        from PySide6.QtWidgets import QMenu

        hdr = self.view.horizontalHeader()
        menu = QMenu(self)
        for c in range(self.model.columnCount()):
            act = menu.addAction(str(self.model.frame.columns[c]))
            act.setCheckable(True)
            act.setChecked(not hdr.isSectionHidden(c))
            act.toggled.connect(lambda on, c=c: (hdr.setSectionHidden(c, not on), self._save_header()))
        menu.addSeparator()
        menu.addAction("Show all columns", lambda: ([hdr.setSectionHidden(c, False)
                                                     for c in range(self.model.columnCount())], self._save_header()))
        menu.addAction("Fit columns to contents", self.fit_columns)
        menu.addAction("Reset column layout", self.reset_columns)
        frz = menu.addAction(f"Freeze the {self.id_col} column")
        frz.setCheckable(True)
        frz.setChecked(self._pref("freeze", True))
        frz.toggled.connect(lambda on: (self._set_pref("freeze", on), self._update_frozen()))
        dens = menu.addMenu("Row density")
        for key, text in (("compact", "Compact"), ("comfortable", "Comfortable")):
            a = dens.addAction(text)
            a.setCheckable(True)
            a.setChecked(self._pref("density", "compact") == key)
            a.triggered.connect(lambda _=False, k=key: (self._set_pref("density", k), self._apply_density()))
        size = menu.addMenu("Table font size")
        for delta, text in ((-1, "Smaller"), (0, "Normal"), (1, "Larger"), (2, "Largest")):
            a = size.addAction(text)
            a.setCheckable(True)
            a.setChecked(int(self._pref("font", 0)) == delta)
            a.triggered.connect(lambda _=False, d=delta: (self._set_pref("font", d), self._apply_density()))
        menu.exec(hdr.mapToGlobal(pos))

    # ---- UI-34: readable defaults (preferences shared by every discovery table) ---------------
    def _pref(self, key: str, default):
        v = self._settings().value(f"discovery/table_{key}", default)
        return v in (True, "true", "1", 1) if isinstance(default, bool) else v

    def _set_pref(self, key: str, value) -> None:
        self._settings().setValue(f"discovery/table_{key}", value)

    def _apply_density(self) -> None:
        f = self.view.font()
        base = QApplication.font().pointSizeF() if QApplication.instance() else 9.0
        f.setPointSizeF(max(6.0, base + int(self._pref("font", 0))))
        self.view.setFont(f)
        pad = 6 if self._pref("density", "compact") == "compact" else 14
        self.view.verticalHeader().setDefaultSectionSize(self.view.fontMetrics().height() + pad)
        if self._frozen is not None:
            self._frozen.setFont(f)
            self._frozen.verticalHeader().setDefaultSectionSize(self.view.verticalHeader().defaultSectionSize())
            self._frozen_geometry()

    def fit_columns(self) -> None:
        """Resize every visible column to its contents (sampled rows), capped so one long formula cannot push the
        others off-screen; formulas keep their full text in the tooltip."""
        hdr = self.view.horizontalHeader()
        for c in range(self.model.columnCount()):
            if not hdr.isSectionHidden(c):
                self.view.resizeColumnToContents(c)
                self.view.setColumnWidth(c, min(self.view.columnWidth(c), 420))
        self._save_header()
        self._update_frozen()

    def reset_columns(self) -> None:
        hdr = self.view.horizontalHeader()
        for c in range(self.model.columnCount()):
            hdr.setSectionHidden(c, False)
            hdr.moveSection(hdr.visualIndex(c), c)
        self.fit_columns()

    def _update_frozen(self) -> None:
        """Keep the identification column visible while scrolling sideways (Qt's frozen-column pattern: a second
        view over the first column, sharing the model and the selection)."""
        cols = list(self.model.frame.columns)
        want = self._pref("freeze", True) and self.id_col in cols and self.model.columnCount() > 1
        if not want:
            if self._frozen is not None:
                self._frozen.hide()
            return
        hdr = self.view.horizontalHeader()
        idc = cols.index(self.id_col)
        if hdr.visualIndex(idc) != 0:
            hdr.moveSection(hdr.visualIndex(idc), 0)
        if self._frozen is None:
            f = QTableView(self.view)
            f.setModel(self.model)
            f.setFocusPolicy(Qt.NoFocus)
            f.verticalHeader().hide()
            f.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            f.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            f.setSelectionBehavior(QAbstractItemView.SelectRows)
            f.setSelectionMode(QAbstractItemView.SingleSelection)
            f.setVerticalScrollMode(self.view.verticalScrollMode())
            f.setStyleSheet("QTableView { border: none; border-right: 2px solid #9aa9bb; }")
            f.horizontalHeader().setSectionsClickable(True)
            f.horizontalHeader().sectionClicked.connect(self._frozen_sort)
            f.doubleClicked.connect(lambda *_: self.activated.emit())
            self.view.viewport().stackUnder(f)
            f.verticalScrollBar().valueChanged.connect(self.view.verticalScrollBar().setValue)
            self.view.verticalScrollBar().valueChanged.connect(f.verticalScrollBar().setValue)
            hdr.sectionResized.connect(lambda *_: self._frozen_geometry())
            # UI-43: re-place the overlay whenever the viewport or headers change size (maximize, scrollbars
            # appearing, row-number width growing), not only when the view itself is resized
            hdr.geometriesChanged.connect(self._frozen_geometry)
            self.view.verticalHeader().geometriesChanged.connect(self._frozen_geometry)
            self.view.viewport().installEventFilter(self)
            self.view.installEventFilter(self)
            self._frozen = f
        f = self._frozen
        f.setSelectionModel(self.view.selectionModel())
        f.setFont(self.view.font())
        f.verticalHeader().setDefaultSectionSize(self.view.verticalHeader().defaultSectionSize())
        for c in range(self.model.columnCount()):
            f.setColumnHidden(c, c != idc)
        f.show()
        self._frozen_geometry()

    def _frozen_sort(self, c: int) -> None:
        hdr = self.view.horizontalHeader()
        same = hdr.sortIndicatorSection() == c and hdr.sortIndicatorOrder() == Qt.AscendingOrder
        self.view.sortByColumn(c, Qt.DescendingOrder if same else Qt.AscendingOrder)

    def _frozen_geometry(self) -> None:
        f = self._frozen
        if f is None or not f.isVisible():
            return
        idc = list(self.model.frame.columns).index(self.id_col) if self.id_col in self.model.frame.columns else 0
        w = self.view.columnWidth(idc)
        f.setColumnWidth(idc, w)
        fw = self.view.frameWidth()
        vp = self.view.viewport().geometry()          # excludes the horizontal scrollbar: never covers it
        f.setGeometry(self.view.verticalHeader().width() + fw, fw, w, vp.bottom() - fw + 1)
        f.verticalScrollBar().setValue(self.view.verticalScrollBar().value())

    def eventFilter(self, obj, ev) -> bool:  # noqa: N802 (Qt override)
        from PySide6.QtCore import QEvent

        if obj in (self.view, self.view.viewport()) and ev.type() in (QEvent.Resize, QEvent.Show):
            self._frozen_geometry()
        return super().eventFilter(obj, ev)

    def csv_text(self) -> str:
        """The rows shown as CSV. The Highlight column carries the status text the table displays (UI-42: it used to
        be blank because the text lives in the view, not in the frame)."""
        hdr = self.view.horizontalHeader()
        cols = [self.model.frame.columns[hdr.logicalIndex(v)] for v in range(hdr.count())
                if not hdr.isSectionHidden(hdr.logicalIndex(v))]
        order = list(self.model._order)
        rows = self.model.frame.iloc[order][cols].copy() if len(self.model.frame) else self.model.frame
        for name, fn in self.model.lazy.items():     # V133-UI-06: lazily displayed text is exported too, never blank
            if name in rows.columns and len(rows):
                rows[name] = [self.model._lazy_cache.get((i, name)) or fn(i) for i in order]
        if STATUS_COL in rows.columns and len(rows):
            st = self.model.status_text
            txt = [str(st[i]) if st is not None and i < len(st) and st[i] else "" for i in order]
            rows[STATUS_COL] = [t or ("highlighted" if self._highlight[i] else "not highlighted")
                                for t, i in zip(txt, order)] if len(self._highlight) else txt
        return rows.to_csv(index=False)

    def save_csv(self, path: str | None = None) -> str | None:
        if path is None:
            from PySide6.QtWidgets import QFileDialog

            path, _ = QFileDialog.getSaveFileName(self, "Save table as CSV", f"{self.name or 'table'}.csv", "CSV (*.csv)")
            if not path:
                return None
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(self.csv_text())
        self.copy_note.setText(f"Saved {max(self.csv_text().count(chr(10)) - 1, 0):,} rows to {path}.")
        return path

    def copy_csv(self) -> None:
        text = self.csv_text()
        QApplication.clipboard().setText(text)
        n = max(text.count("\n") - 1, 0)
        self.copy_note.setText(f"Copied {n:,} rows." if QApplication.clipboard().text() == text
                               else "Copy failed: the clipboard did not accept the text.")

    def _query(self, text: str) -> None:
        keep = self.selected_id()
        self.model.set_query(text)
        if keep is not None:
            self.select_id(keep)

    def set_frame(self, df: pd.DataFrame, highlight=None, warning=None, text: list[str] | None = None,
                  lazy: dict[str, Any] | None = None) -> None:
        keep = self.selected_id()
        cols = [c for c in self.columns if c in df.columns]
        frame = df[cols].rename(columns=self.columns) if len(df) else pd.DataFrame(columns=list(self.columns.values()))
        lazy = {self.columns[k]: f for k, f in (lazy or {}).items() if k in self.columns}
        n = len(frame)
        from ..models import LOG_P_TWIN

        self.model.log_twins = {self.columns[k]: pd.to_numeric(df[lk], errors="coerce").to_numpy(float)
                                for k, lk in LOG_P_TWIN.items() if k in self.columns and lk in df.columns and len(df)}
        self._highlight = np.zeros(n, bool) if highlight is None else np.asarray(highlight, bool)
        self.model.status_text = text
        self.model.set_frame(frame, self._highlight, warning, tips=None if text is None else [t or None for t in text],
                             lazy=lazy)
        self._apply_only()
        self._restore_header()
        sm = self.view.selectionModel()
        if sm is not None and not getattr(self, "_sel_connected", False):
            sm.selectionChanged.connect(lambda *_: self.selected.emit())
            self._sel_connected = True
        if 0 < n <= 5000 and not getattr(self, "_sized", False):
            # sizing scans every row: only once, and never for very large searches
            self.view.resizeColumnsToContents()
            for c in range(self.model.columnCount()):
                self.view.setColumnWidth(c, min(self.view.columnWidth(c), 320))
            self._sized = True
        self._update_frozen()
        if keep is not None:
            self.select_id(keep)

    def set_status(self, highlight=None, warning=None, text: list[str] | None = None) -> None:
        n = len(self.model.frame)
        self._highlight = np.zeros(n, bool) if highlight is None else np.asarray(highlight, bool)
        self.model.set_row_status(self._highlight, warning, text=text,
                                  tips=None if text is None else [t or None for t in text])
        if self.highlight_only:
            self._apply_only()
        elif STATUS_COL in self.model.frame.columns and \
                self.model.sortColumn() == list(self.model.frame.columns).index(STATUS_COL):
            self.model.sort(self.model.sortColumn(), self.model.sortOrder())     # sorted by the status text

    def set_highlight_only(self, on: bool) -> None:
        self.highlight_only = on
        self._apply_only()

    def _apply_only(self) -> None:
        keep = self.selected_id()
        self.model.set_only_rows(np.flatnonzero(self._highlight) if self.highlight_only else None)
        if keep is not None:
            self.select_id(keep)

    def counts(self) -> tuple[int, int, int]:
        """(highlighted among shown rows, shown rows, total rows)."""
        shown = self.model._order
        hl = int(self._highlight[shown].sum()) if len(self._highlight) and len(shown) else 0
        return hl, len(shown), len(self.model.frame)

    def selected_id(self):
        sm = self.view.selectionModel()
        sel = sm.selectedRows() if sm else []
        if not sel or self.id_col not in self.model.frame.columns or sel[0].row() >= self.model.rowCount():
            return None
        return self.model.frame.iloc[self.model.source_row(sel[0].row())][self.id_col]

    def select_id(self, value) -> bool:
        if self.id_col not in self.model.frame.columns:
            return False
        hits = np.flatnonzero((self.model.frame[self.id_col] == value).to_numpy())
        if not len(hits):
            return False
        row = self.model.view_row(int(hits[0]))
        if row < 0:
            return False
        self.view.selectRow(row)
        self.view.scrollTo(self.model.index(row, 0))
        return True

    def selected_idx(self) -> int | None:
        sm = self.view.selectionModel()
        sel = sm.selectedRows() if sm else []
        if not sel:
            return None
        return int(self.model.frame.iloc[self.model.source_row(sel[0].row())]["#"])


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
        self.outcome = make_searchable(QComboBox())
        self.outcome.currentIndexChanged.connect(self._outcome_changed)
        form.addRow("Dependent variable", self.outcome)
        self.outcome_hint = QLabel()
        self.outcome_hint.setObjectName("hint")
        self.outcome_hint.setWordWrap(True)
        form.addRow(self.outcome_hint)
        self.family = QComboBox()
        self.family.addItem("— confirm the outcome model family —", "")
        for key, label in FAMILIES.items():
            self.family.addItem(label, key)
        self.family.setToolTip("Required: StatExplore suggests a family from the outcome's measurement level and "
                               "treatment, but never chooses one silently.")
        self.family.currentIndexChanged.connect(self._family_changed)
        form.addRow("Outcome model family", self.family)
        self.family_hint = QLabel()
        self.family_hint.setTextFormat(Qt.RichText)    # escaped text: Qt's auto format showed &#x27; literally (UI-30)
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
        self.cand_sel = VariableSelector(selected_first=True, orderable=True)
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
        self.resume_btn.setToolTip("Continue a partial search from where it stopped, with the model / time budget "
                                   "currently set above (it applies to the models added by this continuation).")
        self.resume_btn.setEnabled(False)
        self.resume_btn.clicked.connect(self._resume)
        for b in (self.estimate_btn, self.start_btn, self.cancel_btn, self.resume_btn):
            btns.addWidget(b)
        cl.addLayout(btns)
        self.resume_reason = QLabel()          # V134-009: why Resume is unavailable for the selected search
        self.resume_reason.setWordWrap(True)
        self.resume_reason.setStyleSheet("color: #666;")
        cl.addWidget(self.resume_reason)
        cl.addWidget(self.estimate)
        self.progress = QProgressBar()
        self.progress_label = QLabel()
        cl.addWidget(self.progress)
        cl.addWidget(self.progress_label)
        scroll = section_scroll(cfg, 430)

        # ---------- results ----------
        right = QWidget()
        rl = QVBoxLayout(right)
        top = QHBoxLayout()
        top.addWidget(QLabel("Search:"))
        self.history = QComboBox()
        self.history.activated.connect(self._pick_history)
        top.addWidget(self.history, 1)
        self._top_row = top
        rl.addLayout(top)
        hl = QHBoxLayout()
        hl.addWidget(QLabel("<b>Highlight findings:</b>"))
        self.hl_mode = QComboBox()
        for key, text in engine.HIGHLIGHT_MODES.items():
            self.hl_mode.addItem(text, key)
        self.hl_mode.setToolTip("Presentation only: highlighting never refits a model or changes a p-value.\n"
                                "Strict mode needs a confidence interval for the SAME coefficient the p-value "
                                "tests;\njoint (multi-df) tests have none and are never highlighted in strict mode.")
        for combo in (self.hl_mode,):          # long item texts must not force a wide window (UI-28)
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(16)
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
        self.diag_filter = QComboBox()
        for text, key in (("All diagnostics", None), ("Clean fits only", "clean"), ("Flagged fits only", "flagged")):
            self.diag_filter.addItem(text, key)
        self.ci_filter = QComboBox()
        for text, key in (("Any CI", None), ("Same-estimand CI available", "same_estimand"),
                          ("Joint tests (no single CI)", "joint")):
            self.ci_filter.addItem(text, key)
        frow.addWidget(self.diag_filter)
        frow.addWidget(self.ci_filter)
        rl.addLayout(frow)
        self.scope_label = QLabel()          # P2-03: which filters apply to the visible tab, and its unit
        self.scope_label.setObjectName("hint")
        self.scope_label.setWordWrap(True)
        rl.addWidget(self.scope_label)
        self.tabs = QTabWidget()
        self.summary_view = QTextBrowser()
        self.discovery_table = ResultTable(DISCOVERY_COLS, id_col="Test ID", name="ranking")
        self.support_table = ResultTable(SUPPORT_COLS, id_col="Test ID", name="support")
        self.all_table = ResultTable(ALL_COLS, name="attempts")
        self.compare_table = ResultTable(COMPARE_COLS, name="compare")
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
        self.diag_filter.currentIndexChanged.connect(self._refresh_tables)
        self.ci_filter.currentIndexChanged.connect(self._refresh_tables)
        self.tabs.addTab(self.summary_view, "Summary")
        self.tabs.addTab(self.discovery_table, "Discovery significance ranking")
        self.tabs.addTab(self.support_table, "Statistical support ranking")
        self.tabs.addTab(compare_box, "Model comparison")
        self.tabs.addTab(self.all_table, "All attempts")
        from ..widgets.validation_panel import ValidationPanel

        # V133-VAL-01: validation lives in AMD (configurable before the search, runnable after it)
        self.validation_panel = ValidationPanel(state, lambda: self.run, lambda: self.state.dataset)
        from PySide6.QtWidgets import QScrollArea

        val_scroll = QScrollArea()          # never imposes a minimum height on the results splitter (UI-28)
        val_scroll.setWidgetResizable(True)
        val_scroll.setWidget(self.validation_panel)
        self.tabs.addTab(val_scroll, "Validation")
        # UI-28: results over a collapsible details pane for the selected model (uses the space under short tables)
        self.details = QTextBrowser()
        self.details.setOpenExternalLinks(False)
        self.details.setHtml("<p class='note'>Select a model or test row to see its estimates, all of its "
                             "hypothesis tests, fit statistics and diagnostics here (double-click also opens this "
                             "pane).</p>")
        self.results_split = QSplitter(Qt.Vertical)
        self.results_split.addWidget(self.tabs)
        self.results_split.addWidget(self.details)
        self.results_split.setStretchFactor(0, 3)
        self.results_split.setStretchFactor(1, 1)
        self.results_split.setChildrenCollapsible(True)
        self.results_split.setSizes([600, 220])
        rl.addWidget(self.results_split, 1)
        self.tabs.currentChanged.connect(self._update_counts)
        self.tabs.currentChanged.connect(lambda *_: self._update_details())
        for t in (self.discovery_table, self.support_table, self.all_table, self.compare_table):
            t.selected.connect(self._update_details)
            t.activated.connect(self._expand_details)
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
        self.teach_btn.setToolTip("Linear and binary logistic candidates only. Longitudinal candidates: use 'Open full "
                                  "analysis' (Mixed workspace, exact formula and structure, refit verified).")
        self.teach_btn.clicked.connect(self._to_teaching)
        self.set_btn = QPushButton("Add to comparison")
        self.set_btn.clicked.connect(self._to_set)
        self.main_btn = QPushButton("Compare with main-effects model")
        self.main_btn.setToolTip("Adds this model and the same model without any interaction (same rows, same "
                                 "variables) to the Model Comparison set.")
        self.main_btn.clicked.connect(self._compare_main)
        self.audit_btn = QPushButton("Export audit trail…")
        self.audit_btn.setToolTip("Write a versioned JSON audit record of the selected search (identities, versions, "
                                  "grammar, every candidate fit with diagnostics and cache provenance, every hypothesis "
                                  "with raw / log / adjusted p, validation runs) and a readable Markdown report. No "
                                  "row-level data and no personal paths are included.")
        self.audit_btn.clicked.connect(self._export_audit)
        self.details_btn = QPushButton("Details pane")
        self.details_btn.setCheckable(True)
        self.details_btn.setChecked(True)
        self.details_btn.setToolTip("Show or hide the selected-model details under the tables (drag the divider to "
                                    "resize; the layout is remembered).")
        self.details_btn.toggled.connect(self._toggle_details)
        sort_row.addWidget(self.details_btn)
        rl.addLayout(sort_row)
        # UI-28: actions on their own row so the results panel stays usable on narrow / scaled screens
        act_row = QHBoxLayout()
        for b in (self.open_btn, self.code_btn, self.teach_btn, self.set_btn, self.main_btn):
            act_row.addWidget(b)
        self._top_row.addWidget(self.audit_btn)     # beside the search picker: it acts on the whole search
        act_row.addStretch()
        rl.addLayout(act_row)

        split = QSplitter()
        split.addWidget(scroll)
        split.addWidget(right)
        split.setSizes([460, 900])
        split.setChildrenCollapsible(True)    # the configuration can be collapsed to give results the full width
        self.main_split = split
        self._restore_layout()
        split.splitterMoved.connect(lambda *_: self._save_layout())
        self.results_split.splitterMoved.connect(lambda *_: (setattr(self, "_user_split", True), self._save_layout()))
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
        state.metadata_changed.connect(lambda: self._update_resume())
        state.result_added.connect(lambda r: self._show_run() if r.kind == "discovery_stability" and self.run else None)
        self._reload()

    # ---------- study structure (M §8.1) ----------
    def _build_study_box(self, layout: QVBoxLayout) -> None:
        box = QGroupBox("Study structure")
        f = QFormLayout(box)
        self._study_form = f
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
        self.cluster_combo = QComboBox()
        self.cluster_combo.setToolTip("Optional higher-level grouping (e.g. school, hospital, unit): participants are "
                                      "nested in it and it gets a random intercept in every model. It is a grouping "
                                      "level, never a predictor or interaction term. Gaussian outcomes only.")
        self.id_policy = QComboBox()
        self.id_policy.addItem("IDs are unique across clusters (checked: each ID in one cluster)", "global")
        self.id_policy.addItem("IDs are numbered within each cluster (same ID in two clusters = two people)", "local")
        self.id_policy.setToolTip("How participant IDs relate to the cluster. With numbering within clusters, the "
                                  "participant is the (cluster, ID) pair. People who move between clusters (crossed "
                                  "designs) are not supported and are refused, never reinterpreted.")
        self.cluster_compare = QCheckBox("Exploratory: also fit each structure without the cluster")
        self.cluster_compare.setToolTip("Workflow B only. The study design may require the cluster regardless of this "
                                        "comparison. Off by default. The cluster is never removed automatically: with this option "
                                        "each structure is fitted with and without the cluster on the same rows and "
                                        "composite participants, and the cluster variance is tested at the boundary.")
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
        # V134-007: opt-in combined grammar (off = the v1.3.3 grammar, also for reopened searches and saved forms)
        self.time_elig = QCheckBox("Include time in eligible interactions (workflow A)")
        self.time_elig.setToolTip("Time stays a required main effect (counted once among the predictors) and also takes "
                                  "part in every subset's interaction sets: predictor × predictor and time × predictor "
                                  "terms in one model (and permitted three-way terms with all their parents).\n"
                                  "Applies to 'All eligible interaction combinations' and 'Full factorial'; needs 'Include "
                                  "time as a fixed effect'. Required covariates never become eligible.")
        self.role_btn = QPushButton("Mark this column as the participant ID in the project")
        self.role_btn.setAutoDefault(False)
        self.role_btn.clicked.connect(self._mark_id_role)
        self.long_info = QTextBrowser()
        self.long_info.setMaximumHeight(170)
        self._long_widgets = [self.participant, self.time_combo, self.time_type, self.time_fixed, self.workflow,
                              self.structure, self.structures_list, self.time_inter, self.time_elig, self.role_btn,
                              self.long_info,
                              self.cluster_combo, self.id_policy, self.cluster_compare]
        for label, w in [("Participant ID", self.participant), ("", self.role_btn),
                         ("Higher-level cluster", self.cluster_combo), ("Participant IDs", self.id_policy),
                         ("", self.cluster_compare), ("Time", self.time_combo),
                         ("Time enters as", self.time_type), ("", self.time_fixed), ("", self.time_elig),
                         ("Workflow", self.workflow),
                         ("Structure (workflow A)", self.structure), ("Structures (workflow B)", self.structures_list),
                         ("", self.time_inter), ("Preflight", self.long_info)]:
            if label:
                f.addRow(label, w)
            else:
                f.addRow(w)
        layout.addWidget(box)
        for w in (self.study, self.participant, self.time_combo, self.time_type, self.workflow, self.structure,
                  self.cluster_combo, self.id_policy):
            w.currentIndexChanged.connect(self._study_changed)
        self.cluster_combo.currentIndexChanged.connect(self._fill_structures)
        self.cluster_compare.toggled.connect(self._study_changed)
        self.time_fixed.toggled.connect(self._study_changed)
        self.time_inter.toggled.connect(self._invalidate_estimate)
        self.time_elig.toggled.connect(self._invalidate_estimate)
        self.structures_list.itemChanged.connect(self._invalidate_estimate)

    def _fill_structures(self, *_: Any) -> None:
        """Only structures valid for the family (and, with a cluster, the Gaussian RI/DIAG/RS x residual table).
        Checked states and the locked structure are kept when the list refreshes (one configuration state)."""
        from ...analysis.covstruct import GAUSSIAN_STRUCTURES
        from ...discovery.space import DEFAULT_GAUSSIAN_B

        fam = self.family.currentData() or "gaussian"
        clustered = bool(self.cluster_combo.currentData()) if hasattr(self, "cluster_combo") else False
        keep, keep_list = self.structure.currentData(), set(self._checked_structures())
        self.structure.blockSignals(True)
        self.structure.clear()
        self.structures_list.blockSignals(True)
        self.structures_list.clear()
        for key in structures_for(fam):
            if clustered and key not in GAUSSIAN_STRUCTURES:
                continue        # clustered GEE / mmrm / GLMM / CLMM are not part of this release (shown as refused)
            label = STRUCTURES[key][0]
            self.structure.addItem(label, key)
            it = QListWidgetItem(label)
            it.setData(Qt.UserRole, key)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            default = (key in DEFAULT_GAUSSIAN_B + ["glmer_ri", "glmer_rs", "clmm_ri", "clmm_rs"]) if not keep_list \
                else key in keep_list
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
            self.time_elig.setEnabled(wf == "fixed" and self.time_fixed.isChecked() and bool(self.time_combo.currentData()))
            has_cluster = bool(self.cluster_combo.currentData())
            self.id_policy.setEnabled(has_cluster)           # hidden choices keep their values (one state)
            self.cluster_compare.setEnabled(has_cluster and wf == "structure")

        # progressive disclosure (UI-28): the cluster sub-options take space only once a cluster is chosen
        show = long and bool(self.cluster_combo.currentData())
        for w in (self.id_policy, self.cluster_compare):
            self._study_form.setRowVisible(w, show)
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
            cl = self.cluster_combo.currentData()
            preds = [n for n in self.cand_sel.selected() + self.req_sel.selected() if n not in (out, pid, time, cl)]
            rep = preflight(v, out, pid, time, preds, cluster=cl)
            rng = rep._range_text(v)
            crng = rep._range_text(v, rep.complete_range)
            parts = [f"<p><b>Rows</b> {rep.n_records:,} · <b>Participants</b> {rep.n_participants:,} · "
                     f"<b>Waves</b> {len(rep.time_points) if time else '—'} · participants with 2+ observations "
                     f"{rep.n_multi_obs:,} · duplicate participant-time rows {rep.duplicates}</p>",
                     f"<p><b>Outcome observed</b>: {rng} · <b>complete model rows</b> (outcome, time, every candidate "
                     f"and required predictor, ID{', cluster' if cl else ''}): {rep.n_complete:,} rows, time range "
                     f"{crng}, {rep.n_usable:,} participants ({rep.n_usable_one:,} with a single occasion)</p>"]
            if cl:
                sizes = f"{rep.cluster_sizes[0]}–{rep.cluster_sizes[1]}" if rep.cluster_sizes else "—"
                key = ("(cluster, ID) pairs — composite key; IDs numbered within each cluster"
                       if self.id_policy.currentData() == "local" else
                       "(cluster, ID) pairs — composite key; IDs declared unique across clusters")
                parts.append(f"<p><b>Nested design</b>: {rep.n_clusters:,} {html.escape(v.var(cl).display_name)} "
                             f"clusters · participants per cluster {sizes} · participants = {key} · rows without a "
                             f"cluster (excluded) {rep.n_cluster_missing:,} · IDs seen in 2+ clusters "
                             f"{rep.n_crossing:,}</p>")
                if rep.n_crossing and self.id_policy.currentData() == "global":
                    parts.append(f"<p style='color:#c62828'>⚠ {rep.n_crossing:,} IDs occur in more than one cluster: "
                                 "declare IDs as numbered within each cluster, or the design is crossed (not "
                                 "supported). The search will not start.</p>")
            parts += [f"<p style='color:#8d6e00'>⚠ {html.escape(w)}</p>" for w in rep.warnings]
            parts += [f"<p class='note'>{html.escape(n)}</p>" for n in rep.notes]
            if time:
                parts.append(table_html(rep.tables(v)[1]))
            self.long_info.setHtml("".join(parts))
        except Exception as e:  # an incomplete selection must never break the form
            self.long_info.setHtml(f"<i>{html.escape(str(e))}</i>")

    # ---------- variable lists ----------
    def _reload(self) -> None:
        # UI-49: a reopened project shows its most recent search (settings, results, summary) immediately instead of
        # an empty page until a history entry is picked
        p = self.state.project
        self.run = p.searches[-1] if p is not None and p.searches else None
        self._load_highlight_prefs()
        self._reload_lists()
        self._refresh_history()
        self._show_run()

    def _reload_lists(self) -> None:
        ds = self.state.dataset
        keep_outcome = self.outcome.currentData()
        keep_part, keep_time = self.participant.currentData(), self.time_combo.currentData()
        keep_cluster = self.cluster_combo.currentData()
        self.participant.clear()
        self.time_combo.clear()
        self.time_combo.addItem("(none)", None)
        self.cluster_combo.blockSignals(True)
        self.cluster_combo.clear()
        self.cluster_combo.addItem("(none — participants only)", None)
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
        # any column may be the cluster if the nesting is valid; cluster / ID roles are suggested first
        for v in sorted(ids + others, key=lambda v: (v.role.value != "cluster", v.role.value != "id")):
            self.cluster_combo.addItem(f"{v.display_name}  ({v.role.value} role)" if v.role.value in ("cluster", "id")
                                       else v.display_name, v.name)
        self.cluster_combo.setCurrentIndex(max(0, self.cluster_combo.findData(keep_cluster)))
        self.cluster_combo.blockSignals(False)
        self.participant.setCurrentIndex(max(0, self.participant.findData(keep_part)))
        self.time_combo.setCurrentIndex(max(0, self.time_combo.findData(keep_time)))
        idx = self.outcome.findData(keep_outcome)
        self.outcome.setCurrentIndex(max(0, idx))
        self.outcome.blockSignals(False)
        from ..widgets.searchable import excluded_outcomes_hint

        text, tip = excluded_outcomes_hint(ds, {self.outcome.itemData(i) for i in range(self.outcome.count())})
        self.outcome_hint.setText(text)
        self.outcome_hint.setToolTip(tip)
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
    # ---------- form persistence (UI-48 / CD-12) ----------
    def form_state(self) -> dict[str, Any]:
        return {"outcome": self.outcome.currentData(), "family": self.family.currentData(),
                "candidates": self.cand_sel.selected(), "required": self.req_sel.selected(),
                "study": self.study.currentData(), "participant": self.participant.currentData(),
                "time": self.time_combo.currentData(), "time_type": self.time_type.currentData(),
                "time_fixed": self.time_fixed.isChecked(), "workflow": self.workflow.currentData(),
                "structure": self.structure.currentData(), "structures": self._checked_structures(),
                "time_interactions": self.time_inter.isChecked(), "time_eligible": self.time_elig.isChecked(),
                "cluster": self.cluster_combo.currentData(),
                "id_policy": self.id_policy.currentData(), "cluster_compare": self.cluster_compare.isChecked(),
                "strategy": self.strategy.currentData()}

    def restore_form(self, st: dict[str, Any]) -> None:
        def pick(combo, val):
            i = combo.findData(val)
            if i >= 0:
                combo.setCurrentIndex(i)
        pick(self.outcome, st.get("outcome"))
        pick(self.family, st.get("family"))
        pick(self.study, st.get("study"))
        pick(self.participant, st.get("participant"))
        pick(self.time_combo, st.get("time"))
        pick(self.time_type, st.get("time_type"))
        pick(self.cluster_combo, st.get("cluster"))
        pick(self.id_policy, st.get("id_policy"))
        pick(self.workflow, st.get("workflow"))
        pick(self.strategy, st.get("strategy"))
        self.time_fixed.setChecked(bool(st.get("time_fixed", True)))
        self.time_inter.setChecked(bool(st.get("time_interactions")))
        self.time_elig.setChecked(bool(st.get("time_eligible")))
        self.cluster_compare.setChecked(bool(st.get("cluster_compare")))
        self._fill_structures()
        pick(self.structure, st.get("structure"))
        wanted = set(st.get("structures") or [])
        if wanted:
            lst = self.structures_list
            for i in range(lst.count()):
                lst.item(i).setCheckState(Qt.Checked if lst.item(i).data(Qt.UserRole) in wanted else Qt.Unchecked)
        self.cand_sel.set_selected(list(st.get("candidates") or []))
        self.req_sel.set_selected(list(st.get("required") or []))
        self._study_changed()

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
        if long and time and self.strategy.currentData() == "time":
            required = [n for n in required if n != time]   # V133-UI-09: time enters every focal set; counted once
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
            cluster=self.cluster_combo.currentData() if long else None,
            id_policy=self.id_policy.currentData() or "global",
            cluster_compare=bool(long and self.cluster_combo.currentData() and self.workflow.currentData() == "structure"
                                 and self.cluster_compare.isChecked()),
            all_two_way=bool(self.all2.currentData()) and self.cov_mode.currentData() == "subsets"
            and self.strategy.currentData() == "focal",
            strategy=self.strategy.currentData(), include_main=self.include_main.isChecked(),
            time_eligible=bool(long and time and self.time_elig.isChecked() and self.time_elig.isEnabled()
                               and self.strategy.currentData() in ("combinations", "factorial")),
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
        fam = f"{html.escape(c['family'])}; " + ("comparison" if self._settings().workflow == "structure" and
                                                  self._settings().participant else "focal test") + \
            f": {html.escape(c['focal_test'])}"
        extra = []
        for key, label in (("event", "event"), ("reference", "reference"), ("ordering", "order")):
            if c.get(key):
                val = " < ".join(c[key]) if isinstance(c[key], list) else c[key]
                extra.append(f"{label}: {html.escape(str(val))}")
        extra += [html.escape(t) for t in c.get("treatments", [])]
        st = self._settings()
        if st.participant and st.time and st.strategy == "time":
            extra.append(f"{html.escape(ds.var(st.time).display_name)} is in every focal interaction (its main effect "
                         "added by hierarchy) and counts once toward the predictor limit; it is not an ordinary candidate")
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
            # V133-UI-03: a blocked start never leaves the previous run's progress looking like the current one
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            self.progress_label.setText("Not started: the settings are not valid (see the message above). "
                                        "Earlier results stay listed under Searches.")
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

    def _update_resume(self) -> None:
        run = self.run
        why = "no search selected" if run is None else run.resume_reason(self.state.dataset)
        running = self.cancel_btn.isEnabled()
        self.resume_btn.setEnabled(not why and not running)
        txt = "" if (run is None or not why or run.status == "complete") else f"Resume unavailable: {why}."
        if running:
            txt = ""
        self.resume_reason.setText(txt)
        self.resume_btn.setToolTip("Continue the selected partial search (same search ID, same specification) from "
                                   "where it stopped, with the model / time budget currently set above."
                                   + (f"\nUnavailable: {why}." if why else ""))

    def _resume(self) -> None:
        if self.run is not None and self.run.resumable and not self.run.resume_reason(self.state.dataset):
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
        for w in (self.start_btn, self.estimate_btn):
            w.setEnabled(not on)
        self.cancel_btn.setEnabled(on)
        self._update_resume()        # never re-enabled blindly: only a truly resumable search can resume

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
        if self.validation_panel.auto.isChecked() and run.status == "complete" and run is self.run:
            self.validation_panel.start(run=run)

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
        self.validation_panel.set_run(run)
        self._update_resume()
        if run is None:
            self.summary_view.setHtml("<p>Configure a search on the left, calculate the search space, then start.</p>")
            self.frame, self.tests = pd.DataFrame(), pd.DataFrame()
            for t in (self.discovery_table, self.support_table, self.all_table, self.compare_table):
                t.set_frame(pd.DataFrame())
            self._update_counts()
            return
        tf = engine.tests_frame(run, interpret=False)       # interpretations: computed for visible rows only
        self.frame = engine.add_display_columns(engine.results_frame(run, tf), run)
        self.tests = engine.add_display_columns(tf, run)
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
        if not len(df):
            return df
        f = self.view_filter.currentData()
        if f:
            col, val = f.split("=", 1)
            df = df[df[col] == val] if col in df else df
        diag = self.diag_filter.currentData()
        if diag:
            flagged = engine.warning_mask(df)
            df = df[flagged] if diag == "flagged" else df[~flagged & df["status"].isin(["ok"])]
        ci = self.ci_filter.currentData()
        if ci and "ci_applicable" in df:
            df = df[df["ci_applicable"] == ci]
        return df

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
            lazy = None
            if table in (self.discovery_table, self.support_table) and len(df):
                run = self.run
                sub = df.reset_index(drop=True)
                lazy = {"interpretation": lambda i, sub=sub, run=run: engine.interpretations(sub.iloc[[i]], run)[0]}
            table.highlight_only = self.hl_only.isChecked()
            table.set_frame(df, hl, warn, text, lazy=lazy)
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

    SCOPES = {
        "tests": "Rows are hypothesis tests (a model can carry several). 'Show', diagnostics and CI filters apply here; "
                 "highlights count tests.",
        "attempts": "Rows are candidate models (one per fitted or attempted model), highlighted by each model's focal "
                    "test. Only 'Show: Models …' and the diagnostics / CI filters apply; 'Show: Hypothesis …' filters "
                    "act on the ranking tabs, not here.",
        "compare": "Rows are fitted models with information criteria. Only 'Show: Models …' applies here.",
    }

    def _update_counts(self) -> None:
        table = self._current_table()
        if not isinstance(table, ResultTable):
            self.hl_count.setText("")
            self.scope_label.setText("")
            return
        hl, shown, total = table.counts()
        unit = "tests" if table in (self.discovery_table, self.support_table) else "models"
        self.hl_count.setText(f"{hl:,} highlighted {unit} of {shown:,} shown / {total:,} total")
        scope = self.SCOPES["tests" if unit == "tests" else "attempts" if table is self.all_table else "compare"]
        note = ""
        if self.hl_mode.currentData() == "strict" and unit == "tests" and len(self.tests):
            df = self._table_frames.get(id(table), pd.DataFrame())
            if len(df):
                sig = int(engine.highlight_mask(df, "adjusted", self._alpha()).sum())
                if sig > hl:
                    note = (f" Strict mode: {sig:,} shown tests have adjusted p < α but {sig - hl:,} of them lack a "
                            "same-estimand single-coefficient CI (joint or covariance-structure tests), so they are "
                            "not strictly highlighted. Zero strict highlights does not mean zero significant tests.")
        self.scope_label.setText(scope + note)

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
                f"<p><b>Models:</b> {sm['attempted']:,} of {sm['total']:,} candidate models attempted · "
                f"{sm['ok'] + sm['warning']:,} fitted"
                + (f" ({sm['reused']:,} reused from an identical earlier fit)" if sm.get('reused') else "")
                + f" · {sm['invalid']:,} invalid · {sm['failed'] - sm.get('timed_out', 0):,} failed · "
                f"{sm.get('timed_out', 0):,} timed out · "
                f"{sm['warning']:,} flagged by screening"
                + (f" · {sm['speculative']:,} evaluated after the stop point (excluded)" if sm['speculative'] else "")
                + (f" · {sm['total'] - sm['attempted']:,} not yet visited" if sm['total'] > sm['attempted'] else "")
                + "</p>")
            fams = sm.get("families") or []
            parts.append(f"<p><b>Hypothesis tests:</b> {sum(f['tests'] for f in fams):,} in {len(fams)} "
                         f"famil{'y' if len(fams) == 1 else 'ies'} (a model can carry several tests; highlights count "
                         "tests, not models).</p>")
            parts += [f"<p><b>{e(t)}</b></p>" for t in engine.headline(run, fams)]
            if fams:
                parts.append(
                    f"<p><b>Hypotheses by family</b> — {corr} adjustment (selected; the "
                    f"{'Holm' if s.correction == 'BH' else 'Benjamini-Hochberg'} column is shown for comparison) is "
                    "applied separately within each "
                    "prespecified hypothesis family; families are never ranked together. 'Expected by chance' "
                    "(α × tests) refers to that family only:</p><table cellpadding='3' border='0'><tr><th align='left'>"
                    "Family</th><th>Tests</th><th>Raw p &lt; α</th><th>BH</th><th>Holm</th><th>≈ chance</th>"
                    "<th>No same-estimand CI</th></tr>" + "".join(
                        f"<tr><td>{e(f['label'])}</td><td align='right'>{f['tests']:,}</td>"
                        f"<td align='right'>{f['sig_raw']:,}</td><td align='right'>{f['sig_bh']:,}</td>"
                        f"<td align='right'>{f['sig_holm']:,}</td><td align='right'>{f['expected_false_raw']:.1f}</td>"
                        f"<td align='right'>{f['no_ci']:,}</td></tr>" for f in fams) + "</table>")
            if sm["provisional"]:
                parts.append("<p style='color:#8d6e00'>Adjusted p-values are <b>provisional</b>: they are "
                             "conditional on the candidates evaluated, not the full search space.</p>")
        structure_run = bool(s.participant) and s.workflow == "structure"
        parts.append(
            "<p class='note'>Multiple-testing adjustment controls false discoveries among the tests run; it does "
            "not remove post-selection bias. "
            + ("Covariance-structure comparisons describe the dependence between a participant's observations under "
               "one locked fixed part; check that the preferred structure is stable (e.g. with new data) before "
               "relying on it.</p>" if structure_run else
               "Effect sizes and confidence intervals of the top-ranked models are optimistic. Confirm promising "
               "findings (of the tested type: interaction, main-effect or whole-model) on new data or a holdout "
               "sample.</p>"))
        fam = {"gaussian": "Linear models, focal F-test (nested comparison)",
               "binomial": "Logistic models, focal likelihood-ratio test",
               "ordinal": "Ordinal logistic models (ordinal::clm), focal likelihood-ratio test",
               "multinomial": "Multinomial logistic models (nnet::multinom), focal likelihood-ratio test "
                              "(joint over all outcome contrasts)",
               "lmm": ("Gaussian longitudinal models (structures below), REML fits with the fixed part locked; "
                       "covariance-structure likelihood-ratio tests and REML AIC/BIC (Groups column = participants)"
                       if s.participant and s.workflow == "structure" else
                       "Gaussian longitudinal models (structure below); likelihood-ratio tests on ML fits "
                       "(Groups column = participants; never a denominator df)"),
               "glmm": "Binary logistic mixed models (lme4::glmer, Laplace ML); likelihood-ratio tests; "
                       "subject-specific log-odds",
               "clmm": "Ordinal cumulative-logit mixed models (ordinal::clmm); likelihood-ratio tests; "
                       "subject-specific log-odds; proportional odds assumed",
               "gee": "GEE (geepack::geeglm): population-average effects, robust sandwich Wald tests; no likelihood "
                      "(QIC instead of AIC)"}[run.family]
        effect = {"gaussian": "ΔR² (variance uniquely explained by the tested terms — an interaction, a main-effect term or "
                              "the whole model, as each test's type says)",
                  "binomial": "ΔMcFadden R² (deviance reduction / null deviance)",
                  "ordinal": "ΔMcFadden pseudo-R² (LR statistic / -2 log-likelihood of the null model)",
                  "multinomial": "ΔMcFadden pseudo-R² (deviance reduction / null deviance)",
                  "lmm": ("not applicable: covariance-structure comparisons have no effect size (LR statistic and "
                          "AIC/BIC instead)" if s.participant and s.workflow == "structure" else
                          "Δ marginal R² of the tested terms (direct variance decomposition, ML fits, every Gaussian "
                          "structure)"),
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
                ("Budgets", e(run.budget_text())),
                ("Order", "deterministic: lexicographic by candidate position (see docs/formula-grammar.md)"),
                ("Elapsed", f"{run.elapsed_s:.1f} s"),
                ("Engine", e(str(run.engine.get('r_version', '')))),
            ]) + "</table>")
        parts.append("<h3>Support tiers</h3><ul>" + "".join(f"<li><b>{k}</b>: {v}</li>" for k, v in engine.TIERS.items())
                     + "</ul><p class='note'>Within a tier, models are ordered by effect size, then by fewer "
                       "parameters. No single score is computed; the lowest p-value is not labelled 'best'.</p>")
        return "".join(parts)

    # ---------- layout (UI-28) ----------
    def _qsettings(self):
        from PySide6.QtCore import QSettings

        return QSettings("StatExplore", "StatExplore")

    def _save_layout(self) -> None:
        st = self._qsettings()
        st.setValue("discovery/main_split", self.main_split.saveState())
        st.setValue("discovery/results_split", self.results_split.saveState())
        st.setValue("discovery/details_visible", self.details_btn.isChecked())

    def _restore_layout(self) -> None:
        st = self._qsettings()
        for key, sp in (("discovery/main_split", self.main_split), ("discovery/results_split", self.results_split)):
            v = st.value(key)
            if v is not None:
                sp.restoreState(v)
        vis = st.value("discovery/details_visible")
        if vis is not None and str(vis).lower() in ("false", "0"):
            self.details_btn.setChecked(False)

    def _toggle_details(self, on: bool) -> None:
        self.details.setVisible(on)
        if on:
            self._update_details()
        self._save_layout()

    def _expand_details(self) -> None:
        if not self.details_btn.isChecked():
            self.details_btn.setChecked(True)
        sizes = self.results_split.sizes()
        if len(sizes) == 2 and sizes[1] < 120:
            total = sum(sizes) or max(self.results_split.height(), 600)
            self.results_split.setSizes([int(total * 0.6), int(total * 0.4)])
        self._update_details()

    def _update_details(self) -> None:
        if not self.details_btn.isChecked() or self.run is None:
            return
        table = self._current_table()
        idx = table.selected_idx() if isinstance(table, ResultTable) else None
        if idx is None:
            return
        self.details.setHtml(self._details_html(idx))
        # UI-28: with only a few rows, give the otherwise empty space to the details
        sizes = self.results_split.sizes()
        if getattr(self, "_user_split", False):
            return          # the user dragged the divider: never override it
        if len(sizes) == 2 and sum(sizes) and table.model.rowCount() <= 5 and sizes[1] < 0.45 * sum(sizes):
            rows_h = 60 + table.model.rowCount() * table.view.verticalHeader().defaultSectionSize()
            top = max(rows_h + 80, int(sum(sizes) * 0.35))
            self.results_split.setSizes([top, sum(sizes) - top])

    def _details_html(self, idx: int) -> str:
        """Everything already computed for one candidate (no refit): model, fit, parameters, every test."""
        from ...analysis.descriptor import TransferError, from_discovery

        e = html.escape
        run = self.run
        row = next((r for r in run.rows if int(r["idx"]) == int(idx)), None)
        if row is None:
            return ""
        def fmt(v, d=3):
            if isinstance(v, (np.floating, np.integer)):
                v = v.item()
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                return "—"
            if isinstance(v, float) and v.is_integer() and abs(v) < 1e7 and d != 4:
                return f"{int(v):,}"            # counts (n, groups, parameters) and integral statistics
            return f"{v:,.{d}f}" if isinstance(v, float) else f"{v:,}" if isinstance(v, int) else e(str(v))
        parts = [f"<h3>Candidate #{idx}: {e(str(row.get('formula', '')))}</h3>"]
        try:
            parts.append(f"<p><b>Model identity:</b> {e(from_discovery(run, row).summary())}</p>")
        except TransferError as err:
            parts.append(f"<p><b>Not fitted:</b> {e(str(err))}</p>")
            return "".join(parts)
        fr = self.frame[self.frame["idx"] == idx] if len(self.frame) else pd.DataFrame()
        f0 = fr.iloc[0].to_dict() if len(fr) else {}
        cells = [("Status", row.get("status")), ("Estimation", f0.get("estimation")), ("n", row.get("n")),
                 ("Groups", row.get("n_groups")), ("Fixed coefficients", row.get("n_params")),
                 ("Covariance parameters", row.get("n_re_params")), ("Likelihood parameters", f0.get("n_lik")),
                 ("logLik", row.get("loglik")), ("AIC", row.get("aic")), ("BIC", row.get("bic")),
                 ("QIC", row.get("qic")), ("R² / pseudo-R²", row.get("r2"))]
        parts.append("<table cellpadding='3'>" + "".join(
            f"<tr><td><b>{k}</b></td><td>{fmt(v)}</td></tr>" for k, v in cells if v is not None
            and not (isinstance(v, float) and not np.isfinite(v))) + "</table>")
        if row.get("flags"):
            parts.append(f"<p style='color:#8d6e00'>⚠ Screening / diagnostics: {e(str(row['flags']))}</p>")
        tf = self.tests[self.tests["idx"] == idx] if len(self.tests) else pd.DataFrame()
        if len(tf):
            parts.append("<p><b>Hypothesis tests of this model</b> (each adjusted within its own family):</p>"
                         "<table cellpadding='3' border='0'><tr><th align='left'>Test</th><th align='left'>Type</th>"
                         "<th>Statistic</th><th>df</th><th>p</th><th>adjusted p</th><th>Estimate [CI]</th></tr>")
            for t in tf.to_dict("records"):
                est = t.get("estimate")
                ci = (f"{est:.3f} [{t['ci_low']:.3f}, {t['ci_high']:.3f}]"
                      if isinstance(est, float) and np.isfinite(est) and np.isfinite(t.get("ci_low", np.nan)) else "—")
                parts.append(f"<tr><td>{e(str(t.get('label', '')))}</td><td>{e(str(t.get('test_label', '')))}</td>"
                             f"<td align='right'>{fmt(t.get('statistic'))}</td><td align='right'>{fmt(t.get('df1'), 0)}"
                             f"</td><td align='right'>{fmt(t.get('p'), 4)}</td><td align='right'>"
                             f"{fmt(t.get('p_adj'), 4)}</td><td>{ci}</td></tr>")
            parts.append("</table>")
            prim = tf[tf["primary"]].head(1)
            if len(prim):
                parts.append(f"<p class='note'>{e(engine.interpretations(prim.reset_index(drop=True), run)[0])}</p>")
        if any(len(t) > 1 for t in (row.get("inter") or [])):
            parts.append("<p class='note'>Interaction model: the main-effect coefficients are conditional effects at "
                         "the reference level (or zero) of the interacting variables, not overall effects; no "
                         "coefficient alone identifies the 'strongest predictor'.</p>")
        if run.family in ("ordinal", "clmm"):
            parts.append("<p class='note'>Proportional odds is assumed by the search and is not tested per candidate "
                         "(no diagnostic is shown rather than a blank result). 'Open full analysis' runs "
                         "ordinal::nominal_test for single-level models and says why when it cannot.</p>")
        if run.family == "multinomial":
            ref = (run.coding or {}).get("reference")
            parts.append(f"<p class='note'>Multinomial: every coefficient compares one outcome category with the "
                         f"reference category{(' ' + e(str(ref))) if ref is not None else ''}; full analysis lists the "
                         "comparison category on every row.</p>")
        parts.append("<p class='note'>Use 'Open full analysis' for coefficients, CIs, diagnostic plots and conditional "
                     "effects (refit on the same rows, checked against these statistics).</p>")
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

    def _historical_view(self, action: str):
        """(dataset, new-analysis reasons) for refitting a candidate of this search, or None when cancelled.
        Current project = the search's view: refit as is. Changed project: the user chooses between rebuilding the
        search's original data view and an explicitly NEW analysis under the current settings (STAT-39, H5)."""
        from ...discovery.history import context

        ctx = context(self.state.dataset, self.run)
        if ctx.current:
            return self.state.dataset, None
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("The project changed since this search")
        diffs = "\n".join(f"• {d}" for d in ctx.differences)
        original = None
        if ctx.original is not None:
            original = box.addButton("Refit the original model (rebuild the search's data view)",
                                     QMessageBox.ButtonRole.AcceptRole)
        new = box.addButton("New analysis with the current settings", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setText(f"{action}: the data view this search used is no longer the project's current one.\n\n{diffs}\n\n"
                    + ("Refitting the original rebuilds the search's filters, coding and variables from the snapshot "
                       "stored with it (same source data) and checks the refit against the stored identity. "
                       if ctx.original is not None else
                       "The original model cannot be rebuilt, so it cannot be refitted or verified; its stored result "
                       "stays available unchanged in this search. ")
                    + "A new analysis uses the current data and settings and is labelled as a new analysis, never as "
                      "the historical model.")
        box.exec()
        if original is not None and box.clickedButton() is original:
            return ctx.original, None
        if box.clickedButton() is new:
            return self.state.dataset, list(ctx.differences)
        return None

    def _open_detail(self) -> None:
        row = self._selected_row()
        if row is None:
            return
        if row["status"] in ("invalid", "failed"):
            QMessageBox.information(self, "Discovery", f"This candidate was not fitted: {row['reason']}")
            return
        run = self.run
        hv = self._historical_view("Open full analysis")
        if hv is None:
            return
        ds, reasons = hv
        if run.settings.participant:
            self._to_mixed(row, ds, reasons)
            return
        cand = candidate_from_row(run, row)
        self.state.run_r(lambda s: full_analysis(s, ds, run, cand, reasons), self._detail_done,
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

    def _candidate_model(self, action: str):
        """The selected candidate as a single-level ModelSpec carrying its descriptor. Longitudinal candidates are
        never rerouted or replaced: the user is told which destinations can represent them (P0-01)."""
        from ...analysis.descriptor import TransferError, from_discovery

        row = self._selected_row()
        if row is None:
            return None
        if row["status"] in ("invalid", "failed"):
            QMessageBox.information(self, "Discovery", f"This candidate was not fitted: {row['reason']}")
            return None
        if self.run.settings.participant:
            try:
                d = from_discovery(self.run, row)
                what = d.summary()
            except TransferError as e:
                what = str(e)
            QMessageBox.information(
                self, "Discovery",
                f"{action} works with single-level models (lm / glm / clm / multinom). Candidate #{row['idx']} is a "
                f"longitudinal model — {what} — so it was not sent there, and no other model was substituted.\n\n"
                "Compare longitudinal candidates in this search's 'Model comparison' tab (identical rows; the "
                "estimation method is labelled), or use 'Open full analysis' to refit this exact model in the Mixed "
                "workspace, where the refit is checked against the search.")
            return None
        try:
            m = from_candidate(self.run, row)
        except TransferError as e:
            QMessageBox.information(self, "Discovery", str(e))
            return None
        hv = self._historical_view(action)
        if hv is None:
            return None
        ds, reasons = hv
        if ds is not self.state.dataset:
            QMessageBox.information(self, "Discovery", f"{action} fits models on the project's current data. To use the "
                                    "original model there, restore the search's filters and variable settings in the "
                                    "project first (or open the full analysis, which can rebuild the original view).")
            return None
        if reasons and isinstance(getattr(m, "descriptor", None), dict):
            m.descriptor = dict(m.descriptor, new_analysis=True, new_analysis_reasons=list(reasons))
        return m

    def _to_mixed(self, row: dict[str, Any], ds=None, reasons: list[str] | None = None) -> None:
        """Longitudinal candidates: refit the exact formula and structure in the Mixed workspace, which draws
        population-average and participant-specific trajectories (M §8.7). The result is labelled a refit. On a
        rebuilt historical view the refit runs here, on that view, never on the current project data."""
        from ...analysis import mixed as mx

        if row["status"] in ("invalid", "failed"):
            QMessageBox.information(self, "Discovery", f"This candidate was not fitted: {row['reason']}")
            return
        try:
            spec = mx.spec_from_discovery(self.run, row)
        except mx.MixedInputError as e:
            QMessageBox.information(self, "Discovery", str(e))
            return
        if reasons:
            spec.descriptor = dict(spec.descriptor or {}, new_analysis=True, new_analysis_reasons=list(reasons))
            spec.source = f"NEW analysis under the current settings, started from {spec.source.split(' (')[0]}"
        if ds is not None and ds is not self.state.dataset:      # rebuilt original view
            self.state.run_r(lambda s: mx.run(s, ds, spec), self._detail_done,
                             lambda m: QMessageBox.warning(self, "Analysis failed", m))
            return
        self.state.show_in_mixed.emit(spec)

    def _to_teaching(self) -> None:
        m = self._candidate_model("The Teaching Studio")
        if m is None:
            return
        if m.family not in ("gaussian", "binomial"):
            QMessageBox.information(self, "Discovery", f"The Teaching Studio draws linear and binary logistic models; "
                                    f"this is a {m.family} model, so it was not sent there (no binary or linear "
                                    "substitute is made). Use 'Open full analysis' or 'Add to comparison'.")
            return
        self.state.show_in_teaching.emit(m)

    def _to_set(self) -> None:
        m = self._candidate_model("Model Comparison")
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
        m = self._candidate_model("Compare with main-effects model")
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

    def _export_audit(self, folder: str | None = None) -> list | None:
        """V134-015: audit JSON + report of the selected search."""
        from PySide6.QtWidgets import QFileDialog

        from ...export.audit import write_audit

        if self.run is None:
            QMessageBox.information(self, "Audit trail", "Select a search first.")
            return None
        folder = folder or QFileDialog.getExistingDirectory(self, "Folder for the audit trail")
        if not folder:
            return None
        try:
            files = write_audit(self.run, self.state.dataset, Path(folder))
        except Exception as e:  # noqa: BLE001 - user-facing file error
            QMessageBox.warning(self, "Audit trail", f"Could not write the audit trail: {e}")
            return None
        if not getattr(self, "_quiet", False):
            QMessageBox.information(self, "Audit trail", "Written:\n" + "\n".join(str(f) for f in files))
        return files

    def _view_code(self) -> None:
        row = self._selected_row()
        if row is None:
            return
        run = self.run
        hv = self._historical_view("View candidate code")
        if hv is None:
            return
        ds, reasons = hv
        if reasons:
            QMessageBox.information(self, "View code", "The candidate code reproduces the search's model on the "
                                    "search's data view, which can no longer be rebuilt from this project:\n\n"
                                    + "\n".join(f"• {r}" for r in reasons) + "\n\nThe stored result is unchanged; "
                                    "run a new search to obtain code for the current data.")
            return
        cand = candidate_from_row(run, row)

        comparisons = (engine.structure_tests(run).get(int(row["idx"]), [])
                       if run.settings.participant and run.settings.workflow == "structure" else [])
        adjusted = []
        if not self.tests.empty and "idx" in self.tests:
            cols = ["label", "test_label", "p", "p_adj", "adj_method", "family_size", "search_scope"]
            mine = self.tests[self.tests["idx"] == row["idx"]]
            adjusted = mine[[c for c in cols if c in mine]].to_dict("records")

        from ...analysis.descriptor import TransferError, from_discovery, provenance
        from ...export.exporters import FAMILY_PACKAGES

        try:
            prov = provenance(from_discovery(run, row))
        except TransferError:
            prov = None

        def build(session):
            v = search_view(ds, run.settings)
            session.ensure_data(v)
            return candidate_script(session.load_script, v, run.settings, run.family, cand, comparisons,
                                    expected=row, adjusted=adjusted, provenance=prov)

        from ...discovery.rcode import educational_script

        try:
            edu = educational_script(ds, run.settings, run.family, cand)
        except Exception as e:  # noqa: BLE001 - the educational view never blocks the exact script
            edu = f"# Educational code unavailable for this candidate: {e}\n"
        self.state.run_r(build, lambda script: CodeDialog(
            None, self, title=f"Candidate #{cand.index}", script=script, dataset=ds, provenance=prov, educational=edu,
            identity=row.get("identity"), engine=run.engine, packages=FAMILY_PACKAGES.get(run.family),
            extra={"candidate": int(cand.index), "search_id": run.id, "formula": row.get("formula"),
                   "structure": row.get("structure"), "status": row.get("status"), "flags": row.get("flags")},
            header="The search evaluated this candidate with exactly this function and arguments "
                   "(inside sx_discover_batch, which also records warnings)"
                   + ("; the reported covariance-structure comparisons are recomputed at the end." if comparisons
                      else ".") + " The script stops with a message if n, the analysis rows, logLik/AIC/BIC or any "
                   "test differ from the recorded values beyond the stated tolerances.").exec(),
            lambda m: QMessageBox.warning(self, "Code", m))
