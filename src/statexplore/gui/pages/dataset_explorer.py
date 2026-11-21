"""Dataset Explorer: overview, spreadsheet preview, and variable management."""
from __future__ import annotations

from PySide6.QtCore import QSortFilterProxyModel, Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit, QPushButton, QSplitter, QTableView, QTextBrowser,
    QTabWidget, QVBoxLayout, QWidget,
)

from ...core import missing
from ...core.dataset import DatasetError
from ..derive_dialogs import DeriveDialog, FilterDialog
from ...core.overview import as_frame, summarize
from ...core.variables import Measure, Role
from ..models import DataPreviewModel, FrameModel, fmt
from ..state import AppState

OVERVIEW_COLUMNS = {
    "display_name": "Variable", "label": "Label", "storage": "Type", "measure": "Measure",
    "role": "Role", "n_valid": "Valid N", "n_missing": "Missing", "pct_missing": "% missing",
    "n_unique": "Unique", "mean": "Mean", "sd": "SD", "min": "Min", "max": "Max",
    "n_outliers": "Potential outliers (Tukey)", "n_user_missing_codes": "User-missing codes",
}


class DatasetExplorerPage(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_overview(), "Overview")
        self.tabs.addTab(self._build_preview(), "Data")
        self.tabs.addTab(self._build_variables(), "Variables")
        self.tabs.addTab(self._build_derived(), "Derived && Filters")
        self.missing_view = QTextBrowser()
        self.tabs.addTab(self.missing_view, "Missing data")
        layout = QVBoxLayout(self)
        self.title = QLabel("No dataset loaded")
        self.title.setObjectName("pageTitle")
        layout.addWidget(self.title)
        layout.addWidget(self.tabs)
        state.project_changed.connect(self.reload)
        state.metadata_changed.connect(self.refresh_metadata)
        self.reload()

    # ---- overview ---------------------------------------------------
    def _build_overview(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.stats_label = QLabel()
        self.stats_label.setTextFormat(Qt.RichText)
        lay.addWidget(self.stats_label)
        self.overview_model = FrameModel()
        proxy = QSortFilterProxyModel()
        proxy.setSourceModel(self.overview_model)
        proxy.setSortRole(Qt.UserRole)
        self.overview_view = QTableView()
        self.overview_view.setModel(proxy)
        self.overview_view.setSortingEnabled(True)
        self.overview_view.sortByColumn(-1, Qt.AscendingOrder)  # dataset order until the user sorts
        self.overview_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.overview_view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        lay.addWidget(self.overview_view)
        return w

    # ---- preview ----------------------------------------------------
    def _build_preview(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        bar = QHBoxLayout()
        self.labels_box = QCheckBox("Show value labels")
        self.labels_box.toggled.connect(lambda on: self.preview_model.set_show_labels(on))
        bar.addWidget(self.labels_box)
        bar.addWidget(QLabel("User-missing codes are shown in red; blank cells are system-missing."))
        bar.addStretch()
        lay.addLayout(bar)
        self.preview_model = DataPreviewModel()
        view = QTableView()
        view.setModel(self.preview_model)
        view.horizontalHeader().setDefaultSectionSize(110)
        view.verticalHeader().setDefaultSectionSize(22)
        lay.addWidget(view)
        return w

    # ---- variables --------------------------------------------------
    def _build_variables(self) -> QWidget:
        split = QSplitter()
        self.var_list = QListWidget()
        self.var_list.currentRowChanged.connect(self._show_variable)
        split.addWidget(self.var_list)

        right = QWidget()
        rl = QVBoxLayout(right)
        box = QGroupBox("Variable properties")
        form = QFormLayout(box)
        self.f_name = QLabel()
        self.f_label = QLabel()
        self.f_label.setWordWrap(True)
        self.f_display = QLineEdit()
        self.f_measure = QComboBox()
        self.f_measure.addItems([m.value for m in Measure])
        self.f_role = QComboBox()
        self.f_role.addItems([r.value for r in Role])
        self.f_ref = QComboBox()
        self.f_values = QPlainTextEdit()
        self.f_values.setReadOnly(True)
        self.f_values.setMaximumHeight(140)
        self.f_missing = QLabel()
        form.addRow("Original name", self.f_name)
        form.addRow("Label", self.f_label)
        form.addRow("Display name", self.f_display)
        form.addRow("Measurement level", self.f_measure)
        form.addRow("Role", self.f_role)
        form.addRow("Reference level", self.f_ref)
        form.addRow("Value labels", self.f_values)
        form.addRow("User-missing", self.f_missing)
        btns = QHBoxLayout()
        self.apply_btn = QPushButton("Apply changes")
        self.apply_btn.clicked.connect(self._apply)
        self.undo_btn = QPushButton("Undo last change")
        self.undo_btn.clicked.connect(self._undo)
        btns.addWidget(self.apply_btn)
        btns.addWidget(self.undo_btn)
        btns.addStretch()
        rl.addWidget(box)
        rl.addLayout(btns)
        rl.addWidget(QLabel("Change log (original data is never modified):"))
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        rl.addWidget(self.log_view)
        split.addWidget(right)
        split.setSizes([260, 600])
        return split

    # ---- derived variables & filters ---------------------------------
    def _build_derived(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("<b>Derived variables</b> — computed from source variables; the original data "
                             "is unchanged. They are recomputed in R (shown in View Code)."))
        self.derived_list = QListWidget()
        lay.addWidget(self.derived_list)
        row = QHBoxLayout()
        add = QPushButton("New derived variable…")
        add.clicked.connect(self._new_derived)
        rem = QPushButton("Remove selected")
        rem.clicked.connect(self._remove_derived)
        row.addWidget(add)
        row.addWidget(rem)
        row.addStretch()
        lay.addLayout(row)
        lay.addWidget(QLabel("<b>Row filters</b> — all filters must hold (AND). Applied after derived "
                             "variables are computed; filtered rows are greyed in the Data tab."))
        self.filter_list = QListWidget()
        lay.addWidget(self.filter_list)
        row = QHBoxLayout()
        addf = QPushButton("Add filter…")
        addf.clicked.connect(self._new_filter)
        remf = QPushButton("Remove selected")
        remf.clicked.connect(self._remove_filter)
        row.addWidget(addf)
        row.addWidget(remf)
        row.addStretch()
        lay.addLayout(row)
        return w

    def _refresh_derived(self) -> None:
        ds = self.state.dataset
        self.derived_list.clear()
        self.filter_list.clear()
        if ds is None:
            return
        for spec in ds.derived.values():
            it = QListWidgetItem(f"{ds.var(spec.name).display_name} = {spec.describe()}")
            it.setData(Qt.UserRole, spec.name)
            self.derived_list.addItem(it)
        for i, f in enumerate(ds.filters):
            it = QListWidgetItem(f"Keep rows where {ds.var(f.variable).display_name} "
                                 f"{f.describe().split(' ', 1)[1]}")
            it.setData(Qt.UserRole, i)
            self.filter_list.addItem(it)

    def _new_derived(self) -> None:
        ds = self.state.dataset
        if ds is None:
            return
        dlg = DeriveDialog(ds, self)
        while dlg.exec() == QDialog.Accepted:
            try:
                ds.add_derived(dlg.spec, measure=dlg.measure_value)
            except DatasetError as e:
                dlg.error.setText(str(e))
                continue
            self.state.notify_metadata_changed()
            return

    def _remove_derived(self) -> None:
        ds, it = self.state.dataset, self.derived_list.currentItem()
        if ds is None or it is None:
            return
        try:
            ds.remove_derived(it.data(Qt.UserRole))
        except DatasetError as e:
            QMessageBox.warning(self, "Cannot remove", str(e))
            return
        self.state.notify_metadata_changed()

    def _new_filter(self) -> None:
        ds = self.state.dataset
        if ds is None:
            return
        dlg = FilterDialog(ds, self)
        if dlg.exec() == QDialog.Accepted and dlg.spec:
            try:
                ds.add_filter(dlg.spec)
            except DatasetError as e:
                QMessageBox.warning(self, "Filter", str(e))
                return
            self.state.notify_metadata_changed()

    def _remove_filter(self) -> None:
        ds, it = self.state.dataset, self.filter_list.currentItem()
        if ds is None or it is None:
            return
        ds.remove_filter(it.data(Qt.UserRole))
        self.state.notify_metadata_changed()

    # ---- missing data panel (§17) -------------------------------------
    def _refresh_missing(self) -> None:
        import html as _html

        ds, p = self.state.dataset, self.state.project
        if ds is None:
            self.missing_view.setHtml("")
            return
        names = ds.names()
        bv = missing.by_variable(ds, names)
        top = [n for n in bv.head(12)["variable"]]
        top_names = [v.name for v in ds.variables.values() if v.display_name in top]
        pat = missing.patterns(ds, top_names)
        models = [(m.label or m.id, [m.outcome] + m.variables + m.sample_vars) for m in (p.models if p else [])]
        models += [(f"Search {r.id}", [r.settings.outcome] + r.settings.candidates + r.settings.required)
                   for r in (p.searches if p else [])]
        un = missing.usable_n(ds, models) if models else None

        def table(df) -> str:
            head = "".join(f"<th>{_html.escape(str(c))}</th>" for c in df.columns)
            rows = "".join("<tr>" + "".join(
                f"<td>{_html.escape(f'{v:.1f}' if isinstance(v, float) else str(v))}</td>" for v in r) + "</tr>"
                for r in df.itertuples(index=False))
            return f"<table border='1' cellspacing='0' cellpadding='3'><tr>{head}</tr>{rows}</table>"

        parts = ["<h3>Missing values by variable</h3>", table(bv),
                 "<h3>Most common missingness patterns (12 most-missing variables; X = missing)</h3>",
                 table(pat) if not pat.empty else "<p>—</p>"]
        if un is not None:
            parts += ["<h3>Usable observations per model (complete cases)</h3>", table(un),
                      "<p>Models fitted to different rows are not directly comparable.</p>"]
        parts.append("<p>Analyses use complete cases on their variables; mixed models use all available "
                     "occasions (MAR assumption). Nothing is imputed automatically.</p>")
        self.missing_view.setHtml("".join(parts))

    # ---- refresh ----------------------------------------------------
    def reload(self) -> None:
        ds = self.state.dataset
        self.preview_model.set_dataset(ds)
        self.var_list.clear()
        if ds is None:
            self.title.setText("No dataset loaded — use File ▸ Import Dataset or drag a file onto the window.")
            self.overview_model.set_frame(as_frame_empty())
            self.stats_label.setText("")
            return
        self.title.setText(f"Dataset: {ds.source.path}")
        self.refresh_metadata()

    def refresh_metadata(self) -> None:
        ds = self.state.dataset
        if ds is None:
            return
        s = summarize(ds)
        rows = (f"<b>{s.n_rows:,}</b> of {s.n_rows_total:,} rows (filters active)" if ds.filters
                else f"<b>{s.n_rows:,}</b> rows")
        self.stats_label.setText(
            f"{rows} &nbsp;·&nbsp; <b>{s.n_cols}</b> columns &nbsp;·&nbsp; "
            f"<b>{s.n_missing_cells:,}</b> missing cells ({s.pct_missing_cells:.1f}%) &nbsp;·&nbsp; "
            f"<b>{s.n_complete_rows:,}</b> complete rows &nbsp;·&nbsp; source format: {ds.source.format.upper()}"
        )
        frame = as_frame(s)[list(OVERVIEW_COLUMNS)].rename(columns=OVERVIEW_COLUMNS)
        self.overview_model.set_frame(frame)
        self.preview_model.refresh_all()
        self._refresh_derived()
        self._refresh_missing()
        row = self.var_list.currentRow()
        self.var_list.blockSignals(True)
        self.var_list.clear()
        for v in ds.variables.values():
            tag = ", derived" if ds.is_derived(v.name) else ""
            self.var_list.addItem(f"{v.display_name}  [{v.measure.value}{'' if v.role is Role.NONE else ', ' + v.role.value}{tag}]")
        self.var_list.blockSignals(False)
        self.var_list.setCurrentRow(max(row, 0))
        self._show_variable(self.var_list.currentRow())
        self.log_view.setPlainText("\n".join(
            f"{c.at}  {c.variable}.{c.field}: {c.old!r} → {c.new!r}" for c in ds.changes))

    def _current_name(self) -> str | None:
        ds = self.state.dataset
        row = self.var_list.currentRow()
        if ds is None or row < 0:
            return None
        return list(ds.variables)[row]

    def _show_variable(self, row: int) -> None:
        ds = self.state.dataset
        name = self._current_name()
        if ds is None or name is None:
            return
        v = ds.var(name)
        self.f_name.setText(f"{v.name}  (internal: {v.alias})")
        self.f_label.setText(v.label or "—")
        self.f_display.setText(v.display_name)
        self.f_measure.setCurrentText(v.measure.value)
        self.f_role.setCurrentText(v.role.value)
        self.f_ref.clear()
        self.f_ref.addItem("(first level)", None)
        if v.is_categorical:
            levels = ds.levels(name)
            if len(levels) <= 200:
                for code in levels:
                    self.f_ref.addItem(f"{fmt(code)} — {ds.level_label(name, code)}", code)
            if v.reference_level is not None:
                idx = self.f_ref.findData(v.reference_level)
                self.f_ref.setCurrentIndex(max(idx, 0))
        self.f_ref.setEnabled(v.is_categorical)
        self.f_values.setPlainText("\n".join(f"{fmt(k)} = {lab}" for k, lab in v.value_labels.items()) or "—")
        miss = [fmt(x) for x in v.missing_values] + [f"{fmt(lo)}–{fmt(hi)}" for lo, hi in v.missing_ranges]
        self.f_missing.setText(", ".join(miss) or "none declared")

    def _apply(self) -> None:
        ds = self.state.dataset
        name = self._current_name()
        if ds is None or name is None:
            return
        updates = [
            ("display_name", self.f_display.text()),
            ("measure", self.f_measure.currentText()),
            ("role", self.f_role.currentText()),
        ]
        if ds.var(name).is_categorical and Measure(self.f_measure.currentText()) is not Measure.CONTINUOUS:
            updates.append(("reference_level", self.f_ref.currentData()))
        elif ds.var(name).reference_level is not None:
            updates.append(("reference_level", None))
        try:
            for field_name, value in updates:
                ds.set_variable_field(name, field_name, value)
        except DatasetError as e:
            QMessageBox.warning(self, "Cannot apply change", str(e))
        self.state.notify_metadata_changed()

    def _undo(self) -> None:
        ds = self.state.dataset
        if ds and ds.undo():
            self.state.notify_metadata_changed()


def as_frame_empty():
    import pandas as pd

    return pd.DataFrame(columns=list(OVERVIEW_COLUMNS.values()))
