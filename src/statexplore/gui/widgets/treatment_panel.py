"""Variable Treatment Manager: per-analysis "treat as continuous" controls (v1.1 §6, v1.2 M §6).

Every variable of the current model is always shown with a status — the outcome first, then
every selected predictor — so nothing disappears silently:
  eligible     numeric ordinal / nominal codes: a checkbox (needs the §6.2 acknowledgement, and for
               project-nominal codes a second confirmation that the codes are meaningful numbers);
  continuous   "Already continuous — no override needed";
  unavailable  disabled, with the exact reason (text categories, constant, identifier, …).
A second view lists every eligible variable in the dataset for preconfiguration; a preconfigured
treatment applies only when that variable is part of the fitted model. An acknowledgement is kept
only while the variable's coding (values, labels, missing codes, level) is unchanged.

The panel only produces VariableTreatment objects; it never changes the dataset.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QGroupBox, QHBoxLayout, QLabel, QListWidgetItem, QPushButton,
    QScrollArea, QTextBrowser, QVBoxLayout, QWidget,
)

from ...analysis import measurement as M
from ...core.dataset import Dataset
from ...core.variables import Storage
from .variable_selector import VariableSelector

CHECKBOX_TEXT = "Treat {name} as continuous for this analysis"
OUTCOME_TEXT = "Treat outcome {name} as continuous for this analysis"
VIEW_MODEL, VIEW_ALL = "model", "all"


def _fmt(code) -> str:
    return str(int(code)) if isinstance(code, float) and code.is_integer() else str(code)


def eligible_for_continuous(ds: Dataset, name: str) -> bool:
    """Numeric categorical variables with valid numeric codes (§6: only offer when valid)."""
    return M.treatment_status(ds, name)[0] == "eligible"


def codes_html(ds: Dataset, name: str, limit: int = 30) -> str:
    v = ds.var(name)
    counts = ds.analysis_column(name).value_counts(dropna=True)
    codes = ds.levels(name) if v.is_categorical else sorted(counts.index.tolist())[:limit]
    rows = "".join(f"<tr><td align='right'>{_fmt(c)}</td><td>{ds.level_label(name, c)}</td>"
                   f"<td align='right'>{int(counts.get(c, 0)):,}</td></tr>" for c in codes[:limit])
    more = f"<p>… {len(codes) - limit} more</p>" if len(codes) > limit else ""
    return (f"<b>{v.display_name}</b>" + (f" — {v.label}" if v.label else "")
            + f"<br>Project level: {v.measure.value}" + (f"; source file: {v.source_measure}" if v.source_measure else "")
            + f"; storage: {v.storage.value}" + (f"<br>Direction: {v.coding_direction}" if v.coding_direction else "")
            + "<table cellpadding='2'><tr><th>code</th><th>label</th><th>rows</th></tr>" + rows + "</table>" + more
            + "<p class='note'>Category order can be changed in Dataset Explorer ▸ Measurement Review.</p>")


class AcknowledgeDialog(QDialog):
    """Shows the codes, the §6.2 warning, and the required confirmations for one or more variables."""

    def __init__(self, ds: Dataset, names: str | list[str], parent: QWidget | None = None):
        super().__init__(parent)
        names = [names] if isinstance(names, str) else list(names)
        self.setWindowTitle("Treat as continuous: " + ", ".join(ds.var(n).display_name for n in names))
        lay = QVBoxLayout(self)
        body = QWidget()
        bl = QVBoxLayout(body)
        for name in names:
            v = ds.var(name)
            codes = ds.levels(name)
            listing = "<br>".join(f"{_fmt(c)} = {ds.level_label(name, c)}" for c in codes[:20])
            if len(codes) > 20:
                listing += f"<br>… {len(codes) - 20} more"
            info = QLabel(
                f"<b>{v.display_name}</b>" + (f" — {v.label}" if v.label else "")
                + f"<br>Project level: {v.measure.value}" + (f"; source file: {v.source_measure}" if v.source_measure else "")
                + f"<br><br>The analysis will use these numeric codes as values:<br>{listing}"
                + (f"<br><br>Direction: {v.coding_direction}." if v.coding_direction else ""))
            info.setWordWrap(True)
            bl.addWidget(info)
        if len(names) > 3:
            scroll = QScrollArea()
            scroll.setWidget(body)
            scroll.setWidgetResizable(True)
            scroll.setMinimumHeight(300)
            lay.addWidget(scroll)
        else:
            lay.addWidget(body)
        lay.addWidget(QLabel(f"<i>{M.EQUAL_INTERVALS}</i>"))
        self.ack = QCheckBox(M.CONTINUOUS_WARNING)
        self.ack.setStyleSheet("QCheckBox { font-weight: bold }")
        lay.addWidget(self.ack)
        self.confirm = None
        nominal = [n for n in names if M.needs_numeric_confirmation(ds, n)]
        if nominal:
            who = ", ".join(ds.var(n).display_name for n in nominal)
            self.confirm = QCheckBox(f"I confirm the codes of {who} are meaningful numbers, not arbitrary category "
                                     "identifiers. " + M.NOMINAL_NUMERIC_WARNING)
            lay.addWidget(self.confirm)
        for cb in (self.ack, self.confirm):
            if cb:
                cb.setToolTip(cb.text())
                cb.toggled.connect(self._update)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(self.buttons)
        self._update()

    def _update(self) -> None:
        ok = self.ack.isChecked() and (self.confirm is None or self.confirm.isChecked())
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(ok)


class _TreatmentSelector(VariableSelector):
    """VariableSelector whose bulk buttons only touch eligible (checkable) rows."""

    def _set_items(self, items, state) -> None:
        super()._set_items([it for it in items if it.flags() & Qt.ItemIsUserCheckable], state)


class TreatmentPanel(QGroupBox):
    changed = Signal()

    def __init__(self, parent: QWidget | None = None, title: str = "Variable treatment (this analysis only)"):
        super().__init__(title, parent)
        self.ds: Dataset | None = None
        self.outcome: str | None = None
        self.predictors: list[str] = []
        self._acks: dict[str, tuple[M.VariableTreatment, str]] = {}   # name -> (treatment, coding fingerprint)
        self._status: dict[str, tuple[str, str]] = {}
        self.ask = True          # tests set False and call set_treated()
        lay = QVBoxLayout(self)

        # A. outcome
        self.outcome_box = QWidget()
        ol = QVBoxLayout(self.outcome_box)
        ol.setContentsMargins(0, 0, 0, 0)
        self.outcome_info = QLabel()
        self.outcome_info.setWordWrap(True)
        self.outcome_info.setTextFormat(Qt.RichText)
        ol.addWidget(self.outcome_info)
        orow = QHBoxLayout()
        self.outcome_check = QCheckBox()
        self.outcome_check.toggled.connect(self._outcome_toggled)
        orow.addWidget(self.outcome_check, 1)
        self.codes_btn = QPushButton("View codes and labels")
        self.codes_btn.setAutoDefault(False)
        self.codes_btn.clicked.connect(lambda: self.outcome and self.show_codes(self.outcome))
        orow.addWidget(self.codes_btn)
        ol.addLayout(orow)
        lay.addWidget(self.outcome_box)

        # B / C. predictors (current model) or all eligible dataset variables
        vrow = QHBoxLayout()
        vrow.addWidget(QLabel("Show:"))
        self.view_mode = QComboBox()
        self.view_mode.addItem("Current model variables", VIEW_MODEL)
        self.view_mode.addItem("All eligible dataset variables (preconfigure)", VIEW_ALL)
        self.view_mode.setToolTip("Preconfigured treatments apply only when the variable is part of the fitted "
                                  "model; they never change the dataset.")
        self.view_mode.currentIndexChanged.connect(self._render_predictors)
        vrow.addWidget(self.view_mode, 1)
        lay.addLayout(vrow)
        self.pred_sel = _TreatmentSelector(max_height=200, placeholder="Search selected predictors…",
                                           show_level_filter=False)
        self.pred_sel.buttons["check_all"].setText("Check all eligible")
        self.pred_sel.changed.connect(self._predictors_changed)
        self.pred_sel.list.itemDoubleClicked.connect(lambda it: self.show_codes(it.data(Qt.UserRole)))
        lay.addWidget(self.pred_sel)
        self.counts = QLabel()
        self.counts.setObjectName("hint")
        self.counts.setWordWrap(True)
        lay.addWidget(self.counts)
        self._hint = QLabel("Select an outcome or predictors to see their treatment options.")
        self._hint.setObjectName("hint")
        self._hint.setWordWrap(True)
        lay.addWidget(self._hint)
        self._rendering = False
        self.outcome_box.setVisible(False)

    # ---- content --------------------------------------------------------------------
    def set_model(self, ds: Dataset | None, outcome: str | None, predictors: list[str]) -> None:
        self.set_variables(ds, list(predictors), outcome=outcome)

    def set_variables(self, ds: Dataset | None, names: list[str], outcome: str | None = None) -> None:
        """Show `outcome` (if any) and every variable in `names`; earlier decisions are kept while the
        variable's coding is unchanged."""
        self.ds = ds
        known = (lambda n: ds is not None and n in ds.variables)
        self.outcome = outcome if outcome and known(outcome) else None
        self.predictors = [n for n in dict.fromkeys(names) if known(n) and n != self.outcome]
        self._status = {n: M.treatment_status(ds, n) for n in self._names_in_scope()} if ds is not None else {}
        # an acknowledgement is valid only for the exact coding it was given for
        if ds is not None:
            for n in list(self._acks):
                if n not in ds.variables or self._acks[n][1] != M.coding_fingerprint(ds, n) \
                        or M.treatment_status(ds, n)[0] != "eligible":
                    self._acks.pop(n)
        self._render_outcome()
        self._render_predictors()
        self.setVisible(True)

    def _names_in_scope(self) -> list[str]:
        names = ([self.outcome] if self.outcome else []) + list(self.predictors)
        if self.ds is not None and self.view_mode.currentData() == VIEW_ALL:
            names += [n for n, v in self.ds.variables.items()
                      if n not in names and v.is_categorical and v.storage is Storage.NUMERIC]
        return names

    def status(self, name: str) -> tuple[str, str]:
        if name not in self._status and self.ds is not None and name in self.ds.variables:
            self._status[name] = M.treatment_status(self.ds, name)
        return self._status.get(name, ("unavailable", ""))

    def is_offered(self, name: str) -> bool:
        """True if a 'treat as continuous' control is offered for `name` in the current model."""
        return name in self.model_variables() and self.status(name)[0] == "eligible"

    def model_variables(self) -> list[str]:
        return ([self.outcome] if self.outcome else []) + list(self.predictors)

    def _render_outcome(self) -> None:
        ds, out = self.ds, self.outcome
        self.outcome_box.setVisible(bool(out))
        if not out:
            return
        v = ds.var(out)
        st, why = self.status(out)
        self.outcome_info.setText(
            f"<b>Outcome: {v.display_name}</b>" + (f" — {v.label}" if v.label else "")
            + f"<br><small>project level: {v.measure.value}"
            + (f" · source file: {v.source_measure}" if v.source_measure else "")
            + f" · storage: {v.storage.value} · analysis treatment: "
            + ("continuous (this analysis)" if out in self.treated() else v.measure.value) + "</small>"
            + ("" if st == "eligible" else f"<br><i>{why}</i>"))
        self.outcome_check.blockSignals(True)
        self.outcome_check.setText(OUTCOME_TEXT.format(name=v.display_name))
        self.outcome_check.setVisible(st == "eligible")
        self.outcome_check.setChecked(out in self._acks)
        self.outcome_check.setToolTip(why)
        self.outcome_check.blockSignals(False)

    def _render_predictors(self, *_args) -> None:
        ds = self.ds
        if ds is None:
            self.pred_sel.set_variables(None, [])
            self._update_counts()
            return
        mode = self.view_mode.currentData()
        if mode == VIEW_ALL:
            names = [n for n in self._names_in_scope() if n != self.outcome and self.status(n)[0] == "eligible"]
        else:
            names = list(self.predictors)
        in_model = set(self.model_variables())

        def deco(n: str) -> str:
            st, why = self.status(n)
            tag = {"eligible": "", "continuous": "  ·  already continuous — no override needed",
                   "unavailable": "  ·  " + why.split(":", 1)[0].lower()}[st]
            if mode == VIEW_ALL:
                tag += "  ·  active model term" if n in in_model else "  ·  preconfigured (not in the model)"
            return f"  ·  {ds.var(n).measure.value}{tag}"

        self._rendering = True
        self.pred_sel.set_variables(ds, names, keep=False, decorate=deco)
        lst = self.pred_sel.list
        lst.blockSignals(True)
        for i in range(lst.count()):
            it: QListWidgetItem = lst.item(i)
            n = it.data(Qt.UserRole)
            st, why = self.status(n)
            it.setToolTip(it.toolTip() + "\n" + why)
            if st == "eligible":
                it.setCheckState(Qt.Checked if n in self._acks else Qt.Unchecked)
            else:
                it.setFlags(it.flags() & ~Qt.ItemIsUserCheckable & ~Qt.ItemIsEnabled)
                it.setData(Qt.CheckStateRole, None)
        lst.blockSignals(False)
        self.pred_sel._sync_order()
        self.pred_sel._update_count()
        self._rendering = False
        self._update_counts()

    def _update_counts(self) -> None:
        if self.ds is None or not self.model_variables():
            self._hint.setVisible(True)
            self.counts.setText("")
            return
        self._hint.setVisible(False)
        st = [self.status(n)[0] for n in self.model_variables()]
        self.counts.setText(f"Model variables: {len(st)} · eligible for continuous treatment: {st.count('eligible')} · "
                            f"already continuous: {st.count('continuous')} · unavailable: {st.count('unavailable')} · "
                            f"treated as continuous: {len(self.treated())}")

    # ---- decisions ----------------------------------------------------------------------
    def _acknowledge(self, names: list[str]) -> list[str]:
        """Collect the acknowledgement for newly checked variables; return those accepted."""
        names = [n for n in names if n not in self._acks and self.status(n)[0] == "eligible"]
        if not names:
            return []
        if self.ask and AcknowledgeDialog(self.ds, names, self).exec() != QDialog.Accepted:
            return []
        for n in names:
            self._acks[n] = (M.VariableTreatment(n, "continuous", acknowledged=True,
                                                 numeric_confirmed=M.needs_numeric_confirmation(self.ds, n)),
                             M.coding_fingerprint(self.ds, n))
        return names

    def _outcome_toggled(self, on: bool) -> None:
        out = self.outcome
        if not out:
            return
        if on and not self._acknowledge([out]):
            self.outcome_check.blockSignals(True)
            self.outcome_check.setChecked(False)
            self.outcome_check.blockSignals(False)
            return
        if not on:
            self._acks.pop(out, None)
        self._render_outcome()
        self._update_counts()
        self.changed.emit()

    def _predictors_changed(self) -> None:
        if self._rendering:
            return
        lst = self.pred_sel.list
        shown = {lst.item(i).data(Qt.UserRole): lst.item(i).checkState() == Qt.Checked
                 for i in range(lst.count()) if lst.item(i).flags() & Qt.ItemIsUserCheckable}
        new = [n for n, on in shown.items() if on and n not in self._acks]
        accepted = set(self._acknowledge(new))
        rejected = [n for n in new if n not in accepted]
        if rejected:
            self._rendering = True
            lst.blockSignals(True)
            for i in range(lst.count()):
                if lst.item(i).data(Qt.UserRole) in rejected:
                    lst.item(i).setCheckState(Qt.Unchecked)
            lst.blockSignals(False)
            self.pred_sel._sync_order()
            self.pred_sel._update_count()
            self._rendering = False
        for n, on in shown.items():          # hidden-by-search rows keep their state (never unchecked here)
            if not on:
                self._acks.pop(n, None)
        self._update_counts()
        self.changed.emit()

    def set_treated(self, name: str, on: bool = True) -> None:
        """Programmatic check (the caller has collected the acknowledgement when ask is False)."""
        if name == self.outcome:
            self.outcome_check.setChecked(on)
            return
        lst = self.pred_sel.list
        for i in range(lst.count()):
            if lst.item(i).data(Qt.UserRole) == name:
                if not lst.item(i).flags() & Qt.ItemIsUserCheckable:
                    raise ValueError(f"{name} cannot be treated as continuous: {self.status(name)[1]}")
                lst.item(i).setCheckState(Qt.Checked if on else Qt.Unchecked)
                return
        if on:   # not shown in the current view: preconfigure directly
            self._acknowledge([name])
        else:
            self._acks.pop(name, None)
        self.changed.emit()

    def load_treatments(self, items: list | None) -> None:
        """Programmatic restore of a model's continuous treatments (transfer, project reload): exactly these, no
        dialog; non-continuous encodings are not represented here (the caller keeps them)."""
        self._acks = {}
        for t in M.as_treatments(items):
            if t.treatment == "continuous" and self.ds is not None and t.variable in self.ds.variables:
                self._acks[t.variable] = (t, M.coding_fingerprint(self.ds, t.variable))
        self._render_outcome()
        self._render_predictors()
        self.changed.emit()

    def treated(self) -> list[str]:
        """Variables treated as continuous in the CURRENT model (preconfigured others do not apply)."""
        return [n for n in self.model_variables() if n in self._acks]

    def preconfigured(self) -> list[str]:
        return [n for n in self._acks if n not in self.model_variables()]

    def treatments(self) -> list[M.VariableTreatment]:
        return [self._acks[n][0] for n in self.treated()]

    def reset(self) -> None:
        self._acks.clear()
        self._render_outcome()
        self._render_predictors()

    def show_codes(self, name: str) -> None:
        if self.ds is None or not name:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Codes and labels — {self.ds.var(name).display_name}")
        dlg.resize(460, 420)
        lay = QVBoxLayout(dlg)
        view = QTextBrowser()
        view.setHtml(codes_html(self.ds, name))
        lay.addWidget(view)
        btn = QDialogButtonBox(QDialogButtonBox.Close)
        btn.rejected.connect(dlg.reject)
        lay.addWidget(btn)
        dlg.show()
