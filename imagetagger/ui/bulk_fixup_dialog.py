from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Callable

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from imagetagger.utils.theme_colors import danger_accent_color, success_accent_color

if TYPE_CHECKING:
    from imagetagger.ui.main_window import MainWindow
    from imagetagger.ui.models import ImageRecord

_ADD_ICON = "←"
_DEL_ICON = "✕"


class BulkFixupDialog(QDialog):
    """Bulk fixup dialog — aggregates tag proposals across all pending-fixup images."""

    def __init__(
        self,
        records: list["ImageRecord"],
        all_records: list["ImageRecord"],
        parse_tags: Callable[[str], list[str]],
        is_description_like: Callable[[str], bool],
        parent: "MainWindow",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Bulk Fixup")
        self.resize(860, 520)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        self._parse_tags = parse_tags
        self._is_description_like = is_description_like
        self._main_window = parent

        rows = self._compute_proposals(records, all_records, parse_tags, is_description_like)

        self._table = self._build_table(rows)
        layout.addWidget(self._table)

        footer = QHBoxLayout()
        self._resolve_label = QLabel()
        self._resolve_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        footer.addWidget(self._resolve_label)
        footer.addStretch()
        self._apply_button = QPushButton("Apply")
        self._apply_button.setDefault(True)
        self._apply_button.clicked.connect(self._on_apply)
        footer.addWidget(self._apply_button)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.reject)
        footer.addWidget(close_button)
        layout.addLayout(footer)

        self._update_resolve_label()

    # ------------------------------------------------------------------
    # Data assembly
    # ------------------------------------------------------------------

    def _compute_proposals(
        self,
        records: list["ImageRecord"],
        all_records: list["ImageRecord"],
        parse_tags: Callable[[str], list[str]],
        is_description_like: Callable[[str], bool],
    ) -> list[tuple[str, int, str, int, int, int]]:
        """Return [(tag, count, kind, resolve_count, current_count, result_count), ...]"""
        add_counts: Counter[str] = Counter()
        del_counts: Counter[str] = Counter()
        # Per-image pending changes: (pending_adds, pending_dels)
        self._image_changes: list[tuple[set[str], set[str]]] = []
        self._fixup_records: list["ImageRecord"] = []
        image_changes = self._image_changes

        for record in records:
            from imagetagger.utils.sidecar import read_sidecar_data as _read
            sidecar = _read(record.image_path)
            if not sidecar.has_pending_fixup or not sidecar.fixup_tags:
                continue

            current_tags = {
                t.lower()
                for t in parse_tags(record.text)
                if not is_description_like(t)
            }
            proposed_tags = {
                t.lower()
                for t in sidecar.fixup_tags
                if not is_description_like(t)
            }

            pending_adds = proposed_tags - current_tags
            pending_dels = current_tags - proposed_tags
            image_changes.append((pending_adds, pending_dels))
            self._fixup_records.append(record)

            for tag in pending_adds:
                add_counts[tag] += 1
            for tag in pending_dels:
                del_counts[tag] += 1

        self._total_pending = len(image_changes)

        # Current tag counts across the full visible dataset
        current_counts: Counter[str] = Counter()
        for record in all_records:
            for t in parse_tags(record.text):
                if not is_description_like(t):
                    current_counts[t.lower()] += 1

        def _resolve_count(tag: str, kind: str) -> int:
            return sum(
                1 for adds, dels in image_changes
                if (
                    (kind == "add" and tag in adds and len(adds) + len(dels) == 1)
                    or (kind == "del" and tag in dels and len(adds) + len(dels) == 1)
                )
            )

        all_tags = set(add_counts) | set(del_counts)
        rows: list[tuple[str, int, str, int, int, int]] = []
        for tag in all_tags:
            if tag in add_counts:
                cur = current_counts[tag]
                rows.append((tag, add_counts[tag], "add", _resolve_count(tag, "add"), cur, cur + add_counts[tag]))
            if tag in del_counts:
                cur = current_counts[tag]
                rows.append((tag, del_counts[tag], "del", _resolve_count(tag, "del"), cur, cur - del_counts[tag]))
        return sorted(rows, key=lambda x: (-x[1], x[0]))

    # ------------------------------------------------------------------
    # Apply action
    # ------------------------------------------------------------------

    def _on_apply(self) -> None:
        """Apply all accepted/rejected decisions to .txt files and sidecars."""
        from imagetagger.ui.merge_actions import clear_fixup_sidecar
        from imagetagger.utils.sidecar import read_sidecar_data, write_sidecar_data_async

        # Collect the global per-tag decisions from the table.
        accepted_adds: set[str] = set()
        rejected_adds: set[str] = set()
        accepted_dels: set[str] = set()
        rejected_dels: set[str] = set()

        for (tag, kind), widget in zip(self._row_keys, self._decision_widgets):
            state = widget.state
            if kind == "add":
                if state == _DecisionWidget.ACCEPT:
                    accepted_adds.add(tag)
                elif state == _DecisionWidget.REJECT:
                    rejected_adds.add(tag)
            else:  # kind == "del"
                if state == _DecisionWidget.ACCEPT:
                    accepted_dels.add(tag)
                elif state == _DecisionWidget.REJECT:
                    rejected_dels.add(tag)

        any_decision = accepted_adds or rejected_adds or accepted_dels or rejected_dels
        if not any_decision:
            self.accept()
            return

        mw = self._main_window
        any_changed = False
        changed_paths: list = []

        for record, (img_pending_adds, img_pending_dels) in zip(
            self._fixup_records, self._image_changes
        ):
            image_path = record.image_path

            # Intersect this image's pending changes with the global decisions.
            img_accepted_adds = img_pending_adds & accepted_adds
            img_rejected_adds = img_pending_adds & rejected_adds
            img_accepted_dels = img_pending_dels & accepted_dels
            img_rejected_dels = img_pending_dels & rejected_dels

            if not (img_accepted_adds or img_rejected_adds or img_accepted_dels or img_rejected_dels):
                continue

            any_changed = True
            changed_paths.append(image_path)

            # --- Update .txt ---
            all_annotations = self._parse_tags(record.text)
            description_part = [t for t in all_annotations if self._is_description_like(t)]
            current_tag_set = {t for t in all_annotations if not self._is_description_like(t)}
            new_tag_set = (current_tag_set | img_accepted_adds) - img_accepted_dels
            new_annotations = description_part + sorted(new_tag_set)
            mw._set_tags_for_image_path(image_path, new_annotations, "Bulk fixup")

            # --- Update sidecar fixup_tags ---
            sidecar = read_sidecar_data(image_path)
            if sidecar.fixup_tags:
                fixup_desc = [t for t in sidecar.fixup_tags if self._is_description_like(t)]
                fixup_tag_set = {t.lower() for t in sidecar.fixup_tags if not self._is_description_like(t)}
                new_fixup_set = (fixup_tag_set - img_rejected_adds) | img_rejected_dels
            else:
                fixup_desc = []
                new_fixup_set = set(img_rejected_dels)

            # Resolve if all of this image's pending adds/dels now have a decision.
            is_resolved = (
                img_pending_adds <= (img_accepted_adds | img_rejected_adds)
                and img_pending_dels <= (img_accepted_dels | img_rejected_dels)
            )

            if is_resolved:
                clear_fixup_sidecar(image_path)
            else:
                new_fixup_tags = fixup_desc + sorted(new_fixup_set)
                sidecar.fixup_tags = new_fixup_tags if new_fixup_tags else None
                write_sidecar_data_async(image_path, sidecar)

        if any_changed:
            for path in changed_paths:
                mw._on_fixup_state_changed(path)

        self.accept()

    # ------------------------------------------------------------------
    # Table construction
    # ------------------------------------------------------------------

    def _build_table(self, rows: list[tuple[str, int, str, int, int, int]]) -> QTableWidget:
        palette = self.palette()
        del_color = danger_accent_color(palette)
        acc_color = success_accent_color(palette)

        table = QTableWidget(len(rows), 5, self)
        table.setShowGrid(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        table.verticalHeader().hide()
        table.setHorizontalHeaderLabels(["Tag", "Resolves", "Current", "Result", "Action"])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col, width in ((1, 80), (2, 72), (3, 72), (4, 112)):
            table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.Fixed)
            table.setColumnWidth(col, width)

        self._decision_widgets: list[_DecisionWidget] = []
        self._row_keys: list[tuple[str, str]] = []  # (tag, kind) per row

        for row_idx, (tag, count, kind, resolve_count, current_count, result_count) in enumerate(rows):
            cell = _TagProposalCell(tag, count, kind, del_color)
            table.setCellWidget(row_idx, 0, cell)

            for col, value in ((1, resolve_count), (2, current_count), (3, result_count)):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(
                    int(Qt.AlignmentFlag.AlignVCenter) | int(Qt.AlignmentFlag.AlignHCenter)
                )
                item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                table.setItem(row_idx, col, item)

            decision = _DecisionWidget(acc_color, del_color)
            decision.state_changed.connect(self._update_resolve_label)
            table.setCellWidget(row_idx, 4, decision)
            self._decision_widgets.append(decision)
            self._row_keys.append((tag, kind))

            table.setRowHeight(row_idx, 26)

        return table

    def _update_resolve_label(self) -> None:
        """Recompute how many images would be fully resolved by all accepted decisions."""
        accepted_adds: set[str] = set()
        accepted_dels: set[str] = set()
        for (tag, kind), widget in zip(self._row_keys, self._decision_widgets):
            if widget.state == _DecisionWidget.ACCEPT:
                (accepted_adds if kind == "add" else accepted_dels).add(tag)

        resolved = sum(
            1
            for adds, dels in self._image_changes
            if adds <= accepted_adds and dels <= accepted_dels
        )
        self._resolve_label.setText(
            f"Images to resolve: {resolved} of {self._total_pending}"
        )


class _TagProposalCell(QWidget):
    """Single-row widget: 'tag text (N)' left + icon right."""

    def __init__(
        self,
        tag: str,
        count: int,
        kind: str,
        del_color: QColor,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(6)

        text_label = QLabel(f"{tag}  ({count})")
        text_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(text_label, stretch=1)

        icon = _ADD_ICON if kind == "add" else _DEL_ICON
        icon_label = QLabel(icon)
        if kind == "del":
            icon_label.setStyleSheet(
                f"color: {del_color.name(QColor.NameFormat.HexRgb)}; font-weight: bold;"
            )
        icon_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)
        layout.addWidget(icon_label)


class _DecisionWidget(QWidget):
    """Segmented 3-button group: Accept (←) | Unresolved (–) | Reject (✕)."""

    state_changed = pyqtSignal()

    ACCEPT = "accept"
    UNRESOLVED = "unresolved"
    REJECT = "reject"

    def __init__(self, acc_color: QColor, del_color: QColor, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._acc_color = acc_color
        self._del_color = del_color
        self._state = self.UNRESOLVED

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(0)

        self._btn_accept = QPushButton("✓")
        self._btn_skip = QPushButton("–")
        self._btn_reject = QPushButton("✕")

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        for btn in (self._btn_accept, self._btn_skip, self._btn_reject):
            btn.setCheckable(True)
            btn.setFixedHeight(22)
            self._group.addButton(btn)
            layout.addWidget(btn)

        self._btn_skip.setChecked(True)
        self._apply_styles()

        self._btn_accept.toggled.connect(lambda on: self._on_toggled(self.ACCEPT, on))
        self._btn_skip.toggled.connect(lambda on: self._on_toggled(self.UNRESOLVED, on))
        self._btn_reject.toggled.connect(lambda on: self._on_toggled(self.REJECT, on))

    def _on_toggled(self, state: str, checked: bool) -> None:
        if checked:
            self._state = state
            self._apply_styles()
            self.state_changed.emit()

    def _apply_styles(self) -> None:
        palette = self.palette()
        base = palette.color(QPalette.ColorRole.Button).name()
        text = palette.color(QPalette.ColorRole.ButtonText).name()

        acc_bg = self._acc_color.name(QColor.NameFormat.HexRgb)
        del_bg = self._del_color.name(QColor.NameFormat.HexRgb)
        hi_text = palette.color(QPalette.ColorRole.HighlightedText).name()
        neu_bg = palette.color(QPalette.ColorRole.Highlight).name()

        _base = (
            "QPushButton { border: 1px solid palette(mid); padding: 0px 4px;"
            f" background: {base}; color: {text}; font-weight: normal; }}"
            "QPushButton:hover { background: palette(light); }"
        )

        def _active(bg: str, fg: str) -> str:
            return (
                f"QPushButton {{ border: 1px solid palette(mid); padding: 0px 4px;"
                f" background: {bg}; color: {fg}; font-weight: bold; }}"
            )

        if self._state == self.ACCEPT:
            self._btn_accept.setStyleSheet(_active(acc_bg, hi_text))
            self._btn_skip.setStyleSheet(_base)
            self._btn_reject.setStyleSheet(_base)
        elif self._state == self.REJECT:
            self._btn_accept.setStyleSheet(_base)
            self._btn_skip.setStyleSheet(_base)
            self._btn_reject.setStyleSheet(_active(del_bg, hi_text))
        else:
            self._btn_accept.setStyleSheet(_base)
            self._btn_skip.setStyleSheet(_active(neu_bg, hi_text))
            self._btn_reject.setStyleSheet(_base)

    @property
    def state(self) -> str:
        return self._state
