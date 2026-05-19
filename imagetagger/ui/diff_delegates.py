from __future__ import annotations

from typing import Callable
from PyQt6.QtCore import QEvent, Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QPainter,
    QPalette,
    QTextCharFormat,
    QTextCursor,
    QTextDocument,
)
from PyQt6.QtWidgets import (
    QApplication,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTextEdit,
)
from imagetagger.utils.theme_colors import danger_accent_color

DIFF_RANGES_ROLE = int(Qt.ItemDataRole.UserRole) + 1
ITEM_TEXT_ROLE = int(Qt.ItemDataRole.UserRole) + 2
IS_SEARCH_MATCH_ROLE = int(Qt.ItemDataRole.UserRole) + 3
ITEM_BADGE_ROLE = int(Qt.ItemDataRole.UserRole) + 4


def _blend_colors(base: QColor, overlay: QColor, alpha: float) -> QColor:
    """Return an opaque blend of two colors using the provided alpha factor."""
    clamped_alpha = max(0.0, min(1.0, alpha))
    inv = 1.0 - clamped_alpha
    return QColor(
        round(base.red() * inv + overlay.red() * clamped_alpha),
        round(base.green() * inv + overlay.green() * clamped_alpha),
        round(base.blue() * inv + overlay.blue() * clamped_alpha),
    )


def _diff_highlight_colors(palette: QPalette) -> tuple[QColor, QColor]:
    """Build diff highlight colors that stay readable in both light and dark themes."""
    base = palette.color(QPalette.ColorRole.Base)
    highlight = palette.color(QPalette.ColorRole.Highlight)
    text = palette.color(QPalette.ColorRole.Text)
    background = _blend_colors(base, highlight, 0.5)
    return background, text


def _danger_text_color(palette: QPalette) -> QColor:
    """Build a readable danger color used for delete actions."""
    return danger_accent_color(palette)


def _danger_button_stylesheet(palette: QPalette) -> str:
    color = _danger_text_color(palette).name(QColor.NameFormat.HexRgb)
    return f"QPushButton {{ color: {color}; font-weight: bold; padding: 2px; }}"


class DiffHighlightDelegate(QStyledItemDelegate):
    _DOC_CACHE_MAX = 256

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._doc_cache: dict[tuple, QTextDocument] = {}

    def _normalized_ranges(self, text: str, raw_ranges: object) -> list[tuple[int, int]]:
        if not isinstance(raw_ranges, list):
            return []
        normalized: list[tuple[int, int]] = []
        text_len = len(text)
        for entry in raw_ranges:
            if not isinstance(entry, tuple) or len(entry) != 2:
                continue
            start, end = entry
            if not isinstance(start, int) or not isinstance(end, int):
                continue
            bounded_start = max(0, min(text_len, start))
            bounded_end = max(bounded_start, min(text_len, end))
            if bounded_end > bounded_start:
                normalized.append((bounded_start, bounded_end))
        return normalized

    def _build_document(self, text: str, ranges: list[tuple[int, int]], option: QStyleOptionViewItem) -> QTextDocument:
        document = QTextDocument()
        document.setPlainText(text)
        document.setDefaultFont(option.font)

        if ranges:
            cursor = QTextCursor(document)
            char_format = QTextCharFormat()
            highlight_bg, highlight_text = _diff_highlight_colors(option.palette)
            char_format.setBackground(highlight_bg)
            char_format.setForeground(highlight_text)
            for start, end in ranges:
                cursor.setPosition(start)
                cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
                cursor.mergeCharFormat(char_format)

        return document

    def _cached_document(
        self,
        text: str,
        ranges: list[tuple[int, int]],
        option: QStyleOptionViewItem,
    ) -> QTextDocument:
        highlight_bg, _ = _diff_highlight_colors(option.palette)
        key = (text, tuple(ranges), option.font.key(), highlight_bg.name())
        doc = self._doc_cache.get(key)
        if doc is None:
            doc = self._build_document(text, ranges, option)
            if len(self._doc_cache) >= self._DOC_CACHE_MAX:
                self._doc_cache.pop(next(iter(self._doc_cache)))
            self._doc_cache[key] = doc
        return doc

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:  # type: ignore[override]
        style = option.widget.style() if option.widget is not None else None
        if style is None:
            style = QApplication.style()

        base = QStyleOptionViewItem(option)
        self.initStyleOption(base, index)
        base.state &= ~QStyle.StateFlag.State_HasFocus
        text = base.text
        base.text = ""
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, base, painter, option.widget)

        text_rect = style.subElementRect(
            QStyle.SubElement.SE_ItemViewItemText,
            base,
            option.widget,
        )
        if text_rect.isEmpty():
            return

        if option.state & QStyle.StateFlag.State_Selected:
            palette = option.palette
            painter.save()
            painter.fillRect(
                text_rect,
                palette.brush(QPalette.ColorRole.Highlight),
            )
            painter.restore()

        if not text:
            return

        # When selected, keep text plain for readability on highlight background.
        ranges = [] if (option.state & QStyle.StateFlag.State_Selected) else self._normalized_ranges(text, index.data(DIFF_RANGES_ROLE))
        document = self._cached_document(text, ranges, option)
        document.setTextWidth(float(max(10, text_rect.width())))

        painter.save()
        painter.translate(text_rect.topLeft())
        document.drawContents(painter)
        painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index) -> object:  # type: ignore[override]
        base = QStyleOptionViewItem(option)
        self.initStyleOption(base, index)
        text = base.text
        if not text:
            return super().sizeHint(option, index)

        width = option.rect.width()
        if width <= 0 and option.widget is not None:
            width = option.widget.viewport().width()
        width = max(40, width - 8)

        ranges = self._normalized_ranges(text, index.data(DIFF_RANGES_ROLE))
        document = self._cached_document(text, ranges, option)
        document.setTextWidth(float(width))
        size = document.size().toSize()
        size.setHeight(max(size.height() + 6, 24))
        return size


class EditableDiffDelegate(DiffHighlightDelegate):
    _CLOSED_BY_DELEGATE_PROP = "_closed_by_editable_diff_delegate"
    editor_created = pyqtSignal(QTextEdit)

    def _resize_editor(self, editor: QTextEdit) -> None:
        margins = editor.contentsMargins()
        document = editor.document()
        document.setTextWidth(max(20.0, editor.viewport().width()))
        doc_height = int(document.size().height())
        frame = editor.frameWidth() * 2
        vertical_margins = margins.top() + margins.bottom()
        min_height = editor.fontMetrics().lineSpacing() + 10
        editor.setFixedHeight(max(min_height, doc_height + frame + vertical_margins + 4))

    def createEditor(self, parent, option, index):  # type: ignore[override]
        editor = QTextEdit(parent)
        editor.setAcceptRichText(False)
        editor.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        editor.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        editor.setFrameStyle(0)
        editor.installEventFilter(self)
        editor.textChanged.connect(lambda: self._resize_editor(editor))
        self.editor_created.emit(editor)
        return editor

    def setEditorData(self, editor, index) -> None:  # type: ignore[override]
        if isinstance(editor, QTextEdit):
            editor.setPlainText(index.data(Qt.ItemDataRole.EditRole) or "")
            self._resize_editor(editor)
            editor.selectAll()

    def setModelData(self, editor, model, index) -> None:  # type: ignore[override]
        if isinstance(editor, QTextEdit):
            model.setData(index, editor.toPlainText(), Qt.ItemDataRole.EditRole)

    def updateEditorGeometry(self, editor, option, index) -> None:  # type: ignore[override]
        if isinstance(editor, QTextEdit):
            editor.setGeometry(option.rect)
            self._resize_editor(editor)

    def eventFilter(self, editor, event) -> bool:
        if isinstance(editor, QTextEdit):
            if event.type() == QEvent.Type.KeyPress:
                key = event.key()
                if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                    editor.setProperty(self._CLOSED_BY_DELEGATE_PROP, True)
                    self.commitData.emit(editor)
                    self.closeEditor.emit(editor, QStyledItemDelegate.EndEditHint.SubmitModelCache)
                    return True
                if key == Qt.Key.Key_Escape:
                    editor.setProperty(self._CLOSED_BY_DELEGATE_PROP, True)
                    self.closeEditor.emit(editor, QStyledItemDelegate.EndEditHint.RevertModelCache)
                    return True
            if event.type() == QEvent.Type.FocusOut:
                if bool(editor.property(self._CLOSED_BY_DELEGATE_PROP)):
                    return True
                editor.setProperty(self._CLOSED_BY_DELEGATE_PROP, True)
                self.commitData.emit(editor)
                self.closeEditor.emit(editor, QStyledItemDelegate.EndEditHint.SubmitModelCache)
                return True
        return super().eventFilter(editor, event)
