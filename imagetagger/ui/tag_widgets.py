from __future__ import annotations

from PyQt6.QtCore import QEvent, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QListWidget,
    QStyledItemDelegate,
    QTextEdit,
    QWidget,
)


class TagListWidget(QListWidget):
    tags_reordered = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)

    def dropEvent(self, event) -> None:  # type: ignore[override]
        super().dropEvent(event)
        self.tags_reordered.emit()


class WrappedTagItemDelegate(QStyledItemDelegate):
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
                    self.commitData.emit(editor)
                    self.closeEditor.emit(editor, QStyledItemDelegate.EndEditHint.SubmitModelCache)
                    return True
                if key == Qt.Key.Key_Escape:
                    self.closeEditor.emit(editor, QStyledItemDelegate.EndEditHint.RevertModelCache)
                    return True
            if event.type() == QEvent.Type.FocusOut:
                self.commitData.emit(editor)
                self.closeEditor.emit(editor, QStyledItemDelegate.EndEditHint.SubmitModelCache)
        return super().eventFilter(editor, event)
