"""Movable fixed-size crop frame drawn over the merge dialog's image preview.

The overlay is a transparent child of the preview ``QLabel``.  The frame has
a fixed size in image pixels (the largest crop for the chosen ratio); the
user only decides *where* it sits, i.e. which part of the image is discarded.
All state is kept in image pixels so the result is independent of how the
preview happens to be scaled.
"""

from __future__ import annotations

from PyQt6.QtCore import QEvent, QObject, QPoint, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QMouseEvent, QPainter, QPainterPath, QPen, QWheelEvent
from PyQt6.QtWidgets import QLabel, QStyle, QWidget


class CropOverlay(QWidget):
    """Dims the discarded area and lets the user drag the kept frame."""

    frame_moved = pyqtSignal()

    def __init__(self, image_label: QLabel) -> None:
        super().__init__(image_label)
        self._label = image_label
        self._image_w = 0
        self._image_h = 0
        self._crop_w = 0
        self._crop_h = 0
        self._origin_x = 0  # frame top-left, image pixels
        self._origin_y = 0
        self._drag_anchor: QPointF | None = None
        self._drag_origin = (0, 0)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        image_label.installEventFilter(self)
        self.hide()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def begin(self, image_width: int, image_height: int) -> None:
        self._image_w = max(0, int(image_width))
        self._image_h = max(0, int(image_height))
        self._crop_w = self._crop_h = 0
        self._drag_anchor = None
        self.setGeometry(self._label.rect())
        self.show()
        self.raise_()
        self.setFocus(Qt.FocusReason.OtherFocusReason)

    def end(self) -> None:
        self._drag_anchor = None
        self.hide()

    def set_crop_size(self, crop_width: int, crop_height: int) -> None:
        """Set the frame size in image pixels, keeping its centre where possible."""
        crop_width = max(1, min(int(crop_width), self._image_w))
        crop_height = max(1, min(int(crop_height), self._image_h))
        if self._crop_w > 0 and self._crop_h > 0:
            center_x = self._origin_x + self._crop_w / 2.0
            center_y = self._origin_y + self._crop_h / 2.0
        else:
            center_x = self._image_w / 2.0
            center_y = self._image_h / 2.0
        self._crop_w, self._crop_h = crop_width, crop_height
        self._set_origin(round(center_x - crop_width / 2.0), round(center_y - crop_height / 2.0))
        self._update_cursor()

    def image_size(self) -> tuple[int, int]:
        """The image size the frame was laid out on."""
        return (self._image_w, self._image_h)

    def crop_box(self) -> tuple[int, int, int, int]:
        """(left, top, right, bottom) in image pixels."""
        return (
            self._origin_x,
            self._origin_y,
            self._origin_x + self._crop_w,
            self._origin_y + self._crop_h,
        )

    def nudge(self, step_x: int, step_y: int, *, big: bool = False) -> None:
        """Move the frame by whole preview pixels (x10 when ``big``)."""
        scale = self._scale()
        image_step = max(1, round(1.0 / scale)) if scale > 0 else 1
        if big:
            image_step *= 10
        self._set_origin(self._origin_x + step_x * image_step, self._origin_y + step_y * image_step)

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def _display_rect(self) -> QRectF | None:
        """Where the label paints the scaled image, in overlay coordinates."""
        pixmap = self._label.pixmap()
        if pixmap is None or pixmap.isNull():
            return None
        size = pixmap.deviceIndependentSize().toSize()
        if size.isEmpty():
            return None
        aligned = QStyle.alignedRect(
            self._label.layoutDirection(),
            self._label.alignment(),
            size,
            self._label.contentsRect(),
        )
        return QRectF(aligned)

    def _scale(self) -> float:
        rect = self._display_rect()
        if rect is None or self._image_w <= 0:
            return 0.0
        return rect.width() / float(self._image_w)

    def _frame_rect(self, display: QRectF) -> QRectF:
        scale_x = display.width() / float(self._image_w)
        scale_y = display.height() / float(self._image_h)
        return QRectF(
            display.x() + self._origin_x * scale_x,
            display.y() + self._origin_y * scale_y,
            self._crop_w * scale_x,
            self._crop_h * scale_y,
        )

    def _set_origin(self, origin_x: int, origin_y: int) -> None:
        origin_x = max(0, min(int(origin_x), self._image_w - self._crop_w))
        origin_y = max(0, min(int(origin_y), self._image_h - self._crop_h))
        if (origin_x, origin_y) == (self._origin_x, self._origin_y):
            self.update()
            return
        self._origin_x, self._origin_y = origin_x, origin_y
        self.update()
        self.frame_moved.emit()

    def _slack(self) -> tuple[int, int]:
        """Room the frame has to move along x and y, in image pixels.

        A maximal crop spans the image along one axis, so only the other axis
        has room to move.
        """
        return (self._image_w - self._crop_w, self._image_h - self._crop_h)

    def _update_cursor(self) -> None:
        slack_x, slack_y = self._slack()
        if slack_x > slack_y:
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        elif slack_y > slack_x:
            self.setCursor(Qt.CursorShape.SizeVerCursor)
        else:
            self.setCursor(Qt.CursorShape.SizeAllCursor)

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self._label and event.type() == QEvent.Type.Resize and self.isVisible():
            self.setGeometry(self._label.rect())
        return super().eventFilter(watched, event)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() != Qt.MouseButton.LeftButton:
            event.ignore()
            return
        display = self._display_rect()
        if display is None or self._crop_w <= 0:
            return
        position = event.position()
        if not self._frame_rect(display).contains(position):
            # Clicking outside the frame re-centres it under the pointer.
            scale_x = display.width() / float(self._image_w)
            scale_y = display.height() / float(self._image_h)
            self._set_origin(
                round((position.x() - display.x()) / scale_x - self._crop_w / 2.0),
                round((position.y() - display.y()) / scale_y - self._crop_h / 2.0),
            )
        self._drag_anchor = position
        self._drag_origin = (self._origin_x, self._origin_y)
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if self._drag_anchor is None:
            return
        display = self._display_rect()
        if display is None:
            return
        scale_x = display.width() / float(self._image_w)
        scale_y = display.height() / float(self._image_h)
        delta = event.position() - self._drag_anchor
        self._set_origin(
            self._drag_origin[0] + round(delta.x() / scale_x),
            self._drag_origin[1] + round(delta.y() / scale_y),
        )
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_anchor = None
            event.accept()
            return
        event.ignore()

    def wheelEvent(self, event: QWheelEvent) -> None:  # type: ignore[override]
        # Treat the frame as a viewport over the image: scrolling moves it.
        delta: QPoint = event.pixelDelta()
        if delta.isNull():
            delta = event.angleDelta() / 4
        scale = self._scale()
        if scale <= 0 or delta.isNull():
            event.accept()
            return
        # Whichever way the wheel/trackpad is scrolled, move along the axis
        # that actually has room.
        slack_x, slack_y = self._slack()
        dominant = delta.y() if abs(delta.y()) >= abs(delta.x()) else delta.x()
        move_x = move_y = 0.0
        if slack_x > slack_y:
            move_x = -dominant
        else:
            move_y = -dominant
        self._set_origin(
            self._origin_x + round(move_x / scale),
            self._origin_y + round(move_y / scale),
        )
        event.accept()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        display = self._display_rect()
        if display is None or self._crop_w <= 0 or self._image_w <= 0 or self._image_h <= 0:
            return
        frame = self._frame_rect(display)

        painter = QPainter(self)
        discarded = QPainterPath()
        discarded.addRect(display)
        kept = QPainterPath()
        kept.addRect(frame)
        painter.fillPath(discarded.subtracted(kept), QColor(0, 0, 0, 165))

        # Rule-of-thirds guides help judge the composition that is kept.
        painter.setPen(QPen(QColor(255, 255, 255, 70), 1))
        for index in (1, 2):
            x = frame.x() + frame.width() * index / 3.0
            y = frame.y() + frame.height() * index / 3.0
            painter.drawLine(QPointF(x, frame.top()), QPointF(x, frame.bottom()))
            painter.drawLine(QPointF(frame.left(), y), QPointF(frame.right(), y))

        # Dark halo + light line stays visible on any image content.
        outline = frame.adjusted(0.5, 0.5, -0.5, -0.5)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(0, 0, 0, 200), 3))
        painter.drawRect(outline)
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.drawRect(outline)
        painter.end()
