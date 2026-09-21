"""Batch "Autofix ratio" for the main window.

Crops every listed image whose closest allowed ratio keeps at least
``AUTOFIX_MIN_KEPT_FRACTION`` of the pixels (see ``utils.aspect_ratio``) at
the centre, the same rule the merge dialog's Autofix ratio button applies to
one image.  The crops run on a worker thread behind a modal progress dialog
so the UI stays responsive and the user can stop early; a stop leaves the
images already cropped as they are and skips the rest.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import QMessageBox, QProgressDialog, QWidget

from imagetagger.utils.aspect_ratio import CropCandidate, centered_crop_box
from imagetagger.utils.image_crop import ImageCropError, crop_image_file


@dataclass(frozen=True)
class RatioAutofixTarget:
    image_path: Path
    image_size: tuple[int, int]  # size the candidate was computed for
    candidate: CropCandidate


@dataclass(frozen=True)
class RatioAutofixResult:
    image_path: Path
    new_size: tuple[int, int] | None  # None when the crop failed
    error: str = ""


class _RatioAutofixWorker(QObject):
    progress = pyqtSignal(int, int, str)   # (done, total, file name)
    item_done = pyqtSignal(object)         # RatioAutofixResult
    finished = pyqtSignal(int, int, bool)  # (cropped, failed, stopped)

    def __init__(self, targets: list[RatioAutofixTarget]) -> None:
        super().__init__()
        self._targets = targets
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        total = len(self._targets)
        cropped = failed = 0
        stopped = False
        for done, target in enumerate(self._targets):
            if self._stop_requested:
                stopped = True
                break
            self.progress.emit(done, total, target.image_path.name)
            width, height = target.image_size
            box = centered_crop_box(width, height, target.candidate.width, target.candidate.height)
            try:
                # expected_size refuses the crop if the file changed since the
                # size was read (an external edit, or an earlier crop).
                new_size = crop_image_file(target.image_path, box, expected_size=target.image_size)
            except ImageCropError as exc:
                failed += 1
                self.item_done.emit(RatioAutofixResult(target.image_path, None, str(exc)))
                continue
            cropped += 1
            self.item_done.emit(RatioAutofixResult(target.image_path, (int(new_size[0]), int(new_size[1]))))
        self.finished.emit(cropped, failed, stopped)


def run_batch_autofix_ratio(
    parent: QWidget,
    targets: list[RatioAutofixTarget],
    on_item_done: Callable[[RatioAutofixResult], None],
    on_finished: Callable[[int, int, bool, list[RatioAutofixResult]], None],
) -> tuple[QThread, _RatioAutofixWorker]:
    """Crop ``targets`` on a worker thread behind a modal progress dialog.

    ``on_item_done`` runs on the GUI thread after every crop attempt (so the
    list can update as the batch proceeds); ``on_finished`` runs once at the
    end with the counts and every failed result.  The caller must keep the
    returned thread and worker referenced until ``on_finished`` has run.
    """
    total = len(targets)
    dialog = QProgressDialog("Cropping images…", "Stop", 0, total, parent)
    dialog.setWindowTitle("Batch Autofix ratio")
    dialog.setWindowModality(Qt.WindowModality.WindowModal)
    dialog.setMinimumDuration(0)
    dialog.setAutoClose(False)
    dialog.setAutoReset(False)
    dialog.setValue(0)

    worker = _RatioAutofixWorker(targets)
    thread = QThread(parent)
    worker.moveToThread(thread)
    failures: list[RatioAutofixResult] = []

    def _on_progress(done: int, count: int, name: str) -> None:
        dialog.setLabelText(f"Cropping {name} ({done + 1} of {count})…")
        dialog.setValue(done)

    def _on_item_done(result: object) -> None:
        if isinstance(result, RatioAutofixResult):
            if result.new_size is None:
                failures.append(result)
            on_item_done(result)

    def _on_finished(cropped: int, failed: int, stopped: bool) -> None:
        dialog.setValue(total)
        dialog.close()
        dialog.deleteLater()
        thread.quit()
        on_finished(cropped, failed, stopped, failures)

    # A plain callable runs in the GUI thread that emits the signal.  A slot on
    # the worker would be queued to its thread, whose event loop is busy in
    # run() until the batch ends, so Stop would never arrive.
    dialog.canceled.connect(lambda: worker.request_stop())
    worker.progress.connect(_on_progress)
    worker.item_done.connect(_on_item_done)
    worker.finished.connect(_on_finished)
    thread.started.connect(worker.run)
    thread.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    thread.start()
    dialog.show()
    return thread, worker


def show_batch_autofix_summary(
    parent: QWidget,
    cropped: int,
    failed: int,
    stopped: bool,
    failures: list[RatioAutofixResult],
) -> str:
    """Show a failure list when needed; return the status-bar summary line."""
    summary = f"Batch Autofix ratio: cropped {cropped} image(s)"
    if failed:
        summary += f", {failed} failed"
    if stopped:
        summary += " (stopped)"
    if failures:
        lines = [f"{result.image_path.name}: {result.error}" for result in failures[:15]]
        if len(failures) > 15:
            lines.append(f"… and {len(failures) - 15} more")
        QMessageBox.warning(
            parent,
            "Batch Autofix ratio",
            f"{cropped} image(s) cropped. Could not crop {failed} image(s):\n\n" + "\n".join(lines),
        )
    return summary
